"""Toy demonstration of FARAD on CPU (synthetic data, no downloads).

1. Builds Fourier clouds from a synthetic image batch (Listing 1) and checks that their amplitude
   spectrum equals the batch-averaged amplitude (Eq. 3), and that they match no class shape (Thm. 1).
2. Trains a linear classifier on a synthetic long-tailed set, so that it is biased toward the head
   classes, and prints per-class test accuracy before and after subtracting z_bias = g(x_rand) (Eq. 4-8).

Run:  python demo.py
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from farad import bias_logits, farad_predict, random_phase_reference

H = W = 32
K = 5                                   # classes: bar, column, diagonal, disk, ring
TRAIN_COUNTS = [400, 160, 64, 26, 10]   # long-tailed labeled set (imbalance ratio 40), head -> tail
TEST_PER_CLASS = 200                    # balanced test set


def texture(n, gen):
    """Background shared by all classes: 1/f noise with random phase, amplitude ~ 1/(f + 0.05), RGB tint."""
    fy = torch.fft.fftfreq(H)[:, None]
    fx = torch.fft.rfftfreq(W)[None, :]
    amp = 1.0 / (torch.sqrt(fx ** 2 + fy ** 2) + 0.05)
    amp[0, 0] = 0.0                                                    # zero-mean background
    phase = 2 * math.pi * torch.rand((n, 1, H, W // 2 + 1), generator=gen)
    base = torch.fft.irfft2(amp * torch.exp(1j * phase), s=(H, W))     # [n, 1, H, W]
    base = base / base.flatten(1).std(dim=1).view(-1, 1, 1, 1)
    tint = torch.tensor([0.9, 1.0, 0.8]).view(1, 3, 1, 1)
    return base * tint                                                 # [n, 3, H, W]


def shapes(labels, gen, jitter=2):
    """Class-specific shape (the semantics), jittered by up to `jitter` pixels: [n, 1, H, W]."""
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32),
                            indexing='ij')
    out = torch.zeros(len(labels), 1, H, W)
    shift = torch.randint(-jitter, jitter + 1, (len(labels), 2), generator=gen).float()
    for i, c in enumerate(labels.tolist()):
        y = yy - (H / 2 + shift[i, 0])
        x = xx - (W / 2 + shift[i, 1])
        r = torch.sqrt(x ** 2 + y ** 2)
        if c == 0:
            m = (y.abs() < 2) & (x.abs() < 10)
        elif c == 1:
            m = (x.abs() < 2) & (y.abs() < 10)
        elif c == 2:
            m = ((x - y).abs() < 2.5) & (r < 11)
        elif c == 3:
            m = r < 6
        else:
            m = (r > 7) & (r < 10)
        out[i, 0] = m.float()
    return out


def make_set(counts, gen, noise=1.0):
    labels = torch.cat([torch.full((n,), c, dtype=torch.long) for c, n in enumerate(counts)])
    x = shapes(labels, gen) + noise * texture(len(labels), gen)       # [N, 3, H, W]
    return x, labels


def corr(a, b):
    """|cosine similarity| between mean-centred gray-level images: [len(a), len(b)]."""
    a = a.mean(dim=1).flatten(1)
    b = b.mean(dim=1).flatten(1)
    a = F.normalize(a - a.mean(dim=1, keepdim=True), dim=1)
    b = F.normalize(b - b.mean(dim=1, keepdim=True), dim=1)
    return (a @ b.T).abs()


def per_class_acc(pred, y):
    return [(pred[y == c] == c).float().mean().item() for c in range(K)]


def main():
    t0 = time.time()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    # ------------------------------------------------------------ 1. Fourier clouds from a synthetic batch
    x, y = make_set([16] * K, gen)                                    # 80 images, 3 x 32 x 32
    clouds = random_phase_reference(x, generator=gen)                 # Listing 1: one cloud per image
    a_bar = torch.fft.rfft2(x).abs().mean(dim=0)                      # batch-averaged amplitude, Eq. (3)
    a_cloud = torch.fft.rfft2(clouds).abs()
    inner = slice(1, W // 2)                                          # columns 0 < v < W/2: exact by construction
    rel = (a_cloud[..., inner] - a_bar[..., inner]).abs() / a_bar[..., inner].clamp_min(1e-6)
    edge = [0, W // 2]                                                # self-conjugate columns, symmetrized by irfft2
    ratio = (a_cloud[..., edge].mean() / a_bar[..., edge].mean()).item()
    templates = shapes(torch.arange(K), gen, jitter=0)                # shape dictionary Psi_k
    som_cloud = corr(clouds, templates).max(dim=1).values             # semantic overlap of each cloud
    own = corr(x, templates)[torch.arange(len(y)), y]                 # each image vs. its own class shape
    print(f'[1] {len(clouds)} Fourier clouds from {len(x)} synthetic images ({x.shape[1]}x{H}x{W})')
    print(f'    amplitude vs batch mean, bins 0 < v < W/2 ({inner.stop - inner.start}/{W // 2 + 1} columns): '
          f'max relative error {rel.max().item():.1e}')
    print(f'    columns v = 0 and v = W/2 (symmetrized by irfft2, as in Listing 1): mean amplitude ratio {ratio:.2f}')
    print(f'    max |corr| with any class shape: clouds {som_cloud.max().item():.3f}, '
          f'real images (own class, mean) {own.mean().item():.3f}')

    # ------------------------------------------------------------ 2. biased toy classifier, with and without z_bias
    x_tr, y_tr = make_set(TRAIN_COUNTS, gen)
    x_te, y_te = make_set([TEST_PER_CLASS] * K, gen)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * H * W, K))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(300):                                              # full-batch training on the long-tailed set
        opt.zero_grad()
        F.cross_entropy(model(x_tr), y_tr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred_plain = model(x_te).argmax(dim=1)
        # the same seed reproduces the clouds used inside farad_predict, so the printed z_bias is the one subtracted
        z_bias = bias_logits(model, random_phase_reference(x_te, num_refs=32,
                                                           generator=torch.Generator().manual_seed(1)))
    pred_farad = farad_predict(model, x_te, num_refs=32,                 # Eq. (6)-(8): clouds from the test batch
                               generator=torch.Generator().manual_seed(1))
    before, after = per_class_acc(pred_plain, y_te), per_class_acc(pred_farad, y_te)

    print(f'\n[2] linear classifier trained on a long-tailed set, train counts {TRAIN_COUNTS}')
    print('    z_bias = g(x_rand), head -> tail: ' + ' '.join(f'{v:+.2f}' for v in z_bias[0].tolist()))
    print('    class  train  acc g(x)  acc g(x) - z_bias')
    for c in range(K):
        print(f'    {c:5d}  {TRAIN_COUNTS[c]:5d}  {before[c]:8.3f}  {after[c]:17.3f}')
    print(f'    balanced accuracy  {sum(before) / K:.3f} -> {sum(after) / K:.3f}')
    print(f'    test predictions per class  {torch.bincount(pred_plain, minlength=K).tolist()} -> '
          f'{torch.bincount(pred_farad, minlength=K).tolist()}  (true: {[TEST_PER_CLASS] * K})')
    print(f'\ndone in {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()

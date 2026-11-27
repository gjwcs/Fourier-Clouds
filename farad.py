"""FARAD: Fourier-Adapted Reference for Accurate Debiasing.

Concise PyTorch implementation of the bias correction proposed in
"Fourier Clouds: Fast Bias Correction for Imbalanced Semi-Supervised Learning" (NeurIPS 2025).

A Fourier cloud is a random-phase image: it keeps the batch-averaged amplitude spectrum of real images
(Eq. 3) and replaces the phase by i.i.d. uniform noise (Eq. 2), so it carries the color, texture and
frequency statistics of the data but no shapes. The classifier's logits on it estimate the class bias
z_bias = g(x_rand) (Eq. 4), which is subtracted from the logits of real inputs when pseudo-labelling
unlabeled data (Eq. 5, Listing 3) and before the arg max at test time (Eq. 6-8).

Equation, listing and algorithm numbers refer to the paper.
"""
import math
from typing import Optional

import torch

__all__ = ['random_phase_reference', 'bias_logits', 'debias_logits', 'debiased_pseudo_labels', 'farad_predict']


def random_phase_reference(x: torch.Tensor, num_refs: Optional[int] = None,
                           generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Build Fourier clouds (random-phase references) from a batch of real images.

    Follows Listing 1 (Eq. 2-3, Algorithm 1 Steps 1-2):
        freq_r2c  = rfft2(x)                         # batched real-to-complex FFT, half spectrum (Eq. 9)
        amp_mean  = |freq_r2c|.mean(batch)           # batch-averaged amplitude A_bar(u, v), Eq. (3)
        rand_phase ~ Uniform[-pi, pi], i.i.d.        # per bin and per channel (App. G.5), Eq. (2)
        x_rand    = irfft2(amp_mean * exp(j * rand_phase))

    Args:
        x: real tensor [B, C, H, W]. Algorithm 1 averages the amplitude over the labeled and the
            unlabeled images of the current batch.
        num_refs: number of clouds to draw. Default (None): one per input image, as in Listing 1.
            Algorithm 1 uses a single cloud per iteration (num_refs=1).
        generator: optional torch.Generator for reproducible phases.

    Returns:
        real tensor [num_refs, C, H, W] with the dtype of ``x``.

    Notes:
        * Listing 1 writes ``torch.rand_like(freq_r2c)``; ``freq_r2c`` is complex, so the real-valued
          phase of Eq. (2) is drawn with the matching real dtype and the same shape.
        * As in Listing 1, every bin is randomized, including the zero-frequency (DC) bin. For bins that
          are their own conjugate partner (DC, and the Nyquist bins when H or W is even) ``irfft2`` keeps
          only the real part, and it Hermitian-symmetrizes the v = 0 and v = W/2 columns of the half
          spectrum; every other bin of the returned clouds has amplitude exactly A_bar(u, v).
    """
    if x.dim() != 4:
        raise ValueError(f'expected a batch of images [B, C, H, W], got shape {tuple(x.shape)}')
    xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    size = xf.shape[-2:]
    freq_r2c = torch.fft.rfft2(xf, dim=(-2, -1))                       # R2C transform
    amp_mean = freq_r2c.abs().mean(dim=0, keepdim=True)                # average amplitude, Eq. (3)
    n = xf.shape[0] if num_refs is None else int(num_refs)
    shape = (n,) + tuple(freq_r2c.shape[1:])
    rand_phase = (2 * math.pi * torch.rand(shape, generator=generator, dtype=amp_mean.dtype,
                                           device=xf.device) - math.pi)  # Uniform[-pi, pi], Eq. (2)
    freq_rand = amp_mean * torch.exp(1j * rand_phase)                  # combine amplitude & random phase
    x_rand = torch.fft.irfft2(freq_rand, s=size, dim=(-2, -1))         # back to the image domain
    return x_rand.to(x.dtype)


def bias_logits(model: torch.nn.Module, x_rand: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Class-bias estimate z_bias = g_theta(x_rand), Eq. (4) / Algorithm 1 Step 3.

    Args:
        model: classifier returning logits [N, K].
        x_rand: Fourier clouds from :func:`random_phase_reference`.
        reduce: if True, average the logits over the clouds and return one bias vector [1, K]
            (with a single cloud this is exactly Algorithm 1); if False, return one row per cloud,
            which Listing 3 pairs with the samples of the batch.

    The model is called in its current mode, as ``model(x_rand)`` in Listing 3. Wrap the call in
    ``torch.no_grad()`` when the bias only enters targets (pseudo-labels) or predictions.
    """
    z = model(x_rand)
    return z.mean(dim=0, keepdim=True) if reduce else z


def debias_logits(logits: torch.Tensor, z_bias: torch.Tensor) -> torch.Tensor:
    """Bias-corrected logits g*_theta(x) = g_theta(x) - z_bias: Eq. (5) in training, Eq. (7) at test time.

    ``z_bias`` is [1, K] (one bias vector, broadcast over the batch) or [N, K] (one per sample, Listing 3).
    """
    return logits - z_bias


@torch.no_grad()
def debiased_pseudo_labels(model: torch.nn.Module, u_weak: torch.Tensor, ref_images: Optional[torch.Tensor] = None,
                           num_refs: int = 1, per_sample: bool = False,
                           generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Soft pseudo-labels softmax(g(u_weak) - g(x_rand)) for a FixMatch / ReMixMatch step (Listing 3, App. B).

    Args:
        model: classifier returning logits.
        u_weak: weakly augmented unlabeled batch [N, C, H, W].
        ref_images: images whose batch-averaged amplitude defines the clouds; Algorithm 1 uses the
            labeled and unlabeled images of the batch. Defaults to ``u_weak``.
        num_refs: clouds averaged into one bias vector (Algorithm 1: 1).
        per_sample: draw one cloud per unlabeled sample and subtract row by row, as in Listing 3
            (``num_refs`` is then ignored).
        generator: optional torch.Generator for reproducible phases.

    Returns:
        soft pseudo-labels [N, K]. App. B uses FixMatch with threshold tau = 0, i.e. all samples are kept;
        apply a confidence mask afterwards if a threshold is wanted.
    """
    ref = u_weak if ref_images is None else ref_images
    n = u_weak.shape[0] if per_sample else num_refs
    x_rand = random_phase_reference(ref, num_refs=n, generator=generator)
    z_bias = bias_logits(model, x_rand, reduce=not per_sample)
    return torch.softmax(debias_logits(model(u_weak), z_bias), dim=-1)


@torch.no_grad()
def farad_predict(model: torch.nn.Module, x: torch.Tensor, num_refs: int = 1,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Test-time prediction f*(x) = argmax_c [g(x) - g(x_rand)]_c, Eq. (6)-(8).

    A fresh cloud is built from the amplitude of the test batch itself ("a fresh random-phase image per
    batch", App. B); ``num_refs`` > 1 averages several clouds into one bias vector. Call ``model.eval()``
    first.
    """
    x_rand = random_phase_reference(x, num_refs=num_refs, generator=generator)
    return debias_logits(model(x), bias_logits(model, x_rand)).argmax(dim=-1)

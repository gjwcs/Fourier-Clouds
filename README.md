# Fourier Clouds: Fast Bias Correction for Imbalanced Semi-Supervised Learning

Jiawei Gu, Yidi Wang, Qingqiang Sun, Xinming Li, Xiao Luo, Ziyue Qiao · NeurIPS 2025 · [Paper](https://neurips.cc/virtual/2025/poster/118646)

In class-imbalanced semi-supervised learning, pseudo-labels inherit and amplify the classifier's bias toward
majority classes. Reference-based debiasing estimates this bias from the classifier's logits on an input without
semantic content and subtracts it, but blank or uniform references (as used by CDMAD) also lack the color, texture
and frequency statistics of real data. FARAD (Fourier-Adapted Reference for Accurate Debiasing) uses random-phase
images instead, which we call Fourier clouds: they keep the batch-averaged amplitude spectrum of real images and
draw the phase uniformly at random, so they are statistically representative yet semantically irrelevant. The
logits of a Fourier cloud are subtracted from those of real inputs when pseudo-labelling unlabeled data and at test
time, which improves minority-class accuracy, while batched real-to-complex FFTs keep the extra cost small.

This repository provides a concise implementation of the method. The detailed version with the full experimental code will be released later.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python demo.py
```

`demo.py` runs on CPU with synthetic data and downloads nothing. It builds Fourier clouds from a synthetic image
batch, checks that their amplitude spectrum equals the batch average and that they match none of the class shapes,
then trains a linear classifier on a long-tailed set and prints its per-class test accuracy before and after
subtracting the bias logits `z_bias`.

## Usage

FARAD plugs into any pseudo-labelling SSL loop (FixMatch, ReMixMatch, ...) and into test-time prediction.

```python
import torch
import torch.nn.functional as F
from farad import debiased_pseudo_labels, farad_predict

# --- one FixMatch-style training step (Listing 3 of the paper) ---
# Fourier cloud from the labeled + unlabeled batch, z_bias = g(x_rand), pseudo-labels softmax(g(u_weak) - z_bias)
pseudo = debiased_pseudo_labels(model, u_weak, ref_images=torch.cat([x_lb, u_weak]))
loss_u = F.cross_entropy(model(u_strong), pseudo)      # soft targets; the paper keeps all samples (tau = 0)
loss_x = F.cross_entropy(model(x_lb), y_lb)
loss = loss_x + lambda_u * loss_u
optimizer.zero_grad()
loss.backward()
optimizer.step()

# --- test time: a fresh Fourier cloud per test batch, argmax of g(x) - g(x_rand) ---
model.eval()
pred = farad_predict(model, x_test)
```

The same steps with the low-level functions:

```python
from farad import random_phase_reference, bias_logits, debias_logits

x_rand = random_phase_reference(torch.cat([x_lb, u_weak]), num_refs=1)     # Eq. (2)-(3)
with torch.no_grad():
    z_bias = bias_logits(model, x_rand)                                     # Eq. (4), shape [1, K]
    pseudo = torch.softmax(debias_logits(model(u_weak), z_bias), dim=-1)    # Eq. (5)
```

## API

| Function | Returns | Paper |
|---|---|---|
| `random_phase_reference(x, num_refs=None, generator=None)` | Fourier clouds `[num_refs, C, H, W]` built from the batch `x` `[B, C, H, W]` (default: one per input image) | Eq. (2)–(3), Listing 1 |
| `bias_logits(model, x_rand, reduce=True)` | `z_bias = g(x_rand)`, averaged to `[1, K]` (or one row per cloud) | Eq. (4) |
| `debias_logits(logits, z_bias)` | `g(x) - z_bias` | Eq. (5), (7) |
| `debiased_pseudo_labels(model, u_weak, ref_images=None, num_refs=1, per_sample=False)` | soft pseudo-labels `[N, K]`; `per_sample=True` pairs one cloud with each sample | Listing 3 |
| `farad_predict(model, x, num_refs=1)` | predicted labels `[N]` | Eq. (6)–(8) |

As in Listing 1, the phase of every frequency bin, including the zero-frequency bin, is drawn i.i.d. from
Uniform[−π, π], independently per channel; see the docstring of `random_phase_reference`.

## Citation

```bibtex
@inproceedings{gu2025fourier,
  title     = {Fourier Clouds: Fast Bias Correction for Imbalanced Semi-Supervised Learning},
  author    = {Gu, Jiawei and Wang, Yidi and Sun, Qingqiang and Li, Xinming and Luo, Xiao and Qiao, Ziyue},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {38},
  year      = {2025}
}
```

## Contact

gjwcs@outlook.com

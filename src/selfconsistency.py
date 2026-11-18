"""Self-consistency selection without anomalous data.

Leave-one-view-out (LOVO): every view of every shot is scored against a memory built from all *other*
views (and shots); the LOVO image scores estimate the normal-score distribution for free.
  (i)  rotation gate: arbitrary-angle rotations are admitted to the memory only if they "look normal",
       i.e. the median score of the rotated images (against the view memory) does not exceed the
       `gate_quantile` (0.95) quantile of the LOVO scores of the unrotated views.
       NB: the literal criterion "adding rotations lowers LOVO scores" is always satisfied for a nearest-
       neighbour memory (adding memory items can never increase a min distance), so the quantile form is used.
  (ii) tau, d and aggregation quantile are selected by minimising the dispersion of the LOVO image
       scores, measured scale-free as the coefficient of variation std/mean (raw std would reward collapse).
  (iii) an operating threshold at a chosen false-positive rate = (1 - fpr) quantile of LOVO scores.
"""
from __future__ import annotations

import torch

from .score import knn_dist


@torch.no_grad()
def lovo_image_scores(Z: torch.Tensor, row_view: torch.Tensor, n: int, n_views: int, fracs: list[float],
                      k: int = 1) -> dict[float, torch.Tensor]:
    """Z: [Nmax, d] projected (or raw) view table of ONE category; row_view: [Nmax]; n valid rows.
    Returns {frac: [n_views] LOVO image scores}."""
    z, rv = Z[:n], row_view[:n]
    s = knn_dist(z, z, k=k, exclude=(rv, rv))
    out = {}
    for f in fracs:
        vals = []
        for v in range(n_views):
            sv = s[rv == v]
            if sv.numel() == 0:
                continue
            kk = max(1, int(sv.numel() * f))
            vals.append(sv.topk(kk).values.mean())
        out[f] = torch.stack(vals)
    return out


@torch.no_grad()
def rotation_image_scores(Zr: torch.Tensor, r_img: torch.Tensor, nr: int, Z: torch.Tensor, n: int,
                          frac: float, k: int = 1) -> torch.Tensor:
    s = knn_dist(Zr[:nr], Z[:n], k=k)
    ri = r_img[:nr]
    vals = []
    for i in torch.unique(ri):
        si = s[ri == i]
        vals.append(si.topk(max(1, int(si.numel() * frac))).values.mean())
    return torch.stack(vals)


def gate(lovo: torch.Tensor, rot: torch.Tensor, q: float) -> tuple[bool, dict]:
    thr = torch.quantile(lovo.float(), q)
    med = rot.float().median()
    return bool(med <= thr), {"rot_median": float(med), "lovo_q": float(thr)}


def dispersion(x: torch.Tensor) -> float:
    x = x.float()
    return float(x.std() / x.mean().clamp(min=1e-12))


def threshold_at_fpr(lovo: torch.Tensor, fpr: float) -> float:
    return float(torch.quantile(lovo.float(), 1 - fpr))

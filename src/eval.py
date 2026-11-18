"""Metrics. Image: AUROC, AP. Pixel (on GPU, exact, from sorted scores): AUROC, AP, F1-max,
AU-PRO integrated up to an FPR limit (0.3 standard; 0.05 for MVTec AD 2) and normalised by the limit.
All maps and masks are evaluated at eval_res x eval_res (identical for every method)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


def gaussian_kernel(sigma: float, device) -> torch.Tensor:
    r = int(math.ceil(4 * sigma))   # scipy.ndimage.gaussian_filter default truncate=4.0
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def patch_to_map(s: torch.Tensor, grid: tuple, res: int, sigma: float) -> torch.Tensor:
    """s: [T, gh*gw] patch scores -> [T, res, res] bilinear upsampled + Gaussian smoothed (separable, reflect)."""
    T = s.shape[0]
    m = F.interpolate(s.float().view(T, 1, *grid), size=(res, res), mode="bilinear", align_corners=False)
    if sigma > 0:
        k = gaussian_kernel(sigma, s.device)
        r = (len(k) - 1) // 2
        m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        m = F.conv2d(F.pad(m, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return m[:, 0]


def image_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    if labels.min() == labels.max():
        return {"img_auroc": float("nan"), "img_ap": float("nan")}
    return {"img_auroc": 100 * roc_auc_score(labels, scores), "img_ap": 100 * average_precision_score(labels, scores)}


def _sorted_curves(s: torch.Tensor, y: torch.Tensor):
    order = torch.argsort(s, descending=True)
    s, y = s[order], y[order]
    # distinct-threshold group ends
    last = torch.ones_like(s, dtype=torch.bool)
    last[:-1] = s[1:] != s[:-1]
    return order, s, y, last


def region_labels(masks: torch.Tensor):
    """Connected components (8-connectivity) of every GT mask -> (flat label tensor on GPU, sizes, n_regions)."""
    m_np = masks.cpu().numpy().astype(bool)
    lab = np.zeros(m_np.shape, dtype=np.int64)
    nxt, sizes = 0, []
    st = np.ones((3, 3), dtype=bool)
    for i in range(m_np.shape[0]):
        if m_np[i].any():
            l, n = ndimage.label(m_np[i], structure=st)
            l[l > 0] += nxt
            lab[i] = l
            sizes += list(np.bincount(l.reshape(-1))[nxt + 1:nxt + n + 1])
            nxt += n
    return torch.from_numpy(lab.reshape(-1)).to(masks.device), \
        torch.tensor([0] + sizes, device=masks.device, dtype=torch.float64), nxt


@torch.no_grad()
def pixel_metrics(maps: torch.Tensor, masks: torch.Tensor, pro_limits=(0.3,), regions=None) -> dict:
    """maps: [T,H,W] float (GPU); masks: [T,H,W] {0,1}. Exact threshold sweep over all distinct scores."""
    y = masks.reshape(-1).bool()
    s = maps.reshape(-1).float()
    P, Nn = int(y.sum()), int((~y).sum())
    out = {}
    if P == 0:
        return {"px_auroc": float("nan"), "px_ap": float("nan"), "px_f1max": float("nan"),
                **{f"aupro_{int(l * 100)}": float("nan") for l in pro_limits}}
    order, ss, ys, last = _sorted_curves(s, y)
    tp = torch.cumsum(ys.double(), 0)[last]
    fp = torch.cumsum((~ys).double(), 0)[last]
    tpr, fpr = tp / P, fp / Nn
    tpr = torch.cat([tpr.new_zeros(1), tpr])
    fpr = torch.cat([fpr.new_zeros(1), fpr])
    out["px_auroc"] = 100 * float(torch.trapz(tpr, fpr))
    prec = tp / (tp + fp)
    rec = tp / P
    rec_prev = torch.cat([rec.new_zeros(1), rec[:-1]])
    out["px_ap"] = 100 * float(((rec - rec_prev) * prec).sum())
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
    out["px_f1max"] = 100 * float(f1.max())
    # AU-PRO: per-region overlap; regions = connected components (8-connectivity) of every GT mask
    lab_all, sizes_t, nxt = regions if regions is not None else region_labels(masks)
    lab_t = lab_all[order]
    w = torch.where(lab_t > 0, 1.0 / (nxt * sizes_t[lab_t]), torch.zeros((), dtype=torch.float64, device=maps.device))
    pro = torch.cumsum(w, 0)[last]
    pro = torch.cat([pro.new_zeros(1), pro])
    for lim in pro_limits:
        keep = fpr <= lim
        f, p = fpr[keep], pro[keep]
        j = int(keep.sum())
        if j < len(fpr):  # interpolate the curve at the limit
            f0, f1_, p0, p1 = fpr[j - 1], fpr[j], pro[j - 1], pro[j]
            pl = p0 + (p1 - p0) * (lim - f0) / (f1_ - f0).clamp(min=1e-12)
            f = torch.cat([f, f.new_tensor([lim])])
            p = torch.cat([p, pl.view(1)])
        out[f"aupro_{int(round(lim * 100))}"] = 100 * float(torch.trapz(p, f)) / lim
    return out


def top_frac_mean(s: torch.Tensor, frac: float) -> torch.Tensor:
    """s: [T, N] patch scores -> mean of top max(1, int(N*frac)) scores (AnomalyDINO mean_top1p)."""
    k = max(1, int(s.shape[-1] * frac))
    return s.topk(k, dim=-1).values.mean(-1)

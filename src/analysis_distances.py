"""Nearest-neighbour distance distributions of normal and anomalous test patches, raw tokens vs contrastive head.
Same setting as src/analysis_mechanism.py: DINOv2 ViT-B/14 blocks 9-12,
16-view memory, head trained per draw with fixed defaults. Distances are standardised per category with the mean and std of
that category's normal patches, so categories can be pooled. Normal patches are subsampled to 5,000 per category and draw.
Test labels are used only to measure. Writes results/distances.npz.

usage: python -m src.analysis_distances --datasets mvtec_ad visa --draws 5-9 --k 1
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .bank import load_bank, load_test
from .common import load_config, setup_torch
from .run_ours import parse_range, proj
from .score import knn_dist
from .train import TrainSpec, train_heads_chunked


def collect(cfg, ds, k, draw, dev, out):
    bank = load_bank(cfg, "dinov2_vitb14", "b9_12", ds, k, draw, dev)
    h = train_heads_chunked(bank, TrainSpec(taus=[cfg["method"]["tau"]], d=cfg["method"]["d"], seed=0), cfg)
    gen = torch.Generator(device=dev).manual_seed(draw)
    torch.set_grad_enabled(False)
    for c, cat in enumerate(bank.cats):
        n = int(bank.n[c])
        X, labels, masks, entries, grid = load_test(cfg, "dinov2_vitb14", "b9_12", ds, cat, dev)
        T, N, D = X.shape
        cov = F.avg_pool2d(masks.float()[:, None], masks.shape[-1] // grid[0])[:, 0].reshape(T, -1).flatten()
        anom, norm = cov >= 0.5, cov == 0
        if not anom.any():
            continue
        mem = bank.X[c, :n]
        d_raw = knn_dist(X.reshape(-1, D), mem, 1)
        d_head = knn_dist(proj(h, c, X.reshape(-1, D), bank.mu[c])[0], proj(h, c, mem, bank.mu[c])[0], 1)
        n_idx = norm.nonzero().squeeze(1)
        n_idx = n_idx[torch.randperm(len(n_idx), generator=gen, device=dev)[:5000]]
        for name, d in (("raw", d_raw), ("head", d_head)):
            mu, sd = d[norm].mean(), d[norm].std().clamp(min=1e-8)
            z = (d - mu) / sd
            out[f"{name}_norm"].append(z[n_idx].cpu().numpy().astype(np.float32))
            out[f"{name}_anom"].append(z[anom].cpu().numpy().astype(np.float32))
        out["meta"].append(f"{ds}|{cat}|{draw}")
    torch.set_grad_enabled(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mvtec_ad", "visa"])
    ap.add_argument("--draws", nargs="+", default=["5-9"])
    ap.add_argument("--k", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config()
    dev = setup_torch()
    out = {key: [] for key in ("raw_norm", "raw_anom", "head_norm", "head_anom", "meta")}
    for ds in a.datasets:
        for d in parse_range(a.draws):
            collect(cfg, ds, a.k, d, dev, out)
            print(ds, d, len(out["meta"]), flush=True)
    np.savez_compressed("results/distances.npz", **{key: np.concatenate(v) for key, v in out.items() if key != "meta"},
                        meta=np.array(out["meta"]))


if __name__ == "__main__":
    main()

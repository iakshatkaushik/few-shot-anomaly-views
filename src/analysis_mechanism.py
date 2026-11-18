"""Mechanism analysis of the patch-contrastive head: what does the projection change?

For one shot draw (k shots) and every category, compare raw DINOv2 ViT-B/14 tokens (blocks 9-12) with the trained MS-MVPC
head (fixed defaults tau=0.1, d=128, seed 0; trained on that draw only), both scored against the same 16-view memory:
  * patch separation: AUROC between anomalous patches (GT coverage >= 50 %) and normal patches (coverage 0), and the
    standardised gap (mean d_anom - mean d_norm) / std d_norm of the nearest-neighbour distance d
  * what is kept: Spearman correlation between raw and head distances (anomalous / normal patches separately)
  * invariance: cosine similarity of a memory token with its correspondence in another view of the same shot (positive)
    vs a random token of the same shot (negative); the head is trained to raise the first and lower the second
  * local vs global evidence: image AUROC with top-1 % aggregation (local) and mean aggregation (global), raw vs head;
    MVTec LOCO additionally split into logical and structural anomalies
Uses test labels only to *measure*; nothing here is used for training or selection. Writes results/mechanism.jsonl.

usage: python -m src.analysis_mechanism --dataset mvtec_ad --draws 5-9 --k 1
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .bank import load_bank, load_test
from .common import ResultWriter, load_config, setup_torch
from .eval import top_frac_mean
from .run_ours import parse_range, proj
from .score import knn_dist
from .train import TrainSpec, train_heads_chunked


def auroc(y, s):
    y = np.asarray(y)
    return float(100 * roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


@torch.no_grad()
def invariance(Z: torch.Tensor, bank, c: int, n: int, gen, m: int = 4096):
    """mean cosine to the corresponding token in another view vs to a random token of the same shot"""
    rv, rc = bank.row_view[c, :n], bank.row_canon[c, :n]
    a = torch.randint(0, n, (m,), generator=gen, device=Z.device)
    va, ca = rv[a], rc[a]
    shot = va // bank.V
    vb = shot * bank.V + torch.randint(0, bank.V, (m,), generator=gen, device=Z.device)
    pb = bank.L[c, vb, ca]
    ok = (pb >= 0) & (vb != va)
    a, pb = a[ok], pb[ok]
    rnd = torch.randint(0, n, (len(a),), generator=gen, device=Z.device)
    same_shot = (rv[rnd] // bank.V) == (rv[a] // bank.V)
    za = F.normalize(Z[a].float(), dim=-1)
    pos = (za * F.normalize(Z[pb].float(), dim=-1)).sum(-1).mean()
    neg = (za[same_shot] * F.normalize(Z[rnd[same_shot]].float(), dim=-1)).sum(-1).mean()
    return float(pos), float(neg)


def run(cfg, ds, k, draw, dev, writer):
    t0 = time.perf_counter()
    bank = load_bank(cfg, "dinov2_vitb14", "b9_12", ds, k, draw, dev)
    h = train_heads_chunked(bank, TrainSpec(taus=[cfg["method"]["tau"]], d=cfg["method"]["d"], seed=0), cfg)
    gen = torch.Generator(device=dev).manual_seed(draw)
    rows = []
    for c, cat in enumerate(bank.cats):
        n = int(bank.n[c])
        X, labels, masks, entries, grid = load_test(cfg, "dinov2_vitb14", "b9_12", ds, cat, dev)
        T, N, D = X.shape
        cov = F.avg_pool2d(masks.float()[:, None], masks.shape[-1] // grid[0])[:, 0].reshape(T, -1)    # [T, N]
        anom, norm = (cov >= 0.5).flatten(), (cov == 0).flatten()
        mem_raw = bank.X[c, :n]
        Zv = proj(h, c, mem_raw, bank.mu[c])[0]
        Zt = proj(h, c, X.reshape(-1, D), bank.mu[c])[0]
        d_raw = knn_dist(X.reshape(-1, D), mem_raw, 1)
        d_head = knn_dist(Zt, Zv, 1)
        out = dict(dataset=ds, category=cat, k=k, draw=draw, n_anom_patches=int(anom.sum()))
        lab = labels.cpu().numpy()
        subsets = np.array([e.get("subset", "") for e in entries])
        for name, d, Z in (("raw", d_raw, mem_raw), ("head", d_head, Zv)):
            dn, da = d[norm].float(), d[anom].float()
            if anom.any():
                y = torch.cat([torch.zeros_like(dn), torch.ones_like(da)]).cpu().numpy()
                out[f"{name}_patch_auroc"] = auroc(y, torch.cat([dn, da]).cpu().numpy())
                out[f"{name}_gap"] = float((da.mean() - dn.mean()) / dn.std().clamp(min=1e-8))
            out[f"{name}_inv_pos"], out[f"{name}_inv_neg"] = invariance(Z, bank, c, n, gen)
            p = d.view(T, N)
            for agg, s in (("top1", top_frac_mean(p, 0.01)), ("mean", p.float().mean(1))):
                s = s.cpu().numpy()
                out[f"{name}_img_auroc_{agg}"] = auroc(lab, s)
                if ds == "mvtec_loco":
                    for sub in ("logical", "structural"):
                        keep = np.isin(subsets, [sub, "good"])
                        out[f"{name}_{sub}_img_auroc_{agg}"] = auroc(lab[keep], s[keep])
        if anom.any():
            a_idx = anom.nonzero().squeeze(1)
            n_idx = norm.nonzero().squeeze(1)
            n_idx = n_idx[torch.randperm(len(n_idx), generator=gen, device=dev)[:20000]]
            out["spearman_anom"] = float(spearmanr(d_raw[a_idx].cpu().numpy(), d_head[a_idx].cpu().numpy())[0])
            out["spearman_norm"] = float(spearmanr(d_raw[n_idx].cpu().numpy(), d_head[n_idx].cpu().numpy())[0])
        rows.append(out)
    writer.write(rows)
    keys = ["raw_patch_auroc", "head_patch_auroc", "raw_gap", "head_gap", "raw_inv_pos", "head_inv_pos"]
    print(json.dumps({"dataset": ds, "draw": draw, "s": round(time.perf_counter() - t0, 1),
                      **{kk: round(float(np.nanmean([r.get(kk, np.nan) for r in rows])), 3) for kk in keys}}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--draws", nargs="+", default=["5-9"])
    ap.add_argument("--k", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config()
    dev = setup_torch()
    w = ResultWriter("mechanism", cfg)
    for d in parse_range(a.draws):
        run(cfg, a.dataset, a.k, d, dev, w)


if __name__ == "__main__":
    main()

"""AnomalyDINO (Damm et al., WACV 2025) re-implemented from the official code (github.com/dammsi/AnomalyDINO,
src/detection.py, src/backbones.py, src/utils.py; read 8 Nov 2026), run on our identical shot draws.

Faithful to the official defaults ('agnostic' preprocessing): DINOv2 ViT-S/14, shorter edge resized to 448
(bicubic), crop to a multiple of 14, last block with final LayerNorm; reference images rotated by
0,45,...,315 degrees (cv2.warpAffine, reflect-101) and not masked; test-time PCA background mask
(first PC > 10, centre-coverage sign check, 3x3 dilation + closing) for the categories listed in the official
masking table (MVTec: capsule, hazelnut, pill, screw, toothbrush; VisA: all; other datasets: none);
1-NN cosine distance; image score = mean of the top 1% patch distances.
Deviations: timm weights/pos-embed interpolation instead of torch.hub; bf16 autocast instead of fp32;
anomaly maps evaluated with the common protocol (bilinear to 448x448 + Gaussian sigma 4) like every method.

usage: python -m src.baselines.anomalydino --dataset mvtec_ad
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..common import ResultWriter, append_log, load_config, setup_torch
from ..draws import all_draws
from ..encoders import Encoder
from ..eval import patch_to_map, top_frac_mean
from ..featstore import load_tensors, store_dir
from ..score import knn_dist
from .common import MaskSet, iter_aug, rotate_cv2

MASKING = {"mvtec_ad": {"capsule", "hazelnut", "pill", "screw", "toothbrush"},
           "visa": {"candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2",
                    "pcb3", "pcb4", "pipe_fryum"}}


def pca_mask(tok: torch.Tensor, grid: tuple, threshold: float = 10.0, border: float = 0.2) -> np.ndarray:
    """tok: [T, N, D] raw (LayerNorm-ed, not L2) tokens -> bool masks [T, N] (True = foreground)."""
    x = tok.float()
    xc = x - x.mean(1, keepdim=True)
    cov = xc.transpose(1, 2) @ xc
    _, V = torch.linalg.eigh(cov)
    v = V[:, :, -1]                                                       # first principal axis
    # sklearn >= 1.5 svd_flip(u_based_decision=False): largest-|.| entry of each component is positive
    sgn = torch.sign(v.gather(1, v.abs().argmax(1, keepdim=True)))
    pc = (xc @ (v * sgn)[:, :, None]).squeeze(-1).cpu().numpy()          # [T, N]
    gh, gw = grid
    out = []
    k = np.ones((3, 3), np.uint8)
    for p in pc:
        m = p > threshold
        m2 = m.reshape(gh, gw)[int(gh * border):int(gh * (1 - border)), int(gw * border):int(gw * (1 - border))]
        if m2.sum() <= m2.size * 0.35:
            m = -p > threshold
        m = cv2.dilate(m.reshape(gh, gw).astype(np.uint8), k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k).astype(bool)
        out.append(m.reshape(-1))
    return np.stack(out)


def run(ds: str, cfg: dict, dev, ks, draws_sel, encoder=None, test_kind="test_ad", name="AnomalyDINO",
        out="baselines"):
    bc = dict(cfg["baselines"]["anomalydino"])
    if encoder:
        bc["encoder"] = encoder
        bc["smaller_edge"] = cfg["encoders"][encoder]["img_size"]
    enc = Encoder(bc["encoder"], cfg, dev, compile=False, batch_size=256, sets=["last_raw"])
    draws = all_draws(ds, cfg)
    w = ResultWriter(out, cfg)
    for cat in cfg["datasets"][ds]["categories"]:
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        te = load_tensors(store_dir(cfg, bc["encoder"], ds, cat, test_kind), ["last_raw"], device=dev)["last_raw"]
        man = json.load(open(store_dir(cfg, bc["encoder"], ds, cat, test_kind) / "manifest.json"))
        grid = tuple(man["grid"])
        ms = MaskSet(cfg, ds, cat, dev)
        assert ms.paths == [e["path"] for e in man["entries"]]
        T, N, D = te.shape
        fg = pca_mask(te, grid) if cat in MASKING.get(ds, set()) else np.ones((T, N), bool)
        fg_t = torch.from_numpy(fg).to(dev)
        q = F.normalize(te.float(), dim=-1).half().reshape(-1, D)
        # reference features: every rotation of every shot used by any selected draw, encoded once
        jobs = [(d, k) for k in ks for d in draws_sel]
        paths = sorted({r["path"] for d, k in jobs for r in draws[(cat, d, k)]})
        items = [(p, {"angle": a}) for p in paths for a in bc["rotations"]]
        ref = {}
        t_ref = time.perf_counter()
        for idx, x in iter_aug(items, lambda im, s: rotate_cv2(im, s["angle"]), cfg, bc["smaller_edge"], dev,
                               "shorter_edge", 14, 256):
            o = enc(x)["last_raw"]
            for b, i in enumerate(idx):
                ref[(items[i][0], items[i][1]["angle"])] = F.normalize(o[b].float(), dim=-1).half()
        t_ref = time.perf_counter() - t_ref
        rows = []
        for d, k in jobs:
            ts = time.perf_counter()
            mem = torch.cat([ref[(r["path"], a)] for r in draws[(cat, d, k)] for a in bc["rotations"]])
            s = knn_dist(q, mem, 1).view(T, N) * fg_t                      # background patches -> 0
            img = top_frac_mean(s, 0.01).cpu().numpy()
            maps = patch_to_map(s, grid, cfg["method"]["eval_res"], cfg["method"]["map_sigma"])
            m = ms.evaluate(maps, img)
            rows.append(dict(method=name, resolution=bc["smaller_edge"], dataset=ds, category=cat, k=k, draw=d, encoder=bc["encoder"],
                             masking=cat in MASKING.get(ds, set()), rotation=True, n_mem=int(mem.shape[0]),
                             score_s=round(time.perf_counter() - ts, 3), ref_encode_s=round(t_ref / len(jobs), 3),
                             peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), seed=0, **m))
        w.write(rows)
        rec = {"baseline": name, "dataset": ds, "category": cat, "runs": len(rows),
               "wall_s": round(time.perf_counter() - t0, 1),
               "img_auroc_mean": round(float(np.mean([r["img_auroc"] for r in rows])), 2)}
        print(json.dumps(rec), flush=True)
        append_log("baselines", rec, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--draws", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--test_kind", default="test_ad")
    ap.add_argument("--name", default="AnomalyDINO")
    ap.add_argument("--out", default="baselines")
    a = ap.parse_args()
    cfg = load_config()
    dev = setup_torch()
    run(a.dataset, cfg, dev, a.ks, a.draws, a.encoder, a.test_kind, a.name, a.out)


if __name__ == "__main__":
    main()

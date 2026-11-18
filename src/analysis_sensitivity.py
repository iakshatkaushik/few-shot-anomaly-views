"""Sensitivity of the view memory: number of correspondence-preserving views per shot
V' in {1, 2, 4, 8, 16} (view 0 = the shot itself) and number of arbitrary-angle rotations per shot R' in {0, 10, 30},
for raw DINOv2 ViT-B/14 tokens (blocks 9-12) with 1-NN cosine scoring (control C1). Reports image AUROC,
AU-PRO@30 %, memory size (tokens, MB fp16) and the nearest-neighbour time per test image. Analysis only, nothing here feeds
back into the methods. Writes results/sensitivity.jsonl.

usage: python -m src.analysis_sensitivity --dataset mvtec_ad --ks 1 4 --draws 5-9
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .bank import load_bank
from .common import ResultWriter, load_config, setup_torch
from .run_ours import CatEval, parse_range
from .score import knn_dist

VIEWS = (1, 2, 4, 8, 16)
ROTS = (0, 10, 30)


@torch.no_grad()
def run(cfg, ds, k, draw, dev, writer):
    t0 = time.perf_counter()
    bank = load_bank(cfg, "dinov2_vitb14", "b9_12", ds, k, draw, dev, with_synth=False)
    q0 = cfg["method"]["top_frac"]
    rows = []
    for c, cat in enumerate(bank.cats):
        ce = CatEval(cfg, "dinov2_vitb14", "b9_12", ds, cat, dev)
        n, nr = int(bank.n[c]), int(bank.r_n[c])
        X, rv = bank.X[c, :n], bank.row_view[c, :n]
        R, ri = bank.Rt[c, :nr], bank.r_img[c, :nr]
        n_rot = cfg["views"]["n_rotations"]
        for v in VIEWS:
            for r in ROTS:
                mem = X[(rv % bank.V) < v]
                if r:
                    mem = torch.cat([mem, R[(ri % n_rot) < r]])
                torch.cuda.synchronize()
                ts = time.perf_counter()
                s = knn_dist(ce.flat, mem, 1)
                torch.cuda.synchronize()
                knn_ms = 1000 * (time.perf_counter() - ts) / ce.T
                m = ce.metrics(s, [q0], False)[q0]
                rows.append(dict(method="sens_raw", dataset=ds, category=cat, k=k, draw=draw, views=v, rotations=r,
                                 mem_tokens=int(mem.shape[0]), mem_mb=round(mem.numel() * 2 / 2**20, 2),
                                 knn_ms_per_image=round(knn_ms, 3), img_auroc=m["img_auroc"], aupro_30=m["aupro_30"],
                                 px_auroc=m["px_auroc"]))
    writer.write(rows)
    summ = {}
    for r in rows:
        summ.setdefault((r["views"], r["rotations"]), []).append(r["img_auroc"])
    print(json.dumps({"dataset": ds, "k": k, "draw": draw, "s": round(time.perf_counter() - t0, 1),
                      **{f"v{v}r{r}": round(float(np.mean(x)), 2) for (v, r), x in summ.items()}}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--draws", nargs="+", default=["5-9"])
    a = ap.parse_args()
    cfg = load_config()
    dev = setup_torch()
    w = ResultWriter("sensitivity", cfg)
    for k in a.ks:
        for d in parse_range(a.draws):
            run(cfg, a.dataset, k, d, dev, w)


if __name__ == "__main__":
    main()

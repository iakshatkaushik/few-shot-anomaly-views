"""Development runs (MVTec AD / VisA, draws 0-4): score-level fusion of training-free nearest-neighbour
scorers built on different frozen backbones, each with its own 16-view memory (+ all rotations).
Calibration uses normal data only: every scorer's patch distances are divided by the mean leave-one-view-out patch
distance of that scorer on the shot's own views (no test images, no labels). Fused map = mean of the calibrated maps at
eval resolution; image score = mean of the top 1% of fused-map pixels.

usage: python -m src.explore_fusion --dataset mvtec_ad --draws 0-4 --scorers B:dinov2_vitb14/b9_12 D3:dinov3_vitb16/b9_12 ...
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .bank import load_bank, load_test
from .common import ResultWriter, load_config, setup_torch
from .eval import image_metrics, pixel_metrics, region_labels
from .run_ours import parse_range
from .score import knn_dist


def scorer_maps(cfg, enc, fset, ds, cat, k, draw, dev, res, rot=True, head=False):
    """head=True: the contrastive head (fixed defaults tau=0.1, d=128), trained on this shot draw only,
    scored with the same projected view memory (rotations enter only through the rotation gate, so not here)."""
    bank = load_bank(cfg, enc, fset, ds, k, draw, dev, with_synth=head, cats=[cat])
    n, nr = int(bank.n[0]), int(bank.r_n[0])
    X = bank.X[0, :n]
    if head:
        from .run_ours import proj
        from .train import TrainSpec, train_heads
        h = train_heads(bank, TrainSpec(taus=[cfg["method"]["tau"]], d=cfg["method"]["d"]), cfg)
        f = lambda t: proj(h, 0, t, bank.mu[0])[0]
        rot = False
    else:
        f = lambda t: t
    with torch.no_grad():
        Z = f(X)
        mem = torch.cat([Z, f(bank.Rt[0, :nr])]) if rot else Z
        rv = bank.row_view[0, :n]
        lovo = knn_dist(Z, Z, 1, exclude=(rv, rv)).mean().clamp(min=1e-6)   # normal-only scale
        Xt, labels, masks, entries, grid = load_test(cfg, enc, fset, ds, cat, dev)
        T, N, D = Xt.shape
        s = knn_dist(f(Xt.reshape(-1, D)), mem, 1).view(T, 1, *grid) / lovo
        m = F.interpolate(s, size=(res, res), mode="bilinear", align_corners=False)[:, 0]
    return m, labels.cpu().numpy(), masks


def gauss(m, sigma=4.0):
    from .eval import gaussian_kernel
    k = gaussian_kernel(sigma, m.device)
    r = (len(k) - 1) // 2
    x = F.conv2d(F.pad(m[:, None], (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--draws", nargs="+", default=["0-4"])
    ap.add_argument("--scorers", nargs="+", required=True, help="NAME:encoder/fset")
    ap.add_argument("--max_combo", type=int, default=4)
    ap.add_argument("--must", nargs="*", default=[], help="only combinations containing all of these scorers")
    a = ap.parse_args()
    cfg = load_config()
    dev = setup_torch()
    w = ResultWriter("explore", cfg)
    res = cfg["method"]["eval_res"]
    sc = [(s.split(":")[0], *s.split(":")[1].split("/")) for s in a.scorers]   # NAME:enc/fset[/head]
    limits = (cfg["eval"]["pro_fpr"][0],)
    for draw in parse_range(a.draws):
        t0 = time.perf_counter()
        rows = []
        for cat in cfg["datasets"][a.dataset]["categories"]:
            maps, lab, masks = {}, None, None
            for name, enc, fset, *opt in sc:
                maps[name], lab, masks = scorer_maps(cfg, enc, fset, a.dataset, cat, a.k, draw, dev, res,
                                                     head=bool(opt and opt[0] == "head"))
            reg = region_labels(masks)
            for r in range(1, min(a.max_combo, len(sc)) + 1):
                for combo in itertools.combinations([s[0] for s in sc], r):
                    if not set(a.must) <= set(combo):
                        continue
                    fused = gauss(torch.stack([maps[c] for c in combo]).mean(0))
                    flat = fused.flatten(1)
                    img = flat.topk(max(1, int(flat.shape[1] * 0.01)), 1).values.mean(1).cpu().numpy()
                    m = dict(image_metrics(img, lab), **pixel_metrics(fused, masks, limits, reg))
                    rows.append(dict(method="xf_" + "+".join(combo), dataset=a.dataset, category=cat, k=a.k,
                                     draw=draw, **m))
            del maps
        w.write(rows)
        summ = {}
        for r in rows:
            summ.setdefault(r["method"], []).append(r["img_auroc"])
        print(json.dumps({"draw": draw, "s": round(time.perf_counter() - t0, 1),
                          **{k_: round(float(np.mean(v)), 2) for k_, v in summ.items()}}), flush=True)


if __name__ == "__main__":
    main()

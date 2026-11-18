"""VisionAD (Wang, Guo, et al., arXiv 2504.11895, 2025) re-implemented from the official code
(github.com/Qiqigeww/VisionAD: dade_*_test.py, models/uad.py (DADE), utils.py (evaluation_batch_vis, color_jitter_batch,
value_limit); read 9 Nov 2026), run on our identical shot draws.

Official settings: DINOv2-reg ViT-L/14, features = mean of raw block outputs 4..18 (0-indexed, no final norm);
support augmentation: shot + rot90 x3 + vertical flip + horizontal flip; query augmentation: three passes whose maps are
summed -- (i) original, (ii) vertical flip of query and supports ('color_jitter_batch' in the released code only flips),
map flipped back, (iii) value_limit = clamp(min=0) of the normalised query and supports; 1-NN cosine distance;
image score = mean of the top 1% of the map.
Deviations: the official transform resizes to 448 and centre-crops 392 (evaluating only the central crop of image and mask);
we resize to 448x448 without crop so that all methods are scored on the full image; maps are evaluated with the common
protocol (bilinear to 448 + Gaussian sigma 4, as in the official evaluation_batch); timm weights; bf16 autocast.

usage: python -m src.baselines.visionad --dataset visa
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F

from ..common import ResultWriter, append_log, load_config, setup_torch
from ..draws import all_draws
from ..eval import patch_to_map, top_frac_mean
from ..imageio import iter_resized, normalize
from ..score import knn_dist
from .common import MaskSet

SIZE, LAYERS = 448, list(range(4, 19))


class VitL:
    def __init__(self, dev):
        self.m = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, img_size=SIZE,
                                   num_classes=0, dynamic_img_size=True).to(dev).eval().requires_grad_(False)

    @torch.no_grad()
    def feats(self, x_norm: torch.Tensor) -> torch.Tensor:
        """x_norm: ImageNet-normalised [B,3,H,W] -> [B, N, 1024] mean of raw blocks 4..18, l2-normalised"""
        out = []
        for s in range(0, x_norm.shape[0], 32):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inter = self.m.forward_intermediates(x_norm[s:s + 32], indices=LAYERS, norm=False, output_fmt="NLC",
                                                     intermediates_only=True, stop_early=True)
            f = inter[0].float()
            for t in inter[1:]:
                f = f + t.float()
            out.append(F.normalize(f / len(inter), dim=-1).half())
        return torch.cat(out)


def variants(x01: torch.Tensor):
    """the three query/support passes of the released code, on [B,3,H,W] in [0,1]"""
    xn = normalize(x01, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return {"ori": xn, "vflip": torch.flip(xn, dims=[-2]), "clamp": xn.clamp(min=0)}


def support_aug(x01: torch.Tensor) -> torch.Tensor:
    return torch.cat([x01] + [torch.rot90(x01, k=i + 1, dims=[-2, -1]) for i in range(3)] +
                     [torch.flip(x01, dims=[-2]), torch.flip(x01, dims=[-1])])


def run(ds, cfg, dev, ks, draws_sel):
    net = VitL(dev)
    draws = all_draws(ds, cfg)
    w = ResultWriter("baselines", cfg)
    g = SIZE // 14
    for cat in cfg["datasets"][ds]["categories"]:
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        ms = MaskSet(cfg, ds, cat, dev)
        q = {k_: [] for k_ in ("ori", "vflip", "clamp")}
        order = []
        for idx, x in iter_resized(ms.paths, cfg, SIZE, dev, 128, mode="square", patch=14):
            for k_, v in variants(x).items():
                q[k_].append(net.feats(v))
            order += idx
        inv = torch.tensor(order, device=dev).argsort()
        q = {k_: torch.cat(v)[inv] for k_, v in q.items()}
        T, N, D = q["ori"].shape
        jobs = [(d, k) for k in ks for d in draws_sel]
        paths = sorted({r["path"] for d, k in jobs for r in draws[(cat, d, k)]})
        sup = {}
        for idx, x in iter_resized(paths, cfg, SIZE, dev, 16, mode="square", patch=14):
            for b, i in enumerate(idx):
                xs = support_aug(x[b:b + 1])
                sup[paths[i]] = {k_: net.feats(v).reshape(-1, D) for k_, v in variants(xs).items()}
        rows = []
        for d, k in jobs:
            ts = time.perf_counter()
            mem = {k_: torch.cat([sup[r["path"]][k_] for r in draws[(cat, d, k)]]) for k_ in q}
            maps = 0
            for k_ in q:
                s = knn_dist(q[k_].reshape(-1, D), mem[k_], 1).view(T, g, g)
                if k_ == "vflip":
                    s = torch.flip(s, dims=[-2])
                maps = maps + s
            maps = maps.reshape(T, -1)
            full = patch_to_map(maps, (g, g), cfg["method"]["eval_res"], cfg["method"]["map_sigma"])
            img = top_frac_mean(full.flatten(1), 0.01).cpu().numpy()
            m = ms.evaluate(full, img)
            rows.append(dict(method="VisionAD", dataset=ds, category=cat, k=k, draw=d, encoder="dinov2_vitl14_reg",
                             score_s=round(time.perf_counter() - ts, 3),
                             peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), seed=0, **m))
        w.write(rows)
        rec = {"baseline": "VisionAD", "dataset": ds, "category": cat, "runs": len(rows),
               "wall_s": round(time.perf_counter() - t0, 1),
               "img_auroc_mean": round(float(np.mean([r["img_auroc"] for r in rows])), 2)}
        print(json.dumps(rec), flush=True)
        append_log("baselines", rec, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--draws", nargs="+", type=int, default=list(range(10)))
    a = ap.parse_args()
    cfg = load_config()
    dev = setup_torch()
    run(a.dataset, cfg, dev, a.ks, a.draws)


if __name__ == "__main__":
    main()

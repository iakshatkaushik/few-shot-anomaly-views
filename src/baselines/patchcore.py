"""PatchCore (Roth et al., CVPR 2022) few-shot, re-implemented following anomalib's implementation
(open-edge-platform/anomalib src/anomalib/models/image/patchcore/{torch_model,lightning_model}.py; read 8 Nov 2026):
WideResNet-50-2 (ImageNet, timm 'wide_resnet50_2.tv_in1k'), layers layer2+layer3, 3x3 average pooling,
layer3 bilinearly upsampled and concatenated (1536-d), input resized to 256x256 (anomalib default, no crop),
greedy k-center coreset (10%) on a sparse random projection (eps=0.9), 1-NN Euclidean patch scores,
image score re-weighted with the 9 nearest memory neighbours (anomalib compute_anomaly_score).
Runs the CNN in channels_last + bf16 autocast.

usage: python -m src.baselines.patchcore --dataset mvtec_ad
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F

from ..common import ResultWriter, append_log, load_config, setup_torch
from ..draws import all_draws
from ..eval import patch_to_map
from ..imageio import iter_resized, normalize
from .common import MaskSet


class WRN:
    def __init__(self, dev):
        self.m = timm.create_model("wide_resnet50_2.tv_in1k", pretrained=True, features_only=True,
                                   out_indices=(2, 3)).to(dev).eval().to(memory_format=torch.channels_last)
        self.pool = torch.nn.AvgPool2d(3, 1, 1)

    @torch.no_grad()
    def __call__(self, x):
        x = normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f2, f3 = self.m(x)
        f2, f3 = self.pool(f2.float()), self.pool(f3.float())
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear")
        e = torch.cat([f2, f3], 1)
        return e.permute(0, 2, 3, 1).reshape(e.shape[0], -1, e.shape[1]), tuple(e.shape[-2:])


def euclid(x, y):
    d = (x * x).sum(-1, keepdim=True) - 2 * x @ y.t() + (y * y).sum(-1)[None]
    return d.clamp_min_(0).sqrt_()


def kcenter(emb: torch.Tensor, ratio: float, gen: torch.Generator) -> torch.Tensor:
    """anomalib KCenterGreedy: SparseRandomProjection(eps=0.9) then greedy farthest-point selection."""
    n, D = emb.shape
    k = max(1, int(n * ratio))
    nc = int(4 * math.log(n) / (0.9 ** 2 / 2 - 0.9 ** 3 / 3))
    nc = min(nc, D)
    dens = 1 / math.sqrt(D)
    mask = torch.rand(D, nc, generator=gen, device=emb.device) < dens
    sign = torch.where(torch.rand(D, nc, generator=gen, device=emb.device) < 0.5, -1.0, 1.0)
    R = mask * sign * math.sqrt(1 / dens) / math.sqrt(nc)
    z = emb @ R
    sel = [int(torch.randint(0, n, (1,), generator=gen, device=emb.device))]
    mind = ((z - z[sel[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        i = int(mind.argmax())
        sel.append(i)
        mind = torch.minimum(mind, ((z - z[i]) ** 2).sum(1))
    return emb[torch.tensor(sel, device=emb.device)]


def scores(emb: torch.Tensor, mem: torch.Tensor, nn_k: int = 9):
    """emb: [T, P, D] -> patch scores [T, P], image scores [T] (anomalib re-weighting)."""
    T, Pn, D = emb.shape
    ps, loc = [], []
    flat = emb.reshape(-1, D)
    for s in range(0, flat.shape[0], 8192):
        d = euclid(flat[s:s + 8192], mem)
        v, i = d.min(1)
        ps.append(v)
        loc.append(i)
    ps, loc = torch.cat(ps).view(T, Pn), torch.cat(loc).view(T, Pn)
    mp = ps.argmax(1)
    ar = torch.arange(T, device=emb.device)
    feat = emb[ar, mp]
    sc = ps[ar, mp]
    nn_s = mem[loc[ar, mp]]
    kk = min(nn_k, mem.shape[0])
    sup = euclid(nn_s, mem).topk(kk, largest=False, dim=1).indices
    dist = torch.linalg.vector_norm(feat[:, None] - mem[sup], dim=-1)
    w = 1 - F.softmax(dist, 1)[:, 0]
    return ps, w * sc


def run(ds, cfg, dev, ks, draws_sel, coreset=None):
    bc = dict(cfg["baselines"]["patchcore"])
    if coreset is not None:
        bc["coreset_ratio"] = coreset
    name = "PatchCore" if bc["coreset_ratio"] == cfg["baselines"]["patchcore"]["coreset_ratio"] else f"PatchCore-cs{bc['coreset_ratio']:g}"
    net = WRN(dev)
    draws = all_draws(ds, cfg)
    from ..datasets import load_index
    rows_all = load_index(ds, cfg)
    w = ResultWriter("baselines", cfg)
    for cat in cfg["datasets"][ds]["categories"]:
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        ms = MaskSet(cfg, ds, cat, dev)
        test_paths = ms.paths
        te, grid = [], None
        for idx, x in iter_resized(test_paths, cfg, bc["resize"], dev, 256, mode="square", patch=1, ):
            e, grid = net(x)
            te.append((idx, e))
        T = len(test_paths)
        emb = torch.empty(T, te[0][1].shape[1], te[0][1].shape[2], device=dev)
        for idx, e in te:
            emb[torch.tensor(idx, device=dev)] = e
        jobs = [(d, k) for k in ks for d in draws_sel]
        paths = sorted({r["path"] for d, k in jobs for r in draws[(cat, d, k)]})
        ref = {}
        for idx, x in iter_resized(paths, cfg, bc["resize"], dev, 256, mode="square", patch=1):
            e, _ = net(x)
            for b, i in enumerate(idx):
                ref[paths[i]] = e[b]
        rows = []
        for d, k in jobs:
            ts = time.perf_counter()
            gen = torch.Generator(device=dev).manual_seed(1000 * d + 10 * k)
            full = torch.cat([ref[r["path"]] for r in draws[(cat, d, k)]])
            mem = full if bc["coreset_ratio"] >= 1.0 else kcenter(full, bc["coreset_ratio"], gen)
            ps, img = scores(emb, mem, bc["num_neighbors"])
            maps = patch_to_map(ps, grid, cfg["method"]["eval_res"], 4.0)
            m = ms.evaluate(maps, img.cpu().numpy())
            rows.append(dict(method=name, dataset=ds, category=cat, k=k, draw=d, backbone=bc["backbone"],
                             n_mem=int(mem.shape[0]), score_s=round(time.perf_counter() - ts, 3),
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
    ap.add_argument("--coreset", type=float, default=None)
    a = ap.parse_args()
    cfg = load_config()
    dev = setup_torch()
    run(a.dataset, cfg, dev, a.ks, a.draws, a.coreset)


if __name__ == "__main__":
    main()

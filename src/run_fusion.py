"""Held-out evaluation of view-memory fusion (VMF), with the settings picked on the development split:
  scorers  : DINOv3 ViT-B/16 @448, DINOv3 ViT-B/16 @672, DINOv2 ViT-G/14-reg @672 (blocks 37-40), optionally the
             contrastive head (ViT-B/14, fixed defaults); all choices made on development data only
  memory   : for every shot of the draw, the 16 correspondence-preserving views + 30 arbitrary rotations (valid tokens)
  score    : 1-NN cosine distance, divided by the mean leave-one-view-out distance of the same scorer (normal-only
             calibration), bilinear to 448, mean over scorers, Gaussian sigma 4, image score = top 1% of the fused map.
Views/rotations are generated and encoded on the fly (caching ViT-G views for every k would need ~1 TB); test-image
features come from the caches. Writes results/fusion.jsonl (method 'VMF' or 'VMF+H').

usage: python -m src.run_fusion --dataset mpdd --ks 1 2 4 --draws 0-9 [--head]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from features.views import load_shot_images, make_view, rotate_batch, view_params, rotation_angles
from .bank import load_bank, load_test
from .common import ResultWriter, load_config, setup_torch
from .draws import all_draws, draw_seed
from .encoders import Encoder
from .eval import image_metrics, pixel_metrics, region_labels
from .explore_fusion import gauss, scorer_maps
from .run_ours import parse_range
from .score import knn_dist

SCORERS = [("D3", "dinov3_vitb16", "b9_12"), ("D6", "dinov3_vitb16_672", "b9_12"), ("G", "dinov2_vitg14_reg", "late4")]


def test_kind(cfg, enc, fset, ds, cat):
    """the ViT-G 'test' cache of MPDD/LOCO/AD2 holds only SubspaceAD's block set; late blocks live in 'test_l4'."""
    from .featstore import read_manifest, store_dir
    m = read_manifest(store_dir(cfg, enc, ds, cat, "test"))
    return "test" if m is not None and fset in m["sets"] else "test_l4"


@torch.no_grad()
def memory_on_the_fly(enc: Encoder, fset: str, imgs, shots, seed, cfg):
    """16 views + 30 rotations of every shot -> (memory tokens, view id per token) for the views part, rotation tokens."""
    g = enc.size // enc.patch
    vx, vid, rx, valid = [], [], [], []
    for si, r in enumerate(shots):
        x = imgs[r["path"]]
        for vi, p in enumerate(view_params(seed, si, cfg)):
            vx.append(make_view(x, p, enc.patch, cfg))
        ang = rotation_angles(seed, si, cfg)
        xr, vr = rotate_batch(x.unsqueeze(0).expand(len(ang), -1, -1, -1), ang, enc.patch, cfg["views"]["rotation_valid_frac"])
        rx.append(xr)
        valid.append(vr)
    from features.views import index_map
    idx = torch.stack([index_map(g, p, cfg, device=enc.device) for si in range(len(shots))
                       for p in view_params(seed, si, cfg)])
    Fv = enc(torch.stack(vx))[fset]
    keep = idx >= 0
    vi_, ti_ = keep.nonzero(as_tuple=True)
    X, rv = Fv[vi_, ti_], vi_
    Fr = enc(torch.cat(rx))[fset]
    R = Fr[torch.cat(valid)]
    return X, rv, R


def run(cfg, ds, ks, draws_sel, dev, use_head, writer):
    res = cfg["method"]["eval_res"]
    limits = tuple(cfg["eval"]["pro_fpr"]) if ds == "mvtec_ad2" else (cfg["eval"]["pro_fpr"][0],)
    encs = {n: Encoder(e, cfg, dev, compile=False, batch_size=64, sets=[f]) for n, e, f in SCORERS}
    draws = all_draws(ds, cfg)
    name = "VMF+H" if use_head else "VMF"
    for cat in cfg["datasets"][ds]["categories"]:
        t0 = time.perf_counter()
        tests = {n: load_test(cfg, e, f, ds, cat, dev, kind=test_kind(cfg, e, f, ds, cat)) for n, e, f in SCORERS}
        labels, masks = tests["D3"][1].cpu().numpy(), tests["D3"][2]
        reg = region_labels(masks)
        subsets = [e_.get("subset", "") for e_ in tests["D3"][3]]
        imgs = {n: load_shot_images(ds, cat, draws, encs[n], cfg) for n, _, _ in SCORERS}
        rows = []
        for k in ks:
            for d in draws_sel:
                shots = draws[(cat, d, k)]
                maps = []
                for n, e, f in SCORERS:
                    X, rv, R = memory_on_the_fly(encs[n], f, imgs[n], shots, draw_seed(d, k), cfg)
                    lovo = knn_dist(X, X, 1, exclude=(rv, rv)).mean().clamp(min=1e-6)
                    Xt, _, _, _, grid = tests[n]
                    T, N, D = Xt.shape
                    s = knn_dist(Xt.reshape(-1, D), torch.cat([X, R]), 1).view(T, 1, *grid) / lovo
                    maps.append(F.interpolate(s, size=(res, res), mode="bilinear", align_corners=False)[:, 0])
                if use_head:
                    m_h, _, _ = scorer_maps(cfg, "dinov2_vitb14", "b9_12", ds, cat, k, d, dev, res, head=True)
                    maps.append(m_h)
                fused = gauss(torch.stack(maps).mean(0))
                flat = fused.flatten(1)
                img = flat.topk(max(1, int(flat.shape[1] * 0.01)), 1).values.mean(1).cpu().numpy()
                m = dict(image_metrics(img, labels), **pixel_metrics(fused, masks, limits, reg))
                if ds == "mvtec_loco":
                    for sub in ("logical", "structural"):
                        keep = np.array([s_ in (sub, "good") for s_ in subsets])
                        m[f"{sub}_img_auroc"] = image_metrics(img[keep], labels[keep])["img_auroc"]
                        km = torch.tensor(keep, device=fused.device)
                        m.update({f"{sub}_{kk}": vv for kk, vv in pixel_metrics(fused[km], masks[km], limits).items()})
                rows.append(dict(method=name, dataset=ds, category=cat, k=k, draw=d, seed=0,
                                 heldout=not (ds in ("mvtec_ad", "visa") and k == 1 and d < 5), **m))
        writer.write(rows)
        print(json.dumps({"method": name, "dataset": ds, "category": cat, "runs": len(rows),
                          "wall_s": round(time.perf_counter() - t0, 1),
                          "img_auroc": round(float(np.mean([r["img_auroc"] for r in rows])), 2)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--draws", nargs="+", default=["0-9"])
    ap.add_argument("--head", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    dev = setup_torch()
    run(cfg, a.dataset, a.ks, parse_range(a.draws), dev, a.head, ResultWriter("fusion", cfg))


if __name__ == "__main__":
    main()

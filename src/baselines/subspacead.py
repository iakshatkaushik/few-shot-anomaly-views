"""SubspaceAD (Lendering, Akdag, Bondarev, CVPR 2026) re-implemented from the official code
(github.com/CLendering/SubspaceAD: main.py, scripts/benchmark_few_shot.sh, src/subspacead/core/{extractor,pca}.py,
post_process/scoring.py; read 8 Nov 2026), run on our identical shot draws.

Official few-shot settings: DINOv2-G with registers, 672x672 (no crop), hidden states -12..-18 (outputs of
blocks 23..29) mean-aggregated without final norm; each shot plus aug_count=30 augmentations with
T.RandomRotation(degrees=(0, 345)) (nearest interpolation, zero fill, applied at original resolution);
no augmentation for 'transistor'; PCA with 99% explained variance (GPU, float64 covariance);
score = squared reconstruction residual; map: resize to 672 (bilinear) + cv2.GaussianBlur(3x3, sigma 4);
image score = mean of the top 1% of map values (mtop1p). No background mask (bg_mask_method default None).
Deviations: timm weights instead of HF transformers; GPU bicubic resize instead of the HF processor;
rotation angles drawn from our per-draw seed instead of torch.manual_seed(42); bf16 autocast.

usage: python -m src.baselines.subspacead --dataset mvtec_ad
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
from ..draws import all_draws, draw_seed
from ..encoders import Encoder
from ..featstore import load_tensors, store_dir
from ..score import PCAResidual
from .common import MaskSet, iter_aug

ENC = "dinov2_vitg14_reg"


def rot_nearest(im: np.ndarray, spec: dict) -> np.ndarray:
    a = spec["angle"]
    if a is None:
        return im
    c = (im.shape[1] / 2, im.shape[0] / 2)
    M = cv2.getRotationMatrix2D(c, a, 1.0)
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def post_process(patch: torch.Tensor, grid, res: int = 672) -> torch.Tensor:
    """[T, gh*gw] -> [T, res, res]: bilinear resize + 3x3 Gaussian (sigma 4), reflect-101 border (cv2 defaults)."""
    T = patch.shape[0]
    m = F.interpolate(patch.float().view(T, 1, *grid), size=(res, res), mode="bilinear", align_corners=False)
    k = torch.tensor(cv2.getGaussianKernel(3, 4.0)[:, 0], device=patch.device, dtype=torch.float32)
    m = F.conv2d(F.pad(m, (1, 1, 0, 0), mode="reflect"), k.view(1, 1, 1, 3))
    m = F.conv2d(F.pad(m, (0, 0, 1, 1), mode="reflect"), k.view(1, 1, 3, 1))
    return m[:, 0]


def run(ds, cfg, dev, ks, draws_sel):
    bc = cfg["baselines"]["subspacead"]
    # eager bf16, batch 64: the max-autotune CUDA-graph pools of ViT-G (~46 GB) left no room for PCA (OOM on 8 Nov)
    enc = Encoder(ENC, cfg, dev, compile=False, batch_size=64, sets=[bc["set"]])
    draws = all_draws(ds, cfg)
    w = ResultWriter("baselines", cfg)
    for cat in cfg["datasets"][ds]["categories"]:
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        sd = store_dir(cfg, ENC, ds, cat, "test")
        te = load_tensors(sd, [bc["set"]], device=dev)[bc["set"]]
        man = json.load(open(sd / "manifest.json"))
        grid = tuple(man["grid"])
        ms = MaskSet(cfg, ds, cat, dev)
        assert ms.paths == [e["path"] for e in man["entries"]]
        T, N, D = te.shape
        aug = 0 if cat in bc["no_aug_categories"] else bc["aug_count"]
        jobs = [(d, k) for k in ks for d in draws_sel]
        items, owner = [], []
        for d, k in jobs:
            rng = np.random.default_rng([draw_seed(d, k), 42])
            for si, r in enumerate(draws[(cat, d, k)]):
                items.append((r["path"], {"angle": None}))
                owner.append((d, k))
                for a in rng.uniform(0, bc["rot_max"], size=aug):
                    items.append((r["path"], {"angle": float(a)}))
                    owner.append((d, k))
        feats = {j: [] for j in jobs}
        t_enc = time.perf_counter()
        for idx, x in iter_aug(items, rot_nearest, cfg, enc.size, dev, "square", enc.patch, enc.batch_size):
            o = enc(x)[bc["set"]]
            for b, i in enumerate(idx):
                feats[owner[i]].append(o[b])
        t_enc = time.perf_counter() - t_enc
        rows = []
        for d, k in jobs:
            ts = time.perf_counter()
            tr = torch.cat(feats[(d, k)]).float()
            pca = PCAResidual(tr, bc["pca_ev"])
            s = pca(te.reshape(-1, D)).view(T, N)
            maps = post_process(s, grid)
            flat = maps.flatten(1)
            img = flat.topk(max(1, int(flat.shape[1] * 0.01)), dim=1).values.mean(1).cpu().numpy()
            m = ms.evaluate(maps, img)
            rows.append(dict(method="SubspaceAD", dataset=ds, category=cat, k=k, draw=d, encoder=ENC,
                             aug_count=aug, pca_k=pca.k, n_train_tokens=int(tr.shape[0]),
                             score_s=round(time.perf_counter() - ts, 3), ref_encode_s=round(t_enc / len(jobs), 3),
                             peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), seed=0, **m))
        w.write(rows)
        rec = {"baseline": "SubspaceAD", "dataset": ds, "category": cat, "runs": len(rows),
               "wall_s": round(time.perf_counter() - t0, 1), "aug_images": len(items),
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

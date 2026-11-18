"""WinCLIP+ (Jeong et al., CVPR 2023) few-shot, re-implemented following anomalib's implementation
(open-edge-platform/anomalib src/anomalib/models/image/winclip/{torch_model,utils,prompting,lightning_model}.py,
read 8 Nov 2026): open_clip ViT-B-16-plus-240 (laion400m_e31), 240x240 bicubic input, compositional prompt
ensemble (7 normal / 4 anomalous states x 21 templates), window embeddings at scales 2 and 3 (stride 1) by
re-running the CLIP transformer on masked token subsets, zero-shot class scores (temperature 0.07) with harmonic
aggregation, few-shot visual association scores (patch + window level), final scores averaged as in anomalib.

usage: python -m src.baselines.winclip --dataset mvtec_ad
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from ..common import ResultWriter, append_log, load_config, setup_torch
from ..draws import all_draws
from ..imageio import iter_resized, normalize
from .common import MaskSet

NORMAL_STATES = ["{}", "flawless {}", "perfect {}", "unblemished {}", "{} without flaw", "{} without defect",
                 "{} without damage"]
ANOMALOUS_STATES = ["damaged {}", "{} with flaw", "{} with defect", "{} with damage"]
TEMPLATES = ["a cropped photo of the {}.", "a close-up photo of a {}.", "a close-up photo of the {}.",
             "a bright photo of a {}.", "a bright photo of the {}.", "a dark photo of the {}.", "a dark photo of a {}.",
             "a jpeg corrupted photo of the {}.", "a jpeg corrupted photo of the {}.", "a blurry photo of the {}.",
             "a blurry photo of a {}.", "a photo of a {}.", "a photo of the {}.", "a photo of a small {}.",
             "a photo of the small {}.", "a photo of a large {}.", "a photo of the large {}.",
             "a photo of the {} for visual inspection.", "a photo of a {} for visual inspection.",
             "a photo of the {} for anomaly detection.", "a photo of a {} for anomaly detection."]
TEMP = 0.07
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def prompts(cls):
    n = [t.format(s.format(cls)) for s in NORMAL_STATES for t in TEMPLATES]
    a = [t.format(s.format(cls)) for s in ANOMALOUS_STATES for t in TEMPLATES]
    return n, a


def make_masks(gs, k):
    grid = torch.arange(gs[0] * gs[1]).reshape(1, *gs).float()
    return F.unfold(grid, kernel_size=k, stride=1).int()           # [k*k, n_windows]


def cos(a, b):
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-1, -2)


def class_scores(img, txt):
    return (cos(img, txt) / TEMP).softmax(-1)[..., 1]


def harmonic(window_scores, gs, masks):
    B = window_scores.shape[0]
    pid = torch.arange(gs[0] * gs[1], device=masks.device).view(-1, 1, 1)
    cover = (masks.unsqueeze(0) == pid).any(1)                       # [P, n_windows]
    inv = torch.where(cover, (1 / window_scores).unsqueeze(1), 0).sum(-1)
    return (cover.sum(1) / inv).reshape(B, *gs).nan_to_num(posinf=0.0)


def assoc(emb, ref):
    ref = ref.reshape(-1, emb.shape[-1])
    return (1 - cos(emb, ref)).min(-1)[0] / 2


class WinCLIP:
    def __init__(self, dev, scales=(2, 3)):
        self.clip, _, _ = open_clip.create_model_and_transforms("ViT-B-16-plus-240", pretrained="laion400m_e31")
        self.clip = self.clip.to(dev).eval()
        self.clip.visual.output_tokens = True
        self.gs = self.clip.visual.grid_size
        self.masks = [make_masks(self.gs, s).to(dev) for s in scales]
        self.dev = dev
        self.tok = open_clip.get_tokenizer("ViT-B-16-plus-240")

    @torch.no_grad()
    def text(self, cls):
        n, a = prompts(cls)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ne = self.clip.encode_text(self.tok(n).to(self.dev)).float().mean(0, keepdim=True)
            ae = self.clip.encode_text(self.tok(a).to(self.dev)).float().mean(0, keepdim=True)
        return torch.cat([ne, ae])

    def _windows(self, fmap, masks):
        B, nm = fmap.shape[0], masks.shape[1]
        ci = torch.zeros(1, nm, dtype=torch.long, device=fmap.device)
        idx = torch.cat((ci, masks.long() + 1)).T                        # [n_windows, 1 + k*k]
        x = fmap[:, idx].reshape(B * nm, idx.shape[1], -1)               # windows x tokens
        v = self.clip.visual
        x = v.patch_dropout(x)
        x = v.ln_pre(x)
        x = v.transformer(x)          # open_clip >= 2.2x: NLD in, handles LND internally
        x = v.ln_post(x)
        pooled, _ = v._global_pool(x)
        if v.proj is not None:
            pooled = pooled @ v.proj
        return pooled.reshape(B, nm, -1)

    @torch.no_grad()
    def encode(self, x):
        out = {}
        h = self.clip.visual.patch_dropout.register_forward_hook(lambda m, i, o: out.__setitem__("f", i[0].detach()))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                img, patch = self.clip.encode_image(normalize(x, CLIP_MEAN, CLIP_STD))
            fmap = out["f"]
        finally:
            h.remove()   # must be removed before the window passes (they call patch_dropout again)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wins = [self._windows(fmap, m) for m in self.masks]
        return img.float(), [w_.float() for w_ in wins], patch.float()

    @torch.no_grad()
    def score(self, enc, txt, ref):
        img, wins, patch = enc
        iscore = class_scores(img, txt)
        ms = [iscore.view(-1, 1, 1).repeat(1, *self.gs)]
        for w_, m in zip(wins, self.masks):
            ms.append(harmonic(class_scores(w_, txt), self.gs, m))
        zs = (len(self.masks) + 1) / (1 / torch.stack(ms)).sum(0)
        _, rwins, rpatch = ref
        fs = [assoc(patch, rpatch).reshape(-1, *self.gs)]
        for w_, rw, m in zip(wins, rwins, self.masks):
            fs.append(harmonic(assoc(w_, rw), self.gs, m))
        fs = torch.stack(fs).mean(0)
        pix = (zs + fs) / 2
        return pix, (iscore + fs.amax(dim=(-2, -1))) / 2


def run(ds, cfg, dev, ks, draws_sel):
    bc = cfg["baselines"]["winclip"]
    model = WinCLIP(dev, tuple(bc["scales"]))
    draws = all_draws(ds, cfg)
    w = ResultWriter("baselines", cfg)
    for cat in cfg["datasets"][ds]["categories"]:
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        ms = MaskSet(cfg, ds, cat, dev)
        txt = model.text(cat.replace("_", " "))
        encs = []
        for idx, x in iter_resized(ms.paths, cfg, bc["img_size"], dev, 64, mode="square", patch=1):
            encs.append((idx, model.encode(x)))
        order = torch.tensor([i for idx, _ in encs for i in idx], device=dev).argsort()
        te = tuple(torch.cat([e[1][j] for e in encs])[order] if j != 1 else
                   [torch.cat([e[1][1][s] for e in encs])[order] for s in range(len(model.masks))] for j in range(3))
        jobs = [(d, k) for k in ks for d in draws_sel]
        paths = sorted({r["path"] for d, k in jobs for r in draws[(cat, d, k)]})
        ref = {}
        for idx, x in iter_resized(paths, cfg, bc["img_size"], dev, 64, mode="square", patch=1):
            img, wins, patch = model.encode(x)
            for b, i in enumerate(idx):
                ref[paths[i]] = (img[b], [w_[b] for w_ in wins], patch[b])
        rows = []
        for d, k in jobs:
            ts = time.perf_counter()
            shots = [ref[r["path"]] for r in draws[(cat, d, k)]]
            R = (None, [torch.stack([s[1][j] for s in shots]) for j in range(len(model.masks))],
                 torch.stack([s[2] for s in shots]))
            pix, img = [], []
            for s in range(0, te[0].shape[0], 64):
                p_, i_ = model.score((te[0][s:s + 64], [x_[s:s + 64] for x_ in te[1]], te[2][s:s + 64]), txt, R)
                pix.append(p_)
                img.append(i_)
            pix = torch.cat(pix)
            maps = F.interpolate(pix[:, None], size=(cfg["method"]["eval_res"],) * 2, mode="bilinear")[:, 0]
            m = ms.evaluate(maps, torch.cat(img).cpu().numpy())
            rows.append(dict(method="WinCLIP+", dataset=ds, category=cat, k=k, draw=d, backbone=bc["model"],
                             score_s=round(time.perf_counter() - ts, 3),
                             peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), seed=0, **m))
        w.write(rows)
        rec = {"baseline": "WinCLIP+", "dataset": ds, "category": cat, "runs": len(rows),
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

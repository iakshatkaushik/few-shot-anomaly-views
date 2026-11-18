"""Flownomaly (Perrone Ribeiro, Akdag, Bondarev, 2026) re-implemented from the official code
(github.com/b-rbmp/Flownomaly: src/flownomaly/{protocol,network,detector,rotation,calibration,evaluation}.py,
read 8 Nov 2026), run on our identical shot draws.

Same constants as protocol.py: DINOv2-G/14-reg at 672x672 (bicubic), block outputs of 0-indexed layers
15,19,23,27,31 (no final norm), per-layer standardisation with support statistics, averaged; 30 training
rotations (+ identity), 3 validation and 3 calibration rotations with the official deterministic angle rule
(md5 image-key hash / sha256 material, uniform in [0, 345]); rotations disabled for MVTec transistor;
rotated tokens kept only if their centre maps inside the source image; velocity MLP (bottleneck 768,
4 time-conditioned residual blocks, mlp ratio 4, sinusoidal time embedding 256, zero-init output and gate);
conditional flow matching, AdamW lr 1e-3 wd 1e-4, cosine to 1e-6 over 1000 epochs, batch 10240, fp16 AMP with
GradScaler, grad clip 1.0, validation every 3 epochs, early stopping patience 10 (min delta 1e-4), best weights;
operating time chosen by the normal-only Jacobian-gain (JGN) rule over t in {0.05..0.95}; score = squared
single-step recovery error; map = Gaussian sigma 0.8 tokens then bilinear upsampling; image score = mean of the
top 1% of the upsampled raw map.
Deviations: timm weights instead of torch.hub; GPU-vectorised JGN (identical estimator, different RNG streams);
evaluation maps at 448x448 like every method; image paths are hashed with the same last-4-components rule.

usage: python -m src.baselines.flownomaly --dataset mvtec_ad
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import ROOT, ResultWriter, append_log, load_config, setup_torch
from ..draws import all_draws, draw_seed
from ..encoders import Encoder
from ..featstore import load_tensors, store_dir
from ..imageio import iter_resized
from .common import MaskSet

ENC = "dinov2_vitg14_reg"
ROOT_RESULTS = ROOT / "results"
LAYERS = ["l16", "l20", "l24", "l28", "l32"]
P = dict(train_rot=30, val_rot=3, cal_rot=3, max_angle=345.0, train_seed=0, val_seed=1234, val_off=100000,
         cal_seed=5678, cal_off=200000, bottleneck=768, blocks=4, mlp_ratio=4, temb=256, batch=10240, epochs=1000,
         lr=1e-3, min_lr=1e-6, wd=1e-4, val_freq=3, patience=10, min_delta=1e-4,
         jgn=[i / 100 for i in range(5, 100, 5)], jgn_pairs=4, jgn_probes=4, jgn_frac=0.01, jgn_eps=1e-8,
         sigma_tok=0.8, top=0.01)


# ------------------------------------------------------------------------------------------- rotations
def image_key_hash(key: str) -> int:
    k = "/".join(os.path.normpath(key).split(os.sep)[-4:])
    return int(hashlib.md5(k.encode("utf-8")).hexdigest(), 16)


def rotation_angle(key, namespace, aug_idx, seed, offset, max_angle):
    rot_idx = int(offset) + int(aug_idx)
    if namespace == "train" and seed == 0 and offset == 0:
        ps = image_key_hash(key) + int(aug_idx)
    else:
        mat = f"{image_key_hash(key)}\0{namespace}\0{aug_idx}\0{seed}\0{rot_idx}".encode("utf-8")
        ps = int.from_bytes(hashlib.sha256(mat).digest()[:8], "big")
    return random.Random(ps).uniform(0.0, float(max_angle))


def valid_token_mask(angle, grid, device):
    c = (torch.arange(grid, dtype=torch.float32, device=device) + 0.5) / grid - 0.5
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    a = math.radians(angle)
    sx, sy = math.cos(a) * xx - math.sin(a) * yy, math.sin(a) * xx + math.cos(a) * yy
    return ((sx.abs() <= 0.5) & (sy.abs() <= 0.5)).reshape(-1)


def rotate_fill_mean(x, angle, mean):
    """TF.rotate on the *normalised* tensor with fill 0 == rotate the [0,1] image with fill = ImageNet mean."""
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode
    if angle == 0.0:
        return x
    m = torch.tensor(mean, device=x.device).view(3, 1, 1)
    return TF.rotate(x - m, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0) + m


# ------------------------------------------------------------------------------------------- network
class SinTime(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, t):
        h = self.d // 2
        f = torch.exp(torch.arange(h, device=t.device) * -(math.log(10000) / (h - 1)))
        e = t * f.unsqueeze(0)
        return torch.cat((e.sin(), e.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, d, td):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.mod = nn.Linear(td, 2 * d)
        self.mlp = nn.Sequential(nn.Linear(d, d * P["mlp_ratio"]), nn.GELU(), nn.Linear(d * P["mlp_ratio"], d))
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, s, te):
        sc, sh = self.mod(te).chunk(2, dim=-1)
        return s + self.mlp(self.norm(s) * (1.0 + sc) + sh)


class Velocity(nn.Module):
    def __init__(self, D=1536):
        super().__init__()
        td, b = P["temb"], P["bottleneck"]
        self.temb = nn.Sequential(SinTime(td), nn.Linear(td, td), nn.GELU())
        self.inp = nn.Sequential(nn.Linear(D + td, b), nn.LayerNorm(b), nn.GELU())
        self.blocks = nn.ModuleList(Block(b, td) for _ in range(P["blocks"]))
        self.onorm = nn.LayerNorm(b)
        self.out = nn.Linear(b, D)
        self.gate = nn.Sequential(nn.Linear(td, td), nn.GELU(), nn.Linear(td, 1))
        for m in (self.out, self.gate[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, s, t):
        te = self.temb(t)
        h = self.inp(torch.cat((s, te), dim=-1))
        for b in self.blocks:
            h = b(h, te)
        return self.out(self.onorm(h)) + self.gate(te) * s


# ------------------------------------------------------------------------------------------- fit
def fit(train_layers, val_layers, dev, seed):
    """train_layers / val_layers: {layer: [n, D] float tensors}"""
    stats = {l: (v.mean(0, keepdim=True), v.std(0, keepdim=True)) for l, v in train_layers.items()}
    norm = lambda layers: torch.stack([(layers[l] - stats[l][0]) / (stats[l][1] + 1e-6) for l in LAYERS]).mean(0)
    bank, vbank = norm(train_layers), norm(val_layers)
    torch.manual_seed(seed)
    net = Velocity(bank.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=P["lr"], weight_decay=P["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=P["epochs"], eta_min=P["min_lr"])
    gperm = torch.Generator(device=dev).manual_seed(P["train_seed"])
    scaler = torch.amp.GradScaler("cuda")
    best, best_w, pat, ep_run = float("inf"), None, 0, 0

    def vloss():
        net.eval()
        g = torch.Generator(device="cpu").manual_seed(P["val_seed"])
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(vbank), P["batch"]):
                tg = vbank[s:s + P["batch"]].float()
                nz = torch.randn(tg.shape, generator=g).to(dev)
                tm = torch.rand((len(tg), 1), generator=g).to(dev)
                with torch.amp.autocast("cuda"):
                    pr = net(tm * tg + (1 - tm) * nz, tm)
                tot += float(F.mse_loss(pr.float(), (tg - nz).float())) * len(tg)
                cnt += len(tg)
        net.train()
        return tot / cnt

    for ep in range(P["epochs"]):
        ep_run = ep + 1
        net.train()
        perm = torch.randperm(len(bank), generator=gperm, device=dev)
        for s in range(0, len(bank), P["batch"]):
            tg = bank.index_select(0, perm[s:s + P["batch"]]).float()
            nz = torch.randn_like(tg)
            tm = torch.rand((len(tg), 1), device=dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                pr = net(tm * tg + (1 - tm) * nz, tm)
                loss = F.mse_loss(pr.float(), (tg - nz).float())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        if (ep + 1) % P["val_freq"] != 0 and ep + 1 != P["epochs"]:
            continue
        vl = vloss()
        if vl < best - P["min_delta"]:
            best, best_w, pat = vl, copy.deepcopy(net.state_dict()), 0
        else:
            pat += 1
        if pat >= P["patience"]:
            break
    net.load_state_dict(best_w)
    net.eval().requires_grad_(False)
    return net, stats, ep_run


def jgn_time(net, cal_feats, dev, gen):
    """Normal-only Jacobian-gain calibration (calibration.py), vectorised over probes on GPU."""
    cand = P["jgn"]
    view_scores = {c: [] for c in cand}
    for f in cal_feats:
        f = f.float()
        n, D = f.shape
        top = max(1, int(n * P["jgn_frac"]))
        ratios = {c: [] for c in cand}
        for _ in range(P["jgn_pairs"]):
            noise = torch.randn(f.shape, generator=gen, device=dev)
            probes = (torch.randint(0, 2, (P["jgn_probes"], n, D), generator=gen, device=dev).float() * 2 - 1) / math.sqrt(D)
            for c in cand:
                def resid(q):
                    st = c * q + (1 - c) * noise
                    tt = torch.full((len(q), 1), c, device=dev)
                    v = net(st, tt)
                    return q - (st + (1 - c) * v)
                gains = []
                r0 = None
                for p in probes:
                    r, jv = torch.func.jvp(resid, (f,), (p,))
                    r0 = r if r0 is None else r0
                    gains.append(float(jv.double().square().sum()) / n)
                gain = sum(gains) / len(gains)
                floor = float(r0.double().square().sum(-1).topk(top).values.mean())
                ratios[c].append(gain / (floor + P["jgn_eps"]))
        for c in cand:
            view_scores[c].append(float(torch.tensor(ratios[c], dtype=torch.float64).quantile(0.5)))
    agg = {c: float(torch.tensor(view_scores[c], dtype=torch.float64).quantile(0.5)) for c in cand}
    best = max(agg.values())
    return min(c for c, v in agg.items() if v == best)


def gauss_tokens(m: torch.Tensor, sigma: float) -> torch.Tensor:
    """cv2.GaussianBlur(ksize=(0,0), sigma) for float maps: ksize = round(sigma*4*2+1)|1, reflect-101."""
    ks = int(round(sigma * 4 * 2 + 1)) | 1
    k = torch.tensor(cv2.getGaussianKernel(ks, sigma)[:, 0], device=m.device, dtype=torch.float32)
    r = ks // 2
    x = F.conv2d(F.pad(m[:, None], (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x[:, 0]


# ------------------------------------------------------------------------------------------- driver
def run(ds, cfg, dev, ks, draws_sel, cats=None, out="baselines", batch=16):
    enc = Encoder(ENC, cfg, dev, compile=False, batch_size=batch, sets=LAYERS)   # same reason as subspacead.py
    mean = cfg["extract"]["mean"]
    draws = all_draws(ds, cfg)
    w = ResultWriter(out, cfg)
    g = enc.size // enc.patch
    res = cfg["method"]["eval_res"]
    for cat in (cats or cfg["datasets"][ds]["categories"]):
        t0 = time.perf_counter()
        rot_on = not (ds == "mvtec_ad" and "transistor" in cat)
        jobs = [(d, k) for k in ks for d in draws_sel]
        paths = sorted({r["path"] for d, k in jobs for r in draws[(cat, d, k)]})
        imgs = {}
        for idx, x in iter_resized(paths, cfg, enc.size, dev, 4096, mode="square", patch=enc.patch):
            for b, i in enumerate(idx):
                imgs[paths[i]] = x[b]

        def support(p):
            """identity + 30 train + 3 val + 3 calibration rotations of one shot, encoded when the fit needs them
            (pre-encoding every shot of every draw needed ~90 GB host RAM per category; the container limit is 193 GB)"""
            specs = [("id", 0, 0.0)]
            specs += [("train", a, rotation_angle(p, "train", a, P["train_seed"], 0, P["max_angle"])) for a in range(P["train_rot"])] if rot_on else []
            ma = P["max_angle"] if rot_on else 0.0
            specs += [("val", a, rotation_angle(p, "cfm_val", a, P["val_seed"], P["val_off"], ma)) for a in range(P["val_rot"])]
            specs += [("cal", a, rotation_angle(p, "calibration", a, P["cal_seed"], P["cal_off"], ma)) for a in range(P["cal_rot"])]
            X = torch.stack([rotate_fill_mean(imgs[p], ang, mean) for _, _, ang in specs])
            o = enc(X)
            return [(kind, ang, {l: o[l][j] for l in LAYERS}) for j, (kind, _, ang) in enumerate(specs)]

        sd = store_dir(cfg, ENC, ds, cat, "test")
        from ..featstore import read_manifest
        man0 = read_manifest(sd)
        if man0 is None or not all(l in man0["sets"] for l in LAYERS):   # MPDD/LOCO/AD2: separate 5-layer cache
            sd = store_dir(cfg, ENC, ds, cat, "test_flow")
        te = load_tensors(sd, LAYERS, device=dev)
        man = json.load(open(sd / "manifest.json"))
        ms = MaskSet(cfg, ds, cat, dev)
        assert ms.paths == [e["path"] for e in man["entries"]]
        T = te[LAYERS[0]].shape[0]
        rows = []
        done = set()
        bf = ROOT_RESULTS / "baselines.jsonl"
        if bf.exists():   # skip fits already in the results (lets two processes share the work without duplicates)
            for l in open(bf):
                r_ = json.loads(l)
                if r_.get("method") == "Flownomaly" and r_["dataset"] == ds and r_["category"] == cat:
                    done.add((r_["draw"], r_["k"]))
        for d, k in jobs:
            if (d, k) in done:
                continue
            ts = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            shots = [r["path"] for r in draws[(cat, d, k)]]
            tr = {l: [] for l in LAYERS}
            va = {l: [] for l in LAYERS}
            cal = []
            for p in shots:
                for kind, ang, feats in support(p):
                    vm = valid_token_mask(ang, g, dev) if kind != "id" else torch.ones(g * g, dtype=torch.bool, device=dev)
                    if kind in ("id", "train"):
                        for l in LAYERS:
                            tr[l].append(feats[l][vm].float())
                    elif kind == "val":
                        for l in LAYERS:
                            va[l].append(feats[l][vm].float())
                    else:
                        cal.append({l: feats[l][vm].half().float() for l in LAYERS})
            tr = {l: torch.cat(v) for l, v in tr.items()}
            va = {l: torch.cat(v) for l, v in va.items()}
            seed = draw_seed(d, k)
            net, stats, ep = fit(tr, va, dev, seed)
            nrm = lambda layers: torch.stack([(layers[l].float() - stats[l][0]) / (stats[l][1] + 1e-6) for l in LAYERS]).mean(0)
            gen = torch.Generator(device=dev).manual_seed(seed + 7)
            t_op = jgn_time(net, [nrm(c) for c in cal], dev, gen)
            # inference
            scores = []
            gq = torch.Generator(device=dev).manual_seed(seed + 11)
            with torch.no_grad():
                for s in range(0, T, 64):
                    q = nrm({l: te[l][s:s + 64] for l in LAYERS}).reshape(-1, 1536)
                    nz = torch.randn(q.shape, generator=gq, device=dev)
                    st = t_op * q + (1 - t_op) * nz
                    with torch.amp.autocast("cuda"):
                        v = net(st, torch.full((len(q), 1), t_op, device=dev))
                    rec = st + (1 - t_op) * v.float()
                    scores.append((q - rec).square().sum(-1).view(-1, g, g))
            raw = torch.cat(scores)
            up_raw = F.interpolate(raw[:, None], size=(res, res), mode="bilinear", align_corners=False)[:, 0]
            img = up_raw.flatten(1).topk(max(1, int(res * res * P["top"])), dim=1).values.mean(1).cpu().numpy()
            maps = F.interpolate(gauss_tokens(raw, P["sigma_tok"])[:, None], size=(res, res), mode="bilinear",
                                 align_corners=False)[:, 0]
            m = ms.evaluate(maps, img)
            rows.append(dict(method="Flownomaly", dataset=ds, category=cat, k=k, draw=d, encoder=ENC, epochs=ep,
                             t_op=t_op, n_train_tokens=int(tr[LAYERS[0]].shape[0]), rotations=rot_on,
                             fit_s=round(time.perf_counter() - ts, 2),
                             peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), seed=seed, **m))
            print(json.dumps({"cat": cat, "d": d, "k": k, "ep": ep, "t": t_op, "auroc": round(m["img_auroc"], 2),
                              "s": rows[-1]["fit_s"]}), flush=True)
            w.write(rows[-1:])
        if not rows:
            continue
        rec = {"baseline": "Flownomaly", "dataset": ds, "category": cat, "runs": len(rows),
               "wall_s": round(time.perf_counter() - t0, 1),
               "img_auroc_mean": round(float(np.mean([r["img_auroc"] for r in rows])), 2)}
        print(json.dumps(rec), flush=True)
        append_log("baselines", rec, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--draws", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--out", default="baselines")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    cfg = load_config()
    dev = setup_torch()
    run(a.dataset, cfg, dev, a.ks, a.draws, a.cats, a.out, a.batch)


if __name__ == "__main__":
    main()

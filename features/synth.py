"""Synthetic anomalies on every view of every shot: CutPaste-style transplants, NSA Poisson blending,
DRAEM-Perlin texture blending (DTD textures). Synthesis runs in a CPU process pool; encoding on GPU.

Stored per (draw, k): tokens of all patches touched by a synthetic mask (coverage > 0) together with
token-level coverage and mean absolute intensity change (used by the pixel-change filter at mining time),
and the full token-level coverage map of every synthetic image.

usage: python -m features.synth --encoder dinov2_vitb14 --datasets visa [--sets b9_12]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.common import Timer, append_log, load_config, setup_torch, ROOT
from src.draws import all_draws, draw_seed
from src.encoders import Encoder
from src.featstore import store_dir
from features.views import save_file, wait_io

FAMILIES = ("cutpaste", "nsa", "perlin")


# ------------------------------------------------------------------------------------------ synthesis
def _rect(rng, S, area_rng, aspect_rng):
    for _ in range(50):
        area = rng.uniform(*area_rng) * S * S
        ar = math.exp(rng.uniform(math.log(aspect_rng[0]), math.log(aspect_rng[1])))
        h, w = int(round(math.sqrt(area / ar))), int(round(math.sqrt(area * ar)))
        if 4 <= h < S - 4 and 4 <= w < S - 4:
            return h, w
    return S // 8, S // 8


def _jitter(img, rng, s=0.1):
    x = img.astype(np.float32) / 255.0
    x = x * rng.uniform(1 - s, 1 + s)
    m = x.mean()
    x = (x - m) * rng.uniform(1 - s, 1 + s) + m
    g = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    x = g + (x - g) * rng.uniform(1 - s, 1 + s)
    hsv = cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-s, s) * 360) % 360
    return (np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1) * 255).astype(np.uint8)


def cutpaste(img, rng, c):
    S = img.shape[0]
    h, w = _rect(rng, S, c["cutpaste_area"], c["cutpaste_aspect"])
    y0, x0 = rng.integers(0, S - h), rng.integers(0, S - w)
    patch = _jitter(img[y0:y0 + h, x0:x0 + w], rng)
    for _ in range(20):
        y1, x1 = rng.integers(0, S - h), rng.integers(0, S - w)
        if abs(int(y1) - int(y0)) > h // 2 or abs(int(x1) - int(x0)) > w // 2:
            break
    out = img.copy()
    out[y1:y1 + h, x1:x1 + w] = patch
    m = np.zeros((S, S), np.float32)
    m[y1:y1 + h, x1:x1 + w] = 1
    return out, m


def nsa(img, src, rng, c):
    """NSA-style: a patch from another normal image (another shot when available) resized and
    Poisson-blended (cv2.seamlessClone, NORMAL_CLONE) at a random location."""
    S = img.shape[0]
    h, w = _rect(rng, S, c["nsa_area"], c["cutpaste_aspect"])
    ys, xs = rng.integers(0, S - h), rng.integers(0, S - w)
    patch = src[ys:ys + h, xs:xs + w]
    sc = rng.uniform(0.8, 1.2)
    ph, pw = max(4, min(S - 8, int(h * sc))), max(4, min(S - 8, int(w * sc)))
    patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)
    cy = int(rng.integers(ph // 2 + 2, S - ph // 2 - 2))
    cx = int(rng.integers(pw // 2 + 2, S - pw // 2 - 2))
    pm = np.full((ph, pw), 255, np.uint8)
    pm[0, :] = pm[-1, :] = 0
    pm[:, 0] = pm[:, -1] = 0
    try:
        out = cv2.seamlessClone(patch, img, pm, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        out = img.copy()
    m = np.zeros((S, S), np.float32)
    y0, x0 = cy - ph // 2, cx - pw // 2
    m[y0:y0 + ph, x0:x0 + pw] = (pm > 0)
    return out, m


def _perlin(shape, res, rng):
    """rand_perlin_2d_np from DRAEM (Zavrtanik et al., ICCV 2021)."""
    fade = lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * math.pi * rng.random((res[0] + 1, res[1] + 1))
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tile = lambda s1, s2: np.repeat(np.repeat(gradients[s1[0]:s1[1], s2[0]:s2[1]], d[0], axis=0), d[1], axis=1)
    dot = lambda grad, sh: (np.stack((grid[:shape[0], :shape[1], 0] + sh[0], grid[:shape[0], :shape[1], 1] + sh[1]),
                                     axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)
    n00 = dot(tile([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile([0, -1], [1, None]), [0, -1])
    n11 = dot(tile([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    lerp = lambda a, b, w: (b - a) * w + a
    return math.sqrt(2) * lerp(lerp(n00, n10, t[..., 0]), lerp(n01, n11, t[..., 0]), t[..., 1])


def perlin(img, tex_path, rng, c):
    S = img.shape[0]
    S64 = int(math.ceil(S / 64) * 64)
    tex = cv2.cvtColor(cv2.imread(tex_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    tex = _jitter(cv2.resize(tex, (S, S), interpolation=cv2.INTER_LINEAR), rng, 0.2)
    for _ in range(20):
        sx, sy = 2 ** int(rng.integers(c["perlin_scale"][0], c["perlin_scale"][1] + 1)), \
                 2 ** int(rng.integers(c["perlin_scale"][0], c["perlin_scale"][1] + 1))
        noise = _perlin((S64, S64), (sx, sy), rng)[:S, :S]
        ang = rng.uniform(-90, 90)
        M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, 1.0)
        noise = cv2.warpAffine(noise.astype(np.float32), M, (S, S), borderMode=cv2.BORDER_REFLECT)
        m = (noise > c["perlin_threshold"]).astype(np.float32)
        if m.sum() > 0.002 * S * S:
            break
    beta = rng.uniform(*c["perlin_beta"])
    x, t = img.astype(np.float32), tex.astype(np.float32)
    out = x * (1 - m[..., None]) + (1 - beta) * t * m[..., None] + beta * x * m[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), m


def _job(args):
    img, src, tex_path, fam, seed, c = args
    rng = np.random.default_rng(seed)
    if fam == "cutpaste":
        return cutpaste(img, rng, c)
    if fam == "nsa":
        return nsa(img, src, rng, c)
    return perlin(img, tex_path, rng, c)


def _job_shm(args):
    """Images are exchanged through memory-mapped files in /dev/shm instead of pickling through pipes."""
    fin, fout, fmask, j, vi, si, tex_path, fam, seed, c = args
    V = np.load(fin, mmap_mode="r")
    out, m = _job((np.asarray(V[vi]), np.asarray(V[si]), tex_path, fam, seed, c))
    O = np.load(fout, mmap_mode="r+")
    M = np.load(fmask, mmap_mode="r+")
    O[j] = out
    M[j] = (m > 0.5).astype(np.uint8)
    return j


# ------------------------------------------------------------------------------------------ driver
def run(enc: Encoder, datasets: list[str], cfg: dict, pool, ks: list | None = None):
    from features.views import load_shot_images, make_view, view_params, StreamEncoder
    c = cfg["synth"]
    textures = sorted(str(p) for p in (ROOT / c["dtd_root"]).rglob("*.jpg"))
    P = enc.patch
    for ds in datasets:
        draws = all_draws(ds, cfg)
        for cat in cfg["datasets"][ds]["categories"]:
            t = Timer()
            torch.cuda.reset_peak_memory_stats()
            base = store_dir(cfg, enc.name, ds, cat, "synth")
            need = [(d, k) for (cc, d, k) in draws if cc == cat and (ks is None or k in ks)
                    and not (base / f"d{d}_k{k}.safetensors").exists()]
            if not need:
                continue
            base.mkdir(parents=True, exist_ok=True)
            imgs = load_shot_images(ds, cat, draws, enc, cfg)
            se = StreamEncoder(enc)
            meta, n_syn, t_syn = {}, 0, 0.0
            for d, k in need:
                seed = draw_seed(d, k)
                shots = draws[(cat, d, k)]
                views = []
                for si, r in enumerate(shots):
                    for p in view_params(seed, si, cfg):
                        views.append(make_view(imgs[r["path"]], p, P, cfg))
                V = torch.stack(views)                                   # [k*16,3,S,S]
                vnp = (V.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().byte().cpu().numpy()
                nv = len(vnp)
                rng = np.random.default_rng([seed, 7])
                tag = f"/dev/shm/fsad_synth_{os.getpid()}"
                fin, fout, fmask = tag + "_in.npy", tag + "_out.npy", tag + "_mask.npy"
                np.save(fin, vnp)
                S_ = vnp.shape[1]
                n_out = nv * len(FAMILIES)
                np.lib.format.open_memmap(fout, mode="w+", dtype=np.uint8, shape=(n_out, S_, S_, 3)).flush()
                np.lib.format.open_memmap(fmask, mode="w+", dtype=np.uint8, shape=(n_out, S_, S_)).flush()
                args = []
                for vi in range(nv):
                    shot = vi // cfg["views"]["n_views"]
                    other = [j for j in range(nv) if j // cfg["views"]["n_views"] != shot] or list(range(nv))
                    for fi, fam in enumerate(FAMILIES):
                        src = int(rng.choice(other))
                        tex = textures[int(rng.integers(len(textures)))]
                        args.append((fin, fout, fmask, vi * len(FAMILIES) + fi, vi, src, tex, fam, [seed, vi, fi, 11], c))
                ts = Timer()
                pool.map(_job_shm, args, chunksize=2)
                t_syn += ts()
                syn = torch.from_numpy(np.load(fout)).to(enc.device).permute(0, 3, 1, 2).float() / 255
                msk = torch.from_numpy(np.load(fmask)).to(enc.device).unsqueeze(1).float()
                for f_ in (fin, fout, fmask):
                    os.remove(f_)
                n_syn += n_out
                base_rep = V.repeat_interleave(len(FAMILIES), 0)
                cov = F.avg_pool2d(msk, P).flatten(1)                                   # [n, N]
                chg = F.avg_pool2d((syn - base_rep).abs().mean(1, keepdim=True), P).flatten(1)
                meta[f"s{d}_{k}"] = (cov, chg)
                for job, o in se.add(f"s{d}_{k}", syn):
                    _save(job, o, meta, base)
            for job, o in se.flush():
                _save(job, o, meta, base)
            wait_io()
            rec = {"step": "synth", "encoder": enc.name, "dataset": ds, "category": cat, "jobs": len(need),
                   "synthetic_images": n_syn, "cpu_synth_s": round(t_syn, 2), "wall_s": round(t(), 2),
                   "img_per_s": round(n_syn / t(), 1), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            print(json.dumps(rec), flush=True)
            append_log("synth", rec, cfg)


def _save(job, o, meta, base):
    d, k = job[1:].split("_")
    cov, chg = meta.pop(job)
    keep = cov > 0                                          # candidate tokens: touched by a synthetic mask
    img_i, tok_i = keep.nonzero(as_tuple=True)
    t = {f"feat_{s}": v[img_i, tok_i].cpu() for s, v in o.items()}
    t.update(img=img_i.to(torch.int32).cpu(), tok=tok_i.to(torch.int16).cpu(),
             cov=cov[img_i, tok_i].half().cpu(), chg=chg[img_i, tok_i].half().cpu(),
             cov_map=cov.half().cpu(),
             family=(img_i % len(FAMILIES)).to(torch.int8).cpu(), view=(img_i // len(FAMILIES)).to(torch.int16).cpu())
    save_file(t, str(base / f"d{d}_k{k}.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--procs", type=int, default=96)
    ap.add_argument("--ks", nargs="*", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    pool = mp.get_context("spawn").Pool(a.procs)
    dev = setup_torch()
    enc = Encoder(a.encoder, cfg, dev, sets=a.sets)
    run(enc, a.datasets, cfg, pool, ks=a.ks)
    pool.close()


if __name__ == "__main__":
    main()

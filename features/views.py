"""Correspondence-preserving views and arbitrary-angle rotations of every shot.

For each (dataset, category, draw, k) with draw_seed = 1000*draw + 10*k and each of the k shots:
  * 16 views: view 0 = identity; views 1..7 = the other 7 elements of the dihedral group D4
    (h-flip x rot90^r); views 8..15 = random D4 elements. Views 1..15 are additionally translated by whole
    patch strides (sy, sx in [-2, 2]) with reflect padding, and receive mild colour jitter / Gaussian blur.
    `idx` maps every token of the view grid to its canonical token (-1 = token entered from padding).
  * 30 arbitrary-angle rotations (memory-only candidates, never positives) with a per-token validity mask.

usage: python -m features.views --encoder dinov2_vitb14 --datasets mvtec_ad visa [--sets b9_12 last]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.common import Timer, append_log, load_config, setup_torch, ROOT
from src.draws import all_draws, draw_seed
from src.encoders import Encoder
from src.featstore import store_dir
from src.imageio import iter_resized
from safetensors.torch import save_file as _save_file
from concurrent.futures import ThreadPoolExecutor

_IO = ThreadPoolExecutor(6)
_FUTS = []


def save_file(t, path):
    """Asynchronous cache write (keeps the GPU busy while shards go to disk)."""
    _FUTS.append(_IO.submit(_save_file, t, path))


def wait_io():
    for f in _FUTS:
        f.result()
    _FUTS.clear()

D4 = [(r, f) for f in (0, 1) for r in range(4)]  # (rot90 count, hflip); (0, 0) = identity


# ----------------------------------------------------------------------------------------------- params
def view_params(seed: int, shot: int, cfg: dict) -> list[dict]:
    cv = cfg["views"]
    rng = np.random.default_rng([seed, shot, 1])
    out = []
    for v in range(cv["n_views"]):
        if v == 0:
            out.append(dict(rot=0, flip=0, sy=0, sx=0, jit=None, blur=None))
            continue
        r, f = D4[v] if v < 8 else D4[int(rng.integers(0, 8))]
        ms = cv["max_shift_tokens"]
        sy, sx = (int(t) for t in rng.integers(-ms, ms + 1, size=2))
        jit = None
        if rng.random() < cv["jitter_prob"]:
            s = cv["jitter_strength"]
            jit = [float(t) for t in rng.uniform(1 - s, 1 + s, size=3)]  # brightness, contrast, saturation
        blur = float(rng.uniform(*cv["blur_sigma"])) if rng.random() < cv["blur_prob"] else None
        out.append(dict(rot=int(r), flip=int(f), sy=sy, sx=sx, jit=jit, blur=blur))
    return out


def rotation_angles(seed: int, shot: int, cfg: dict) -> list[float]:
    rng = np.random.default_rng([seed, shot, 2])
    return [float(a) for a in rng.uniform(0.0, cfg["views"]["rotation_max_deg"], size=cfg["views"]["n_rotations"])]


# ----------------------------------------------------------------------------------------------- ops
def _shift(x: torch.Tensor, sy: int, sx: int, pad: int, mode: str, value=0):
    """out[.., y, x] = in[.., y - sy, x - sx]; x: [..., H, W]; shifts in the same units as `pad`."""
    if sy == 0 and sx == 0:
        return x
    H, W = x.shape[-2:]
    if mode == "reflect":
        xp = F.pad(x.unsqueeze(0) if x.dim() == 3 else x, (pad, pad, pad, pad), mode="reflect")
        xp = xp.squeeze(0) if x.dim() == 3 else xp
    else:
        xp = F.pad(x, (pad, pad, pad, pad), mode="constant", value=value)
    return xp[..., pad - sy: pad - sy + H, pad - sx: pad - sx + W]


def apply_geom(x: torch.Tensor, p: dict, unit: int, max_shift: int, mode: str, value=0) -> torch.Tensor:
    """Apply flip -> rot90 -> shift. `unit` = pixels per token (patch) for images, 1 for token grids."""
    if p["flip"]:
        x = x.flip(-1)
    if p["rot"]:
        x = torch.rot90(x, p["rot"], dims=(-2, -1))
    return _shift(x, p["sy"] * unit, p["sx"] * unit, max_shift * unit, mode, value)


def apply_photo(x: torch.Tensor, p: dict) -> torch.Tensor:
    """x: [3,H,W] in [0,1]"""
    if p["jit"] is not None:
        b, c, s = p["jit"]
        x = x * b
        m = x.mean()
        x = (x - m) * c + m
        g = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0)
        x = (g + (x - g) * s).clamp(0, 1)
    if p["blur"] is not None:
        k = 2 * math.ceil(3 * p["blur"]) + 1
        x = TF.gaussian_blur(x.unsqueeze(0), [k, k], [p["blur"], p["blur"]]).squeeze(0)
    return x


def make_view(x: torch.Tensor, p: dict, patch: int, cfg: dict) -> torch.Tensor:
    return apply_photo(apply_geom(x, p, patch, cfg["views"]["max_shift_tokens"], "reflect"), p)


def index_map(g: int, p: dict, cfg: dict, device="cpu") -> torch.Tensor:
    G = torch.arange(g * g, device=device, dtype=torch.int32).view(g, g)
    return apply_geom(G, p, 1, cfg["views"]["max_shift_tokens"], "constant", -1).reshape(-1)


def rotate_batch(x: torch.Tensor, angles: list[float], patch: int, valid_frac: float):
    """x: [n,3,S,S]; bilinear rotation about the centre with zero fill (as T.RandomRotation / TF.rotate).
    Returns rotated images and per-token validity (>= valid_frac of the patch comes from the image)."""
    n, _, S, _ = x.shape
    th = torch.tensor(angles, device=x.device, dtype=torch.float32) * math.pi / 180
    c, s = torch.cos(th), torch.sin(th)
    # affine_grid maps output -> input coords; TF.rotate rotates counter-clockwise for positive angles
    A = torch.stack([torch.stack([c, -s, torch.zeros_like(c)], 1), torch.stack([s, c, torch.zeros_like(c)], 1)], 1)
    grid = F.affine_grid(A, (n, 3, S, S), align_corners=False)
    xr = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    ones = torch.ones(n, 1, S, S, device=x.device)
    vr = F.grid_sample(ones, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    valid = F.avg_pool2d(vr, patch).flatten(1) >= valid_frac
    return xr, valid


# ----------------------------------------------------------------------------------------------- driver
class StreamEncoder:
    """Accumulate image tensors from many small jobs, encode in full fixed-size batches, emit per job."""

    def __init__(self, enc: Encoder):
        self.enc = enc
        self.queue: list[tuple[str, int, torch.Tensor]] = []  # (job, row, image)
        self.parts: dict[str, dict[int, dict]] = {}
        self.sizes: dict[str, int] = {}
        self.n_images = 0

    def add(self, job: str, x: torch.Tensor):
        self.sizes[job] = x.shape[0]
        self.parts[job] = {}
        self.queue += [(job, i, x[i]) for i in range(x.shape[0])]
        return self._flush(force=False)

    def _flush(self, force: bool):
        done = []
        B = self.enc.batch_size
        while len(self.queue) >= B or (force and self.queue):
            chunk, self.queue = self.queue[:B], self.queue[B:]
            o = self.enc(torch.stack([c[2] for c in chunk]))
            self.n_images += len(chunk)
            for b, (job, row, _) in enumerate(chunk):
                self.parts[job][row] = {k: v[b] for k, v in o.items()}
            for job in {c[0] for c in chunk}:
                if len(self.parts[job]) == self.sizes[job]:
                    rows = self.parts.pop(job)
                    done.append((job, {k: torch.stack([rows[i][k] for i in range(self.sizes[job])])
                                       for k in rows[0]}))
        return done

    def flush(self):
        return self._flush(force=True)


def load_shot_images(dataset: str, category: str, draws: dict, enc: Encoder, cfg: dict) -> dict[str, torch.Tensor]:
    paths = sorted({r["path"] for (c, d, k), rows in draws.items() if c == category for r in rows})
    out = {}
    for idx, x in iter_resized(paths, cfg, enc.size, enc.device, 4096, mode="square", patch=enc.patch):
        for b, i in enumerate(idx):
            out[paths[i]] = x[b]
    return out


def run(enc: Encoder, datasets: list[str], cfg: dict, do_rot: bool = True, ks: list | None = None):
    g = enc.size // enc.patch
    for ds in datasets:
        draws = all_draws(ds, cfg)
        for cat in cfg["datasets"][ds]["categories"]:
            t = Timer()
            base = store_dir(cfg, enc.name, ds, cat, "views")
            rbase = store_dir(cfg, enc.name, ds, cat, "rot")
            jobs = [(d, k) for (c, d, k) in draws if c == cat and (ks is None or k in ks)]
            need = [(d, k) for d, k in jobs if not (base / f"d{d}_k{k}.safetensors").exists()
                    or (do_rot and not (rbase / f"d{d}_k{k}.safetensors").exists())]
            if not need:
                continue
            base.mkdir(parents=True, exist_ok=True)
            rbase.mkdir(parents=True, exist_ok=True)
            imgs = load_shot_images(ds, cat, draws, enc, cfg)
            se = StreamEncoder(enc)
            meta = {}
            pending = []
            for d, k in need:
                seed = draw_seed(d, k)
                shots = draws[(cat, d, k)]
                vx, idxs, params, rx, valid, angles = [], [], [], [], [], []
                for si, r in enumerate(shots):
                    x = imgs[r["path"]]
                    for p in view_params(seed, si, cfg):
                        vx.append(make_view(x, p, enc.patch, cfg))
                        idxs.append(index_map(g, p, cfg))
                        params.append(dict(p, shot=si, path=r["path"]))
                    if do_rot:
                        ang = rotation_angles(seed, si, cfg)
                        xr, vr = rotate_batch(x.unsqueeze(0).expand(len(ang), -1, -1, -1), ang, enc.patch,
                                              cfg["views"]["rotation_valid_frac"])
                        rx.append(xr)
                        valid.append(vr)
                        angles += [dict(shot=si, angle=a, path=r["path"]) for a in ang]
                meta[f"v{d}_{k}"] = (torch.stack(idxs).to(torch.int16), params)
                pending += se.add(f"v{d}_{k}", torch.stack(vx))
                if do_rot:
                    meta[f"r{d}_{k}"] = (torch.cat(valid).cpu(), angles)
                    pending += se.add(f"r{d}_{k}", torch.cat(rx))
                pending = _save(pending, meta, base, rbase)
            pending += se.flush()
            _save(pending, meta, base, rbase)
            wait_io()
            rec = {"step": "views", "encoder": enc.name, "dataset": ds, "category": cat, "jobs": len(need),
                   "images": se.n_images, "wall_s": round(t(), 2), "img_per_s": round(se.n_images / t(), 1),
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            print(json.dumps(rec), flush=True)
            append_log("views", rec, cfg)


def _save(pending, meta, base, rbase):
    for job, o in pending:
        d, k = job[1:].split("_")
        if job[0] == "v":
            idx, params = meta.pop(job)
            t = {f"feat_{s}": v.cpu() for s, v in o.items()} | {"idx": idx}
            save_file(t, str(base / f"d{d}_k{k}.safetensors"))
            json.dump(params, open(base / f"d{d}_k{k}.json", "w"))
        else:
            valid, angles = meta.pop(job)
            t = {f"feat_{s}": v.cpu() for s, v in o.items()} | {"valid": valid}
            save_file(t, str(rbase / f"d{d}_k{k}.safetensors"))
            json.dump(angles, open(rbase / f"d{d}_k{k}.json", "w"))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--no_rot", action="store_true")
    ap.add_argument("--ks", nargs="*", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    dev = setup_torch()
    enc = Encoder(a.encoder, cfg, dev, sets=a.sets)
    run(enc, a.datasets, cfg, do_rot=not a.no_rot, ks=a.ks)


if __name__ == "__main__":
    main()

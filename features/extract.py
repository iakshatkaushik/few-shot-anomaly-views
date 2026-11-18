"""Encode test images and canonical shot images, cache fp16 tokens in safetensors shards + manifest.

usage: python -m features.extract --encoder dinov2_vitb14 --datasets mvtec_ad visa --kinds test shots
       python -m features.extract --masks --datasets mvtec_ad        (eval masks at eval_res)
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from src.common import Timer, append_log, config_hash, load_config, setup_torch, ROOT
from src.datasets import load_index
from src.draws import all_draws
from src.encoders import Encoder
from src.featstore import read_manifest, save_shard, store_dir, write_manifest, load_single
from src.imageio import iter_resized


def unique_shot_rows(dataset: str, cfg: dict) -> dict[str, list[dict]]:
    out: dict[str, dict] = {}
    for (c, d, k), rows in all_draws(dataset, cfg).items():
        for r in rows:
            out.setdefault(c, {})[r["path"]] = r
    return {c: sorted(v.values(), key=lambda r: r["path"]) for c, v in out.items()}


def encode_groups(enc: Encoder, groups: list[tuple], cfg: dict, meta: dict, mode: str = "square",
                  profile: bool = False) -> list[dict]:
    """groups: [(store_dir, rows)]. One DataLoader for all groups (no per-category worker start-up);
    images are accumulated on the GPU up to the probed batch size; outputs are routed back per group."""
    todo = []
    for d, rows in groups:
        m = read_manifest(d)
        if m is not None and m.get("n") == len(rows) and m.get("config_hash_enc") == meta["config_hash_enc"]:
            continue
        todo.append((d, rows))
    if not todo:
        return []
    paths, owner = [], []
    for g, (d, rows) in enumerate(todo):
        for j, r in enumerate(rows):
            paths.append(r["path"])
            owner.append((g, j))
    out: list[dict] = [dict() for _ in todo]
    grids: list = [None] * len(todo)
    t = Timer()
    prof = None
    if profile:
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                      schedule=torch.profiler.schedule(wait=1, warmup=1, active=20))
        prof.start()
    n_done = 0
    torch.cuda.reset_peak_memory_stats()
    for idx, x in iter_resized(paths, cfg, enc.size, enc.device, enc.batch_size, mode=mode, patch=enc.patch):
        o = enc(x)
        pinned = {k: torch.empty(v.shape, dtype=v.dtype, pin_memory=True) for k, v in o.items()}
        for k, v in o.items():
            pinned[k].copy_(v, non_blocking=True)
        torch.cuda.synchronize()
        o = pinned
        for b, i in enumerate(idx):
            g, j = owner[i]
            grids[g] = (x.shape[2] // enc.patch, x.shape[3] // enc.patch)
            for k, v in o.items():
                out[g].setdefault(k, {})[j] = v[b]
        n_done += len(idx)
        if prof is not None:
            prof.step()
    if prof is not None:
        prof.stop()
        tbl = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
        (ROOT / "logs").mkdir(exist_ok=True)
        open(ROOT / "logs" / f"profile_extract_{enc.name}.txt", "w").write(tbl)
    dt = t()
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    infos = []
    shard_n = cfg["extract"]["shard_images"]
    for g, (d, rows) in enumerate(todo):
        shards = []
        for s0 in range(0, len(rows), shard_n):
            tens = {k: torch.stack([v[j] for j in range(s0, min(s0 + shard_n, len(rows)))]) for k, v in out[g].items()}
            shards.append(save_shard(d, len(shards), tens))
        man = dict(meta, n=len(rows), shards=shards, grid=list(grids[g]), entries=rows)
        write_manifest(d, man)
        infos.append({"store": str(d.relative_to(ROOT / cfg["paths"]["cache_root"])), "n": len(rows)})
    rec = {"encoder": enc.name, "n_images": n_done, "wall_s": round(dt, 2), "img_per_s": round(n_done / dt, 1),
           "peak_mem_gb": peak, "batch": enc.batch_size, "groups": len(todo)}
    print(json.dumps(rec), flush=True)
    append_log("extract", rec, cfg)
    return infos


def run_masks(dataset: str, cfg: dict):
    rows = [r for r in load_index(dataset, cfg) if r["split"] == "test"]
    res = cfg["method"]["eval_res"]
    for c in cfg["datasets"][dataset]["categories"]:
        rc = [r for r in rows if r["category"] == c]
        out = ROOT / cfg["paths"]["cache_root"] / "masks" / dataset / f"{c}.safetensors"
        if out.exists():
            continue

        def one(r):
            if r["mask"] is None:
                return np.zeros((res, res), dtype=np.uint8)
            import cv2
            m = cv2.imread(str(ROOT / r["mask"]), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize((m > 0).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
            return m

        with ThreadPoolExecutor(32) as ex:
            ms = list(ex.map(one, rc))
        out.parent.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file
        save_file({"masks": torch.from_numpy(np.stack(ms)), "labels": torch.tensor([r["label"] for r in rc])}, str(out))
        json.dump([r["path"] for r in rc], open(out.with_suffix(".json"), "w"))
        print(dataset, c, len(rc), "masks", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+", default=["test", "shots"])
    ap.add_argument("--mode", default="square", choices=["square", "shorter_edge"])
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--masks", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.masks:
        for ds in a.datasets:
            run_masks(ds, cfg)
        return
    dev = setup_torch()
    enc = Encoder(a.encoder, cfg, dev, compile=not a.no_compile, sets=a.sets)
    meta = {"model": a.encoder, "timm": cfg["encoders"][a.encoder]["timm"], "img_size": enc.size, "mode": a.mode,
            "sets": {k: v for k, v in enc.net.sets.items()}, "config_hash_enc": config_hash(cfg["encoders"][a.encoder]),
            "batch_size": enc.batch_size}
    kind_suffix = (f"_{a.tag}" if a.tag else "")
    groups = []
    for ds in a.datasets:
        rows_all = load_index(ds, cfg)
        shots = unique_shot_rows(ds, cfg) if "shots" in a.kinds else {}
        for c in cfg["datasets"][ds]["categories"]:
            for kind in a.kinds:
                rows = [r for r in rows_all if r["category"] == c and r["split"] == "test"] if kind == "test" else shots[c]
                groups.append((store_dir(cfg, a.encoder, ds, c, kind + kind_suffix), rows))
    encode_groups(enc, groups, cfg, meta, mode=a.mode, profile=a.profile)


if __name__ == "__main__":
    main()

"""fp16 safetensors shards + JSON manifest; loaded via memory mapping (safe_open)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .common import ROOT, load_config


def store_dir(cfg: dict, encoder: str, dataset: str, category: str, kind: str) -> Path:
    return ROOT / cfg["paths"]["cache_root"] / encoder / dataset / category / kind


def save_shard(d: Path, shard: int, tensors: dict[str, torch.Tensor]) -> str:
    d.mkdir(parents=True, exist_ok=True)
    name = f"shard_{shard:04d}.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(d / name))
    return name


def write_manifest(d: Path, manifest: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "manifest.json.tmp"
    json.dump(manifest, open(tmp, "w"))
    tmp.replace(d / "manifest.json")


def read_manifest(d: Path) -> dict | None:
    p = d / "manifest.json"
    return json.load(open(p)) if p.exists() else None


def load_tensors(d: Path, keys: list[str] | None = None, device=None) -> dict[str, torch.Tensor]:
    """Concatenate all shards of a store along dim 0 (mmap-backed reads)."""
    man = read_manifest(d)
    if man is None:
        raise FileNotFoundError(d)
    out: dict[str, list] = {}
    for sh in man["shards"]:
        with safe_open(str(d / sh), framework="pt", device="cpu") as f:
            for k in (keys or f.keys()):
                out.setdefault(k, []).append(f.get_tensor(k))
    res = {k: torch.cat(v) if len(v) > 1 else v[0] for k, v in out.items()}
    if device is not None:
        res = {k: v.to(device, non_blocking=True) for k, v in res.items()}
    return res


def load_single(path: Path, keys: list[str] | None = None, device=None) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        res = {k: f.get_tensor(k) for k in (keys or f.keys())}
    if device is not None:
        res = {k: v.to(device, non_blocking=True) for k, v in res.items()}
    return res

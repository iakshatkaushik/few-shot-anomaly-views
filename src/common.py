"""Shared utilities: config, hashing, seeds, git commit, result rows, GPU logging."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before the first CUDA allocation
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict | None = None) -> str:
    cfg = cfg if cfg is not None else load_config()
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        c = out.stdout.strip() or "nocommit"
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return c + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_torch() -> torch.device:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def path(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.t0


@contextmanager
def gpu_step(name: str, log: list | None = None, n_items: int | None = None):
    """Log wall-clock, throughput and peak memory of a GPU step."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    rec = {"step": name}
    yield rec
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    rec.update(wall_s=round(dt, 4), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3))
    if n_items:
        rec["items_per_s"] = round(n_items / max(dt, 1e-9), 2)
    if log is not None:
        log.append(rec)
    print(json.dumps(rec), flush=True)


class ResultWriter:
    """Append-only JSONL result rows; every row carries config hash and commit."""

    def __init__(self, name: str, cfg: dict | None = None):
        self.cfg = cfg if cfg is not None else load_config()
        self.path = path(self.cfg, "results_dir") / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.base = {"config_hash": config_hash(self.cfg), "commit": git_commit()}

    def write(self, rows: list[dict]) -> None:
        with open(self.path, "a") as f:
            for r in rows:
                f.write(json.dumps({**self.base, **r}) + "\n")


def append_log(name: str, rec: dict, cfg: dict | None = None) -> None:
    cfg = cfg if cfg is not None else load_config()
    p = path(cfg, "logs_dir") / f"{name}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")

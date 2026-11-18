"""Shot draws: 10 draws x k in {1,2,4} per category, seed = 1000*draw + 10*k.
Shots are drawn without replacement from the sorted list of normal training images."""
from __future__ import annotations

import numpy as np

from .datasets import load_index


def draw_seed(draw: int, k: int) -> int:
    return 1000 * draw + 10 * k


def train_pool(rows: list[dict], category: str) -> list[dict]:
    return sorted((r for r in rows if r["category"] == category and r["split"] == "train" and r["label"] == 0),
                  key=lambda r: r["path"])


def shots(rows: list[dict], category: str, draw: int, k: int) -> list[dict]:
    pool = train_pool(rows, category)
    rng = np.random.default_rng(draw_seed(draw, k))
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in sorted(idx.tolist())]


def all_draws(dataset: str, cfg: dict) -> dict:
    """{(category, draw, k): [rows]}"""
    rows = load_index(dataset, cfg)
    out = {}
    for c in cfg["datasets"][dataset]["categories"]:
        for k in cfg["draws"]["shots"]:
            for d in range(cfg["draws"]["n_draws"]):
                out[(c, d, k)] = shots(rows, c, d, k)
    return out

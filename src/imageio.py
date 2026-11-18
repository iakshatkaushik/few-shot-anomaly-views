"""Image decoding on CPU workers, resizing on GPU in batches."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .common import ROOT

cv2.setNumThreads(1)


def read_rgb(path: str) -> np.ndarray:
    im = cv2.imread(str(ROOT / path), cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def read_mask(path: str | None, hw: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(hw, dtype=bool)
    m = cv2.imread(str(ROOT / path), cv2.IMREAD_GRAYSCALE)
    return m > 0


class ImageList(Dataset):
    """Returns uint8 HWC numpy decoded on a CPU worker; no resizing here."""

    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return i, torch.from_numpy(read_rgb(self.paths[i]))


def _collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def iter_resized(paths: list[str], cfg: dict, size: int, device, big_batch: int, mode: str = "square",
                 patch: int = 14, loader_batch: int | None = None):
    """Decode on CPU workers (small batches), resize on GPU, yield (indices, [B,3,H,W]) with B <= big_batch.
    Images of different output sizes are never mixed in one yielded batch."""
    if loader_batch is None:  # keep decoded batches in flight below ~2 GB of /dev/shm (4 GB on this host)
        h, w = read_rgb(paths[0]).shape[:2]
        ex = cfg["extract"]
        loader_batch = int(max(1, min(16, 2e9 / (ex["num_workers"] * ex["prefetch_factor"] * h * w * 3 * 1.5))))
    loader = make_loader(paths, loader_batch, cfg)
    acc: dict[tuple, tuple[list, list]] = {}
    for idx, imgs in loader:
        xs = gpu_resize(imgs, size, device, mode=mode, patch=patch, stack=False)
        for i, x in zip(idx, xs):
            ai, ax = acc.setdefault(tuple(x.shape), ([], []))
            ai.append(i)
            ax.append(x)
            if len(ai) >= big_batch:
                yield list(ai), torch.stack(ax)
                ai.clear()
                ax.clear()
    for ai, ax in acc.values():
        if ai:
            yield list(ai), torch.stack(ax)


def make_loader(paths: list[str], batch_size: int, cfg: dict) -> DataLoader:
    nw = cfg["extract"]["num_workers"]
    return DataLoader(ImageList(paths), batch_size=batch_size, shuffle=False, num_workers=nw,
                      pin_memory=True, persistent_workers=nw > 0, prefetch_factor=cfg["extract"]["prefetch_factor"] if nw > 0 else None,
                      collate_fn=_collate)


def target_hw(h: int, w: int, size: int, mode: str, patch: int) -> tuple[int, int]:
    if mode == "square":
        return size, size
    if mode == "shorter_edge":  # AnomalyDINO: shorter edge -> size, keep aspect, crop to multiple of patch
        if h <= w:
            nh, nw = size, int(round(w * size / h))
        else:
            nh, nw = int(round(h * size / w)), size
        return nh, nw
    raise ValueError(mode)


def gpu_resize(imgs: list[torch.Tensor], size: int, device, mode: str = "square", patch: int = 14,
               interp: str = "bicubic", stack: bool = True):
    """uint8 HWC list -> float [B,3,H,W] in [0,1] on GPU. Same-size images are resized together."""
    out = [None] * len(imgs)
    groups: dict[tuple, list[int]] = {}
    for i, im in enumerate(imgs):
        groups.setdefault(tuple(im.shape), []).append(i)
    for shp, idx_all in groups.items():
        th, tw = target_hw(shp[0], shp[1], size, mode, patch)
        chunk = max(1, int(2.5e8 // (shp[0] * shp[1] * 3)))  # bound the full-res float buffer (~1 GB)
        for c0 in range(0, len(idx_all), chunk):
            idx = idx_all[c0:c0 + chunk]
            # upload each pinned uint8 image separately (a CPU-side stack would be pageable), stack on GPU
            x = torch.stack([imgs[i].to(device, non_blocking=True) for i in idx])
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            if (th, tw) != (shp[0], shp[1]):
                x = F.interpolate(x, size=(th, tw), mode=interp, antialias=True, align_corners=False).clamp_(0, 1)
            if mode == "shorter_edge":
                x = x[:, :, : th - th % patch, : tw - tw % patch]
            for j, i in enumerate(idx):
                out[i] = x[j]
    if not stack:
        return out
    shapes = {tuple(o.shape) for o in out}
    if len(shapes) != 1:
        raise ValueError(f"mixed output sizes in one batch: {shapes}")
    return torch.stack(out)


def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s

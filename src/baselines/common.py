"""Shared helpers for re-implemented baselines: identical draws, identical evaluation (masks at eval_res,
same metric code as the proposed method), result rows with config hash / commit."""
from __future__ import annotations

import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..common import ROOT
from ..eval import image_metrics, pixel_metrics, region_labels
from ..featstore import load_single
from ..imageio import gpu_resize, read_rgb

cv2.setNumThreads(1)


class MaskSet:
    def __init__(self, cfg: dict, ds: str, cat: str, dev):
        mf = ROOT / cfg["paths"]["cache_root"] / "masks" / ds / f"{cat}.safetensors"
        m = load_single(mf, device=dev)
        self.paths = json.load(open(mf.with_suffix(".json")))
        self.masks, self.labels = m["masks"], m["labels"].cpu().numpy()
        self.res = cfg["method"]["eval_res"]
        self.limits = tuple(cfg["eval"]["pro_fpr"]) if ds == "mvtec_ad2" else (cfg["eval"]["pro_fpr"][0],)
        self.regions = region_labels(self.masks)
        from ..datasets import load_index
        sub = {r["path"]: r.get("subset", "") for r in load_index(ds, cfg)}
        self.subsets = [sub[p] for p in self.paths]
        self.loco = ds == "mvtec_loco"

    def evaluate(self, maps: torch.Tensor, img_scores: np.ndarray) -> dict:
        """maps: [T, H, W] (any res; resized bilinearly to eval_res if needed)."""
        if maps.shape[-1] != self.res or maps.shape[-2] != self.res:
            maps = F.interpolate(maps[:, None].float(), size=(self.res, self.res), mode="bilinear",
                                 align_corners=False)[:, 0]
        m = dict(image_metrics(img_scores, self.labels), **pixel_metrics(maps, self.masks, self.limits, self.regions))
        if self.loco:
            for sub in ("logical", "structural"):
                keep = np.array([s in (sub, "good") for s in self.subsets])
                m[f"{sub}_img_auroc"] = image_metrics(img_scores[keep], self.labels[keep])["img_auroc"]
                km = torch.tensor(keep, device=maps.device)
                pm = pixel_metrics(maps[km], self.masks[km], self.limits)
                m.update({f"{sub}_{k}": v for k, v in pm.items()})
        return m


def rotate_cv2(img: np.ndarray, angle: float, border=cv2.BORDER_DEFAULT) -> np.ndarray:
    """AnomalyDINO utils.rotate_image (rotation about the image centre, bilinear, reflect-101 border)."""
    c = tuple(np.array(img.shape[1::-1]) / 2)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    return cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=border)


class AugImages(Dataset):
    """(path, fn-spec) pairs; decoding and CPU augmentation in workers."""

    def __init__(self, items: list[tuple[str, dict]], fn):
        self.items, self.fn = items, fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, spec = self.items[i]
        return i, torch.from_numpy(np.ascontiguousarray(self.fn(read_rgb(p), spec)))


def iter_aug(items, fn, cfg, size, dev, mode, patch, big_batch, interp="bicubic"):
    """Yield (indices, resized [B,3,H,W]) for augmented images; groups by output shape."""
    from ..imageio import _collate
    ex = cfg["extract"]
    h, w = read_rgb(items[0][0]).shape[:2]
    lb = int(max(1, min(16, 2e9 / (ex["num_workers"] * ex["prefetch_factor"] * h * w * 3 * 1.5))))
    dl = DataLoader(AugImages(items, fn), batch_size=lb, num_workers=ex["num_workers"], pin_memory=True,
                    prefetch_factor=ex["prefetch_factor"], persistent_workers=False, collate_fn=_collate)
    acc: dict = {}
    for idx, imgs in dl:
        xs = gpu_resize(imgs, size, dev, mode=mode, patch=patch, interp=interp, stack=False)
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

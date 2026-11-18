"""Dataset indexing. Produces data/index/<dataset>.json with one entry per image:
{path, category, split (train|val|test), label (0 normal / 1 anomalous), mask (path or None), defect, subset}
`subset` is used for MVTec LOCO (logical / structural / good).
All paths are relative to the repository root.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import zlib
import io
from collections import Counter
from pathlib import Path

import numpy as np

from .common import ROOT, load_config

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT))


def _imgs(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def index_mvtec_like(root: Path, categories: list[str], mask_suffix: str = "_mask") -> list[dict]:
    """MVTec AD and MPDD layout: cat/train/good, cat/test/<type>, cat/ground_truth/<type>/<id>_mask.png"""
    out = []
    for c in categories:
        for p in _imgs(root / c / "train" / "good"):
            out.append(dict(path=_rel(p), category=c, split="train", label=0, mask=None, defect="good", subset="good"))
        for t in sorted(x.name for x in (root / c / "test").iterdir() if x.is_dir()):
            for p in _imgs(root / c / "test" / t):
                if t == "good":
                    out.append(dict(path=_rel(p), category=c, split="test", label=0, mask=None, defect="good", subset="good"))
                else:
                    m = root / c / "ground_truth" / t / f"{p.stem}{mask_suffix}.png"
                    if not m.exists():
                        raise FileNotFoundError(m)
                    out.append(dict(path=_rel(p), category=c, split="test", label=1, mask=_rel(m), defect=t, subset="anomaly"))
    return out


def index_loco(root: Path, categories: list[str]) -> list[dict]:
    """MVTec LOCO: ground_truth/<type>/<id>/000.png (possibly several masks per image -> merged mask written)."""
    from PIL import Image
    out = []
    merged_root = ROOT / "data" / "mvtec_loco_merged_masks"
    for c in categories:
        for split, sd in (("train", "train"), ("val", "validation")):
            for p in _imgs(root / c / sd / "good"):
                out.append(dict(path=_rel(p), category=c, split=split, label=0, mask=None, defect="good", subset="good"))
        for t in ("good", "logical_anomalies", "structural_anomalies"):
            for p in _imgs(root / c / "test" / t):
                if t == "good":
                    out.append(dict(path=_rel(p), category=c, split="test", label=0, mask=None, defect="good", subset="good"))
                    continue
                md = root / c / "ground_truth" / t / p.stem
                ms = _imgs(md)
                if not ms:
                    raise FileNotFoundError(md)
                mo = merged_root / c / t / f"{p.stem}.png"
                if not mo.exists():
                    acc = None
                    for m in ms:
                        a = np.array(Image.open(m)) > 0
                        acc = a if acc is None else (acc | a)
                    mo.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray((acc * 255).astype(np.uint8)).save(mo)
                out.append(dict(path=_rel(p), category=c, split="test", label=1, mask=_rel(mo), defect=t,
                                subset="logical" if t.startswith("logical") else "structural"))
    return out


def index_visa(root: Path, categories: list[str]) -> list[dict]:
    """VisA standard one-class split (split_csv/1cls.csv)."""
    out = []
    with open(root / "split_csv" / "1cls.csv") as f:
        for r in csv.DictReader(f):
            if r["object"] not in categories:
                continue
            lab = 0 if r["label"] == "normal" else 1
            m = (root / r["mask"]) if (lab and r["mask"]) else None
            if lab and not m.exists():
                raise FileNotFoundError(m)
            out.append(dict(path=_rel(root / r["image"]), category=r["object"], split=r["split"], label=lab,
                            mask=_rel(m) if m else None, defect="bad" if lab else "good",
                            subset="anomaly" if lab else "good"))
    return out


def _fo_mask(blob: dict) -> np.ndarray:
    """FiftyOne stores segmentation masks as zlib-compressed np.save bytes (base64 in JSON)."""
    raw = base64.b64decode(blob["$binary"]["base64"])
    return np.load(io.BytesIO(zlib.decompress(raw)), allow_pickle=False)


def index_ad2_fo(fo_root: Path, categories: list[str]) -> list[dict]:
    """MVTec AD 2 from the FiftyOne export (flattened file names, masks embedded in samples.json).
    Masks are decoded and written to data/mvtec_ad2_masks/<category>/<file>.png."""
    from PIL import Image
    samples = json.load(open(fo_root / "samples.json"))["samples"]
    split_map = {"train": "train", "validation": "val", "test_public": "test"}
    out = []
    mroot = ROOT / "data" / "mvtec_ad2_masks"
    for s in samples:
        c = s["category_label"]["label"]
        if c not in categories:
            continue
        split = split_map[s["folder_type"]]
        p = fo_root / s["filepath"]
        lab = 0 if s["defect_label"]["label"] == "good" else 1
        mpath = None
        if lab:
            mo = mroot / c / (Path(s["filepath"]).stem + ".png")
            if not mo.exists():
                m = _fo_mask(s["segmentation"]["mask"])
                mo.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(((m > 0) * 255).astype(np.uint8)).save(mo)
            mpath = _rel(mo)
        out.append(dict(path=_rel(p), category=c, split=split, label=lab, mask=mpath,
                        defect=s["defect_label"]["label"], subset=s.get("shift_type", "")))
    return out


ROOTS = {
    "mvtec_ad": "data/mvtec_ad",
    "mpdd": "data/mpdd_raw/MPDD",
    "visa": "data/visa_raw",
    "mvtec_loco": "data/mvtec_loco",
    "mvtec_ad2": "data/mvtec_ad2_fo",
}


def build(name: str, cfg: dict) -> list[dict]:
    cats = cfg["datasets"][name]["categories"]
    root = ROOT / ROOTS[name]
    if name in ("mvtec_ad", "mpdd"):
        rows = index_mvtec_like(root, cats)
    elif name == "mvtec_loco":
        rows = index_loco(root, cats)
    elif name == "visa":
        rows = index_visa(root, cats)
    elif name == "mvtec_ad2":
        rows = index_ad2_fo(root, cats)
    else:
        raise ValueError(name)
    for r in rows:
        if not (ROOT / r["path"]).exists():
            raise FileNotFoundError(r["path"])
    return rows


def load_index(name: str, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    return json.load(open(ROOT / cfg["paths"]["index_dir"] / f"{name}.json"))


def summary(rows: list[dict]) -> dict:
    c = Counter((r["split"], r["label"]) for r in rows)
    return {f"{s}_{'anom' if l else 'norm'}": v for (s, l), v in sorted(c.items())} | {"total": len(rows)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    a = ap.parse_args()
    cfg = load_config()
    od = ROOT / cfg["paths"]["index_dir"]
    od.mkdir(parents=True, exist_ok=True)
    for n in a.datasets:
        rows = build(n, cfg)
        json.dump(rows, open(od / f"{n}.json", "w"))
        print(n, json.dumps(summary(rows)))

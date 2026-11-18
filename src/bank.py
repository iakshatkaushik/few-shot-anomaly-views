"""Assemble all cached features for one (encoder, feature set, dataset, k, draw) into padded GPU tensors,
one row per category, so that all categories are trained and scored in a single batched pass."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .common import ROOT
from .featstore import load_single, load_tensors, store_dir


@dataclass
class Bank:
    cats: list
    textures: list            # bool per category
    V: int                    # views per shot (16)
    k: int
    g: int                    # grid side
    X: torch.Tensor           # [C, Nmax, D] normal view tokens (valid only), fp16
    n: torch.Tensor           # [C] valid rows
    row_view: torch.Tensor    # [C, Nmax] view index (0..k*V-1)
    row_canon: torch.Tensor   # [C, Nmax] canonical token index
    L: torch.Tensor           # [C, k*V, N] canonical token -> row (or -1)
    mu: torch.Tensor          # [C, D] mean of all normal tokens
    S: torch.Tensor           # [C, Mmax, D] synthetic candidate tokens
    s_cov: torch.Tensor       # [C, Mmax]
    s_chg: torch.Tensor       # [C, Mmax]
    s_valid: torch.Tensor     # [C, Mmax] bool (padding mask)
    s_family: torch.Tensor    # [C, Mmax]
    Rt: torch.Tensor | None   # [C, Rmax, D] valid rotation tokens
    r_n: torch.Tensor | None  # [C]
    r_img: torch.Tensor | None  # [C, Rmax] rotation image index (k*30)
    view_params: list = field(default_factory=list)


def _pad(tensors: list[torch.Tensor], value=0) -> torch.Tensor:
    m = max(t.shape[0] for t in tensors)
    out = []
    for t in tensors:
        pad = [0, 0] * (t.dim() - 1) + [0, m - t.shape[0]]
        out.append(F.pad(t, pad, value=value))
    return torch.stack(out)


def load_bank(cfg: dict, enc: str, fset: str, dataset: str, k: int, draw: int, device, with_rot: bool = True,
              with_synth: bool = True, cats: list | None = None) -> Bank:
    cats = cats or cfg["datasets"][dataset]["categories"]
    tex = set(cfg["datasets"][dataset]["textures"])
    V = cfg["views"]["n_views"]
    Xs, rv, rc, Ls, mus, Ss, cov, chg, sval, fam, Rs, rimg, vparams = [], [], [], [], [], [], [], [], [], [], [], [], []
    g = None
    for c in cats:
        vf = store_dir(cfg, enc, dataset, c, "views") / f"d{draw}_k{k}.safetensors"
        t = load_single(vf, [f"feat_{fset}", "idx"])
        F_, idx = t[f"feat_{fset}"].to(device, non_blocking=True), t["idx"].to(device).long()
        nv, N, D = F_.shape
        g = int(round(N ** 0.5))
        valid = idx >= 0
        vi, ti = valid.nonzero(as_tuple=True)
        X = F_[vi, ti]
        Lc = torch.full((nv, N), -1, dtype=torch.long, device=device)
        Lc[vi, idx[vi, ti]] = torch.arange(len(vi), device=device)
        Xs.append(X)
        rv.append(vi)
        rc.append(idx[vi, ti])
        Ls.append(Lc)
        mus.append(X.float().mean(0))
        vparams.append(json.load(open(vf.with_suffix(".json"))))
        if with_synth:
            sf = store_dir(cfg, enc, dataset, c, "synth") / f"d{draw}_k{k}.safetensors"
            s = load_single(sf, [f"feat_{fset}", "cov", "chg", "family"], device=device)
            Ss.append(s[f"feat_{fset}"])
            cov.append(s["cov"].float())
            chg.append(s["chg"].float())
            fam.append(s["family"].long())
            sval.append(torch.ones(len(s["cov"]), dtype=torch.bool, device=device))
        if with_rot:
            rf = store_dir(cfg, enc, dataset, c, "rot") / f"d{draw}_k{k}.safetensors"
            r = load_single(rf, [f"feat_{fset}", "valid"], device=device)
            ri, rt = r["valid"].nonzero(as_tuple=True)
            Rs.append(r[f"feat_{fset}"][ri, rt])
            rimg.append(ri)
    n = torch.tensor([x.shape[0] for x in Xs], device=device)
    b = Bank(cats=cats, textures=[c in tex for c in cats], V=V, k=k, g=g,
             X=_pad(Xs), n=n, row_view=_pad(rv), row_canon=_pad(rc), L=_pad(Ls, -1) if len({l.shape for l in Ls}) == 1 else None,
             mu=torch.stack(mus).half(),
             S=_pad(Ss) if with_synth else None, s_cov=_pad(cov) if with_synth else None,
             s_chg=_pad(chg) if with_synth else None, s_valid=_pad(sval, False) if with_synth else None,
             s_family=_pad(fam) if with_synth else None,
             Rt=_pad(Rs) if with_rot else None,
             r_n=torch.tensor([r.shape[0] for r in Rs], device=device) if with_rot else None,
             r_img=_pad(rimg, -1) if with_rot else None, view_params=vparams)
    if b.L is None:
        b.L = torch.stack(Ls)
    return b


def load_test(cfg: dict, enc: str, fset: str, dataset: str, category: str, device, kind: str = "test"):
    """Returns tokens [T, N, D] fp16 on device, image labels [T], masks [T, H, W] uint8 (eval_res), entries."""
    d = store_dir(cfg, enc, dataset, category, kind)
    t = load_tensors(d, [fset], device=device)
    man = json.load(open(d / "manifest.json"))
    mf = ROOT / cfg["paths"]["cache_root"] / "masks" / dataset / f"{category}.safetensors"
    m = load_single(mf, device=device)
    paths = json.load(open(mf.with_suffix(".json")))
    assert paths == [e["path"] for e in man["entries"]], "mask/feature order mismatch"
    return t[fset], m["labels"], m["masks"], man["entries"], tuple(man["grid"])

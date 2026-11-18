"""Full protocol for the proposed method, its controls and its ablations on cached features.

For each (dataset, k, draw), all categories at once:
  1. rotation gate per category from raw-feature LOVO scores (selfconsistency.py)
  2. train heads for the selection grid d in {64,128,256} x tau in {0.05,0.1,0.2} (3 batched runs, R=3 taus)
  3. self-consistency selection of (tau, d, q) per category (min LOVO dispersion)
  4. score the test set for every grid config; rows:
       ours        selected config (the method's number)
       ours_fixed  fixed defaults tau=0.1, d=128, q=1%
       ours_oracle best test image AUROC over the grid (flagged oracle=True; never used as the method)
       ours_pca    PCA-subspace residual in projected space (fixed defaults)
       C1_nohead   raw features, same views + gated rotations memory, same scoring rule
       C2_shots    raw features, memory = canonical shots (+ gated rotations), no views
       C3_randhead frozen random head (d=128), mean-shifted, same memory
  --ablation NAME trains a single fixed-default variant instead (see ABLATIONS).

usage: python -m src.run_ours --dataset mvtec_ad --ks 1 2 4 --draws 0-9
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .bank import load_bank, load_test
from .common import ResultWriter, append_log, load_config, seed_everything, setup_torch
from .eval import image_metrics, patch_to_map, pixel_metrics, region_labels, top_frac_mean
from .head import BatchedHeads
from .score import PCAResidual, knn_dist
from .selfconsistency import dispersion, gate, lovo_image_scores, rotation_image_scores, threshold_at_fpr
from .train import TrainSpec, train_heads_chunked as train_heads

ABLATIONS = {
    "fixed": dict(),                      # fixed defaults; also emits the raw no-head control (backbone/block ablations)
    "no_shift": dict(mean_shift=False),
    "neg_random": dict(negatives="random"),
    "neg_unfiltered": dict(negatives="unfiltered"),
    "neg_extreme": dict(negatives="extreme"),
    "neg_none": dict(negatives="none"),
    "d32": dict(d=32),
    "d512": dict(d=512),
    "part_off": dict(part_aware=False),
    "seed1": dict(seed=1),
    "seed2": dict(seed=2),
    "onefa": dict(onefa=True),
}


def head_slice(h: BatchedHeads, c: int):
    c = 0 if h.W1.shape[0] == 1 else c
    return h.W1[c], h.b1[c], h.W2[c], h.b2[c]


@torch.no_grad()
def proj(h: BatchedHeads, c: int, x: torch.Tensor, mu: torch.Tensor | None, chunk: int = 131072) -> torch.Tensor:
    """x: [n, D] -> [R, n, d] unit fp16 using the head(s) of category c."""
    W1, b1, W2, b2 = head_slice(h, c)
    g = lambda t: torch.matmul(F.gelu(torch.matmul(t, W1) + b1), W2) + b2
    gm = g(mu.float()[None, None, :]) if mu is not None else 0
    out = []
    for s in range(0, x.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(F.normalize(g(x[None, s:s + chunk]) - gm, dim=-1).half())
    return torch.cat(out, 1)


class CatEval:
    """Holds one category's test data and evaluates patch scores."""

    def __init__(self, cfg, enc, fset, ds, cat, dev):
        self.X, labels, masks, self.entries, self.grid = load_test(cfg, enc, fset, ds, cat, dev)
        self.labels = labels.cpu().numpy()
        self.masks = masks
        self.T, self.N, self.D = self.X.shape
        self.res = cfg["method"]["eval_res"]
        self.sigma = cfg["method"]["map_sigma"]
        self.limits = tuple(cfg["eval"]["pro_fpr"]) if ds == "mvtec_ad2" else (cfg["eval"]["pro_fpr"][0],)
        self.regions = region_labels(masks)
        self.subsets = [e.get("subset", "") for e in self.entries]
        self.flat = self.X.reshape(-1, self.D)

    def metrics(self, patch: torch.Tensor, fracs: list[float], subset_split: bool = False) -> dict[float, dict]:
        """patch: [T*N] -> {frac: metrics}. Pixel metrics are frac-independent."""
        p = patch.view(self.T, self.N)
        maps = patch_to_map(p, self.grid, self.res, self.sigma)
        pix = pixel_metrics(maps, self.masks, self.limits, self.regions)
        out = {}
        for f in fracs:
            sc = top_frac_mean(p, f).cpu().numpy()
            m = dict(image_metrics(sc, self.labels), **pix)
            if subset_split:  # MVTec LOCO: logical / structural evaluated separately against all good images
                for sub in ("logical", "structural"):
                    keep = np.array([s in (sub, "good") for s in self.subsets])
                    m[f"{sub}_img_auroc"] = image_metrics(sc[keep], self.labels[keep])["img_auroc"]
                    km = torch.tensor(keep, device=maps.device)
                    pm = pixel_metrics(maps[km], self.masks[km], self.limits)
                    m.update({f"{sub}_{kk}": vv for kk, vv in pm.items()})
            out[f] = m
        return out


def run_one(cfg, ds, k, draw, enc, fset, dev, writer, ablation=None, extra_tag=""):
    t_all = time.perf_counter()
    mc, sc = cfg["method"], cfg["selection"]
    torch.cuda.reset_peak_memory_stats()
    bank = load_bank(cfg, enc, fset, ds, k, draw, dev)
    C = len(bank.cats)
    fr_all = sc["quantile"]
    q0 = mc["top_frac"]
    base = dict(dataset=ds, k=k, draw=draw, encoder=enc, fset=fset, tag=extra_tag)
    # ---------------- 1. rotation gate (raw features)
    gates, ginfo, lovo_raw = [], [], []
    for c in range(C):
        n = int(bank.n[c])
        lv = lovo_image_scores(bank.X[c], bank.row_view[c], n, k * bank.V, [q0])[q0]
        rs = rotation_image_scores(bank.Rt[c], bank.r_img[c], int(bank.r_n[c]), bank.X[c], n, q0)
        o, info = gate(lv, rs, sc["gate_quantile"])
        gates.append(o)
        ginfo.append(info)
        lovo_raw.append(lv)
    # ---------------- 2. training
    t_tr = time.perf_counter()
    logs = {}
    if ablation is None:
        heads = {}
        for d in sc["d"]:
            lg = []
            heads[d] = train_heads(bank, TrainSpec(taus=sc["tau"], d=d, seed=0), cfg, lg)
            logs[d] = lg
        grid = [(d, t, r) for d in sc["d"] for r, t in enumerate(sc["tau"])]
    else:
        spec = dict(taus=[mc["tau"]], d=mc["d"], seed=0, mean_shift=mc["mean_shift"], negatives=mc["negatives"],
                    part_aware=mc["part_aware"])
        spec.update(ABLATIONS[ablation])
        lg = []
        h = train_heads(bank, TrainSpec(**spec), cfg, lg)
        heads = {spec["d"]: h}
        logs[spec["d"]] = lg
        grid = [(spec["d"], spec["taus"][0], 0)]
    train_s = time.perf_counter() - t_tr
    use_shift = mc["mean_shift"] if ablation is None else ABLATIONS[ablation].get("mean_shift", mc["mean_shift"])
    onefa = ablation == "onefa"
    mu_all = None
    if onefa:
        mu_all = ((bank.mu.float() * bank.n[:, None]).sum(0) / bank.n.sum()).half()
    # random head control
    rand_head = BatchedHeads(C, 1, bank.X.shape[2], mc["head_hidden"], mc["d"], seed=12345, device=dev) if ablation is None else None
    rows = []
    # ---------------- per category: selection + test scoring
    ev_cache = {}
    if onefa:  # one memory for all categories: union of every category's views (+ gated rotations), shared head
        h = heads[grid[0][0]]
        mem_union = [proj(h, 0, bank.X[c, :int(bank.n[c])], mu_all if use_shift else None)[0] for c in range(C)]
        mem_union += [proj(h, 0, bank.Rt[c, :int(bank.r_n[c])], mu_all if use_shift else None)[0] for c in range(C) if gates[c]]
        mem_union = torch.cat(mem_union)
        raw_union = torch.cat([bank.X[c, :int(bank.n[c])] for c in range(C)] +
                              [bank.Rt[c, :int(bank.r_n[c])] for c in range(C) if gates[c]])
    for c, cat in enumerate(bank.cats):
        ce = CatEval(cfg, enc, fset, ds, cat, dev)
        loco = ds == "mvtec_loco"
        n, nr = int(bank.n[c]), int(bank.r_n[c])
        cb = dict(base, category=cat, gate_open=gates[c], **{f"gate_{kk}": vv for kk, vv in ginfo[c].items()})
        if onefa:
            h = heads[grid[0][0]]
            zt = proj(h, 0, ce.flat, mu_all if use_shift else None)[0]
            s = knn_dist(zt, mem_union, mc["knn_k"])
            m = ce.metrics(s, [q0], loco)[q0]
            rows.append(dict(cb, method="ours_onefa", d=grid[0][0], tau=grid[0][1], q=q0, **m))
            s = knn_dist(ce.flat, raw_union, mc["knn_k"])
            m = ce.metrics(s, [q0], loco)[q0]
            rows.append(dict(cb, method="C1_nohead_onefa", q=q0, **m))
            continue
        mu_c = bank.mu[c] if use_shift else None
        # --- grid configs
        res_cfg = {}
        for (d, tau, r) in grid:
            h = heads[d]
            Zv = proj(h, c, bank.X[c, :n], mu_c)[r]
            lv = lovo_image_scores(Zv, bank.row_view[c], n, k * bank.V, fr_all)
            mem = Zv if not gates[c] else torch.cat([Zv, proj(h, c, bank.Rt[c, :nr], mu_c)[r]])
            zt = proj(h, c, ce.flat, mu_c)[r]
            s = knn_dist(zt, mem, mc["knn_k"])
            mets = ce.metrics(s, fr_all, loco)
            for q in fr_all:
                res_cfg[(d, tau, q)] = dict(mets[q], lovo_disp=dispersion(lv[q]),
                                            thr_fpr=threshold_at_fpr(lv[q], sc["threshold_fpr"]))
            if ablation is None and d == mc["d"] and tau == mc["tau"]:
                p = PCAResidual(mem.float(), 0.99)
                m = ce.metrics(p(zt), [q0], loco)[q0]
                rows.append(dict(cb, method="ours_pca", d=d, tau=tau, q=q0, pca_k=p.k, **m))
        if ablation is None:
            sel = min(res_cfg, key=lambda key: res_cfg[key]["lovo_disp"])
            fix = (mc["d"], mc["tau"], q0)
            orc = max(res_cfg, key=lambda key: res_cfg[key]["img_auroc"])
            rows.append(dict(cb, method="ours", d=sel[0], tau=sel[1], q=sel[2], selected=True, **res_cfg[sel]))
            rows.append(dict(cb, method="ours_fixed", d=fix[0], tau=fix[1], q=fix[2], **res_cfg[fix]))
            rows.append(dict(cb, method="ours_oracle", oracle=True, d=orc[0], tau=orc[1], q=orc[2], **res_cfg[orc]))
            for key, v in res_cfg.items():
                rows.append(dict(cb, method="ours_grid", d=key[0], tau=key[1], q=key[2], grid_point=True, **v))
            # --- controls
            raw_mem = bank.X[c, :n] if not gates[c] else torch.cat([bank.X[c, :n], bank.Rt[c, :nr]])
            s = knn_dist(ce.flat, raw_mem, mc["knn_k"])
            m = ce.metrics(s, [q0], loco)[q0]
            rows.append(dict(cb, method="C1_nohead", q=q0, **m))
            canon = bank.X[c, :n][bank.row_view[c, :n] % bank.V == 0]
            shot_mem = canon if not gates[c] else torch.cat([canon, bank.Rt[c, :nr]])
            s = knn_dist(ce.flat, shot_mem, mc["knn_k"])
            m = ce.metrics(s, [q0], loco)[q0]
            rows.append(dict(cb, method="C2_shots", q=q0, **m))
            Zr = proj(rand_head, c, bank.X[c, :n], bank.mu[c])[0]
            memr = Zr if not gates[c] else torch.cat([Zr, proj(rand_head, c, bank.Rt[c, :nr], bank.mu[c])[0]])
            s = knn_dist(proj(rand_head, c, ce.flat, bank.mu[c])[0], memr, mc["knn_k"])
            m = ce.metrics(s, [q0], loco)[q0]
            rows.append(dict(cb, method="C3_randhead", d=mc["d"], q=q0, **m))
        else:
            (d, tau, _), = grid
            rows.append(dict(cb, method=f"abl_{ablation}", d=d, tau=tau, q=q0, **res_cfg[(d, tau, q0)]))
            if ablation == "fixed":
                raw_mem = bank.X[c, :n] if not gates[c] else torch.cat([bank.X[c, :n], bank.Rt[c, :nr]])
                s = knn_dist(ce.flat, raw_mem, mc["knn_k"])
                rows.append(dict(cb, method="C1_nohead", q=q0, **ce.metrics(s, [q0], loco)[q0]))
    wall = time.perf_counter() - t_all
    peak = torch.cuda.max_memory_allocated() / 1e9
    for r in rows:
        r.update(train_s=round(train_s, 2), wall_s=round(wall, 2), peak_mem_gb=round(peak, 2),
                 seed=ABLATIONS.get(ablation, {}).get("seed", 0) if ablation else 0, ablation=ablation)
    writer.write(rows)
    rec = dict(base, ablation=ablation, n_rows=len(rows), train_s=round(train_s, 2), wall_s=round(wall, 2),
               peak_mem_gb=round(peak, 2), train_log={str(kk): v for kk, v in logs.items()})
    append_log("run_ours", rec, cfg)
    summ = {}
    for r in rows:
        if r["method"] in ("ours", "ours_fixed", "C1_nohead", "C3_randhead") or r["method"].startswith("abl_") \
                or r["method"].endswith("onefa"):
            summ.setdefault(r["method"], []).append(r["img_auroc"])
    print(json.dumps({**base, "ablation": ablation, "wall_s": round(wall, 1), "train_s": round(train_s, 1),
                      "peak_gb": round(peak, 1), **{kk: round(float(np.mean(v)), 2) for kk, v in summ.items()}}), flush=True)


def parse_range(s):
    out = []
    for p in s:
        if "-" in p:
            a, b = p.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--draws", nargs="+", default=["0-9"])
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--fset", default="b9_12")
    ap.add_argument("--ablation", default=None, choices=[None] + list(ABLATIONS))
    ap.add_argument("--out", default="ours")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_config()
    dev = setup_torch()
    seed_everything(cfg["seed"])
    w = ResultWriter(a.out, cfg)
    for k in a.ks:
        for d in parse_range(a.draws):
            run_one(cfg, a.dataset, k, d, a.encoder, a.fset, dev, w, a.ablation, a.tag)


if __name__ == "__main__":
    main()

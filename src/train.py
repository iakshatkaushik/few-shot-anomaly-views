"""Training of the projection heads for all categories of a dataset (and several replicas, e.g. the
tau grid or head seeds) in one batched run: AdamW lr 1e-3, 8,192 anchors per category per step,
300 steps, negatives re-mined every 50 steps."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .bank import Bank
from .head import BatchedHeads, project
from .loss import multi_positive_infonce, uniformity
from .mining import candidate_mask, mine


@dataclass
class TrainSpec:
    taus: list            # one replica per entry
    d: int
    mean_shift: bool = True
    negatives: str = "band"     # band | extreme | unfiltered | random | none
    part_aware: bool = True
    steps: int = 300
    seed: int = 0
    nn_positive: bool = False   # ablation: add the nearest normal token in each other shot as a positive
    onefa: bool = False         # one shared head for all categories, category-aware negatives


def _gather_rows(Z: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Z: [C,R,N,d]; rows: [C,*S] (clamped >= 0) -> [C,R,*S,d]"""
    C, R, N, d = Z.shape
    flat = rows.reshape(C, 1, -1, 1).expand(C, R, -1, d)
    return torch.gather(Z, 2, flat).reshape(C, R, *rows.shape[1:], d)


def train_heads(bank: Bank, spec: TrainSpec, cfg: dict, log: list | None = None) -> BatchedHeads:
    mcfg = cfg["method"]
    dev = bank.X.device
    # cast the cached fp16 tables to bf16 once (autocast would otherwise re-cast them at every step)
    import copy
    bank = copy.copy(bank)
    bank.X = bank.X.to(torch.bfloat16)
    if bank.S is not None:
        bank.S = bank.S.to(torch.bfloat16)
    C, Nmax, D = bank.X.shape
    R = len(spec.taus)
    tau = torch.tensor(spec.taus, device=dev, dtype=torch.float32)
    Ch = 1 if spec.onefa else C
    heads = BatchedHeads(Ch, R, D, mcfg["head_hidden"], spec.d, seed=spec.seed, device=dev)
    opt = torch.optim.AdamW(heads.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
    gen = torch.Generator(device=dev).manual_seed(10_000 + spec.seed)
    mu = bank.mu.to(torch.bfloat16) if spec.mean_shift else None
    if spec.onefa and mu is not None:  # one head: shift by the dataset-wide normal mean
        mu = (bank.mu.float() * bank.n[:, None]).sum(0, keepdim=True).div(bank.n.sum()).to(torch.bfloat16)

    def heads_apply(x, mu_):
        """x: [C, 1|R, n, D] -> [C,R,n,d]; a one-for-all head has leading dim 1 and broadcasts over categories."""
        return heads(x, mu_)

    B, Vt = mcfg["anchors"], bank.L.shape[1]
    n = bank.n
    use_neg = spec.negatives != "none" and bank.S is not None
    cand = candidate_mask(bank, spec.negatives, cfg) if use_neg else None
    if use_neg:  # bound the mining cost: keep at most max_mining_candidates per category (deterministic)
        cap = mcfg["max_mining_candidates"]
        if bank.S.shape[1] > cap:
            keep = torch.zeros_like(cand)
            for c in range(C):
                ids = cand[c].nonzero().squeeze(1)
                if len(ids) > cap:
                    ids = ids[torch.randperm(len(ids), generator=gen, device=dev)[:cap]]
                keep[c, ids] = True
            cand = keep
    m, disc = mcfg["m"], mcfg["band_discard"]
    texture = torch.tensor(bank.textures, device=dev)
    g = bank.g
    yy, xx = bank.row_canon // g, bank.row_canon % g
    shot_of_row = bank.row_view // bank.V
    neg_idx = neg_valid = None
    t0 = time.perf_counter()
    for step in range(spec.steps):
        if use_neg and step % mcfg["refresh_every"] == 0:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                Zt = torch.cat([heads_apply(bank.X[:, None, s:s + 32768], mu) for s in range(0, Nmax, 32768)], 2)
                Zs = torch.cat([heads_apply(bank.S[:, None, s:s + 32768], mu)
                                for s in range(0, bank.S.shape[1], 32768)], 2)
                neg_idx, neg_valid = mine(Zs, Zt, n, cand, spec.negatives, m, disc, gen)
                del Zt, Zs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # anchors (uniform over each category's valid normal tokens)
            a = (torch.rand(C, B, device=dev, generator=gen) * n[:, None]).long()
            a_view = torch.gather(bank.row_view, 1, a)
            a_can = torch.gather(bank.row_canon, 1, a)
            a_shot = a_view // bank.V
            # positives: the same canonical token in every other view of the same shot
            vv = a_shot[:, :, None] * bank.V + torch.arange(bank.V, device=dev)[None, None, :]          # [C,B,V]
            prow = bank.L[torch.arange(C, device=dev)[:, None, None], vv, a_can[:, :, None]]           # [C,B,V]
            pmask = (prow >= 0) & (vv != a_view[:, :, None])
            # project the whole normal table once (with grad), then gather anchors / positives / parts / unif
            need = [a[:, :, None], prow.clamp(min=0)]
            if spec.part_aware:
                Q = mcfg["part_negatives"]
                cy = torch.randint(0, g, (C, B, Q), device=dev, generator=gen)
                cx = torch.randint(0, g, (C, B, Q), device=dev, generator=gen)
                far = torch.maximum((cy - torch.gather(yy, 1, a)[:, :, None]).abs(),
                                    (cx - torch.gather(xx, 1, a)[:, :, None]).abs()) >= mcfg["part_min_dist"]
                qrow = bank.L[torch.arange(C, device=dev)[:, None, None], a_view[:, :, None].expand(-1, -1, Q), cy * g + cx]
                qmask = far & (qrow >= 0) & (~texture[:, None, None])
                need.append(qrow.clamp(min=0))
            U = mcfg["unif_subset"]
            u = (torch.rand(C, U, device=dev, generator=gen) * n[:, None]).long()
            need.append(u[:, :, None])
            sizes = [t.shape[1] * t.shape[2] for t in need]
            rows = torch.cat([t.reshape(C, -1) for t in need], 1)                                      # [C, Ntot]
            Ztab = heads_apply(bank.X[:, None], mu)                                                      # [C,R,Nmax,d]
            z = _gather_rows(Ztab, rows)                                                                 # [C,R,Ntot,d]
            parts = list(torch.split(z, sizes, dim=2))
            za = parts[0]
            zp = parts[1].reshape(C, R, B, bank.V, -1)
            negs = []
            if spec.part_aware:
                negs.append((parts[2].reshape(C, R, B, mcfg["part_negatives"], -1), qmask[:, None].expand(-1, R, -1, -1)))
            zu = parts[-1]
            if use_neg:
                sx = torch.gather(bank.S, 1, neg_idx.reshape(C, -1)[:, :, None].expand(-1, -1, D)).view(C, R, m, D)
                zn = heads_apply(sx, mu)
                negs.append((zn, neg_valid))
            if spec.onefa and C > 1:  # category-aware negatives: normal tokens of the other categories
                oc = (torch.arange(C, device=dev)[:, None] + 1 + torch.randint(0, C - 1, (C, 256), device=dev, generator=gen)) % C
                orow = (torch.rand(C, 256, device=dev, generator=gen) * n[oc]).long()
                xo = bank.X[oc, orow]                                                                    # [C,256,D]
                negs.append((heads_apply(xo[:, None], mu).expand(C, R, -1, -1), None))
            lcon = multi_positive_infonce(za, zp, pmask[:, None].expand(-1, R, -1, -1), negs, tau)
            lun = uniformity(zu.float())
            loss = (lcon + mcfg["unif_weight"] * lun).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log is not None and (step % 50 == 0 or step == spec.steps - 1):
            log.append({"step": step, "loss_con": float(lcon.mean()), "loss_unif": float(lun.mean()),
                        "t": round(time.perf_counter() - t0, 3)})
    return heads


def slice_bank(bank: Bank, idx: list[int]) -> Bank:
    import copy
    b = copy.copy(bank)
    for f in ("X", "n", "row_view", "row_canon", "L", "mu", "S", "s_cov", "s_chg", "s_valid", "s_family", "Rt", "r_n", "r_img"):
        v = getattr(bank, f)
        if v is not None:
            setattr(b, f, v[idx])
    b.cats = [bank.cats[i] for i in idx]
    b.textures = [bank.textures[i] for i in idx]
    return b


def train_heads_chunked(bank: Bank, spec: TrainSpec, cfg: dict, log: list | None = None,
                        budget_gb: float = 45.0) -> BatchedHeads:
    """Per-class heads are independent: when C x R x anchors x d is large, train them in category chunks
    (identical result per category up to RNG streams) to stay inside GPU memory. One-for-all is never chunked."""
    C = len(bank.cats)
    # measured: ~5.5 GB peak per category for 3 replicas, d=256, 8,192 anchors (VisA k=4: 62 GB for 12 categories)
    per_cat_gb = 5.5 * len(spec.taus) / 3 * max(spec.d, 128) / 256 * cfg["method"]["anchors"] / 8192
    chunk = C if spec.onefa else max(1, min(C, int(budget_gb // per_cat_gb)))
    if chunk >= C:
        return train_heads(bank, spec, cfg, log)
    parts = []
    for s in range(0, C, chunk):
        idx = list(range(s, min(C, s + chunk)))
        parts.append(train_heads(slice_bank(bank, idx), spec, cfg, log))
        torch.cuda.empty_cache()
    h = parts[0]
    for name in ("W1", "b1", "W2", "b2"):
        setattr(h, name, torch.nn.Parameter(torch.cat([getattr(p, name).data for p in parts], 0)))
    h.C = C
    return h

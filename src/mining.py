"""Hard-negative mining of synthetic anomaly tokens.

Candidates: tokens touched by a synthetic mask. Filter (a): coverage >= min_cov and mean absolute
intensity change >= min_change. Rank by cosine distance to the nearest normal token (all views, all shots)
in the *current projected space*. Modes:
  band       : filtered, drop hardest `band_discard` fraction, keep next m        (method default)
  extreme    : filtered, keep the hardest m
  unfiltered : no pixel-change filter (coverage > 0), band
  random     : filtered, m uniformly random
"""
from __future__ import annotations

import torch


@torch.no_grad()
def nn_sim(q: torch.Tensor, mem: torch.Tensor, mem_n: torch.Tensor, budget: float = 4e9) -> torch.Tensor:
    """q: [C,R,M,d], mem: [C,R,N,d] (unit, padded), mem_n: [C] valid rows -> max cosine [C,R,M].
    The query chunk is sized so that one similarity block stays below `budget` bytes."""
    C, R, N, _ = mem.shape
    chunk = max(128, int(budget / (C * R * N * 2)))
    pad = torch.arange(N, device=mem.device)[None, :] >= mem_n[:, None]          # [C,N]
    out = []
    for s in range(0, q.shape[2], chunk):
        sim = torch.matmul(q[:, :, s:s + chunk], mem.transpose(-1, -2))           # [C,R,m,N]
        sim.masked_fill_(pad[:, None, None, :], -2.0)
        out.append(sim.amax(-1))
    return torch.cat(out, dim=2)


def candidate_mask(bank, mode: str, cfg: dict) -> torch.Tensor:
    c = cfg["synth"]
    base = bank.s_valid & (bank.s_cov > 0)
    if mode == "unfiltered":
        return base
    return base & (bank.s_cov >= c["min_cov"]) & (bank.s_chg >= c["min_change"])


@torch.no_grad()
def mine(Zs: torch.Tensor, Zt: torch.Tensor, nt: torch.Tensor, cand: torch.Tensor, mode: str, m: int,
         discard: float, gen: torch.Generator):
    """Zs: projected candidates [C,R,M,d]; Zt: projected normal table [C,R,N,d]; cand: [C,M] bool.
    Returns idx [C,R,m] into the candidate axis and a validity mask [C,R,m]."""
    C, R, M, _ = Zs.shape
    if mode == "random":
        score = torch.rand(C, R, M, device=Zs.device, generator=gen)
    else:
        score = 1.0 - nn_sim(Zs, Zt, nt)                                            # distance to nearest normal
    score = score.masked_fill(~cand[:, None, :], float("inf"))
    order = score.argsort(dim=-1)                                                   # hardest (closest) first
    navail = cand.sum(-1)                                                           # [C]
    start = (navail.float() * discard).floor().long() if mode in ("band", "unfiltered") else torch.zeros_like(navail)
    pos = start[:, None] + torch.arange(m, device=Zs.device)[None, :]               # [C,m]
    valid = pos < navail[:, None]
    pos = pos.clamp(max=M - 1)
    idx = torch.gather(order, 2, pos[:, None, :].expand(C, R, m))
    return idx, valid[:, None, :].expand(C, R, m)

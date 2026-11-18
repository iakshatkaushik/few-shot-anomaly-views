"""Mean-shifted multi-positive patch InfoNCE + uniformity regulariser.

L_con(i) = -(1/|P(i)|) sum_{p in P(i)} log[ exp(z_i.z_p/tau) / (sum_{p' in P(i)} exp(z_i.z_p'/tau) + sum_{n in N(i)} exp(z_i.z_n/tau)) ]
L_unif   = log E_{i,j normal} exp(-2 ||z_i - z_j||^2)     (Wang and Isola, ICML 2020)
L        = mean_i L_con(i) + w_unif * L_unif
All tensors carry leading dims [C, R] (categories x replicas); tau has shape [R] or scalar.
"""
from __future__ import annotations

import torch


def multi_positive_infonce(za, zp, pmask, zn, tau):
    """za: [C,R,B,d]; zp: [C,R,B,P,d] with pmask [C,1|R,B,P] (bool);
    zn: list of (negatives [C,R,M,d] shared by all anchors, mask [C,R,M] or None)
                or (per-anchor negatives [C,R,B,Q,d], mask [C,1|R,B,Q] or None).
    The denominator log-sum-exp is computed per block and combined (no concatenation of the logit blocks).
    Returns per-(C,R) mean loss [C,R] over anchors with at least one positive."""
    t = tau.view(1, -1, 1, 1) if torch.is_tensor(tau) else tau
    za = (za.float() / t).to(za.dtype)        # fold 1/tau into the anchors: logits stay in bf16, no big casts
    sp = torch.matmul(zp, za.unsqueeze(-1)).squeeze(-1).float()                     # [C,R,B,P]
    sp = sp.masked_fill(~pmask, float("-inf"))
    lses = [torch.logsumexp(sp, dim=-1)]
    for item, mask in zn:
        if item.dim() == 4:              # shared negatives [C,R,M,d]
            s_ = torch.matmul(za, item.transpose(-1, -2))
            if mask is not None and not bool(mask.all()):
                s_ = s_.masked_fill(~mask[:, :, None, :], float("-inf"))
        else:                            # per-anchor negatives [C,R,B,Q,d]
            s_ = torch.matmul(item, za.unsqueeze(-1)).squeeze(-1)
            if mask is not None:
                s_ = s_.masked_fill(~mask, float("-inf"))
        lses.append(torch.logsumexp(s_, dim=-1).float())
    lse = torch.logsumexp(torch.stack(lses, dim=-1), dim=-1, keepdim=True)          # [C,R,B,1]
    lp = (sp - lse).masked_fill(~pmask, 0.0)
    npos = pmask.sum(-1).clamp(min=1)
    li = -lp.sum(-1) / npos                                                         # [C,R,B]
    valid = (pmask.sum(-1) > 0).float().expand_as(li)
    return (li * valid).sum(-1) / valid.sum(-1).clamp(min=1)


def uniformity(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """z: [C,R,U,d] unit vectors -> [C,R]; log mean_{i != j} exp(-t ||z_i - z_j||^2)."""
    sq = (2 - 2 * torch.matmul(z, z.transpose(-1, -2))).clamp(min=0)
    U = z.shape[-2]
    eye = torch.eye(U, device=z.device, dtype=torch.bool)
    e = torch.exp(-t * sq).masked_fill(eye, 0.0)
    return torch.log(e.sum((-1, -2)) / (U * (U - 1)))

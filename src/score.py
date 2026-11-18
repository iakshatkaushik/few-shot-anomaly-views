"""Scoring rules: kNN cosine distance to the normal memory (k=1 default),
PCA-subspace residual (SubspaceAD-style) as an alternative rule, top-q image aggregation,
anomaly maps = bilinear upsampling + Gaussian sigma 4. All chunked matmuls on GPU, no Python loops over patches."""
from __future__ import annotations

import torch

from .eval import top_frac_mean


@torch.no_grad()
def knn_dist(q: torch.Tensor, mem: torch.Tensor, k: int = 1, chunk: int = 16384,
             exclude: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """q: [Q,d], mem: [M,d] unit vectors (fp16/bf16) -> 1 - mean of top-k cosine similarities, [Q] float32.
    exclude=(q_group [Q], mem_group [M]): memory items with the same group id as the query are ignored
    (used for leave-one-view-out)."""
    out = []
    memT = mem.t().contiguous()
    for s in range(0, q.shape[0], chunk):
        sim = torch.matmul(q[s:s + chunk], memT)
        if exclude is not None:
            sim.masked_fill_(exclude[0][s:s + chunk, None] == exclude[1][None, :], -2.0)
        v = sim.topk(k, dim=1).values.float().mean(1) if k > 1 else sim.amax(1).float()
        out.append(1.0 - v)
    return torch.cat(out)


class PCAResidual:
    """PCA fitted on the normal memory, keeping components that explain `ev` of the variance;
    score = squared reconstruction residual (SubspaceAD 'reconstruction')."""

    def __init__(self, mem: torch.Tensor, ev: float = 0.99):
        x = mem.double()
        self.mu = x.mean(0)
        xc = x - self.mu
        cov = xc.t() @ xc / max(1, x.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(cov)
        evals, evecs = evals.flip(0), evecs.flip(1)
        cum = torch.cumsum(evals, 0) / evals.sum()
        self.k = int(torch.searchsorted(cum, torch.tensor([ev], dtype=cum.dtype, device=cum.device)).item()) + 1
        self.P = evecs[:, : self.k].float()
        self.mu = self.mu.float()

    @torch.no_grad()
    def __call__(self, q: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
        out = []
        for s in range(0, q.shape[0], chunk):
            x = q[s:s + chunk].float() - self.mu
            r = x - (x @ self.P) @ self.P.t()
            out.append((r * r).sum(1))
        return torch.cat(out)


def image_scores(patch: torch.Tensor, fracs: list[float]) -> dict[float, torch.Tensor]:
    """patch: [T, N] -> {frac: [T]}"""
    return {f: top_frac_mean(patch, f) for f in fracs}

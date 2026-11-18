"""Projection heads g: Linear(D_in -> 512) -> GELU -> Linear(512 -> d), batched over
(categories C) x (replicas R, e.g. different tau or seeds) and evaluated with broadcast matmuls,
so all heads of a dataset train in one forward/backward pass. Mean shift: z = normalise(g(x) - g(mu_c))."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedHeads(nn.Module):
    def __init__(self, C: int, R: int, d_in: int, hidden: int, d: int, seed: int = 0, device="cuda"):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)

        def lin(fan_in, fan_out):  # PyTorch nn.Linear default init (kaiming-uniform a=sqrt(5) -> U(+-1/sqrt(fan_in)))
            b = 1.0 / math.sqrt(fan_in)
            W = (torch.rand(C, R, fan_in, fan_out, generator=g) * 2 - 1) * b
            bias = (torch.rand(C, R, 1, fan_out, generator=g) * 2 - 1) * b
            return nn.Parameter(W.to(device)), nn.Parameter(bias.to(device))

        self.W1, self.b1 = lin(d_in, hidden)
        self.W2, self.b2 = lin(hidden, d)
        self.C, self.R, self.d = C, R, d

    def g(self, x: torch.Tensor) -> torch.Tensor:
        """x: [C, 1 or R, N, D_in] -> [C, R, N, d] (unnormalised)"""
        h = torch.matmul(x, self.W1)
        h = F.gelu(h + self.b1.to(h.dtype))       # keep activations in the autocast dtype (no fp32 promotion)
        o = torch.matmul(h, self.W2)
        return o + self.b2.to(o.dtype)

    def forward(self, x: torch.Tensor, mu: torch.Tensor | None) -> torch.Tensor:
        """x: [C, 1|R, N, D_in]; mu: [C, D_in] or None (no mean shift). Returns unit vectors [C, R, N, d]."""
        z = self.g(x)
        if mu is not None:
            z = z - self.g(mu[:, None, None, :])
        return F.normalize(z, dim=-1)

    def n_params_per_head(self) -> int:
        return sum(p[0, 0].numel() for p in (self.W1, self.b1, self.W2, self.b2))


@torch.no_grad()
def project(heads: BatchedHeads, x: torch.Tensor, mu: torch.Tensor | None, chunk: int = 65536) -> torch.Tensor:
    """Chunked inference projection of a large token set. x: [C, N, D_in] -> [C, R, N, d] fp16."""
    outs = []
    for s in range(0, x.shape[1], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(heads(x[:, None, s:s + chunk], mu).half())
    return torch.cat(outs, dim=2)

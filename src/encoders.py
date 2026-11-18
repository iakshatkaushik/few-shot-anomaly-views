"""Frozen ViT encoders (timm) producing named token sets.
bf16 autocast, TF32, fused SDPA attention (timm uses F.scaled_dot_product_attention),
torch.compile(max-autotune) at a fixed batch size chosen by an auto-probe."""
from __future__ import annotations

import json
import os
import time

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ROOT, append_log
from .imageio import normalize

os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
# real path (features/cache is a symlink): g++ invoked by Inductor breaks on paths containing spaces
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.realpath(str(ROOT / "features" / "cache")) + "/_inductor")


class TokenSets(nn.Module):
    """Wraps a timm ViT; forward(images in [0,1]) -> {set_name: [B, N, D]}"""

    def __init__(self, enc_name: str, cfg: dict, sets: list[str] | None = None):
        super().__init__()
        e = cfg["encoders"][enc_name]
        self.e = e
        self.name = enc_name
        self.model = timm.create_model(e["timm"], pretrained=True, img_size=e["img_size"], num_classes=0,
                                       dynamic_img_size=True)
        self.model.eval().requires_grad_(False)
        self.sets = {k: v for k, v in e["sets"].items() if sets is None or k in sets}
        blocks = sorted({b for s in self.sets.values() for b in s["blocks"]})
        self.idx = [b - 1 for b in blocks]  # timm indices are 0-based
        self.pos = {b: i for i, b in enumerate(blocks)}
        self.mean, self.std = cfg["extract"]["mean"], cfg["extract"]["std"]
        self.patch = e["patch"]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = normalize(x, self.mean, self.std)
        need_norm = any(s["norm"] for s in self.sets.values())
        need_raw = any(not s["norm"] for s in self.sets.values())
        if need_norm and need_raw:   # one backbone pass; apply the final norm only where needed
            raw = self.model.forward_intermediates(x, indices=self.idx, norm=False, output_fmt="NLC",
                                                   intermediates_only=True, stop_early=True)
            outs = {False: raw, True: [self.model.norm(t) for t in raw]}
        else:
            outs = {need_norm: self.model.forward_intermediates(x, indices=self.idx, norm=need_norm, output_fmt="NLC",
                                                                intermediates_only=True, stop_early=True)}
        res = {}
        for name, s in self.sets.items():
            feats = outs[s["norm"]]
            t = feats[self.pos[s["blocks"][0]]].float()
            for b in s["blocks"][1:]:          # running sum: no [n_blocks, B, N, D] fp32 stack
                t = t + feats[self.pos[b]].float()
            t = t / len(s["blocks"])
            if s["l2"]:
                t = F.normalize(t, dim=-1)
            res[name] = t.half()
        return res


class Encoder:
    def __init__(self, enc_name: str, cfg: dict, device, compile: bool = True, batch_size: int | None = None,
                 sets: list[str] | None = None):
        self.cfg = cfg
        self.device = device
        self.net = TokenSets(enc_name, cfg, sets).to(device)
        self.name = enc_name
        self.size = cfg["encoders"][enc_name]["img_size"]
        self.patch = cfg["encoders"][enc_name]["patch"]
        self.dim = cfg["encoders"][enc_name]["dim"]
        self.batch_size = batch_size or self.probe()
        self.fn = self.net
        self.compiled = False
        if compile:
            self.fn = torch.compile(self.net, mode=cfg["extract"]["compile_mode"], dynamic=False)
            self.compiled = True

    @torch.inference_mode()
    def _run(self, fn, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return fn(x)

    def probe(self, hw: tuple[int, int] | None = None) -> int:
        """Increase the batch until peak memory exceeds the limit (or OOM / max), then back off one step."""
        ex = self.cfg["extract"]
        hw = hw or (self.size, self.size)
        limit = ex["probe_mem_limit_gb"] * 1e9
        b, best, log = ex["probe_start_batch"], None, []
        while b <= ex["probe_max_batch"]:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                x = torch.rand(b, 3, *hw, device=self.device)
                self._run(self.net, x[: min(b, 8)])  # warm-up
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                self._run(self.net, x)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated()
                log.append({"batch": b, "peak_gb": round(peak / 1e9, 2), "img_s_eager": round(b / dt, 1)})
                del x
                if peak > limit:
                    break
                best = b
                b *= 2
            except torch.cuda.OutOfMemoryError:
                log.append({"batch": b, "oom": True})
                break
        torch.cuda.empty_cache()
        best = best or ex["probe_start_batch"]
        rec = {"encoder": self.name, "hw": hw, "probe": log, "chosen": best}
        print(json.dumps(rec), flush=True)
        append_log("batch_probe", rec, self.cfg)
        return best

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [n,3,H,W] float in [0,1] on GPU (any n). Pads to the fixed batch for the compiled graph."""
        n = x.shape[0]
        B = self.batch_size
        outs = []
        s = 0
        while s < n:
            B = self.batch_size
            xb = x[s:s + B]
            m = xb.shape[0]
            if self.compiled and m < B:
                xb = torch.cat([xb, xb[-1:].expand(B - m, *xb.shape[1:])])
            try:
                o = self._run(self.fn, xb)
            except torch.cuda.OutOfMemoryError:
                # compiled graphs (max-autotune + CUDA graphs) need more memory than the eager probe measured:
                # back off by halving the batch and recompiling
                if B <= 1:
                    raise
                del xb
                torch._dynamo.reset()
                torch.cuda.empty_cache()
                self.batch_size = B // 2
                self.fn = torch.compile(self.net, mode=self.cfg["extract"]["compile_mode"], dynamic=False) \
                    if self.compiled else self.net
                rec = {"encoder": self.name, "compiled_oom_backoff": B, "new_batch": self.batch_size}
                print(json.dumps(rec), flush=True)
                append_log("batch_probe", rec, self.cfg)
                continue
            outs.append({k: v[:m].clone() for k, v in o.items()})
            s += m
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}

    def grid(self, hw: tuple[int, int] | None = None) -> tuple[int, int]:
        h, w = hw or (self.size, self.size)
        return h // self.patch, w // self.patch

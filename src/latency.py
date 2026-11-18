"""Latency (ms / image at batch size 1, after warm-up, CUDA events) and per-category memory footprint
for every method on the same GPU, same batch size (1), each at its own published input resolution.
All encoders run eagerly in bf16 autocast (no torch.compile) so that timing is comparable.
Category: MVTec AD 'bottle', k = 1, draw 0. Writes results/latency.json.

usage: python -m src.latency
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from .bank import load_bank
from .common import ROOT, config_hash, git_commit, load_config, setup_torch
from .datasets import load_index
from .draws import shots
from .encoders import Encoder
from .eval import patch_to_map, top_frac_mean
from .head import BatchedHeads
from .imageio import gpu_resize, read_rgb
from .score import PCAResidual, knn_dist

N_IMG, WARM = 50, 10


def timeit(fn, xs):
    for x in xs[:WARM]:
        fn(x)
    torch.cuda.synchronize()
    ts = []
    for x in xs[WARM:WARM + N_IMG]:
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn(x)
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.mean(ts)), float(np.std(ts))


def main():
    cfg = load_config()
    dev = setup_torch()
    ds, cat, k, d = "mvtec_ad", "bottle", 1, 0
    rows = [r for r in load_index(ds, cfg) if r["category"] == cat and r["split"] == "test"][:N_IMG + WARM]
    raw = [torch.from_numpy(read_rgb(r["path"])).pin_memory() for r in rows]
    res = cfg["method"]["eval_res"]
    out = {"config_hash": config_hash(cfg), "commit": git_commit(), "gpu": torch.cuda.get_device_name(),
           "dataset": ds, "category": cat, "k": k, "draw": d, "batch_size": 1, "methods": {}}

    def rec(name, fn, mem_bytes, note=""):
        torch.cuda.reset_peak_memory_stats()
        m, s = timeit(fn, raw)
        out["methods"][name] = {"ms_per_image": round(m, 2), "ms_std": round(s, 2), "memory_mb": round(mem_bytes / 2**20, 2),
                                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "note": note}
        print(name, out["methods"][name], flush=True)

    # ---- ours (ViT-B/14 448, head d=128, kNN) and C1 raw
    enc = Encoder("dinov2_vitb14", cfg, dev, compile=False, batch_size=1, sets=["b9_12"])
    bank = load_bank(cfg, "dinov2_vitb14", "b9_12", ds, k, d, dev, with_synth=False, cats=[cat])
    n = int(bank.n[0])
    head = BatchedHeads(1, 1, 768, 512, 128, device=dev)
    from .run_ours import proj
    mem = proj(head, 0, bank.X[0, :n], bank.mu[0])[0]
    g = enc.grid()

    def ours(x):
        f = enc(gpu_resize([x], enc.size, dev))["b9_12"][0]
        s = knn_dist(proj(head, 0, f, bank.mu[0])[0], mem, 1)[None]
        top_frac_mean(s, 0.01)
        patch_to_map(s, g, res, 4)
    rec("ours (ViT-B/14 448 + head d=128)", ours, mem.numel() * 2 + sum(p.numel() for p in head.parameters()) * 4,
        "memory = 16 views x valid tokens x 128 (fp16) + head parameters (fp32)")
    memr = bank.X[0, :n]

    def c1(x):
        f = enc(gpu_resize([x], enc.size, dev))["b9_12"][0]
        s = knn_dist(f, memr, 1)[None]
        top_frac_mean(s, 0.01)
        patch_to_map(s, g, res, 4)
    rec("C1 no head (ViT-B/14 448, raw 768-d)", c1, memr.numel() * 2)
    del enc

    # ---- AnomalyDINO (ViT-S/14, shorter edge 448)
    enc = Encoder("dinov2_vits14", cfg, dev, compile=False, batch_size=1, sets=["last_raw"])
    sh = shots(load_index(ds, cfg), cat, d, k)
    from .baselines.common import rotate_cv2
    refs = []
    for r in sh:
        im = read_rgb(r["path"])
        for a in cfg["baselines"]["anomalydino"]["rotations"]:
            x = gpu_resize([torch.from_numpy(np.ascontiguousarray(rotate_cv2(im, a)))], 448, dev, mode="shorter_edge")
            refs.append(F.normalize(enc(x)["last_raw"][0].float(), dim=-1).half())
    mad = torch.cat(refs)

    def adino(x):
        xx = gpu_resize([x], 448, dev, mode="shorter_edge")
        f = enc(xx)["last_raw"][0]
        s = knn_dist(F.normalize(f.float(), dim=-1).half(), mad, 1)[None]
        top_frac_mean(s, 0.01)
        patch_to_map(s, (xx.shape[2] // 14, xx.shape[3] // 14), res, 4)
    rec("AnomalyDINO (ViT-S/14 448)", adino, mad.numel() * 2, "rotation-augmented memory; masking off for bottle")
    del enc

    # ---- SubspaceAD (ViT-G/14-reg 672, PCA)
    enc = Encoder("dinov2_vitg14_reg", cfg, dev, compile=False, batch_size=1, sets=["sub23_29"])
    from .baselines.subspacead import post_process, rot_nearest
    rng = np.random.default_rng(0)
    tr = []
    for r in sh:
        im = read_rgb(r["path"])
        for a in [None] + list(rng.uniform(0, 345, 30)):
            xx = gpu_resize([torch.from_numpy(np.ascontiguousarray(rot_nearest(im, {"angle": a})))], 672, dev)
            tr.append(enc(xx)["sub23_29"][0].float())
    pca = PCAResidual(torch.cat(tr), 0.99)

    def sub(x):
        f = enc(gpu_resize([x], 672, dev))["sub23_29"][0]
        mp = post_process(pca(f)[None], (48, 48))
        mp.flatten(1).topk(int(672 * 672 * 0.01), 1).values.mean(1)
    rec("SubspaceAD (ViT-G/14-reg 672, PCA)", sub, (pca.P.numel() + pca.mu.numel()) * 4, f"PCA k={pca.k}")
    del enc

    # ---- PatchCore (WRN50-2 256)
    from .baselines.patchcore import WRN, kcenter, scores
    net = WRN(dev)
    ref = torch.cat([net(gpu_resize([torch.from_numpy(read_rgb(r["path"]))], 256, dev))[0][0] for r in sh])
    pmem = kcenter(ref, 0.1, torch.Generator(device=dev).manual_seed(0))

    def pc(x):
        e, gr = net(gpu_resize([x], 256, dev))
        ps, _ = scores(e, pmem, 9)
        patch_to_map(ps, gr, res, 4)
    rec("PatchCore (WRN50-2 256, coreset 10%)", pc, pmem.numel() * 4)

    # ---- WinCLIP+ (ViT-B/16+ 240)
    from .baselines.winclip import WinCLIP
    wc = WinCLIP(dev)
    txt = wc.text(cat)
    r_img, r_w, r_p = wc.encode(gpu_resize([torch.from_numpy(read_rgb(sh[0]["path"]))], 240, dev))
    R = (None, [w_[None] for w_ in r_w], r_p[None])

    def win(x):
        e = wc.encode(gpu_resize([x], 240, dev))
        wc.score(e, txt, R)
    rec("WinCLIP+ (ViT-B/16+ 240)", win, sum(t.numel() for t in r_w) * 4 + r_p.numel() * 4)
    del wc
    torch.cuda.empty_cache()

    # ---- VMF (DINOv3-B @448, DINOv3-B @672, DINOv2 ViT-G @672; view+rotation memories; fused map)
    from .run_fusion import SCORERS, memory_on_the_fly
    from .draws import all_draws, draw_seed
    from features.views import load_shot_images
    from .explore_fusion import gauss
    draws = all_draws(ds, cfg)
    vm, vmem = {}, 0
    for n_, e_, f_ in SCORERS:
        enc_ = Encoder(e_, cfg, dev, compile=False, batch_size=64, sets=[f_])
        imgs = load_shot_images(ds, cat, {(cat, d, k): draws[(cat, d, k)]}, enc_, cfg)
        X, rv, R = memory_on_the_fly(enc_, f_, imgs, draws[(cat, d, k)], draw_seed(d, k), cfg)
        lv = knn_dist(X, X, 1, exclude=(rv, rv)).mean()
        enc_.batch_size = 1
        vm[n_] = (enc_, f_, torch.cat([X, R]), lv)
        vmem += torch.cat([X, R]).numel() * 2

    def vmf(x):
        maps = []
        for n_, (enc_, f_, mem_, lv) in vm.items():
            f = enc_(gpu_resize([x], enc_.size, dev))[f_][0]
            g_ = enc_.grid()
            s_ = knn_dist(f, mem_, 1).view(1, 1, *g_) / lv
            maps.append(F.interpolate(s_, size=(res, res), mode="bilinear", align_corners=False)[:, 0])
        m_ = gauss(torch.stack(maps).mean(0))
        m_.flatten(1).topk(int(res * res * 0.01), 1).values.mean(1)
    rec("VMF (DINOv3-B 448 + DINOv3-B 672 + ViT-G 672)", vmf, vmem, "three view memories (16 views + 30 rotations, fp16)")
    del vm
    torch.cuda.empty_cache()

    # ---- Flownomaly (ViT-G 672 + velocity MLP, single-step recovery); untrained net has the same cost as a fitted one
    from .baselines.flownomaly import LAYERS, Velocity
    enc_ = Encoder("dinov2_vitg14_reg", cfg, dev, compile=False, batch_size=1, sets=LAYERS)
    net = Velocity().to(dev).eval()
    stats = {l: (torch.zeros(1, 1536, device=dev), torch.ones(1, 1536, device=dev)) for l in LAYERS}

    @torch.no_grad()
    def flow(x):
        o = enc_(gpu_resize([x], 672, dev))
        q = torch.stack([(o[l][0].float() - stats[l][0]) / (stats[l][1] + 1e-6) for l in LAYERS]).mean(0)
        nz = torch.randn_like(q)
        st = 0.5 * q + 0.5 * nz
        with torch.amp.autocast("cuda"):
            v = net(st, torch.full((len(q), 1), 0.5, device=dev))
        ((q - (st + 0.5 * v.float())) ** 2).sum(-1)
    rec("Flownomaly (ViT-G 672 + flow MLP)", flow, sum(p_.numel() for p_ in net.parameters()) * 4,
        "per-category velocity network parameters (fp32)")
    json.dump(out, open(ROOT / "results" / "latency.json", "w"), indent=1)


if __name__ == "__main__":
    main()

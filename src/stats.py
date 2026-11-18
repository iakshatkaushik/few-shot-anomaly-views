"""Statistics over shot draws.

For each (dataset, k, metric) and method: category-averaged score per draw -> mean, std, 10th percentile
over the draws; paired Wilcoxon signed-rank test over draws of the category-averaged score against every
baseline/control, Holm-corrected across the comparisons of one (dataset, k, metric); bootstrap 95% CIs over draws;
mean paired differences. Also per-category draw std (variance figure).

usage: python -m src.stats            -> results/summary.csv, results/tests.csv, results/per_category.csv, results/*.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .common import ROOT, config_hash, git_commit, load_config

METRICS = ["img_auroc", "img_ap", "px_auroc", "aupro_30", "aupro_5", "px_ap", "px_f1max",
           "logical_img_auroc", "structural_img_auroc", "logical_aupro_30", "structural_aupro_30"]


# methods in the Holm family: all baselines and controls, the contrastive head and VMF+H
TEST_FAMILY = {"SubspaceAD", "AnomalyDINO", "Flownomaly", "VisionAD", "WinCLIP+", "PatchCore", "PatchCore-cs1",
               "C1_nohead", "C2_shots", "C3_randhead", "ours", "VMF+H"}


def load_rows(heldout: bool = True) -> pd.DataFrame:
    """heldout=True: for MVTec AD and VisA at k=1, draws 0-4 were used to pick VMF,
    so every method is compared on draws 5-9 there; all other (dataset, k) use draws 0-9."""
    rows = []
    for name in ("ours", "baselines", "ablations", "fusion"):  # conformal.jsonl summarised separately
        p = ROOT / "results" / f"{name}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in open(p)]
    df = pd.DataFrame(rows)
    if "oracle" not in df:
        df["oracle"] = False
    df["oracle"] = df["oracle"].fillna(False)
    if "grid_point" in df:
        df = df[df["grid_point"].isna() | (df["grid_point"] == False)]  # noqa: E712
    if "tag" in df:
        df["tag"] = df["tag"].fillna("")
        df.loc[df["tag"] != "", "method"] = df["method"] + "@" + df["tag"]
    if "encoder" in df and "fset" in df:
        alt = df["method"].str.startswith(("ours", "C1", "C2", "C3", "abl_")) & (
            (df["encoder"] != "dinov2_vitb14") | (df["fset"].fillna("b9_12") != "b9_12"))
        df.loc[alt, "method"] = df.loc[alt, "method"] + "@" + df.loc[alt, "encoder"] + "/" + df.loc[alt, "fset"]
    # keep the latest row for duplicated (method, dataset, category, k, draw)
    df = df.drop_duplicates(["method", "dataset", "category", "k", "draw"], keep="last")
    if heldout:
        dev = df["dataset"].isin(["mvtec_ad", "visa"]) & (df["k"] == 1) & (df["draw"] < 5)
        df = df[~dev]
    return df


def holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return float("nan"), float("nan")
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def draw_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """[method, dataset, k, draw] -> category-averaged metric (only complete category sets)."""
    d = df.dropna(subset=[metric])
    ncat = d.groupby(["dataset"])["category"].nunique()
    g = d.groupby(["method", "dataset", "k", "draw"]).agg(v=(metric, "mean"), n=("category", "nunique")).reset_index()
    g = g[g["n"] == g["dataset"].map(ncat)]
    return g


def summarise(df: pd.DataFrame, ref: str = "VMF") -> tuple[pd.DataFrame, pd.DataFrame]:
    summ, tests = [], []
    for metric in METRICS:
        if metric not in df:
            continue
        g = draw_means(df, metric)
        for (ds, k), gg in g.groupby(["dataset", "k"]):
            piv = gg.pivot(index="draw", columns="method", values="v")
            for meth in piv.columns:
                x = piv[meth].dropna().values
                lo, hi = boot_ci(x)
                summ.append(dict(dataset=ds, k=k, metric=metric, method=meth, n_draws=len(x), mean=x.mean(),
                                 std=x.std(ddof=1) if len(x) > 1 else np.nan, p10=np.percentile(x, 10),
                                 ci_lo=lo, ci_hi=hi))
            if ref not in piv:
                continue
            min_draws = 5
            comps = []
            for meth in piv.columns:
                if meth == ref or meth not in TEST_FAMILY:
                    continue
                pair = piv[[ref, meth]].dropna()
                if len(pair) < min_draws:
                    continue
                diff = pair[ref].values - pair[meth].values
                try:
                    p = wilcoxon(pair[ref], pair[meth], zero_method="wilcox", alternative="two-sided").pvalue
                except ValueError:
                    p = 1.0
                lo, hi = boot_ci(diff)
                comps.append(dict(dataset=ds, k=k, metric=metric, ref=ref, other=meth, n_draws=len(pair),
                                  mean_diff=diff.mean(), diff_ci_lo=lo, diff_ci_hi=hi, p=p))
            if comps:
                adj = holm(np.array([c["p"] for c in comps]))
                for c, a in zip(comps, adj):
                    c["p_holm"] = a
                    c["significant"] = bool(a < 0.05)
                tests += comps
    return pd.DataFrame(summ), pd.DataFrame(tests)


def per_category(df: pd.DataFrame) -> pd.DataFrame:
    keep = [m for m in METRICS if m in df]
    g = df.groupby(["method", "dataset", "category", "k"])
    out = g[keep].agg(["mean", "std", lambda s: np.percentile(s.dropna(), 10) if s.notna().any() else np.nan,
                       "count"])
    out.columns = [f"{a}_{'p10' if b == '<lambda_0>' else b}" for a, b in out.columns]
    return out.reset_index()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_draws", action="store_true", help="include development draws (supplementary only)")
    a = ap.parse_args()
    cfg = load_config()
    df = load_rows(heldout=not a.all_draws)
    s, t = summarise(df)
    pc = per_category(df)
    meta = {"config_hash": config_hash(cfg), "commit": git_commit()}
    for name, frame in (("summary", s), ("tests", t), ("per_category", pc)):
        frame = frame.assign(**meta)
        suf = "_alldraws" if a.all_draws else ""
        frame.to_csv(ROOT / "results" / f"{name}{suf}.csv", index=False)
        frame.to_json(ROOT / "results" / f"{name}{suf}.json", orient="records", indent=0)
    long = df.melt(id_vars=[c for c in ("method", "dataset", "category", "k", "draw", "seed", "config_hash", "commit",
                                         "wall_s", "peak_mem_gb") if c in df],
                   value_vars=[m for m in METRICS if m in df], var_name="metric", value_name="value").dropna(subset=["value"])
    long.to_csv(ROOT / "results" / ("all_rows_long_alldraws.csv" if a.all_draws else "all_rows_long.csv"), index=False)
    print(f"summary rows {len(s)}, tests {len(t)}, per-category {len(pc)}, long rows {len(long)}")


if __name__ == "__main__":
    main()

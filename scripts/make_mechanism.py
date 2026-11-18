"""Aggregate results/mechanism.jsonl and results/sensitivity.jsonl into: results/mechanism_summary.csv,
results/sensitivity_summary.csv, outputs/tables/table_mech.tex, outputs/numbers_mech.tex (macros \\Mx{dataset|stat}) and
figures/fig3_sensitivity.pdf. Statistics: per-category means over draws; paired Wilcoxon over categories (raw vs head);
share of (category, draw) pairs in which the head is lower than raw."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DS = {"mvtec_ad": "MVTec AD", "visa": "VisA", "mpdd": "MPDD", "mvtec_loco": "LOCO", "mvtec_ad2": "MVTec AD 2"}


def mechanism():
    df = pd.DataFrame([json.loads(l) for l in open(ROOT / "results" / "mechanism.jsonl")])
    df = df.drop_duplicates(["dataset", "category", "k", "draw"], keep="last")
    pairs = [("patch_auroc", "raw_patch_auroc", "head_patch_auroc"), ("gap", "raw_gap", "head_gap"),
             ("inv_pos", "raw_inv_pos", "head_inv_pos"), ("inv_neg", "raw_inv_neg", "head_inv_neg"),
             ("img_top1", "raw_img_auroc_top1", "head_img_auroc_top1"), ("img_mean", "raw_img_auroc_mean", "head_img_auroc_mean"),
             ("logical_top1", "raw_logical_img_auroc_top1", "head_logical_img_auroc_top1"),
             ("logical_mean", "raw_logical_img_auroc_mean", "head_logical_img_auroc_mean"),
             ("structural_top1", "raw_structural_img_auroc_top1", "head_structural_img_auroc_top1"),
             ("structural_mean", "raw_structural_img_auroc_mean", "head_structural_img_auroc_mean")]
    rows = []
    for (ds, k), g in df.groupby(["dataset", "k"]):
        for name, a, b in pairs:
            if a not in g or g[a].isna().all():
                continue
            gc = g.groupby("category")[[a, b]].mean().dropna()
            p = wilcoxon(gc[a], gc[b]).pvalue if len(gc) >= 5 and (gc[a] != gc[b]).any() else np.nan
            share = float((g[b] < g[a]).mean())
            rows.append(dict(dataset=ds, k=k, stat=name, raw=gc[a].mean(), head=gc[b].mean(), p_categories=p,
                             share_head_lower=share, n_categories=len(gc), n_draws=g["draw"].nunique()))
    if "spearman_anom" in df:
        for (ds, k), g in df.groupby(["dataset", "k"]):
            rows.append(dict(dataset=ds, k=k, stat="spearman_anom", raw=np.nan, head=g["spearman_anom"].mean(),
                             p_categories=np.nan, share_head_lower=np.nan, n_categories=g["category"].nunique(),
                             n_draws=g["draw"].nunique()))
    # pooled over all categories of all datasets (one value per category: mean over its draws), raw vs head
    for k, g in df.groupby("k"):
        for name, a, b in pairs[:6]:
            gc = g.groupby(["dataset", "category"])[[a, b]].mean().dropna()
            if len(gc) >= 5:
                rows.append(dict(dataset="all", k=k, stat=name, raw=gc[a].mean(), head=gc[b].mean(),
                                 p_categories=wilcoxon(gc[a], gc[b]).pvalue, share_head_lower=float((gc[b] < gc[a]).mean()),
                                 n_categories=len(gc), n_draws=g["draw"].nunique()))
    S = pd.DataFrame(rows)
    S.to_csv(ROOT / "results" / "mechanism_summary.csv", index=False)
    return S


def table_and_macros(S):
    k1 = S[S.k == 1]
    get = lambda ds, st, col: (k1[(k1.dataset == ds) & (k1.stat == st)][col].iloc[0]
                               if not k1[(k1.dataset == ds) & (k1.stat == st)].empty else np.nan)
    lines = ["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{lcccccc}", "\\hline",
             " & \\multicolumn{2}{c}{patch AUROC} & \\multicolumn{2}{c}{std.\\ gap} & \\multicolumn{2}{c}{view invariance}\\\\",
             "Dataset (1-shot) & raw & head & raw & head & raw & head\\\\", "\\hline"]
    for ds, nm in DS.items():
        vals = [get(ds, "patch_auroc", "raw"), get(ds, "patch_auroc", "head"), get(ds, "gap", "raw"), get(ds, "gap", "head"),
                get(ds, "inv_pos", "raw"), get(ds, "inv_pos", "head")]
        if np.isnan(vals[0]):
            continue
        lines.append(f"{nm} & {vals[0]:.1f} & {vals[1]:.1f} & {vals[2]:.2f} & {vals[3]:.2f} & {vals[4]:.2f} & {vals[5]:.2f}\\\\")
    lines += ["\\hline"]
    lo = [("logical", "LOCO logical"), ("structural", "LOCO structural")]
    lines += [" & \\multicolumn{2}{c}{image AUROC, top-1\\%} & \\multicolumn{2}{c}{image AUROC, mean} & & \\\\",
              " & raw & head & raw & head & & \\\\", "\\hline"]
    for key, nm in lo:
        v = [get("mvtec_loco", f"{key}_top1", "raw"), get("mvtec_loco", f"{key}_top1", "head"),
             get("mvtec_loco", f"{key}_mean", "raw"), get("mvtec_loco", f"{key}_mean", "head")]
        if not np.isnan(v[0]):
            lines.append(f"{nm} & {v[0]:.1f} & {v[1]:.1f} & {v[2]:.1f} & {v[3]:.1f} & & \\\\")
    lines += ["\\hline", "\\end{tabular}",
              "\\caption{What the contrastive head changes (1-shot, held-out draws, same 16-view memory). Patch AUROC: anomalous "
              "(mask coverage $\\geq$50\\%) vs normal patches; std.\\ gap: (mean anomalous $-$ mean normal NN distance)/std normal; "
              "view invariance: cosine between a token and its correspondence in another view. Image AUROC with local (top-1\\%) "
              "and global (mean) aggregation on LOCO.}\\label{tab:mech}", "\\end{table}"]
    (ROOT / "outputs" / "tables" / "table_mech.tex").write_text("\n".join(lines))
    mac = [r"\makeatletter", r"\def\Mx#1{\@ifundefined{mx@#1}{??}{\csname mx@#1\endcsname}}"]
    for _, r in S.iterrows():
        two = r.stat in ("gap", "inv_pos", "inv_neg", "spearman_anom")
        for col, fmt in (("raw", "{:.2f}" if two else "{:.1f}"), ("head", "{:.2f}" if two else "{:.1f}"),
                         ("p_categories", "{:.3f}"), ("share_head_lower", "{:.0f}"), ("n_categories", "{:.0f}")):
            v = r[col]
            if pd.notna(v):
                key = f"{r.dataset}|{int(r.k)}|{r.stat}|{col}"
                v = 100 * v if col == "share_head_lower" else v
                s = (("$<$0.001" if v < 0.001 else "=\\,{:.3f}".format(v)) if col == "p_categories" else fmt.format(v))
                mac.append(rf"\expandafter\def\csname mx@{key}\endcsname{{{s}}}")
    mac.append(r"\makeatother")
    (ROOT / "outputs" / "numbers_mech.tex").write_text("\n".join(mac) + "\n")


def sensitivity():
    df = pd.DataFrame([json.loads(l) for l in open(ROOT / "results" / "sensitivity.jsonl")])
    df = df.drop_duplicates(["dataset", "category", "k", "draw", "views", "rotations"], keep="last")
    g = df.groupby(["dataset", "k", "views", "rotations", "draw"]).agg(img=("img_auroc", "mean"), pro=("aupro_30", "mean"),
                                                                       mb=("mem_mb", "mean"), ms=("knn_ms_per_image", "mean"))
    g = g.groupby(["dataset", "k", "views", "rotations"]).agg(img=("img", "mean"), img_std=("img", "std"), pro=("pro", "mean"),
                                                              mb=("mb", "mean"), ms=("ms", "mean")).reset_index()
    g.to_csv(ROOT / "results" / "sensitivity_summary.csv", index=False)
    mac = [r"\makeatletter", r"\def\Sx#1{\@ifundefined{sx@#1}{??}{\csname sx@#1\endcsname}}"]
    for _, r in g.iterrows():
        key = f"{r.dataset}|{int(r.k)}|{int(r.views)}|{int(r.rotations)}"
        mac += [rf"\expandafter\def\csname sx@{key}|img\endcsname{{{r.img:.1f}}}",
                rf"\expandafter\def\csname sx@{key}|mb\endcsname{{{r.mb:.1f}}}",
                rf"\expandafter\def\csname sx@{key}|ms\endcsname{{{r.ms:.2f}}}"]
    (ROOT / "outputs" / "numbers_sens.tex").write_text("\n".join(mac + [r"\makeatother"]) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.7, "xtick.labelsize": 8.5, "pdf.fonttype": 42,
                         "ytick.labelsize": 8.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    colors, markers = ["#1d4ed8", "#c2410c", "#15803d"], ["o", "s", "^"]
    dss = [d for d in DS if d in set(g.dataset)]
    fig, axes = plt.subplots(1, len(dss), figsize=(12.2 / 2.54, 4.3 / 2.54), sharey=True)
    handles = []
    for ax, ds in zip(np.atleast_1d(axes), dss):
        base = float(g[(g.dataset == ds) & (g.k == 1) & (g.views == 1) & (g.rotations == 0)].img.iloc[0])
        ax.axhline(0, color="#9a9a9a", lw=0.6)
        for i, r in enumerate(sorted(g.rotations.unique())):
            d = g[(g.dataset == ds) & (g.k == 1) & (g.rotations == r)].sort_values("views")
            h, = ax.plot(d.views, d.img - base, marker=markers[i], color=colors[i], ms=3.6, lw=1.1,
                         label=f"+ {r} rotations" if r else "views only")
            if ds == dss[0]:
                handles.append(h)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16])
        ax.set_xticklabels(["1", "4", "16"])
        ax.minorticks_off()
        ax.set_ylim(-2, 3.6)
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(length=2.5, pad=1.5)
        ax.set_title(DS[ds].replace("MVTec AD 2", "AD 2"), fontsize=9, pad=2)
        ax.grid(axis="y", color="#e2e2e2", lw=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("$\\Delta$ image AUROC", fontsize=9)
    fig.supxlabel("views per shot (log scale)", fontsize=9, y=0.01)
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.55, 1.0),
               handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(pad=0.25, w_pad=0.9, rect=(0, 0, 1, 0.86))
    fig.savefig(ROOT / "figures" / "fig3_sensitivity.pdf")
    return g


def latency_macros():
    """\\Lat{method|stat}: ms per image and memory per category from results/latency.json."""
    mac = [r"\makeatletter", r"\def\Lat#1{\@ifundefined{lat@#1}{??}{\csname lat@#1\endcsname}}"]
    lat = json.load(open(ROOT / "results" / "latency.json"))["methods"]
    short = {"VMF": "VMF", "Flownomaly": "Flownomaly", "SubspaceAD": "SubspaceAD", "AnomalyDINO": "AnomalyDINO",
             "C1 no head": "C1", "ours": "head"}
    for name, v in lat.items():
        for pre, key in short.items():
            if name.startswith(pre):
                mac.append(rf"\expandafter\def\csname lat@{key}|ms\endcsname{{{v['ms_per_image']:.1f}}}")
                mac.append(rf"\expandafter\def\csname lat@{key}|mb\endcsname{{{v['memory_mb']:.0f}}}")
    mac.append(r"\makeatother")
    (ROOT / "outputs" / "numbers_latency.tex").write_text("\n".join(mac) + "\n")


if __name__ == "__main__":
    latency_macros()
    S = mechanism()
    table_and_macros(S)
    G = sensitivity()
    print(S[S.k == 1].pivot_table(index="stat", columns="dataset", values=["raw", "head"]).round(2).to_string())
    print(G[G.k == 1].pivot_table(index=["views"], columns=["dataset", "rotations"], values="img").round(2).to_string())

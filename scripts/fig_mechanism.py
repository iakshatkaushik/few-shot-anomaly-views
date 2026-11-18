"""Figure: what the contrastive head does to nearest-neighbour (NN) distances.
(a, b) NN distances of normal and anomalous test patches, raw tokens vs head (results/distances.npz; MVTec AD and VisA,
draws 5-9, k = 1, standardised per category with the normal patches). (c) patch AUROC per dataset
(results/mechanism_summary.csv, k = 1). Also writes outputs/numbers_dist.tex with the medians."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS, FS_S = 9.0, 8.5
plt.rcParams.update({"font.family": "serif", "font.size": FS, "axes.linewidth": 0.7, "mathtext.fontset": "cm",
                     "pdf.fonttype": 42, "xtick.labelsize": FS_S, "ytick.labelsize": FS_S, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
GREY, GREY_EDGE, RAW, HEAD, GRID = "#c9ccd1", "#7a7f87", "#1f5bd6", "#c2410c", "#e3e3e3"
D = np.load(ROOT / "results/distances.npz")
S = pd.read_csv(ROOT / "results/mechanism_summary.csv")

fig = plt.figure(figsize=(12.2 / 2.54, 4.75 / 2.54))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.08], left=0.095, right=0.955, top=0.765, bottom=0.25, wspace=0.16)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
bins = np.linspace(-3, 12, 61)
for ax, name, col, title in ((ax_a, "raw", RAW, "(a) raw tokens"), (ax_b, "head", HEAD, "(b) with head")):
    n, a = D[f"{name}_norm"], D[f"{name}_anom"]
    ax.hist(n, bins=bins, density=True, histtype="stepfilled", color=GREY, ec=GREY_EDGE, lw=0.7)
    ax.hist(a, bins=bins, density=True, histtype="step", color=col, lw=1.3)
    med = float(np.median(a))
    ax.axvline(med, color=col, lw=0.8, ls=(0, (2.5, 1.5)))
    ax.text(med + 0.5, 0.47, f"median\n{med:.1f}", fontsize=FS_S, color=col, va="center", ha="left", linespacing=1.05)
    ax.set_title(title, fontsize=FS, pad=4)
    ax.set_xlim(-3, 12)
    ax.set_ylim(0, 0.62)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 0.3, 0.6])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
ax_a.set_ylabel("density", fontsize=FS, labelpad=2)
plt.setp(ax_b.get_yticklabels(), visible=False)
# one shared x label centred under (a) and (b)
xa, xb = ax_a.get_position(), ax_b.get_position()
fig.text((xa.x0 + xb.x1) / 2, 0.025, "NN distance (s.d. of normal patches)", ha="center", va="bottom", fontsize=FS)

ax = fig.add_subplot(gs[0, 2])
order = [("mvtec_ad", "MVTec AD"), ("visa", "VisA"), ("mpdd", "MPDD"), ("mvtec_ad2", "AD 2"), ("mvtec_loco", "LOCO")]
P = S[(S.k == 1) & (S.stat == "patch_auroc")].set_index("dataset")
for j, (ds, _) in enumerate(order):
    r, h = P.loc[ds, "raw"], P.loc[ds, "head"]
    ax.plot([h, r], [j, j], color=GRID, lw=2.4, solid_capstyle="round", zorder=1)
    ax.scatter([r], [j], s=24, color=RAW, zorder=3, lw=0)
    ax.scatter([h], [j], s=24, color=HEAD, marker="s", zorder=3, lw=0)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([o[1] for o in order], fontsize=FS_S)
ax.tick_params(axis="y", length=0, pad=3)
ax.set_ylim(len(order) - 0.5, -0.6)
ax.set_xlim(66, 100)
ax.set_xticks([70, 80, 90, 100])
ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.set_xlabel("AUROC (%)", fontsize=FS)
ax.set_title("(c) patch AUROC", fontsize=FS, pad=4)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
# move (c) right so that its dataset labels have their own space next to (b)
pc = ax.get_position()
ax.set_position([pc.x0 + 0.075, pc.y0, pc.width - 0.075, pc.height])

handles = [Patch(fc=GREY, ec=GREY_EDGE, lw=0.7, label="normal patches"),
           Line2D([], [], color=RAW, lw=1.3, marker="o", ms=4.5, mew=0, label="raw tokens"),
           Line2D([], [], color=HEAD, lw=1.3, marker="s", ms=4.5, mew=0, label="with head")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=FS, bbox_to_anchor=(0.5, 1.0),
           handlelength=1.8, columnspacing=1.6, handletextpad=0.5, borderaxespad=0.2)
fig.savefig(ROOT / "figures/fig2_mechanism.pdf")

mac = [r"\makeatletter", r"\def\Dd#1{\@ifundefined{dd@#1}{??}{\csname dd@#1\endcsname}}"]
for k in ("raw_norm", "raw_anom", "head_norm", "head_anom"):
    mac.append(rf"\expandafter\def\csname dd@{k}\endcsname{{{np.median(D[k]):.1f}}}")
mac += [rf"\expandafter\def\csname dd@n_anom\endcsname{{{len(D['raw_anom']):,}}}".replace(",", "{,}"),
        rf"\expandafter\def\csname dd@n_runs\endcsname{{{len(D['meta'])}}}", r"\makeatother"]
(ROOT / "outputs/numbers_dist.tex").write_text("\n".join(mac) + "\n")
print("written figures/fig2_mechanism.pdf")

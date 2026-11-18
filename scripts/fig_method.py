"""Figure: overview of the two ways of using views.
The thumbnails are MVTec AD hazelnut/train/good/000.png and views/rotations from features/views.py
(seed 5010, i.e. draw 5, k = 1). Drawn 12.2 cm wide. Writes figures/fig1_design.pdf."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.views import make_view, rotate_batch, rotation_angles, view_params  # noqa: E402
from src.common import load_config  # noqa: E402

SEED, VIEWS = 1000 * 5 + 10, (1, 5, 9)
cfg = load_config()
img = Image.open(ROOT / "data/mvtec_ad/hazelnut/train/good/000.png").convert("RGB").resize((224, 224), Image.BICUBIC)
x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
ps = view_params(SEED, 0, cfg)
views = [make_view(x, ps[v], 14, cfg) for v in VIEWS]
ANGLE = rotation_angles(SEED, 0, cfg)[0]
rot, _ = rotate_batch(x[None], [ANGLE], 14, 0.5)
thumbs = [x] + views + [rot[0]]
to_np = lambda t: t.clamp(0, 1).permute(1, 2, 0).numpy()

if __name__ == "__main__":
    W, H = 12.2 / 2.54, 5.8 / 2.54
    BIG, SMALL = 9.0, 8.5
    plt.rcParams.update({"font.family": "serif", "font.size": BIG, "mathtext.fontset": "cm", "pdf.fonttype": 42})
    INK, MUTED, A_COL, B_COL, EDGE = "#1a1a1a", "#444444", "#b23a0b", "#1d4ed8", "#8a8a8a"
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0.20, H + 0.20)
    ax.axis("off")

    def box(x0, y0, w, h, text, col=EDGE, fc="white", bold=False, size=None):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", lw=0.9, ec=col, fc=fc))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=size or BIG, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.15)

    def arrow(x0, y0, x1, y1, col=MUTED):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=col,
                                     shrinkA=0, shrinkB=0))

    def thumb(t, x0, y0, s):
        ax.imshow(to_np(t), extent=(x0, x0 + s, y0, y0 + s), interpolation="none")
        ax.add_patch(plt.Rectangle((x0, y0), s, s, fill=False, lw=0.6, ec=EDGE))

    # input shot
    thumb(thumbs[0], 0.02, 1.12, 0.52)
    ax.text(0.28, 1.04, "$k$ normal\nshots", ha="center", va="top", fontsize=BIG, color=INK)
    arrow(0.57, 1.38, 0.65, 1.38)
    # views and rotations
    s2, x2 = 0.335, 0.68
    for i, t in enumerate(thumbs[1:4]):
        thumb(t, x2 + i * (s2 + 0.03), 1.62, s2)
    ax.text(x2 + 1.5 * s2 + 0.03, 2.00, "16 views per shot", ha="center", va="bottom", fontsize=BIG, color=INK)
    ax.text(x2, 1.56, "flips, rot90, shifts,\ncolour jitter, blur", ha="left", va="top", fontsize=SMALL, color=MUTED)
    thumb(thumbs[4], x2, 0.44, s2)
    ax.text(x2 + s2 + 0.05, 0.61, "30 rotations\n(memory\nonly)", ha="left", va="center", fontsize=SMALL, color=MUTED)
    arrow(1.76, 1.38, 1.86, 1.38)
    # frozen encoder
    box(1.88, 1.02, 0.60, 0.72, "frozen\nViT\nencoder", bold=True)
    ax.text(2.18, 0.96, "patch\ntokens", ha="center", va="top", fontsize=SMALL, color=MUTED)
    # branch A
    xb, wb, yA, yB, hb = 2.68, 1.24, 1.60, 0.40, 0.60
    arrow(2.48, 1.55, 2.66, yA + hb / 2, A_COL)
    box(xb, yA, wb, hb, "contrastive head $g$\n(Eq. 1); 1-NN\nin $g$-space", col=A_COL)
    ax.text(xb, yA + hb + 0.06, "A: views as training signal", ha="left", va="bottom", fontsize=BIG, color=A_COL,
            fontweight="bold")
    ax.text(xb + wb / 2, yA - 0.04, "pos.: view partners\nneg.: synthetic defects", ha="center", va="top", fontsize=SMALL,
            color=MUTED)
    # branch B
    arrow(2.48, 1.20, 2.66, yB + hb / 2, B_COL)
    box(xb, yB, wb, hb, "raw-token memory\n(views, rotations);\n1-NN, Eq. (2)", col=B_COL)
    ax.text(xb, yB + hb + 0.06, "B: views as memory", ha="left", va="bottom", fontsize=BIG, color=B_COL,
            fontweight="bold")
    ax.text(xb + wb / 2, yB - 0.04, "VMF: 3 backbones, averaged", ha="center", va="top", fontsize=SMALL, color=MUTED)
    # output
    xo, yo, wo, ho = 4.12, 0.82, 0.66, 0.96
    arrow(xb + wb, yA + hb / 2, xo, yo + ho - 0.12, A_COL)
    arrow(xb + wb, yB + hb / 2, xo, yo + 0.12, B_COL)
    box(xo, yo, wo, ho, "anomaly\nmap;\nimage\nscore:\ntop-1%\nmean", fc="#f2f2f0")
    fig.savefig(ROOT / "figures/fig1_design.pdf")
    print("written figures/fig1_design.pdf", W, H)

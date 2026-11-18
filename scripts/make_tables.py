"""Generate all tables from results/ only (summary.csv, tests.csv). Writes outputs/tables/*.tex and results/tables.md.
Cells: mean over 10 draws of the category-averaged metric; +-std over draws; (p10) = 10th percentile over draws.
Marks: * = paired Wilcoxon over draws vs 'ours', Holm-corrected p < 0.05 (sign of the difference shown by the numbers)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import config_hash, git_commit, load_config  # noqa: E402

S = pd.read_csv(ROOT / "results" / "summary.csv")
T = pd.read_csv(ROOT / "results" / "tests.csv") if (ROOT / "results" / "tests.csv").exists() else pd.DataFrame()
OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
MD = []

NAMES = {"VMF": "\\textbf{VMF (ours)}", "VMF+H": "VMF+H (ours, + contrastive head)",
         "ours": "Contrastive head (self-consistency selection)", "ours_fixed": "Ours, fixed defaults", "ours_oracle": "Ours, test oracle$^\\dagger$",
         "C1_nohead": "C1 no head (raw ViT-B)", "C2_shots": "C2 shot-only memory", "C3_randhead": "C3 random head",
         "ours_pca": "Ours, PCA-residual scoring", "AnomalyDINO": "AnomalyDINO$^\\ddagger$", "SubspaceAD": "SubspaceAD$^\\ddagger$",
         "PatchCore": "PatchCore$^\\ddagger$ (10\\% coreset)", "PatchCore-cs1": "PatchCore$^\\ddagger$ (full memory)",
         "WinCLIP+": "WinCLIP+$^\\ddagger$", "Flownomaly": "Flownomaly$^\\ddagger$", "VisionAD": "VisionAD$^\\ddagger$"}
MAIN = ["VMF", "VMF+H", "SubspaceAD", "AnomalyDINO", "VisionAD", "Flownomaly", "WinCLIP+", "PatchCore-cs1", "C1_nohead", "ours"]
NOT_RUN = {"PromptAD": "not run (see text)"}
REPORTED = {  # numbers reported in the original papers (different draws / protocol)
    "mvtec_ad": {"Flownomaly (reported)": {1: 97.8, 2: 98.4, 4: 98.5}, "ANoCo (reported)": {1: 97.9, 2: 98.4, 4: 98.7}},
    "visa": {"Flownomaly (reported)": {1: 95.3, 2: 96.1, 4: 96.3}},
}


def get(ds, k, metric, meth):
    r = S[(S.dataset == ds) & (S.k == k) & (S.metric == metric) & (S.method == meth)]
    return None if r.empty else r.iloc[0]


def sig(ds, k, metric, meth):
    if T.empty or meth == "VMF":
        return ""
    r = T[(T.dataset == ds) & (T.k == k) & (T.metric == metric) & (T.other == meth)]
    return "*" if (not r.empty and bool(r.iloc[0]["significant"])) else ""


def cell(ds, k, metric, meth, full=True):
    r = get(ds, k, metric, meth)
    if r is None:
        return "--"
    s = sig(ds, k, metric, meth)
    if full:
        return f"{r['mean']:.1f}$\\pm${r['std']:.1f} ({r['p10']:.1f}){s}"
    return f"{r['mean']:.1f}{s}"


def table_main(dss, name, caption, metric_b="aupro_30", methods=MAIN):
    lines = ["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}",
             "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l" + "cc" * 3 + "}", "\\hline"]
    md = [f"### {name}", ""]
    for ds in dss:
        lines += [f"\\multicolumn{{7}}{{l}}{{\\textbf{{{ds.replace('_', ' ')}}}}}\\\\",
                  "Method & \\multicolumn{2}{c}{1-shot} & \\multicolumn{2}{c}{2-shot} & \\multicolumn{2}{c}{4-shot}\\\\",
                  " & I-AUROC & PRO & I-AUROC & PRO & I-AUROC & PRO\\\\", "\\hline"]
        md += [f"**{ds}**", "", "| Method | 1-shot I-AUROC | PRO | 2-shot I-AUROC | PRO | 4-shot I-AUROC | PRO |",
               "|---|---|---|---|---|---|---|"]
        for m in methods:
            if all(get(ds, k, "img_auroc", m) is None for k in (1, 2, 4)):
                continue
            cells = []
            for k in (1, 2, 4):
                cells += [cell(ds, k, "img_auroc", m), cell(ds, k, metric_b, m, full=False)]
            lines.append(NAMES.get(m, m) + " & " + " & ".join(cells) + "\\\\")
            md.append("| " + NAMES.get(m, m).replace("$^\\ddagger$", "‡").replace("$^\\dagger$", "†").replace("\\%", "%")
                      + " | " + " | ".join(c.replace("$\\pm$", "±") for c in cells) + " |")
        for m, why in NOT_RUN.items():
            lines.append(f"{m} & \\multicolumn{{6}}{{l}}{{{why}}}\\\\")
            md.append(f"| {m} | {why} | | | | | |")
        for m, vals in REPORTED.get(ds, {}).items():
            lines.append(f"\\textit{{{m}}} & " + " & ".join(f"{vals[k]:.1f} & --" for k in (1, 2, 4)) + "\\\\")
            md.append(f"| *{m}, different draws* | " + " | ".join(f"{vals[k]:.1f} | --" for k in (1, 2, 4)) + " |")
        lines.append("\\hline")
        md.append("")
    lines += ["\\end{tabular}}", f"\\caption{{{caption}}}\\label{{tab:main}}", "\\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines))
    MD.extend(md)


def table_bench():
    """Table 2: un-saturated benchmarks."""
    blocks = [("mpdd", "img_auroc", "aupro_30", "MPDD"),
              ("mvtec_loco", "structural_img_auroc", "structural_aupro_30", "LOCO structural"),
              ("mvtec_loco", "logical_img_auroc", "logical_aupro_30", "LOCO logical"),
              ("mvtec_ad2", "img_auroc", "aupro_5", "MVTec AD 2 (public, PRO@5\\%)")]
    lines = ["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}",
             "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l" + "cc" * 3 + "}", "\\hline"]
    md = ["### table2_benchmarks", ""]
    for ds, ma, mb, title in blocks:
        lines += [f"\\multicolumn{{7}}{{l}}{{\\textbf{{{title}}}}}\\\\", "\\hline"]
        md += [f"**{title}**", "", "| Method | 1-shot I-AUROC | PRO | 2-shot | PRO | 4-shot | PRO |", "|---|---|---|---|---|---|---|"]
        for m in MAIN:
            if all(get(ds, k, ma, m) is None for k in (1, 2, 4)):
                continue
            cells = []
            for k in (1, 2, 4):
                cells += [cell(ds, k, ma, m), cell(ds, k, mb, m, full=False)]
            lines.append(NAMES.get(m, m) + " & " + " & ".join(cells) + "\\\\")
            md.append("| " + NAMES.get(m, m).replace("$^\\ddagger$", "‡").replace("\\%", "%") + " | " +
                      " | ".join(c.replace("$\\pm$", "±") for c in cells) + " |")
        lines.append("\\hline")
        md.append("")
    lines.append("\\multicolumn{7}{l}{Real-IAD: access pending. MVTec AD 2 private split: pending (server account).}\\\\")
    lines += ["\\end{tabular}}", "\\caption{Un-saturated benchmarks (10 draws; mean$\\pm$std (10th percentile) of the "
              "category-averaged image AUROC; PRO = AU-PRO@30\\% unless stated). $^\\ddagger$re-implemented, identical draws.}"
              "\\label{tab:bench}", "\\end{table}"]
    (OUT / "table2_benchmarks.tex").write_text("\n".join(lines))
    MD.extend(md)


ABL = [("ours_fixed", "Contrastive head, default settings"), ("ours", "\\quad + self-consistency selection"),
       ("ours_oracle", "\\quad + selection on test labels$^\\dagger$"), ("abl_no_shift", "\\quad without mean shift"),
       ("abl_neg_random", "\\quad random synthetic negatives"), ("ours_onefa", "\\quad one head for all categories"),
       ("C3_randhead", "C3: frozen random head"), ("C2_shots", "C2: no head, shot-only memory"),
       ("C1_nohead", "C1: no head, 16-view memory")]
BACK = [("abl_fixed@dinov2_vitg14_reg/late4", "ViT-G/14 (672 px), head"), ("C1_nohead@dinov2_vitg14_reg/late4", "ViT-G/14 (672 px), no head")]


def table_ablation():
    lines = ["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lcccc}", "\\hline",
             " & \\multicolumn{2}{c}{MVTec AD} & \\multicolumn{2}{c}{VisA}\\\\",
             "Variant & I-AUROC & PRO & I-AUROC & PRO\\\\", "\\hline"]
    md = ["### table3_ablations (1-shot, mean over 10 draws)", "", "| Variant | MVTec I-AUROC | PRO | VisA I-AUROC | PRO |",
          "|---|---|---|---|---|"]
    for key, label in ABL + BACK:
        cells = []
        for ds in ("mvtec_ad", "visa"):
            for metric in ("img_auroc", "aupro_30"):
                r = get(ds, 1, metric, key)
                cells.append("--" if r is None else f"{r['mean']:.1f}")
        if all(c == "--" for c in cells):
            continue
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
        md.append(f"| {label.replace('$', '').replace(chr(92), '')} | " + " | ".join(cells) + " |")
    lines += ["\\hline", "\\end{tabular}",
              "\\caption{Contrastive-head variants and controls on MVTec AD and VisA ($k{=}1$, mean over held-out draws 5--9). "
              "I-AUROC: image-level AUROC; PRO: AU-PRO up to 30\\% false-positive rate; both in \\%. Unless stated, the backbone "
              "is DINOv2 ViT-B/14 at 448 px. $^\\dagger$Uses test labels; an upper bound, not a valid method.}""\\label{tab:abl}", "\\end{table}"]
    (OUT / "table3_ablations.tex").write_text("\n".join(lines))
    MD.extend(md + [""])


COMPACT = ["VMF", "Flownomaly", "SubspaceAD", "VisionAD", "AnomalyDINO", "WinCLIP+", "PatchCore-cs1", "C1_nohead", "ours"]
CNAMES = {"C1_nohead": "C1: raw ViT-B + 16 views", "ours": "C1 + contrastive head", "VMF": "VMF",
          "PatchCore-cs1": "PatchCore$^\\ddagger$"}


def table_compact():
    """Compact table: image AUROC mean over draws, 1/4 shots, five benchmarks; * Holm p<0.05 vs VMF."""
    cols = [("mvtec_ad", "img_auroc", "MVTec AD"), ("visa", "img_auroc", "VisA"), ("mpdd", "img_auroc", "MPDD"),
            ("mvtec_loco", "structural_img_auroc", "LOCO str."), ("mvtec_loco", "logical_img_auroc", "LOCO log."),
            ("mvtec_ad2", "img_auroc", "AD 2")]
    lines = ["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.2pt}", "\\resizebox{\\textwidth}{!}{%",
             "\\begin{tabular}{l" + "cc" * len(cols) + "}", "\\hline",
             "Method & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for _, _, t in cols) + "\\\\",
             "$k$ & " + " & ".join("1 & 4" for _ in cols) + "\\\\", "\\hline"]
    for m in COMPACT:
        cells = []
        for ds, met, _ in cols:
            for k in (1, 4):
                r = get(ds, k, met, m)
                c = "--" if r is None else f"{r['mean']:.1f}{sig(ds, k, met, m)}"
                cells.append(c)
        lines.append(CNAMES.get(m, NAMES.get(m, m)) + " & " + " & ".join(cells) + "\\\\")
        if m == "PatchCore-cs1":
            lines.append("\\hline")
    lines += ["\\hline", "\\end{tabular}}",
              "\\caption{Image AUROC (\\%), mean over 10 shot draws (MVTec AD, VisA at $k{=}1$: held-out draws 5--9). LOCO str./log.: "
              "structural/logical anomalies; AD\\,2: MVTec AD\\,2 public test set. *: differs from VMF (paired Wilcoxon, Holm, "
              "$p<0.05$). $^\\ddagger$Re-implemented, same draws; PatchCore with full memory. C1: no head, 16-view memory; "
              "C1 + contrastive head: with self-consistency selection.}\\label{tab:main}", "\\end{table}"]
    (OUT / "table_compact.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    cfg = load_config()
    table_main(["mvtec_ad", "visa"], "table1_main",
               "MVTec AD and VisA, 10 shot draws per category. I-AUROC: mean$\\pm$std (10th percentile) over draws of the "
               "category-averaged image AUROC; PRO: AU-PRO@30\\% (mean). * Holm-corrected paired Wilcoxon $p<0.05$ vs VMF. "
               "1-shot: held-out draws 5--9 (draws 0--4 were VMF's development split); 2/4-shot: 10 draws. "
               "$^\\ddagger$re-implemented from the official code, identical draws.")
    table_bench()
    table_ablation()
    table_compact()
    hdr = f"<!-- generated by scripts/make_tables.py from results/summary.csv; config {config_hash(cfg)}, commit {git_commit()} -->\n"
    (ROOT / "results" / "tables.md").write_text(hdr + "\n".join(MD))
    print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))

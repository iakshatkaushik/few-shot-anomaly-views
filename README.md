<div align="center">

# Views as Memory vs. Contrastive Training

### for few-shot visual anomaly detection on frozen ViT features

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/backbones-DINOv2%20%7C%20DINOv3-8A2BE2)](https://github.com/huggingface/pytorch-image-models)
[![Benchmarks](https://img.shields.io/badge/benchmarks-5-0E8A16)](#results)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7B93E)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-this%20work-B31B1B)](#citation)

[Results](#results) · [Installation](#installation) · [Usage](#usage) · [Protocol](#protocol) · [Citation](#citation)

</div>

With only a handful of normal images, few-shot anomaly detectors augment them into many views. This repository
asks a simple question: **should those views train a representation, or should they just be remembered?**
Both options run on the same frozen encoder, with the same scoring rule, shot draws and evaluation code.

**A. Views as a training signal.** A small patch-level contrastive head is trained on the views (positives: the
same patch in another view; negatives: synthetic defects). Test patches are scored by 1-NN distance in the head's
space.

**B. Views as memory.** The raw patch tokens of all views and rotations go straight into the nearest-neighbour
memory, with no training. **VMF** averages three such memories (DINOv3 ViT-B/16 at 448 and 672 px, DINOv2
ViT-G/14 at 672 px).

The evaluation covers **5 industrial benchmarks** (MVTec AD, VisA, MPDD, MVTec LOCO AD, MVTec AD 2) at 1, 2 and
4 shots, with **10 random shot draws** each. **6 published methods** are re-implemented and run on exactly the
same draws.

*The accompanying paper is currently under review. This repository is a partial release: it contains the core
method, the baselines, the evaluation and statistics code, and all result rows needed to reproduce the reported
numbers. Some parts of the codebase (experiment orchestration and exploratory code) are not included yet. The
remaining code will be released after the review process.*

---

## Results

Image AUROC (%) at **k = 1**, mean over 10 shot draws; **bold** marks the best value in each column.

| Method | MVTec AD | VisA | MPDD | LOCO str. | LOCO log. | MVTec AD 2 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **VMF** (view memory, 3 backbones) | 97.5 | 93.0 | 73.5 | 75.8 | 55.3 | **71.0** |
| Raw view memory, ViT-B (B) | 96.3 | 89.3 | 71.8 | 72.3 | 53.9 | 67.4 |
| Contrastive head, ViT-B (A) | 89.9 | 81.1 | 73.6 | 71.6 | 60.5 | 61.0 |
| Flownomaly\* | **97.7** | **94.7** | 70.4 | **79.7** | 53.8 | **71.0** |
| VisionAD\* | 96.9 | 92.9 | **75.9** | 78.9 | 57.6 | 69.6 |
| SubspaceAD\* | 96.5 | 92.6 | 67.7 | 75.7 | 50.2 | 68.5 |
| AnomalyDINO\* | 96.6 | 87.2 | 70.6 | 70.9 | 55.5 | 67.1 |
| WinCLIP+\* | 93.4 | 82.6 | 68.1 | 67.7 | 59.1 | 59.5 |
| PatchCore\* (full memory) | 83.2 | 79.5 | 58.6 | 61.4 | **64.6** | 59.4 |

<sub>\* re-implemented here and run on the same shot draws. For MVTec AD and VisA the mean is over draws 5-9,
because draws 0-4 were used to design VMF. LOCO str./log.: structural and logical anomalies of MVTec LOCO AD.
MVTec AD 2: public test split. Source: <code>results/summary.csv</code>, which also holds k = 2 and 4, pixel
AUROC, AU-PRO, AP, F1-max, standard deviations and 10th percentiles. Paired tests: <code>results/tests.csv</code>.</sub>

**Takeaways**

- Keeping the views as memory (B) has a higher mean than training on them (A) on MVTec AD, VisA, LOCO
  structural and MVTec AD 2.
- The contrastive head is ahead on LOCO logical anomalies and on MPDD.
- The head pulls the anomalous patches closer to normal ones: the median nearest-neighbour distance of anomalous
  patches drops from 3.3 to 2.7 normal-patch standard deviations (MVTec AD and VisA, k = 1).

---

## Installation

```bash
git clone https://github.com/iakshatkaushik/few-shot-anomaly-views.git
cd few-shot-anomaly-views
pip install -r requirements.txt
python scripts/fetch_weights.py      # backbone weights from the Hugging Face hub
```

Python 3.10 and PyTorch 2.4 with CUDA. All experiments were run on a single NVIDIA H100 (80 GB).

---

## Data

The datasets are not included; each has its own licence. Place them under `data/`:

| Dataset | Path |
|---|---|
| MVTec AD | `data/mvtec_ad/<category>/{train,test,ground_truth}` |
| VisA | `data/visa_raw/` (extracted `VisA_20220922.tar`; uses `split_csv/1cls.csv`) |
| MPDD | `data/mpdd_raw/MPDD/<category>/...` |
| MVTec LOCO AD | `data/mvtec_loco/<category>/{train,validation,test,ground_truth}` |
| MVTec AD 2 | `data/mvtec_ad2_fo/` (FiftyOne export with `samples.json`; public test split) |
| DTD | `data/dtd/images/` (texture source for synthetic anomalies) |

Then build the image index:

```bash
python -m src.datasets mvtec_ad visa mpdd mvtec_loco mvtec_ad2
```

---

## Usage

Patch tokens are cached once and shared by all methods. The cache is large, so point `paths.cache_root` in
`config.yaml` at a disk with plenty of space. `make all` runs the full pipeline. The individual steps are:

```bash
# 1. cache tokens, views and synthetic anomalies
scripts/extract_features.sh dinov2_vitb14 all mvtec_ad visa

# 2. contrastive head (branch A) with controls C1-C3 (C1 = raw view memory, branch B)
python -m src.run_ours --dataset mvtec_ad --ks 1 2 4 --draws 0-9

# 3. view-memory fusion; --head adds the contrastive head as a fourth scorer (VMF+H)
python -m src.run_fusion --dataset mvtec_ad --ks 1 2 4 --draws 0-9

# 4. baselines
python -m src.baselines.anomalydino --dataset mvtec_ad   # also subspacead, patchcore, winclip, visionad, flownomaly

# 5. feature-space analyses and timing
python -m src.analysis_mechanism   --dataset mvtec_ad
python -m src.analysis_distances   --datasets mvtec_ad visa
python -m src.analysis_sensitivity --dataset mvtec_ad
python -m src.latency
```

Each run appends rows to `results/*.jsonl`. A row holds the dataset, category, k, draw, all metrics, the config
hash and the git commit.

### Reproducing the numbers without a GPU

All result rows are included, so the statistics, tables and figures can be rebuilt on a CPU:

```bash
make stats tables figures
```

This writes `results/summary.csv`, `results/tests.csv` and `results/per_category.csv`, LaTeX tables and macros in
`outputs/`, and PDF figures in `figures/`. Only the overview figure needs the datasets (it uses one MVTec AD image).

---

## Protocol

<details>
<summary><b>Shot draws, leakage control, statistics and baselines</b></summary>
<br>

- **Shot draws.** Seed `1000 * draw + 10 * k`, 10 draws for each k in {1, 2, 4}. Every method gets exactly the
  same shots.
- **No test leakage.** No test image or label is used for training, model selection or thresholds. The head's
  temperature, width and aggregation, and the rotation gate, are chosen by leave-one-view-out consistency on the
  shots. The `ours_oracle` rows select on test labels and exist only to measure the selection gap.
- **Development split.** VMF was designed on MVTec AD and VisA, draws 0-4, k = 1. Those cells are reported on
  draws 5-9. The development runs are in `results/explore.jsonl`.
- **Evaluation.** All methods share one evaluation code path: maps at 448 x 448, Gaussian smoothing with
  sigma 4, and the same masks and metrics.
- **Statistics.** Paired Wilcoxon signed-rank tests over draws, Holm-corrected across methods (`src/stats.py`).
- **Baselines.** Re-implemented from the official code (anomalib for PatchCore and WinCLIP+). The docstring at the
  top of each file in `src/baselines/` lists the settings used and every deviation from the original.

</details>

---

## Repository structure

```
├── config.yaml            all hyper-parameters (their hash is stored with every result row)
├── features/              token caching, views and rotations, synthetic anomalies
├── src/
│   ├── head.py, loss.py, mining.py, train.py     contrastive head and its training
│   ├── score.py, selfconsistency.py, eval.py     scoring, label-free selection, metrics
│   ├── run_ours.py, run_fusion.py                branch A with controls, view-memory fusion
│   ├── analysis_*.py, latency.py, stats.py       analyses, timing, statistics
│   └── baselines/                                AnomalyDINO, SubspaceAD, PatchCore, WinCLIP+, VisionAD, Flownomaly
├── scripts/               feature caching, weight download, tables and figures
└── results/               all result rows (*.jsonl) and summaries
```

---

## Citation

If you use this code, the result files or any part of this repository in your work, please give credit to the
authors and cite the paper. The paper is currently under review; the full BibTeX entry will be added here once it
is published. Until then, please cite this repository (the **"Cite this repository"** button in the sidebar gives
the reference from [`CITATION.cff`](CITATION.cff)).

---

## License

Released under the [MIT License](LICENSE): free to use, modify and share, provided the copyright notice is kept.
The datasets and pretrained weights keep their own licences.

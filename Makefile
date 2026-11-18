# Full pipeline. Expects the datasets in data/ (see README.md). Every step can be re-run:
# cached shards are skipped and result rows are appended (the newest row per run is used).
PY ?= python
DATASETS := mvtec_ad visa mpdd mvtec_loco mvtec_ad2

.PHONY: all index weights features ours fusion baselines ablations analyses stats tables figures

all: index features ours fusion baselines ablations analyses stats tables figures

index:
	$(PY) -m src.datasets $(DATASETS)

weights:
	$(PY) scripts/fetch_weights.py

features: index weights
	PY=$(PY) scripts/extract_features.sh dinov2_vitb14 all mvtec_ad visa
	PY=$(PY) scripts/extract_features.sh dinov2_vitb14 b9_12 mpdd mvtec_loco mvtec_ad2
	$(PY) -m features.extract --encoder dinov2_vits14 --datasets $(DATASETS) --kinds test --mode shorter_edge --sets last_raw --tag ad --no_compile
	$(PY) -m features.extract --encoder dinov2_vitg14_reg --datasets mvtec_ad visa --kinds test --sets sub23_29 late4 last l16 l20 l24 l28 l32
	$(PY) -m features.extract --encoder dinov2_vitg14_reg --datasets mpdd mvtec_loco mvtec_ad2 --kinds test --sets sub23_29
	for E in dinov2_vits14 dinov3_vitb16; do \
	  $(PY) -m features.extract --encoder $$E --datasets mvtec_ad visa --kinds test shots --sets b9_12; \
	  $(PY) -m features.views --encoder $$E --datasets mvtec_ad visa --sets b9_12 --ks 1; \
	  $(PY) -m features.synth --encoder $$E --datasets mvtec_ad visa --sets b9_12 --ks 1; done
	$(PY) -m features.extract --encoder dinov2_vitg14_reg --datasets mvtec_ad visa --kinds shots --sets late4
	$(PY) -m features.views --encoder dinov2_vitg14_reg --datasets mvtec_ad visa --sets late4 --ks 1
	$(PY) -m features.synth --encoder dinov2_vitg14_reg --datasets mvtec_ad visa --sets late4 --ks 1

ours:
	for D in $(DATASETS); do $(PY) -m src.run_ours --dataset $$D; done

fusion:
	for D in $(DATASETS); do $(PY) -m src.run_fusion --dataset $$D; $(PY) -m src.run_fusion --dataset $$D --head; done

baselines:
	for D in $(DATASETS); do \
	  $(PY) -m src.baselines.anomalydino --dataset $$D; \
	  $(PY) -m src.baselines.subspacead --dataset $$D; \
	  $(PY) -m src.baselines.patchcore --dataset $$D; \
	  $(PY) -m src.baselines.patchcore --dataset $$D --coreset 1.0; \
	  $(PY) -m src.baselines.winclip --dataset $$D; \
	  $(PY) -m src.baselines.visionad --dataset $$D; \
	  $(PY) -m src.baselines.flownomaly --dataset $$D; done

ablations:
	for D in mvtec_ad visa; do \
	  for A in no_shift neg_random neg_unfiltered neg_extreme d32 d512 part_off onefa; do \
	    $(PY) -m src.run_ours --dataset $$D --ablation $$A --out ablations; done; \
	  $(PY) -m src.run_ours --dataset $$D --ablation seed1 --draws 0 --out ablations; \
	  $(PY) -m src.run_ours --dataset $$D --ablation seed2 --draws 0 --out ablations; \
	  $(PY) -m src.run_ours --dataset $$D --ablation fixed --fset last --out ablations; \
	  $(PY) -m src.run_ours --dataset $$D --ks 1 --ablation fixed --encoder dinov2_vits14 --fset b9_12 --out ablations; \
	  $(PY) -m src.run_ours --dataset $$D --ks 1 --ablation fixed --encoder dinov3_vitb16 --fset b9_12 --out ablations; \
	  $(PY) -m src.run_ours --dataset $$D --ks 1 --ablation fixed --encoder dinov2_vitg14_reg --fset late4 --out ablations; done
	for D in mpdd mvtec_loco mvtec_ad2; do $(PY) -m src.run_ours --dataset $$D --ablation onefa --out ablations; done

analyses:
	for D in $(DATASETS); do $(PY) -m src.analysis_mechanism --dataset $$D; done
	for D in $(DATASETS); do $(PY) -m src.analysis_sensitivity --dataset $$D; done
	$(PY) -m src.analysis_distances
	$(PY) -m src.latency

# everything below runs on CPU from the files in results/
stats:
	$(PY) -m src.stats

tables:
	$(PY) scripts/make_numbers.py
	$(PY) scripts/make_tables.py
	$(PY) scripts/make_mechanism.py

figures:
	$(PY) scripts/fig_method.py
	$(PY) scripts/fig_mechanism.py

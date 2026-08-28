<p align="right">
  <strong>English</strong> · <a href="README_zh.md">中文</a>
</p>

# FLaG — Frequency-Domain Latent Attention Gating for Cross-Domain Token Aggregation

*Built on [AMPCliff](https://github.com/Kewei2023/AMPCliff)*

**FLaG** (Frequency-Domain Latent Attention Gating for Cross-Domain Token Aggregation) is an independent research project built on [AMPCliff](https://github.com/Kewei2023/AMPCliff) for antimicrobial peptide **activity cliff** prediction. AMPCliff provides the activity cliff dataset, benchmark framework, and ESM2 downstream training infrastructure; FLaG introduces the core pooling method `FLaG` along with mechanism analysis and ablation experiments.

Code is hosted on the **`FLaG` branch** of the AMPCliff repository (the `AMPCliff` Python package name and directory layout are retained for historical reasons).

---

## Table of Contents

- [Setup & Environment](#setup--environment)
- [Data & Model Weights](#data--model-weights)
- [Training FLaG](#training-flag)
- [Running](#running)
- [Five Mechanism Experiments](#five-mechanism-experiments)
- [Ablation Scripts](#ablation-scripts)
- [Key Configuration](#key-configuration)
- [Script Reference](#script-reference)
- [Citation & Acknowledgements](#citation--acknowledgements)
- [Contact](#contact)

---

## Setup & Environment

```bash
git clone git@github.com:Kewei2023/AMPCliff.git
cd AMPCliff
git checkout FLaG
```

> **Note:** The local directory name must be `AMPCliff` (required for absolute imports: `from AMPCliff.xxx import ...`).

```bash
conda env create -f environment.yaml
conda activate AMPCliff
```

After cloning, set the repository root (batch scripts default to HPC paths):

```bash
export REPO_ROOT=/path/to/AMPCliff
```

Common environment variables:

| Variable | Description |
|----------|-------------|
| `REPO_ROOT` | Project root directory |
| `OUTPUT_ROOT` | Output root for training/analysis (default `${REPO_ROOT}/outputs/ablation_new_data`, etc.) |
| `CONFIG_DIR` | ESM2 pretrained weights directory (falls back to `factory/initializer.py` defaults if unset) |
| `DRY_RUN=1` | Print commands only, do not execute |
| `SKIP_IF_DONE=1` | Skip runs when outputs already exist |

---

## Data & Model Weights

- **Data:** Place under `./data/`. Activity cliff splits are generated via [AMPCliff-generation](https://github.com/Kewei2023/AMPCliff-generation).
- **Default AC fix split path:**
  ```
  data/blosum62 average/diff_5-trd_0.9/grampa_{dataset}_7_25-{train,valid,test}.csv
  ```
  where `{dataset}` is `s_aureus` or `e_coli`.
- **ESM2 weights** (HPC defaults):
  - `esm2_t6` → `/data/public/models/facebook/esm2_t6_8M_UR50D/`
  - `esm2_t12` → `/data/public/models/facebook/esm2_t12_35M_UR50D/`
- Override at training time if your paths differ:
  ```bash
  python downstream_train.py model.config_dir="/your/path/to/esm2_t12_35M_UR50D/"
  ```

---

## Training FLaG

### Core Method

FLaG pooling is implemented as `FFTLatentAttentionGatePooling` in `factory/pooling/flag_pooling.py`:

**rFFT → latent attention → gate → iFFT → time pooling**

Config key: `model.regression.pooling=FLaG`

### Single Run

```bash
export REPO_ROOT=/path/to/AMPCliff
cd "${REPO_ROOT}"

POOLING=FLaG MODEL_TYPE=esm2_t6 DATASET=s_aureus \
  bash evaluation_scripts/run_baseline_pooling_train.sh
```

- Training entry: `downstream_train.py` + `configs/downstream.yaml`
- Evaluation entry: `downstream_evaluate.py`

### Multi-Seed 2×2 Grid (esm2_t6/t12 × e_coli/s_aureus)

Sharded parallel scripts (set `POOLINGS` for baseline comparison):

```bash
# FLaG only
POOLINGS="FLaG" bash evaluation_scripts/run_baseline_pooling_grid_2x2_seeds_1.sh

# Compare against mean / max / attn_structured
POOLINGS="mean max attn_structured FLaG" bash evaluation_scripts/run_baseline_pooling_grid_2x2_seeds_1.sh
# seeds_2.sh ~ seeds_4.sh are the remaining shards
```

### Aggregate Seed Metrics

```bash
bash evaluation_scripts/run_aggregate_baseline_pooling_seed_metrics.sh
# or
python evaluation_scripts/aggregate_pooling_seed_metrics.py
```

---

## Running

Quick entry points for baseline comparison and the five mechanism experiments (each with run + plot). More background: [Five Mechanism Experiments](#five-mechanism-experiments).

### 1. Baseline comparison (train + aggregate)

Train FLaG against common pooling baselines on the 2×2 grid (esm2_t6/t12 × e_coli/s_aureus) with multiple seeds (shards 1–4):

```bash
POOLINGS="mean max attn_structured last latent_attn mltp_paper FLaG" \
  bash evaluation_scripts/run_baseline_pooling_grid_2x2_seeds_1.sh
# Also run seeds_2 / _3 / _4 (same POOLINGS) in parallel terminals or jobs.

# Single-run smoke test
POOLING=FLaG MODEL_TYPE=esm2_t6 DATASET=s_aureus \
  bash evaluation_scripts/run_baseline_pooling_train.sh
```

Aggregate seed metrics and export tables:

```bash
bash evaluation_scripts/run_aggregate_baseline_pooling_seed_metrics.sh
# or
python evaluation_scripts/aggregate_pooling_seed_metrics.py
python evaluation_scripts/merge_seed_metrics_pooling_csv_to_xlsx.py
```

Outputs land under `outputs/ablation_new_data/` (and related statics paths used by the aggregate scripts).

### 2. Five mechanism experiments (run + plot)

Requires trained FLaG checkpoints (`POOLING=FLaG`). Exp1–4 outputs go under `outputs/analysis/fftlag_mechanism/`; Exp5 under `outputs/analysis/dc_validation/` (presets also in [`paper/results/exp5/`](paper/results/exp5/)). Background and configs: [Five Mechanism Experiments](#five-mechanism-experiments).

**Exp1 — Band knockout**

```bash
bash evaluation_scripts/run_fftlag_exp1_fulltest.sh
# optional: sbatch evaluation_scripts/run_fftlag_exp1_fulltest_slurm.sh
python evaluation_scripts/aggregate_fftlag_mechanism_seeds.py
python evaluation_scripts/plot_fftlag_exp1_fulltest_violin.py
python evaluation_scripts/plot_fftlag_exp1_representative_heatmaps.py
python evaluation_scripts/plot_exp1_abs_mse_violin_heatmap_composite_allblue_violin_u1.py
```

**Exp2 — Gate PSD**

```bash
bash evaluation_scripts/run_fftlag_exp2_fulltest.sh
python evaluation_scripts/plot_fftlag_exp2_fulltest_combined.py
```

**Exp3 — Token knockout (|ΔMSE|)**

```bash
bash evaluation_scripts/run_fftlag_exp3_fulltest.sh
# or two-GPU revised poolings:
# bash evaluation_scripts/run_fftlag_exp3_revised_poolings_parallel.sh
python evaluation_scripts/aggregate_exp3_token_knockout_mse_diff.py --force
python evaluation_scripts/plot_fftlag_exp3_mse_diff_violin_revised.py --force
python evaluation_scripts/export_fftlag_exp3_token_knockout_data.py
```

**Exp4 — Latent viz**

```bash
bash evaluation_scripts/run_fftlag_exp4_fulltest.sh
python evaluation_scripts/plot_fftlag_exp4_fulltest_latent_query_dist.py
# optional attn-score variant / combined figure:
# bash evaluation_scripts/run_fftlag_exp4_attn_score_raw.sh
# python evaluation_scripts/plot_fftlag_exp4_attn_score_raw.py
# python evaluation_scripts/aggregate_fftlag_exp4_attn_score_raw.py
python evaluation_scripts/plot_fftlag_exp4_frequency_attention_combined.py
```

**Exp5 — DC–property validation**

```bash
bash evaluation_scripts/run_dc_validation_v2.sh
python evaluation_scripts/plot_dc_validation_combined_figure_v3.py
python evaluation_scripts/plot_property_dc_knockout.py
python evaluation_scripts/plot_multi_property_band_sensitivity_combined.py
```

---

## Five Mechanism Experiments

Mechanism experiments analyze FLaG's internal behavior on the activity cliff task. **Train FLaG models first** (checkpoints are resolved by default from `outputs/ablation_new_data/{model}_FLaG_{dataset}_diff5/seed_*/data/model.pth`).

### Exp1–4: mechanism statistics on the full test set

Exp1–4 reuse the original mechanism probes; the main update is **full-test coverage** (all test peptides) instead of a 30-peptide manifest subset. Exp1 plots report **absolute** $\lvert\Delta\mathrm{P}\rvert$ (CSV column remains `mse_diff`; figure labels use $|\Delta\mathrm{P}|$).

| Exp | Purpose | Main Python Script | Full-test batch |
|-----|---------|-------------------|-----------------|
| **Exp1** Band knockout | Sequence band notch sensitivity | `downstream_evaluate_spectrual_filter.py` | `evaluation_scripts/run_fftlag_exp1_fulltest.sh` (+ `_slurm`) |
| **Exp2** Gate PSD | Spectral energy change before/after gate | `downstream_evaluate_psd_gate.py` | `evaluation_scripts/run_fftlag_exp2_fulltest.sh` (+ `_slurm`) |
| **Exp3** Token knockout | Token perturbation response ($\lvert\Delta\mathrm{MSE}\rvert$) | `downstream_evaluate_knockout.py` | `evaluation_scripts/run_fftlag_exp3_fulltest.sh` (+ `_slurm`); parallel revised poolings: `run_fftlag_exp3_revised_poolings_parallel.sh` |
| **Exp4** Latent viz | Latent query band mass distribution | `downstream_evaluate_fft_lag_latent.py` | `evaluation_scripts/run_fftlag_exp4_fulltest.sh` |

Legacy subset orchestration (30 peptides): `evaluation_scripts/run_fftlag_mechanism_experiments.sh` (`RUN_EXPS=1,2,4` by default; Exp3 separate).

Plot / aggregate helpers: `plot_fftlag_exp{1,2,4}_*`, `aggregate_fftlag_mechanism_seeds.py`. **Exp3** uses the `|ΔMSE|` pipeline:

```bash
bash evaluation_scripts/run_fftlag_exp3_fulltest.sh
# or (mltp_paper + attn_structured on two GPUs)
bash evaluation_scripts/run_fftlag_exp3_revised_poolings_parallel.sh

python evaluation_scripts/aggregate_exp3_token_knockout_mse_diff.py --force
python evaluation_scripts/plot_fftlag_exp3_mse_diff_violin_revised.py --force
python evaluation_scripts/export_fftlag_exp3_token_knockout_data.py
```

The revised combined violin plots 7 poolings (mean / max / attn / last / MLTP / latent_attn / FLaG) and writes `outputs/analysis/fftlag_mechanism/figures/exp3/combined/exp3_token_knockout_mse_diff_violinplot_combined_no_swe_ot.png` (plus `.svg` / `.xlsx` export).

### Exp5 (upgraded): DC–property validation

**Exp5** is the full DC interpretability pipeline (property table → DCT features → DC decoding → species×property effects → property-bucket knockout). It answers:

1. Which physicochemical properties can be decoded from the DC component?
2. Do those properties relate to *E. coli* vs *S. aureus* activity in the same way?
3. Does DC knockout follow the same property pattern?

| Step | Role | Scripts |
|------|------|---------|
| 1 | Build `dc_property_table.csv` | `build_dc_property_table.py`, `dc_property_utils.py` |
| 2 | Last-layer DCT $\mathcal{B}_0-\mathcal{B}_3$ | `extract_dct_coefficient_features.py` |
| 3 | **Main exp. 1** DC property decoding | `analyze_dc_property_encoding.py`, `dc_property_probe.py` |
| 4 | **Main exp. 2A** Species×property activity | `analyze_species_property_effects.py` |
| 5 | **Main exp. 2B** Property-bucket band/DC KO | `run_dc_property_knockout_fulltest.sh`, `analyze_property_dc_tables.py`, `plot_property_dc_knockout.py` |

Official Exp5 orchestration:

```bash
bash evaluation_scripts/run_dc_validation_v2.sh
```

Exp5 reports **both** signed $\Delta\mathrm{MSE}$ and $\lvert\Delta\mathrm{MSE}\rvert$ (see Step-5 tables/figures). Preset result snapshots (no large `.npz` features) live under [`paper/results/exp5/`](paper/results/exp5/).

Optional / legacy helix-structure bucketing from Exp1/2/4 aggregates: `run_exp5_structure_fulltest.sh` / `analyze_fft_lag_mechanism_by_structure.py` (not required for the core Exp5 evidence chain).

### Important Notes

- Configs: Exp1/4 `configs/evaluate_fftlag_mechanism.yaml`; Exp2 `configs/evaluate_psd_gate.yaml`; Exp3 `configs/downstream_knockout.yaml` (paths use relative `./data/...`).
- Shared peptide subset (legacy): `evaluation_scripts/select_knockout_peptide_subset.py`
- Exp5 needs `biopython` and `statsmodels` (listed in `environment.yaml`).

### Quick Examples

```bash
export REPO_ROOT=/path/to/AMPCliff
cd "${REPO_ROOT}"

# Exp1–4 full test set (one experiment)
bash evaluation_scripts/run_fftlag_exp1_fulltest.sh
# sbatch evaluation_scripts/run_fftlag_exp1_fulltest_slurm.sh

# Exp5 official pipeline (Steps 1–5)
bash evaluation_scripts/run_dc_validation_v2.sh

# Legacy 30-peptide Exp1/2/4 subset
bash evaluation_scripts/run_fftlag_mechanism_experiments.sh
```

---

## Ablation Scripts

### A. Pooling Baseline Comparison (mean / max / attn_structured / FLaG)

| Script | Role |
|--------|------|
| `evaluation_scripts/run_baseline_pooling_train.sh` | Single run; set `POOLING=` |
| `evaluation_scripts/run_baseline_pooling_grid_2x2_seeds_{1-4}.sh` | 2×2 × multi-seed batch training |
| `evaluation_scripts/run_pooling_main.sh` | Shortcut to `evaluation_scripts/run_pooling_baseline_ablation.sh` |
| `evaluation_scripts/run_pooling_baseline_ablation.sh` | SpectralAnchor ablation (num_anchor / use_fft sweep) |

### B. Results Aggregation & Statistics

| Script | Role |
|--------|------|
| `extract_ablation_metrics.py` | Extract pearson / spearman / recall from `outputs/ablation/` |
| `aggregate_ablation_results.py` | Aggregate mean±std by config × seed |
| `analyze_ablation_results.py` | Analyze summary CSV |
| `collect_ablation_summary.py` | Collect ablation summary |
| `split_ablation_metrics_to_excel.py` | Export to Excel |
| `evaluation_scripts/compute_pooling_ablation_stats_new_data.py` | Pooling ablation statistics |
| `evaluation_scripts/ablation_new_data_stats_to_xlsx.py` | Export statistics to xlsx |
| `evaluation_scripts/merge_seed_metrics_pooling_csv_to_xlsx.py` | Merge seed metrics |
| `run_ablation_protein.sh` | Multi-component pooling ablation batch training (see note below) |

Supported poolings: `mean`, `max`, `last`, `latent_attn`, `attn_structured`, `mltp_paper`, `FLaG`.

```bash
POOLING=attn_structured MODEL_TYPE=esm2_t6 DATASET=s_aureus bash evaluation_scripts/run_baseline_pooling_train.sh
POOLING=mltp_paper MODEL_TYPE=esm2_t6 DATASET=s_aureus bash evaluation_scripts/run_baseline_pooling_train.sh
```

> **Note:** `run_pooling_baseline_ablation.sh` and `run_ablation_protein.sh` are legacy spectral ablation scripts; this release no longer registers `spectral_anchor` and related old poolings.

---

## Key Configuration

FLaG-related keys in `configs/downstream.yaml`:

```yaml
features.type: LLM
model.regression.version: esm2_t6   # or esm2_t12
model.regression.pooling: FLaG
model.regression.apply: none
model.regression.pooling_config.FLaG:
  num_heads: 4
  num_latents: 8
  time_pool: max        # max | mean
  gate_residual: true
  use_gate: true
  use_latent: true
data.regression.mode: fix
```

**Load checkpoint** (evaluation / mechanism experiments):

```yaml
model.regression.check_point.load: true
model.regression.check_point.path: /path/to/seed_0/data/model.pth
```

Or via Hydra CLI:

```bash
python downstream_evaluate.py \
  model.regression.check_point.load=true \
  model.regression.check_point.path="/path/to/model.pth"
```

**Debug mode:** `other.debug=true` (uses a reduced dataset for quick smoke tests).

---

## Script Reference

| Script Path | Type | Description |
|-------------|------|-------------|
| `downstream_train.py` | Core | Hydra training entry |
| `downstream_evaluate.py` | Core | Standard evaluation entry |
| `downstream_evaluate_spectrual_filter.py` | Mechanism | Exp1: band knockout |
| `downstream_evaluate_psd_gate.py` | Mechanism | Exp2: gate PSD |
| `downstream_evaluate_knockout.py` | Mechanism | Exp3: token knockout |
| `downstream_evaluate_fft_lag_latent.py` | Mechanism | Exp4: latent visualization |
| `evaluation_scripts/run_fftlag_exp{1,2,3,4}_fulltest.sh` | Mechanism | Exp1–4 full-test batch runners |
| `evaluation_scripts/plot_fftlag_exp1_fulltest_violin.py` | Mechanism | Exp1 $|\Delta\mathrm{P}|$ violin plots |
| `evaluation_scripts/plot_exp1_abs_mse_violin_heatmap_composite_allblue_violin_u1.py` | Mechanism | Exp1 violin+heatmap composite |
| `evaluation_scripts/plot_fftlag_exp4_frequency_attention_combined.py` | Mechanism | Exp4 frequency-attention combined figure |
| `analyze_fft_lag_mechanism_by_structure.py` | Mechanism | Exp5 optional structure bucketing |
| `evaluation_scripts/run_dc_validation_v2.sh` | Mechanism | Exp5: official pipeline (Steps 1–5) |
| `build_dc_property_table.py` | Mechanism | Exp5 Step 1: property table |
| `extract_dct_coefficient_features.py` | Mechanism | Exp5 Step 2: DCT features |
| `analyze_dc_property_encoding.py` | Mechanism | Exp5 Step 3 / main exp. 1: DC decoding |
| `analyze_species_property_effects.py` | Mechanism | Exp5 Step 4 / main exp. 2A: species effects |
| `evaluation_scripts/run_dc_property_knockout_fulltest.sh` | Mechanism | Exp5 Step 5 / main exp. 2B: property buckets |
| `evaluation_scripts/run_fftlag_mechanism_experiments.sh` | Mechanism | Legacy Exp1/2/4 subset orchestration |
| `evaluation_scripts/run_fftlag_exp4_only.sh` | Mechanism | Exp4 only (SLURM-friendly) |
| `evaluation_scripts/run_fftlag_exp4_attn_score_raw.sh` | Mechanism | Exp4 attn-score variant |
| `evaluation_scripts/select_knockout_peptide_subset.py` | Mechanism | Build shared peptide manifest |
| `evaluation_scripts/aggregate_fftlag_mechanism_seeds.py` | Mechanism | Cross-seed mechanism metric aggregation |
| `evaluation_scripts/plot_fftlag_mechanism_per_sample_seeds.py` | Mechanism | Per-sample mechanism plots |
| `evaluation_scripts/aggregate_amp_knockout_seed_csvs.py` | Mechanism | Exp3 cross-seed aggregation |
| `evaluation_scripts/plot_amp_knockout_figure.py` | Mechanism | Exp3 result plotting |
| `downstream_evaluate_knockout.sh` | Mechanism | Exp3 SLURM/batch entry |
| `evaluation_scripts/run_baseline_pooling_train.sh` | Training | Single pooling training run |
| `evaluation_scripts/run_baseline_pooling_grid_2x2_seeds_{1-4}.sh` | Training | 2×2 multi-seed grid training |
| `evaluation_scripts/run_aggregate_baseline_pooling_seed_metrics.sh` | Training | Aggregate baseline seed metrics |
| `evaluation_scripts/aggregate_pooling_seed_metrics.py` | Training | Seed metric aggregation (Python) |
| `evaluation_scripts/run_pooling_main.sh` | Ablation | Pooling ablation shortcut |
| `evaluation_scripts/run_pooling_baseline_ablation.sh` | Ablation | SpectralAnchor hyperparameter sweep |
| `run_ablation_protein.sh` | Ablation | Component pooling ablation (see registry note) |
| `extract_ablation_metrics.py` | Analytics | Extract ablation metrics |
| `aggregate_ablation_results.py` | Analytics | Aggregate ablation results |
| `analyze_ablation_results.py` | Analytics | Analyze ablation CSV |
| `collect_ablation_summary.py` | Analytics | Collect ablation summary |
| `split_ablation_metrics_to_excel.py` | Analytics | Export to Excel |
| `evaluation_scripts/compute_pooling_ablation_stats_new_data.py` | Analytics | New-data pooling statistics |
| `evaluation_scripts/ablation_new_data_stats_to_xlsx.py` | Analytics | Export statistics to xlsx |
| `evaluation_scripts/merge_seed_metrics_pooling_csv_to_xlsx.py` | Analytics | Merge seed metrics to xlsx |
| `downstream_train.sh` | Shell | SLURM training wrapper |
| `downstream_evaluate.sh` | Shell | SLURM evaluation wrapper |

---

## Citation & Acknowledgements

### FLaG

If you use the FLaG pooling method or this codebase, please cite our paper:

https://arxiv.org/abs/2606.08191

```bibtex
@article{li2026flag,
  title={Frequency-Domain Latent Attention Gating for Cross-Domain Token Aggregation},
  author={Li, Kewei and Zhang, Rongying and Wang, Xueli and Gong, Xiwen and Wang, Zhongjian and Huang, Lan and Zhang, Ruochi and Zhou, Fengfeng},
  journal={arXiv preprint arXiv:2606.08191},
  year={2026},
  url={https://arxiv.org/abs/2606.08191}
}
```

### AMPCliff (Foundation)

FLaG is built upon AMPCliff. If you use the activity cliff dataset or benchmark framework, please also cite the published AMPCliff paper (*Journal of Advanced Research*, 2026):

```bibtex
@article{AMPCliff,
  title={AMPCliff: Quantitative definition and benchmarking of activity cliffs in antimicrobial peptides},
  author={Li, Kewei and Wu, Yuqian and Li, Yinheng and Guo, Yutong and Kong, Yanwen and Wang, Yan and Liang, Yiyang and Fan, Yusi and Huang, Lan and Zhang, Ruochi and Zhou, Fengfeng},
  journal={Journal of Advanced Research},
  volume={80},
  pages={287--300},
  year={2026},
  issn={2090-1232},
  doi={10.1016/j.jare.2025.04.046}
}
```

https://doi.org/10.1016/j.jare.2025.04.046

---

## Contact

kwbb1997@gmail.com or FengfengZhou@gmail.com

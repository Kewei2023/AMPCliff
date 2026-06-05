<p align="right">
  <strong>English</strong> · <a href="README_zh.md">中文</a>
</p>

# FLaG — Frequency-Domain Latent Attention Gating for Cross-Domain Token Aggregation

*Built on [AMPCliff](https://github.com/Kewei2023/AMPCliff)*

**FLaG** (Frequency-Domain Latent Attention Gating for Cross-Domain Token Aggregation) is an independent research project built on [AMPCliff](https://github.com/Kewei2023/AMPCliff) for antimicrobial peptide **activity cliff** prediction. AMPCliff provides the activity cliff dataset, benchmark framework, and ESM2 downstream training infrastructure; FLaG introduces the core pooling method `fft_latent_attn_gate` along with mechanism analysis and ablation experiments.

Code is hosted on the **`FLaG` branch** of the AMPCliff repository (the `AMPCliff` Python package name and directory layout are retained for historical reasons).

---

## Table of Contents

- [Setup & Environment](#setup--environment)
- [Data & Model Weights](#data--model-weights)
- [Training FLaG](#training-flag)
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

FLaG pooling is implemented as `FFTLatentAttentionGatePooling` in `factory/pooling/spectral_anchor_v2.py`:

**rFFT → latent attention → gate → iFFT → time pooling**

Config key: `model.regression.pooling=fft_latent_attn_gate`

### Single Run

```bash
export REPO_ROOT=/path/to/AMPCliff
cd "${REPO_ROOT}"

POOLING=fft_latent_attn_gate MODEL_TYPE=esm2_t6 DATASET=s_aureus \
  bash evaluation_scripts/run_baseline_pooling_train.sh
```

- Training entry: `downstream_train.py` + `configs/downstream.yaml`
- Evaluation entry: `downstream_evaluate.py`

### Multi-Seed 2×2 Grid (esm2_t6/t12 × e_coli/s_aureus)

Sharded parallel scripts (set `POOLINGS` for baseline comparison):

```bash
# FLaG only
POOLINGS="fft_latent_attn_gate" bash evaluation_scripts/run_baseline_pooling_grid_2x2_seeds_1.sh

# Compare against mean / max / attn
POOLINGS="mean max attn fft_latent_attn_gate" bash evaluation_scripts/run_baseline_pooling_grid_2x2_seeds_1.sh
# seeds_2.sh ~ seeds_4.sh are the remaining shards
```

### Aggregate Seed Metrics

```bash
bash evaluation_scripts/run_aggregate_baseline_pooling_seed_metrics.sh
# or
python evaluation_scripts/aggregate_pooling_seed_metrics.py
```

---

## Five Mechanism Experiments

Mechanism experiments analyze FLaG's internal behavior on the activity cliff task. **Train FLaG models first** (checkpoints are resolved by default from `outputs/ablation_new_data/{model}_fft_latent_attn_gate_{dataset}_diff5/seed_*/data/model.pth`).

| Exp | Purpose | Main Python Script | Batch / Utilities |
|-----|---------|-------------------|-------------------|
| **Exp1** Band knockout | Sequence band notch sensitivity | `downstream_evaluate_spectrual_filter.py` | `evaluation_scripts/run_fftlag_mechanism_experiments.sh` (`RUN_EXPS` includes `1`) |
| **Exp2** Gate PSD | Spectral energy change before/after gate | `downstream_evaluate_psd_gate.py` | Same script (`RUN_EXPS` includes `2`) |
| **Exp3** Token knockout | Token perturbation response distribution | `downstream_evaluate_knockout.py` | `downstream_evaluate_knockout.sh`; aggregate `evaluation_scripts/aggregate_amp_knockout_seed_csvs.py`; plot `evaluation_scripts/plot_amp_knockout_figure.py` |
| **Exp4** Latent viz | Latent query band mass distribution | `downstream_evaluate_fft_lag_latent.py` | Same script (`RUN_EXPS=4`) or `evaluation_scripts/run_fftlag_exp4_only.sh`; attn-score variant `evaluation_scripts/run_fftlag_exp4_attn_score_raw.sh` |
| **Exp5** Structure | Structure-stratified secondary analysis | `analyze_fft_lag_mechanism_by_structure.py` | Called at end of `evaluation_scripts/run_fftlag_mechanism_experiments.sh` when `RUN_EXP5=1` |

### Important Notes

- `evaluation_scripts/run_fftlag_mechanism_experiments.sh` runs **Exp1/2/4 by default** (`RUN_EXPS=1,2,4`) and **does not include Exp3**; run the knockout pipeline separately for Exp3.
- Shared peptide subset manifest: `evaluation_scripts/select_knockout_peptide_subset.py`
- Config files:
  - Exp1/4: `configs/evaluate_fftlag_mechanism.yaml`
  - Exp2: `configs/evaluate_psd_gate.yaml`
  - Exp3: `configs/downstream_knockout.yaml`
- Cross-seed aggregation/plotting:
  - `evaluation_scripts/aggregate_fftlag_mechanism_seeds.py`
  - `evaluation_scripts/plot_fftlag_mechanism_per_sample_seeds.py`

### Quick Examples

```bash
export REPO_ROOT=/path/to/AMPCliff
cd "${REPO_ROOT}"

# Exp1 + Exp2 + Exp4 + Exp5 (default)
bash evaluation_scripts/run_fftlag_mechanism_experiments.sh

# Exp4 only
RUN_EXPS=4 bash evaluation_scripts/run_fftlag_mechanism_experiments.sh

# Exp3 (requires checkpoint; see Hydra overrides in configs/downstream_knockout.yaml)
bash downstream_evaluate_knockout.sh
```

---

## Ablation Scripts

### A. Pooling Baseline Comparison (mean / max / attn / FLaG)

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

> **Note:** `run_ablation_protein.sh` references component pooling names such as `fft_latent_only`, but the FLaG branch `factory/pooling/registry.py` only registers standard poolings (including `fft_latent_attn_gate`). Unregistered names will fail at runtime; restore the corresponding pooling in the registry if you need component ablations.

---

## Key Configuration

FLaG-related keys in `configs/downstream.yaml`:

```yaml
features.type: LLM
model.regression.version: esm2_t6   # or esm2_t12
model.regression.pooling: fft_latent_attn_gate
model.regression.apply: none
model.regression.pooling_config.fft_latent_attn_gate:
  num_heads: 4
  num_latents: 8
  time_pool: max        # max | mean | attn
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
| `analyze_fft_lag_mechanism_by_structure.py` | Mechanism | Exp5: structure-stratified analysis |
| `evaluation_scripts/run_fftlag_mechanism_experiments.sh` | Mechanism | Exp1/2/4/5 batch orchestration |
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

If you use the FLaG pooling method or this codebase, please cite our paper (bibtex to be updated).

### AMPCliff (Foundation)

FLaG is built upon AMPCliff. If you use the activity cliff dataset or benchmark framework, please also cite:

```bibtex
@article{AMPCliff,
  title={AMPCliff: quantitative definition and benchmarking of activity cliffs in antimicrobial peptides},
  author={Kewei Li, Yuqian Wu, Yinheng Li, Yutong Guo, Yan Wang, Yiyang Liang, Yusi Fan, Lan Huang, Ruochi Zhang, Fengfeng Zhou},
  journal={arXiv},
  year={2024}
}
```

---

## Contact

kwbb1997@gmail.com or FengfengZhou@gmail.com

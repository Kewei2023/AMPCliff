# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AMPCliff is a PyTorch-based machine learning project for predicting Antimicrobial Peptide (AMP) activity cliffs. It combines traditional ML methods with deep learning using protein language models (ESM2, GPT2, ProGen2, ProtGPT2, BERT) and custom architectures (CellFree, AMPSpace, peptimizer, TaiChiNet2).

## Common Commands

### Environment Setup
```bash
conda env create -f environment.yaml
conda activate AMPCliff
```

### Training

**Deep Learning / LLM Training:**
```bash
# Local training (single GPU)
python downstream_train.py

# With config override (recommended for switching models)
python downstream_train.py model.config_dir="/path/to/model/" model.regression.version="esm2_t12"

# Cluster training (SLURM)
sbatch --gpus=1 downstream_train.sh
```

**Traditional ML Training:**
```bash
python machine_learning_train.py
sbatch --gpus=1 machine_learning_train.sh  # Cluster
```

### Evaluation
```bash
python downstream_evaluate.py                    # Main evaluation
python downstream_evaluate_interpret_features.py # Feature interpretation
python downstream_evaluate_knockout.py           # Attention knockout
python downstream_evaluate_spectrual_filter.py   # Spectral filtering
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_pooling_registry.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## Architecture Overview

### Entry Points and Configuration

The project uses **Hydra** for configuration management. Main config file is `configs/downstream.yaml`.

Key configuration patterns:
- `mode.ddp`: Set to `true` for multi-GPU distributed training
- `mode.amp`: Set to `true` for automatic mixed precision
- `task.type`: `regression` or `classification`
- `features.type`: Feature extraction type (see Model Pairing below)
- `model.regression.version`: Model architecture to use
- `model.regression.apply`: DC component filtering strategy (`filter_dc`, `scale_dc`, `concat_dc`, `distill_vc`, `none`)
- `model.regression.pooling`: Pooling method (`mean`, `max`, `attn`, `spectral_anchor`, …)
- `model.regression.pooling_common` / `model.regression.pooling_config.<pooling>`: 分层 pooling 超参（`resolve_pooling_kwargs` 合并；扁平 `model.regression.num_anchor=` 等 CLI 仍可覆盖）
- `data.regression.mode`: Data split mode (`random` for 5-fold CV, `fix` for AC split)
- `data.regression.dataset`: Target dataset (`s_aureus` or `e_coli`)

### Model and Feature Type Pairing

**Critical:** `features.type` must match `model.regression.version`:

| Model Type | `features.type` | `model.regression.version` |
|------------|-----------------|---------------------------|
| LLMs | `LLM` | `esm2_t6`, `esm2_t12`, `esm2_t33`, `esm2_t48`, `gpt2-base`, `gpt2-large`, `progen2-small`, `progen2-base`, `progen2-medium`, `protgpt2`, `bert-base` |
| Custom | `CellFree-cnn` | `CellFree-cnn` |
| Custom | `CellFree-rnn` | `CellFree-rnn` |
| Custom | `AMPSpace` | `AMPSpace` |
| Custom | `peptimizer` | `peptimizer` |

**ESM2 Model Paths:** On the HPC shared filesystem, pretrained weights live under **`/data/public/models/facebook/`**. The `config_dir_mapping` in [`factory/initializer.py`](factory/initializer.py) (ESM2 branch) resolves:

| `model.regression.version` | Directory (append trailing `/` for Hydra) |
|-----------------------------|-------------------------------------------|
| `esm2_t6` | `/data/public/models/facebook/esm2_t6_8M_UR50D/` |
| `esm2_t12` | `/data/public/models/facebook/esm2_t12_35M_UR50D/` |
| `esm2_t33` | `/data/public/models/facebook/esm2_t33_650M_UR50D/` |
| `esm2_t48` | `/data/public/models/facebook/esm2_t48_15B_UR50D/` |

If your machine does not mount `/data/public`, override at runtime, e.g. `python downstream_train.py model.config_dir="/path/to/esm2_t12_35M_UR50D/"`, or edit `config_dir_mapping` locally. Older paths such as `/mnt/g/...` are no longer the defaults in this repo.

### Module Structure

**`factory/`** - Model initialization and architecture
- `initializer.py`: Main entry point for model loading. Handles LLMs (ESM2, GPT2, ProGen2, ProtGPT2, BERT) and custom models. Contains version-to-path mappings.
- `regression.py`: Regression wrappers (`RegModel_v1`, `RegModel_v2`, `RegModel_bagua`, `RegModel_KnockDC`, `RegModel_ScaleDC`, `RegModel_ConcatDC`, `RegModel_DistillVC`)
- `classification.py`: Classification wrappers
- `pooling/`: Pooling method registry and implementations (`mean`, `max`, `attn`, `spectral_anchor`)
- `AMPSpace.py`: LSTM-based architecture
- `CellTree.py`: CNN/RNN regressors
- `peptimizer.py`: Peptimizer regressor

**`loader/`** - Data pipeline
- `utils.py`: Data loader with collate functions
- `dataset.py`: Dataset classes (`LSDataset`, `ACDataset`)
- `split.py`: Data splitting strategies (random, stratified, fixed cluster)
- `preprocess_data.py`: Preprocessing utilities

**`features/`** - Feature extraction
- `feature_fetcher.py`: Unified interface for feature extraction (LLM features vs hand-crafted features)
- Hand-crafted features: `AAComposition`, `Autocorrelation`, `BasicDes`, `CTD`, `PseudoAAC`, `QuasiSequenceOrder`, `fingerprint_2d`

**`taichinet/`** - TaiChiNet2 (Fourier-transform-based architecture)
- `fourier_transform.py`: FFT processing utilities
- Inter-layer and intra-layer spectral analysis
- `visual.py`, `visual_2d.py`: Visualization tools

**`spectrual_filter/`** - Spectral filtering techniques
- `filter.py`: DC component filtering (DCT/FFT)
- `rotary_knock.py`: Rotary position knockout
- `hidden_energy.py`: Hidden state energy analysis

**`attention_knockout/`** - Attention mechanism analysis
- `as_ko.py`: Attention score knockout
- `hs_ko.py`: Hidden state knockout

**`utils/`** - Core utilities
- `trainer.py`: Training loop implementation
- `evaluator.py`: Evaluation metrics (Spearman, Pearson, MSE)
- `std_logger.py`: Logging system
- `path_helper.py`: Path resolution utilities
- `orthogonal_constraint.py`: Orthogonal constraint loss implementation
- `orthogonal_metrics.py`: Metrics for measuring orthogonality

**`visualization/`** - Plotting tools
- `plot.py`: t-SNE, UMAP visualizations

### Training Workflow

1. Configuration via `downstream.yaml`
2. Data loading through `loader/` with specified split mode
3. Feature extraction based on `features.type`
4. Model initialization via `factory/initializer.py`
5. Training loop in `utils/trainer.py` with optional DDP/AMP
6. Evaluation at intervals using `utils/evaluator.py`
7. MLFlow tracking (configured via conda env vars)

### DC Component Processing (Key Research Feature)

The project implements several novel techniques for processing DC (direct current) components in hidden states:
- `filter_dc`: Remove DC component using DCT/FFT
- `scale_dc`: Scale DC component
- `concat_dc`: Concatenate DC with filtered signal
- `distill_vc`: Amplify variable components

These are applied in the regression models and controlled via `model.regression.apply` in configs.

### Pooling Methods

Pooling is managed through `factory/pooling/registry.py`. Supported methods:
- `mean`: Average pooling over sequence length
- `max`: Max pooling over sequence length
- `attn`: Attention-based pooling (learnable)
- `spectral_anchor`: Spectral anchor pooling using FFT-based anchors

### Orthogonal Constraint

Optional regularization to constrain different layers' output representations to be orthogonal. Controlled via `orthogonal_constraint` in config:
```yaml
orthogonal_constraint:
  enabled: true
  weight: 0.01                # Regularization strength
  layer_indices: null         # null = all layers, or [0,2,4,6]
  constraint_type: "pairwise" # "pairwise", "sequential", or "gram"
  normalize: true
```

## Code Style

### Imports
```python
# Order: stdlib → external → local (absolute imports with AMPCliff prefix)
import os
import torch
from transformers import AutoModel
from AMPCliff.utils.std_logger import Logger
from AMPCliff.factory.initializer import ModelInitializer
```

### Naming Conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/Variables: `snake_case`

### Logging
Use `Logger.info()` and `Logger.error()` from `AMPCliff.utils.std_logger`. Avoid print statements.

## Important Notes

- **Import style:** All local imports use absolute paths with `AMPCliff` prefix
- **Debug mode:** Set `other.debug: true` in config for reduced dataset/sample testing
- **Checkpoint loading:** Set `model.regression.check_point.load: true` and provide path to resume/evaluate
- **Output directories:** Hydra auto-creates timestamped output directories under `outputs/` with format: `outputs/YYYY-MM-DD/HH-MM-SS_{dataset}_{model}_{apply}_{pooling}/`
- **No ipdb in production:** Debug code with `ipdb` should not be committed

## Data Location

Data should be placed in `./data/` folder. Activity cliff generation is handled by the separate [AMPCliff-generation](https://github.com/Kewei2023/AMPCliff-generation) repository.

## Common Issues

**ESM2 Model Paths:** Defaults use `/data/public/models/facebook/...` (see **Model and Feature Type Pairing** above). On machines without that mount, override:

```bash
python downstream_train.py model.config_dir="/your/path/to/esm2_t12_35M_UR50D/"
```

## Session Continuity

### 新会话开始时
使用触发词查询进度: "现在什么情况了" 或 "这个项目做到哪了"

### 完成重要工作后
使用触发词记录进度: "记录实验" 或 "你记一下"

### 统计缺失实验
使用触发词: "统计一下还缺什么训练结果" 或 "还剩多少训练实验没跑"

### 进度文件位置
- 快照: `.claude/progress/progress_snapshot_*.md`
- 日志: `.claude/progress/knockout_quick_*.log`

# AMPCliff Agent Guide

## Overview
AMPCliff is a PyTorch-based deep learning project for AMP (Antimicrobial Peptide) activity cliff prediction. The project uses pre-trained protein language models (ESM2, GPT2, ProGen2, ProtGPT2) and custom architectures (TaiChiNet, CellFree-cnn/rnn, AMPSpace, peptimizer).

## Environment Setup

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate AMPCliff
```

## Build & Run Commands

### Training (Deep Learning / LLMs)
```bash
# Local machine
python downstream_train.py

# SLURM cluster (with GPU)
sbatch --gpus=1 downstream_train.sh

# With specific model configuration
python downstream_train.py model.config_dir="/path/to/model/" model.regression.version="esm2_t12"
```

### Training (Machine Learning)
```bash
# Local machine
python machine_learning_train.py

# SLURM cluster
sbatch --gpus=1 machine_learning_train.sh
```

### Evaluation
```bash
python downstream_evaluate.py
# Or via SLURM
sbatch --gpus=1 downstream_evaluate.sh
```

### Feature Extraction
```bash
python extract_features.py
```

### Running Specific Tests/Scripts
```bash
# Example specific evaluation scripts
python test_bnn.py
python downstream_evaluate_bagua_visual.py
python classification_head_visualization.py
```

## Configuration

### Hydra + OmegaConf
- All configurations in `configs/` directory
- Main configs: `downstream.yaml`, `ML.yaml`, `classification.yaml`
- Modify configs in YAML files or override via command line:
  ```bash
  python script.py config.path=value
  ```

### Key Configuration Sections
- `mode`: Training mode (ddp for distributed, amp for mixed precision)
- `task.type`: regression or classification
- `features.type`: LLM, CellFree-cnn, CellFree-rnn, AMPSpace, peptimizer
- `model.regression.version`: Model version (esm2_t6, gpt2-base, progen2-medium, etc.)
- `model.regression.pooling`: max, mean, attn 等；超参见 `pooling_common` 与 `pooling_config.<pooling>`（或旧版扁平字段 / CLI 覆盖）
- `model.regression.apply`: none, filter_dc, distill_vc, scale_dc

### MLFlow Tracking
Set environment variables before running:
```bash
export MLFLOW_EXPERIMENT_NAME=breeze
export MLFLOW_TRACKING_URI=http://192.168.1.23:5002
```

## Code Style Guidelines

### Imports
```python
# Standard library imports
import os
import sys
from pathlib import Path

# External packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel

# Local project imports (using absolute paths with AMPCliff prefix)
from AMPCliff.utils.std_logger import Logger
from AMPCliff.loader.utils import make_loader
from AMPCliff.factory.initializer import ModelInitializer
```

- **Order**: stdlib → external → local (absolute imports preferred)
- **No unused imports**: Remove imports that are commented out
- **Debug tool**: `ipdb` is used for debugging; do not commit `ipdb.set_trace()` calls

### File & Directory Organization
- Root level: Main scripts (e.g., `downstream_train.py`, `machine_learning_train.py`)
- `configs/`: YAML configuration files
- `factory/`: Model definitions (regression, classification, ML, specialized models)
- `loader/`: Data loading and preprocessing
- `utils/`: Utilities (trainer, evaluator, metrics, logging)
- `taichinet/`: TaiChiNet2 model implementation
- `features/`: Feature extraction modules
- `visualization/`: Plotting and visualization tools

### Naming Conventions
- **Files**: `snake_case.py` (e.g., `downstream_train.py`, `feature_fetcher.py`)
- **Classes**: `PascalCase` (e.g., `ModelInitializer`, `Trainer`, `LSDataset`)
- **Functions/Methods**: `snake_case` (e.g., `get_device`, `load_weights`, `make_loader`)
- **Variables**: `snake_case`
- **Constants**: `UPPER_CASE` in configs

### Logging
```python
from AMPCliff.utils.std_logger import Logger

Logger.info("Training started...")
Logger.error("Failed to load checkpoint")
```
- Use `Logger.info()` for informational messages
- Use `Logger.error()` for error messages
- Avoid print statements in production code

### Error Handling
- Use try-except for I/O operations (file loading, network requests)
- Use assertions for data validation:
  ```python
  assert len(sequence) == len(seqName) == len(seqID)
  ```
- Include meaningful error messages

### Type Hints
- Used in newer modules but not consistently enforced
- Use typing hints when adding new functions:
  ```python
  def get_device(cfg) -> torch.device:
      pass
  ```

### Path Handling
```python
from pathlib import Path

best_model_path = Path(best_model_path) / "data/model.pth"
os.makedirs(savedir, exist_ok=True)
```

## Working with Models

### Loading Pre-trained Models
Models are loaded via `ModelInitializer` in `factory/initializer.py`:
- Supported: ESM2, GPT2, ProGen2, ProtGPT2, BERT-base
- Config paths point to local model directories
- Tokenizers loaded from model paths (GPT2 requires pad_token = eos_token)

### Custom Models
- `factory/regression.py`: Regression models (RegModel_v1, RegModel_bagua, etc.)
- `factory/classification.py`: Classification models
- `factory/AMPSpace.py`, `factory/peptimizer.py`, `factory/CellTree.py`: Specialized architectures
- `taichinet/`: TaiChiNet2 implementation

## Training Patterns

### DDP (Distributed Data Parallel)
```python
if cfg.mode.ddp:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    global_rank = int(os.environ['RANK'])
    device = torch.device("cuda", local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

### Random Seed
```python
from AMPCliff.utils.utils import fix_random_seed

fix_random_seed(random_seed, cuda_deterministic=True)
```

### Mixed Precision (AMP)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Metrics & Evaluation

### Default Metrics
- Regression: `spearman` correlation
- Also tracks MSE, Pearson correlation, top-k metrics

### Evaluating Models
- Use `Evaluator` class from `utils/evaluator.py`
- Run evaluation at specified epochs (`train.eval_epoch` in config)
- Results saved to `outputs/` directory

## Debug Mode
Set `other.debug: true` in config to enable debugging features.

## Important Notes
- The project folder name must be `AMPCliff` (required for absolute imports)
- Set `model.regression.check_point.load: true` and provide path for loading checkpoints
- Data files should be in `./data/` directory
- Model outputs saved to `./outputs/` directory with timestamp
- Ignore `slurm-*.out` and `nohup.out` log files

## HPC 实验脚本编写规范

### 1. 优先写 Shell 脚本而非 Python 包装脚本
- 所有批量实验（ablation、多 seed、多 dataset）**必须**写 `.sh` 脚本 + Hydra CLI override，复用项目已有的 `downstream_train.py` / `downstream_evaluate.py` 入口。
- **禁止**写 Python 包装脚本绕过 `downstream_train.py`（如手写 `run_xxx.py` 调用 `Trainer`/`Evaluator`），这会导致：
  - 需要 `OmegaConf.set_struct(False)` hack
  - 需要手动设 `orig_cwd`
  - 产出格式与项目其他实验不一致
- Shell 脚本通过 `python -u downstream_train.py key=value key2=value2` 传递所有配置。

### 2. SLURM 脚本模板
```bash
#!/bin/bash
#SBATCH --job-name=<short_name>
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0                # 用 0，不自定义内存（N26区每卡默认38GB，不允许超额）
#SBATCH --time=72:00:00
#SBATCH --chdir=/data/home/scv6872/run/kwli/AMPCliff
#SBATCH --output=logs/<name>_%j.out
#SBATCH --error=logs/<name>_%j.err
set -uo pipefail
```

### 3. 环境激活（固定的 module load + conda）
```bash
module load miniforge gcc/11.1.0 cuda/11.1 gcc/11.1.0
module load cudnn/8.1.0.77_CUDA11.1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate AMPCliff
```
- 注意：旧脚本中 `source activate AMPCliff` 也能用，但 `eval + conda activate` 更标准。

### 4. Hydra CLI Override 传参
Shell 脚本中通过 Hydra CLI override 修改 OmegaConf 配置，**不需要** `set_struct(False)`：
```bash
python -u downstream_train.py \
    model.config_dir="/data/public/models/facebook/esm2_t6_8M_UR50D/" \
    model.regression.version="esm2_t6" \
    model.regression.pooling="${POOLING}" \
    model.regression.apply="${APPLY}" \
    model.regression.check_point.load=false \
    data.regression.dataset="${DATASET}" \
    data.regression.mode=fix \
    data.regression.fix.train_file="${TRAIN_FILE}" \
    data.regression.fix.valid_file="${VALID_FILE}" \
    data.regression.fix.test_file="${TEST_FILE}" \
    data.diff="[5]" \
    data.threshold="0.9" \
    data.regression.condition="[blosum62 average]" \
    train.random_seed="${SEED}" \
    mode.ddp=false \
    mode.amp=false \
    logger.log=false \
    other.debug=false \
    hydra.run.dir="${SEED_DIR}"
```

- 列表参数用 `"[val1, val2]"` 语法。
- Hydra 在 CLI override 阶段内部处理了 struct mode，所以不需要 `set_struct`。

### 5. 固定路径

| 用途 | 路径 |
|------|------|
| 数据根目录 | `/data/home/scv6872/run/kwli/AMPCliff/data/blosum62 average/diff_5-trd_0.9` |
| ESM2-8M | `/data/public/models/facebook/esm2_t6_8M_UR50D/` |
| ESM2-35M | `/data/public/models/facebook/esm2_t12_35M_UR50D/` |
| ESM2-650M | `/data/public/models/facebook/esm2_t33_650M_UR50D/` |
| 输出目录 | `${REPO_ROOT}/outputs/<experiment_name>/` |
| Checkpoint 约定 | `${OUTPUT_ROOT}/${MODEL}_${POOLING}_${DATASET}_diff${DIFF}/seed_${SEED}/` |
| 日志目录 | `${REPO_ROOT}/logs/` |

- 数据文件命名：`grampa_${DATASET}_7_25-{train,valid,test,all-train}.csv`
- **不要**用 `/data/run01/scv6872/kwli/AMPCliff/data/blast/blosum62 average/diff_5`，也不要用 `/mnt/g/` 开头的路径。

### 6. Shell 脚本常见 Bug 与防范

#### `set -u` 导致未定义变量报错
- **现象**：`line N: VAR_NAME: unbound variable`
- **原因**：`set -u` 要求所有变量必须已定义
- **修复**：用 `${VAR:-default}` 语法给默认值
```bash
DRY_RUN="${DRY_RUN:-}"          # 默认空字符串
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/ablation}"
SEEDS="${SEEDS:-0 1 2 3 4}"
```

#### SLURM 内存超额
- **现象**：`不允许超额申请内存`
- **修复**：用 `--mem=0`，不指定具体数值

#### Python 相对导入错误
- **现象**：`ValueError: attempted relative import beyond top-level package`
- **修复**：从项目父目录运行，用 `from AMPCliff.xxx import yyy` 绝对导入
```bash
cd /data/home/scv6872/run/kwli  # 而非 cd AMPCliff
python -m AMPCliff.downstream_train  # 或直接在 AMPCliff 目录下运行
```

#### Hydra `get_original_cwd()` 报错
- **现象**：`ValueError: get_original_cwd() must only be used after HydraConfig is initialized`
- **原因**：在非 `@hydra.main` 的 Python 代码中调用了 `hydra.utils.get_original_cwd()`
- **修复**：不要绕过 `downstream_train.py`；如果必须，用 `os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))`

#### OmegaConf struct mode 阻止修改
- **现象**：`omegaconf.errors.ConfigAttributeError: Key 'xxx' is not in struct`
- **原因**：在 Python 中直接对 `DictConfig` 对象赋值
- **修复**：**用 Shell 脚本 + Hydra CLI override**，不需要 `set_struct(False)`

### 7. 实验脚本标准结构
```bash
#!/bin/bash
#SBATCH --job-name=xxx
#SBATCH --gpus=1 ...
set -uo pipefail

# 1. 环境激活
module load miniforge gcc/11.1.0 cuda/11.1 gcc/11.1.0
module load cudnn/8.1.0.77_CUDA11.1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate AMPCliff

# 2. 路径和参数（全部用 ${VAR:-default} 语法）
REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"
DRY_RUN="${DRY_RUN:-}"
SEEDS="${SEEDS:-0 1 2 3 4}"

# 3. 打印配置摘要
echo "========== Config =========="
echo "REPO_ROOT: ${REPO_ROOT}"
echo "SEEDS: ${SEEDS}"

# 4. 循环执行（带 skip 已完成、错误捕获、计数）
cd "${REPO_ROOT}"
for ... ; do
    # 检查是否已完成（检测输出文件）
    if [[ -f "${RESULT_FILE}" ]]; then
        echo "[SKIP] ..."
        continue
    fi

    # DRY_RUN 支持
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY] ..."
        continue
    fi

    mkdir -p "${OUT_DIR}"
    _rc=0
    python -u downstream_train.py ... 2>&1 | tee "${LOG_FILE}" || _rc=$?

    if (( _rc != 0 )); then
        echo "[FAIL] ..."
    else
        echo "[RUN] ..."
    fi
done

# 5. 汇总统计
echo "Summary: RUN=${RUN} SKIP=${SKIP} FAIL=${FAIL}"
```

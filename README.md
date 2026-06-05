# FLaG — FFT Latent Attention Gate for Activity Cliff Prediction

*Built on [AMPCliff](https://github.com/Kewei2023/AMPCliff)*

**FLaG**（FFT Latent Attention Gate）是基于 [AMPCliff](https://github.com/Kewei2023/AMPCliff) 开发的独立研究项目，面向 antimicrobial peptide **activity cliff** 预测。AMPCliff 提供 activity cliff 数据集、benchmark 框架与 ESM2 下游训练基础设施；FLaG 在此基础上提出核心 pooling 方法 `fft_latent_attn_gate`，以及配套的机制分析与 ablation 实验。

代码托管于 AMPCliff 仓库的 **`FLaG` 分支**（历史原因保留 `AMPCliff` Python 包名与目录结构）。

---

## 目录

- [获取代码与环境](#获取代码与环境)
- [数据与模型权重](#数据与模型权重)
- [训练 FLaG 模型](#训练-flag-模型)
- [五个机制实验](#五个机制实验)
- [Ablation 实验脚本](#ablation-实验脚本)
- [关键配置](#关键配置)
- [脚本速查表](#脚本速查表)
- [引用与致谢](#引用与致谢)
- [Contact](#contact)

---

## 获取代码与环境

```bash
git clone git@github.com:Kewei2023/AMPCliff.git
cd AMPCliff
git checkout FLaG
```

> **注意**：本地目录名必须为 `AMPCliff`（Python 绝对 import 要求 `from AMPCliff.xxx import ...`）。

```bash
conda env create -f environment.yaml
conda activate AMPCliff
```

Clone 后请设置仓库根路径（批处理脚本默认指向超算路径）：

```bash
export REPO_ROOT=/path/to/AMPCliff
```

常用环境变量：

| 变量 | 说明 |
|------|------|
| `REPO_ROOT` | 项目根目录 |
| `OUTPUT_ROOT` | 训练/分析输出根目录（默认 `${REPO_ROOT}/outputs/ablation_new_data` 等） |
| `CONFIG_DIR` | ESM2 预训练权重目录（未设置时使用 `factory/initializer.py` 默认路径） |
| `DRY_RUN=1` | 只打印命令，不实际执行 |
| `SKIP_IF_DONE=1` | 检测到已有结果则跳过 |

---

## 数据与模型权重

- **数据**：放入 `./data/`。Activity cliff 划分生成见 [AMPCliff-generation](https://github.com/Kewei2023/AMPCliff-generation)。
- **默认 AC fix split 路径**：
  ```
  data/blosum62 average/diff_5-trd_0.9/grampa_{dataset}_7_25-{train,valid,test}.csv
  ```
  其中 `{dataset}` 为 `s_aureus` 或 `e_coli`。
- **ESM2 权重**（超算默认）：
  - `esm2_t6` → `/data/public/models/facebook/esm2_t6_8M_UR50D/`
  - `esm2_t12` → `/data/public/models/facebook/esm2_t12_35M_UR50D/`
- 本地路径不同时，训练时 override：
  ```bash
  python downstream_train.py model.config_dir="/your/path/to/esm2_t12_35M_UR50D/"
  ```

---

## 训练 FLaG 模型

### 核心方法

FLaG pooling 实现在 `factory/pooling/spectral_anchor_v2.py` 的 `FFTLatentAttentionGatePooling`：

**rFFT → latent attention → gate → iFFT → time pooling**

配置键：`model.regression.pooling=fft_latent_attn_gate`

### 单次训练

```bash
export REPO_ROOT=/path/to/AMPCliff
cd "${REPO_ROOT}"

POOLING=fft_latent_attn_gate MODEL_TYPE=esm2_t6 DATASET=s_aureus \
  bash evaluation_scripts/run_baseline_pooling_train.sh
```

- 训练入口：`downstream_train.py` + `configs/downstream.yaml`
- 评估入口：`downstream_evaluate.py`

### 多种子 2×2 网格（esm2_t6/t12 × e_coli/s_aureus）

分片并行脚本（可设 `POOLINGS` 做 baseline 对比）：

```bash
# 仅 FLaG
POOLINGS="fft_latent_attn_gate" bash evaluation_scripts/run_baseline_pooling_grid_2x2_seeds_1.sh

# 与 mean/max/attn 对比
POOLINGS="mean max attn fft_latent_attn_gate" bash evaluation_scripts/run_baseline_pooling_grid_2x2_seeds_1.sh
# seeds_2.sh ~ seeds_4.sh 为其余分片
```

### 汇总 seed 指标

```bash
bash evaluation_scripts/run_aggregate_baseline_pooling_seed_metrics.sh
# 或
python evaluation_scripts/aggregate_pooling_seed_metrics.py
```

---

## 五个机制实验

机制实验用于分析 FLaG 在 activity cliff 任务上的内部行为，需**先完成 FLaG 模型训练**（checkpoint 默认从 `outputs/ablation_new_data/{model}_fft_latent_attn_gate_{dataset}_diff5/seed_*/data/model.pth` 解析）。

| Exp | 目的 | 主 Python 脚本 | 批处理 / 辅助 |
|-----|------|----------------|---------------|
| **Exp1** Band knockout | 序列频带 notch 敏感性 | `downstream_evaluate_spectrual_filter.py` | `evaluation_scripts/run_fftlag_mechanism_experiments.sh`（`RUN_EXPS` 含 `1`） |
| **Exp2** Gate PSD | gate 前后频谱能量变化 | `downstream_evaluate_psd_gate.py` | 同上（`RUN_EXPS` 含 `2`） |
| **Exp3** Token knockout | token 扰动响应分布 | `downstream_evaluate_knockout.py` | `downstream_evaluate_knockout.sh`；聚合 `evaluation_scripts/aggregate_amp_knockout_seed_csvs.py`；绘图 `evaluation_scripts/plot_amp_knockout_figure.py` |
| **Exp4** Latent viz | latent query 频带质量分布 | `downstream_evaluate_fft_lag_latent.py` | 同上（`RUN_EXPS=4`）或 `evaluation_scripts/run_fftlag_exp4_only.sh`；attn score 变体 `evaluation_scripts/run_fftlag_exp4_attn_score_raw.sh` |
| **Exp5** Structure | 按结构分桶的二阶分析 | `analyze_fft_lag_mechanism_by_structure.py` | `evaluation_scripts/run_fftlag_mechanism_experiments.sh` 末尾、`RUN_EXP5=1` 时自动调用 |

### 重要说明

- `evaluation_scripts/run_fftlag_mechanism_experiments.sh` **默认跑 Exp1/2/4**（`RUN_EXPS=1,2,4`），**不包含 Exp3**；Exp3 需单独跑 knockout 流程。
- 共用 peptide 子集由 `evaluation_scripts/select_knockout_peptide_subset.py` 生成 manifest。
- 配置文件：
  - Exp1/4：`configs/evaluate_fftlag_mechanism.yaml`
  - Exp2：`configs/evaluate_psd_gate.yaml`
  - Exp3：`configs/downstream_knockout.yaml`
- 跨 seed 聚合/绘图：
  - `evaluation_scripts/aggregate_fftlag_mechanism_seeds.py`
  - `evaluation_scripts/plot_fftlag_mechanism_per_sample_seeds.py`

### 一键示例

```bash
export REPO_ROOT=/path/to/AMPCliff
cd "${REPO_ROOT}"

# Exp1 + Exp2 + Exp4 + Exp5（默认）
bash evaluation_scripts/run_fftlag_mechanism_experiments.sh

# 仅 Exp4
RUN_EXPS=4 bash evaluation_scripts/run_fftlag_mechanism_experiments.sh

# Exp3（需先有 checkpoint，Hydra override 见 configs/downstream_knockout.yaml）
bash downstream_evaluate_knockout.sh
```

---

## Ablation 实验脚本

### A. Pooling baseline 对比（mean / max / attn / FLaG）

| 脚本 | 作用 |
|------|------|
| `evaluation_scripts/run_baseline_pooling_train.sh` | 单次训练，设 `POOLING=` |
| `evaluation_scripts/run_baseline_pooling_grid_2x2_seeds_{1-4}.sh` | 2×2 × 多 seed 批量训练 |
| `evaluation_scripts/run_pooling_main.sh` | 调用 `evaluation_scripts/run_pooling_baseline_ablation.sh` 的快捷入口 |
| `evaluation_scripts/run_pooling_baseline_ablation.sh` | SpectralAnchor 相关 ablation（num_anchor / use_fft sweep） |

### B. 结果汇总与统计

| 脚本 | 作用 |
|------|------|
| `extract_ablation_metrics.py` | 从 `outputs/ablation/` 抽取 pearson / spearman / recall |
| `aggregate_ablation_results.py` | 按 config × seed 聚合 mean±std |
| `analyze_ablation_results.py` | 分析汇总 CSV |
| `collect_ablation_summary.py` | 收集 ablation 摘要 |
| `split_ablation_metrics_to_excel.py` | 导出 Excel |
| `evaluation_scripts/compute_pooling_ablation_stats_new_data.py` | pooling ablation 统计 |
| `evaluation_scripts/ablation_new_data_stats_to_xlsx.py` | 导出 xlsx |
| `evaluation_scripts/merge_seed_metrics_pooling_csv_to_xlsx.py` | 合并 seed metrics |
| `run_ablation_protein.sh` | 多 pooling 组件 ablation 批量训练（见下方注意） |

> **注意**：`run_ablation_protein.sh` 内含 `fft_latent_only` 等组件 pooling 名，但当前 FLaG 分支 `factory/pooling/registry.py` 仅注册标准 pooling（含 `fft_latent_attn_gate`）。直接运行该脚本中未注册的 pooling 名会报错；如需组件 ablation，须在 registry 中恢复对应实现。

---

## 关键配置

`configs/downstream.yaml` 中 FLaG 相关关键项：

```yaml
features.type: LLM
model.regression.version: esm2_t6   # 或 esm2_t12
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

**加载 checkpoint**（评估 / 机制实验）：

```yaml
model.regression.check_point.load: true
model.regression.check_point.path: /path/to/seed_0/data/model.pth
```

或通过 Hydra CLI：

```bash
python downstream_evaluate.py \
  model.regression.check_point.load=true \
  model.regression.check_point.path="/path/to/model.pth"
```

**Debug 模式**：`other.debug=true`（缩小数据集用于快速验证）。

---

## 脚本速查表

| 脚本路径 | 类型 | 说明 |
|----------|------|------|
| `downstream_train.py` | Core | Hydra 训练入口 |
| `downstream_evaluate.py` | Core | 标准评估入口 |
| `downstream_evaluate_spectrual_filter.py` | Mechanism | Exp1：频带 knockout |
| `downstream_evaluate_psd_gate.py` | Mechanism | Exp2：gate 前后 PSD |
| `downstream_evaluate_knockout.py` | Mechanism | Exp3：token knockout |
| `downstream_evaluate_fft_lag_latent.py` | Mechanism | Exp4：latent 可视化 |
| `analyze_fft_lag_mechanism_by_structure.py` | Mechanism | Exp5：结构分桶分析 |
| `evaluation_scripts/run_fftlag_mechanism_experiments.sh` | Mechanism | Exp1/2/4/5 批处理调度 |
| `evaluation_scripts/run_fftlag_exp4_only.sh` | Mechanism | 仅 Exp4（SLURM 友好） |
| `evaluation_scripts/run_fftlag_exp4_attn_score_raw.sh` | Mechanism | Exp4 attn score 变体 |
| `evaluation_scripts/select_knockout_peptide_subset.py` | Mechanism | 生成共用 peptide manifest |
| `evaluation_scripts/aggregate_fftlag_mechanism_seeds.py` | Mechanism | 跨 seed 聚合机制指标 |
| `evaluation_scripts/plot_fftlag_mechanism_per_sample_seeds.py` | Mechanism | 机制实验 per-sample 绘图 |
| `evaluation_scripts/aggregate_amp_knockout_seed_csvs.py` | Mechanism | Exp3 跨 seed 聚合 |
| `evaluation_scripts/plot_amp_knockout_figure.py` | Mechanism | Exp3 结果绘图 |
| `downstream_evaluate_knockout.sh` | Mechanism | Exp3 SLURM/批处理入口 |
| `evaluation_scripts/run_baseline_pooling_train.sh` | Training | 单次 pooling 训练 |
| `evaluation_scripts/run_baseline_pooling_grid_2x2_seeds_{1-4}.sh` | Training | 2×2 网格多种子训练 |
| `evaluation_scripts/run_aggregate_baseline_pooling_seed_metrics.sh` | Training | 汇总 baseline seed 指标 |
| `evaluation_scripts/aggregate_pooling_seed_metrics.py` | Training | seed 指标聚合 Python 脚本 |
| `evaluation_scripts/run_pooling_main.sh` | Ablation | pooling ablation 快捷入口 |
| `evaluation_scripts/run_pooling_baseline_ablation.sh` | Ablation | SpectralAnchor 超参 sweep |
| `run_ablation_protein.sh` | Ablation | 组件 pooling ablation（见 registry 限制） |
| `extract_ablation_metrics.py` | Analytics | 抽取 ablation 指标 |
| `aggregate_ablation_results.py` | Analytics | 聚合 ablation 结果 |
| `analyze_ablation_results.py` | Analytics | 分析 ablation CSV |
| `collect_ablation_summary.py` | Analytics | 收集 ablation 摘要 |
| `split_ablation_metrics_to_excel.py` | Analytics | 导出 Excel |
| `evaluation_scripts/compute_pooling_ablation_stats_new_data.py` | Analytics | 新数据 pooling 统计 |
| `evaluation_scripts/ablation_new_data_stats_to_xlsx.py` | Analytics | 统计结果导出 xlsx |
| `evaluation_scripts/merge_seed_metrics_pooling_csv_to_xlsx.py` | Analytics | 合并 seed metrics 到 xlsx |
| `downstream_train.sh` | Shell | SLURM 训练 wrapper |
| `downstream_evaluate.sh` | Shell | SLURM 评估 wrapper |

---

## 引用与致谢

### FLaG

若使用 FLaG pooling 方法或本仓库代码，请引用我们的论文（待更新 bibtex）。

### AMPCliff（基础平台）

FLaG is built upon AMPCliff. 若使用 activity cliff 数据集或 benchmark 框架，请同时引用：

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

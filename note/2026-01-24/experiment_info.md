# AMPCliff Pooling 实验工作流说明

> 返回 [[../2026-01-23/summary|上一日实验总结]]

---

## 1. experiments.csv 使用方法

`experiments.csv` 是实验配置的**核心文件**，定义了：

### 前四列：实验组合（用于训练）
- `datasets`: 数据集列表（e_coli, s_aureus）
- `models`: 模型列表（esm2_t6, esm2_t12）
- `pooling`: 池化方式（max, mean, attn）
- `strategies`: 策略列表（none, filter_dc, scale_dc, concat_dc, distill_vc）

这四列构成**笛卡尔积**，生成所有可能的训练组合。

### 后两列：分析任务（用于训练后评估）
- `experiments`: 分析类型名称（如 attention_knockout, spectrual_filter, interprete_features）
- `files`: 对应的评估脚本文件名

### 工作原理
1. **生成训练任务**：扫描前四列的所有组合
2. **跳过已完成的实验**：检查 `paths.csv` 中是否已有对应模型路径
3. **执行训练**：对未完成的组合执行训练脚本
4. **执行分析**：训练完成后，根据后两列执行对应的分析脚本

---

## 2. Project's Experiment Workflow

### 阶段 A：�
1. 定义 `experiments.csv` 的实验矩阵
2. 创建 `paths.csv` 用于记录已完成的模型路径

### 阶段 B：训练
1. 根据 `experiments.csv` 的前四列生成训练任务
2. 对每个训练任务：
   - 检查 `paths.csv` 是否已有匹配项
   - 如果有，跳过训练
   - 如果没有，执行训练
3. 训练完成后，将模型路径添加到 `paths.csv`

### 阶段 C：分析
1. 根据 `experiments.csv` 的后两列执行分析任务
2. 从 `paths.csv` 查找对应模型路径
3. 如果找不到，使用默认路径或报错
4. 生成分析报告

### 阶段 D：迭代
1. 根据分析结果调整 `experiments.csv`
2. 重新运行阶段 B 和 C
3. 循环直到所有实验完成

---

## 3. 关键文件说明

| 文件 | 作用 |
|------|------|
| `experiments.csv` | 定义实验矩阵和分析任务 |
| `paths.csv` | 记录已完成模型的路径，避免重复训练 |
| `downstream_evaluate_analysis_local.sh` | 根据上述逻辑生成的分析脚本 |

---

## 4. 当前状态

- **训练脚本**：`downstream_train_rerun.sh` (72 个实验) 已完成
- **输出目录**：
  - `/mnt/g/likw/AMPCliff/AMPCliff/outputs/2026-01-23/` (44 个实验)
  - `/mnt/g/likw/AMPCliff/AMPCliff/outputs/2026-01-24/` (18 个实验)
  - `/mnt/g/likw/AMPCliff/AMPCliff/outputs/scale_ablation/` (12 个实验)

### 待办事项

- [ ] 扫描所有实验目录，更新 `paths.csv`
- [ ] 更新 `downstream_evaluate_analysis_local.sh`
- [ ] 运行 3 个验证实验

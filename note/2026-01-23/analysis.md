> 返回 [[summary]] 查看今日概要

# 实验深度分析与审计报告 (2026-01-23)

## 实验矩阵

本批次实验采用全正交设计，旨在通过多维度对比验证频域特征提取策略的有效性。

| 维度 | 配置内容 | 备注 |
|------|----------|------|
| **数据集 (2)** | e_coli, s_aureus | 覆盖革兰氏阴性/阳性菌 |
| **模型 (2)** | esm2_t6 (8M), esm2_t12 (35M) | 验证模型规模影响 |
| **策略 (5)** | none, filter_dc, scale_dc, concat_dc, distill_vc | 核心实验变量 |
| **池化 (3)** | max, mean, attn | 结构稳定性验证 |

**总计**: 2 * 2 * 5 * 3 = 60 个基础组合。每个组合包含 AC Split 重复实验。

---

## 审计结果表

| 实验 ID | 模型 | 策略 | 池化 | Train Spearman | 状态 | 备注 |
|---------|------|------|------|----------------|------|------|
| 00-02-38 | esm2_t6 | filter_dc | max | 0.984 | ✅ GOOD | 紧密对角线 |
| 03-14-29 | esm2_t6 | filter_dc | attn | 0.980 | ✅ GOOD | 紧密对角线 |
| 00-40-03 | esm2_t6 | distill_vc | max | 0.992 | ✅ GOOD | 紧密对角线 |
| 01-32-10 | esm2_t6 | filter_dc | mean | 0.056 | ❌ COLLAPSED | 完全水平分布 |
| 01-43-46 | esm2_t6 | concat_dc | mean | 0.719 | ⚠️ PLATEAU | 高MIC区域预测饱和 |
| 01-56-31 | esm2_t6 | scale_dc | mean | 0.922 | ⚠️ FAIL | 未达0.95标准 |
| 02-10-22 | esm2_t6 | distill_vc | mean | 0.922 | ⚠️ FAIL | 未达0.95标准 |
| 08-48-50 | esm2_t12 | filter_dc | mean | 0.060 | ❌ COLLAPSED | 完全水平分布 |
| 09-27-09 | esm2_t12 | scale_dc | mean | 0.806 | ⚠️ PLATEAU | 高MIC区域预测饱和 |
| 09-48-07 | esm2_t12 | distill_vc | mean | 1.000 | ✅ GOOD | 完美对角线 |

## 关键发现

### 结论性发现（重要）
**`filter_dc + mean + opposite=false` 在 esm2_t6 和 esm2_t12 上都失败（崩溃）**
- esm2_t6: Spearman = 0.056
- esm2_t12: Spearman = 0.060
- 表现：预测值几乎为常数，完全水平分布
- **后续结果分析时需着重分析此现象的原因**

### 池化稳定性差异
| 池化方式 | 稳定性 | Train Spearman 范围 |
|---------|--------|-------------------|
| Max | 极佳 | > 0.98 |
| Attn | 极佳 | > 0.98 |
| Mean | **极不稳定** | 0.05 ~ 1.0（方差大） |

---

## 数学分析

### scale_dc + mean pooling 深度分析

**结论：scale_dc + mean pooling 等价于对 mean 结果做 scale**

数学推导：
1. DCT 变换（ortho 模式）：C₀ = √T · mean(x)
2. Scale 操作：C'₀ = scale · C₀
3. IDCT 逆变换后：mean(x') = scale · mean(x)

**等价公式**：
```
mean(scale_dc(x, s)) = s · mean(x)
```

**失败原因分析**：
1. **梯度消失**：特征被压缩 100 倍（scale=0.01），梯度也相应缩小。
2. **数值不匹配**：回归头需要学习极大权重来补偿，训练不稳定。
3. **高原效应**：模型陷入局部最优，高 MIC 区域预测饱和。

### DC/VC 拼接策略 + Mean Pooling 深度分析

**数学原理**：
- VC 分量 = 去除 0 频率后的信号。
- 理论上 VC 的全局均值 = 0（因为直流分量被移除）。
- 但由于 `masked_mean_pooling` 只在有效长度内计算，VC 均值 ≠ 0。

**实际问题**：
```
concat_dc/distill_vc 输出: [VC, DC]  维度 2d
                            ↓
mean pooling 后:           [P_vc, P_dc]
                            ↓
P_vc ≈ mean(x) × (1 - N/L)  ← 与 DC 是冗余信息！
P_dc ≈ mean(x) × (N/L)
```

**结论**：在 Mean Pooling 下，VC 和 DC 部分提供的是**冗余信息**（都是均值的不同缩放版本），VC 丢失了其代表的时间序列波动特征。所有 DC/VC 拼接策略 + Mean Pooling 的组合都存在结构性问题。

---

## 操作规程与后续计划

### 更新 paths.csv 规程
1. **新路径格式**：`outputs/2026-01-23/{HH-MM-SS}_{dataset}_{model}_{strategy}_{pooling}/`
2. **脚本思路**：遍历 `outputs/2026-01-23/*/`，检查 `train*.png`，从日志提取 best model 路径，解析目录名获取元数据，追加到 `paths.csv`。

### 重记与消融实验规划
1. **Part 1**: 未完成实验重跑 (Exp 50, 51)，数量：2。
2. **Part 2**: e_coli scale_dc 消融实验 (scale=0.5, 1.0, 2.0)，数量：6。
3. **Part 3**: e_coli 剩余未开始实验 (Exp 52-60)，数量：9。
4. **Part 4**: s_aureus scale_dc 消融实验 (scale=0.5, 1.0, 2.0)，数量：6。
5. **Part 5**: s_aureus 主线实验 (Exp 61-120)，数量：49。

---

## 实验重跑清单

### 必须重跑（未完成）
| 实验 ID | 数据集 | 模型 | 策略 | 池化 | 原因 |
|---------|--------|------|------|------|------|
| 00-00-19 | e_coli | esm2_t12 | filter_dc | attn | epoch 5 中断 |
| 11-29-10 | e_coli | esm2_t12 | none | mean | epoch 25 中断 |

### 可选重跑（scale 参数消融）
| 数据集 | 模型 | 策略 | 池化 | scale 值 |
|--------|------|------|------|---------|
| e_coli | esm2_t6 | scale_dc | mean | 0.5, 1.0, 2.0 |
| e_coli | esm2_t12 | scale_dc | mean | 0.5, 1.0, 2.0 |

# 2026-01-23 Pooling 实验日志

> 本文件是实验记录的**目录索引**，按实验流程组织，点击链接跳转详情。

---

## 📋 目录

1. [[#实验背景]]
2. [[#实验设计]]
3. [[#执行状态]]
4. [[#审计结果]]
5. [[#关键发现]]
6. [[#后续计划]]

---

## 实验背景

**项目**: AMPCliff Pooling 策略对比研究
**目标**: 评估不同池化方式（max, mean, attn）与频域策略（filter_dc, scale_dc, concat_dc, distill_vc）的组合效果

---

## 实验设计

| 维度 | 选项 |
|------|------|
| 数据集 | e_coli, s_aureus |
| 模型 | esm2_t6, esm2_t12 |
| 池化 | max, mean, attn |
| 策略 | none, filter_dc, scale_dc, concat_dc, distill_vc |

**实验总数**: 120 个（60 e_coli + 60 s_aureus）

详细配置见 [[analysis#实验矩阵]]

---

## 执行状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| e_coli 主线 | ✅ 44/60 完成 | 16 个待重跑 |
| s_aureus 主线 | ⏳ 0/60 | 待启动 |
| scale_dc 消融 | ⏳ 0/12 | 待启动 |

**当前运行**: `downstream_train_rerun.sh` (72 个实验，预计 14.4h)

执行命令见 [[commands]]

---

## 审计结果

| 状态 | 数量 | 
|------|------|
| ✅ 成功 (Spearman > 0.95) | 38 |
| ⚠️ 欠拟合 (0.5 < Spearman < 0.95) | 4 |
| ❌ 崩溃 (Spearman < 0.5) | 2 |

详细审计表见 [[analysis#审计结果表]]

---

## 关键发现

### 结论：Mean Pooling 与频域策略不兼容

| 策略 + Mean | 现象 | 原因 |
|-------------|------|------|
| filter_dc | ❌ 崩溃 (Spearman ~0.06) | 数学互斥 |
| scale_dc | ⚠️ 梯度消失 | DC 压缩 100 倍 |
| concat_dc | ⚠️ 高原效应 | VC 信息丢失 |
| distill_vc | ⚠️ 欠拟合 | 同 concat_dc |

**推荐**: Max 或 Attn Pooling（Spearman > 0.98）

数学证明见 [[analysis#数学分析]]

---

## 后续计划

- [x] 完成 e_coli 审计
- [x] 生成重跑脚本 (72 实验)
- [ ] 运行重跑脚本 ← **进行中**
- [ ] 审计 scale_dc 消融结果
- [ ] 启动 s_aureus 实验
- [ ] 汇总最终报告

---

## 文件索引

| 文件 | 内容 |
|------|------|
| [[analysis]] | 审计详情、数学证明、操作规程 |
| [[commands]] | 原始命令记录 |

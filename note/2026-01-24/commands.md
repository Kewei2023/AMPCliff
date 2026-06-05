# 命令记录

> 返回 [[summary]]

---

## [17:48] 读取原始路径文件

```bash
# 读取 paths.csv 并保存为时间戳版本
cp .opencode/skill/project-pipeline/assets/paths.csv .opencode/skill/project-pipeline/assets/paths-20260124-17.48.csv
```

**输出摘要**: 创建原始路径文件的备份，共 133 行记录

---

## [18:51] 路径存在性检查与审计

```bash
# 审计 CSV 文件中的路径是否存在
# 检查 paths-20260124-17.48.csv 中的所有模型路径
# 发现 21 个路径不存在，主要是 2025 年数据缺少配置目录前缀
```

**输出摘要**:
- 总计检查 132 个路径（1 行标题 + 131 行数据）
- 发现 21 个路径不存在
- 问题原因：路径缺少 `_unknown_esm2_t6_*` 配置目录前缀
- 错误示例：`outputs/2025-11-11/20-29-08/esm2_t6/...` 应为 `outputs/2025-11-11/20-29-08_unknown_esm2_t6_distill_vc_max/esm2_t6/...`

---

## [19:06] 生成路径映射表

```bash
# 分析缺失路径与实际存在路径的对应关系
# 生成路径映射表 paths_mapping.txt
# 对每个缺失路径进行模糊匹配，查找实际存在的路径
```

**输出摘要**: 生成 paths_mapping.txt，包含：
- 19 个完全匹配的路径映射（添加前缀）
- 1 个相近匹配（Line 121：时间戳不同但配置相同）
- 1 个未找到（Line 120：完全无法匹配）

**关键匹配示例**:
```
Line 113: 20-29-08_esm2_t6_distill_vc_max
      → 20-29-08_unknown_esm2_t6_distill_vc_max ✓
Line 119: 12-03-58_esm2_t6_filter_dc_attn
      → 12-03-58_unknown_esm2_t6_filter_dc_attn ✓
Line 121: 23-04-19_esm2_t6_scale_dc_max
      → 22-37-23_unknown_esm2_t6_scale_dc_max ≈ (时间不同)
```

---

## [19:11] 应用路径修复并更新文件

```bash
# 根据路径映射表更新 CSV 文件
# 为 19 个完全匹配的路径添加前缀配置目录
# 删除 2 个无法匹配的路径（Lines 120, 121）
# 生成修复后的文件 paths-20260124-19.11.csv
```

**输出摘要**:
- 修复前：133 行（包含 21 个无效路径）
- 修复后：131 行（删除 2 个，修复 19 个）
- 删除行号：120, 121（s_aureus, esm2_t6, attn/max/filter_dc 相关路径）
- 修复方式：在时间戳后插入配置目录前缀（如 `_unknown_esm2_t6_distill_vc_max`）

**修复示例**:
```diff
- outputs/2025-11-11/20-29-08/esm2_t6/blosum62 average/diff5-trd0.9/model_step_21_spearman_0.735
+ outputs/2025-11-11/20-29-08_unknown_esm2_t6_distill_vc_max/esm2_t6/blosum62 average/diff5-trd0.9/model_step_21_spearman_0.735
```

---

## [19:21] 路径文件排序

```bash
# 按字母顺序对路径文件进行排序
# 排序优先级：datasets → models → pooling → strategies
# 生成排序后的文件 paths-sorted-20260124-19.21.csv
```

**输出摘要**:
- 输入文件：paths-20260124-19.11.csv (131 行)
- 输出文件：paths-sorted-20260124-19.21.csv (131 行)
- 排序规则：CSV 多列排序（第 1-4 列升序）

**排序后首行**:
```
e_coli,esm2_t6,max,none,/mnt/g/likw/AMPCliff/AMPCliff/outputs/2025-11-01/10-01-22_unknown_esm2_t6_none_max/esm2_t6/blosum62 average/diff5-trd0.9/model_step_25_spearman_0.756
```

---

## [16:29] 实验配置更新（experiments.csv）

```bash
# 编辑 experiments.csv
# 在 scale_dc 行添加了 4 个新的 scale 参数配置
# 配置内容：
#   - scale_dc, scale=0.01
#   - scale_dc, scale=0.5
#   - scale_dc, scale=1.0
#   - scale_dc, scale=2.0
```

**输出摘要**: experiments.csv 从 14 行扩展到包含 scale 参数消融实验配置

**新增配置**:
```csv
,,"scale_dc, scale=0.01"
,,"scale_dc, scale=0.5"
,,"scale_dc, scale=1.0"
,,"scale_dc, scale=2.0"
```

---

## 文件清单

| 文件 | 大小 | 时间戳 | 说明 |
|------|------|--------|------|
| paths-20260124-17.48.csv | 22KB | 17:48 | 原始路径文件备份 |
| paths-20260124-18.51.csv | 25KB | 18:51 | 中间处理版本 |
| paths_mapping.txt | 4.9KB | 19:06 | 路径映射表 |
| paths-20260124-19.11.csv | 25KB | 19:11 | 修复后的路径文件 |
| paths-sorted-20260124-19.21.csv | 25KB | 19:21 | 排序后的路径文件 |
| experiments.csv | 571B | 16:29 | 更新后的实验配置 |

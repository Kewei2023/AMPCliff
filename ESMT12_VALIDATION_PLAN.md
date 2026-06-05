# ESM2-T12 验证脚本执行方案

## ✅ 已完成

1. **生成了3个新的 esm2_t12 验证脚本**:
   - `run_evaluate_knockout_esm2_t12.sh` (66个实验)
   - `run_evaluate_spectrual_filter_esm2_t12.sh` (12个实验)
   - `run_evaluate_interpret_features_esm2_t12.sh` (66个实验)

2. **修复了参数错误**:
   - scale_dc 参数: `distill_vc.scale` → `scale_dc.scale`
   - MLFlow URI: `file:////` → `file:///`
   - 输出目录: 添加 `_esm2_t12` 后缀

3. **验证了脚本正确性**:
   - ✅ 模型路径正确: `esm2_t12_35M_UR50D/`
   - ✅ Checkpoint 加载成功
   - ✅ 评估正常运行

---

## 🚀 执行方案

### 方案A: 快速测试（推荐先执行）

```bash
# 设置环境变量
export MLFLOW_EXPERIMENT_NAME=breeze
export MLFLOW_TRACKING_URI="file:///mnt/g/likw/AMPCliff/AMPCliff/mlruns"

# 测试 knockout
bash -c "$(sed -n '28,42p' run_evaluate_knockout_esm2_t12.sh)" 2>&1 | tee /tmp/test_knockout.log
grep "Loading ESM2 model" /tmp/test_knockout.log

# 测试 spectrual_filter
bash -c "$(sed -n '28,42p' run_evaluate_spectrual_filter_esm2_t12.sh)" 2>&1 | tee /tmp/test_spectrual.log
grep "Loading ESM2 model" /tmp/test_spectrual.log

# 测试 interpret_features
bash -c "$(sed -n '28,42p' run_evaluate_interpret_features_esm2_t12.sh)" 2>&1 | tee /tmp/test_interpret.log
grep "Loading ESM2 model" /tmp/test_interpret.log
```

**预期输出**: `Loading ESM2 model: version=esm2_t12, path=.../esm2_t12_35M_UR50D/`

---

### 方案B: 完整串行运行（测试通过后）

```bash
# 依次运行三个脚本
bash run_evaluate_knockout_esm2_t12.sh > /tmp/knockout_esm2_t12.log 2>&1 && \
  echo "✓ Knockout complete (66/66)" && \
  bash run_evaluate_spectrual_filter_esm2_t12.sh > /tmp/spectrual_filter_esm2_t12.log 2>&1 && \
  echo "✓ Spectrual_filter complete (12/12)" && \
  bash run_evaluate_interpret_features_esm2_t12.sh > /tmp/interpret_features_esm2_t12.log 2>&1 && \
  echo "✓ Interpret_features complete (66/66)" && \
  echo "✅ All 144 experiments complete!"
```

**运行时间**: 约 5-11小时

---

### 方案C: 并行运行（需要足够GPU内存）

```bash
# 同时运行三个脚本
bash run_evaluate_knockout_esm2_t12.sh > /tmp/knockout_esm2_t12.log 2>&1 &
bash run_evaluate_spectrual_filter_esm2_t12.sh > /tmp/spectrual_filter_esm2_t12.log 2>&1 &
bash run_evaluate_interpret_features_esm2_t12.sh > /tmp/interpret_features_esm2_t12.log 2>&1 &

echo "All scripts started. Monitor: tail -f /tmp/*_esm2_t12.log"
```

---

## 📊 监控命令

```bash
# 查看进度
watch -n 10 'echo "Knockout: $(grep -c "✓ Completed" /tmp/knockout_esm2_t12.log)/66"; echo "Spectrual: $(grep -c "✓ Completed" /tmp/spectrual_filter_esm2_t12.log)/12"; echo "Interpret: $(grep -c "✓ Completed" /tmp/interpret_features_esm2_t12.log)/66"'

# 查看实时日志
tail -f /tmp/knockout_esm2_t12.log

# 检查错误
grep -i "error\|traceback" /tmp/*_esm2_t12.log
```

---

## 📁 生成的文件

| 文件 | 大小 | 实验数 | 状态 |
|------|------|--------|------|
| run_evaluate_knockout_esm2_t12.sh | 54K | 66 | ✅ 就绪 |
| run_evaluate_spectrual_filter_esm2_t12.sh | 11K | 12 | ✅ 就绪 |
| run_evaluate_interpret_features_esm2_t12.sh | 55K | 66 | ✅ 就绪 |
| **总计** | **120K** | **144** | ✅ 就绪 |

---

## 🎯 推荐执行顺序

1. **先执行方案A**（快速测试，约10分钟）
2. **确认测试通过后，执行方案B**（完整运行，约5-11小时）
3. **监控进度和错误**
4. **验证结果**

---

**准备就绪！请选择执行方案并运行。**

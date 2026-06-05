# Re-training Scripts for esm2_t12 Experiments

**Date**: 2026-01-25
**Purpose**: Re-train all 119 mismatched esm2_t12 experiments with correct model (480 hidden_dim)

---

## Problem Summary

**Root Cause**: Checkpoints trained before config_dir mapping fix (commit 9bf9bfd on 2026-01-25) used wrong model architecture
- Claimed: esm2_t12 (480 hidden_dim)
- Actually used: esm2_t6 (320 hidden_dim)

**Impact**: All evaluation and analysis results from these checkpoints are invalid

---

## Deletion Summary

✅ **Deleted**: 100 directories (all mismatched checkpoints removed)
- `outputs/scale_ablation/`: 16 directories (12 e_coli + 4 e_coli nested)
- `outputs/analysis/spectrual_filter/`: 15 directories (e_coli + s_aureus)
- `outputs/analysis/interprete_features/`: 68 directories (e_coli + s_aureus)
- `outputs/test/`: 1 directory (test reference)

---

## Re-training Scripts

### Overview

Three scripts created to re-train all 107 experiments (excluding test directory):
1. `re_train_esm2_t12_experiments.sh` - Part 1 (24 experiments)
2. `re_train_esm2_t12_part2.sh` - Part 2 (15 experiments)
3. `re_train_esm2_t12_part3.sh` - Part 3 (68 experiments)
4. `re_train_all.sh` - Master script to run all parts

### Master Script

Use the master script to re-train all experiments:

```bash
bash re_train_all.sh
```

This will execute all three parts in sequence, training a total of 107 experiments.

---

## Script Details

### Part 1: Scale Ablation (24 experiments)

**Dataset**: e_coli, s_aureus
**Experiments**: 12 each (total 24)
- `scale=0.01` (3 experiments per dataset): pooling=attn, max, mean
- `scale=0.5` (3 experiments per dataset): pooling=attn, max, mean
- `scale=1.0` (3 experiments per dataset): pooling=attn, max, mean
- `scale=2.0` (3 experiments per dataset): pooling=attn, max, mean

**Output**: `outputs/scale_ablation/`

### Part 2: Spectrual Filter (15 experiments)

**Dataset**: e_coli, s_aureus
**Experiments**: e_coli=9, s_aureus=6 (total 15)
- `pooling=max, mean, attn`
- `apply`: none, concat_dc, scale_dc, filter_dc (opposite=false/true)

**Output**: `outputs/analysis/spectrual_filter/`

### Part 3: Interpret Features (68 experiments)

**Dataset**: e_coli, s_aureus
**Experiments**: e_coli=34, s_aureus=34 (total 68)
- `pooling=max, mean, attn`
- `apply`: none, concat_dc, scale_dc, filter_dc
- `apply`: distill_vc (scale=1.5, 2.0, 2.5, 3.0, 3.5)

**Output**: `outputs/analysis/interprete_features/`

---

## Configuration Verification

All experiments use:
- `model.regression.version=esm2_t12` ✅ (now correctly maps to esm2_t12_35M_UR50D/)
- `mode.ddp=false`
- `data.regression.mode=fix`
- `logger.log=true`

After training, verify checkpoints use correct model:
```bash
# Check a few new checkpoints
grep "config_dir:" outputs/<new_dir>/.hydra/config.yaml
# Should show: /mnt/.../esm2_t12_35M_UR50D/
```

---

## Execution Time Estimate

**Per experiment**: ~10-15 minutes (depending on dataset size and convergence)
**Part 1 total**: ~4-6 hours
**Part 2 total**: ~2.5-3.75 hours
**Part 3 total**: ~11-17 hours
**Grand Total**: ~17.5-26.75 hours

---

## Important Notes

⚠️ **GPU Resources**: Training 107 experiments requires significant GPU time. Run sequentially or monitor GPU usage.

✅ **No Code Changes Needed**: The existing config_dir mapping fix (commit 9bf9bfd) is working correctly.

📊 **Results Monitoring**: Track training progress with MLFlow if enabled.

🎯 **Next Steps**: After re-training complete, all evaluation scripts will have valid checkpoints and run correctly.

---

## Scripts Created

1. `re_train_all.sh` - Master script
2. `re_train_esm2_t12_experiments.sh` - Scale ablation (partially complete)
3. `re_train_esm2_t12_part2.sh` - Spectrual filter
4. `re_train_esm2_t12_part3.sh` - Interpret features

**Total experiments to re-train**: 107

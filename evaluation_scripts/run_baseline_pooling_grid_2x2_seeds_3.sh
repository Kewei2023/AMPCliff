#!/bin/bash
# =============================================================================
# run_baseline_pooling_grid_2x2_seeds.sh
#
# 对 POOLINGS（默认 mean max attn）各跑一遍 2×2 网格（esm2_t6/t12 × s_aureus/e_coli），
# 每格多个随机种子；子脚本为 evaluation_scripts/run_baseline_pooling_train.sh。
#
# 汇总各 seed 的 test/valid Spearman、Pearson（均值、样本方差、样本标准差），对每个
# (POOLING, MODEL_TYPE, DATASET) 实验目录执行：
#   python evaluation_scripts/aggregate_se_pooling_seed_metrics.py \
#     --exp-dir "${OUTPUT_ROOT}/${MODEL_TYPE}_${POOLING}_${DATASET}_diff${DIFF}"
#
# 环境变量：
#   POOLINGS             默认 "mean max attn"（空格分隔）
#   SEEDS                默认 "0 1 2 3 4 5 6 7 8 9"
#   REPO_ROOT            默认 /data/home/scv6872/run/kwli/AMPCliff（可 export 覆盖）
#   OUTPUT_ROOT          默认 ${REPO_ROOT}/outputs/ablation（与 run_local_stft_grid_2x2_seeds.sh 一致）
#   CONDITION, DIFF, GRID_USE_REPO_CSV
#   DRY_RUN, SKIP_IF_DONE, NUM_EPOCH 等 — 传给子脚本
#
# 用法：
#   bash evaluation_scripts/run_baseline_pooling_grid_2x2_seeds.sh
#   DRY_RUN=1 bash evaluation_scripts/run_baseline_pooling_grid_2x2_seeds.sh
#
# 每一格可整段注释掉以跳过（与 run_se_pooling_grid_2x2_seeds.sh 相同）。
# =============================================================================
set -uo pipefail

module load miniforge gcc/11.1.0 cuda/11.1 gcc/11.1.0
module load cudnn/8.1.0.77_CUDA11.1
source activate AMPCliff

REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"

CHILD_SH="${REPO_ROOT}/evaluation_scripts/run_baseline_pooling_train.sh"

CONDITION="${CONDITION:-blosum62 average}"
DIFF="${DIFF:-5}"
GRID_USE_REPO_CSV="${GRID_USE_REPO_CSV:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/ablation-new-data}"
export OUTPUT_ROOT

RUN_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0
declare -a GRID_RESULT_LINES=()

# shellcheck disable=SC2086
# SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SEEDS="${SEEDS:-5 6 7 8 9}"
# shellcheck disable=SC2086
# POOLINGS="${POOLINGS:-mean max attn}"
POOLINGS="${POOLINGS:-attn last}"

_default_csv_for_dataset() {
  local ds="$1"
  local base="${REPO_ROOT}/data/${CONDITION}/diff_${DIFF}-trd_0.9"
  export TRAIN_FILE="${base}/grampa_${ds}_7_25-train.csv"
  export VALID_FILE="${base}/grampa_${ds}_7_25-valid.csv"
  export TEST_FILE="${base}/grampa_${ds}_7_25-test.csv"
}

_run_cell_seeds() {
  export CONDITION
  export DIFF
  export OUTPUT_ROOT
  export POOLING
  if [[ "${GRID_USE_REPO_CSV}" == "1" ]]; then
    _default_csv_for_dataset "${DATASET}"
  else
    unset TRAIN_FILE VALID_FILE TEST_FILE 2>/dev/null || true
  fi

  # shellcheck disable=SC2086
  for RANDOM_SEED in ${SEEDS}; do
    export RANDOM_SEED
    echo ""
    echo ">>> Run: POOLING=${POOLING} MODEL_TYPE=${MODEL_TYPE} DATASET=${DATASET} RANDOM_SEED=${RANDOM_SEED}"
    if [[ "${GRID_USE_REPO_CSV}" == "1" ]]; then
      echo "    TRAIN_FILE=${TRAIN_FILE}"
    fi
    _log=$(bash "${CHILD_SH}" 2>&1) || _rc=$?
    _rc=${_rc:-0}
    echo "${_log}"
    if (( _rc != 0 )); then
      FAIL_COUNT=$((FAIL_COUNT + 1))
      GRID_RESULT_LINES+=("[FAIL] POOLING=${POOLING} ${MODEL_TYPE}/${DATASET}/seed_${RANDOM_SEED}")
    elif grep -q "^\[SKIP\] Already done:" <<< "${_log}"; then
      SKIP_COUNT=$((SKIP_COUNT + 1))
      GRID_RESULT_LINES+=("[SKIP] POOLING=${POOLING} ${MODEL_TYPE}/${DATASET}/seed_${RANDOM_SEED}")
    else
      RUN_COUNT=$((RUN_COUNT + 1))
      GRID_RESULT_LINES+=("[RUN ] POOLING=${POOLING} ${MODEL_TYPE}/${DATASET}/seed_${RANDOM_SEED}")
    fi
    unset _rc
  done
}

_run_grid_for_pooling() {
  export POOLING="$1"

  echo ""
  echo "###################################################################"
  echo "# POOLING=${POOLING} — 2×2 × seeds"
  echo "###################################################################"

  # ---------------------------------------------------------------------------
  # (1/4) esm2_t6 + s_aureus
  # ---------------------------------------------------------------------------
  export MODEL_TYPE=esm2_t6
  export DATASET=s_aureus
  _run_cell_seeds

  # ---------------------------------------------------------------------------
  # (2/4) esm2_t6 + e_coli
  # ---------------------------------------------------------------------------
  export MODEL_TYPE=esm2_t6
  export DATASET=e_coli
  _run_cell_seeds

  # ---------------------------------------------------------------------------
  # (3/4) esm2_t12 + s_aureus
  # ---------------------------------------------------------------------------
  export MODEL_TYPE=esm2_t12
  export DATASET=s_aureus
  _run_cell_seeds

  # ---------------------------------------------------------------------------
  # (4/4) esm2_t12 + e_coli
  # ---------------------------------------------------------------------------
  export MODEL_TYPE=esm2_t12
  export DATASET=e_coli
  _run_cell_seeds
}

echo "=========================================="
echo "Baseline pooling (mean / max / attn / last / latent_attn / fft_latent_attn_gate) — 2×2 grid × seeds"
echo "=========================================="
echo "REPO_ROOT:     ${REPO_ROOT}"
echo "POOLINGS:      ${POOLINGS}"
echo "CONDITION:     ${CONDITION}"
echo "DIFF:          ${DIFF}"
echo "SEEDS:         ${SEEDS}"
echo "Repo CSV:      ${GRID_USE_REPO_CSV}"
echo "OUTPUT_ROOT:   ${OUTPUT_ROOT}"
echo "Child script:  ${CHILD_SH}"
echo "=========================================="

# shellcheck disable=SC2086
for POOLING in ${POOLINGS}; do
  _run_grid_for_pooling "${POOLING}"
done

echo ""
echo "=========================================="
echo "Summary: RUN=${RUN_COUNT}  SKIP=${SKIP_COUNT}  FAIL=${FAIL_COUNT}"
echo "------------------------------------------"
for _line in "${GRID_RESULT_LINES[@]}"; do
  echo "  ${_line}"
done
echo "=========================================="
echo "Per-experiment aggregate (examples):"
echo "  python evaluation_scripts/aggregate_se_pooling_seed_metrics.py --exp-dir \"\${OUTPUT_ROOT}/esm2_t12_mean_e_coli_diff${DIFF}\""
echo "  python evaluation_scripts/aggregate_se_pooling_seed_metrics.py --exp-dir \"\${OUTPUT_ROOT}/esm2_t12_max_e_coli_diff${DIFF}\""
echo "  python evaluation_scripts/aggregate_se_pooling_seed_metrics.py --exp-dir \"\${OUTPUT_ROOT}/esm2_t12_attn_e_coli_diff${DIFF}\""
echo "  # EXP_TAG = \${MODEL_TYPE}_\${POOLING}_\${DATASET}_diff\${DIFF}"
echo "=========================================="

exit $(( FAIL_COUNT > 0 ? 1 : 0 ))

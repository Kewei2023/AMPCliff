#!/bin/bash
module load miniforge gcc/11.1.0 cuda/11.1 gcc/11.1.0
module load cudnn/8.1.0.77_CUDA11.1
# =============================================================================
# run_new_poolings_train.sh
#
# 对新增 pooling（last / latent_attn / swe_ot / mltp）在 2 dataset × 2 version 上 sweep，
# 每个组合跑多个随机种子；单次训练见 evaluation_scripts/run_new_poolings_train_single.sh。
#
# 输出：${OUTPUT_ROOT}/${EXP_TAG}/seed_${RANDOM_SEED}/，其中
#   EXP_TAG=${MODEL_TYPE}_${POOLING}_${DATASET}_diff${DIFF}
#   hydra.run.dir 固定为上述路径（无时间戳子目录）。
#
# 汇总（每个 EXP_TAG 一次，共 2×2×4=16 组）：
#   python evaluation_scripts/aggregate_se_pooling_seed_metrics.py --exp-dir "${OUTPUT_ROOT}/${EXP_TAG}"
#
# 环境变量：
#   SEEDS                默认 "0 1 2 3 4 5 6 7 8 9"
#   OUTPUT_ROOT          默认 ${REPO_ROOT}/outputs/ablation
#   AMPCLIFF_ABLATION_ROOT — 若设置且 OUTPUT_ROOT 未设置，可作为默认 ablation 根
#   CONDITION, DIFF, THRESHOLD, NUM_EPOCH, EVAL_EPOCH
#   DRY_RUN — 1 / true 或 --dry-run 仅打印子脚本命令
#   SKIP_IF_DONE — 传给子脚本
#
# 旧版脚本已备份为：evaluation_scripts/run_new_poolings_train.sh.bak
#
# 用法：
#   bash evaluation_scripts/run_new_poolings_train.sh
#   DRY_RUN=1 bash evaluation_scripts/run_new_poolings_train.sh
#   --dry-run  bash evaluation_scripts/run_new_poolings_train.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"
CHILD_SH="${REPO_ROOT}/evaluation_scripts/run_new_poolings_train_single.sh"
DRY_RUN="${DRY_RUN:-0}"

DATASETS=("e_coli" "s_aureus")
VERSIONS=("esm2_t6" "esm2_t12")
POOLINGS=("last" "latent_attn" "swe_ot" "mltp")

CONDITION="${CONDITION:-blosum62 average}"
DIFF="${DIFF:-5}"
THRESHOLD="${THRESHOLD:-0.9}"

AMPCLIFF_ABLATION_ROOT="${AMPCLIFF_ABLATION_ROOT:-}"
if [[ -n "${AMPCLIFF_ABLATION_ROOT}" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-${AMPCLIFF_ABLATION_ROOT}}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/ablation}"
fi
export OUTPUT_ROOT

# shellcheck disable=SC2086
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

_run_one_sweep() {
  export CONDITION
  export DIFF
  export THRESHOLD
  export OUTPUT_ROOT
  export SKIP_IF_DONE
  export DRY_RUN
  export NUM_EPOCH
  export EVAL_EPOCH
  unset RANDOM_SEED EXP_TAG OUTPUT_DIR 2>/dev/null || true

  # shellcheck disable=SC2086
  for RANDOM_SEED in ${SEEDS}; do
    export RANDOM_SEED
    echo ""
    echo ">>> Run: MODEL_TYPE=${MODEL_TYPE} DATASET=${DATASET} POOLING=${POOLING} RANDOM_SEED=${RANDOM_SEED}"
    echo "    OUTPUT_ROOT=${OUTPUT_ROOT}"
    bash "${CHILD_SH}"
  done
}

NUM_EPOCH="${NUM_EPOCH:-50}"
EVAL_EPOCH="${EVAL_EPOCH:-2}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"

echo "=========================================="
echo "New poolings — dataset × version × pooling × seeds"
echo "=========================================="
echo "REPO_ROOT:     ${REPO_ROOT}"
echo "CONDITION:     ${CONDITION}"
echo "DIFF:          ${DIFF}"
echo "THRESHOLD:     ${THRESHOLD}"
echo "SEEDS:         ${SEEDS}"
echo "OUTPUT_ROOT:   ${OUTPUT_ROOT}"
echo "NUM_EPOCH:     ${NUM_EPOCH}  EVAL_EPOCH: ${EVAL_EPOCH}"
echo "Child script:  ${CHILD_SH}"
echo "DRY_RUN:       ${DRY_RUN}"
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
  export DATASET="${dataset}"
  for version in "${VERSIONS[@]}"; do
    export MODEL_TYPE="${version}"
    for pooling in "${POOLINGS[@]}"; do
      export POOLING="${pooling}"
      _run_one_sweep
    done
  done
done

echo ""
echo "=========================================="
echo "All sweep steps finished."
echo "Per-experiment aggregate (EXP_TAG=\${MODEL_TYPE}_\${POOLING}_\${DATASET}_diff${DIFF}):"
for dataset in "${DATASETS[@]}"; do
  for version in "${VERSIONS[@]}"; do
    for pooling in "${POOLINGS[@]}"; do
      _exp_tag="${version}_${pooling}_${dataset}_diff${DIFF}"
      echo "  python evaluation_scripts/aggregate_se_pooling_seed_metrics.py --exp-dir \"${OUTPUT_ROOT}/${_exp_tag}\""
    done
  done
done
echo "=========================================="

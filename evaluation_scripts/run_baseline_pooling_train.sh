#!/bin/bash
module load miniforge gcc/11.1.0 cuda/11.1 gcc/11.1.0
module load cudnn/8.1.0.77_CUDA11.1
source activate AMPCliff
set -euo pipefail

# =============================================================================
# run_baseline_pooling_train.sh
#
# 单次回归训练：model.regression.pooling 为 mean / max / attn（经典 pooling，非 se_pooling）。
#
# 必设环境变量：
#   POOLING — mean | max | attn
#
# 其他环境变量（常用）：
#   MODEL_TYPE, DATASET, CONDITION, DIFF, THRESHOLD
#   TRAIN_FILE, VALID_FILE, TEST_FILE  — 未设置时按 REPO_ROOT 下默认 CSV
#   CONFIG_DIR — 未设置时按 MODEL_TYPE 选超算共享路径（与 factory/initializer.py、run_new_poolings_train_single.sh 一致）
#   OUTPUT_ROOT, EXP_TAG, OUTPUT_DIR — 输出目录；默认 hydra.run.dir=OUTPUT_DIR
#   RANDOM_SEED — 若设置：追加 train.random_seed，且在未显式设置 OUTPUT_DIR 时
#                 使用 ${OUTPUT_ROOT}/${EXP_TAG}/seed_${RANDOM_SEED}，避免多种子互相覆盖
#   DROPOUT — 回归头 dropout（默认 0.0，与 run_se_pooling_train.sh 一致）
#   OTHER_DEBUG, OTHER_DEBUG_SAMPLES — 对应 Hydra other.debug / other.debug_samples
#   PYTHON_BIN, DRY_RUN, NUM_EPOCH, SKIP_IF_DONE（为 1 时若 *-test_result.csv 已存在则跳过训练）
#
# 默认 EXP_TAG：${MODEL_TYPE}_${POOLING}_${DATASET}_diff${DIFF}
#
# 示例：
#   DRY_RUN=1 POOLING=mean MODEL_TYPE=esm2_t12 DATASET=e_coli bash evaluation_scripts/run_baseline_pooling_train.sh
#   DRY_RUN=1 POOLING=attn RANDOM_SEED=3 MODEL_TYPE=esm2_t12 DATASET=e_coli bash evaluation_scripts/run_baseline_pooling_train.sh
# =============================================================================

REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"


if [[ -z "${POOLING:-}" ]]; then
  echo "ERROR: POOLING must be set to one of: mean, max, attn" >&2
  exit 1
fi
case "${POOLING}" in
  mean|max|attn|last|swe_ot|mltp|latent_attn|fft_latent_attn_gate|fft_latent_attn_gate_v2) ;;
  *)
    echo "ERROR: Invalid POOLING='${POOLING}' (expected mean, max, or attn)" >&2
    exit 1
    ;;
esac

# PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/AMPCliff/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_TYPE="${MODEL_TYPE:-esm2_t12}"
DATASET="${DATASET:-e_coli}"
CONDITION="${CONDITION:-blosum62 average}"
DIFF="${DIFF:-5}"
THRESHOLD="${THRESHOLD:-0.9}"

NUM_EPOCH="${NUM_EPOCH:-50}"
EVAL_EPOCH="${EVAL_EPOCH:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-1}"

OTHER_DEBUG="${OTHER_DEBUG:-false}"
OTHER_DEBUG_SAMPLES="${OTHER_DEBUG_SAMPLES:-16}"

DROPOUT="${DROPOUT:-0.0}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"

_BASE_CSV="${REPO_ROOT}/data/${CONDITION}/diff_${DIFF}-trd_0.9"
TRAIN_FILE="${TRAIN_FILE:-${_BASE_CSV}/grampa_${DATASET}_7_25-train.csv}"
VALID_FILE="${VALID_FILE:-${_BASE_CSV}/grampa_${DATASET}_7_25-valid.csv}"
TEST_FILE="${TEST_FILE:-${_BASE_CSV}/grampa_${DATASET}_7_25-test.csv}"

if [[ -z "${CONFIG_DIR:-}" ]]; then
  case "${MODEL_TYPE}" in
    esm2_t6)
      CONFIG_DIR="/data/public/models/facebook/esm2_t6_8M_UR50D/"
      ;;
    esm2_t12)
      CONFIG_DIR="/data/public/models/facebook/esm2_t12_35M_UR50D/"
      ;;
    *)
      CONFIG_DIR="/data/public/models/facebook/esm2_t12_35M_UR50D/"
      ;;
  esac
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablation-new-data}"
EXP_TAG="${EXP_TAG:-${MODEL_TYPE}_${POOLING}_${DATASET}_diff${DIFF}}"

if [[ -n "${RANDOM_SEED:-}" ]]; then
  _DEFAULT_OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_TAG}/seed_${RANDOM_SEED}"
else
  _DEFAULT_OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_TAG}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${_DEFAULT_OUTPUT_DIR}}"

# DONE marker = downstream_train.py (per-seed *-test_result.csv), not metrics.json
DONE_MARKER="${OUTPUT_DIR}/${MODEL_TYPE}-${CONDITION}-diff${DIFF}-trd${THRESHOLD}-test_result.csv"

if [[ "${SKIP_IF_DONE}" == "1" && "${DRY_RUN}" != "1" && -f "${DONE_MARKER}" ]]; then
  echo "[SKIP] Already done: ${DONE_MARKER}"
  exit 0
fi

ARGS=(
  mode.ddp=false
  mode.amp=false
  logger.log=false
  "other.debug=${OTHER_DEBUG}"
  "other.debug_samples=${OTHER_DEBUG_SAMPLES}"
  "train.batch_size=${BATCH_SIZE}"
  "train.num_workers=${NUM_WORKERS}"
  "train.num_epoch=${NUM_EPOCH}"
  "train.eval_epoch=${EVAL_EPOCH}"
  "data.regression.mode=fix"
  "data.regression.dataset=${DATASET}"
  "data.regression.condition=[\"${CONDITION}\"]"
  "data.diff=[${DIFF}]"
  "data.threshold=${THRESHOLD}"
  "data.regression.fix.train_file=${TRAIN_FILE}"
  "data.regression.fix.valid_file=${VALID_FILE}"
  "data.regression.fix.test_file=${TEST_FILE}"
  "model.config_dir=${CONFIG_DIR}"
  "model.regression.version=${MODEL_TYPE}"
  "model.regression.apply=none"
  "model.regression.pooling=${POOLING}"
  "model.regression.pooling_common.dropout=${DROPOUT}"
  "model.regression.check_point.load=false"
  "hydra.run.dir=${OUTPUT_DIR}"
)

if [[ -n "${RANDOM_SEED:-}" ]]; then
  ARGS+=("train.random_seed=${RANDOM_SEED}")
fi

echo "=========================================="
echo "Baseline pooling train (mean / max / attn)"
echo "=========================================="
echo "REPO_ROOT=${REPO_ROOT}"
# echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MODEL_TYPE=${MODEL_TYPE} DATASET=${DATASET}"
echo "CONDITION=${CONDITION} DIFF=${DIFF} THRESHOLD=${THRESHOLD}"
echo "POOLING=${POOLING} DROPOUT=${DROPOUT}"
echo "RANDOM_SEED=${RANDOM_SEED:-<unset>}"
echo "CONFIG_DIR=${CONFIG_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SKIP_IF_DONE=${SKIP_IF_DONE} DONE_MARKER=${DONE_MARKER}"
echo "=========================================="

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY RUN] python -u downstream_train.py ${ARGS[*]}"
  exit 0
fi

python -u downstream_train.py "${ARGS[@]}"

echo "[DONE] Training finished. Output directory: ${OUTPUT_DIR}"

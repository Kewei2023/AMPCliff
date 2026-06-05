#!/bin/bash
module load miniforge gcc/11.1.0 cuda/11.1 gcc/11.1.0
module load cudnn/8.1.0.77_CUDA11.1
source activate AMPCliff
set -euo pipefail

# =============================================================================
# run_new_poolings_train_single.sh
#
# 单次训练：last / latent_attn / swe_ot / mltp 等 readout pooling，与 run_se_pooling_train.sh 同构
#（fix CSV、OUTPUT_ROOT/EXP_TAG/seed、hydra.run.dir 无时间戳目录）。
#
# 必设环境变量：POOLING — last | latent_attn | swe_ot | mltp
# 常用：MODEL_TYPE, DATASET, CONDITION, DIFF, THRESHOLD, RANDOM_SEED,
#   OUTPUT_ROOT, EXP_TAG, OUTPUT_DIR, CONFIG_DIR,
#   NUM_EPOCH, EVAL_EPOCH, BATCH_SIZE, NUM_WORKERS,
#   OTHER_DEBUG, OTHER_DEBUG_SAMPLES, PYTHON_BIN, DRY_RUN, SKIP_IF_DONE
#
# EXP_TAG 默认：${MODEL_TYPE}_${POOLING}_${DATASET}_diff${DIFF}
# CONFIG_DIR 未设置：/data/public/models/facebook/ 下 esm2_t6 / esm2_t12（与 initializer 超算路径一致）
#
# 示例：
#   POOLING=last MODEL_TYPE=esm2_t12 DATASET=e_coli RANDOM_SEED=0 DRY_RUN=1 bash evaluation_scripts/run_new_poolings_train_single.sh
# =============================================================================

REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${POOLING:-}" ]]; then
  echo "ERROR: POOLING must be set (e.g. last, latent_attn, swe_ot, mltp)" >&2
  exit 1
fi
case "${POOLING}" in
  last|latent_attn|swe_ot|mltp) ;;
  *)
    echo "ERROR: Invalid POOLING='${POOLING}'" >&2
    exit 1
    ;;
esac

if [[ "${DRY_RUN}" == "true" ]]; then
  DRY_RUN=1
fi

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

SKIP_IF_DONE="${SKIP_IF_DONE:-1}"

_BASE_CSV="${REPO_ROOT}/data/blast/${CONDITION}/diff_${DIFF}"
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

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/ablation}"
EXP_TAG="${EXP_TAG:-${MODEL_TYPE}_${POOLING}_${DATASET}_diff${DIFF}}"

if [[ -n "${RANDOM_SEED:-}" ]]; then
  _DEFAULT_OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_TAG}/seed_${RANDOM_SEED}"
else
  _DEFAULT_OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_TAG}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${_DEFAULT_OUTPUT_DIR}}"

METRICS_MARKER="${OUTPUT_DIR}/metrics.json"

if [[ "${SKIP_IF_DONE}" == "1" && "${DRY_RUN}" != "1" && -f "${METRICS_MARKER}" ]]; then
  echo "[SKIP] Already done: ${METRICS_MARKER}"
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
  "model.regression.check_point.load=false"
  "hydra.run.dir=${OUTPUT_DIR}"
)

if [[ -n "${RANDOM_SEED:-}" ]]; then
  ARGS+=("train.random_seed=${RANDOM_SEED}")
fi

echo "=========================================="
echo "New poolings train (single run)"
echo "=========================================="
echo "REPO_ROOT=${REPO_ROOT}"
# echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MODEL_TYPE=${MODEL_TYPE} DATASET=${DATASET}"
echo "POOLING=${POOLING}"
echo "CONDITION=${CONDITION} DIFF=${DIFF} THRESHOLD=${THRESHOLD}"
echo "RANDOM_SEED=${RANDOM_SEED:-<unset>}"
echo "CONFIG_DIR=${CONFIG_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SKIP_IF_DONE=${SKIP_IF_DONE} METRICS_MARKER=${METRICS_MARKER}"
echo "=========================================="

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY RUN] python -u downstream_train.py ${ARGS[*]}"
  exit 0
fi

python -u downstream_train.py "${ARGS[@]}"

echo "[DONE] Training finished. Output directory: ${OUTPUT_DIR}"

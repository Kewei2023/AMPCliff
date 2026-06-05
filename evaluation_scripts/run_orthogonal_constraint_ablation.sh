#!/usr/bin/env bash
set -euo pipefail

# One-click runner for orthogonal constraint ablation:
# 1) Baseline without orthogonal constraint
# 2) Weight sweep (lambda)
# 3) Constraint type sweep (pairwise / sequential / gram)
# 4) Layer index sweep (all vs selected layers)

PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/AMPCliff/bin/python}"
SCRIPT="${SCRIPT:-downstream_train.py}"
DRY_RUN="${DRY_RUN:-0}"

# Dataset/model settings (override when launching)
MODEL_TYPE="${MODEL_TYPE:-esm2_t6}"
MODEL_DIR="${MODEL_DIR:-/mnt/d/AMPCliff/models/facebook/${MODEL_TYPE}_8M_UR50D}"
USE_MODEL_DIR="${USE_MODEL_DIR:-0}" # set 1 to force model.config_dir override
DATASET="${DATASET:-s_aureus}"
CONDITION="${CONDITION:-blosum62 average}"
DIFF="${DIFF:-5}"
TRAIN_FILE="${TRAIN_FILE:-/mnt/d/AMPCliff/data/blast/${CONDITION}/diff_${DIFF}/grampa_${DATASET}_7_25-train.csv}"
VALID_FILE="${VALID_FILE:-/mnt/d/AMPCliff/data/blast/${CONDITION}/diff_${DIFF}/grampa_${DATASET}_7_25-valid.csv}"
TEST_FILE="${TEST_FILE:-/mnt/d/AMPCliff/data/blast/${CONDITION}/diff_${DIFF}/grampa_${DATASET}_7_25-test.csv}"

# Train settings
POOLING="${POOLING:-mean}"
NUM_EPOCH="${NUM_EPOCH:-50}"
EVAL_EPOCH="${EVAL_EPOCH:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

# Orthogonal settings
ORTHO_NORMALIZE="${ORTHO_NORMALIZE:-true}"
ORTHO_LOG_METRICS="${ORTHO_LOG_METRICS:-true}"
ORTHO_WEIGHT_BASE="${ORTHO_WEIGHT_BASE:-0.01}"
ORTHO_TYPE_BASE="${ORTHO_TYPE_BASE:-pairwise}"
ORTHO_LAYER_BASE="${ORTHO_LAYER_BASE:-null}" # null or list like [0,2,4]

OUTPUT_ROOT="${OUTPUT_ROOT:-${AMPCLIFF_ABLATION_ROOT:-outputs/ablation}}"
EXP_TAG="${EXP_TAG:-${MODEL_TYPE}_orthogonal_ablation_${DATASET}_diff${DIFF}}"
ENABLE_MLFLOW="${ENABLE_MLFLOW:-auto}"

COMMON_ARGS=(
  "model.regression.version=${MODEL_TYPE}"
  "model.regression.apply=none"
  "model.regression.pooling=${POOLING}"
  "data.regression.mode=fix"
  "data.regression.fix.train_file=${TRAIN_FILE}"
  "data.regression.fix.valid_file=${VALID_FILE}"
  "data.regression.fix.test_file=${TEST_FILE}"
  "train.num_epoch=${NUM_EPOCH}"
  "train.eval_epoch=${EVAL_EPOCH}"
  "train.batch_size=${BATCH_SIZE}"
  "train.learning_rate=${LEARNING_RATE}"
  "other.debug=false"
  "logger.log=true"
  "hydra.run.dir=${OUTPUT_ROOT}/${EXP_TAG}/enabled_\${orthogonal_constraint.enabled}_w\${orthogonal_constraint.weight}_t\${orthogonal_constraint.constraint_type}_l\${orthogonal_constraint.layer_indices}"
  "hydra.sweep.dir=${OUTPUT_ROOT}/${EXP_TAG}/multirun"
  "hydra.sweep.subdir=enabled_\${orthogonal_constraint.enabled}_w\${orthogonal_constraint.weight}_t\${orthogonal_constraint.constraint_type}_l\${orthogonal_constraint.layer_indices}"
)

if [[ "${USE_MODEL_DIR}" == "1" ]]; then
  COMMON_ARGS+=("model.config_dir=${MODEL_DIR}")
fi

LOGGER_LOG_VALUE="true"
if [[ "${ENABLE_MLFLOW}" == "false" ]]; then
  LOGGER_LOG_VALUE="false"
elif [[ "${ENABLE_MLFLOW}" == "true" ]]; then
  LOGGER_LOG_VALUE="true"
else
  if [[ -z "${MLFLOW_TRACKING_URI:-}" || -z "${MLFLOW_EXPERIMENT_NAME:-}" ]]; then
    LOGGER_LOG_VALUE="false"
  fi
fi

for idx in "${!COMMON_ARGS[@]}"; do
  if [[ "${COMMON_ARGS[$idx]}" == logger.log=* ]]; then
    COMMON_ARGS[$idx]="logger.log=${LOGGER_LOG_VALUE}"
  fi
done

run_cmd() {
  echo "=================================================="
  echo "[RUN] $*"
  echo "=================================================="
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" "${SCRIPT}" "$@"
}

echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Script: ${SCRIPT}"
echo "[INFO] Dataset/Condition/Diff: ${DATASET} / ${CONDITION} / ${DIFF}"
echo "[INFO] Train/Valid/Test:"
echo "       ${TRAIN_FILE}"
echo "       ${VALID_FILE}"
echo "       ${TEST_FILE}"
echo "[INFO] logger.log: ${LOGGER_LOG_VALUE}"
echo "[INFO] Dry run: ${DRY_RUN}"
echo "[INFO] Use model.config_dir override: ${USE_MODEL_DIR}"
if [[ "${USE_MODEL_DIR}" == "1" ]]; then
  echo "[INFO] Model dir: ${MODEL_DIR}"
fi

echo
echo "[STAGE 1/4] Baseline (orthogonal disabled)"
run_cmd "${COMMON_ARGS[@]}" \
  "orthogonal_constraint.enabled=false"

echo
echo "[STAGE 2/4] Orthogonal weight sweep (type=${ORTHO_TYPE_BASE}, layers=${ORTHO_LAYER_BASE})"
for w in 0.001 0.005 0.02 0.05; do
  run_cmd "${COMMON_ARGS[@]}" \
    "orthogonal_constraint.enabled=true" \
    "orthogonal_constraint.weight=${w}" \
    "orthogonal_constraint.constraint_type=${ORTHO_TYPE_BASE}" \
    "orthogonal_constraint.layer_indices=${ORTHO_LAYER_BASE}" \
    "orthogonal_constraint.normalize=${ORTHO_NORMALIZE}" \
    "orthogonal_constraint.log_metrics=${ORTHO_LOG_METRICS}"
done

echo
echo "[STAGE 3/4] Constraint type sweep (weight=${ORTHO_WEIGHT_BASE}, layers=${ORTHO_LAYER_BASE})"
for ctype in sequential gram; do
  run_cmd "${COMMON_ARGS[@]}" \
    "orthogonal_constraint.enabled=true" \
    "orthogonal_constraint.weight=${ORTHO_WEIGHT_BASE}" \
    "orthogonal_constraint.constraint_type=${ctype}" \
    "orthogonal_constraint.layer_indices=${ORTHO_LAYER_BASE}" \
    "orthogonal_constraint.normalize=${ORTHO_NORMALIZE}" \
    "orthogonal_constraint.log_metrics=${ORTHO_LOG_METRICS}"
done

echo
echo "[STAGE 4/4] Layer index sweep (weight=${ORTHO_WEIGHT_BASE}, type=${ORTHO_TYPE_BASE})"
for layers in "[0,2,4]" "[1,3,5]"; do
  run_cmd "${COMMON_ARGS[@]}" \
    "orthogonal_constraint.enabled=true" \
    "orthogonal_constraint.weight=${ORTHO_WEIGHT_BASE}" \
    "orthogonal_constraint.constraint_type=${ORTHO_TYPE_BASE}" \
    "orthogonal_constraint.layer_indices=${layers}" \
    "orthogonal_constraint.normalize=${ORTHO_NORMALIZE}" \
    "orthogonal_constraint.log_metrics=${ORTHO_LOG_METRICS}"
done

echo
echo "[DONE] Orthogonal constraint ablation finished."

#!/usr/bin/env bash
set -euo pipefail

# One-click runner for:
# 1) Baseline pooling: mean / max / attn
# 2) SpectralAnchor ablation: num_anchor sweep
# 3) SpectralAnchor ablation: use_fft true/false

PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/AMPCliff/bin/python}"
SCRIPT="${SCRIPT:-downstream_train.py}"
DRY_RUN="${DRY_RUN:-0}"

# You can override these paths when launching the script.
MODEL_TYPE="${MODEL_TYPE:-esm2_t6}"
MODEL_DIR="${MODEL_DIR:-/mnt/d/AMPCliff/models/facebook/${MODEL_TYPE}_8M_UR50D}"
DATASET="${DATASET:-s_aureus}"
CONDITION="${CONDITION:-blosum62 average}"
DIFF="${DIFF:-5}"
TRAIN_FILE="${TRAIN_FILE:-/mnt/d/AMPCliff/data/blast/${CONDITION}/diff_${DIFF}/grampa_${DATASET}_7_25-train.csv}"
VALID_FILE="${VALID_FILE:-/mnt/d/AMPCliff/data/blast/${CONDITION}/diff_${DIFF}/grampa_${DATASET}_7_25-valid.csv}"
TEST_FILE="${TEST_FILE:-/mnt/d/AMPCliff/data/blast/${CONDITION}/diff_${DIFF}/grampa_${DATASET}_7_25-test.csv}"
NUM_EPOCH="${NUM_EPOCH:-50}"
EVAL_EPOCH="${EVAL_EPOCH:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AMPCLIFF_ABLATION_ROOT:-outputs/ablation}}"
EXP_TAG="${EXP_TAG:-${MODEL_TYPE}_pooling_baseline_ablation_${DATASET}_diff${DIFF}}"
ENABLE_MLFLOW="${ENABLE_MLFLOW:-auto}"

COMMON_ARGS=(
  # "model.config_dir=${MODEL_DIR}"
  "model.regression.version=${MODEL_TYPE}"
  "model.regression.apply=none"
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
  "hydra.run.dir=${OUTPUT_ROOT}/${EXP_TAG}/\${model.regression.pooling}_k\${model.regression.pooling_common.num_anchor}_fft\${model.regression.pooling_common.use_fft}"
  "hydra.sweep.dir=${OUTPUT_ROOT}/${EXP_TAG}/multirun"
  "hydra.sweep.subdir=\${model.regression.pooling}_k\${model.regression.pooling_common.num_anchor}_fft\${model.regression.pooling_common.use_fft}"
)

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
echo "[INFO] Model dir: ${MODEL_DIR}"
echo "[INFO] Train/Valid/Test:"
echo "       ${TRAIN_FILE}"
echo "       ${VALID_FILE}"
echo "       ${TEST_FILE}"
echo "[INFO] logger.log: ${LOGGER_LOG_VALUE}"
echo "[INFO] Dry run: ${DRY_RUN}"

echo
# echo "[STAGE 1/3] Baseline pooling: mean / max / attn"
# for pooling in mean max attn; do
#   run_cmd "${COMMON_ARGS[@]}" \
#     "model.regression.pooling=${pooling}"
# done

echo
echo "[STAGE 2/3] SpectralAnchor num_anchor ablation (use_fft=true)"
for k in 1 2 4 8 16; do
  run_cmd "${COMMON_ARGS[@]}" \
    "model.regression.pooling=spectral_anchor" \
    "model.regression.pooling_common.num_anchor=${k}" \
    "model.regression.pooling_config.spectral_anchor.use_fft=true"
done

echo
# echo "[STAGE 3/3] SpectralAnchor use_fft ablation (num_anchor=8)"
# for k in 1 2 4 8 16; do
#   run_cmd "${COMMON_ARGS[@]}" \
#     "model.regression.pooling=spectral_anchor" \
#     "model.regression.num_anchor=${k}" \
#     "model.regression.use_fft=false"
# done

echo
echo "[DONE] Baseline + ablation runs finished."

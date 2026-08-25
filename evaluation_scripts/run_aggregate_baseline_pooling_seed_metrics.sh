#!/bin/bash
module load miniforge gcc/11.1.0 cuda/11.1 gcc/11.1.0
module load cudnn/8.1.0.77_CUDA11.1
source activate AMPCliff
# Do not use `set -e`: run all 12 configs even if one experiment dir is missing.
set -uo pipefail

# Aggregate test/valid Spearman, Pearson, top-K recall per seed_* for baseline pooling
# (attn_structured, …) × 2×2 (esm2_t6/t12 × e_coli/s_aureus); extend the for-loop as needed.
#
# Writes CSV + JSON under ${STATICS_ROOT}/<pooling>/ (one subdir per pooling name):
#   statics/attn_structured/seed_metrics_attn_structured_pooling_esm2_t6_e_coli.csv / .json
#   statics/local_stft/seed_metrics_local_stft_pooling_esm2_t6_e_coli.csv / .json
#   ... (per model×dataset combo in the loop below)
#
# Environment:
#   AMPCLIFF_ABLATION_ROOT   (default below)
#   STATICS_ROOT             default ${REPO_ROOT}/statics — parent for pooling subdirs
#   DIFF                     default 5
#   TOPK                     recall top-K (default 50, passed to --topk)
#   DRY_RUN=1                print only, no files
#
# Usage:
#   bash evaluation_scripts/run_aggregate_baseline_pooling_seed_metrics.sh
#   DRY_RUN=1 bash evaluation_scripts/run_aggregate_baseline_pooling_seed_metrics.sh
#   bash evaluation_scripts/run_aggregate_baseline_pooling_seed_metrics.sh -- --exp-dir ... --out-csv ...


REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"
STATICS_ROOT="${STATICS_ROOT:-${REPO_ROOT}/statics}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PY="${REPO_ROOT}/evaluation_scripts/aggregate_pooling_seed_metrics.py"

if [[ "${#}" -gt 0 ]]; then
  exec "${PYTHON_BIN}" "${PY}" "$@"
fi

AMPCLIFF_ABLATION_ROOT="${AMPCLIFF_ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation}"
ROOT="${AMPCLIFF_ABLATION_ROOT}"
DIFF="${DIFF:-5}"
TOPK="${TOPK:-50}"

EXTRA_PY_ARGS=(--topk "${TOPK}")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_PY_ARGS+=(--dry-run)
fi

run_one() {
  local exp_subdir="$1"
  local model_type="$2"
  local dataset="$3"
  local pooling="$4"
  local out_csv="${STATICS_ROOT}/${pooling}/seed_metrics_${pooling}_pooling_${model_type}_${dataset}.csv"
  "${PYTHON_BIN}" "${PY}" \
    --exp-dir "${ROOT}/${exp_subdir}" \
    --out-csv "${out_csv}" \
    "${EXTRA_PY_ARGS[@]}"
}

failed=0
for POOLING in mean max attn_structured last mltp_paper latent_attn FLaG; do
  run_one "esm2_t6_${POOLING}_e_coli_diff${DIFF}" "esm2_t6" "e_coli" "${POOLING}" || failed=1
  run_one "esm2_t6_${POOLING}_s_aureus_diff${DIFF}" "esm2_t6" "s_aureus" "${POOLING}" || failed=1
  run_one "esm2_t12_${POOLING}_e_coli_diff${DIFF}" "esm2_t12" "e_coli" "${POOLING}" || failed=1
  run_one "esm2_t12_${POOLING}_s_aureus_diff${DIFF}" "esm2_t12" "s_aureus" "${POOLING}" || failed=1
done
exit "${failed}"

#!/usr/bin/env bash
# =============================================================================
# run_fftlag_exp3_revised_poolings_parallel.sh
#
# Exp3 full-test token knockout for revised poolings on two GPUs in parallel:
#   GPU0 (default): mltp_paper
#   GPU1 (default): attn_structured
#
# Usage:
#   DRY_RUN=1 bash evaluation_scripts/run_fftlag_exp3_revised_poolings_parallel.sh
#   bash evaluation_scripts/run_fftlag_exp3_revised_poolings_parallel.sh
#   GPU0=0 GPU1=1 SEEDS="0 1" bash evaluation_scripts/run_fftlag_exp3_revised_poolings_parallel.sh
# =============================================================================
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
SEEDS="${SEEDS:-}"
DATASETS="${DATASETS:-}"

RUN_SCRIPT="${REPO_ROOT}/evaluation_scripts/run_fftlag_exp3_fulltest.sh"
COMMON_ENV=(
  "REPO_ROOT=${REPO_ROOT}"
  "DRY_RUN=${DRY_RUN}"
  "SKIP_IF_DONE=${SKIP_IF_DONE}"
  "MODEL_VERSION=${MODEL_VERSION:-esm2_t6}"
)

if [[ -n "${SEEDS}" ]]; then
  COMMON_ENV+=("SEEDS=${SEEDS}")
fi
if [[ -n "${DATASETS}" ]]; then
  COMMON_ENV+=("DATASETS=${DATASETS}")
fi

echo "========== Exp3 revised poolings parallel knockout =========="
echo "REPO_ROOT=${REPO_ROOT}"
echo "GPU0=${GPU0} -> mltp_paper"
echo "GPU1=${GPU1} -> attn_structured"
echo "DRY_RUN=${DRY_RUN} SKIP_IF_DONE=${SKIP_IF_DONE}"
echo "============================================================="

env "${COMMON_ENV[@]}" CUDA_VISIBLE_DEVICES="${GPU0}" POOLINGS=mltp_paper \
  bash "${RUN_SCRIPT}" > logs/exp3_knockout_mltp_paper_gpu"${GPU0}".log 2>&1 &
PID0=$!

env "${COMMON_ENV[@]}" CUDA_VISIBLE_DEVICES="${GPU1}" POOLINGS=attn_structured \
  bash "${RUN_SCRIPT}" > logs/exp3_knockout_attn_structured_gpu"${GPU1}".log 2>&1 &
PID1=$!

FAIL=0
wait "${PID0}" || FAIL=1
wait "${PID1}" || FAIL=1

echo ""
echo "Logs:"
echo "  logs/exp3_knockout_mltp_paper_gpu${GPU0}.log"
echo "  logs/exp3_knockout_attn_structured_gpu${GPU1}.log"
echo "Done. FAIL=${FAIL}"
echo "Next:"
echo "  python evaluation_scripts/aggregate_exp3_token_knockout_mse_diff.py --force"
echo "  python evaluation_scripts/plot_fftlag_exp3_mse_diff_violin_revised.py --force"
echo "  python evaluation_scripts/export_fftlag_exp3_token_knockout_data.py"
exit "${FAIL}"

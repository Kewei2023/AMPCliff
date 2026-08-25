#!/usr/bin/env bash
#SBATCH --job-name=fftlag_attn_agg
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/fftlag_attn_agg_%j.out
#SBATCH --error=logs/fftlag_attn_agg_%j.err
#
# CPU-only: aggregate Exp4 attn_score_raw across train seeds (per dataset).
# e_coli and s_aureus are aggregated separately under aggregated/{dataset}/exp4/...
#
# Usage:
#   bash evaluation_scripts/run_aggregate_fftlag_exp4_attn_score_raw.sh
#   sbatch evaluation_scripts/run_aggregate_fftlag_exp4_attn_score_raw.sh
#   SEEDS="0 1 2 3 4" DATASETS=s_aureus bash evaluation_scripts/run_aggregate_fftlag_exp4_attn_score_raw.sh
#   FORCE=1 DRY_RUN=1 bash evaluation_scripts/run_aggregate_fftlag_exp4_attn_score_raw.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load miniforge gcc/11.1.0 cuda/11.1 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate AMPCliff 2>/dev/null || source activate AMPCliff 2>/dev/null || true
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${REPO_ROOT}/outputs/analysis/fftlag_mechanism}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
MIN_SEEDS="${MIN_SEEDS:-1}"

read -r -a DATASETS <<< "${DATASETS:-e_coli s_aureus}"

EXTRA_ARGS=(--min-seeds "${MIN_SEEDS}")
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
  EXTRA_ARGS+=(--seeds "${SEED_LIST[@]}")
fi
if [[ -n "${IDX:-}" ]]; then
  read -r -a IDX_LIST <<< "${IDX}"
  EXTRA_ARGS+=(--idx "${IDX_LIST[@]}")
fi
if [[ "${FORCE}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

echo "========== Exp4 attn_score_raw aggregate (per dataset) =========="
echo "REPO_ROOT=${REPO_ROOT}"
echo "ANALYSIS_ROOT=${ANALYSIS_ROOT}"
echo "DATASETS=${DATASETS[*]}"
echo "SEEDS=${SEEDS:-all seed_*}"
echo "MIN_SEEDS=${MIN_SEEDS} FORCE=${FORCE} DRY_RUN=${DRY_RUN}"
echo "Output: aggregated/{dataset}/exp4/per_sample/idx_*/plots/mean_style/attn_score_raw.png"
echo "================================================================"

"${PYTHON_BIN}" "${REPO_ROOT}/evaluation_scripts/aggregate_fftlag_exp4_attn_score_raw.py" \
  --analysis-root "${ANALYSIS_ROOT}" \
  --datasets "${DATASETS[@]}" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

exit $?

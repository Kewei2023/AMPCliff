#!/usr/bin/env bash
#SBATCH --job-name=fftlag_attn_raw
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --chdir=/data/home/scv6872/run/kwli/AMPCliff
#SBATCH --output=logs/fftlag_attn_raw_%j.out
#SBATCH --error=logs/fftlag_attn_raw_%j.err
#
# CPU-only offline replot: latent_attn_weights.pt -> per_sample/idx_*/attn_score_raw.png
# Requires prior Exp4 runs that saved latent_attn_weights.pt under exp4_latent/.
#
# Usage:
#   bash evaluation_scripts/run_fftlag_exp4_attn_score_raw.sh
#   sbatch evaluation_scripts/run_fftlag_exp4_attn_score_raw.sh
#   SEEDS="0 1" DATASETS=s_aureus bash evaluation_scripts/run_fftlag_exp4_attn_score_raw.sh
#   FORCE=1 DRY_RUN=1 bash evaluation_scripts/run_fftlag_exp4_attn_score_raw.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"
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

read -r -a DATASETS <<< "${DATASETS:-e_coli s_aureus}"

EXTRA_ARGS=()
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
  EXTRA_ARGS+=(--seeds "${SEED_LIST[@]}")
fi
if [[ "${FORCE}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

echo "========== Exp4 attn_score_raw offline replot =========="
echo "REPO_ROOT=${REPO_ROOT}"
echo "ANALYSIS_ROOT=${ANALYSIS_ROOT}"
echo "DATASETS=${DATASETS[*]}"
echo "SEEDS=${SEEDS:-all seed_*}"
echo "FORCE=${FORCE} DRY_RUN=${DRY_RUN}"
echo "========================================================"

"${PYTHON_BIN}" "${REPO_ROOT}/evaluation_scripts/plot_fftlag_exp4_attn_score_raw.py" \
  --analysis-root "${ANALYSIS_ROOT}" \
  --datasets "${DATASETS[@]}" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

exit $?

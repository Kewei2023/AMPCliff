#!/usr/bin/env bash
#SBATCH --job-name=fftlag_exp4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/fftlag_exp4_%j.out
#SBATCH --error=logs/fftlag_exp4_%j.err
#
# Exp4 only: training-aligned latent visualization (all seeds, s_aureus + e_coli).
# Does not run Exp1/Exp2/Exp5. Aggregates Exp4 CSVs across seeds per dataset.
#
# Usage:
#   sbatch --gpus=1 evaluation_scripts/run_fftlag_exp4_only.sh
#   SEEDS="0 1 2" DATASETS=s_aureus sbatch evaluation_scripts/run_fftlag_exp4_only.sh
#   DRY_RUN=1 bash evaluation_scripts/run_fftlag_exp4_only.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

export RUN_EXPS=4
export RUN_EXP5=0
export SKIP_IF_DONE="${SKIP_IF_DONE:-0}"

echo "========== FFT-LAG Exp4 only (latent viz) =========="
echo "RUN_EXPS=${RUN_EXPS} RUN_EXP5=${RUN_EXP5} SKIP_IF_DONE=${SKIP_IF_DONE}"
echo "SEEDS=${SEEDS:-0..9 default} DATASETS=${DATASETS:-e_coli s_aureus}"
echo "===================================================="

bash "${REPO_ROOT}/evaluation_scripts/run_fftlag_mechanism_experiments.sh"
_rc=$?

echo ""
echo "Exp4 batch finished with exit code ${_rc}"
exit "${_rc}"

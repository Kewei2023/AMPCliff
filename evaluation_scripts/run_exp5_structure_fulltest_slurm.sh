#!/usr/bin/env bash
#SBATCH --job-name=exp5_fulltest
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp5_structure_fulltest_%j.out
#SBATCH --error=logs/exp5_structure_fulltest_%j.err
#
# CPU post-processing after Exp2 fulltest GPU job completes.
#
# Usage:
#   sbatch evaluation_scripts/run_exp5_structure_fulltest_slurm.sh
#   sbatch --dependency=afterok:JOBID evaluation_scripts/run_exp5_structure_fulltest_slurm.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load miniforge gcc/11.1.0 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate AMPCliff 2>/dev/null || source activate AMPCliff 2>/dev/null || true
fi

echo "========== SLURM Exp5 structure fulltest (CPU) =========="
echo "JOB_ID=${SLURM_JOB_ID:-local}"
echo "========================================================="

bash "${REPO_ROOT}/evaluation_scripts/run_exp5_structure_fulltest.sh"

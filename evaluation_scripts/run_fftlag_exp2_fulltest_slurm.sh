#!/usr/bin/env bash
#SBATCH --job-name=fftlag_exp2_ft
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --chdir=/data/home/scv6872/run/kwli/AMPCliff
#SBATCH --output=logs/fftlag_exp2_fulltest_%j.out
#SBATCH --error=logs/fftlag_exp2_fulltest_%j.err
#
# Single GPU job: serially runs seeds 0-9 (both datasets on full test set).
#
# Usage:
#   sbatch evaluation_scripts/run_fftlag_exp2_fulltest_slurm.sh
#   SEEDS="0 1" sbatch evaluation_scripts/run_fftlag_exp2_fulltest_slurm.sh  # optional subset

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"
cd "${REPO_ROOT}"
mkdir -p logs

echo "========== SLURM Exp2 fulltest (serial) =========="
echo "JOB_ID=${SLURM_JOB_ID:-local}"
echo "SEEDS=${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
echo "================================================="

bash "${REPO_ROOT}/evaluation_scripts/run_fftlag_exp2_fulltest.sh"

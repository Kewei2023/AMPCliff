#!/usr/bin/env bash
#SBATCH --job-name=fftlag_exp3_ft
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --chdir=/data/home/scv6872/run/kwli/AMPCliff
#SBATCH --output=logs/fftlag_exp3_fulltest_%j.out
#SBATCH --error=logs/fftlag_exp3_fulltest_%j.err
#
# SLURM wrapper for Exp3 full-test token knockout.
#
# Usage:
#   sbatch evaluation_scripts/run_fftlag_exp3_fulltest_slurm.sh
#   POOLINGS=fft_latent_attn_gate SEEDS="0 1" sbatch evaluation_scripts/run_fftlag_exp3_fulltest_slurm.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"
bash "${REPO_ROOT}/evaluation_scripts/run_fftlag_exp3_fulltest.sh"

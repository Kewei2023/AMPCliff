#!/usr/bin/env bash
#SBATCH --job-name=fftlag_exp2_ecoli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/rerun_fftlag_exp2_ecoli_%j.out
#SBATCH --error=logs/rerun_fftlag_exp2_ecoli_%j.err
#
# Re-run Exp2 (Gate PSD) for e_coli seeds that failed partially (torch.cat bug).
# Deletes exp2_psd_gate/ per seed before running; does not touch Exp1/Exp4.
#
# Usage:
#   sbatch --gpus=1 evaluation_scripts/rerun_fftlag_exp2_ecoli.sh
#   SEEDS="0 2 4" DRY_RUN=1 bash evaluation_scripts/rerun_fftlag_exp2_ecoli.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load miniforge gcc/11.1.0 cuda/11.1 2>/dev/null || true
  module load cudnn/8.1.0.77_CUDA11.1 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate AMPCliff 2>/dev/null || source activate AMPCliff 2>/dev/null || true
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_VERSION="${MODEL_VERSION:-esm2_t6}"
POOLING="${POOLING:-FLaG}"
DIFF="${DIFF:-5}"
THRESHOLD="${THRESHOLD:-0.9}"
DATASET="${DATASET:-e_coli}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4 5}"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-${REPO_ROOT}/outputs/analysis/fftlag_mechanism}"
ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_new_data}"
MANIFEST_DIR="${MANIFEST_DIR:-${ABLATION_ROOT}/_amp_knockout_seed_runs/_peptide_manifest}"
CONFIG_DIR="/data/public/models/facebook/esm2_t6_8M_UR50D/"

_BASE_CSV="${REPO_ROOT}/data/blosum62 average/diff_${DIFF}-trd_${THRESHOLD}"
train_csv="${_BASE_CSV}/grampa_${DATASET}_7_25-train.csv"
valid_csv="${_BASE_CSV}/grampa_${DATASET}_7_25-valid.csv"
test_csv="${_BASE_CSV}/grampa_${DATASET}_7_25-test.csv"
manifest="${MANIFEST_DIR}/${MODEL_VERSION}_${DATASET}_diff${DIFF}.json"

resolve_ckpt_dir() {
  local seed_root="$1"
  local mp
  mp="$(find "${seed_root}" -type f -name model.pth 2>/dev/null | head -1)"
  if [[ -z "${mp}" ]]; then
    return 1
  fi
  dirname "$(dirname "${mp}")"
}

RUN=0
SKIP=0
FAIL=0

echo "========== Rerun Exp2 (e_coli) =========="
echo "REPO_ROOT=${REPO_ROOT}"
echo "ANALYSIS_ROOT=${ANALYSIS_ROOT}"
echo "SEEDS=${SEED_LIST[*]} DATASET=${DATASET}"
echo "manifest=${manifest}"
echo "=========================================="

if [[ ! -f "${manifest}" ]]; then
  echo "[ERROR] manifest not found: ${manifest}"
  exit 1
fi

for seed in "${SEED_LIST[@]}"; do
  exp_root="${ABLATION_ROOT}/${MODEL_VERSION}_${POOLING}_${DATASET}_diff${DIFF}/seed_${seed}"
  seed_out="${ANALYSIS_ROOT}/seed_${seed}/${DATASET}"
  exp2_dir="${seed_out}/exp2_psd_gate"

  ckpt=""
  if ! ckpt="$(resolve_ckpt_dir "${exp_root}")" || [[ -z "${ckpt}" ]]; then
    echo "[WARN] no checkpoint: ${exp_root} (skip seed=${seed})"
    SKIP=$((SKIP + 1))
    continue
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY] would rm -rf ${exp2_dir} && run Exp2 seed=${seed} ckpt=${ckpt}"
    RUN=$((RUN + 1))
    continue
  fi

  echo "[RUN] Exp2 seed=${seed} ds=${DATASET} (remove ${exp2_dir})"
  rm -rf "${exp2_dir}"
  mkdir -p "${exp2_dir}"

  _rc=0
  "${PYTHON_BIN}" -u downstream_evaluate_psd_gate.py \
    --config-name=evaluate_psd_gate \
    mode.ddp=false mode.amp=false logger.log=false other.debug=false \
    "train.random_seed=${seed}" \
    "model.config_dir=${CONFIG_DIR}" \
    "model.regression.version=${MODEL_VERSION}" \
    "model.regression.pooling=${POOLING}" \
    model.regression.check_point.load=true \
    "model.regression.check_point.path=${ckpt}" \
    "data.regression.dataset=${DATASET}" \
    "data.diff=[${DIFF}]" \
    "data.threshold=${THRESHOLD}" \
    'data.regression.condition=["blosum62 average"]' \
    "data.regression.fix.train_file=${train_csv}" \
    "data.regression.fix.valid_file=${valid_csv}" \
    "data.regression.fix.test_file=${test_csv}" \
    "analysis.peptide_manifest=${manifest}" \
    'psd_gate_analysis.splits=[test]' \
    "hydra.run.dir=${exp2_dir}" \
    hydra.output_subdir=null || _rc=$?

  if (( _rc != 0 )); then
    echo "[FAIL] Exp2 seed=${seed} ds=${DATASET} (exit ${_rc})"
    FAIL=$((FAIL + 1))
    continue
  fi

  _png_count=0
  if [[ -d "${exp2_dir}" ]]; then
    _png_count="$(find "${exp2_dir}" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)"
  fi
  if [[ ! -f "${exp2_dir}/per_sample_gate_by_band.csv" ]] || (( _png_count < 3 )); then
    echo "[FAIL] Exp2 seed=${seed} incomplete outputs (png=${_png_count})"
    FAIL=$((FAIL + 1))
    continue
  fi

  echo "[OK] Exp2 seed=${seed} ds=${DATASET} (png=${_png_count})"
  RUN=$((RUN + 1))
done

echo ""
echo "Done. RUN=${RUN} SKIP=${SKIP} FAIL=${FAIL}"
exit $(( FAIL > 0 ? 1 : 0 ))

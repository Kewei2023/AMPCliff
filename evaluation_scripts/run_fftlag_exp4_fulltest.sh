#!/usr/bin/env bash
#SBATCH --job-name=fftlag_exp4_ft
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/fftlag_exp4_fulltest_%j.out
#SBATCH --error=logs/fftlag_exp4_fulltest_%j.err
#
# Exp4 full test set: latent-query cross-attention for ALL test peptides (no manifest).
# Saves latent_attn_weights.pt under exp4_latent_fulltest/ (does not overwrite exp4_latent/).
#
# Usage:
#   sbatch --gpus=1 evaluation_scripts/run_fftlag_exp4_fulltest.sh
#   SEEDS="0" DATASETS=s_aureus sbatch evaluation_scripts/run_fftlag_exp4_fulltest.sh
#   DRY_RUN=1 bash evaluation_scripts/run_fftlag_exp4_fulltest.sh
#   SKIP_IF_DONE=0 bash evaluation_scripts/run_fftlag_exp4_fulltest.sh  # force rerun
#
# After all seeds finish, run distribution plots (CPU):
#   python evaluation_scripts/plot_fftlag_exp4_fulltest_latent_query_dist.py

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
DRY_RUN="${DRY_RUN:-0}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${REPO_ROOT}/outputs/analysis/fftlag_mechanism}"
EXP4_SUBDIR="${EXP4_SUBDIR:-exp4_latent_fulltest}"

if [[ -d "${REPO_ROOT}/outputs/ablation_new_data" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_new_data}"
elif [[ -d "${REPO_ROOT}/outputs/ablation-new-data" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation-new-data}"
elif [[ -d "${REPO_ROOT}/outputs/ablation_layernorm" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_layernorm}"
else
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_new_data}"
fi
EXP_SUFFIX="${EXP_SUFFIX:-}"

resolve_exp_root() {
  local ds="$1"
  local seed="$2"
  local candidates=(
    "${ABLATION_ROOT}/${MODEL_VERSION}_${POOLING}_${ds}_diff${DIFF}${EXP_SUFFIX}/seed_${seed}"
    "${ABLATION_ROOT}/${MODEL_VERSION}_${POOLING}_${ds}_diff${DIFF}_layernorm/seed_${seed}"
    "${ABLATION_ROOT}/${MODEL_VERSION}_${POOLING}_${ds}_diff${DIFF}/seed_${seed}"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -d "${c}" ]] && find "${c}" -type f -name model.pth 2>/dev/null | head -1 | grep -q .; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

read -r -a DATASETS <<< "${DATASETS:-e_coli s_aureus}"
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
else
  SEED_LIST=(0 1 2 3 4 5 6 7 8 9)
fi

_BASE_CSV="${REPO_ROOT}/data/blosum62 average/diff_${DIFF}-trd_${THRESHOLD}"

config_dir_for_model() {
  case "${MODEL_VERSION}" in
    esm2_t6) echo "/data/public/models/facebook/esm2_t6_8M_UR50D/" ;;
    esm2_t12) echo "/data/public/models/facebook/esm2_t12_35M_UR50D/" ;;
    *) echo "/data/public/models/facebook/esm2_t6_8M_UR50D/" ;;
  esac
}

resolve_ckpt_dir() {
  local seed_root="$1"
  local mp
  mp="$(find "${seed_root}" -type f -name model.pth 2>/dev/null | head -1)"
  if [[ -z "${mp}" ]]; then
    return 1
  fi
  dirname "$(dirname "${mp}")"
}

expected_test_size() {
  local ds="$1"
  local test_csv="${_BASE_CSV}/grampa_${ds}_7_25-test.csv"
  if [[ ! -f "${test_csv}" ]]; then
    echo "0"
    return
  fi
  # subtract header line
  echo $(( $(wc -l < "${test_csv}") - 1 ))
}

CONFIG_DIR="$(config_dir_for_model)"
RUN=0
SKIP=0
FAIL=0

echo "========== FFT-LAG Exp4 full test set =========="
echo "REPO_ROOT=${REPO_ROOT}"
echo "ANALYSIS_ROOT=${ANALYSIS_ROOT}"
echo "ABLATION_ROOT=${ABLATION_ROOT}"
echo "MODEL=${MODEL_VERSION} POOLING=${POOLING}"
echo "SEEDS=${SEED_LIST[*]} DATASETS=${DATASETS[*]}"
echo "EXP4_SUBDIR=${EXP4_SUBDIR} SKIP_IF_DONE=${SKIP_IF_DONE}"
echo "================================================"

for ds in "${DATASETS[@]}"; do
  train_csv="${_BASE_CSV}/grampa_${ds}_7_25-train.csv"
  valid_csv="${_BASE_CSV}/grampa_${ds}_7_25-valid.csv"
  test_csv="${_BASE_CSV}/grampa_${ds}_7_25-test.csv"
  expected_n="$(expected_test_size "${ds}")"

  for seed in "${SEED_LIST[@]}"; do
    exp_root=""
    if ! exp_root="$(resolve_exp_root "${ds}" "${seed}")" || [[ -z "${exp_root}" ]]; then
      echo "[WARN] no checkpoint dir for seed=${seed} ds=${ds} under ${ABLATION_ROOT}"
      SKIP=$((SKIP + 1))
      continue
    fi

    ckpt=""
    if ! ckpt="$(resolve_ckpt_dir "${exp_root}")" || [[ -z "${ckpt}" ]]; then
      echo "[WARN] no checkpoint: ${exp_root} (skip seed=${seed} ds=${ds})"
      SKIP=$((SKIP + 1))
      continue
    fi

    seed_out="${ANALYSIS_ROOT}/seed_${seed}/${ds}"
    exp4_dir="${seed_out}/${EXP4_SUBDIR}"
    done_pt="${exp4_dir}/latent_attn_weights.pt"

    if [[ "${SKIP_IF_DONE}" == "1" && -f "${done_pt}" ]]; then
      n_saved="$("${PYTHON_BIN}" -c "
import torch
p=torch.load('${done_pt}', map_location='cpu')
print(len(p.get('samples') or []))
" 2>/dev/null || echo 0)"
      if [[ "${expected_n}" -gt 0 && "${n_saved}" -ge "${expected_n}" ]]; then
        echo "[SKIP] Exp4 fulltest seed=${seed} ds=${ds} (${n_saved}/${expected_n} samples)"
        SKIP=$((SKIP + 1))
        continue
      fi
      echo "[RERUN] Exp4 fulltest seed=${seed} ds=${ds} (${n_saved}/${expected_n} samples incomplete)"
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY] Exp4 fulltest seed=${seed} ds=${ds} ckpt=${ckpt} -> ${exp4_dir}"
      RUN=$((RUN + 1))
      continue
    fi

    mkdir -p "${exp4_dir}"
    if "${PYTHON_BIN}" -u downstream_evaluate_fft_lag_latent.py \
      --config-name=evaluate_fftlag_mechanism \
      mode.ddp=false mode.amp=false logger.log=false other.debug=false \
      "train.random_seed=${seed}" \
      "model.config_dir=${CONFIG_DIR}" \
      "model.regression.version=${MODEL_VERSION}" \
      "model.regression.pooling=${POOLING}" \
      model.regression.check_point.load=true \
      "model.regression.check_point.path=${ckpt}" \
      "data.regression.dataset=${ds}" \
      "data.diff=[${DIFF}]" \
      "data.threshold=${THRESHOLD}" \
      'data.regression.condition=["blosum62 average"]' \
      "data.regression.fix.train_file=${train_csv}" \
      "data.regression.fix.valid_file=${valid_csv}" \
      "data.regression.fix.test_file=${test_csv}" \
      analysis.peptide_manifest="" \
      'latent_analysis.splits=[test]' \
      'latent_analysis.primary_plots=[]' \
      "hydra.run.dir=${exp4_dir}" \
      hydra.output_subdir=null; then
      n_saved="$("${PYTHON_BIN}" -c "
import torch
p=torch.load('${done_pt}', map_location='cpu')
print(len(p.get('samples') or []))
" 2>/dev/null || echo 0)"
      echo "[OK] Exp4 fulltest seed=${seed} ds=${ds} (${n_saved}/${expected_n} samples)"
      RUN=$((RUN + 1))
    else
      echo "[FAIL] Exp4 fulltest seed=${seed} ds=${ds}"
      FAIL=$((FAIL + 1))
    fi
  done
done

echo ""
echo "Done. RUN=${RUN} SKIP=${SKIP} FAIL=${FAIL}"
echo "Next: python evaluation_scripts/plot_fftlag_exp4_fulltest_latent_query_dist.py"
exit $(( FAIL > 0 ? 1 : 0 ))

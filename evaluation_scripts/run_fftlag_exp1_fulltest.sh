#!/usr/bin/env bash
# Exp1 full test set: band knockout for ALL test peptides (no manifest).
# Saves under exp1_band_knockout_fulltest/ (does not overwrite exp1_band_knockout/).
#
# Usage:
#   bash evaluation_scripts/run_fftlag_exp1_fulltest.sh
#   SEEDS="0" DATASETS=s_aureus bash evaluation_scripts/run_fftlag_exp1_fulltest.sh
#   DRY_RUN=1 bash evaluation_scripts/run_fftlag_exp1_fulltest.sh
#   SKIP_IF_DONE=0 bash evaluation_scripts/run_fftlag_exp1_fulltest.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"
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
POOLING="${POOLING:-fft_latent_attn_gate}"
APPLY="${APPLY:-none}"
DIFF="${DIFF:-5}"
THRESHOLD="${THRESHOLD:-0.9}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${REPO_ROOT}/outputs/analysis/fftlag_mechanism}"
EXP1_SUBDIR="${EXP1_SUBDIR:-exp1_band_knockout_fulltest}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/outputs/ablation_new_data/_amp_knockout_seed_runs/_peptide_manifest}"

if [[ -d "${REPO_ROOT}/outputs/ablation_new_data" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_new_data}"
elif [[ -d "${REPO_ROOT}/outputs/ablation-new-data" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation-new-data}"
else
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_new_data}"
fi

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
  echo $(( $(wc -l < "${test_csv}") - 1 ))
}

count_unique_idx() {
  local csv_path="$1"
  if [[ ! -f "${csv_path}" ]]; then
    echo "0"
    return
  fi
  "${PYTHON_BIN}" -c "
import pandas as pd
df = pd.read_csv('${csv_path}')
print(df['idx'].nunique() if 'idx' in df.columns else 0)
" 2>/dev/null || echo "0"
}

CONFIG_DIR="$(config_dir_for_model)"
RUN=0
SKIP=0
FAIL=0

echo "========== FFT-LAG Exp1 full test set =========="
echo "REPO_ROOT=${REPO_ROOT}"
echo "ANALYSIS_ROOT=${ANALYSIS_ROOT}"
echo "ABLATION_ROOT=${ABLATION_ROOT}"
echo "MODEL=${MODEL_VERSION} POOLING=${POOLING}"
echo "SEEDS=${SEED_LIST[*]} DATASETS=${DATASETS[*]}"
echo "EXP1_SUBDIR=${EXP1_SUBDIR} SKIP_IF_DONE=${SKIP_IF_DONE}"
echo "================================================"

# Optional: write fulltest manifest JSON for traceability (Exp1 does not use it).
for ds in "${DATASETS[@]}"; do
  manifest="${MANIFEST_DIR}/${MODEL_VERSION}_${ds}_diff${DIFF}_fulltest.json"
  if [[ ! -f "${manifest}" && "${DRY_RUN}" != "1" ]]; then
    echo "[RUN] fulltest manifest ${ds}"
    "${PYTHON_BIN}" "${REPO_ROOT}/evaluation_scripts/select_knockout_peptide_subset.py" \
      --model-version "${MODEL_VERSION}" \
      --dataset "${ds}" \
      --diff "${DIFF}" \
      --threshold "${THRESHOLD}" \
      --all-test \
      --out-dir "${MANIFEST_DIR}" || true
  fi
done

for ds in "${DATASETS[@]}"; do
  train_csv="${_BASE_CSV}/grampa_${ds}_7_25-train.csv"
  valid_csv="${_BASE_CSV}/grampa_${ds}_7_25-valid.csv"
  test_csv="${_BASE_CSV}/grampa_${ds}_7_25-test.csv"
  expected_n="$(expected_test_size "${ds}")"

  for seed in "${SEED_LIST[@]}"; do
    exp_root="${ABLATION_ROOT}/${MODEL_VERSION}_${POOLING}_${ds}_diff${DIFF}/seed_${seed}"
    if [[ ! -d "${exp_root}" ]]; then
      echo "[WARN] no checkpoint dir for seed=${seed} ds=${ds}: ${exp_root}"
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
    exp1_dir="${seed_out}/${EXP1_SUBDIR}"
    done_csv="${exp1_dir}/per_sample_band_sensitivity.csv"

    if [[ "${SKIP_IF_DONE}" == "1" && -f "${done_csv}" ]]; then
      n_saved="$(count_unique_idx "${done_csv}")"
      if [[ "${expected_n}" -gt 0 && "${n_saved}" -ge "${expected_n}" ]]; then
        echo "[SKIP] Exp1 fulltest seed=${seed} ds=${ds} (${n_saved}/${expected_n} idx)"
        SKIP=$((SKIP + 1))
        continue
      fi
      echo "[RERUN] Exp1 fulltest seed=${seed} ds=${ds} (${n_saved}/${expected_n} idx incomplete)"
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY] Exp1 fulltest seed=${seed} ds=${ds} ckpt=${ckpt} -> ${exp1_dir}"
      RUN=$((RUN + 1))
      continue
    fi

    mkdir -p "${exp1_dir}"
    if "${PYTHON_BIN}" -u downstream_evaluate_spectrual_filter.py \
      --config-name=evaluate_fftlag_mechanism \
      mode.ddp=false mode.amp=false logger.log=false other.debug=false \
      "train.random_seed=${seed}" \
      "model.config_dir=${CONFIG_DIR}" \
      "model.regression.version=${MODEL_VERSION}" \
      "model.regression.pooling=${POOLING}" \
      "model.regression.apply=${APPLY}" \
      model.regression.check_point.load=true \
      "model.regression.check_point.path=${ckpt}" \
      "model.regression.spectrual_filter.dim=seq_len" \
      'model.regression.spectrual_filter.split_for_plot=[test]' \
      "data.regression.dataset=${ds}" \
      "data.diff=[${DIFF}]" \
      "data.threshold=${THRESHOLD}" \
      'data.regression.condition=["blosum62 average"]' \
      "data.regression.fix.train_file=${train_csv}" \
      "data.regression.fix.valid_file=${valid_csv}" \
      "data.regression.fix.test_file=${test_csv}" \
      analysis.peptide_manifest="" \
      analysis.shuffle_dataloader=false \
      "hydra.run.dir=${exp1_dir}" \
      hydra.output_subdir=null; then
      n_saved="$(count_unique_idx "${done_csv}")"
      echo "[OK] Exp1 fulltest seed=${seed} ds=${ds} (${n_saved}/${expected_n} idx)"
      RUN=$((RUN + 1))
    else
      echo "[FAIL] Exp1 fulltest seed=${seed} ds=${ds}"
      FAIL=$((FAIL + 1))
    fi
  done
done

echo ""
echo "Done. RUN=${RUN} SKIP=${SKIP} FAIL=${FAIL}"
echo "Next: bash evaluation_scripts/run_dc_property_knockout_fulltest.sh"
exit $(( FAIL > 0 ? 1 : 0 ))

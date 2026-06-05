#!/usr/bin/env bash
# FFT-LAG mechanism experiments (Exp1/2/4/5) on knockout peptide manifest subset.
#
# Usage:
#   bash evaluation_scripts/run_fftlag_mechanism_experiments.sh
#   RUN_EXPS=4 bash evaluation_scripts/run_fftlag_mechanism_experiments.sh   # Exp4 only
#   DRY_RUN=1 bash evaluation_scripts/run_fftlag_mechanism_experiments.sh
#   SEEDS="0" DATASETS=s_aureus bash evaluation_scripts/run_fftlag_mechanism_experiments.sh
#   sbatch evaluation_scripts/run_fftlag_exp4_only.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"
cd "${REPO_ROOT}"

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
CONDITION="${CONDITION:-blosum62 average}"
SUBSET_SEED="${SUBSET_SEED:-42}"
N_PEPTIDES="${N_PEPTIDES:-30}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
# Comma-separated experiment ids to run: 1=band knockout, 2=gate PSD, 4=latent viz, 5=structure (after aggregate)
RUN_EXPS="${RUN_EXPS:-1,2,4}"
RUN_EXP5="${RUN_EXP5:-1}"

_should_run_exp() {
  local id="$1"
  [[ ",${RUN_EXPS}," == *",${id},"* ]]
}

read -r -a DATASETS <<< "${DATASETS:-e_coli s_aureus}"
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
else
  SEED_LIST=(0 1 2 3 4 5 6 7 8 9)
fi

ANALYSIS_ROOT="${ANALYSIS_ROOT:-${REPO_ROOT}/outputs/analysis/fftlag_mechanism}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/outputs/ablation_new_data/_amp_knockout_seed_runs/_peptide_manifest}"

if [[ -d "${REPO_ROOT}/outputs/ablation_new_data" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_new_data}"
elif [[ -d "${REPO_ROOT}/outputs/ablation-new-data" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation-new-data}"
else
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_new_data}"
fi

_BASE_CSV="${REPO_ROOT}/data/blosum62 average/diff_${DIFF}-trd_${THRESHOLD}"

resolve_ckpt_dir() {
  local seed_root="$1"
  local mp
  mp="$(find "${seed_root}" -type f -name model.pth 2>/dev/null | head -1)"
  if [[ -z "${mp}" ]]; then
    return 1
  fi
  dirname "$(dirname "${mp}")"
}

ensure_manifest() {
  local ds="$1"
  local manifest="${MANIFEST_DIR}/${MODEL_VERSION}_${ds}_diff${DIFF}.json"
  if [[ -f "${manifest}" ]]; then
    echo "${manifest}"
    return 0
  fi
  echo "Generating manifest: ${manifest}" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${manifest}"
    return 0
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation_scripts/select_knockout_peptide_subset.py" \
    --model-version "${MODEL_VERSION}" \
    --dataset "${ds}" \
    --diff "${DIFF}" \
    --subset-seed "${SUBSET_SEED}" \
    --n-peptides "${N_PEPTIDES}" \
    --out-dir "${MANIFEST_DIR}"
  echo "${manifest}"
}

config_dir_for_model() {
  case "${MODEL_VERSION}" in
    esm2_t6) echo "/data/public/models/facebook/esm2_t6_8M_UR50D/" ;;
    esm2_t12) echo "/data/public/models/facebook/esm2_t12_35M_UR50D/" ;;
    *) echo "/data/public/models/facebook/esm2_t6_8M_UR50D/" ;;
  esac
}

RUN=0
SKIP=0
FAIL=0

echo "========== FFT-LAG Mechanism Experiments =========="
echo "REPO_ROOT=${REPO_ROOT}"
echo "ANALYSIS_ROOT=${ANALYSIS_ROOT}"
echo "ABLATION_ROOT=${ABLATION_ROOT}"
echo "MODEL=${MODEL_VERSION} POOLING=${POOLING} SEEDS=${SEED_LIST[*]}"
echo "DATASETS=${DATASETS[*]} SUBSET_SEED=${SUBSET_SEED} N_PEPTIDES=${N_PEPTIDES}"
echo "RUN_EXPS=${RUN_EXPS} RUN_EXP5=${RUN_EXP5}"
echo "==================================================="

CONFIG_DIR="$(config_dir_for_model)"

for ds in "${DATASETS[@]}"; do
  manifest="$(ensure_manifest "${ds}")"
  train_csv="${_BASE_CSV}/grampa_${ds}_7_25-train.csv"
  valid_csv="${_BASE_CSV}/grampa_${ds}_7_25-valid.csv"
  test_csv="${_BASE_CSV}/grampa_${ds}_7_25-test.csv"

  for seed in "${SEED_LIST[@]}"; do
    exp_root="${ABLATION_ROOT}/${MODEL_VERSION}_${POOLING}_${ds}_diff${DIFF}/seed_${seed}"
    seed_out="${ANALYSIS_ROOT}/seed_${seed}/${ds}"
    ckpt=""
    if ! ckpt="$(resolve_ckpt_dir "${exp_root}")" || [[ -z "${ckpt}" ]]; then
      echo "[WARN] no checkpoint: ${exp_root} (skip seed=${seed} ds=${ds})"
      SKIP=$((SKIP + 1))
      continue
    fi

    # ---- Exp1: seq_len band knockout ----
    if ! _should_run_exp 1; then
      :
    else
    exp1_dir="${seed_out}/exp1_band_knockout"
    exp1_done="${exp1_dir}/per_sample_band_sensitivity.csv"
    if [[ "${SKIP_IF_DONE}" == "1" && -f "${exp1_done}" ]]; then
      echo "[SKIP] Exp1 seed=${seed} ds=${ds}"
      SKIP=$((SKIP + 1))
    elif [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY] Exp1 seed=${seed} ds=${ds} ckpt=${ckpt}"
      RUN=$((RUN + 1))
    else
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
        "analysis.peptide_manifest=${manifest}" \
        "hydra.run.dir=${exp1_dir}" \
        hydra.output_subdir=null; then
        echo "[OK] Exp1 seed=${seed} ds=${ds}"
        RUN=$((RUN + 1))
      else
        echo "[FAIL] Exp1 seed=${seed} ds=${ds}"
        FAIL=$((FAIL + 1))
      fi
    fi
    fi

    # ---- Exp2: gate PSD ----
    if ! _should_run_exp 2; then
      :
    else
    exp2_dir="${seed_out}/exp2_psd_gate"
    exp2_done="${exp2_dir}/per_sample_gate_by_band.csv"
    if [[ "${SKIP_IF_DONE}" == "1" && -f "${exp2_done}" ]]; then
      echo "[SKIP] Exp2 seed=${seed} ds=${ds}"
      SKIP=$((SKIP + 1))
    elif [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY] Exp2 seed=${seed} ds=${ds} ckpt=${ckpt}"
      RUN=$((RUN + 1))
    else
      mkdir -p "${exp2_dir}"
      if "${PYTHON_BIN}" -u downstream_evaluate_psd_gate.py \
        --config-name=evaluate_psd_gate \
        mode.ddp=false mode.amp=false logger.log=false other.debug=false \
        "train.random_seed=${seed}" \
        "model.config_dir=${CONFIG_DIR}" \
        "model.regression.version=${MODEL_VERSION}" \
        "model.regression.pooling=${POOLING}" \
        "model.regression.apply=${APPLY}" \
        model.regression.check_point.load=true \
        "model.regression.check_point.path=${ckpt}" \
        "data.regression.dataset=${ds}" \
        "data.diff=[${DIFF}]" \
        "data.threshold=${THRESHOLD}" \
        'data.regression.condition=["blosum62 average"]' \
        "data.regression.fix.train_file=${train_csv}" \
        "data.regression.fix.valid_file=${valid_csv}" \
        "data.regression.fix.test_file=${test_csv}" \
        "analysis.peptide_manifest=${manifest}" \
        'psd_gate_analysis.splits=[test]' \
        "hydra.run.dir=${exp2_dir}" \
        hydra.output_subdir=null; then
        echo "[OK] Exp2 seed=${seed} ds=${ds}"
        RUN=$((RUN + 1))
      else
        echo "[FAIL] Exp2 seed=${seed} ds=${ds}"
        FAIL=$((FAIL + 1))
      fi
    fi
    fi

    # ---- Exp4: latent query ----
    if ! _should_run_exp 4; then
      :
    else
    exp4_dir="${seed_out}/exp4_latent"
    exp4_done="${exp4_dir}/latent_query_band_mass.csv"
    if [[ "${SKIP_IF_DONE}" == "1" && -f "${exp4_done}" ]]; then
      echo "[SKIP] Exp4 seed=${seed} ds=${ds}"
      SKIP=$((SKIP + 1))
    elif [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY] Exp4 seed=${seed} ds=${ds} ckpt=${ckpt}"
      RUN=$((RUN + 1))
    else
      mkdir -p "${exp4_dir}"
      if "${PYTHON_BIN}" -u downstream_evaluate_fft_lag_latent.py \
        --config-name=evaluate_fftlag_mechanism \
        mode.ddp=false mode.amp=false logger.log=false other.debug=false \
        "train.random_seed=${seed}" \
        "model.config_dir=${CONFIG_DIR}" \
        "model.regression.version=${MODEL_VERSION}" \
        "model.regression.pooling=${POOLING}" \
        "model.regression.apply=${APPLY}" \
        model.regression.check_point.load=true \
        "model.regression.check_point.path=${ckpt}" \
        "data.regression.dataset=${ds}" \
        "data.diff=[${DIFF}]" \
        "data.threshold=${THRESHOLD}" \
        'data.regression.condition=["blosum62 average"]' \
        "data.regression.fix.train_file=${train_csv}" \
        "data.regression.fix.valid_file=${valid_csv}" \
        "data.regression.fix.test_file=${test_csv}" \
        "analysis.peptide_manifest=${manifest}" \
        'latent_analysis.splits=[test]' \
        "hydra.run.dir=${exp4_dir}" \
        hydra.output_subdir=null; then
        echo "[OK] Exp4 seed=${seed} ds=${ds}"
        RUN=$((RUN + 1))
      else
        echo "[FAIL] Exp4 seed=${seed} ds=${ds}"
        FAIL=$((FAIL + 1))
      fi
    fi
    fi
  done

  # ---- Aggregate across seeds ----
  if [[ "${DRY_RUN}" != "1" ]] && { _should_run_exp 1 || _should_run_exp 2 || _should_run_exp 4; }; then
    "${PYTHON_BIN}" "${REPO_ROOT}/evaluation_scripts/aggregate_fftlag_mechanism_seeds.py" \
      --analysis-root "${ANALYSIS_ROOT}" \
      --dataset "${ds}" || true
    "${PYTHON_BIN}" "${REPO_ROOT}/evaluation_scripts/plot_fftlag_mechanism_per_sample_seeds.py" \
      --analysis-root "${ANALYSIS_ROOT}" \
      --dataset "${ds}" || true
  fi

  if [[ "${DRY_RUN}" != "1" && "${RUN_EXP5}" == "1" ]] && _should_run_exp 5; then
    agg_dir="${ANALYSIS_ROOT}/aggregated/${ds}"
    exp5_dir="${ANALYSIS_ROOT}/exp5_structure/${ds}"
    mkdir -p "${exp5_dir}"
    "${PYTHON_BIN}" "${REPO_ROOT}/analyze_fft_lag_mechanism_by_structure.py" \
      --manifest "${manifest}" \
      --aggregated-dir "${agg_dir}" \
      --output-dir "${exp5_dir}" || true
  fi
done

echo ""
echo "Done. RUN=${RUN} SKIP=${SKIP} FAIL=${FAIL}"
echo "Outputs: ${ANALYSIS_ROOT}"

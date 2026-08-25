#!/usr/bin/env bash
# Exp3 full test set: token-position HS knockout for ALL test peptides (no manifest).
# Saves under exp3_fulltest/{pooling}/seed_{k}/{dataset}/ (does not overwrite other experiments).
#
# Usage:
#   bash evaluation_scripts/run_fftlag_exp3_fulltest.sh
#   POOLINGS=fft_latent_attn_gate SEEDS="0" DATASETS=s_aureus bash evaluation_scripts/run_fftlag_exp3_fulltest.sh
#   DRY_RUN=1 bash evaluation_scripts/run_fftlag_exp3_fulltest.sh
#   SKIP_IF_DONE=0 bash evaluation_scripts/run_fftlag_exp3_fulltest.sh
#   sbatch evaluation_scripts/run_fftlag_exp3_fulltest_slurm.sh
#
# After runs finish (or incrementally as poolings are added):
#   python evaluation_scripts/aggregate_exp3_token_knockout_mse_diff.py --force
#   python evaluation_scripts/plot_fftlag_exp3_mse_diff_violin_revised.py --force
#   python evaluation_scripts/export_fftlag_exp3_token_knockout_data.py

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
DIFF="${DIFF:-5}"
THRESHOLD="${THRESHOLD:-0.9}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${REPO_ROOT}/outputs/analysis/fftlag_mechanism}"
EXP3_ROOT="${EXP3_ROOT:-${ANALYSIS_ROOT}/exp3_fulltest}"

if [[ -d "${REPO_ROOT}/outputs/ablation_new_data" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_new_data}"
elif [[ -d "${REPO_ROOT}/outputs/ablation-new-data" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation-new-data}"
else
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_new_data}"
fi

read -r -a POOLINGS <<< "${POOLINGS:-fft_latent_attn_gate}"
read -r -a DATASETS <<< "${DATASETS:-e_coli s_aureus}"
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
else
  SEED_LIST=(0 1 2 3 4 5 6 7 8 9)
fi

_BASE_CSV="${REPO_ROOT}/data/blosum62 average/diff_${DIFF}-trd_${THRESHOLD}"

config_dir_for_model() {
  local cluster_dir local_dir
  case "${MODEL_VERSION}" in
    esm2_t6)
      cluster_dir="/data/public/models/facebook/esm2_t6_8M_UR50D/"
      local_dir="${REPO_ROOT}/models/facebook/esm2_t6_8M_UR50D"
      ;;
    esm2_t12)
      cluster_dir="/data/public/models/facebook/esm2_t12_35M_UR50D/"
      local_dir="${REPO_ROOT}/models/facebook/esm2_t12_35M_UR50D"
      ;;
    *)
      cluster_dir="/data/public/models/facebook/esm2_t6_8M_UR50D/"
      local_dir="${REPO_ROOT}/models/facebook/esm2_t6_8M_UR50D"
      ;;
  esac
  # Prefer explicit override, then cluster shared weights, then local/WSL checkout.
  if [[ -n "${CONFIG_DIR:-}" ]]; then
    echo "${CONFIG_DIR}"
  elif [[ -d "${cluster_dir}" ]]; then
    echo "${cluster_dir}"
  else
    echo "${local_dir}"
  fi
}

revised_pooling_output_root() {
  local pool="$1"
  case "${pool}" in
    mltp_paper)
      echo "${REVISED_POOLING_ROOT:-${REPO_ROOT}/outputs/mltp_paper_${MODEL_VERSION}}"
      ;;
    attn_structured)
      echo "${REVISED_POOLING_ROOT:-${REPO_ROOT}/outputs/attn_structured_${MODEL_VERSION}}"
      ;;
    *)
      echo ""
      ;;
  esac
}

resolve_exp_root() {
  local pool="$1"
  local ds="$2"
  local seed="$3"
  local candidates=(
    "${ABLATION_ROOT}/${MODEL_VERSION}_${pool}_${ds}_diff${DIFF}/seed_${seed}"
    "${ABLATION_ROOT}/${MODEL_VERSION}_${pool}_${ds}_diff${DIFF}_layernorm/seed_${seed}"
  )
  local revised_root
  revised_root="$(revised_pooling_output_root "${pool}")"
  if [[ -n "${revised_root}" ]]; then
    candidates+=(
      "${revised_root}/${MODEL_VERSION}_${pool}_${ds}_diff${DIFF}/seed_${seed}"
    )
  fi
  local c
  for c in "${candidates[@]}"; do
    if [[ -d "${c}" ]] && find "${c}" -type f -name model.pth 2>/dev/null | head -1 | grep -q .; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

resolve_ckpt_dir() {
  local seed_root="$1"
  local mp
  # Prefer largest model.pth when multiple checkpoints exist (e.g. last pooling).
  mp="$(find "${seed_root}" -type f -name model.pth -printf '%s %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "${mp}" ]]; then
    return 1
  fi
  dirname "$(dirname "${mp}")"
}

extra_knockout_args_for_pool() {
  local pool="$1"
  case "${pool}" in
    attn_structured)
      # downstream_knockout.yaml lacks pooling_config (struct mode); append with + prefix.
      echo \
        +model.regression.pooling_config.attn_structured.attention_size=350 \
        +model.regression.pooling_config.attn_structured.attention_hops=30 \
        +model.regression.pooling_config.attn_structured.attention_dropout=0.5 \
        +model.regression.pooling_config.attn_structured.penalization_coeff=1.0 \
        +model.regression.pooling_config.attn_structured.use_bias=false \
        +model.regression.pooling_config.attn_structured.hop_output=flatten
      ;;
    *)
      ;;
  esac
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

echo "========== Exp3 token knockout full test set =========="
echo "REPO_ROOT=${REPO_ROOT}"
echo "ANALYSIS_ROOT=${ANALYSIS_ROOT}"
echo "EXP3_ROOT=${EXP3_ROOT}"
echo "ABLATION_ROOT=${ABLATION_ROOT}"
echo "MODEL=${MODEL_VERSION} POOLINGS=${POOLINGS[*]}"
echo "SEEDS=${SEED_LIST[*]} DATASETS=${DATASETS[*]}"
echo "SKIP_IF_DONE=${SKIP_IF_DONE} DRY_RUN=${DRY_RUN}"
echo "======================================================="

mkdir -p "${EXP3_ROOT}"

for pool in "${POOLINGS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    train_csv="${_BASE_CSV}/grampa_${ds}_7_25-train.csv"
    valid_csv="${_BASE_CSV}/grampa_${ds}_7_25-valid.csv"
    test_csv="${_BASE_CSV}/grampa_${ds}_7_25-test.csv"
    expected_n="$(expected_test_size "${ds}")"

    for seed in "${SEED_LIST[@]}"; do
      exp_root=""
      if ! exp_root="$(resolve_exp_root "${pool}" "${ds}" "${seed}")" || [[ -z "${exp_root}" ]]; then
        echo "[WARN] no checkpoint: pool=${pool} seed=${seed} ds=${ds}"
        SKIP=$((SKIP + 1))
        continue
      fi

      ckpt=""
      if ! ckpt="$(resolve_ckpt_dir "${exp_root}")" || [[ -z "${ckpt}" ]]; then
        echo "[WARN] no model.pth: ${exp_root}"
        SKIP=$((SKIP + 1))
        continue
      fi

      exp3_dir="${EXP3_ROOT}/${pool}/seed_${seed}/${ds}"
      done_csv="${exp3_dir}/knockout_lastlayer_HS.csv"

      if [[ "${SKIP_IF_DONE}" == "1" && -f "${done_csv}" ]]; then
        n_saved="$(count_unique_idx "${done_csv}")"
        if [[ "${expected_n}" -gt 0 && "${n_saved}" -ge "${expected_n}" ]]; then
          echo "[SKIP] Exp3 pool=${pool} seed=${seed} ds=${ds} (${n_saved}/${expected_n} idx)"
          SKIP=$((SKIP + 1))
          continue
        fi
        echo "[RERUN] Exp3 pool=${pool} seed=${seed} ds=${ds} (${n_saved}/${expected_n} idx incomplete)"
      fi

      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY] Exp3 pool=${pool} seed=${seed} ds=${ds} ckpt=${ckpt} -> ${exp3_dir}"
        RUN=$((RUN + 1))
        continue
      fi

      mkdir -p "${exp3_dir}"
      echo "==== Exp3 pool=${pool} seed=${seed} ds=${ds} -> ${exp3_dir} ===="
      read -r -a POOL_EXTRA_ARGS <<< "$(extra_knockout_args_for_pool "${pool}")"
      if "${PYTHON_BIN}" -u "${REPO_ROOT}/downstream_evaluate_knockout.py" \
        knockout.enabled=true \
        knockout.mode=HS \
        knockout.split=test \
        knockout.last_layer_only=true \
        knockout.save_per_peptide_heatmap=false \
        knockout.peptide_manifest="" \
        "knockout.output_dir=${exp3_dir}" \
        mode.ddp=false \
        mode.amp=false \
        logger.log=false \
        other.debug=false \
        "train.random_seed=${seed}" \
        "model.config_dir=${CONFIG_DIR}" \
        "model.regression.version=${MODEL_VERSION}" \
        "model.regression.pooling=${pool}" \
        model.regression.check_point.load=true \
        "model.regression.check_point.path=${ckpt}" \
        "data.regression.dataset=${ds}" \
        "data.diff=[${DIFF}]" \
        "data.threshold=${THRESHOLD}" \
        'data.regression.condition=["blosum62 average"]' \
        "data.regression.fix.train_file=${train_csv}" \
        "data.regression.fix.valid_file=${valid_csv}" \
        "data.regression.fix.test_file=${test_csv}" \
        "hydra.run.dir=${exp3_dir}" \
        hydra.output_subdir=null \
        "${POOL_EXTRA_ARGS[@]}"; then
        n_saved="$(count_unique_idx "${done_csv}")"
        echo "[OK] Exp3 pool=${pool} seed=${seed} ds=${ds} (${n_saved}/${expected_n} idx)"
        RUN=$((RUN + 1))
      else
        echo "[FAIL] Exp3 pool=${pool} seed=${seed} ds=${ds}"
        FAIL=$((FAIL + 1))
      fi
    done
  done
done

echo ""
echo "Done. RUN=${RUN} SKIP=${SKIP} FAIL=${FAIL}"
echo "Next:"
echo "  python evaluation_scripts/aggregate_exp3_token_knockout_mse_diff.py --force"
echo "  python evaluation_scripts/plot_fftlag_exp3_mse_diff_violin_revised.py --force"
echo "  python evaluation_scripts/export_fftlag_exp3_token_knockout_data.py"
exit $(( FAIL > 0 ? 1 : 0 ))

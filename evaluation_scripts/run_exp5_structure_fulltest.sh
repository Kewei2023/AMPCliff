#!/usr/bin/env bash
# Aggregate fulltest Exp1/2/4 results and run Exp5 structure-bucket analysis.
#
# Prerequisites: all 10 seeds × 2 datasets finished Exp1/2/4 fulltest.
#
# Usage:
#   bash evaluation_scripts/run_exp5_structure_fulltest.sh
#   REQUIRE_UPSTREAM=0 bash evaluation_scripts/run_exp5_structure_fulltest.sh
#   ANALYZE_ONLY=1 bash evaluation_scripts/run_exp5_structure_fulltest.sh  # skip aggregate

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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
DIFF="${DIFF:-5}"
THRESHOLD="${THRESHOLD:-0.9}"
MODEL_VERSION="${MODEL_VERSION:-esm2_t6}"
REQUIRE_UPSTREAM="${REQUIRE_UPSTREAM:-1}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
read -r -a SEED_LIST <<< "${SEEDS}"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-${REPO_ROOT}/outputs/analysis/fftlag_mechanism}"
AGGREGATED_SUBDIR="${AGGREGATED_SUBDIR:-aggregated_fulltest}"
EXP1_SUBDIR="${EXP1_SUBDIR:-exp1_band_knockout_fulltest}"
EXP2_SUBDIR="${EXP2_SUBDIR:-exp2_psd_gate_fulltest}"
EXP4_SUBDIR="${EXP4_SUBDIR:-exp4_latent_fulltest}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ANALYSIS_ROOT}/exp5_structure_fulltest}"
FIGURES_ROOT="${FIGURES_ROOT:-${ANALYSIS_ROOT}/figures/exp5}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/outputs/ablation_new_data/_amp_knockout_seed_runs/_peptide_manifest}"

_BASE_CSV="${REPO_ROOT}/data/blosum62 average/diff_${DIFF}-trd_${THRESHOLD}"

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

check_upstream_ready() {
  local missing=0
  for ds in e_coli s_aureus; do
    local expected_n
    expected_n="$(expected_test_size "${ds}")"

    local agg_exp1="${ANALYSIS_ROOT}/${AGGREGATED_SUBDIR}/${ds}/exp1/per_sample_band_sensitivity_aggregated.csv"
    local n_agg
    n_agg="$(count_unique_idx "${agg_exp1}")"
    if [[ "${expected_n}" -gt 0 && "${n_agg}" -lt "${expected_n}" ]]; then
      echo "[NOT READY] Exp1 aggregated ${ds}: ${n_agg}/${expected_n} idx (${agg_exp1})"
      missing=$((missing + 1))
    fi

    for seed in "${SEED_LIST[@]}"; do
      local exp2_csv="${ANALYSIS_ROOT}/seed_${seed}/${ds}/${EXP2_SUBDIR}/per_sample_gate_by_band.csv"
      local n_exp2
      n_exp2="$(count_unique_idx "${exp2_csv}")"
      if [[ "${expected_n}" -gt 0 && "${n_exp2}" -lt "${expected_n}" ]]; then
        echo "[NOT READY] Exp2 seed=${seed} ds=${ds}: ${n_exp2}/${expected_n} idx (${exp2_csv})"
        missing=$((missing + 1))
      fi

      local exp4_csv="${ANALYSIS_ROOT}/seed_${seed}/${ds}/${EXP4_SUBDIR}/latent_query_diversity.csv"
      local n_exp4
      n_exp4="$(count_unique_idx "${exp4_csv}")"
      if [[ "${expected_n}" -gt 0 && "${n_exp4}" -lt "${expected_n}" ]]; then
        echo "[NOT READY] Exp4 seed=${seed} ds=${ds}: ${n_exp4}/${expected_n} idx (${exp4_csv})"
        missing=$((missing + 1))
      fi
    done
  done
  return "${missing}"
}

verify_exp5_output() {
  local ds="$1"
  local expected_n
  expected_n="$(expected_test_size "${ds}")"
  local summary_csv="${OUTPUT_ROOT}/${ds}/bucketwise_band_sensitivity_summary.csv"
  if [[ ! -f "${summary_csv}" ]]; then
    echo "[VERIFY FAIL] missing ${summary_csv}"
    return 1
  fi
  local total_n
  total_n="$("${PYTHON_BIN}" -c "
import pandas as pd
df = pd.read_csv('${summary_csv}')
print(int(df['n_samples'].sum()) if 'n_samples' in df.columns else 0)
" 2>/dev/null || echo 0)"
  if [[ "${expected_n}" -gt 0 && "${total_n}" -ne "${expected_n}" ]]; then
    echo "[VERIFY FAIL] ${ds} bucket n_samples sum=${total_n}, expected=${expected_n}"
    return 1
  fi
  local fig_ok=1
  local fig_dir="${FIGURES_ROOT}/${ds}"
  for fig in bucketwise_band_sensitivity_combined.png bucketwise_gate_effect_combined.png bucketwise_latent_diversity.png; do
    if [[ ! -f "${fig_dir}/${fig}" ]]; then
      echo "[VERIFY FAIL] missing figure ${fig_dir}/${fig}"
      fig_ok=0
    fi
  done
  if [[ "${fig_ok}" -eq 0 ]]; then
    return 1
  fi
  echo "[VERIFY OK] ${ds}: n_samples=${total_n}, key figures present under ${fig_dir}"
  return 0
}

RUN=0
SKIP=0
FAIL=0

echo "========== Exp5 Structure Fulltest =========="
echo "ANALYSIS_ROOT=${ANALYSIS_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "FIGURES_ROOT=${FIGURES_ROOT}"
echo "AGGREGATED_SUBDIR=${AGGREGATED_SUBDIR}"
echo "ANALYZE_ONLY=${ANALYZE_ONLY} REQUIRE_UPSTREAM=${REQUIRE_UPSTREAM}"
echo "============================================="

if [[ "${REQUIRE_UPSTREAM}" == "1" ]]; then
  if ! check_upstream_ready; then
    echo ""
    echo "Upstream fulltest incomplete. Submit Exp2 GPU job first if needed:"
    echo "  sbatch evaluation_scripts/run_fftlag_exp2_fulltest_slurm.sh"
    exit 1
  fi
  echo "[OK] Upstream fulltest readiness check passed"
fi

if [[ "${ANALYZE_ONLY}" != "1" ]]; then
  for ds in e_coli s_aureus; do
    agg_exp2="${ANALYSIS_ROOT}/${AGGREGATED_SUBDIR}/${ds}/exp2/per_sample_gate_by_band_aggregated.csv"
    agg_exp4="${ANALYSIS_ROOT}/${AGGREGATED_SUBDIR}/${ds}/exp4/latent_query_diversity_aggregated.csv"
    need_agg=0
    if [[ ! -f "${agg_exp2}" || ! -f "${agg_exp4}" ]]; then
      need_agg=1
    fi
    if [[ "${need_agg}" -eq 0 ]]; then
      echo "[SKIP] aggregate ${ds} (exp2+exp4 CSVs exist)"
      SKIP=$((SKIP + 1))
    else
      echo "[RUN] aggregate ${ds} (exp1+exp2+exp4)"
      if "${PYTHON_BIN}" -u "${REPO_ROOT}/evaluation_scripts/aggregate_fftlag_mechanism_seeds.py" \
        --analysis-root "${ANALYSIS_ROOT}" \
        --dataset "${ds}" \
        --exp1-subdir "${EXP1_SUBDIR}" \
        --exp2-subdir "${EXP2_SUBDIR}" \
        --exp4-subdir "${EXP4_SUBDIR}" \
        --aggregated-subdir "${AGGREGATED_SUBDIR}" \
        --exp-only all; then
        RUN=$((RUN + 1))
      else
        echo "[FAIL] aggregate ${ds}"
        FAIL=$((FAIL + 1))
      fi
    fi
  done
fi

for ds in e_coli s_aureus; do
  manifest="${MANIFEST_DIR}/${MODEL_VERSION}_${ds}_diff${DIFF}_fulltest.json"
  agg_dir="${ANALYSIS_ROOT}/${AGGREGATED_SUBDIR}/${ds}"
  out_dir="${OUTPUT_ROOT}/${ds}"
  fig_dir="${FIGURES_ROOT}/${ds}"
  done_marker="${out_dir}/bucketwise_band_sensitivity_summary.csv"
  gate_fig="${fig_dir}/bucketwise_gate_effect_combined.png"
  exp5_complete=0
  if [[ -f "${done_marker}" && -f "${gate_fig}" && -f "${fig_dir}/bucketwise_latent_diversity.png" ]]; then
    exp5_complete=1
  fi

  if [[ ! -f "${manifest}" ]]; then
    echo "[FAIL] missing manifest: ${manifest}"
    FAIL=$((FAIL + 1))
    continue
  fi

  if [[ "${exp5_complete}" -eq 1 && "${ANALYZE_ONLY}" != "1" ]]; then
    echo "[SKIP] Exp5 analyze ${ds} (outputs complete)"
    SKIP=$((SKIP + 1))
  else
    echo "[RUN] Exp5 analyze ${ds}"
    mkdir -p "${out_dir}" "${fig_dir}"
    if "${PYTHON_BIN}" -u "${REPO_ROOT}/analyze_fft_lag_mechanism_by_structure.py" \
      --manifest "${manifest}" \
      --aggregated-dir "${agg_dir}" \
      --output-dir "${out_dir}" \
      --figures-dir "${fig_dir}"; then
      RUN=$((RUN + 1))
    else
      echo "[FAIL] Exp5 analyze ${ds}"
      FAIL=$((FAIL + 1))
      continue
    fi
  fi

  if ! verify_exp5_output "${ds}"; then
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "Summary: RUN=${RUN} SKIP=${SKIP} FAIL=${FAIL}"
echo "Outputs: ${OUTPUT_ROOT}"
if (( FAIL > 0 )); then
  exit 1
fi

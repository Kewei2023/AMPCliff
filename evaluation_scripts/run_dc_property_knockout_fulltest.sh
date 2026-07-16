#!/usr/bin/env bash
# Exp5 / DC validation design v2 — Step 5 / 主实验二 Part B: fulltest property-bucket KO.
# Aggregate fulltest Exp1 results and run property-stratified DC knockout analysis.
#
# Prerequisites: all 10 seeds × 2 datasets finished Exp1 fulltest knockout.
#
# Usage:
#   bash evaluation_scripts/run_dc_property_knockout_fulltest.sh
#   REQUIRE_EXP1=0 bash evaluation_scripts/run_dc_property_knockout_fulltest.sh  # skip readiness check
#   PLOT_ONLY=1 bash evaluation_scripts/run_dc_property_knockout_fulltest.sh

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
DIFF="${DIFF:-5}"
THRESHOLD="${THRESHOLD:-0.9}"
REQUIRE_EXP1="${REQUIRE_EXP1:-1}"
PLOT_ONLY="${PLOT_ONLY:-0}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
read -r -a SEED_LIST <<< "${SEEDS}"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-${REPO_ROOT}/outputs/analysis/fftlag_mechanism}"
AGGREGATED_SUBDIR="${AGGREGATED_SUBDIR:-aggregated_fulltest}"
EXP1_SUBDIR="${EXP1_SUBDIR:-exp1_band_knockout_fulltest}"
PROPERTY_TABLE="${PROPERTY_TABLE:-${REPO_ROOT}/outputs/analysis/dc_validation/dc_property_table.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/analysis/dc_validation/property_dc_knockout_fulltest}"

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

check_exp1_ready() {
  local missing=0
  for ds in e_coli s_aureus; do
    local expected_n
    expected_n="$(expected_test_size "${ds}")"
    for seed in "${SEED_LIST[@]}"; do
      local csv="${ANALYSIS_ROOT}/seed_${seed}/${ds}/${EXP1_SUBDIR}/per_sample_band_sensitivity.csv"
      local n_saved
      n_saved="$(count_unique_idx "${csv}")"
      if [[ "${expected_n}" -gt 0 && "${n_saved}" -lt "${expected_n}" ]]; then
        echo "[NOT READY] seed=${seed} ds=${ds}: ${n_saved}/${expected_n} idx (${csv})"
        missing=$((missing + 1))
      fi
    done
  done
  return "${missing}"
}

RUN=0
SKIP=0
FAIL=0

echo "========== Property DC Knockout Fulltest =========="
echo "ANALYSIS_ROOT=${ANALYSIS_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "AGGREGATED_SUBDIR=${AGGREGATED_SUBDIR}"
echo "PLOT_ONLY=${PLOT_ONLY} REQUIRE_EXP1=${REQUIRE_EXP1}"
echo "==================================================="

if [[ "${REQUIRE_EXP1}" == "1" ]]; then
  if ! check_exp1_ready; then
    echo ""
    echo "Exp1 fulltest incomplete. Submit GPU job first (single job, serial seeds 0-9):"
    echo "  sbatch evaluation_scripts/run_fftlag_exp1_fulltest_slurm.sh"
    exit 1
  fi
  echo "[OK] Exp1 fulltest readiness check passed"
fi

if [[ "${PLOT_ONLY}" != "1" ]]; then
  for ds in e_coli s_aureus; do
    agg_csv="${ANALYSIS_ROOT}/${AGGREGATED_SUBDIR}/${ds}/exp1/per_sample_band_sensitivity_aggregated.csv"
    if [[ -f "${agg_csv}" ]]; then
      echo "[SKIP] aggregate ${ds}"
      SKIP=$((SKIP + 1))
    else
      echo "[RUN] aggregate ${ds}"
      if "${PYTHON_BIN}" -u "${REPO_ROOT}/evaluation_scripts/aggregate_fftlag_mechanism_seeds.py" \
        --analysis-root "${ANALYSIS_ROOT}" \
        --dataset "${ds}" \
        --exp1-subdir "${EXP1_SUBDIR}" \
        --aggregated-subdir "${AGGREGATED_SUBDIR}" \
        --exp-only 1; then
        RUN=$((RUN + 1))
      else
        echo "[FAIL] aggregate ${ds}"
        FAIL=$((FAIL + 1))
      fi
    fi
  done

  for ds in e_coli s_aureus; do
    agg_dir="${ANALYSIS_ROOT}/${AGGREGATED_SUBDIR}/${ds}"
    out_intermediate="${OUTPUT_ROOT}/${ds}/intermediate"
    sens_csv="${out_intermediate}/property_dc_knockout_sensitivity.csv"
    if [[ -f "${sens_csv}" ]]; then
      echo "[SKIP] analyze tables ${ds}"
      SKIP=$((SKIP + 1))
    else
      echo "[RUN] analyze tables ${ds}"
      mkdir -p "${out_intermediate}"
      if "${PYTHON_BIN}" -u "${REPO_ROOT}/evaluation_scripts/analyze_property_dc_tables.py" \
        --aggregated-dir "${agg_dir}" \
        --property-table "${PROPERTY_TABLE}" \
        --output-dir "${out_intermediate}" \
        --species "${ds}"; then
        RUN=$((RUN + 1))
      else
        echo "[FAIL] analyze tables ${ds}"
        FAIL=$((FAIL + 1))
      fi
    fi
  done
fi

for ds in e_coli s_aureus; do
  out_intermediate="${OUTPUT_ROOT}/${ds}/intermediate"
  out_figures="${OUTPUT_ROOT}/${ds}/figures"
  if [[ ! -d "${out_intermediate}" ]]; then
    echo "[FAIL] missing intermediate dir: ${out_intermediate}"
    FAIL=$((FAIL + 1))
    continue
  fi
  echo "[RUN] plot ${ds}"
  if "${PYTHON_BIN}" -u "${REPO_ROOT}/evaluation_scripts/plot_property_dc_knockout.py" \
    --intermediate-dir "${out_intermediate}" \
    --figures-dir "${out_figures}"; then
    RUN=$((RUN + 1))
  else
    echo "[FAIL] plot ${ds}"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "Summary: RUN=${RUN} SKIP=${SKIP} FAIL=${FAIL}"
echo "Outputs: ${OUTPUT_ROOT}"
echo ""
echo "Re-plot only (after editing plot script):"
echo "  PLOT_ONLY=1 bash evaluation_scripts/run_dc_property_knockout_fulltest.sh"
if (( FAIL > 0 )); then
  exit 1
fi

#!/usr/bin/env bash
# Exp5 / DC validation design v2 — official Exp5 pipeline:
#   Step1 property table -> Step2 DCT -> Step3 main1 encoding ->
#   Step4 main2A species effects -> Step5 main2B property-bucket KO.
#
# Usage:
#   bash evaluation_scripts/run_dc_validation_v2.sh
#   DRY_RUN=1 bash evaluation_scripts/run_dc_validation_v2.sh
#   SEEDS="0 1 2" bash evaluation_scripts/run_dc_validation_v2.sh
#   RUN_STEPS="2 3" bash evaluation_scripts/run_dc_validation_v2.sh
#   RUN_STEPS="3a" bash evaluation_scripts/run_dc_validation_v2.sh
#   SKIP_IF_DONE=0 bash evaluation_scripts/run_dc_validation_v2.sh  # force rerun completed steps
#
# Defaults align with FFT-LAG Exp1-Exp5: 10 train seeds (0-9).
# Step 2 expects 20 npz files (2 species x 10 seeds) under dct_features/.

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
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_STEPS="${RUN_STEPS:-0 1 2 3 3a 4 5}"
MODEL_VERSION="${MODEL_VERSION:-esm2_t6}"
POOLING="${POOLING:-fft_latent_attn_gate}"
DIFF="${DIFF:-5}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/blosum62 average/diff_${DIFF}-trd_0.9}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/analysis/dc_validation}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${REPO_ROOT}/outputs/analysis/fftlag_mechanism}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/outputs/ablation_new_data/_amp_knockout_seed_runs/_peptide_manifest}"
CONFIG_DIR="${CONFIG_DIR:-/data/public/models/facebook/esm2_t6_8M_UR50D/}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SUBSET_SEED="${SUBSET_SEED:-42}"
N_PEPTIDES="${N_PEPTIDES:-30}"
THRESHOLD="${THRESHOLD:-0.9}"

if [[ -d "${REPO_ROOT}/outputs/ablation_new_data" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_new_data}"
elif [[ -d "${REPO_ROOT}/outputs/ablation-new-data" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation-new-data}"
else
  ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/ablation_new_data}"
fi

PROPERTY_TABLE="${OUTPUT_ROOT}/dc_property_table.csv"
FEATURE_DIR="${OUTPUT_ROOT}/dct_features"
PROBE_DIR="${OUTPUT_ROOT}/dc_property_encoding"
SPECIES_DIR="${OUTPUT_ROOT}/species_property_effects"
PROPERTY_KO_DIR="${OUTPUT_ROOT}/property_dc_knockout"

_should_run_step() {
  local id="$1"
  local steps=",${RUN_STEPS// /,},"
  [[ "${steps}" == *",${id},"* ]]
}

_is_done() {
  [[ "${SKIP_IF_DONE}" == "1" && -f "$1" ]]
}

_print_preflight_summary() {
  read -r -a _pf_seed_list <<< "${SEEDS}"
  local _pf_n_seeds="${#_pf_seed_list[@]}"
  local _pf_n_npz_expected=$((2 * _pf_n_seeds))

  if _should_run_step 0 || _should_run_step 5; then
    local _pf_agg_done=0
    for _pf_ds in e_coli s_aureus; do
      local _pf_agg="${ANALYSIS_ROOT}/aggregated/${_pf_ds}/exp1/per_sample_band_sensitivity_aggregated.csv"
      if _is_done "${_pf_agg}"; then
        _pf_agg_done=$((_pf_agg_done + 1))
      fi
    done
    if [[ "${_pf_agg_done}" -eq 2 ]]; then
      echo "[PLAN] Step 0: SKIP (2/2 agg csv)"
    elif [[ "${_pf_agg_done}" -gt 0 ]]; then
      echo "[PLAN] Step 0: PARTIAL (${_pf_agg_done}/2 agg csv, will run missing)"
    else
      echo "[PLAN] Step 0: RUN"
    fi
  fi

  if _should_run_step 1; then
    if _is_done "${PROPERTY_TABLE}"; then
      echo "[PLAN] Step 1: SKIP"
    else
      echo "[PLAN] Step 1: RUN"
    fi
  fi

  if _should_run_step 2; then
    local _pf_npz_done=0
    for _pf_ds in e_coli s_aureus; do
      for _pf_seed in "${_pf_seed_list[@]}"; do
        if _is_done "${FEATURE_DIR}/${_pf_ds}_seed${_pf_seed}.npz"; then
          _pf_npz_done=$((_pf_npz_done + 1))
        fi
      done
    done
    if [[ "${_pf_npz_done}" -eq "${_pf_n_npz_expected}" ]]; then
      echo "[PLAN] Step 2: SKIP (${_pf_npz_done}/${_pf_n_npz_expected} npz)"
    elif [[ "${_pf_npz_done}" -gt 0 ]]; then
      echo "[PLAN] Step 2: PARTIAL (${_pf_npz_done}/${_pf_n_npz_expected} npz, will run missing)"
    else
      echo "[PLAN] Step 2: RUN (${_pf_n_npz_expected} npz)"
    fi
  fi

  if _should_run_step 3; then
    if _is_done "${PROBE_DIR}/dc_property_probe_results.csv" && _is_done "${PROBE_DIR}/dc_c0_vs_c1_delta_ci.csv"; then
      echo "[PLAN] Step 3: SKIP"
    else
      echo "[PLAN] Step 3: RUN"
    fi
  fi

  if _should_run_step 3a; then
    if _is_done "${PROBE_DIR}/aac_property_probe_results.csv" && _is_done "${PROBE_DIR}/aac_vs_dc_comparison.csv"; then
      echo "[PLAN] Step 3a: SKIP"
    else
      echo "[PLAN] Step 3a: RUN"
    fi
  fi

  if _should_run_step 4; then
    if _is_done "${SPECIES_DIR}/species_property_activity_effects.csv"; then
      echo "[PLAN] Step 4: SKIP"
    else
      echo "[PLAN] Step 4: RUN"
    fi
  fi

  if _should_run_step 5; then
    local _pf_ko_done=0
    for _pf_ds in e_coli s_aureus; do
      if _is_done "${PROPERTY_KO_DIR}/${_pf_ds}/property_dc_knockout_sensitivity.csv"; then
        _pf_ko_done=$((_pf_ko_done + 1))
      fi
    done
    if [[ "${_pf_ko_done}" -eq 2 ]]; then
      echo "[PLAN] Step 5: SKIP (2/2 species)"
    elif [[ "${_pf_ko_done}" -gt 0 ]]; then
      echo "[PLAN] Step 5: PARTIAL (${_pf_ko_done}/2 species, will run missing)"
    else
      echo "[PLAN] Step 5: RUN"
    fi
  fi
}

resolve_ckpt_dir() {
  local ds="$1"
  local seed="$2"
  local seed_root="${ABLATION_ROOT}/${MODEL_VERSION}_${POOLING}_${ds}_diff${DIFF}/seed_${seed}"
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
    --threshold "${THRESHOLD}" \
    --subset-seed "${SUBSET_SEED}" \
    --n-peptides "${N_PEPTIDES}" \
    --out-dir "${MANIFEST_DIR}" >&2
  echo "${manifest}"
}

RUN=0
SKIP=0
FAIL=0

echo "========== DC Validation v2 =========="
echo "REPO_ROOT: ${REPO_ROOT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "ABLATION_ROOT: ${ABLATION_ROOT}"
echo "SEEDS: ${SEEDS}"
echo "RUN_STEPS: ${RUN_STEPS}"
echo "DRY_RUN: ${DRY_RUN}"
echo "SKIP_IF_DONE: ${SKIP_IF_DONE}"
echo ""

mkdir -p "${OUTPUT_ROOT}" "${FEATURE_DIR}" "${PROBE_DIR}" "${SPECIES_DIR}" "${PROPERTY_KO_DIR}"

_print_preflight_summary
echo ""

# ---- Step 0: ensure aggregated Exp1 CSV exists ----
if _should_run_step 0 || _should_run_step 5; then
  for ds in e_coli s_aureus; do
    agg_csv="${ANALYSIS_ROOT}/aggregated/${ds}/exp1/per_sample_band_sensitivity_aggregated.csv"
    if _is_done "${agg_csv}"; then
      echo "[SKIP] aggregated exp1 exists: ${agg_csv}"
      SKIP=$((SKIP + 1))
      continue
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY] aggregate_fftlag_mechanism_seeds.py --dataset ${ds}"
      continue
    fi
    echo "[RUN] aggregate exp1 for ${ds}"
    if "${PYTHON_BIN}" "${REPO_ROOT}/evaluation_scripts/aggregate_fftlag_mechanism_seeds.py" \
      --analysis-root "${ANALYSIS_ROOT}" \
      --dataset "${ds}"; then
      RUN=$((RUN + 1))
    else
      echo "[FAIL] aggregate exp1 for ${ds}"
      FAIL=$((FAIL + 1))
    fi
  done
fi

# ---- Step 1: property table ----
if _should_run_step 1; then
  if _is_done "${PROPERTY_TABLE}"; then
    echo "[SKIP] ${PROPERTY_TABLE}"
    SKIP=$((SKIP + 1))
  elif [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY] build_dc_property_table.py -> ${PROPERTY_TABLE}"
  else
    echo "[RUN] build_dc_property_table.py"
    if "${PYTHON_BIN}" -u "${REPO_ROOT}/build_dc_property_table.py" \
      --e-coli-data-dir "${DATA_DIR}" \
      --s-aureus-data-dir "${DATA_DIR}" \
      --output "${PROPERTY_TABLE}"; then
      RUN=$((RUN + 1))
    else
      echo "[FAIL] build_dc_property_table.py"
      FAIL=$((FAIL + 1))
    fi
  fi
fi

# ---- Step 2: DCT feature extraction ----
if _should_run_step 2; then
  if [[ ! -f "${PROPERTY_TABLE}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY] extract_dct_coefficient_features.py (property table will be built in step 1)"
    else
      echo "[FAIL] missing property table: ${PROPERTY_TABLE}"
      FAIL=$((FAIL + 1))
    fi
  else
    read -r -a SEED_LIST <<< "${SEEDS}"
    for ds in e_coli s_aureus; do
      for seed in "${SEED_LIST[@]}"; do
        out_npz="${FEATURE_DIR}/${ds}_seed${seed}.npz"
        if _is_done "${out_npz}"; then
          echo "[SKIP] ${out_npz}"
          SKIP=$((SKIP + 1))
          continue
        fi
        if ! ckpt_dir="$(resolve_ckpt_dir "${ds}" "${seed}")"; then
          echo "[FAIL] checkpoint not found: ${ds} seed=${seed}"
          FAIL=$((FAIL + 1))
          continue
        fi
        if [[ "${DRY_RUN}" == "1" ]]; then
          echo "[DRY] extract_dct_coefficient_features.py ${ds} seed=${seed} ckpt=${ckpt_dir}"
          continue
        fi
        echo "[RUN] extract DCT features ${ds} seed=${seed}"
        if "${PYTHON_BIN}" -u "${REPO_ROOT}/extract_dct_coefficient_features.py" \
          --property-table "${PROPERTY_TABLE}" \
          --species "${ds}" \
          --checkpoint "${ckpt_dir}" \
          --output "${out_npz}" \
          --model-version "${MODEL_VERSION}" \
          --pooling "${POOLING}" \
          --config-dir "${CONFIG_DIR}" \
          --batch-size "${BATCH_SIZE}"; then
          RUN=$((RUN + 1))
        else
          echo "[FAIL] extract DCT ${ds} seed=${seed}"
          FAIL=$((FAIL + 1))
        fi
      done
    done
  fi
fi

# ---- Step 3: property probes (+ C0 vs C1 delta CI) ----
if _should_run_step 3; then
  probe_csv="${PROBE_DIR}/dc_property_probe_results.csv"
  delta_csv="${PROBE_DIR}/dc_c0_vs_c1_delta_ci.csv"
  if _is_done "${probe_csv}" && _is_done "${delta_csv}"; then
    echo "[SKIP] ${probe_csv} and ${delta_csv}"
    SKIP=$((SKIP + 1))
  elif [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY] analyze_dc_property_encoding.py"
  else
    seed_args=()
    read -r -a SEED_LIST <<< "${SEEDS}"
    for seed in "${SEED_LIST[@]}"; do
      seed_args+=("${seed}")
    done
    echo "[RUN] analyze_dc_property_encoding.py"
    if "${PYTHON_BIN}" -u "${REPO_ROOT}/analyze_dc_property_encoding.py" \
      --feature-dir "${FEATURE_DIR}" \
      --property-table "${PROPERTY_TABLE}" \
      --output-dir "${PROBE_DIR}" \
      --seeds "${seed_args[@]}"; then
      RUN=$((RUN + 1))
    else
      echo "[FAIL] analyze_dc_property_encoding.py"
      FAIL=$((FAIL + 1))
    fi
  fi
fi

# ---- Step 3a: AAC composition baseline + comparison ----
if _should_run_step 3a; then
  aac_csv="${PROBE_DIR}/aac_property_probe_results.csv"
  compare_csv="${PROBE_DIR}/aac_vs_dc_comparison.csv"
  dc_probe_csv="${PROBE_DIR}/dc_property_probe_results.csv"
  if _is_done "${aac_csv}" && _is_done "${compare_csv}"; then
    echo "[SKIP] ${aac_csv} and ${compare_csv}"
    SKIP=$((SKIP + 1))
  elif [[ ! -f "${PROPERTY_TABLE}" && "${DRY_RUN}" != "1" ]]; then
    echo "[FAIL] missing property table for step 3a: ${PROPERTY_TABLE}"
    FAIL=$((FAIL + 1))
  elif [[ ! -f "${dc_probe_csv}" && "${DRY_RUN}" != "1" ]]; then
    echo "[FAIL] missing DC probe results for step 3a: ${dc_probe_csv} (run step 3 first)"
    FAIL=$((FAIL + 1))
  elif [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY] analyze_aac_property_baseline.py + compare_probe_baselines.py"
  else
    echo "[RUN] analyze_aac_property_baseline.py"
    _rc_aac=0
    if ! _is_done "${aac_csv}"; then
      "${PYTHON_BIN}" -u "${REPO_ROOT}/analyze_aac_property_baseline.py" \
        --property-table "${PROPERTY_TABLE}" \
        --output-dir "${PROBE_DIR}" || _rc_aac=$?
    fi
    echo "[RUN] compare_probe_baselines.py"
    _rc_cmp=0
    if ! _is_done "${compare_csv}"; then
      "${PYTHON_BIN}" -u "${REPO_ROOT}/evaluation_scripts/compare_probe_baselines.py" \
        --aac-csv "${aac_csv}" \
        --dc-csv "${dc_probe_csv}" \
        --output-dir "${PROBE_DIR}" || _rc_cmp=$?
    fi
    if (( _rc_aac != 0 || _rc_cmp != 0 )); then
      echo "[FAIL] step 3a (aac=${_rc_aac}, compare=${_rc_cmp})"
      FAIL=$((FAIL + 1))
    else
      RUN=$((RUN + 1))
    fi
  fi
fi

# ---- Step 4: species property effects ----
if _should_run_step 4; then
  effects_csv="${SPECIES_DIR}/species_property_activity_effects.csv"
  if _is_done "${effects_csv}"; then
    echo "[SKIP] ${effects_csv}"
    SKIP=$((SKIP + 1))
  elif [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY] analyze_species_property_effects.py"
  else
    echo "[RUN] analyze_species_property_effects.py"
    if "${PYTHON_BIN}" -u "${REPO_ROOT}/analyze_species_property_effects.py" \
      --property-table "${PROPERTY_TABLE}" \
      --output-dir "${SPECIES_DIR}"; then
      RUN=$((RUN + 1))
    else
      echo "[FAIL] analyze_species_property_effects.py"
      FAIL=$((FAIL + 1))
    fi
  fi
fi

# ---- Step 5: property-stratified DC knockout ----
if _should_run_step 5; then
  for ds in e_coli s_aureus; do
    manifest="$(ensure_manifest "${ds}")"
    out_dir="${PROPERTY_KO_DIR}/${ds}"
    ko_csv="${out_dir}/property_dc_knockout_sensitivity.csv"
    if [[ ! -f "${manifest}" && "${DRY_RUN}" != "1" ]]; then
      echo "[FAIL] missing manifest: ${manifest}"
      FAIL=$((FAIL + 1))
      continue
    fi
    if _is_done "${ko_csv}"; then
      echo "[SKIP] ${ko_csv}"
      SKIP=$((SKIP + 1))
      continue
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY] property knockout analysis ${ds}"
      continue
    fi
    mkdir -p "${out_dir}"
    echo "[RUN] property DC knockout ${ds}"
    if "${PYTHON_BIN}" -u "${REPO_ROOT}/analyze_fft_lag_mechanism_by_structure.py" \
      --analysis-mode property \
      --manifest "${manifest}" \
      --property-table "${PROPERTY_TABLE}" \
      --aggregated-dir "${ANALYSIS_ROOT}/aggregated/${ds}" \
      --properties net_charge mean_hydrophobicity \
      --bands 0 1 \
      --species "${ds}" \
      --output-dir "${out_dir}"; then
      RUN=$((RUN + 1))
    else
      echo "[FAIL] property DC knockout ${ds}"
      FAIL=$((FAIL + 1))
    fi
  done
fi

echo ""
echo "Summary: RUN=${RUN} SKIP=${SKIP} FAIL=${FAIL}"
echo "Outputs: ${OUTPUT_ROOT}"
if (( FAIL > 0 )); then
  exit 1
fi

# Replot figures only (after label/style changes, no probe recomputation):
#   python evaluation_scripts/replot_dc_validation_figures.py
#   PLOT_ONLY=1 bash evaluation_scripts/run_dc_property_knockout_fulltest.sh

#!/bin/bash
#SBATCH --job-name=abl_prot
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --chdir=/data/home/scv6872/run/kwli/AMPCliff
#SBATCH --output=logs/ablation_protein_%j.out
#SBATCH --error=logs/ablation_protein_%j.err
# =============================================================================
# run_ablation_protein.sh
#
# Ablation study for protein (peptide) experiments.
# ESM2-8M / E. coli, 12 pooling configs x N seeds.
#
# Uses Hydra CLI override to pass config to downstream_train.py,
# consistent with the project's other experiment scripts.
#
# Environment variables:
#   REPO_ROOT       default /data/home/scv6872/run/kwli/AMPCliff
#   OUTPUT_ROOT     default ${REPO_ROOT}/outputs/ablation_protein
#   SEEDS           default "0 1 2 3 4"
#   DATASETS        default "e_coli"
#   MODEL           default "esm2_t6"
#   NUM_SEEDS       default 5 (overrides SEEDS if set)
#   DRY_RUN         default "" (set to "1" for dry run)
#
# Usage:
#   sbatch run_ablation_protein.sh
#   DRY_RUN=1 bash run_ablation_protein.sh
#   SEEDS="0 1 2" bash run_ablation_protein.sh
# =============================================================================
set -uo pipefail

module load miniforge gcc/11.1.0 cuda/11.1 gcc/11.1.0
module load cudnn/8.1.0.77_CUDA11.1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate AMPCliff

REPO_ROOT="${REPO_ROOT:-/data/home/scv6872/run/kwli/AMPCliff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/ablation_protein}"
MODEL="${MODEL:-esm2_t6}"
DATASETS="${DATASETS:-e_coli}"
DRY_RUN="${DRY_RUN:-}"
DIFF="5"
THRESHOLD="0.9"
CONDITION="blosum62 average"

# Resolve model config dir
case "${MODEL}" in
    esm2_t6)  CONFIG_DIR="/data/public/models/facebook/esm2_t6_8M_UR50D/" ;;
    esm2_t12) CONFIG_DIR="/data/public/models/facebook/esm2_t12_35M_UR50D/" ;;
    esm2_t33) CONFIG_DIR="/data/public/models/facebook/esm2_t33_650M_UR50D/" ;;
    *) echo "Unknown model: ${MODEL}"; exit 1 ;;
esac

# Data base path
DATA_BASE="/data/home/scv6872/run/kwli/AMPCliff/data/blosum62 average/diff_5-trd_0.9"

# Seeds
if [[ -n "${NUM_SEEDS:-}" ]]; then
    SEEDS=$(seq 0 $((NUM_SEEDS - 1)))
else
    SEEDS="${SEEDS:-0 1 2 3 4}"
fi

# Ablation configs: each line is "POOLING_NAME:pooling_value"
# The part before : is a human-readable tag, after : is the Hydra pooling value
# Set POOLINGS env var to override (space-separated list of tag names)
_ALL_CONFIGS=(
    "mean:mean"
    "max:max"
    "attn_structured:attn_structured"
    "last:last"
    "latent_attn:latent_attn"
    "mltp_paper:mltp_paper"
    "FLaG:FLaG"
)

if [[ -n "${POOLINGS:-}" ]]; then
    # Filter to only requested configs
    ABLATION_CONFIGS=()
    for requested in ${POOLINGS}; do
        for entry in "${_ALL_CONFIGS[@]}"; do
            tag="${entry%%:*}"
            if [[ "${tag}" == "${requested}" ]]; then
                ABLATION_CONFIGS+=("${entry}")
                break
            fi
        done
    done
else
    ABLATION_CONFIGS=("${_ALL_CONFIGS[@]}")
fi

RUN_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0
declare -a RESULT_LINES=()

echo "=========================================="
echo "Ablation Study - Protein (Peptide)"
echo "=========================================="
echo "REPO_ROOT:     ${REPO_ROOT}"
echo "OUTPUT_ROOT:   ${OUTPUT_ROOT}"
echo "MODEL:         ${MODEL}"
echo "CONFIG_DIR:    ${CONFIG_DIR}"
echo "DATA_BASE:     ${DATA_BASE}"
echo "SEEDS:         ${SEEDS}"
echo "APPLY:         ${APPLY}"
echo "Configs:       ${#ABLATION_CONFIGS[@]}"
echo "=========================================="

cd "${REPO_ROOT}"

for DATASET in ${DATASETS}; do
    for ENTRY in "${ABLATION_CONFIGS[@]}"; do
        TAG="${ENTRY%%:*}"
        POOLING="${ENTRY#*:}"

        for SEED in ${SEEDS}; do

            EXP_TAG="${MODEL}_${TAG}_${DATASET}_diff${DIFF}"
            SEED_DIR="${OUTPUT_ROOT}/${EXP_TAG}/seed_${SEED}"

            # Skip if already done (look for test result CSV)
            TEST_RESULT="${SEED_DIR}/${MODEL}-blosum62 average-diff${DIFF}-trd${THRESHOLD}-test_result.csv"
            if [[ -f "${TEST_RESULT}" ]]; then
                echo "[SKIP] ${EXP_TAG}/seed_${SEED} — already done"
                SKIP_COUNT=$((SKIP_COUNT + 1))
                RESULT_LINES+=("[SKIP] ${EXP_TAG}/seed_${SEED}")
                continue
            fi

            echo ""
            echo ">>> [RUN] ${EXP_TAG}/seed_${SEED}"
            echo "    pooling:  ${POOLING}"
            echo "    output:  ${SEED_DIR}"

            if [[ "${DRY_RUN}" == "1" ]]; then
                echo "    (dry run, skipping execution)"
                RUN_COUNT=$((RUN_COUNT + 1))
                RESULT_LINES+=("[DRY ] ${EXP_TAG}/seed_${SEED}")
                continue
            fi

            mkdir -p "${SEED_DIR}"

            # Resolve data files for this dataset
            TRAIN_FILE="${DATA_BASE}/grampa_${DATASET}_7_25-train.csv"
            VALID_FILE="${DATA_BASE}/grampa_${DATASET}_7_25-valid.csv"
            TEST_FILE="${DATA_BASE}/grampa_${DATASET}_7_25-test.csv"

            _rc=0
            python -u downstream_train.py \
                model.config_dir="${CONFIG_DIR}" \
                model.regression.version="${MODEL}" \
                model.regression.pooling="${POOLING}" \
                model.regression.check_point.load=false \
                data.regression.dataset="${DATASET}" \
                data.regression.mode=fix \
                data.regression.fix.train_file="${TRAIN_FILE}" \
                data.regression.fix.valid_file="${VALID_FILE}" \
                data.regression.fix.test_file="${TEST_FILE}" \
                data.diff="[${DIFF}]" \
                data.threshold="${THRESHOLD}" \
                data.regression.condition="[${CONDITION}]" \
                train.random_seed="${SEED}" \
                mode.ddp=false \
                mode.amp=false \
                logger.log=false \
                other.debug=false \
                hydra.run.dir="${SEED_DIR}" \
                2>&1 | tee "${SEED_DIR}/downstream_train.log" || _rc=$?

            if (( _rc != 0 )); then
                FAIL_COUNT=$((FAIL_COUNT + 1))
                RESULT_LINES+=("[FAIL] ${EXP_TAG}/seed_${SEED}")
            else
                RUN_COUNT=$((RUN_COUNT + 1))
                RESULT_LINES+=("[RUN ] ${EXP_TAG}/seed_${SEED}")
            fi

            unset _rc

        done
    done
done

echo ""
echo "=========================================="
echo "Summary: RUN=${RUN_COUNT}  SKIP=${SKIP_COUNT}  FAIL=${FAIL_COUNT}"
echo "------------------------------------------"
for _line in "${RESULT_LINES[@]}"; do
    echo "  ${_line}"
done
echo "=========================================="
echo ""
echo "To analyze results:"
echo "  python ${REPO_ROOT}/analyze_ablation_results.py --input_dir ${OUTPUT_ROOT}"
echo "=========================================="

exit $(( FAIL_COUNT > 0 ? 1 : 0 ))

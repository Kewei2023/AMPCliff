#/usr/bin/env bash
set -euo pipefail

DATASET="s_aureus" MODEL_TYPE="esm2_t6" bash evaluation_scripts/run_pooling_baseline_ablation.sh
DATASET="e_coli" MODEL_TYPE="esm2_t12" bash evaluation_scripts/run_pooling_baseline_ablation.sh
DATASET="s_aureus" MODEL_TYPE="esm2_t12" bash evaluation_scripts/run_pooling_baseline_ablation.sh
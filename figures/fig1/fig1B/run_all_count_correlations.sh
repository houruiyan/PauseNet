#!/usr/bin/env bash
set -euo pipefail

# Batch plot count-task correlations for all completed PauseNet evaluations.
#
# Optional positional arguments:
#   1. evaluation root
#   2. output directory
#
# PYTHON_BIN can be overridden in the environment when using another conda env.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_ROOT="${1:-/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/HDD8TB/houruiyan/env/conda_envs/testPauseNet/bin/python}"

DATASETS=(
    "hek293t_groseq"
    "hek293t_netseq"
    "hek293t_proseq"
    "helas3_netseq"
    "k562_mnetseq"
    "molt4_netseq"
)

NPZ_FILES=()
for dataset in "${DATASETS[@]}"; do
    npz_path="${EVALUATION_ROOT}/${dataset}/test_profiles.npz"
    if [[ ! -s "${npz_path}" ]]; then
        echo "Missing or empty input: ${npz_path}" >&2
        exit 1
    fi
    NPZ_FILES+=("${npz_path}")
done

mkdir -p "${OUTPUT_DIR}"
export PYTHONNOUSERSITE=1

"${PYTHON_BIN}" "${SCRIPT_DIR}/1.plot_count_correlation.py" \
    --npz "${NPZ_FILES[@]}" \
    --outdir "${OUTPUT_DIR}" \
    --shared-axis-limit

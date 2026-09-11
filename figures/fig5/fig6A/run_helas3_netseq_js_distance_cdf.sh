#!/usr/bin/env bash
set -euo pipefail

readonly FIG6_DIR="/mnt/HDD8TB/houruiyan/pausing_site/4_plot_figure/fig6"
readonly PYTHON_BIN="/mnt/HDD8TB/houruiyan/env/conda_envs/testPauseNet/bin/python"
readonly PLOT_SCRIPT="/mnt/HDD8TB/houruiyan/pausing_site/4_plot_figure/fig1/fig1C/1.plot_main_js_distance_cdf.py"
readonly NPZ="/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/helas3_netseq/test_profiles.npz"
readonly POS_BW="/mnt/HDD8TB/houruiyan/pausing_site/data/HeLaS3_NETseq/merge/merged/HeLaS3.merged.pos.bw"
readonly NEG_BW="/mnt/HDD8TB/houruiyan/pausing_site/data/HeLaS3_NETseq/merge/merged/HeLaS3.merged.neg.bw"
readonly OUTPUT_PDF="${FIG6_DIR}/helas3_netseq_js_distance_cdf_res20bp_count_ge_50.pdf"

for required_file in \
    "${PYTHON_BIN}" \
    "${PLOT_SCRIPT}" \
    "${NPZ}" \
    "${POS_BW}" \
    "${NEG_BW}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file is missing: ${required_file}" >&2
        exit 1
    fi
done

mkdir -p "${FIG6_DIR}"

"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
    --npz "${NPZ}" \
    --pos-bw "${POS_BW}" \
    --neg-bw "${NEG_BW}" \
    --signal-unit 1 \
    --pseudoreplicate-repeats 10 \
    --resolution 20 \
    --count-threshold 50 \
    --output-pdf "${OUTPUT_PDF}"

echo "Completed: ${OUTPUT_PDF}"

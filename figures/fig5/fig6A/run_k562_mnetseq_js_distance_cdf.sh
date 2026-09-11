#!/usr/bin/env bash
set -euo pipefail

readonly FIG6_DIR="/mnt/HDD8TB/houruiyan/pausing_site/4_plot_figure/fig6"
readonly PYTHON_BIN="/mnt/HDD8TB/houruiyan/env/conda_envs/testPauseNet/bin/python"
readonly PLOT_SCRIPT="/mnt/HDD8TB/houruiyan/pausing_site/4_plot_figure/fig1/fig1C/1.plot_main_js_distance_cdf.py"
readonly NPZ="/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/k562_mnetseq/test_profiles.npz"
readonly POS_BW_REP1="/mnt/HDD8TB/houruiyan/pausing_site/data/K562_mNETseq/hg19_coordinate/GSM3518117_N_tCTD_1_CGGAAT_1_K562_wildtype.plus.hg19.bw"
readonly NEG_BW_REP1="/mnt/HDD8TB/houruiyan/pausing_site/data/K562_mNETseq/hg19_coordinate/GSM3518117_N_tCTD_1_CGGAAT_1_K562_wildtype.minus.hg19.bw"
readonly POS_BW_REP2="/mnt/HDD8TB/houruiyan/pausing_site/data/K562_mNETseq/hg19_coordinate/GSM3518118_N_tCTD_2_CTAGCT_2_K562_wildtype.plus.hg19.bw"
readonly NEG_BW_REP2="/mnt/HDD8TB/houruiyan/pausing_site/data/K562_mNETseq/hg19_coordinate/GSM3518118_N_tCTD_2_CTAGCT_2_K562_wildtype.minus.hg19.bw"
readonly OUTPUT_PDF="${FIG6_DIR}/k562_mnetseq_js_distance_cdf_res20bp_count_ge_50.pdf"

for required_file in \
    "${PYTHON_BIN}" \
    "${PLOT_SCRIPT}" \
    "${NPZ}" \
    "${POS_BW_REP1}" \
    "${NEG_BW_REP1}" \
    "${POS_BW_REP2}" \
    "${NEG_BW_REP2}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file is missing: ${required_file}" >&2
        exit 1
    fi
done

mkdir -p "${FIG6_DIR}"

"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
    --npz "${NPZ}" \
    --pos-bw "${POS_BW_REP1}" "${POS_BW_REP2}" \
    --neg-bw "${NEG_BW_REP1}" "${NEG_BW_REP2}" \
    --signal-unit 1.0430108308792114 0.9082756638526917 \
    --pseudoreplicate-repeats 10 \
    --resolution 20 \
    --count-threshold 50 \
    --output-pdf "${OUTPUT_PDF}"

echo "Completed: ${OUTPUT_PDF}"

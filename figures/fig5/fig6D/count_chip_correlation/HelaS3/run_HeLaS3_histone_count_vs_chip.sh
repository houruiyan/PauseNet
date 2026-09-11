#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NPZ="/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/helas3_netseq/test_profiles.npz"
PYTHON_BIN="${PYTHON_BIN:-/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python}"

SIGNALS=(
  "H3K27ac|/mnt/HDD8TB/houruiyan/pausing_site/data/histone_chipseq/H3K27ac/ENCFF388WMD_HelaS3_hg19.bigWig"
  "H3K27me3|/mnt/HDD8TB/houruiyan/pausing_site/data/histone_chipseq/H3K27me3/ENCFF958BAN_HelaS3_hg19.bigWig"
  "H3K36me3|/mnt/HDD8TB/houruiyan/pausing_site/data/histone_chipseq/H3K36me3/ENCFF559FSM_HelaS3_hg19.bigWig"
  "H3K4me3|/mnt/HDD8TB/houruiyan/pausing_site/data/histone_chipseq/H3K4me3/ENCFF699TXY_HelaS3_hg19.bigWig"
  "H3K79me2|/mnt/HDD8TB/houruiyan/pausing_site/data/histone_chipseq/H3K79me2/ENCFF432DSJ_HelaS3_hg19.bigWig"
  "H3K9ac|/mnt/HDD8TB/houruiyan/pausing_site/data/histone_chipseq/H3K9ac/ENCFF573VUK_HelaS3_hg19.bigWig"
  "H3K9me3|/mnt/HDD8TB/houruiyan/pausing_site/data/histone_chipseq/H3K9me3/ENCFF761QZP_HelaS3_hg19.bigWig"
  "H4K20me1|/mnt/HDD8TB/houruiyan/pausing_site/data/histone_chipseq/H4K20me1/ENCFF114EGM_HelaS3_hg19.bigWig"
)

for entry in "${SIGNALS[@]}"; do
  IFS='|' read -r signal bigwig <<< "${entry}"
  prefix="helas3_${signal,,}_pausing_count_vs_chip"
  echo "Running HeLaS3 ${signal}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_pausing_count_vs_chip.py" \
    --npz "${NPZ}" \
    --signal-bw "${bigwig}" \
    --signal-name "${signal}" \
    --cell-label HeLaS3 \
    --count-source observed \
    --fit linear \
    --expected-assembly hg19 \
    --output-pdf "${SCRIPT_DIR}/${prefix}_scatter.pdf" \
    --output-tsv "${SCRIPT_DIR}/${prefix}_values.tsv" \
    --output-summary "${SCRIPT_DIR}/${prefix}_summary.tsv"
done

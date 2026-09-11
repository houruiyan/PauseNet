#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/plot_residual_vs_chip_difference.py" \
  --cell-a-npz /mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/k562_mnetseq/test_profiles.npz \
  --cell-b-npz /mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/helas3_netseq/test_profiles.npz \
  --cell-a-bigwig /mnt/HDD8TB/houruiyan/pausing_site/data/histone_chipseq/H3K4me3/ENCFF715DGL_K562_hg19.bigWig \
  --cell-b-bigwig /mnt/HDD8TB/houruiyan/pausing_site/data/histone_chipseq/H3K4me3/ENCFF699TXY_HelaS3_hg19.bigWig \
  --cell-a-label K562 \
  --cell-b-label HeLaS3 \
  --signal-name H3K4me3 \
  --fit linear \
  --output-pdf "${SCRIPT_DIR}/h3k4me3_k562_minus_helas3_residual_vs_chip_scatter.pdf" \
  --output-tsv "${SCRIPT_DIR}/h3k4me3_k562_minus_helas3_residual_vs_chip_values.tsv"

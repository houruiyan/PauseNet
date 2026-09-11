#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NPZ="/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/helas3_netseq/test_profiles.npz"
PYTHON_BIN="${PYTHON_BIN:-/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python}"

SIGNALS=(
  "CTCF|/mnt/HDD8TB/houruiyan/pausing_site/data/TF/CTCF/ENCFF179RSE_HelaS3_hg19.bigWig"
  "CHD2|/mnt/HDD8TB/houruiyan/pausing_site/data/TF/CHD2/ENCFF239ZWG_CHD2_HeLaS3_hg19.bigWig"
  "SWI|/mnt/HDD8TB/houruiyan/pausing_site/data/TF/SWI/ENCFF805DTS_HelaS3_hg19.bigWig"
  "TFIIF|/mnt/HDD8TB/houruiyan/pausing_site/data/TF/TFIIF/ENCFF541OKE_TFIIF_HelaS3_hg19.bigWig"
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

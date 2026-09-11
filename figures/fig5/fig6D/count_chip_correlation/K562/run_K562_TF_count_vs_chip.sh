#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NPZ="/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/k562_mnetseq/test_profiles.npz"
PYTHON_BIN="${PYTHON_BIN:-/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python}"

SIGNALS=(
  "CTCF|/mnt/HDD8TB/houruiyan/pausing_site/data/TF/CTCF/ENCFF979PWH_K562_hg19.bigWig"
  "CHD2|/mnt/HDD8TB/houruiyan/pausing_site/data/TF/CHD2/ENCFF032HVZ_CHD2_K562_hg19.bigWig"
  "SWI|/mnt/HDD8TB/houruiyan/pausing_site/data/TF/SWI/ENCFF424PDB_K562_hg19.bigWig"
  "TFIIF|/mnt/HDD8TB/houruiyan/pausing_site/data/TF/TFIIF/ENCFF050ZBY_TFIIF_K562_hg19.bigWig"
)

for entry in "${SIGNALS[@]}"; do
  IFS='|' read -r signal bigwig <<< "${entry}"
  prefix="k562_${signal,,}_pausing_count_vs_chip"
  echo "Running ${signal}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_pausing_count_vs_chip.py" \
    --npz "${NPZ}" \
    --signal-bw "${bigwig}" \
    --signal-name "${signal}" \
    --cell-label K562 \
    --count-source observed \
    --fit linear \
    --expected-assembly hg19 \
    --output-pdf "${SCRIPT_DIR}/${prefix}_scatter.pdf" \
    --output-tsv "${SCRIPT_DIR}/${prefix}_values.tsv" \
    --output-summary "${SCRIPT_DIR}/${prefix}_summary.tsv"
done

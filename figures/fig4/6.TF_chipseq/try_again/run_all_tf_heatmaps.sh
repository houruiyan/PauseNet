#!/usr/bin/env bash
# Draw 4-TF ChIP-seq heatmaps (PATZ1/ZNF610/SP1/SP2) for count P2/P7/P8/P10.
# Rows sorted by each pattern's matched TF central signal; +-500 bp window.
set -u
cd "$(dirname "$0")"
PY=/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python
SCRIPT=../chipseq/plot_tf_seqlet_chip_heatmap.py

run() {  # pattern_idx matched_tf
  $PY "$SCRIPT" --task count --pattern-index "$1" --matched-tf "$2" \
      --tfs PATZ1 ZNF610 SP1 SP2 --half-win 500 --bin 10 \
      --out "count_P${1}_allTF_chip_heatmap.pdf" --preview-png
}

run 2  PATZ1
run 7  PATZ1
run 8  SP1
run 10 PATZ1
echo ALL_DONE

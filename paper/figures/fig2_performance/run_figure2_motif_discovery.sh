#!/usr/bin/env bash
# Run Figure 2 expected-gradient attribution and TF-MoDISco motif discovery.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 DATA_DIR CHECKPOINT OUTPUT_DIR [DEVICE]" >&2
  exit 2
fi

DATA_DIR=$1
CHECKPOINT=$2
OUTPUT_DIR=$3
DEVICE=${4:-cuda:0}
PYTHON_BIN=${PYTHON_BIN:-python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ATTRIBUTION_DIR="$OUTPUT_DIR/deepshap"
mkdir -p "$ATTRIBUTION_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/01_compute_deepshap.py" \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$ATTRIBUTION_DIR" \
  --split test \
  --background-split train \
  --device "$DEVICE" \
  --batch-size 8 \
  --num-backgrounds 4 \
  --num-steps 8 \
  --background-pool-size 2048 \
  --tasks count profile \
  --overwrite

for TASK in count profile; do
  "$PYTHON_BIN" "$SCRIPT_DIR/02_run_tfmodisco.py" \
    --task "$TASK" \
    --data-dir "$DATA_DIR" \
    --contrib-dir "$ATTRIBUTION_DIR" \
    --output-dir "$OUTPUT_DIR/tfmodisco/$TASK" \
    --split test \
    --max-samples 1000 \
    --sample-selection top_abs_contrib \
    --sliding-window-size 21 \
    --flank-size 10 \
    --target-seqlet-fdr 0.2 \
    --min-passing-windows-frac 0.03 \
    --max-passing-windows-frac 0.2 \
    --max-seqlets-per-metacluster 3000 \
    --min-metacluster-size 100 \
    --n-leiden-runs 5 \
    --nearest-neighbors-to-compute 150 \
    --final-min-cluster-size 20 \
    --verbose
done

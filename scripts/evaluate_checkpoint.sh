#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:?Usage: scripts/evaluate_checkpoint.sh DATA_DIR CHECKPOINT OUTPUT_DIR [SPLIT]}
CHECKPOINT=${2:?Usage: scripts/evaluate_checkpoint.sh DATA_DIR CHECKPOINT OUTPUT_DIR [SPLIT]}
OUTPUT_DIR=${3:?Usage: scripts/evaluate_checkpoint.sh DATA_DIR CHECKPOINT OUTPUT_DIR [SPLIT]}
SPLIT=${4:-test}

pausenet evaluate \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT"

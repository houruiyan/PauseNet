#!/usr/bin/env bash
set -euo pipefail
PY=/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python
BASE=/mnt/HDD8TB/houruiyan/pausing_site/data/NET_seq/HEK293T/final/tfmodisco
OUT=$BASE/results/smoke_count_top1000
LOG=$BASE/logs/smoke_count_top1000.log
mkdir -p "$OUT" "$(dirname "$LOG")"
"$PY" "$BASE/code/run_tfmodisco_from_contribs.py" \
  --task count \
  --output-dir "$OUT" \
  --max-samples 1000 \
  --sample-selection top_abs_contrib \
  --n-leiden-runs 3 \
  --nearest-neighbors-to-compute 100 \
  --max-seqlets-per-metacluster 2000 \
  --verbose \
  > "$LOG" 2>&1

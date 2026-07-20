#!/usr/bin/env bash
set -euo pipefail
PY=/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python
BASE=/mnt/HDD8TB/houruiyan/pausing_site/data/NET_seq/HEK293T/final/tfmodisco
mkdir -p "$BASE/results" "$BASE/logs"
for TASK in count profile; do
  OUT=$BASE/results/${TASK}_top5000_test_bg4_steps8
  LOG=$BASE/logs/${TASK}_top5000_test_bg4_steps8.log
  mkdir -p "$OUT"
  echo "[$(date)] Starting TF-MoDISco top5000 for ${TASK}" | tee "$LOG"
  "$PY" "$BASE/code/run_tfmodisco_from_contribs.py" \
    --task "$TASK" \
    --output-dir "$OUT" \
    --max-samples 5000 \
    --sample-selection top_abs_contrib \
    --sliding-window-size 21 \
    --flank-size 10 \
    --target-seqlet-fdr 0.2 \
    --min-passing-windows-frac 0.03 \
    --max-passing-windows-frac 0.2 \
    --max-seqlets-per-metacluster 10000 \
    --min-metacluster-size 100 \
    --n-leiden-runs 10 \
    --nearest-neighbors-to-compute 300 \
    --final-min-cluster-size 20 \
    --verbose \
    >> "$LOG" 2>&1
  echo "[$(date)] Finished TF-MoDISco top5000 for ${TASK}" | tee -a "$LOG"
done

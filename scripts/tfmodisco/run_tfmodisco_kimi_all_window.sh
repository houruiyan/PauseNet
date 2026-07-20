#!/usr/bin/env bash
set -euo pipefail
PY=/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python
SRC=/mnt/HDD8TB/houruiyan/pausing_site/data/NET_seq/HEK293T/final/tfmodisco
BASE=/mnt/HDD8TB/houruiyan/pausing_site/data/NET_seq/HEK293T/final/tfmodisco_kimi_all_window
mkdir -p "$BASE/results" "$BASE/logs"
for TASK in count profile; do
  OUT=$BASE/results/${TASK}_all_test_bg4_steps8
  LOG=$BASE/logs/${TASK}_all_test_bg4_steps8.log
  mkdir -p "$OUT"
  echo "[$(date)] Starting TF-MoDISco all-window (n=57590) for ${TASK}" | tee "$LOG"
  "$PY" "$SRC/code/run_tfmodisco_from_contribs.py" \
    --task "$TASK" \
    --output-dir "$OUT" \
    --sample-selection all \
    --sliding-window-size 21 \
    --flank-size 10 \
    --target-seqlet-fdr 0.2 \
    --min-passing-windows-frac 0.03 \
    --max-passing-windows-frac 0.2 \
    --max-seqlets-per-metacluster 20000 \
    --min-metacluster-size 100 \
    --n-leiden-runs 20 \
    --nearest-neighbors-to-compute 500 \
    --final-min-cluster-size 20 \
    --verbose \
    >> "$LOG" 2>&1
  echo "[$(date)] Finished TF-MoDISco all-window for ${TASK}" | tee -a "$LOG"
done
echo "[$(date)] ALL DONE" | tee -a "$BASE/logs/all_window_run.status"

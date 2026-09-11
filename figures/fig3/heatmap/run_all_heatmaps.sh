#!/usr/bin/env bash
# Run plot_pattern_mfe_signal_heatmap.py over selected TF-MoDISco patterns:
#   profile pos_patterns : 6 7 8 13 19 35 37
#   count   pos_patterns : 2 4 5 6 7 8 9 10 11
# Outputs land in this folder as pattern<N>_mfe_signal_heatmap.{pdf,png}
# and count_pos_pattern<N>_mfe_signal_heatmap.{pdf,png}.
# Logs go to ./logs/; a failed pattern does not stop the loop.

set -u
cd "$(dirname "$0")"

PY=/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python
SCRIPT=plot_pattern_mfe_signal_heatmap.py
JOBS=${JOBS:-8}          # fold workers per pattern; override with: JOBS=16 bash run_all_heatmaps.sh
mkdir -p logs

PROFILE_PATTERNS="6 7 8 13 19 35 37"
COUNT_POS_PATTERNS="2 4 5 6 7 8 9 10 11"

run_one() {
    local tag="$1"; shift   # tag only used for the log file name
    local log="logs/${tag}.log"
    echo "[$(date '+%F %T')] START ${tag}"
    if "$PY" "$SCRIPT" --jobs "$JOBS" "$@" >"$log" 2>&1; then
        echo "[$(date '+%F %T')] OK    ${tag}"
    else
        echo "[$(date '+%F %T')] FAIL  ${tag} (see $log)"
    fi
}

for i in $PROFILE_PATTERNS; do
    run_one "profile_pattern_${i}" --task profile --pattern-index "$i"
done

for i in $COUNT_POS_PATTERNS; do
    run_one "count_pos_pattern_${i}" --task count --sign pos --pattern-index "$i"
done

echo "[$(date '+%F %T')] ALL DONE"

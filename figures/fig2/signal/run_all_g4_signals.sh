#!/usr/bin/env bash
# Run plot_profile_pattern_pausing_signal.py over selected TF-MoDISco patterns:
#   profile pos_patterns : 6 7 8 13 19 35 37
#   count   pos_patterns : 2 4 5 6 7 8 9 10 11
# Outputs land in this folder as profile_pattern_<N>_observed_predicted_signal.pdf
# and count_pos_pattern_<N>_observed_predicted_signal.pdf.
# Logs go to ./logs/; a failed pattern does not stop the loop.

set -u
cd "$(dirname "$0")"

PY=/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python
SCRIPT=plot_profile_pattern_pausing_signal.py
mkdir -p logs

PROFILE_PATTERNS="6 7 8 13 19 35 37"
COUNT_POS_PATTERNS="2 4 5 6 7 8 9 10 11"

run_one() {
    local tag="$1"; shift   # tag only used for the log file name
    local log="logs/${tag}.log"
    echo "[$(date '+%F %T')] START ${tag}"
    if "$PY" "$SCRIPT" "$@" >"$log" 2>&1; then
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

#!/usr/bin/env bash
# Run pausenet_pattern_knockout.py for all patterns of interest (fig2 panel).
#   cP = count-branch pattern, pP = profile-branch pattern.
# Each run writes <task>_pattern_<N>_knockout_summary.tsv and
# <task>_pattern_<N>_knockout_mean_profile.pdf/.png into this folder.
# Log per run: logs/<task>_pattern_<N>.log

set -u
cd "$(dirname "$0")"

PY=/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python
SCRIPT=pausenet_pattern_knockout.py

COUNT_PATTERNS=(2 4 5 6 7 8 9 10 11)
PROFILE_PATTERNS=(6 7 8 13 19 35 37)

mkdir -p logs

run_one() {
    local task=$1 idx=$2
    local tag="${task}_pattern_${idx}"
    echo "=== [$(date '+%H:%M:%S')] ${tag} ==="
    "$PY" "$SCRIPT" --task "$task" --pattern-index "$idx" \
        > "logs/${tag}.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "    OK  -> ${tag}_knockout_mean_profile.pdf"
    else
        echo "    FAILED (see logs/${tag}.log)"
    fi
}

for i in "${PROFILE_PATTERNS[@]}"; do run_one profile "$i"; done
for i in "${COUNT_PATTERNS[@]}";   do run_one count   "$i"; done

echo "=== [$(date '+%H:%M:%S')] all done ==="

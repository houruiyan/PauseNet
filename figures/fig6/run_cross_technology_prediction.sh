#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /mnt/HDD8TB/houruiyan/env/conda_envs/testPauseNet/bin/python -u \
  "$SCRIPT_DIR/plot_cross_technology_prediction.py" --device auto --batch-size 128 --num-workers 2 "$@"

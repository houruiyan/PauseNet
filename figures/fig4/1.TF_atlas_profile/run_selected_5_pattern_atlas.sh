#!/usr/bin/env bash
set -euo pipefail

PYTHON="/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/plot_selected_5_pattern_atlas.py"

"${PYTHON}" "${SCRIPT}" --outdir "${HERE}"

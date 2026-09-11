#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_tf_pearson_r_lollipop.py"

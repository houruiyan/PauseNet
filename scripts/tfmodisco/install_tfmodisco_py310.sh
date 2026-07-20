#!/usr/bin/env bash
set -euo pipefail
PY=/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python
PIP=/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/pip
LOG=/mnt/HDD8TB/houruiyan/pausing_site/data/NET_seq/HEK293T/final/tfmodisco/logs/install_tfmodisco_py310.log
{
  echo "[$(date)] Installing TF-MoDISco tooling into PY310"
  "$PY" --version
  "$PIP" --version
  "$PIP" install --upgrade h5py modisco-lite==2.4.0
  "$PY" - <<'PY'
import importlib, sys
for m in ['h5py', 'modiscolite', 'numpy', 'scipy', 'sklearn']:
    mod = importlib.import_module(m)
    print(m, getattr(mod, '__version__', 'version_unknown'))
PY
  echo "[$(date)] Done"
} 2>&1 | tee "$LOG"

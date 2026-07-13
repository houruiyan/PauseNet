#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/hek293t_netseq.yaml}

pausenet train --config "$CONFIG"

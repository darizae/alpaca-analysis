#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_analysis_common.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <analysis-id> <launch-dir> [config-path]" >&2
  exit 1
fi

ANALYSIS_ID="$1"
LAUNCH_DIR="$2"
CONFIG_PATH="${3:-$DEFAULT_CONFIG_PATH}"

RESOLVED=()
while IFS= read -r line; do
  RESOLVED+=("$line")
done < <("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH" "$LAUNCH_DIR"
import sys
from pathlib import Path

from stv_analysis.framework import remote_launch_dir_for_paths
from stv_backend.config import load_config

config = load_config(Path(sys.argv[1]))
launch_dir = Path(sys.argv[2]).resolve()
print(config.analysis_hpc.host)
print(remote_launch_dir_for_paths(config, launch_dir))
print(launch_dir)
PY
)

HOST="${HOST_OVERRIDE:-${RESOLVED[0]}}"
REMOTE_LAUNCH_DIR="${REMOTE_LAUNCH_DIR_OVERRIDE:-${RESOLVED[1]}}"
LOCAL_LAUNCH_DIR="${RESOLVED[2]}"

if [[ ! -d "$LOCAL_LAUNCH_DIR" ]]; then
  echo "Local launch dir does not exist: $LOCAL_LAUNCH_DIR" >&2
  exit 1
fi

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "mkdir -p $(printf '%q' "$REMOTE_LAUNCH_DIR")"
rsync -avP \
  --include "base_snapshot/" \
  --include "base_snapshot/***" \
  --include "configs/" \
  --include "configs/***" \
  --include "hpc_bundle/" \
  --include "hpc_bundle/***" \
  --include "specs/" \
  --include "specs/***" \
  --exclude "*" \
  "${LOCAL_LAUNCH_DIR}/" "${HOST}:${REMOTE_LAUNCH_DIR}/"

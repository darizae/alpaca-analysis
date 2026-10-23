#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_analysis_common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <local-run-dir> [config-path] [--mode status|full]" >&2
  exit 1
fi

LOCAL_RUN_DIR_INPUT="$1"
shift

CONFIG_PATH="$DEFAULT_CONFIG_PATH"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG_PATH="$1"
  shift
fi

MODE="status"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    *)
      echo "Usage: $0 <local-run-dir> [config-path] [--mode status|full]" >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "status" && "$MODE" != "full" ]]; then
  echo "Unsupported mode: $MODE" >&2
  exit 1
fi

RESOLVED=()
while IFS= read -r line; do
  RESOLVED+=("$line")
done < <("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH" "$LOCAL_RUN_DIR_INPUT"
import sys
from pathlib import Path

from stv_backend.config import load_config

config = load_config(Path(sys.argv[1]))
local_run_dir = Path(sys.argv[2]).resolve()
repo_root = Path.cwd().resolve()
relative = local_run_dir.relative_to(repo_root)
print(config.analysis_hpc.host)
print(config.analysis_hpc.remote_repo_root.rstrip("/") + "/" + relative.as_posix())
print(local_run_dir)
PY
)

HOST="${HOST_OVERRIDE:-${RESOLVED[0]}}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR_OVERRIDE:-${RESOLVED[1]}}"
LOCAL_RUN_DIR="${RESOLVED[2]}"

mkdir -p "$LOCAL_RUN_DIR"

if [[ "$MODE" == "full" ]]; then
  rsync -avP "${HOST}:${REMOTE_RUN_DIR}/" "${LOCAL_RUN_DIR}/"
else
  rsync -avP \
    --include "*/" \
    --include "input_manifest.json" \
    --include "run_summary.json" \
    --include "hpc_bundle/submission.json" \
    --include "hpc_bundle/logs/***" \
    --include "full/**/tables/*manifest*.csv" \
    --include "full/**/cache/*progress*.csv" \
    --include "full/**/cache/**/*progress*.csv" \
    --include "smoke/**/tables/*manifest*.csv" \
    --include "smoke/**/cache/*progress*.csv" \
    --include "smoke/**/cache/**/*progress*.csv" \
    --exclude "*" \
    "${HOST}:${REMOTE_RUN_DIR}/" "${LOCAL_RUN_DIR}/"
fi

if [[ -f "${LOCAL_RUN_DIR}/hpc_bundle/submission.json" ]]; then
  cat "${LOCAL_RUN_DIR}/hpc_bundle/submission.json"
fi
if [[ -f "${LOCAL_RUN_DIR}/run_summary.json" ]]; then
  echo
  cat "${LOCAL_RUN_DIR}/run_summary.json"
fi

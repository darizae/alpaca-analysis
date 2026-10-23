#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_analysis_common.sh"

CONFIG_PATH="${1:-$DEFAULT_CONFIG_PATH}"

RESOLVED=()
while IFS= read -r line; do
  RESOLVED+=("$line")
done < <("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH"
import sys
from pathlib import Path

from stv_backend.config import load_config

config = load_config(Path(sys.argv[1]))
print(config.analysis_hpc.host)
print(config.analysis_hpc.remote_repo_root)
PY
)

HOST="${HOST_OVERRIDE:-${RESOLVED[0]}}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT_OVERRIDE:-${RESOLVED[1]}}"

if [[ -z "$REMOTE_REPO_ROOT" ]]; then
  echo "Missing analysis_hpc.remote_repo_root in $CONFIG_PATH" >&2
  exit 1
fi

rsync -avP \
  --exclude ".git" \
  --exclude ".claude" \
  --exclude ".idea" \
  --exclude ".venv" \
  --exclude "backend/.venv" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude "*.pyo" \
  --exclude "backend/.pytest_cache" \
  --exclude "backend/tests/__pycache__" \
  --exclude "data/" \
  --exclude "backend/data/cache/" \
  --exclude "backend/data/processed/" \
  ./ "${HOST}:${REMOTE_REPO_ROOT}/"

#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG_PATH="${1:-config/hpc_full_grid.yaml}"
HOST="${HOST_OVERRIDE:-$("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH"
import sys
import yaml
from pathlib import Path
raw = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print((raw.get("hpc", {}) or {}).get("host", "hpc-cluster"))
PY
)}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT_OVERRIDE:-$("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH"
import sys
import yaml
from pathlib import Path
raw = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print((raw.get("hpc", {}) or {}).get("remote_repo_root", ""))
PY
)}"

if [[ -z "$REMOTE_REPO_ROOT" ]]; then
  echo "Missing hpc.remote_repo_root in $CONFIG_PATH" >&2
  exit 1
fi

rsync -avP \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude ".pytest_cache" \
  --exclude "__pycache__" \
  --exclude "artifacts/dedup_full_corpus/" \
  ./ "${HOST}:${REMOTE_REPO_ROOT}/"

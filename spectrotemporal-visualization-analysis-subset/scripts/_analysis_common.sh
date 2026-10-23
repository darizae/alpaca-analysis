#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "This script requires bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/backend"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-${BACKEND_DIR}/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Expected Python interpreter at $PYTHON_BIN" >&2
  echo "Create the backend venv first, then rerun." >&2
  exit 1
fi

DEFAULT_CONFIG_PATH="${BACKEND_DIR}/config/default.toml"


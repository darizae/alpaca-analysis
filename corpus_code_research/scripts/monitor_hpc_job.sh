#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG_PATH="${1:-config/hpc_full_grid.yaml}"
JOB_ID="${2:-}"

if [[ -z "$JOB_ID" ]]; then
  echo "Usage: $0 <config-path> <job-id> [--stage <stage>]" >&2
  exit 1
fi

shift 2 || true
"$PYTHON_BIN" -m corpus_dedup.cli show-hpc-status --config "$CONFIG_PATH" --job-id "$JOB_ID" "$@"

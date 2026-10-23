#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG_PATH="${1:-config/hpc_full_grid.yaml}"
shift || true

"$PYTHON_BIN" -m corpus_dedup.cli show-hpc-status --config "$CONFIG_PATH" "$@"

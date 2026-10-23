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

"$PYTHON_BIN" -m stv_analysis.cli --config "$CONFIG_PATH" launch status "$ANALYSIS_ID" --launch-dir "$LAUNCH_DIR"


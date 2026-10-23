#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG_PATH="${1:-config/hpc_full_grid.yaml}"
SOURCE_LAUNCH="${2:-}"
shift 2 || true

FORCE_FLAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE_FLAG="--force"
      shift
      ;;
    *)
      echo "Usage: $0 <config-path> <source-launch-dir-or-label> [--force]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SOURCE_LAUNCH" ]]; then
  echo "Source launch is required." >&2
  exit 1
fi

RESOLVED=()
while IFS= read -r line; do
  RESOLVED+=("$line")
done < <("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH" "$SOURCE_LAUNCH"
import sys
from pathlib import Path, PurePosixPath

from corpus_dedup.config import load_config
from corpus_dedup.features import launch_paths_for_config
from corpus_dedup.hpc import _remote_launch_dir

config = load_config(Path(sys.argv[1]))
paths = launch_paths_for_config(config)
source_arg = sys.argv[2].strip()

repo_root = Path.cwd().resolve()
source_value = Path(source_arg)
if source_value.is_absolute() or "/" in source_arg:
    source_local_launch_dir = source_value.resolve()
else:
    source_local_launch_dir = (config.artifact_root / source_arg).resolve()

try:
    source_relative_launch_path = source_local_launch_dir.relative_to(repo_root)
except ValueError:
    source_remote_launch_dir = str(
        PurePosixPath(config.hpc.remote_repo_root)
        / config.artifact_root.name
        / source_local_launch_dir.name
    )
else:
    source_remote_launch_dir = str(
        PurePosixPath(config.hpc.remote_repo_root) / PurePosixPath(source_relative_launch_path.as_posix())
    )

print(config.hpc.host)
print(config.hpc.remote_repo_root)
print(config.hpc.remote_venv_path)
print(_remote_launch_dir(config, paths))
print(source_remote_launch_dir)
PY
)

HOST="${HOST_OVERRIDE:-${RESOLVED[0]}}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT_OVERRIDE:-${RESOLVED[1]}}"
REMOTE_VENV_PATH="${REMOTE_VENV_PATH_OVERRIDE:-${RESOLVED[2]}}"
TARGET_REMOTE_LAUNCH_DIR="${TARGET_REMOTE_LAUNCH_DIR_OVERRIDE:-${RESOLVED[3]}}"
SOURCE_REMOTE_LAUNCH_DIR="${SOURCE_REMOTE_LAUNCH_DIR_OVERRIDE:-${RESOLVED[4]}}"

if [[ -z "$REMOTE_REPO_ROOT" || -z "$REMOTE_VENV_PATH" ]]; then
  echo "Missing remote repo root or remote venv path in $CONFIG_PATH" >&2
  exit 1
fi

ssh -oBatchMode=yes -oConnectTimeout=15 "$HOST" "bash -lc 'set -euo pipefail; source \"$REMOTE_VENV_PATH/bin/activate\"; cd \"$REMOTE_REPO_ROOT\"; python -m corpus_dedup.cli seed-feature-artifacts --launch-dir \"$TARGET_REMOTE_LAUNCH_DIR\" --source-launch-dir \"$SOURCE_REMOTE_LAUNCH_DIR\" $FORCE_FLAG'"

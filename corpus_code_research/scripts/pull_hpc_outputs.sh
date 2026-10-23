#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG_PATH="${1:-config/hpc_full_grid.yaml}"
shift || true

MODE="launch"
LAUNCH_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --launch)
      LAUNCH_ARG="${2:-}"
      shift 2
      ;;
    *)
      echo "Usage: $0 <config-path> [--mode status|launch|all] [--launch <launch-dir-or-label>]" >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "status" && "$MODE" != "launch" && "$MODE" != "all" ]]; then
  echo "Unsupported mode: $MODE" >&2
  exit 1
fi

RESOLVED=()
while IFS= read -r line; do
  RESOLVED+=("$line")
done < <("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH" "$LAUNCH_ARG"
import sys
from pathlib import Path, PurePosixPath

from corpus_dedup.config import load_config
from corpus_dedup.features import launch_paths_for_config
from corpus_dedup.hpc import _remote_launch_dir

config = load_config(Path(sys.argv[1]))
paths = launch_paths_for_config(config)
launch_arg = sys.argv[2].strip()

if launch_arg:
    launch_value = Path(launch_arg)
    if launch_value.is_absolute() or "/" in launch_arg:
        local_launch_dir = launch_value.resolve()
    else:
        local_launch_dir = (config.artifact_root / launch_arg).resolve()
else:
    local_launch_dir = paths.launch_dir.resolve()

repo_root = Path.cwd().resolve()
if local_launch_dir == paths.launch_dir.resolve():
    remote_launch_dir = _remote_launch_dir(config, paths)
else:
    try:
        relative_launch_path = local_launch_dir.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit(
            f"Launch dir must be inside the repo when overriding the default launch: {local_launch_dir}"
        ) from exc
    remote_launch_dir = str(PurePosixPath(config.hpc.remote_repo_root) / PurePosixPath(relative_launch_path.as_posix()))

print(config.hpc.host)
print(config.hpc.remote_repo_root)
print(config.artifact_root.resolve())
print(local_launch_dir)
print(remote_launch_dir)
PY
)

HOST="${HOST_OVERRIDE:-${RESOLVED[0]}}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT_OVERRIDE:-${RESOLVED[1]}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT_OVERRIDE:-${RESOLVED[2]}}"
LOCAL_LAUNCH_DIR="${LAUNCH_DIR_OVERRIDE:-${RESOLVED[3]}}"
REMOTE_LAUNCH_DIR="${REMOTE_LAUNCH_DIR_OVERRIDE:-${RESOLVED[4]}}"

if [[ -z "$REMOTE_REPO_ROOT" ]]; then
  echo "Missing hpc.remote_repo_root in $CONFIG_PATH" >&2
  exit 1
fi

if [[ "$MODE" == "all" ]]; then
  mkdir -p "$ARTIFACT_ROOT"
  rsync -avP "${HOST}:${REMOTE_REPO_ROOT}/artifacts/dedup_full_corpus/" "${ARTIFACT_ROOT}/"
  exit 0
fi

mkdir -p "$LOCAL_LAUNCH_DIR"
if [[ "$MODE" == "launch" ]]; then
  rsync -avP "${HOST}:${REMOTE_LAUNCH_DIR}/" "${LOCAL_LAUNCH_DIR}/"
  exit 0
fi

rsync -avP \
  --include "configs/" \
  --include "configs/launch_config.yaml" \
  --include "progress/" \
  --include "progress/***" \
  --include "hpc_bundle/" \
  --include "hpc_bundle/*/" \
  --include "hpc_bundle/*/submission.json" \
  --include "hpc_bundle/*/logs/" \
  --include "hpc_bundle/*/logs/***" \
  --exclude "*" \
  "${HOST}:${REMOTE_LAUNCH_DIR}/" "${LOCAL_LAUNCH_DIR}/"

#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_analysis_common.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <analysis-id> <launch-dir> [config-path] [--start-stage <stage>]" >&2
  exit 1
fi

ANALYSIS_ID="$1"
LAUNCH_DIR="$2"
shift 2

CONFIG_PATH="$DEFAULT_CONFIG_PATH"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG_PATH="$1"
  shift
fi

START_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-stage)
      START_STAGE="${2:-}"
      shift 2
      ;;
    *)
      echo "Usage: $0 <analysis-id> <launch-dir> [config-path] [--start-stage <stage>]" >&2
      exit 1
      ;;
  esac
done

"$PYTHON_BIN" -m stv_analysis.cli --config "$CONFIG_PATH" launch prepare-hpc "$ANALYSIS_ID" --launch-dir "$LAUNCH_DIR" >/dev/null
"${SCRIPT_DIR}/analysis_sync_repo_to_hpc.sh" "$CONFIG_PATH"
"${SCRIPT_DIR}/analysis_sync_launch_to_hpc.sh" "$ANALYSIS_ID" "$LAUNCH_DIR" "$CONFIG_PATH"

CMD=("$PYTHON_BIN" -m stv_analysis.cli --config "$CONFIG_PATH" launch submit-hpc "$ANALYSIS_ID" --launch-dir "$LAUNCH_DIR")
if [[ -n "$START_STAGE" ]]; then
  CMD+=(--start-stage "$START_STAGE")
fi
"${CMD[@]}"

FAILOVER_ARGS_FILE="$(mktemp)"
trap 'rm -f "$FAILOVER_ARGS_FILE"' EXIT
"$PYTHON_BIN" - "$LAUNCH_DIR" "$START_STAGE" > "$FAILOVER_ARGS_FILE" <<'PY'
import json
import sys
from pathlib import Path

launch_dir = Path(sys.argv[1]).resolve()
start_stage = sys.argv[2]
manifest = json.loads((launch_dir / "configs" / "launch_manifest.json").read_text(encoding="utf8"))
stage_specs = list(manifest.get("stage_specs", []))
stage_ids = [str(item["stage_id"]) for item in stage_specs]
failover_profiles = {"robustness", "model_fit"}
targets = [str(item["stage_id"]) for item in stage_specs if str(item.get("resource_profile")) in failover_profiles]
if not targets:
    raise SystemExit(0)
target_stage = targets[0]
if start_stage:
    if start_stage not in stage_ids:
        raise SystemExit(0)
    if stage_ids.index(start_stage) > stage_ids.index(target_stage):
        raise SystemExit(0)

bundle_dir = launch_dir / "hpc_bundle"
required = stage_ids[stage_ids.index(target_stage):]
payloads = {}
for stage_id in required:
    path = bundle_dir / stage_id / "submission.json"
    if not path.is_file():
        raise SystemExit(0)
    payloads[stage_id] = json.loads(path.read_text(encoding="utf8"))

primary_job_id = str(payloads[target_stage].get("job_id", "")).strip()
downstream_job_ids = [
    str(payloads[stage_id].get("job_id", "")).strip()
    for stage_id in required[1:]
]
if not primary_job_id or any(not job_id for job_id in downstream_job_ids):
    raise SystemExit(0)

existing_path = bundle_dir / f"failover_{target_stage}" / "submission.json"
if existing_path.is_file():
    existing = json.loads(existing_path.read_text(encoding="utf8"))
    if str(existing.get("primary_job_id", "")).strip() == primary_job_id:
        raise SystemExit(0)

print(target_stage)
print(primary_job_id)
for job_id in downstream_job_ids:
    print(job_id)
PY

FAILOVER_JOB_ARGS=()
while IFS= read -r line; do
  FAILOVER_JOB_ARGS+=("$line")
done < "$FAILOVER_ARGS_FILE"

if [[ ${#FAILOVER_JOB_ARGS[@]} -gt 0 ]]; then
  "${SCRIPT_DIR}/analysis_submit_hpc_failover.sh" "$ANALYSIS_ID" "$LAUNCH_DIR" "${FAILOVER_JOB_ARGS[@]}"
fi

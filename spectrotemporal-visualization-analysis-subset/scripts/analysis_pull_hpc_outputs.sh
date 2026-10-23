#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_analysis_common.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <analysis-id> <launch-dir> [config-path] [--mode status|outputs|launch]" >&2
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

MODE="launch"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    *)
      echo "Usage: $0 <analysis-id> <launch-dir> [config-path] [--mode status|outputs|launch]" >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "status" && "$MODE" != "outputs" && "$MODE" != "launch" ]]; then
  echo "Unsupported mode: $MODE" >&2
  exit 1
fi

RESOLVED=()
while IFS= read -r line; do
  RESOLVED+=("$line")
done < <("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH" "$LAUNCH_DIR"
import sys
from pathlib import Path

from stv_analysis.framework import remote_launch_dir_for_paths
from stv_backend.config import load_config

config = load_config(Path(sys.argv[1]))
launch_dir = Path(sys.argv[2]).resolve()
print(config.analysis_hpc.host)
print(remote_launch_dir_for_paths(config, launch_dir))
print(launch_dir)
PY
)

HOST="${HOST_OVERRIDE:-${RESOLVED[0]}}"
REMOTE_LAUNCH_DIR="${REMOTE_LAUNCH_DIR_OVERRIDE:-${RESOLVED[1]}}"
LOCAL_LAUNCH_DIR="${RESOLVED[2]}"

mkdir -p "$LOCAL_LAUNCH_DIR"

if [[ "$MODE" == "outputs" || "$MODE" == "launch" ]]; then
  rsync -avP "${HOST}:${REMOTE_LAUNCH_DIR}/" "${LOCAL_LAUNCH_DIR}/"
  "$PYTHON_BIN" - <<'PY' "$LOCAL_LAUNCH_DIR" "$ANALYSIS_ID" "$MODE"
import json
import sys
from pathlib import Path

launch_dir = Path(sys.argv[1]).resolve()
analysis_id = sys.argv[2]
mode = sys.argv[3]
progress_path = launch_dir / "progress" / "launch_progress.json"
if not progress_path.is_file():
    raise SystemExit(f"refusing to promote launch without progress: {progress_path}")
progress = json.loads(progress_path.read_text(encoding="utf8"))
stages = progress.get("stages", {})
if progress.get("stage_status") != "completed" or not stages or any(
    row.get("status") != "completed" for row in stages.values()
):
    raise SystemExit("refusing to promote incomplete launch")

manifest_path = launch_dir / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"refusing to promote launch without manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf8"))
if manifest.get("analysis_id") != analysis_id:
    raise SystemExit("launch manifest analysis_id does not match requested analysis")
for name, artifact in manifest.get("artifacts", {}).items():
    artifact_path = launch_dir / str(artifact.get("path", ""))
    if artifact.get("canonical_status") != "current-canonical" or not artifact_path.is_file():
        raise SystemExit(f"invalid canonical artifact {name}: {artifact_path}")

if analysis_id == "rq1-post-hum-cohesion":
    launch_manifest = json.loads(
        (launch_dir / "configs" / "launch_manifest.json").read_text(encoding="utf8")
    )
    required_tables = {
        "model_input.parquet",
        "model_input_summary.csv",
        "matched_control_windows.parquet",
        "matched_control_windows_summary.csv",
        "matching_feasibility.csv",
        "baseline_covariates.parquet",
        "baseline_covariates_summary.csv",
        "window_inventory.parquet",
        "observed_centroid_distance.parquet",
        "observed_centroid_distance_summary.csv",
        "rq1_directional_movement_window_level.csv",
        "rq1_directional_movement_summary.csv",
        "rq1_directional_movement_outcome_summary.csv",
        "filtering_attrition.csv",
        "hard_gps_filter_summary.csv",
        "rq1_artifact_freshness_and_reuse.csv",
        "rq1_final_output_index.csv",
    }
    missing = sorted(name for name in required_tables if not (launch_dir / "tables" / name).is_file())
    required_reports = {
        "rq1_model_input_build_report.md",
        "README.md",
    }
    missing.extend(
        sorted(name for name in required_reports if not (launch_dir / "reports" / name).is_file())
    )
    if missing:
        raise SystemExit("refusing to promote RQ1 launch with missing artifacts: " + ", ".join(missing))
    pd = __import__("pandas")
    model_input = pd.read_parquet(launch_dir / "tables" / "model_input.parquet")
    if set(pd.to_numeric(model_input["condition"], errors="coerce").dropna().astype(int).unique()) != {0, 1}:
        raise SystemExit("refusing to promote RQ1 launch with non-binary condition encoding")
    required_directional_columns = {
        "analysis_window_id", "condition", "window_s", "focal_id", "session_id",
        "pre_distance_m", "post_distance_m", "delta_D_m", "focal_distance_reduction_m",
        "centroid_distance_reduction_m", "directional_category",
        "directional_category_clean", "directional_geometry_valid", "directional_geometry_invalid_reason",
    }
    directional = pd.read_csv(launch_dir / "tables" / "rq1_directional_movement_window_level.csv")
    if not required_directional_columns.issubset(directional.columns):
        raise SystemExit("refusing to promote RQ1 launch with an invalid directional movement table")
    if mode == "launch" and launch_manifest.get("profile") != "full":
        raise SystemExit("refusing to promote a non-full RQ1 launch")

if mode == "launch":
    latest = launch_dir.parent.parent / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_dir() and not latest.is_symlink():
            raise SystemExit(f"refusing to replace non-symlink latest directory: {latest}")
        latest.unlink()
    latest.symlink_to(Path("launches") / launch_dir.name)
PY
  exit 0
fi

rsync -avP \
  --include "configs/" \
  --include "configs/***" \
  --include "progress/" \
  --include "progress/***" \
  --include "hpc_bundle/" \
  --include "hpc_bundle/*/" \
  --include "hpc_bundle/*/submission.json" \
  --include "hpc_bundle/*/logs/" \
  --include "hpc_bundle/*/logs/***" \
  --exclude "*" \
  "${HOST}:${REMOTE_LAUNCH_DIR}/" "${LOCAL_LAUNCH_DIR}/"

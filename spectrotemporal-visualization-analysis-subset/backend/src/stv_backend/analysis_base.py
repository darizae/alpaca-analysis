from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from stv_core.geometry import compute_bbox

from .config import BackendConfig
from .source_index import sidecar_cache_path


ANALYSIS_BASE_VERSION = 1
OUTPUT_DIRNAME = "base"


def build_analysis_base(
    config: BackendConfig,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    artifact_dir = (output_dir or (config.processed_dir / "analysis" / OUTPUT_DIRNAME)).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    recorders = pd.read_parquet(
        config.intermediate_dir / "recorders.parquet",
        columns=["recorder_id", "recorder_display_name", "device_id"],
    ).sort_values(["recorder_id"]).reset_index(drop=True)
    segments = pd.read_parquet(
        config.intermediate_dir / "segments.parquet",
        columns=[
            "segment_id",
            "recorder_id",
            "recorder_display_name",
            "wav_basename",
            "csv_hpc_path",
            "sample_rate_estimate",
            "session_id",
            "day_session_id",
            "date_local",
        ],
    ).sort_values(["date_local", "recorder_id", "segment_id"]).reset_index(drop=True)
    events = pd.read_parquet(
        config.intermediate_dir / "events.parquet",
        columns=[
            "review_row_key",
            "recorder_id",
            "recorder_display_name",
            "device_id",
            "reviewed_resolved_label_semantic",
            "timestamp_start_local",
            "timestamp_end_local",
            "date_local",
            "session_id",
            "day_session_id",
            "seconds_since_midnight_local",
        ],
    ).sort_values(["date_local", "seconds_since_midnight_local", "review_row_key"]).reset_index(drop=True)
    trace_points_filtered = pd.read_parquet(
        config.intermediate_dir / "trace_points.parquet",
        columns=[
            "segment_id",
            "point_index_in_segment",
            "recorder_id",
            "recorder_display_name",
            "wav_basename",
            "timestamp_utc",
            "timestamp_local",
            "date_local",
            "seconds_since_midnight_local",
            "latitude",
            "longitude",
            "gps_status",
            "timer_count",
            "buffers_filled",
            "buffers_written",
            "sample_rate_estimate",
            "session_id",
            "day_session_id",
        ],
    ).sort_values(["date_local", "recorder_id", "timestamp_local", "point_index_in_segment"]).reset_index(drop=True)

    sidecar_index = segments[
        ["segment_id", "recorder_id", "wav_basename", "date_local", "session_id", "day_session_id", "csv_hpc_path"]
    ].copy()
    sidecar_index["cached_csv_path"] = [
        str(sidecar_cache_path(config.sidecar_cache_dir, str(csv_hpc_path)))
        if sidecar_cache_path(config.sidecar_cache_dir, str(csv_hpc_path)) is not None
        else None
        for csv_hpc_path in sidecar_index["csv_hpc_path"]
    ]
    sidecar_index = sidecar_index.sort_values(["date_local", "recorder_id", "segment_id"]).reset_index(drop=True)

    recorders_path = artifact_dir / "recorders.parquet"
    segments_path = artifact_dir / "segments.parquet"
    events_path = artifact_dir / "events.parquet"
    trace_points_path = artifact_dir / "trace_points_filtered.parquet"
    sidecar_index_path = artifact_dir / "sidecar_index.parquet"
    gps_blocker_rule_path = artifact_dir / "gps_blocker_rule.json"
    manifest_path = artifact_dir / "manifest.json"

    recorders.to_parquet(recorders_path, index=False)
    segments.to_parquet(segments_path, index=False)
    events.to_parquet(events_path, index=False)
    trace_points_filtered.to_parquet(trace_points_path, index=False)
    sidecar_index.to_parquet(sidecar_index_path, index=False)
    gps_blocker_rule_path.write_text(config.gps_blocker_path.read_text(encoding="utf8"), encoding="utf8")

    bbox = compute_bbox(
        list(
            zip(
                trace_points_filtered["longitude"].astype(float).tolist(),
                trace_points_filtered["latitude"].astype(float).tolist(),
                strict=False,
            )
        )
    )
    manifest = {
        "bundle_name": "analysis_base",
        "bundle_version": ANALYSIS_BASE_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "git_commit": _git_commit(config.config_path.resolve().parents[2]),
        "dataset_name": config.dataset_name,
        "timezone": config.timezone_name,
        "config_path": str(config.config_path),
        "artifact_dir": str(artifact_dir),
        "source_intermediate_dir": str(config.intermediate_dir),
        "source_dataset_manifest_path": str((config.processed_dir / "manifest.json").resolve()),
        "gps_blocker_rule_path": str(gps_blocker_rule_path),
        "bbox": bbox,
        "tables": {
            "recorders": {
                "path": str(recorders_path),
                "rows": int(len(recorders.index)),
                "primary_key": ["recorder_id"],
            },
            "segments": {
                "path": str(segments_path),
                "rows": int(len(segments.index)),
                "primary_key": ["segment_id"],
            },
            "events": {
                "path": str(events_path),
                "rows": int(len(events.index)),
                "primary_key": ["review_row_key"],
            },
            "trace_points_filtered": {
                "path": str(trace_points_path),
                "rows": int(len(trace_points_filtered.index)),
            },
            "sidecar_index": {
                "path": str(sidecar_index_path),
                "rows": int(len(sidecar_index.index)),
                "primary_key": ["segment_id"],
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return {
        "analysis_base_dir": str(artifact_dir),
        "manifest_path": str(manifest_path),
        "recorders": int(len(recorders.index)),
        "segments": int(len(segments.index)),
        "events": int(len(events.index)),
        "trace_points_filtered": int(len(trace_points_filtered.index)),
        "sidecar_index_rows": int(len(sidecar_index.index)),
    }


def _git_commit(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None

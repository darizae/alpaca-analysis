from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from .contracts import BaseAnalysisBundle
from ..paths import repo_root


def load_base_bundle(manifest_path: Path) -> BaseAnalysisBundle:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"analysis-base manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    _validate_base_manifest(manifest, manifest_path)
    base_dir = manifest_path.parent

    recorders = pd.read_parquet(_resolve_manifest_path(base_dir, str(manifest["tables"]["recorders"]["path"])))
    segments = pd.read_parquet(_resolve_manifest_path(base_dir, str(manifest["tables"]["segments"]["path"])))
    events = pd.read_parquet(_resolve_manifest_path(base_dir, str(manifest["tables"]["events"]["path"])))
    trace_points_filtered = pd.read_parquet(_resolve_manifest_path(base_dir, str(manifest["tables"]["trace_points_filtered"]["path"])))
    sidecar_index = pd.read_parquet(_resolve_manifest_path(base_dir, str(manifest["tables"]["sidecar_index"]["path"])))
    gps_blocker_rule = json.loads(_resolve_manifest_path(base_dir, str(manifest["gps_blocker_rule_path"])).read_text(encoding="utf8"))

    _require_columns(recorders, ["recorder_id", "recorder_display_name", "device_id"], "recorders")
    _require_columns(
        segments,
        [
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
        "segments",
    )
    _require_columns(
        events,
        [
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
        "events",
    )
    _require_columns(
        trace_points_filtered,
        [
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
        "trace_points_filtered",
    )
    _require_columns(
        sidecar_index,
        [
            "segment_id",
            "recorder_id",
            "wav_basename",
            "date_local",
            "session_id",
            "day_session_id",
            "csv_hpc_path",
            "cached_csv_path",
        ],
        "sidecar_index",
    )
    _require_gps_blocker_rule(gps_blocker_rule)

    return BaseAnalysisBundle(
        manifest_path=manifest_path,
        manifest=manifest,
        timezone_name=str(manifest["timezone"]),
        dataset_name=str(manifest["dataset_name"]),
        recorders=recorders,
        segments=segments,
        events=events,
        trace_points_filtered=trace_points_filtered,
        sidecar_index=sidecar_index,
        gps_blocker_rule=gps_blocker_rule,
    )


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=path.suffix or ".parquet",
        dir=path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        df.to_parquet(temp_path, index=False)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf8",
        suffix=path.suffix or ".csv",
        dir=path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        df.to_csv(temp_path, index=index)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf8",
        suffix=path.suffix or ".json",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(serialized)
        temp_path = Path(handle.name)
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf8"))


def resolve_cached_sidecar_path(value: object) -> Path | None:
    if value in (None, "") or pd.isna(value):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def resolve_sidecar_source(row: pd.Series | Any) -> Path:
    cached_value = getattr(row, "cached_csv_path", None)
    cached_path = resolve_cached_sidecar_path(cached_value)
    if cached_path is not None and cached_path.exists():
        return cached_path

    remote_value = getattr(row, "csv_hpc_path", None)
    if remote_value not in (None, "") and not pd.isna(remote_value):
        remote_path = Path(str(remote_value))
        if remote_path.exists():
            return remote_path
    raise FileNotFoundError(
        f"missing sidecar source for segment {getattr(row, 'segment_id', '<unknown>')}: "
        f"cached_csv_path={cached_value!r}, csv_hpc_path={remote_value!r}"
    )


def _validate_base_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    required = [
        "bundle_name",
        "bundle_version",
        "dataset_name",
        "timezone",
        "artifact_dir",
        "gps_blocker_rule_path",
        "tables",
    ]
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError(f"{manifest_path} is missing required fields: {missing}")
    if manifest["bundle_name"] not in {"analysis_base", "analysis_base_snapshot"}:
        raise ValueError(f"{manifest_path} has unsupported bundle_name: {manifest['bundle_name']}")
    table_names = {"recorders", "segments", "events", "trace_points_filtered", "sidecar_index"}
    missing_tables = sorted(table_names - set(manifest["tables"]))
    if missing_tables:
        raise ValueError(f"{manifest_path} is missing required tables: {missing_tables}")


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _require_gps_blocker_rule(rule: dict[str, Any]) -> None:
    required = ["name", "description", "selection_source", "buffer_meters", "points"]
    missing = [field for field in required if field not in rule]
    if missing:
        raise ValueError(f"gps blocker rule is missing required fields: {missing}")


def _resolve_manifest_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()

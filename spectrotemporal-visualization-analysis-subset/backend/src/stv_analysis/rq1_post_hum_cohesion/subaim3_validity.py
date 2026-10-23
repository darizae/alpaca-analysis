from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stv_core.audiomoth import summarize_audiomoth_csv, valid_trace_points
from stv_core.geometry import buffered_bbox_polygon

from .contracts import PRIMARY_NONFOCAL_THRESHOLD
from .io import resolve_sidecar_source


def decorate_window_validity(window_inventory: pd.DataFrame) -> pd.DataFrame:
    enriched = window_inventory.copy()
    for prefix in ("filtered", "raw"):
        if f"{prefix}_focal_pre_fix_count" not in enriched.columns:
            continue
        threshold = PRIMARY_NONFOCAL_THRESHOLD
        enriched[f"{prefix}_gps_valid_min_nonfocal_{threshold}"] = (
            enriched[f"{prefix}_focal_pre_fix_count"].gt(0)
            & enriched[f"{prefix}_focal_post_fix_count"].gt(0)
            & enriched[f"{prefix}_nonfocal_pre_unique_count"].ge(threshold)
            & enriched[f"{prefix}_nonfocal_post_unique_count"].ge(threshold)
            & enriched[f"{prefix}_aligned_pre_sample_count"].gt(0)
            & enriched[f"{prefix}_aligned_post_sample_count"].gt(0)
        )

    if "raw_gps_valid_min_nonfocal_3" in enriched.columns:
        enriched["blocker_changed_validity_primary"] = (
            enriched["raw_gps_valid_min_nonfocal_3"] & ~enriched["filtered_gps_valid_min_nonfocal_3"]
        )
    else:
        enriched["blocker_changed_validity_primary"] = False
    if "raw_focal_pre_fix_count" in enriched.columns:
        enriched["blocker_removed_trace_points"] = (
            (enriched["raw_focal_pre_fix_count"] - enriched["filtered_focal_pre_fix_count"]).clip(lower=0)
            + (enriched["raw_focal_post_fix_count"] - enriched["filtered_focal_post_fix_count"]).clip(lower=0)
        )
    else:
        enriched["blocker_removed_trace_points"] = 0

    enriched["filtered_gps_valid_reason_tags"] = enriched.apply(validity_reason_tags, axis=1)
    return enriched.sort_values(["date_local", "event_id", "window_s"]).reset_index(drop=True)


def validity_reason_tags(row: pd.Series) -> str:
    tags: list[str] = []
    if int(row.get("filtered_focal_pre_fix_count", 0)) == 0:
        tags.append("missing_focal_pre")
    if int(row.get("filtered_focal_post_fix_count", 0)) == 0:
        tags.append("missing_focal_post")
    if int(row.get("filtered_nonfocal_pre_unique_count", 0)) < PRIMARY_NONFOCAL_THRESHOLD:
        tags.append("insufficient_nonfocal_pre")
    if int(row.get("filtered_nonfocal_post_unique_count", 0)) < PRIMARY_NONFOCAL_THRESHOLD:
        tags.append("insufficient_nonfocal_post")
    if int(row.get("filtered_aligned_pre_sample_count", 0)) == 0:
        tags.append("no_aligned_pre_seconds")
    if int(row.get("filtered_aligned_post_sample_count", 0)) == 0:
        tags.append("no_aligned_post_seconds")
    return "|".join(tags)


def build_blocked_trace_points(
    *,
    segments: pd.DataFrame,
    sidecar_index: pd.DataFrame,
    timezone_name: str,
    blocker_rule: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    blocker_polygon = buffered_bbox_polygon(
        [(float(lon), float(lat)) for lon, lat in blocker_rule["points"]],
        float(blocker_rule["buffer_meters"]),
    )
    lons = [lon for lon, _ in blocker_polygon]
    lats = [lat for _, lat in blocker_polygon]

    segments_with_sidecars = segments.merge(
        sidecar_index[["segment_id", "cached_csv_path"]],
        on="segment_id",
        how="left",
        validate="one_to_one",
    )

    blocked_frames: list[pd.DataFrame] = []
    segment_summaries: list[dict[str, Any]] = []
    for segment in segments_with_sidecars.itertuples(index=False):
        blocked, segment_summary = scan_blocked_trace_points_for_segment(
            segment=pd.Series(segment._asdict()),
            timezone_name=timezone_name,
            blocker_rule=blocker_rule,
        )
        segment_summaries.append(segment_summary)
        if not blocked.empty:
            blocked_frames.append(blocked)

    blocked_trace_points = (
        pd.concat(blocked_frames, ignore_index=True).sort_values(["timestamp_local", "recorder_id", "point_index_in_segment"])
        if blocked_frames
        else pd.DataFrame(
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
            ]
        )
    )
    audit = blocker_audit_from_segment_summaries(blocked_trace_points, segment_summaries, blocker_rule, blocker_polygon)
    return blocked_trace_points.reset_index(drop=True), audit


def scan_blocked_trace_points_for_segment(
    *,
    segment: pd.Series,
    timezone_name: str,
    blocker_rule: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    blocker_polygon = buffered_bbox_polygon(
        [(float(lon), float(lat)) for lon, lat in blocker_rule["points"]],
        float(blocker_rule["buffer_meters"]),
    )
    lons = [lon for lon, _ in blocker_polygon]
    lats = [lat for _, lat in blocker_polygon]
    sidecar_source = resolve_sidecar_source(segment)
    csv_df, summary = summarize_audiomoth_csv(sidecar_source)
    valid_points = valid_trace_points(csv_df, timezone_name)
    total_raw_trace_points = int(len(valid_points.index))
    if valid_points.empty:
        return (
            pd.DataFrame(columns=_blocked_trace_point_columns()),
            {
                "segment_id": str(segment.segment_id),
                "recorder_id": str(segment.recorder_id),
                "day_session_id": str(segment.day_session_id),
                "raw_valid_trace_points": 0,
                "blocked_trace_points": 0,
                "first_blocked_timestamp_local": None,
                "last_blocked_timestamp_local": None,
            },
        )

    valid_points = valid_points.reset_index(drop=True).copy()
    valid_points["point_index_in_segment"] = np.arange(1, len(valid_points.index) + 1, dtype=np.int32)
    blocked_mask = (
        valid_points["longitude"].between(min(lons), max(lons), inclusive="both")
        & valid_points["latitude"].between(min(lats), max(lats), inclusive="both")
    )
    blocked = valid_points.loc[blocked_mask].copy()
    if blocked.empty:
        return (
            pd.DataFrame(columns=_blocked_trace_point_columns()),
            {
                "segment_id": str(segment.segment_id),
                "recorder_id": str(segment.recorder_id),
                "day_session_id": str(segment.day_session_id),
                "raw_valid_trace_points": total_raw_trace_points,
                "blocked_trace_points": 0,
                "first_blocked_timestamp_local": None,
                "last_blocked_timestamp_local": None,
            },
        )

    blocked["segment_id"] = str(segment.segment_id)
    blocked["recorder_id"] = str(segment.recorder_id)
    blocked["recorder_display_name"] = str(segment.recorder_display_name)
    blocked["wav_basename"] = str(segment.wav_basename)
    blocked["gps_status"] = "A"
    blocked["timer_count"] = pd.to_numeric(blocked.get("TIMER_COUNT"), errors="coerce").astype("Int64")
    blocked["buffers_filled"] = pd.to_numeric(blocked.get("BUFFERS_FILLED"), errors="coerce").astype("Int64")
    blocked["buffers_written"] = pd.to_numeric(blocked.get("BUFFERS_WRITTEN"), errors="coerce").astype("Int64")
    blocked["sample_rate_estimate"] = summary.sample_rate_estimate
    blocked["session_id"] = str(segment.session_id)
    blocked["day_session_id"] = str(segment.day_session_id)
    blocked = blocked[_blocked_trace_point_columns()].reset_index(drop=True)
    return (
        blocked,
        {
            "segment_id": str(segment.segment_id),
            "recorder_id": str(segment.recorder_id),
            "day_session_id": str(segment.day_session_id),
            "raw_valid_trace_points": total_raw_trace_points,
            "blocked_trace_points": int(len(blocked.index)),
            "first_blocked_timestamp_local": blocked["timestamp_local"].min().isoformat(),
            "last_blocked_timestamp_local": blocked["timestamp_local"].max().isoformat(),
        },
    )


def blocker_audit_from_segment_summaries(
    blocked_trace_points: pd.DataFrame,
    segment_summaries: list[dict[str, Any]],
    blocker_rule: dict[str, Any],
    blocker_polygon: list[tuple[float, float]],
) -> dict[str, Any]:
    lons = [lon for lon, _ in blocker_polygon]
    lats = [lat for _, lat in blocker_polygon]
    total_raw_trace_points = int(sum(int(item["raw_valid_trace_points"]) for item in segment_summaries))
    audit = {
        "rule_name": blocker_rule["name"],
        "rule_description": blocker_rule["description"],
        "selection_source": blocker_rule["selection_source"],
        "buffer_meters": float(blocker_rule["buffer_meters"]),
        "seed_points_lon_lat": blocker_rule["points"],
        "derived_bbox_polygon_lon_lat": blocker_polygon,
        "derived_bbox_bounds": {
            "min_lon": min(lons),
            "max_lon": max(lons),
            "min_lat": min(lats),
            "max_lat": max(lats),
        },
        "total_raw_trace_points": total_raw_trace_points,
        "blocked_trace_points": int(len(blocked_trace_points.index)),
        "blocked_trace_share": float(len(blocked_trace_points.index) / total_raw_trace_points) if total_raw_trace_points else 0.0,
        "affected_recorders": int(blocked_trace_points["recorder_id"].nunique()) if not blocked_trace_points.empty else 0,
        "affected_day_sessions": sorted(blocked_trace_points["day_session_id"].dropna().astype(str).unique().tolist()),
        "first_blocked_timestamp_local": blocked_trace_points["timestamp_local"].min().isoformat()
        if not blocked_trace_points.empty
        else None,
        "last_blocked_timestamp_local": blocked_trace_points["timestamp_local"].max().isoformat()
        if not blocked_trace_points.empty
        else None,
        "blocked_points_by_recorder": {
            key: int(value)
            for key, value in blocked_trace_points.groupby("recorder_id").size().sort_values(ascending=False).items()
        },
        "segments_scanned": len(segment_summaries),
    }
    return audit


def _blocked_trace_point_columns() -> list[str]:
    return [
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
    ]

from __future__ import annotations

from typing import Any

import pandas as pd


def local_seconds(series: pd.Series) -> pd.Series:
    return (
        (series.dt.hour * 3600)
        + (series.dt.minute * 60)
        + series.dt.second
        + (series.dt.microsecond / 1_000_000)
        + (series.dt.nanosecond / 1_000_000_000)
    )


def build_analysis_units(
    events: pd.DataFrame,
    recorders: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    units = events.copy()
    explicit_non_hum_labels = {"human_talking", "rumble"}
    explicit_non_hum_mask = units["reviewed_resolved_label_semantic"].isin(explicit_non_hum_labels)
    units = units.loc[~explicit_non_hum_mask].copy()
    units["event_id"] = units["review_row_key"].astype(str)
    units["focal_id"] = units["recorder_id"].astype(str)
    midpoint_delta = (units["timestamp_end_local"] - units["timestamp_start_local"]) / 2
    units["event_midpoint_local"] = units["timestamp_start_local"] + midpoint_delta
    units["event_midpoint_seconds_local"] = local_seconds(units["event_midpoint_local"])
    units["analysis_unit_type"] = "hum_event_focal_recorder_window"
    units["focal_mapping_method"] = "recorder_id_as_focal_collar_id"
    units["focal_mapping_complete"] = units["focal_id"].notna()
    units = units[
        [
            "event_id",
            "focal_id",
            "recorder_display_name",
            "device_id",
            "date_local",
            "session_id",
            "day_session_id",
            "timestamp_start_local",
            "timestamp_end_local",
            "event_midpoint_local",
            "event_midpoint_seconds_local",
            "analysis_unit_type",
            "focal_mapping_method",
            "focal_mapping_complete",
        ]
    ].sort_values(["event_midpoint_local", "event_id"]).reset_index(drop=True)

    recorder_to_device = recorders.groupby("recorder_id")["device_id"].nunique()
    device_to_recorder = recorders.groupby("device_id")["recorder_id"].nunique()
    summary = {
        "events": int(len(units.index)),
        "focal_recorders": int(units["focal_id"].nunique()),
        "day_sessions": int(units["day_session_id"].nunique()),
        "missing_focal_mapping": int((~units["focal_mapping_complete"]).sum()),
        "excluded_explicit_non_hum_rows": int(explicit_non_hum_mask.sum()),
        "recorder_to_device_is_one_to_one": bool((recorder_to_device == 1).all()),
        "device_to_recorder_is_one_to_one": bool((device_to_recorder == 1).all()),
    }

    feasibility_rows = [
        {
            "variable_name": "event_id",
            "exists": "yes",
            "source_table_or_file": "analysis/base/events.parquet",
            "derivation": "review_row_key copied to event_id",
            "assumptions": "review_row_key is stable and unique per hum detection",
            "failure_modes": "none observed in current corpus",
            "recommended_alternative_if_not_feasible": "",
        },
        {
            "variable_name": "focal_id",
            "exists": "yes",
            "source_table_or_file": "analysis/base/events.parquet",
            "derivation": "recorder_id used directly as focal collar identity",
            "assumptions": "each recorder maps to one collar deployment; no separate animal_id table exists",
            "failure_modes": "would fail if recorder_id reused across animals within-session",
            "recommended_alternative_if_not_feasible": "fallback to recorder-level rather than animal-level interpretation",
        },
        {
            "variable_name": "event_midpoint_local",
            "exists": "derived",
            "source_table_or_file": "analysis/base/events.parquet",
            "derivation": "timestamp_start_local + (timestamp_end_local - timestamp_start_local) / 2",
            "assumptions": "event midpoint is the correct t0 because start and end are available",
            "failure_modes": "missing start/end timestamps",
            "recommended_alternative_if_not_feasible": "use event start timestamp",
        },
        {
            "variable_name": "gps_trace_points",
            "exists": "yes",
            "source_table_or_file": "analysis/base/trace_points_filtered.parquet",
            "derivation": "canonical filtered 1 Hz valid GPS trace table",
            "assumptions": "filtered trace_points are the analysis-ready GPS source",
            "failure_modes": "none observed in current corpus",
            "recommended_alternative_if_not_feasible": "",
        },
        {
            "variable_name": "recorder_device_lookup",
            "exists": "yes",
            "source_table_or_file": "analysis/base/recorders.parquet",
            "derivation": "recorder_id <-> device_id lookup",
            "assumptions": "device_id is stable across the deployment window",
            "failure_modes": "recorder or device reuse across animals",
            "recommended_alternative_if_not_feasible": "join through source segment metadata",
        },
        {
            "variable_name": "audio_coverage_intervals",
            "exists": "yes",
            "source_table_or_file": "analysis/base/segments.parquet",
            "derivation": "recording windows define coverage; session_id/day_session_id inherit from segment availability",
            "assumptions": "segment recording windows are authoritative for coverage",
            "failure_modes": "missing or truncated source segment metadata",
            "recommended_alternative_if_not_feasible": "derive coverage directly from source WAV spans",
        },
        {
            "variable_name": "hard_gps_spatial_filter",
            "exists": "yes",
            "source_table_or_file": "analysis/base/gps_blocker_rule.json",
            "derivation": "buffered bounding box around reviewed outlier-hum GPS points",
            "assumptions": "the blocker polygon is the intended hard filter for impossible GPS positions",
            "failure_modes": "sidecar cache missing for exact attrition audit",
            "recommended_alternative_if_not_feasible": "rebuild or refresh sidecar cache before blocker audit",
        },
        {
            "variable_name": "shared_metric_coordinates",
            "exists": "derived",
            "source_table_or_file": "analysis/base/trace_points_filtered.parquet",
            "derivation": "local equirectangular projection anchored at the corpus mean latitude/longitude",
            "assumptions": "paddock extent is small enough for a local linear projection",
            "failure_modes": "would distort distances only if the study area were much larger",
            "recommended_alternative_if_not_feasible": "switch to a formal projected CRS",
        },
    ]
    feasibility_log = pd.DataFrame(feasibility_rows)
    return units, summary, feasibility_log

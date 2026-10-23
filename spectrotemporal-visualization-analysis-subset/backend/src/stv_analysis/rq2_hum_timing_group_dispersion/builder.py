from __future__ import annotations

import json
from math import cos, isfinite, pi
import os
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from ..framework import AnalysisDefinition, AnalysisStageSpec, LaunchContext, read_json, refresh_latest_alias, stage_cache_dir, slugify_token, write_json
from ..rq1_post_hum_cohesion.contracts import LATITUDE_METERS_PER_DEGREE
from ..rq1_post_hum_cohesion.io import load_base_bundle, write_csv, write_parquet
from ..rq1_post_hum_cohesion.subaim1_units import local_seconds
from ..rq1_post_hum_cohesion.subaim2_windows import (
    build_session_trace_grids,
    projection_origin,
    read_session_trace_grid,
    write_session_trace_grid,
)
from .contracts import (
    ANALYSIS_ID,
    ANALYSIS_VERSION,
    BIOLOGICALLY_IMPOSSIBLE_SPEED_MPS,
    DOCUMENTED_PADDOCK_AREA_M2,
    DOCUMENTED_PADDOCK_LENGTH_M,
    DOCUMENTED_PADDOCK_WIDTH_M,
    HEAVY_COMPUTE_LABEL,
    MODEL_FIT_RESOURCE_PROFILE,
    OUTPUT_DIRNAME,
    PRIMARY_WINDOW_LENGTH_S,
    RESIDUAL_GPS_FILTER_VERSION,
    STAGE_IDS,
    WINDOW_LENGTHS_SECONDS,
)


DOC_PATHS = (
    "PRIMER.md",
    "docs/rq2_hum_timing_group_dispersion/rq2_hum_timing_group_dispersion_experimental_plan.md",
    "docs/rq2_hum_timing_group_dispersion/ARTIFACT_CONTRACT.md",
)

STAGE_SPECS = (
    AnalysisStageSpec("units", "Build fixed-window RQ2 units and feasibility report.", HEAVY_COMPUTE_LABEL, "heavy"),
    AnalysisStageSpec("gps_filter_audit", "Audit hard GPS filter and materialize launch-local residual GPS-QC trace table.", HEAVY_COMPUTE_LABEL, "heavy", ("units",)),
    AnalysisStageSpec("trace_grids", "Build per-session GPS trace grids from residual-GPS-QC traces.", HEAVY_COMPUTE_LABEL, "heavy", ("gps_filter_audit",)),
    AnalysisStageSpec("window_grid", "Decorate windows with GPS/audio coverage and eligibility.", HEAVY_COMPUTE_LABEL, "heavy", ("trace_grids",)),
    AnalysisStageSpec("dispersion_features", "Compute observed area, spread change, speed, and centroid-position change.", HEAVY_COMPUTE_LABEL, "heavy", ("window_grid",)),
    AnalysisStageSpec("hum_rate_features", "Assign canonical hum events to windows and compute hum counts/rates.", HEAVY_COMPUTE_LABEL, "heavy", ("dispersion_features",)),
    AnalysisStageSpec("model_input", "Assemble the 60 s RQ2 count-GLMM input table.", HEAVY_COMPUTE_LABEL, "heavy", ("hum_rate_features",)),
    AnalysisStageSpec("scientific_glmm", "Fit the 60 s glmmTMB Poisson and NB1 models with maximal session random slopes and DHARMa diagnostics.", HEAVY_COMPUTE_LABEL, MODEL_FIT_RESOURCE_PROFILE, ("model_input",)),
    AnalysisStageSpec("diagnostics_outputs", "Write launch diagnostics and output index.", HEAVY_COMPUTE_LABEL, "heavy", ("scientific_glmm",)),
    AnalysisStageSpec("reports", "Write final runtime README and manifest.", HEAVY_COMPUTE_LABEL, "heavy", ("diagnostics_outputs",)),
)

ANALYSIS_DEFINITION = AnalysisDefinition(
    analysis_id=ANALYSIS_ID,
    output_dirname=OUTPUT_DIRNAME,
    docs_to_snapshot=DOC_PATHS,
    stage_specs=STAGE_SPECS,
    initialize_launch=lambda ctx, profile: initialize_launch(ctx, profile),
    run_stage=lambda ctx, spec, task_value: run_rq2_stage(ctx, spec.stage_id, task_value),
)


def _base_snapshot_manifest_path(ctx: LaunchContext) -> Path:
    manifest_path = Path(ctx.launch_manifest["base_snapshot_manifest_path"])
    if manifest_path.is_absolute():
        return manifest_path
    return (ctx.paths.launch_dir / manifest_path).resolve()


def initialize_launch(ctx: LaunchContext, profile: str) -> dict[str, Any]:
    bundle = load_base_bundle(_base_snapshot_manifest_path(ctx))
    available_sessions = sorted(bundle.trace_points_filtered["day_session_id"].dropna().astype(str).unique().tolist())
    if not available_sessions:
        raise ValueError("base snapshot has no day_session_id values in trace_points_filtered")
    if profile == "smoke":
        selected_sessions = available_sessions[: min(5, len(available_sessions))]
    elif profile == "full":
        selected_sessions = available_sessions
    else:
        raise ValueError(f"unsupported profile: {profile}")
    selected_trace = _select_sessions(bundle.trace_points_filtered, selected_sessions)
    selected_segments = _select_sessions(bundle.segments, selected_sessions)
    selected_events = _select_sessions(bundle.events, selected_sessions)
    stage_tasks = {stage_id: [] for stage_id in STAGE_IDS}
    stage_tasks["trace_grids"] = selected_sessions
    return {
        "analysis": {
            "analysis_version": ANALYSIS_VERSION,
            "profile": profile,
            "selected_day_session_ids": selected_sessions,
            "selected_event_rows": int(len(selected_events.index)),
            "selected_segment_rows": int(len(selected_segments.index)),
            "selected_trace_point_rows": int(len(selected_trace.index)),
            "projection_origin": projection_origin(selected_trace),
            "window_lengths_s": list(WINDOW_LENGTHS_SECONDS),
            "primary_window_length_s": PRIMARY_WINDOW_LENGTH_S,
            "hum_event_semantics": "canonical hum events retain alpaca_hum, None, and unlabeled non-explicit rows; explicit rumble and human_talking are excluded",
            "heavy_compute_stage_ids": [spec.stage_id for spec in STAGE_SPECS],
            "model_fit_stage_id": "scientific_glmm",
            "model_fit_resource_profile": MODEL_FIT_RESOURCE_PROFILE,
            "random_effects_structure_version": "rq2_session_maximal_random_slopes_v1",
            "documented_paddock_dimensions_m": {"length": DOCUMENTED_PADDOCK_LENGTH_M, "width": DOCUMENTED_PADDOCK_WIDTH_M},
            "documented_paddock_area_m2": DOCUMENTED_PADDOCK_AREA_M2,
            "residual_gps_filter_version": RESIDUAL_GPS_FILTER_VERSION,
            "smoke_session_rule": "first up to five available day_session_id values in analysis-base trace_points_filtered",
        },
        "stage_tasks": stage_tasks,
    }


def run_rq2_stage(ctx: LaunchContext, stage_id: str, task_value: str | None) -> dict[str, Any]:
    if stage_id == "units":
        return _run_units_stage(ctx)
    if stage_id == "gps_filter_audit":
        return _run_gps_filter_audit_stage(ctx)
    if stage_id == "trace_grids":
        assert task_value is not None
        return _run_trace_grid_task(ctx, task_value)
    if stage_id == "window_grid":
        return _run_window_grid_stage(ctx)
    if stage_id == "dispersion_features":
        return _run_dispersion_features_stage(ctx)
    if stage_id == "hum_rate_features":
        return _run_hum_rate_features_stage(ctx)
    if stage_id == "model_input":
        return _run_model_input_stage(ctx)
    if stage_id == "scientific_glmm":
        return _run_scientific_glmm_stage(ctx)
    if stage_id == "diagnostics_outputs":
        return _run_diagnostics_outputs_stage(ctx)
    if stage_id == "reports":
        return _run_reports_stage(ctx)
    raise ValueError(f"unsupported stage_id: {stage_id}")


def build_rq2_units(
    trace_points: pd.DataFrame,
    events: pd.DataFrame,
    segments: pd.DataFrame,
    recorders: pd.DataFrame,
    *,
    selected_sessions: list[str],
    timezone_name: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_trace = _select_sessions(trace_points, selected_sessions)
    session_bounds = (
        selected_trace.groupby("day_session_id", sort=True)
        .agg(
            date_local=("date_local", "first"),
            session_id=("session_id", "first"),
            min_second=("seconds_since_midnight_local", "min"),
            max_second=("seconds_since_midnight_local", "max"),
            trace_points=("latitude", "size"),
            recorder_count=("recorder_id", "nunique"),
        )
        .reset_index()
    )
    for session in session_bounds.itertuples(index=False):
        first_second = int(round(float(session.min_second)))
        last_second_exclusive = int(round(float(session.max_second))) + 1
        midnight = pd.Timestamp(str(session.date_local), tz=timezone_name)
        for window_length_s in WINDOW_LENGTHS_SECONDS:
            for start_s in range(first_second, last_second_exclusive - int(window_length_s) + 1, int(window_length_s)):
                end_s = start_s + int(window_length_s)
                rows.append(
                    {
                        "rq2_window_id": f"{session.day_session_id}:w{window_length_s}:{start_s}",
                        "day_session_id": str(session.day_session_id),
                        "date_local": str(session.date_local),
                        "session_id": str(session.session_id),
                        "window_length_s": int(window_length_s),
                        "window_start_seconds_local": float(start_s),
                        "window_end_seconds_local": float(end_s),
                        "window_midpoint_seconds_local": float(start_s + (int(window_length_s) / 2.0)),
                        "window_start_local": midnight + pd.to_timedelta(start_s, unit="s"),
                        "window_end_local": midnight + pd.to_timedelta(end_s, unit="s"),
                    }
                )
    units = pd.DataFrame(rows).sort_values(
        ["date_local", "day_session_id", "window_length_s", "window_start_seconds_local"]
    ).reset_index(drop=True)
    hum_events = hum_event_frame(events)
    feasibility = pd.DataFrame(
        [
            _feasibility_row("event_corpus_version", "yes", "base_snapshot/manifest.json", f"{dataset_name} analysis-base snapshot", "deduplicated event rows are already materialized by build-analysis-base"),
            _feasibility_row("canonical_hum_event_count", "derived", "base_snapshot/events.parquet", "retain all labels except explicit rumble/human_talking", "None/unlabeled rows are canonical alpaca hum events for RQ2"),
            _feasibility_row("fixed_window_unit", "derived", "base_snapshot/trace_points_filtered.parquet", "non-overlapping 60 s windows from first rounded session GPS second", "terminal incomplete windows are discarded"),
            _feasibility_row("hard_gps_filter", "yes", "base_snapshot/trace_points_filtered.parquet + gps_blocker_rule.json", "analysis-base trace points are already spatially hard-filtered", "RQ2 additionally audits residual extreme positions"),
            _feasibility_row("residual_gps_qc_table", "derived", "cache/gps_filter_audit/trace_points_rq2_gps_qc.parquet", "remove residual extreme-position GPS fingerprints before trace grids", "raw analysis-base snapshot remains unchanged"),
            _feasibility_row("count_glmm_response", "derived", "tables/rq2_model_input.parquet", "hum_count with offset(log(window_length_minutes))", "window exposure is retained explicitly instead of fitting hum_rate_per_min"),
        ]
    )
    trace_availability = session_bounds.assign(
        variable_name=lambda frame: "trace_point_availability_session_" + frame["day_session_id"].astype(str),
        exists="yes",
        source_table_or_file="base_snapshot/trace_points_filtered.parquet",
        derivation="session trace-point counts after the canonical hard GPS spatial filter; residual GPS QC is applied in gps_filter_audit",
        assumptions="filtered trace_points are the hard-filter input, not the final RQ2 GPS input",
        failure_modes="zero trace points would exclude a session from launch selection",
        recommended_alternative_if_not_feasible="rebuild analysis-base before RQ2 launch",
    )[
        [
            "variable_name",
            "exists",
            "source_table_or_file",
            "derivation",
            "assumptions",
            "failure_modes",
            "recommended_alternative_if_not_feasible",
            "trace_points",
            "recorder_count",
        ]
    ]
    feasibility = pd.concat([feasibility, trace_availability], ignore_index=True)
    summary = {
        "event_corpus_version": dataset_name,
        "canonical_hum_event_count": int(len(hum_events.index)),
        "raw_event_rows": int(len(events.index)),
        "excluded_explicit_non_hum_rows": int(len(events.index) - len(hum_events.index)),
        "event_label_policy": "alpaca_hum, None, and unlabeled non-explicit rows are hums; rumble and human_talking are excluded",
        "session_count": int(len(selected_sessions)),
        "recorder_count": int(recorders["recorder_id"].nunique()),
        "segment_count": int(_select_sessions(segments, selected_sessions)["segment_id"].nunique()),
        "window_rows": int(len(units.index)),
        "window_lengths_s": list(WINDOW_LENGTHS_SECONDS),
    }
    return units, feasibility, summary


def _run_units_stage(ctx: LaunchContext) -> dict[str, Any]:
    bundle = load_base_bundle(_base_snapshot_manifest_path(ctx))
    sessions = _selected_sessions(ctx)
    units, feasibility_log, summary = build_rq2_units(
        bundle.trace_points_filtered,
        bundle.events,
        bundle.segments,
        bundle.recorders,
        selected_sessions=sessions,
        timezone_name=bundle.timezone_name,
        dataset_name=bundle.dataset_name,
    )
    write_parquet(units, ctx.paths.tables_dir / "rq2_units.parquet")
    write_csv(feasibility_log, ctx.paths.tables_dir / "variable_feasibility_log.csv")
    write_json(stage_cache_dir(ctx.paths, "units") / "unit_summary.json", summary)
    _write_report(ctx.paths.reports_dir / "subaim_1_units.md", _subaim_1_report(summary))
    return {"stage_id": "units", "rq2_windows": int(len(units.index)), "day_sessions": len(sessions)}


def _run_gps_filter_audit_stage(ctx: LaunchContext) -> dict[str, Any]:
    bundle = load_base_bundle(_base_snapshot_manifest_path(ctx))
    trace = _select_sessions(bundle.trace_points_filtered, _selected_sessions(ctx))
    origin = _projection_origin(ctx)
    qc_trace, audit, outliers, point_summary, paddock = gps_residual_qc_tables(trace, origin)
    cache_dir = stage_cache_dir(ctx.paths, "gps_filter_audit")
    write_parquet(qc_trace, cache_dir / "trace_points_rq2_gps_qc.parquet")
    write_csv(audit, ctx.paths.tables_dir / "rq2_gps_residual_outlier_audit.csv")
    write_parquet(outliers, ctx.paths.tables_dir / "rq2_gps_residual_outlier_points.parquet")
    write_csv(point_summary, ctx.paths.tables_dir / "rq2_gps_qc_trace_point_summary.csv")
    write_csv(paddock, ctx.paths.tables_dir / "rq2_paddock_scale_sanity.csv")
    _write_gps_audit_figures(ctx.paths.launch_dir / "figures", trace, qc_trace, audit, origin)
    write_json(
        cache_dir / "residual_gps_filter_manifest.json",
        {
            "residual_gps_filter_version": RESIDUAL_GPS_FILTER_VERSION,
            "projection_origin": origin,
            "documented_paddock_area_m2": DOCUMENTED_PADDOCK_AREA_M2,
            "rule": "flag x_m < -100 or y_m < -100 plus points beyond session robust 0.1%-99.9% envelope by max(10 m, 25% robust span)",
            "input_rows": int(len(trace.index)),
            "removed_rows": int(len(outliers.index)),
            "output_rows": int(len(qc_trace.index)),
        },
    )
    _write_report(ctx.paths.reports_dir / "subaim_2_gps_audit.md", _subaim_2_gps_audit_report(audit, point_summary, paddock))
    return {"stage_id": "gps_filter_audit", "input_rows": int(len(trace.index)), "removed_rows": int(len(outliers.index)), "qc_rows": int(len(qc_trace.index))}


def gps_residual_qc_tables(trace_points: pd.DataFrame, origin: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    projected = add_projected_coordinates(trace_points, origin)
    outlier_masks: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []
    paddock_rows: list[dict[str, Any]] = []
    for session_id, frame in projected.groupby("day_session_id", sort=True):
        x = frame["x_m"].to_numpy(dtype=float)
        y = frame["y_m"].to_numpy(dtype=float)
        qx001, qx999 = np.nanquantile(x, [0.001, 0.999])
        qy001, qy999 = np.nanquantile(y, [0.001, 0.999])
        span_x = max(0.0, float(qx999 - qx001))
        span_y = max(0.0, float(qy999 - qy001))
        margin_x = max(10.0, span_x * 0.25)
        margin_y = max(10.0, span_y * 0.25)
        mask = (
            (frame["x_m"] < -100.0)
            | (frame["y_m"] < -100.0)
            | (frame["x_m"] < qx001 - margin_x)
            | (frame["x_m"] > qx999 + margin_x)
            | (frame["y_m"] < qy001 - margin_y)
            | (frame["y_m"] > qy999 + margin_y)
        )
        outlier_masks.append(mask)
        all_hull_area = _convex_hull_area(frame[["x_m", "y_m"]].to_numpy(dtype=float))
        robust_envelope_area = span_x * span_y
        affected_seconds = frame.loc[mask, "seconds_since_midnight_local"].round().nunique()
        audit_rows.append(
            {
                "day_session_id": str(session_id),
                "input_points": int(len(frame.index)),
                "residual_outlier_points": int(mask.sum()),
                "affected_seconds": int(affected_seconds),
                "all_point_hull_area_m2": float(all_hull_area) if isfinite(all_hull_area) else np.nan,
                "robust_0p1_99p9_envelope_area_m2": float(robust_envelope_area),
                "all_point_hull_to_documented_paddock_ratio": float(all_hull_area / DOCUMENTED_PADDOCK_AREA_M2) if isfinite(all_hull_area) else np.nan,
                "robust_envelope_to_documented_paddock_ratio": float(robust_envelope_area / DOCUMENTED_PADDOCK_AREA_M2),
                "q001_x_m": float(qx001),
                "q999_x_m": float(qx999),
                "q001_y_m": float(qy001),
                "q999_y_m": float(qy999),
                "rule_x_lt_minus_100_or_y_lt_minus_100_count": int(((frame["x_m"] < -100.0) | (frame["y_m"] < -100.0)).sum()),
                "residual_filter_version": RESIDUAL_GPS_FILTER_VERSION,
            }
        )
        per_second = _per_second_polygon_areas(frame.loc[~mask].copy())
        paddock_rows.append(
            {
                "day_session_id": str(session_id),
                "documented_paddock_area_m2": DOCUMENTED_PADDOCK_AREA_M2,
                "remaining_per_second_polygon_exceeds_9800_m2": bool((per_second["per_second_polygon_area_m2"] > DOCUMENTED_PADDOCK_AREA_M2).any()) if not per_second.empty else False,
                "max_remaining_per_second_polygon_area_m2": float(per_second["per_second_polygon_area_m2"].max()) if not per_second.empty else np.nan,
                "outlier_removed": bool(mask.any()),
            }
        )
    combined_mask = pd.concat(outlier_masks).sort_index() if outlier_masks else pd.Series(False, index=projected.index)
    outlier_cols = [
        "day_session_id",
        "date_local",
        "session_id",
        "recorder_id",
        "recorder_display_name",
        "seconds_since_midnight_local",
        "latitude",
        "longitude",
        "x_m",
        "y_m",
    ]
    outliers = projected.loc[combined_mask, [c for c in outlier_cols if c in projected.columns]].copy().reset_index(drop=True)
    qc_trace = projected.loc[~combined_mask, trace_points.columns].copy().reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    point_summary = _gps_qc_trace_point_summary(projected, qc_trace, outliers)
    paddock = pd.DataFrame(paddock_rows)
    return qc_trace, audit, outliers, point_summary, paddock


def _run_trace_grid_task(ctx: LaunchContext, day_session_id: str) -> dict[str, Any]:
    cache_dir = stage_cache_dir(ctx.paths, "trace_grids")
    token = slugify_token(day_session_id)
    grid_path = cache_dir / f"{token}.npz"
    summary_path = cache_dir / f"{token}.json"
    if grid_path.exists() and summary_path.exists():
        return {"stage_id": "trace_grids", "day_session_id": day_session_id, "grid_path": str(grid_path), "status": "cached"}
    qc_path = stage_cache_dir(ctx.paths, "gps_filter_audit") / "trace_points_rq2_gps_qc.parquet"
    if not qc_path.is_file():
        raise FileNotFoundError(f"missing residual GPS-QC trace table: {qc_path}")
    trace_points = pd.read_parquet(qc_path)
    session_points = trace_points.loc[trace_points["day_session_id"].astype(str) == day_session_id].copy()
    origin = _projection_origin(ctx)
    grids, sync_summary = build_session_trace_grids(
        session_points,
        origin_lat=float(origin["origin_latitude"]),
        origin_lon=float(origin["origin_longitude"]),
    )
    grid = grids[day_session_id]
    write_session_trace_grid(grid, grid_path)
    write_json(summary_path, sync_summary)
    return {"stage_id": "trace_grids", "day_session_id": day_session_id, "grid_path": str(grid_path)}


def _run_window_grid_stage(ctx: LaunchContext) -> dict[str, Any]:
    bundle = load_base_bundle(_base_snapshot_manifest_path(ctx))
    units = pd.read_parquet(ctx.paths.tables_dir / "rq2_units.parquet")
    grids = _load_grid_cache(ctx, "trace_grids")
    qc_trace = pd.read_parquet(stage_cache_dir(ctx.paths, "gps_filter_audit") / "trace_points_rq2_gps_qc.parquet")
    window_grid = build_window_grid(units, grids, qc_trace, bundle.segments, bundle.gps_blocker_rule)
    write_parquet(window_grid, ctx.paths.tables_dir / "rq2_window_grid.parquet")
    write_csv(window_coverage_summary(window_grid), ctx.paths.tables_dir / "rq2_window_coverage_summary.csv")
    write_csv(hard_gps_filter_summary(bundle, window_grid), ctx.paths.tables_dir / "hard_gps_filter_summary.csv")
    _write_report(ctx.paths.reports_dir / "subaim_3_window_grid.md", _subaim_3_window_grid_report(window_grid, bundle.gps_blocker_rule))
    return {"stage_id": "window_grid", "eligible_windows": int(window_grid["window_eligible"].sum()), "window_rows": int(len(window_grid.index))}


def build_window_grid(units: pd.DataFrame, grids: dict[str, Any], trace_points: pd.DataFrame, segments: pd.DataFrame, gps_blocker_rule: dict[str, Any]) -> pd.DataFrame:
    coverage = audio_coverage_intervals(trace_points, segments)
    rows: list[dict[str, Any]] = []
    for row in units.itertuples(index=False):
        grid = grids.get(str(row.day_session_id))
        if grid is None:
            rows.append(_empty_window_grid_row(row, gps_blocker_rule, "missing_trace_grid"))
            continue
        mask = _second_window_mask(grid, row.window_start_seconds_local, row.window_end_seconds_local)
        counts = grid.valid[mask].sum(axis=1) if np.any(mask) else np.array([], dtype=int)
        seconds_with_gps = int(np.count_nonzero(counts > 0))
        seconds_with_three = int(np.count_nonzero(counts >= 3))
        unique_collar_count = int(np.count_nonzero(grid.valid[mask].any(axis=0))) if np.any(mask) else 0
        active_recorders = active_audio_recorders(coverage.get(str(row.day_session_id), {}), float(row.window_start_seconds_local), float(row.window_end_seconds_local))
        audio_coverage_fraction = float(len(active_recorders) / max(1, len(grid.recorder_ids)))
        gps_coverage_fraction = float(seconds_with_three / max(1.0, float(row.window_length_s)))
        audio_coverage_ok = len(active_recorders) > 0
        eligible = gps_coverage_fraction > 0 and unique_collar_count >= 3 and audio_coverage_ok
        rows.append(
            {
                **_unit_fields(row),
                "n_collars_available": unique_collar_count,
                "n_valid_gps_seconds": seconds_with_three,
                "n_any_gps_seconds": seconds_with_gps,
                "gps_coverage_fraction": gps_coverage_fraction,
                "audio_coverage_ok": bool(audio_coverage_ok),
                "n_audio_recorders_active": int(len(active_recorders)),
                "audio_coverage_fraction": audio_coverage_fraction,
                "window_eligible": bool(eligible),
                "window_ineligibility_reason": "" if eligible else _ineligibility_reason(gps_coverage_fraction, unique_collar_count, audio_coverage_ok),
                "hard_gps_filter_version": str(gps_blocker_rule.get("name", "unknown")),
                "residual_gps_filter_version": RESIDUAL_GPS_FILTER_VERSION,
            }
        )
    return pd.DataFrame(rows).sort_values(["date_local", "day_session_id", "window_length_s", "window_start_seconds_local"]).reset_index(drop=True)


def _run_dispersion_features_stage(ctx: LaunchContext) -> dict[str, Any]:
    window_grid = pd.read_parquet(ctx.paths.tables_dir / "rq2_window_grid.parquet")
    grids = _load_grid_cache(ctx, "trace_grids")
    features = build_dispersion_features(window_grid, grids)
    write_parquet(features, ctx.paths.tables_dir / "rq2_dispersion_features.parquet")
    write_csv(dispersion_feature_summary(features), ctx.paths.tables_dir / "rq2_dispersion_feature_summary.csv")
    write_csv(dispersion_change_summary(features), ctx.paths.tables_dir / "rq2_dispersion_change_summary.csv")
    _write_report(ctx.paths.reports_dir / "subaim_4_dispersion_features.md", _subaim_4_dispersion_report(features))
    return {"stage_id": "dispersion_features", "feature_rows": int(len(features.index))}


def build_dispersion_features(window_grid: pd.DataFrame, grids: dict[str, Any], *, dropped_recorder_id: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = window_grid.loc[window_grid["window_eligible"].astype(bool)].copy()
    for row in eligible.itertuples(index=False):
        grid = grids[str(row.day_session_id)]
        dropped_index = grid.recorder_index.get(dropped_recorder_id) if dropped_recorder_id is not None else None
        metrics = _window_dispersion_metrics(grid, float(row.window_start_seconds_local), float(row.window_end_seconds_local), dropped_index=dropped_index)
        if metrics["n_collars_available"] < 3:
            continue
        rows.append({**{column: getattr(row, column) for column in window_grid.columns}, **metrics})
    features = pd.DataFrame(rows)
    if features.empty:
        return _empty_dispersion_features()
    features = features.sort_values(["day_session_id", "window_length_s", "window_start_seconds_local"]).reset_index(drop=True)
    grouped = features.groupby(["day_session_id", "window_length_s"], sort=False)
    features["recent_change_in_polygon_area_m2"] = grouped["observed_polygon_area_m2"].diff().fillna(0.0)
    features["dispersion_change_m2_per_min"] = features["recent_change_in_polygon_area_m2"] / (features["window_length_s"].astype(float) / 60.0)
    features["recent_change_in_centroid_distance_m"] = grouped["mean_distance_to_observed_centroid_m"].diff().fillna(0.0)
    features["centroid_change_in_position_m"] = _centroid_position_change(features)
    features["centroid_change_in_position_m_per_min"] = features["centroid_change_in_position_m"] / (features["window_length_s"].astype(float) / 60.0)
    if dropped_recorder_id is not None:
        features["dropped_recorder_id"] = dropped_recorder_id
    return features


def _run_hum_rate_features_stage(ctx: LaunchContext) -> dict[str, Any]:
    bundle = load_base_bundle(_base_snapshot_manifest_path(ctx))
    dispersion_features = pd.read_parquet(ctx.paths.tables_dir / "rq2_dispersion_features.parquet")
    hum_features = build_hum_rate_features(dispersion_features, _select_sessions(bundle.events, _selected_sessions(ctx)))
    write_parquet(hum_features, ctx.paths.tables_dir / "rq2_hum_rate_features.parquet")
    write_csv(hum_rate_summary(hum_features), ctx.paths.tables_dir / "rq2_hum_rate_summary.csv")
    _write_report(ctx.paths.reports_dir / "subaim_5_hum_counts.md", _subaim_5_hum_report(hum_features))
    return {"stage_id": "hum_rate_features", "feature_rows": int(len(hum_features.index)), "hum_count_total": int(hum_features["hum_count"].sum()) if not hum_features.empty else 0}


def build_hum_rate_features(dispersion_features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    windows = dispersion_features.copy()
    if windows.empty:
        for col in ("hum_count", "hum_presence", "hum_rate_per_min"):
            windows[col] = []
        return windows
    hum_events = hum_event_frame(events)
    if hum_events.empty:
        windows["hum_count"] = 0
    else:
        hum_events = hum_events.copy()
        hum_events["event_midpoint_local"] = hum_events["timestamp_start_local"] + (hum_events["timestamp_end_local"] - hum_events["timestamp_start_local"]) / 2
        hum_events["event_midpoint_seconds_local"] = local_seconds(hum_events["event_midpoint_local"])
        counts: list[int] = []
        grouped = {str(k): v for k, v in hum_events.groupby(hum_events["day_session_id"].astype(str), sort=False)}
        for row in windows.itertuples(index=False):
            session_events = grouped.get(str(row.day_session_id), hum_events.iloc[0:0])
            mask = (session_events["event_midpoint_seconds_local"] >= float(row.window_start_seconds_local)) & (session_events["event_midpoint_seconds_local"] < float(row.window_end_seconds_local))
            counts.append(int(mask.sum()))
        windows["hum_count"] = counts
    windows["hum_presence"] = (windows["hum_count"] > 0).astype(int)
    windows["hum_rate_per_min"] = windows["hum_count"] / windows["window_length_s"].astype(float) * 60.0
    return windows.sort_values(["day_session_id", "window_length_s", "window_start_seconds_local"]).reset_index(drop=True)


def _run_model_input_stage(ctx: LaunchContext) -> dict[str, Any]:
    hum_features = pd.read_parquet(ctx.paths.tables_dir / "rq2_hum_rate_features.parquet")
    model_input, summary = build_model_input(hum_features)
    write_parquet(model_input, ctx.paths.tables_dir / "rq2_model_input.parquet")
    write_csv(model_input, ctx.paths.tables_dir / "rq2_model_input.csv")
    write_csv(summary, ctx.paths.tables_dir / "rq2_model_input_summary.csv")
    _write_report(ctx.paths.reports_dir / "subaim_6_model_input.md", _subaim_6_model_input_report(model_input, summary))
    return {"stage_id": "model_input", "model_input_rows": int(len(model_input.index))}


def build_model_input(hum_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = hum_features.copy()
    if df.empty:
        return _empty_model_input(), pd.DataFrame([{"window_length_s": int(w), "rows": 0} for w in WINDOW_LENGTHS_SECONDS])
    first_by_session = df.groupby("day_session_id")["window_start_seconds_local"].transform("min")
    df["minutes_since_session_start"] = (df["window_midpoint_seconds_local"] - first_by_session) / 60.0
    df["time_of_day"] = pd.to_datetime(df["window_midpoint_seconds_local"], unit="s").dt.strftime("%H:%M:%S")
    df["window_length_minutes"] = df["window_length_s"].astype(float) / 60.0
    df["area_m2"] = pd.to_numeric(df["observed_polygon_area_m2"], errors="coerce")
    df["spread_change_m2"] = pd.to_numeric(df["recent_change_in_polygon_area_m2"], errors="coerce")
    df["spread_change_m2_per_min"] = pd.to_numeric(df["dispersion_change_m2_per_min"], errors="coerce")
    df["speed_mps"] = pd.to_numeric(df["mean_speed_mps"], errors="coerce")
    df["centroid_change_in_position_m"] = pd.to_numeric(df["centroid_change_in_position_m"], errors="coerce")
    df["canonical_event_policy"] = "alpaca_hum and None are canonical hums; rumble and human_talking are excluded"
    df["model_family"] = "count_glmm"
    df["model_offset"] = "offset(log(window_length_minutes))"
    required = [
        "rq2_window_id",
        "day_session_id",
        "date_local",
        "session_id",
        "window_length_s",
        "window_length_minutes",
        "window_start_seconds_local",
        "window_end_seconds_local",
        "window_midpoint_seconds_local",
        "time_of_day",
        "minutes_since_session_start",
        "hum_count",
        "hum_presence",
        "hum_rate_per_min",
        "area_m2",
        "spread_change_m2",
        "spread_change_m2_per_min",
        "speed_mps",
        "centroid_change_in_position_m",
        "observed_polygon_area_m2",
        "mean_distance_to_observed_centroid_m",
        "median_distance_to_observed_centroid_m",
        "window_centroid_x_m",
        "window_centroid_y_m",
        "recent_change_in_polygon_area_m2",
        "dispersion_change_m2_per_min",
        "recent_change_in_centroid_distance_m",
        "centroid_change_in_position_m_per_min",
        "n_collars_available",
        "n_audio_recorders_active",
        "audio_coverage_fraction",
        "gps_coverage_fraction",
        "n_valid_gps_seconds",
        "n_any_gps_seconds",
        "spread_valid_seconds",
        "biologically_impossible_speed_samples",
        "hard_gps_filter_version",
        "residual_gps_filter_version",
        "canonical_event_policy",
        "model_family",
        "model_offset",
    ]
    output = df[[column for column in required if column in df.columns]].copy()
    complete_cols = ["hum_count", "window_length_minutes", "session_id", "area_m2", "spread_change_m2_per_min", "speed_mps", "centroid_change_in_position_m"]
    summary = output.groupby("window_length_s", sort=True).agg(
        rows=("rq2_window_id", "size"),
        complete_model_rows=("rq2_window_id", lambda values: int(output.loc[values.index, complete_cols].notna().all(axis=1).sum())),
        sessions=("session_id", "nunique"),
        hum_count_total=("hum_count", "sum"),
        zero_hum_fraction=("hum_count", lambda values: float((values == 0).mean())),
        median_area_m2=("area_m2", "median"),
        median_spread_change_m2_per_min=("spread_change_m2_per_min", "median"),
        median_within_group_centroid_distance_m=("mean_distance_to_observed_centroid_m", "median"),
        median_speed_mps=("speed_mps", "median"),
        median_centroid_change_in_position_m=("centroid_change_in_position_m", "median"),
    ).reset_index()
    return output.reset_index(drop=True), summary


def _run_scientific_glmm_stage(ctx: LaunchContext) -> dict[str, Any]:
    input_path = ctx.paths.tables_dir / "rq2_model_input.csv"
    if not input_path.is_file():
        raise FileNotFoundError(f"missing RQ2 model input CSV: {input_path}")
    script_path = Path(__file__).resolve().with_name("local_glmmtmb_60s_workflow.R")
    if not script_path.is_file():
        raise FileNotFoundError(f"missing R workflow: {script_path}")
    mode = "smoke" if ctx.launch_manifest.get("profile") == "smoke" else "full"
    max_rows = "1200" if mode == "smoke" else "NA"
    command = [
        "Rscript",
        str(script_path),
        "--input",
        str(input_path),
        "--output-dir",
        str(ctx.paths.launch_dir),
        "--mode",
        mode,
        "--max-rows",
        max_rows,
    ]
    env = dict(**os.environ, STV_ANALYSIS_RUN_CONTEXT="hpc_slurm" if os.environ.get("SLURM_JOB_ID") else "local_launch_stage")
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[4], text=True, capture_output=True, check=False, env=env)
    log_dir = stage_cache_dir(ctx.paths, "scientific_glmm")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "rq2_glmmtmb_stdout.log").write_text(completed.stdout, encoding="utf8")
    (log_dir / "rq2_glmmtmb_stderr.log").write_text(completed.stderr, encoding="utf8")
    if completed.returncode != 0:
        raise RuntimeError(f"RQ2 R GLMM workflow failed with exit code {completed.returncode}; see {log_dir}")
    manifest_path = ctx.paths.tables_dir / "rq2_model_run_manifest.csv"
    run_manifest = pd.read_csv(manifest_path) if manifest_path.is_file() else pd.DataFrame()
    return {
        "stage_id": "scientific_glmm",
        "mode": mode,
        "model_runs": int(len(run_manifest.index)),
        "statuses": sorted(run_manifest["fit_status"].dropna().astype(str).unique().tolist()) if not run_manifest.empty and "fit_status" in run_manifest.columns else [],
    }


def _run_diagnostics_outputs_stage(ctx: LaunchContext) -> dict[str, Any]:
    model_input = pd.read_parquet(ctx.paths.tables_dir / "rq2_model_input.parquet")
    window_grid = pd.read_parquet(ctx.paths.tables_dir / "rq2_window_grid.parquet")
    availability = data_availability_table(model_input, window_grid)
    attrition = filtering_attrition_table(window_grid, model_input)
    diagnostics = model_diagnostics_table(ctx.paths.tables_dir)
    figure_index = write_diagnostic_figures(ctx.paths.launch_dir / "figures", model_input)
    final_index = final_output_index(ctx.paths.launch_dir)
    write_csv(availability, ctx.paths.tables_dir / "rq2_data_availability.csv")
    write_csv(attrition, ctx.paths.tables_dir / "rq2_filtering_attrition.csv")
    write_csv(diagnostics, ctx.paths.tables_dir / "rq2_model_diagnostics.csv")
    write_csv(figure_index, ctx.paths.tables_dir / "rq2_diagnostic_figure_index.csv")
    write_csv(final_index, ctx.paths.tables_dir / "rq2_final_output_index.csv")
    _write_report(ctx.paths.reports_dir / "subaim_8_diagnostics.md", _subaim_8_diagnostics_report(availability, attrition, diagnostics, figure_index))
    return {"stage_id": "diagnostics_outputs", "figures": int(len(figure_index.index)), "final_outputs": int(len(final_index.index))}


def _run_reports_stage(ctx: LaunchContext) -> dict[str, Any]:
    manifest = runtime_manifest(ctx)
    (ctx.paths.launch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    _write_report(ctx.paths.reports_dir / "README.md", final_readme(ctx))
    refresh_latest_alias(ctx.paths)
    return {"stage_id": "reports", "manifest_path": str(ctx.paths.launch_dir / "manifest.json")}


def add_projected_coordinates(trace_points: pd.DataFrame, origin: dict[str, float]) -> pd.DataFrame:
    projected = trace_points.copy()
    origin_lat = float(origin["origin_latitude"])
    origin_lon = float(origin["origin_longitude"])
    lon_scale = max(1e-9, LATITUDE_METERS_PER_DEGREE * cos(origin_lat * pi / 180.0))
    projected["x_m"] = (projected["longitude"] - origin_lon) * lon_scale
    projected["y_m"] = (projected["latitude"] - origin_lat) * LATITUDE_METERS_PER_DEGREE
    return projected


def _window_dispersion_metrics(grid: Any, start_s: float, end_s: float, *, dropped_index: int | None = None) -> dict[str, Any]:
    mask = _second_window_mask(grid, start_s, end_s)
    valid = grid.valid[mask].copy()
    x = grid.x_m[mask].copy()
    y = grid.y_m[mask].copy()
    if dropped_index is not None and 0 <= dropped_index < valid.shape[1]:
        valid[:, dropped_index] = False
        x[:, dropped_index] = np.nan
        y[:, dropped_index] = np.nan
    areas: list[float] = []
    distances: list[float] = []
    centroid_x: list[float] = []
    centroid_y: list[float] = []
    speeds: list[float] = []
    impossible_speed_samples = 0
    for idx in range(valid.shape[0]):
        point_mask = valid[idx]
        if int(point_mask.sum()) < 3:
            continue
        points = np.column_stack([x[idx, point_mask], y[idx, point_mask]])
        area = _convex_hull_area(points)
        if isfinite(area):
            areas.append(area)
        centroid = points.mean(axis=0)
        centroid_x.append(float(centroid[0]))
        centroid_y.append(float(centroid[1]))
        distances.extend(np.hypot(points[:, 0] - centroid[0], points[:, 1] - centroid[1]).tolist())
    for recorder_index in range(valid.shape[1]):
        recorder_mask = valid[:, recorder_index]
        if int(recorder_mask.sum()) < 2:
            continue
        seconds = grid.second_bins[mask][recorder_mask].astype(float)
        recorder_x = x[:, recorder_index][recorder_mask].astype(float)
        recorder_y = y[:, recorder_index][recorder_mask].astype(float)
        dt = np.diff(seconds)
        dist = np.hypot(np.diff(recorder_x), np.diff(recorder_y))
        speed = np.divide(dist, dt, out=np.full_like(dist, np.nan), where=dt > 0)
        impossible_speed_samples += int(np.count_nonzero(speed > BIOLOGICALLY_IMPOSSIBLE_SPEED_MPS))
        speed = speed[np.isfinite(speed) & (speed <= BIOLOGICALLY_IMPOSSIBLE_SPEED_MPS)]
        speeds.extend(speed.tolist())
    unique_collar_count = int(np.count_nonzero(valid.any(axis=0))) if valid.size else 0
    return {
        "observed_polygon_area_m2": float(np.nanmedian(areas)) if areas else np.nan,
        "mean_distance_to_observed_centroid_m": float(np.nanmean(distances)) if distances else np.nan,
        "median_distance_to_observed_centroid_m": float(np.nanmedian(distances)) if distances else np.nan,
        "window_centroid_x_m": float(np.nanmean(centroid_x)) if centroid_x else np.nan,
        "window_centroid_y_m": float(np.nanmean(centroid_y)) if centroid_y else np.nan,
        "mean_speed_mps": float(np.nanmean(speeds)) if speeds else 0.0,
        "n_collars_available": unique_collar_count,
        "spread_valid_seconds": int(len(areas)),
        "biologically_impossible_speed_samples": impossible_speed_samples,
    }


def _centroid_position_change(features: pd.DataFrame) -> pd.Series:
    changes = pd.Series(np.nan, index=features.index, dtype=float)
    for (_session, _window_length), idx in features.groupby(["day_session_id", "window_length_s"], sort=False).groups.items():
        ordered = features.loc[list(idx)].sort_values("window_start_seconds_local")
        dx = ordered["window_centroid_x_m"].diff()
        dy = ordered["window_centroid_y_m"].diff()
        current = np.hypot(dx, dy).fillna(0.0)
        changes.loc[ordered.index] = current
    return changes


def _per_second_polygon_areas(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["day_session_id", "second_bin", "per_second_polygon_area_m2"])
    collapsed = frame.copy()
    collapsed["second_bin"] = collapsed["seconds_since_midnight_local"].round().astype(int)
    per_recorder = (
        collapsed.groupby(["day_session_id", "second_bin", "recorder_id"], sort=False, as_index=False)
        .agg(x_m=("x_m", "median"), y_m=("y_m", "median"))
    )
    bbox = (
        per_recorder.groupby(["day_session_id", "second_bin"], sort=False)
        .agg(
            n_recorders=("recorder_id", "nunique"),
            x_min=("x_m", "min"),
            x_max=("x_m", "max"),
            y_min=("y_m", "min"),
            y_max=("y_m", "max"),
        )
        .reset_index()
    )
    bbox["bbox_area_m2"] = (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"])
    candidates = bbox.loc[(bbox["n_recorders"] >= 3) & (bbox["bbox_area_m2"] > DOCUMENTED_PADDOCK_AREA_M2)]
    rows = []
    for candidate in candidates.itertuples(index=False):
        unique = per_recorder.loc[
            (per_recorder["day_session_id"].astype(str) == str(candidate.day_session_id))
            & (per_recorder["second_bin"].astype(int) == int(candidate.second_bin))
        ]
        rows.append(
            {
                "day_session_id": str(candidate.day_session_id),
                "second_bin": int(candidate.second_bin),
                "per_second_polygon_area_m2": _convex_hull_area(unique[["x_m", "y_m"]].to_numpy(dtype=float)),
                "area_type": "exact_hull_candidate_bbox_exceeded_paddock",
            }
        )
    if rows:
        return pd.DataFrame(rows)
    max_bbox = bbox.loc[bbox["n_recorders"] >= 3, "bbox_area_m2"].max() if not bbox.empty else np.nan
    return pd.DataFrame(
        [
            {
                "day_session_id": str(frame["day_session_id"].iloc[0]) if "day_session_id" in frame.columns and not frame.empty else "",
                "second_bin": -1,
                "per_second_polygon_area_m2": float(max_bbox) if pd.notna(max_bbox) else np.nan,
                "area_type": "bbox_upper_bound_no_candidate_exceeded_paddock",
            }
        ]
    )


def _convex_hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return np.nan
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return np.nan
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except QhullError:
        return 0.0


def audio_coverage_intervals(trace_points: pd.DataFrame, segments: pd.DataFrame) -> dict[str, dict[str, list[tuple[float, float]]]]:
    trace_bounds = trace_points.groupby(["day_session_id", "recorder_id", "segment_id"], sort=True).agg(start_s=("seconds_since_midnight_local", "min"), end_s=("seconds_since_midnight_local", "max")).reset_index()
    merged = trace_bounds.merge(segments[["segment_id", "day_session_id", "recorder_id"]], on=["segment_id", "day_session_id", "recorder_id"], how="inner")
    coverage: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for row in merged.itertuples(index=False):
        coverage.setdefault(str(row.day_session_id), {}).setdefault(str(row.recorder_id), []).append((float(row.start_s), float(row.end_s) + 1.0))
    return coverage


def active_audio_recorders(session_coverage: dict[str, list[tuple[float, float]]], start_s: float, end_s: float) -> list[str]:
    return [recorder_id for recorder_id, intervals in session_coverage.items() if any(interval_start < end_s and interval_end > start_s for interval_start, interval_end in intervals)]


def hum_event_frame(events: pd.DataFrame) -> pd.DataFrame:
    explicit_non_hum_labels = {"human_talking", "rumble"}
    label = events["reviewed_resolved_label_semantic"]
    return events.loc[~label.isin(explicit_non_hum_labels)].copy()


def _load_grid_cache(ctx: LaunchContext, stage_id: str) -> dict[str, Any]:
    cache_dir = stage_cache_dir(ctx.paths, stage_id)
    grids: dict[str, Any] = {}
    for day_session_id in ctx.stage_task_values(stage_id):
        path = cache_dir / f"{slugify_token(day_session_id)}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing cached grid for stage {stage_id} and session {day_session_id}: {path}")
        grids[day_session_id] = read_session_trace_grid(path)
    return grids


def _selected_sessions(ctx: LaunchContext) -> list[str]:
    return [str(value) for value in ctx.launch_manifest["analysis"]["selected_day_session_ids"]]


def _projection_origin(ctx: LaunchContext) -> dict[str, float]:
    raw_origin = dict(ctx.launch_manifest["analysis"]["projection_origin"])
    return {"origin_latitude": float(raw_origin["origin_latitude"]), "origin_longitude": float(raw_origin["origin_longitude"])}


def _select_sessions(df: pd.DataFrame, selected_sessions: list[str]) -> pd.DataFrame:
    if "day_session_id" not in df.columns:
        return df.copy()
    return df.loc[df["day_session_id"].astype(str).isin(selected_sessions)].copy()


def _second_window_mask(grid: Any, start_s: float, end_s: float) -> np.ndarray:
    return (grid.second_bins.astype(float) >= float(start_s)) & (grid.second_bins.astype(float) < float(end_s))


def _unit_fields(row: Any) -> dict[str, Any]:
    return {
        "rq2_window_id": str(row.rq2_window_id),
        "day_session_id": str(row.day_session_id),
        "date_local": str(row.date_local),
        "session_id": str(row.session_id),
        "window_length_s": int(row.window_length_s),
        "window_start_seconds_local": float(row.window_start_seconds_local),
        "window_end_seconds_local": float(row.window_end_seconds_local),
        "window_midpoint_seconds_local": float(row.window_midpoint_seconds_local),
        "window_start_local": row.window_start_local,
        "window_end_local": row.window_end_local,
    }


def _empty_window_grid_row(row: Any, gps_blocker_rule: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **_unit_fields(row),
        "n_collars_available": 0,
        "n_valid_gps_seconds": 0,
        "n_any_gps_seconds": 0,
        "gps_coverage_fraction": 0.0,
        "audio_coverage_ok": False,
        "n_audio_recorders_active": 0,
        "audio_coverage_fraction": 0.0,
        "window_eligible": False,
        "window_ineligibility_reason": reason,
        "hard_gps_filter_version": str(gps_blocker_rule.get("name", "unknown")),
        "residual_gps_filter_version": RESIDUAL_GPS_FILTER_VERSION,
    }


def _ineligibility_reason(gps_coverage_fraction: float, n_collars_available: int, audio_coverage_ok: bool) -> str:
    reasons = []
    if gps_coverage_fraction <= 0:
        reasons.append("no_valid_gps_seconds_with_min_3_collars")
    if n_collars_available < 3:
        reasons.append("insufficient_collars_available")
    if not audio_coverage_ok:
        reasons.append("audio_coverage_not_ok")
    return "|".join(reasons)


def _feasibility_row(variable_name: str, exists: str, source: str, derivation: str, assumptions: str) -> dict[str, str]:
    return {"variable_name": variable_name, "exists": exists, "source_table_or_file": source, "derivation": derivation, "assumptions": assumptions, "failure_modes": "", "recommended_alternative_if_not_feasible": ""}


def _empty_dispersion_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "rq2_window_id",
            "day_session_id",
            "date_local",
            "session_id",
            "window_length_s",
            "window_start_seconds_local",
            "window_end_seconds_local",
            "window_midpoint_seconds_local",
            "observed_polygon_area_m2",
            "mean_distance_to_observed_centroid_m",
            "median_distance_to_observed_centroid_m",
            "window_centroid_x_m",
            "window_centroid_y_m",
            "mean_speed_mps",
            "n_collars_available",
            "recent_change_in_polygon_area_m2",
            "dispersion_change_m2_per_min",
            "recent_change_in_centroid_distance_m",
            "centroid_change_in_position_m",
            "centroid_change_in_position_m_per_min",
            "spread_valid_seconds",
            "biologically_impossible_speed_samples",
        ]
    )


def _empty_model_input() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "rq2_window_id",
            "day_session_id",
            "date_local",
            "session_id",
            "window_length_s",
            "window_length_minutes",
            "hum_count",
            "hum_rate_per_min",
            "area_m2",
            "spread_change_m2",
            "spread_change_m2_per_min",
            "speed_mps",
            "centroid_change_in_position_m",
            "observed_polygon_area_m2",
            "mean_distance_to_observed_centroid_m",
            "median_distance_to_observed_centroid_m",
            "recent_change_in_polygon_area_m2",
            "dispersion_change_m2_per_min",
        ]
    )


def _gps_qc_trace_point_summary(projected: pd.DataFrame, qc_trace: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    before = projected.groupby("day_session_id", sort=True).size().rename("input_points")
    after = qc_trace.groupby("day_session_id", sort=True).size().rename("qc_points") if not qc_trace.empty else pd.Series(dtype=int, name="qc_points")
    removed = outliers.groupby("day_session_id", sort=True).size().rename("removed_points") if not outliers.empty else pd.Series(dtype=int, name="removed_points")
    summary = pd.concat([before, after, removed], axis=1).fillna(0).reset_index()
    summary["input_points"] = summary["input_points"].astype(int)
    summary["qc_points"] = summary["qc_points"].astype(int)
    summary["removed_points"] = summary["removed_points"].astype(int)
    summary["removed_fraction"] = summary["removed_points"] / summary["input_points"].clip(lower=1)
    return summary


def _write_gps_audit_figures(figures_dir: Path, trace: pd.DataFrame, qc_trace: pd.DataFrame, audit: pd.DataFrame, origin: dict[str, float]) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    projected_raw = add_projected_coordinates(trace, origin)
    projected_qc = add_projected_coordinates(qc_trace, origin)
    sample_raw = projected_raw.sample(n=min(5000, len(projected_raw.index)), random_state=1) if len(projected_raw.index) else projected_raw
    sample_qc = projected_qc.sample(n=min(5000, len(projected_qc.index)), random_state=1) if len(projected_qc.index) else projected_qc
    plt.figure(figsize=(6, 5))
    plt.scatter(sample_raw["x_m"], sample_raw["y_m"], s=2, alpha=0.2, label="hard-filter input")
    plt.scatter(sample_qc["x_m"], sample_qc["y_m"], s=2, alpha=0.2, label="RQ2 residual-QC")
    plt.xlabel("x_m")
    plt.ylabel("y_m")
    plt.legend(markerscale=4)
    plt.tight_layout()
    plt.savefig(figures_dir / "rq2_gps_qc_cloud.png", dpi=160)
    plt.close()
    plt.figure(figsize=(7, 4))
    x = np.arange(len(audit.index))
    labels = audit["day_session_id"].astype(str).tolist()
    plt.bar(x - 0.15, audit["all_point_hull_to_documented_paddock_ratio"], width=0.3, label="all hull / paddock")
    plt.bar(x + 0.15, audit["robust_envelope_to_documented_paddock_ratio"], width=0.3, label="robust envelope / paddock")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "rq2_gps_session_hull_audit.png", dpi=160)
    plt.close()


def window_coverage_summary(window_grid: pd.DataFrame) -> pd.DataFrame:
    return window_grid.groupby("window_length_s", sort=True).agg(
        windows=("rq2_window_id", "size"),
        eligible_windows=("window_eligible", "sum"),
        median_gps_coverage_fraction=("gps_coverage_fraction", "median"),
        median_audio_recorders_active=("n_audio_recorders_active", "median"),
        median_collars_available=("n_collars_available", "median"),
    ).reset_index()


def hard_gps_filter_summary(bundle: Any, window_grid: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hard_gps_filter_version": str(bundle.gps_blocker_rule.get("name", "unknown")),
                "description": str(bundle.gps_blocker_rule.get("description", "")),
                "selection_source": str(bundle.gps_blocker_rule.get("selection_source", "")),
                "buffer_meters": float(bundle.gps_blocker_rule.get("buffer_meters", np.nan)),
                "filtered_trace_points": int(len(bundle.trace_points_filtered.index)),
                "eligible_windows": int(window_grid["window_eligible"].sum()),
            }
        ]
    )


def dispersion_feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["window_length_s", "rows", "median_polygon_area_m2"])
    return features.groupby("window_length_s", sort=True).agg(
        rows=("rq2_window_id", "size"),
        median_polygon_area_m2=("observed_polygon_area_m2", "median"),
        median_within_group_spread_m=("mean_distance_to_observed_centroid_m", "median"),
        median_speed_mps=("mean_speed_mps", "median"),
        median_centroid_change_in_position_m=("centroid_change_in_position_m", "median"),
        impossible_speed_samples=("biologically_impossible_speed_samples", "sum"),
    ).reset_index()


def dispersion_change_summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["window_length_s", "rows"])
    return features.groupby("window_length_s", sort=True).agg(
        rows=("rq2_window_id", "size"),
        mean_recent_change_in_polygon_area_m2=("recent_change_in_polygon_area_m2", "mean"),
        median_recent_change_in_polygon_area_m2=("recent_change_in_polygon_area_m2", "median"),
        mean_dispersion_change_m2_per_min=("dispersion_change_m2_per_min", "mean"),
        median_dispersion_change_m2_per_min=("dispersion_change_m2_per_min", "median"),
        median_centroid_change_in_position_m=("centroid_change_in_position_m", "median"),
    ).reset_index()


def hum_rate_summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["window_length_s", "windows"])
    return features.groupby("window_length_s", sort=True).agg(
        windows=("rq2_window_id", "size"),
        hum_count_total=("hum_count", "sum"),
        hum_presence_windows=("hum_presence", "sum"),
        mean_hum_rate_per_min=("hum_rate_per_min", "mean"),
        zero_hum_fraction=("hum_count", lambda values: float((values == 0).mean())),
        median_audio_coverage_fraction=("audio_coverage_fraction", "median"),
    ).reset_index()


def data_availability_table(model_input: pd.DataFrame, window_grid: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "window_grid_rows", "value": int(len(window_grid.index))},
            {"metric": "eligible_window_rows", "value": int(window_grid["window_eligible"].sum())},
            {"metric": "model_input_rows", "value": int(len(model_input.index))},
            {"metric": "model_input_sessions", "value": int(model_input["session_id"].nunique()) if not model_input.empty else 0},
            {"metric": "model_input_hum_count_total", "value": int(model_input["hum_count"].sum()) if not model_input.empty else 0},
        ]
    )


def filtering_attrition_table(window_grid: pd.DataFrame, model_input: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"step": "rq2_units", "rows": int(len(window_grid.index))},
            {"step": "eligible_window_grid", "rows": int(window_grid["window_eligible"].sum())},
            {"step": "rq2_model_input", "rows": int(len(model_input.index))},
        ]
    )


def model_diagnostics_table(tables_dir: Path) -> pd.DataFrame:
    run_manifest = _safe_read_csv(tables_dir / "rq2_model_run_manifest.csv")
    dharma = _safe_read_csv(tables_dir / "rq2_dharma_standard_tests.csv")
    temporal = _safe_read_csv(tables_dir / "rq2_dharma_temporal_autocorrelation.csv")
    rows: list[dict[str, Any]] = []
    for frame, source in ((run_manifest, "run_manifest"), (dharma, "dharma_standard_tests"), (temporal, "dharma_temporal_autocorrelation")):
        if frame.empty:
            rows.append({"diagnostic": source, "status": "missing", "model_id": "", "value": ""})
            continue
        for row in frame.to_dict("records"):
            rows.append(
                {
                    "diagnostic": source,
                    "status": row.get("fit_status", row.get("status", "computed")),
                    "model_id": row.get("model_id", ""),
                    "value": row.get("p_value", row.get("AIC", "")),
                }
            )
    return pd.DataFrame(rows)


def write_diagnostic_figures(figures_dir: Path, model_input: pd.DataFrame) -> pd.DataFrame:
    figures_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("rq2_hum_count_distribution.png", lambda p: _hist(p, model_input["hum_count"], "Hum count", "Windows")),
        ("rq2_area_distribution.png", lambda p: _hist(p, model_input["area_m2"], "Observed polygon area m2", "Windows")),
        ("rq2_spread_change_distribution.png", lambda p: _hist(p, model_input["spread_change_m2_per_min"], "Observed polygon-area change m2/min", "Windows")),
        ("rq2_speed_distribution.png", lambda p: _hist(p, model_input["speed_mps"], "Mean group member speed m/s", "Windows")),
        ("rq2_centroid_position_change_distribution.png", lambda p: _hist(p, model_input["centroid_change_in_position_m"], "Centroid position change m", "Windows")),
    ]
    rows = []
    for filename, writer in specs:
        path = figures_dir / filename
        if not path.is_file():
            writer(path)
        rows.append({"figure_name": filename, "path": f"figures/{filename}", "status": "written" if path.is_file() else "missing"})
    extra_patterns = (
        "rq2_dharma_*.png",
        "rq2_residuals_vs_window_*.png",
        "rq2_residual_acf_*.png",
        "rq2_fixed_effects_*.png",
    )
    extra_paths: list[Path] = []
    for pattern in extra_patterns:
        extra_paths.extend(sorted(figures_dir.glob(pattern)))
    for path in extra_paths:
        rel = f"figures/{path.name}"
        if rel not in {row["path"] for row in rows}:
            rows.append({"figure_name": path.name, "path": rel, "status": "written"})
    return pd.DataFrame(rows)


def _hist(path: Path, values: pd.Series, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    clean = pd.to_numeric(values, errors="coerce").dropna()
    plt.hist(clean, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def final_output_index(launch_dir: Path) -> pd.DataFrame:
    rows = []
    for subdir in ("tables", "reports", "figures"):
        for path in sorted((launch_dir / subdir).glob("*")):
            if path.is_file():
                rows.append({"artifact_type": subdir.rstrip("s"), "path": str(path.relative_to(launch_dir)), "bytes": int(path.stat().st_size)})
    return pd.DataFrame(rows)


def runtime_manifest(ctx: LaunchContext) -> dict[str, Any]:
    tables = sorted(path.name for path in ctx.paths.tables_dir.glob("*") if path.is_file())
    reports = sorted(path.name for path in ctx.paths.reports_dir.glob("*") if path.is_file())
    figures = sorted(path.name for path in (ctx.paths.launch_dir / "figures").glob("*") if path.is_file())
    row_counts: dict[str, int] = {}
    for table in ctx.paths.tables_dir.glob("*.csv"):
        try:
            row_counts[table.name] = int(len(pd.read_csv(table).index))
        except Exception:
            row_counts[table.name] = -1
    for table in ctx.paths.tables_dir.glob("*.parquet"):
        try:
            row_counts[table.name] = int(len(pd.read_parquet(table).index))
        except Exception:
            row_counts[table.name] = -1
    stage_status = {spec.stage_id: read_json(ctx.paths.stage_progress_dir / f"{spec.stage_id}.json").get("status") for spec in ctx.definition.stage_specs}
    if stage_status.get("reports") == "running":
        stage_status["reports"] = "completed"
    return {
        "analysis_id": ANALYSIS_ID,
        "analysis_version": ANALYSIS_VERSION,
        "launch_label": str(ctx.launch_manifest["launch_label"]),
        "profile": str(ctx.launch_manifest["profile"]),
        "selected_sessions": list(ctx.launch_manifest["analysis"]["selected_day_session_ids"]),
        "event_corpus_version": "analysis-base launch snapshot",
        "event_corpus_count": _manifest_hum_count(ctx),
        "stage_status": stage_status,
        "canonical_artifact_paths": {"tables": [f"tables/{name}" for name in tables], "reports": [f"reports/{name}" for name in reports], "figures": [f"figures/{name}" for name in figures]},
        "row_counts": row_counts,
        "documented_paddock_dimensions_m": {"length": DOCUMENTED_PADDOCK_LENGTH_M, "width": DOCUMENTED_PADDOCK_WIDTH_M},
        "documented_paddock_area_m2": DOCUMENTED_PADDOCK_AREA_M2,
        "existing_hard_gps_filter_version": _hard_filter_version(ctx),
        "residual_gps_filter_version": RESIDUAL_GPS_FILTER_VERSION,
        "residual_gps_filter_thresholds": read_json(stage_cache_dir(ctx.paths, "gps_filter_audit") / "residual_gps_filter_manifest.json"),
        "model_fit_resource_profile": MODEL_FIT_RESOURCE_PROFILE,
        "random_effects_structure_version": "rq2_session_maximal_random_slopes_v1",
        "model_input_path": "tables/rq2_model_input.parquet",
        "glmmtmb_model_runs": _safe_csv_records(ctx.paths.tables_dir / "rq2_model_run_manifest.csv"),
        "predictor_standardization": _safe_csv_records(ctx.paths.tables_dir / "rq2_predictor_standardization.csv"),
        "dharma_standard_tests": _safe_csv_records(ctx.paths.tables_dir / "rq2_dharma_standard_tests.csv"),
        "input_manifest_path": "base_snapshot/manifest.json",
        "specs_manifest": ctx.launch_manifest.get("specs_manifest", {}),
    }


def final_readme(ctx: LaunchContext) -> str:
    run_manifest = _safe_read_csv(ctx.paths.tables_dir / "rq2_model_run_manifest.csv")
    primary_status = ", ".join(run_manifest["fit_status"].dropna().astype(str).unique().tolist()) if not run_manifest.empty and "fit_status" in run_manifest.columns else "unknown"
    return f"""# RQ2 Hum Timing / Group Dispersion GLMM

- Analysis ID: `{ANALYSIS_ID}`
- Analysis version: `{ANALYSIS_VERSION}`
- Primary window: `{PRIMARY_WINDOW_LENGTH_S}` seconds
- Model statuses: `{primary_status}`
- Response: `hum_count`
- Offset: `log(window_length_minutes)`
- Random-effects structure: session-level random intercept plus maximal session-level random slopes matching each fixed-effect surface.
- Fixed effects: interaction and additive variants of `area_m2`, `spread_change_m2_per_min`, `speed_mps`, and `centroid_change_in_position_m`.
- Families: Poisson and negative binomial NB1, both fitted with `glmmTMB`.
- Diagnostics: DHARMa standard tests, DHARMa temporal autocorrelation tests, residual-vs-window plots, and residual ACF plots.
- Event policy: `None`/unlabeled non-explicit rows and `alpaca_hum` are canonical hums; `rumble` and `human_talking` are excluded.
- Interpretation boundary: observed collared-subnetwork association only; not caller identity, not full-herd movement, and not a causal claim.

The detailed R report is `reports/rq2_methods_results.md`.
"""


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict("records")


def _hard_filter_version(ctx: LaunchContext) -> str:
    path = ctx.paths.base_snapshot_dir / "gps_blocker_rule.json"
    return str(read_json(path).get("name", "unknown")) if path.is_file() else "unknown"


def _manifest_hum_count(ctx: LaunchContext) -> int:
    summary_path = stage_cache_dir(ctx.paths, "units") / "unit_summary.json"
    if not summary_path.is_file():
        return 0
    return int(json.loads(summary_path.read_text(encoding="utf8")).get("canonical_hum_event_count", 0))


def _write_report(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf8")


def _subaim_1_report(summary: dict[str, Any]) -> str:
    return f"""# Sub-aim 1: RQ2 Units

- Event corpus version: {summary['event_corpus_version']}
- Canonical hum-event count used: {summary['canonical_hum_event_count']}
- Event policy: {summary['event_label_policy']}
- Session count: {summary['session_count']}
- Recorder count: {summary['recorder_count']}
- Window rows: {summary['window_rows']}
- Window lengths: {summary['window_lengths_s']} (60 s only)
"""


def _subaim_2_gps_audit_report(audit: pd.DataFrame, point_summary: pd.DataFrame, paddock: pd.DataFrame) -> str:
    removed = int(audit["residual_outlier_points"].sum()) if not audit.empty else 0
    exceeds = bool(paddock["remaining_per_second_polygon_exceeds_9800_m2"].any()) if not paddock.empty else False
    return f"""# Sub-aim 2: GPS Audit

- Existing hard GPS filter is consumed from `base_snapshot/trace_points_filtered.parquet`.
- Residual GPS filter version: {RESIDUAL_GPS_FILTER_VERSION}
- Residual extreme-position points removed: {removed}
- Documented paddock area: {DOCUMENTED_PADDOCK_AREA_M2} m2 ({DOCUMENTED_PADDOCK_LENGTH_M} m x {DOCUMENTED_PADDOCK_WIDTH_M} m)
- Remaining per-second observed polygon exceeds documented paddock area: {exceeds}
- QC trace rows after filtering: {int(point_summary['qc_points'].sum()) if not point_summary.empty else 0}
- Trace grids consume `cache/gps_filter_audit/trace_points_rq2_gps_qc.parquet`.
"""


def _subaim_3_window_grid_report(window_grid: pd.DataFrame, gps_blocker_rule: dict[str, Any]) -> str:
    return f"""# Sub-aim 3: Trace Grids and Valid Window Grid

- Window rows: {len(window_grid.index)}
- Eligible windows: {int(window_grid['window_eligible'].sum())}
- Hard GPS filter: {gps_blocker_rule.get('name', 'unknown')}
- Residual GPS filter: {RESIDUAL_GPS_FILTER_VERSION}
- Interpolation: none; only available aligned GPS fixes / aligned seconds are used.
"""


def _subaim_4_dispersion_report(features: pd.DataFrame) -> str:
    return f"""# Sub-aim 4: Dispersion and Movement Predictors

- Feature rows: {len(features.index)}
- Median observed polygon area m2: {float(features['observed_polygon_area_m2'].median()) if not features.empty else 'NA'}
- Median spread change m2/min: {float(features['dispersion_change_m2_per_min'].median()) if not features.empty else 'NA'}
- Median within-group centroid distance m (descriptive only): {float(features['mean_distance_to_observed_centroid_m'].median()) if not features.empty else 'NA'}
- Median speed m/s: {float(features['mean_speed_mps'].median()) if not features.empty else 'NA'}
- Median centroid position change m: {float(features['centroid_change_in_position_m'].median()) if not features.empty else 'NA'}
- Interpretation boundary: all predictors describe the observed collared animals, not the full herd.
"""


def _subaim_5_hum_report(features: pd.DataFrame) -> str:
    return f"""# Sub-aim 5: Hum Counts

- Feature rows: {len(features.index)}
- Total assigned hum events in eligible windows: {int(features['hum_count'].sum()) if not features.empty else 0}
- Event assignment: midpoint within same day_session_id and fixed window.
- Count response retained as `hum_count`; `hum_rate_per_min` is descriptive only because exposure enters through the offset.
"""


def _subaim_6_model_input_report(model_input: pd.DataFrame, summary: pd.DataFrame) -> str:
    return f"""# Sub-aim 6: Count GLMM Input

- Model input rows: {len(model_input.index)}
- Summary rows: {len(summary.index)}
- Response: `hum_count`
- Offset: `log(window_length_minutes)`
- Fixed effects: `area_m2 * spread_change_m2_per_min + speed_mps + centroid_change_in_position_m`
- Additive companion fixed effects: `area_m2 + spread_change_m2_per_min + speed_mps + centroid_change_in_position_m`
- Random effect base for R: random intercept by `session_id`.
"""


def _subaim_8_diagnostics_report(availability: pd.DataFrame, attrition: pd.DataFrame, diagnostics: pd.DataFrame, figure_index: pd.DataFrame) -> str:
    return f"""# Sub-aim 8: Diagnostics and Output Index

- Data availability rows: {len(availability.index)}
- Filtering attrition rows: {len(attrition.index)}
- Model diagnostic rows: {len(diagnostics.index)}
- Diagnostic figures: {len(figure_index.index)}
- Count-model diagnostics are DHARMa tests and residual temporal plots.
"""

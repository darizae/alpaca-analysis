from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from stv_core.geometry import buffered_bbox_polygon

from ..framework import AnalysisDefinition, AnalysisStageSpec, LaunchContext, stage_cache_dir, slugify_token, write_json
from .contracts import ANALYSIS_ID, HEAVY_COMPUTE_LABEL, LOCAL_OK_LABEL, OUTPUT_DIRNAME
from .io import load_base_bundle, write_parquet
from .reports import build_runtime_manifest, write_runtime_reports
from .subaim1_units import build_analysis_units
from .subaim2_windows import (
    build_session_trace_grids,
    projection_origin,
    read_session_trace_grid,
    summarize_event_windows,
    trace_sync_summary,
    write_session_trace_grid,
)
from .subaim3_validity import (
    blocker_audit_from_segment_summaries,
    decorate_window_validity,
    scan_blocked_trace_points_for_segment,
)
from .subaim4_centroid import (
    centroid_summary,
    compute_observed_centroid_distance,
    directional_movement_tables,
)
from .subaim5_covariates import (
    compute_baseline_covariates_for_session,
    covariate_summary,
)
from .subaim6_matching import global_match_scales, match_controls_for_session, match_quality_summary
from .subaim7_model_input import build_model_input


DOC_PATHS = (
    "PRIMER.md",
    "docs/rq1_post_hum_cohesion/rq1_post_hum_cohesion_experimental_plan.md",
    "docs/rq1_post_hum_cohesion/ARTIFACT_CONTRACT.md",
)

STAGE_SPECS = (
    AnalysisStageSpec(
        stage_id="units",
        description="Build Sub-aim 1 analysis units and feasibility log.",
        compute_class=LOCAL_OK_LABEL,
        resource_profile="light",
    ),
    AnalysisStageSpec(
        stage_id="filtered_trace_grids",
        description="Build filtered per-session trace grids for Sub-aim 2.",
        compute_class=HEAVY_COMPUTE_LABEL,
        resource_profile="heavy",
        dependencies=("units",),
    ),
    AnalysisStageSpec(
        stage_id="filtered_window_inventory",
        description="Aggregate filtered per-session grids into the Sub-aim 2 window inventory.",
        compute_class=LOCAL_OK_LABEL,
        resource_profile="light",
        dependencies=("filtered_trace_grids",),
    ),
    AnalysisStageSpec(
        stage_id="blocker_scan",
        description="Scan segment sidecars for blocked raw GPS fixes for Sub-aim 3.",
        compute_class=HEAVY_COMPUTE_LABEL,
        resource_profile="heavy",
        dependencies=("units",),
    ),
    AnalysisStageSpec(
        stage_id="blocker_merge",
        description="Merge blocker-scan outputs into the canonical blocked trace-point table and audit.",
        compute_class=LOCAL_OK_LABEL,
        resource_profile="light",
        dependencies=("blocker_scan",),
    ),
    AnalysisStageSpec(
        stage_id="raw_trace_grids",
        description="Rebuild raw per-session grids after blocker replay for Sub-aim 3.",
        compute_class=HEAVY_COMPUTE_LABEL,
        resource_profile="heavy",
        dependencies=("blocker_merge",),
    ),
    AnalysisStageSpec(
        stage_id="window_validity",
        description="Decorate the final Sub-aim 3 validity inventory.",
        compute_class=LOCAL_OK_LABEL,
        resource_profile="light",
        dependencies=("filtered_window_inventory", "raw_trace_grids"),
    ),
    AnalysisStageSpec(
        stage_id="centroid_distance",
        description="Compute per-session Sub-aim 4 centroid-distance outputs.",
        compute_class=HEAVY_COMPUTE_LABEL,
        resource_profile="heavy",
        dependencies=("window_validity",),
    ),
    AnalysisStageSpec(
        stage_id="baseline_covariates",
        description="Materialize per-session Sub-aim 5 baseline spatial-state covariates.",
        compute_class=HEAVY_COMPUTE_LABEL,
        resource_profile="heavy",
        dependencies=("centroid_distance",),
    ),
    AnalysisStageSpec(
        stage_id="covariate_table",
        description="Assemble the canonical Sub-aim 5 baseline covariate table.",
        compute_class=LOCAL_OK_LABEL,
        resource_profile="light",
        dependencies=("baseline_covariates",),
    ),
    AnalysisStageSpec(
        stage_id="matched_controls",
        description="Construct matched control windows without hums for Sub-aim 6.",
        compute_class=HEAVY_COMPUTE_LABEL,
        resource_profile="heavy",
        dependencies=("covariate_table",),
    ),
    AnalysisStageSpec(
        stage_id="model_input",
        description="Assemble the retained RQ1 scientific-LMM model input.",
        compute_class=LOCAL_OK_LABEL,
        resource_profile="light",
        dependencies=("matched_controls",),
    ),
    AnalysisStageSpec(
        stage_id="reports",
        description="Write model-input launch reports and manifest.",
        compute_class=LOCAL_OK_LABEL,
        resource_profile="light",
        dependencies=("model_input", "blocker_merge"),
    ),
)


ANALYSIS_DEFINITION = AnalysisDefinition(
    analysis_id=ANALYSIS_ID,
    output_dirname=OUTPUT_DIRNAME,
    docs_to_snapshot=DOC_PATHS,
    stage_specs=STAGE_SPECS,
    initialize_launch=lambda ctx, profile: initialize_launch(ctx, profile),
    run_stage=lambda ctx, spec, task_value: run_rq1_stage(ctx, spec.stage_id, task_value),
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
        blocker_session = "2025-02-03:session-001"
        selected_sessions = [blocker_session] if blocker_session in available_sessions else [available_sessions[0]]
    elif profile == "full":
        selected_sessions = available_sessions
    else:
        raise ValueError(f"unsupported profile: {profile}")

    selected_trace_points = _select_sessions(bundle.trace_points_filtered, selected_sessions)
    selected_segments = _select_sessions(bundle.segments, selected_sessions)
    selected_events = _select_sessions(bundle.events, selected_sessions)
    origin = projection_origin(selected_trace_points)
    stage_tasks = {
        "units": [],
        "filtered_trace_grids": selected_sessions,
        "filtered_window_inventory": [],
        "blocker_scan": selected_segments["segment_id"].astype(str).sort_values().tolist(),
        "blocker_merge": [],
        "raw_trace_grids": selected_sessions,
        "window_validity": [],
        "centroid_distance": selected_sessions,
        "baseline_covariates": selected_sessions,
        "covariate_table": [],
        "matched_controls": selected_sessions,
        "model_input": [],
        "reports": [],
    }
    return {
        "analysis": {
            "profile": profile,
            "selected_day_session_ids": selected_sessions,
            "selected_event_rows": int(len(selected_events.index)),
            "selected_segment_rows": int(len(selected_segments.index)),
            "selected_trace_point_rows": int(len(selected_trace_points.index)),
            "projection_origin": origin,
            "heavy_compute_stage_ids": [
                spec.stage_id for spec in STAGE_SPECS if spec.compute_class == HEAVY_COMPUTE_LABEL
            ],
            "smoke_session_rule": "prefer blocker-affected session 2025-02-03:session-001",
        },
        "stage_tasks": stage_tasks,
    }


def run_rq1_stage(ctx: LaunchContext, stage_id: str, task_value: str | None) -> dict[str, Any]:
    if stage_id == "units":
        return _run_units_stage(ctx)
    if stage_id == "filtered_trace_grids":
        assert task_value is not None
        return _run_filtered_trace_grid_task(ctx, task_value)
    if stage_id == "filtered_window_inventory":
        return _run_filtered_window_inventory_stage(ctx)
    if stage_id == "blocker_scan":
        assert task_value is not None
        return _run_blocker_scan_task(ctx, task_value)
    if stage_id == "blocker_merge":
        return _run_blocker_merge_stage(ctx)
    if stage_id == "raw_trace_grids":
        assert task_value is not None
        return _run_raw_trace_grid_task(ctx, task_value)
    if stage_id == "window_validity":
        return _run_window_validity_stage(ctx)
    if stage_id == "centroid_distance":
        assert task_value is not None
        return _run_centroid_distance_task(ctx, task_value)
    if stage_id == "baseline_covariates":
        assert task_value is not None
        return _run_baseline_covariates_task(ctx, task_value)
    if stage_id == "covariate_table":
        return _run_covariate_table_stage(ctx)
    if stage_id == "matched_controls":
        assert task_value is not None
        return _run_matched_controls_task(ctx, task_value)
    if stage_id == "model_input":
        return _run_model_input_stage(ctx)
    if stage_id == "reports":
        return _run_reports_stage(ctx)
    raise ValueError(f"unsupported stage_id: {stage_id}")


def _run_units_stage(ctx: LaunchContext) -> dict[str, Any]:
    bundle = load_base_bundle(_base_snapshot_manifest_path(ctx))
    sessions = _selected_sessions(ctx)
    analysis_units, unit_summary, feasibility_log = build_analysis_units(
        _select_sessions(bundle.events, sessions),
        bundle.recorders,
    )
    write_parquet(analysis_units, ctx.paths.tables_dir / "analysis_units.parquet")
    feasibility_log.to_csv(ctx.paths.tables_dir / "variable_feasibility_log.csv", index=False)
    write_json(stage_cache_dir(ctx.paths, "units") / "unit_summary.json", unit_summary)
    return {
        "stage_id": "units",
        "analysis_units": int(len(analysis_units.index)),
        "day_sessions": int(analysis_units["day_session_id"].nunique()),
    }


def _run_filtered_trace_grid_task(ctx: LaunchContext, day_session_id: str) -> dict[str, Any]:
    cache_dir = stage_cache_dir(ctx.paths, "filtered_trace_grids")
    token = slugify_token(day_session_id)
    grid_path = cache_dir / f"{token}.npz"
    summary_path = cache_dir / f"{token}.json"
    if grid_path.exists() and summary_path.exists():
        return {
            "stage_id": "filtered_trace_grids",
            "day_session_id": day_session_id,
            "grid_path": str(grid_path),
            "status": "cached",
        }

    bundle = load_base_bundle(_base_snapshot_manifest_path(ctx))
    session_points = bundle.trace_points_filtered.loc[
        bundle.trace_points_filtered["day_session_id"].astype(str) == day_session_id
    ].copy()
    origin = _projection_origin(ctx)
    grids, sync_summary = build_session_trace_grids(
        session_points,
        origin_lat=float(origin["origin_latitude"]),
        origin_lon=float(origin["origin_longitude"]),
    )
    grid = grids[day_session_id]
    write_session_trace_grid(grid, grid_path)
    write_json(summary_path, sync_summary)
    return {
        "stage_id": "filtered_trace_grids",
        "day_session_id": day_session_id,
        "grid_path": str(grid_path),
        "summary_path": str(summary_path),
    }


def _run_filtered_window_inventory_stage(ctx: LaunchContext) -> dict[str, Any]:
    analysis_units = pd.read_parquet(ctx.paths.tables_dir / "analysis_units.parquet")
    grids = _load_grid_cache(ctx, "filtered_trace_grids")
    filtered_window_inventory = summarize_event_windows(analysis_units, grids, prefix="filtered")
    cache_dir = stage_cache_dir(ctx.paths, "filtered_window_inventory")
    write_parquet(filtered_window_inventory, cache_dir / "filtered_window_inventory.parquet")

    bundle = load_base_bundle(_base_snapshot_manifest_path(ctx))
    sessions = _selected_sessions(ctx)
    sync_summary = trace_sync_summary(_select_sessions(bundle.trace_points_filtered, sessions))
    write_json(cache_dir / "sync_summary.json", sync_summary)
    return {
        "stage_id": "filtered_window_inventory",
        "analysis_windows": int(len(filtered_window_inventory.index)),
    }


def _run_blocker_scan_task(ctx: LaunchContext, segment_id: str) -> dict[str, Any]:
    cache_dir = stage_cache_dir(ctx.paths, "blocker_scan")
    token = slugify_token(segment_id)
    parquet_path = cache_dir / f"{token}.parquet"
    summary_path = cache_dir / f"{token}.json"
    if parquet_path.exists() and summary_path.exists():
        return {
            "stage_id": "blocker_scan",
            "segment_id": segment_id,
            "parquet_path": str(parquet_path),
            "status": "cached",
        }

    bundle = load_base_bundle(_base_snapshot_manifest_path(ctx))
    sessions = _selected_sessions(ctx)
    segments_with_sidecars = _segments_with_sidecars(bundle, sessions)
    segment_rows = segments_with_sidecars.loc[segments_with_sidecars["segment_id"].astype(str) == segment_id]
    if segment_rows.empty:
        raise KeyError(f"segment_id {segment_id} is not part of the selected launch")
    blocked, summary = scan_blocked_trace_points_for_segment(
        segment=segment_rows.iloc[0],
        timezone_name=bundle.timezone_name,
        blocker_rule=bundle.gps_blocker_rule,
    )
    write_parquet(blocked, parquet_path)
    write_json(summary_path, summary)
    return {
        "stage_id": "blocker_scan",
        "segment_id": segment_id,
        "blocked_trace_points": int(len(blocked.index)),
    }


def _run_blocker_merge_stage(ctx: LaunchContext) -> dict[str, Any]:
    bundle = load_base_bundle(_base_snapshot_manifest_path(ctx))
    cache_dir = stage_cache_dir(ctx.paths, "blocker_scan")
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for segment_id in ctx.stage_task_values("blocker_scan"):
        token = slugify_token(segment_id)
        frames.append(pd.read_parquet(cache_dir / f"{token}.parquet"))
        summaries.append(json.loads((cache_dir / f"{token}.json").read_text(encoding="utf8")))
    blocked_trace_points = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not blocked_trace_points.empty:
        blocked_trace_points = blocked_trace_points.sort_values(
            ["timestamp_local", "recorder_id", "point_index_in_segment"]
        ).reset_index(drop=True)
    else:
        blocked_trace_points = pd.DataFrame(columns=bundle.trace_points_filtered.columns)
    blocker_polygon = buffered_bbox_polygon(
        [(float(lon), float(lat)) for lon, lat in bundle.gps_blocker_rule["points"]],
        float(bundle.gps_blocker_rule["buffer_meters"]),
    )
    blocker_audit = blocker_audit_from_segment_summaries(
        blocked_trace_points,
        summaries,
        bundle.gps_blocker_rule,
        blocker_polygon,
    )
    write_parquet(blocked_trace_points, ctx.paths.tables_dir / "blocked_trace_points.parquet")
    (ctx.paths.tables_dir / "gps_blocker_audit.json").write_text(json.dumps(blocker_audit, indent=2) + "\n", encoding="utf8")
    return {
        "stage_id": "blocker_merge",
        "blocked_trace_points": int(len(blocked_trace_points.index)),
    }


def _run_raw_trace_grid_task(ctx: LaunchContext, day_session_id: str) -> dict[str, Any]:
    cache_dir = stage_cache_dir(ctx.paths, "raw_trace_grids")
    token = slugify_token(day_session_id)
    grid_path = cache_dir / f"{token}.npz"
    summary_path = cache_dir / f"{token}.json"
    if grid_path.exists() and summary_path.exists():
        return {
            "stage_id": "raw_trace_grids",
            "day_session_id": day_session_id,
            "grid_path": str(grid_path),
            "status": "cached",
        }

    bundle = load_base_bundle(_base_snapshot_manifest_path(ctx))
    blocked_trace_points = pd.read_parquet(ctx.paths.tables_dir / "blocked_trace_points.parquet")
    filtered_points = bundle.trace_points_filtered.loc[
        bundle.trace_points_filtered["day_session_id"].astype(str) == day_session_id
    ].copy()
    blocked_points = blocked_trace_points.loc[
        blocked_trace_points["day_session_id"].astype(str) == day_session_id
    ].copy()
    raw_trace_points = pd.concat([filtered_points, blocked_points], ignore_index=True).sort_values(
        ["day_session_id", "recorder_id", "timestamp_local", "point_index_in_segment"]
    )
    origin = _projection_origin(ctx)
    grids, sync_summary = build_session_trace_grids(
        raw_trace_points,
        origin_lat=float(origin["origin_latitude"]),
        origin_lon=float(origin["origin_longitude"]),
    )
    grid = grids[day_session_id]
    write_session_trace_grid(grid, grid_path)
    write_json(summary_path, sync_summary)
    return {
        "stage_id": "raw_trace_grids",
        "day_session_id": day_session_id,
        "grid_path": str(grid_path),
    }


def _run_window_validity_stage(ctx: LaunchContext) -> dict[str, Any]:
    analysis_units = pd.read_parquet(ctx.paths.tables_dir / "analysis_units.parquet")
    filtered_window_inventory = pd.read_parquet(
        stage_cache_dir(ctx.paths, "filtered_window_inventory") / "filtered_window_inventory.parquet"
    )
    raw_grids = _load_grid_cache(ctx, "raw_trace_grids")
    raw_window_inventory = summarize_event_windows(analysis_units, raw_grids, prefix="raw")
    window_inventory = filtered_window_inventory.merge(
        raw_window_inventory,
        on=[
            "analysis_window_id",
            "event_id",
            "focal_id",
            "day_session_id",
            "date_local",
            "window_s",
            "event_midpoint_local",
            "event_midpoint_seconds_local",
        ],
        how="left",
        validate="one_to_one",
    )
    window_inventory = decorate_window_validity(window_inventory)
    write_parquet(window_inventory, ctx.paths.tables_dir / "window_inventory.parquet")
    return {
        "stage_id": "window_validity",
        "analysis_windows": int(len(window_inventory.index)),
    }


def _run_centroid_distance_task(ctx: LaunchContext, day_session_id: str) -> dict[str, Any]:
    cache_dir = stage_cache_dir(ctx.paths, "centroid_distance")
    token = slugify_token(day_session_id)
    parquet_path = cache_dir / f"{token}.parquet"
    if parquet_path.exists():
        return {
            "stage_id": "centroid_distance",
            "day_session_id": day_session_id,
            "parquet_path": str(parquet_path),
            "status": "cached",
        }
    window_inventory = pd.read_parquet(ctx.paths.tables_dir / "window_inventory.parquet")
    session_windows = window_inventory.loc[window_inventory["day_session_id"].astype(str) == day_session_id].copy()
    grid = _load_grid_cache(ctx, "filtered_trace_grids")[day_session_id]
    observed_centroid_distance = compute_observed_centroid_distance(
        session_windows,
        {day_session_id: grid},
    )
    write_parquet(observed_centroid_distance, parquet_path)
    return {
        "stage_id": "centroid_distance",
        "day_session_id": day_session_id,
        "rows": int(len(observed_centroid_distance.index)),
    }


def _run_baseline_covariates_task(ctx: LaunchContext, day_session_id: str) -> dict[str, Any]:
    cache_dir = stage_cache_dir(ctx.paths, "baseline_covariates")
    token = slugify_token(day_session_id)
    parquet_path = cache_dir / f"{token}.parquet"
    if parquet_path.exists():
        return {
            "stage_id": "baseline_covariates",
            "day_session_id": day_session_id,
            "parquet_path": str(parquet_path),
            "status": "cached",
        }
    observed_centroid_distance = pd.read_parquet(
        stage_cache_dir(ctx.paths, "centroid_distance") / f"{slugify_token(day_session_id)}.parquet"
    )
    window_inventory = pd.read_parquet(ctx.paths.tables_dir / "window_inventory.parquet")
    grid = _load_grid_cache(ctx, "filtered_trace_grids")[day_session_id]
    covariates = compute_baseline_covariates_for_session(
        observed_centroid_distance=observed_centroid_distance,
        window_inventory=window_inventory.loc[window_inventory["day_session_id"].astype(str) == day_session_id].copy(),
        grid=grid,
    )
    write_parquet(covariates, parquet_path)
    return {
        "stage_id": "baseline_covariates",
        "day_session_id": day_session_id,
        "rows": int(len(covariates.index)),
    }


def _run_covariate_table_stage(ctx: LaunchContext) -> dict[str, Any]:
    frames = [
        pd.read_parquet(stage_cache_dir(ctx.paths, "baseline_covariates") / f"{slugify_token(day_session_id)}.parquet")
        for day_session_id in ctx.stage_task_values("baseline_covariates")
    ]
    baseline_covariates = (
        pd.concat(frames, ignore_index=True).sort_values(["date_local", "event_id", "window_s"]).reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )
    write_parquet(baseline_covariates, ctx.paths.tables_dir / "baseline_covariates.parquet")
    covariate_summary(baseline_covariates).to_csv(ctx.paths.tables_dir / "baseline_covariates_summary.csv", index=False)
    return {
        "stage_id": "covariate_table",
        "baseline_covariate_rows": int(len(baseline_covariates.index)),
    }


def _run_matched_controls_task(ctx: LaunchContext, day_session_id: str) -> dict[str, Any]:
    cache_dir = stage_cache_dir(ctx.paths, "matched_controls")
    token = slugify_token(day_session_id)
    parquet_path = cache_dir / f"{token}.parquet"
    summary_path = cache_dir / f"{token}.json"
    if parquet_path.exists() and summary_path.exists():
        return {
            "stage_id": "matched_controls",
            "day_session_id": day_session_id,
            "parquet_path": str(parquet_path),
            "status": "cached",
        }
    baseline_covariates = pd.read_parquet(ctx.paths.tables_dir / "baseline_covariates.parquet")
    analysis_units = pd.read_parquet(ctx.paths.tables_dir / "analysis_units.parquet")
    session_covariates = baseline_covariates.loc[baseline_covariates["day_session_id"].astype(str) == day_session_id]
    all_hum_times = pd.to_numeric(
        analysis_units.loc[
            analysis_units["day_session_id"].astype(str).eq(day_session_id),
            "event_midpoint_seconds_local",
        ],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    date_local = str(session_covariates["date_local"].iloc[0]) if not session_covariates.empty else day_session_id.split(":", maxsplit=1)[0]
    grid = _load_grid_cache(ctx, "filtered_trace_grids")[day_session_id]
    matched_controls, summary = match_controls_for_session(
        hum_covariates=baseline_covariates,
        grid=grid,
        date_local=date_local,
        global_scales=global_match_scales(baseline_covariates),
        all_hum_event_times_seconds=all_hum_times,
    )
    write_parquet(matched_controls, parquet_path)
    write_json(summary_path, summary)
    return {
        "stage_id": "matched_controls",
        "day_session_id": day_session_id,
        "matched_controls": int(len(matched_controls.index)),
        "summary_path": str(summary_path),
    }


def _run_model_input_stage(ctx: LaunchContext) -> dict[str, Any]:
    baseline_covariates = pd.read_parquet(ctx.paths.tables_dir / "baseline_covariates.parquet")
    frames = [
        pd.read_parquet(stage_cache_dir(ctx.paths, "matched_controls") / f"{slugify_token(day_session_id)}.parquet")
        for day_session_id in ctx.stage_task_values("matched_controls")
    ]
    matched_controls = (
        pd.concat(frames, ignore_index=True).sort_values(["date_local", "hum_event_id", "window_s", "match_rank"]).reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )
    write_parquet(matched_controls, ctx.paths.tables_dir / "matched_control_windows.parquet")
    match_summary = match_quality_summary(matched_controls, baseline_covariates)
    rule_failure_totals: dict[str, int] = {}
    for day_session_id in ctx.stage_task_values("matched_controls"):
        summary_path = stage_cache_dir(ctx.paths, "matched_controls") / f"{slugify_token(day_session_id)}.json"
        session_summary = json.loads(summary_path.read_text(encoding="utf8"))
        for rule_name, count in dict(session_summary.get("rule_failure_counts", {})).items():
            rule_failure_totals[rule_name] = rule_failure_totals.get(rule_name, 0) + int(count)
    for rule_name, count in sorted(rule_failure_totals.items()):
        match_summary[f"rule_failures_{rule_name}"] = count
    match_summary.to_csv(ctx.paths.tables_dir / "matched_control_windows_summary.csv", index=False)
    match_summary.to_csv(ctx.paths.tables_dir / "matching_feasibility.csv", index=False)
    model_input, model_summary = build_model_input(
        hum_covariates=baseline_covariates,
        matched_controls=matched_controls,
        hard_gps_filter_version=str(
            json.loads((ctx.paths.tables_dir / "gps_blocker_audit.json").read_text(encoding="utf8"))["rule_name"]
        ),
    )
    write_parquet(model_input, ctx.paths.tables_dir / "model_input.parquet")
    model_summary.to_csv(ctx.paths.tables_dir / "model_input_summary.csv", index=False)
    directional_window, directional_summary, directional_outcome_summary = directional_movement_tables(model_input)
    directional_window.to_csv(
        ctx.paths.tables_dir / "rq1_directional_movement_window_level.csv", index=False
    )
    directional_summary.to_csv(
        ctx.paths.tables_dir / "rq1_directional_movement_summary.csv", index=False
    )
    directional_outcome_summary.to_csv(
        ctx.paths.tables_dir / "rq1_directional_movement_outcome_summary.csv", index=False
    )
    return {
        "stage_id": "model_input",
        "matched_controls": int(len(matched_controls.index)),
        "model_input_rows": int(len(model_input.index)),
    }


def _run_reports_stage(ctx: LaunchContext) -> dict[str, Any]:
    analysis_units = pd.read_parquet(ctx.paths.tables_dir / "analysis_units.parquet")
    window_inventory = pd.read_parquet(ctx.paths.tables_dir / "window_inventory.parquet")
    blocked_trace_points = pd.read_parquet(ctx.paths.tables_dir / "blocked_trace_points.parquet")
    centroid_frames = [
        pd.read_parquet(stage_cache_dir(ctx.paths, "centroid_distance") / f"{slugify_token(day_session_id)}.parquet")
        for day_session_id in ctx.stage_task_values("centroid_distance")
    ]
    observed_centroid_distance = (
        pd.concat(centroid_frames, ignore_index=True).sort_values(["date_local", "event_id", "window_s"]).reset_index(drop=True)
        if centroid_frames
        else pd.DataFrame()
    )
    centroid_summary_df = centroid_summary(observed_centroid_distance)
    write_parquet(observed_centroid_distance, ctx.paths.tables_dir / "observed_centroid_distance.parquet")
    centroid_summary_df.to_csv(ctx.paths.tables_dir / "observed_centroid_distance_summary.csv", index=False)
    baseline_covariates = pd.read_parquet(ctx.paths.tables_dir / "baseline_covariates.parquet")
    matched_controls = pd.read_parquet(ctx.paths.tables_dir / "matched_control_windows.parquet")
    model_input = pd.read_parquet(ctx.paths.tables_dir / "model_input.parquet")
    blocker_audit = json.loads((ctx.paths.tables_dir / "gps_blocker_audit.json").read_text(encoding="utf8"))
    _write_filtering_tables(
        tables_dir=ctx.paths.tables_dir,
        window_inventory=window_inventory,
        model_input=model_input,
        blocker_audit=blocker_audit,
    )
    _write_freshness_and_reuse_table(ctx)
    write_runtime_reports(
        report_dir=ctx.paths.reports_dir,
        tables_dir=ctx.paths.tables_dir,
        context={
            "analysis_units": len(analysis_units),
            "blocked_trace_points": len(blocked_trace_points),
            "baseline_covariate_rows": len(baseline_covariates),
            "matched_controls": len(matched_controls),
            "model_input_rows": len(model_input),
        },
    )
    manifest = build_runtime_manifest(
        launch_label=str(ctx.launch_manifest["launch_label"]),
        launch_dir=ctx.paths.launch_dir,
        input_manifest_path=_base_snapshot_manifest_path(ctx),
        specs_manifest=dict(ctx.launch_manifest["specs_manifest"]),
        source_code_manifest=dict(ctx.launch_manifest["source_code_manifest"]),
        tables_dir=ctx.paths.tables_dir,
        reports_dir=ctx.paths.reports_dir,
    )
    (ctx.paths.launch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    return {
        "stage_id": "reports",
        "analysis_units": int(len(analysis_units.index)),
        "analysis_windows": int(len(window_inventory.index)),
        "observed_centroid_distance_rows": int(len(observed_centroid_distance.index)),
        "baseline_covariate_rows": int(len(baseline_covariates.index)),
        "matched_controls": int(len(matched_controls.index)),
        "model_input_rows": int(len(model_input.index)),
        "manifest_path": str(ctx.paths.launch_dir / "manifest.json"),
    }


def _write_filtering_tables(
    *,
    tables_dir: Path,
    window_inventory: pd.DataFrame,
    model_input: pd.DataFrame,
    blocker_audit: dict[str, Any],
) -> None:
    attrition_rows: list[dict[str, Any]] = []
    for window_s, group in window_inventory.groupby("window_s", sort=True):
        current = model_input.loc[pd.to_numeric(model_input["window_s"], errors="coerce").eq(window_s)]
        attrition_rows.extend(
            [
                {"window_s": int(window_s), "step": "all_windows", "rows": int(len(group.index))},
                {
                    "window_s": int(window_s),
                    "step": "gps_valid_min_3_nonfocal",
                    "rows": int(group["filtered_gps_valid_min_nonfocal_3"].sum()),
                },
                {
                    "window_s": int(window_s),
                    "step": "matched_hum_model_rows",
                    "rows": int(pd.to_numeric(current["condition"], errors="coerce").eq(1).sum()),
                },
                {
                    "window_s": int(window_s),
                    "step": "matched_control_model_rows",
                    "rows": int(pd.to_numeric(current["condition"], errors="coerce").eq(0).sum()),
                },
            ]
        )
    pd.DataFrame(attrition_rows).to_csv(tables_dir / "filtering_attrition.csv", index=False)
    pd.DataFrame(
        [
            {
                "blocked_trace_points": blocker_audit.get("blocked_trace_points", 0),
                "total_raw_trace_points": blocker_audit.get("total_raw_trace_points", 0),
                "blocked_trace_share": blocker_audit.get("blocked_trace_share", pd.NA),
                "affected_recorders": blocker_audit.get("affected_recorders", 0),
                "affected_day_sessions": ",".join(blocker_audit.get("affected_day_sessions", [])),
                "derived_bbox_bounds_json": json.dumps(blocker_audit.get("derived_bbox_bounds", {}), sort_keys=True),
            }
        ]
    ).to_csv(tables_dir / "hard_gps_filter_summary.csv", index=False)


def _write_freshness_and_reuse_table(ctx: LaunchContext) -> None:
    reuse_path = ctx.paths.configs_dir / "cache_reuse_manifest.json"
    rows: list[dict[str, Any]] = []
    if reuse_path.is_file():
        reuse = json.loads(reuse_path.read_text(encoding="utf8"))
        rows.extend(
            [
                {
                    "artifact_scope": "source launch: units through raw_trace_grids",
                    "source_launch": reuse.get("source_launch", ""),
                    "classification": "current-valid",
                    "reuse_decision": "reused",
                    "reason": reuse.get("semantic_basis", ""),
                },
                {
                    "artifact_scope": "source launch: window_validity through model_input",
                    "source_launch": reuse.get("source_launch", ""),
                    "classification": "not-used",
                    "reuse_decision": "not_reused",
                    "reason": "The retained workflow rebuilds matched-control model input and delegates fitting to scientific_lmm_guide_workflow.R.",
                },
            ]
        )
    rows.append(
        {
            "artifact_scope": "current launch: window_validity through model_input",
            "source_launch": str(ctx.launch_manifest["launch_label"]),
            "classification": "current-valid",
            "reuse_decision": "generated",
            "reason": "Generated under the retained RQ1 model-input contract; no model-grid fitting is run in this launch.",
        }
    )
    pd.DataFrame(rows).to_csv(
        ctx.paths.tables_dir / "rq1_artifact_freshness_and_reuse.csv", index=False
    )


def _load_grid_cache(ctx: LaunchContext, stage_id: str) -> dict[str, Any]:
    cache_dir = stage_cache_dir(ctx.paths, stage_id)
    grids: dict[str, Any] = {}
    for day_session_id in ctx.stage_task_values(stage_id):
        path = cache_dir / f"{slugify_token(day_session_id)}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing cached grid for stage {stage_id} and session {day_session_id}: {path}")
        grids[day_session_id] = read_session_trace_grid(path)
    return grids


def _segments_with_sidecars(bundle: Any, selected_sessions: list[str]) -> pd.DataFrame:
    segments = _select_sessions(bundle.segments, selected_sessions)
    sidecar_index = _select_sessions(bundle.sidecar_index, selected_sessions)
    return segments.merge(
        sidecar_index[["segment_id", "cached_csv_path"]],
        on="segment_id",
        how="left",
        validate="one_to_one",
    )


def _selected_sessions(ctx: LaunchContext) -> list[str]:
    return [str(value) for value in ctx.launch_manifest["analysis"]["selected_day_session_ids"]]


def _projection_origin(ctx: LaunchContext) -> dict[str, float]:
    raw_origin = dict(ctx.launch_manifest["analysis"]["projection_origin"])
    return {
        "origin_latitude": float(raw_origin["origin_latitude"]),
        "origin_longitude": float(raw_origin["origin_longitude"]),
    }


def _select_sessions(df: pd.DataFrame, selected_sessions: list[str]) -> pd.DataFrame:
    return df.loc[df["day_session_id"].astype(str).isin(selected_sessions)].copy()

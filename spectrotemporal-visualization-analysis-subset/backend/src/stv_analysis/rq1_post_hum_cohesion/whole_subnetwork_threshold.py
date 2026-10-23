from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contracts import SUMMARY_METHOD, SessionTraceGrid, WINDOW_LENGTHS_SECONDS
from .subaim2_windows import (
    build_session_trace_grids,
    projection_origin,
    read_session_trace_grid,
    write_session_trace_grid,
)
from .subaim5_covariates import IMPOSSIBLE_SPEED_M_PER_S, _convex_hull_area, _round_or_nan, _safe_nanmedian
from .subaim4_centroid import directional_movement_tables, distance_contribution_components


GLOBAL_OUTLIER_THRESHOLD_M = 49.358079
GLOBAL_OUTLIER_THRESHOLD_ROLE = "global_outlier_filtered"
MIN_RETAINED_COMPANIONS = 2
THRESHOLD_SELECTION_RULE = (
    "KDE trough on pooled frozen W60 row-candidate leave-one-candidate-out median distances "
    "when two prominent modes are separated by at least 10 m and trough density is <=70% of "
    "the lower peak; otherwise Q3 + 1.5*IQR. The frozen fallback value is 49.358079 m."
)

RETAINED_MODEL_PREDICTORS = [
    "condition_hum",
    "starting_centroid_distance_m",
    "mean_speed_pre_m_s",
    "pre_delta_centroid_distance_m",
    "observed_polygon_area_pre_m2",
    "recent_change_in_polygon_area_m2",
]

EXCLUDED_PREDICTOR_COLUMNS = {
    "time_of_day_minutes",
    "local_collared_density_per_1000m2",
    "n_collars_pre",
    "collar_availability",
    "minutes_since_session_start",
    "time_of_day",
    "local_density",
}

MODEL_INPUT_COLUMNS = [
    "analysis_window_id",
    "primary_analysis_window_id",
    "condition",
    "condition_label",
    "match_set_id",
    "event_id",
    "hum_event_id",
    "focal_id",
    "focal_alpaca_id",
    "day_session_id",
    "session_id",
    "date_local",
    "window_s",
    "window_length_s",
    "window_code",
    "window_center_seconds_local",
    "window_center_timestamp",
    "Delta_D_w",
    "Delta_D_cm",
    "change_in_centroid_distance_w",
    "change_in_centroid_distance_loo_m",
    "pre_distance_loo_centroid_m",
    "post_distance_loo_centroid_m",
    "starting_centroid_distance_m",
    "pre_delta_centroid_distance_m",
    "mean_speed_pre_m_s",
    "observed_polygon_area_pre_m2",
    "recent_change_in_polygon_area_m2",
    "starting_distance",
    "pre_window_speed",
    "recent_pre_event_distance_change",
    "observed_polygon_area_pre",
    "recent_change_in_polygon_area",
    "directional_summary_method",
    "focal_distance_reduction_m",
    "centroid_distance_reduction_m",
    "position_distance_interaction_m",
    "distance_change_outcome",
    "directional_category",
    "directional_category_clean",
    "focal_contribution_weight",
    "centroid_contribution_weight",
    "directional_assignment_low_support",
    "directional_geometry_valid",
    "directional_geometry_invalid_reason",
    "distance_sample_count_pre",
    "distance_sample_count_post",
    "matching_rule_used",
    "buffer_rule_used",
    "buffer_seconds",
    "match_rank",
    "control_match_use_count",
    "standardized_match_distance",
    "hard_gps_filter_version",
    "source_confidence",
    "matching_policy",
    "response_unit",
    "response_transform",
    "canonical_event_policy",
    "centroid_rule",
    "threshold_m",
    "threshold_role",
    "threshold_selection_rule",
    "retained_companion_count_pre",
    "retained_companion_ids",
    "retained_companion_scores_m",
    "candidate_companion_count_pre",
    "finite_candidate_score_count_pre",
    "available_retained_companions_pre_median",
    "available_retained_companions_post_median",
    "available_retained_companions_previous_median",
    "filtered_centroid_sample_count_previous",
    "filtered_polygon_sample_count_pre",
    "filtered_polygon_sample_count_previous",
    "threshold_density_pre_per_1000m2",
    "threshold_centroid_displacement_m",
    "threshold_model_row",
    "threshold_exclusion_reason",
]


def threshold_label(threshold_m: float = GLOBAL_OUTLIER_THRESHOLD_M) -> str:
    return f"{float(threshold_m):.6f}".rstrip("0").rstrip(".").replace(".", "p") + "m"


def threshold_token(threshold_m: float = GLOBAL_OUTLIER_THRESHOLD_M) -> str:
    return f"T{threshold_label(threshold_m)}"


def build_filtered_grids_from_base(
    *,
    trace_points: pd.DataFrame,
    cache_dir: Path,
    selected_sessions: Iterable[str] | None = None,
) -> tuple[dict[str, SessionTraceGrid], dict[str, float]]:
    sessions = (
        sorted(str(value) for value in selected_sessions)
        if selected_sessions is not None
        else sorted(trace_points["day_session_id"].dropna().astype(str).unique().tolist())
    )
    selected = trace_points.loc[trace_points["day_session_id"].astype(str).isin(sessions)].copy()
    if selected.empty:
        raise ValueError("no trace points are available for the selected sessions")
    origin = projection_origin(selected)

    cache_dir.mkdir(parents=True, exist_ok=True)
    grids: dict[str, SessionTraceGrid] = {}
    for day_session_id, session_points in selected.groupby("day_session_id", sort=True):
        session_id = str(day_session_id)
        path = cache_dir / f"{_slug(session_id)}.npz"
        if path.exists():
            grids[session_id] = read_session_trace_grid(path)
            continue
        built, _summary = build_session_trace_grids(
            session_points.copy(),
            origin_lat=float(origin["origin_latitude"]),
            origin_lon=float(origin["origin_longitude"]),
        )
        grid = built[session_id]
        write_session_trace_grid(grid, path)
        grids[session_id] = grid
    return grids, origin


def materialize_whole_subnetwork_threshold(
    *,
    repo_root: Path,
    primary_rows: pd.DataFrame,
    trace_points: pd.DataFrame,
    output_dir: Path,
    threshold_m: float = GLOBAL_OUTLIER_THRESHOLD_M,
    selected_sessions: Iterable[str] | None = None,
    hard_gps_filter_version: str = "",
    calibration_source_dir: Path | None = None,
) -> dict[str, Any]:
    if not np.isclose(float(threshold_m), GLOBAL_OUTLIER_THRESHOLD_M):
        raise ValueError("the RQ1 whole-subnetwork threshold supplement accepts exactly one frozen threshold")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    cache_dir = output_dir / "cache"
    calibration_dir = output_dir / "threshold_calibration"
    for directory in (data_dir, tables_dir, reports_dir, cache_dir, calibration_dir):
        directory.mkdir(parents=True, exist_ok=True)

    primary = normalize_primary_rows(primary_rows)
    if selected_sessions is not None:
        session_set = set(str(value) for value in selected_sessions)
        primary = primary.loc[primary["day_session_id"].astype(str).isin(session_set)].copy()
        trace_points = trace_points.loc[trace_points["day_session_id"].astype(str).isin(session_set)].copy()
    if primary.empty:
        raise ValueError("no primary rows are available for threshold materialization")

    grids, origin = build_filtered_grids_from_base(
        trace_points=trace_points,
        cache_dir=cache_dir / "filtered_trace_grids",
        selected_sessions=selected_sessions,
    )

    rows = [
        threshold_features_for_primary_row(
            row=row,
            grid=grids.get(str(row.day_session_id)),
            threshold_m=float(threshold_m),
            hard_gps_filter_version=hard_gps_filter_version,
        )
        for row in primary.itertuples(index=False)
    ]
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError("threshold materialization produced no row diagnostics")

    _assert_threshold_invariants(comparison, threshold_m=float(threshold_m))
    model_input = comparison.loc[comparison["threshold_model_row"].astype(bool)].copy()
    model_input = model_input.reindex(columns=MODEL_INPUT_COLUMNS)
    model_input = model_input.sort_values(["window_s", "match_set_id", "condition", "match_rank", "analysis_window_id"]).reset_index(drop=True)
    _assert_no_excluded_predictors(model_input)

    write_model_input(model_input, data_dir / "h1_threshold_filtered_model_input")
    comparison.to_csv(tables_dir / "rq1_threshold_row_comparison.csv.gz", index=False)

    summary = threshold_model_input_summary(comparison)
    summary.to_csv(tables_dir / "rq1_threshold_model_input_summary.csv", index=False)
    retained_distribution(comparison).to_csv(tables_dir / "rq1_threshold_retained_companion_distribution.csv", index=False)
    geometry_exclusions(comparison).to_csv(tables_dir / "rq1_threshold_geometry_exclusions.csv", index=False)
    matched_set_completeness(comparison).to_csv(tables_dir / "rq1_threshold_matched_set_completeness.csv", index=False)
    pre_event_balance(model_input).to_csv(tables_dir / "rq1_threshold_pre_event_balance.csv", index=False)
    raw_change_summary(model_input).to_csv(tables_dir / "rq1_threshold_raw_change_summary.csv", index=False)
    directional_window, directional_summary, directional_outcome_summary = directional_movement_tables(model_input)
    directional_window.to_csv(
        tables_dir / "rq1_threshold_directional_movement_window_level.csv", index=False
    )
    directional_summary.to_csv(
        tables_dir / "rq1_threshold_directional_movement_summary.csv", index=False
    )
    directional_outcome_summary.to_csv(
        tables_dir / "rq1_threshold_directional_movement_outcome_summary.csv", index=False
    )
    run_config = run_config_rows(relative_to_repo(output_dir, repo_root), threshold_m=float(threshold_m))
    run_config.to_csv(tables_dir / "rq1_threshold_scientific_lmm_run_config.csv", index=False)

    copy_calibration_artifacts(calibration_source_dir, calibration_dir)
    manifest = {
        "output_dir": str(output_dir),
        "threshold_m": float(threshold_m),
        "threshold_role": GLOBAL_OUTLIER_THRESHOLD_ROLE,
        "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
        "projection_origin": origin,
        "primary_rows": int(len(primary.index)),
        "model_rows": int(len(model_input.index)),
        "model_input_csv": str(data_dir / "h1_threshold_filtered_model_input.csv"),
        "model_input_parquet": str(data_dir / "h1_threshold_filtered_model_input.parquet"),
        "row_comparison_csv_gz": str(tables_dir / "rq1_threshold_row_comparison.csv.gz"),
        "run_config_csv": str(tables_dir / "rq1_threshold_scientific_lmm_run_config.csv"),
        "retained_fixed_effects": RETAINED_MODEL_PREDICTORS,
        "excluded_predictor_columns": sorted(EXCLUDED_PREDICTOR_COLUMNS),
        "model_policy": {
            "matching_policy": "standard_matching",
            "control_regeneration": "forbidden",
            "control_replacement": "forbidden",
            "match_set_random_effect": "not fitted",
            "random_effect_base": ["focal_id", "session_id"],
            "optimizer": "nloptwrap",
        },
    }
    (output_dir / "threshold_materialization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    write_report(
        reports_dir / "rq1_threshold_materialized_views.md",
        output_dir=output_dir,
        summary=summary,
        run_config=run_config,
    )
    return manifest


def normalize_primary_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "analysis_window_id",
        "condition",
        "match_set_id",
        "focal_id",
        "day_session_id",
        "session_id",
        "date_local",
        "window_s",
        "window_center_seconds_local",
        "mean_speed_pre_m_s",
        "matching_rule_used",
        "buffer_rule_used",
        "buffer_seconds",
        "match_rank",
        "control_match_use_count",
        "standardized_match_distance",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"primary model input is missing required columns: {missing}")
    out = df.copy()
    out["condition"] = pd.to_numeric(out["condition"], errors="raise").astype("int8")
    out["condition_label"] = out["condition"].map({0: "control", 1: "hum"})
    out["window_s"] = pd.to_numeric(out["window_s"], errors="raise").astype(int)
    out["window_length_s"] = out["window_s"]
    out["window_code"] = "W" + out["window_s"].astype(str)
    out["day_session_id"] = out["day_session_id"].astype(str)
    out["session_id"] = out["session_id"].fillna(out["day_session_id"]).astype(str)
    out["focal_id"] = out["focal_id"].astype(str)
    out["focal_alpaca_id"] = out.get("focal_alpaca_id", out["focal_id"]).fillna(out["focal_id"]).astype(str)
    out["match_set_id"] = out["match_set_id"].astype(str)
    out["analysis_window_id"] = out["analysis_window_id"].astype(str)
    for column in [
        "window_center_seconds_local",
        "mean_speed_pre_m_s",
        "buffer_seconds",
        "match_rank",
        "control_match_use_count",
        "standardized_match_distance",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["window_s", "match_set_id", "condition", "match_rank", "analysis_window_id"]).reset_index(drop=True)


def threshold_features_for_primary_row(
    *,
    row: Any,
    grid: SessionTraceGrid | None,
    threshold_m: float,
    hard_gps_filter_version: str,
) -> dict[str, Any]:
    base = _primary_identity(row, threshold_m=threshold_m, hard_gps_filter_version=hard_gps_filter_version)
    if grid is None:
        return _excluded_row(base, "missing_session_grid")
    focal_id = str(row.focal_id)
    if focal_id not in grid.recorder_index:
        return _excluded_row(base, "missing_focal_in_grid")

    focal_index = grid.recorder_index[focal_id]
    window_s = int(row.window_s)
    t0 = float(row.window_center_seconds_local)
    pre_indices = _window_indices(grid, t0, window_s, side="pre")
    post_indices = _window_indices(grid, t0, window_s, side="post")
    previous_indices = _previous_window_indices(grid, t0, window_s)

    scores = companion_outlier_scores(grid, focal_index, pre_indices)
    retained = [slot for slot, score in sorted(scores.items(), key=lambda item: (item[1], grid.recorder_ids[item[0]])) if score <= threshold_m]
    if focal_index in retained:
        raise AssertionError("focal alpaca entered retained threshold companion set")

    retained_ids = [grid.recorder_ids[slot] for slot in retained]
    base.update(
        {
            "retained_companion_count_pre": int(len(retained)),
            "retained_companion_ids": "|".join(retained_ids),
            "retained_companion_scores_m": ";".join(f"{grid.recorder_ids[slot]}:{scores[slot]:.6f}" for slot in retained),
            "candidate_companion_count_pre": int(len([slot for slot in range(len(grid.recorder_ids)) if slot != focal_index])),
            "finite_candidate_score_count_pre": int(len(scores)),
        }
    )
    if len(retained) < MIN_RETAINED_COMPANIONS:
        return _excluded_row(base, "fewer_than_2_retained_companions_pre")

    pre_distances, pre_centroids = filtered_centroid_distances(grid, focal_index, retained, pre_indices)
    post_distances, post_centroids = filtered_centroid_distances(grid, focal_index, retained, post_indices)
    previous_distances, _previous_centroids = filtered_centroid_distances(grid, focal_index, retained, previous_indices)
    pre_distance = _safe_nanmedian(pre_distances)
    post_distance = _safe_nanmedian(post_distances)
    previous_distance = _safe_nanmedian(previous_distances)
    pre_delta = np.nan if np.isnan(pre_distance) or np.isnan(previous_distance) else pre_distance - previous_distance

    polygon_pre = filtered_polygon_areas(grid, focal_index, retained, pre_indices)
    polygon_previous = filtered_polygon_areas(grid, focal_index, retained, previous_indices)
    polygon_area_pre = _safe_nanmedian(polygon_pre)
    polygon_area_previous = _safe_nanmedian(polygon_previous)
    recent_polygon_change = (
        np.nan if np.isnan(polygon_area_pre) or np.isnan(polygon_area_previous) else polygon_area_pre - polygon_area_previous
    )

    speed_values = focal_speed_values_for_indices(grid, focal_index, pre_indices)
    plausible_speeds = speed_values[speed_values <= IMPOSSIBLE_SPEED_M_PER_S]
    primary_speed = _float_or_nan(getattr(row, "mean_speed_pre_m_s", np.nan))
    mean_speed_pre = primary_speed if np.isfinite(primary_speed) else (
        float(np.nanmean(plausible_speeds)) if plausible_speeds.size else np.nan
    )

    pre_available = available_retained_counts(grid, retained, pre_indices)
    post_available = available_retained_counts(grid, retained, post_indices)
    previous_available = available_retained_counts(grid, retained, previous_indices)
    delta_d = np.nan if np.isnan(post_distance) or np.isnan(pre_distance) else post_distance - pre_distance
    direction = filtered_directional_movement_components(grid, focal_index, retained, pre_indices, post_indices, delta_d_m=delta_d)
    centroid_displacement = median_centroid_displacement(pre_centroids, post_centroids)
    threshold_density = np.nan
    total_count_pre = np.asarray(pre_available, dtype=float) + 1.0
    n_collars_pre = _safe_nanmedian(total_count_pre[np.isfinite(total_count_pre)])
    if np.isfinite(n_collars_pre) and np.isfinite(polygon_area_pre) and polygon_area_pre > 0:
        threshold_density = n_collars_pre / (polygon_area_pre / 1000.0)

    model_values = {
        "Delta_D_w": _round_or_nan(delta_d),
        "Delta_D_cm": _round_or_nan(delta_d * 100.0 if np.isfinite(delta_d) else np.nan),
        "change_in_centroid_distance_w": _round_or_nan(delta_d),
        "change_in_centroid_distance_loo_m": _round_or_nan(delta_d),
        "pre_distance_loo_centroid_m": _round_or_nan(pre_distance),
        "post_distance_loo_centroid_m": _round_or_nan(post_distance),
        "starting_centroid_distance_m": _round_or_nan(pre_distance),
        "pre_delta_centroid_distance_m": _round_or_nan(pre_delta),
        "mean_speed_pre_m_s": _round_or_nan(mean_speed_pre),
        "observed_polygon_area_pre_m2": _round_or_nan(polygon_area_pre),
        "recent_change_in_polygon_area_m2": _round_or_nan(recent_polygon_change),
        "starting_distance": _round_or_nan(pre_distance),
        "pre_window_speed": _round_or_nan(mean_speed_pre),
        "recent_pre_event_distance_change": _round_or_nan(pre_delta),
        "observed_polygon_area_pre": _round_or_nan(polygon_area_pre),
        "recent_change_in_polygon_area": _round_or_nan(recent_polygon_change),
    }
    base.update(model_values)
    base.update(direction)
    base.update(
        {
            "distance_sample_count_pre": int(pre_distances.size),
            "distance_sample_count_post": int(post_distances.size),
            "available_retained_companions_pre_median": _round_or_nan(_safe_nanmedian(np.asarray(pre_available, dtype=float))),
            "available_retained_companions_post_median": _round_or_nan(_safe_nanmedian(np.asarray(post_available, dtype=float))),
            "available_retained_companions_previous_median": _round_or_nan(_safe_nanmedian(np.asarray(previous_available, dtype=float))),
            "filtered_centroid_sample_count_previous": int(previous_distances.size),
            "filtered_polygon_sample_count_pre": int(polygon_pre.size),
            "filtered_polygon_sample_count_previous": int(polygon_previous.size),
            "threshold_density_pre_per_1000m2": _round_or_nan(threshold_density),
            "threshold_centroid_displacement_m": _round_or_nan(centroid_displacement),
        }
    )

    exclusion = complete_case_exclusion(base)
    base["threshold_exclusion_reason"] = exclusion
    base["threshold_model_row"] = exclusion == ""
    return base


def companion_outlier_scores(grid: SessionTraceGrid, focal_index: int, pre_indices: np.ndarray) -> dict[int, float]:
    if pre_indices.size == 0:
        return {}
    valid = grid.valid[pre_indices] & np.isfinite(grid.x_m[pre_indices]) & np.isfinite(grid.y_m[pre_indices])
    x_fill = np.where(valid, grid.x_m[pre_indices], 0.0)
    y_fill = np.where(valid, grid.y_m[pre_indices], 0.0)
    total_x = x_fill.sum(axis=1)
    total_y = y_fill.sum(axis=1)
    total_count = valid.sum(axis=1).astype(float)
    focal_valid = valid[:, focal_index]
    focal_x = x_fill[:, focal_index]
    focal_y = y_fill[:, focal_index]

    scores: dict[int, float] = {}
    for candidate in range(len(grid.recorder_ids)):
        if candidate == focal_index:
            continue
        candidate_valid = valid[:, candidate]
        other_count = total_count - focal_valid.astype(float) - candidate_valid.astype(float)
        scorable = candidate_valid & (other_count >= 1)
        if not np.any(scorable):
            continue
        centroid_x = (total_x[scorable] - focal_x[scorable] - x_fill[scorable, candidate]) / other_count[scorable]
        centroid_y = (total_y[scorable] - focal_y[scorable] - y_fill[scorable, candidate]) / other_count[scorable]
        distances = np.hypot(grid.x_m[pre_indices[scorable], candidate] - centroid_x, grid.y_m[pre_indices[scorable], candidate] - centroid_y)
        median = _safe_nanmedian(distances)
        if np.isfinite(median):
            scores[candidate] = float(median)
    return scores


def filtered_centroid_distances(
    grid: SessionTraceGrid,
    focal_index: int,
    retained: list[int],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if indices.size == 0 or not retained:
        return np.asarray([], dtype=float), np.empty((0, 2), dtype=float)
    focal_valid = valid_slot_mask_at_indices(grid, focal_index, indices)
    companion_valid = selected_valid_matrix_for_indices(grid, retained, indices)
    counts = companion_valid.sum(axis=1)
    usable = focal_valid & (counts >= 1)
    if not np.any(usable):
        return np.asarray([], dtype=float), np.empty((0, 2), dtype=float)
    selected_x = np.where(companion_valid, grid.x_m[np.ix_(indices, retained)], 0.0)
    selected_y = np.where(companion_valid, grid.y_m[np.ix_(indices, retained)], 0.0)
    centroid_x = selected_x[usable].sum(axis=1) / counts[usable]
    centroid_y = selected_y[usable].sum(axis=1) / counts[usable]
    usable_indices = indices[usable]
    distances = np.hypot(grid.x_m[usable_indices, focal_index] - centroid_x, grid.y_m[usable_indices, focal_index] - centroid_y)
    centroids = np.column_stack([centroid_x, centroid_y])
    return distances[np.isfinite(distances)], centroids[np.isfinite(distances)]


def filtered_polygon_areas(
    grid: SessionTraceGrid,
    focal_index: int,
    retained: list[int],
    indices: np.ndarray,
) -> np.ndarray:
    areas: list[float] = []
    for row_index in indices:
        if not valid_slot_at_row(grid, focal_index, row_index):
            continue
        valid_retained = valid_companion_slots_at_row(grid, retained, row_index)
        if len(valid_retained) < MIN_RETAINED_COMPANIONS:
            continue
        slots = [focal_index, *valid_retained.tolist()]
        points = [(float(grid.x_m[row_index, slot]), float(grid.y_m[row_index, slot])) for slot in slots]
        areas.append(_convex_hull_area(points))
    return np.asarray(areas, dtype=float)


def filtered_directional_movement_components(
    grid: SessionTraceGrid,
    focal_index: int,
    retained: list[int],
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
    delta_d_m: float | None = None,
) -> dict[str, Any]:
    pre_distances, pre_centroids = filtered_centroid_distances(grid, focal_index, retained, pre_indices)
    post_distances, post_centroids = filtered_centroid_distances(grid, focal_index, retained, post_indices)
    if pre_distances.size == 0 or pre_centroids.size == 0:
        return _invalid_directional_result("missing_pre_positions")
    if post_distances.size == 0 or post_centroids.size == 0:
        return _invalid_directional_result("missing_post_positions")
    pre_usable = usable_centroid_indices(grid, focal_index, retained, pre_indices)
    post_usable = usable_centroid_indices(grid, focal_index, retained, post_indices)
    if pre_usable.size == 0 or post_usable.size == 0:
        return _invalid_directional_result("missing_positions")
    focal_pre = np.nanmedian(np.column_stack([grid.x_m[pre_usable, focal_index], grid.y_m[pre_usable, focal_index]]), axis=0)
    focal_post = np.nanmedian(np.column_stack([grid.x_m[post_usable, focal_index], grid.y_m[post_usable, focal_index]]), axis=0)
    centroid_pre = np.nanmedian(pre_centroids, axis=0)
    centroid_post = np.nanmedian(post_centroids, axis=0)
    return distance_contribution_components(
        focal_pre=focal_pre,
        centroid_pre=centroid_pre,
        focal_post=focal_post,
        centroid_post=centroid_post,
        delta_d_m=delta_d_m,
    )


def complete_case_exclusion(row: dict[str, Any]) -> str:
    if int(row.get("retained_companion_count_pre", 0)) < MIN_RETAINED_COMPANIONS:
        return "fewer_than_2_retained_companions_pre"
    if (
        int(row.get("distance_sample_count_pre", 0)) <= 0
        or int(row.get("distance_sample_count_post", 0)) <= 0
        or int(row.get("filtered_centroid_sample_count_previous", 0)) <= 0
    ):
        return "centroid_coverage"
    if (
        int(row.get("filtered_polygon_sample_count_pre", 0)) <= 0
        or int(row.get("filtered_polygon_sample_count_previous", 0)) <= 0
    ):
        return "polygon_coverage"
    for column in [
        "Delta_D_cm",
        "starting_centroid_distance_m",
        "mean_speed_pre_m_s",
        "pre_delta_centroid_distance_m",
        "observed_polygon_area_pre_m2",
        "recent_change_in_polygon_area_m2",
    ]:
        if not np.isfinite(_float_or_nan(row.get(column, np.nan))):
            return f"nonfinite_{column}"
    return ""


def threshold_model_input_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (window_s, condition), group in comparison.groupby(["window_s", "condition_label"], sort=True, dropna=False):
        model_rows = group.loc[group["threshold_model_row"].astype(bool)]
        rows.append(
            {
                "threshold_m": GLOBAL_OUTLIER_THRESHOLD_M,
                "threshold_role": GLOBAL_OUTLIER_THRESHOLD_ROLE,
                "window_s": int(window_s),
                "condition_label": str(condition),
                "primary_rows": int(len(group.index)),
                "model_rows": int(len(model_rows.index)),
                "excluded_rows": int(len(group.index) - len(model_rows.index)),
                "median_retained_companion_count_pre": float(group["retained_companion_count_pre"].median()),
                "min_retained_companion_count_pre": int(group["retained_companion_count_pre"].min()),
                "median_delta_D_cm": float(model_rows["Delta_D_cm"].median()) if not model_rows.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def retained_distribution(comparison: pd.DataFrame) -> pd.DataFrame:
    out = (
        comparison.groupby(["threshold_m", "threshold_role", "window_s", "condition_label", "retained_companion_count_pre"], sort=True, dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    out["retained_companion_count_bin"] = pd.cut(
        pd.to_numeric(out["retained_companion_count_pre"], errors="coerce"),
        bins=[-1, 0, 1, 2, np.inf],
        labels=["0", "1", "2", "at_least_3"],
    ).astype(str)
    return out


def geometry_exclusions(comparison: pd.DataFrame) -> pd.DataFrame:
    reason = comparison["threshold_exclusion_reason"].replace("", "model_row")
    out = (
        comparison.assign(exclusion_reason=reason)
        .groupby(["threshold_m", "threshold_role", "window_s", "condition_label", "exclusion_reason"], sort=True, dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    return out


def matched_set_completeness(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_s, group in comparison.groupby("window_s", sort=True):
        total_sets = group["match_set_id"].nunique()
        model = group.loc[group["threshold_model_row"].astype(bool)]
        hum_sets = set(model.loc[model["condition"].eq(1), "match_set_id"].astype(str))
        control_sets = set(model.loc[model["condition"].eq(0), "match_set_id"].astype(str))
        failed_sets = set(group.loc[~group["threshold_model_row"].astype(bool), "match_set_id"].astype(str))
        rows.append(
            {
                "threshold_m": GLOBAL_OUTLIER_THRESHOLD_M,
                "threshold_role": GLOBAL_OUTLIER_THRESHOLD_ROLE,
                "window_s": int(window_s),
                "primary_match_sets": int(total_sets),
                "model_match_sets": int(model["match_set_id"].nunique()),
                "model_match_sets_with_hum_and_control": int(len(hum_sets & control_sets)),
                "model_hum_sets_without_control": int(len(hum_sets - control_sets)),
                "model_control_sets_without_hum": int(len(control_sets - hum_sets)),
                "primary_match_sets_with_one_or_more_excluded_rows": int(len(failed_sets)),
            }
        )
    return pd.DataFrame(rows)


def pre_event_balance(model_input: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    predictors = [
        "starting_centroid_distance_m",
        "mean_speed_pre_m_s",
        "pre_delta_centroid_distance_m",
        "observed_polygon_area_pre_m2",
        "recent_change_in_polygon_area_m2",
    ]
    for (window_s, condition), group in model_input.groupby(["window_s", "condition_label"], sort=True):
        for predictor in predictors:
            values = pd.to_numeric(group[predictor], errors="coerce").dropna()
            rows.append(
                {
                    "threshold_m": GLOBAL_OUTLIER_THRESHOLD_M,
                    "threshold_role": GLOBAL_OUTLIER_THRESHOLD_ROLE,
                    "window_s": int(window_s),
                    "condition_label": condition,
                    "predictor": predictor,
                    "rows": int(len(values.index)),
                    "median": float(values.median()) if len(values.index) else np.nan,
                    "q25": float(values.quantile(0.25)) if len(values.index) else np.nan,
                    "q75": float(values.quantile(0.75)) if len(values.index) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def raw_change_summary(model_input: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (window_s, condition), group in model_input.groupby(["window_s", "condition_label"], sort=True):
        values = pd.to_numeric(group["Delta_D_cm"], errors="coerce").dropna()
        rows.append(
            {
                "threshold_m": GLOBAL_OUTLIER_THRESHOLD_M,
                "threshold_role": GLOBAL_OUTLIER_THRESHOLD_ROLE,
                "window_s": int(window_s),
                "condition_label": condition,
                "rows": int(len(values.index)),
                "median_delta_D_cm": float(values.median()) if len(values.index) else np.nan,
                "q25_delta_D_cm": float(values.quantile(0.25)) if len(values.index) else np.nan,
                "q75_delta_D_cm": float(values.quantile(0.75)) if len(values.index) else np.nan,
                "mean_delta_D_cm": float(values.mean()) if len(values.index) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_config_rows(output_dir_rel: Path, threshold_m: float = GLOBAL_OUTLIER_THRESHOLD_M) -> pd.DataFrame:
    data_path = output_dir_rel / "data" / "h1_threshold_filtered_model_input.csv"
    rows = []
    for window_s in WINDOW_LENGTHS_SECONDS:
        rows.append(
            {
                "model_id": f"threshold_49m_h1_{int(window_s)}s",
                "threshold_m": float(threshold_m),
                "threshold_role": GLOBAL_OUTLIER_THRESHOLD_ROLE,
                "window_s": int(window_s),
                "matching_policy": "standard_matching",
                "include_match_set": False,
                "data_path": str(data_path),
                "predictor_set": "rq1_predictor_revamp",
                "match_set_random_effect": "standard_model",
                "matching_requirement": "standard_matching",
                "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
            }
        )
    return pd.DataFrame(rows)


def write_model_input(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    df.to_parquet(stem.with_suffix(".parquet"), index=False)


def write_report(path: Path, *, output_dir: Path, summary: pd.DataFrame, run_config: pd.DataFrame) -> None:
    def frame_text(frame: pd.DataFrame, limit: int = 30) -> str:
        if frame.empty:
            return "No rows."
        shown = frame.head(limit)
        suffix = "" if len(frame.index) <= limit else f"\n... {len(frame.index) - limit} additional rows omitted."
        return "```text\n" + shown.to_string(index=False) + suffix + "\n```"

    lines = [
        "# RQ1 Whole-Subnetwork Threshold Materialized Views",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Threshold: `{GLOBAL_OUTLIER_THRESHOLD_M:.6f} m`.",
        "- Threshold role: `global_outlier_filtered`.",
        "- Companion scoring: leave-one-candidate-out distance to other non-focal companions during the pre-event interval.",
        "- Companion membership is selected only from pre-event positions and then frozen across preceding, pre-event, and post-event intervals.",
        "- Primary hum/control row IDs and assignments are preserved; controls are not regenerated, replaced, or rematched.",
        "- Geometry exclusion rule: exclude rows with fewer than two retained companions, failed recalculated centroid coverage, failed recalculated polygon coverage, or another non-finite required model variable.",
        "- Retained model predictors: `condition_hum`, `starting_centroid_distance_m`, `mean_speed_pre_m_s`, `pre_delta_centroid_distance_m`, `observed_polygon_area_pre_m2`, `recent_change_in_polygon_area_m2`.",
        "",
        "## Model Input Summary",
        "",
        frame_text(summary),
        "",
        "## R Run Config",
        "",
        frame_text(run_config),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf8")


def copy_calibration_artifacts(source_dir: Path | None, calibration_dir: Path) -> None:
    if source_dir is None or not source_dir.exists():
        return
    for child in source_dir.iterdir():
        target = calibration_dir / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _primary_identity(row: Any, *, threshold_m: float, hard_gps_filter_version: str) -> dict[str, Any]:
    condition = int(row.condition)
    delta = _float_or_nan(getattr(row, "Delta_D_w", np.nan))
    return {
        "analysis_window_id": str(row.analysis_window_id),
        "primary_analysis_window_id": str(row.analysis_window_id),
        "condition": condition,
        "condition_label": "hum" if condition == 1 else "control",
        "match_set_id": str(row.match_set_id),
        "event_id": str(getattr(row, "event_id", "")) if pd.notna(getattr(row, "event_id", "")) else "",
        "hum_event_id": str(getattr(row, "hum_event_id", "")) if pd.notna(getattr(row, "hum_event_id", "")) else "",
        "focal_id": str(row.focal_id),
        "focal_alpaca_id": str(getattr(row, "focal_alpaca_id", row.focal_id)),
        "day_session_id": str(row.day_session_id),
        "session_id": str(row.session_id),
        "date_local": str(row.date_local),
        "window_s": int(row.window_s),
        "window_length_s": int(row.window_s),
        "window_code": f"W{int(row.window_s)}",
        "window_center_seconds_local": _float_or_nan(row.window_center_seconds_local),
        "window_center_timestamp": str(getattr(row, "window_center_timestamp", "")),
        "Delta_D_w": _round_or_nan(delta),
        "Delta_D_cm": _round_or_nan(delta * 100.0 if np.isfinite(delta) else np.nan),
        "change_in_centroid_distance_w": _round_or_nan(delta),
        "change_in_centroid_distance_loo_m": _round_or_nan(delta),
        "pre_distance_loo_centroid_m": _float_or_nan(getattr(row, "pre_distance_loo_centroid_m", np.nan)),
        "post_distance_loo_centroid_m": _float_or_nan(getattr(row, "post_distance_loo_centroid_m", np.nan)),
        "starting_centroid_distance_m": np.nan,
        "pre_delta_centroid_distance_m": np.nan,
        "mean_speed_pre_m_s": _float_or_nan(getattr(row, "mean_speed_pre_m_s", np.nan)),
        "observed_polygon_area_pre_m2": np.nan,
        "recent_change_in_polygon_area_m2": np.nan,
        "starting_distance": np.nan,
        "pre_window_speed": _float_or_nan(getattr(row, "mean_speed_pre_m_s", np.nan)),
        "recent_pre_event_distance_change": np.nan,
        "observed_polygon_area_pre": np.nan,
        "recent_change_in_polygon_area": np.nan,
        "directional_summary_method": "fixed_pre_position_distance_contribution",
        "focal_distance_reduction_m": np.nan,
        "centroid_distance_reduction_m": np.nan,
        "position_distance_interaction_m": np.nan,
        "distance_change_outcome": "invalid",
        "directional_category": None,
        "directional_category_clean": "invalid",
        "focal_contribution_weight": np.nan,
        "centroid_contribution_weight": np.nan,
        "directional_assignment_low_support": True,
        "directional_geometry_valid": False,
        "directional_geometry_invalid_reason": "",
        "distance_sample_count_pre": 0,
        "distance_sample_count_post": 0,
        "matching_rule_used": str(row.matching_rule_used),
        "buffer_rule_used": str(row.buffer_rule_used),
        "buffer_seconds": _float_or_nan(row.buffer_seconds),
        "match_rank": int(row.match_rank) if pd.notna(row.match_rank) else -1,
        "control_match_use_count": int(row.control_match_use_count) if pd.notna(row.control_match_use_count) else 0,
        "standardized_match_distance": _float_or_nan(row.standardized_match_distance),
        "hard_gps_filter_version": hard_gps_filter_version,
        "source_confidence": "threshold_recalculated_from_primary_row",
        "matching_policy": "standard_matching",
        "response_unit": "centimeters",
        "response_transform": "Delta_D_cm = Delta_D_w * 100; no distributional transform",
        "canonical_event_policy": "alpaca_hum and None are canonical; rumble is excluded",
        "centroid_rule": "outlier-filtered retained non-focal collared companions, excluding focal",
        "threshold_m": float(threshold_m),
        "threshold_role": GLOBAL_OUTLIER_THRESHOLD_ROLE,
        "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
        "retained_companion_count_pre": 0,
        "retained_companion_ids": "",
        "retained_companion_scores_m": "",
        "candidate_companion_count_pre": 0,
        "finite_candidate_score_count_pre": 0,
        "available_retained_companions_pre_median": np.nan,
        "available_retained_companions_post_median": np.nan,
        "available_retained_companions_previous_median": np.nan,
        "filtered_centroid_sample_count_previous": 0,
        "filtered_polygon_sample_count_pre": 0,
        "filtered_polygon_sample_count_previous": 0,
        "threshold_density_pre_per_1000m2": np.nan,
        "threshold_centroid_displacement_m": np.nan,
        "threshold_model_row": False,
        "threshold_exclusion_reason": "",
    }


def _excluded_row(base: dict[str, Any], reason: str) -> dict[str, Any]:
    base["threshold_model_row"] = False
    base["threshold_exclusion_reason"] = reason
    return base


def _assert_threshold_invariants(comparison: pd.DataFrame, *, threshold_m: float) -> None:
    values = pd.to_numeric(comparison["threshold_m"], errors="raise").dropna().unique()
    if len(values) != 1 or not np.isclose(float(values[0]), threshold_m):
        raise AssertionError(f"expected one threshold {threshold_m}, found {values}")
    missing_reason = comparison.loc[~comparison["threshold_model_row"].astype(bool) & comparison["threshold_exclusion_reason"].astype(str).eq("")]
    if not missing_reason.empty:
        raise AssertionError("one or more excluded rows lack threshold_exclusion_reason")
    focal_in_retained = comparison.apply(
        lambda row: str(row["focal_id"]) in set(str(row["retained_companion_ids"]).split("|")) if str(row["retained_companion_ids"]) else False,
        axis=1,
    )
    if focal_in_retained.any():
        raise AssertionError("focal alpaca appears in a retained companion set")


def _assert_no_excluded_predictors(model_input: pd.DataFrame) -> None:
    present = sorted(EXCLUDED_PREDICTOR_COLUMNS & set(model_input.columns))
    if present:
        raise AssertionError(f"excluded predictor columns are present in threshold model input: {present}")


def _window_indices(grid: SessionTraceGrid, t0_seconds_local: float, window_s: int, *, side: str) -> np.ndarray:
    times = grid.bin_times_s
    if side == "pre":
        start = float(t0_seconds_local) - int(window_s)
        stop = float(t0_seconds_local)
        include_start = True
        include_stop = False
    elif side == "post":
        start = float(t0_seconds_local)
        stop = float(t0_seconds_local) + int(window_s)
        include_start = False
        include_stop = True
    else:
        raise ValueError(f"unsupported side: {side}")
    lo = max(0, int(np.searchsorted(grid.second_bins, np.floor(start) - 2, side="left")))
    hi = min(len(times), int(np.searchsorted(grid.second_bins, np.ceil(stop) + 2, side="right")))
    if hi <= lo:
        return np.asarray([], dtype=int)
    indices = np.arange(lo, hi, dtype=int)
    finite = np.isfinite(times[indices])
    lower = times[indices] >= start if include_start else times[indices] > start
    upper = times[indices] <= stop if include_stop else times[indices] < stop
    return indices[finite & lower & upper]


def _previous_window_indices(grid: SessionTraceGrid, t0_seconds_local: float, window_s: int) -> np.ndarray:
    times = grid.bin_times_s
    start = float(t0_seconds_local) - 2 * int(window_s)
    stop = float(t0_seconds_local) - int(window_s)
    lo = max(0, int(np.searchsorted(grid.second_bins, np.floor(start) - 2, side="left")))
    hi = min(len(times), int(np.searchsorted(grid.second_bins, np.ceil(stop) + 2, side="right")))
    if hi <= lo:
        return np.asarray([], dtype=int)
    indices = np.arange(lo, hi, dtype=int)
    return indices[np.isfinite(times[indices]) & (times[indices] >= start) & (times[indices] < stop)]


def focal_speed_values_for_indices(grid: SessionTraceGrid, focal_index: int, indices: np.ndarray) -> np.ndarray:
    if indices.size < 2:
        return np.asarray([], dtype=float)
    valid = valid_slot_mask_at_indices(grid, focal_index, indices) & np.isfinite(grid.bin_times_s[indices])
    focal_indices = indices[valid]
    if len(focal_indices) < 2:
        return np.asarray([], dtype=float)
    dx = np.diff(grid.x_m[focal_indices, focal_index])
    dy = np.diff(grid.y_m[focal_indices, focal_index])
    dt = np.diff(grid.bin_times_s[focal_indices])
    good = (dt > 0) & np.isfinite(dx) & np.isfinite(dy)
    if not np.any(good):
        return np.asarray([], dtype=float)
    return np.hypot(dx[good], dy[good]) / dt[good]


def available_retained_counts(grid: SessionTraceGrid, retained: list[int], indices: np.ndarray) -> np.ndarray:
    if indices.size == 0 or not retained:
        return np.asarray([], dtype=int)
    return selected_valid_matrix_for_indices(grid, retained, indices).sum(axis=1)


def median_centroid_displacement(pre_centroids: np.ndarray, post_centroids: np.ndarray) -> float:
    if pre_centroids.size == 0 or post_centroids.size == 0:
        return np.nan
    pre = np.nanmedian(pre_centroids, axis=0)
    post = np.nanmedian(post_centroids, axis=0)
    if not np.all(np.isfinite(pre)) or not np.all(np.isfinite(post)):
        return np.nan
    return float(np.linalg.norm(post - pre))


def usable_centroid_indices(grid: SessionTraceGrid, focal_index: int, retained: list[int], indices: np.ndarray) -> np.ndarray:
    if indices.size == 0 or not retained:
        return np.asarray([], dtype=int)
    focal_valid = valid_slot_mask_at_indices(grid, focal_index, indices)
    companion_valid = selected_valid_matrix_for_indices(grid, retained, indices)
    counts = companion_valid.sum(axis=1)
    return indices[focal_valid & (counts >= 1)]


def selected_valid_matrix_for_indices(grid: SessionTraceGrid, retained: list[int], indices: np.ndarray) -> np.ndarray:
    if indices.size == 0 or not retained:
        return np.zeros((len(indices), 0), dtype=bool)
    valid = grid.valid[np.ix_(indices, retained)].copy()
    valid &= np.isfinite(grid.x_m[np.ix_(indices, retained)])
    valid &= np.isfinite(grid.y_m[np.ix_(indices, retained)])
    return valid


def valid_companion_slots_at_row(grid: SessionTraceGrid, retained: list[int], row_index: int) -> np.ndarray:
    if not retained:
        return np.asarray([], dtype=int)
    slots = np.asarray(retained, dtype=int)
    valid = grid.valid[row_index, slots] & np.isfinite(grid.x_m[row_index, slots]) & np.isfinite(grid.y_m[row_index, slots])
    return slots[valid]


def valid_slot_mask_at_indices(grid: SessionTraceGrid, slot: int, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.asarray([], dtype=bool)
    return grid.valid[indices, slot] & np.isfinite(grid.x_m[indices, slot]) & np.isfinite(grid.y_m[indices, slot])


def valid_slot_at_row(grid: SessionTraceGrid, slot: int, row_index: int) -> bool:
    return bool(grid.valid[row_index, slot] and np.isfinite(grid.x_m[row_index, slot]) and np.isfinite(grid.y_m[row_index, slot]))


def _invalid_directional_result(reason: str) -> dict[str, Any]:
    return {
        "directional_summary_method": "fixed_pre_position_distance_contribution",
        "focal_distance_reduction_m": np.nan,
        "centroid_distance_reduction_m": np.nan,
        "position_distance_interaction_m": np.nan,
        "distance_change_outcome": "invalid",
        "directional_category": None,
        "directional_category_clean": "invalid",
        "focal_contribution_weight": np.nan,
        "centroid_contribution_weight": np.nan,
        "directional_assignment_low_support": True,
        "directional_geometry_valid": False,
        "directional_geometry_invalid_reason": reason,
    }


def relative_to_repo(path: Path, repo_root: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path


def _float_or_nan(value: Any) -> float:
    if not pd.notna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))

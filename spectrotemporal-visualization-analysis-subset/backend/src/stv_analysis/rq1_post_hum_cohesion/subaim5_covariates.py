from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .contracts import PRIMARY_NONFOCAL_THRESHOLD, SUMMARY_METHOD, SessionTraceGrid
from .subaim2_windows import window_mask
from .subaim4_centroid import directional_movement_components


IMPOSSIBLE_SPEED_M_PER_S = 10.0


def compute_baseline_covariates_for_session(
    *,
    observed_centroid_distance: pd.DataFrame,
    window_inventory: pd.DataFrame,
    grid: SessionTraceGrid,
) -> pd.DataFrame:
    session_distances = observed_centroid_distance.loc[
        observed_centroid_distance["day_session_id"].astype(str) == grid.day_session_id
    ].copy()
    if session_distances.empty:
        return _empty_covariate_frame()

    inventory_columns = [
        "analysis_window_id",
        "event_midpoint_local",
        "event_midpoint_seconds_local",
        "filtered_focal_pre_fix_count",
        "filtered_focal_post_fix_count",
        "filtered_nonfocal_pre_unique_count",
        "filtered_nonfocal_post_unique_count",
        "filtered_aligned_pre_sample_count",
        "filtered_aligned_post_sample_count",
    ]
    joined = session_distances.merge(
        window_inventory[inventory_columns],
        on="analysis_window_id",
        how="left",
        validate="one_to_one",
    )
    rows = [
        {
            **_base_window_identity(row),
            "condition": "hum",
            "window_center_timestamp": row.event_midpoint_local,
            "window_center_seconds_local": float(row.event_midpoint_seconds_local),
            **_outcome_fields(row),
            **window_covariates(
                grid=grid,
                focal_id=str(row.focal_id),
                t0_seconds_local=float(row.event_midpoint_seconds_local),
                window_s=int(row.window_s),
            ),
            "filtered_focal_pre_fix_count": int(row.filtered_focal_pre_fix_count),
            "filtered_focal_post_fix_count": int(row.filtered_focal_post_fix_count),
            "filtered_nonfocal_pre_unique_count": int(row.filtered_nonfocal_pre_unique_count),
            "filtered_nonfocal_post_unique_count": int(row.filtered_nonfocal_post_unique_count),
            "filtered_aligned_pre_sample_count": int(row.filtered_aligned_pre_sample_count),
            "filtered_aligned_post_sample_count": int(row.filtered_aligned_post_sample_count),
        }
        for row in joined.itertuples(index=False)
    ]
    return pd.DataFrame(rows).sort_values(["date_local", "event_id", "window_s"]).reset_index(drop=True)


def covariate_summary(covariates: pd.DataFrame) -> pd.DataFrame:
    if covariates.empty:
        return pd.DataFrame(
            columns=[
                "window_s",
                "rows",
                "median_starting_centroid_distance_m",
                "median_mean_speed_pre_m_s",
                "median_n_collars_pre",
                "polygon_area_available_rows",
                "pre_delta_available_rows",
            ]
        )
    rows: list[dict[str, Any]] = []
    for window_s, group in covariates.groupby("window_s", sort=True):
        rows.append(
            {
                "window_s": int(window_s),
                "rows": int(len(group.index)),
                "median_starting_centroid_distance_m": float(group["starting_centroid_distance_m"].median()),
                "median_mean_speed_pre_m_s": float(group["mean_speed_pre_m_s"].median()),
                "median_n_collars_pre": float(group["n_collars_pre"].median()),
                "polygon_area_available_rows": int(group["observed_polygon_area_pre_m2"].notna().sum()),
                "pre_delta_available_rows": int(group["pre_delta_centroid_distance_m"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def window_covariates(
    *,
    grid: SessionTraceGrid,
    focal_id: str,
    t0_seconds_local: float,
    window_s: int,
) -> dict[str, Any]:
    focal_index = grid.recorder_index[focal_id]
    pre_mask = window_mask(grid.bin_times_s, t0_seconds_local, window_s, side="pre")
    previous_mask = (
        np.isfinite(grid.bin_times_s)
        & (grid.bin_times_s >= (t0_seconds_local - 2 * window_s))
        & (grid.bin_times_s < (t0_seconds_local - window_s))
    )
    eligible_pre = _eligible_distance_mask(grid, focal_index, pre_mask)
    eligible_previous = _eligible_distance_mask(grid, focal_index, previous_mask)

    pre_distances = grid.loo_distance_m[eligible_pre, focal_index]
    previous_distances = grid.loo_distance_m[eligible_previous, focal_index]
    starting_distance = _safe_nanmedian(pre_distances)
    previous_distance = _safe_nanmedian(previous_distances)
    pre_delta = np.nan if np.isnan(starting_distance) or np.isnan(previous_distance) else starting_distance - previous_distance

    pre_valid = pre_mask & grid.valid[:, focal_index]
    speed_values = _focal_speed_values(grid, focal_index, pre_valid)
    speed_impossible_count = int(np.count_nonzero(speed_values > IMPOSSIBLE_SPEED_M_PER_S))
    plausible_speeds = speed_values[speed_values <= IMPOSSIBLE_SPEED_M_PER_S]

    total_collar_counts = grid.valid[pre_mask].sum(axis=1) if np.any(pre_mask) else np.array([], dtype=np.int16)
    polygon_areas = _polygon_areas_for_mask(grid, pre_mask)
    previous_polygon_areas = _polygon_areas_for_mask(grid, previous_mask)
    polygon_area_pre = _safe_nanmedian(polygon_areas)
    previous_polygon_area = _safe_nanmedian(previous_polygon_areas)
    recent_polygon_change = (
        np.nan if np.isnan(polygon_area_pre) or np.isnan(previous_polygon_area) else polygon_area_pre - previous_polygon_area
    )
    n_collars_pre = _safe_nanmedian(total_collar_counts[total_collar_counts > 0])
    density = np.nan
    if not np.isnan(n_collars_pre) and not np.isnan(polygon_area_pre) and polygon_area_pre > 0:
        density = n_collars_pre / (polygon_area_pre / 1000.0)

    return {
        "starting_centroid_distance_m": _round_or_nan(starting_distance),
        "pre_delta_centroid_distance_m": _round_or_nan(pre_delta),
        "mean_speed_pre_m_s": _round_or_nan(float(np.nanmean(plausible_speeds)) if plausible_speeds.size else np.nan),
        "speed_sample_count_pre": int(plausible_speeds.size),
        "speed_impossible_count_pre": speed_impossible_count,
        "n_collars_pre": _round_or_nan(n_collars_pre),
        "observed_polygon_area_pre_m2": _round_or_nan(polygon_area_pre),
        "recent_change_in_polygon_area_m2": _round_or_nan(recent_polygon_change),
        "time_of_day_minutes": round(float(t0_seconds_local) / 60.0, 6),
        "minutes_since_session_start": _round_or_nan((float(t0_seconds_local) - _session_start_seconds(grid)) / 60.0),
        "local_collared_density_per_1000m2": _round_or_nan(density),
        "covariate_summary_method": SUMMARY_METHOD,
    }


def control_window_features(
    *,
    grid: SessionTraceGrid,
    focal_id: str,
    day_session_id: str,
    date_local: str,
    window_s: int,
    t0_seconds_local: float,
) -> dict[str, Any] | None:
    if focal_id not in grid.recorder_index:
        return None
    focal_index = grid.recorder_index[focal_id]
    pre_mask = window_mask(grid.bin_times_s, t0_seconds_local, window_s, side="pre")
    post_mask = window_mask(grid.bin_times_s, t0_seconds_local, window_s, side="post")
    eligible_pre = _eligible_distance_mask(grid, focal_index, pre_mask)
    eligible_post = _eligible_distance_mask(grid, focal_index, post_mask)
    if not np.any(eligible_pre) or not np.any(eligible_post):
        return None

    pre_loo = float(np.nanmedian(grid.loo_distance_m[eligible_pre, focal_index]))
    post_loo = float(np.nanmedian(grid.loo_distance_m[eligible_post, focal_index]))
    delta_d = post_loo - pre_loo
    covariates = window_covariates(
        grid=grid,
        focal_id=focal_id,
        t0_seconds_local=t0_seconds_local,
        window_s=window_s,
    )
    candidate_second = int(round(float(t0_seconds_local)))
    return {
        "analysis_window_id": f"control:{day_session_id}:{focal_id}:w{int(window_s)}:t{candidate_second}",
        "event_id": "",
        "focal_id": focal_id,
        "day_session_id": day_session_id,
        "date_local": date_local,
        "window_s": int(window_s),
        "condition": "matched_non_hum",
        "window_center_seconds_local": float(t0_seconds_local),
        "distance_summary_method": SUMMARY_METHOD,
        "pre_distance_loo_centroid_m": round(pre_loo, 6),
        "post_distance_loo_centroid_m": round(post_loo, 6),
        "change_in_centroid_distance_loo_m": round(delta_d, 6),
        "distance_sample_count_pre": int(eligible_pre.sum()),
        "distance_sample_count_post": int(eligible_post.sum()),
        "directional_summary_method": "fixed_pre_position_distance_contribution",
        **directional_movement_components(
            grid=grid,
            focal_id=focal_id,
            t0_seconds_local=t0_seconds_local,
            window_s=window_s,
            delta_d_m=delta_d,
        ),
        **covariates,
    }


def _base_window_identity(row: Any) -> dict[str, Any]:
    return {
        "analysis_window_id": row.analysis_window_id,
        "event_id": row.event_id,
        "focal_id": row.focal_id,
        "day_session_id": row.day_session_id,
        "date_local": row.date_local,
        "window_s": int(row.window_s),
    }


def _outcome_fields(row: Any) -> dict[str, Any]:
    return {
        "distance_summary_method": row.distance_summary_method,
        "pre_distance_loo_centroid_m": float(row.pre_distance_loo_centroid_m),
        "post_distance_loo_centroid_m": float(row.post_distance_loo_centroid_m),
        "change_in_centroid_distance_loo_m": float(row.change_in_centroid_distance_loo_m),
        "distance_sample_count_pre": int(row.distance_sample_count_pre),
        "distance_sample_count_post": int(row.distance_sample_count_post),
        "directional_summary_method": row.directional_summary_method,
        "focal_distance_reduction_m": float(row.focal_distance_reduction_m),
        "centroid_distance_reduction_m": float(row.centroid_distance_reduction_m),
        "position_distance_interaction_m": float(row.position_distance_interaction_m),
        "distance_change_outcome": row.distance_change_outcome,
        "directional_category": row.directional_category,
        "directional_category_clean": row.directional_category_clean,
        "focal_contribution_weight": _float_or_nan(row.focal_contribution_weight),
        "centroid_contribution_weight": _float_or_nan(row.centroid_contribution_weight),
        "directional_assignment_low_support": bool(row.directional_assignment_low_support),
        "directional_geometry_valid": bool(row.directional_geometry_valid),
        "directional_geometry_invalid_reason": row.directional_geometry_invalid_reason,
    }


def _eligible_distance_mask(grid: SessionTraceGrid, focal_index: int, mask: np.ndarray) -> np.ndarray:
    return (
        mask
        & np.isfinite(grid.loo_distance_m[:, focal_index])
        & grid.valid[:, focal_index]
        & (grid.nonfocal_count[:, focal_index] >= PRIMARY_NONFOCAL_THRESHOLD)
    )


def _focal_speed_values(grid: SessionTraceGrid, focal_index: int, mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask & grid.valid[:, focal_index] & np.isfinite(grid.bin_times_s))
    if len(indices) < 2:
        return np.array([], dtype=np.float64)
    dx = np.diff(grid.x_m[indices, focal_index])
    dy = np.diff(grid.y_m[indices, focal_index])
    dt = np.diff(grid.bin_times_s[indices])
    good = (dt > 0) & np.isfinite(dx) & np.isfinite(dy)
    if not np.any(good):
        return np.array([], dtype=np.float64)
    return np.hypot(dx[good], dy[good]) / dt[good]


def _polygon_areas_for_mask(grid: SessionTraceGrid, mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask)
    areas: list[float] = []
    for row_index in rows:
        valid_slots = np.flatnonzero(grid.valid[row_index])
        if len(valid_slots) < 3:
            continue
        points = [
            (float(grid.x_m[row_index, slot]), float(grid.y_m[row_index, slot]))
            for slot in valid_slots
            if np.isfinite(grid.x_m[row_index, slot]) and np.isfinite(grid.y_m[row_index, slot])
        ]
        if len(points) < 3:
            continue
        areas.append(_convex_hull_area(points))
    return np.asarray(areas, dtype=np.float64)


def _convex_hull_area(points: list[tuple[float, float]]) -> float:
    unique_points = sorted(set(points))
    if len(unique_points) < 3:
        return 0.0

    def cross(origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(hull):
        next_point = hull[(index + 1) % len(hull)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) / 2.0


def _safe_nanmedian(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(np.nanmedian(finite))


def _round_or_nan(value: float) -> float:
    if value is None or not np.isfinite(value):
        return np.nan
    return round(float(value), 6)


def _float_or_nan(value: Any) -> float:
    return float(value) if pd.notna(value) else np.nan


def _session_start_seconds(grid: SessionTraceGrid) -> float:
    finite = grid.bin_times_s[np.isfinite(grid.bin_times_s)]
    return float(finite.min()) if finite.size else 0.0


def _empty_covariate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "analysis_window_id",
            "event_id",
            "focal_id",
            "day_session_id",
            "date_local",
            "window_s",
            "condition",
            "window_center_timestamp",
            "window_center_seconds_local",
            "distance_summary_method",
            "pre_distance_loo_centroid_m",
            "post_distance_loo_centroid_m",
            "change_in_centroid_distance_loo_m",
            "distance_sample_count_pre",
            "distance_sample_count_post",
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
            "starting_centroid_distance_m",
            "pre_delta_centroid_distance_m",
            "mean_speed_pre_m_s",
            "speed_sample_count_pre",
            "speed_impossible_count_pre",
            "n_collars_pre",
            "observed_polygon_area_pre_m2",
            "recent_change_in_polygon_area_m2",
            "time_of_day_minutes",
            "minutes_since_session_start",
            "local_collared_density_per_1000m2",
            "covariate_summary_method",
        ]
    )

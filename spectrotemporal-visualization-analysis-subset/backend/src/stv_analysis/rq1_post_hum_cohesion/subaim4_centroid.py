from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import PRIMARY_NONFOCAL_THRESHOLD, SUMMARY_METHOD, SessionTraceGrid
from .subaim2_windows import window_mask

DIRECTIONAL_EPSILON_M = 0.05
DIRECTIONAL_OUTPUT_COLUMNS = [
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
]


def compute_observed_centroid_distance(
    window_inventory: pd.DataFrame,
    filtered_grids: dict[str, SessionTraceGrid],
) -> pd.DataFrame:
    valid_windows = window_inventory.loc[window_inventory["filtered_gps_valid_min_nonfocal_3"]].copy()
    rows: list[dict[str, object]] = []
    for row in valid_windows.itertuples(index=False):
        grid = filtered_grids[str(row.day_session_id)]
        focal_index = grid.recorder_index[str(row.focal_id)]
        event_midpoint_s = float(row.event_midpoint_seconds_local)
        pre_mask = window_mask(grid.bin_times_s, event_midpoint_s, int(row.window_s), side="pre")
        post_mask = window_mask(grid.bin_times_s, event_midpoint_s, int(row.window_s), side="post")
        eligible_pre = pre_mask & np.isfinite(grid.loo_distance_m[:, focal_index]) & (grid.nonfocal_count[:, focal_index] >= PRIMARY_NONFOCAL_THRESHOLD)
        eligible_post = post_mask & np.isfinite(grid.loo_distance_m[:, focal_index]) & (grid.nonfocal_count[:, focal_index] >= PRIMARY_NONFOCAL_THRESHOLD)
        pre_loo = float(np.nanmedian(grid.loo_distance_m[eligible_pre, focal_index]))
        post_loo = float(np.nanmedian(grid.loo_distance_m[eligible_post, focal_index]))
        delta_d = post_loo - pre_loo
        movement = directional_movement_components(
            grid=grid,
            focal_id=str(row.focal_id),
            t0_seconds_local=event_midpoint_s,
            window_s=int(row.window_s),
            delta_d_m=delta_d,
        )
        rows.append(
            {
                "analysis_window_id": row.analysis_window_id,
                "event_id": row.event_id,
                "focal_id": row.focal_id,
                "day_session_id": row.day_session_id,
                "date_local": row.date_local,
                "window_s": int(row.window_s),
                "distance_summary_method": SUMMARY_METHOD,
                "pre_distance_loo_centroid_m": round(pre_loo, 6),
                "post_distance_loo_centroid_m": round(post_loo, 6),
                "change_in_centroid_distance_loo_m": round(delta_d, 6),
                "distance_sample_count_pre": int(eligible_pre.sum()),
                "distance_sample_count_post": int(eligible_post.sum()),
                "directional_summary_method": "fixed_pre_position_distance_contribution",
                **movement,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "analysis_window_id",
                "event_id",
                "focal_id",
                "day_session_id",
                "date_local",
                "window_s",
                "distance_summary_method",
                "pre_distance_loo_centroid_m",
                "post_distance_loo_centroid_m",
                "change_in_centroid_distance_loo_m",
                "distance_sample_count_pre",
                "distance_sample_count_post",
                "directional_summary_method",
                *DIRECTIONAL_OUTPUT_COLUMNS,
            ]
        )
    return pd.DataFrame(rows).sort_values(["date_local", "event_id", "window_s"]).reset_index(drop=True)


def centroid_summary(observed_centroid_distance: pd.DataFrame) -> pd.DataFrame:
    if observed_centroid_distance.empty:
        return pd.DataFrame(
            columns=[
                "window_s",
                "rows",
                "median_pre_distance_loo_m",
                "median_post_distance_loo_m",
                "median_change_distance_loo_m",
                "directional_geometry_valid_rows",
            ]
        )

    rows: list[dict[str, object]] = []
    for window_s, group in observed_centroid_distance.groupby("window_s", sort=True):
        rows.append(
            {
                "window_s": int(window_s),
                "rows": int(len(group.index)),
                "median_pre_distance_loo_m": float(group["pre_distance_loo_centroid_m"].median()),
                "median_post_distance_loo_m": float(group["post_distance_loo_centroid_m"].median()),
                "median_change_distance_loo_m": float(group["change_in_centroid_distance_loo_m"].median()),
                "directional_geometry_valid_rows": int(group["directional_geometry_valid"].fillna(False).astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows)


def directional_movement_components(
    *,
    grid: SessionTraceGrid,
    focal_id: str,
    t0_seconds_local: float,
    window_s: int,
    delta_d_m: float | None = None,
    epsilon_m: float = DIRECTIONAL_EPSILON_M,
) -> dict[str, object]:
    focal_index = grid.recorder_index[focal_id]
    pre_mask = window_mask(grid.bin_times_s, t0_seconds_local, window_s, side="pre")
    post_mask = window_mask(grid.bin_times_s, t0_seconds_local, window_s, side="post")

    def median_positions(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        rows = np.flatnonzero(mask & grid.valid[:, focal_index])
        focal_positions: list[tuple[float, float]] = []
        centroid_positions: list[tuple[float, float]] = []
        for row_index in rows:
            other = grid.valid[row_index].copy()
            other[focal_index] = False
            other &= np.isfinite(grid.x_m[row_index]) & np.isfinite(grid.y_m[row_index])
            if int(other.sum()) < PRIMARY_NONFOCAL_THRESHOLD:
                continue
            fx = float(grid.x_m[row_index, focal_index])
            fy = float(grid.y_m[row_index, focal_index])
            if not np.isfinite(fx) or not np.isfinite(fy):
                continue
            focal_positions.append((fx, fy))
            centroid_positions.append((float(np.mean(grid.x_m[row_index, other])), float(np.mean(grid.y_m[row_index, other]))))
        if not focal_positions:
            return None
        return np.median(np.asarray(focal_positions), axis=0), np.median(np.asarray(centroid_positions), axis=0)

    pre = median_positions(pre_mask)
    post = median_positions(post_mask)
    if pre is None:
        return _invalid_directional_result("missing_pre_positions")
    if post is None:
        return _invalid_directional_result("missing_post_positions")
    focal_pre, centroid_pre = pre
    focal_post, centroid_post = post
    return distance_contribution_components(
        focal_pre=focal_pre,
        centroid_pre=centroid_pre,
        focal_post=focal_post,
        centroid_post=centroid_post,
        delta_d_m=delta_d_m,
        epsilon_m=epsilon_m,
    )


def distance_contribution_components(
    *,
    focal_pre: np.ndarray,
    centroid_pre: np.ndarray,
    focal_post: np.ndarray,
    centroid_post: np.ndarray,
    delta_d_m: float | None,
    epsilon_m: float = DIRECTIONAL_EPSILON_M,
) -> dict[str, object]:
    points = [focal_pre, centroid_pre, focal_post, centroid_post]
    if any(not np.all(np.isfinite(point)) for point in points):
        return _invalid_directional_result("non_finite_positions")

    pre_distance = float(np.linalg.norm(focal_pre - centroid_pre))
    focal_post_to_centroid_pre = float(np.linalg.norm(focal_post - centroid_pre))
    focal_pre_to_centroid_post = float(np.linalg.norm(focal_pre - centroid_post))
    position_post_distance = float(np.linalg.norm(focal_post - centroid_post))
    if not all(np.isfinite(value) for value in [pre_distance, focal_post_to_centroid_pre, focal_pre_to_centroid_post, position_post_distance]):
        return _invalid_directional_result("non_finite_distances")

    focal_reduction = pre_distance - focal_post_to_centroid_pre
    centroid_reduction = pre_distance - focal_pre_to_centroid_post
    position_delta = position_post_distance - pre_distance
    delta = position_delta if delta_d_m is None or not np.isfinite(delta_d_m) else float(delta_d_m)
    outcome, category, focal_weight, centroid_weight, low_support = _classify_distance_contributions(
        delta_d_m=delta,
        focal_reduction=focal_reduction,
        centroid_reduction=centroid_reduction,
        epsilon_m=epsilon_m,
    )
    return {
        "directional_summary_method": "fixed_pre_position_distance_contribution",
        "focal_distance_reduction_m": round(focal_reduction, 6),
        "centroid_distance_reduction_m": round(centroid_reduction, 6),
        "position_distance_interaction_m": round(position_delta + focal_reduction + centroid_reduction, 6),
        "distance_change_outcome": outcome,
        "directional_category": category,
        "directional_category_clean": category,
        "focal_contribution_weight": _round_or_nan(focal_weight),
        "centroid_contribution_weight": _round_or_nan(centroid_weight),
        "directional_assignment_low_support": bool(low_support),
        "directional_geometry_valid": True,
        "directional_geometry_invalid_reason": "",
    }


def _classify_distance_contributions(
    *,
    delta_d_m: float,
    focal_reduction: float,
    centroid_reduction: float,
    epsilon_m: float,
) -> tuple[str, str, float, float, bool]:
    if not np.isfinite(delta_d_m):
        return "invalid", "invalid", np.nan, np.nan, True
    if delta_d_m < -epsilon_m:
        focal_amount = max(float(focal_reduction), 0.0)
        centroid_amount = max(float(centroid_reduction), 0.0)
        low_support = focal_amount <= epsilon_m and centroid_amount <= epsilon_m
        if focal_amount > epsilon_m and centroid_amount > epsilon_m:
            category = "both moved toward each other"
        elif focal_amount > epsilon_m:
            category = "focal moved toward centroid"
        elif centroid_amount > epsilon_m:
            category = "centroid moved toward focal"
        else:
            category = "focal moved toward centroid" if abs(focal_reduction) >= abs(centroid_reduction) else "centroid moved toward focal"
        total = focal_amount + centroid_amount
        return "convergence", category, _safe_weight(focal_amount, total), _safe_weight(centroid_amount, total), low_support
    if delta_d_m > epsilon_m:
        focal_amount = max(-float(focal_reduction), 0.0)
        centroid_amount = max(-float(centroid_reduction), 0.0)
        low_support = focal_amount <= epsilon_m and centroid_amount <= epsilon_m
        if focal_amount > epsilon_m and centroid_amount > epsilon_m:
            category = "both moved apart"
        elif focal_amount > epsilon_m:
            category = "focal moved away from centroid"
        elif centroid_amount > epsilon_m:
            category = "centroid moved away from focal"
        else:
            category = "focal moved away from centroid" if abs(focal_reduction) >= abs(centroid_reduction) else "centroid moved away from focal"
        total = focal_amount + centroid_amount
        return "separation", category, _safe_weight(focal_amount, total), _safe_weight(centroid_amount, total), low_support
    return "stable", "stable / no material distance change", np.nan, np.nan, False


def directional_movement_tables(model_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition = pd.to_numeric(model_input["condition"], errors="coerce")
    window_level = pd.DataFrame(
        {
            "analysis_window_id": model_input["analysis_window_id"],
            "condition": condition.astype("Int64"),
            "condition_label": model_input["condition_label"],
            "window_s": pd.to_numeric(model_input["window_s"], errors="coerce").astype("Int64"),
            "focal_id": model_input["focal_id"],
            "session_id": model_input["day_session_id"],
            "fit_complete_case": _fit_complete_case(model_input),
            "pre_distance_m": pd.to_numeric(model_input["pre_distance_loo_centroid_m"], errors="coerce"),
            "post_distance_m": pd.to_numeric(model_input["post_distance_loo_centroid_m"], errors="coerce"),
            "delta_D_m": pd.to_numeric(model_input["change_in_centroid_distance_loo_m"], errors="coerce"),
            "focal_distance_reduction_m": pd.to_numeric(model_input["focal_distance_reduction_m"], errors="coerce"),
            "centroid_distance_reduction_m": pd.to_numeric(model_input["centroid_distance_reduction_m"], errors="coerce"),
            "position_distance_interaction_m": pd.to_numeric(model_input["position_distance_interaction_m"], errors="coerce"),
            "distance_change_outcome": model_input["distance_change_outcome"],
            "directional_category": model_input["directional_category"],
            "directional_category_clean": model_input["directional_category_clean"].fillna("invalid"),
            "focal_contribution_weight": pd.to_numeric(model_input["focal_contribution_weight"], errors="coerce"),
            "centroid_contribution_weight": pd.to_numeric(model_input["centroid_contribution_weight"], errors="coerce"),
            "directional_assignment_low_support": model_input["directional_assignment_low_support"].fillna(False).astype(bool),
            "directional_geometry_valid": model_input["directional_geometry_valid"].fillna(False).astype(bool),
            "directional_geometry_invalid_reason": model_input["directional_geometry_invalid_reason"].fillna(""),
        }
    )
    summary_input = window_level.loc[
        window_level["condition"].eq(1)
        & window_level["fit_complete_case"]
        & window_level["distance_change_outcome"].isin(["convergence", "separation"])
        & window_level["directional_geometry_valid"]
    ].copy()
    counts = (
        summary_input.groupby(
            [
                "window_s",
                "condition",
                "condition_label",
                "distance_change_outcome",
                "directional_category_clean",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            n_windows=("analysis_window_id", "size"),
            median_delta_D_m=("delta_D_m", "median"),
            median_focal_distance_reduction_m=("focal_distance_reduction_m", "median"),
            median_centroid_distance_reduction_m=("centroid_distance_reduction_m", "median"),
            median_focal_contribution_weight=("focal_contribution_weight", "median"),
            median_centroid_contribution_weight=("centroid_contribution_weight", "median"),
            low_support_windows=("directional_assignment_low_support", "sum"),
        )
        .reset_index()
    )
    totals = (
        summary_input.groupby(
            ["window_s", "condition", "condition_label", "distance_change_outcome"],
            observed=True,
        )
        .size()
        .rename("outcome_windows")
        .reset_index()
    )
    summary = counts.merge(
        totals,
        on=["window_s", "condition", "condition_label", "distance_change_outcome"],
        how="left",
    )
    summary["percent_of_outcome"] = summary["n_windows"] / summary["outcome_windows"] * 100.0

    outcome_input = window_level.loc[
        window_level["condition"].eq(1)
        & window_level["fit_complete_case"]
        & window_level["directional_geometry_valid"]
        & window_level["distance_change_outcome"].isin(["convergence", "separation", "stable"])
    ].copy()
    outcome_counts = (
        outcome_input.groupby(
            ["window_s", "condition", "condition_label", "distance_change_outcome"],
            observed=True,
        )
        .size()
        .rename("n_windows")
        .reset_index()
    )
    condition_totals = (
        outcome_input.groupby(["window_s", "condition", "condition_label"], observed=True)
        .size()
        .rename("condition_window_rows")
        .reset_index()
    )
    outcome_summary = outcome_counts.merge(
        condition_totals,
        on=["window_s", "condition", "condition_label"],
        how="left",
    )
    outcome_summary["percent_of_condition_window"] = (
        outcome_summary["n_windows"] / outcome_summary["condition_window_rows"] * 100.0
    )
    return window_level, summary, outcome_summary


def _fit_complete_case(model_input: pd.DataFrame) -> pd.Series:
    required = [
        "change_in_centroid_distance_loo_m",
        "starting_centroid_distance_m",
        "pre_delta_centroid_distance_m",
        "mean_speed_pre_m_s",
        "observed_polygon_area_pre_m2",
        "recent_change_in_polygon_area_m2",
        "condition",
        "window_s",
        "focal_alpaca_id",
        "day_session_id",
    ]
    numeric = [
        "change_in_centroid_distance_loo_m",
        "starting_centroid_distance_m",
        "pre_delta_centroid_distance_m",
        "mean_speed_pre_m_s",
        "observed_polygon_area_pre_m2",
        "recent_change_in_polygon_area_m2",
        "condition",
        "window_s",
    ]
    complete = pd.Series(True, index=model_input.index)
    for column in numeric:
        complete &= pd.to_numeric(model_input[column], errors="coerce").notna()
    for column in set(required) - set(numeric):
        complete &= model_input[column].notna()
    return complete


def _invalid_directional_result(reason: str) -> dict[str, object]:
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


def _safe_weight(value: float, total: float) -> float:
    return value / total if total > 0 else np.nan


def _round_or_nan(value: float) -> float:
    return round(float(value), 6) if np.isfinite(value) else np.nan

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import WINDOW_LENGTHS_SECONDS, SessionTraceGrid
from .subaim5_covariates import control_window_features


TARGET_CONTROLS_PER_HUM = 3
MIN_CONTROLS_PER_HUM = 1
CONTROL_WINDOW_RULE = "analysis_window_only"


@dataclass(frozen=True)
class MatchRule:
    name: str
    time_caliper_min: float
    continuous_caliper_sd: float
    n_collars_caliper: float
    same_focal_required: bool


MATCH_RULES = (
    MatchRule("primary_same_focal", 60.0, 0.5, 2.0, True),
    MatchRule("relax_time_120m", 120.0, 0.5, 2.0, True),
    MatchRule("relax_continuous_1sd", 120.0, 1.0, 2.0, True),
    MatchRule("relax_n_collars_3", 120.0, 1.0, 3.0, True),
    MatchRule("relax_different_focal_same_session", 120.0, 1.0, 3.0, False),
)


def match_controls_for_session(
    *,
    hum_covariates: pd.DataFrame,
    grid: SessionTraceGrid,
    date_local: str,
    global_scales: dict[str, float],
    all_hum_event_times_seconds: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    session_hums = hum_covariates.loc[hum_covariates["day_session_id"].astype(str) == grid.day_session_id].copy()
    if session_hums.empty:
        return _empty_matched_controls(), {
            "day_session_id": grid.day_session_id, "hum_windows": 0, "matched_controls": 0,
            "rule_failure_counts": {rule.name: 0 for rule in MATCH_RULES},
        }

    candidate_frames = [
        _generate_candidates_for_window(
            grid=grid,
            date_local=date_local,
            window_s=window_s,
            session_hums=session_hums.loc[session_hums["window_s"] == window_s],
            all_hum_event_times_seconds=all_hum_event_times_seconds,
        )
        for window_s in WINDOW_LENGTHS_SECONDS
    ]
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else _empty_matched_controls()
    if candidates.empty:
        return _empty_matched_controls(), {
            "day_session_id": grid.day_session_id,
            "hum_windows": int(len(session_hums.index)),
            "candidate_windows": 0,
            "matched_controls": 0,
            "unmatched_hum_windows": int(len(session_hums.index)),
            "rule_failure_counts": {rule.name: 0 for rule in MATCH_RULES},
        }

    selected_rows: list[dict[str, Any]] = []
    reuse_counts: dict[str, int] = {}
    failure_counts = {rule.name: 0 for rule in MATCH_RULES}
    unmatched = 0

    for hum in session_hums.sort_values(["window_s", "event_id"]).itertuples(index=False):
        selected = _select_matches_for_hum(
            hum=hum,
            candidates=candidates,
            reuse_counts=reuse_counts,
            global_scales=global_scales,
            failure_counts=failure_counts,
        )
        if len(selected) < MIN_CONTROLS_PER_HUM:
            unmatched += 1
            continue
        selected_rows.extend(selected)
        for row in selected:
            reuse_key = str(row["control_window_id"])
            reuse_counts[reuse_key] = reuse_counts.get(reuse_key, 0) + 1

    if not selected_rows:
        matched = _empty_matched_controls()
    else:
        matched = pd.DataFrame(selected_rows)
        matched["control_match_use_count"] = [
            int(reuse_counts.get(str(row.control_window_id), 0))
            for row in matched.itertuples(index=False)
        ]
        matched = matched.sort_values(
            ["date_local", "hum_event_id", "window_s", "match_rank"]
        ).reset_index(drop=True)

    summary = {
        "day_session_id": grid.day_session_id,
        "hum_windows": int(len(session_hums.index)),
        "candidate_windows": int(len(candidates.index)),
        "matched_controls": int(len(matched.index)),
        "unmatched_hum_windows": int(unmatched),
        "hum_windows_with_at_least_1_control": int(matched["match_set_id"].nunique()) if not matched.empty else 0,
        "max_control_match_use_count": int(max(reuse_counts.values())) if reuse_counts else 0,
        "control_window_rule": CONTROL_WINDOW_RULE,
        "rule_failure_counts": failure_counts,
    }
    return matched, summary


def match_quality_summary(matched_controls: pd.DataFrame, hum_covariates: pd.DataFrame) -> pd.DataFrame:
    if matched_controls.empty:
        hum_counts = hum_covariates.groupby("window_s", sort=True)["analysis_window_id"].nunique().to_dict()
        return pd.DataFrame([
            {
                "control_window_rule": CONTROL_WINDOW_RULE, "window_s": int(window_s),
                "hum_windows": int(hum_counts.get(window_s, 0)), "hum_windows_with_at_least_1_control": 0,
                "hum_windows_with_3_controls": 0, "matched_controls": 0, "max_control_match_use_count": 0,
                "median_standardized_match_distance": np.nan,
                "primary_time_caliper_minutes": 60.0, "relaxed_time_caliper_minutes": 120.0,
                "primary_continuous_caliper_sd": 0.5, "relaxed_continuous_caliper_sd": 1.0,
                "primary_n_collars_caliper": 2.0, "relaxed_n_collars_caliper": 3.0,
            }
            for window_s in WINDOW_LENGTHS_SECONDS
        ])
    hum_counts = hum_covariates.groupby("window_s", sort=True)["analysis_window_id"].nunique().to_dict()
    rows: list[dict[str, Any]] = []
    set_counts = (
        matched_controls.groupby(["window_s", "match_set_id"], sort=True)
        .size()
        .reset_index(name="controls")
    )
    for window_s, group in matched_controls.groupby("window_s", sort=True):
        window_sets = set_counts.loc[set_counts["window_s"] == window_s]
        rows.append(
            {
                "control_window_rule": CONTROL_WINDOW_RULE,
                "window_s": int(window_s),
                "hum_windows": int(hum_counts.get(window_s, 0)),
                "hum_windows_with_at_least_1_control": int((window_sets["controls"] >= 1).sum()),
                "hum_windows_with_3_controls": int((window_sets["controls"] >= TARGET_CONTROLS_PER_HUM).sum()),
                "matched_controls": int(len(group.index)),
                "max_control_match_use_count": int(group["control_match_use_count"].max()),
                "median_standardized_match_distance": float(group["standardized_match_distance"].median()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["primary_time_caliper_minutes"] = 60.0
    summary["relaxed_time_caliper_minutes"] = 120.0
    summary["primary_continuous_caliper_sd"] = 0.5
    summary["relaxed_continuous_caliper_sd"] = 1.0
    summary["primary_n_collars_caliper"] = 2.0
    summary["relaxed_n_collars_caliper"] = 3.0
    return summary


def _generate_candidates_for_window(
    *,
    grid: SessionTraceGrid,
    date_local: str,
    window_s: int,
    session_hums: pd.DataFrame,
    all_hum_event_times_seconds: np.ndarray,
) -> pd.DataFrame:
    finite_times = grid.bin_times_s[np.isfinite(grid.bin_times_s)]
    if finite_times.size == 0 or session_hums.empty:
        return pd.DataFrame()
    start = int(np.ceil(float(finite_times.min()) + window_s))
    stop = int(np.floor(float(finite_times.max()) - window_s))
    if stop <= start:
        return pd.DataFrame()
    focal_ids = sorted(set(session_hums["focal_id"].astype(str).tolist()) | set(grid.recorder_ids))
    rows: list[dict[str, Any]] = []
    for focal_id in focal_ids:
        if focal_id not in grid.recorder_index:
            continue
        for t0 in range(start, stop + 1, max(30, int(window_s))):
            if _overlaps_hum_window(float(t0), all_hum_event_times_seconds, int(window_s)):
                continue
            features = control_window_features(
                grid=grid,
                focal_id=focal_id,
                day_session_id=grid.day_session_id,
                date_local=date_local,
                window_s=int(window_s),
                t0_seconds_local=float(t0),
            )
            if features is not None:
                features["buffer_rule_used"] = CONTROL_WINDOW_RULE
                features["buffer_seconds"] = 0.0
                rows.append(features)
    return pd.DataFrame(rows)


def _overlaps_hum_window(t0: float, hum_times: np.ndarray, window_s: int) -> bool:
    if hum_times.size == 0:
        return False
    return bool(np.any(np.abs(hum_times - t0) < (2 * window_s)))


def _select_matches_for_hum(
    *,
    hum: Any,
    candidates: pd.DataFrame,
    reuse_counts: dict[str, int],
    global_scales: dict[str, float],
    failure_counts: dict[str, int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for rule in MATCH_RULES:
        eligible = _eligible_candidates(hum=hum, candidates=candidates, rule=rule, global_scales=global_scales)
        if eligible.empty:
            failure_counts[rule.name] += 1
            continue
        eligible = eligible.assign(
            _reuse_count=[
                int(reuse_counts.get(str(analysis_window_id), 0))
                for analysis_window_id in eligible["analysis_window_id"].astype(str)
            ]
        )
        eligible = eligible.sort_values(["_reuse_count", "standardized_match_distance", "analysis_window_id"])
        for candidate in eligible.itertuples(index=False):
            if len(selected) >= TARGET_CONTROLS_PER_HUM:
                break
            if any(row["control_window_id"] == candidate.analysis_window_id for row in selected):
                continue
            selected.append(_matched_row(hum, candidate, len(selected) + 1, rule.name))
        if len(selected) >= MIN_CONTROLS_PER_HUM:
            break
    return selected


def _eligible_candidates(
    *,
    hum: Any,
    candidates: pd.DataFrame,
    rule: MatchRule,
    global_scales: dict[str, float],
) -> pd.DataFrame:
    group = candidates.loc[(candidates["window_s"] == int(hum.window_s)) & (candidates["day_session_id"] == hum.day_session_id)].copy()
    if rule.same_focal_required:
        group = group.loc[group["focal_id"].astype(str) == str(hum.focal_id)].copy()
    if group.empty:
        return group
    group["time_difference_minutes"] = (group["window_center_seconds_local"].astype(float) - float(hum.window_center_seconds_local)) / 60.0
    group["starting_centroid_distance_difference_m"] = (
        group["starting_centroid_distance_m"].astype(float) - float(hum.starting_centroid_distance_m)
    )
    group["speed_difference_m_s"] = group["mean_speed_pre_m_s"].astype(float) - float(hum.mean_speed_pre_m_s)
    group["n_collars_difference"] = group["n_collars_pre"].astype(float) - float(hum.n_collars_pre)
    start_scale = max(float(global_scales.get("starting_centroid_distance_m", 1.0)), 1e-9)
    speed_scale = max(float(global_scales.get("log1p_mean_speed_pre_m_s", 1.0)), 1e-9)
    start_z = group["starting_centroid_distance_difference_m"].abs() / start_scale
    speed_z = (
        np.log1p(group["mean_speed_pre_m_s"].clip(lower=0).astype(float)) - np.log1p(max(float(hum.mean_speed_pre_m_s), 0.0))
    ).abs() / speed_scale
    passes = (
        group["time_difference_minutes"].abs().le(rule.time_caliper_min)
        & group["n_collars_difference"].abs().le(rule.n_collars_caliper)
        & start_z.le(rule.continuous_caliper_sd)
        & speed_z.le(rule.continuous_caliper_sd)
    )
    group = group.loc[passes].copy()
    if group.empty:
        return group
    group["standardized_match_distance"] = np.sqrt(
        (group["time_difference_minutes"].abs() / rule.time_caliper_min) ** 2
        + (group["n_collars_difference"].abs() / max(rule.n_collars_caliper, 1e-9)) ** 2
        + start_z.loc[group.index] ** 2
        + speed_z.loc[group.index] ** 2
    )
    return group


def _matched_row(hum: Any, candidate: Any, rank: int, rule_name: str) -> dict[str, Any]:
    return {
        "match_set_id": f"{hum.event_id}:w{int(hum.window_s)}",
        "hum_analysis_window_id": hum.analysis_window_id,
        "hum_event_id": hum.event_id,
        "control_window_id": candidate.analysis_window_id,
        "condition": "matched_non_hum",
        "focal_id": candidate.focal_id,
        "hum_focal_id": hum.focal_id,
        "same_focal_match": str(candidate.focal_id) == str(hum.focal_id),
        "day_session_id": candidate.day_session_id,
        "date_local": candidate.date_local,
        "window_s": int(candidate.window_s),
        "window_center_seconds_local": float(candidate.window_center_seconds_local),
        "hum_window_center_seconds_local": float(hum.window_center_seconds_local),
        "time_difference_minutes": round(float(candidate.time_difference_minutes), 6),
        "standardized_match_distance": round(float(candidate.standardized_match_distance), 6),
        "starting_centroid_distance_difference_m": round(float(candidate.starting_centroid_distance_difference_m), 6),
        "speed_difference_m_s": round(float(candidate.speed_difference_m_s), 6),
        "n_collars_difference": round(float(candidate.n_collars_difference), 6),
        "matching_rule_used": rule_name,
        "buffer_rule_used": CONTROL_WINDOW_RULE,
        "buffer_seconds": _float_or_nan(getattr(candidate, "buffer_seconds", np.nan)),
        "match_rank": int(rank),
        "control_match_use_count": 0,
        "distance_summary_method": candidate.distance_summary_method,
        "pre_distance_loo_centroid_m": float(candidate.pre_distance_loo_centroid_m),
        "post_distance_loo_centroid_m": float(candidate.post_distance_loo_centroid_m),
        "change_in_centroid_distance_loo_m": float(candidate.change_in_centroid_distance_loo_m),
        "distance_sample_count_pre": int(candidate.distance_sample_count_pre),
        "distance_sample_count_post": int(candidate.distance_sample_count_post),
        "directional_summary_method": candidate.directional_summary_method,
        "focal_distance_reduction_m": _float_or_nan(candidate.focal_distance_reduction_m),
        "centroid_distance_reduction_m": _float_or_nan(candidate.centroid_distance_reduction_m),
        "position_distance_interaction_m": _float_or_nan(candidate.position_distance_interaction_m),
        "distance_change_outcome": candidate.distance_change_outcome,
        "directional_category": candidate.directional_category,
        "directional_category_clean": candidate.directional_category_clean,
        "focal_contribution_weight": _float_or_nan(candidate.focal_contribution_weight),
        "centroid_contribution_weight": _float_or_nan(candidate.centroid_contribution_weight),
        "directional_assignment_low_support": bool(candidate.directional_assignment_low_support),
        "directional_geometry_valid": bool(candidate.directional_geometry_valid),
        "directional_geometry_invalid_reason": candidate.directional_geometry_invalid_reason,
        "starting_centroid_distance_m": float(candidate.starting_centroid_distance_m),
        "pre_delta_centroid_distance_m": _float_or_nan(candidate.pre_delta_centroid_distance_m),
        "mean_speed_pre_m_s": float(candidate.mean_speed_pre_m_s),
        "speed_sample_count_pre": int(candidate.speed_sample_count_pre),
        "speed_impossible_count_pre": int(candidate.speed_impossible_count_pre),
        "n_collars_pre": float(candidate.n_collars_pre),
        "observed_polygon_area_pre_m2": _float_or_nan(candidate.observed_polygon_area_pre_m2),
        "recent_change_in_polygon_area_m2": _float_or_nan(candidate.recent_change_in_polygon_area_m2),
        "time_of_day_minutes": float(candidate.time_of_day_minutes),
        "minutes_since_session_start": float(candidate.minutes_since_session_start),
        "local_collared_density_per_1000m2": _float_or_nan(candidate.local_collared_density_per_1000m2),
        "covariate_summary_method": candidate.covariate_summary_method,
    }


def global_match_scales(hum_covariates: pd.DataFrame) -> dict[str, float]:
    if hum_covariates.empty:
        return {"starting_centroid_distance_m": 1.0, "log1p_mean_speed_pre_m_s": 1.0}
    start_sd = float(hum_covariates["starting_centroid_distance_m"].astype(float).std(skipna=True))
    speed_sd = float(np.log1p(hum_covariates["mean_speed_pre_m_s"].clip(lower=0).astype(float)).std(skipna=True))
    return {
        "starting_centroid_distance_m": start_sd if np.isfinite(start_sd) and start_sd > 0 else 1.0,
        "log1p_mean_speed_pre_m_s": speed_sd if np.isfinite(speed_sd) and speed_sd > 0 else 1.0,
    }


def _float_or_nan(value: Any) -> float:
    return float(value) if pd.notna(value) else np.nan


def _empty_matched_controls() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_set_id",
            "hum_analysis_window_id",
            "hum_event_id",
            "control_window_id",
            "condition",
            "focal_id",
            "hum_focal_id",
            "same_focal_match",
            "day_session_id",
            "date_local",
            "window_s",
            "window_center_seconds_local",
            "hum_window_center_seconds_local",
            "time_difference_minutes",
            "standardized_match_distance",
            "starting_centroid_distance_difference_m",
            "speed_difference_m_s",
            "n_collars_difference",
            "matching_rule_used",
            "buffer_rule_used",
            "buffer_seconds",
            "match_rank",
            "control_match_use_count",
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

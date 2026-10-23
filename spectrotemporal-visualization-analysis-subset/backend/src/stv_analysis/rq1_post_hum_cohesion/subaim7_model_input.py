from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_INPUT_COLUMNS = [
    "analysis_window_id",
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
    "change_in_centroid_distance_loo_m",
    "pre_distance_loo_centroid_m",
    "post_distance_loo_centroid_m",
    "Delta_D_w",
    "change_in_centroid_distance_w",
    "starting_centroid_distance_m",
    "pre_delta_centroid_distance_m",
    "mean_speed_pre_m_s",
    "n_collars_pre",
    "observed_polygon_area_pre_m2",
    "recent_change_in_polygon_area_m2",
    "time_of_day_minutes",
    "minutes_since_session_start",
    "local_collared_density_per_1000m2",
    "starting_distance",
    "pre_window_speed",
    "recent_pre_event_distance_change",
    "observed_polygon_area_pre",
    "recent_change_in_polygon_area",
    "time_of_day",
    "collar_availability",
    "local_density",
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
]


def build_model_input(
    *,
    hum_covariates: pd.DataFrame,
    matched_controls: pd.DataFrame,
    hard_gps_filter_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hum_covariates.empty or matched_controls.empty:
        model_input = pd.DataFrame(columns=MODEL_INPUT_COLUMNS)
        return model_input, model_input_summary(model_input)

    matched_hum_ids = sorted(matched_controls["hum_analysis_window_id"].astype(str).unique().tolist())
    hum_rows = hum_covariates.loc[hum_covariates["analysis_window_id"].astype(str).isin(matched_hum_ids)].copy()
    hum_rows["match_set_id"] = hum_rows["event_id"].astype(str) + ":w" + hum_rows["window_s"].astype(str)
    hum_rows["hum_event_id"] = hum_rows["event_id"]
    hum_rows["matching_rule_used"] = "observed_hum"
    hum_rows["buffer_rule_used"] = "analysis_window_only"
    hum_rows["buffer_seconds"] = 0.0
    hum_rows["match_rank"] = 0
    hum_rows["control_match_use_count"] = 0
    hum_rows["standardized_match_distance"] = 0.0
    hum_rows["condition"] = 1
    hum_rows["condition_label"] = "hum"

    control_rows = matched_controls.rename(
        columns={
            "control_window_id": "analysis_window_id",
        }
    ).copy()
    control_rows["event_id"] = ""
    control_rows["condition"] = 0
    control_rows["condition_label"] = "control"

    model_input = pd.concat(
        [hum_rows, control_rows],
        ignore_index=True,
        sort=False,
    ).reindex(columns=MODEL_INPUT_COLUMNS)
    model_input["Delta_D_w"] = pd.to_numeric(model_input["change_in_centroid_distance_loo_m"], errors="coerce")
    model_input["change_in_centroid_distance_w"] = model_input["Delta_D_w"]
    model_input["focal_alpaca_id"] = model_input["focal_id"]
    model_input["session_id"] = model_input["day_session_id"]
    model_input["window_length_s"] = pd.to_numeric(model_input["window_s"], errors="coerce").astype("Int64")
    model_input["window_code"] = "W" + model_input["window_length_s"].astype(str)
    model_input["window_center_timestamp"] = pd.to_datetime(model_input["date_local"], errors="coerce") + pd.to_timedelta(
        pd.to_numeric(model_input["window_center_seconds_local"], errors="coerce"), unit="s"
    )
    model_input["starting_distance"] = model_input["starting_centroid_distance_m"]
    model_input["pre_window_speed"] = model_input["mean_speed_pre_m_s"]
    model_input["recent_pre_event_distance_change"] = model_input["pre_delta_centroid_distance_m"]
    model_input["observed_polygon_area_pre"] = model_input["observed_polygon_area_pre_m2"]
    model_input["recent_change_in_polygon_area"] = model_input["recent_change_in_polygon_area_m2"]
    model_input["time_of_day"] = model_input["time_of_day_minutes"]
    model_input["collar_availability"] = model_input["n_collars_pre"]
    model_input["local_density"] = model_input["local_collared_density_per_1000m2"]
    model_input["hard_gps_filter_version"] = hard_gps_filter_version
    model_input["source_confidence"] = "not_available"
    model_input["condition"] = pd.to_numeric(model_input["condition"], errors="raise").astype("int8")
    model_input = model_input.sort_values(
        ["date_local", "match_set_id", "condition", "match_rank"]
    ).reset_index(drop=True)
    return model_input, model_input_summary(model_input)


def model_input_summary(model_input: pd.DataFrame) -> pd.DataFrame:
    if model_input.empty:
        return pd.DataFrame(
            columns=[
                "window_s",
                "control_window_rule",
                "match_sets",
                "hum_rows",
                "control_rows",
                "median_hum_change_m",
                "median_control_change_m",
                "median_hum_minus_control_change_m",
            ]
        )
    rows: list[dict[str, Any]] = []
    for window_s, group in model_input.groupby("window_s", sort=True):
        hum = group.loc[group["condition"].eq(1)]
        controls = group.loc[group["condition"].eq(0)]
        control_by_set = controls.groupby("match_set_id")["change_in_centroid_distance_loo_m"].mean()
        hum_by_set = hum.set_index("match_set_id")["change_in_centroid_distance_loo_m"]
        paired = hum_by_set.subtract(control_by_set, fill_value=np.nan).dropna()
        rows.append(
            {
                "window_s": int(window_s),
                "control_window_rule": "analysis_window_only",
                "match_sets": int(group["match_set_id"].nunique()),
                "hum_rows": int(len(hum.index)),
                "control_rows": int(len(controls.index)),
                "median_hum_change_m": float(hum["change_in_centroid_distance_loo_m"].median()) if not hum.empty else np.nan,
                "median_control_change_m": float(controls["change_in_centroid_distance_loo_m"].median())
                if not controls.empty
                else np.nan,
                "median_hum_minus_control_change_m": float(paired.median()) if not paired.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)

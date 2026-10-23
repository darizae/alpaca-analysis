from __future__ import annotations

from math import cos, pi
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import LATITUDE_METERS_PER_DEGREE, PRIMARY_NONFOCAL_THRESHOLD, SessionTraceGrid, WINDOW_LENGTHS_SECONDS


def projection_origin(trace_points: pd.DataFrame) -> dict[str, float]:
    return {
        "origin_latitude": float(trace_points["latitude"].mean()),
        "origin_longitude": float(trace_points["longitude"].mean()),
    }


def build_session_trace_grids(
    trace_points: pd.DataFrame,
    *,
    origin_lat: float,
    origin_lon: float,
) -> tuple[dict[str, SessionTraceGrid], dict[str, Any]]:
    lon_scale = max(1e-9, LATITUDE_METERS_PER_DEGREE * cos(origin_lat * pi / 180.0))
    collapsed = trace_points[
        [
            "day_session_id",
            "recorder_id",
            "seconds_since_midnight_local",
            "latitude",
            "longitude",
        ]
    ].copy()
    collapsed["second_bin"] = collapsed["seconds_since_midnight_local"].round().astype("int32")
    collapsed = (
        collapsed.groupby(["day_session_id", "recorder_id", "second_bin"], sort=True, as_index=False)
        .agg(
            fix_time_s=("seconds_since_midnight_local", "median"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            collapsed_fix_count=("latitude", "size"),
        )
    )
    collapsed["x_m"] = (collapsed["longitude"] - origin_lon) * lon_scale
    collapsed["y_m"] = (collapsed["latitude"] - origin_lat) * LATITUDE_METERS_PER_DEGREE

    alignment = (
        collapsed.groupby(["day_session_id", "second_bin"], sort=True, as_index=False)
        .agg(
            recorder_count=("recorder_id", "nunique"),
            median_fix_time_s=("fix_time_s", "median"),
            min_fix_time_s=("fix_time_s", "min"),
            max_fix_time_s=("fix_time_s", "max"),
        )
    )
    alignment["time_spread_s"] = alignment["max_fix_time_s"] - alignment["min_fix_time_s"]

    grids: dict[str, SessionTraceGrid] = {}
    for day_session_id, session_points in collapsed.groupby("day_session_id", sort=True):
        recorder_ids = tuple(sorted(session_points["recorder_id"].unique().tolist()))
        recorder_index = {recorder_id: index for index, recorder_id in enumerate(recorder_ids)}
        min_second = int(session_points["second_bin"].min())
        max_second = int(session_points["second_bin"].max())
        second_bins = np.arange(min_second, max_second + 1, dtype=np.int32)
        n_seconds = len(second_bins)
        n_recorders = len(recorder_ids)

        valid = np.zeros((n_seconds, n_recorders), dtype=bool)
        x_m = np.full((n_seconds, n_recorders), np.nan, dtype=np.float64)
        y_m = np.full((n_seconds, n_recorders), np.nan, dtype=np.float64)
        bin_times_s = np.full(n_seconds, np.nan, dtype=np.float64)

        session_alignment = alignment.loc[alignment["day_session_id"] == day_session_id]
        second_offsets = session_alignment["second_bin"].to_numpy(dtype=np.int32) - min_second
        bin_times_s[second_offsets] = session_alignment["median_fix_time_s"].to_numpy(dtype=np.float64)

        for row in session_points.itertuples(index=False):
            second_index = int(row.second_bin) - min_second
            recorder_slot = recorder_index[str(row.recorder_id)]
            valid[second_index, recorder_slot] = True
            x_m[second_index, recorder_slot] = float(row.x_m)
            y_m[second_index, recorder_slot] = float(row.y_m)

        nonfocal_count, loo_distance_m = distance_arrays(x_m, y_m, valid)
        grids[day_session_id] = SessionTraceGrid(
            day_session_id=day_session_id,
            recorder_ids=recorder_ids,
            recorder_index=recorder_index,
            second_bins=second_bins,
            bin_times_s=bin_times_s,
            valid=valid,
            x_m=x_m,
            y_m=y_m,
            nonfocal_count=nonfocal_count,
            loo_distance_m=loo_distance_m,
        )

    sync_summary = trace_sync_summary_from_collapsed(trace_points, collapsed, alignment)
    return grids, sync_summary


def trace_sync_summary(trace_points: pd.DataFrame) -> dict[str, Any]:
    collapsed = trace_points[
        [
            "day_session_id",
            "recorder_id",
            "seconds_since_midnight_local",
            "latitude",
        ]
    ].copy()
    collapsed["second_bin"] = collapsed["seconds_since_midnight_local"].round().astype("int32")
    collapsed = (
        collapsed.groupby(["day_session_id", "recorder_id", "second_bin"], sort=True, as_index=False)
        .agg(
            fix_time_s=("seconds_since_midnight_local", "median"),
            collapsed_fix_count=("latitude", "size"),
        )
    )
    alignment = (
        collapsed.groupby(["day_session_id", "second_bin"], sort=True, as_index=False)
        .agg(
            recorder_count=("recorder_id", "nunique"),
            median_fix_time_s=("fix_time_s", "median"),
            min_fix_time_s=("fix_time_s", "min"),
            max_fix_time_s=("fix_time_s", "max"),
        )
    )
    alignment["time_spread_s"] = alignment["max_fix_time_s"] - alignment["min_fix_time_s"]
    return trace_sync_summary_from_collapsed(trace_points, collapsed, alignment)


def trace_sync_summary_from_collapsed(
    trace_points: pd.DataFrame,
    collapsed: pd.DataFrame,
    alignment: pd.DataFrame,
) -> dict[str, Any]:
    duplicate_groups = collapsed.loc[collapsed["collapsed_fix_count"] > 1, "collapsed_fix_count"]
    fix_intervals = (
        trace_points.sort_values(["day_session_id", "recorder_id", "seconds_since_midnight_local"])
        .groupby(["day_session_id", "recorder_id"])["seconds_since_midnight_local"]
        .diff()
        .dropna()
    )
    aligned = alignment.loc[alignment["recorder_count"] >= 2, "time_spread_s"]
    return {
        "rounded_second_alignment_rule": "nearest local second",
        "median_fix_interval_s": float(fix_intervals.median()) if not fix_intervals.empty else 0.0,
        "duplicate_recorder_second_groups": int((collapsed["collapsed_fix_count"] > 1).sum()),
        "max_fixes_in_recorder_second_group": int(duplicate_groups.max()) if not duplicate_groups.empty else 1,
        "time_spread_quantiles_s": {
            "p50": float(aligned.quantile(0.50)) if not aligned.empty else 0.0,
            "p95": float(aligned.quantile(0.95)) if not aligned.empty else 0.0,
            "p99": float(aligned.quantile(0.99)) if not aligned.empty else 0.0,
            "max": float(aligned.max()) if not aligned.empty else 0.0,
        },
    }


def distance_arrays(
    x_m: np.ndarray,
    y_m: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_fill = np.where(valid, x_m, 0.0)
    y_fill = np.where(valid, y_m, 0.0)
    total_x = x_fill.sum(axis=1)
    total_y = y_fill.sum(axis=1)
    total_count = valid.sum(axis=1)

    n_seconds, n_recorders = valid.shape
    nonfocal_count = np.zeros((n_seconds, n_recorders), dtype=np.int16)
    loo_distance_m = np.full((n_seconds, n_recorders), np.nan, dtype=np.float64)

    for recorder_index in range(n_recorders):
        focal_valid = valid[:, recorder_index]
        focal_x = x_fill[:, recorder_index]
        focal_y = y_fill[:, recorder_index]
        recorder_nonfocal_count = total_count - focal_valid.astype(np.int16)
        nonfocal_count[:, recorder_index] = recorder_nonfocal_count

        loo_mask = focal_valid & (recorder_nonfocal_count >= 1)
        if np.any(loo_mask):
            loo_centroid_x = (total_x[loo_mask] - focal_x[loo_mask]) / recorder_nonfocal_count[loo_mask]
            loo_centroid_y = (total_y[loo_mask] - focal_y[loo_mask]) / recorder_nonfocal_count[loo_mask]
            loo_distance_m[loo_mask, recorder_index] = np.hypot(
                focal_x[loo_mask] - loo_centroid_x,
                focal_y[loo_mask] - loo_centroid_y,
            )

    return nonfocal_count, loo_distance_m


def summarize_event_windows(
    analysis_units: pd.DataFrame,
    grids: dict[str, SessionTraceGrid],
    *,
    prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in analysis_units.itertuples(index=False):
        day_session_id = str(row.day_session_id)
        focal_id = str(row.focal_id)
        for window_s in WINDOW_LENGTHS_SECONDS:
            base_row = {
                "analysis_window_id": f"{row.event_id}:w{window_s}",
                "event_id": row.event_id,
                "focal_id": focal_id,
                "day_session_id": day_session_id,
                "date_local": row.date_local,
                "window_s": int(window_s),
                "event_midpoint_local": row.event_midpoint_local,
                "event_midpoint_seconds_local": float(row.event_midpoint_seconds_local),
            }
            grid = grids.get(day_session_id)
            if grid is None or focal_id not in grid.recorder_index:
                rows.append(
                    {
                        **base_row,
                        f"{prefix}_focal_pre_fix_count": 0,
                        f"{prefix}_focal_post_fix_count": 0,
                        f"{prefix}_nonfocal_pre_unique_count": 0,
                        f"{prefix}_nonfocal_post_unique_count": 0,
                        f"{prefix}_aligned_pre_sample_count": 0,
                        f"{prefix}_aligned_post_sample_count": 0,
                    }
                )
                continue

            focal_index = grid.recorder_index[focal_id]
            pre_mask = window_mask(grid.bin_times_s, float(row.event_midpoint_seconds_local), window_s, side="pre")
            post_mask = window_mask(grid.bin_times_s, float(row.event_midpoint_seconds_local), window_s, side="post")

            focal_pre_fix_count = int(grid.valid[pre_mask, focal_index].sum())
            focal_post_fix_count = int(grid.valid[post_mask, focal_index].sum())
            nonfocal_pre_unique_count = nonfocal_unique_count(grid.valid[pre_mask], focal_index)
            nonfocal_post_unique_count = nonfocal_unique_count(grid.valid[post_mask], focal_index)
            aligned_pre_sample_count = int(
                np.count_nonzero(pre_mask & grid.valid[:, focal_index] & (grid.nonfocal_count[:, focal_index] >= PRIMARY_NONFOCAL_THRESHOLD))
            )
            aligned_post_sample_count = int(
                np.count_nonzero(post_mask & grid.valid[:, focal_index] & (grid.nonfocal_count[:, focal_index] >= PRIMARY_NONFOCAL_THRESHOLD))
            )
            rows.append(
                {
                    **base_row,
                    f"{prefix}_focal_pre_fix_count": focal_pre_fix_count,
                    f"{prefix}_focal_post_fix_count": focal_post_fix_count,
                    f"{prefix}_nonfocal_pre_unique_count": nonfocal_pre_unique_count,
                    f"{prefix}_nonfocal_post_unique_count": nonfocal_post_unique_count,
                    f"{prefix}_aligned_pre_sample_count": aligned_pre_sample_count,
                    f"{prefix}_aligned_post_sample_count": aligned_post_sample_count,
                }
            )
    return pd.DataFrame(rows)


def window_mask(
    bin_times_s: np.ndarray,
    t0_seconds_local: float,
    window_s: int,
    *,
    side: str,
) -> np.ndarray:
    if side == "pre":
        return np.isfinite(bin_times_s) & (bin_times_s >= (t0_seconds_local - window_s)) & (bin_times_s < t0_seconds_local)
    if side == "post":
        return np.isfinite(bin_times_s) & (bin_times_s > t0_seconds_local) & (bin_times_s <= (t0_seconds_local + window_s))
    raise ValueError(f"unsupported side: {side}")


def nonfocal_unique_count(valid_window: np.ndarray, focal_index: int) -> int:
    if valid_window.size == 0:
        return 0
    any_valid = valid_window.any(axis=0)
    any_valid[focal_index] = False
    return int(any_valid.sum())


def write_session_trace_grid(grid: SessionTraceGrid, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        day_session_id=np.array([grid.day_session_id], dtype=object),
        recorder_ids=np.array(grid.recorder_ids, dtype=object),
        second_bins=grid.second_bins,
        bin_times_s=grid.bin_times_s,
        valid=grid.valid,
        x_m=grid.x_m,
        y_m=grid.y_m,
        nonfocal_count=grid.nonfocal_count,
        loo_distance_m=grid.loo_distance_m,
    )


def read_session_trace_grid(path: Path) -> SessionTraceGrid:
    data = np.load(path, allow_pickle=True)
    recorder_ids = tuple(str(value) for value in data["recorder_ids"].tolist())
    return SessionTraceGrid(
        day_session_id=str(data["day_session_id"].tolist()[0]),
        recorder_ids=recorder_ids,
        recorder_index={recorder_id: index for index, recorder_id in enumerate(recorder_ids)},
        second_bins=np.asarray(data["second_bins"], dtype=np.int32),
        bin_times_s=np.asarray(data["bin_times_s"], dtype=np.float64),
        valid=np.asarray(data["valid"], dtype=bool),
        x_m=np.asarray(data["x_m"], dtype=np.float64),
        y_m=np.asarray(data["y_m"], dtype=np.float64),
        nonfocal_count=np.asarray(data["nonfocal_count"], dtype=np.int16),
        loo_distance_m=np.asarray(data["loo_distance_m"], dtype=np.float64),
    )

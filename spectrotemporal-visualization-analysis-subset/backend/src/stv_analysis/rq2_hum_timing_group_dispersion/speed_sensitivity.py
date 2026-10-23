from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import subprocess
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..rq1_post_hum_cohesion.subaim2_windows import read_session_trace_grid
from .contracts import BIOLOGICALLY_IMPOSSIBLE_SPEED_MPS


DEFAULT_STATIC_GPS_DIR = Path("data/attenuation_raw_data_bundle/gps_precision_sanity_check_20260819")
DEFAULT_OUTPUT_SUBDIR = Path("speed_sensitivity_attenuation_conservative_nb1_interaction")

STRICT_ATTENUATION_THRESHOLD_ID = "attenuation_conservative"
SPEED_INTERVALS_SECONDS = (1, 5, 10, 20)
WINDOW_LENGTH_S = 60
RANDOM_EFFECTS_STRUCTURE_VERSION = "rq2_session_maximal_random_slopes_v1"
ALPHA = 0.05
TERM_ORDER = ("(Intercept)", "z_area", "z_spread_change", "z_speed", "z_centroid_change", "z_area:z_spread_change")
EFFECT_TERMS = ("z_area", "z_spread_change", "z_area:z_spread_change", "z_speed", "z_centroid_change")


def run_speed_sensitivity_analysis(
    *,
    launch_dir: Path,
    attenuation_filter_dir: Path,
    static_gps_dir: Path,
    output_dir: Path,
    threshold_id: str = STRICT_ATTENUATION_THRESHOLD_ID,
    intervals_s: Iterable[int] = SPEED_INTERVALS_SECONDS,
    fit_models: bool = True,
    nsim: int = 1000,
    seed: int = 20260823,
) -> dict[str, Any]:
    intervals = tuple(int(value) for value in intervals_s)
    if not intervals:
        raise ValueError("at least one speed interval is required")
    for interval in intervals:
        if interval < 1:
            raise ValueError(f"speed intervals must be positive integers: {interval}")
        if WINDOW_LENGTH_S % interval != 0:
            raise ValueError(f"speed interval must divide {WINDOW_LENGTH_S} s windows: {interval}")

    for subdir in ("inputs", "tables", "figures", "reports", "cache"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    strict_input = load_strict_attenuation_input(attenuation_filter_dir, threshold_id)
    threshold_meta = load_threshold_metadata(attenuation_filter_dir, threshold_id)
    grids = load_trace_grids(launch_dir, strict_input["day_session_id"].dropna().astype(str).unique())
    speed_by_window = deployment_speed_by_window(strict_input, grids, intervals)
    recompute_audit = one_second_recompute_audit(strict_input, speed_by_window)
    model_input = speed_sensitivity_model_input(strict_input, speed_by_window)
    deployment_summary = summarize_deployment_speed(model_input)
    static_steps, static_summary, static_receiver_summary = static_apparent_speed_tables(static_gps_dir, intervals)
    figure_index = write_speed_figures(
        output_dir=output_dir,
        model_input=model_input,
        deployment_summary=deployment_summary,
        static_summary=static_summary,
    )

    write_base_tables(
        output_dir=output_dir,
        model_input=model_input,
        speed_by_window=speed_by_window,
        deployment_summary=deployment_summary,
        static_steps=static_steps,
        static_summary=static_summary,
        static_receiver_summary=static_receiver_summary,
        threshold_meta=threshold_meta,
        recompute_audit=recompute_audit,
        figure_index=figure_index,
    )

    r_result: dict[str, Any] | None = None
    if fit_models:
        if r_outputs_exist(output_dir):
            r_result = {
                "status": "reused_existing",
                "dharma_simulations_per_model": int(nsim),
                "seed": int(seed),
                "fixed_effects": "tables/rq2_speed_sensitivity_fixed_effects.csv",
                "model_run_manifest": "tables/rq2_speed_sensitivity_model_run_manifest.csv",
            }
        else:
            r_result = run_r_workflow(
                output_dir=output_dir,
                input_path=output_dir / "inputs" / "rq2_speed_sensitivity_model_input_60s.csv",
                intervals_s=intervals,
                nsim=nsim,
                seed=seed,
            )

    comparison_outputs = write_comparison_tables(
        launch_dir=launch_dir,
        attenuation_filter_dir=attenuation_filter_dir,
        output_dir=output_dir,
        threshold_id=threshold_id,
        deployment_summary=deployment_summary,
        static_summary=static_summary,
    )
    write_reports(
        launch_dir=launch_dir,
        attenuation_filter_dir=attenuation_filter_dir,
        static_gps_dir=static_gps_dir,
        output_dir=output_dir,
        threshold_id=threshold_id,
        intervals_s=intervals,
        threshold_meta=threshold_meta,
        recompute_audit=recompute_audit,
        deployment_summary=deployment_summary,
        static_summary=static_summary,
        comparison_outputs=comparison_outputs,
        fit_models=fit_models,
        nsim=nsim,
        seed=seed,
    )

    manifest = {
        "bundle_name": "rq2_speed_sensitivity_attenuation_conservative_nb1_interaction",
        "bundle_version": 1,
        "generated_at_local": pd.Timestamp.now().isoformat(),
        "launch_dir": str(launch_dir.resolve()),
        "attenuation_filter_dir": str(attenuation_filter_dir.resolve()),
        "static_gps_dir": str(static_gps_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "threshold_id": threshold_id,
        "speed_intervals_s": list(intervals),
        "model_family": "NB1",
        "fixed_surface": "interaction",
        "formula": "hum_count ~ z_area * z_spread_change + z_speed + z_centroid_change + offset(log(window_length_minutes)) + (1 + z_area * z_spread_change + z_speed + z_centroid_change | session_id)",
        "random_effects_structure_version": RANDOM_EFFECTS_STRUCTURE_VERSION,
        "random_effects": "session-level random intercept plus maximal session-level random slopes for z_area, z_spread_change, their interaction, z_speed, and z_centroid_change",
        "response_source": "attenuation-calibrated amplitude filter response",
        "response_threshold": threshold_meta.to_dict(orient="records"),
        "speed_algorithm": "within each existing 60 s window, aggregate per-collar GPS positions into interval-specific bins, take median x/y and median observed time per bin, compute distances between consecutive aggregate positions divided by elapsed time, drop step speeds above 12 m/s, and average remaining step speeds across collars",
        "static_gps_algorithm": "same median-bin step-speed algorithm applied to primary static fence GPS receivers from the attenuation GPS precision bundle",
        "models_fit": bool(fit_models),
        "r_workflow": r_result,
        "outputs": {
            "model_input_csv": "inputs/rq2_speed_sensitivity_model_input_60s.csv",
            "model_input_parquet": "inputs/rq2_speed_sensitivity_model_input_60s.parquet",
            "deployment_speed_by_window": "tables/rq2_speed_sensitivity_window_speeds.csv",
            "deployment_speed_summary": "tables/rq2_speed_sensitivity_deployment_speed_summary.csv",
            "static_speed_summary": "tables/rq2_speed_sensitivity_static_apparent_speed_summary.csv",
            "model_run_manifest": "tables/rq2_speed_sensitivity_model_run_manifest.csv",
            "fixed_effects": "tables/rq2_speed_sensitivity_fixed_effects.csv",
            "speed_effect_summary": "tables/rq2_speed_sensitivity_speed_effect_summary.csv",
            "effect_comparison": "tables/rq2_speed_sensitivity_effect_comparison.csv",
            "external_effect_comparison": "tables/rq2_speed_sensitivity_external_effect_comparison.csv",
            "methods": "METHODS.md",
            "results": "RESULTS.md",
            "readme": "README.md",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    return manifest


def load_strict_attenuation_input(attenuation_filter_dir: Path, threshold_id: str) -> pd.DataFrame:
    parquet_path = attenuation_filter_dir / "inputs" / "rq2_attenuation_filter_model_input_60s.parquet"
    csv_path = attenuation_filter_dir / "inputs" / "rq2_attenuation_filter_model_input_60s.csv"
    if parquet_path.is_file():
        model_input = pd.read_parquet(parquet_path)
    elif csv_path.is_file():
        model_input = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"missing attenuation-filter model input under {attenuation_filter_dir / 'inputs'}")
    if "threshold_id" not in model_input.columns:
        raise ValueError("attenuation-filter model input is missing threshold_id")
    strict = model_input.loc[
        (model_input["threshold_id"].astype(str) == threshold_id)
        & (pd.to_numeric(model_input["window_length_s"], errors="coerce").astype("Int64") == WINDOW_LENGTH_S)
    ].copy()
    if strict.empty:
        raise ValueError(f"no 60 s attenuation-filter rows found for threshold_id={threshold_id}")
    strict["rq2_window_id"] = strict["rq2_window_id"].astype(str)
    strict["day_session_id"] = strict["day_session_id"].astype(str)
    strict = strict.sort_values(["day_session_id", "window_start_seconds_local", "rq2_window_id"]).reset_index(drop=True)
    return strict


def load_threshold_metadata(attenuation_filter_dir: Path, threshold_id: str) -> pd.DataFrame:
    path = attenuation_filter_dir / "tables" / "rq2_attenuation_threshold_definitions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing attenuation threshold definitions: {path}")
    thresholds = pd.read_csv(path)
    out = thresholds.loc[thresholds["threshold_id"].astype(str) == threshold_id].copy()
    if out.empty:
        raise ValueError(f"threshold_id={threshold_id} not found in {path}")
    return out.reset_index(drop=True)


def load_trace_grids(launch_dir: Path, day_session_ids: Iterable[str]) -> dict[str, Any]:
    requested = {str(value) for value in day_session_ids}
    candidates = [
        launch_dir / "cache" / "trace_grids",
        launch_dir.parent.parent / "cache" / "trace_grids",
    ]
    trace_grid_dir = next((path for path in candidates if path.is_dir()), None)
    if trace_grid_dir is None:
        rendered = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"missing trace-grid cache; checked {rendered}")
    grids: dict[str, Any] = {}
    for path in sorted(trace_grid_dir.glob("*.npz")):
        grid = read_session_trace_grid(path)
        if str(grid.day_session_id) in requested:
            grids[str(grid.day_session_id)] = grid
    missing = sorted(requested - set(grids))
    if missing:
        raise FileNotFoundError(f"missing trace grids for day_session_id values: {missing}")
    return grids


def deployment_speed_by_window(model_input: pd.DataFrame, grids: dict[str, Any], intervals_s: Iterable[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    windows = model_input[
        ["rq2_window_id", "day_session_id", "window_start_seconds_local", "window_end_seconds_local"]
    ].drop_duplicates("rq2_window_id")
    for interval_s in intervals_s:
        for day_session_id, session_windows in windows.groupby("day_session_id", sort=True):
            rows.extend(deployment_speed_by_session_interval(session_windows, grids[str(day_session_id)], int(interval_s)))
    return pd.DataFrame(rows).sort_values(["speed_interval_s", "day_session_id", "window_start_seconds_local"]).reset_index(drop=True)


def deployment_speed_by_session_interval(session_windows: pd.DataFrame, grid: Any, interval_s: int) -> list[dict[str, Any]]:
    ordered_windows = session_windows.sort_values("window_start_seconds_local").reset_index(drop=True)
    first_start = float(ordered_windows["window_start_seconds_local"].min())
    last_end = float(ordered_windows["window_end_seconds_local"].max())
    bins_per_window = WINDOW_LENGTH_S // int(interval_s)

    window_meta: dict[int, dict[str, Any]] = {}
    accumulators: dict[int, dict[str, Any]] = {}
    for row in ordered_windows.itertuples(index=False):
        window_number = int(round((float(row.window_start_seconds_local) - first_start) / WINDOW_LENGTH_S))
        window_meta[window_number] = {
            "rq2_window_id": str(row.rq2_window_id),
            "day_session_id": str(row.day_session_id),
            "window_start_seconds_local": float(row.window_start_seconds_local),
            "window_end_seconds_local": float(row.window_end_seconds_local),
        }
        accumulators[window_number] = {
            "speeds": [],
            "valid_recorders": 0,
            "aggregate_positions": 0,
            "impossible_speed_samples": 0,
        }

    start_idx = int(np.searchsorted(grid.second_bins, int(np.ceil(first_start)), side="left"))
    end_idx = int(np.searchsorted(grid.second_bins, int(np.ceil(last_end)), side="left"))
    seconds = grid.second_bins[start_idx:end_idx].astype(float)
    valid = grid.valid[start_idx:end_idx]
    x = grid.x_m[start_idx:end_idx]
    y = grid.y_m[start_idx:end_idx]

    for recorder_index in range(valid.shape[1] if valid.size else 0):
        recorder_mask = valid[:, recorder_index]
        if int(recorder_mask.sum()) < 2:
            continue
        binned = aggregate_recorder_position_arrays(
            seconds[recorder_mask],
            x[:, recorder_index][recorder_mask].astype(float),
            y[:, recorder_index][recorder_mask].astype(float),
            start_s=first_start,
            interval_s=interval_s,
        )
        bin_index = binned["bin_index"]
        if len(bin_index) == 0:
            continue
        position_windows = np.floor_divide(bin_index, bins_per_window).astype(int)
        for window_number in np.unique(position_windows):
            if int(window_number) in accumulators:
                accumulators[int(window_number)]["aggregate_positions"] += int(np.count_nonzero(position_windows == window_number))
        if len(bin_index) < 2:
            continue

        recorder_window_counts: dict[int, int] = {}
        dt = np.diff(binned["bin_time_s"])
        dist = np.hypot(np.diff(binned["x_m"]), np.diff(binned["y_m"]))
        speed = np.divide(dist, dt, out=np.full_like(dist, np.nan, dtype=float), where=dt > 0)
        left_windows = np.floor_divide(bin_index[:-1], bins_per_window).astype(int)
        right_windows = np.floor_divide(bin_index[1:], bins_per_window).astype(int)
        same_window = left_windows == right_windows
        for idx, step_speed in enumerate(speed):
            if not same_window[idx]:
                continue
            window_number = int(left_windows[idx])
            if window_number not in accumulators:
                continue
            if not np.isfinite(step_speed):
                continue
            if step_speed > BIOLOGICALLY_IMPOSSIBLE_SPEED_MPS:
                accumulators[window_number]["impossible_speed_samples"] += 1
                continue
            accumulators[window_number]["speeds"].append(float(step_speed))
            recorder_window_counts[window_number] = recorder_window_counts.get(window_number, 0) + 1
        for window_number, count in recorder_window_counts.items():
            if count > 0 and window_number in accumulators:
                accumulators[window_number]["valid_recorders"] += 1

    rows: list[dict[str, Any]] = []
    for window_number, meta in window_meta.items():
        acc = accumulators[window_number]
        speeds = acc["speeds"]
        rows.append(
            {
                **meta,
                "speed_interval_s": int(interval_s),
                "speed_metric_id": speed_metric_id(interval_s),
                "speed_mps": float(np.nanmean(speeds)) if speeds else np.nan,
                "speed_sample_count": int(len(speeds)),
                "speed_valid_recorders": int(acc["valid_recorders"]),
                "speed_aggregate_position_count": int(acc["aggregate_positions"]),
                "biologically_impossible_speed_samples": int(acc["impossible_speed_samples"]),
            }
        )
    return rows


def aggregated_window_speed_metrics(grid: Any, start_s: float, end_s: float, interval_s: int) -> dict[str, Any]:
    start_idx = int(np.searchsorted(grid.second_bins, int(np.ceil(start_s)), side="left"))
    end_idx = int(np.searchsorted(grid.second_bins, int(np.ceil(end_s)), side="left"))
    valid = grid.valid[start_idx:end_idx].copy()
    seconds = grid.second_bins[start_idx:end_idx].astype(float)
    x = grid.x_m[start_idx:end_idx].copy()
    y = grid.y_m[start_idx:end_idx].copy()
    speeds: list[float] = []
    impossible_speed_samples = 0
    aggregate_positions = 0
    recorder_speed_sample_counts: list[int] = []

    if valid.size:
        for recorder_index in range(valid.shape[1]):
            recorder_mask = valid[:, recorder_index]
            if int(recorder_mask.sum()) < 2:
                recorder_speed_sample_counts.append(0)
                continue
            binned = aggregate_recorder_position_arrays(
                seconds[recorder_mask],
                x[:, recorder_index][recorder_mask].astype(float),
                y[:, recorder_index][recorder_mask].astype(float),
                start_s=start_s,
                interval_s=interval_s,
            )
            aggregate_positions += int(len(binned["bin_time_s"]))
            if len(binned["bin_time_s"]) < 2:
                recorder_speed_sample_counts.append(0)
                continue
            dt = np.diff(binned["bin_time_s"])
            dist = np.hypot(
                np.diff(binned["x_m"]),
                np.diff(binned["y_m"]),
            )
            speed = np.divide(dist, dt, out=np.full_like(dist, np.nan, dtype=float), where=dt > 0)
            impossible_speed_samples += int(np.count_nonzero(speed > BIOLOGICALLY_IMPOSSIBLE_SPEED_MPS))
            speed = speed[np.isfinite(speed) & (speed <= BIOLOGICALLY_IMPOSSIBLE_SPEED_MPS)]
            recorder_speed_sample_counts.append(int(len(speed)))
            speeds.extend(speed.tolist())

    valid_speed_recorders = int(sum(count > 0 for count in recorder_speed_sample_counts))
    return {
        "speed_mps": float(np.nanmean(speeds)) if speeds else np.nan,
        "speed_sample_count": int(len(speeds)),
        "speed_valid_recorders": valid_speed_recorders,
        "speed_aggregate_position_count": int(aggregate_positions),
        "biologically_impossible_speed_samples": int(impossible_speed_samples),
    }


def aggregate_recorder_positions(
    seconds: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    start_s: float,
    interval_s: int,
) -> pd.DataFrame:
    arrays = aggregate_recorder_position_arrays(seconds, x_m, y_m, start_s=start_s, interval_s=interval_s)
    return pd.DataFrame(arrays)


def aggregate_recorder_position_arrays(
    seconds: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    start_s: float,
    interval_s: int,
) -> dict[str, np.ndarray]:
    seconds = np.asarray(seconds, dtype=float)
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    finite = np.isfinite(seconds) & np.isfinite(x_m) & np.isfinite(y_m)
    if not np.any(finite):
        return {
            "bin_index": np.array([], dtype=int),
            "bin_time_s": np.array([], dtype=float),
            "x_m": np.array([], dtype=float),
            "y_m": np.array([], dtype=float),
            "n_fixes": np.array([], dtype=int),
        }
    seconds = seconds[finite]
    x_m = x_m[finite]
    y_m = y_m[finite]
    bin_index = np.floor((seconds - float(start_s)) / float(interval_s)).astype(int)
    if len(seconds) > 200:
        frame = pd.DataFrame({"bin_index": bin_index, "second": seconds, "x_m": x_m, "y_m": y_m})
        grouped = (
            frame.groupby("bin_index", sort=True)
            .agg(bin_time_s=("second", "median"), x_m=("x_m", "median"), y_m=("y_m", "median"), n_fixes=("second", "size"))
            .reset_index()
        )
        return {
            "bin_index": grouped["bin_index"].to_numpy(dtype=int),
            "bin_time_s": grouped["bin_time_s"].to_numpy(dtype=float),
            "x_m": grouped["x_m"].to_numpy(dtype=float),
            "y_m": grouped["y_m"].to_numpy(dtype=float),
            "n_fixes": grouped["n_fixes"].to_numpy(dtype=int),
        }
    order = np.argsort(bin_index, kind="mergesort")
    bin_index = bin_index[order]
    seconds = seconds[order]
    x_m = x_m[order]
    y_m = y_m[order]
    unique_bins, starts, counts = np.unique(bin_index, return_index=True, return_counts=True)
    if np.all(counts == 1):
        return {
            "bin_index": unique_bins.astype(int),
            "bin_time_s": seconds[starts].astype(float),
            "x_m": x_m[starts].astype(float),
            "y_m": y_m[starts].astype(float),
            "n_fixes": counts.astype(int),
        }
    bin_time = np.empty(len(unique_bins), dtype=float)
    x_out = np.empty(len(unique_bins), dtype=float)
    y_out = np.empty(len(unique_bins), dtype=float)
    for idx, start in enumerate(starts):
        end = start + counts[idx]
        bin_time[idx] = float(np.median(seconds[start:end]))
        x_out[idx] = float(np.median(x_m[start:end]))
        y_out[idx] = float(np.median(y_m[start:end]))
    return {
        "bin_index": unique_bins.astype(int),
        "bin_time_s": bin_time,
        "x_m": x_out,
        "y_m": y_out,
        "n_fixes": counts.astype(int),
    }


def one_second_recompute_audit(model_input: pd.DataFrame, speed_by_window: pd.DataFrame) -> pd.DataFrame:
    current = model_input[["rq2_window_id", "speed_mps"]].copy()
    current["current_1s_speed_mps"] = pd.to_numeric(current["speed_mps"], errors="coerce")
    recomputed = speed_by_window.loc[speed_by_window["speed_interval_s"] == 1, ["rq2_window_id", "speed_mps"]].copy()
    recomputed = recomputed.rename(columns={"speed_mps": "recomputed_1s_speed_mps"})
    audit_rows = current.merge(recomputed, on="rq2_window_id", how="left")
    audit_rows["absolute_delta_mps"] = (audit_rows["current_1s_speed_mps"] - audit_rows["recomputed_1s_speed_mps"]).abs()
    delta = pd.to_numeric(audit_rows["absolute_delta_mps"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "rows": int(len(audit_rows.index)),
                "matched_recomputed_rows": int(audit_rows["recomputed_1s_speed_mps"].notna().sum()),
                "max_absolute_delta_mps": float(delta.max()) if not delta.dropna().empty else np.nan,
                "median_absolute_delta_mps": float(delta.median()) if not delta.dropna().empty else np.nan,
                "mean_absolute_delta_mps": float(delta.mean()) if not delta.dropna().empty else np.nan,
            }
        ]
    )


def speed_sensitivity_model_input(model_input: pd.DataFrame, speed_by_window: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = model_input.copy()
    base["speed_mps_original_1s"] = pd.to_numeric(base["speed_mps"], errors="coerce")
    for interval_s, speeds in speed_by_window.groupby("speed_interval_s", sort=True):
        frame = base.drop(columns=[column for column in ("speed_sample_count", "speed_valid_recorders", "speed_aggregate_position_count") if column in base.columns]).merge(
            speeds[
                [
                    "rq2_window_id",
                    "speed_metric_id",
                    "speed_mps",
                    "speed_sample_count",
                    "speed_valid_recorders",
                    "speed_aggregate_position_count",
                    "biologically_impossible_speed_samples",
                ]
            ].rename(
                columns={
                    "speed_mps": "speed_mps_recomputed",
                    "biologically_impossible_speed_samples": "biologically_impossible_speed_samples_recomputed",
                }
            ),
            on="rq2_window_id",
            how="left",
        )
        frame["speed_interval_s"] = int(interval_s)
        frame["speed_mps"] = pd.to_numeric(frame["speed_mps_recomputed"], errors="coerce")
        frame["speed_algorithm"] = "window_relative_median_position_bins"
        rows.append(frame)
    output = pd.concat(rows, ignore_index=True)
    return output.sort_values(["speed_interval_s", "day_session_id", "window_start_seconds_local", "rq2_window_id"]).reset_index(drop=True)


def static_apparent_speed_tables(static_gps_dir: Path, intervals_s: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points_path = static_gps_dir / "tables" / "attenuation_gps_points.parquet"
    if not points_path.is_file():
        raise FileNotFoundError(f"missing static GPS points table: {points_path}")
    points = pd.read_parquet(points_path)
    required = {"receiver_id", "primary_static_fence_geometry", "has_valid_fix", "seconds_since_recording_start", "local_x_m", "local_y_m"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"static GPS points table is missing columns: {missing}")
    primary = points.loc[
        points["primary_static_fence_geometry"].astype(bool)
        & points["has_valid_fix"].astype(bool)
        & points["seconds_since_recording_start"].notna()
        & points["local_x_m"].notna()
        & points["local_y_m"].notna()
    ].copy()
    if primary.empty:
        raise ValueError("static GPS points table has no primary static fence fixes")

    step_rows: list[dict[str, Any]] = []
    receiver_summary_rows: list[dict[str, Any]] = []
    for interval_s in intervals_s:
        for receiver_id, receiver_points in primary.groupby("receiver_id", sort=True):
            receiver_steps = static_receiver_step_speeds(receiver_points, int(interval_s))
            for idx, speed in enumerate(receiver_steps):
                step_rows.append(
                    {
                        "receiver_id": str(receiver_id),
                        "speed_interval_s": int(interval_s),
                        "speed_metric_id": speed_metric_id(interval_s),
                        "step_index": int(idx),
                        "apparent_speed_mps": float(speed),
                    }
                )
            values = pd.Series(receiver_steps, dtype=float)
            receiver_summary_rows.append(summary_row_for_values(values, {"receiver_id": str(receiver_id), "speed_interval_s": int(interval_s), "speed_metric_id": speed_metric_id(interval_s)}))

    steps = pd.DataFrame(step_rows)
    receiver_summary = pd.DataFrame(receiver_summary_rows)
    summary_rows = []
    for interval_s, group in steps.groupby("speed_interval_s", sort=True):
        values = pd.to_numeric(group["apparent_speed_mps"], errors="coerce")
        summary_rows.append(
            summary_row_for_values(
                values,
                {
                    "speed_interval_s": int(interval_s),
                    "speed_metric_id": speed_metric_id(interval_s),
                    "static_receiver_scope": "primary_static_fence_geometry",
                    "static_receiver_count": int(group["receiver_id"].nunique()),
                },
            )
        )
    return steps, pd.DataFrame(summary_rows), receiver_summary


def static_receiver_step_speeds(points: pd.DataFrame, interval_s: int) -> list[float]:
    start_s = float(pd.to_numeric(points["seconds_since_recording_start"], errors="coerce").min())
    binned = aggregate_recorder_position_arrays(
        pd.to_numeric(points["seconds_since_recording_start"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(points["local_x_m"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(points["local_y_m"], errors="coerce").to_numpy(dtype=float),
        start_s=start_s,
        interval_s=interval_s,
    )
    if len(binned["bin_time_s"]) < 2:
        return []
    dt = np.diff(binned["bin_time_s"])
    dist = np.hypot(
        np.diff(binned["x_m"]),
        np.diff(binned["y_m"]),
    )
    speed = np.divide(dist, dt, out=np.full_like(dist, np.nan, dtype=float), where=dt > 0)
    speed = speed[np.isfinite(speed) & (speed <= BIOLOGICALLY_IMPOSSIBLE_SPEED_MPS)]
    return speed.tolist()


def summary_row_for_values(values: pd.Series, base: dict[str, Any]) -> dict[str, Any]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = dict(base)
    out.update(
        {
            "speed_sample_count": int(len(finite.index)),
            "mean_speed_mps": float(finite.mean()) if not finite.empty else np.nan,
            "median_speed_mps": float(finite.median()) if not finite.empty else np.nan,
            "p25_speed_mps": float(finite.quantile(0.25)) if not finite.empty else np.nan,
            "p75_speed_mps": float(finite.quantile(0.75)) if not finite.empty else np.nan,
            "p95_speed_mps": float(finite.quantile(0.95)) if not finite.empty else np.nan,
            "max_speed_mps": float(finite.max()) if not finite.empty else np.nan,
        }
    )
    return out


def summarize_deployment_speed(model_input: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for interval_s, group in model_input.groupby("speed_interval_s", sort=True):
        values = pd.to_numeric(group["speed_mps"], errors="coerce")
        base = {
            "speed_interval_s": int(interval_s),
            "speed_metric_id": speed_metric_id(int(interval_s)),
            "model_rows": int(len(group.index)),
            "sessions": int(group["session_id"].nunique()),
            "hum_count_total": int(pd.to_numeric(group["hum_count"], errors="coerce").sum()),
            "zero_hum_fraction": float((pd.to_numeric(group["hum_count"], errors="coerce") == 0).mean()),
        }
        rows.append(summary_row_for_values(values, base))
    return pd.DataFrame(rows)


def write_base_tables(
    *,
    output_dir: Path,
    model_input: pd.DataFrame,
    speed_by_window: pd.DataFrame,
    deployment_summary: pd.DataFrame,
    static_steps: pd.DataFrame,
    static_summary: pd.DataFrame,
    static_receiver_summary: pd.DataFrame,
    threshold_meta: pd.DataFrame,
    recompute_audit: pd.DataFrame,
    figure_index: pd.DataFrame,
) -> None:
    inputs_dir = output_dir / "inputs"
    tables_dir = output_dir / "tables"
    model_input.to_csv(inputs_dir / "rq2_speed_sensitivity_model_input_60s.csv", index=False)
    model_input.to_parquet(inputs_dir / "rq2_speed_sensitivity_model_input_60s.parquet", index=False)
    speed_by_window.to_csv(tables_dir / "rq2_speed_sensitivity_window_speeds.csv", index=False)
    speed_by_window.to_parquet(tables_dir / "rq2_speed_sensitivity_window_speeds.parquet", index=False)
    deployment_summary.to_csv(tables_dir / "rq2_speed_sensitivity_deployment_speed_summary.csv", index=False)
    static_steps.to_csv(tables_dir / "rq2_speed_sensitivity_static_apparent_speed_steps.csv", index=False)
    static_summary.to_csv(tables_dir / "rq2_speed_sensitivity_static_apparent_speed_summary.csv", index=False)
    static_receiver_summary.to_csv(tables_dir / "rq2_speed_sensitivity_static_receiver_speed_summary.csv", index=False)
    threshold_meta.to_csv(tables_dir / "rq2_speed_sensitivity_threshold_metadata.csv", index=False)
    recompute_audit.to_csv(tables_dir / "rq2_speed_sensitivity_1s_recompute_audit.csv", index=False)
    figure_index.to_csv(tables_dir / "rq2_speed_sensitivity_python_figure_index.csv", index=False)


def run_r_workflow(*, output_dir: Path, input_path: Path, intervals_s: Iterable[int], nsim: int, seed: int) -> dict[str, Any]:
    script_path = Path(__file__).resolve().with_name("speed_sensitivity_nb1_interaction_workflow.R")
    if not script_path.is_file():
        raise FileNotFoundError(f"missing R workflow: {script_path}")
    command = [
        "Rscript",
        str(script_path),
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--speed-intervals",
        ",".join(str(value) for value in intervals_s),
        "--nsim",
        str(nsim),
        "--seed",
        str(seed),
    ]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[4], text=True, capture_output=True, check=False)
    log_dir = output_dir / "cache"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "rq2_speed_sensitivity_stdout.log"
    stderr_log = log_dir / "rq2_speed_sensitivity_stderr.log"
    stdout_log.write_text(completed.stdout, encoding="utf8")
    stderr_log.write_text(completed.stderr, encoding="utf8")
    if completed.returncode != 0:
        raise RuntimeError(f"speed-sensitivity R workflow failed with exit code {completed.returncode}; see {stderr_log}")
    return {
        "script": str(script_path),
        "exit_code": int(completed.returncode),
        "stdout_log": "cache/rq2_speed_sensitivity_stdout.log",
        "stderr_log": "cache/rq2_speed_sensitivity_stderr.log",
        "dharma_simulations_per_model": int(nsim),
        "seed": int(seed),
    }


def r_outputs_exist(output_dir: Path) -> bool:
    tables_dir = output_dir / "tables"
    required = (
        "rq2_speed_sensitivity_model_run_manifest.csv",
        "rq2_speed_sensitivity_fixed_effects.csv",
        "rq2_speed_sensitivity_predictor_standardization.csv",
        "rq2_speed_sensitivity_dharma_standard_tests.csv",
        "rq2_speed_sensitivity_dharma_temporal_autocorrelation.csv",
        "rq2_speed_sensitivity_residual_acf.csv",
    )
    if not all((tables_dir / name).is_file() for name in required):
        return False
    try:
        manifest = pd.read_csv(tables_dir / "rq2_speed_sensitivity_model_run_manifest.csv")
    except Exception:
        return False
    if "random_effects_structure_version" not in manifest.columns:
        return False
    return set(manifest["random_effects_structure_version"].dropna().astype(str)) == {RANDOM_EFFECTS_STRUCTURE_VERSION}


def write_comparison_tables(
    *,
    launch_dir: Path,
    attenuation_filter_dir: Path,
    output_dir: Path,
    threshold_id: str,
    deployment_summary: pd.DataFrame,
    static_summary: pd.DataFrame,
) -> dict[str, str | None]:
    tables_dir = output_dir / "tables"
    fixed_path = tables_dir / "rq2_speed_sensitivity_fixed_effects.csv"
    manifest_path = tables_dir / "rq2_speed_sensitivity_model_run_manifest.csv"
    if not fixed_path.is_file():
        return {"status": "skipped_no_model_fits"}

    fixed = pd.read_csv(fixed_path)
    fixed["significant_alpha_0p05"] = pd.to_numeric(fixed["p_value"], errors="coerce") < ALPHA
    fixed["term_order"] = fixed["term"].map({term: idx for idx, term in enumerate(TERM_ORDER)})
    fixed = fixed.sort_values(["speed_interval_s", "term_order"]).reset_index(drop=True)

    model_runs = pd.read_csv(manifest_path) if manifest_path.is_file() else pd.DataFrame()
    speed_effect = fixed.loc[fixed["term"] == "z_speed"].copy()
    speed_effect = speed_effect.merge(
        deployment_summary.rename(
            columns={
                "mean_speed_mps": "deployment_mean_speed_mps",
                "median_speed_mps": "deployment_median_speed_mps",
                "p95_speed_mps": "deployment_p95_speed_mps",
            }
        )[["speed_interval_s", "deployment_mean_speed_mps", "deployment_median_speed_mps", "deployment_p95_speed_mps"]],
        on="speed_interval_s",
        how="left",
    )
    speed_effect = speed_effect.merge(
        static_summary.rename(
            columns={
                "mean_speed_mps": "static_mean_apparent_speed_mps",
                "median_speed_mps": "static_median_apparent_speed_mps",
                "p95_speed_mps": "static_p95_apparent_speed_mps",
            }
        )[["speed_interval_s", "static_mean_apparent_speed_mps", "static_median_apparent_speed_mps", "static_p95_apparent_speed_mps"]],
        on="speed_interval_s",
        how="left",
    )

    effect_comparison = fixed.loc[fixed["term"].isin(EFFECT_TERMS)].copy()
    effect_comparison.to_csv(tables_dir / "rq2_speed_sensitivity_effect_comparison.csv", index=False)
    speed_effect.to_csv(tables_dir / "rq2_speed_sensitivity_speed_effect_summary.csv", index=False)

    external = external_effect_comparison(
        launch_dir=launch_dir,
        attenuation_filter_dir=attenuation_filter_dir,
        speed_effects=fixed,
        threshold_id=threshold_id,
    )
    external.to_csv(tables_dir / "rq2_speed_sensitivity_external_effect_comparison.csv", index=False)

    diagnostics = diagnostic_summary(output_dir)
    diagnostics.to_csv(tables_dir / "rq2_speed_sensitivity_diagnostic_summary.csv", index=False)

    return {
        "fixed_effects": "tables/rq2_speed_sensitivity_fixed_effects.csv",
        "model_run_manifest": "tables/rq2_speed_sensitivity_model_run_manifest.csv" if not model_runs.empty else None,
        "effect_comparison": "tables/rq2_speed_sensitivity_effect_comparison.csv",
        "speed_effect_summary": "tables/rq2_speed_sensitivity_speed_effect_summary.csv",
        "external_effect_comparison": "tables/rq2_speed_sensitivity_external_effect_comparison.csv",
        "diagnostic_summary": "tables/rq2_speed_sensitivity_diagnostic_summary.csv",
    }


def external_effect_comparison(
    *,
    launch_dir: Path,
    attenuation_filter_dir: Path,
    speed_effects: pd.DataFrame,
    threshold_id: str,
) -> pd.DataFrame:
    rows = []
    primary_path = launch_dir / "tables" / "rq2_fixed_effects.csv"
    if primary_path.is_file():
        primary = pd.read_csv(primary_path)
        primary = primary.loc[
            (primary["family"].astype(str) == "NB1")
            & (primary["fixed_surface"].astype(str) == "interaction")
            & (primary["term"].isin(EFFECT_TERMS))
        ].copy()
        primary["analysis_context"] = "primary_no_threshold_current_1s"
        primary["threshold_id"] = "none"
        primary["speed_interval_s"] = 1
        primary["speed_metric_id"] = "current_1s_speed"
        rows.append(primary)

    attenuation_path = attenuation_filter_dir / "tables" / "rq2_attenuation_fixed_effects.csv"
    if attenuation_path.is_file():
        attenuation = pd.read_csv(attenuation_path)
        attenuation = attenuation.loc[
            (attenuation["family"].astype(str) == "NB1")
            & (attenuation["fixed_surface"].astype(str) == "interaction")
            & (attenuation["term"].isin(EFFECT_TERMS))
        ].copy()
        attenuation["analysis_context"] = attenuation["threshold_id"].astype(str) + "_current_1s_existing"
        attenuation["speed_interval_s"] = 1
        attenuation["speed_metric_id"] = "current_1s_speed"
        rows.append(attenuation)

    new_effects = speed_effects.loc[speed_effects["term"].isin(EFFECT_TERMS)].copy()
    new_effects["analysis_context"] = threshold_id + "_speed_" + new_effects["speed_interval_s"].astype(int).astype(str) + "s"
    if "threshold_id" not in new_effects.columns:
        new_effects["threshold_id"] = threshold_id
    rows.append(new_effects)
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["significant_alpha_0p05"] = pd.to_numeric(out["p_value"], errors="coerce") < ALPHA
    out["term_order"] = out["term"].map({term: idx for idx, term in enumerate(TERM_ORDER)})
    context_order = {
        "primary_no_threshold_current_1s": 0,
        "attenuation_lenient_current_1s_existing": 1,
        "attenuation_primary_current_1s_existing": 2,
        "attenuation_conservative_current_1s_existing": 3,
        f"{threshold_id}_speed_1s": 4,
        f"{threshold_id}_speed_5s": 5,
        f"{threshold_id}_speed_10s": 6,
        f"{threshold_id}_speed_20s": 7,
    }
    out["analysis_context_order"] = out["analysis_context"].map(context_order).fillna(99).astype(int)
    return out.sort_values(["analysis_context_order", "term_order"]).reset_index(drop=True)


def diagnostic_summary(output_dir: Path) -> pd.DataFrame:
    standard_path = output_dir / "tables" / "rq2_speed_sensitivity_dharma_standard_tests.csv"
    temporal_path = output_dir / "tables" / "rq2_speed_sensitivity_dharma_temporal_autocorrelation.csv"
    if not standard_path.is_file():
        return pd.DataFrame()
    standard = pd.read_csv(standard_path)
    temporal = pd.read_csv(temporal_path) if temporal_path.is_file() else pd.DataFrame()
    rows = []
    for interval_s, group in standard.groupby("speed_interval_s", sort=True):
        row: dict[str, Any] = {"speed_interval_s": int(interval_s), "model_id": str(group["model_id"].iloc[0])}
        for test in ("uniformity", "dispersion", "zero_inflation", "outliers_bootstrap"):
            test_rows = group.loc[group["test"].astype(str) == test]
            row[f"{test}_p_value"] = float(test_rows["p_value"].iloc[0]) if not test_rows.empty and pd.notna(test_rows["p_value"].iloc[0]) else np.nan
            row[f"{test}_status"] = str(test_rows["status"].iloc[0]) if not test_rows.empty else ""
        if not temporal.empty:
            temp = temporal.loc[(temporal["speed_interval_s"].astype(int) == int(interval_s)) & (temporal["status"].astype(str) == "computed")]
            p = pd.to_numeric(temp["p_value"], errors="coerce").dropna()
            row["temporal_sessions_tested"] = int(len(p.index))
            row["temporal_sessions_significant_alpha_0p05"] = int((p < ALPHA).sum()) if len(p.index) else 0
            row["temporal_median_p_value"] = float(p.median()) if len(p.index) else np.nan
            row["temporal_min_p_value"] = float(p.min()) if len(p.index) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_speed_figures(
    *,
    output_dir: Path,
    model_input: pd.DataFrame,
    deployment_summary: pd.DataFrame,
    static_summary: pd.DataFrame,
) -> pd.DataFrame:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = [
        pd.to_numeric(group["speed_mps"], errors="coerce").dropna().to_numpy()
        for _, group in model_input.groupby("speed_interval_s", sort=True)
    ]
    labels = [f"{int(interval)} s" for interval in sorted(model_input["speed_interval_s"].unique())]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title("Deployment Mean Group Speed By GPS Aggregation Interval")
    ax.set_xlabel("GPS aggregation interval")
    ax.set_ylabel("Window mean group member speed (m/s)")
    ax.grid(axis="y", alpha=0.25)
    path = figures_dir / "rq2_speed_sensitivity_deployment_speed_boxplot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    rows.append({"figure": path.name, "relative_path": str(path.relative_to(output_dir)), "figure_type": "Deployment speed distribution", "description": "Window-level mean group member speed by GPS aggregation interval."})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    dep = deployment_summary.sort_values("speed_interval_s")
    stat = static_summary.sort_values("speed_interval_s")
    ax.plot(dep["speed_interval_s"], dep["median_speed_mps"], marker="o", label="Deployment median")
    ax.plot(stat["speed_interval_s"], stat["median_speed_mps"], marker="o", label="Static fence median")
    ax.set_xscale("log")
    ax.set_xticks(list(dep["speed_interval_s"]))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("GPS aggregation interval (s)")
    ax.set_ylabel("Median apparent speed (m/s)")
    ax.set_title("Speed Metric Declines With Coarser GPS Aggregation")
    ax.grid(alpha=0.25)
    ax.legend()
    path = figures_dir / "rq2_speed_sensitivity_deployment_vs_static_speed.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    rows.append({"figure": path.name, "relative_path": str(path.relative_to(output_dir)), "figure_type": "Deployment versus static speed", "description": "Median deployment group member speed and static-fence apparent speed by aggregation interval."})
    return pd.DataFrame(rows)


def write_reports(
    *,
    launch_dir: Path,
    attenuation_filter_dir: Path,
    static_gps_dir: Path,
    output_dir: Path,
    threshold_id: str,
    intervals_s: Iterable[int],
    threshold_meta: pd.DataFrame,
    recompute_audit: pd.DataFrame,
    deployment_summary: pd.DataFrame,
    static_summary: pd.DataFrame,
    comparison_outputs: dict[str, str | None],
    fit_models: bool,
    nsim: int,
    seed: int,
) -> None:
    methods = methods_markdown(
        launch_dir=launch_dir,
        attenuation_filter_dir=attenuation_filter_dir,
        static_gps_dir=static_gps_dir,
        output_dir=output_dir,
        threshold_id=threshold_id,
        intervals_s=tuple(intervals_s),
        threshold_meta=threshold_meta,
        recompute_audit=recompute_audit,
        nsim=nsim,
        seed=seed,
    )
    results = results_markdown(
        output_dir=output_dir,
        threshold_id=threshold_id,
        threshold_meta=threshold_meta,
        deployment_summary=deployment_summary,
        static_summary=static_summary,
        fit_models=fit_models,
        comparison_outputs=comparison_outputs,
    )
    readme = readme_markdown(output_dir=output_dir, threshold_id=threshold_id, intervals_s=tuple(intervals_s), fit_models=fit_models)
    (output_dir / "METHODS.md").write_text(methods, encoding="utf8")
    (output_dir / "RESULTS.md").write_text(results, encoding="utf8")
    (output_dir / "README.md").write_text(readme, encoding="utf8")
    (output_dir / "reports" / "METHODS.md").write_text(methods, encoding="utf8")
    (output_dir / "reports" / "RESULTS.md").write_text(results, encoding="utf8")


def methods_markdown(
    *,
    launch_dir: Path,
    attenuation_filter_dir: Path,
    static_gps_dir: Path,
    output_dir: Path,
    threshold_id: str,
    intervals_s: tuple[int, ...],
    threshold_meta: pd.DataFrame,
    recompute_audit: pd.DataFrame,
    nsim: int,
    seed: int,
) -> str:
    threshold = threshold_meta.iloc[0]
    audit = recompute_audit.iloc[0]
    intervals = ", ".join(f"{interval} s" for interval in intervals_s)
    command = (
        "PYTHONPATH=backend/src python -m "
        "stv_analysis.rq2_hum_timing_group_dispersion.speed_sensitivity "
        f"--launch-dir {launch_dir} "
        f"--attenuation-filter-dir {attenuation_filter_dir} "
        f"--static-gps-dir {static_gps_dir} "
        f"--output-dir {output_dir} "
        f"--threshold-id {threshold_id} "
        f"--speed-intervals {','.join(str(interval) for interval in intervals_s)} "
        f"--nsim {nsim} --seed {seed}"
    )
    return f"""# RQ2 Speed Sensitivity Methods

## Scope

This sensitivity analysis follows the retained RQ2 attenuation random-slopes protocol. It keeps the 60 s RQ2 H2 windows, the attenuation-calibrated amplitude-filtered response, the fixed-effect surface, the offset, and the maximal session-level random-slope structure fixed. Only the `speed_mps` predictor is replaced by temporally aggregated GPS speed metrics.

The response is restricted to `{threshold_id}`, the strictest attenuation-calibrated amplitude filter in the current attenuation analysis. Its cutoff is {float(threshold["cutoff_dbfs"]):.6g} dBFS on `{threshold["primary_metric"]}`, defined as the {float(threshold["external_fence_quantile"]) * 100:.1f}th percentile of known external fence-recorder hum-band amplitudes.

## Inputs

- Primary RQ2 launch: `{launch_dir}`
- Attenuation-filter analysis: `{attenuation_filter_dir}`
- Static attenuation GPS precision bundle: `{static_gps_dir}`
- Output folder: `{output_dir}`

## Speed Recalculation

Speed intervals are {intervals}. For each existing 60 s window and each collar:

1. GPS positions are grouped into window-relative bins of the requested interval.
2. Each bin is represented by median `x_m`, median `y_m`, and median observed GPS second.
3. Step speed is the Euclidean distance between consecutive binned median positions divided by elapsed time.
4. Step speeds above {BIOLOGICALLY_IMPOSSIBLE_SPEED_MPS:.1f} m/s are removed, matching the primary RQ2 speed filter.
5. Remaining step speeds are pooled across collars and averaged within the 60 s window, matching the current implementation's averaging behavior.

The 1 s recomputation is included as an audit against the current `speed_mps`. Across {int(audit["rows"])} rows, the maximum absolute difference from the existing 1 s speed was {float(audit["max_absolute_delta_mps"]):.6g} m/s.

## Static Apparent Speed

Static apparent speed uses the same median-bin step-speed algorithm on the attenuation GPS precision bundle. The primary static fence receivers are used: `fence_1m`, `fence_2m`, `fence_4m`, and `fence_8m`. The problematic `fence_0.5m` unit remains excluded, following the GPS precision bundle's own receiver-inclusion policy.

## Model

For each speed interval, the model is:

`hum_count ~ z_area * z_spread_change + z_speed + z_centroid_change + offset(log(window_length_minutes)) + (1 + z_area * z_spread_change + z_speed + z_centroid_change | session_id)`

The model family is `glmmTMB::nbinom1(link = "log")`. Predictors are standardized within each speed-interval-specific model input, as in the existing RQ2 and attenuation-filter workflows. Random effects use `{RANDOM_EFFECTS_STRUCTURE_VERSION}`. Coefficient p-values are conditional-model Wald z tests from `glmmTMB`. DHARMa residual diagnostics use {nsim} simulations per fitted model with seed {seed}.

## Reproducibility

Run from the repository root:

```bash
{command}
```

Key outputs are:

- `inputs/rq2_speed_sensitivity_model_input_60s.csv`
- `tables/rq2_speed_sensitivity_fixed_effects.csv`
- `tables/rq2_speed_sensitivity_speed_effect_summary.csv`
- `tables/rq2_speed_sensitivity_external_effect_comparison.csv`
- `RESULTS.md`
"""


def results_markdown(
    *,
    output_dir: Path,
    threshold_id: str,
    threshold_meta: pd.DataFrame,
    deployment_summary: pd.DataFrame,
    static_summary: pd.DataFrame,
    fit_models: bool,
    comparison_outputs: dict[str, str | None],
) -> str:
    threshold = threshold_meta.iloc[0]
    speed_effect_path = output_dir / "tables" / "rq2_speed_sensitivity_speed_effect_summary.csv"
    effect_path = output_dir / "tables" / "rq2_speed_sensitivity_effect_comparison.csv"
    external_path = output_dir / "tables" / "rq2_speed_sensitivity_external_effect_comparison.csv"
    diagnostic_path = output_dir / "tables" / "rq2_speed_sensitivity_diagnostic_summary.csv"
    model_manifest_path = output_dir / "tables" / "rq2_speed_sensitivity_model_run_manifest.csv"

    lines = [
        "# RQ2 Speed Sensitivity Results",
        "",
        "## Scope",
        "",
        f"This analysis uses `{threshold_id}`, the strictest attenuation-calibrated threshold: {float(threshold['external_fence_quantile']) * 100:.1f}th percentile of external fence-recorder `{threshold['primary_metric']}` values, cutoff {float(threshold['cutoff_dbfs']):.6g} dBFS.",
        "",
        "## Speed Summary",
        "",
    ]
    if speed_effect_path.is_file():
        speed_effect = pd.read_csv(speed_effect_path)
        summary = speed_effect_summary_for_report(speed_effect)
        lines.extend(markdown_table(summary))
        speed_rows = speed_effect.sort_values("speed_interval_s")
        positive = (pd.to_numeric(speed_rows["incidence_rate_ratio"], errors="coerce") > 1).all()
        significant = (pd.to_numeric(speed_rows["p_value"], errors="coerce") < ALPHA).all()
        first = speed_rows.iloc[0]
        last = speed_rows.iloc[-1]
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                (
                    "The speed coefficient remains positive across all tested GPS aggregation intervals."
                    if positive
                    else "The speed coefficient does not remain positive across all tested GPS aggregation intervals."
                ),
                (
                    "It remains significant at alpha = 0.05 across all intervals."
                    if significant
                    else "It is not significant at alpha = 0.05 across all intervals."
                ),
                f"The speed IRR changes from {float(first['incidence_rate_ratio']):.4g} at {int(first['speed_interval_s'])} s to {float(last['incidence_rate_ratio']):.4g} at {int(last['speed_interval_s'])} s.",
            ]
        )
    elif fit_models:
        lines.append("Model-fit outputs were requested but the speed-effect table was not found.")
    else:
        combined = deployment_summary.merge(static_summary, on="speed_interval_s", suffixes=("_deployment", "_static"))
        lines.extend(markdown_table(combined))
        lines.append("")
        lines.append("Model fitting was skipped, so no coefficient comparisons are available.")

    if effect_path.is_file():
        effects = pd.read_csv(effect_path)
        report = effects.loc[effects["term"].isin(EFFECT_TERMS)].copy()
        report = report[["speed_interval_s", "term", "incidence_rate_ratio", "irr_95_low", "irr_95_high", "p_value", "significant_alpha_0p05"]]
        report = format_effect_table(report)
        lines.extend(["", "## Model Effects", "", *markdown_table(report, max_rows=60)])

    if external_path.is_file():
        external = pd.read_csv(external_path)
        speed_external = external.loc[external["term"] == "z_speed"].copy()
        speed_external = speed_external[["analysis_context", "incidence_rate_ratio", "irr_95_low", "irr_95_high", "p_value", "significant_alpha_0p05"]]
        speed_external = format_effect_table(speed_external)
        lines.extend(["", "## External Context", "", "Speed effect compared with the existing no-threshold and attenuation-filter NB1 interaction results.", "", *markdown_table(speed_external, max_rows=20)])
        external_effects = external.loc[external["term"].isin(EFFECT_TERMS)].copy()
        external_effects = external_effects[["analysis_context", "term", "incidence_rate_ratio", "irr_95_low", "irr_95_high", "p_value", "significant_alpha_0p05"]]
        external_effects = format_effect_table(external_effects)
        lines.extend(["", "All fixed effects and alpha = 0.05 significance flags for the same comparison set.", "", *markdown_table(external_effects, max_rows=60)])

    if diagnostic_path.is_file():
        diagnostics = pd.read_csv(diagnostic_path)
        if not diagnostics.empty:
            diagnostics = diagnostics[["speed_interval_s", "uniformity_p_value", "dispersion_p_value", "zero_inflation_p_value", "outliers_bootstrap_p_value", "temporal_sessions_significant_alpha_0p05", "temporal_sessions_tested"]]
            diagnostics = format_numeric_columns(diagnostics)
            lines.extend(["", "## DHARMa Summary", "", *markdown_table(diagnostics)])

    if model_manifest_path.is_file():
        manifest = pd.read_csv(model_manifest_path)
        manifest = manifest[["speed_interval_s", "n_rows", "n_sessions", "hum_count_total", "zero_hum_fraction", "fit_status", "AIC", "BIC", "dispersion_parameter"]]
        manifest = format_numeric_columns(manifest)
        lines.extend(["", "## Model Fit Summary", "", *markdown_table(manifest)])

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- Model input: `inputs/rq2_speed_sensitivity_model_input_60s.csv`",
            "- Speed effect summary: `tables/rq2_speed_sensitivity_speed_effect_summary.csv`",
            "- All model effects: `tables/rq2_speed_sensitivity_effect_comparison.csv`",
            "- External effect comparison: `tables/rq2_speed_sensitivity_external_effect_comparison.csv`",
            "- DHARMa diagnostics: `tables/rq2_speed_sensitivity_diagnostic_summary.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def speed_effect_summary_for_report(speed_effect: pd.DataFrame) -> pd.DataFrame:
    report = speed_effect[
        [
            "speed_interval_s",
            "static_mean_apparent_speed_mps",
            "static_p95_apparent_speed_mps",
            "deployment_mean_speed_mps",
            "deployment_median_speed_mps",
            "incidence_rate_ratio",
            "irr_95_low",
            "irr_95_high",
            "p_value",
            "significant_alpha_0p05",
        ]
    ].copy()
    return format_effect_table(report)


def readme_markdown(*, output_dir: Path, threshold_id: str, intervals_s: tuple[int, ...], fit_models: bool) -> str:
    intervals = ", ".join(str(value) for value in intervals_s)
    return f"""# RQ2 Speed Sensitivity: Strict Attenuation Filter

This folder contains the speed sensitivity analysis for `{threshold_id}` with the NB1 interaction model.

GPS aggregation intervals: {intervals} seconds.

Models fit: {fit_models}.

Primary documents:

- `METHODS.md`
- `RESULTS.md`
- `manifest.json`

Primary tables:

- `inputs/rq2_speed_sensitivity_model_input_60s.csv`
- `tables/rq2_speed_sensitivity_speed_effect_summary.csv`
- `tables/rq2_speed_sensitivity_effect_comparison.csv`
- `tables/rq2_speed_sensitivity_external_effect_comparison.csv`
"""


def format_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ("incidence_rate_ratio", "irr_95_low", "irr_95_high"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    if "p_value" in out.columns:
        out["p_value"] = pd.to_numeric(out["p_value"], errors="coerce").map(format_p_value)
    for column in ("static_median_apparent_speed_mps", "deployment_median_speed_mps"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    for column in ("static_mean_apparent_speed_mps", "static_p95_apparent_speed_mps", "deployment_mean_speed_mps"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return out


def format_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_numeric_dtype(out[column]):
            if column.endswith("p_value") or column == "p_value":
                out[column] = pd.to_numeric(out[column], errors="coerce").map(format_p_value)
            else:
                numeric = pd.to_numeric(out[column], errors="coerce")
                nonnull = numeric.dropna()
                integer_like = bool(len(nonnull.index) and np.all(np.isclose(nonnull.to_numpy(dtype=float), np.round(nonnull.to_numpy(dtype=float)))))
                if integer_like:
                    out[column] = numeric.map(lambda value: "" if pd.isna(value) else str(int(round(float(value)))))
                else:
                    out[column] = numeric.map(lambda value: "" if pd.isna(value) else f"{float(value):.4g}")
    return out


def format_p_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    numeric = float(value)
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.4g}"


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> list[str]:
    if df.empty:
        return ["No rows."]
    out = df.head(max_rows).copy()
    out = out.astype(str)
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for row in out.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def speed_metric_id(interval_s: int) -> str:
    return "current_1s_speed" if int(interval_s) == 1 else f"median_aggregated_{int(interval_s)}s_speed"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RQ2 speed sensitivity for the strict attenuation-calibrated amplitude threshold.")
    parser.add_argument("--launch-dir", type=Path, required=True)
    parser.add_argument("--attenuation-filter-dir", type=Path, required=True)
    parser.add_argument("--static-gps-dir", type=Path, default=DEFAULT_STATIC_GPS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold-id", default=STRICT_ATTENUATION_THRESHOLD_ID)
    parser.add_argument("--speed-intervals", default=",".join(str(value) for value in SPEED_INTERVALS_SECONDS))
    parser.add_argument("--skip-fits", action="store_true")
    parser.add_argument("--nsim", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    launch_dir = args.launch_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (launch_dir / DEFAULT_OUTPUT_SUBDIR).resolve()
    intervals = tuple(int(value.strip()) for value in str(args.speed_intervals).split(",") if value.strip())
    manifest = run_speed_sensitivity_analysis(
        launch_dir=launch_dir,
        attenuation_filter_dir=args.attenuation_filter_dir.resolve(),
        static_gps_dir=args.static_gps_dir.resolve(),
        output_dir=output_dir,
        threshold_id=str(args.threshold_id),
        intervals_s=intervals,
        fit_models=not args.skip_fits,
        nsim=int(args.nsim),
        seed=int(args.seed),
    )
    print(json.dumps({"output_dir": manifest["output_dir"], "models_fit": manifest["models_fit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

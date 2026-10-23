from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import wave

import numpy as np
import pandas as pd
from scipy import signal

from .builder import hum_event_frame


PRIMARY_METRIC = "hum_band_frame_rms_p95_dbfs"
HUM_BAND_LOW_HZ = 220.0
HUM_BAND_HIGH_HZ = 5000.0
FRAME_LENGTH_S = 0.0417
FRAME_HOP_S = 0.0208
DBFS_FLOOR = 1e-12
WINDOW_LENGTH_S = 60


def load_rq2_rich_events(launch_dir: Path, intermediate_events_path: Path) -> pd.DataFrame:
    base_events_path = launch_dir / "base_snapshot" / "events.parquet"
    if not base_events_path.is_file():
        raise FileNotFoundError(f"missing launch event snapshot: {base_events_path}")
    if not intermediate_events_path.is_file():
        raise FileNotFoundError(f"missing rich intermediate events table: {intermediate_events_path}")

    base_events = pd.read_parquet(base_events_path)
    canonical = hum_event_frame(base_events)
    rich_columns = [
        "review_row_key",
        "clip_path",
        "clip_duration_s",
        "source_audio_file",
        "source_start_s",
        "source_end_s",
        "detection_score",
        "model_positive_score",
        "Low Freq (Hz)",
        "High Freq (Hz)",
    ]
    rich = pd.read_parquet(intermediate_events_path, columns=rich_columns)
    merged = canonical.merge(rich, on="review_row_key", how="left", validate="one_to_one")
    missing_clip = merged["clip_path"].isna()
    if missing_clip.any():
        examples = merged.loc[missing_clip, "review_row_key"].head(5).tolist()
        raise ValueError(f"canonical RQ2 events are missing clip paths, examples: {examples}")
    return merged.sort_values(["day_session_id", "seconds_since_midnight_local", "review_row_key"]).reset_index(drop=True)


def compute_event_amplitude_metrics(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sos_cache: dict[int, np.ndarray] = {}
    total = len(events.index)
    for index, row in enumerate(events[["review_row_key", "clip_path"]].itertuples(index=False), start=1):
        path = Path(str(row.clip_path))
        result: dict[str, Any] = {"review_row_key": str(row.review_row_key), "clip_read_status": "ok"}
        try:
            samples, sample_rate = read_wav_mono_float(path)
            result.update(amplitude_metrics(samples, sample_rate, sos_cache))
        except Exception as exc:
            result["clip_read_status"] = f"error:{type(exc).__name__}"
            result["clip_read_error"] = str(exc)
        rows.append(result)
        if index % 5000 == 0 or index == total:
            print(f"computed amplitude metrics for {index}/{total} clips", file=sys.stderr)
    return pd.DataFrame(rows)


def read_wav_mono_float(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = int(wav.getframerate())
        channels = int(wav.getnchannels())
        sample_width = int(wav.getsampwidth())
        frame_count = int(wav.getnframes())
        raw = wav.readframes(frame_count)
    data = pcm_bytes_to_float(raw, sample_width)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32, copy=False), sample_rate


def pcm_bytes_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        return int24_bytes_to_float(raw)
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported WAV sample width: {sample_width}")


def int24_bytes_to_float(raw: bytes) -> np.ndarray:
    bytes_view = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    ints = (
        bytes_view[:, 0].astype(np.int32)
        | (bytes_view[:, 1].astype(np.int32) << 8)
        | (bytes_view[:, 2].astype(np.int32) << 16)
    )
    sign_bit = 1 << 23
    ints = (ints ^ sign_bit) - sign_bit
    return ints.astype(np.float32) / float(1 << 23)


def amplitude_metrics(samples: np.ndarray, sample_rate: int, sos_cache: dict[int, np.ndarray] | None = None) -> dict[str, Any]:
    if samples.size == 0:
        return {
            "sample_rate_hz": int(sample_rate),
            "clip_sample_count": 0,
            "clip_duration_from_samples_s": 0.0,
        }

    y = np.asarray(samples, dtype=np.float32)
    abs_y = np.abs(y)
    frame = frame_rms_values(y, sample_rate)
    sos = hum_band_sos(sample_rate, sos_cache)
    hum_band = signal.sosfilt(sos, y).astype(np.float32, copy=False)
    abs_band = np.abs(hum_band)
    band_frame = frame_rms_values(hum_band, sample_rate)

    rms = rms_amplitude(y)
    band_rms = rms_amplitude(hum_band)
    peak = float(np.max(abs_y))
    band_peak = float(np.max(abs_band))
    frame_p95 = percentile_or_nan(frame, 95)
    band_frame_p95 = percentile_or_nan(band_frame, 95)
    return {
        "sample_rate_hz": int(sample_rate),
        "clip_sample_count": int(samples.size),
        "clip_duration_from_samples_s": float(samples.size / sample_rate),
        "rms_linear": rms,
        "rms_dbfs": dbfs(rms),
        "peak_linear": peak,
        "peak_dbfs": dbfs(peak),
        "abs_p95_linear": percentile_or_nan(abs_y, 95),
        "abs_p95_dbfs": dbfs(percentile_or_nan(abs_y, 95)),
        "frame_rms_p95_linear": frame_p95,
        "frame_rms_p95_dbfs": dbfs(frame_p95),
        "frame_rms_max_linear": float(np.max(frame)) if frame.size else np.nan,
        "frame_rms_max_dbfs": dbfs(float(np.max(frame))) if frame.size else np.nan,
        "hum_band_rms_linear": band_rms,
        "hum_band_rms_dbfs": dbfs(band_rms),
        "hum_band_peak_linear": band_peak,
        "hum_band_peak_dbfs": dbfs(band_peak),
        "hum_band_frame_rms_p95_linear": band_frame_p95,
        "hum_band_frame_rms_p95_dbfs": dbfs(band_frame_p95),
        "hum_band_frame_rms_max_linear": float(np.max(band_frame)) if band_frame.size else np.nan,
        "hum_band_frame_rms_max_dbfs": dbfs(float(np.max(band_frame))) if band_frame.size else np.nan,
        "crest_factor_db": dbfs(peak) - dbfs(rms),
        "clipped_sample_fraction": float(np.mean(abs_y >= 0.999)),
        "zero_sample_fraction": float(np.mean(abs_y == 0.0)),
    }


def hum_band_sos(sample_rate: int, sos_cache: dict[int, np.ndarray] | None) -> np.ndarray:
    cache = sos_cache if sos_cache is not None else {}
    if sample_rate in cache:
        return cache[sample_rate]
    high = min(HUM_BAND_HIGH_HZ, sample_rate / 2.0 * 0.95)
    if HUM_BAND_LOW_HZ >= high:
        raise ValueError(f"invalid hum band for sample rate {sample_rate}")
    sos = signal.butter(4, [HUM_BAND_LOW_HZ, high], btype="bandpass", fs=sample_rate, output="sos")
    cache[sample_rate] = sos
    return sos


def frame_rms_values(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(1, int(round(FRAME_LENGTH_S * sample_rate)))
    hop = max(1, int(round(FRAME_HOP_S * sample_rate)))
    y = np.asarray(samples, dtype=np.float32)
    if y.size == 0:
        return np.array([], dtype=np.float32)
    if y.size <= frame:
        return np.array([rms_amplitude(y)], dtype=np.float32)
    power = np.square(y, dtype=np.float64)
    cumsum = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    starts = np.arange(0, y.size - frame + 1, hop)
    means = (cumsum[starts + frame] - cumsum[starts]) / frame
    return np.sqrt(np.maximum(means, 0.0)).astype(np.float32)


def rms_amplitude(samples: np.ndarray) -> float:
    if samples.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return np.nan
    return float(np.nanpercentile(values, percentile))


def dbfs(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(20.0 * np.log10(max(float(value), DBFS_FLOOR)))


def ensure_window_assignment(events: pd.DataFrame, launch_dir: Path) -> pd.DataFrame:
    if "inside_existing_rq2_window" in events.columns and "rq2_window_id" in events.columns:
        return events.copy()
    return assign_events_to_windows(events, launch_dir)


def assign_events_to_windows(events: pd.DataFrame, launch_dir: Path) -> pd.DataFrame:
    windows_path = launch_dir / "tables" / "rq2_model_input.parquet"
    if not windows_path.is_file():
        raise FileNotFoundError(f"missing RQ2 model input windows: {windows_path}")
    windows = pd.read_parquet(
        windows_path,
        columns=[
            "rq2_window_id",
            "day_session_id",
            "window_start_seconds_local",
            "window_end_seconds_local",
            "area_m2",
            "spread_change_m2_per_min",
            "speed_mps",
            "centroid_change_in_position_m",
            "hum_count",
        ],
    )
    out = events.copy()
    starts = pd.to_datetime(out["timestamp_start_local"])
    ends = pd.to_datetime(out["timestamp_end_local"])
    out["event_midpoint_seconds_local"] = out["seconds_since_midnight_local"].astype(float) + (
        (ends - starts).dt.total_seconds().astype(float) / 2.0
    )
    out["rq2_window_id"] = None
    out["inside_existing_rq2_window"] = False

    window_groups = {
        str(session): group.sort_values("window_start_seconds_local").reset_index(drop=True)
        for session, group in windows.groupby("day_session_id", sort=False)
    }
    for session, index_values in out.groupby(out["day_session_id"].astype(str), sort=False).groups.items():
        group_windows = window_groups.get(str(session))
        if group_windows is None or group_windows.empty:
            continue
        starts_s = group_windows["window_start_seconds_local"].astype(float).to_numpy()
        ends_s = group_windows["window_end_seconds_local"].astype(float).to_numpy()
        window_ids = group_windows["rq2_window_id"].astype(str).to_numpy()
        event_seconds = out.loc[index_values, "event_midpoint_seconds_local"].astype(float).to_numpy()
        positions = np.searchsorted(starts_s, event_seconds, side="right") - 1
        valid = (positions >= 0) & (event_seconds < ends_s[np.clip(positions, 0, len(ends_s) - 1)])
        valid_indices = np.asarray(list(index_values))[valid]
        out.loc[valid_indices, "rq2_window_id"] = window_ids[positions[valid]]
        out.loc[valid_indices, "inside_existing_rq2_window"] = True

    return out.merge(windows, on=["rq2_window_id", "day_session_id"], how="left", suffixes=("", "_window"))


def load_primary_models(launch_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = pd.read_csv(launch_dir / "tables" / "rq2_model_run_manifest.csv")
    fixed = pd.read_csv(launch_dir / "tables" / "rq2_fixed_effects.csv")
    run["source"] = "primary_unthresholded"
    run["threshold_id"] = "unthresholded"
    fixed["source"] = "primary_unthresholded"
    fixed["threshold_id"] = "unthresholded"
    return run, fixed


def compare_to_primary(thresholded_fixed: pd.DataFrame, primary_fixed: pd.DataFrame) -> pd.DataFrame:
    baseline = primary_fixed[
        ["family", "fixed_surface", "term", "estimate", "p_value", "incidence_rate_ratio", "irr_95_low", "irr_95_high"]
    ].rename(
        columns={
            "estimate": "primary_estimate",
            "p_value": "primary_p_value",
            "incidence_rate_ratio": "primary_irr",
            "irr_95_low": "primary_irr_95_low",
            "irr_95_high": "primary_irr_95_high",
        }
    )
    rows = thresholded_fixed.merge(baseline, on=["family", "fixed_surface", "term"], how="left")
    rows["estimate_delta_vs_primary"] = rows["estimate"] - rows["primary_estimate"]
    rows["p_value_delta_vs_primary"] = rows["p_value"] - rows["primary_p_value"]
    rows["irr_ratio_vs_primary"] = rows["incidence_rate_ratio"] / rows["primary_irr"]
    rows["primary_significant_0_05"] = rows["primary_p_value"] < 0.05
    rows["thresholded_significant_0_05"] = rows["p_value"] < 0.05
    rows["direction_changed_vs_primary"] = np.sign(rows["estimate"]) != np.sign(rows["primary_estimate"])
    return rows

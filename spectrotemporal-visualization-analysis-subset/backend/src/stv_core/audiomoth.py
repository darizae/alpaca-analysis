from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass(frozen=True)
class AudioMothCsvSummary:
    csv_path: str
    recording_start_time: str | None
    recording_end_time: str | None
    total_samples: int | None
    sample_rate_estimate: float | None
    gps_valid_fraction: float
    first_valid_latitude: float | None
    first_valid_longitude: float | None
    last_valid_latitude: float | None
    last_valid_longitude: float | None


def _to_signed_coordinate(deg: object, minutes: object, direction: object) -> float | None:
    if pd.isna(deg) or pd.isna(minutes) or pd.isna(direction):
        return None
    try:
        degrees = float(deg)
        mins = float(minutes)
    except (TypeError, ValueError):
        return None
    value = degrees + (mins / 60.0)
    direction_text = str(direction).strip().upper()
    if direction_text in {"S", "W"}:
        value *= -1.0
    return value


def load_audiomoth_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["AUDIOMOTH_TIME"] = pd.to_datetime(df["AUDIOMOTH_TIME"], errors="coerce", utc=True, format="mixed")
    for column in ("SAMPLES", "TOTAL_SAMPLES", "TIMER_COUNT", "BUFFERS_FILLED", "BUFFERS_WRITTEN"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["latitude"] = [
        _to_signed_coordinate(deg, mins, direction)
        for deg, mins, direction in zip(df.get("LAT_DEG"), df.get("LAT_MIN"), df.get("LAT_DIR"), strict=False)
    ]
    df["longitude"] = [
        _to_signed_coordinate(deg, mins, direction)
        for deg, mins, direction in zip(df.get("LONG_DEG"), df.get("LONG_MIN"), df.get("LONG_DIR"), strict=False)
    ]
    if "LAST_RMC_AUDIOMOTH_TIME" in df.columns:
        df["LAST_RMC_AUDIOMOTH_TIME"] = pd.to_datetime(
            df["LAST_RMC_AUDIOMOTH_TIME"], errors="coerce", utc=True, format="mixed"
        )
    if "LAST_RMC_GPS_TIME" in df.columns:
        df["LAST_RMC_GPS_TIME"] = pd.to_datetime(df["LAST_RMC_GPS_TIME"], errors="coerce", utc=True, format="mixed")
    return df


def summarize_audiomoth_csv(csv_path: Path) -> tuple[pd.DataFrame, AudioMothCsvSummary]:
    df = load_audiomoth_csv(csv_path)
    recording_start_time = df["AUDIOMOTH_TIME"].dropna().min()
    total_samples_series = df["TOTAL_SAMPLES"].dropna() if "TOTAL_SAMPLES" in df.columns else pd.Series(dtype="float64")
    total_samples = int(total_samples_series.max()) if not total_samples_series.empty else None

    sample_rate_estimate = None
    if df["AUDIOMOTH_TIME"].notna().any() and "TOTAL_SAMPLES" in df.columns:
        time_seconds = df["AUDIOMOTH_TIME"].astype("int64") / 1_000_000_000
        deltas = df["TOTAL_SAMPLES"].diff() / time_seconds.diff()
        deltas = deltas.dropna()
        deltas = deltas[(deltas > 1000) & (deltas < 200000)]
        if not deltas.empty:
            sample_rate_estimate = float(deltas.median())

    recording_end_time = None
    if recording_start_time is not None and sample_rate_estimate and total_samples is not None:
        recording_end_time = recording_start_time + pd.to_timedelta(total_samples / sample_rate_estimate, unit="s")
    else:
        candidate_end = df["AUDIOMOTH_TIME"].dropna().max()
        if pd.notna(candidate_end):
            recording_end_time = candidate_end

    status_series = df["STATUS"] if "STATUS" in df.columns else pd.Series([""] * len(df.index), index=df.index)
    valid_gps = df[status_series.fillna("").astype(str).str.upper() == "A"].copy()
    valid_gps = valid_gps[valid_gps["latitude"].notna() & valid_gps["longitude"].notna()]
    gps_valid_fraction = float(len(valid_gps.index) / len(df.index)) if len(df.index) else 0.0
    first_valid = valid_gps.iloc[0] if not valid_gps.empty else None
    last_valid = valid_gps.iloc[-1] if not valid_gps.empty else None
    summary = AudioMothCsvSummary(
        csv_path=str(csv_path),
        recording_start_time=recording_start_time.isoformat() if pd.notna(recording_start_time) else None,
        recording_end_time=recording_end_time.isoformat() if recording_end_time is not None and pd.notna(recording_end_time) else None,
        total_samples=total_samples,
        sample_rate_estimate=sample_rate_estimate,
        gps_valid_fraction=gps_valid_fraction,
        first_valid_latitude=float(first_valid["latitude"]) if first_valid is not None else None,
        first_valid_longitude=float(first_valid["longitude"]) if first_valid is not None else None,
        last_valid_latitude=float(last_valid["latitude"]) if last_valid is not None else None,
        last_valid_longitude=float(last_valid["longitude"]) if last_valid is not None else None,
    )
    return df, summary


def valid_trace_points(csv_df: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    zone = ZoneInfo(timezone_name)
    status_series = csv_df["STATUS"] if "STATUS" in csv_df.columns else pd.Series([""] * len(csv_df.index), index=csv_df.index)
    points = csv_df[
        status_series.fillna("").astype(str).str.upper().eq("A")
        & csv_df["AUDIOMOTH_TIME"].notna()
        & csv_df["latitude"].notna()
        & csv_df["longitude"].notna()
    ].copy()
    points = points.sort_values("AUDIOMOTH_TIME").reset_index(drop=True)
    if points.empty:
        return points

    points["timestamp_utc"] = points["AUDIOMOTH_TIME"]
    points["timestamp_local"] = points["timestamp_utc"].dt.tz_convert(zone)
    points["date_local"] = points["timestamp_local"].dt.strftime("%Y-%m-%d")
    points["seconds_since_midnight_local"] = (
        (points["timestamp_local"].dt.hour * 3600)
        + (points["timestamp_local"].dt.minute * 60)
        + points["timestamp_local"].dt.second
        + (points["timestamp_local"].dt.microsecond / 1_000_000)
    )
    return points

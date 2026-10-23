from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    df["AUDIOMOTH_TIME"] = pd.to_datetime(df["AUDIOMOTH_TIME"], errors="coerce", utc=True)
    for column in ("SAMPLES", "TOTAL_SAMPLES", "TIMER_COUNT", "BUFFERS_FILLED", "BUFFERS_WRITTEN"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["latitude"] = [
        _to_signed_coordinate(deg, mins, direction)
        for deg, mins, direction in zip(df.get("LAT_DEG"), df.get("LAT_MIN"), df.get("LAT_DIR"), strict=False)
    ]
    df["longitude"] = [
        _to_signed_coordinate(deg, mins, direction)
        for deg, mins, direction in zip(df.get("LONG_DEG"), df.get("LONG_MIN"), df.get("LONG_DIR"), strict=False)
    ]
    df["LAST_RMC_AUDIOMOTH_TIME"] = pd.to_datetime(df.get("LAST_RMC_AUDIOMOTH_TIME"), errors="coerce", utc=True)
    df["LAST_RMC_GPS_TIME"] = pd.to_datetime(df.get("LAST_RMC_GPS_TIME"), errors="coerce", utc=True)
    return df



def summarize_audiomoth_csv(csv_path: Path) -> tuple[pd.DataFrame, AudioMothCsvSummary]:
    df = load_audiomoth_csv(csv_path)
    recording_start_time = df["AUDIOMOTH_TIME"].dropna().min()
    total_samples_series = df["TOTAL_SAMPLES"].dropna()
    total_samples = int(total_samples_series.max()) if not total_samples_series.empty else None

    sample_rate_estimate = None
    time_seconds = (
        df["AUDIOMOTH_TIME"].astype("int64") / 1_000_000_000
        if df["AUDIOMOTH_TIME"].notna().any()
        else pd.Series(dtype=float)
    )
    deltas = df["TOTAL_SAMPLES"].diff() / time_seconds.diff() if not time_seconds.empty else pd.Series(dtype=float)
    deltas = deltas.replace([pd.NA, pd.NaT], pd.NA).dropna()
    deltas = deltas[(deltas > 1000) & (deltas < 200000)]
    if not deltas.empty:
        sample_rate_estimate = float(deltas.median())
    elif recording_start_time is not None and total_samples is not None:
        recording_end_time_guess = df["AUDIOMOTH_TIME"].dropna().max()
        duration_s = (recording_end_time_guess - recording_start_time).total_seconds() if pd.notna(recording_end_time_guess) else None
        if duration_s and duration_s > 0:
            sample_rate_estimate = float(total_samples / duration_s)

    recording_end_time = None
    if recording_start_time is not None and sample_rate_estimate and total_samples is not None:
        recording_end_time = recording_start_time + pd.to_timedelta(total_samples / sample_rate_estimate, unit="s")
    else:
        candidate_end = df["AUDIOMOTH_TIME"].dropna().max()
        if pd.notna(candidate_end):
            recording_end_time = candidate_end

    status_series = df["STATUS"] if "STATUS" in df.columns else pd.Series([""] * len(df.index), index=df.index)
    valid_gps = df[status_series.fillna("").astype(str).str.upper() == "A"]
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
        first_valid_latitude=float(first_valid["latitude"]) if first_valid is not None and pd.notna(first_valid["latitude"]) else None,
        first_valid_longitude=float(first_valid["longitude"]) if first_valid is not None and pd.notna(first_valid["longitude"]) else None,
        last_valid_latitude=float(last_valid["latitude"]) if last_valid is not None and pd.notna(last_valid["latitude"]) else None,
        last_valid_longitude=float(last_valid["longitude"]) if last_valid is not None and pd.notna(last_valid["longitude"]) else None,
    )
    return df, summary



def annotate_detections_with_csv(
    detections_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    summary: AudioMothCsvSummary,
) -> pd.DataFrame:
    if detections_df.empty:
        return detections_df.copy()

    df = detections_df.copy()
    recording_start_time = pd.to_datetime(summary.recording_start_time, utc=True) if summary.recording_start_time else pd.NaT
    df["absolute_detection_start_time"] = pd.NaT
    df["absolute_detection_end_time"] = pd.NaT
    if pd.notna(recording_start_time):
        df["absolute_detection_start_time"] = recording_start_time + pd.to_timedelta(df["source_start_s"], unit="s")
        df["absolute_detection_end_time"] = recording_start_time + pd.to_timedelta(df["source_end_s"], unit="s")

    csv_df = csv_df.copy()
    csv_df = csv_df.sort_values("AUDIOMOTH_TIME").reset_index(drop=True)
    detection_rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        target_time = row["absolute_detection_start_time"]
        if pd.isna(target_time) or csv_df["AUDIOMOTH_TIME"].isna().all():
            detection_rows.append(
                {
                    "gps_status_near_detection": None,
                    "latitude_near_detection": None,
                    "longitude_near_detection": None,
                    "timer_count_near_detection": None,
                    "buffers_filled_near_detection": None,
                    "buffers_written_near_detection": None,
                    "last_rmc_audiomoth_time_near_detection": None,
                    "last_rmc_gps_time_near_detection": None,
                }
            )
            continue
        deltas = (csv_df["AUDIOMOTH_TIME"] - target_time).abs()
        idx = int(deltas.idxmin())
        match = csv_df.loc[idx]
        detection_rows.append(
            {
                "gps_status_near_detection": None if pd.isna(match.get("STATUS")) else str(match.get("STATUS")),
                "latitude_near_detection": None if pd.isna(match.get("latitude")) else float(match["latitude"]),
                "longitude_near_detection": None if pd.isna(match.get("longitude")) else float(match["longitude"]),
                "timer_count_near_detection": None if pd.isna(match.get("TIMER_COUNT")) else int(match["TIMER_COUNT"]),
                "buffers_filled_near_detection": None if pd.isna(match.get("BUFFERS_FILLED")) else int(match["BUFFERS_FILLED"]),
                "buffers_written_near_detection": None if pd.isna(match.get("BUFFERS_WRITTEN")) else int(match["BUFFERS_WRITTEN"]),
                "last_rmc_audiomoth_time_near_detection": None
                if pd.isna(match.get("LAST_RMC_AUDIOMOTH_TIME"))
                else pd.Timestamp(match["LAST_RMC_AUDIOMOTH_TIME"]).isoformat(),
                "last_rmc_gps_time_near_detection": None
                if pd.isna(match.get("LAST_RMC_GPS_TIME"))
                else pd.Timestamp(match["LAST_RMC_GPS_TIME"]).isoformat(),
            }
        )

    annotation_df = pd.DataFrame(detection_rows, index=df.index)
    df = pd.concat([df, annotation_df], axis=1)
    df["recording_start_time"] = summary.recording_start_time
    df["recording_end_time"] = summary.recording_end_time
    absolute_start = pd.to_datetime(df["absolute_detection_start_time"], utc=True, errors="coerce")
    df["detection_date"] = absolute_start.dt.strftime("%Y-%m-%d")
    df["detection_time_of_day"] = absolute_start.dt.strftime("%H:%M:%S.%f").str.rstrip("0").str.rstrip(".")
    df.loc[absolute_start.isna(), "detection_date"] = None
    df.loc[absolute_start.isna(), "detection_time_of_day"] = None
    df["detection_hour"] = absolute_start.dt.hour.astype("Int64")
    df["detection_seconds_since_midnight"] = (
        (absolute_start.dt.hour * 3600)
        + (absolute_start.dt.minute * 60)
        + absolute_start.dt.second
        + (absolute_start.dt.microsecond / 1_000_000)
    )
    df.loc[absolute_start.isna(), "detection_seconds_since_midnight"] = None
    df["timestamp_source"] = None
    df.loc[absolute_start.notna(), "timestamp_source"] = "audiomoth_csv"
    df["timestamp_timezone"] = None
    df.loc[absolute_start.notna(), "timestamp_timezone"] = "unknown_or_utc_or_local_assumed"
    df["sample_rate_estimate"] = summary.sample_rate_estimate
    df["total_samples"] = summary.total_samples
    df["gps_valid_fraction"] = summary.gps_valid_fraction
    return df

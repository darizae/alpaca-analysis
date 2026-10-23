from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
from typing import Any
import wave
from zoneinfo import ZoneInfo

import pandas as pd

from stv_core.audiomoth import summarize_audiomoth_csv, valid_trace_points
from stv_core.geometry import buffered_bbox_polygon, compute_bbox, hull_geojson

from .config import BackendConfig
from .source_index import sidecar_cache_path


@dataclass(frozen=True)
class BuildResult:
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class JoinedWavPart:
    path: Path
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class JoinedWavParts:
    manifest_path: Path
    params: Any
    frame_rate: int
    total_frames: int
    parts: tuple[JoinedWavPart, ...]


CONTRACT_VERSION = 5
SECONDS_PER_DAY = 86_399.999
SESSION_MERGE_GAP_SECONDS = 300.0
MOVING_SPEED_THRESHOLD_MPS = 0.03
HIGH_HUM_BIN_SECONDS = 900.0
HIGH_HUM_TOP_N = 5
COMPANY_SAMPLE_SECONDS = 60.0
COMPANY_DISTANCE_METERS = 40.0
DUPLICATE_HUM_WINDOW_SECONDS = 2.0
POLYGON_DISTANCE_METERS = {
    "min": 10,
    "max": 150,
    "step": 5,
    "default": 40,
    "low_zoom_min": 16,
}


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _read_cleaned_detections(raw_corpus_dir: Path) -> pd.DataFrame:
    path = raw_corpus_dir / "metadata_backfill" / "cleaned_enriched_detections.csv"
    df = pd.read_csv(path)
    _require_columns(
        df,
        [
            "review_row_key",
            "audio_collection_name",
            "device_id",
            "source_audio_file_normalized",
            "source_audio_file_original",
            "wav_basename",
            "source_start_s",
            "source_end_s",
            "detection_score",
            "reviewed_resolved_label_semantic",
            "absolute_detection_start_time",
            "absolute_detection_end_time",
            "latitude_near_detection",
            "longitude_near_detection",
        ],
        str(path),
    )
    return df


def _read_source_index(raw_corpus_dir: Path) -> pd.DataFrame:
    path = raw_corpus_dir / "metadata_backfill" / "source_wav_metadata_index.csv"
    df = pd.read_csv(path)
    _require_columns(
        df,
        [
            "source_audio_file_original",
            "source_audio_file_normalized",
            "audio_collection_name",
            "wav_basename",
            "csv_hpc_path",
            "recording_start_time",
            "recording_end_time",
            "total_samples",
            "sample_rate_estimate",
            "gps_valid_fraction",
            "first_valid_latitude",
            "first_valid_longitude",
            "last_valid_latitude",
            "last_valid_longitude",
            "device_id",
        ],
        str(path),
    )
    return df


def _read_export_manifest(raw_corpus_dir: Path) -> dict[str, Any]:
    path = raw_corpus_dir / "cleaned_alpaca_hum_corpus" / "manifest.json"
    return json.loads(path.read_text(encoding="utf8"))


def _read_selection_table(raw_corpus_dir: Path) -> pd.DataFrame:
    path = raw_corpus_dir / "cleaned_alpaca_hum_corpus" / "review_selection_table.txt"
    df = pd.read_csv(path, sep="\t")
    rename_map = {
        "Selection": "export_selection",
        "Begin Time (s)": "export_begin_time_s",
        "End Time (s)": "export_end_time_s",
    }
    df = df.rename(columns=rename_map)
    _require_columns(
        df,
        [
            "review_row_key",
            "export_selection",
            "export_begin_time_s",
            "export_end_time_s",
        ],
        str(path),
    )
    deduped = df.sort_values("export_selection").drop_duplicates(subset=["review_row_key"], keep="first")
    return deduped


def _cleaned_corpus_dir(raw_corpus_dir: Path) -> Path:
    return raw_corpus_dir / "cleaned_alpaca_hum_corpus"


def _load_cleaned_corpus_wav_parts(raw_corpus_dir: Path) -> JoinedWavParts:
    corpus_dir = _cleaned_corpus_dir(raw_corpus_dir)
    manifest_path = corpus_dir / "review_concat_parts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    parts: list[JoinedWavPart] = []
    for item in manifest["parts"]:
        part_path = corpus_dir / str(item["file"])
        if not part_path.is_file():
            raise FileNotFoundError(f"cleaned corpus WAV part not found: {part_path}")
        parts.append(
            JoinedWavPart(
                path=part_path,
                start_frame=int(item["start_frame_in_joined_wav"]),
                end_frame=int(item["end_frame_in_joined_wav"]),
            )
        )
    if not parts:
        raise ValueError(f"no WAV parts listed in {manifest_path}")

    with wave.open(str(parts[0].path), "rb") as handle:
        params = handle.getparams()
        frame_rate = handle.getframerate()

    return JoinedWavParts(
        manifest_path=manifest_path,
        params=params,
        frame_rate=frame_rate,
        total_frames=int(manifest["total_frames"]),
        parts=tuple(parts),
    )


def _read_cleaned_corpus_frames(audio: JoinedWavParts, start_frame: int, end_frame: int) -> bytes:
    bytes_per_frame = audio.params.nchannels * audio.params.sampwidth
    chunks: list[bytes] = []
    for part in audio.parts:
        overlap_start = max(start_frame, part.start_frame)
        overlap_end = min(end_frame, part.end_frame)
        if overlap_end <= overlap_start:
            continue
        with wave.open(str(part.path), "rb") as source:
            source.setpos(overlap_start - part.start_frame)
            chunks.append(source.readframes(overlap_end - overlap_start))

    frames_read = sum(len(chunk) // bytes_per_frame for chunk in chunks)
    expected_frames = end_frame - start_frame
    if frames_read != expected_frames:
        raise ValueError(f"read {frames_read} corpus frames, expected {expected_frames}")
    return b"".join(chunks)


def _parse_timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, format="mixed", errors="coerce")


def _seconds_since_midnight(ts: pd.Series) -> pd.Series:
    return (
        (ts.dt.hour * 3600)
        + (ts.dt.minute * 60)
        + ts.dt.second
        + (ts.dt.microsecond / 1_000_000)
    )


def _safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _safe_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _safe_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _clip_seconds(value: float) -> float:
    return max(0.0, min(float(value), SECONDS_PER_DAY))


def _local_seconds(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    return (
        timestamp.hour * 3600
        + timestamp.minute * 60
        + timestamp.second
        + timestamp.microsecond / 1_000_000
        + timestamp.nanosecond / 1_000_000_000
    )


def _window_payload(session_id: str, start_s: float, end_s: float, source_count: int = 0) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "start_s": round(_clip_seconds(start_s), 3),
        "end_s": round(_clip_seconds(end_s), 3),
        "duration_s": round(max(0.0, end_s - start_s), 3),
        "source_count": int(source_count),
    }


def _dead_windows(alive_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    cursor = 0.0
    for index, window in enumerate(alive_windows, start=1):
        start_s = float(window["start_s"])
        if start_s > cursor:
            windows.append(_window_payload(f"dead-{index:03d}", cursor, start_s))
        cursor = max(cursor, float(window["end_s"]))
    if cursor < SECONDS_PER_DAY:
        windows.append(_window_payload(f"dead-{len(windows) + 1:03d}", cursor, SECONDS_PER_DAY))
    return windows


def _merge_windows(
    rows: list[tuple[float, float]],
    *,
    date_local: str,
    recorder_id: str | None = None,
) -> list[dict[str, Any]]:
    clean_rows = sorted(
        (max(0.0, start), min(SECONDS_PER_DAY, end))
        for start, end in rows
        if pd.notna(start) and pd.notna(end) and end > start
    )
    merged: list[dict[str, Any]] = []
    for start_s, end_s in clean_rows:
        if not merged or start_s > float(merged[-1]["end_s"]) + SESSION_MERGE_GAP_SECONDS:
            prefix = f"{date_local}:{recorder_id}" if recorder_id else date_local
            merged.append(_window_payload(f"{prefix}:session-{len(merged) + 1:03d}", start_s, end_s, 1))
        else:
            merged[-1]["end_s"] = round(max(float(merged[-1]["end_s"]), end_s), 3)
            merged[-1]["duration_s"] = round(float(merged[-1]["end_s"]) - float(merged[-1]["start_s"]), 3)
            merged[-1]["source_count"] = int(merged[-1]["source_count"]) + 1
    return merged


def _find_window(windows: list[dict[str, Any]], seconds: float | None) -> dict[str, Any] | None:
    if seconds is None or pd.isna(seconds):
        return None
    for window in windows:
        if float(window["start_s"]) <= float(seconds) <= float(window["end_s"]):
            return window
    return None


def _availability_payload(
    segments: pd.DataFrame,
    *,
    date_local: str,
    recorder_id: str | None = None,
) -> dict[str, Any]:
    rows = [
        (float(row.recording_start_seconds_local), float(row.recording_end_seconds_local))
        for row in segments.itertuples(index=False)
    ]
    alive_windows = _merge_windows(rows, date_local=date_local, recorder_id=recorder_id)
    return {
        "date_local": date_local,
        "timezone": segments["timezone"].dropna().astype(str).iloc[0] if "timezone" in segments and not segments.empty else None,
        "alive_windows": alive_windows,
        "dead_windows": _dead_windows(alive_windows),
    }


def _distance_meters(left: tuple[float, float], right: tuple[float, float]) -> float:
    from math import cos, hypot, pi

    mean_latitude = (left[1] + right[1]) / 2.0
    lat_scale = 111_320.0
    lon_scale = max(1e-9, lat_scale * cos(mean_latitude * pi / 180.0))
    return hypot((right[0] - left[0]) * lon_scale, (right[1] - left[1]) * lat_scale)


def _clip_key(event_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", event_id).strip("_") or "event"


def _load_gps_blocker(config: BackendConfig) -> list[tuple[float, float]]:
    raw = json.loads(config.gps_blocker_path.read_text(encoding="utf8"))
    points = [(float(lon), float(lat)) for lon, lat in raw["points"]]
    return buffered_bbox_polygon(points, float(raw["buffer_meters"]))


def _to_iso(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _iso_series(series: pd.Series) -> pd.Series:
    return series.map(_to_iso)


def _stable_segment_id(recorder_id: str, wav_basename: str) -> str:
    return f"{recorder_id}:{wav_basename}"


def _display_name(recorder_id: str) -> str:
    stem = recorder_id.removeprefix("audio_collection_")
    return stem.split("_", 1)[0]


def _prepare_events(config: BackendConfig, selection_df: pd.DataFrame) -> pd.DataFrame:
    events = _read_cleaned_detections(config.raw_corpus_dir).copy()
    zone = ZoneInfo(config.timezone_name)

    numeric_columns = ["source_start_s", "source_end_s", "detection_score", "latitude_near_detection", "longitude_near_detection"]
    for column in numeric_columns:
        events[column] = pd.to_numeric(events[column], errors="coerce")

    events["timestamp_start_utc"] = _parse_timestamps(events["absolute_detection_start_time"])
    events["timestamp_end_utc"] = _parse_timestamps(events["absolute_detection_end_time"])
    events["timestamp_start_local"] = events["timestamp_start_utc"].dt.tz_convert(zone)
    events["timestamp_end_local"] = events["timestamp_end_utc"].dt.tz_convert(zone)
    events["date_local"] = events["timestamp_start_local"].dt.strftime("%Y-%m-%d")
    events["seconds_since_midnight_local"] = _seconds_since_midnight(events["timestamp_start_local"])
    events["recorder_id"] = events["audio_collection_name"].astype(str)
    events["recorder_display_name"] = events["recorder_id"].map(_display_name)
    events["segment_id"] = [
        _stable_segment_id(recorder_id, wav_basename)
        for recorder_id, wav_basename in zip(events["recorder_id"], events["wav_basename"], strict=False)
    ]

    events = events.merge(selection_df, on="review_row_key", how="left", suffixes=("", "_selection"))
    missing_audio = events["export_selection"].isna() | events["export_begin_time_s"].isna() | events["export_end_time_s"].isna()
    if missing_audio.any():
        missing_keys = events.loc[missing_audio, "review_row_key"].head(5).tolist()
        raise ValueError(f"selection table join missing export audio fields for rows such as: {missing_keys}")

    return events


def _prepare_segments(config: BackendConfig) -> pd.DataFrame:
    segments = _read_source_index(config.raw_corpus_dir).copy()
    zone = ZoneInfo(config.timezone_name)

    segments["recording_start_utc"] = _parse_timestamps(segments["recording_start_time"])
    segments["recording_end_utc"] = _parse_timestamps(segments["recording_end_time"])
    segments["recording_start_local"] = segments["recording_start_utc"].dt.tz_convert(zone)
    segments["recording_end_local"] = segments["recording_end_utc"].dt.tz_convert(zone)
    segments["date_local"] = segments["recording_start_local"].dt.strftime("%Y-%m-%d")
    segments["timezone"] = config.timezone_name
    segments["recording_start_seconds_local"] = segments["recording_start_local"].map(_local_seconds)
    segments["recording_end_seconds_local"] = segments["recording_end_local"].map(_local_seconds)
    segments["segment_id"] = [
        _stable_segment_id(recorder_id, wav_basename)
        for recorder_id, wav_basename in zip(segments["audio_collection_name"], segments["wav_basename"], strict=False)
    ]
    segments["recorder_id"] = segments["audio_collection_name"].astype(str)
    segments["recorder_display_name"] = segments["recorder_id"].map(_display_name)
    segments["duration_s"] = (
        segments["recording_end_utc"] - segments["recording_start_utc"]
    ).dt.total_seconds()
    return segments


def _prepare_recorders(segments: pd.DataFrame) -> pd.DataFrame:
    columns = ["recorder_id", "recorder_display_name", "device_id"]
    recorders = segments[columns].drop_duplicates().sort_values("recorder_id").reset_index(drop=True)
    return recorders


def _fill_missing_event_coordinates(events: pd.DataFrame, trace_points: pd.DataFrame) -> pd.DataFrame:
    enriched = events.copy()
    enriched["coordinate_source"] = "detection"
    enriched["nearest_trace_offset_s"] = None
    missing_mask = enriched["latitude_near_detection"].isna() | enriched["longitude_near_detection"].isna()
    if not missing_mask.any():
        return enriched

    trace_groups = {
        segment_id: group.sort_values("seconds_since_midnight_local").reset_index(drop=True)
        for segment_id, group in trace_points.groupby("segment_id", sort=False)
    }

    for idx in enriched.index[missing_mask]:
        row = enriched.loc[idx]
        segment_points = trace_groups.get(row["segment_id"])
        if segment_points is None or segment_points.empty or pd.isna(row["seconds_since_midnight_local"]):
            continue
        deltas = (segment_points["seconds_since_midnight_local"] - float(row["seconds_since_midnight_local"])).abs()
        nearest = segment_points.loc[int(deltas.idxmin())]
        nearest_offset_s = float(nearest["seconds_since_midnight_local"]) - float(row["seconds_since_midnight_local"])
        if pd.isna(enriched.at[idx, "latitude_near_detection"]):
            enriched.at[idx, "latitude_near_detection"] = float(nearest["latitude"])
        if pd.isna(enriched.at[idx, "longitude_near_detection"]):
            enriched.at[idx, "longitude_near_detection"] = float(nearest["longitude"])
        enriched.at[idx, "coordinate_source"] = "nearest_trace_backfill"
        enriched.at[idx, "nearest_trace_offset_s"] = round(nearest_offset_s, 3)
    return enriched


def _assign_session_ids(
    segments: pd.DataFrame,
    trace_points: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enriched_segments = segments.copy()
    session_by_segment: dict[str, str] = {}
    day_session_by_segment: dict[str, str] = {}

    for (date_local, recorder_id), group in enriched_segments.groupby(["date_local", "recorder_id"], sort=False):
        availability = _availability_payload(group, date_local=str(date_local), recorder_id=str(recorder_id))
        windows = availability["alive_windows"]
        for idx, row in group.iterrows():
            window = _find_window(windows, float(row["recording_start_seconds_local"]))
            if window is None:
                raise ValueError(f"segment is outside recorder availability: {row['segment_id']}")
            session_id = str(window["session_id"])
            enriched_segments.at[idx, "session_id"] = session_id
            session_by_segment[str(row["segment_id"])] = session_id

    for date_local, group in enriched_segments.groupby("date_local", sort=False):
        availability = _availability_payload(group, date_local=str(date_local))
        windows = availability["alive_windows"]
        for idx, row in group.iterrows():
            window = _find_window(windows, float(row["recording_start_seconds_local"]))
            if window is None:
                raise ValueError(f"segment is outside day availability: {row['segment_id']}")
            day_session_id = str(window["session_id"])
            enriched_segments.at[idx, "day_session_id"] = day_session_id
            day_session_by_segment[str(row["segment_id"])] = day_session_id

    enriched_trace_points = trace_points.copy()
    enriched_trace_points["session_id"] = enriched_trace_points["segment_id"].map(session_by_segment)
    enriched_trace_points["day_session_id"] = enriched_trace_points["segment_id"].map(day_session_by_segment)
    if enriched_trace_points["session_id"].isna().any():
        missing = sorted(enriched_trace_points.loc[enriched_trace_points["session_id"].isna(), "segment_id"].unique().tolist())[:5]
        raise ValueError(f"trace points missing session ids for segments: {missing}")
    if enriched_trace_points["day_session_id"].isna().any():
        missing = sorted(enriched_trace_points.loc[enriched_trace_points["day_session_id"].isna(), "segment_id"].unique().tolist())[:5]
        raise ValueError(f"trace points missing day session ids for segments: {missing}")

    enriched_events = events.copy()
    enriched_events["session_id"] = enriched_events["segment_id"].map(session_by_segment)
    enriched_events["day_session_id"] = enriched_events["segment_id"].map(day_session_by_segment)
    if enriched_events["session_id"].isna().any():
        missing = sorted(enriched_events.loc[enriched_events["session_id"].isna(), "segment_id"].unique().tolist())[:5]
        raise ValueError(f"events missing session ids for segments: {missing}")
    if enriched_events["day_session_id"].isna().any():
        missing = sorted(enriched_events.loc[enriched_events["day_session_id"].isna(), "segment_id"].unique().tolist())[:5]
        raise ValueError(f"events missing day session ids for segments: {missing}")

    windows_by_session: dict[str, tuple[float, float]] = {}
    windows_by_day_session: dict[str, tuple[float, float]] = {}
    for row in enriched_segments.itertuples(index=False):
        session_id = str(row.session_id)
        day_session_id = str(row.day_session_id)
        start_s = float(row.recording_start_seconds_local)
        end_s = float(row.recording_end_seconds_local)
        if session_id in windows_by_session:
            previous_start, previous_end = windows_by_session[session_id]
            windows_by_session[session_id] = (min(previous_start, start_s), max(previous_end, end_s))
        else:
            windows_by_session[session_id] = (start_s, end_s)
        if day_session_id in windows_by_day_session:
            previous_start, previous_end = windows_by_day_session[day_session_id]
            windows_by_day_session[day_session_id] = (min(previous_start, start_s), max(previous_end, end_s))
        else:
            windows_by_day_session[day_session_id] = (start_s, end_s)
    outside = []
    for row in enriched_events.itertuples(index=False):
        start_s, end_s = windows_by_session[str(row.session_id)]
        event_s = float(row.seconds_since_midnight_local)
        if event_s < start_s or event_s > end_s:
            outside.append(row.event_id if hasattr(row, "event_id") else row.review_row_key)
    outside_day = []
    for row in enriched_events.itertuples(index=False):
        start_s, end_s = windows_by_day_session[str(row.day_session_id)]
        event_s = float(row.seconds_since_midnight_local)
        if event_s < start_s or event_s > end_s:
            outside_day.append(row.event_id if hasattr(row, "event_id") else row.review_row_key)
    if outside:
        raise ValueError(f"events outside assigned session windows: {outside[:5]}")
    if outside_day:
        raise ValueError(f"events outside assigned day session windows: {outside_day[:5]}")

    return enriched_segments, enriched_trace_points, enriched_events

def _write_media_assets(config: BackendConfig, events: pd.DataFrame) -> pd.DataFrame:
    audio = _load_cleaned_corpus_wav_parts(config.raw_corpus_dir)

    clips_dir = config.processed_dir / "audio_clips"
    if clips_dir.exists():
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    enriched = events.copy()
    enriched["clip_key"] = enriched["review_row_key"].astype(str).map(_clip_key)
    duplicate_clip_keys = enriched["clip_key"].duplicated(keep=False)
    if duplicate_clip_keys.any():
        for ordinal, idx in enumerate(enriched.index[duplicate_clip_keys], start=1):
            enriched.at[idx, "clip_key"] = f"{enriched.at[idx, 'clip_key']}-{ordinal}"
    enriched["clip_path"] = None
    enriched["clip_url_path"] = None
    enriched["clip_duration_s"] = None

    frame_rate = float(audio.frame_rate)
    for idx, row in enriched.sort_values("export_begin_time_s").iterrows():
        start_s = float(row["export_begin_time_s"])
        end_s = float(row["export_end_time_s"])
        if end_s <= start_s:
            raise ValueError(f"invalid clip bounds for event {row['review_row_key']}: {start_s} to {end_s}")
        start_frame = max(0, min(audio.total_frames, int(start_s * frame_rate)))
        end_frame = max(start_frame, min(audio.total_frames, int(end_s * frame_rate + 0.999999)))
        frames = _read_cleaned_corpus_frames(audio, start_frame, end_frame)
        clip_path = clips_dir / f"{row['clip_key']}.wav"
        with wave.open(str(clip_path), "wb") as clip:
            clip.setparams(audio.params)
            clip.writeframes(frames)
        enriched.at[idx, "clip_path"] = str(clip_path.resolve())
        enriched.at[idx, "clip_url_path"] = f"/api/audio/clips/{row['clip_key']}"
        enriched.at[idx, "clip_duration_s"] = (end_frame - start_frame) / frame_rate

    return enriched


def _movement_summaries(trace_points: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if trace_points.empty:
        return summaries

    grouped = trace_points.groupby(
        ["recorder_id", "recorder_display_name", "session_id", "day_session_id"],
        sort=True,
        dropna=False,
    )
    for (recorder_id, recorder_display_name, session_id, day_session_id), group in grouped:
        ordered = group.sort_values("seconds_since_midnight_local")
        distance = 0.0
        previous: tuple[float, float] | None = None
        for row in ordered.itertuples(index=False):
            point = (float(row.longitude), float(row.latitude))
            if previous is not None:
                distance += _distance_meters(previous, point)
            previous = point
        start_s = float(ordered["seconds_since_midnight_local"].min())
        end_s = float(ordered["seconds_since_midnight_local"].max())
        duration_s = max(0.0, end_s - start_s)
        speed_mps = distance / duration_s if duration_s > 0 else 0.0
        label = "unknown"
        if duration_s > 0 and len(ordered.index) >= 2:
            label = "moving" if speed_mps >= MOVING_SPEED_THRESHOLD_MPS else "still"
        summaries.append(
            {
                "recorder_id": recorder_id,
                "recorder_display_name": recorder_display_name,
                "session_id": session_id,
                "day_session_id": day_session_id,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "distance_m": round(distance, 2),
                "duration_s": round(duration_s, 3),
                "speed_mps": round(speed_mps, 4),
                "state": label,
            }
        )
    return summaries


def _high_hum_intervals(events: pd.DataFrame) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    if events.empty:
        return intervals

    for (recorder_id, recorder_display_name, session_id, day_session_id), group in events.groupby(
        ["recorder_id", "recorder_display_name", "session_id", "day_session_id"],
        sort=True,
        dropna=False,
    ):
        binned = group.copy()
        binned["bin_start_s"] = (
            (binned["seconds_since_midnight_local"] // HIGH_HUM_BIN_SECONDS) * HIGH_HUM_BIN_SECONDS
        )
        counts = (
            binned.groupby("bin_start_s", sort=True)
            .size()
            .reset_index(name="event_count")
            .sort_values(["event_count", "bin_start_s"], ascending=[False, True])
            .head(HIGH_HUM_TOP_N)
        )
        for row in counts.itertuples(index=False):
            start_s = float(row.bin_start_s)
            intervals.append(
                {
                    "recorder_id": recorder_id,
                    "recorder_display_name": recorder_display_name,
                    "session_id": session_id,
                    "day_session_id": day_session_id,
                    "start_s": round(start_s, 3),
                    "end_s": round(min(SECONDS_PER_DAY, start_s + HIGH_HUM_BIN_SECONDS), 3),
                    "event_count": int(row.event_count),
                }
            )
    return intervals


def _duplicate_hum_count(left: pd.DataFrame, right: pd.DataFrame) -> int:
    left_times = sorted(float(value) for value in left["seconds_since_midnight_local"].dropna().tolist())
    right_times = sorted(float(value) for value in right["seconds_since_midnight_local"].dropna().tolist())
    count = 0
    right_index = 0
    for left_time in left_times:
        while right_index < len(right_times) and right_times[right_index] < left_time - DUPLICATE_HUM_WINDOW_SECONDS:
            right_index += 1
        if right_index < len(right_times) and abs(right_times[right_index] - left_time) <= DUPLICATE_HUM_WINDOW_SECONDS:
            count += 1
    return count


def _company_summaries(trace_points: pd.DataFrame, events: pd.DataFrame) -> list[dict[str, Any]]:
    if trace_points.empty:
        return []

    samples = trace_points.copy()
    samples["sample_s"] = (
        (samples["seconds_since_midnight_local"] // COMPANY_SAMPLE_SECONDS) * COMPANY_SAMPLE_SECONDS
    )
    samples = (
        samples.sort_values("seconds_since_midnight_local")
        .drop_duplicates(["day_session_id", "recorder_id", "sample_s"], keep="first")
        .reset_index(drop=True)
    )

    event_groups = {
        (str(recorder_id), str(day_session_id)): group
        for (recorder_id, day_session_id), group in events.groupby(["recorder_id", "day_session_id"], sort=False)
    }
    pair_counts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (day_session_id, sample_s), group in samples.groupby(["day_session_id", "sample_s"], sort=True):
        rows = list(group.itertuples(index=False))
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                distance = _distance_meters(
                    (float(left.longitude), float(left.latitude)),
                    (float(right.longitude), float(right.latitude)),
                )
                if distance > COMPANY_DISTANCE_METERS:
                    continue
                key = tuple(sorted([str(left.recorder_id), str(right.recorder_id)])) + (str(day_session_id),)
                entry = pair_counts.setdefault(
                    key,
                    {
                        "day_session_id": str(day_session_id),
                        "recorder_ids": sorted([str(left.recorder_id), str(right.recorder_id)]),
                        "recorder_display_names": sorted(
                            [str(left.recorder_display_name), str(right.recorder_display_name)]
                        ),
                        "sample_count": 0,
                        "min_distance_m": distance,
                        "mean_distance_m_total": 0.0,
                    },
                )
                entry["sample_count"] += 1
                entry["min_distance_m"] = min(float(entry["min_distance_m"]), distance)
                entry["mean_distance_m_total"] += distance

    summaries: list[dict[str, Any]] = []
    for key, entry in sorted(pair_counts.items()):
        recorder_ids = entry["recorder_ids"]
        day_session_id = str(entry["day_session_id"])
        duplicate_hums = _duplicate_hum_count(
            event_groups.get((recorder_ids[0], day_session_id), pd.DataFrame()),
            event_groups.get((recorder_ids[1], day_session_id), pd.DataFrame()),
        )
        sample_count = int(entry["sample_count"])
        summaries.append(
            {
                "day_session_id": day_session_id,
                "recorder_ids": recorder_ids,
                "recorder_display_names": entry["recorder_display_names"],
                "threshold_m": COMPANY_DISTANCE_METERS,
                "together_duration_s": round(sample_count * COMPANY_SAMPLE_SECONDS, 3),
                "sample_count": sample_count,
                "min_distance_m": round(float(entry["min_distance_m"]), 2),
                "mean_distance_m": round(float(entry["mean_distance_m_total"]) / sample_count, 2),
                "duplicate_hum_count": duplicate_hums,
            }
        )
    return summaries


def _sampled_positions(trace_points: pd.DataFrame) -> list[dict[str, Any]]:
    if trace_points.empty:
        return []

    samples = trace_points.copy()
    samples["sample_s"] = (
        (samples["seconds_since_midnight_local"] // COMPANY_SAMPLE_SECONDS) * COMPANY_SAMPLE_SECONDS
    )
    samples = (
        samples.sort_values("seconds_since_midnight_local")
        .drop_duplicates(["day_session_id", "recorder_id", "sample_s"], keep="first")
        .reset_index(drop=True)
    )

    return [
        {
            "recorder_id": str(row.recorder_id),
            "recorder_display_name": str(row.recorder_display_name),
            "session_id": str(row.session_id),
            "day_session_id": str(row.day_session_id),
            "sample_s": round(float(row.sample_s), 3),
            "latitude": round(float(row.latitude), 7),
            "longitude": round(float(row.longitude), 7),
        }
        for row in samples.itertuples(index=False)
    ]


def _event_time_summaries(events: pd.DataFrame) -> list[dict[str, Any]]:
    if events.empty:
        return []

    summaries: list[dict[str, Any]] = []
    grouped = events.groupby(
        ["recorder_id", "recorder_display_name", "day_session_id"],
        sort=True,
        dropna=False,
    )
    for (recorder_id, recorder_display_name, day_session_id), group in grouped:
        summaries.append(
            {
                "recorder_id": str(recorder_id),
                "recorder_display_name": str(recorder_display_name),
                "day_session_id": str(day_session_id),
                "event_times_s": [
                    round(float(value), 3)
                    for value in sorted(group["seconds_since_midnight_local"].dropna().tolist())
                ],
            }
        )
    return summaries


def _build_trace_points(config: BackendConfig, segments: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing_csvs: list[str] = []
    for segment in segments.itertuples(index=False):
        csv_path = sidecar_cache_path(config.sidecar_cache_dir, getattr(segment, "csv_hpc_path"))
        if csv_path is None or not csv_path.is_file():
            missing_csvs.append(str(getattr(segment, "csv_hpc_path")))
            continue
        csv_df, summary = summarize_audiomoth_csv(csv_path)
        points = valid_trace_points(csv_df, config.timezone_name)
        if points.empty:
            continue
        frame = pd.DataFrame(
            {
                "segment_id": segment.segment_id,
                "point_index_in_segment": range(1, len(points.index) + 1),
                "recorder_id": segment.recorder_id,
                "recorder_display_name": segment.recorder_display_name,
                "wav_basename": segment.wav_basename,
                "timestamp_utc": points["timestamp_utc"],
                "timestamp_local": points["timestamp_local"],
                "date_local": points["date_local"],
                "seconds_since_midnight_local": points["seconds_since_midnight_local"].astype("float64"),
                "latitude": points["latitude"].astype("float64"),
                "longitude": points["longitude"].astype("float64"),
                "gps_status": "A",
                "timer_count": pd.to_numeric(points.get("TIMER_COUNT"), errors="coerce").astype("Int64"),
                "buffers_filled": pd.to_numeric(points.get("BUFFERS_FILLED"), errors="coerce").astype("Int64"),
                "buffers_written": pd.to_numeric(points.get("BUFFERS_WRITTEN"), errors="coerce").astype("Int64"),
                "sample_rate_estimate": summary.sample_rate_estimate,
            }
        )
        frames.append(frame)
    if missing_csvs:
        preview = [path for path in missing_csvs if path][:5]
        raise FileNotFoundError(f"missing cached AudioMoth CSV sidecars for segments such as: {preview}")

    trace_points = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if trace_points.empty:
        raise ValueError("no valid trace points were produced from cached AudioMoth CSV sidecars")
    return trace_points.sort_values(["date_local", "recorder_id", "timestamp_local"]).reset_index(drop=True)


def _filter_trace_points_with_blocker(
    trace_points: pd.DataFrame,
    blocker_polygon: list[tuple[float, float]],
) -> pd.DataFrame:
    if trace_points.empty:
        return trace_points

    lons = [lon for lon, _ in blocker_polygon]
    lats = [lat for _, lat in blocker_polygon]
    blocked_mask = (
        trace_points["longitude"].between(min(lons), max(lons), inclusive="both")
        & trace_points["latitude"].between(min(lats), max(lats), inclusive="both")
    )
    filtered = trace_points.loc[~blocked_mask].copy()
    if filtered.empty:
        raise ValueError("gps blocker removed every trace point from the dataset")
    return filtered.reset_index(drop=True)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def _manifest_audio_export(raw_corpus_dir: Path, export_manifest: dict[str, Any]) -> dict[str, Any]:
    parts_manifest = raw_corpus_dir / "cleaned_alpaca_hum_corpus" / "review_concat_parts" / "manifest.json"
    return {
        "path": str(parts_manifest.resolve()),
        "parts_dir": str(parts_manifest.parent.resolve()),
        "row_count": export_manifest.get("row_count"),
        "duration_s": export_manifest.get("duration_s"),
        "gap_duration_s": export_manifest.get("gap_duration_s"),
    }


def _segment_payload(day_segments: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in day_segments.itertuples(index=False):
        records.append(
            {
                "segment_id": row.segment_id,
                "session_id": row.session_id,
                "day_session_id": row.day_session_id,
                "recorder_id": row.recorder_id,
                "recorder_display_name": row.recorder_display_name,
                "device_id": row.device_id,
                "wav_basename": row.wav_basename,
                "source_audio_file": row.source_audio_file_normalized,
                "recording_start_utc": _to_iso(row.recording_start_utc),
                "recording_end_utc": _to_iso(row.recording_end_utc),
                "recording_start_local": _to_iso(row.recording_start_local),
                "recording_end_local": _to_iso(row.recording_end_local),
                "recording_start_seconds_local": _safe_float(row.recording_start_seconds_local),
                "recording_end_seconds_local": _safe_float(row.recording_end_seconds_local),
                "duration_s": _safe_float(row.duration_s),
                "gps_valid_fraction": _safe_float(row.gps_valid_fraction),
                "first_valid_latitude": _safe_float(row.first_valid_latitude),
                "first_valid_longitude": _safe_float(row.first_valid_longitude),
                "last_valid_latitude": _safe_float(row.last_valid_latitude),
                "last_valid_longitude": _safe_float(row.last_valid_longitude),
            }
        )
    return records


def _trace_payload(day_trace_points: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    grouped = day_trace_points.groupby(
        ["segment_id", "session_id", "day_session_id", "recorder_id", "recorder_display_name", "wav_basename"],
        sort=True,
        dropna=False,
    )
    for (segment_id, session_id, day_session_id, recorder_id, recorder_display_name, wav_basename), group in grouped:
        ordered = group.sort_values("point_index_in_segment")
        records.append(
            {
                "segment_id": segment_id,
                "session_id": session_id,
                "day_session_id": day_session_id,
                "recorder_id": recorder_id,
                "recorder_display_name": recorder_display_name,
                "wav_basename": wav_basename,
                "point_count": int(len(ordered.index)),
                "times_s": [round(float(value), 3) for value in ordered["seconds_since_midnight_local"].tolist()],
                "latitudes": [round(float(value), 7) for value in ordered["latitude"].tolist()],
                "longitudes": [round(float(value), 7) for value in ordered["longitude"].tolist()],
            }
        )
    return records


def _event_payload(day_events: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in day_events.itertuples(index=False):
        records.append(
            {
                "event_id": row.review_row_key,
                "segment_id": row.segment_id,
                "session_id": row.session_id,
                "day_session_id": row.day_session_id,
                "recorder_id": row.recorder_id,
                "recorder_display_name": row.recorder_display_name,
                "device_id": row.device_id,
                "wav_basename": row.wav_basename,
                "source_audio_file": row.source_audio_file_normalized,
                "timestamp_start_utc": _to_iso(row.timestamp_start_utc),
                "timestamp_end_utc": _to_iso(row.timestamp_end_utc),
                "timestamp_start_local": _to_iso(row.timestamp_start_local),
                "timestamp_end_local": _to_iso(row.timestamp_end_local),
                "seconds_since_midnight_local": _safe_float(row.seconds_since_midnight_local),
                "latitude": _safe_float(row.latitude_near_detection),
                "longitude": _safe_float(row.longitude_near_detection),
                "coordinate_source": _safe_str(row.coordinate_source),
                "nearest_trace_offset_s": _safe_float(row.nearest_trace_offset_s),
                "detection_score": _safe_float(row.detection_score),
                "label": _safe_str(row.reviewed_resolved_label_semantic),
                "source_start_s": _safe_float(row.source_start_s),
                "source_end_s": _safe_float(row.source_end_s),
                "export_selection": _safe_int(row.export_selection),
                "export_begin_time_s": _safe_float(row.export_begin_time_s),
                "export_end_time_s": _safe_float(row.export_end_time_s),
                "clip_key": row.clip_key,
                "clip_path": row.clip_path,
                "clip_url_path": row.clip_url_path,
                "clip_duration_s": _safe_float(row.clip_duration_s),
            }
        )
    return records


def _counts_payload(
    *,
    recorder_count: int,
    segment_count: int,
    trace_point_count: int,
    trace_segment_count: int,
    event_count: int,
) -> dict[str, int]:
    return {
        "recorders": int(recorder_count),
        "segments": int(segment_count),
        "trace_points": int(trace_point_count),
        "trace_segments": int(trace_segment_count),
        "events": int(event_count),
    }


def _write_day_shards(
    config: BackendConfig,
    segments: pd.DataFrame,
    trace_points: pd.DataFrame,
    events: pd.DataFrame,
    sampled_positions: list[dict[str, Any]],
    event_time_summaries: list[dict[str, Any]],
    movement_summaries: list[dict[str, Any]],
    high_hum_intervals: list[dict[str, Any]],
    company_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    days_dir = config.processed_dir / "days"
    if days_dir.exists():
        shutil.rmtree(days_dir)
    days_dir.mkdir(parents=True, exist_ok=True)

    day_rows: list[dict[str, Any]] = []
    for date_local in sorted(events["date_local"].dropna().unique().tolist()):
        day_segments = segments[segments["date_local"] == date_local].copy()
        day_trace_points = trace_points[trace_points["date_local"] == date_local].copy()
        day_events = events[events["date_local"] == date_local].copy()
        day_availability = _availability_payload(day_segments, date_local=date_local)
        day_company_summaries = [
            item for item in company_summaries if str(item["day_session_id"]).startswith(f"{date_local}:")
        ]
        day_sampled_positions = [
            item for item in sampled_positions if str(item["day_session_id"]).startswith(f"{date_local}:")
        ]
        day_event_time_summaries = [
            item for item in event_time_summaries if str(item["day_session_id"]).startswith(f"{date_local}:")
        ]
        day_dir = days_dir / date_local
        recorder_dir = day_dir / "recorders"
        recorder_dir.mkdir(parents=True, exist_ok=True)
        day_path = day_dir / "index.json"

        trace_encoding = {
            "kind": "per-segment-array",
            "time_unit": "seconds_since_midnight_local",
            "coordinate_order": ["latitude", "longitude"],
        }
        recorder_rows: list[dict[str, Any]] = []
        for recorder_id in sorted(day_events["recorder_id"].dropna().unique().tolist()):
            recorder_segments = day_segments[day_segments["recorder_id"] == recorder_id].copy()
            recorder_trace_points = day_trace_points[day_trace_points["recorder_id"] == recorder_id].copy()
            recorder_events = day_events[day_events["recorder_id"] == recorder_id].copy()
            recorder_display_name = recorder_events["recorder_display_name"].dropna().astype(str).iloc[0]
            recorder_availability = _availability_payload(
                recorder_segments,
                date_local=date_local,
                recorder_id=recorder_id,
            )
            recorder_session_ids = {window["session_id"] for window in recorder_availability["alive_windows"]}
            recorder_day_session_ids = {segment["day_session_id"] for segment in _segment_payload(recorder_segments)}
            recorder_movement = [
                item
                for item in movement_summaries
                if item["recorder_id"] == recorder_id and item["session_id"] in recorder_session_ids
            ]
            recorder_high_hums = [
                item
                for item in high_hum_intervals
                if item["recorder_id"] == recorder_id and item["session_id"] in recorder_session_ids
            ]
            recorder_company = [
                item
                for item in day_company_summaries
                if recorder_id in item["recorder_ids"] and item["day_session_id"] in recorder_day_session_ids
            ]
            recorder_samples = [
                item
                for item in day_sampled_positions
                if item["recorder_id"] == recorder_id and item["session_id"] in recorder_session_ids
            ]
            recorder_event_times = [
                item
                for item in day_event_time_summaries
                if item["recorder_id"] == recorder_id and item["day_session_id"] in recorder_day_session_ids
            ]
            recorder_path = recorder_dir / f"{recorder_id}.json"
            recorder_payload = {
                "date_local": date_local,
                "timezone": config.timezone_name,
                "day_start_local": f"{date_local}T00:00:00",
                "day_end_local": f"{date_local}T23:59:59.999999",
                "trace_encoding": trace_encoding,
                "availability": recorder_availability,
                "recorder_id": recorder_id,
                "recorder_display_name": recorder_display_name,
                "segments": _segment_payload(recorder_segments),
                "trace_segments": _trace_payload(recorder_trace_points),
                "events": _event_payload(recorder_events),
                "sampled_positions": recorder_samples,
                "event_time_summaries": recorder_event_times,
                "movement_summaries": recorder_movement,
                "high_hum_intervals": recorder_high_hums,
                "company_summaries": recorder_company,
            }
            recorder_path.write_text(
                json.dumps(recorder_payload, separators=(",", ":"), allow_nan=False),
                encoding="utf8",
            )
            recorder_rows.append(
                {
                    "recorder_id": recorder_id,
                    "recorder_display_name": recorder_display_name,
                    "path": str(recorder_path.resolve()),
                    "segment_count": len(recorder_payload["segments"]),
                    "trace_point_count": int(
                        sum(item["point_count"] for item in recorder_payload["trace_segments"])
                    ),
                    "trace_segment_count": len(recorder_payload["trace_segments"]),
                    "event_count": len(recorder_payload["events"]),
                }
            )

        day_counts = _counts_payload(
            recorder_count=len(recorder_rows),
            segment_count=len(day_segments.index),
            trace_point_count=len(day_trace_points.index),
            trace_segment_count=int(day_trace_points["segment_id"].nunique()),
            event_count=len(day_events.index),
        )
        payload = {
            "date_local": date_local,
            "timezone": config.timezone_name,
            "day_start_local": f"{date_local}T00:00:00",
            "day_end_local": f"{date_local}T23:59:59.999999",
            "trace_encoding": trace_encoding,
            "availability": day_availability,
            "recorder_ids": [row["recorder_id"] for row in recorder_rows],
            "recorder_files": recorder_rows,
            "sampled_positions": day_sampled_positions,
            "event_time_summaries": day_event_time_summaries,
            "counts": day_counts,
            "company_summaries": day_company_summaries,
        }
        day_path.write_text(
            json.dumps(payload, separators=(",", ":"), allow_nan=False),
            encoding="utf8",
        )
        day_rows.append(
            {
                "date_local": date_local,
                "path": str(day_path.resolve()),
                "recorder_count": day_counts["recorders"],
                "segment_count": day_counts["segments"],
                "trace_point_count": day_counts["trace_points"],
                "trace_segment_count": day_counts["trace_segments"],
                "event_count": day_counts["events"],
            }
        )
    return day_rows


def build_dataset(config: BackendConfig) -> BuildResult:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.intermediate_dir.mkdir(parents=True, exist_ok=True)

    selection_df = _read_selection_table(config.raw_corpus_dir)
    export_manifest = _read_export_manifest(config.raw_corpus_dir)
    events = _prepare_events(config, selection_df)
    segments = _prepare_segments(config)
    recorders = _prepare_recorders(segments)
    trace_points = _build_trace_points(config, segments)
    blocker_polygon = _load_gps_blocker(config)
    trace_points = _filter_trace_points_with_blocker(trace_points, blocker_polygon)
    events = _fill_missing_event_coordinates(events, trace_points)
    segments, trace_points, events = _assign_session_ids(segments, trace_points, events)
    events = _write_media_assets(config, events)
    sampled_positions = _sampled_positions(trace_points)
    event_time_summaries = _event_time_summaries(events)
    movement_summaries = _movement_summaries(trace_points)
    high_hum_intervals = _high_hum_intervals(events)
    company_summaries = _company_summaries(trace_points, events)

    missing_segments = set(events["segment_id"]) - set(segments["segment_id"])
    if missing_segments:
        preview = sorted(missing_segments)[:5]
        raise ValueError(f"events refer to unknown segments: {preview}")

    if events["date_local"].nunique() != segments["date_local"].nunique():
        # This is an integrity hint, not a failure condition.
        pass

    _write_parquet(recorders, config.intermediate_dir / "recorders.parquet")
    _write_parquet(segments, config.intermediate_dir / "segments.parquet")
    _write_parquet(trace_points, config.intermediate_dir / "trace_points.parquet")
    _write_parquet(events, config.intermediate_dir / "events.parquet")
    _write_parquet(
        pd.DataFrame(
            movement_summaries,
            columns=[
                "recorder_id",
                "recorder_display_name",
                "session_id",
                "day_session_id",
                "start_s",
                "end_s",
                "distance_m",
                "duration_s",
                "speed_mps",
                "state",
            ],
        ),
        config.intermediate_dir / "movement_summaries.parquet",
    )
    _write_parquet(
        pd.DataFrame(
            high_hum_intervals,
            columns=[
                "recorder_id",
                "recorder_display_name",
                "session_id",
                "day_session_id",
                "start_s",
                "end_s",
                "event_count",
            ],
        ),
        config.intermediate_dir / "high_hum_intervals.parquet",
    )
    _write_parquet(
        pd.DataFrame(
            company_summaries,
            columns=[
                "day_session_id",
                "recorder_ids",
                "recorder_display_names",
                "threshold_m",
                "together_duration_s",
                "sample_count",
                "min_distance_m",
                "mean_distance_m",
                "duplicate_hum_count",
            ],
        ),
        config.intermediate_dir / "company_summaries.parquet",
    )

    all_points = [(row.longitude, row.latitude) for row in trace_points.itertuples(index=False)]
    bbox = compute_bbox(all_points)
    hull = hull_geojson(all_points)
    hull_path = config.processed_dir / "paddock_hull.geojson"
    hull_path.write_text(json.dumps(hull, indent=2, allow_nan=False), encoding="utf8")

    day_rows = _write_day_shards(
        config,
        segments,
        trace_points,
        events,
        sampled_positions,
        event_time_summaries,
        movement_summaries,
        high_hum_intervals,
        company_summaries,
    )
    manifest = {
        "dataset_name": config.dataset_name,
        "version": CONTRACT_VERSION,
        "timezone": config.timezone_name,
        "source_date_generated_utc": pd.Timestamp.utcnow().isoformat(),
        "available_days": [row["date_local"] for row in day_rows],
        "recorders": recorders.to_dict(orient="records"),
        "day_files": day_rows,
        "spatial_thresholds": {
            "polygon_link_meters": POLYGON_DISTANCE_METERS,
            "company_distance_meters": {
                "min": 5,
                "max": 120,
                "step": 5,
                "default": COMPANY_DISTANCE_METERS,
            },
        },
        "counts": _counts_payload(
            recorder_count=len(recorders.index),
            segment_count=len(segments.index),
            trace_point_count=len(trace_points.index),
            trace_segment_count=int(trace_points["segment_id"].nunique()),
            event_count=len(events.index),
        ),
        "bbox": bbox,
        "paddock_hull_path": str(hull_path.resolve()),
        "audio_export": _manifest_audio_export(config.raw_corpus_dir, export_manifest),
        "audio_clips": {
            "path": str((config.processed_dir / "audio_clips").resolve()),
            "count": int(len(events.index)),
        },
    }
    manifest_path = config.processed_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf8")
    return BuildResult(manifest_path=manifest_path, manifest=manifest)


def verify_dataset(config: BackendConfig) -> dict[str, Any]:
    manifest_path = config.processed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"processed manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if int(manifest.get("version", 0)) != CONTRACT_VERSION:
        raise ValueError(f"unsupported manifest version for verifier: {manifest.get('version')}")
    required_paths = [
        config.intermediate_dir / "recorders.parquet",
        config.intermediate_dir / "segments.parquet",
        config.intermediate_dir / "trace_points.parquet",
        config.intermediate_dir / "events.parquet",
        config.intermediate_dir / "movement_summaries.parquet",
        config.intermediate_dir / "high_hum_intervals.parquet",
        config.intermediate_dir / "company_summaries.parquet",
        Path(manifest["paddock_hull_path"]),
        Path(manifest["audio_clips"]["path"]),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"processed outputs are missing files: {missing}")

    recorders = pd.read_parquet(config.intermediate_dir / "recorders.parquet")
    segments = pd.read_parquet(config.intermediate_dir / "segments.parquet")
    trace_points = pd.read_parquet(config.intermediate_dir / "trace_points.parquet")
    events = pd.read_parquet(config.intermediate_dir / "events.parquet")
    blocker_polygon = _load_gps_blocker(config)
    lons = [lon for lon, _ in blocker_polygon]
    lats = [lat for _, lat in blocker_polygon]

    if len(events.index) != int(manifest["counts"]["events"]):
        raise ValueError("event count mismatch between manifest and intermediate parquet")
    if len(segments.index) != int(manifest["counts"]["segments"]):
        raise ValueError("segment count mismatch between manifest and intermediate parquet")
    if len(recorders.index) != int(manifest["counts"]["recorders"]):
        raise ValueError("recorder count mismatch between manifest and intermediate parquet")
    if len(trace_points.index) != int(manifest["counts"]["trace_points"]):
        raise ValueError("trace-point count mismatch between manifest and intermediate parquet")
    if int(trace_points["segment_id"].nunique()) != int(manifest["counts"]["trace_segments"]):
        raise ValueError("trace-segment count mismatch between manifest and intermediate parquet")
    if (
        events["session_id"].isna().any()
        or segments["session_id"].isna().any()
        or trace_points["session_id"].isna().any()
        or events["day_session_id"].isna().any()
        or segments["day_session_id"].isna().any()
        or trace_points["day_session_id"].isna().any()
    ):
        raise ValueError("session ids and day session ids must be populated for segments, trace points, and events")
    if (
        events["clip_path"].isna().any()
        or events["clip_url_path"].isna().any()
        or events["clip_duration_s"].isna().any()
    ):
        raise ValueError("some events are missing generated clip metadata")
    missing_clips = [path for path in events["clip_path"].dropna().astype(str).tolist() if not Path(path).is_file()]
    if missing_clips:
        raise FileNotFoundError(f"generated clip files are missing: {missing_clips[:5]}")

    event_segment_diff = set(events["segment_id"]) - set(segments["segment_id"])
    if event_segment_diff:
        raise ValueError(f"events reference missing segments: {sorted(event_segment_diff)[:5]}")

    trace_segment_diff = set(trace_points["segment_id"]) - set(segments["segment_id"])
    if trace_segment_diff:
        raise ValueError(f"trace points reference missing segments: {sorted(trace_segment_diff)[:5]}")

    if events["latitude_near_detection"].isna().any() or events["longitude_near_detection"].isna().any():
        raise ValueError("some events still have missing overlay coordinates after trace-based backfill")
    if (
        trace_points["longitude"].between(min(lons), max(lons), inclusive="both")
        & trace_points["latitude"].between(min(lats), max(lats), inclusive="both")
    ).any():
        raise ValueError("trace points inside the gps blocker survived dataset filtering")

    day_total_events = 0
    day_total_trace_points = 0
    day_total_segments = 0
    day_total_recorders = 0
    for day in manifest["day_files"]:
        day_path = Path(day["path"])
        payload = json.loads(day_path.read_text(encoding="utf8"))
        if payload["date_local"] != day["date_local"]:
            raise ValueError(f"day shard date mismatch in {day_path}")
        recorder_files = payload.get("recorder_files", [])
        if len(recorder_files) != int(payload["counts"]["recorders"]):
            raise ValueError(f"recorder count mismatch inside day index {day_path}")

        shard_events = 0
        shard_trace_points = 0
        shard_segments = 0
        shard_recorders = 0
        recorder_ids_seen: set[str] = set()
        for recorder_entry in recorder_files:
            recorder_path = Path(recorder_entry["path"])
            recorder_payload = json.loads(recorder_path.read_text(encoding="utf8"))
            recorder_id = str(recorder_entry["recorder_id"])
            if recorder_payload["date_local"] != payload["date_local"]:
                raise ValueError(f"recorder shard date mismatch in {recorder_path}")
            if recorder_payload["recorder_id"] != recorder_id:
                raise ValueError(f"recorder shard id mismatch in {recorder_path}")

            payload_segment_ids = {
                item["segment_id"] for item in recorder_payload["segments"] if item["recorder_id"] == recorder_id
            }
            payload_session_ids = {item["session_id"] for item in recorder_payload["segments"]}
            payload_day_session_ids = {item["day_session_id"] for item in recorder_payload["segments"]}
            availability_session_ids = {
                item["session_id"] for item in recorder_payload["availability"]["alive_windows"]
            }
            availability_day_session_ids = {
                item["session_id"] for item in payload["availability"]["alive_windows"]
            }
            if len(payload_segment_ids) != len(recorder_payload["segments"]):
                raise ValueError(f"recorder shard includes segment from another recorder: {recorder_path}")
            if not payload_session_ids.issubset(availability_session_ids):
                raise ValueError(f"recorder shard includes segment outside availability: {recorder_path}")
            if not payload_day_session_ids.issubset(availability_day_session_ids):
                raise ValueError(f"recorder shard includes segment outside day availability: {recorder_path}")
            trace_recorder_ids = {item["recorder_id"] for item in recorder_payload["trace_segments"]}
            if trace_recorder_ids != {recorder_id}:
                raise ValueError(f"recorder shard includes trace from another recorder: {recorder_path}")
            event_recorder_ids = {item["recorder_id"] for item in recorder_payload["events"]}
            if event_recorder_ids != {recorder_id}:
                raise ValueError(f"recorder shard includes event from another recorder: {recorder_path}")

            trace_segment_ids = {item["segment_id"] for item in recorder_payload["trace_segments"]}
            event_segment_ids = {item["segment_id"] for item in recorder_payload["events"]}
            trace_session_ids = {item["session_id"] for item in recorder_payload["trace_segments"]}
            event_session_ids = {item["session_id"] for item in recorder_payload["events"]}
            trace_day_session_ids = {item["day_session_id"] for item in recorder_payload["trace_segments"]}
            event_day_session_ids = {item["day_session_id"] for item in recorder_payload["events"]}
            if not trace_segment_ids.issubset(payload_segment_ids):
                raise ValueError(f"trace segment references missing segment in {recorder_path}")
            if not event_segment_ids.issubset(payload_segment_ids):
                raise ValueError(f"event references missing segment in {recorder_path}")
            if not trace_session_ids.issubset(availability_session_ids):
                raise ValueError(f"trace segment references unavailable session in {recorder_path}")
            if not event_session_ids.issubset(availability_session_ids):
                raise ValueError(f"event references unavailable session in {recorder_path}")
            if not trace_day_session_ids.issubset(availability_day_session_ids):
                raise ValueError(f"trace segment references unavailable day session in {recorder_path}")
            if not event_day_session_ids.issubset(availability_day_session_ids):
                raise ValueError(f"event references unavailable day session in {recorder_path}")
            for event in recorder_payload["events"]:
                clip_path = Path(event["clip_path"])
                if not clip_path.is_file() or event["clip_duration_s"] <= 0:
                    raise ValueError(f"invalid clip metadata for event in {recorder_path}: {event['event_id']}")

            shard_recorders += 1
            recorder_ids_seen.add(recorder_id)
            shard_events += len(recorder_payload["events"])
            shard_trace_points += sum(int(item["point_count"]) for item in recorder_payload["trace_segments"])
            shard_segments += len(recorder_payload["segments"])

            if len(recorder_payload["segments"]) != int(recorder_entry["segment_count"]):
                raise ValueError(f"segment count mismatch in recorder shard {recorder_path}")
            if len(recorder_payload["trace_segments"]) != int(recorder_entry["trace_segment_count"]):
                raise ValueError(f"trace segment count mismatch in recorder shard {recorder_path}")
            if len(recorder_payload["events"]) != int(recorder_entry["event_count"]):
                raise ValueError(f"event count mismatch in recorder shard {recorder_path}")
            if sum(int(item["point_count"]) for item in recorder_payload["trace_segments"]) != int(
                recorder_entry["trace_point_count"]
            ):
                raise ValueError(f"trace point count mismatch in recorder shard {recorder_path}")

        if sorted(recorder_ids_seen) != sorted(payload["recorder_ids"]):
            raise ValueError(f"recorder ids mismatch in day index {day_path}")
        if shard_recorders != int(payload["counts"]["recorders"]):
            raise ValueError(f"recorder totals mismatch in day index {day_path}")
        if shard_segments != int(payload["counts"]["segments"]):
            raise ValueError(f"segment totals mismatch in day index {day_path}")
        if shard_trace_points != int(payload["counts"]["trace_points"]):
            raise ValueError(f"trace point totals mismatch in day index {day_path}")
        if shard_events != int(payload["counts"]["events"]):
            raise ValueError(f"event totals mismatch in day index {day_path}")

        day_total_recorders += shard_recorders
        day_total_events += shard_events
        day_total_trace_points += shard_trace_points
        day_total_segments += shard_segments

    if day_total_events != len(events.index):
        raise ValueError("sum of day-shard events does not match intermediate events")
    if day_total_trace_points != len(trace_points.index):
        raise ValueError("sum of day-shard trace points does not match intermediate trace points")
    if day_total_segments != len(segments.index):
        raise ValueError("sum of day-shard segments does not match intermediate segments")
    if day_total_recorders != sum(int(day["recorder_count"]) for day in manifest["day_files"]):
        raise ValueError("sum of day-shard recorders does not match manifest day metadata")

    return {
        "manifest_path": str(manifest_path),
        "version": int(manifest["version"]),
        "events": len(events.index),
        "segments": len(segments.index),
        "trace_points": len(trace_points.index),
        "recorders": len(recorders.index),
        "days": len(manifest["day_files"]),
        "gps_blocker_path": str(config.gps_blocker_path),
    }

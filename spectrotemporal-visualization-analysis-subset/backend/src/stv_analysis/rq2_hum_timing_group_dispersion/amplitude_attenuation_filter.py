from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import subprocess
from typing import Any
import wave

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from .amplitude_audio import (
    FRAME_HOP_S,
    FRAME_LENGTH_S,
    PRIMARY_METRIC,
    WINDOW_LENGTH_S,
    amplitude_metrics,
    compare_to_primary,
    compute_event_amplitude_metrics,
    dbfs,
    ensure_window_assignment,
    frame_rms_values,
    hum_band_sos,
    load_primary_models,
    load_rq2_rich_events,
    pcm_bytes_to_float,
)


DEFAULT_ATTENUATION_DIR = Path("data/attenuation_raw_data_bundle/raw_audio_attenuation")
DEFAULT_INTERMEDIATE_EVENTS = Path("backend/data/processed/intermediate/events.parquet")
DEFAULT_OUTPUT_SUBDIR = Path("amplitude_attenuation_filter")

SEARCH_PRE_S = 3.0
SEARCH_POST_S = 1.0
MEASUREMENT_PRE_S = 0.75
MEASUREMENT_POST_S = 0.75

PRIMARY_CALIBRATION_ANIMALS = ("7026", "7460")
EXCLUDED_FOCAL_ANIMAL = "7935"

FENCE_RECORDERS = {
    "fence_0.5m": {
        "folder": "0.5m_SD30_13-02-25",
        "position_m_from_zero": 0.5,
        "primary_external_calibration": False,
        "exclusion_reason": "Documented recorder malfunction; retained only as diagnostic.",
    },
    "fence_1m": {
        "folder": "1m_SD12_13-02-25",
        "position_m_from_zero": 1.0,
        "primary_external_calibration": True,
        "exclusion_reason": "",
    },
    "fence_2m": {
        "folder": "2m_SD21_13-02-25",
        "position_m_from_zero": 2.0,
        "primary_external_calibration": True,
        "exclusion_reason": "",
    },
    "fence_4m": {
        "folder": "4m_SD26_13-02-25",
        "position_m_from_zero": 4.0,
        "primary_external_calibration": True,
        "exclusion_reason": "",
    },
    "fence_8m": {
        "folder": "8m_SD34_13-02-25",
        "position_m_from_zero": 8.0,
        "primary_external_calibration": True,
        "exclusion_reason": "",
    },
}

COLLAR_RECORDERS = {
    "7935": {"folder": "7935_SD24_13-02-25", "sd_card": "24"},
    "7026": {"folder": "7026_SD39_13-02-25", "sd_card": "39"},
    "7460": {"folder": "7460_SD22_13-02-25", "sd_card": "22"},
}

THRESHOLD_SPECS = (
    {
        "threshold_id": "attenuation_lenient",
        "external_fence_quantile": 0.90,
        "description": "Cutoff at the 90th percentile of known external fence-recorder hum amplitudes.",
    },
    {
        "threshold_id": "attenuation_primary",
        "external_fence_quantile": 0.95,
        "description": "Cutoff at the 95th percentile of known external fence-recorder hum amplitudes.",
    },
    {
        "threshold_id": "attenuation_conservative",
        "external_fence_quantile": 0.975,
        "description": "Cutoff at the 97.5th percentile of known external fence-recorder hum amplitudes.",
    },
)
THRESHOLD_IDS = tuple(str(spec["threshold_id"]) for spec in THRESHOLD_SPECS)
KEY_TERMS = ("z_area", "z_spread_change", "z_area:z_spread_change", "z_speed", "z_centroid_change")
AMPLITUDE_AXIS_LABEL = "Hum amplitude (dBFS)"
CALIBRATION_BY_RECEIVER_DPI = 1200

FIGURE_COLORS = {
    "self_collar": "#0b7285",
    "external_fence": "#c2410c",
    "attenuation_lenient": "#2563eb",
    "attenuation_primary": "#047857",
    "attenuation_conservative": "#b45309",
    "removed": "#cbd5e1",
    "retained": "#0f766e",
}


def run_attenuation_filter_analysis(
    *,
    launch_dir: Path,
    attenuation_dir: Path,
    intermediate_events_path: Path,
    output_dir: Path,
    event_metrics_path: Path | None = None,
    fit_models: bool = True,
    nsim: int = 1000,
    seed: int = 20260817,
) -> dict[str, Any]:
    for subdir in ("inputs", "tables", "figures", "reports", "cache"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    timestamp_events = parse_attenuation_timestamp_events(attenuation_dir)
    recorder_manifest = attenuation_recorder_manifest(attenuation_dir)
    calibration_measurements = compute_attenuation_calibration_measurements(
        attenuation_dir=attenuation_dir,
        timestamp_events=timestamp_events,
        recorder_manifest=recorder_manifest,
    )
    threshold_definitions = attenuation_threshold_definitions(calibration_measurements)
    (
        model_input,
        input_summary,
        area_summary,
        session_summary,
        recorder_summary,
        event_count_audit,
        rq2_event_flags,
    ) = materialize_rq2_attenuation_inputs(
        launch_dir=launch_dir,
        intermediate_events_path=intermediate_events_path,
        output_dir=output_dir,
        threshold_definitions=threshold_definitions,
        event_metrics_path=event_metrics_path,
    )
    figure_index = write_figures(
        output_dir=output_dir,
        calibration_measurements=calibration_measurements,
        threshold_definitions=threshold_definitions,
        rq2_event_flags=rq2_event_flags,
        area_summary=area_summary,
        session_summary=session_summary,
    )

    write_tables(
        output_dir=output_dir,
        timestamp_events=timestamp_events,
        recorder_manifest=recorder_manifest,
        calibration_measurements=calibration_measurements,
        threshold_definitions=threshold_definitions,
        model_input=model_input,
        input_summary=input_summary,
        area_summary=area_summary,
        session_summary=session_summary,
        recorder_summary=recorder_summary,
        event_count_audit=event_count_audit,
        rq2_event_flags=rq2_event_flags,
        figure_index=figure_index,
    )
    write_reports(
        launch_dir=launch_dir,
        attenuation_dir=attenuation_dir,
        intermediate_events_path=intermediate_events_path,
        output_dir=output_dir,
        timestamp_events=timestamp_events,
        recorder_manifest=recorder_manifest,
        calibration_measurements=calibration_measurements,
        threshold_definitions=threshold_definitions,
        input_summary=input_summary,
        area_summary=area_summary,
        session_summary=session_summary,
        recorder_summary=recorder_summary,
        event_count_audit=event_count_audit,
        figure_index=figure_index,
    )
    r_result: dict[str, Any] | None = None
    fit_reporting: dict[str, Any] | None = None
    if fit_models:
        r_result = run_attenuation_r_workflow(
            output_dir=output_dir,
            input_path=output_dir / "inputs" / "rq2_attenuation_filter_model_input_60s.csv",
            nsim=nsim,
            seed=seed,
        )
        fit_reporting = write_model_fit_reports(
            launch_dir=launch_dir,
            attenuation_dir=attenuation_dir,
            output_dir=output_dir,
            threshold_definitions=threshold_definitions,
            input_summary=input_summary,
            area_summary=area_summary,
            session_summary=session_summary,
            recorder_summary=recorder_summary,
            figure_index=figure_index,
        )

    manifest = {
        "bundle_name": "rq2_attenuation_calibrated_amplitude_filter_60s",
        "bundle_version": 2,
        "generated_at_local": pd.Timestamp.now().isoformat(),
        "launch_dir": str(launch_dir.resolve()),
        "attenuation_dir": str(attenuation_dir.resolve()),
        "intermediate_events_path": str(intermediate_events_path.resolve()),
        "event_metrics_path": str(event_metrics_path.resolve()) if event_metrics_path is not None else None,
        "output_dir": str(output_dir.resolve()),
        "primary_metric": PRIMARY_METRIC,
        "calibration_rule": "external fence-recorder upper-tail quantiles after self-collar timestamp realignment",
        "thresholds": threshold_definitions.to_dict(orient="records"),
        "models_fit": bool(fit_models),
        "model_constraints": {
            "window_length_s": WINDOW_LENGTH_S,
            "families": ["Poisson", "NB1"],
            "fixed_surfaces": ["interaction", "additive"],
            "dharma_tests_run": bool(fit_models),
            "dharma_simulations_per_model": int(nsim) if fit_models else 0,
            "seed": int(seed),
        },
        "r_workflow": r_result,
        "fit_reporting": fit_reporting,
        "outputs": {
            "model_input_csv": "inputs/rq2_attenuation_filter_model_input_60s.csv",
            "model_input_parquet": "inputs/rq2_attenuation_filter_model_input_60s.parquet",
            "rq2_event_flags": "tables/rq2_attenuation_event_flags.parquet",
            "calibration_measurements": "tables/attenuation_calibration_measurements.parquet",
            "threshold_definitions": "tables/rq2_attenuation_threshold_definitions.csv",
            "input_summary": "tables/rq2_attenuation_input_summary.csv",
            "area_summary": "tables/rq2_attenuation_area_decile_summary.csv",
            "session_summary": "tables/rq2_attenuation_session_summary.csv",
            "recorder_summary": "tables/rq2_attenuation_recorder_summary.csv",
            "model_run_manifest": "tables/rq2_attenuation_model_run_manifest.csv",
            "fixed_effects": "tables/rq2_attenuation_fixed_effects.csv",
            "dharma_standard_tests": "tables/rq2_attenuation_dharma_standard_tests.csv",
            "dharma_temporal_autocorrelation": "tables/rq2_attenuation_dharma_temporal_autocorrelation.csv",
            "methods_results": "reports/METHODS_RESULTS.md",
            "methods": "METHODS.md",
            "readme": "README.md",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    return manifest


def parse_attenuation_timestamp_events(attenuation_dir: Path) -> pd.DataFrame:
    path = attenuation_dir / "timestamps_raw_attenuationtestonly.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing attenuation timestamp table: {path}")
    raw = pd.read_csv(path)
    raw["name"] = raw["name"].fillna("").astype(str).str.strip()
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp"], unit="s", utc=True)
    raw["timestamp_adelaide"] = raw["timestamp_utc"].dt.tz_convert("Australia/Adelaide")
    raw["modified_utc"] = pd.to_datetime(raw["modified"], unit="s", utc=True)

    rows: list[dict[str, Any]] = []
    current_animal_id: str | None = None
    current_sd_card: str | None = None
    for row in raw.sort_values("timestamp").itertuples(index=False):
        name = str(row.name).strip()
        marker = re.search(r"ID[: ]+\s*(\d+)\s*Moth[: ]+\s*(\d+)", name, flags=re.I)
        if marker:
            current_animal_id = marker.group(1)
            current_sd_card = marker.group(2)
            rows.append(
                {
                    "timestamp_row_id": int(row.id),
                    "row_type": "focal_marker",
                    "source_animal_id": current_animal_id,
                    "source_animal_sd_card": current_sd_card,
                    "raw_name": name,
                    "timestamp_utc": row.timestamp_utc,
                    "timestamp_adelaide": row.timestamp_adelaide,
                    "modified_utc": row.modified_utc,
                    "source_position_m_from_zero": np.nan,
                    "facing_direction": None,
                    "is_hum_annotation": False,
                    "is_primary_calibration_event": False,
                    "exclusion_reason": "",
                }
            )
            continue

        is_na = "NA" in name.upper()
        looks_like_hum = bool(re.search(r"\bHum(?:m)?\b", name, flags=re.I) or re.match(r"^\d+(?:\.\d+)?\s*m?\b", name, flags=re.I))
        distance_match = re.search(r"(\d+(?:\.\d+)?)\s*m?", name, flags=re.I) if looks_like_hum else None
        direction_match = re.search(r"\b([TALR])\b\s*$", name, flags=re.I) if looks_like_hum else None
        distance_m = float(distance_match.group(1)) if distance_match else np.nan
        direction = direction_match.group(1).upper() if direction_match else None

        exclusion_reason = ""
        if not looks_like_hum:
            row_type = "note"
            exclusion_reason = "not a hum annotation"
        elif is_na:
            row_type = "excluded_hum_annotation"
            exclusion_reason = "observer marked row as NA"
        elif current_animal_id == EXCLUDED_FOCAL_ANIMAL:
            row_type = "excluded_hum_annotation"
            exclusion_reason = "first focal animal excluded from primary calibration: no facing direction and early distance convention changed"
        elif current_animal_id not in PRIMARY_CALIBRATION_ANIMALS:
            row_type = "excluded_hum_annotation"
            exclusion_reason = "not one of the primary calibration focal animals"
        elif not np.isfinite(distance_m):
            row_type = "excluded_hum_annotation"
            exclusion_reason = "missing mouth distance from zero point"
        elif direction is None:
            row_type = "excluded_hum_annotation"
            exclusion_reason = "missing facing direction"
        else:
            row_type = "hum_annotation"

        rows.append(
            {
                "timestamp_row_id": int(row.id),
                "row_type": row_type,
                "source_animal_id": current_animal_id,
                "source_animal_sd_card": current_sd_card,
                "raw_name": name,
                "timestamp_utc": row.timestamp_utc,
                "timestamp_adelaide": row.timestamp_adelaide,
                "modified_utc": row.modified_utc,
                "source_position_m_from_zero": distance_m,
                "facing_direction": direction,
                "is_hum_annotation": bool(looks_like_hum and not is_na),
                "is_primary_calibration_event": row_type == "hum_annotation",
                "exclusion_reason": exclusion_reason,
            }
        )
    return pd.DataFrame(rows)


def attenuation_recorder_manifest(attenuation_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for receiver_id, spec in FENCE_RECORDERS.items():
        rows.extend(
            recorder_rows(
                attenuation_dir=attenuation_dir,
                receiver_id=receiver_id,
                folder=str(spec["folder"]),
                receiver_role="fence",
                receiver_position_m_from_zero=float(spec["position_m_from_zero"]),
                receiver_animal_id=None,
                primary_external_calibration=bool(spec["primary_external_calibration"]),
                exclusion_reason=str(spec["exclusion_reason"]),
            )
        )
    for animal_id, spec in COLLAR_RECORDERS.items():
        rows.extend(
            recorder_rows(
                attenuation_dir=attenuation_dir,
                receiver_id=f"collar_{animal_id}",
                folder=str(spec["folder"]),
                receiver_role="collar",
                receiver_position_m_from_zero=np.nan,
                receiver_animal_id=animal_id,
                primary_external_calibration=False,
                exclusion_reason="" if animal_id in PRIMARY_CALIBRATION_ANIMALS else "not used for primary calibration",
            )
        )
    return pd.DataFrame(rows).sort_values(["receiver_role", "receiver_id", "wav_start_utc"]).reset_index(drop=True)


def recorder_rows(
    *,
    attenuation_dir: Path,
    receiver_id: str,
    folder: str,
    receiver_role: str,
    receiver_position_m_from_zero: float,
    receiver_animal_id: str | None,
    primary_external_calibration: bool,
    exclusion_reason: str,
) -> list[dict[str, Any]]:
    folder_path = attenuation_dir / folder
    if not folder_path.is_dir():
        raise FileNotFoundError(f"missing attenuation recorder folder: {folder_path}")
    device_id = read_device_id(folder_path / "DEVICE.TXT")
    rows = []
    for wav_path in sorted(folder_path.glob("*.WAV")):
        info = wav_metadata(wav_path)
        rows.append(
            {
                "receiver_id": receiver_id,
                "receiver_role": receiver_role,
                "receiver_animal_id": receiver_animal_id,
                "folder": folder,
                "device_id": device_id,
                "receiver_position_m_from_zero": receiver_position_m_from_zero,
                "primary_external_calibration": primary_external_calibration,
                "exclusion_reason": exclusion_reason,
                **info,
            }
        )
    return rows


def read_device_id(path: Path) -> str:
    if not path.is_file():
        return ""
    match = re.search(r"Device:\s*([A-Za-z0-9]+)", path.read_text(encoding="utf8", errors="replace"))
    return match.group(1) if match else ""


def wav_metadata(path: Path) -> dict[str, Any]:
    start = pd.to_datetime(path.stem, format="%Y%m%d_%H%M%S", utc=True)
    with wave.open(str(path), "rb") as wav:
        sample_rate = int(wav.getframerate())
        channels = int(wav.getnchannels())
        sample_width = int(wav.getsampwidth())
        frames = int(wav.getnframes())
    duration_s = frames / sample_rate
    return {
        "wav_path": str(path),
        "wav_basename": path.name,
        "wav_start_utc": start,
        "wav_end_utc": start + pd.to_timedelta(duration_s, unit="s"),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frames,
        "duration_s": duration_s,
    }


def compute_attenuation_calibration_measurements(
    *,
    attenuation_dir: Path,
    timestamp_events: pd.DataFrame,
    recorder_manifest: pd.DataFrame,
) -> pd.DataFrame:
    primary_events = timestamp_events.loc[timestamp_events["is_primary_calibration_event"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    sos_cache: dict[int, np.ndarray] = {}
    for event in primary_events.itertuples(index=False):
        source_collar = select_recorder_segment(
            recorder_manifest,
            receiver_id=f"collar_{event.source_animal_id}",
            timestamp=event.timestamp_utc,
        )
        if source_collar is None:
            rows.append(missing_measurement_row(event, "self_collar", "self_collar", "missing source-collar WAV coverage"))
            continue

        refined = refine_event_time_from_self_collar(
            event_timestamp=event.timestamp_utc,
            source_collar=source_collar,
            sos_cache=sos_cache,
        )
        receiver_rows = [source_collar]
        receiver_rows.extend(
            row
            for row in iter_recorder_segments_at_time(recorder_manifest, event.timestamp_utc)
            if str(row.receiver_role) == "fence"
        )
        for receiver in receiver_rows:
            row = measure_receiver_at_refined_time(
                event=event,
                receiver=receiver,
                refined=refined,
                sos_cache=sos_cache,
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["timestamp_row_id", "receiver_role", "receiver_id"]).reset_index(drop=True)


def select_recorder_segment(recorder_manifest: pd.DataFrame, *, receiver_id: str, timestamp: pd.Timestamp) -> Any | None:
    subset = recorder_manifest.loc[
        (recorder_manifest["receiver_id"].astype(str) == receiver_id)
        & (recorder_manifest["wav_start_utc"] <= timestamp)
        & (timestamp < recorder_manifest["wav_end_utc"])
    ]
    if subset.empty:
        return None
    return next(subset.sort_values("wav_start_utc").itertuples(index=False))


def iter_recorder_segments_at_time(recorder_manifest: pd.DataFrame, timestamp: pd.Timestamp) -> list[Any]:
    subset = recorder_manifest.loc[
        (recorder_manifest["wav_start_utc"] <= timestamp) & (timestamp < recorder_manifest["wav_end_utc"])
    ]
    return list(subset.sort_values(["receiver_role", "receiver_id", "wav_start_utc"]).itertuples(index=False))


def missing_measurement_row(event: Any, receiver_id: str, calibration_class: str, status: str) -> dict[str, Any]:
    return {
        "timestamp_row_id": int(event.timestamp_row_id),
        "source_animal_id": str(event.source_animal_id),
        "raw_name": str(event.raw_name),
        "event_timestamp_utc": event.timestamp_utc,
        "source_position_m_from_zero": float(event.source_position_m_from_zero),
        "facing_direction": event.facing_direction,
        "receiver_id": receiver_id,
        "receiver_role": "",
        "receiver_position_m_from_zero": np.nan,
        "receiver_animal_id": "",
        "device_id": "",
        "calibration_class": calibration_class,
        "source_to_receiver_nominal_distance_m": np.nan,
        "measurement_status": status,
        "refined_event_timestamp_utc": pd.NaT,
        "refined_delta_from_observer_timestamp_s": np.nan,
    }


def refine_event_time_from_self_collar(*, event_timestamp: pd.Timestamp, source_collar: Any, sos_cache: dict[int, np.ndarray]) -> dict[str, Any]:
    path = Path(str(source_collar.wav_path))
    offset_s = (event_timestamp - source_collar.wav_start_utc).total_seconds()
    samples, sample_rate = read_wav_segment_mono_float(path, offset_s - SEARCH_PRE_S, SEARCH_PRE_S + SEARCH_POST_S)
    if samples.size == 0:
        return {
            "timestamp": event_timestamp,
            "delta_s": 0.0,
            "self_search_max_dbfs": np.nan,
            "self_search_status": "empty search window",
        }
    hum_band = signal.sosfilt(hum_band_sos(sample_rate, sos_cache), samples)
    frames = frame_rms_values(hum_band, sample_rate)
    if frames.size == 0:
        return {
            "timestamp": event_timestamp,
            "delta_s": 0.0,
            "self_search_max_dbfs": np.nan,
            "self_search_status": "no frames",
        }
    max_index = int(np.nanargmax(frames))
    frame_hop_samples = max(1, int(round(FRAME_HOP_S * sample_rate)))
    frame_length_samples = max(1, int(round(FRAME_LENGTH_S * sample_rate)))
    frame_center_s = (max_index * frame_hop_samples + (frame_length_samples / 2.0)) / sample_rate
    delta_s = -SEARCH_PRE_S + frame_center_s
    refined_timestamp = event_timestamp + pd.to_timedelta(delta_s, unit="s")
    return {
        "timestamp": refined_timestamp,
        "delta_s": float(delta_s),
        "self_search_max_dbfs": dbfs(float(frames[max_index])),
        "self_search_status": "ok",
    }


def measure_receiver_at_refined_time(
    *,
    event: Any,
    receiver: Any,
    refined: dict[str, Any],
    sos_cache: dict[int, np.ndarray],
) -> dict[str, Any]:
    receiver_role = str(receiver.receiver_role)
    receiver_id = str(receiver.receiver_id)
    receiver_animal_id = "" if pd.isna(receiver.receiver_animal_id) else str(receiver.receiver_animal_id)
    if receiver_role == "collar":
        calibration_class = "self_collar" if receiver_animal_id == str(event.source_animal_id) else "other_collar"
        nominal_distance = np.nan
    elif receiver_role == "fence":
        if bool(receiver.primary_external_calibration):
            calibration_class = "external_fence"
        else:
            calibration_class = "excluded_fence_diagnostic"
        nominal_distance = abs(float(receiver.receiver_position_m_from_zero) - float(event.source_position_m_from_zero))
    else:
        calibration_class = "unknown"
        nominal_distance = np.nan

    result: dict[str, Any] = {
        "timestamp_row_id": int(event.timestamp_row_id),
        "source_animal_id": str(event.source_animal_id),
        "raw_name": str(event.raw_name),
        "event_timestamp_utc": event.timestamp_utc,
        "source_position_m_from_zero": float(event.source_position_m_from_zero),
        "facing_direction": event.facing_direction,
        "receiver_id": receiver_id,
        "receiver_role": receiver_role,
        "receiver_position_m_from_zero": receiver.receiver_position_m_from_zero,
        "receiver_animal_id": receiver_animal_id,
        "device_id": str(receiver.device_id),
        "wav_basename": str(receiver.wav_basename),
        "calibration_class": calibration_class,
        "source_to_receiver_nominal_distance_m": nominal_distance,
        "measurement_status": "ok",
        "refined_event_timestamp_utc": refined["timestamp"],
        "refined_delta_from_observer_timestamp_s": float(refined["delta_s"]),
        "self_search_max_dbfs": float(refined["self_search_max_dbfs"]) if np.isfinite(refined["self_search_max_dbfs"]) else np.nan,
        "self_search_status": str(refined["self_search_status"]),
        "measurement_window_pre_s": MEASUREMENT_PRE_S,
        "measurement_window_post_s": MEASUREMENT_POST_S,
    }
    try:
        offset_s = (refined["timestamp"] - receiver.wav_start_utc).total_seconds()
        samples, sample_rate = read_wav_segment_mono_float(
            Path(str(receiver.wav_path)),
            offset_s - MEASUREMENT_PRE_S,
            MEASUREMENT_PRE_S + MEASUREMENT_POST_S,
        )
        result.update(amplitude_metrics(samples, sample_rate, sos_cache))
    except Exception as exc:
        result["measurement_status"] = f"error:{type(exc).__name__}"
        result["measurement_error"] = str(exc)
    return result


def read_wav_segment_mono_float(path: Path, start_s: float, duration_s: float) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = int(wav.getframerate())
        channels = int(wav.getnchannels())
        sample_width = int(wav.getsampwidth())
        total_frames = int(wav.getnframes())
        start_frame = max(0, int(round(start_s * sample_rate)))
        frame_count = max(0, int(round(duration_s * sample_rate)))
        if start_frame >= total_frames or frame_count == 0:
            return np.array([], dtype=np.float32), sample_rate
        frame_count = min(frame_count, total_frames - start_frame)
        wav.setpos(start_frame)
        raw = wav.readframes(frame_count)
    data = pcm_bytes_to_float(raw, sample_width)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32, copy=False), sample_rate


def attenuation_threshold_definitions(calibration_measurements: pd.DataFrame) -> pd.DataFrame:
    self_values = pd.to_numeric(
        calibration_measurements.loc[calibration_measurements["calibration_class"] == "self_collar", PRIMARY_METRIC],
        errors="coerce",
    ).dropna()
    external_values = pd.to_numeric(
        calibration_measurements.loc[calibration_measurements["calibration_class"] == "external_fence", PRIMARY_METRIC],
        errors="coerce",
    ).dropna()
    if self_values.empty:
        raise ValueError("no valid self-collar calibration amplitudes")
    if external_values.empty:
        raise ValueError("no valid external fence calibration amplitudes")

    rows = []
    for spec in THRESHOLD_SPECS:
        cutoff = float(external_values.quantile(float(spec["external_fence_quantile"])))
        self_retained = int((self_values >= cutoff).sum())
        external_pass = int((external_values >= cutoff).sum())
        rows.append(
            {
                "threshold_id": spec["threshold_id"],
                "description": spec["description"],
                "primary_metric": PRIMARY_METRIC,
                "cutoff_dbfs": cutoff,
                "threshold_scope": "attenuation_experiment_external_fence_quantile",
                "external_fence_quantile": float(spec["external_fence_quantile"]),
                "external_fence_measurements": int(external_values.shape[0]),
                "external_fence_measurements_passing_cutoff": external_pass,
                "external_fence_pass_fraction": external_pass / external_values.shape[0],
                "self_collar_measurements": int(self_values.shape[0]),
                "self_collar_measurements_retained": self_retained,
                "self_collar_retained_fraction": self_retained / self_values.shape[0],
                "calibration_animals": ",".join(PRIMARY_CALIBRATION_ANIMALS),
                "excluded_focal_animal": EXCLUDED_FOCAL_ANIMAL,
                "timestamp_alignment": f"self-collar max {SEARCH_PRE_S:.1f}s before to {SEARCH_POST_S:.1f}s after observer timestamp",
                "measurement_window_s": MEASUREMENT_PRE_S + MEASUREMENT_POST_S,
            }
        )
    return pd.DataFrame(rows)


def materialize_rq2_attenuation_inputs(
    *,
    launch_dir: Path,
    intermediate_events_path: Path,
    output_dir: Path,
    threshold_definitions: pd.DataFrame,
    event_metrics_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_path = launch_dir / "tables" / "rq2_model_input.parquet"
    if not primary_path.is_file():
        raise FileNotFoundError(f"missing primary RQ2 model input: {primary_path}")

    primary = pd.read_parquet(primary_path)
    primary = primary.loc[primary["window_length_s"].astype(int) == WINDOW_LENGTH_S].copy()
    primary["rq2_window_id"] = primary["rq2_window_id"].astype(str)
    primary["area_decile"] = pd.qcut(primary["area_m2"], q=10, labels=False, duplicates="drop") + 1
    primary["area_label"] = primary["area_decile"].map(lambda value: f"D{int(value)}")

    events = load_or_compute_rq2_event_metrics(
        launch_dir=launch_dir,
        intermediate_events_path=intermediate_events_path,
        output_dir=output_dir,
        event_metrics_path=event_metrics_path,
    )
    events = ensure_window_assignment(events, launch_dir)
    events = events.loc[events["inside_existing_rq2_window"].astype(bool) & events["rq2_window_id"].notna()].copy()
    events["rq2_window_id"] = events["rq2_window_id"].astype(str)
    events = events.drop(columns=[column for column in ("area_decile", "area_label") if column in events.columns])
    events = events.merge(primary[["rq2_window_id", "area_decile", "area_label"]], on="rq2_window_id", how="left")
    rq2_event_flags = add_attenuation_event_flags(events, threshold_definitions)

    all_event_counts = (
        rq2_event_flags.groupby("rq2_window_id", sort=False).size().rename("attenuation_event_count_all").reset_index()
    )
    event_count_audit = primary[["rq2_window_id", "day_session_id", "hum_count"]].merge(all_event_counts, on="rq2_window_id", how="left")
    event_count_audit["attenuation_event_count_all"] = event_count_audit["attenuation_event_count_all"].fillna(0).astype(int)
    event_count_audit["count_delta_primary_minus_attenuation_events"] = (
        event_count_audit["hum_count"].astype(int) - event_count_audit["attenuation_event_count_all"].astype(int)
    )

    window_metric = (
        rq2_event_flags.groupby("rq2_window_id", sort=False)
        .agg(primary_metric_median_dbfs=(PRIMARY_METRIC, "median"), primary_metric_max_dbfs=(PRIMARY_METRIC, "max"))
        .reset_index()
    )
    rows = []
    summary_rows = []
    for threshold in threshold_definitions.itertuples(index=False):
        flag = f"passes_{threshold.threshold_id}"
        counts = (
            rq2_event_flags.loc[rq2_event_flags[flag].astype(bool)]
            .groupby("rq2_window_id", sort=False)
            .size()
            .rename("thresholded_hum_count")
            .reset_index()
        )
        frame = primary.merge(counts, on="rq2_window_id", how="left").merge(window_metric, on="rq2_window_id", how="left")
        frame["thresholded_hum_count"] = frame["thresholded_hum_count"].fillna(0).astype(int)
        frame["primary_hum_count"] = frame["hum_count"].astype(int)
        frame["hum_count_removed_by_threshold"] = frame["primary_hum_count"] - frame["thresholded_hum_count"]
        frame["hum_count"] = frame["thresholded_hum_count"].astype(int)
        frame["hum_presence"] = (frame["hum_count"] > 0).astype(int)
        frame["hum_rate_per_min"] = frame["hum_count"] / frame["window_length_minutes"].astype(float)
        frame["threshold_id"] = threshold.threshold_id
        frame["threshold_description"] = threshold.description
        frame["amplitude_threshold_primary_metric"] = PRIMARY_METRIC
        frame["amplitude_threshold_rule"] = "attenuation_external_fence_quantile"
        frame["threshold_cutoff_dbfs"] = float(threshold.cutoff_dbfs)
        frame["threshold_external_fence_quantile"] = float(threshold.external_fence_quantile)
        frame["calibration_self_collar_retained_fraction"] = float(threshold.self_collar_retained_fraction)
        frame["calibration_external_fence_pass_fraction"] = float(threshold.external_fence_pass_fraction)
        rows.append(frame)
        summary_rows.append(summarize_model_input_threshold(frame, threshold))

    model_input = pd.concat(rows, ignore_index=True)
    input_summary = pd.DataFrame(summary_rows)
    area_summary = summarize_area_deciles(model_input)
    session_summary = summarize_sessions(model_input)
    recorder_summary = summarize_recorders(rq2_event_flags, threshold_definitions)
    return model_input, input_summary, area_summary, session_summary, recorder_summary, event_count_audit, rq2_event_flags


def load_or_compute_rq2_event_metrics(
    *,
    launch_dir: Path,
    intermediate_events_path: Path,
    output_dir: Path,
    event_metrics_path: Path | None,
) -> pd.DataFrame:
    if event_metrics_path is not None:
        if not event_metrics_path.is_file():
            raise FileNotFoundError(f"missing supplied event metrics table: {event_metrics_path}")
        events = pd.read_parquet(event_metrics_path)
    else:
        events = load_rq2_rich_events(launch_dir, intermediate_events_path)
        metrics = compute_event_amplitude_metrics(events)
        events = events.merge(metrics, on="review_row_key", how="left")
    if PRIMARY_METRIC not in events.columns:
        raise ValueError(f"RQ2 event metrics are missing {PRIMARY_METRIC}")
    return events


def add_attenuation_event_flags(events: pd.DataFrame, threshold_definitions: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    values = pd.to_numeric(out[PRIMARY_METRIC], errors="coerce")
    for threshold in threshold_definitions.itertuples(index=False):
        flag = f"passes_{threshold.threshold_id}"
        out[flag] = values.notna() & (values >= float(threshold.cutoff_dbfs))
        out[f"{threshold.threshold_id}_cutoff_dbfs"] = float(threshold.cutoff_dbfs)
    return out


def summarize_model_input_threshold(frame: pd.DataFrame, threshold: Any) -> dict[str, Any]:
    primary_hum_windows = frame["primary_hum_count"].astype(int) > 0
    compact = frame["area_decile"].astype(float) <= 3
    retained_total = int(frame["hum_count"].sum())
    compact_retained = int(frame.loc[compact, "hum_count"].sum())
    primary_total = int(frame["primary_hum_count"].sum())
    return {
        "threshold_id": threshold.threshold_id,
        "threshold_description": threshold.description,
        "rows": int(len(frame.index)),
        "sessions": int(frame["session_id"].nunique()),
        "primary_hum_count_total": primary_total,
        "thresholded_hum_count_total": retained_total,
        "removed_hum_count_total": int(frame["hum_count_removed_by_threshold"].sum()),
        "retained_hum_fraction": retained_total / max(primary_total, 1),
        "removed_hum_fraction": float(frame["hum_count_removed_by_threshold"].sum() / max(primary_total, 1)),
        "zero_hum_fraction": float((frame["hum_count"] == 0).mean()),
        "hum_windows": int((frame["hum_count"] > 0).sum()),
        "primary_hum_windows": int(primary_hum_windows.sum()),
        "hum_windows_lost_all_hums": int((primary_hum_windows & (frame["hum_count"] == 0)).sum()),
        "cutoff_dbfs": float(threshold.cutoff_dbfs),
        "external_fence_quantile": float(threshold.external_fence_quantile),
        "calibration_self_collar_retained_fraction": float(threshold.self_collar_retained_fraction),
        "calibration_external_fence_pass_fraction": float(threshold.external_fence_pass_fraction),
        "compact_d1_d3_primary_hums": int(frame.loc[compact, "primary_hum_count"].sum()),
        "compact_d1_d3_retained_hums": compact_retained,
        "compact_d1_d3_retained_share_of_retained": compact_retained / max(retained_total, 1),
    }


def summarize_area_deciles(model_input: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold_id, group in model_input.groupby("threshold_id", sort=False):
        area = (
            group.groupby("area_decile", observed=True, sort=True)
            .agg(
                windows=("rq2_window_id", "size"),
                area_min_m2=("area_m2", "min"),
                area_median_m2=("area_m2", "median"),
                area_max_m2=("area_m2", "max"),
                primary_hums=("primary_hum_count", "sum"),
                retained_hums=("hum_count", "sum"),
                removed_hums=("hum_count_removed_by_threshold", "sum"),
                primary_hum_windows=("primary_hum_count", lambda values: int((values > 0).sum())),
                retained_hum_windows=("hum_count", lambda values: int((values > 0).sum())),
                cutoff_dbfs=("threshold_cutoff_dbfs", "first"),
            )
            .reset_index()
        )
        area["threshold_id"] = threshold_id
        area["area_label"] = area["area_decile"].map(lambda value: f"D{int(value)}")
        area["retained_hum_fraction"] = area["retained_hums"] / area["primary_hums"]
        area["removed_hum_fraction"] = area["removed_hums"] / area["primary_hums"]
        area["hum_windows_lost_all_hums"] = area["primary_hum_windows"] - area["retained_hum_windows"]
        rows.append(area)
    return pd.concat(rows, ignore_index=True)


def summarize_sessions(model_input: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold_id, group in model_input.groupby("threshold_id", sort=False):
        session = (
            group.groupby("day_session_id", sort=True)
            .agg(
                windows=("rq2_window_id", "size"),
                primary_hums=("primary_hum_count", "sum"),
                retained_hums=("hum_count", "sum"),
                removed_hums=("hum_count_removed_by_threshold", "sum"),
                primary_hum_windows=("primary_hum_count", lambda values: int((values > 0).sum())),
                retained_hum_windows=("hum_count", lambda values: int((values > 0).sum())),
            )
            .reset_index()
        )
        session["threshold_id"] = threshold_id
        session["retained_hum_fraction"] = session["retained_hums"] / session["primary_hums"]
        session["removed_hum_fraction"] = session["removed_hums"] / session["primary_hums"]
        session["hum_windows_lost_all_hums"] = session["primary_hum_windows"] - session["retained_hum_windows"]
        rows.append(session)
    return pd.concat(rows, ignore_index=True)


def summarize_recorders(rq2_event_flags: pd.DataFrame, threshold_definitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in threshold_definitions.itertuples(index=False):
        flag = f"passes_{threshold.threshold_id}"
        group = (
            rq2_event_flags.assign(passes_threshold=rq2_event_flags[flag].astype(bool))
            .groupby(["recorder_id", "recorder_display_name", "device_id"], dropna=False, sort=True)
            .agg(primary_hums=("review_row_key", "size"), retained_hums=("passes_threshold", "sum"))
            .reset_index()
        )
        group["threshold_id"] = threshold.threshold_id
        group["removed_hums"] = group["primary_hums"] - group["retained_hums"]
        group["retained_hum_fraction"] = group["retained_hums"] / group["primary_hums"]
        group["removed_hum_fraction"] = group["removed_hums"] / group["primary_hums"]
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def write_tables(
    *,
    output_dir: Path,
    timestamp_events: pd.DataFrame,
    recorder_manifest: pd.DataFrame,
    calibration_measurements: pd.DataFrame,
    threshold_definitions: pd.DataFrame,
    model_input: pd.DataFrame,
    input_summary: pd.DataFrame,
    area_summary: pd.DataFrame,
    session_summary: pd.DataFrame,
    recorder_summary: pd.DataFrame,
    event_count_audit: pd.DataFrame,
    rq2_event_flags: pd.DataFrame,
    figure_index: pd.DataFrame,
) -> None:
    tables_dir = output_dir / "tables"
    inputs_dir = output_dir / "inputs"
    timestamp_events.to_csv(tables_dir / "attenuation_timestamp_events.csv", index=False)
    recorder_manifest.to_csv(tables_dir / "attenuation_recorder_manifest.csv", index=False)
    calibration_measurements.to_parquet(tables_dir / "attenuation_calibration_measurements.parquet", index=False)
    calibration_measurements.to_csv(tables_dir / "attenuation_calibration_measurements.csv", index=False)
    calibration_summary_tables(calibration_measurements).to_csv(tables_dir / "attenuation_calibration_summary.csv", index=False)
    threshold_definitions.to_csv(tables_dir / "rq2_attenuation_threshold_definitions.csv", index=False)
    rq2_event_flags.to_parquet(tables_dir / "rq2_attenuation_event_flags.parquet", index=False)
    model_input.to_csv(inputs_dir / "rq2_attenuation_filter_model_input_60s.csv", index=False)
    model_input.to_parquet(inputs_dir / "rq2_attenuation_filter_model_input_60s.parquet", index=False)
    input_summary.to_csv(tables_dir / "rq2_attenuation_input_summary.csv", index=False)
    area_summary.to_csv(tables_dir / "rq2_attenuation_area_decile_summary.csv", index=False)
    session_summary.to_csv(tables_dir / "rq2_attenuation_session_summary.csv", index=False)
    recorder_summary.to_csv(tables_dir / "rq2_attenuation_recorder_summary.csv", index=False)
    event_count_audit.to_csv(tables_dir / "rq2_attenuation_event_count_audit.csv", index=False)
    figure_index.to_csv(tables_dir / "rq2_attenuation_figure_index.csv", index=False)


def calibration_summary_tables(calibration_measurements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in calibration_measurements.groupby(["calibration_class", "receiver_id"], dropna=False, sort=True):
        values = pd.to_numeric(group[PRIMARY_METRIC], errors="coerce").dropna()
        rows.append(
            {
                "calibration_class": keys[0],
                "receiver_id": keys[1],
                "measurements": int(len(group.index)),
                "valid_metric_measurements": int(values.shape[0]),
                "mean_dbfs": float(values.mean()) if not values.empty else np.nan,
                "sd_dbfs": float(values.std()) if values.shape[0] > 1 else np.nan,
                "q05_dbfs": float(values.quantile(0.05)) if not values.empty else np.nan,
                "q10_dbfs": float(values.quantile(0.10)) if not values.empty else np.nan,
                "q25_dbfs": float(values.quantile(0.25)) if not values.empty else np.nan,
                "q50_dbfs": float(values.quantile(0.50)) if not values.empty else np.nan,
                "q75_dbfs": float(values.quantile(0.75)) if not values.empty else np.nan,
                "q90_dbfs": float(values.quantile(0.90)) if not values.empty else np.nan,
                "q95_dbfs": float(values.quantile(0.95)) if not values.empty else np.nan,
                "max_dbfs": float(values.max()) if not values.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_figures(
    *,
    output_dir: Path,
    calibration_measurements: pd.DataFrame,
    threshold_definitions: pd.DataFrame,
    rq2_event_flags: pd.DataFrame,
    area_summary: pd.DataFrame,
    session_summary: pd.DataFrame,
) -> pd.DataFrame:
    figures_dir = output_dir / "figures"
    rows = [
        plot_calibration_distribution(calibration_measurements, threshold_definitions, figures_dir),
        plot_calibration_by_receiver(calibration_measurements, threshold_definitions, figures_dir),
        plot_calibration_amplitude_by_distance(calibration_measurements, threshold_definitions, figures_dir),
        plot_rq2_distribution(rq2_event_flags, threshold_definitions, figures_dir),
        plot_removed_fraction_by_area(area_summary, figures_dir),
        plot_retained_removed_counts_by_area(area_summary, figures_dir),
        plot_removed_fraction_by_session(session_summary, figures_dir),
    ]
    return pd.DataFrame(rows)


def plot_calibration_distribution(measurements: pd.DataFrame, thresholds: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for calibration_class, color, label in [
        ("self_collar", FIGURE_COLORS["self_collar"], "known self-collar hums"),
        ("external_fence", FIGURE_COLORS["external_fence"], "known external fence recordings"),
    ]:
        values = pd.to_numeric(measurements.loc[measurements["calibration_class"] == calibration_class, PRIMARY_METRIC], errors="coerce").dropna()
        ax.hist(values, bins=46, alpha=0.58, color=color, label=f"{label} (n={len(values)})")
    add_threshold_lines(ax, thresholds, label_lines=False)
    add_threshold_line_labels(ax, thresholds)
    ax.set_title("Attenuation Calibration: Self-Collar Versus External Fence Levels")
    ax.set_xlabel(AMPLITUDE_AXIS_LABEL)
    ax.set_ylabel("Measurement count")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.90, loc="upper left")
    path = figures_dir / "rq2_attenuation_calibration_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "figure": path.name,
        "relative_path": str(path.relative_to(figures_dir.parent)),
        "figure_type": "Calibration amplitude distribution",
        "description": "Known self-collar and external fence-recorder amplitude distributions with attenuation-derived cutoffs.",
    }


def plot_calibration_amplitude_by_distance(measurements: pd.DataFrame, thresholds: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    subset = measurements.loc[
        measurements["calibration_class"].eq("external_fence")
        & measurements["source_to_receiver_nominal_distance_m"].notna()
        & measurements[PRIMARY_METRIC].notna()
    ].copy()
    subset["distance_m"] = pd.to_numeric(subset["source_to_receiver_nominal_distance_m"], errors="coerce")
    subset["amplitude_dbfs"] = pd.to_numeric(subset[PRIMARY_METRIC], errors="coerce")
    subset = subset.loc[subset["distance_m"].notna() & subset["amplitude_dbfs"].notna()].copy()
    if subset.empty:
        raise ValueError("no external-fence calibration measurements available for amplitude-by-distance plot")

    rng = np.random.default_rng(20260817)
    x = subset["distance_m"].to_numpy(dtype=float) + rng.uniform(-0.035, 0.035, size=len(subset.index))
    y = subset["amplitude_dbfs"].to_numpy(dtype=float)

    bins = np.arange(0.0, np.ceil(float(subset["distance_m"].max()) * 2.0) / 2.0 + 0.75, 0.5)
    subset["distance_bin"] = pd.cut(subset["distance_m"], bins=bins, include_lowest=True, right=False)
    summary = (
        subset.groupby("distance_bin", observed=True)
        .agg(
            distance_mid_m=("distance_m", "median"),
            amplitude_q25_dbfs=("amplitude_dbfs", lambda values: float(np.quantile(values, 0.25))),
            amplitude_median_dbfs=("amplitude_dbfs", "median"),
            amplitude_q75_dbfs=("amplitude_dbfs", lambda values: float(np.quantile(values, 0.75))),
            measurements=("amplitude_dbfs", "size"),
        )
        .reset_index(drop=True)
    )
    summary = summary.loc[summary["measurements"] >= 5].copy()

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.scatter(x, y, s=12, alpha=0.22, color="#64748b", edgecolors="none", label=f"External fence recordings (n={len(subset.index)})")
    if not summary.empty:
        ax.fill_between(
            summary["distance_mid_m"].to_numpy(dtype=float),
            summary["amplitude_q25_dbfs"].to_numpy(dtype=float),
            summary["amplitude_q75_dbfs"].to_numpy(dtype=float),
            color="#0f766e",
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            summary["distance_mid_m"],
            summary["amplitude_median_dbfs"],
            color="#0f766e",
            linewidth=2.2,
            marker="o",
            markersize=4.2,
            label="Median and IQR in 0.5 m bins",
        )
    ax.set_title("Attenuation Calibration Amplitude By Distance")
    ax.set_xlabel("Nominal mouth-to-recorder distance (m)")
    ax.set_ylabel(AMPLITUDE_AXIS_LABEL)
    ax.set_xlim(-0.15, max(8.2, float(subset["distance_m"].max()) + 0.3))
    add_threshold_lines(ax, thresholds, orientation="horizontal", label_lines=False)
    add_horizontal_threshold_line_labels(ax, thresholds)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.90, loc="lower left")
    path = figures_dir / "rq2_attenuation_calibration_amplitude_by_distance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "figure": path.name,
        "relative_path": str(path.relative_to(figures_dir.parent)),
        "figure_type": "Calibration amplitude by distance",
        "description": "Known external fence-recorder hum amplitudes against nominal mouth-to-recorder distance; line shows median and IQR in 0.5 m bins.",
    }


def plot_calibration_by_receiver(measurements: pd.DataFrame, thresholds: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    subset = measurements.loc[measurements["calibration_class"].isin(["self_collar", "external_fence"])].copy()
    subset["plot_receiver"] = np.where(subset["calibration_class"] == "self_collar", "self collar", subset["receiver_id"].str.replace("fence_", "", regex=False))
    order = ["self collar", "1m", "2m", "4m", "8m"]
    data = [
        pd.to_numeric(subset.loc[subset["plot_receiver"] == receiver, PRIMARY_METRIC], errors="coerce").dropna().to_numpy()
        for receiver in order
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    parts = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for index, patch in enumerate(parts["boxes"]):
        patch.set_facecolor(FIGURE_COLORS["self_collar"] if index == 0 else "#f4a261")
        patch.set_alpha(0.75)
    add_threshold_lines(ax, thresholds, orientation="horizontal")
    ax.set_xlabel("Recording device distance to focal")
    ax.set_ylabel(AMPLITUDE_AXIS_LABEL)
    ax.grid(axis="y", alpha=0.2)
    path = figures_dir / "rq2_attenuation_calibration_by_receiver.png"
    fig.tight_layout()
    fig.savefig(path, dpi=CALIBRATION_BY_RECEIVER_DPI)
    plt.close(fig)
    return {
        "figure": path.name,
        "relative_path": str(path.relative_to(figures_dir.parent)),
        "figure_type": "Calibration by receiver",
        "description": "Self-collar and fence-recorder hum amplitude spread in the attenuation experiment.",
    }


def plot_rq2_distribution(events: pd.DataFrame, thresholds: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    values = pd.to_numeric(events[PRIMARY_METRIC], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.hist(values, bins=70, color="#64748b", alpha=0.72)
    add_threshold_lines(ax, thresholds)
    ax.set_title("Hum Amplitude Distribution With Attenuation Cutoffs")
    ax.set_xlabel(AMPLITUDE_AXIS_LABEL)
    ax.set_ylabel("Hum events per amplitude bin")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    path = figures_dir / "rq2_attenuation_rq2_amplitude_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "figure": path.name,
        "relative_path": str(path.relative_to(figures_dir.parent)),
        "figure_type": "RQ2 amplitude distribution",
        "description": "RQ2 complete-window hum amplitude distribution with attenuation-derived cutoffs.",
    }


def plot_removed_fraction_by_area(area_summary: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for threshold_id, subset in area_summary.groupby("threshold_id", sort=False):
        subset = subset.sort_values("area_decile")
        ax.plot(
            subset["area_decile"].astype(int),
            subset["removed_hum_fraction"],
            marker="o",
            linewidth=2.0,
            color=FIGURE_COLORS.get(str(threshold_id), "#111827"),
            label=threshold_display_label(subset.iloc[0]),
        )
    ax.set_title("Attenuation Filter: Proportion Of Hums Removed By Area")
    ax.set_xlabel("60 s window area decile, low to high area")
    ax.set_ylabel("Removed hum fraction")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"D{value}" for value in range(1, 11)])
    ax.set_ylim(0, min(1.0, max(0.1, float(area_summary["removed_hum_fraction"].max()) * 1.08)))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    path = figures_dir / "rq2_attenuation_removed_fraction_by_area.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "figure": path.name,
        "relative_path": str(path.relative_to(figures_dir.parent)),
        "figure_type": "Removed hum fraction by area decile",
        "description": "Proportion of primary hum events removed in each area decile.",
    }


def plot_retained_removed_counts_by_area(area_summary: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    threshold_order = [spec["threshold_id"] for spec in THRESHOLD_SPECS]
    fig, axes = plt.subplots(len(threshold_order), 1, figsize=(10.0, 9.0), sharex=True)
    for ax, threshold_id in zip(axes, threshold_order, strict=False):
        subset = area_summary.loc[area_summary["threshold_id"] == threshold_id].sort_values("area_decile")
        x = np.arange(len(subset.index))
        retained = subset["retained_hums"].to_numpy(dtype=float)
        removed = subset["removed_hums"].to_numpy(dtype=float)
        total = retained + removed
        ax.bar(x, retained, color=FIGURE_COLORS["retained"], label="retained")
        ax.bar(x, removed, bottom=retained, color=FIGURE_COLORS["removed"], label="removed")
        ax.set_ylabel(threshold_display_label(subset.iloc[0]) if not subset.empty else threshold_id)
        ax.grid(axis="y", alpha=0.2)
        if threshold_id == threshold_order[0]:
            ax.legend(frameon=False, ncol=2, loc="upper right")
        for index, row in enumerate(subset.itertuples(index=False)):
            ax.text(index, total[index] + max(total) * 0.02, f"{row.removed_hum_fraction:.0%}", ha="center", va="bottom", fontsize=8)
    axes[-1].set_xticks(np.arange(10))
    axes[-1].set_xticklabels([f"D{value}" for value in range(1, 11)])
    axes[-1].set_xlabel("60 s window area decile, low to high area")
    fig.suptitle("Attenuation Filter: Retained And Removed Hums By Area Decile", y=0.995)
    path = figures_dir / "rq2_attenuation_area_decile_counts.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "figure": path.name,
        "relative_path": str(path.relative_to(figures_dir.parent)),
        "figure_type": "Retained and removed hum counts by area decile",
        "description": "Stacked retained/removed hum counts by area decile; labels are removed fractions.",
    }


def plot_removed_fraction_by_session(session_summary: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    sessions = sorted(session_summary["day_session_id"].astype(str).unique())
    x = np.arange(len(sessions))
    offsets = np.linspace(-0.24, 0.24, len(THRESHOLD_SPECS))
    width = 0.18
    for offset, spec in zip(offsets, THRESHOLD_SPECS, strict=False):
        threshold_id = str(spec["threshold_id"])
        subset = session_summary.loc[session_summary["threshold_id"] == threshold_id].set_index("day_session_id").reindex(sessions)
        label = threshold_display_label(subset.iloc[0]) if not subset.empty else threshold_id
        ax.bar(x + offset, subset["removed_hum_fraction"], width=width, color=FIGURE_COLORS[threshold_id], label=label)
    ax.set_title("Attenuation Filter: Removed Hum Fraction By Session")
    ax.set_xlabel("Day session")
    ax.set_ylabel("Removed hum fraction")
    ax.set_xticks(x)
    ax.set_xticklabels([session.split(":")[0][5:] for session in sessions], rotation=45, ha="right")
    ax.set_ylim(0, min(1.0, max(0.1, float(session_summary["removed_hum_fraction"].max()) * 1.08)))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    path = figures_dir / "rq2_attenuation_removed_fraction_by_session.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "figure": path.name,
        "relative_path": str(path.relative_to(figures_dir.parent)),
        "figure_type": "Removed hum fraction by session",
        "description": "Session-level removal fraction for each attenuation-derived threshold.",
    }


def add_threshold_lines(ax: Any, thresholds: pd.DataFrame, *, orientation: str = "vertical", label_lines: bool = True) -> None:
    for threshold in thresholds.itertuples(index=False):
        threshold_id = str(threshold.threshold_id)
        line_kwargs = {
            "color": FIGURE_COLORS.get(threshold_id, "#111827"),
            "linewidth": 1.8,
            "linestyle": "--",
        }
        if label_lines:
            line_kwargs["label"] = f"{threshold_display_label(threshold)}: {threshold.cutoff_dbfs:.2f} dBFS"
        if orientation == "vertical":
            ax.axvline(float(threshold.cutoff_dbfs), **line_kwargs)
        elif orientation == "horizontal":
            ax.axhline(float(threshold.cutoff_dbfs), **line_kwargs)
        else:
            raise ValueError(f"unsupported threshold line orientation: {orientation}")


def add_threshold_line_labels(ax: Any, thresholds: pd.DataFrame) -> None:
    y_top = ax.get_ylim()[1]
    for threshold in thresholds.itertuples(index=False):
        threshold_id = str(threshold.threshold_id)
        label = threshold_display_label(threshold).replace("External-fence ", "")
        ax.text(
            float(threshold.cutoff_dbfs) + 0.18,
            y_top * 0.985,
            f"{label}\n{threshold.cutoff_dbfs:.2f} dBFS",
            rotation=90,
            ha="left",
            va="top",
            fontsize=9,
            color=FIGURE_COLORS.get(threshold_id, "#111827"),
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )


def add_horizontal_threshold_line_labels(ax: Any, thresholds: pd.DataFrame) -> None:
    x_max = ax.get_xlim()[1]
    for threshold in thresholds.itertuples(index=False):
        threshold_id = str(threshold.threshold_id)
        label = threshold_display_label(threshold).replace("External-fence ", "")
        ax.text(
            x_max - 0.08,
            float(threshold.cutoff_dbfs) + 0.25,
            f"{label} {threshold.cutoff_dbfs:.2f}",
            ha="right",
            va="bottom",
            fontsize=9,
            color=FIGURE_COLORS.get(threshold_id, "#111827"),
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        )


def threshold_display_label(threshold: Any) -> str:
    threshold_id = getattr(threshold, "threshold_id", "")
    if not threshold_id and isinstance(threshold, pd.Series):
        threshold_id = threshold.get("threshold_id", "")
    quantile = getattr(threshold, "external_fence_quantile", None)
    if quantile is None and isinstance(threshold, pd.Series):
        quantile = threshold.get("external_fence_quantile")
    if quantile is None or (isinstance(quantile, float) and not np.isfinite(quantile)):
        for spec in THRESHOLD_SPECS:
            if str(spec["threshold_id"]) == str(threshold_id):
                quantile = spec["external_fence_quantile"]
                break
    try:
        q_value = float(quantile) * 100.0
    except (TypeError, ValueError):
        q_value = float("nan")
    if np.isfinite(q_value):
        return f"External-fence Q{q_value:g}"
    return str(threshold_id).replace("attenuation_", "")


def write_reports(
    *,
    launch_dir: Path,
    attenuation_dir: Path,
    intermediate_events_path: Path,
    output_dir: Path,
    timestamp_events: pd.DataFrame,
    recorder_manifest: pd.DataFrame,
    calibration_measurements: pd.DataFrame,
    threshold_definitions: pd.DataFrame,
    input_summary: pd.DataFrame,
    area_summary: pd.DataFrame,
    session_summary: pd.DataFrame,
    recorder_summary: pd.DataFrame,
    event_count_audit: pd.DataFrame,
    figure_index: pd.DataFrame,
) -> None:
    calibration_summary = calibration_summary_tables(calibration_measurements)
    readme = readme_markdown(output_dir, threshold_definitions, input_summary, figure_index)
    methods = methods_markdown(
        launch_dir=launch_dir,
        attenuation_dir=attenuation_dir,
        intermediate_events_path=intermediate_events_path,
        output_dir=output_dir,
        timestamp_events=timestamp_events,
        recorder_manifest=recorder_manifest,
        calibration_measurements=calibration_measurements,
        calibration_summary=calibration_summary,
        threshold_definitions=threshold_definitions,
        input_summary=input_summary,
        area_summary=area_summary,
        session_summary=session_summary,
        recorder_summary=recorder_summary,
        event_count_audit=event_count_audit,
        figure_index=figure_index,
    )
    (output_dir / "README.md").write_text(readme, encoding="utf8")
    (output_dir / "METHODS.md").write_text(methods, encoding="utf8")


def run_attenuation_r_workflow(*, output_dir: Path, input_path: Path, nsim: int, seed: int) -> dict[str, Any]:
    script_path = Path(__file__).resolve().with_name("amplitude_threshold_glmmtmb_workflow.R")
    if not script_path.is_file():
        raise FileNotFoundError(f"missing R workflow: {script_path}")
    command = [
        "Rscript",
        str(script_path),
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--thresholds",
        ",".join(THRESHOLD_IDS),
        "--output-prefix",
        "rq2_attenuation",
        "--model-id-tag",
        "attenuation_amplitude",
        "--response-label",
        "attenuation_thresholded_hum_count",
        "--run-label",
        "attenuation-threshold",
        "--plot-title-prefix",
        "Hum-rate fixed effects after attenuation-calibrated amplitude filtering",
        "--nsim",
        str(nsim),
        "--seed",
        str(seed),
    ]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[4], text=True, capture_output=True, check=False)
    log_dir = output_dir / "cache"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "rq2_attenuation_stdout.log").write_text(completed.stdout, encoding="utf8")
    (log_dir / "rq2_attenuation_stderr.log").write_text(completed.stderr, encoding="utf8")
    if completed.returncode != 0:
        raise RuntimeError(f"attenuation-threshold RQ2 workflow failed with exit code {completed.returncode}; see {log_dir}")
    return {
        "script": str(script_path),
        "exit_code": int(completed.returncode),
        "stdout_log": "cache/rq2_attenuation_stdout.log",
        "stderr_log": "cache/rq2_attenuation_stderr.log",
        "dharma_simulations_per_model": int(nsim),
        "seed": int(seed),
    }


def write_model_fit_reports(
    *,
    launch_dir: Path,
    attenuation_dir: Path,
    output_dir: Path,
    threshold_definitions: pd.DataFrame,
    input_summary: pd.DataFrame,
    area_summary: pd.DataFrame,
    session_summary: pd.DataFrame,
    recorder_summary: pd.DataFrame,
    figure_index: pd.DataFrame,
) -> dict[str, Any]:
    run_manifest = pd.read_csv(output_dir / "tables" / "rq2_attenuation_model_run_manifest.csv")
    fixed_effects = pd.read_csv(output_dir / "tables" / "rq2_attenuation_fixed_effects.csv")
    dharma_standard = pd.read_csv(output_dir / "tables" / "rq2_attenuation_dharma_standard_tests.csv")
    dharma_temporal = pd.read_csv(output_dir / "tables" / "rq2_attenuation_dharma_temporal_autocorrelation.csv")
    raw_r_figures = pd.read_csv(output_dir / "tables" / "rq2_attenuation_raw_figure_index.csv")
    full_figure_index = pd.concat([figure_index, raw_r_figures], ignore_index=True, sort=False).fillna("")
    full_figure_index.to_csv(output_dir / "tables" / "rq2_attenuation_figure_index.csv", index=False)

    primary_run, primary_fixed = load_primary_models(launch_dir)
    primary_deltas = compare_to_primary(fixed_effects, primary_fixed)
    run_comparison = build_source_run_comparison(run_manifest, primary_run)
    key_comparison = build_source_key_effect_comparison(fixed_effects, primary_fixed)
    significance_changes = build_source_significance_changes(primary_deltas)

    primary_deltas.to_csv(output_dir / "tables" / "rq2_attenuation_vs_primary_effect_deltas.csv", index=False)
    run_comparison.to_csv(output_dir / "tables" / "rq2_attenuation_run_comparison.csv", index=False)
    key_comparison.to_csv(output_dir / "tables" / "rq2_attenuation_key_effect_comparison.csv", index=False)
    significance_changes.to_csv(output_dir / "tables" / "rq2_attenuation_significance_changes_vs_primary.csv", index=False)

    readme = readme_with_models_markdown(output_dir, threshold_definitions, input_summary, run_manifest, fixed_effects, dharma_standard, full_figure_index)
    methods_results = model_methods_results_markdown(
        launch_dir=launch_dir,
        attenuation_dir=attenuation_dir,
        output_dir=output_dir,
        threshold_definitions=threshold_definitions,
        input_summary=input_summary,
        area_summary=area_summary,
        session_summary=session_summary,
        recorder_summary=recorder_summary,
        run_manifest=run_manifest,
        fixed_effects=fixed_effects,
        dharma_standard=dharma_standard,
        dharma_temporal=dharma_temporal,
        primary_deltas=primary_deltas,
        run_comparison=run_comparison,
        figure_index=full_figure_index,
    )

    (output_dir / "README.md").write_text(readme, encoding="utf8")
    (output_dir / "reports" / "METHODS_RESULTS.md").write_text(methods_results, encoding="utf8")
    return {
        "attenuation_models": int(run_manifest.shape[0]),
        "comparison_rows": int(key_comparison.shape[0]),
        "figures_indexed": int(full_figure_index.shape[0]),
        "methods_results": str((output_dir / "reports" / "METHODS_RESULTS.md").resolve()),
    }


def build_source_run_comparison(attenuation_run: pd.DataFrame, primary_run: pd.DataFrame) -> pd.DataFrame:
    attenuation = attenuation_run.copy()
    attenuation["source"] = "attenuation_calibrated"
    primary = primary_run.copy()
    keep = [
        "source",
        "threshold_id",
        "model_id",
        "family",
        "fixed_surface",
        "n_rows",
        "n_sessions",
        "hum_count_total",
        "zero_hum_fraction",
        "fit_status",
        "AIC",
        "BIC",
        "dispersion_parameter",
    ]
    return pd.concat([primary[keep], attenuation[keep]], ignore_index=True, sort=False)


def build_source_key_effect_comparison(attenuation_fixed: pd.DataFrame, primary_fixed: pd.DataFrame) -> pd.DataFrame:
    attenuation = attenuation_fixed.copy()
    attenuation["source"] = "attenuation_calibrated"
    primary = primary_fixed.copy()
    rows = pd.concat([primary, attenuation], ignore_index=True, sort=False)
    rows = rows.loc[rows["term"].isin(KEY_TERMS)].copy()
    keep = [
        "source",
        "threshold_id",
        "model_id",
        "family",
        "fixed_surface",
        "term",
        "estimate",
        "std_error",
        "p_value",
        "incidence_rate_ratio",
        "irr_95_low",
        "irr_95_high",
    ]
    return rows[keep].sort_values(["family", "fixed_surface", "term", "source", "threshold_id"]).reset_index(drop=True)


def build_source_significance_changes(primary_deltas: pd.DataFrame) -> pd.DataFrame:
    rows = primary_deltas.copy()
    rows = rows.loc[rows["primary_p_value"].notna()].copy()
    rows["primary_significant_0_05"] = rows["primary_p_value"] < 0.05
    rows["attenuation_significant_0_05"] = rows["p_value"] < 0.05
    rows["significance_change_0_05"] = np.select(
        [
            rows["primary_significant_0_05"] & rows["attenuation_significant_0_05"],
            rows["primary_significant_0_05"] & ~rows["attenuation_significant_0_05"],
            ~rows["primary_significant_0_05"] & rows["attenuation_significant_0_05"],
        ],
        ["unchanged_significant", "lost_significance", "gained_significance"],
        default="unchanged_not_significant",
    )
    return rows[
        [
            "threshold_id",
            "model_id",
            "family",
            "fixed_surface",
            "term",
            "estimate",
            "p_value",
            "incidence_rate_ratio",
            "primary_estimate",
            "primary_p_value",
            "primary_irr",
            "estimate_delta_vs_primary",
            "p_value_delta_vs_primary",
            "irr_ratio_vs_primary",
            "direction_changed_vs_primary",
            "significance_change_0_05",
        ]
    ]


def readme_with_models_markdown(
    output_dir: Path,
    thresholds: pd.DataFrame,
    input_summary: pd.DataFrame,
    run_manifest: pd.DataFrame,
    fixed_effects: pd.DataFrame,
    dharma_standard: pd.DataFrame,
    figure_index: pd.DataFrame,
) -> str:
    threshold_cols = [
        "threshold_id",
        "external_fence_quantile",
        "cutoff_dbfs",
        "self_collar_retained_fraction",
        "external_fence_pass_fraction",
    ]
    input_cols = [
        "threshold_id",
        "thresholded_hum_count_total",
        "removed_hum_count_total",
        "retained_hum_fraction",
        "removed_hum_fraction",
        "zero_hum_fraction",
        "hum_windows_lost_all_hums",
    ]
    run_cols = [
        "threshold_id",
        "family",
        "fixed_surface",
        "hum_count_total",
        "zero_hum_fraction",
        "fit_status",
        "AIC",
        "dispersion_parameter",
    ]
    key = fixed_effects.loc[fixed_effects["term"].isin(["z_area", "z_spread_change", "z_area:z_spread_change"])].copy()
    key_cols = ["threshold_id", "family", "fixed_surface", "term", "incidence_rate_ratio", "p_value"]
    dharma_cols = ["threshold_id", "family", "fixed_surface", "test", "p_value", "status"]
    figure_cols = ["figure_type", "relative_path", "description"]
    return f"""# RQ2 Attenuation-Calibrated Amplitude Filter Bundle

This bundle uses fixed dBFS cutoffs derived from Stephan/Riley's 13 February 2025 attenuation experiment. The full fit surface is Poisson and NB1, each with interaction and additive fixed effects, for every attenuation threshold. DHARMa diagnostics were run for every fitted model.

## Threshold Values

{markdown_table(thresholds[threshold_cols])}

## Input Summary

{markdown_table(input_summary[input_cols])}

## Fit Summary

{markdown_table(run_manifest[run_cols])}

## Key Effects

{markdown_table(key[key_cols])}

## DHARMa Standard Tests

{markdown_table(dharma_standard[dharma_cols])}

## Figures

{markdown_table(figure_index[figure_cols])}

Full reporting: `reports/METHODS_RESULTS.md`

No-threshold comparison tables are in `tables/rq2_attenuation_vs_primary_effect_deltas.csv`, `tables/rq2_attenuation_key_effect_comparison.csv`, and `tables/rq2_attenuation_significance_changes_vs_primary.csv`.

Output bundle: `{output_dir.resolve()}`
"""


def model_methods_results_markdown(
    *,
    launch_dir: Path,
    attenuation_dir: Path,
    output_dir: Path,
    threshold_definitions: pd.DataFrame,
    input_summary: pd.DataFrame,
    area_summary: pd.DataFrame,
    session_summary: pd.DataFrame,
    recorder_summary: pd.DataFrame,
    run_manifest: pd.DataFrame,
    fixed_effects: pd.DataFrame,
    dharma_standard: pd.DataFrame,
    dharma_temporal: pd.DataFrame,
    primary_deltas: pd.DataFrame,
    run_comparison: pd.DataFrame,
    figure_index: pd.DataFrame,
) -> str:
    threshold_cols = [
        "threshold_id",
        "external_fence_quantile",
        "cutoff_dbfs",
        "external_fence_measurements",
        "external_fence_pass_fraction",
        "self_collar_measurements",
        "self_collar_retained_fraction",
    ]
    input_cols = [
        "threshold_id",
        "primary_hum_count_total",
        "thresholded_hum_count_total",
        "removed_hum_count_total",
        "retained_hum_fraction",
        "removed_hum_fraction",
        "zero_hum_fraction",
        "hum_windows_lost_all_hums",
    ]
    area_cols = [
        "threshold_id",
        "area_label",
        "area_median_m2",
        "primary_hums",
        "retained_hums",
        "removed_hums",
        "removed_hum_fraction",
        "hum_windows_lost_all_hums",
    ]
    session_cols = ["threshold_id", "day_session_id", "primary_hums", "retained_hums", "removed_hum_fraction", "hum_windows_lost_all_hums"]
    recorder_cols = ["threshold_id", "recorder_display_name", "primary_hums", "retained_hums", "removed_hum_fraction"]
    run_cols = ["threshold_id", "family", "fixed_surface", "hum_count_total", "zero_hum_fraction", "fit_status", "AIC", "BIC", "dispersion_parameter"]
    key_effect_cols = [
        "threshold_id",
        "family",
        "fixed_surface",
        "term",
        "estimate",
        "p_value",
        "incidence_rate_ratio",
        "irr_95_low",
        "irr_95_high",
    ]
    key_effects = fixed_effects.loc[fixed_effects["term"].isin(KEY_TERMS)].copy()
    dharma_cols = ["threshold_id", "family", "fixed_surface", "test", "statistic_name", "statistic", "estimate", "p_value", "status"]
    temporal_summary = (
        dharma_temporal.groupby(["threshold_id", "family", "fixed_surface", "status"], dropna=False)
        .agg(
            session_tests=("p_value", "size"),
            median_temporal_p_value=("p_value", "median"),
            min_temporal_p_value=("p_value", "min"),
            max_temporal_p_value=("p_value", "max"),
        )
        .reset_index()
    )
    primary_effects = primary_deltas.loc[primary_deltas["term"].isin(KEY_TERMS)].copy()
    primary_cols = [
        "threshold_id",
        "family",
        "fixed_surface",
        "term",
        "estimate",
        "p_value",
        "incidence_rate_ratio",
        "primary_estimate",
        "primary_p_value",
        "primary_irr",
        "estimate_delta_vs_primary",
        "p_value_delta_vs_primary",
        "direction_changed_vs_primary",
    ]

    return f"""# Methods And Results: RQ2 Attenuation-Calibrated Amplitude Threshold

## Scope

This artifact fits the attenuation-calibrated amplitude-threshold responses for the RQ2 60 s hum-count dataset. The filter is applied only to the hum-count response. Area, spread-change, speed, centroid-change, exposure, session grouping, and 60 s window eligibility are unchanged from the primary no-threshold RQ2 launch.

The threshold derivation is documented in `METHODS.md`; this report documents the model fits and diagnostics.

## Inputs

- Primary RQ2 launch: `{launch_dir.resolve()}`
- Attenuation bundle: `{attenuation_dir.resolve()}`
- Output bundle: `{output_dir.resolve()}`
- Materialized model rows: {int(input_summary["rows"].sum())} threshold-specific rows, representing {int(input_summary["rows"].iloc[0]) if not input_summary.empty else 0} unique 60 s windows per threshold.

## Threshold Values

{markdown_table(threshold_definitions[threshold_cols])}

## Applied Response Summary

For each threshold, retained event counts are aggregated back to the original 60 s windows. The response `hum_count` is replaced by the retained hum count. `primary_hum_count` preserves the original unfiltered response, and `hum_count_removed_by_threshold` records the difference.

{markdown_table(input_summary[input_cols])}

## Area-Decile Exclusion Summary

`D1` is the lowest `area_m2` decile across all 60 s model windows; `D10` is the highest. `removed_hum_fraction` is `removed_hums / primary_hums` within each area decile.

{markdown_table(area_summary[area_cols])}

## Session Exclusion Summary

{markdown_table(session_summary[session_cols])}

## Recorder Exclusion Summary

{markdown_table(recorder_summary[recorder_cols], max_rows=45)}

## Model Surface

For each attenuation threshold, four `glmmTMB` models were fitted locally:

- `Poisson interaction`: `hum_count ~ z_area * z_spread_change + z_speed + z_centroid_change + offset(log(window_length_minutes)) + (1 | session_id)`.
- `Poisson additive`: the same model without `z_area:z_spread_change`.
- `NB1 interaction`: the interaction fixed surface with `glmmTMB::nbinom1(link = "log")`.
- `NB1 additive`: the additive fixed surface with `glmmTMB::nbinom1(link = "log")`.

Predictors were standardized within each threshold-specific model input to one SD units. Coefficient p-values are conditional-model Wald z tests from `glmmTMB`.

## Fit Summary

AIC/BIC are useful within a fixed response definition and family context, but should not be overinterpreted across different thresholded responses because the response itself changes.

{markdown_table(run_manifest[run_cols])}

## Fixed Effects

{markdown_table(key_effects[key_effect_cols])}

## DHARMa Diagnostics

DHARMa residual diagnostics were run for every fitted model. Standard checks used `testUniformity`, `testDispersion`, `testZeroInflation`, and bootstrap `testOutliers`. Temporal autocorrelation was evaluated separately within each session using `window_start_seconds_local` as the time coordinate.

{markdown_table(dharma_standard[dharma_cols])}

## Temporal Autocorrelation Summary

{markdown_table(temporal_summary)}

## Primary No-Threshold Comparison Table

This is a mechanical coefficient comparison against the primary no-threshold RQ2 fits.

{markdown_table(primary_effects[primary_cols])}

## Run Comparison Table

{markdown_table(run_comparison)}

## Figures

{markdown_table(figure_index[["figure_type", "relative_path", "description"]])}

## Reproducibility

- Python module: `backend/src/stv_analysis/rq2_hum_timing_group_dispersion/amplitude_attenuation_filter.py`
- Shared R workflow: `backend/src/stv_analysis/rq2_hum_timing_group_dispersion/amplitude_threshold_glmmtmb_workflow.R`
- Model input: `inputs/rq2_attenuation_filter_model_input_60s.csv`
- Event flags: `tables/rq2_attenuation_event_flags.parquet`
- Fixed effects: `tables/rq2_attenuation_fixed_effects.csv`
- DHARMa tests: `tables/rq2_attenuation_dharma_standard_tests.csv`
- Temporal autocorrelation: `tables/rq2_attenuation_dharma_temporal_autocorrelation.csv`
"""


def markdown_table(frame: pd.DataFrame, *, floatfmt: str = ".4f", max_rows: int | None = None) -> str:
    data = frame if max_rows is None else frame.head(max_rows)
    if data.empty:
        return "(empty)"
    return data.to_markdown(index=False, floatfmt=floatfmt)


def readme_markdown(output_dir: Path, thresholds: pd.DataFrame, input_summary: pd.DataFrame, figure_index: pd.DataFrame) -> str:
    threshold_cols = [
        "threshold_id",
        "external_fence_quantile",
        "cutoff_dbfs",
        "self_collar_retained_fraction",
        "external_fence_pass_fraction",
    ]
    input_cols = [
        "threshold_id",
        "thresholded_hum_count_total",
        "removed_hum_count_total",
        "retained_hum_fraction",
        "removed_hum_fraction",
        "zero_hum_fraction",
        "hum_windows_lost_all_hums",
    ]
    return f"""# RQ2 Attenuation-Calibrated Amplitude Filter

This bundle uses fixed dBFS cutoffs derived from Stephan/Riley's 13 February 2025 attenuation experiment. The base attenuation data views are documented here; when model fits are generated, `README.md` is refreshed with fit summaries and `reports/METHODS_RESULTS.md` contains the model report.

## Threshold Values

{markdown_table(thresholds[threshold_cols])}

## RQ2 Input Effect

{markdown_table(input_summary[input_cols])}

## Figures

{markdown_table(figure_index[["figure_type", "relative_path", "description"]])}

Full method details: `METHODS.md`

Output bundle: `{output_dir.resolve()}`
"""


def methods_markdown(
    *,
    launch_dir: Path,
    attenuation_dir: Path,
    intermediate_events_path: Path,
    output_dir: Path,
    timestamp_events: pd.DataFrame,
    recorder_manifest: pd.DataFrame,
    calibration_measurements: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    threshold_definitions: pd.DataFrame,
    input_summary: pd.DataFrame,
    area_summary: pd.DataFrame,
    session_summary: pd.DataFrame,
    recorder_summary: pd.DataFrame,
    event_count_audit: pd.DataFrame,
    figure_index: pd.DataFrame,
) -> str:
    primary_events = timestamp_events.loc[timestamp_events["is_primary_calibration_event"].astype(bool)]
    hum_annotations = timestamp_events.loc[timestamp_events["is_hum_annotation"].astype(bool)]
    event_count_max_abs_delta = float(event_count_audit["count_delta_primary_minus_attenuation_events"].abs().max())
    threshold_cols = [
        "threshold_id",
        "external_fence_quantile",
        "cutoff_dbfs",
        "external_fence_measurements",
        "external_fence_pass_fraction",
        "self_collar_measurements",
        "self_collar_retained_fraction",
    ]
    input_cols = [
        "threshold_id",
        "primary_hum_count_total",
        "thresholded_hum_count_total",
        "removed_hum_count_total",
        "retained_hum_fraction",
        "removed_hum_fraction",
        "zero_hum_fraction",
        "hum_windows_lost_all_hums",
    ]
    area_cols = [
        "threshold_id",
        "area_label",
        "area_median_m2",
        "primary_hums",
        "retained_hums",
        "removed_hums",
        "removed_hum_fraction",
    ]
    return f"""# METHODS: RQ2 Attenuation-Calibrated Amplitude Filter

## Scope

This artifact creates an attenuation-calibrated amplitude filter for the RQ2 60 s hum-count dataset using known self-collar hums and simultaneous known external fence-recorder captures from the attenuation experiment.

The amplitude filter is materialized first as thresholded data views and exclusion/effect summaries. The same artifact can then fit the Poisson and NB1 RQ2 model surface; those fit results and DHARMa diagnostics are reported separately in `reports/METHODS_RESULTS.md`.

## Inputs

- Primary RQ2 launch: `{launch_dir.resolve()}`
- Rich event table used if cached RQ2 event metrics are unavailable: `{intermediate_events_path.resolve()}`
- Attenuation bundle: `{attenuation_dir.resolve()}`
- Output bundle: `{output_dir.resolve()}`

The RQ2 deployment event metric remains `{PRIMARY_METRIC}`: the 95th percentile of short-time RMS after 220-5000 Hz bandpass filtering, expressed in dBFS. Keeping the same metric lets the attenuation-derived cutoffs be applied directly to the existing deployment hum-event table.

## Attenuation Experiment Interpretation

The attenuation experiment used AudioMoth-GPS-Sync recorders at gain 3 and 48 kHz, matching the deployment configuration. Five fence recorders were placed on the marked line at 0.5, 1, 2, 4, and 8 m from a zero point, and three focal animals wore collar recorders. Observer timestamps recorded hum time, mouth distance from the zero point, and for the second and third focal animals, facing direction relative to the recorder line.

The primary calibration uses focal animals 7026 and 7460 only. This follows Stephan's note that animal 1 should not be the starting point because facing direction was not recorded and the distance convention changed during that block. Animal 7935 is parsed and documented but excluded from threshold derivation.

The 0.5 m fence recorder is also excluded from threshold derivation because the method file and AudioMoth log document microphone-change failures. It is retained as diagnostic metadata but not used in the external-fence distribution.

## Timestamp Alignment

The phone-app timestamp marks the observer event time, but a direct check of the self-collar audio showed that the strongest hum frame was usually before that timestamp. For each primary calibration hum, the code searched the source animal's own collar recording from {SEARCH_PRE_S:.1f} s before to {SEARCH_POST_S:.1f} s after the observer timestamp and set the refined hum time to the maximum hum frame RMS in that search window.

Amplitude was then measured on every eligible receiver in a {MEASUREMENT_PRE_S + MEASUREMENT_POST_S:.1f} s window centered on the refined self-collar hum time. This keeps source loudness fixed within event and compares known self-collar reception against simultaneous external fence-recorder reception.

Refined timestamp deltas from the observer timestamp, in seconds:

{markdown_table(calibration_measurements.loc[calibration_measurements["calibration_class"] == "self_collar", ["refined_delta_from_observer_timestamp_s"]].describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).T.reset_index().rename(columns={"index": "metric"}))}

## Calibration Counts

- Hum annotation rows parsed: {len(hum_annotations)}
- Primary calibration hums used for thresholds: {len(primary_events)}
- Calibration measurements materialized: {len(calibration_measurements)}

Calibration measurement summary:

{markdown_table(calibration_summary)}

## Threshold Rule

Thresholds are upper-tail quantiles of the known external fence-recorder amplitude distribution from the primary calibration subset. A deployment hum is retained when `{PRIMARY_METRIC} >= cutoff_dbfs`.

This rule treats the attenuation experiment as a caller-proximity calibration, not as a perfect caller-identity classifier. Very close non-collared animals can still be loud enough to pass. The filter is intended to reduce detections whose received level is typical of known off-collar/fence recordings, while retaining most known self-collar hums in the calibration data.

The threshold tiers are:

- `attenuation_lenient`: external-fence q90; allows about 10% of known external fence measurements to pass.
- `attenuation_primary`: external-fence q95; allows about 5% of known external fence measurements to pass.
- `attenuation_conservative`: external-fence q97.5; allows about 2.5% of known external fence measurements to pass.

{markdown_table(threshold_definitions[threshold_cols])}

## Application To RQ2

The threshold is applied only to the hum-count response. Window eligibility, session grouping, area, spread change, speed, centroid displacement, exposure, and all other RQ2 predictors remain unchanged from the primary 60 s launch.

For each threshold, retained event counts are aggregated back to the original 60 s windows. The output keeps the original response in `primary_hum_count`, replaces `hum_count` with the retained count, and records `hum_count_removed_by_threshold`.

The event-count audit maximum absolute difference between primary `hum_count` and complete-window event rows is {event_count_max_abs_delta:.0f}.

{markdown_table(input_summary[input_cols])}

## Area-Decile Effect

`D1` is the lowest 60 s `area_m2` decile and `D10` is the highest. The table reports removal fractions by decile before any model fitting.

{markdown_table(area_summary[area_cols], max_rows=30)}

## Session Effect

{markdown_table(session_summary[["threshold_id", "day_session_id", "primary_hums", "retained_hums", "removed_hum_fraction", "hum_windows_lost_all_hums"]])}

## Recorder Effect

{markdown_table(recorder_summary[["threshold_id", "recorder_display_name", "primary_hums", "retained_hums", "removed_hum_fraction"]], max_rows=45)}

## Figures

{markdown_table(figure_index[["figure_type", "relative_path", "description"]])}

## Online Method Check

The design follows the general passive-acoustic monitoring logic that recorder detection distances should be empirically evaluated rather than assumed. Haupert et al. describe propagation experiments and level-difference approaches for distance-dependent attenuation, and Winiarska et al. emphasize that detection range is central for interpreting autonomous recorder data. Lapp et al. show that AudioMoth frequency response and orientation can affect recorded levels, which is why this filter uses the experiment's observed orientation mix instead of an idealized free-field attenuation curve.

Sources checked:

- Haupert et al., physics-based detection distance and sound attenuation protocol: https://arxiv.org/abs/2211.16077
- Winiarska et al., detection ranges for passive acoustic monitoring: https://www.nature.com/articles/s41598-024-51297-z
- Lapp, Stahlman and Kitzes, AudioMoth performance evaluation: https://doi.org/10.3390/s23115254

## Reproducibility

- Python module: `backend/src/stv_analysis/rq2_hum_timing_group_dispersion/amplitude_attenuation_filter.py`
- Timestamp event table: `tables/attenuation_timestamp_events.csv`
- Calibration measurements: `tables/attenuation_calibration_measurements.parquet`
- RQ2 event flags: `tables/rq2_attenuation_event_flags.parquet`
- Thresholded 60 s input: `inputs/rq2_attenuation_filter_model_input_60s.parquet`
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RQ2 attenuation-calibrated amplitude-filter data views and optionally fit Poisson/NB1 models.")
    parser.add_argument("--launch-dir", type=Path, required=True)
    parser.add_argument("--attenuation-dir", type=Path, default=DEFAULT_ATTENUATION_DIR)
    parser.add_argument("--intermediate-events-path", type=Path, default=DEFAULT_INTERMEDIATE_EVENTS)
    parser.add_argument("--event-metrics-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-fits", action="store_true", help="Only materialize the attenuation-filter data views and base methods report.")
    parser.add_argument("--nsim", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    launch_dir = args.launch_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (launch_dir / DEFAULT_OUTPUT_SUBDIR).resolve()
    manifest = run_attenuation_filter_analysis(
        launch_dir=launch_dir,
        attenuation_dir=args.attenuation_dir.resolve(),
        intermediate_events_path=args.intermediate_events_path.resolve(),
        output_dir=output_dir,
        event_metrics_path=args.event_metrics_path.resolve() if args.event_metrics_path is not None else None,
        fit_models=not args.skip_fits,
        nsim=args.nsim,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": manifest["output_dir"], "models_fit": manifest["models_fit"], "fit_reporting": manifest["fit_reporting"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

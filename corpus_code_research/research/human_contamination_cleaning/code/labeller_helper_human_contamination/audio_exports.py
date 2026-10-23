from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .bundle_audio import ConcatWavReader, read_bundle_audio_info
from .manifest_utils import json_ready, write_json

OUTPUT_ROUND_DECIMALS = 6
ALL_EXCLUDED_EXPORT_NAME = "all_excluded_from_working_core"
MANUAL_SEEDS_EXPORT_NAME = "manual_human_seeds"
SAME_SOURCE_EXPORT_NAME = "same_source_as_manual_human"
ACOUSTIC_SIMILAR_EXPORT_NAME = "acoustically_similar_to_manual_human"


@dataclass(frozen=True)
class AudioExportSummary:
    subset_name: str
    description: str
    row_count: int
    duration_s: float
    gap_duration_s: float
    wav_path: Path
    selection_table_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class AudioExportSubsetSpec:
    subset_name: str
    description: str
    export_df: pd.DataFrame
    extra_selection_columns: tuple[str, ...] = ()


def _pcm16_bytes(samples: np.ndarray) -> bytes:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    return pcm.tobytes()


def _selection_table_rows(
    export_df: pd.DataFrame,
    *,
    gap_duration_s: float,
    extra_selection_columns: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    unique_extra_columns = [column for column in dict.fromkeys(extra_selection_columns) if column]
    cursor_s = 0.0
    for export_index, record in enumerate(export_df.to_dict(orient="records"), start=1):
        duration_s = float(record["concat_end_s"]) - float(record["concat_begin_s"])
        begin_s = round(cursor_s, OUTPUT_ROUND_DECIMALS)
        end_s = round(begin_s + duration_s, OUTPUT_ROUND_DECIMALS)
        row = {
            "Selection": export_index,
            "View": record["View"],
            "Channel": record["Channel"],
            "Begin Time (s)": begin_s,
            "End Time (s)": end_s,
            "Low Freq (Hz)": float(record["Low Freq (Hz)"]),
            "High Freq (Hz)": float(record["High Freq (Hz)"]),
            "review_label": record.get("review_label", ""),
            "primary_quarantine_reason": record.get("primary_quarantine_reason", ""),
            "quarantine_reason_tags": record.get("quarantine_reason_tags", ""),
            "original_selection": int(record["selection"]),
            "review_row_key": record["review_row_key"],
            "wav_basename": record["wav_basename"],
            "source_audio_file": record["source_audio_file"],
            "source_start_s": float(record["source_start_s"]),
            "source_end_s": float(record["source_end_s"]),
            "source_detection_rank": int(record.get("source_detection_rank", 0)),
            "detection_score": float(record.get("detection_score", 0.0)),
            "original_concat_begin_s": float(record["concat_begin_s"]),
            "original_concat_end_s": float(record["concat_end_s"]),
        }
        for column in unique_extra_columns:
            if column not in row:
                row[column] = record.get(column, "")
        rows.append(row)
        cursor_s = end_s
        if export_index < len(export_df.index):
            cursor_s = round(cursor_s + gap_duration_s, OUTPUT_ROUND_DECIMALS)
    return rows, round(cursor_s, OUTPUT_ROUND_DECIMALS)


def _write_subset_export(
    export_root: Path,
    export_df: pd.DataFrame,
    *,
    subset_name: str,
    description: str,
    gap_duration_s: float,
    review_bundle: Path,
    extra_selection_columns: Iterable[str] = (),
) -> AudioExportSummary:
    subset_dir = export_root / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    audio_info = read_bundle_audio_info(review_bundle)
    gap_frame_count = int(round(gap_duration_s * audio_info.sample_rate_hz))
    actual_gap_duration_s = (
        round(gap_frame_count / audio_info.sample_rate_hz, OUTPUT_ROUND_DECIMALS) if audio_info.sample_rate_hz else 0.0
    )
    gap_bytes = _pcm16_bytes(np.zeros(gap_frame_count, dtype=np.float32))

    ordered_df = export_df.sort_values(["selection", "review_row_key"], kind="stable").reset_index(drop=True)
    selection_rows, export_duration_s = _selection_table_rows(
        ordered_df,
        gap_duration_s=actual_gap_duration_s,
        extra_selection_columns=extra_selection_columns,
    )
    selection_table_df = pd.DataFrame(selection_rows)

    wav_path = subset_dir / "review_concat.wav"
    with ConcatWavReader(audio_info) as reader, wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(audio_info.sample_rate_hz)
        for export_index, row in enumerate(ordered_df.itertuples(index=False), start=1):
            samples = reader.read_mono(float(row.concat_begin_s), float(row.concat_end_s))
            handle.writeframes(_pcm16_bytes(samples))
            if export_index < len(ordered_df.index) and gap_frame_count > 0:
                handle.writeframes(gap_bytes)

    selection_table_path = subset_dir / "review_selection_table.txt"
    selection_table_df.to_csv(selection_table_path, sep="\t", index=False)

    manifest_path = subset_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_name": "human_contamination_audio_export",
        "subset_name": subset_name,
        "description": description,
        "row_count": int(len(ordered_df.index)),
        "sample_rate_hz": int(audio_info.sample_rate_hz),
        "gap_duration_s": actual_gap_duration_s,
        "duration_s": export_duration_s,
        "source_bundle_dir": str(review_bundle.resolve()),
        "wav_path": str(wav_path.resolve()),
        "selection_table_path": str(selection_table_path.resolve()),
        "rows": [{key: json_ready(value) for key, value in record.items()} for record in selection_rows],
    }
    write_json(manifest_path, manifest)

    return AudioExportSummary(
        subset_name=subset_name,
        description=description,
        row_count=int(len(ordered_df.index)),
        duration_s=export_duration_s,
        gap_duration_s=actual_gap_duration_s,
        wav_path=wav_path,
        selection_table_path=selection_table_path,
        manifest_path=manifest_path,
    )


def write_named_subset_audio_exports(
    review_bundle: Path,
    export_root: Path,
    subsets: Iterable[AudioExportSubsetSpec],
    *,
    gap_duration_s: float,
) -> list[AudioExportSummary]:
    if gap_duration_s < 0.0:
        raise ValueError(f"gap_duration_s must be >= 0.0, got {gap_duration_s}")
    summaries: list[AudioExportSummary] = []
    for subset in subsets:
        subset_df = subset.export_df.copy()
        if subset_df.empty:
            continue
        summaries.append(
            _write_subset_export(
                export_root,
                subset_df,
                subset_name=subset.subset_name,
                description=subset.description,
                gap_duration_s=gap_duration_s,
                review_bundle=review_bundle,
                extra_selection_columns=subset.extra_selection_columns,
            )
        )
    return summaries


def write_quarantine_audio_exports(
    review_bundle: Path,
    export_root: Path,
    assessment_df: pd.DataFrame,
    *,
    gap_duration_s: float,
) -> list[AudioExportSummary]:
    return write_named_subset_audio_exports(
        review_bundle,
        export_root,
        [
            AudioExportSubsetSpec(
                subset_name=ALL_EXCLUDED_EXPORT_NAME,
                description="All detections excluded from the working clean core.",
                export_df=assessment_df.loc[~assessment_df["working_core_keep"]].copy(),
            ),
            AudioExportSubsetSpec(
                subset_name=MANUAL_SEEDS_EXPORT_NAME,
                description="Manual human-speaking contamination seed detections only.",
                export_df=assessment_df.loc[assessment_df["is_manual_human_contamination"]].copy(),
            ),
            AudioExportSubsetSpec(
                subset_name=SAME_SOURCE_EXPORT_NAME,
                description="All detections that come from a source WAV containing at least one manual human seed. Overlaps are allowed.",
                export_df=assessment_df.loc[assessment_df["same_source_as_manual_human"]].copy(),
            ),
            AudioExportSubsetSpec(
                subset_name=ACOUSTIC_SIMILAR_EXPORT_NAME,
                description="Detections tagged as acoustically similar to the manual human seeds. Overlaps with same-source detections are allowed.",
                export_df=assessment_df.loc[
                    assessment_df["is_feature_neighbor_candidate"] & ~assessment_df["is_manual_human_contamination"]
                ].copy(),
            ),
        ],
        gap_duration_s=gap_duration_s,
    )

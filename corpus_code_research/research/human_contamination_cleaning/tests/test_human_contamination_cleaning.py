from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import pandas as pd

from labeller_helper_human_contamination.pipeline import (
    HumanContaminationCleaningConfig,
    run_human_contamination_cleaning,
)


def _write_bundle(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    review_manifest = {
        "schema_version": 1,
        "package_format": "prediction_review_raven_bundle_v2",
        "session_id": "session-1",
        "prediction_run_id": "run-1",
        "editable_table_filename": "review_selection_table.txt",
        "metadata_table_filename": "review_selection_table_metadata.tsv",
        "row_count": 4,
    }
    (bundle_dir / "review_import_manifest.json").write_text(
        json.dumps(review_manifest, indent=2) + "\n", encoding="utf8"
    )
    (bundle_dir / "review_selection_table.txt").write_text(
        "\n".join(
            [
                "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\treview_label",
                "1\t1\t1\t0.0\t0.5\t0\t4000\t",
                "2\t1\t1\t0.5\t1.0\t0\t4000\t",
                "3\t1\t1\t1.0\t1.5\t0\t4000\t",
                "4\t1\t1\t1.5\t2.0\t0\t4000\t",
            ]
        )
        + "\n",
        encoding="utf8",
    )
    (bundle_dir / "review_selection_table_metadata.tsv").write_text(
        "\n".join(
            [
                "Selection\tsession_id\tsession_item_id\tcanonical_detection_id\tcanonical_state_version\titem_index\tdetection_index\tsource_audio_file\tsource_start_s\tsource_end_s\tdetection_score",
                "1\tsession-1\titem-1\tcd-1\t1\t0\t0\t/source_a.wav\t60.0\t60.5\t0.6",
                "2\tsession-1\titem-2\tcd-2\t1\t1\t1\t/source_a.wav\t120.0\t120.5\t0.7",
                "3\tsession-1\titem-3\tcd-3\t1\t2\t2\t/source_b.wav\t180.0\t180.5\t0.8",
                "4\tsession-1\titem-4\tcd-4\t1\t3\t3\t/source_b.wav\t240.0\t240.5\t0.55",
            ]
        )
        + "\n",
        encoding="utf8",
    )


def _write_split_review_dir(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    selection_table_path = split_dir / "review_concat_part_01_of_05_selection_table.txt"
    selection_table_path.write_text(
        "\n".join(
            [
                "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tDelta Time (s)\tDelta Freq (Hz)\tAvg Power Density (dB FS/Hz)\treview_label",
                "1\tSpectrogram 1\t1\t0.0\t0.5\t0\t4000\t0.5\t4000\t-70\t0",
                "2\tSpectrogram 1\t1\t0.5\t1.0\t0\t4000\t0.5\t4000\t-71\t",
                "3\tSpectrogram 1\t1\t1.0\t1.5\t0\t4000\t0.5\t4000\t-72\t",
                "4\tSpectrogram 1\t1\t1.5\t2.0\t0\t4000\t0.5\t4000\t-73\t",
            ]
        )
        + "\n",
        encoding="utf8",
    )
    split_manifest = {
        "boundaries_s": [0.0, 2.0],
        "parts": 1,
        "outputs": [
            {
                "part": 1,
                "start_s_original": 0.0,
                "end_s_original": 2.0,
                "selection_rows": 4,
                "selection_table": str(selection_table_path.resolve()),
                "wav": str((split_dir / "review_concat_part_01_of_05.wav").resolve()),
            }
        ],
    }
    (split_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf8")


def _write_review_concat_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate_hz = 48000
    duration_s = 2.0
    frame_count = int(sample_rate_hz * duration_s)
    samples = bytearray()
    for frame_index in range(frame_count):
        amplitude = int(12000 * math.sin(2.0 * math.pi * 440.0 * (frame_index / sample_rate_hz)))
        samples.extend(int(amplitude).to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(bytes(samples))


def _write_artifacts(artifacts_root: Path, bundle_name: str) -> None:
    run_root = artifacts_root / bundle_name
    backfill_dir = run_root / "metadata_backfill"
    analysis_dir = run_root / "acoustic_analysis" / "tables"
    backfill_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    enriched = pd.DataFrame(
        [
            {
                "Selection": "1",
                "selection": 1,
                "review_row_key": "row-1",
                "source_audio_file": "/source_a.wav",
                "wav_basename": "source_a.wav",
                "audio_collection_name": "collection_a",
                "device_id": "device_a",
                "source_start_s": 60.0,
                "source_end_s": 60.5,
                "sample_rate_estimate": 48000.0,
                "total_samples": 48000.0 * 600.0,
                "detection_score": 0.6,
                "concat_begin_s": 0.0,
                "concat_end_s": 0.5,
                "Low Freq (Hz)": 0.0,
                "High Freq (Hz)": 4000.0,
                "View": 1,
                "Channel": 1,
                "review_label": "",
            },
            {
                "Selection": "2",
                "selection": 2,
                "review_row_key": "row-2",
                "source_audio_file": "/source_a.wav",
                "wav_basename": "source_a.wav",
                "audio_collection_name": "collection_a",
                "device_id": "device_a",
                "source_start_s": 120.0,
                "source_end_s": 120.5,
                "sample_rate_estimate": 48000.0,
                "total_samples": 48000.0 * 600.0,
                "detection_score": 0.7,
                "concat_begin_s": 0.5,
                "concat_end_s": 1.0,
                "Low Freq (Hz)": 0.0,
                "High Freq (Hz)": 4000.0,
                "View": 1,
                "Channel": 1,
                "review_label": "",
            },
            {
                "Selection": "3",
                "selection": 3,
                "review_row_key": "row-3",
                "source_audio_file": "/source_b.wav",
                "wav_basename": "source_b.wav",
                "audio_collection_name": "collection_b",
                "device_id": "device_b",
                "source_start_s": 180.0,
                "source_end_s": 180.5,
                "sample_rate_estimate": 48000.0,
                "total_samples": 48000.0 * 600.0,
                "detection_score": 0.8,
                "concat_begin_s": 1.0,
                "concat_end_s": 1.5,
                "Low Freq (Hz)": 0.0,
                "High Freq (Hz)": 4000.0,
                "View": 1,
                "Channel": 1,
                "review_label": "",
            },
            {
                "Selection": "4",
                "selection": 4,
                "review_row_key": "row-4",
                "source_audio_file": "/source_b.wav",
                "wav_basename": "source_b.wav",
                "audio_collection_name": "collection_b",
                "device_id": "device_b",
                "source_start_s": 240.0,
                "source_end_s": 240.5,
                "sample_rate_estimate": 48000.0,
                "total_samples": 48000.0 * 600.0,
                "detection_score": 0.55,
                "concat_begin_s": 1.5,
                "concat_end_s": 2.0,
                "Low Freq (Hz)": 0.0,
                "High Freq (Hz)": 4000.0,
                "View": 1,
                "Channel": 1,
                "review_label": "",
            },
        ]
    )
    enriched.to_parquet(backfill_dir / "enriched_detections.parquet", index=False)

    analysis_rows = enriched.copy()
    analysis_rows["feature_status"] = "ok"
    analysis_rows["skip_reason"] = ""
    analysis_rows["audio_source_strategy"] = "review_concat_wav"
    analysis_rows["snippet_relpath"] = ""
    analysis_rows["cluster_id"] = -1
    analysis_rows["is_noise"] = True
    analysis_rows["membership_strength"] = 0.0
    analysis_rows["pca_x"] = 0.0
    analysis_rows["pca_y"] = 0.0
    analysis_rows.to_parquet(analysis_dir / "analysis_rows.parquet", index=False)

    features = pd.DataFrame(
        [
            {
                "review_row_key": "row-1",
                "selection": 1,
                "snippet_relpath": "",
                "duration_s": 0.5,
                "rms": 0.10,
                "zero_crossing_rate": 0.10,
                "spectral_centroid_mean": 100.0,
                "spectral_centroid_std": 10.0,
                "spectral_bandwidth_mean": 200.0,
                "spectral_bandwidth_std": 20.0,
                "spectral_rolloff_mean": 300.0,
                "spectral_rolloff_std": 30.0,
                "spectral_flatness_mean": 0.01,
                "spectral_flatness_std": 0.001,
                "mfcc_01_mean": -10.0,
                "mfcc_01_std": 1.0,
                "mfcc_02_mean": 2.0,
                "mfcc_02_std": 0.5,
                "mfcc_03_mean": 3.0,
                "mfcc_03_std": 0.5,
                "mfcc_04_mean": 4.0,
                "mfcc_04_std": 0.5,
                "mfcc_05_mean": 5.0,
                "mfcc_05_std": 0.5,
            },
            {
                "review_row_key": "row-2",
                "selection": 2,
                "snippet_relpath": "",
                "duration_s": 0.5,
                "rms": 0.15,
                "zero_crossing_rate": 0.15,
                "spectral_centroid_mean": 500.0,
                "spectral_centroid_std": 10.0,
                "spectral_bandwidth_mean": 800.0,
                "spectral_bandwidth_std": 20.0,
                "spectral_rolloff_mean": 1200.0,
                "spectral_rolloff_std": 30.0,
                "spectral_flatness_mean": 0.20,
                "spectral_flatness_std": 0.001,
                "mfcc_01_mean": -5.0,
                "mfcc_01_std": 1.0,
                "mfcc_02_mean": 4.0,
                "mfcc_02_std": 0.5,
                "mfcc_03_mean": 6.0,
                "mfcc_03_std": 0.5,
                "mfcc_04_mean": 8.0,
                "mfcc_04_std": 0.5,
                "mfcc_05_mean": 10.0,
                "mfcc_05_std": 0.5,
            },
            {
                "review_row_key": "row-3",
                "selection": 3,
                "snippet_relpath": "",
                "duration_s": 0.5,
                "rms": 0.30,
                "zero_crossing_rate": 0.30,
                "spectral_centroid_mean": 1500.0,
                "spectral_centroid_std": 10.0,
                "spectral_bandwidth_mean": 2200.0,
                "spectral_bandwidth_std": 20.0,
                "spectral_rolloff_mean": 2800.0,
                "spectral_rolloff_std": 30.0,
                "spectral_flatness_mean": 0.50,
                "spectral_flatness_std": 0.001,
                "mfcc_01_mean": 2.0,
                "mfcc_01_std": 1.0,
                "mfcc_02_mean": 6.0,
                "mfcc_02_std": 0.5,
                "mfcc_03_mean": 9.0,
                "mfcc_03_std": 0.5,
                "mfcc_04_mean": 12.0,
                "mfcc_04_std": 0.5,
                "mfcc_05_mean": 15.0,
                "mfcc_05_std": 0.5,
            },
            {
                "review_row_key": "row-4",
                "selection": 4,
                "snippet_relpath": "",
                "duration_s": 0.5,
                "rms": 0.10,
                "zero_crossing_rate": 0.10,
                "spectral_centroid_mean": 100.0,
                "spectral_centroid_std": 10.0,
                "spectral_bandwidth_mean": 200.0,
                "spectral_bandwidth_std": 20.0,
                "spectral_rolloff_mean": 300.0,
                "spectral_rolloff_std": 30.0,
                "spectral_flatness_mean": 0.01,
                "spectral_flatness_std": 0.001,
                "mfcc_01_mean": -10.0,
                "mfcc_01_std": 1.0,
                "mfcc_02_mean": 2.0,
                "mfcc_02_std": 0.5,
                "mfcc_03_mean": 3.0,
                "mfcc_03_std": 0.5,
                "mfcc_04_mean": 4.0,
                "mfcc_04_std": 0.5,
                "mfcc_05_mean": 5.0,
                "mfcc_05_std": 0.5,
            },
        ]
    )
    features.to_parquet(analysis_dir / "acoustic_features.parquet", index=False)


def test_human_contamination_cleaning_builds_core_views_and_review_queue(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    artifacts_root = tmp_path / "artifacts"
    _write_bundle(bundle_dir)
    _write_split_review_dir(bundle_dir / "review_concat_split_5")
    _write_artifacts(artifacts_root, bundle_dir.name)

    output_dir = run_human_contamination_cleaning(
        review_bundle=bundle_dir,
        config=HumanContaminationCleaningConfig(
            human_label_value="0",
            feature_neighbor_quantile=0.25,
            boundary_rank_review=1,
            boundary_minutes_review=2.0,
        ),
        output_root=artifacts_root,
        backfill_dir=artifacts_root / bundle_dir.name / "metadata_backfill",
        analysis_dir=artifacts_root / bundle_dir.name / "acoustic_analysis",
    )

    assessment = pd.read_parquet(output_dir / "tables" / "detection_contamination_assessment.parquet")
    assert assessment.loc[assessment["selection"] == 1, "primary_quarantine_reason"].item() == "confirmed_human_contamination"
    assert assessment.loc[assessment["selection"] == 2, "primary_quarantine_reason"].item() == "source_quarantine"
    assert assessment.loc[assessment["selection"] == 4, "primary_quarantine_reason"].item() == "feature_neighbor_quarantine"
    assert (
        assessment.loc[assessment["selection"] == 1, "quarantine_reason_tags"].item()
        == "all_excluded_from_working_core|manual_human_seeds|same_source_as_manual_human"
    )
    assert (
        assessment.loc[assessment["selection"] == 2, "quarantine_reason_tags"].item()
        == "all_excluded_from_working_core|same_source_as_manual_human"
    )
    assert (
        assessment.loc[assessment["selection"] == 4, "quarantine_reason_tags"].item()
        == "all_excluded_from_working_core|acoustically_similar_to_manual_human"
    )

    conservative = pd.read_parquet(output_dir / "tables" / "clean_core_conservative.parquet")
    working = pd.read_parquet(output_dir / "tables" / "clean_core_working.parquet")
    review_queue = pd.read_parquet(output_dir / "tables" / "review_priority_candidates.parquet")

    assert conservative["selection"].tolist() == [3, 4]
    assert working["selection"].tolist() == [3]
    assert review_queue["selection"].tolist() == [4]

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf8"))
    assert manifest["counts"]["manual_human_rows"] == 1
    assert manifest["counts"]["review_queue_rows"] == 1


def test_human_contamination_cleaning_writes_quarantine_audio_exports(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    artifacts_root = tmp_path / "artifacts"
    _write_bundle(bundle_dir)
    _write_review_concat_wav(bundle_dir / "review_concat.wav")
    _write_split_review_dir(bundle_dir / "review_concat_split_5")
    _write_artifacts(artifacts_root, bundle_dir.name)

    output_dir = run_human_contamination_cleaning(
        review_bundle=bundle_dir,
        config=HumanContaminationCleaningConfig(
            human_label_value="0",
            feature_neighbor_quantile=0.25,
            boundary_rank_review=1,
            boundary_minutes_review=2.0,
            write_quarantine_audio_exports=True,
            audio_export_gap_s=0.1,
        ),
        output_root=artifacts_root,
        backfill_dir=artifacts_root / bundle_dir.name / "metadata_backfill",
        analysis_dir=artifacts_root / bundle_dir.name / "acoustic_analysis",
    )

    quarantine_manifest = json.loads(
        (output_dir / "audio_exports" / "all_excluded_from_working_core" / "manifest.json").read_text(encoding="utf8")
    )
    assert quarantine_manifest["row_count"] == 3
    assert quarantine_manifest["gap_duration_s"] == 0.1

    quarantine_table = pd.read_csv(
        output_dir / "audio_exports" / "all_excluded_from_working_core" / "review_selection_table.txt",
        sep="\t",
    )
    assert quarantine_table["original_selection"].tolist() == [1, 2, 4]
    assert quarantine_table["Begin Time (s)"].round(6).tolist() == [0.0, 0.6, 1.2]
    assert quarantine_table["End Time (s)"].round(6).tolist() == [0.5, 1.1, 1.7]
    assert quarantine_table["quarantine_reason_tags"].tolist() == [
        "all_excluded_from_working_core|manual_human_seeds|same_source_as_manual_human",
        "all_excluded_from_working_core|same_source_as_manual_human",
        "all_excluded_from_working_core|acoustically_similar_to_manual_human",
    ]

    with wave.open(str(output_dir / "audio_exports" / "all_excluded_from_working_core" / "review_concat.wav"), "rb") as handle:
        assert handle.getframerate() == 48000
        assert handle.getnchannels() == 1
        assert handle.getnframes() == 81600

    same_source_manifest = json.loads(
        (output_dir / "audio_exports" / "same_source_as_manual_human" / "manifest.json").read_text(encoding="utf8")
    )
    assert same_source_manifest["row_count"] == 2

    acoustic_similar_manifest = json.loads(
        (output_dir / "audio_exports" / "acoustically_similar_to_manual_human" / "manifest.json").read_text(
            encoding="utf8"
        )
    )
    assert acoustic_similar_manifest["row_count"] == 1

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf8"))
    assert manifest["counts"]["audio_export_sets"] == 4

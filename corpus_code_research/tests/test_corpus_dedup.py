from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.io import wavfile

import corpus_dedup.features as feature_module
import corpus_dedup.hpc as hpc_module
from corpus_dedup.audio import compute_features
from corpus_dedup.cache import file_lock
from corpus_dedup.constants import ROBUSTNESS_METHOD_VERSION
from corpus_dedup.config import load_config
from corpus_dedup.features import prepare_features, run_features
from corpus_dedup.grid import (
    _build_batch_context,
    _evaluate_model_from_batch,
    _interval_overlap_s,
    _overlap_fraction_short,
    build_candidate_pair_cache,
)
from corpus_dedup.manifest_utils import write_json


def _write_wav(path: Path, *, duration_s: float, sample_rate: int, frequency_hz: float, amplitude: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = int(duration_s * sample_rate)
    time_axis = np.arange(sample_count) / sample_rate
    samples = amplitude * np.sin(2.0 * math.pi * frequency_hz * time_axis)
    wavfile.write(path, sample_rate, np.asarray(samples * 32767.0, dtype=np.int16))


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    sample_rate = 48_000
    wav_a = tmp_path / "audio" / "device_a.wav"
    wav_b = tmp_path / "audio" / "device_b.wav"
    wav_c = tmp_path / "audio" / "device_c.wav"
    wav_d = tmp_path / "audio" / "device_d.wav"
    wav_e = tmp_path / "audio" / "device_e.wav"
    wav_f = tmp_path / "audio" / "device_f.wav"

    _write_wav(wav_a, duration_s=10.0, sample_rate=sample_rate, frequency_hz=220.0)
    _write_wav(wav_b, duration_s=10.0, sample_rate=sample_rate, frequency_hz=255.0)
    _write_wav(wav_c, duration_s=10.0, sample_rate=sample_rate, frequency_hz=330.0)
    _write_wav(wav_d, duration_s=10.0, sample_rate=sample_rate, frequency_hz=345.0)
    _write_wav(wav_e, duration_s=10.0, sample_rate=sample_rate, frequency_hz=900.0)
    _write_wav(wav_f, duration_s=10.0, sample_rate=sample_rate, frequency_hz=440.0)

    detections = pd.DataFrame(
        [
            {
                "canonical_detection_id": "hum-001",
                "review_row_key": "row-001",
                "device_id": "device_a",
                "wav_basename": wav_a.name,
                "source_audio_file": str(wav_a),
                "source_start_s": 1.00,
                "source_end_s": 1.60,
                "sample_rate_estimate": float(sample_rate),
                "total_samples": float(sample_rate * 10.0),
                "absolute_detection_start_time": "2025-02-01T00:00:01Z",
                "absolute_detection_end_time": "2025-02-01T00:00:01.6Z",
                "latitude_near_detection": -34.524700,
                "longitude_near_detection": 138.689700,
                "model_positive_score": 0.91,
                "audio_collection_name": "collection_a",
                "metadata_join_status": "matched",
            },
            {
                "canonical_detection_id": "hum-002",
                "review_row_key": "row-002",
                "device_id": "device_b",
                "wav_basename": wav_b.name,
                "source_audio_file": str(wav_b),
                "source_start_s": 1.05,
                "source_end_s": 1.65,
                "sample_rate_estimate": float(sample_rate),
                "total_samples": float(sample_rate * 10.0),
                "absolute_detection_start_time": "2025-02-01T00:00:01.05Z",
                "absolute_detection_end_time": "2025-02-01T00:00:01.65Z",
                "latitude_near_detection": -34.524705,
                "longitude_near_detection": 138.689705,
                "model_positive_score": 0.95,
                "audio_collection_name": "collection_b",
                "metadata_join_status": "matched",
            },
            {
                "canonical_detection_id": "hum-003",
                "review_row_key": "row-003",
                "device_id": "device_c",
                "wav_basename": wav_c.name,
                "source_audio_file": str(wav_c),
                "source_start_s": 3.00,
                "source_end_s": 3.60,
                "sample_rate_estimate": float(sample_rate),
                "total_samples": float(sample_rate * 10.0),
                "absolute_detection_start_time": "2025-02-01T00:00:03Z",
                "absolute_detection_end_time": "2025-02-01T00:00:03.6Z",
                "latitude_near_detection": -34.624700,
                "longitude_near_detection": 138.789700,
                "model_positive_score": 0.88,
                "audio_collection_name": "collection_c",
                "metadata_join_status": "matched",
            },
            {
                "canonical_detection_id": "hum-004",
                "review_row_key": "row-004",
                "device_id": "device_d",
                "wav_basename": wav_d.name,
                "source_audio_file": str(wav_d),
                "source_start_s": 3.04,
                "source_end_s": 3.64,
                "sample_rate_estimate": float(sample_rate),
                "total_samples": float(sample_rate * 10.0),
                "absolute_detection_start_time": "2025-02-01T00:00:03.04Z",
                "absolute_detection_end_time": "2025-02-01T00:00:03.64Z",
                "latitude_near_detection": -34.624706,
                "longitude_near_detection": 138.789706,
                "model_positive_score": 0.96,
                "audio_collection_name": "collection_d",
                "metadata_join_status": "matched",
            },
            {
                "canonical_detection_id": "hum-005",
                "review_row_key": "row-005",
                "device_id": "device_e",
                "wav_basename": wav_e.name,
                "source_audio_file": str(wav_e),
                "source_start_s": 3.08,
                "source_end_s": 3.68,
                "sample_rate_estimate": float(sample_rate),
                "total_samples": float(sample_rate * 10.0),
                "absolute_detection_start_time": "2025-02-01T00:00:03.08Z",
                "absolute_detection_end_time": "2025-02-01T00:00:03.68Z",
                "latitude_near_detection": -34.624712,
                "longitude_near_detection": 138.789712,
                "model_positive_score": 0.83,
                "audio_collection_name": "collection_e",
                "metadata_join_status": "matched",
            },
            {
                "canonical_detection_id": "hum-006",
                "review_row_key": "row-006",
                "device_id": "device_a",
                "wav_basename": wav_a.name,
                "source_audio_file": str(wav_a),
                "source_start_s": 6.00,
                "source_end_s": 6.60,
                "sample_rate_estimate": float(sample_rate),
                "total_samples": float(sample_rate * 10.0),
                "absolute_detection_start_time": "2025-02-01T00:00:06Z",
                "absolute_detection_end_time": "2025-02-01T00:00:06.6Z",
                "latitude_near_detection": -34.724700,
                "longitude_near_detection": 138.889700,
                "model_positive_score": 0.60,
                "audio_collection_name": "collection_a",
                "metadata_join_status": "matched",
            },
            {
                "canonical_detection_id": "hum-007",
                "review_row_key": "row-007",
                "device_id": "device_f",
                "wav_basename": wav_f.name,
                "source_audio_file": str(wav_f),
                "source_start_s": 7.00,
                "source_end_s": 7.60,
                "sample_rate_estimate": float(sample_rate),
                "total_samples": float(sample_rate * 10.0),
                "absolute_detection_start_time": "2025-02-01T00:00:07Z",
                "absolute_detection_end_time": "2025-02-01T00:00:07.6Z",
                "latitude_near_detection": np.nan,
                "longitude_near_detection": np.nan,
                "model_positive_score": 0.72,
                "audio_collection_name": "collection_f",
                "metadata_join_status": "matched",
            },
        ]
    )
    source_wav_index = pd.DataFrame(
        [
            {
                "source_audio_file_original": str(wav_a),
                "wav_hpc_path": str(wav_a),
                "wav_basename": wav_a.name,
                "first_valid_latitude": -34.524700,
                "first_valid_longitude": 138.689700,
                "last_valid_latitude": -34.524710,
                "last_valid_longitude": 138.689710,
                "gps_valid_fraction": 1.0,
                "audio_collection_name": "collection_a",
                "device_id": "device_a",
                "firmware_version": "fw-a",
                "firmware_description": "firmware a",
                "settings_hash": "hash-a",
            },
            {
                "source_audio_file_original": str(wav_b),
                "wav_hpc_path": str(wav_b),
                "wav_basename": wav_b.name,
                "first_valid_latitude": -34.524705,
                "first_valid_longitude": 138.689705,
                "last_valid_latitude": -34.524715,
                "last_valid_longitude": 138.689715,
                "gps_valid_fraction": 1.0,
                "audio_collection_name": "collection_b",
                "device_id": "device_b",
                "firmware_version": "fw-b",
                "firmware_description": "firmware b",
                "settings_hash": "hash-b",
            },
            {
                "source_audio_file_original": str(wav_c),
                "wav_hpc_path": str(wav_c),
                "wav_basename": wav_c.name,
                "first_valid_latitude": -34.624700,
                "first_valid_longitude": 138.789700,
                "last_valid_latitude": -34.624710,
                "last_valid_longitude": 138.789710,
                "gps_valid_fraction": 1.0,
                "audio_collection_name": "collection_c",
                "device_id": "device_c",
                "firmware_version": "fw-c",
                "firmware_description": "firmware c",
                "settings_hash": "hash-c",
            },
            {
                "source_audio_file_original": str(wav_d),
                "wav_hpc_path": str(wav_d),
                "wav_basename": wav_d.name,
                "first_valid_latitude": -34.624706,
                "first_valid_longitude": 138.789706,
                "last_valid_latitude": -34.624716,
                "last_valid_longitude": 138.789716,
                "gps_valid_fraction": 1.0,
                "audio_collection_name": "collection_d",
                "device_id": "device_d",
                "firmware_version": "fw-d",
                "firmware_description": "firmware d",
                "settings_hash": "hash-d",
            },
            {
                "source_audio_file_original": str(wav_e),
                "wav_hpc_path": str(wav_e),
                "wav_basename": wav_e.name,
                "first_valid_latitude": -34.624712,
                "first_valid_longitude": 138.789712,
                "last_valid_latitude": -34.624722,
                "last_valid_longitude": 138.789722,
                "gps_valid_fraction": 1.0,
                "audio_collection_name": "collection_e",
                "device_id": "device_e",
                "firmware_version": "fw-e",
                "firmware_description": "firmware e",
                "settings_hash": "hash-e",
            },
            {
                "source_audio_file_original": str(wav_f),
                "wav_hpc_path": str(wav_f),
                "wav_basename": wav_f.name,
                "first_valid_latitude": -34.824700,
                "first_valid_longitude": 138.989700,
                "last_valid_latitude": -34.824710,
                "last_valid_longitude": 138.989710,
                "gps_valid_fraction": 1.0,
                "audio_collection_name": "collection_f",
                "device_id": "device_f",
                "firmware_version": "fw-f",
                "firmware_description": "firmware f",
                "settings_hash": "hash-f",
            },
        ]
    )

    input_table = tmp_path / "cleaned_enriched_detections.parquet"
    source_wav_index_path = tmp_path / "source_wav_metadata_index.parquet"
    detections.to_parquet(input_table, index=False)
    source_wav_index.to_parquet(source_wav_index_path, index=False)

    config_path = tmp_path / "smoke.yaml"
    launch_dir = tmp_path / "artifacts" / "dedup_full_corpus" / "launch_test_smoke"
    config = {
        "launch_label": "launch_test_smoke",
        "artifact_root": str((tmp_path / "artifacts" / "dedup_full_corpus").resolve()),
        "launch_dir": str(launch_dir.resolve()),
        "input_table": str(input_table.resolve()),
        "source_wav_index": str(source_wav_index_path.resolve()),
        "algorithm_version": "overlap_temporal_v4",
        "overlap_fraction_short_thresholds": [0.05, 0.10, 0.25, 0.50, 0.75],
        "gps_thresholds_m": [2.0, 3.0, 4.0, 5.0, 6.0],
        "relative_difference_thresholds": [None, 0.10, 0.20, 0.30, 0.40],
        "group_acoustic_thresholds": [None, "q90", "q75", "q50"],
        "top_k_models": 2,
        "exemplars_per_model": 4,
        "robustness": {
            "acoustic_margin_quantile": 0.25,
            "target_acoustic_margin": 0.10,
            "min_final_selectable_acoustic_coverage": 0.95,
            "decision_acoustic_core_weight": 0.75,
            "removal_count_stability_weight": 0.25,
        },
        "hpc": {
            "host": "hpc-cluster",
            "partition": "compute",
            "account": "alpaca-project",
            "remote_repo_root": "/cluster/projects/alpaca/corpus_code_research",
            "remote_venv_path": "/cluster/projects/alpaca/alpaca-pipelines/.venv",
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
    return config_path, launch_dir


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "corpus_dedup.cli", *args], check=True, text=True, capture_output=True)


def _prepare_feature_tables(config_path: Path, launch_dir: Path) -> None:
    _run_cli("prepare-features", "--config", str(config_path))
    _run_cli("run-features", "--launch-dir", str(launch_dir))
    _run_cli("finalize-features", "--launch-dir", str(launch_dir))


def _execute_full_pipeline(config_path: Path, launch_dir: Path) -> None:
    _prepare_feature_tables(config_path, launch_dir)
    _run_cli("prepare-grid", "--launch-dir", str(launch_dir))
    _run_cli("run-grid", "--launch-dir", str(launch_dir))
    _run_cli("summarize-run", "--launch-dir", str(launch_dir))
    _run_cli("export-top-exemplars", "--launch-dir", str(launch_dir), "--top-k", "2")


def test_compute_features_extracts_expected_research_measurements() -> None:
    sample_rate = 48_000
    duration_s = 0.8
    time_axis = np.arange(int(sample_rate * duration_s)) / sample_rate
    snippet = (0.2 * np.sin(2.0 * math.pi * 220.0 * time_axis)).astype(np.float32)

    config = load_config(Path("config/local_smoke.yaml")).feature
    features = compute_features(snippet, sample_rate, config)

    assert set(features) == {
        "snippet_duration_s",
        "snippet_rms",
        "snippet_snr_db",
        "f0_hz",
        "duration90_s",
        "peak_frequency_hz",
        "frequency5_hz",
        "frequency95_hz",
        "bandwidth90_hz",
        "average_entropy",
    }
    assert abs(features["snippet_duration_s"] - duration_s) < 0.01
    assert abs(features["f0_hz"] - 220.0) < 40.0
    assert abs(features["peak_frequency_hz"] - 220.0) < 40.0
    assert 0.60 <= features["duration90_s"] <= 0.80
    assert features["bandwidth90_hz"] >= 0.0
    assert 0.0 <= features["average_entropy"] <= 1.0


def test_interval_overlap_and_candidate_pair_cache(tmp_path: Path) -> None:
    assert abs(_interval_overlap_s(0.0, 1.0, 0.5, 1.5) - 0.5) < 1e-9
    assert _interval_overlap_s(0.0, 1.0, 1.15, 2.0) == 0.0
    assert abs(_overlap_fraction_short(0.0, 1.0, 0.5, 1.5) - 0.5) < 1e-9
    assert _overlap_fraction_short(0.0, 1.0, 1.15, 2.0) == 0.0

    config_path, launch_dir = _build_fixture(tmp_path)
    _prepare_feature_tables(config_path, launch_dir)
    config = load_config(config_path)
    candidate_df = build_candidate_pair_cache(config, launch_dir_to_paths(launch_dir))

    candidate_pairs = {tuple(sorted((row.hum_id_i, row.hum_id_j))) for row in candidate_df.itertuples(index=False)}
    assert ("hum-001", "hum-002") in candidate_pairs
    assert ("hum-003", "hum-004") in candidate_pairs
    assert ("hum-001", "hum-003") not in candidate_pairs


def test_size2_relative_difference_gating(tmp_path: Path) -> None:
    config_path, launch_dir = _build_fixture(tmp_path)
    _prepare_feature_tables(config_path, launch_dir)
    config = load_config(config_path)
    paths = launch_dir_to_paths(launch_dir)

    hum_features_df = pd.read_parquet(launch_dir / "tables" / "hum_features.parquet")
    candidate_df = build_candidate_pair_cache(config, paths)
    batch_row = {
        "batch_id": "batch_ovshort010_g02",
        "T_overlap_short": 0.10,
        "T_gps_m": 2.0,
    }
    batch_candidate_df, provisional_groups_df = _build_batch_context(hum_features_df, candidate_df, batch_row)
    models_df = pd.DataFrame(
        [
            {
                "dedup_model_id": "model_rel10",
                "batch_id": batch_row["batch_id"],
                "T_overlap_short": 0.10,
                "T_gps_m": 2.0,
                "relative_difference_setting": "rel10",
                "relative_difference_threshold": 0.10,
                "group_acoustic_setting": "acnone",
                "group_acoustic_threshold": None,
            },
            {
                "dedup_model_id": "model_rel30",
                "batch_id": batch_row["batch_id"],
                "T_overlap_short": 0.10,
                "T_gps_m": 2.0,
                "relative_difference_setting": "rel30",
                "relative_difference_threshold": 0.30,
                "group_acoustic_setting": "acnone",
                "group_acoustic_threshold": None,
            },
        ]
    )

    result_rel10 = _evaluate_model_from_batch(
        hum_features_df,
        batch_candidate_df,
        provisional_groups_df,
        models_df.iloc[0].to_dict(),
        materialize_details=False,
    )
    result_rel20 = _evaluate_model_from_batch(
        hum_features_df,
        batch_candidate_df,
        provisional_groups_df,
        models_df.iloc[1].to_dict(),
        materialize_details=False,
    )

    assert result_rel10["summary"]["accepted_edge_count_size_2"] == 0
    assert result_rel20["summary"]["accepted_edge_count_size_2"] == 1
    assert result_rel10["summary"]["accepted_edge_acoustic_clearances"] == []
    assert result_rel20["summary"]["accepted_edge_acoustic_clearances"]
    assert result_rel20["summary"]["duplicate_pair_count"] > 0


def test_size3plus_quantile_gating_and_traceability(tmp_path: Path) -> None:
    config_path, launch_dir = _build_fixture(tmp_path)
    _prepare_feature_tables(config_path, launch_dir)
    config = load_config(config_path)
    paths = launch_dir_to_paths(launch_dir)

    hum_features_df = pd.read_parquet(launch_dir / "tables" / "hum_features.parquet")
    candidate_df = build_candidate_pair_cache(config, paths)
    batch_row = {
        "batch_id": "batch_ovshort005_g02",
        "T_overlap_short": 0.05,
        "T_gps_m": 2.0,
    }
    batch_candidate_df, provisional_groups_df = _build_batch_context(hum_features_df, candidate_df, batch_row)
    model_none = {
        "dedup_model_id": "model_acnone",
        "batch_id": batch_row["batch_id"],
        "T_overlap_short": 0.05,
        "T_gps_m": 2.0,
        "relative_difference_setting": "relnone",
        "relative_difference_threshold": None,
        "group_acoustic_setting": "acnone",
        "group_acoustic_threshold": None,
    }
    model_q50 = {
        "dedup_model_id": "model_acq50",
        "batch_id": batch_row["batch_id"],
        "T_overlap_short": 0.05,
        "T_gps_m": 2.0,
        "relative_difference_setting": "relnone",
        "relative_difference_threshold": None,
        "group_acoustic_setting": "acq50",
        "group_acoustic_threshold": "q50",
    }

    result_none = _evaluate_model_from_batch(
        hum_features_df,
        batch_candidate_df,
        provisional_groups_df,
        model_none,
        materialize_details=True,
    )
    result_q50 = _evaluate_model_from_batch(
        hum_features_df,
        batch_candidate_df,
        provisional_groups_df,
        model_q50,
        materialize_details=True,
    )

    assert result_none["summary"]["N_accepted_edges"] > result_q50["summary"]["N_accepted_edges"]
    assert set(result_q50["members_df"]["path_to_primary_type"]) >= {"primary", "direct"}
    assert result_q50["links_df"]["is_component_tree_edge"].all()
    assert set(result_q50["links_df"]["path_to_primary_type"]) <= {"direct", "redundant"}


def test_summary_uses_new_robustness_terms(tmp_path: Path) -> None:
    config_path, launch_dir = _build_fixture(tmp_path)
    _execute_full_pipeline(config_path, launch_dir)

    summary_df = pd.read_parquet(launch_dir / "tables" / "model_sensitivity_summary.parquet")
    top_rankings_df = pd.read_csv(launch_dir / "tables" / "top_model_rankings.csv")
    fully_unfiltered = summary_df.loc[
        (summary_df["group_acoustic_setting"] == "acnone")
        & (summary_df["relative_difference_setting"] == "relnone")
    ].copy()

    assert len(summary_df.index) == 500
    assert not fully_unfiltered.empty
    assert "local_decision_stability" in summary_df.columns
    assert "removal_count_stability" in summary_df.columns
    assert "acoustic_margin_q25" in summary_df.columns
    assert "acoustic_threshold_coverage" in summary_df.columns
    assert "acoustic_strength" in summary_df.columns
    assert "decision_acoustic_core" in summary_df.columns
    assert "final_selectable" in summary_df.columns
    assert "ineligibility_reason" in summary_df.columns
    assert "robustness_method_version" in summary_df.columns
    assert "duplicate_pair_count" in summary_df.columns
    assert "event_count_stability" not in summary_df.columns
    assert "acoustic_consistency" not in summary_df.columns
    assert "coverage_adjusted_acoustic_margin" not in summary_df.columns
    assert summary_df["local_decision_stability"].between(0.0, 1.0).all()
    assert summary_df["removal_count_stability"].between(0.0, 1.0).all()
    assert summary_df["acoustic_margin_q25"].between(0.0, 1.0).all()
    assert summary_df["acoustic_threshold_coverage"].between(0.0, 1.0).all()
    assert summary_df["acoustic_strength"].between(0.0, 1.0).all()
    assert summary_df["decision_acoustic_core"].between(0.0, 1.0).all()
    assert summary_df["robustness_method_version"].eq(ROBUSTNESS_METHOD_VERSION).all()
    assert fully_unfiltered["acoustic_margin_q25"].eq(0.0).all()
    assert fully_unfiltered["acoustic_threshold_coverage"].eq(0.0).all()
    assert fully_unfiltered["acoustic_strength"].eq(0.0).all()
    assert fully_unfiltered["decision_acoustic_core"].eq(0.0).all()
    assert fully_unfiltered["final_selectable"].eq(False).all()
    assert fully_unfiltered["robustness_score"].isna().all()
    assert fully_unfiltered["ineligibility_reason"].str.contains("relative_difference_threshold_none").all()
    assert summary_df["final_selectable"].any()
    assert summary_df.loc[summary_df["final_selectable"], "robustness_score"].nunique() > 1
    assert top_rankings_df["final_selectable"].all()


def test_summarize_run_reuses_model_caches_without_rerunning_grid(tmp_path: Path) -> None:
    config_path, launch_dir = _build_fixture(tmp_path)
    _execute_full_pipeline(config_path, launch_dir)

    summary_before = pd.read_parquet(launch_dir / "tables" / "model_sensitivity_summary.parquet")
    sample_summary_path = next((launch_dir / "cache" / "model_runs").rglob("summary.json"))
    sample_mtime_before = sample_summary_path.stat().st_mtime_ns

    for stale_path in [
        launch_dir / "tables" / "model_sensitivity_summary.parquet",
        launch_dir / "tables" / "model_sensitivity_summary.csv",
        launch_dir / "tables" / "top_model_rankings.csv",
        launch_dir / "reports" / "model_ranking_report.md",
        launch_dir / "tables" / "final_model_selection.json",
    ]:
        stale_path.unlink()

    _run_cli("summarize-run", "--launch-dir", str(launch_dir))

    summary_after = pd.read_parquet(launch_dir / "tables" / "model_sensitivity_summary.parquet")
    report_text = (launch_dir / "reports" / "model_ranking_report.md").read_text(encoding="utf8")

    assert sample_summary_path.stat().st_mtime_ns == sample_mtime_before
    assert "acoustic_margin_q25" in summary_after.columns
    assert "acoustic_threshold_coverage" in summary_after.columns
    assert "acoustic_strength" in summary_after.columns
    assert "decision_acoustic_core" in summary_after.columns
    assert "coverage_adjusted_acoustic_margin" not in summary_after.columns
    assert "0.75 * decision_acoustic_core + 0.25 * removal_count_stability" in report_text
    assert "coverage_adjusted_acoustic_margin" not in report_text
    pd.testing.assert_series_equal(summary_before["robustness_score"], summary_after["robustness_score"])


def test_summarize_run_handles_no_final_selectable_models(tmp_path: Path) -> None:
    config_path, launch_dir = _build_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))
    config["relative_difference_thresholds"] = [None]
    config["group_acoustic_thresholds"] = [None]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")

    _prepare_feature_tables(config_path, launch_dir)
    _run_cli("prepare-grid", "--launch-dir", str(launch_dir))
    _run_cli("run-grid", "--launch-dir", str(launch_dir))
    _run_cli("summarize-run", "--launch-dir", str(launch_dir))
    _run_cli("export-top-exemplars", "--launch-dir", str(launch_dir), "--top-k", "2")

    top_rankings_df = pd.read_csv(launch_dir / "tables" / "top_model_rankings.csv")
    final_selection = json.loads((launch_dir / "tables" / "final_model_selection.json").read_text())

    assert top_rankings_df.empty
    assert final_selection["selection_status"] == "no_final_selectable_model"
    assert final_selection["best_model_id"] is None
    assert not (launch_dir / "tables" / "deduplicated_hums.parquet").exists()


def test_seed_feature_artifacts_reuses_existing_feature_outputs(tmp_path: Path) -> None:
    config_path, source_launch_dir = _build_fixture(tmp_path)
    _prepare_feature_tables(config_path, source_launch_dir)

    target_launch_dir = tmp_path / "artifacts" / "dedup_full_corpus" / "launch_test_seeded"
    target_config = yaml.safe_load(config_path.read_text(encoding="utf8"))
    target_config["launch_label"] = "launch_test_seeded"
    target_config["launch_dir"] = str(target_launch_dir.resolve())
    target_config_path = tmp_path / "seed_target.yaml"
    target_config_path.write_text(yaml.safe_dump(target_config, sort_keys=False), encoding="utf8")

    _run_cli("prepare-features", "--config", str(target_config_path))
    _run_cli(
        "seed-feature-artifacts",
        "--launch-dir",
        str(target_launch_dir),
        "--source-launch-dir",
        str(source_launch_dir),
    )

    source_hum_features = pd.read_parquet(source_launch_dir / "tables" / "hum_features.parquet")
    target_hum_features = pd.read_parquet(target_launch_dir / "tables" / "hum_features.parquet")
    feature_merge_progress = json.loads((target_launch_dir / "progress" / "feature_merge_progress.json").read_text())

    pd.testing.assert_frame_equal(source_hum_features, target_hum_features)
    assert feature_merge_progress["stage_status"] == "completed"
    assert (target_launch_dir / "cache" / "features").is_dir()


def test_seed_model_artifacts_reuses_existing_grid_outputs(tmp_path: Path) -> None:
    config_path, source_launch_dir = _build_fixture(tmp_path)
    _prepare_feature_tables(config_path, source_launch_dir)
    _run_cli("prepare-grid", "--launch-dir", str(source_launch_dir))
    _run_cli("run-grid", "--launch-dir", str(source_launch_dir))

    target_launch_dir = tmp_path / "artifacts" / "dedup_full_corpus" / "launch_test_seeded_models"
    target_config = yaml.safe_load(config_path.read_text(encoding="utf8"))
    target_config["launch_label"] = "launch_test_seeded_models"
    target_config["launch_dir"] = str(target_launch_dir.resolve())
    target_config_path = tmp_path / "seed_models_target.yaml"
    target_config_path.write_text(yaml.safe_dump(target_config, sort_keys=False), encoding="utf8")

    _run_cli("prepare-features", "--config", str(target_config_path))
    _run_cli(
        "seed-feature-artifacts",
        "--launch-dir",
        str(target_launch_dir),
        "--source-launch-dir",
        str(source_launch_dir),
    )
    _run_cli("prepare-grid", "--launch-dir", str(target_launch_dir))
    _run_cli(
        "seed-model-artifacts",
        "--launch-dir",
        str(target_launch_dir),
        "--source-launch-dir",
        str(source_launch_dir),
    )
    _run_cli("summarize-run", "--launch-dir", str(target_launch_dir))

    grid_progress = json.loads((target_launch_dir / "progress" / "grid_progress.json").read_text())
    summary_cache_files = list((target_launch_dir / "cache" / "model_runs").rglob("summary.json"))

    assert grid_progress["stage_status"] == "completed"
    assert grid_progress["cached_models_reused"] == 500
    assert len(summary_cache_files) == 500
    assert (target_launch_dir / "tables" / "top_model_rankings.csv").is_file()


def test_feature_stage_race_regression(tmp_path: Path) -> None:
    config_path, launch_dir = _build_fixture(tmp_path)

    _run_cli("prepare-features", "--config", str(config_path))
    _run_cli("run-features", "--launch-dir", str(launch_dir))

    assert not (launch_dir / "tables" / "hum_features.parquet").exists()

    _run_cli("finalize-features", "--launch-dir", str(launch_dir))

    assert (launch_dir / "tables" / "hum_features.parquet").is_file()


def test_run_features_uses_memory_mapped_wav_reads(tmp_path: Path, monkeypatch) -> None:
    config_path, _ = _build_fixture(tmp_path)
    config = load_config(config_path)
    prepare_features(config)

    mmap_flags: list[bool] = []
    original_load_wav = feature_module.load_wav

    def _tracked_load_wav(path: Path, *, mmap: bool = False):
        mmap_flags.append(mmap)
        return original_load_wav(path, mmap=mmap)

    monkeypatch.setattr(feature_module, "load_wav", _tracked_load_wav)

    run_features(config, shard_ids=[0])

    assert mmap_flags
    assert all(mmap_flags)


def test_file_lock_reuses_existing_lock_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache" / "artifact.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    with file_lock(lock_path, timeout_s=0.1):
        assert lock_path.is_file()

    with file_lock(lock_path, timeout_s=0.1):
        assert lock_path.is_file()


def test_write_json_tolerates_concurrent_writers(tmp_path: Path) -> None:
    target = tmp_path / "progress" / "grid_progress.json"

    def _write(value: int) -> None:
        write_json(target, {"writer": value})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_write, range(20)))

    payload = json.loads(target.read_text(encoding="utf8"))
    assert "writer" in payload
    assert not list(target.parent.glob("*.tmp"))


def test_empty_batch_candidates_do_not_crash_model_evaluation(tmp_path: Path) -> None:
    config_path, launch_dir = _build_fixture(tmp_path)
    _prepare_feature_tables(config_path, launch_dir)
    config = load_config(config_path)
    paths = launch_dir_to_paths(launch_dir)

    hum_features_df = pd.read_parquet(launch_dir / "tables" / "hum_features.parquet")
    candidate_df = build_candidate_pair_cache(config, paths)
    batch_row = {
        "batch_id": "batch_ovshort010_g02",
        "T_overlap_short": 0.10,
        "T_gps_m": 2.0,
    }
    batch_candidate_df, provisional_groups_df = _build_batch_context(hum_features_df, candidate_df, batch_row)
    empty_candidates_df = batch_candidate_df.iloc[0:0].copy()
    model_row = {
        "dedup_model_id": "model_empty",
        "batch_id": "batch_empty",
        "T_overlap_short": 0.75,
        "T_gps_m": 0.0,
        "relative_difference_setting": "rel20",
        "relative_difference_threshold": 0.20,
        "group_acoustic_setting": "acq75",
        "group_acoustic_threshold": "q75",
    }

    result = _evaluate_model_from_batch(
        hum_features_df,
        empty_candidates_df,
        provisional_groups_df,
        model_row,
        materialize_details=True,
    )

    assert result["summary"]["N_accepted_edges"] == 0
    assert result["members_df"]["cluster_size"].eq(1).all()
    assert result["links_df"].empty


def test_corpus_dedup_smoke_cli(tmp_path: Path) -> None:
    config_path, launch_dir = _build_fixture(tmp_path)

    _execute_full_pipeline(config_path, launch_dir)
    _run_cli("prepare-hpc", "--launch-dir", str(launch_dir), "--stage", "features")
    _run_cli("prepare-hpc", "--launch-dir", str(launch_dir), "--stage", "feature_merge")
    _run_cli("prepare-hpc", "--launch-dir", str(launch_dir), "--stage", "grid")
    _run_cli("prepare-hpc", "--launch-dir", str(launch_dir), "--stage", "aggregate")
    _run_cli("prepare-hpc", "--launch-dir", str(launch_dir), "--stage", "exemplars")
    dry_run = _run_cli("submit-hpc", "--launch-dir", str(launch_dir), "--stage", "grid", "--dry-run")
    progress = _run_cli("show-progress", "--launch-dir", str(launch_dir))

    summary = pd.read_parquet(launch_dir / "tables" / "model_sensitivity_summary.parquet")
    grid_batches = pd.read_parquet(launch_dir / "manifests" / "grid_batches.parquet")
    deduplicated = pd.read_csv(launch_dir / "tables" / "deduplicated_hums.csv")
    members = pd.read_csv(launch_dir / "tables" / "dedup_event_members.csv")
    dedup_links = pd.read_csv(launch_dir / "tables" / "dedup_links.csv")
    summary_cache_files = list((launch_dir / "cache" / "model_runs").rglob("summary.json"))

    assert len(summary.index) == 500
    assert len(grid_batches.index) == 25
    assert len(summary_cache_files) == 500
    assert not deduplicated.empty
    assert not members.empty
    assert not dedup_links.empty
    assert "sbatch job-array.sbatch" in dry_run.stdout
    assert "launch=" in progress.stdout
    assert (launch_dir / "hpc_bundle" / "features" / "job-array.sbatch").is_file()
    assert (launch_dir / "hpc_bundle" / "feature_merge" / "job-array.sbatch").is_file()
    assert (launch_dir / "hpc_bundle" / "grid" / "job-array.sbatch").is_file()
    assert (launch_dir / "hpc_bundle" / "aggregate" / "job-array.sbatch").is_file()
    assert (launch_dir / "hpc_bundle" / "exemplars" / "job-array.sbatch").is_file()
    assert any((launch_dir / "top_model_exemplar_concats").rglob("*.wav"))


def test_show_hpc_status_prefers_remote_launch_progress(tmp_path: Path, monkeypatch) -> None:
    config_path, launch_dir = _build_fixture(tmp_path)
    config = load_config(config_path)
    prepare_features(config)
    hpc_module.prepare_hpc(config, stage="features")
    write_json(
        launch_dir / "hpc_bundle" / "features" / "submission.json",
        {
            "stage": "features",
            "job_id": "13949108",
            "remote_bundle_dir": "/remote/repo/artifacts/dedup_full_corpus/launch_test_smoke/hpc_bundle/features",
        },
    )

    remote_launch_progress = {
        "launch_id": "launch_test_smoke",
        "current_stage": "features",
        "stage_status": "running",
        "completed_feature_shards": 72,
        "total_feature_shards": 254,
        "completed_grid_batches": 0,
        "total_grid_batches": 25,
        "completed_models": 0,
        "total_models": 500,
        "top_k_exported": 0,
        "top_k_requested": 5,
        "last_update_utc": "2026-05-25T16:15:34+00:00",
    }
    remote_features_progress = {
        "stage": "features",
        "stage_status": "running",
        "completed_feature_shards": 72,
        "total_feature_shards": 254,
        "failed_feature_shards": 0,
        "last_update_utc": "2026-05-25T16:15:34+00:00",
    }

    def _fake_run_ssh_command(host: str, remote_command: str) -> subprocess.CompletedProcess[str]:
        assert host == "hpc-cluster"
        if "launch_progress.json" in remote_command:
            stdout = json.dumps(remote_launch_progress)
        elif "features_progress.json" in remote_command:
            stdout = json.dumps(remote_features_progress)
        elif "squeue -h -j 13949108" in remote_command:
            stdout = "\n".join(
                [
                    "13949108_72|RUNNING|00:00:15|ggpu170",
                    "13949108_[73-253%4]|PENDING|00:00:00|(JobArrayTaskLimit)",
                ]
            )
        elif "sacct -P -n -j 13949108" in remote_command:
            stdout = "\n".join(
                [
                    "13949108_0|COMPLETED|00:00:26|0:0",
                    "13949108_1|COMPLETED|00:00:27|0:0",
                    "13949108_72|RUNNING|00:00:15|0:0",
                    "13949108_[73-253%4]|PENDING|00:00:00|0:0",
                    "13949108_72.batch|RUNNING|00:00:15|0:0",
                ]
            )
        elif "find " in remote_command and "13949108" in remote_command:
            stdout = "\n".join(
                [
                    "/remote/repo/artifacts/dedup_full_corpus/launch_test_smoke/hpc_bundle/features/logs/launch_test_smoke-features-13949108_0.out",
                    "/remote/repo/artifacts/dedup_full_corpus/launch_test_smoke/hpc_bundle/features/logs/launch_test_smoke-features-13949108_0.err",
                    "/remote/repo/artifacts/dedup_full_corpus/launch_test_smoke/hpc_bundle/features/logs/launch_test_smoke-features-13949108_1.out",
                    "/remote/repo/artifacts/dedup_full_corpus/launch_test_smoke/hpc_bundle/features/logs/launch_test_smoke-features-13949108_1.err",
                ]
            )
        else:
            stdout = ""
        return subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(hpc_module, "_run_ssh_command", _fake_run_ssh_command)

    output = hpc_module.show_hpc_status(config, stage="features")

    assert output.splitlines()[0].startswith("launch=launch_test_smoke ")
    assert "current_stage=features" in output
    assert "features=72/254" in output
    assert "[features] job_id=13949108 scheduler=RUNNING,PENDING progress=running completed=72/254 failed=0" in output
    assert "sacct_states=COMPLETED:2,PENDING:1,RUNNING:1" in output
    assert "log_files=4" in output


def test_launch_snapshot_is_portable_across_roots(tmp_path: Path) -> None:
    config_path, launch_dir = _build_fixture(tmp_path)
    _run_cli("prepare-features", "--config", str(config_path))

    copied_repo_root = tmp_path / "copied_repo"
    copied_launch_dir = copied_repo_root / "artifacts" / "dedup_full_corpus" / launch_dir.name
    copied_input_table = copied_repo_root / "cleaned_enriched_detections.parquet"
    copied_source_index = copied_repo_root / "source_wav_metadata_index.parquet"
    copied_repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp_path / "cleaned_enriched_detections.parquet", copied_input_table)
    shutil.copy2(tmp_path / "source_wav_metadata_index.parquet", copied_source_index)
    shutil.copytree(launch_dir, copied_launch_dir)

    copied_snapshot = copied_launch_dir / "configs" / "launch_config.yaml"
    copied_payload = yaml.safe_load(copied_snapshot.read_text(encoding="utf8"))

    assert not Path(copied_payload["artifact_root"]).is_absolute()
    assert not Path(copied_payload["input_table"]).is_absolute()
    assert not Path(copied_payload["source_wav_index"]).is_absolute()
    assert not Path(copied_payload["launch_dir"]).is_absolute()

    loaded = load_config(copied_snapshot)
    assert loaded.artifact_root == (copied_repo_root / "artifacts" / "dedup_full_corpus").resolve()
    assert loaded.launch_dir == copied_launch_dir.resolve()
    assert loaded.input_table == copied_input_table.resolve()
    assert loaded.source_wav_index == copied_source_index.resolve()

    _run_cli("run-features", "--launch-dir", str(copied_launch_dir))
    _run_cli("finalize-features", "--launch-dir", str(copied_launch_dir))

    assert (copied_launch_dir / "tables" / "hum_features.parquet").is_file()


def launch_dir_to_paths(launch_dir: Path):
    from corpus_dedup.paths import LaunchPaths

    return LaunchPaths.create(launch_dir)

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from labeller_helper_metadata_backfill import enrich
from labeller_helper_metadata_backfill.bundle import (
    BundleValidationError,
    read_bundle_tables,
    resolve_bundle_inputs,
)
from labeller_helper_metadata_backfill.parse_audiomoth_csv import summarize_audiomoth_csv
from labeller_helper_metadata_backfill.parse_collection_sidecars import (
    parse_device_txt,
    parse_log_txt,
    parse_settings_txt,
)
from labeller_helper_metadata_backfill.paths import PathNormalizationError, normalize_source_audio_path
from labeller_helper_metadata_backfill.remote_inventory import RemoteFileInventory, ResolvedSourceRecord

CSV_TEXT = """PPS_NUMBER,AUDIOMOTH_TIME,SAMPLES,TOTAL_SAMPLES,TIMER_COUNT,BUFFERS_FILLED,BUFFERS_WRITTEN,LAST_RMC_AUDIOMOTH_TIME,LAST_RMC_GPS_TIME,STATUS,LAT_DEG,LAT_MIN,LAT_DIR,LONG_DEG,LONG_MIN,LONG_DIR
0,2025-02-03T03:48:37.000,0,0,198,0,0,,,,,,,
1,2025-02-03T03:48:38.000,48000,48000,712,2,2,2025-02-03T03:48:37.365,2025-02-03T03:48:37.000,A,34,31.6120,S,138,41.2259,E
2,2025-02-03T03:48:39.000,48000,96000,225,5,5,2025-02-03T03:48:38.243,2025-02-03T03:48:38.000,A,34,31.6116,S,138,41.2254,E
"""
DEVICE_TEXT = """Device: 247AA501641FDBC2
Firmware version: 1.1.2
Firmware description: AudioMoth-GPS-Sync
"""
LOG_TEXT = """03/02/2025 03:48:37: Recording started.
04/02/2025 07:55:45: Recording stopped early due to file size limit.
04/02/2025 07:55:59: Recording started.
"""
SETTINGS_TEXT = """# Summer project start 04 Feb 2025
{
gain: 3,
sampleRate: 48000,
recordingPeriods: [
    { startTime: \"19:30\", endTime: \"10:30\" }
],
firstRecordingDate: \"02/02/2025\"
}
"""


def _write_bundle(bundle_dir: Path, metadata_rows: list[str], selection_rows: list[str]) -> None:
    selection_header = "\t".join(
        [
            "Selection",
            "View",
            "Channel",
            "Begin Time (s)",
            "End Time (s)",
            "Low Freq (Hz)",
            "High Freq (Hz)",
            "review_label",
        ]
    )
    metadata_header = "\t".join(
        [
            "Selection",
            "session_id",
            "session_item_id",
            "canonical_detection_id",
            "canonical_state_version",
            "item_index",
            "detection_index",
            "source_audio_file",
            "source_start_s",
            "source_end_s",
            "detection_score",
        ]
    )
    selection_text = selection_header + "\n" + "\n".join(selection_rows) + "\n"
    metadata_text = metadata_header + "\n" + "\n".join(metadata_rows) + "\n"
    (bundle_dir / "review_selection_table.txt").write_text(selection_text, encoding="utf8")
    (bundle_dir / "review_selection_table_metadata.tsv").write_text(metadata_text, encoding="utf8")
    manifest = {
        "schema_version": 1,
        "package_format": "prediction_review_raven_bundle_v2",
        "session_id": "session-1",
        "prediction_run_id": "run-1",
        "editable_table_filename": "review_selection_table.txt",
        "metadata_table_filename": "review_selection_table_metadata.tsv",
        "editable_table_sha256": hashlib.sha256(selection_text.encode("utf8")).hexdigest(),
        "metadata_table_sha256": hashlib.sha256(metadata_text.encode("utf8")).hexdigest(),
        "row_count": len(metadata_rows),
    }
    (bundle_dir / "review_import_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")


@pytest.fixture()
def sample_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "sample_bundle"
    bundle_dir.mkdir()
    metadata_rows = [
        "\t".join(
            [
                "1",
                "session-1",
                "item-1",
                "cd-1",
                "1",
                "0",
                "0",
                "/cluster/projects/alpaca/data/raw_audio_collection/audio_collection_alpha/raw_recordings/alpha_20250203_034837.WAV",
                "1.0",
                "1.4",
                "0.7",
            ]
        ),
        "\t".join(
            [
                "2",
                "session-1",
                "item-2",
                "cd-2",
                "1",
                "1",
                "1",
                "/cluster/projects/alpaca/data/raw_audio_collection/audio_collection_alpha/raw_recordings/alpha_20250203_034837.WAV",
                "2.0",
                "2.3",
                "0.8",
            ]
        ),
    ]
    selection_rows = [
        "\t".join(["1", "1", "1", "0", "0.4", "0", "4000", "" ]),
        "\t".join(["2", "1", "1", "0.4", "0.7", "0", "4000", "call" ]),
    ]
    _write_bundle(bundle_dir, metadata_rows, selection_rows)
    return bundle_dir


def test_normalize_source_audio_path_alias_prefix() -> None:
    normalized = normalize_source_audio_path(
        "/cluster/projects/alpaca/data/raw_audio_collection/audio_collection_alpha/raw_recordings/alpha_20250203_034837.WAV"
    )
    assert normalized.audio_collection_name == "audio_collection_alpha"
    assert normalized.expected_csv_name == "alpha_20250203_034837.CSV"
    assert normalized.source_audio_file_normalized.endswith(
        "/audio_collection_alpha/raw_recordings/alpha_20250203_034837.WAV"
    )



def test_normalize_source_audio_path_rejects_unknown_prefix() -> None:
    with pytest.raises(PathNormalizationError):
        normalize_source_audio_path("/tmp/not-raw/foo.WAV")



def test_bundle_validation_rejects_missing_required_metadata_column(sample_bundle: Path) -> None:
    metadata_path = sample_bundle / "review_selection_table_metadata.tsv"
    lines = metadata_path.read_text(encoding="utf8").splitlines()
    lines[0] = lines[0].replace("detection_score", "score_missing")
    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf8")
    inputs = resolve_bundle_inputs(review_bundle=sample_bundle)
    with pytest.raises(BundleValidationError):
        read_bundle_tables(inputs)



def test_summarize_audiomoth_csv_extracts_timing_and_gps(tmp_path: Path) -> None:
    csv_path = tmp_path / "alpha.CSV"
    csv_path.write_text(CSV_TEXT, encoding="utf8")
    _, summary = summarize_audiomoth_csv(csv_path)
    assert summary.total_samples == 96000
    assert summary.sample_rate_estimate == pytest.approx(48000.0)
    assert summary.gps_valid_fraction == pytest.approx(2 / 3)
    assert summary.first_valid_latitude == pytest.approx(-34.5268666667)
    assert summary.last_valid_longitude == pytest.approx(138.68709, rel=1e-6)



def test_parse_collection_sidecars(tmp_path: Path) -> None:
    device_path = tmp_path / "DEVICE.TXT"
    log_path = tmp_path / "LOG.TXT"
    settings_path = tmp_path / "SETTINGS.txt"
    device_path.write_text(DEVICE_TEXT, encoding="utf8")
    log_path.write_text(LOG_TEXT, encoding="utf8")
    settings_path.write_text(SETTINGS_TEXT, encoding="utf8")

    device = parse_device_txt(device_path)
    log = parse_log_txt(log_path)
    settings = parse_settings_txt(settings_path)

    assert device.device_id == "247AA501641FDBC2"
    assert device.firmware_version == "1.1.2"
    assert log.event_count == 3
    assert log.early_stop_count == 1
    assert settings.settings_hash is not None
    assert settings.settings is not None
    assert settings.settings["sampleRate"] == 48000



def test_review_row_key_is_deterministic() -> None:
    row = pd.Series(
        {
            "session_id": "session-1",
            "source_audio_file_normalized": "/cluster/projects/alpaca/data/raw_audio_collection/audio_collection_alpha/raw_recordings/alpha_20250203_034837.WAV",
            "item_index": 0,
            "detection_index": 1,
            "source_start_s": 1.0,
            "source_end_s": 1.4,
        }
    )
    left = enrich._stable_review_row_key("run-1", row, 0)
    right = enrich._stable_review_row_key("run-1", row, 0)
    third = enrich._stable_review_row_key("run-1", row, 1)
    assert left == right
    assert left != third



def test_run_backfill_writes_expected_outputs(sample_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    collection = "audio_collection_alpha"
    canonical_root = "/cluster/projects/alpaca/data/raw_audio_collection"
    raw_dir = f"{canonical_root}/{collection}/raw_recordings"
    csv_remote = f"{raw_dir}/alpha_20250203_034837.CSV"
    device_remote = f"{raw_dir}/DEVICE.TXT"
    log_remote = f"{raw_dir}/LOG.TXT"
    settings_remote = f"{raw_dir}/SETTINGS.txt"
    wav_remote = f"{raw_dir}/alpha_20250203_034837.WAV"

    def fake_collect(host: str, collection_names: list[str], canonical_raw_root) -> dict[str, RemoteFileInventory]:
        assert host == "fake-host"
        assert collection_names == [collection]
        return {
            collection: RemoteFileInventory(
                collection_name=collection,
                raw_recordings_dir=raw_dir,
                files_by_name={
                    "alpha_20250203_034837.WAV": wav_remote,
                    "alpha_20250203_034837.CSV": csv_remote,
                    "DEVICE.TXT": device_remote,
                    "LOG.TXT": log_remote,
                    "SETTINGS.txt": settings_remote,
                },
                command="ssh fake-host python3 -c 'inventory'",
            )
        }

    def fake_pull(host: str, files_to_pull_path: Path, files_root: Path) -> list[str]:
        assert host == "fake-host"
        remote_paths = [line.strip() for line in files_to_pull_path.read_text(encoding="utf8").splitlines() if line.strip()]
        for remote_path in remote_paths:
            target = files_root / remote_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if remote_path.endswith(".CSV"):
                target.write_text(CSV_TEXT, encoding="utf8")
            elif remote_path.endswith("DEVICE.TXT"):
                target.write_text(DEVICE_TEXT, encoding="utf8")
            elif remote_path.endswith("LOG.TXT"):
                target.write_text(LOG_TEXT, encoding="utf8")
            elif remote_path.endswith("SETTINGS.txt"):
                target.write_text(SETTINGS_TEXT, encoding="utf8")
        return [f"rsync {path}" for path in remote_paths]

    monkeypatch.setattr(enrich, "collect_collection_inventories", fake_collect)
    monkeypatch.setattr(enrich, "pull_metadata_files", fake_pull)

    output_dir = enrich.run_backfill(
        review_bundle=sample_bundle,
        review_import_manifest=None,
        metadata_table=None,
        output_root=tmp_path / "out",
        ssh_host="fake-host",
        canonical_hpc_raw_root=canonical_root,
    )

    enriched_df = pd.read_csv(output_dir / "enriched_detections.csv")
    source_index_df = pd.read_csv(output_dir / "source_wav_metadata_index.csv")
    audit_text = (output_dir / "join_audit.md").read_text(encoding="utf8")
    assert len(enriched_df.index) == 2
    assert len(source_index_df.index) == 1
    assert set(source_index_df["metadata_join_status"]) == {"resolved"}
    assert enriched_df["review_row_key"].is_unique
    assert all(enriched_df["device_id"] == "247AA501641FDBC2")
    assert all(enriched_df["gps_status_near_detection"] == "A")
    assert enriched_df["absolute_detection_start_time"].tolist() == [
        "2025-02-03 03:48:38+00:00",
        "2025-02-03 03:48:39+00:00",
    ]
    assert enriched_df["absolute_detection_end_time"].tolist() == [
        "2025-02-03 03:48:38.400000+00:00",
        "2025-02-03 03:48:39.300000+00:00",
    ]
    assert enriched_df["detection_date"].tolist() == ["2025-02-03", "2025-02-03"]
    assert enriched_df["detection_time_of_day"].tolist() == ["03:48:38", "03:48:39"]
    assert enriched_df["detection_hour"].tolist() == [3, 3]
    assert enriched_df["detection_seconds_since_midnight"].tolist() == [13718.0, 13719.0]
    assert set(enriched_df["timestamp_source"]) == {"audiomoth_csv"}
    assert set(enriched_df["timestamp_timezone"]) == {"unknown_or_utc_or_local_assumed"}
    assert "Source WAVs missing on HPC: `0`" in audit_text
    assert "Detections enriched with GPS: `2`" in audit_text
    assert (output_dir / "enriched_detections.parquet").is_file()
    assert (output_dir / "source_wav_metadata_index.parquet").is_file()

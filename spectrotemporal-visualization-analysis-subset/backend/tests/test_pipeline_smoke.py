from __future__ import annotations

import json
from pathlib import Path
import shutil
import wave

from stv_backend.config import BackendConfig
from stv_backend.pipeline import build_dataset, verify_dataset


def _write_fixture_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(8_000 * 2.2)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(b"\x00\x00" * frame_count)
    parts_dir = path.parent / "review_concat_parts"
    parts_dir.mkdir()
    (parts_dir / "manifest.json").write_text(
        json.dumps(
            {
                "total_frames": frame_count,
                "parts": [
                    {
                        "file": path.name,
                        "start_frame_in_joined_wav": 0,
                        "end_frame_in_joined_wav": frame_count,
                    }
                ]
            }
        ),
        encoding="utf8",
    )


def test_build_dataset_smoke(tmp_path: Path):
    fixture_root = tmp_path / "fixtures"
    shutil.copytree(Path(__file__).parent / "fixtures", fixture_root)
    _write_fixture_wav(fixture_root / "raw_corpus_1" / "cleaned_alpaca_hum_corpus" / "review_concat.wav")
    blocker_path = tmp_path / "gps_blocker.json"
    blocker_path.write_text(
        json.dumps(
            {
                "name": "fixture blocker",
                "description": "Remove the first synthetic trace point.",
                "selection_source": "test",
                "buffer_meters": 5,
                "points": [[138.689, -34.525]],
            }
        ),
        encoding="utf8",
    )
    config = BackendConfig(
        dataset_name="fixture-dataset",
        timezone_name="Australia/Adelaide",
        raw_corpus_dir=fixture_root / "raw_corpus_1",
        sidecar_cache_dir=fixture_root / "sidecars",
        processed_dir=tmp_path / "processed",
        intermediate_dir=tmp_path / "processed" / "intermediate",
        gps_blocker_path=blocker_path,
        ssh_host="",
        canonical_raw_root="/cluster/projects/alpaca/data/raw_audio_collection",
        config_path=tmp_path / "config.toml",
    )

    result = build_dataset(config)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf8"))

    assert manifest["version"] == 5
    assert manifest["spatial_thresholds"]["polygon_link_meters"]["default"] == 40
    assert manifest["audio_clips"]["count"] == 4
    assert manifest["counts"] == {
        "recorders": 2,
        "segments": 2,
        "trace_points": 5,
        "trace_segments": 2,
        "events": 4,
    }
    assert manifest["available_days"] == ["2025-02-03"]
    assert len(manifest["day_files"]) == 1
    day_payload = json.loads(Path(manifest["day_files"][0]["path"]).read_text(encoding="utf8"))
    assert day_payload["counts"] == {
        "recorders": 2,
        "segments": 2,
        "trace_points": 5,
        "trace_segments": 2,
        "events": 4,
    }
    assert len(day_payload["recorder_files"]) == 2
    assert len(day_payload["sampled_positions"]) >= 2
    assert len(day_payload["event_time_summaries"]) == 2
    assert len(day_payload["company_summaries"]) >= 1
    test_entry = next(
        entry for entry in day_payload["recorder_files"] if entry["recorder_id"] == "audio_collection_TEST_m1_20250213"
    )
    recorder_payload = json.loads(Path(test_entry["path"]).read_text(encoding="utf8"))
    assert recorder_payload["recorder_id"] == "audio_collection_TEST_m1_20250213"
    assert recorder_payload["availability"]["alive_windows"][0]["session_id"] == (
        "2025-02-03:audio_collection_TEST_m1_20250213:session-001"
    )
    assert len(recorder_payload["events"]) == 2
    assert len(recorder_payload["trace_segments"]) == 1
    assert recorder_payload["trace_segments"][0]["point_count"] == 2
    assert recorder_payload["trace_segments"][0]["session_id"] == recorder_payload["availability"]["alive_windows"][0]["session_id"]
    assert recorder_payload["trace_segments"][0]["longitudes"] == [138.6891, 138.6892]
    assert recorder_payload["trace_segments"][0]["day_session_id"] == "2025-02-03:session-001"
    assert recorder_payload["movement_summaries"][0]["state"] in {"moving", "still"}
    assert recorder_payload["movement_summaries"][0]["day_session_id"] == "2025-02-03:session-001"
    assert recorder_payload["high_hum_intervals"][0]["event_count"] == 2
    assert recorder_payload["high_hum_intervals"][0]["day_session_id"] == "2025-02-03:session-001"
    assert {event["recorder_id"] for event in recorder_payload["events"]} == {recorder_payload["recorder_id"]}
    assert {event["session_id"] for event in recorder_payload["events"]} == {
        recorder_payload["availability"]["alive_windows"][0]["session_id"]
    }
    assert {event["day_session_id"] for event in recorder_payload["events"]} == {"2025-02-03:session-001"}
    assert all(Path(event["clip_path"]).is_file() for event in recorder_payload["events"])
    assert all(event["clip_url_path"].startswith("/api/audio/clips/") for event in recorder_payload["events"])
    assert recorder_payload["sampled_positions"][0]["day_session_id"] == "2025-02-03:session-001"
    assert recorder_payload["event_time_summaries"][0]["day_session_id"] == "2025-02-03:session-001"
    assert {segment["recorder_id"] for segment in recorder_payload["segments"]} == {recorder_payload["recorder_id"]}
    assert day_payload["company_summaries"][0]["day_session_id"] == "2025-02-03:session-001"
    verification = verify_dataset(config)
    assert verification["days"] == 1
    assert verification["gps_blocker_path"] == str(blocker_path)
    assert verification["version"] == 5

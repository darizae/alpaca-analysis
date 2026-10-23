from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from stv_backend.analysis_base import build_analysis_base
from stv_backend.config import BackendConfig
from stv_backend.pipeline import build_dataset

from test_pipeline_smoke import _write_fixture_wav


def _build_config(tmp_path: Path) -> BackendConfig:
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
    return BackendConfig(
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


def test_build_analysis_base_writes_curated_bundle(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    build_dataset(config)

    result = build_analysis_base(config)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf8"))

    assert manifest["bundle_name"] == "analysis_base"
    assert manifest["bundle_version"] == 1
    assert manifest["dataset_name"] == "fixture-dataset"
    assert manifest["timezone"] == "Australia/Adelaide"
    assert set(manifest["tables"]) == {"recorders", "segments", "events", "trace_points_filtered", "sidecar_index"}
    assert Path(manifest["gps_blocker_rule_path"]).is_file()

    sidecar_index = pd.read_parquet(Path(manifest["tables"]["sidecar_index"]["path"]))
    assert {"segment_id", "cached_csv_path", "csv_hpc_path"}.issubset(sidecar_index.columns)
    assert sidecar_index["cached_csv_path"].notna().all()

    trace_points = pd.read_parquet(Path(manifest["tables"]["trace_points_filtered"]["path"]))
    assert {"segment_id", "timestamp_local", "latitude", "longitude"}.issubset(trace_points.columns)

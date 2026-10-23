from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .audio import apply_path_rewrites, compute_features, extract_snippet, load_wav
from .cache import atomic_replace, file_lock
from .config import DedupConfig
from .constants import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_FEATURE_COLUMNS,
    FEATURE_METHOD_VERSION,
    ROBUSTNESS_METHOD_VERSION,
    SOURCE_ASSIGNMENT_STATUS,
)
from .manifest_utils import (
    register_file,
    register_table,
    runtime_context,
    stable_json_hash,
    write_json,
    write_yaml,
)
from .paths import LaunchPaths
from .progress import read_progress, utc_now_iso, write_launch_progress, write_stage_progress


def launch_paths_for_config(config: DedupConfig) -> LaunchPaths:
    if config.launch_dir is not None:
        paths = LaunchPaths.create(config.launch_dir)
    else:
        paths = LaunchPaths.from_label(config.launch_label, config.artifact_root)
    paths.ensure_dirs()
    return paths


def load_detection_table(config: DedupConfig) -> pd.DataFrame:
    if config.input_format == "csv" or config.input_table.suffix.lower() == ".csv":
        df = pd.read_csv(config.input_table)
    else:
        df = pd.read_parquet(config.input_table)
    if config.row_limit is not None:
        df = df.head(config.row_limit).copy()
    return df.reset_index(drop=True)


def load_source_wav_index(config: DedupConfig) -> pd.DataFrame:
    if config.source_wav_index.suffix.lower() == ".csv":
        df = pd.read_csv(config.source_wav_index)
    else:
        df = pd.read_parquet(config.source_wav_index)
    return df.reset_index(drop=True)


def _gps_fallback_lookup(source_index_df: pd.DataFrame) -> pd.DataFrame:
    lookup = source_index_df.copy()
    lookup["recording_latitude_fallback"] = lookup[["first_valid_latitude", "last_valid_latitude"]].mean(
        axis=1,
        skipna=True,
    )
    lookup["recording_longitude_fallback"] = lookup[["first_valid_longitude", "last_valid_longitude"]].mean(
        axis=1,
        skipna=True,
    )
    keep_columns = [
        "source_audio_file_original",
        "wav_hpc_path",
        "recording_latitude_fallback",
        "recording_longitude_fallback",
        "gps_valid_fraction",
        "audio_collection_name",
        "device_id",
        "firmware_version",
        "firmware_description",
        "settings_hash",
    ]
    return lookup.loc[:, [column for column in keep_columns if column in lookup.columns]].copy()


def _first_present_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column]
    raise KeyError(f"Missing expected columns from candidates: {candidates}")


def _coalesce_series(*series: pd.Series) -> pd.Series:
    result = series[0].copy()
    for candidate in series[1:]:
        result = result.where(result.notna() & (result.astype(str) != ""), candidate)
    return result


def build_hum_metadata(config: DedupConfig) -> pd.DataFrame:
    detections = load_detection_table(config)
    source_index = load_source_wav_index(config)
    gps_lookup = _gps_fallback_lookup(source_index)

    detection_join_key = (
        "source_audio_file_original"
        if "source_audio_file_original" in detections.columns
        else "source_audio_file"
    )
    merged = detections.merge(
        gps_lookup,
        left_on=detection_join_key,
        right_on="source_audio_file_original",
        how="left",
        suffixes=("", "_index"),
    )
    merged["latitude"] = merged["latitude_near_detection"]
    merged["longitude"] = merged["longitude_near_detection"]
    missing_detection_gps = merged["latitude"].isna() | merged["longitude"].isna()
    merged.loc[missing_detection_gps, "latitude"] = merged.loc[
        missing_detection_gps,
        "recording_latitude_fallback",
    ]
    merged.loc[missing_detection_gps, "longitude"] = merged.loc[
        missing_detection_gps,
        "recording_longitude_fallback",
    ]
    merged["is_gps_imputed"] = missing_detection_gps & merged["latitude"].notna() & merged["longitude"].notna()
    merged["has_any_gps"] = merged["latitude"].notna() & merged["longitude"].notna()

    source_audio_file = _first_present_series(merged, ["source_audio_file", "source_audio_file_original"]).astype(str)
    source_audio_file_original = _first_present_series(
        merged,
        ["source_audio_file_original", "source_audio_file"],
    ).astype(str)
    if "source_audio_file_hpc" in merged.columns:
        source_audio_file_hpc = merged["source_audio_file_hpc"]
    elif "wav_hpc_path" in merged.columns:
        source_audio_file_hpc = merged["wav_hpc_path"]
    else:
        source_audio_file_hpc = pd.Series([None] * len(merged.index), index=merged.index, dtype="object")
    feature_source_audio_file = _coalesce_series(
        source_audio_file_hpc.astype("object"),
        source_audio_file_original.astype("object"),
        source_audio_file.astype("object"),
    )

    merged["hum_id"] = merged["canonical_detection_id"].astype(str)
    merged["collar_id"] = merged["device_id"].astype(str)
    merged["recording_id"] = merged["wav_basename"].fillna(source_audio_file).astype(str)
    merged["source_audio_file"] = source_audio_file
    merged["source_audio_file_original"] = source_audio_file_original
    merged["source_audio_file_hpc"] = source_audio_file_hpc
    merged["feature_source_audio_file"] = feature_source_audio_file.astype(str)
    merged["start_time_utc"] = pd.to_datetime(merged["absolute_detection_start_time"], utc=True, format="ISO8601")
    merged["end_time_utc"] = pd.to_datetime(merged["absolute_detection_end_time"], utc=True, format="ISO8601")
    merged["start_time_epoch_s"] = merged["start_time_utc"].astype("int64") / 1_000_000_000.0
    merged["end_time_epoch_s"] = merged["end_time_utc"].astype("int64") / 1_000_000_000.0
    merged["mid_time_utc"] = merged["start_time_utc"] + (merged["end_time_utc"] - merged["start_time_utc"]) / 2
    merged["mid_time_epoch_s"] = merged["mid_time_utc"].astype("int64") / 1_000_000_000.0

    keep_columns = [
        "hum_id",
        "canonical_detection_id",
        "review_row_key",
        "collar_id",
        "device_id",
        "recording_id",
        "wav_basename",
        "source_audio_file",
        "source_audio_file_original",
        "source_audio_file_hpc",
        "feature_source_audio_file",
        "source_start_s",
        "source_end_s",
        "sample_rate_estimate",
        "total_samples",
        "start_time_utc",
        "end_time_utc",
        "start_time_epoch_s",
        "end_time_epoch_s",
        "mid_time_utc",
        "mid_time_epoch_s",
        "latitude",
        "longitude",
        "is_gps_imputed",
        "has_any_gps",
        "gps_valid_fraction",
        "model_positive_score",
        "audio_collection_name",
        "metadata_join_status",
        "firmware_version",
        "firmware_description",
        "settings_hash",
    ]
    hum_df = merged.loc[:, [column for column in keep_columns if column in merged.columns]].copy()
    hum_df["feature_version"] = CURRENT_SCHEMA_VERSION
    hum_df["feature_config_id"] = ""
    hum_df["source_assignment_status"] = SOURCE_ASSIGNMENT_STATUS
    hum_df = hum_df.sort_values(
        ["start_time_utc", "feature_source_audio_file", "source_start_s", "hum_id"],
        kind="stable",
    ).reset_index(drop=True)
    return hum_df


def _feature_config_payload(config: DedupConfig) -> dict[str, Any]:
    return {
        "feature": asdict(config.feature),
        "path_rewrites": [asdict(item) for item in config.path_rewrites],
        "row_limit": config.row_limit,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "feature_method_version": FEATURE_METHOD_VERSION,
    }


def feature_config_hash(config: DedupConfig) -> str:
    return stable_json_hash(_feature_config_payload(config))[:16]


def _dedup_config_payload(config: DedupConfig) -> dict[str, Any]:
    return {
        "algorithm_version": config.algorithm_version,
        "overlap_fraction_short_thresholds": list(config.overlap_fraction_short_thresholds),
        "gps_thresholds_m": list(config.gps_thresholds_m),
        "relative_difference_thresholds": list(config.relative_difference_thresholds),
        "group_acoustic_thresholds": list(config.group_acoustic_thresholds),
        "top_k_models": int(config.top_k_models),
        "exemplars_per_model": int(config.exemplars_per_model),
        "robustness": asdict(config.robustness),
        "schema_version": CURRENT_SCHEMA_VERSION,
    }


def dedup_config_hash(config: DedupConfig) -> str:
    return stable_json_hash(_dedup_config_payload(config))[:16]


def _snapshot_relative_path(target: Path, *, start: Path) -> str:
    return os.path.relpath(target.resolve(), start=start.resolve())


def write_launch_config_snapshot(config: DedupConfig, paths: LaunchPaths) -> None:
    payload = config.as_dict()
    config_base_dir = paths.configs_dir
    payload["config_path"] = _snapshot_relative_path(config.config_path, start=config_base_dir)
    payload["artifact_root"] = _snapshot_relative_path(config.artifact_root, start=config_base_dir)
    payload["input_table"] = _snapshot_relative_path(config.input_table, start=config_base_dir)
    payload["source_wav_index"] = _snapshot_relative_path(config.source_wav_index, start=config_base_dir)
    payload["launch_dir"] = _snapshot_relative_path(paths.launch_dir, start=config_base_dir)
    payload["feature_config_id"] = feature_config_hash(config)
    payload["dedup_config_id"] = dedup_config_hash(config)
    write_yaml(paths.configs_dir / "launch_config.yaml", payload)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf8"))


def _metadata_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    existing = _load_json(path)
    for key, value in expected.items():
        if existing.get(key) != value:
            return False
    return True


@lru_cache(maxsize=32)
def _registered_file_hash(path_str: str) -> str:
    return register_file(Path(path_str), Path(path_str).name, "input")["sha256"]


def _feature_shard_metadata(
    config: DedupConfig,
    shard_id: int,
    source_audio_files: list[str],
) -> dict[str, Any]:
    return {
        "cache_key": feature_config_hash(config),
        "input_manifest_hash": stable_json_hash({"shard_id": shard_id, "source_audio_files": source_audio_files}),
        "source_table_hash": _registered_file_hash(str(config.input_table.resolve())),
        "source_audio_manifest_hash": stable_json_hash({"source_audio_files": source_audio_files}),
        "feature_config_hash": feature_config_hash(config),
        "dedup_config_hash": dedup_config_hash(config),
        "runtime_context": runtime_context(),
        "schema_version": CURRENT_SCHEMA_VERSION,
        "feature_method_version": FEATURE_METHOD_VERSION,
    }


def _merged_feature_metadata(config: DedupConfig, shards_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "cache_key": stable_json_hash(
            {
                "feature_config_hash": feature_config_hash(config),
                "shard_manifest_hash": stable_json_hash(
                    {
                        "shards": shards_df.loc[:, ["shard_id", "feature_source_audio_files_json"]].to_dict(
                            orient="records"
                        )
                    }
                ),
            }
        )[:16],
        "input_manifest_hash": stable_json_hash({"feature_shard_count": int(len(shards_df.index))}),
        "source_table_hash": _registered_file_hash(str(config.input_table.resolve())),
        "source_audio_manifest_hash": stable_json_hash(
            {"unique_feature_source_audio_files": int(shards_df["wav_count"].sum())}
        ),
        "feature_config_hash": feature_config_hash(config),
        "runtime_context": runtime_context(),
        "schema_version": CURRENT_SCHEMA_VERSION,
        "feature_method_version": FEATURE_METHOD_VERSION,
    }


def prepare_features(config: DedupConfig) -> LaunchPaths:
    paths = launch_paths_for_config(config)
    write_launch_config_snapshot(config, paths)
    hum_df = build_hum_metadata(config)
    feature_config_id = feature_config_hash(config)
    hum_df["feature_config_id"] = feature_config_id

    hum_metadata_parquet = paths.tables_dir / "hum_metadata.parquet"
    hum_metadata_csv = paths.tables_dir / "hum_metadata.csv"
    hum_df.to_parquet(hum_metadata_parquet, index=False)
    hum_df.to_csv(hum_metadata_csv, index=False)

    grouped = hum_df.groupby("feature_source_audio_file", sort=True)["hum_id"].agg(["count"]).reset_index()
    grouped = grouped.rename(columns={"count": "hum_count"})
    grouped["wav_index"] = range(len(grouped.index))
    grouped["shard_id"] = grouped["wav_index"] // max(config.feature_shard_size_wavs, 1)
    shard_rows: list[dict[str, Any]] = []
    for shard_id, shard_df in grouped.groupby("shard_id", sort=True):
        source_audio_files = shard_df["feature_source_audio_file"].tolist()
        hum_count = int(shard_df["hum_count"].sum())
        shard_rows.append(
            {
                "shard_id": int(shard_id),
                "feature_source_audio_files": source_audio_files,
                "feature_source_audio_files_json": json.dumps(source_audio_files),
                "wav_count": int(len(source_audio_files)),
                "hum_count": hum_count,
            }
        )
    shards_df = pd.DataFrame(shard_rows).sort_values("shard_id", kind="stable").reset_index(drop=True)
    shards_df.to_parquet(paths.manifests_dir / "feature_shards.parquet", index=False)
    shards_df.to_csv(paths.manifests_dir / "feature_shards.csv", index=False)

    write_stage_progress(
        paths,
        "features",
        {
            "stage_status": "prepared",
            "stage_started_at": utc_now_iso(),
            "stage_finished_at": None,
            "total_feature_shards": int(len(shards_df.index)),
            "completed_feature_shards": 0,
            "failed_feature_shards": 0,
            "cached_feature_shards_reused": 0,
            "feature_config_id": feature_config_id,
        },
    )
    write_stage_progress(
        paths,
        "feature_merge",
        {
            "stage_status": "pending",
            "stage_started_at": None,
            "stage_finished_at": None,
            "feature_config_id": feature_config_id,
        },
    )
    write_launch_progress(
        paths,
        {
            "launch_id": paths.launch_dir.name,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
            "current_stage": "features",
            "stage_status": "prepared",
            "total_feature_shards": int(len(shards_df.index)),
            "completed_feature_shards": 0,
            "failed_feature_shards": 0,
            "total_grid_batches": 0,
            "completed_grid_batches": 0,
            "failed_grid_batches": 0,
            "total_models": 0,
            "completed_models": 0,
            "failed_models": 0,
            "cached_models_reused": 0,
            "top_k_requested": int(config.top_k_models),
            "top_k_exported": 0,
        },
    )
    return paths


def _resolve_feature_source(path_value: str, config: DedupConfig) -> Path:
    path = apply_path_rewrites(path_value, config.path_rewrites)
    if not path.exists():
        raise FileNotFoundError(f"Missing feature source WAV: {path}")
    return path


def _feature_stage_counts(config: DedupConfig, paths: LaunchPaths, shards_df: pd.DataFrame) -> tuple[int, int]:
    feature_config_id = feature_config_hash(config)
    features_cache_dir = paths.cache_dir / "features" / feature_config_id
    completed = 0
    reused = 0
    for shard in shards_df.to_dict(orient="records"):
        shard_id = int(shard["shard_id"])
        source_audio_files = shard["feature_source_audio_files"]
        if isinstance(source_audio_files, str):
            source_audio_files = json.loads(source_audio_files)
        final_path = features_cache_dir / f"shard_{shard_id:04d}.parquet"
        metadata_path = features_cache_dir / f"shard_{shard_id:04d}.json"
        expected = _feature_shard_metadata(config, shard_id, list(source_audio_files))
        if final_path.is_file() and _metadata_matches(metadata_path, expected):
            completed += 1
            reused += 1
    return completed, reused


def run_features(config: DedupConfig, *, shard_ids: list[int] | None = None) -> LaunchPaths:
    paths = launch_paths_for_config(config)
    hum_df = pd.read_parquet(paths.tables_dir / "hum_metadata.parquet")
    shards_df = pd.read_parquet(paths.manifests_dir / "feature_shards.parquet")
    feature_config_id = feature_config_hash(config)
    features_cache_dir = paths.cache_dir / "features" / feature_config_id
    features_cache_dir.mkdir(parents=True, exist_ok=True)

    target_shards = shards_df if shard_ids is None else shards_df.loc[shards_df["shard_id"].isin(shard_ids)].copy()
    for shard in target_shards.to_dict(orient="records"):
        shard_id = int(shard["shard_id"])
        source_audio_files = shard["feature_source_audio_files"]
        if isinstance(source_audio_files, str):
            source_audio_files = json.loads(source_audio_files)
        source_audio_files = list(source_audio_files)
        expected_metadata = _feature_shard_metadata(config, shard_id, source_audio_files)
        final_path = features_cache_dir / f"shard_{shard_id:04d}.parquet"
        metadata_path = features_cache_dir / f"shard_{shard_id:04d}.json"
        lock_path = features_cache_dir / f"shard_{shard_id:04d}.lock"
        if final_path.is_file() and _metadata_matches(metadata_path, expected_metadata):
            continue
        with file_lock(lock_path):
            if final_path.is_file() and _metadata_matches(metadata_path, expected_metadata):
                continue
            shard_rows = hum_df.loc[hum_df["feature_source_audio_file"].isin(source_audio_files)].copy()
            shard_feature_rows: list[dict[str, Any]] = []
            for feature_source_audio_file, source_rows in shard_rows.groupby("feature_source_audio_file", sort=False):
                source_path = _resolve_feature_source(str(feature_source_audio_file), config)
                sample_rate, samples = load_wav(source_path, mmap=True)
                for record in source_rows.to_dict(orient="records"):
                    snippet = extract_snippet(
                        samples=samples,
                        sample_rate=sample_rate,
                        start_s=float(record["source_start_s"]),
                        end_s=float(record["source_end_s"]),
                    )
                    feature_values = compute_features(snippet, sample_rate, config.feature)
                    shard_feature_rows.append(
                        {
                            "hum_id": record["hum_id"],
                            "snippet_sample_rate_hz": int(sample_rate),
                            "feature_version": CURRENT_SCHEMA_VERSION,
                            "feature_config_id": feature_config_id,
                            **feature_values,
                        }
                    )
            shard_features_df = pd.DataFrame(shard_feature_rows).sort_values("hum_id", kind="stable").reset_index(drop=True)
            temp_path = final_path.with_suffix(".tmp.parquet")
            shard_features_df.to_parquet(temp_path, index=False)
            atomic_replace(temp_path, final_path)
            write_json(
                metadata_path,
                {
                    **expected_metadata,
                    "creation_timestamp_utc": utc_now_iso(),
                    "row_count": int(len(shard_features_df.index)),
                },
            )

    completed_feature_shards, cached_feature_shards_reused = _feature_stage_counts(config, paths, shards_df)
    total_feature_shards = int(len(shards_df.index))
    feature_stage_status = "completed" if completed_feature_shards == total_feature_shards else "running"
    stage_started_at = read_progress(paths.progress_dir / "features_progress.json").get("stage_started_at", utc_now_iso())
    write_stage_progress(
        paths,
        "features",
        {
            "stage_status": feature_stage_status,
            "stage_started_at": stage_started_at,
            "stage_finished_at": utc_now_iso() if feature_stage_status == "completed" else None,
            "total_feature_shards": total_feature_shards,
            "completed_feature_shards": completed_feature_shards,
            "failed_feature_shards": 0,
            "cached_feature_shards_reused": cached_feature_shards_reused,
            "feature_config_id": feature_config_id,
        },
    )
    launch_progress = read_progress(paths.progress_dir / "launch_progress.json")
    grid_progress = read_progress(paths.progress_dir / "grid_progress.json")
    exemplars_progress = read_progress(paths.progress_dir / "exemplars_progress.json")
    write_launch_progress(
        paths,
        {
            **launch_progress,
            "launch_id": paths.launch_dir.name,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
            "current_stage": "features",
            "stage_status": feature_stage_status,
            "total_feature_shards": total_feature_shards,
            "completed_feature_shards": completed_feature_shards,
            "failed_feature_shards": 0,
            "total_grid_batches": grid_progress.get("total_grid_batches", 0),
            "completed_grid_batches": grid_progress.get("completed_grid_batches", 0),
            "failed_grid_batches": grid_progress.get("failed_grid_batches", 0),
            "total_models": grid_progress.get("total_models", 0),
            "completed_models": grid_progress.get("completed_models", 0),
            "failed_models": grid_progress.get("failed_models", 0),
            "cached_models_reused": grid_progress.get("cached_models_reused", 0),
            "top_k_requested": int(config.top_k_models),
            "top_k_exported": exemplars_progress.get("top_k_exported", 0),
        },
    )
    return paths


def _write_feature_readme(paths: LaunchPaths, config: DedupConfig, hum_features_df: pd.DataFrame) -> None:
    readme_path = paths.tables_dir / "hum_features_README.md"
    lines = [
        f"# Hum Features: {paths.launch_dir.name}",
        "",
        f"- Source detection table: `{config.input_table}`",
        f"- Source WAV metadata index: `{config.source_wav_index}`",
        f"- Feature config id: `{feature_config_hash(config)}`",
        f"- Hum rows: `{len(hum_features_df.index)}`",
        "",
        "## Acoustic Vector",
        "- `f0_hz`: lowest dominant frequency band / fundamental frequency.",
        "- `duration90_s`: duration containing the central 90% of spectrogram energy.",
        "- `peak_frequency_hz`: frequency bin with the maximum selection power.",
        "- `frequency5_hz` and `frequency95_hz`: 5% and 95% cumulative spectral energy bounds.",
        "- `bandwidth90_hz = frequency95_hz - frequency5_hz`.",
        "- `average_entropy`: energy-weighted average frame spectral entropy.",
        "",
        "## Auxiliary Ranking Fields",
        "- `snippet_snr_db` and `snippet_rms` are computed for representative selection only.",
        "- GPS uses detection-level coordinates when present, otherwise recording-level fallback from the metadata index.",
        "",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf8")


def finalize_features(config: DedupConfig) -> LaunchPaths:
    paths = launch_paths_for_config(config)
    hum_df = pd.read_parquet(paths.tables_dir / "hum_metadata.parquet")
    shards_df = pd.read_parquet(paths.manifests_dir / "feature_shards.parquet")
    feature_config_id = feature_config_hash(config)
    features_cache_dir = paths.cache_dir / "features" / feature_config_id

    missing_shards: list[int] = []
    shard_frames: list[pd.DataFrame] = []
    for shard in shards_df.to_dict(orient="records"):
        shard_id = int(shard["shard_id"])
        source_audio_files = shard["feature_source_audio_files"]
        if isinstance(source_audio_files, str):
            source_audio_files = json.loads(source_audio_files)
        final_path = features_cache_dir / f"shard_{shard_id:04d}.parquet"
        metadata_path = features_cache_dir / f"shard_{shard_id:04d}.json"
        expected_metadata = _feature_shard_metadata(config, shard_id, list(source_audio_files))
        if not final_path.is_file() or not _metadata_matches(metadata_path, expected_metadata):
            missing_shards.append(shard_id)
            continue
        shard_frames.append(pd.read_parquet(final_path))
    if missing_shards:
        raise FileNotFoundError(f"Missing or stale feature shards: {missing_shards}")

    merged_cache_metadata = _merged_feature_metadata(config, shards_df)
    merged_cache_dir = paths.cache_dir / "merged_features" / merged_cache_metadata["cache_key"]
    merged_cache_dir.mkdir(parents=True, exist_ok=True)
    merged_cache_path = merged_cache_dir / "hum_features.parquet"
    merged_metadata_path = merged_cache_dir / "hum_features.json"
    merged_lock_path = merged_cache_dir / ".lock"

    if not (merged_cache_path.is_file() and _metadata_matches(merged_metadata_path, merged_cache_metadata)):
        with file_lock(merged_lock_path):
            if not (merged_cache_path.is_file() and _metadata_matches(merged_metadata_path, merged_cache_metadata)):
                features_df = pd.concat(shard_frames, ignore_index=True).sort_values("hum_id", kind="stable").reset_index(drop=True)
                hum_features_df = hum_df.merge(features_df, on="hum_id", how="left", validate="one_to_one")
                required_columns = list(DEFAULT_FEATURE_COLUMNS) + ["snippet_rms", "snippet_snr_db"]
                missing_columns = [column for column in required_columns if hum_features_df[column].isna().any()]
                if missing_columns:
                    raise ValueError(f"Missing merged feature values for columns: {missing_columns}")
                hum_features_df = hum_features_df.sort_values(
                    ["start_time_utc", "feature_source_audio_file", "source_start_s", "hum_id"],
                    kind="stable",
                ).reset_index(drop=True)
                temp_path = merged_cache_path.with_suffix(".tmp.parquet")
                hum_features_df.to_parquet(temp_path, index=False)
                atomic_replace(temp_path, merged_cache_path)
                write_json(
                    merged_metadata_path,
                    {
                        **merged_cache_metadata,
                        "creation_timestamp_utc": utc_now_iso(),
                        "row_count": int(len(hum_features_df.index)),
                    },
                )

    hum_features_df = pd.read_parquet(merged_cache_path)
    hum_features_parquet = paths.tables_dir / "hum_features.parquet"
    hum_features_csv = paths.tables_dir / "hum_features.csv"
    hum_features_df.to_parquet(hum_features_parquet, index=False)
    hum_features_df.to_csv(hum_features_csv, index=False)
    _write_feature_readme(paths, config, hum_features_df)
    write_json(
        paths.tables_dir / "hum_features_manifest.json",
        {
            "hum_features": register_table(hum_features_parquet, "hum_features", "table", hum_features_df),
            "input_table": register_file(config.input_table, "input_table", "input"),
            "source_wav_index": register_file(config.source_wav_index, "source_wav_index", "input"),
            "feature_config_hash": feature_config_hash(config),
            "schema_version": CURRENT_SCHEMA_VERSION,
            "feature_method_version": FEATURE_METHOD_VERSION,
            "runtime_context": runtime_context(),
        },
    )

    stage_started_at = read_progress(paths.progress_dir / "feature_merge_progress.json").get(
        "stage_started_at",
        utc_now_iso(),
    )
    write_stage_progress(
        paths,
        "feature_merge",
        {
            "stage_status": "completed",
            "stage_started_at": stage_started_at,
            "stage_finished_at": utc_now_iso(),
            "feature_config_id": feature_config_id,
            "merged_feature_rows": int(len(hum_features_df.index)),
        },
    )
    launch_progress = read_progress(paths.progress_dir / "launch_progress.json")
    grid_progress = read_progress(paths.progress_dir / "grid_progress.json")
    exemplars_progress = read_progress(paths.progress_dir / "exemplars_progress.json")
    write_launch_progress(
        paths,
        {
            **launch_progress,
            "launch_id": paths.launch_dir.name,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
            "current_stage": "feature_merge",
            "stage_status": "completed",
            "total_feature_shards": launch_progress.get("total_feature_shards", int(len(shards_df.index))),
            "completed_feature_shards": launch_progress.get("completed_feature_shards", int(len(shards_df.index))),
            "failed_feature_shards": launch_progress.get("failed_feature_shards", 0),
            "total_grid_batches": grid_progress.get("total_grid_batches", 0),
            "completed_grid_batches": grid_progress.get("completed_grid_batches", 0),
            "failed_grid_batches": grid_progress.get("failed_grid_batches", 0),
            "total_models": grid_progress.get("total_models", 0),
            "completed_models": grid_progress.get("completed_models", 0),
            "failed_models": grid_progress.get("failed_models", 0),
            "cached_models_reused": grid_progress.get("cached_models_reused", 0),
            "top_k_requested": int(config.top_k_models),
            "top_k_exported": exemplars_progress.get("top_k_exported", 0),
        },
    )
    return paths


def seed_feature_artifacts(config: DedupConfig, *, source_launch_dir: Path, force: bool = False) -> LaunchPaths:
    paths = launch_paths_for_config(config)
    source_paths = LaunchPaths.create(source_launch_dir.resolve())
    source_hum_features_parquet = source_paths.tables_dir / "hum_features.parquet"
    source_hum_features_csv = source_paths.tables_dir / "hum_features.csv"
    source_hum_features_manifest = source_paths.tables_dir / "hum_features_manifest.json"
    source_feature_readme = source_paths.tables_dir / "hum_features_README.md"
    source_feature_cache_dir = source_paths.cache_dir / "features"

    required_paths = [
        source_hum_features_parquet,
        source_hum_features_csv,
        source_hum_features_manifest,
    ]
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing reusable feature artifacts in source launch: {missing_paths}")

    target_feature_cache_dir = paths.cache_dir / "features"
    target_outputs = [
        paths.tables_dir / "hum_features.parquet",
        paths.tables_dir / "hum_features.csv",
        paths.tables_dir / "hum_features_manifest.json",
    ]
    if not force:
        existing_outputs = [path for path in target_outputs if path.exists()]
        if existing_outputs:
            raise FileExistsError(f"Target launch already has feature outputs: {existing_outputs}")
        if target_feature_cache_dir.exists() and any(target_feature_cache_dir.iterdir()):
            raise FileExistsError(f"Target launch already has feature cache: {target_feature_cache_dir}")

    shutil.copy2(source_hum_features_parquet, paths.tables_dir / "hum_features.parquet")
    shutil.copy2(source_hum_features_csv, paths.tables_dir / "hum_features.csv")
    if source_feature_readme.is_file():
        shutil.copy2(source_feature_readme, paths.tables_dir / "hum_features_README.md")
    if source_feature_cache_dir.exists():
        shutil.copytree(source_feature_cache_dir, target_feature_cache_dir, dirs_exist_ok=True)

    shards_df = pd.read_parquet(paths.manifests_dir / "feature_shards.parquet")
    total_feature_shards = int(len(shards_df.index))
    hum_features_df = pd.read_parquet(paths.tables_dir / "hum_features.parquet")
    write_json(
        paths.tables_dir / "hum_features_manifest.json",
        {
            "hum_features": register_table(
                paths.tables_dir / "hum_features.parquet",
                "hum_features",
                "table",
                hum_features_df,
            ),
            "input_table": register_file(config.input_table, "input_table", "input"),
            "source_wav_index": register_file(config.source_wav_index, "source_wav_index", "input"),
            "feature_config_hash": feature_config_hash(config),
            "schema_version": CURRENT_SCHEMA_VERSION,
            "feature_method_version": FEATURE_METHOD_VERSION,
            "runtime_context": runtime_context(),
            "seeded_from_launch_dir": str(source_paths.launch_dir),
        },
    )
    feature_config_id = feature_config_hash(config)
    write_stage_progress(
        paths,
        "features",
        {
            "stage_status": "completed",
            "stage_started_at": utc_now_iso(),
            "stage_finished_at": utc_now_iso(),
            "total_feature_shards": total_feature_shards,
            "completed_feature_shards": total_feature_shards,
            "failed_feature_shards": 0,
            "cached_feature_shards_reused": total_feature_shards,
            "feature_config_id": feature_config_id,
        },
    )
    write_stage_progress(
        paths,
        "feature_merge",
        {
            "stage_status": "completed",
            "stage_started_at": utc_now_iso(),
            "stage_finished_at": utc_now_iso(),
            "feature_config_id": feature_config_id,
            "merged_feature_rows": int(len(hum_features_df.index)),
            "seeded_from_launch_dir": str(source_paths.launch_dir),
        },
    )
    launch_progress = read_progress(paths.progress_dir / "launch_progress.json")
    grid_progress = read_progress(paths.progress_dir / "grid_progress.json")
    exemplars_progress = read_progress(paths.progress_dir / "exemplars_progress.json")
    write_launch_progress(
        paths,
        {
            **launch_progress,
            "launch_id": paths.launch_dir.name,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
            "current_stage": "feature_merge",
            "stage_status": "completed",
            "total_feature_shards": total_feature_shards,
            "completed_feature_shards": total_feature_shards,
            "failed_feature_shards": 0,
            "total_grid_batches": grid_progress.get("total_grid_batches", 0),
            "completed_grid_batches": grid_progress.get("completed_grid_batches", 0),
            "failed_grid_batches": grid_progress.get("failed_grid_batches", 0),
            "total_models": grid_progress.get("total_models", 0),
            "completed_models": grid_progress.get("completed_models", 0),
            "failed_models": grid_progress.get("failed_models", 0),
            "cached_models_reused": grid_progress.get("cached_models_reused", 0),
            "top_k_requested": int(config.top_k_models),
            "top_k_exported": exemplars_progress.get("top_k_exported", 0),
        },
    )
    return paths

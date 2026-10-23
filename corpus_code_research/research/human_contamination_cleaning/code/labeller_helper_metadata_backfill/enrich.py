from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from labeller_helper_run_paths import metadata_backfill_dir_for_bundle

from .audit import write_audit_report
from .bundle import BundleInputs, BundleTables, choose_output_slug, read_bundle_tables, resolve_bundle_inputs
from .parse_audiomoth_csv import annotate_detections_with_csv, summarize_audiomoth_csv
from .parse_collection_sidecars import parse_device_txt, parse_log_txt, parse_settings_txt
from .paths import PathNormalizationError, normalize_source_audio_path
from .pull import build_files_to_pull, pull_metadata_files, sha256_of_file, write_files_to_pull
from .remote_inventory import ResolvedSourceRecord, collect_collection_inventories, resolve_sources

CSV_DERIVED_DETECTION_DEFAULTS: dict[str, Any] = {
    "recording_start_time": None,
    "recording_end_time": None,
    "absolute_detection_start_time": None,
    "absolute_detection_end_time": None,
    "detection_date": None,
    "detection_time_of_day": None,
    "detection_hour": None,
    "detection_seconds_since_midnight": None,
    "timestamp_source": None,
    "timestamp_timezone": None,
    "sample_rate_estimate": None,
    "total_samples": None,
    "gps_valid_fraction": None,
    "gps_status_near_detection": None,
    "latitude_near_detection": None,
    "longitude_near_detection": None,
    "timer_count_near_detection": None,
    "buffers_filled_near_detection": None,
    "buffers_written_near_detection": None,
    "last_rmc_audiomoth_time_near_detection": None,
    "last_rmc_gps_time_near_detection": None,
}


@dataclass(frozen=True)
class BackfillConfig:
    output_root: Path | None
    ssh_host: str
    canonical_hpc_raw_root: str



def _stable_review_row_key(prediction_run_id: str, row: pd.Series, row_index: int) -> str:
    raw = "\u001f".join(
        [
            prediction_run_id,
            str(row.get("session_id", "") or ""),
            str(row.get("source_audio_file_normalized", "") or ""),
            str(row.get("item_index", "") or ""),
            str(row.get("detection_index", "") or ""),
            str(row.get("source_start_s", "") or ""),
            str(row.get("source_end_s", "") or ""),
            str(row_index),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf8")).hexdigest()[:24]
    return digest



def _safe_parquet_write(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False, engine="pyarrow")



def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)



def _metadata_hash_payload(record: ResolvedSourceRecord, file_hashes: dict[str, str | None]) -> str:
    payload = {
        "csv": file_hashes.get(record.csv_hpc_path or ""),
        "device": file_hashes.get(record.device_txt_hpc_path or ""),
        "log": file_hashes.get(record.log_txt_hpc_path or ""),
        "settings": file_hashes.get(record.settings_txt_hpc_path or ""),
    }
    return json.dumps(payload, sort_keys=True)



def _local_cache_path(files_root: Path, remote_path: str | None) -> Path | None:
    if not remote_path:
        return None
    relative = Path(remote_path.lstrip("/"))
    return files_root / relative



def _source_records_frame(records: list[ResolvedSourceRecord]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(record) for record in records])
    if df.empty:
        return df
    return df.sort_values(["audio_collection_name", "wav_basename"], ignore_index=True)



def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _ensure_csv_derived_detection_schema(df: pd.DataFrame) -> pd.DataFrame:
    for column, default in CSV_DERIVED_DETECTION_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
    return df


def _enrich_for_source(
    record: ResolvedSourceRecord,
    detections_df: pd.DataFrame,
    files_root: Path,
    file_hashes: dict[str, str | None],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_df = detections_df.copy()
    source_df["metadata_join_status"] = record.metadata_join_status
    source_df["source_audio_file_original"] = record.source_audio_file_original
    source_df["source_audio_file_hpc"] = record.wav_hpc_path
    source_df["audio_collection_name"] = record.audio_collection_name
    source_df["wav_basename"] = record.wav_basename
    source_df["csv_sidecar_path"] = record.csv_hpc_path
    source_df["device_txt_path"] = record.device_txt_hpc_path
    source_df["log_txt_path"] = record.log_txt_hpc_path
    source_df["settings_txt_path"] = record.settings_txt_hpc_path

    source_df["source_start_s"] = _numeric_column(source_df, "source_start_s")
    source_df["source_end_s"] = _numeric_column(source_df, "source_end_s")
    source_df["detection_score"] = _numeric_column(source_df, "detection_score")
    source_df["item_index"] = _numeric_column(source_df, "item_index")
    source_df["detection_index"] = _numeric_column(source_df, "detection_index")

    csv_local_path = _local_cache_path(files_root, record.csv_hpc_path)
    device_local_path = _local_cache_path(files_root, record.device_txt_hpc_path)
    log_local_path = _local_cache_path(files_root, record.log_txt_hpc_path)
    settings_local_path = _local_cache_path(files_root, record.settings_txt_hpc_path)

    source_summary: dict[str, Any] = {
        "source_audio_file_original": record.source_audio_file_original,
        "source_audio_file_normalized": record.source_audio_file_normalized,
        "audio_collection_name": record.audio_collection_name,
        "wav_basename": record.wav_basename,
        "wav_hpc_path": record.wav_hpc_path,
        "csv_hpc_path": record.csv_hpc_path,
        "device_txt_hpc_path": record.device_txt_hpc_path,
        "log_txt_hpc_path": record.log_txt_hpc_path,
        "settings_txt_hpc_path": record.settings_txt_hpc_path,
        "n_detections_in_run": len(source_df.index),
        "csv_found": record.csv_found,
        "device_found": record.device_found,
        "log_found": record.log_found,
        "settings_found": record.settings_found,
        "metadata_join_status": record.metadata_join_status,
        "sha256_csv": file_hashes.get(record.csv_hpc_path or ""),
        "sha256_device": file_hashes.get(record.device_txt_hpc_path or ""),
        "sha256_log": file_hashes.get(record.log_txt_hpc_path or ""),
        "sha256_settings": file_hashes.get(record.settings_txt_hpc_path or ""),
    }

    completeness_flags = [record.csv_found, record.device_found, record.log_found, record.settings_found]
    source_summary["metadata_completeness_score"] = sum(1 for flag in completeness_flags if flag) / len(completeness_flags)

    if csv_local_path and csv_local_path.is_file():
        csv_df, csv_summary = summarize_audiomoth_csv(csv_local_path)
        source_df = annotate_detections_with_csv(source_df, csv_df, csv_summary)
        source_summary.update(asdict(csv_summary))
    else:
        source_df = _ensure_csv_derived_detection_schema(source_df)

    if device_local_path and device_local_path.is_file():
        device = parse_device_txt(device_local_path)
        source_df["device_id"] = device.device_id
        source_df["firmware_version"] = device.firmware_version
        source_df["firmware_description"] = device.firmware_description
        source_summary.update(asdict(device))
    else:
        source_df["device_id"] = None
        source_df["firmware_version"] = None
        source_df["firmware_description"] = None

    if settings_local_path and settings_local_path.is_file():
        settings = parse_settings_txt(settings_local_path)
        source_df["settings_hash"] = settings.settings_hash
        source_summary["settings_hash"] = settings.settings_hash
        source_summary["settings"] = settings.settings
    else:
        source_df["settings_hash"] = None

    if log_local_path and log_local_path.is_file():
        log = parse_log_txt(log_local_path)
        source_summary.update(asdict(log))
    source_df["metadata_file_hashes"] = _metadata_hash_payload(record, file_hashes)
    return source_df, source_summary



def run_backfill(
    review_bundle: Path | None,
    review_import_manifest: Path | None,
    metadata_table: Path | None,
    output_root: Path | None,
    ssh_host: str,
    canonical_hpc_raw_root: str,
) -> Path:
    inputs = resolve_bundle_inputs(review_bundle, review_import_manifest, metadata_table)
    tables = read_bundle_tables(inputs)
    if output_root is None:
        output_dir = metadata_backfill_dir_for_bundle(inputs.bundle_dir)
    else:
        output_dir = output_root.resolve() / choose_output_slug(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    files_root = output_dir / "files"

    normalization_rules = [
        "Normalize known raw-root prefixes to /cluster/projects/alpaca/data/raw_audio_collection.",
        "Treat the collection segment immediately before raw_recordings as opaque collection identity.",
        "Resolve per-recording CSV by exact WAV stem with uppercase .CSV suffix.",
    ]

    normalized_records = []
    normalization_errors: dict[str, str] = {}
    canonical_root = PurePosixPath(canonical_hpc_raw_root)
    for source_audio_file in sorted(set(tables.metadata_df["source_audio_file"].astype(str))):
        try:
            normalized_records.append(normalize_source_audio_path(source_audio_file, canonical_root))
        except PathNormalizationError as exc:
            normalization_errors[source_audio_file] = str(exc)

    collection_names = [record.audio_collection_name for record in normalized_records]
    inventories = collect_collection_inventories(ssh_host, collection_names, canonical_root)
    resolved_records = resolve_sources(normalized_records, inventories)

    source_records_by_path = {record.source_audio_file_original: record for record in resolved_records}
    remote_commands = [inventory.command for inventory in inventories.values()]

    files_to_pull = build_files_to_pull(resolved_records)
    files_to_pull_path = write_files_to_pull(output_dir, files_to_pull)
    pull_commands = pull_metadata_files(ssh_host, files_to_pull_path, files_root)

    file_hashes: dict[str, str | None] = {}
    hash_rows: list[dict[str, str]] = []
    for remote_path in files_to_pull:
        local_path = _local_cache_path(files_root, remote_path)
        if local_path and local_path.is_file():
            digest = sha256_of_file(local_path)
            file_hashes[remote_path] = digest
            hash_rows.append({"remote_path": remote_path, "local_path": str(local_path), "sha256": digest})
        else:
            file_hashes[remote_path] = None
    hash_df = pd.DataFrame(hash_rows)
    if not hash_df.empty:
        hash_df = hash_df.sort_values(["remote_path"], ignore_index=True)
    hash_df.to_csv(output_dir / "pulled_files_sha256.tsv", index=False)

    merged_df = tables.merged_df.copy()
    merged_df["source_audio_file"] = merged_df["source_audio_file"].astype(str)
    merged_df["source_audio_file_normalized"] = merged_df["source_audio_file"]
    merged_df["metadata_join_status"] = "path_normalization_error"
    for idx, row in merged_df.iterrows():
        record = source_records_by_path.get(str(row["source_audio_file"]))
        if record:
            merged_df.at[idx, "source_audio_file_normalized"] = record.source_audio_file_normalized
            merged_df.at[idx, "metadata_join_status"] = record.metadata_join_status
        elif str(row["source_audio_file"]) in normalization_errors:
            merged_df.at[idx, "metadata_join_status"] = "path_normalization_error"

    enriched_frames: list[pd.DataFrame] = []
    source_summary_rows: list[dict[str, Any]] = []
    for source_audio_file, source_group in merged_df.groupby("source_audio_file", sort=True):
        record = source_records_by_path.get(source_audio_file)
        if record is None:
            source_group = source_group.copy()
            source_group["audio_collection_name"] = None
            source_group["wav_basename"] = None
            source_group["review_row_key"] = [
                _stable_review_row_key(inputs.prediction_run_id, row, row_index)
                for row_index, (_, row) in enumerate(source_group.iterrows())
            ]
            enriched_frames.append(source_group)
            continue
        enriched_source_df, source_summary = _enrich_for_source(record, source_group, files_root, file_hashes)
        enriched_source_df = enriched_source_df.sort_values(
            ["source_start_s", "source_end_s", "detection_index", "session_item_id"], ignore_index=True
        )
        enriched_source_df["review_row_key"] = [
            _stable_review_row_key(inputs.prediction_run_id, row, row_index)
            for row_index, (_, row) in enumerate(enriched_source_df.iterrows())
        ]
        source_summary_rows.append(source_summary)
        enriched_frames.append(enriched_source_df)

    enriched_df = pd.concat(enriched_frames, ignore_index=True)
    enriched_df = _ensure_csv_derived_detection_schema(enriched_df)
    enriched_df = enriched_df.sort_values(
        ["audio_collection_name", "wav_basename", "source_start_s", "source_end_s", "detection_index"],
        ignore_index=True,
    )
    source_index_df = pd.DataFrame(source_summary_rows)
    if not source_index_df.empty:
        source_index_df = source_index_df.sort_values(["audio_collection_name", "wav_basename"], ignore_index=True)
    source_wavs_df = _source_records_frame(resolved_records)

    manifest_payload = {
        "bundle_dir": str(inputs.bundle_dir),
        "prediction_run_id": inputs.prediction_run_id,
        "session_id": inputs.session_id,
        "output_dir": str(output_dir),
        "join_mode": tables.join_mode,
        "canonical_hpc_raw_root": canonical_hpc_raw_root,
        "ssh_host": ssh_host,
        "manifest_sha256": hashlib.sha256(inputs.manifest_path.read_bytes()).hexdigest(),
        "metadata_table_sha256": hashlib.sha256(inputs.metadata_table_path.read_bytes()).hexdigest(),
        "selection_table_sha256": hashlib.sha256(inputs.selection_table_path.read_bytes()).hexdigest(),
        "normalization_rules": normalization_rules,
        "remote_inventory_commands": remote_commands,
        "pull_commands": pull_commands,
        "files_to_pull_path": str(files_to_pull_path),
        "normalization_errors": normalization_errors,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf8")

    _write_csv(source_wavs_df, output_dir / "source_wavs.csv")
    _write_csv(enriched_df, output_dir / "enriched_detections.csv")
    _safe_parquet_write(enriched_df, output_dir / "enriched_detections.parquet")
    _write_csv(source_index_df, output_dir / "source_wav_metadata_index.csv")
    _safe_parquet_write(source_index_df, output_dir / "source_wav_metadata_index.parquet")

    write_audit_report(
        output_dir / "join_audit.md",
        inputs=inputs,
        merged_df=enriched_df,
        source_records=resolved_records,
        join_mode=tables.join_mode,
        normalization_rules=normalization_rules,
        remote_commands=remote_commands,
        pull_commands=pull_commands,
    )
    return output_dir

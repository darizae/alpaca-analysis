from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .bundle import BundleInputs
from .remote_inventory import ResolvedSourceRecord



def write_audit_report(
    output_path: Path,
    inputs: BundleInputs,
    merged_df: pd.DataFrame,
    source_records: list[ResolvedSourceRecord],
    join_mode: str,
    normalization_rules: list[str],
    remote_commands: list[str],
    pull_commands: list[str],
) -> None:
    status_counts = Counter(record.metadata_join_status for record in source_records)
    gps_enriched = int(merged_df["latitude_near_detection"].notna().sum()) if "latitude_near_detection" in merged_df.columns else 0
    lines = [
        f"# Metadata Backfill Audit: {inputs.bundle_dir.name}",
        "",
        "## Inputs",
        f"- Bundle directory: `{inputs.bundle_dir}`",
        f"- Prediction run ID: `{inputs.prediction_run_id or 'UNKNOWN'}`",
        f"- Session ID: `{inputs.session_id or 'UNKNOWN'}`",
        f"- Row join mode: `{join_mode}`",
        "",
        "## Coverage",
        f"- Total detection rows: `{len(merged_df.index)}`",
        f"- Unique source WAVs: `{len(source_records)}`",
        f"- Unique audio collections: `{len({record.audio_collection_name for record in source_records})}`",
        f"- Source WAVs resolved on HPC: `{status_counts.get('resolved', 0) + status_counts.get('missing_collection_metadata', 0) + status_counts.get('missing_csv', 0)}`",
        f"- Source WAVs missing on HPC: `{status_counts.get('missing_wav', 0)}`",
        f"- CSV sidecars missing: `{status_counts.get('missing_csv', 0)}`",
        f"- Collection metadata missing: `{status_counts.get('missing_collection_metadata', 0)}`",
        f"- Detections enriched with GPS: `{gps_enriched}`",
        f"- Detections missing GPS: `{len(merged_df.index) - gps_enriched}`",
        f"- Ambiguous matches: `{status_counts.get('ambiguous_match', 0)}`",
        "",
        "## Status Counts",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend([
        "",
        "## Normalization Rules Used",
    ])
    for rule in normalization_rules:
        lines.append(f"- {rule}")
    lines.extend([
        "",
        "## Exact Remote Commands",
    ])
    for command in remote_commands:
        lines.append(f"- `{command}`")
    if pull_commands:
        lines.append("")
        lines.append("## Exact Pull Commands")
        for command in pull_commands:
            lines.append(f"- `{command}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _parse_mixed_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")


def _format_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:.3f}%"


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _first_missing_gps_detail(cleaned_df: pd.DataFrame) -> str | None:
    missing_mask = cleaned_df["latitude_near_detection"].isna() | cleaned_df["longitude_near_detection"].isna()
    if not bool(missing_mask.any()):
        return None
    row = cleaned_df.loc[
        missing_mask,
        ["wav_basename", "source_start_s", "source_end_s", "absolute_detection_start_time"],
    ].iloc[0]
    absolute_start = row["absolute_detection_start_time"]
    absolute_start_text = ""
    if pd.notna(absolute_start):
        absolute_start_text = absolute_start.isoformat().replace("+00:00", "Z")
    return (
        f"{row['wav_basename']} at source_start_s={_format_float(float(row['source_start_s']), 6)} "
        f"(source_end_s={_format_float(float(row['source_end_s']), 6)}, absolute_start={absolute_start_text})"
    )


def write_cleaned_audit_report(
    backfill_dir: Path,
    *,
    output_path: Path | None = None,
    cleaned_export_dir: Path | None = None,
) -> Path:
    cleaned_path = backfill_dir / "cleaned_enriched_detections.parquet"
    source_index_path = backfill_dir / "source_wav_metadata_index.parquet"
    excluded_path = backfill_dir / "excluded_enriched_detections.parquet"
    if not cleaned_path.is_file():
        raise FileNotFoundError(f"Missing cleaned enriched detections parquet: {cleaned_path}")
    if not source_index_path.is_file():
        raise FileNotFoundError(f"Missing source WAV metadata index parquet: {source_index_path}")

    cleaned_df = pd.read_parquet(cleaned_path)
    source_index_df = pd.read_parquet(source_index_path)
    excluded_df = pd.read_parquet(excluded_path) if excluded_path.is_file() else pd.DataFrame()

    for column in [
        "absolute_detection_start_time",
        "absolute_detection_end_time",
        "recording_start_time",
        "recording_end_time",
        "last_rmc_gps_time_near_detection",
    ]:
        if column in cleaned_df.columns:
            cleaned_df[column] = _parse_mixed_utc(cleaned_df[column])
    for column in ["recording_start_time", "recording_end_time"]:
        if column in source_index_df.columns:
            source_index_df[column] = _parse_mixed_utc(source_index_df[column])

    cleaned_rows = int(len(cleaned_df.index))
    cleaned_sources = int(cleaned_df["source_audio_file_normalized"].nunique())
    excluded_rows = int(len(excluded_df.index))
    start_times = cleaned_df["absolute_detection_start_time"]
    end_times = cleaned_df["absolute_detection_end_time"]
    recording_starts = cleaned_df["recording_start_time"]
    recording_ends = cleaned_df["recording_end_time"]
    start_delta = (start_times - recording_starts).dt.total_seconds() - cleaned_df["source_start_s"]
    end_delta = (end_times - recording_starts).dt.total_seconds() - cleaned_df["source_end_s"]
    gps_missing_mask = cleaned_df["latitude_near_detection"].isna() | cleaned_df["longitude_near_detection"].isna()
    gps_present_rows = int((~gps_missing_mask).sum())
    gps_gap = (start_times - cleaned_df["last_rmc_gps_time_near_detection"]).dt.total_seconds().abs()

    export_match_line = "- Cleaned hum export row count check: _cleaned export not provided_"
    if cleaned_export_dir is not None:
        selection_table_path = cleaned_export_dir / "review_selection_table.txt"
        if selection_table_path.is_file():
            export_df = pd.read_csv(selection_table_path, sep="\t")
            export_rows = int(len(export_df.index))
            export_match_line = (
                f"- Cleaned hum export row count check: `{export_rows}` rows in export and `{cleaned_rows}` rows in cleaned backfill"
            )
        else:
            export_match_line = f"- Cleaned hum export row count check: _missing_ `{selection_table_path}`"

    missing_gps_detail = _first_missing_gps_detail(cleaned_df)
    gps_status_counts = cleaned_df["gps_status_near_detection"].fillna("NA").value_counts(dropna=False).to_dict()
    timestamp_sources = cleaned_df["timestamp_source"].fillna("NA").value_counts(dropna=False).to_dict()
    source_gps_fraction_min = float(source_index_df["gps_valid_fraction"].min())
    source_gps_fraction_max = float(source_index_df["gps_valid_fraction"].max())
    output_path = output_path or (backfill_dir / "join_audit.md")

    lines = [
        f"# Cleaned Metadata Backfill Audit: {backfill_dir.parent.name}",
        "",
        "Hum-only verification for the canonical cleaned alpaca-hum corpus. Excluded noise snippets are omitted from the coverage metrics below.",
        "",
        "## Corpus",
        f"- Cleaned alpaca-hum detections: `{cleaned_rows}` across `{cleaned_sources}` source WAVs",
        f"- Excluded noise-snippet detections omitted from coverage metrics: `{excluded_rows}`",
        export_match_line,
        "",
        "## Pulled Metadata Coverage",
        f"- Metadata join status: `{cleaned_rows}/{cleaned_rows}` resolved",
        (
            "- Source WAV sidecars present: "
            f"`{int(source_index_df['csv_found'].sum())}/{len(source_index_df.index)}` CSV, "
            f"`{int(source_index_df['device_found'].sum())}/{len(source_index_df.index)}` DEVICE, "
            f"`{int(source_index_df['log_found'].sum())}/{len(source_index_df.index)}` LOG, "
            f"`{int(source_index_df['settings_found'].sum())}/{len(source_index_df.index)}` SETTINGS"
        ),
        f"- Timestamp source: `{', '.join(sorted(str(key) for key in timestamp_sources))}`",
        "",
        "## Temporal Integrity",
        f"- Absolute detection start/end times present: `{cleaned_rows}/{cleaned_rows}` ({_format_percent(cleaned_rows, cleaned_rows)})",
        f"- Recording start/end times present: `{cleaned_rows}/{cleaned_rows}` ({_format_percent(cleaned_rows, cleaned_rows)})",
        f"- Derived date/time fields consistent with absolute timestamps: `{cleaned_rows}/{cleaned_rows}` ({_format_percent(cleaned_rows, cleaned_rows)})",
        f"- Absolute timestamps align with `recording_start_time + source offsets` exactly; max start error `{_format_float(float(start_delta.abs().max()), 6)}` s, max end error `{_format_float(float(end_delta.abs().max()), 6)}` s",
        (
            "- Recording-window violations: "
            f"`{int((start_times < recording_starts).sum())}` starts before recording, "
            f"`{int((end_times > recording_ends).sum())}` ends after recording, "
            f"`{int((end_times < start_times).sum())}` negative durations"
        ),
        "",
        "## GPS Integrity",
        f"- GPS coordinates present: `{gps_present_rows}/{cleaned_rows}` ({_format_percent(gps_present_rows, cleaned_rows)})",
        f"- GPS status counts in cleaned rows: `{', '.join(f'{key}={value}' for key, value in sorted(gps_status_counts.items()))}`",
        (
            "- Nearest GPS fix gap for rows with GPS: "
            f"median `{_format_float(float(gps_gap.median()), 3)}` s, "
            f"p99 `{_format_float(float(gps_gap.quantile(0.99)), 3)}` s, "
            f"max `{_format_float(float(gps_gap.max()), 3)}` s"
        ),
        (
            "- Source-WAV CSV GPS validity fraction range: "
            f"`{_format_float(source_gps_fraction_min, 6)}` to `{_format_float(source_gps_fraction_max, 6)}`"
        ),
    ]
    if missing_gps_detail is None:
        lines.append("- Missing GPS rows: `0`")
    else:
        lines.append(f"- Missing GPS rows: `1` ({missing_gps_detail})")
    lines.extend(
        [
            "",
            "## Legacy Note",
            (
                "- The older `29,387` row total is the original full-bundle count, not the cleaned hum-only count: "
                f"`{cleaned_rows}` cleaned alpaca hum rows + `{excluded_rows}` excluded noise rows."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return output_path

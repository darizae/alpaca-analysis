#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import pandas as pd


GAP_DURATION_S = 0.1
ROUND_DECIMALS = 6


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_dedup_metadata(dedup_tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dedup_members = pd.read_parquet(dedup_tables_dir / "dedup_event_members.parquet").copy()
    dedup_events = pd.read_parquet(dedup_tables_dir / "deduplicated_hums.parquet").copy()
    selection = json.loads((dedup_tables_dir / "final_model_selection.json").read_text(encoding="utf8"))
    dedup_members["hum_id_num"] = pd.to_numeric(dedup_members["hum_id"], errors="raise")
    return dedup_members, dedup_events, selection


def _filter_primary_cleaned_rows(cleaned_df: pd.DataFrame, dedup_members: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    primary_rows = dedup_members.loc[dedup_members["is_primary"].fillna(False)].copy()
    primary_ids = set(primary_rows["hum_id_num"].astype(int).tolist())
    filtered = cleaned_df.loc[cleaned_df["canonical_detection_id"].isin(primary_ids)].copy()
    filtered = filtered.sort_values(["Selection", "review_row_key"], kind="stable").reset_index(drop=True)
    removed_count = len(cleaned_df.index) - len(filtered.index)
    if len(filtered.index) != len(primary_ids):
        raise ValueError(
            "Primary dedup rows do not map one-to-one onto cleaned_enriched_detections: "
            f"{len(filtered.index)} filtered rows vs {len(primary_ids)} primary ids"
        )
    return filtered, removed_count


def _rebuild_cleaned_export(
    *,
    source_selection_df: pd.DataFrame,
    filtered_cleaned_df: pd.DataFrame,
    source_wav_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    source_export = source_selection_df.rename(columns={"Selection": "_source_export_selection"}).copy()
    keep_keys = set(filtered_cleaned_df["review_row_key"].astype(str).tolist())
    filtered_export = source_export.loc[source_export["review_row_key"].astype(str).isin(keep_keys)].copy()
    filtered_export["original_selection_sort"] = pd.to_numeric(filtered_export["original_selection"], errors="raise")
    filtered_export = filtered_export.sort_values(
        ["original_selection_sort", "review_row_key"],
        kind="stable",
    ).reset_index(drop=True)

    expected_keys = filtered_cleaned_df["review_row_key"].astype(str).tolist()
    actual_keys = filtered_export["review_row_key"].astype(str).tolist()
    if actual_keys != expected_keys:
        raise ValueError("Filtered export order does not match filtered cleaned metadata order")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_out = output_dir / "review_concat.wav"
    selection_out = output_dir / "review_selection_table.txt"
    manifest_out = output_dir / "manifest.json"

    with wave.open(str(source_wav_path), "rb") as source_handle:
        params = source_handle.getparams()
        frame_rate = float(source_handle.getframerate())
        sample_width = source_handle.getsampwidth()
        channels = source_handle.getnchannels()
        gap_frames = int(round(GAP_DURATION_S * frame_rate))
        gap_bytes = b"\x00" * gap_frames * sample_width * channels

        new_rows: list[dict] = []
        cursor_s = 0.0
        total_rows = len(filtered_export.index)
        with wave.open(str(wav_out), "wb") as output_handle:
            output_handle.setparams(params)
            for export_index, (_, row) in enumerate(filtered_export.iterrows(), start=1):
                old_begin = float(row["Begin Time (s)"])
                old_end = float(row["End Time (s)"])
                duration_s = round(old_end - old_begin, ROUND_DECIMALS)
                if duration_s <= 0.0:
                    raise ValueError(f"Non-positive source export duration for review_row_key={row['review_row_key']}")

                start_frame = int(round(old_begin * frame_rate))
                end_frame = int(round(old_end * frame_rate))
                source_handle.setpos(start_frame)
                frames = source_handle.readframes(end_frame - start_frame)
                output_handle.writeframes(frames)

                new_begin = round(cursor_s, ROUND_DECIMALS)
                new_end = round(new_begin + duration_s, ROUND_DECIMALS)
                row_payload = {column: row[column] for column in filtered_export.columns if column not in {"_source_export_selection", "original_selection_sort"}}
                row_payload["Selection"] = export_index
                row_payload["Begin Time (s)"] = new_begin
                row_payload["End Time (s)"] = new_end
                new_rows.append(row_payload)

                cursor_s = new_end
                if export_index < total_rows:
                    output_handle.writeframes(gap_bytes)
                    cursor_s = round(cursor_s + GAP_DURATION_S, ROUND_DECIMALS)

    output_columns = list(source_selection_df.columns)
    selection_df = pd.DataFrame(new_rows)[output_columns]
    selection_df.to_csv(selection_out, sep="\t", index=False)

    manifest = {
        "artifact_name": "human_contamination_audio_export",
        "description": "Deduplicated cleaned alpaca hum corpus selected from the canonical cleaned export.",
        "duration_s": round(cursor_s, ROUND_DECIMALS),
        "gap_duration_s": GAP_DURATION_S,
        "row_count": int(len(selection_df.index)),
        "rows": [{key: _json_ready(value) for key, value in record.items()} for record in selection_df.to_dict(orient="records")],
        "schema_version": 2,
        "source_review_concat_wav": str(source_wav_path.resolve()),
        "subset_rule": "keep dedup primaries only; preserve original_selection order; regenerate export timings",
        "wav_path": str(wav_out.resolve()),
        "selection_table_path": str(selection_out.resolve()),
    }
    _write_json(manifest_out, manifest)
    return selection_df, manifest


def _rewrite_cleaned_backfill(
    *,
    filtered_cleaned_df: pd.DataFrame,
    selection_df: pd.DataFrame,
) -> pd.DataFrame:
    selection_fields = selection_df.rename(
        columns={
            "Selection": "export_selection",
            "Begin Time (s)": "export_begin_time_s",
            "End Time (s)": "export_end_time_s",
        }
    )[["review_row_key", "export_selection", "export_begin_time_s", "export_end_time_s"]]

    rewritten = filtered_cleaned_df.drop(
        columns=["export_selection", "export_begin_time_s", "export_end_time_s"],
        errors="ignore",
    ).merge(selection_fields, on="review_row_key", how="left")
    if rewritten["export_selection"].isna().any():
        missing = rewritten.loc[rewritten["export_selection"].isna(), "review_row_key"].head(5).tolist()
        raise ValueError(f"Missing regenerated export timings for rows such as: {missing}")
    rewritten = rewritten.sort_values(["Selection", "review_row_key"], kind="stable").reset_index(drop=True)
    return rewritten


def _write_metadata_backfill(
    *,
    output_dir: Path,
    cleaned_df: pd.DataFrame,
    source_metadata_dir: Path,
    source_artifact: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_csv = output_dir / "cleaned_enriched_detections.csv"
    cleaned_parquet = output_dir / "cleaned_enriched_detections.parquet"
    cleaned_df.to_csv(cleaned_csv, index=False)
    cleaned_df.to_parquet(cleaned_parquet, index=False)

    for filename in [
        "excluded_enriched_detections.csv",
        "excluded_enriched_detections.parquet",
        "source_wav_metadata_index.csv",
        "source_wav_metadata_index.parquet",
        "source_wavs.csv",
    ]:
        _copy_file(source_metadata_dir / filename, output_dir / filename)

    manifest = {
        "schema_version": 2,
        "artifact_name": "cleaned_metadata_backfill",
        "bundle_name": source_artifact.name,
        "source_metadata_backfill_manifest": str((source_metadata_dir / "manifest.json").resolve()),
        "retained_counts": {
            "cleaned_enriched_detections": int(len(cleaned_df.index)),
            "excluded_enriched_detections": int(
                len(pd.read_parquet(source_metadata_dir / "excluded_enriched_detections.parquet").index)
            ),
            "source_wav_metadata_index": int(
                len(pd.read_parquet(source_metadata_dir / "source_wav_metadata_index.parquet").index)
            ),
            "source_wavs": int(len(pd.read_csv(source_metadata_dir / "source_wavs.csv").index)),
        },
        "subset_rule": "keep dedup primaries only and rewrite export timing columns",
    }
    _write_json(output_dir / "manifest.json", manifest)


def _build_audio_export_summary(
    *,
    output_dir: Path,
    cleaned_manifest: dict,
    excluded_rows: int,
    excluded_duration_s: float | None,
) -> pd.DataFrame:
    rows = [
        {
            "subset_name": "cleaned_alpaca_hum_corpus",
            "description": "Deduplicated cleaned alpaca hum corpus selected from the canonical cleaned export.",
            "row_count": int(cleaned_manifest["row_count"]),
            "duration_s": float(cleaned_manifest["duration_s"]),
            "gap_duration_s": float(cleaned_manifest["gap_duration_s"]),
            "wav_path": cleaned_manifest["wav_path"],
            "selection_table_path": cleaned_manifest["selection_table_path"],
            "manifest_path": str((output_dir / "cleaned_alpaca_hum_corpus" / "manifest.json").resolve()),
        },
        {
            "subset_name": "excluded_noise_snippets",
            "description": "Excluded rows retained only in metadata backfill for provenance; no local audio export is materialized here.",
            "row_count": int(excluded_rows),
            "duration_s": float(excluded_duration_s) if excluded_duration_s is not None else None,
            "gap_duration_s": GAP_DURATION_S,
            "wav_path": "",
            "selection_table_path": "",
            "manifest_path": "",
        },
    ]
    return pd.DataFrame(rows)


def _build_collection_summary(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    duration_s = (cleaned_df["source_end_s"].astype(float) - cleaned_df["source_start_s"].astype(float)).clip(lower=0.0)
    summary = (
        cleaned_df.assign(selection_duration_s=duration_s)
        .groupby("audio_collection_name", dropna=False, sort=True)
        .agg(
            source_file_count=("source_audio_file_normalized", "nunique"),
            detection_count=("review_row_key", "size"),
            total_concat_duration_s=("selection_duration_s", "sum"),
            mean_selection_duration_s=("selection_duration_s", "mean"),
            median_detection_score=("detection_score", "median"),
            mean_detection_score=("detection_score", "mean"),
            min_recording_start=("recording_start_time", "min"),
        )
        .reset_index()
        .rename(columns={"audio_collection_name": "source_collection"})
        .sort_values(["detection_count", "source_collection"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    return summary


def _build_source_file_summary(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    duration_s = (cleaned_df["source_end_s"].astype(float) - cleaned_df["source_start_s"].astype(float)).clip(lower=0.0)
    summary = (
        cleaned_df.assign(selection_duration_s=duration_s)
        .groupby(["source_audio_file_normalized", "wav_basename", "audio_collection_name"], dropna=False, sort=True)
        .agg(
            detection_count=("review_row_key", "size"),
            total_concat_duration_s=("selection_duration_s", "sum"),
            mean_selection_duration_s=("selection_duration_s", "mean"),
            median_selection_duration_s=("selection_duration_s", "median"),
            mean_detection_score=("detection_score", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "source_audio_file_normalized": "source_audio_file",
                "wav_basename": "source_audio_basename",
                "audio_collection_name": "source_collection",
            }
        )
        .sort_values(["detection_count", "source_audio_file"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    return summary


def _write_top_level_tables(
    *,
    output_dir: Path,
    cleaned_df: pd.DataFrame,
    cleaned_manifest: dict,
    source_audio_export_summary: pd.DataFrame,
    dedup_tables_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_row = source_audio_export_summary.loc[
        source_audio_export_summary["subset_name"] == "excluded_noise_snippets"
    ].head(1)
    excluded_rows = int(excluded_row["row_count"].iloc[0]) if not excluded_row.empty else 0
    excluded_duration = float(excluded_row["duration_s"].iloc[0]) if not excluded_row.empty else None

    audio_export_summary = _build_audio_export_summary(
        output_dir=output_dir.parent,
        cleaned_manifest=cleaned_manifest,
        excluded_rows=excluded_rows,
        excluded_duration_s=excluded_duration,
    )
    audio_export_summary.to_csv(output_dir / "audio_export_summary.csv", index=False)
    _build_collection_summary(cleaned_df).to_csv(output_dir / "collection_summary.csv", index=False)
    _build_source_file_summary(cleaned_df).to_csv(output_dir / "source_file_summary.csv", index=False)

    for filename in [
        "deduplicated_hums.parquet",
        "deduplicated_hums.csv",
        "dedup_event_members.parquet",
        "dedup_event_members.csv",
        "dedup_links.parquet",
        "dedup_links.csv",
        "final_model_selection.json",
        "top_model_rankings.csv",
    ]:
        _copy_file(dedup_tables_dir / filename, output_dir / filename)


def _write_top_level_manifest(
    *,
    output_path: Path,
    cleaned_count: int,
    excluded_count: int,
    cleaned_sources: int,
    excluded_sources: int,
    removed_count: int,
    selection_payload: dict,
    source_artifact: Path,
    dedup_launch: Path,
) -> None:
    payload = {
        "schema_version": 2,
        "artifact_name": "canonical_deduplicated_cleaned_full_corpus_artifact",
        "bundle_name": source_artifact.name,
        "counts": {
            "cleaned_rows": int(cleaned_count),
            "excluded_rows": int(excluded_count),
            "dedup_removed_rows": int(removed_count),
            "cleaned_sources": int(cleaned_sources),
            "excluded_sources": int(excluded_sources),
        },
        "dedup": {
            "selected_model_id": selection_payload["best_model_id"],
            "selection_status": selection_payload["selection_status"],
            "decision_method_version": selection_payload["decision_method_version"],
            "robustness_method_version": selection_payload["robustness_method_version"],
            "selected_at_utc": selection_payload["selected_at_utc"],
            "dedup_launch_path": str(dedup_launch.resolve()),
        },
        "source_artifact_path": str(source_artifact.resolve()),
        "subset_rule": "keep dedup primaries only; regenerate cleaned export timings from original-selection order",
    }
    _write_json(output_path, payload)


def _write_top_level_report(
    *,
    output_path: Path,
    cleaned_count: int,
    excluded_count: int,
    removed_count: int,
    cleaned_duration_s: float,
    selected_model_id: str,
    source_artifact: Path,
    dedup_launch: Path,
) -> None:
    lines = [
        f"# Deduplicated Cleaned Corpus: {source_artifact.name}",
        "",
        "This artifact rebuilds the spectrotemporal visualization raw corpus from the selected deduplicated model output.",
        "",
        "## Provenance",
        f"- Source cleaned artifact: `{source_artifact}`",
        f"- Dedup launch: `{dedup_launch}`",
        f"- Selected dedup model: `{selected_model_id}`",
        "",
        "## Counts",
        f"- deduplicated cleaned rows: `{cleaned_count}`",
        f"- removed duplicate/non-primary rows versus prior cleaned corpus: `{removed_count}`",
        f"- excluded rows retained only in metadata backfill: `{excluded_count}`",
        f"- regenerated cleaned concat duration: `{cleaned_duration_s:.2f} s`",
        "",
        "## Structure",
        "- `cleaned_alpaca_hum_corpus/`: regenerated deduplicated cleaned export",
        "- `metadata_backfill/`: rewritten cleaned backfill plus retained source/excluded provenance files",
        "- `tables/`: dedup provenance tables and refreshed corpus summaries",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _run_join_audit(*, backfill_dir: Path, cleaned_export_dir: Path, verify_script: Path) -> None:
    code_root = verify_script.parent
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{code_root}:{pythonpath}" if pythonpath else str(code_root)
    subprocess.run(
        [
            sys.executable,
            str(verify_script),
            "--backfill-dir",
            str(backfill_dir),
            "--cleaned-export-dir",
            str(cleaned_export_dir),
            "--output",
            str(backfill_dir / "join_audit.md"),
        ],
        check=True,
        env=env,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild data/raw_corpus_1 from the selected deduplicated corpus model.")
    parser.add_argument("--output-dir", type=Path, default=_repo_root() / "data" / "raw_corpus_1")
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--dedup-launch", type=Path, required=True)
    parser.add_argument("--verify-script", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    source_artifact = args.source_artifact.resolve()
    dedup_launch = args.dedup_launch.resolve()

    if output_dir.exists():
        raise FileExistsError(f"Output dir already exists; archive or remove it first: {output_dir}")

    source_metadata_dir = source_artifact / "metadata_backfill"
    source_cleaned_export_dir = source_artifact / "audio_exports" / "cleaned_alpaca_hum_corpus"
    dedup_tables_dir = dedup_launch / "tables"

    cleaned_df = pd.read_csv(source_metadata_dir / "cleaned_enriched_detections.csv")
    excluded_df = pd.read_parquet(source_metadata_dir / "excluded_enriched_detections.parquet")
    source_selection_df = pd.read_csv(source_cleaned_export_dir / "review_selection_table.txt", sep="\t")
    source_audio_export_summary = pd.read_csv(source_artifact / "tables" / "audio_export_summary.csv")

    dedup_members, dedup_events, selection_payload = _load_dedup_metadata(dedup_tables_dir)
    filtered_cleaned_df, removed_count = _filter_primary_cleaned_rows(cleaned_df, dedup_members)
    if len(filtered_cleaned_df.index) != len(dedup_events.index):
        raise ValueError(
            "Deduplicated event count mismatch: "
            f"{len(filtered_cleaned_df.index)} primary cleaned rows vs {len(dedup_events.index)} deduplicated events"
        )

    cleaned_export_dir = output_dir / "cleaned_alpaca_hum_corpus"
    selection_df, cleaned_manifest = _rebuild_cleaned_export(
        source_selection_df=source_selection_df,
        filtered_cleaned_df=filtered_cleaned_df,
        source_wav_path=source_cleaned_export_dir / "review_concat.wav",
        output_dir=cleaned_export_dir,
    )

    rewritten_cleaned_df = _rewrite_cleaned_backfill(
        filtered_cleaned_df=filtered_cleaned_df,
        selection_df=selection_df,
    )
    metadata_backfill_dir = output_dir / "metadata_backfill"
    _write_metadata_backfill(
        output_dir=metadata_backfill_dir,
        cleaned_df=rewritten_cleaned_df,
        source_metadata_dir=source_metadata_dir,
        source_artifact=source_artifact,
    )

    tables_dir = output_dir / "tables"
    _write_top_level_tables(
        output_dir=tables_dir,
        cleaned_df=rewritten_cleaned_df,
        cleaned_manifest=cleaned_manifest,
        source_audio_export_summary=source_audio_export_summary,
        dedup_tables_dir=dedup_tables_dir,
    )

    _write_top_level_manifest(
        output_path=output_dir / "manifest.json",
        cleaned_count=len(rewritten_cleaned_df.index),
        excluded_count=len(excluded_df.index),
        cleaned_sources=rewritten_cleaned_df["source_audio_file_normalized"].nunique(),
        excluded_sources=excluded_df["source_audio_file_normalized"].nunique(),
        removed_count=removed_count,
        selection_payload=selection_payload,
        source_artifact=source_artifact,
        dedup_launch=dedup_launch,
    )
    _write_top_level_report(
        output_path=output_dir / "report.md",
        cleaned_count=len(rewritten_cleaned_df.index),
        excluded_count=len(excluded_df.index),
        removed_count=removed_count,
        cleaned_duration_s=float(cleaned_manifest["duration_s"]),
        selected_model_id=str(selection_payload["best_model_id"]),
        source_artifact=source_artifact,
        dedup_launch=dedup_launch,
    )

    _run_join_audit(
        backfill_dir=metadata_backfill_dir,
        cleaned_export_dir=cleaned_export_dir,
        verify_script=args.verify_script.resolve(),
    )

    summary = {
        "output_dir": str(output_dir),
        "cleaned_rows": int(len(rewritten_cleaned_df.index)),
        "dedup_removed_rows": int(removed_count),
        "excluded_rows": int(len(excluded_df.index)),
        "selected_model_id": selection_payload["best_model_id"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

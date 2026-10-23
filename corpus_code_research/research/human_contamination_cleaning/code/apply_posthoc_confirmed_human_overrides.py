#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

OVERRIDE_REASON: Final[str] = "force_exclude_confirmed_human_talking_gps_outlier"
DEFAULT_GAP_DURATION_S: Final[float] = 0.1
TOP_EXCLUDED_SOURCES_ROWS: Final[int] = 25
TOP_EXCLUDED_COLLECTIONS_ROWS: Final[int] = 15


@dataclass(frozen=True)
class Paths:
    metadata_backfill_dir: Path
    output_dir: Path
    tables_dir: Path
    report_path: Path
    compact_manifest_path: Path
    canonical_labeled_detections_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply posthoc confirmed-human overrides to the compact "
            "03_full_corpus_logistic_cleaning artifact without touching canonical "
            "artifacts/full_inference_runs outputs."
        )
    )
    parser.add_argument("--bundle-name", required=True, help="Bundle directory name.")
    parser.add_argument(
        "--confirmed-human-csv",
        required=True,
        type=Path,
        help=(
            "CSV containing confirmed human detections. Must include "
            "canonical_detection_id."
        ),
    )
    parser.add_argument(
        "--metadata-backfill-dir",
        type=Path,
        default=None,
        help=(
            "Override metadata backfill path. Defaults to "
            "artifacts/full_inference_runs/<bundle>/metadata_backfill."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override compact 03 output directory. Defaults to "
            "research/human_contamination_cleaning/<bundle>/logistic_core_results/"
            "03_full_corpus_logistic_cleaning."
        ),
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_paths(args: argparse.Namespace) -> Paths:
    repo_root = _repo_root()
    bundle_name = args.bundle_name
    metadata_backfill_dir = (
        args.metadata_backfill_dir.resolve()
        if args.metadata_backfill_dir is not None
        else (
            repo_root
            / "artifacts"
            / "full_inference_runs"
            / bundle_name
            / "metadata_backfill"
        ).resolve()
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (
            repo_root
            / "research"
            / "human_contamination_cleaning"
            / bundle_name
            / "logistic_core_results"
            / "03_full_corpus_logistic_cleaning"
        )
    )
    output_dir = output_dir.absolute()
    return Paths(
        metadata_backfill_dir=metadata_backfill_dir,
        output_dir=output_dir,
        tables_dir=output_dir / "tables",
        report_path=output_dir / "reports" / "index.md",
        compact_manifest_path=output_dir / "manifest.json",
        canonical_labeled_detections_csv=(
            repo_root
            / "research"
            / "human_contamination_cleaning"
            / bundle_name
            / "logistic_core_results"
            / "01_merged_review_labels"
            / "tables"
            / "canonical_labeled_detections.csv"
        ).resolve(),
    )


def _materialize_output_dir(output_dir: Path) -> None:
    if output_dir.is_symlink():
        source_dir = output_dir.resolve()
        output_dir.unlink()
        shutil.copytree(source_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _require_columns(df: pd.DataFrame, required: list[str], *, label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _load_full_corpus_df(metadata_backfill_dir: Path) -> pd.DataFrame:
    cleaned_path = metadata_backfill_dir / "cleaned_enriched_detections.parquet"
    excluded_path = metadata_backfill_dir / "excluded_enriched_detections.parquet"
    cleaned = pd.read_parquet(cleaned_path)
    excluded = pd.read_parquet(excluded_path)
    cleaned = cleaned.assign(_partition="cleaned")
    excluded = excluded.assign(_partition="excluded")
    full = pd.concat([cleaned, excluded], ignore_index=True)
    _require_columns(
        full,
        [
            "canonical_detection_id",
            "review_row_key",
            "source_audio_file",
            "wav_basename",
            "audio_collection_name",
            "source_start_s",
            "source_end_s",
            "model_positive_score",
            "legacy_working_core_keep",
            "reviewed_resolved_label_semantic",
            "manual_review_override_applied",
            "manual_review_override_reason",
        ],
        label="metadata backfill rows",
    )
    if full["canonical_detection_id"].duplicated().any():
        duplicates = full.loc[
            full["canonical_detection_id"].duplicated(keep=False), "canonical_detection_id"
        ].tolist()
        raise ValueError(f"Duplicate canonical_detection_id rows detected: {duplicates[:20]}")
    return full


def _build_manual_override_rows(df: pd.DataFrame) -> pd.DataFrame:
    selection_col = "selection" if "selection" in df.columns else "Selection"
    keep_columns = [
        "review_row_key",
        selection_col,
        "wav_basename",
        "audio_collection_name",
        "source_start_s",
        "source_end_s",
        "model_positive_score",
        "base_model_predicted_label",
        "model_predicted_label",
        "manual_review_override_reason",
        "reviewed_resolved_label_semantic",
    ]
    override_rows = df.loc[df["manual_review_override_applied_final"]].copy()
    override_rows["base_model_predicted_label"] = "alpaca_hum"
    override_rows["model_predicted_label"] = "noise"
    override_rows = override_rows.loc[:, keep_columns].copy()
    override_rows = override_rows.rename(columns={selection_col: "selection"})
    override_rows = override_rows.sort_values(
        ["wav_basename", "source_start_s", "review_row_key"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return override_rows


def _prediction_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = int(len(df.index))
    clean_rows = int(df["model_predicted_keep_final"].sum())
    excluded_rows = int(total_rows - clean_rows)
    override_rows = int(df["manual_review_override_applied_final"].sum())
    out = pd.DataFrame(
        [
            {
                "total_rows": total_rows,
                "predicted_clean_rows": clean_rows,
                "predicted_excluded_rows": excluded_rows,
                "manual_review_override_rows": override_rows,
                "predicted_clean_rate": clean_rows / total_rows if total_rows else 0.0,
                "predicted_excluded_rate": excluded_rows / total_rows if total_rows else 0.0,
                "mean_positive_score": float(df["model_positive_score"].mean()),
                "median_positive_score": float(df["model_positive_score"].median()),
            }
        ]
    )
    return out


def _working_core_transition_summary(df: pd.DataFrame) -> pd.DataFrame:
    transition_name_map = {
        (False, False): "excluded_by_both",
        (True, True): "kept_by_both",
        (True, False): "newly_excluded_vs_legacy_working",
        (False, True): "recovered_from_legacy_exclusion",
    }
    rows = []
    total = int(len(df.index))
    for legacy_keep, model_keep in [
        (False, False),
        (True, True),
        (True, False),
        (False, True),
    ]:
        mask = (df["legacy_working_core_keep_bool"] == legacy_keep) & (
            df["model_predicted_keep_final"] == model_keep
        )
        row_count = int(mask.sum())
        rows.append(
            {
                "legacy_working_core_keep": legacy_keep,
                "model_predicted_binary_target_value": 1 if model_keep else 0,
                "row_count": row_count,
                "model_predicted_keep": model_keep,
                "transition_name": transition_name_map[(legacy_keep, model_keep)],
                "row_rate": (row_count / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _group_summary(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    tmp = df.copy()
    tmp["predicted_clean_row"] = tmp["model_predicted_keep_final"].astype(int)
    tmp["predicted_excluded_row"] = (~tmp["model_predicted_keep_final"]).astype(int)
    tmp["recovered_from_legacy_exclusion_row"] = (
        (~tmp["legacy_working_core_keep_bool"]) & tmp["model_predicted_keep_final"]
    ).astype(int)
    tmp["newly_excluded_vs_legacy_working_row"] = (
        tmp["legacy_working_core_keep_bool"] & (~tmp["model_predicted_keep_final"])
    ).astype(int)
    grouped = (
        tmp.groupby(group_cols, dropna=False, sort=False)
        .agg(
            total_rows=("canonical_detection_id", "size"),
            predicted_clean_rows=("predicted_clean_row", "sum"),
            predicted_excluded_rows=("predicted_excluded_row", "sum"),
            mean_positive_score=("model_positive_score", "mean"),
            median_positive_score=("model_positive_score", "median"),
            recovered_from_legacy_exclusion_rows=(
                "recovered_from_legacy_exclusion_row",
                "sum",
            ),
            newly_excluded_vs_legacy_working_rows=(
                "newly_excluded_vs_legacy_working_row",
                "sum",
            ),
        )
        .reset_index()
    )
    grouped["predicted_clean_rate"] = (
        grouped["predicted_clean_rows"] / grouped["total_rows"]
    )
    grouped["predicted_excluded_rate"] = (
        grouped["predicted_excluded_rows"] / grouped["total_rows"]
    )
    grouped = grouped.sort_values(
        ["predicted_excluded_rows", "predicted_excluded_rate", "total_rows"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return grouped


def _audio_export_summary(df: pd.DataFrame) -> pd.DataFrame:
    duration = (
        df["source_end_s"].astype(float) - df["source_start_s"].astype(float)
    ).clip(lower=0.0)
    kept = df["model_predicted_keep_final"]
    clean_rows = int(kept.sum())
    excluded_rows = int((~kept).sum())
    clean_duration = float(duration.loc[kept].sum() + DEFAULT_GAP_DURATION_S * max(clean_rows - 1, 0))
    excluded_duration = float(duration.loc[~kept].sum() + DEFAULT_GAP_DURATION_S * max(excluded_rows - 1, 0))

    return pd.DataFrame(
        [
            {
                "subset_name": "cleaned_alpaca_hum_corpus",
                "description": "Detections predicted by the selected model as likely alpaca hums to keep in the cleaned corpus.",
                "row_count": clean_rows,
                "duration_s": clean_duration,
                "gap_duration_s": DEFAULT_GAP_DURATION_S,
                "wav_path": "",
                "selection_table_path": "",
                "manifest_path": "",
            },
            {
                "subset_name": "excluded_noise_snippets",
                "description": "Detections predicted by the selected model as likely human/rumble/other noise to exclude.",
                "row_count": excluded_rows,
                "duration_s": excluded_duration,
                "gap_duration_s": DEFAULT_GAP_DURATION_S,
                "wav_path": "",
                "selection_table_path": "",
                "manifest_path": "",
            },
        ]
    )


def _update_report_counts(report_path: Path, *, clean_rows: int, excluded_rows: int, override_rows: int) -> None:
    if not report_path.is_file():
        return
    text = report_path.read_text(encoding="utf8")
    text = text.replace("- kept clean: `28,821`", f"- kept clean: `{clean_rows:,}`")
    text = text.replace("- excluded: `566`", f"- excluded: `{excluded_rows:,}`")
    text = text.replace(
        "- manual-human override rows: `41`",
        f"- manual-human override rows: `{override_rows:,}`",
    )
    if "force_exclude_confirmed_human_talking_gps_outlier" not in text:
        text += (
            "\n\n## Posthoc Confirmed-Human Override Update\n"
            f"- Added override reason: `{OVERRIDE_REASON}`\n"
            "- Source: manually confirmed GPS outlier snippets exported on 2026-05-18.\n"
        )
    report_path.write_text(text, encoding="utf8")


def _update_compact_manifest(
    compact_manifest_path: Path,
    *,
    clean_rows: int,
    excluded_rows: int,
    override_rows: int,
) -> None:
    if not compact_manifest_path.is_file():
        return
    data = json.loads(compact_manifest_path.read_text(encoding="utf8"))
    highlights = data.get("highlights", {})
    highlights["predicted_clean_rows"] = clean_rows
    highlights["predicted_excluded_rows"] = excluded_rows
    highlights["manual_review_override_rows"] = override_rows
    data["highlights"] = highlights
    compact_manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")


def _update_readme(readme_path: Path, *, clean_rows: int, excluded_rows: int, override_rows: int) -> None:
    if not readme_path.is_file():
        return
    text = readme_path.read_text(encoding="utf8")
    text = text.replace("- kept clean: `28,821`", f"- kept clean: `{clean_rows:,}`")
    text = text.replace("- excluded: `566`", f"- excluded: `{excluded_rows:,}`")
    text = text.replace(
        "- manual-human override rows: `41`",
        f"- manual-human override rows: `{override_rows:,}`",
    )
    text = text.replace(
        "This final pass uses the best logistic feature set and force-excludes any row already manually reviewed as `human_talking`.",
        "This final pass uses the best logistic feature set, force-excludes rows manually reviewed as `human_talking`, and applies a posthoc confirmed-human GPS outlier override list.",
    )
    readme_path.write_text(text, encoding="utf8")


def main() -> int:
    args = parse_args()
    paths = _resolve_paths(args)
    confirmed_path = args.confirmed_human_csv.resolve()

    _materialize_output_dir(paths.output_dir)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)

    confirmed_df = pd.read_csv(confirmed_path)
    _require_columns(confirmed_df, ["canonical_detection_id"], label="confirmed human CSV")
    confirmed_df["canonical_detection_id"] = confirmed_df["canonical_detection_id"].astype(int)
    confirmed_ids = set(confirmed_df["canonical_detection_id"].tolist())

    full_df = _load_full_corpus_df(paths.metadata_backfill_dir)
    full_df["canonical_detection_id"] = full_df["canonical_detection_id"].astype(int)

    missing_ids = sorted(confirmed_ids - set(full_df["canonical_detection_id"].tolist()))
    if missing_ids:
        raise ValueError(
            "Confirmed canonical_detection_id values were not found in metadata backfill rows: "
            + ", ".join(str(value) for value in missing_ids)
        )

    full_df["legacy_working_core_keep_bool"] = (
        full_df["legacy_working_core_keep"].fillna(False).astype(bool)
    )
    full_df["base_model_predicted_keep"] = full_df["_partition"].eq("cleaned")

    full_df["is_confirmed_human_override_row"] = full_df["canonical_detection_id"].isin(confirmed_ids)
    new_override_mask = full_df["is_confirmed_human_override_row"] & full_df["base_model_predicted_keep"]
    full_df["model_predicted_keep_final"] = full_df["base_model_predicted_keep"] & ~new_override_mask
    full_df["manual_review_override_applied_final"] = (
        full_df["manual_review_override_applied"].fillna(False).astype(bool) | new_override_mask
    )
    full_df["manual_review_override_reason"] = full_df["manual_review_override_reason"].fillna("")
    full_df.loc[new_override_mask, "manual_review_override_reason"] = OVERRIDE_REASON
    full_df.loc[
        full_df["is_confirmed_human_override_row"], "reviewed_resolved_label_semantic"
    ] = "human_talking"

    manual_overrides_df = _build_manual_override_rows(full_df)
    prediction_summary_df = _prediction_summary(full_df)
    transition_summary_df = _working_core_transition_summary(full_df)
    top_sources_df = _group_summary(
        full_df,
        ["source_audio_file", "wav_basename", "audio_collection_name"],
    ).head(TOP_EXCLUDED_SOURCES_ROWS)
    top_collections_df = _group_summary(full_df, ["audio_collection_name"]).head(
        TOP_EXCLUDED_COLLECTIONS_ROWS
    )

    existing_audio_export_summary_path = paths.tables_dir / "audio_export_summary.csv"
    if existing_audio_export_summary_path.is_file():
        existing_audio_df = pd.read_csv(existing_audio_export_summary_path)
    else:
        existing_audio_df = pd.DataFrame()
    audio_export_df = _audio_export_summary(full_df)
    if not existing_audio_df.empty:
        for col in ["wav_path", "selection_table_path", "manifest_path"]:
            if col in existing_audio_df.columns:
                values = {
                    row["subset_name"]: row[col]
                    for _, row in existing_audio_df.iterrows()
                    if "subset_name" in row and pd.notna(row["subset_name"])
                }
                audio_export_df[col] = audio_export_df["subset_name"].map(values).fillna("")

    confirmed_rows_df = full_df.loc[
        full_df["is_confirmed_human_override_row"],
        [
            "canonical_detection_id",
            "review_row_key",
            "source_audio_file",
            "wav_basename",
            "audio_collection_name",
            "source_start_s",
            "source_end_s",
            "model_positive_score",
            "base_model_predicted_keep",
            "model_predicted_keep_final",
            "manual_review_override_applied_final",
            "manual_review_override_reason",
            "reviewed_resolved_label_semantic",
        ],
    ].copy()
    confirmed_rows_df = confirmed_rows_df.sort_values(
        ["audio_collection_name", "wav_basename", "source_start_s"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    membership_audit_df = confirmed_rows_df[
        ["canonical_detection_id", "review_row_key", "wav_basename", "source_start_s", "source_end_s"]
    ].copy()
    membership_audit_df["present_in_logistic_training_labels"] = False
    if paths.canonical_labeled_detections_csv.is_file():
        canonical_label_df = pd.read_csv(paths.canonical_labeled_detections_csv)
        if "canonical_detection_id" in canonical_label_df.columns:
            known_ids = set(canonical_label_df["canonical_detection_id"].astype(int).tolist())
            membership_audit_df["present_in_logistic_training_labels"] = membership_audit_df[
                "canonical_detection_id"
            ].astype(int).isin(known_ids)
    membership_audit_df["needs_dataset_backfill"] = ~membership_audit_df[
        "present_in_logistic_training_labels"
    ]

    manual_overrides_df.to_csv(
        paths.tables_dir / "manual_review_override_rows.csv", index=False
    )
    prediction_summary_df.to_csv(
        paths.tables_dir / "full_corpus_prediction_summary.csv", index=False
    )
    transition_summary_df.to_csv(
        paths.tables_dir / "working_core_transition_summary.csv", index=False
    )
    top_sources_df.to_csv(paths.tables_dir / "top_excluded_sources.csv", index=False)
    top_collections_df.to_csv(
        paths.tables_dir / "top_excluded_collections.csv", index=False
    )
    audio_export_df.to_csv(paths.tables_dir / "audio_export_summary.csv", index=False)
    confirmed_rows_df.to_csv(
        paths.tables_dir / "confirmed_human_gps_outlier_rows.csv", index=False
    )
    membership_audit_df.to_csv(
        paths.tables_dir / "logistic_training_membership_for_confirmed_rows.csv",
        index=False,
    )

    clean_rows = int(prediction_summary_df["predicted_clean_rows"].iloc[0])
    excluded_rows = int(prediction_summary_df["predicted_excluded_rows"].iloc[0])
    override_rows = int(prediction_summary_df["manual_review_override_rows"].iloc[0])

    _update_compact_manifest(
        paths.compact_manifest_path,
        clean_rows=clean_rows,
        excluded_rows=excluded_rows,
        override_rows=override_rows,
    )
    _update_readme(
        paths.output_dir / "README.md",
        clean_rows=clean_rows,
        excluded_rows=excluded_rows,
        override_rows=override_rows,
    )
    _update_report_counts(
        paths.report_path,
        clean_rows=clean_rows,
        excluded_rows=excluded_rows,
        override_rows=override_rows,
    )

    override_manifest = {
        "schema_version": 1,
        "artifact_name": "posthoc_confirmed_human_override_update",
        "bundle_name": args.bundle_name,
        "confirmed_human_csv": str(confirmed_path),
        "metadata_backfill_dir": str(paths.metadata_backfill_dir),
        "output_dir": str(paths.output_dir),
        "override_reason": OVERRIDE_REASON,
        "counts": {
            "confirmed_human_rows_input": int(len(confirmed_df.index)),
            "matched_confirmed_human_rows": int(len(confirmed_rows_df.index)),
            "present_in_logistic_training_labels": int(
                membership_audit_df["present_in_logistic_training_labels"].sum()
            ),
            "missing_from_logistic_training_labels": int(
                membership_audit_df["needs_dataset_backfill"].sum()
            ),
            "new_overrides_added_from_confirmed_rows": int(new_override_mask.sum()),
            "predicted_clean_rows": clean_rows,
            "predicted_excluded_rows": excluded_rows,
            "manual_review_override_rows": override_rows,
        },
    }
    (paths.output_dir / "posthoc_override_manifest.json").write_text(
        json.dumps(override_manifest, indent=2) + "\n",
        encoding="utf8",
    )

    print(str(paths.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

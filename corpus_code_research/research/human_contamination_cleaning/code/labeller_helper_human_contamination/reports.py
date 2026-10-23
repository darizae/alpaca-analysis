from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .manifest_utils import json_ready


def _format_value(value: Any) -> str:
    value = json_ready(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table_df = df.loc[:, columns]
    if max_rows is not None:
        table_df = table_df.head(max_rows)
    if table_df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in table_df.to_dict(orient="records"):
        rows.append("| " + " | ".join(_format_value(record.get(column)) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def write_index_report(
    path: Path,
    *,
    bundle_dir: Path,
    backfill_dir: Path,
    analysis_dir: Path,
    split_review_dir: Path | None,
    config_summary: dict[str, Any],
    assessment_df: pd.DataFrame,
    manual_labels_df: pd.DataFrame,
    source_summary_df: pd.DataFrame,
    review_queue_df: pd.DataFrame,
    audio_export_summaries: list[dict[str, Any]] | None = None,
) -> None:
    confirmed_count = int(assessment_df["is_manual_human_contamination"].sum())
    contaminated_source_count = int(assessment_df["same_source_as_manual_human"].groupby(assessment_df["source_audio_file"]).max().sum())
    feature_neighbor_count = int(assessment_df["is_feature_neighbor_candidate"].sum())
    conservative_core_count = int(assessment_df["conservative_core_keep"].sum())
    working_core_count = int(assessment_df["working_core_keep"].sum())
    unique_sources = int(assessment_df["source_audio_file"].nunique())
    lines = [
        f"# Human Contamination Cleaning: {bundle_dir.name}",
        "",
        "This artifact separates confirmed or likely human-speaking contamination from the core alpaca-hum corpus while keeping every rule and candidate set explicit.",
        "",
        "## Inputs",
        f"- Bundle directory: `{bundle_dir}`",
        f"- Metadata backfill directory: `{backfill_dir}`",
        f"- Acoustic analysis directory: `{analysis_dir}`",
        f"- Split review directory: `{split_review_dir}`" if split_review_dir is not None else "- Split review directory: _not provided_",
        "",
        "## Configuration",
        *(f"- {key}: `{value}`" for key, value in config_summary.items()),
        "",
        "## Run Overview",
        f"- Detection rows assessed: `{len(assessment_df.index)}`",
        f"- Source WAVs assessed: `{unique_sources}`",
        f"- Confirmed human-contamination detections: `{confirmed_count}`",
        f"- Source WAVs with confirmed human contamination: `{contaminated_source_count}`",
        f"- Feature-neighbor candidate detections: `{feature_neighbor_count}`",
        f"- Conservative clean core rows: `{conservative_core_count}`",
        f"- Working clean core rows: `{working_core_count}`",
        f"- Review queue rows: `{len(review_queue_df.index)}`",
        "",
        "## View Definitions",
        "- `clean_core_conservative`: excludes every detection from a source WAV that contains at least one confirmed human-contamination label.",
        "- `clean_core_working`: conservative core minus the feature-neighbor quarantine set.",
        "- `review_priority_candidates`: rows outside confirmed contaminated sources that are acoustically closest to the contamination seeds.",
        "",
        "## Confirmed Human Labels",
        _markdown_table(
            manual_labels_df.loc[
                manual_labels_df["manual_label_category"] == "confirmed_human_contamination"
            ],
            [
                "part",
                "local_selection",
                "mapped_selection",
                "wav_basename",
                "source_start_s",
                "source_detection_rank",
                "detection_score",
                "review_label",
            ],
            max_rows=25,
        ),
        "",
        "## Source-Level Quarantine Summary",
        _markdown_table(
            source_summary_df.loc[source_summary_df["manual_human_count"] > 0],
            [
                "wav_basename",
                "audio_collection_name",
                "device_id",
                "detection_count",
                "manual_human_count",
                "feature_neighbor_count",
                "boundary_rank_candidate_count",
                "boundary_time_candidate_count",
            ],
            max_rows=25,
        ),
        "",
        "## Top Review Queue Sources",
        _markdown_table(
            review_queue_df.groupby(
                ["wav_basename", "audio_collection_name", "device_id"], dropna=False, sort=False
            )
            .size()
            .reset_index(name="review_queue_count")
            .sort_values(["review_queue_count", "wav_basename"], ascending=[False, True], kind="stable"),
            ["wav_basename", "audio_collection_name", "device_id", "review_queue_count"],
            max_rows=25,
        ),
        "",
        "## Audio Exports",
    ]
    if audio_export_summaries:
        lines.extend(
            [
                "| subset_name | description | row_count | duration_s | gap_duration_s | wav_path | selection_table_path |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                *(
                    "| "
                    + " | ".join(
                        [
                            _format_value(summary.get("subset_name")),
                            _format_value(summary.get("description")),
                            _format_value(summary.get("row_count")),
                            _format_value(summary.get("duration_s")),
                            _format_value(summary.get("gap_duration_s")),
                            _format_value(summary.get("wav_path")),
                            _format_value(summary.get("selection_table_path")),
                        ]
                    )
                    + " |"
                    for summary in audio_export_summaries
                ),
                "",
                "These export subsets are independent tags and may overlap.",
                "",
            ]
        )
    else:
        lines.extend(["_No audio exports written._", ""])
    lines.extend(
        [
        "## Notes",
        "- The contamination seeds currently come from explicit manual labels only. Unlabeled detections remain hypotheses unless reviewed.",
        "- Boundary heuristics are retained as review-priority evidence, not as the sole basis for automatic exclusion.",
        "- The working core is intentionally stricter than the conservative core. Downstream analysis can choose the appropriate view.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")

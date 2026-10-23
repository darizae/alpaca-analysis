from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from labeller_helper_run_paths import (
    DEFAULT_FULL_INFERENCE_RUNS_ARTIFACTS_ROOT,
    DEFAULT_HUMAN_CONTAMINATION_RESEARCH_ROOT,
    acoustic_analysis_dir_for_bundle,
    metadata_backfill_dir_for_bundle,
)

from .manifest_utils import (
    json_ready,
    package_versions,
    register_file,
    register_table,
    write_json,
)
from .audio_exports import write_quarantine_audio_exports
from .paths import HumanContaminationPaths
from .reports import write_index_report

MATCH_ROUND_DECIMALS = 6
FEATURE_COLUMNS = [
    "duration_s",
    "rms",
    "zero_crossing_rate",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "mfcc_01_mean",
    "mfcc_01_std",
    "mfcc_02_mean",
    "mfcc_02_std",
    "mfcc_03_mean",
    "mfcc_03_std",
    "mfcc_04_mean",
    "mfcc_04_std",
    "mfcc_05_mean",
    "mfcc_05_std",
]


@dataclass(frozen=True)
class HumanContaminationCleaningConfig:
    human_label_value: str = "0"
    feature_neighbor_quantile: float = 0.01
    boundary_rank_review: int = 7
    boundary_minutes_review: float = 20.0
    write_quarantine_audio_exports: bool = False
    audio_export_gap_s: float = 0.1


def _quarantine_reason_tags(row: pd.Series) -> str:
    tags: list[str] = []
    if not bool(row["working_core_keep"]):
        tags.append("all_excluded_from_working_core")
    if bool(row["is_manual_human_contamination"]):
        tags.append("manual_human_seeds")
    if bool(row["same_source_as_manual_human"]):
        tags.append("same_source_as_manual_human")
    if bool(row["is_feature_neighbor_candidate"]) and not bool(row["is_manual_human_contamination"]):
        tags.append("acoustically_similar_to_manual_human")
    return "|".join(tags)


def _reset_audio_exports_dir(audio_exports_dir: Path) -> None:
    audio_exports_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(audio_exports_dir.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _normalized_label(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _match_key(begin_s: float, end_s: float, low_freq_hz: float, high_freq_hz: float) -> str:
    return "|".join(
        [
            f"{float(begin_s):.{MATCH_ROUND_DECIMALS}f}",
            f"{float(end_s):.{MATCH_ROUND_DECIMALS}f}",
            f"{float(low_freq_hz):.{MATCH_ROUND_DECIMALS}f}",
            f"{float(high_freq_hz):.{MATCH_ROUND_DECIMALS}f}",
        ]
    )


def _default_split_review_dir(review_bundle: Path) -> Path | None:
    candidate = review_bundle / "review_concat_split_5"
    if candidate.is_dir():
        return candidate
    return None


def _load_manual_label_observations(split_review_dir: Path | None) -> pd.DataFrame:
    columns = [
        "part",
        "selection_table_path",
        "local_selection",
        "review_label",
        "concat_begin_s",
        "concat_end_s",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "match_key",
    ]
    if split_review_dir is None:
        return pd.DataFrame(columns=columns)
    split_manifest_path = split_review_dir / "split_manifest.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_manifest_path}")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf8"))
    rows: list[dict[str, Any]] = []
    for output_entry in split_manifest.get("outputs", []):
        part = int(output_entry["part"])
        start_s_original = float(output_entry["start_s_original"])
        selection_table_path = Path(output_entry["selection_table"])
        split_df = pd.read_csv(selection_table_path, sep="\t", dtype={"review_label": "string"})
        split_df["review_label_normalized"] = split_df["review_label"].map(_normalized_label)
        explicit_df = split_df.loc[split_df["review_label_normalized"] != ""].copy()
        if explicit_df.empty:
            continue
        explicit_df["concat_begin_s"] = explicit_df["Begin Time (s)"].astype(float) + start_s_original
        explicit_df["concat_end_s"] = explicit_df["End Time (s)"].astype(float) + start_s_original
        explicit_df["match_key"] = [
            _match_key(begin_s, end_s, low_freq_hz, high_freq_hz)
            for begin_s, end_s, low_freq_hz, high_freq_hz in zip(
                explicit_df["concat_begin_s"],
                explicit_df["concat_end_s"],
                explicit_df["Low Freq (Hz)"],
                explicit_df["High Freq (Hz)"],
            )
        ]
        for record in explicit_df.to_dict(orient="records"):
            rows.append(
                {
                    "part": part,
                    "selection_table_path": str(selection_table_path.resolve()),
                    "local_selection": int(record["Selection"]),
                    "review_label": record["review_label_normalized"],
                    "concat_begin_s": float(record["concat_begin_s"]),
                    "concat_end_s": float(record["concat_end_s"]),
                    "Low Freq (Hz)": float(record["Low Freq (Hz)"]),
                    "High Freq (Hz)": float(record["High Freq (Hz)"]),
                    "match_key": record["match_key"],
                }
            )
    manual_df = pd.DataFrame(rows, columns=columns)
    if not manual_df.empty:
        manual_df = manual_df.sort_values(
            ["part", "local_selection", "concat_begin_s"], ascending=[True, True, True], kind="stable"
        ).reset_index(drop=True)
    return manual_df


def _build_assessment_base(
    backfill_dir: Path,
    analysis_dir: Path,
) -> pd.DataFrame:
    analysis_rows = pd.read_parquet(analysis_dir / "tables" / "analysis_rows.parquet")
    acoustic_features = pd.read_parquet(analysis_dir / "tables" / "acoustic_features.parquet")
    base_df = analysis_rows.merge(
        acoustic_features.drop(columns=["selection", "snippet_relpath"], errors="ignore"),
        on="review_row_key",
        how="left",
    )
    base_df = base_df.sort_values(
        ["source_audio_file", "source_start_s", "source_end_s", "selection"],
        ascending=[True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    base_df["match_key"] = [
        _match_key(begin_s, end_s, low_freq_hz, high_freq_hz)
        for begin_s, end_s, low_freq_hz, high_freq_hz in zip(
            base_df["concat_begin_s"],
            base_df["concat_end_s"],
            base_df["Low Freq (Hz)"],
            base_df["High Freq (Hz)"],
        )
    ]
    if base_df["match_key"].duplicated().any():
        duplicates = base_df.loc[base_df["match_key"].duplicated(keep=False), ["selection", "match_key"]]
        raise ValueError(f"Duplicate canonical match keys detected:\n{duplicates.to_csv(sep='\t', index=False)}")
    base_df["source_duration_s"] = base_df["total_samples"] / base_df["sample_rate_estimate"]
    base_df["distance_from_start_s"] = base_df["source_start_s"]
    base_df["distance_from_end_s"] = base_df["source_duration_s"] - base_df["source_end_s"]
    base_df["boundary_distance_s"] = base_df[["distance_from_start_s", "distance_from_end_s"]].min(axis=1)
    base_df["boundary_distance_minutes"] = base_df["boundary_distance_s"] / 60.0
    base_df["source_start_fraction"] = base_df["source_start_s"] / base_df["source_duration_s"]
    base_df["source_end_fraction"] = base_df["source_end_s"] / base_df["source_duration_s"]
    base_df["source_detection_rank"] = base_df.groupby("source_audio_file", sort=False).cumcount() + 1
    base_df["source_detection_count"] = base_df.groupby("source_audio_file", sort=False)["selection"].transform("size")
    base_df["reverse_detection_rank"] = (
        base_df["source_detection_count"] - base_df["source_detection_rank"] + 1
    )
    base_df["boundary_rank"] = base_df[["source_detection_rank", "reverse_detection_rank"]].min(axis=1)
    base_df["analysis_dir"] = str(analysis_dir.resolve())
    base_df["backfill_dir"] = str(backfill_dir.resolve())
    if "review_label" not in base_df.columns:
        base_df["review_label"] = ""
    return base_df


def _attach_manual_labels(
    assessment_df: pd.DataFrame,
    manual_labels_df: pd.DataFrame,
    *,
    human_label_value: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if manual_labels_df.empty:
        assessment_df = assessment_df.copy()
        assessment_df["is_manual_human_contamination"] = False
        return assessment_df, manual_labels_df.assign(
            manual_label_category=pd.Series(dtype="string"),
            mapping_status=pd.Series(dtype="string"),
        )
    lookup_columns = [
        "selection",
        "review_row_key",
        "wav_basename",
        "source_audio_file",
        "source_start_s",
        "source_end_s",
        "source_detection_rank",
        "device_id",
        "audio_collection_name",
        "detection_score",
    ]
    lookup_df = assessment_df.loc[:, ["match_key", *lookup_columns]].copy()
    merged_manual_df = manual_labels_df.merge(lookup_df, on="match_key", how="left")
    missing_mask = merged_manual_df["review_row_key"].isna()
    if bool(missing_mask.any()):
        missing_df = merged_manual_df.loc[
            missing_mask,
            ["part", "local_selection", "review_label", "concat_begin_s", "concat_end_s", "selection_table_path"],
        ]
        raise ValueError(
            "Unable to map manual split-table labels back to canonical detections:\n"
            + missing_df.to_csv(sep="\t", index=False)
        )
    merged_manual_df["manual_label_category"] = np.where(
        merged_manual_df["review_label"].map(_normalized_label) == _normalized_label(human_label_value),
        "confirmed_human_contamination",
        "other_manual_label",
    )
    merged_manual_df["mapping_status"] = "mapped"
    merged_manual_df = merged_manual_df.rename(columns={"selection": "mapped_selection"})
    human_review_row_keys = set(
        merged_manual_df.loc[
            merged_manual_df["manual_label_category"] == "confirmed_human_contamination", "review_row_key"
        ].astype(str)
    )
    assessment_df = assessment_df.copy()
    assessment_df["is_manual_human_contamination"] = assessment_df["review_row_key"].astype(str).isin(
        human_review_row_keys
    )
    return assessment_df, merged_manual_df


def _attach_feature_neighbor_evidence(
    assessment_df: pd.DataFrame,
    *,
    feature_neighbor_quantile: float,
) -> pd.DataFrame:
    assessment_df = assessment_df.copy()
    if not 0.0 < feature_neighbor_quantile <= 1.0:
        raise ValueError(f"feature_neighbor_quantile must be in (0, 1], got {feature_neighbor_quantile}")
    seed_mask = assessment_df["is_manual_human_contamination"].to_numpy(dtype=bool)
    if not seed_mask.any():
        assessment_df["feature_distance_to_manual_human_seed"] = np.nan
        assessment_df["feature_neighbor_distance_threshold"] = np.nan
        assessment_df["feature_distance_rank"] = pd.Series(index=assessment_df.index, dtype="Int64")
        assessment_df["feature_distance_percentile"] = np.nan
        assessment_df["is_feature_neighbor_candidate"] = False
        return assessment_df
    feature_df = assessment_df.loc[:, FEATURE_COLUMNS].astype(float)
    feature_df = feature_df.fillna(feature_df.median())
    scaled = StandardScaler().fit_transform(feature_df)
    seed_vectors = scaled[seed_mask]
    distances = pairwise_distances(scaled, seed_vectors).min(axis=1)
    threshold = float(pd.Series(distances).quantile(feature_neighbor_quantile))
    assessment_df["feature_distance_to_manual_human_seed"] = distances
    assessment_df["feature_neighbor_distance_threshold"] = threshold
    assessment_df["feature_distance_rank"] = (
        pd.Series(distances, index=assessment_df.index)
        .rank(method="first", ascending=True)
        .astype("Int64")
    )
    assessment_df["feature_distance_percentile"] = (
        pd.Series(distances, index=assessment_df.index).rank(method="max", pct=True, ascending=True)
    )
    assessment_df["is_feature_neighbor_candidate"] = assessment_df[
        "feature_distance_to_manual_human_seed"
    ] <= threshold
    return assessment_df


def _build_source_summary(assessment_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        assessment_df.groupby(
            ["source_audio_file", "wav_basename", "audio_collection_name", "device_id"], dropna=False, sort=False
        )
        .agg(
            detection_count=("selection", "size"),
            manual_human_count=("is_manual_human_contamination", "sum"),
            feature_neighbor_count=("is_feature_neighbor_candidate", "sum"),
            boundary_rank_candidate_count=("is_boundary_rank_candidate", "sum"),
            boundary_time_candidate_count=("is_boundary_time_candidate", "sum"),
            conservative_core_keep_count=("conservative_core_keep", "sum"),
            working_core_keep_count=("working_core_keep", "sum"),
            first_source_start_s=("source_start_s", "min"),
            last_source_end_s=("source_end_s", "max"),
            source_duration_s=("source_duration_s", "first"),
        )
        .reset_index()
        .sort_values(["manual_human_count", "feature_neighbor_count", "detection_count", "wav_basename"], ascending=[False, False, False, True], kind="stable")
    )
    return summary_df


def _review_queue_table(assessment_df: pd.DataFrame) -> pd.DataFrame:
    queue_df = assessment_df.loc[
        assessment_df["review_queue_include"]
    ].copy()
    if queue_df.empty:
        return queue_df
    queue_df = queue_df.sort_values(
        [
            "same_device_as_manual_human",
            "same_collection_as_manual_human",
            "boundary_rank",
            "feature_distance_to_manual_human_seed",
            "selection",
        ],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    queue_df["review_priority_rank"] = np.arange(1, len(queue_df.index) + 1, dtype=int)
    return queue_df


def _selection_table_for_review_queue(review_queue_df: pd.DataFrame) -> pd.DataFrame:
    if review_queue_df.empty:
        return pd.DataFrame(
            columns=[
                "Selection",
                "View",
                "Channel",
                "Begin Time (s)",
                "End Time (s)",
                "Low Freq (Hz)",
                "High Freq (Hz)",
                "review_label",
                "primary_quarantine_reason",
                "quarantine_reason_tags",
                "feature_distance_to_manual_human_seed",
                "source_audio_file",
                "source_start_s",
                "source_detection_rank",
                "device_id",
                "audio_collection_name",
            ]
        )
    return (
        review_queue_df.rename(
            columns={
                "selection": "Selection",
                "concat_begin_s": "Begin Time (s)",
                "concat_end_s": "End Time (s)",
            }
        )
        .loc[
            :,
            [
                "Selection",
                "View",
                "Channel",
                "Begin Time (s)",
                "End Time (s)",
                "Low Freq (Hz)",
                "High Freq (Hz)",
                "review_label",
                "primary_quarantine_reason",
                "quarantine_reason_tags",
                "feature_distance_to_manual_human_seed",
                "source_audio_file",
                "source_start_s",
                "source_detection_rank",
                "device_id",
                "audio_collection_name",
            ],
        ]
        .copy()
    )


def _write_outputs(
    paths: HumanContaminationPaths,
    *,
    assessment_df: pd.DataFrame,
    manual_labels_df: pd.DataFrame,
    source_summary_df: pd.DataFrame,
    review_queue_df: pd.DataFrame,
) -> dict[str, tuple[Path, pd.DataFrame]]:
    outputs: dict[str, tuple[Path, pd.DataFrame]] = {}
    tables: list[tuple[str, pd.DataFrame]] = [
        ("manual_label_observations", manual_labels_df),
        ("detection_contamination_assessment", assessment_df),
        ("source_contamination_summary", source_summary_df),
        ("clean_core_conservative", assessment_df.loc[assessment_df["conservative_core_keep"]].copy()),
        ("clean_core_working", assessment_df.loc[assessment_df["working_core_keep"]].copy()),
        ("quarantine_detections", assessment_df.loc[~assessment_df["working_core_keep"]].copy()),
        ("review_priority_candidates", review_queue_df),
        ("review_priority_selection_table", _selection_table_for_review_queue(review_queue_df)),
    ]
    for logical_name, df in tables:
        csv_path = paths.tables_dir / f"{logical_name}.csv"
        parquet_path = paths.tables_dir / f"{logical_name}.parquet"
        df.to_csv(csv_path, index=False)
        outputs[f"{logical_name}_csv"] = (csv_path, df)
        if logical_name != "review_priority_selection_table":
            df.to_parquet(parquet_path, index=False)
            outputs[f"{logical_name}_parquet"] = (parquet_path, df)
    return outputs


def run_human_contamination_cleaning(
    review_bundle: Path,
    config: HumanContaminationCleaningConfig,
    *,
    output_root: Path | None = None,
    backfill_dir: Path | None = None,
    analysis_dir: Path | None = None,
    split_review_dir: Path | None = None,
) -> Path:
    bundle_dir = review_bundle.resolve()
    research_root = (
        output_root.resolve() if output_root is not None else DEFAULT_HUMAN_CONTAMINATION_RESEARCH_ROOT.resolve()
    )
    backfill_dir = (
        backfill_dir.resolve()
        if backfill_dir is not None
        else metadata_backfill_dir_for_bundle(bundle_dir, DEFAULT_FULL_INFERENCE_RUNS_ARTIFACTS_ROOT.resolve())
    )
    analysis_dir = (
        analysis_dir.resolve()
        if analysis_dir is not None
        else acoustic_analysis_dir_for_bundle(bundle_dir, DEFAULT_FULL_INFERENCE_RUNS_ARTIFACTS_ROOT.resolve())
    )
    split_review_dir = (
        split_review_dir.resolve()
        if split_review_dir is not None
        else _default_split_review_dir(bundle_dir)
    )
    paths = HumanContaminationPaths.for_bundle(bundle_dir, research_root=research_root)
    paths.ensure_dirs()

    assessment_df = _build_assessment_base(backfill_dir, analysis_dir)
    manual_labels_df = _load_manual_label_observations(split_review_dir)
    assessment_df, manual_labels_df = _attach_manual_labels(
        assessment_df,
        manual_labels_df,
        human_label_value=config.human_label_value,
    )
    assessment_df = _attach_feature_neighbor_evidence(
        assessment_df,
        feature_neighbor_quantile=config.feature_neighbor_quantile,
    )

    contaminated_sources = set(
        assessment_df.loc[assessment_df["is_manual_human_contamination"], "source_audio_file"].astype(str)
    )
    contaminated_devices = set(
        assessment_df.loc[assessment_df["is_manual_human_contamination"], "device_id"].dropna().astype(str)
    )
    contaminated_collections = set(
        assessment_df.loc[assessment_df["is_manual_human_contamination"], "audio_collection_name"].dropna().astype(str)
    )

    assessment_df["same_source_as_manual_human"] = assessment_df["source_audio_file"].astype(str).isin(
        contaminated_sources
    )
    assessment_df["same_device_as_manual_human"] = assessment_df["device_id"].fillna("").astype(str).isin(
        contaminated_devices
    )
    assessment_df["same_collection_as_manual_human"] = assessment_df[
        "audio_collection_name"
    ].fillna("").astype(str).isin(contaminated_collections)
    assessment_df["is_boundary_rank_candidate"] = assessment_df["boundary_rank"] <= int(
        config.boundary_rank_review
    )
    assessment_df["is_boundary_time_candidate"] = assessment_df["boundary_distance_minutes"] <= float(
        config.boundary_minutes_review
    )
    assessment_df["conservative_core_keep"] = ~assessment_df["same_source_as_manual_human"]
    assessment_df["working_core_keep"] = assessment_df["conservative_core_keep"] & ~assessment_df[
        "is_feature_neighbor_candidate"
    ]
    assessment_df["primary_quarantine_reason"] = "retained"
    assessment_df.loc[
        assessment_df["is_boundary_rank_candidate"] | assessment_df["is_boundary_time_candidate"],
        "primary_quarantine_reason",
    ] = "boundary_priority_only"
    assessment_df.loc[
        assessment_df["is_feature_neighbor_candidate"], "primary_quarantine_reason"
    ] = "feature_neighbor_quarantine"
    assessment_df.loc[
        assessment_df["same_source_as_manual_human"], "primary_quarantine_reason"
    ] = "source_quarantine"
    assessment_df.loc[
        assessment_df["is_manual_human_contamination"], "primary_quarantine_reason"
    ] = "confirmed_human_contamination"
    assessment_df["review_queue_include"] = assessment_df["conservative_core_keep"] & assessment_df[
        "is_feature_neighbor_candidate"
    ]
    assessment_df["quarantine_reason_tags"] = assessment_df.apply(_quarantine_reason_tags, axis=1)

    review_queue_df = _review_queue_table(assessment_df)
    source_summary_df = _build_source_summary(assessment_df)
    output_tables = _write_outputs(
        paths,
        assessment_df=assessment_df,
        manual_labels_df=manual_labels_df,
        source_summary_df=source_summary_df,
        review_queue_df=review_queue_df,
    )
    audio_export_summaries = []
    if config.write_quarantine_audio_exports:
        _reset_audio_exports_dir(paths.audio_exports_dir)
        audio_export_summaries = [
            {
                "subset_name": summary.subset_name,
                "description": summary.description,
                "row_count": summary.row_count,
                "duration_s": summary.duration_s,
                "gap_duration_s": summary.gap_duration_s,
                "wav_path": str(summary.wav_path.resolve()),
                "selection_table_path": str(summary.selection_table_path.resolve()),
                "manifest_path": str(summary.manifest_path.resolve()),
            }
            for summary in write_quarantine_audio_exports(
                bundle_dir,
                paths.audio_exports_dir,
                assessment_df,
                gap_duration_s=float(config.audio_export_gap_s),
            )
        ]

    report_path = paths.reports_dir / "index.md"
    write_index_report(
        report_path,
        bundle_dir=bundle_dir,
        backfill_dir=backfill_dir,
        analysis_dir=analysis_dir,
        split_review_dir=split_review_dir,
        config_summary=asdict(config),
        assessment_df=assessment_df,
        manual_labels_df=manual_labels_df,
        source_summary_df=source_summary_df,
        review_queue_df=review_queue_df,
        audio_export_summaries=audio_export_summaries,
    )

    input_files = []
    candidate_input_paths = [
        bundle_dir / "review_import_manifest.json",
        bundle_dir / "review_selection_table.txt",
        bundle_dir / "review_selection_table_metadata.tsv",
        backfill_dir / "enriched_detections.parquet",
        analysis_dir / "tables" / "analysis_rows.parquet",
        analysis_dir / "tables" / "acoustic_features.parquet",
    ]
    if split_review_dir is not None:
        candidate_input_paths.append(split_review_dir / "split_manifest.json")
        for table_path in sorted(split_review_dir.glob("*selection_table.txt")):
            candidate_input_paths.append(table_path)
    for input_path in candidate_input_paths:
        if input_path.is_file():
            input_files.append(register_file(input_path, logical_name=input_path.name, category="input"))

    output_files = []
    for logical_name, (path, df) in output_tables.items():
        category = "table"
        if path.suffix == ".parquet":
            output_files.append(register_table(path, logical_name, category, df))
        else:
            output_files.append(register_table(path, logical_name, category, df))
    for summary in audio_export_summaries:
        output_files.append(
            register_file(
                Path(summary["wav_path"]),
                logical_name=f'{summary["subset_name"]}_review_concat_wav',
                category="audio_export",
            )
        )
        selection_table_df = pd.read_csv(Path(summary["selection_table_path"]), sep="\t")
        output_files.append(
            register_table(
                Path(summary["selection_table_path"]),
                logical_name=f'{summary["subset_name"]}_review_selection_table',
                category="audio_export_table",
                df=selection_table_df,
            )
        )
        output_files.append(
            register_file(
                Path(summary["manifest_path"]),
                logical_name=f'{summary["subset_name"]}_manifest',
                category="audio_export_manifest",
            )
        )
    output_files.append(register_file(report_path, logical_name="index_report", category="report"))

    manifest_path = paths.root / "manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_name": "human_contamination_cleaning",
        "bundle_dir": str(bundle_dir),
        "backfill_dir": str(backfill_dir),
        "analysis_dir": str(analysis_dir),
        "split_review_dir": str(split_review_dir) if split_review_dir is not None else None,
        "config": asdict(config),
        "counts": {
            "detection_rows": int(len(assessment_df.index)),
            "source_wav_rows": int(assessment_df["source_audio_file"].nunique()),
            "manual_label_rows": int(len(manual_labels_df.index)),
            "manual_human_rows": int(assessment_df["is_manual_human_contamination"].sum()),
            "confirmed_contaminated_sources": int(len(contaminated_sources)),
            "feature_neighbor_candidate_rows": int(assessment_df["is_feature_neighbor_candidate"].sum()),
            "review_queue_rows": int(len(review_queue_df.index)),
            "conservative_core_rows": int(assessment_df["conservative_core_keep"].sum()),
            "working_core_rows": int(assessment_df["working_core_keep"].sum()),
            "audio_export_sets": int(len(audio_export_summaries)),
        },
        "manual_human_seed_selections": assessment_df.loc[
            assessment_df["is_manual_human_contamination"], "selection"
        ].astype(int).tolist(),
        "audio_exports": [{key: json_ready(value) for key, value in summary.items()} for summary in audio_export_summaries],
        "software_versions": package_versions(),
        "inputs": input_files,
        "outputs": output_files,
    }
    write_json(manifest_path, manifest)
    return paths.root

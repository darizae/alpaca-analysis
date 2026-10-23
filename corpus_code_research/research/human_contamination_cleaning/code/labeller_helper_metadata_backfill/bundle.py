from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PREDICTION_REVIEW_PACKAGE_FORMAT = "prediction_review_raven_bundle_v2"
REQUIRED_BUNDLE_FILENAMES = (
    "review_import_manifest.json",
    "review_selection_table.txt",
    "review_selection_table_metadata.tsv",
)
REQUIRED_METADATA_COLUMNS = (
    "source_audio_file",
    "source_start_s",
    "source_end_s",
    "session_id",
    "session_item_id",
    "item_index",
    "detection_index",
    "detection_score",
)
OPTIONAL_METADATA_COLUMNS = (
    "canonical_detection_id",
    "canonical_state_version",
)


@dataclass(frozen=True)
class BundleInputs:
    bundle_dir: Path
    manifest_path: Path
    selection_table_path: Path
    metadata_table_path: Path
    manifest: dict
    prediction_run_id: str
    session_id: str


@dataclass(frozen=True)
class BundleTables:
    selection_df: pd.DataFrame
    metadata_df: pd.DataFrame
    merged_df: pd.DataFrame
    join_mode: str


class BundleValidationError(ValueError):
    """Raised when a review bundle fails deterministic validation."""



def resolve_bundle_inputs(
    review_bundle: Path | None = None,
    review_import_manifest: Path | None = None,
    metadata_table: Path | None = None,
) -> BundleInputs:
    specified = [value is not None for value in (review_bundle, review_import_manifest, metadata_table)]
    if sum(specified) != 1:
        raise BundleValidationError("Specify exactly one of --review-bundle, --review-import-manifest, or --metadata-table.")

    if review_bundle is not None:
        bundle_dir = review_bundle.resolve()
    elif review_import_manifest is not None:
        bundle_dir = review_import_manifest.resolve().parent
    else:
        bundle_dir = metadata_table.resolve().parent  # type: ignore[union-attr]

    manifest_path = bundle_dir / "review_import_manifest.json"
    selection_table_path = bundle_dir / "review_selection_table.txt"
    metadata_table_path = bundle_dir / "review_selection_table_metadata.tsv"
    for file_path in (manifest_path, selection_table_path, metadata_table_path):
        if not file_path.is_file():
            raise BundleValidationError(f"Missing required bundle file: {file_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    package_format = str(manifest.get("package_format", "")).strip()
    if package_format != PREDICTION_REVIEW_PACKAGE_FORMAT:
        raise BundleValidationError(f"Unsupported package format: {package_format!r}")

    prediction_run_id = str(manifest.get("prediction_run_id", "")).strip()
    session_id = str(manifest.get("session_id", "")).strip()
    return BundleInputs(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        selection_table_path=selection_table_path,
        metadata_table_path=metadata_table_path,
        manifest=manifest,
        prediction_run_id=prediction_run_id,
        session_id=session_id,
    )



def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def validate_manifest_integrity(inputs: BundleInputs) -> None:
    manifest = inputs.manifest
    if "editable_table_sha256" in manifest:
        actual = _sha256_of_file(inputs.selection_table_path)
        if actual != manifest["editable_table_sha256"]:
            raise BundleValidationError(
                f"Selection table SHA-256 mismatch: expected {manifest['editable_table_sha256']}, got {actual}"
            )
    if "metadata_table_sha256" in manifest:
        actual = _sha256_of_file(inputs.metadata_table_path)
        if actual != manifest["metadata_table_sha256"]:
            raise BundleValidationError(
                f"Metadata table SHA-256 mismatch: expected {manifest['metadata_table_sha256']}, got {actual}"
            )



def read_bundle_tables(inputs: BundleInputs) -> BundleTables:
    validate_manifest_integrity(inputs)
    selection_df = pd.read_csv(inputs.selection_table_path, sep="\t")
    metadata_df = pd.read_csv(inputs.metadata_table_path, sep="\t")

    if len(selection_df.index) != len(metadata_df.index):
        raise BundleValidationError(
            f"Selection row count {len(selection_df.index)} does not match metadata row count {len(metadata_df.index)}."
        )
    expected_row_count = inputs.manifest.get("row_count")
    if expected_row_count is not None and int(expected_row_count) != len(metadata_df.index):
        raise BundleValidationError(
            f"Manifest row_count {expected_row_count} does not match metadata rows {len(metadata_df.index)}."
        )

    missing_metadata_columns = [column for column in REQUIRED_METADATA_COLUMNS if column not in metadata_df.columns]
    if missing_metadata_columns:
        raise BundleValidationError(
            f"Metadata table missing required columns: {', '.join(missing_metadata_columns)}"
        )

    selection_df = selection_df.copy()
    metadata_df = metadata_df.copy()
    selection_df["Selection"] = selection_df["Selection"].astype(str)
    metadata_df["Selection"] = metadata_df["Selection"].astype(str)

    if metadata_df["Selection"].is_unique and selection_df["Selection"].is_unique:
        merged_df = metadata_df.merge(selection_df, on="Selection", how="left", validate="one_to_one")
        join_mode = "selection"
    else:
        metadata_with_order = metadata_df.copy()
        selection_with_order = selection_df.copy()
        metadata_with_order["_row_order"] = range(len(metadata_with_order.index))
        selection_with_order["_row_order"] = range(len(selection_with_order.index))
        merged_df = metadata_with_order.merge(selection_with_order, on="_row_order", how="left", validate="one_to_one")
        merged_df = merged_df.drop(columns=["_row_order"])
        join_mode = "row_order"

    if "review_label" not in merged_df.columns:
        merged_df["review_label"] = ""
    merged_df["review_label"] = merged_df["review_label"].fillna("")

    return BundleTables(
        selection_df=selection_df,
        metadata_df=metadata_df,
        merged_df=merged_df,
        join_mode=join_mode,
    )



def choose_output_slug(inputs: BundleInputs) -> str:
    if inputs.prediction_run_id and inputs.session_id:
        return f"{inputs.prediction_run_id}_{inputs.session_id}"
    digest = hashlib.sha256(inputs.metadata_table_path.read_bytes()).hexdigest()[:24]
    return digest

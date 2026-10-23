#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from labeller_helper_human_contamination.pipeline import (
    HumanContaminationCleaningConfig,
    run_human_contamination_cleaning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic human-contamination cleaning artifact from a review bundle, "
            "metadata backfill outputs, acoustic features, and optional split-table manual labels."
        )
    )
    parser.add_argument(
        "--review-bundle",
        required=True,
        type=Path,
        help="Path to a prediction-review Raven bundle directory.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        type=Path,
        help=(
            "Optional override for the run-artifact root. "
            "Defaults to artifacts/full_inference_runs/<bundle-name>/human_contamination_cleaning/."
        ),
    )
    parser.add_argument(
        "--backfill-dir",
        default=None,
        type=Path,
        help="Optional override for the metadata backfill directory.",
    )
    parser.add_argument(
        "--analysis-dir",
        default=None,
        type=Path,
        help="Optional override for the acoustic analysis directory.",
    )
    parser.add_argument(
        "--split-review-dir",
        default=None,
        type=Path,
        help=(
            "Optional directory containing split review artifacts and split_manifest.json. "
            "Defaults to <bundle>/review_concat_split_5 when present."
        ),
    )
    parser.add_argument(
        "--human-label-value",
        default="0",
        help="Manual review label value interpreted as confirmed human contamination.",
    )
    parser.add_argument(
        "--feature-neighbor-quantile",
        type=float,
        default=0.01,
        help=(
            "Corpus quantile used to define the feature-neighbor quarantine threshold from "
            "distance-to-seed scores. Smaller values produce a tighter review queue."
        ),
    )
    parser.add_argument(
        "--boundary-rank-review",
        type=int,
        default=7,
        help=(
            "Detections within the first/last N detections of a source WAV are marked as "
            "boundary-priority review candidates."
        ),
    )
    parser.add_argument(
        "--boundary-minutes-review",
        type=float,
        default=20.0,
        help=(
            "Detections within the first/last N minutes of a source WAV are marked as "
            "boundary-priority review candidates."
        ),
    )
    parser.add_argument(
        "--write-quarantine-audio-exports",
        action="store_true",
        help=(
            "Write concatenated review WAVs plus Raven selection tables for quarantine/exclusion subsets "
            "under audio_exports/."
        ),
    )
    parser.add_argument(
        "--audio-export-gap-s",
        type=float,
        default=0.1,
        help="Fixed silence gap inserted between snippets in exported quarantine-review WAVs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = run_human_contamination_cleaning(
        review_bundle=args.review_bundle,
        config=HumanContaminationCleaningConfig(
            human_label_value=args.human_label_value,
            feature_neighbor_quantile=args.feature_neighbor_quantile,
            boundary_rank_review=args.boundary_rank_review,
            boundary_minutes_review=args.boundary_minutes_review,
            write_quarantine_audio_exports=args.write_quarantine_audio_exports,
            audio_export_gap_s=args.audio_export_gap_s,
        ),
        output_root=args.output_root,
        backfill_dir=args.backfill_dir,
        analysis_dir=args.analysis_dir,
        split_review_dir=args.split_review_dir,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

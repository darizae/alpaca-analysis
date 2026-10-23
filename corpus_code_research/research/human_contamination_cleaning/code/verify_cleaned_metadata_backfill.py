#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from labeller_helper_metadata_backfill.audit import write_cleaned_audit_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the cleaned metadata-backfill artifact for the hum-only corpus and write a concise GPS/temporal audit."
        )
    )
    parser.add_argument(
        "--backfill-dir",
        required=True,
        type=Path,
        help="Path to a metadata_backfill artifact directory containing cleaned_enriched_detections.parquet.",
    )
    parser.add_argument(
        "--cleaned-export-dir",
        default=None,
        type=Path,
        help=(
            "Optional path to the cleaned_alpaca_hum_corpus export directory. "
            "If omitted, the report skips the cleaned-export row-count cross-check."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Optional output markdown path. Defaults to <backfill-dir>/join_audit.md.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = write_cleaned_audit_report(
        args.backfill_dir,
        output_path=args.output,
        cleaned_export_dir=args.cleaned_export_dir,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from labeller_helper_metadata_backfill.enrich import run_backfill
from labeller_helper_metadata_backfill.paths import CANONICAL_HPC_RAW_ROOT



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill deterministic recorder metadata for an inference review bundle."
    )
    parser.add_argument("--review-bundle", type=Path)
    parser.add_argument("--review-import-manifest", type=Path)
    parser.add_argument("--metadata-table", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional override for deterministic backfill outputs. Defaults to the bundle's run-local artifact folder.",
    )
    parser.add_argument(
        "--ssh-host",
        default="hpc-cluster",
        help="SSH host used for remote metadata lookup and pulling.",
    )
    parser.add_argument(
        "--hpc-raw-root",
        default=str(CANONICAL_HPC_RAW_ROOT),
        help="Canonical HPC raw audio collection root.",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    output_dir = run_backfill(
        review_bundle=args.review_bundle,
        review_import_manifest=args.review_import_manifest,
        metadata_table=args.metadata_table,
        output_root=args.output_root,
        ssh_host=args.ssh_host,
        canonical_hpc_raw_root=args.hpc_raw_root,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

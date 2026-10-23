#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from labeller_helper_run_paths import (
    DEFAULT_FULL_INFERENCE_RUNS_ARTIFACTS_ROOT,
    LEGACY_HUMAN_CONTAMINATION_RESEARCH_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate bundle-level human-contamination cleaning outputs so there is one "
            "canonical instance under artifacts/full_inference_runs/<bundle>/human_contamination_cleaning, "
            "with research paths rewired to symlinks."
        )
    )
    parser.add_argument(
        "--bundle-name",
        required=True,
        help="Bundle directory name (for example RF_run_on_FULL_Run_1_prediction-review-raven_bundle-...).",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_FULL_INFERENCE_RUNS_ARTIFACTS_ROOT,
        help="Root containing full-inference run artifacts. Default: artifacts/full_inference_runs",
    )
    parser.add_argument(
        "--research-root",
        type=Path,
        default=LEGACY_HUMAN_CONTAMINATION_RESEARCH_ROOT,
        help="Legacy research root that may contain duplicate outputs. Default: research/human_contamination_cleaning",
    )
    return parser.parse_args()


def _replace_with_symlink(link_path: Path, target_path: Path) -> tuple[str, str | None]:
    if link_path.is_symlink():
        current_target = link_path.resolve()
        if current_target == target_path.resolve():
            return "already_symlinked", None
        link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir() and not any(link_path.iterdir()):
            link_path.rmdir()
        else:
            backup_path = link_path.with_name(f"{link_path.name}__pre_canonical_consolidation_backup")
            if backup_path.exists():
                raise RuntimeError(
                    f"Backup path already exists: {backup_path}. "
                    "Move or remove the backup, then rerun consolidation."
                )
            link_path.rename(backup_path)
            link_path.parent.mkdir(parents=True, exist_ok=True)
            relative_target = os.path.relpath(target_path, start=link_path.parent)
            link_path.symlink_to(relative_target)
            return "created_symlink_with_backup", str(backup_path)

    link_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target_path, start=link_path.parent)
    link_path.symlink_to(relative_target)
    return "created_symlink", None


def consolidate_bundle(bundle_name: str, artifacts_root: Path, research_root: Path) -> dict[str, object]:
    artifacts_root = artifacts_root.resolve()
    research_root = research_root.resolve()

    canonical_bundle_root = artifacts_root / bundle_name
    canonical_cleaning_root = canonical_bundle_root / "human_contamination_cleaning"
    research_bundle_root = research_root / bundle_name
    research_cleaning_root = research_bundle_root / "human_contamination_cleaning"
    staged_cleaning_root = research_bundle_root / "logistic_core_results" / "03_full_corpus_logistic_cleaning"

    if not canonical_bundle_root.is_dir():
        raise RuntimeError(f"Canonical bundle root does not exist: {canonical_bundle_root}")
    if not research_bundle_root.is_dir():
        raise RuntimeError(f"Research bundle root does not exist: {research_bundle_root}")

    moved_from: str | None = None
    if not canonical_cleaning_root.exists():
        if staged_cleaning_root.is_dir() and not staged_cleaning_root.is_symlink():
            staged_cleaning_root.rename(canonical_cleaning_root)
            moved_from = str(staged_cleaning_root)
        elif research_cleaning_root.is_dir() and not research_cleaning_root.is_symlink():
            research_cleaning_root.rename(canonical_cleaning_root)
            moved_from = str(research_cleaning_root)
        else:
            raise RuntimeError(
                "No source directory found to populate canonical cleaning root. "
                "Expected either logistic_core_results/03_full_corpus_logistic_cleaning or "
                "research/<bundle>/human_contamination_cleaning."
            )

    staged_link_status, staged_backup = _replace_with_symlink(staged_cleaning_root, canonical_cleaning_root)
    research_link_status, research_backup = _replace_with_symlink(research_cleaning_root, canonical_cleaning_root)

    return {
        "bundle_name": bundle_name,
        "canonical_bundle_root": str(canonical_bundle_root),
        "canonical_cleaning_root": str(canonical_cleaning_root),
        "moved_from": moved_from,
        "staged_cleaning_link": {
            "path": str(staged_cleaning_root),
            "status": staged_link_status,
            "backup_path": staged_backup,
            "target": str(staged_cleaning_root.resolve()),
        },
        "research_cleaning_link": {
            "path": str(research_cleaning_root),
            "status": research_link_status,
            "backup_path": research_backup,
            "target": str(research_cleaning_root.resolve()),
        },
    }


def main() -> int:
    args = parse_args()
    summary = consolidate_bundle(
        bundle_name=args.bundle_name,
        artifacts_root=args.artifacts_root,
        research_root=args.research_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from stv_analysis.rq1_post_hum_cohesion.whole_subnetwork_threshold import (
    GLOBAL_OUTLIER_THRESHOLD_M,
    materialize_whole_subnetwork_threshold,
)


DEFAULT_BASE_DIR = Path("backend/data/processed/analysis/base")
DEFAULT_PRIMARY_INPUT = Path(
    "backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/data/"
    "h1_contact_reinforcement_model_input.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/threshold_supplement"
)
DEFAULT_CALIBRATION_SOURCE_DIR = Path(
    "backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/threshold_diagnostics"
)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    base_dir = resolve_path(repo_root, args.base_dir)
    primary_input = resolve_path(repo_root, args.primary_input)
    output_dir = resolve_path(repo_root, args.output_dir)
    calibration_source_dir = resolve_path(repo_root, args.calibration_source_dir)
    sessions = tuple(value.strip() for value in args.sessions.split(",") if value.strip()) if args.sessions else None

    if not primary_input.is_file():
        raise FileNotFoundError(f"missing frozen primary H1 model input: {primary_input}")

    base = load_base_tables(base_dir)
    primary_rows = read_primary_input(primary_input)
    manifest = materialize_whole_subnetwork_threshold(
        repo_root=repo_root,
        primary_rows=primary_rows,
        trace_points=base["trace_points_filtered"],
        output_dir=output_dir,
        threshold_m=float(args.threshold_m),
        selected_sessions=sessions,
        hard_gps_filter_version=str(base["gps_blocker_rule"].get("name", "analysis_base_filtered_trace_points")),
        calibration_source_dir=calibration_source_dir,
    )
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize RQ1 whole-subnetwork threshold supplementary LMM views.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="Analysis-base directory.")
    parser.add_argument("--primary-input", default=str(DEFAULT_PRIMARY_INPUT), help="Frozen primary H1 model input.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Threshold supplementary output directory.")
    parser.add_argument("--calibration-source-dir", default=str(DEFAULT_CALIBRATION_SOURCE_DIR), help="Pre-fit threshold diagnostics to copy into the materialized output.")
    parser.add_argument("--threshold-m", default=str(GLOBAL_OUTLIER_THRESHOLD_M), help="Frozen global outlier threshold in metres.")
    parser.add_argument("--sessions", default="", help="Optional comma-separated day_session_id subset for smoke/debug materialization.")
    return parser.parse_args()


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def read_primary_input(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_base_tables(base_dir: Path) -> dict[str, Any]:
    required = {
        "trace_points_filtered": base_dir / "trace_points_filtered.parquet",
        "gps_blocker_rule": base_dir / "gps_blocker_rule.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing analysis-base inputs for threshold materialization: {missing}")
    return {
        "trace_points_filtered": pd.read_parquet(required["trace_points_filtered"]),
        "gps_blocker_rule": json.loads(required["gps_blocker_rule"].read_text(encoding="utf8")),
    }


if __name__ == "__main__":
    main()

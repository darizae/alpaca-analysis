from __future__ import annotations

import argparse
from pathlib import Path

from .config import DedupConfig, load_config
from .features import finalize_features, launch_paths_for_config, prepare_features, run_features, seed_feature_artifacts
from .grid import export_top_exemplars, prepare_grid, run_grid, seed_model_artifacts, summarize_run
from .hpc import format_progress, prepare_hpc, show_hpc_status, submit_hpc


def _load_config_for_args(args: argparse.Namespace) -> DedupConfig:
    if getattr(args, "config", None):
        return load_config(Path(args.config))
    launch_dir = Path(args.launch_dir).resolve()
    config_path = launch_dir / "configs" / "launch_config.yaml"
    return load_config(config_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Corpus hum deduplication workflow CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_features_parser = subparsers.add_parser("prepare-features")
    prepare_features_parser.add_argument("--config", required=True, type=Path)

    run_features_parser = subparsers.add_parser("run-features")
    run_features_parser.add_argument("--config", type=Path)
    run_features_parser.add_argument("--launch-dir", type=Path)
    run_features_parser.add_argument("--shard-id", type=int, action="append")

    finalize_features_parser = subparsers.add_parser("finalize-features")
    finalize_features_parser.add_argument("--config", type=Path)
    finalize_features_parser.add_argument("--launch-dir", type=Path)

    seed_features_parser = subparsers.add_parser("seed-feature-artifacts")
    seed_features_parser.add_argument("--config", type=Path)
    seed_features_parser.add_argument("--launch-dir", type=Path)
    seed_features_parser.add_argument("--source-launch-dir", required=True, type=Path)
    seed_features_parser.add_argument("--force", action="store_true")

    seed_model_parser = subparsers.add_parser("seed-model-artifacts")
    seed_model_parser.add_argument("--config", type=Path)
    seed_model_parser.add_argument("--launch-dir", type=Path)
    seed_model_parser.add_argument("--source-launch-dir", required=True, type=Path)
    seed_model_parser.add_argument("--force", action="store_true")

    prepare_grid_parser = subparsers.add_parser("prepare-grid")
    prepare_grid_parser.add_argument("--config", type=Path)
    prepare_grid_parser.add_argument("--launch-dir", type=Path)

    run_grid_parser = subparsers.add_parser("run-grid")
    run_grid_parser.add_argument("--config", type=Path)
    run_grid_parser.add_argument("--launch-dir", type=Path)
    run_grid_parser.add_argument("--batch-id", action="append")

    prepare_hpc_parser = subparsers.add_parser("prepare-hpc")
    prepare_hpc_parser.add_argument("--config", type=Path)
    prepare_hpc_parser.add_argument("--launch-dir", type=Path)
    prepare_hpc_parser.add_argument(
        "--stage",
        required=True,
        choices=["features", "feature_merge", "grid", "aggregate", "exemplars"],
    )

    submit_hpc_parser = subparsers.add_parser("submit-hpc")
    submit_hpc_parser.add_argument("--config", type=Path)
    submit_hpc_parser.add_argument("--launch-dir", type=Path)
    submit_hpc_parser.add_argument(
        "--stage",
        required=True,
        choices=["features", "feature_merge", "grid", "aggregate", "exemplars"],
    )
    submit_hpc_parser.add_argument("--dependency")
    submit_hpc_parser.add_argument("--dry-run", action="store_true")

    show_progress_parser = subparsers.add_parser("show-progress")
    show_progress_parser.add_argument("--config", type=Path)
    show_progress_parser.add_argument("--launch-dir", type=Path)
    show_progress_parser.add_argument(
        "--stage",
        choices=["features", "feature_merge", "grid", "aggregate", "exemplars"],
    )
    show_progress_parser.add_argument("--detail", default="summary", choices=["summary", "stage", "task", "json"])

    show_hpc_parser = subparsers.add_parser("show-hpc-status")
    show_hpc_parser.add_argument("--config", type=Path)
    show_hpc_parser.add_argument("--launch-dir", type=Path)
    show_hpc_parser.add_argument("--stage", choices=["features", "feature_merge", "grid", "aggregate", "exemplars"])
    show_hpc_parser.add_argument("--job-id")
    show_hpc_parser.add_argument("--remote-run-dir")

    summarize_parser = subparsers.add_parser("summarize-run")
    summarize_parser.add_argument("--config", type=Path)
    summarize_parser.add_argument("--launch-dir", type=Path)

    exemplars_parser = subparsers.add_parser("export-top-exemplars")
    exemplars_parser.add_argument("--config", type=Path)
    exemplars_parser.add_argument("--launch-dir", type=Path)
    exemplars_parser.add_argument("--top-k", type=int)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = _load_config_for_args(args)

    if args.command == "prepare-features":
        paths = prepare_features(config)
        print(paths.launch_dir)
        return 0
    if args.command == "run-features":
        run_features(config, shard_ids=args.shard_id)
        return 0
    if args.command == "finalize-features":
        finalize_features(config)
        return 0
    if args.command == "seed-feature-artifacts":
        seed_feature_artifacts(config, source_launch_dir=args.source_launch_dir, force=args.force)
        return 0
    if args.command == "seed-model-artifacts":
        seed_model_artifacts(config, source_launch_dir=args.source_launch_dir, force=args.force)
        return 0
    if args.command == "prepare-grid":
        prepare_grid(config)
        return 0
    if args.command == "run-grid":
        run_grid(config, batch_ids=args.batch_id)
        return 0
    if args.command == "prepare-hpc":
        prepare_hpc(config, stage=args.stage)
        return 0
    if args.command == "submit-hpc":
        print(submit_hpc(config, stage=args.stage, dependency=args.dependency, dry_run=args.dry_run))
        return 0
    if args.command == "show-progress":
        paths = launch_paths_for_config(config)
        print(format_progress(paths, stage=args.stage, detail=args.detail))
        return 0
    if args.command == "show-hpc-status":
        print(show_hpc_status(config, stage=args.stage, job_id=args.job_id, remote_run_dir=args.remote_run_dir))
        return 0
    if args.command == "summarize-run":
        summarize_run(config)
        return 0
    if args.command == "export-top-exemplars":
        export_top_exemplars(config, top_k=args.top_k)
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

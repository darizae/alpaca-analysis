from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from stv_backend.config import load_config

from .framework import create_launch, load_launch_context, prepare_hpc, run_stage, show_status, submit_hpc
from .paths import default_backend_config_path, repo_root
from .registry import get_analysis_definition


def _default_base_manifest_path() -> Path:
    return repo_root() / "backend" / "data" / "processed" / "analysis" / "base" / "manifest.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stv-analysis", description="Run scientific analysis launches.")
    parser.add_argument("--config", type=Path, default=default_backend_config_path())
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_subparsers = launch_parser.add_subparsers(dest="launch_command", required=True)

    create_parser = launch_subparsers.add_parser("create")
    create_parser.add_argument("analysis_id")
    create_parser.add_argument("--label", required=True)
    create_parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    create_parser.add_argument("--input-manifest", type=Path, default=_default_base_manifest_path())

    run_stage_parser = launch_subparsers.add_parser("run-stage")
    run_stage_parser.add_argument("analysis_id")
    run_stage_parser.add_argument("--launch-dir", type=Path, required=True)
    run_stage_parser.add_argument("--stage", required=True)
    run_stage_parser.add_argument("--task-value", default=None)

    prepare_parser = launch_subparsers.add_parser("prepare-hpc")
    prepare_parser.add_argument("analysis_id")
    prepare_parser.add_argument("--launch-dir", type=Path, required=True)

    submit_parser = launch_subparsers.add_parser("submit-hpc")
    submit_parser.add_argument("analysis_id")
    submit_parser.add_argument("--launch-dir", type=Path, required=True)
    submit_parser.add_argument("--start-stage", default=None)

    status_parser = launch_subparsers.add_parser("status")
    status_parser.add_argument("analysis_id")
    status_parser.add_argument("--launch-dir", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command != "launch":
        parser.error(f"unknown command: {args.command}")
        return 2

    definition = get_analysis_definition(args.analysis_id)
    if args.launch_command == "create":
        result = create_launch(
            definition=definition,
            config=config,
            launch_label=args.label,
            profile=args.profile,
            input_manifest_path=args.input_manifest,
        )
    else:
        ctx = load_launch_context(
            definition=definition,
            config=config,
            launch_dir=args.launch_dir,
        )
        if args.launch_command == "run-stage":
            result = run_stage(ctx=ctx, stage_id=args.stage, task_value=args.task_value)
        elif args.launch_command == "prepare-hpc":
            result = prepare_hpc(ctx)
        elif args.launch_command == "submit-hpc":
            result = submit_hpc(ctx, start_stage=args.start_stage)
        elif args.launch_command == "status":
            result = show_status(ctx)
        else:
            parser.error(f"unknown launch command: {args.launch_command}")
            return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())


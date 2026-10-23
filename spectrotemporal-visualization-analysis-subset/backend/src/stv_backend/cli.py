from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from pathlib import PurePosixPath
import sys

import pandas as pd

from .analysis_base import build_analysis_base
from .config import BackendConfig, load_config
from .pipeline import build_dataset, verify_dataset
from .source_index import (
    build_files_to_pull,
    collect_collection_inventories,
    normalize_source_audio_path,
    pull_metadata_files,
    resolve_sources,
    write_files_to_pull,
)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default.toml"


def _load_runtime_config(config_path: Path, ssh_host_override: str | None) -> BackendConfig:
    config = load_config(config_path)
    if ssh_host_override:
        config = replace(config, ssh_host=ssh_host_override)
    return config


def _load_source_paths(config: BackendConfig) -> list[str]:
    path = config.raw_corpus_dir / "metadata_backfill" / "source_wavs.csv"
    df = pd.read_csv(path)
    if "source_audio_file_original" in df.columns:
        return sorted(df["source_audio_file_original"].dropna().astype(str).unique().tolist())
    if "source_audio_file_normalized" in df.columns:
        return sorted(df["source_audio_file_normalized"].dropna().astype(str).unique().tolist())
    raise ValueError(f"{path} must contain source_audio_file_original or source_audio_file_normalized")


def fetch_sidecars(config: BackendConfig) -> dict[str, object]:
    if not config.ssh_host:
        raise ValueError("remote ssh host is empty; set remote.ssh_host in config or pass --ssh-host")

    canonical_root = PurePosixPath(config.canonical_raw_root)
    source_audio_files = _load_source_paths(config)
    normalized_sources = [normalize_source_audio_path(source_audio_file, canonical_root) for source_audio_file in source_audio_files]
    collection_names = [source.audio_collection_name for source in normalized_sources]
    inventories = collect_collection_inventories(config.ssh_host, collection_names, canonical_root)
    resolved = resolve_sources(normalized_sources, inventories)

    unresolved = [record for record in resolved if not record.csv_found]
    if unresolved:
        preview = [record.source_audio_file_original for record in unresolved[:5]]
        raise FileNotFoundError(f"failed to resolve CSV sidecars for sources such as: {preview}")

    files_to_pull = build_files_to_pull(resolved)
    files_to_pull_path = write_files_to_pull(config.processed_dir, files_to_pull)
    commands = pull_metadata_files(config.ssh_host, files_to_pull_path, config.sidecar_cache_dir)

    summary = {
        "resolved_sources": len(resolved),
        "collections": len(set(collection_names)),
        "files_to_pull": len(files_to_pull),
        "sidecar_cache_dir": str(config.sidecar_cache_dir),
        "commands": commands,
    }
    summary_path = config.processed_dir / "fetch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stv-backend", description="Prepare dataset and scientific-analysis inputs.")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--ssh-host", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("fetch-sidecars")
    subparsers.add_parser("build-dataset")
    build_analysis_base_parser = subparsers.add_parser("build-analysis-base")
    build_analysis_base_parser.add_argument("--output-dir", type=Path, default=None)
    subparsers.add_parser("verify-dataset")
    subparsers.add_parser("all")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _load_runtime_config(args.config, args.ssh_host)

    if args.command == "fetch-sidecars":
        result = fetch_sidecars(config)
    elif args.command == "build-dataset":
        result = build_dataset(config).manifest
    elif args.command == "build-analysis-base":
        result = build_analysis_base(config, output_dir=args.output_dir)
    elif args.command == "verify-dataset":
        result = verify_dataset(config)
    elif args.command == "all":
        fetch_sidecars(config)
        build_dataset(config)
        build_analysis_base(config)
        result = verify_dataset(config)
    else:
        parser.error(f"unknown command: {args.command}")
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

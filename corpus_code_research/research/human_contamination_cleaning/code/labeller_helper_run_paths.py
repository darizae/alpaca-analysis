from __future__ import annotations

from pathlib import Path

DEFAULT_FULL_INFERENCE_RUNS_ARTIFACTS_ROOT = Path("artifacts/full_inference_runs")
LEGACY_HUMAN_CONTAMINATION_RESEARCH_ROOT = Path("research/human_contamination_cleaning")

# Canonical location for human-contamination cleaning outputs.
# Kept under the run artifact root so each bundle has one authoritative instance.
DEFAULT_HUMAN_CONTAMINATION_RESEARCH_ROOT = DEFAULT_FULL_INFERENCE_RUNS_ARTIFACTS_ROOT


def run_artifact_root_for_bundle(
    bundle_dir: Path,
    artifacts_root: Path = DEFAULT_FULL_INFERENCE_RUNS_ARTIFACTS_ROOT,
) -> Path:
    return artifacts_root.resolve() / bundle_dir.resolve().name


def metadata_backfill_dir_for_bundle(
    bundle_dir: Path,
    artifacts_root: Path = DEFAULT_FULL_INFERENCE_RUNS_ARTIFACTS_ROOT,
) -> Path:
    return run_artifact_root_for_bundle(bundle_dir, artifacts_root) / "metadata_backfill"


def acoustic_analysis_dir_for_bundle(
    bundle_dir: Path,
    artifacts_root: Path = DEFAULT_FULL_INFERENCE_RUNS_ARTIFACTS_ROOT,
) -> Path:
    return run_artifact_root_for_bundle(bundle_dir, artifacts_root) / "acoustic_analysis"


def human_contamination_cleaning_dir_for_bundle(
    bundle_dir: Path,
    research_root: Path = DEFAULT_HUMAN_CONTAMINATION_RESEARCH_ROOT,
) -> Path:
    return human_contamination_research_root_for_bundle(bundle_dir, research_root) / "human_contamination_cleaning"


def human_contamination_research_root_for_bundle(
    bundle_dir: Path,
    research_root: Path = DEFAULT_HUMAN_CONTAMINATION_RESEARCH_ROOT,
) -> Path:
    return research_root.resolve() / bundle_dir.resolve().name

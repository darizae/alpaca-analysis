from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_ARTIFACT_ROOT


@dataclass(frozen=True)
class LaunchPaths:
    launch_dir: Path
    configs_dir: Path
    manifests_dir: Path
    logs_dir: Path
    slurm_dir: Path
    hpc_bundle_dir: Path
    tables_dir: Path
    reports_dir: Path
    readme_dir: Path
    cache_dir: Path
    progress_dir: Path
    exemplar_manifests_dir: Path
    exemplar_concats_dir: Path

    @classmethod
    def create(cls, launch_dir: Path) -> "LaunchPaths":
        root = launch_dir.resolve()
        return cls(
            launch_dir=root,
            configs_dir=root / "configs",
            manifests_dir=root / "manifests",
            logs_dir=root / "logs",
            slurm_dir=root / "slurm",
            hpc_bundle_dir=root / "hpc_bundle",
            tables_dir=root / "tables",
            reports_dir=root / "reports",
            readme_dir=root / "readme",
            cache_dir=root / "cache",
            progress_dir=root / "progress",
            exemplar_manifests_dir=root / "top_model_exemplar_manifests",
            exemplar_concats_dir=root / "top_model_exemplar_concats",
        )

    @classmethod
    def from_label(cls, label: str, artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> "LaunchPaths":
        return cls.create(artifact_root.resolve() / label)

    def ensure_dirs(self) -> None:
        for path in (
            self.launch_dir,
            self.configs_dir,
            self.manifests_dir,
            self.logs_dir,
            self.slurm_dir,
            self.hpc_bundle_dir,
            self.tables_dir,
            self.reports_dir,
            self.readme_dir,
            self.cache_dir,
            self.progress_dir,
            self.exemplar_manifests_dir,
            self.exemplar_concats_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        for relative in (
            "features",
            "merged_features",
            "candidate_pairs",
            "model_runs",
            "exemplars",
        ):
            (self.cache_dir / relative).mkdir(parents=True, exist_ok=True)

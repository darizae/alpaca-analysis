from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return backend_root().parent


def default_backend_config_path() -> Path:
    return backend_root() / "config" / "default.toml"


@dataclass(frozen=True)
class AnalysisLaunchPaths:
    analysis_root: Path
    launches_root: Path
    latest_path: Path
    launch_dir: Path
    base_snapshot_dir: Path
    tables_dir: Path
    reports_dir: Path
    cache_dir: Path
    progress_dir: Path
    stage_progress_dir: Path
    task_progress_dir: Path
    specs_dir: Path
    configs_dir: Path
    hpc_bundle_dir: Path

    @classmethod
    def create(cls, analysis_root: Path, launch_label: str) -> "AnalysisLaunchPaths":
        analysis_root = analysis_root.resolve()
        launches_root = analysis_root / "launches"
        launch_dir = launches_root / launch_label
        return cls(
            analysis_root=analysis_root,
            launches_root=launches_root,
            latest_path=analysis_root / "latest",
            launch_dir=launch_dir,
            base_snapshot_dir=launch_dir / "base_snapshot",
            tables_dir=launch_dir / "tables",
            reports_dir=launch_dir / "reports",
            cache_dir=launch_dir / "cache",
            progress_dir=launch_dir / "progress",
            stage_progress_dir=launch_dir / "progress" / "stages",
            task_progress_dir=launch_dir / "progress" / "tasks",
            specs_dir=launch_dir / "specs",
            configs_dir=launch_dir / "configs",
            hpc_bundle_dir=launch_dir / "hpc_bundle",
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.analysis_root,
            self.launches_root,
            self.launch_dir,
            self.base_snapshot_dir,
            self.tables_dir,
            self.reports_dir,
            self.cache_dir,
            self.progress_dir,
            self.stage_progress_dir,
            self.task_progress_dir,
            self.specs_dir,
            self.configs_dir,
            self.hpc_bundle_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


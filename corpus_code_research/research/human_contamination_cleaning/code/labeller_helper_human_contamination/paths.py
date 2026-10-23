from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from labeller_helper_run_paths import (
    DEFAULT_HUMAN_CONTAMINATION_RESEARCH_ROOT,
    human_contamination_cleaning_dir_for_bundle,
)


@dataclass(frozen=True)
class HumanContaminationPaths:
    root: Path
    tables_dir: Path
    reports_dir: Path
    audio_exports_dir: Path

    @classmethod
    def for_bundle(
        cls,
        bundle_dir: Path,
        *,
        research_root: Path = DEFAULT_HUMAN_CONTAMINATION_RESEARCH_ROOT,
    ) -> "HumanContaminationPaths":
        root = human_contamination_cleaning_dir_for_bundle(bundle_dir, research_root)
        return cls(
            root=root,
            tables_dir=root / "tables",
            reports_dir=root / "reports",
            audio_exports_dir=root / "audio_exports",
        )

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.audio_exports_dir.mkdir(parents=True, exist_ok=True)

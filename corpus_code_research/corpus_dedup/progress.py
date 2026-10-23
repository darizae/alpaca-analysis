from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest_utils import write_json
from .paths import LaunchPaths


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stage_path(paths: LaunchPaths, stage: str) -> Path:
    return paths.progress_dir / f"{stage}_progress.json"


def write_stage_progress(paths: LaunchPaths, stage: str, payload: dict[str, Any]) -> None:
    write_json(_stage_path(paths, stage), {"stage": stage, **payload, "last_update_utc": utc_now_iso()})


def write_launch_progress(paths: LaunchPaths, payload: dict[str, Any]) -> None:
    write_json(paths.progress_dir / "launch_progress.json", {**payload, "last_update_utc": utc_now_iso()})


def read_progress(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf8"))

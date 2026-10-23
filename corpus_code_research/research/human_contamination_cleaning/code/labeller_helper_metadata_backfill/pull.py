from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable

from .remote_inventory import ResolvedSourceRecord



def build_files_to_pull(records: Iterable[ResolvedSourceRecord]) -> list[str]:
    paths: set[str] = set()
    for record in records:
        for candidate in (
            record.csv_hpc_path,
            record.device_txt_hpc_path,
            record.log_txt_hpc_path,
            record.settings_txt_hpc_path,
        ):
            if candidate:
                paths.add(candidate)
    return sorted(paths)



def write_files_to_pull(output_dir: Path, remote_paths: list[str]) -> Path:
    path = output_dir / "files_to_pull.txt"
    content = "".join(f"{remote_path.lstrip('/')}\n" for remote_path in remote_paths)
    path.write_text(content, encoding="utf8")
    return path


def pull_metadata_files(host: str, files_to_pull_path: Path, files_root: Path) -> list[str]:
    files_root.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "-rt",
        "--no-perms",
        "--no-owner",
        "--no-group",
        "--files-from",
        str(files_to_pull_path),
        f"{host}:/",
        str(files_root),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "rsync pull failed")
    return [" ".join(command)]



def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

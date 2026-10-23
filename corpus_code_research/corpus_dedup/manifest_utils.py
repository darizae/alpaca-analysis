from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        temp_path.write_text(payload, encoding="utf8")
        temp_path.replace(path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, yaml.safe_dump(json_ready(payload), sort_keys=False))


@lru_cache(maxsize=1)
def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
        "pyyaml": yaml.__version__,
    }


@lru_cache(maxsize=1)
def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


@lru_cache(maxsize=1)
def runtime_context() -> dict[str, Any]:
    return {
        "package_versions": package_versions(),
        "git_sha": git_sha(),
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf8"))


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: dict[str, Any]) -> str:
    return sha256_text(json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")))


def register_file(path: Path, logical_name: str, category: str) -> dict[str, Any]:
    return {
        "logical_name": logical_name,
        "category": category,
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_of_file(path),
    }


def register_table(path: Path, logical_name: str, category: str, df: pd.DataFrame) -> dict[str, Any]:
    record = register_file(path, logical_name, category)
    record["row_count"] = int(len(df.index))
    record["columns"] = list(df.columns)
    record["dtypes"] = {column: str(dtype) for column, dtype in df.dtypes.items()}
    return record

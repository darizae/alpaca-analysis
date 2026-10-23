from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from .paths import CANONICAL_HPC_RAW_ROOT, NormalizedSourcePath


@dataclass(frozen=True)
class RemoteFileInventory:
    collection_name: str
    raw_recordings_dir: str
    files_by_name: dict[str, str]
    command: str


@dataclass(frozen=True)
class ResolvedSourceRecord:
    source_audio_file_original: str
    source_audio_file_normalized: str
    audio_collection_name: str
    raw_recordings_dir: str
    wav_basename: str
    wav_stem: str
    expected_csv_name: str
    wav_hpc_path: str | None
    csv_hpc_path: str | None
    device_txt_hpc_path: str | None
    log_txt_hpc_path: str | None
    settings_txt_hpc_path: str | None
    wav_found: bool
    csv_found: bool
    device_found: bool
    log_found: bool
    settings_found: bool
    metadata_join_status: str


class RemoteInventoryError(RuntimeError):
    """Raised when remote inventory collection fails."""



def _run_ssh_json(host: str, script: str) -> str:
    remote_command = f"python3 -c {shlex.quote(script)}"
    command = ["ssh", host, remote_command]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RemoteInventoryError(completed.stderr.strip() or completed.stdout.strip() or "ssh command failed")
    return completed.stdout



def collect_collection_inventories(
    host: str,
    collection_names: Iterable[str],
    canonical_raw_root: PurePosixPath = CANONICAL_HPC_RAW_ROOT,
) -> dict[str, RemoteFileInventory]:
    inventories: dict[str, RemoteFileInventory] = {}
    for collection_name in sorted(set(collection_names)):
        raw_recordings_dir = canonical_raw_root / collection_name / "raw_recordings"
        python_path_literal = repr(str(raw_recordings_dir))
        script = (
            "import json, os, pathlib; "
            f"raw_dir = pathlib.Path({python_path_literal}); "
            "items = []; "
            "\nif raw_dir.is_dir():\n"
            "  for child in sorted(raw_dir.iterdir(), key=lambda item: item.name):\n"
            "    if child.is_file():\n"
            "      items.append({'name': child.name, 'path': str(child)})\n"
            "print(json.dumps({'raw_recordings_dir': str(raw_dir), 'items': items}, sort_keys=True))"
        )
        output = _run_ssh_json(host, script)
        payload = json.loads(output)
        files_by_name = {item["name"]: item["path"] for item in payload.get("items", [])}
        inventories[collection_name] = RemoteFileInventory(
            collection_name=collection_name,
            raw_recordings_dir=payload["raw_recordings_dir"],
            files_by_name=files_by_name,
            command=f"ssh {host} python3 -c {shlex.quote(script)}",
        )
    return inventories



def resolve_sources(
    normalized_sources: list[NormalizedSourcePath],
    inventories: dict[str, RemoteFileInventory],
) -> list[ResolvedSourceRecord]:
    resolved_records: list[ResolvedSourceRecord] = []
    for source in normalized_sources:
        inventory = inventories.get(source.audio_collection_name)
        if inventory is None:
            resolved_records.append(
                ResolvedSourceRecord(
                    source_audio_file_original=source.source_audio_file_original,
                    source_audio_file_normalized=source.source_audio_file_normalized,
                    audio_collection_name=source.audio_collection_name,
                    raw_recordings_dir=source.raw_recordings_dir,
                    wav_basename=source.wav_basename,
                    wav_stem=source.wav_stem,
                    expected_csv_name=source.expected_csv_name,
                    wav_hpc_path=None,
                    csv_hpc_path=None,
                    device_txt_hpc_path=None,
                    log_txt_hpc_path=None,
                    settings_txt_hpc_path=None,
                    wav_found=False,
                    csv_found=False,
                    device_found=False,
                    log_found=False,
                    settings_found=False,
                    metadata_join_status="missing_wav",
                )
            )
            continue
        files_by_name = inventory.files_by_name
        wav_hpc_path = files_by_name.get(source.wav_basename)
        csv_hpc_path = files_by_name.get(source.expected_csv_name)
        device_txt_hpc_path = files_by_name.get("DEVICE.TXT")
        log_txt_hpc_path = files_by_name.get("LOG.TXT")
        settings_txt_hpc_path = files_by_name.get("SETTINGS.txt")

        wav_found = wav_hpc_path is not None
        csv_found = csv_hpc_path is not None
        device_found = device_txt_hpc_path is not None
        log_found = log_txt_hpc_path is not None
        settings_found = settings_txt_hpc_path is not None

        if not wav_found:
            status = "missing_wav"
        elif not csv_found:
            status = "missing_csv"
        elif not (device_found and log_found and settings_found):
            status = "missing_collection_metadata"
        else:
            status = "resolved"

        resolved_records.append(
            ResolvedSourceRecord(
                source_audio_file_original=source.source_audio_file_original,
                source_audio_file_normalized=source.source_audio_file_normalized,
                audio_collection_name=source.audio_collection_name,
                raw_recordings_dir=source.raw_recordings_dir,
                wav_basename=source.wav_basename,
                wav_stem=source.wav_stem,
                expected_csv_name=source.expected_csv_name,
                wav_hpc_path=wav_hpc_path,
                csv_hpc_path=csv_hpc_path,
                device_txt_hpc_path=device_txt_hpc_path,
                log_txt_hpc_path=log_txt_hpc_path,
                settings_txt_hpc_path=settings_txt_hpc_path,
                wav_found=wav_found,
                csv_found=csv_found,
                device_found=device_found,
                log_found=log_found,
                settings_found=settings_found,
                metadata_join_status=status,
            )
        )
    return resolved_records

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeviceMetadata:
    device_id: str | None
    firmware_version: str | None
    firmware_description: str | None


@dataclass(frozen=True)
class SettingsMetadata:
    settings: dict[str, object] | None
    settings_hash: str | None


@dataclass(frozen=True)
class LogMetadata:
    event_count: int
    early_stop_count: int
    first_event_time: str | None
    last_event_time: str | None


LOG_LINE_RE = re.compile(r"^(?P<stamp>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}): (?P<message>.*)$")



def parse_device_txt(path: Path) -> DeviceMetadata:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip()
    return DeviceMetadata(
        device_id=values.get("device"),
        firmware_version=values.get("firmware version"),
        firmware_description=values.get("firmware description"),
    )



def _strip_settings_comments(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        if line.strip():
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)



def parse_settings_txt(path: Path) -> SettingsMetadata:
    raw = path.read_text(encoding="utf8")
    cleaned = _strip_settings_comments(raw)
    settings_start = cleaned.find("{")
    settings_end = cleaned.rfind("}")
    if settings_start < 0 or settings_end < settings_start:
        return SettingsMetadata(settings=None, settings_hash=None)
    settings_text = cleaned[settings_start : settings_end + 1]
    settings_text = re.sub(r"(\b[a-zA-Z_][a-zA-Z0-9_]*\b)\s*:", r'"\1":', settings_text)
    settings_text = settings_text.replace("\t", "    ")
    settings_text = re.sub(r",\s*([}\]])", r"\1", settings_text)
    parsed = ast.literal_eval(settings_text)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return SettingsMetadata(settings=parsed, settings_hash=hashlib.sha256(canonical.encode("utf8")).hexdigest())



def parse_log_txt(path: Path) -> LogMetadata:
    event_count = 0
    early_stop_count = 0
    first_event_time = None
    last_event_time = None
    for line in path.read_text(encoding="utf8", errors="replace").splitlines():
        match = LOG_LINE_RE.match(line.strip())
        if not match:
            continue
        event_count += 1
        stamp = match.group("stamp")
        first_event_time = first_event_time or stamp
        last_event_time = stamp
        if "stopped early due to file size limit" in match.group("message"):
            early_stop_count += 1
    return LogMetadata(
        event_count=event_count,
        early_stop_count=early_stop_count,
        first_event_time=first_event_time,
        last_event_time=last_event_time,
    )

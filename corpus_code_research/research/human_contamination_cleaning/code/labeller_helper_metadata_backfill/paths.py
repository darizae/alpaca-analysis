from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

CANONICAL_HPC_RAW_ROOT = PurePosixPath("/cluster/projects/alpaca/data/raw_audio_collection")
KNOWN_RAW_ROOT_PREFIXES = (
    "/cluster/projects/alpaca/data/raw_audio_collection",
)


@dataclass(frozen=True)
class NormalizedSourcePath:
    source_audio_file_original: str
    source_audio_file_normalized: str
    audio_collection_name: str
    raw_recordings_dir: str
    wav_basename: str
    wav_stem: str
    expected_csv_name: str


class PathNormalizationError(ValueError):
    """Raised when a bundle source path cannot be normalized deterministically."""



def normalize_source_audio_path(
    source_audio_file: str,
    canonical_raw_root: PurePosixPath = CANONICAL_HPC_RAW_ROOT,
) -> NormalizedSourcePath:
    candidate = source_audio_file.strip()
    if not candidate:
        raise PathNormalizationError("source_audio_file is empty")

    matching_prefix = next((prefix for prefix in KNOWN_RAW_ROOT_PREFIXES if candidate.startswith(prefix)), None)
    if matching_prefix is None:
        raise PathNormalizationError(f"source_audio_file does not start with a known raw root: {candidate}")

    relative = candidate[len(matching_prefix) :].lstrip("/")
    relative_path = PurePosixPath(relative)
    parts = relative_path.parts
    if len(parts) < 3:
        raise PathNormalizationError(f"source_audio_file is too short to contain collection/raw_recordings/WAV: {candidate}")
    if parts[1] != "raw_recordings":
        raise PathNormalizationError(f"source_audio_file does not point into raw_recordings: {candidate}")

    audio_collection_name = parts[0]
    wav_basename = parts[-1]
    wav_stem = PurePosixPath(wav_basename).stem
    raw_recordings_dir = canonical_raw_root / audio_collection_name / "raw_recordings"
    normalized_path = raw_recordings_dir / wav_basename
    return NormalizedSourcePath(
        source_audio_file_original=source_audio_file,
        source_audio_file_normalized=str(normalized_path),
        audio_collection_name=audio_collection_name,
        raw_recordings_dir=str(raw_recordings_dir),
        wav_basename=wav_basename,
        wav_stem=wav_stem,
        expected_csv_name=f"{wav_stem}.CSV",
    )

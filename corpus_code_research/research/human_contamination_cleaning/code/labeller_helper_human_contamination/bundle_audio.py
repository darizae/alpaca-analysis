from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BundleAudioInfo:
    wav_path: Path
    sample_rate_hz: int
    sample_width_bytes: int
    n_channels: int
    frame_count: int


def read_bundle_audio_info(review_bundle: Path) -> BundleAudioInfo:
    wav_path = review_bundle / "review_concat.wav"
    if not wav_path.is_file():
        raise FileNotFoundError(f"Missing bundle review concat WAV: {wav_path}")
    with wave.open(str(wav_path), "rb") as handle:
        return BundleAudioInfo(
            wav_path=wav_path,
            sample_rate_hz=handle.getframerate(),
            sample_width_bytes=handle.getsampwidth(),
            n_channels=handle.getnchannels(),
            frame_count=handle.getnframes(),
        )


class ConcatWavReader:
    def __init__(self, audio_info: BundleAudioInfo) -> None:
        self.audio_info = audio_info
        self._samples: np.ndarray | None = None

    def __enter__(self) -> "ConcatWavReader":
        with wave.open(str(self.audio_info.wav_path), "rb") as handle:
            raw = handle.readframes(handle.getnframes())
        if self.audio_info.sample_width_bytes != 2:
            raise ValueError(
                f"Only 16-bit PCM review_concat WAVs are supported, got sample width {self.audio_info.sample_width_bytes}"
            )
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
        if self.audio_info.n_channels > 1:
            samples = samples.reshape(-1, self.audio_info.n_channels).mean(axis=1)
        self._samples = samples
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._samples = None

    def read_mono(self, begin_s: float, end_s: float) -> np.ndarray:
        if self._samples is None:
            raise RuntimeError("ConcatWavReader must be used as a context manager")
        start_index = max(0, int(round(begin_s * self.audio_info.sample_rate_hz)))
        end_index = max(start_index, int(round(end_s * self.audio_info.sample_rate_hz)))
        end_index = min(end_index, len(self._samples))
        return self._samples[start_index:end_index].copy()

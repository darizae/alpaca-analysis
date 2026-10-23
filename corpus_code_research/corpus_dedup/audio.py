from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, spectrogram

from .config import FeatureConfig, PathRewrite


def apply_path_rewrites(path: str, rewrites: list[PathRewrite]) -> Path:
    for rewrite in rewrites:
        if path.startswith(rewrite.source_prefix):
            return Path(rewrite.target_prefix + path[len(rewrite.source_prefix) :])
    return Path(path)


def _to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        scale = float(np.iinfo(samples.dtype).max)
        if scale <= 0:
            scale = 1.0
        return samples.astype(np.float32) / scale
    return samples.astype(np.float32)


def load_wav(path: Path, *, mmap: bool = False) -> tuple[int, np.ndarray]:
    sample_rate, samples = wavfile.read(path, mmap=mmap)
    array = np.asarray(samples)
    if mmap:
        return int(sample_rate), array
    return int(sample_rate), _to_float32(array)


def extract_snippet(samples: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start_index = max(0, int(math.floor(start_s * sample_rate)))
    end_index = min(len(samples), int(math.ceil(end_s * sample_rate)))
    if end_index <= start_index:
        raise ValueError(f"Invalid snippet bounds: start_s={start_s}, end_s={end_s}, sample_rate={sample_rate}")
    return _to_float32(np.asarray(samples[start_index:end_index]))


def _frame_rms(snippet: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(snippet) <= frame_length:
        return np.array([float(np.sqrt(np.mean(np.square(snippet)) + 1e-12))], dtype=np.float64)
    frame_count = 1 + (len(snippet) - frame_length) // hop_length
    values = np.empty(frame_count, dtype=np.float64)
    for frame_index in range(frame_count):
        start = frame_index * hop_length
        frame = snippet[start : start + frame_length]
        values[frame_index] = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
    return values


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return 0.0
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return float(values[-1])
    cumulative = np.cumsum(weights) / total_weight
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    index = min(max(index, 0), len(values) - 1)
    return float(values[index])


def _smooth(values: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 1 or len(values) <= 2:
        return values
    kernel = np.ones(int(bins), dtype=np.float64) / float(bins)
    return np.convolve(values, kernel, mode="same")


def _fundamental_frequency(frequencies: np.ndarray, spectrum_power: np.ndarray, config: FeatureConfig) -> float:
    if len(frequencies) == 0:
        return 0.0
    positive_mask = frequencies > 0.0
    positive_frequencies = frequencies[positive_mask]
    positive_power = spectrum_power[positive_mask]
    if len(positive_frequencies) == 0:
        return 0.0
    smoothed = _smooth(positive_power, config.spectral_smoothing_bins)
    prominence_floor = float(np.max(smoothed) * max(config.f0_min_prominence_ratio, 0.0))
    peaks, properties = find_peaks(smoothed, prominence=prominence_floor)
    if len(peaks) == 0:
        return float(positive_frequencies[int(np.argmax(smoothed))])
    prominences = properties.get("prominences", np.zeros(len(peaks), dtype=np.float64))
    ordering = np.argsort(positive_frequencies[peaks], kind="stable")
    for ordered_index in ordering:
        peak_index = peaks[int(ordered_index)]
        if prominences[int(ordered_index)] >= prominence_floor:
            return float(positive_frequencies[int(peak_index)])
    return float(positive_frequencies[int(peaks[0])])


def _average_entropy(power: np.ndarray) -> float:
    if power.ndim == 1:
        power = power[:, np.newaxis]
    entropy_values: list[float] = []
    weights: list[float] = []
    normalization = math.log(max(power.shape[0], 2))
    for frame_index in range(power.shape[1]):
        frame = power[:, frame_index]
        total = float(np.sum(frame))
        if total <= 0:
            continue
        distribution = frame / total
        entropy = -float(np.sum(distribution * np.log(distribution + 1e-12))) / normalization
        entropy_values.append(entropy)
        weights.append(total)
    if not entropy_values:
        return 0.0
    return float(np.average(np.asarray(entropy_values), weights=np.asarray(weights)))


def compute_features(snippet: np.ndarray, sample_rate: int, config: FeatureConfig) -> dict[str, float]:
    duration_s = float(len(snippet) / sample_rate)
    rms = float(np.sqrt(np.mean(np.square(snippet)) + 1e-12))
    frame_rms = _frame_rms(snippet, config.frame_length, config.hop_length)
    floor_frame_count = max(1, int(math.ceil(len(frame_rms) * config.noise_floor_quantile)))
    noise_rms = float(np.mean(np.sort(frame_rms)[:floor_frame_count]))
    snr_db = float(20.0 * np.log10((rms + 1e-12) / (noise_rms + 1e-12)))

    frequencies, times, spectrum = spectrogram(
        snippet,
        fs=sample_rate,
        window="hann",
        nperseg=min(config.frame_length, len(snippet)),
        noverlap=max(0, min(config.frame_length, len(snippet)) - config.hop_length),
        nfft=max(config.n_fft, min(config.frame_length, len(snippet))),
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if spectrum.ndim == 1:
        spectrum = spectrum[:, np.newaxis]
    power = np.square(spectrum) + 1e-12

    frame_energy = power.sum(axis=0)
    spectrum_energy = power.sum(axis=1)

    frequency5_hz = _weighted_quantile(frequencies, spectrum_energy, 0.05)
    frequency95_hz = _weighted_quantile(frequencies, spectrum_energy, 0.95)
    duration5_s = _weighted_quantile(times, frame_energy, 0.05) if len(times) > 0 else 0.0
    duration95_s = _weighted_quantile(times, frame_energy, 0.95) if len(times) > 0 else duration_s
    duration90_s = max(duration95_s - duration5_s, 0.0)

    peak_frequency_hz = float(frequencies[int(np.argmax(power.max(axis=1)))]) if len(frequencies) > 0 else 0.0
    f0_hz = _fundamental_frequency(frequencies, spectrum_energy, config)
    average_entropy = _average_entropy(power)

    return {
        "snippet_duration_s": duration_s,
        "snippet_rms": rms,
        "snippet_snr_db": snr_db,
        "f0_hz": float(f0_hz),
        "duration90_s": float(min(duration90_s, duration_s)),
        "peak_frequency_hz": float(peak_frequency_hz),
        "frequency5_hz": float(frequency5_hz),
        "frequency95_hz": float(frequency95_hz),
        "bandwidth90_hz": float(max(frequency95_hz - frequency5_hz, 0.0)),
        "average_entropy": float(average_entropy),
    }

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import (
    DEFAULT_ALGORITHM_VERSION,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXEMPLARS_PER_MODEL,
    DEFAULT_GPS_THRESHOLDS_M,
    DEFAULT_GROUP_ACOUSTIC_THRESHOLDS,
    DEFAULT_OVERLAP_FRACTION_SHORT_THRESHOLDS,
    DEFAULT_RELATIVE_DIFFERENCE_THRESHOLDS,
    DEFAULT_TOP_K_MODELS,
)


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_optional_float_list(values: list[Any], default: list[float | None]) -> list[float | None]:
    raw_values = values if values else list(default)
    parsed: list[float | None] = []
    for value in raw_values:
        if value is None or str(value).lower() == "none":
            parsed.append(None)
        else:
            parsed.append(float(value))
    return parsed


def _parse_group_acoustic_thresholds(values: list[Any]) -> list[str | None]:
    parsed: list[str | None] = []
    raw_values = values if values else list(DEFAULT_GROUP_ACOUSTIC_THRESHOLDS)
    for value in raw_values:
        if value is None or str(value).lower() == "none":
            parsed.append(None)
            continue
        label = str(value).strip().lower()
        if label not in {"q90", "q75", "q50"}:
            raise ValueError(f"Unsupported group acoustic threshold: {value!r}")
        parsed.append(label)
    return parsed


@dataclass(frozen=True)
class PathRewrite:
    source_prefix: str
    target_prefix: str


@dataclass(frozen=True)
class FeatureConfig:
    frame_length: int = 1024
    hop_length: int = 512
    n_fft: int = 2048
    noise_floor_quantile: float = 0.20
    f0_min_prominence_ratio: float = 0.10
    spectral_smoothing_bins: int = 5


@dataclass(frozen=True)
class RobustnessConfig:
    acoustic_margin_quantile: float = 0.25
    target_acoustic_margin: float = 0.10
    min_final_selectable_acoustic_coverage: float = 0.95
    decision_acoustic_core_weight: float = 0.75
    removal_count_stability_weight: float = 0.25


@dataclass(frozen=True)
class HPCStageConfig:
    cpus_per_task: int
    memory_gb: int
    time_limit: str
    array_concurrency: int


@dataclass(frozen=True)
class HPCConfig:
    host: str = "hpc-cluster"
    partition: str = "compute"
    account: str = "alpaca-project"
    remote_repo_root: str = ""
    remote_venv_path: str = ""
    features: HPCStageConfig = field(
        default_factory=lambda: HPCStageConfig(cpus_per_task=8, memory_gb=32, time_limit="08:00:00", array_concurrency=4)
    )
    feature_merge: HPCStageConfig = field(
        default_factory=lambda: HPCStageConfig(cpus_per_task=4, memory_gb=16, time_limit="01:00:00", array_concurrency=1)
    )
    grid: HPCStageConfig = field(
        default_factory=lambda: HPCStageConfig(cpus_per_task=4, memory_gb=16, time_limit="02:00:00", array_concurrency=4)
    )
    aggregate: HPCStageConfig = field(
        default_factory=lambda: HPCStageConfig(cpus_per_task=4, memory_gb=16, time_limit="01:00:00", array_concurrency=1)
    )
    exemplars: HPCStageConfig = field(
        default_factory=lambda: HPCStageConfig(cpus_per_task=4, memory_gb=16, time_limit="01:00:00", array_concurrency=1)
    )


@dataclass(frozen=True)
class DedupConfig:
    config_path: Path
    launch_label: str
    artifact_root: Path
    input_table: Path
    source_wav_index: Path
    algorithm_version: str = DEFAULT_ALGORITHM_VERSION
    input_format: str = "parquet"
    launch_dir: Path | None = None
    row_limit: int | None = None
    feature_shard_size_wavs: int = 1
    overlap_fraction_short_thresholds: list[float] = field(
        default_factory=lambda: list(DEFAULT_OVERLAP_FRACTION_SHORT_THRESHOLDS)
    )
    gps_thresholds_m: list[float] = field(default_factory=lambda: list(DEFAULT_GPS_THRESHOLDS_M))
    relative_difference_thresholds: list[float | None] = field(
        default_factory=lambda: list(DEFAULT_RELATIVE_DIFFERENCE_THRESHOLDS)
    )
    group_acoustic_thresholds: list[str | None] = field(
        default_factory=lambda: list(DEFAULT_GROUP_ACOUSTIC_THRESHOLDS)
    )
    top_k_models: int = DEFAULT_TOP_K_MODELS
    exemplars_per_model: int = DEFAULT_EXEMPLARS_PER_MODEL
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    hpc: HPCConfig = field(default_factory=HPCConfig)
    path_rewrites: list[PathRewrite] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_path"] = str(self.config_path)
        payload["artifact_root"] = str(self.artifact_root)
        payload["input_table"] = str(self.input_table)
        payload["source_wav_index"] = str(self.source_wav_index)
        payload["launch_dir"] = str(self.launch_dir) if self.launch_dir is not None else None
        return payload


def _load_stage_config(payload: dict[str, Any], defaults: HPCStageConfig) -> HPCStageConfig:
    return HPCStageConfig(
        cpus_per_task=int(payload.get("cpus_per_task", defaults.cpus_per_task)),
        memory_gb=int(payload.get("memory_gb", defaults.memory_gb)),
        time_limit=str(payload.get("time_limit", defaults.time_limit)),
        array_concurrency=int(payload.get("array_concurrency", defaults.array_concurrency)),
    )


def load_config(path: Path) -> DedupConfig:
    resolved = path.resolve()
    base_dir = resolved.parent
    raw = yaml.safe_load(resolved.read_text(encoding="utf8")) or {}
    forbidden_keys = [
        "gap_thresholds_s",
        "model_batch_grouping",
        "time_thresholds_s",
        "snr_quantiles",
        "acoustic_quantiles",
        "acoustic_distance_columns",
        "inspection",
        "full_review",
    ]
    found_forbidden = [key for key in forbidden_keys if key in raw]
    if found_forbidden:
        joined = ", ".join(sorted(found_forbidden))
        raise ValueError(
            f"Unsupported legacy config keys in {resolved.name}: {joined}. "
            "Use the overlap-temporal v4 config schema."
        )

    feature_raw = raw.get("feature", {}) or {}
    hpc_raw = raw.get("hpc", {}) or {}
    robustness_raw = raw.get("robustness", {}) or {}
    algorithm_version = str(raw.get("algorithm_version", DEFAULT_ALGORITHM_VERSION))
    if algorithm_version != DEFAULT_ALGORITHM_VERSION:
        raise ValueError(
            f"Unsupported algorithm_version {algorithm_version!r}. "
            f"This workflow implements {DEFAULT_ALGORITHM_VERSION!r}."
        )
    default_hpc = HPCConfig()
    hpc = HPCConfig(
        host=str(hpc_raw.get("host", default_hpc.host)),
        partition=str(hpc_raw.get("partition", default_hpc.partition)),
        account=str(hpc_raw.get("account", default_hpc.account)),
        remote_repo_root=str(hpc_raw.get("remote_repo_root", default_hpc.remote_repo_root)),
        remote_venv_path=str(hpc_raw.get("remote_venv_path", default_hpc.remote_venv_path)),
        features=_load_stage_config(hpc_raw.get("features", {}) or {}, default_hpc.features),
        feature_merge=_load_stage_config(hpc_raw.get("feature_merge", {}) or {}, default_hpc.feature_merge),
        grid=_load_stage_config(hpc_raw.get("grid", {}) or {}, default_hpc.grid),
        aggregate=_load_stage_config(hpc_raw.get("aggregate", {}) or {}, default_hpc.aggregate),
        exemplars=_load_stage_config(hpc_raw.get("exemplars", {}) or {}, default_hpc.exemplars),
    )
    path_rewrites = [
        PathRewrite(
            source_prefix=str(item["source_prefix"]),
            target_prefix=str(item["target_prefix"]),
        )
        for item in (raw.get("path_rewrites", []) or [])
    ]
    artifact_root = _resolve_path(base_dir, raw.get("artifact_root")) or DEFAULT_ARTIFACT_ROOT.resolve()
    launch_dir = _resolve_path(base_dir, raw.get("launch_dir"))
    return DedupConfig(
        config_path=resolved,
        launch_label=str(raw.get("launch_label", resolved.stem)),
        artifact_root=artifact_root,
        input_table=_resolve_path(base_dir, raw["input_table"]),
        source_wav_index=_resolve_path(base_dir, raw["source_wav_index"]),
        algorithm_version=algorithm_version,
        input_format=str(raw.get("input_format", "parquet")),
        launch_dir=launch_dir,
        row_limit=int(raw["row_limit"]) if raw.get("row_limit") is not None else None,
        feature_shard_size_wavs=int(raw.get("feature_shard_size_wavs", 1)),
        overlap_fraction_short_thresholds=[
            float(value)
            for value in raw.get(
                "overlap_fraction_short_thresholds",
                DEFAULT_OVERLAP_FRACTION_SHORT_THRESHOLDS,
            )
        ],
        gps_thresholds_m=[float(value) for value in raw.get("gps_thresholds_m", DEFAULT_GPS_THRESHOLDS_M)],
        relative_difference_thresholds=_parse_optional_float_list(
            list(raw.get("relative_difference_thresholds", []) or []),
            list(DEFAULT_RELATIVE_DIFFERENCE_THRESHOLDS),
        ),
        group_acoustic_thresholds=_parse_group_acoustic_thresholds(
            list(raw.get("group_acoustic_thresholds", []) or [])
        ),
        top_k_models=int(raw.get("top_k_models", DEFAULT_TOP_K_MODELS)),
        exemplars_per_model=int(raw.get("exemplars_per_model", DEFAULT_EXEMPLARS_PER_MODEL)),
        feature=FeatureConfig(
            frame_length=int(feature_raw.get("frame_length", 1024)),
            hop_length=int(feature_raw.get("hop_length", 512)),
            n_fft=int(feature_raw.get("n_fft", 2048)),
            noise_floor_quantile=float(feature_raw.get("noise_floor_quantile", 0.20)),
            f0_min_prominence_ratio=float(feature_raw.get("f0_min_prominence_ratio", 0.10)),
            spectral_smoothing_bins=int(feature_raw.get("spectral_smoothing_bins", 5)),
        ),
        robustness=RobustnessConfig(
            acoustic_margin_quantile=float(
                robustness_raw.get("acoustic_margin_quantile", 0.25)
            ),
            target_acoustic_margin=float(
                robustness_raw.get("target_acoustic_margin", 0.10)
            ),
            min_final_selectable_acoustic_coverage=float(
                robustness_raw.get("min_final_selectable_acoustic_coverage", 0.95)
            ),
            decision_acoustic_core_weight=float(
                robustness_raw.get("decision_acoustic_core_weight", 0.75)
            ),
            removal_count_stability_weight=float(
                robustness_raw.get("removal_count_stability_weight", 0.25)
            ),
        ),
        hpc=hpc,
        path_rewrites=path_rewrites,
    )

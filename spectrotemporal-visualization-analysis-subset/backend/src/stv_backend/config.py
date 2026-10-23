from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AnalysisHpcResourceProfile:
    cpus_per_task: int = 2
    memory_gb: int = 8
    time_limit: str = "00:30:00"
    array_concurrency: int = 1


@dataclass(frozen=True)
class AnalysisHpcConfig:
    host: str = "hpc-cluster"
    partition: str = "compute"
    account: str = "alpaca-project"
    remote_repo_root: str = "/cluster/projects/alpaca/spectrotemporal-visualization"
    remote_venv_path: str = "/cluster/projects/alpaca/alpaca-pipelines/.venv"
    module_loads: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    resources: dict[str, AnalysisHpcResourceProfile] = field(
        default_factory=lambda: {
            "light": AnalysisHpcResourceProfile(cpus_per_task=2, memory_gb=8, time_limit="00:20:00", array_concurrency=1),
            "heavy": AnalysisHpcResourceProfile(cpus_per_task=4, memory_gb=16, time_limit="02:00:00", array_concurrency=8),
            "materialize_48h": AnalysisHpcResourceProfile(
                cpus_per_task=4,
                memory_gb=64,
                time_limit="48:00:00",
                array_concurrency=1,
            ),
            "robustness": AnalysisHpcResourceProfile(
                cpus_per_task=4,
                memory_gb=16,
                time_limit="12:00:00",
                array_concurrency=4,
            ),
            "model_fit": AnalysisHpcResourceProfile(
                cpus_per_task=4,
                memory_gb=64,
                time_limit="48:00:00",
                array_concurrency=3,
            ),
        }
    )


@dataclass(frozen=True)
class BackendConfig:
    dataset_name: str
    timezone_name: str
    raw_corpus_dir: Path
    sidecar_cache_dir: Path
    processed_dir: Path
    intermediate_dir: Path
    gps_blocker_path: Path
    ssh_host: str
    canonical_raw_root: str
    config_path: Path
    analysis_hpc: AnalysisHpcConfig = field(default_factory=AnalysisHpcConfig)


def _resolve_path(base_dir: Path, value: str) -> Path:
    return (base_dir / value).resolve()


def _load_analysis_hpc_config(raw: dict[str, object], ssh_host: str) -> AnalysisHpcConfig:
    analysis_raw = (raw.get("analysis_hpc") or {}) if isinstance(raw, dict) else {}
    if not isinstance(analysis_raw, dict):
        raise ValueError("analysis_hpc must be a table")

    defaults = AnalysisHpcConfig()
    resources_raw = analysis_raw.get("resources") or {}
    if resources_raw and not isinstance(resources_raw, dict):
        raise ValueError("analysis_hpc.resources must be a table of named profiles")

    resources = dict(defaults.resources)
    for name, payload in resources_raw.items():
        if not isinstance(payload, dict):
            raise ValueError(f"analysis_hpc.resources.{name} must be a table")
        fallback = resources.get(str(name), AnalysisHpcResourceProfile())
        resources[str(name)] = AnalysisHpcResourceProfile(
            cpus_per_task=int(payload.get("cpus_per_task", fallback.cpus_per_task)),
            memory_gb=int(payload.get("memory_gb", fallback.memory_gb)),
            time_limit=str(payload.get("time_limit", fallback.time_limit)),
            array_concurrency=int(payload.get("array_concurrency", fallback.array_concurrency)),
        )

    return AnalysisHpcConfig(
        host=os.environ.get("ANALYSIS_HPC_HOST", str(analysis_raw.get("host", ssh_host or defaults.host))),
        partition=os.environ.get("ANALYSIS_HPC_PARTITION", str(analysis_raw.get("partition", defaults.partition))),
        account=os.environ.get("ANALYSIS_HPC_ACCOUNT", str(analysis_raw.get("account", defaults.account))),
        remote_repo_root=os.environ.get(
            "ANALYSIS_HPC_REMOTE_REPO_ROOT",
            str(analysis_raw.get("remote_repo_root", defaults.remote_repo_root)),
        ),
        remote_venv_path=os.environ.get(
            "ANALYSIS_HPC_REMOTE_VENV_PATH",
            str(analysis_raw.get("remote_venv_path", defaults.remote_venv_path)),
        ),
        module_loads=tuple(str(value) for value in analysis_raw.get("module_loads", defaults.module_loads)),
        environment={str(key): str(value) for key, value in dict(analysis_raw.get("environment", {})).items()},
        resources=resources,
    )


def load_config(config_path: Path) -> BackendConfig:
    config_path = config_path.resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf8"))
    base_dir = config_path.parent

    dataset = raw["dataset"]
    paths = raw["paths"]
    remote = raw["remote"]

    return BackendConfig(
        dataset_name=str(dataset["name"]),
        timezone_name=str(dataset["timezone"]),
        raw_corpus_dir=_resolve_path(base_dir, str(paths["raw_corpus_dir"])),
        sidecar_cache_dir=_resolve_path(base_dir, str(paths["sidecar_cache_dir"])),
        processed_dir=_resolve_path(base_dir, str(paths["processed_dir"])),
        intermediate_dir=_resolve_path(base_dir, str(paths["intermediate_dir"])),
        gps_blocker_path=_resolve_path(base_dir, str(paths["gps_blocker_path"])),
        ssh_host=str(remote.get("ssh_host", "") or ""),
        canonical_raw_root=str(remote["canonical_raw_root"]),
        config_path=config_path,
        analysis_hpc=_load_analysis_hpc_config(raw, str(remote.get("ssh_host", "") or "")),
    )

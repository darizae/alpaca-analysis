from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import tempfile
import re
import shlex
import subprocess
from typing import Any, Callable

import pandas as pd

from stv_backend.config import AnalysisHpcResourceProfile, BackendConfig

from .paths import AnalysisLaunchPaths, default_backend_config_path, repo_root


COMPUTE_CLASS_HEAVY = "heavy_compute"
COMPUTE_CLASS_LOCAL_OK = "local_ok"
SINGLE_TASK_SENTINEL = "__single__"


@dataclass(frozen=True)
class AnalysisStageSpec:
    stage_id: str
    description: str
    compute_class: str
    resource_profile: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisDefinition:
    analysis_id: str
    output_dirname: str
    docs_to_snapshot: tuple[str, ...]
    stage_specs: tuple[AnalysisStageSpec, ...]
    initialize_launch: Callable[["LaunchContext", str], dict[str, Any]]
    run_stage: Callable[["LaunchContext", AnalysisStageSpec, str | None], dict[str, Any]]


@dataclass(frozen=True)
class LaunchContext:
    definition: AnalysisDefinition
    config: BackendConfig
    paths: AnalysisLaunchPaths
    launch_manifest: dict[str, Any]
    launch_manifest_path: Path

    def stage_spec(self, stage_id: str) -> AnalysisStageSpec:
        for spec in self.definition.stage_specs:
            if spec.stage_id == stage_id:
                return spec
        raise KeyError(f"unknown stage: {stage_id}")

    def stage_task_values(self, stage_id: str) -> list[str]:
        task_values = self.launch_manifest.get("stage_tasks", {}).get(stage_id, [])
        if not isinstance(task_values, list):
            raise ValueError(f"launch manifest stage_tasks.{stage_id} must be a list")
        return [str(value) for value in task_values]


def utc_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf8", dir=path.parent, delete=False) as handle:
        handle.write(serialized)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def slugify_token(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not safe:
        safe = "task"
    digest = hashlib.sha1(value.encode("utf8")).hexdigest()[:8]
    return f"{safe}__{digest}"


def stage_progress_path(paths: AnalysisLaunchPaths, stage_id: str) -> Path:
    return paths.stage_progress_dir / f"{stage_id}.json"


def stage_task_progress_path(paths: AnalysisLaunchPaths, stage_id: str, task_value: str) -> Path:
    return paths.task_progress_dir / stage_id / f"{slugify_token(task_value)}.json"


def stage_cache_dir(paths: AnalysisLaunchPaths, stage_id: str) -> Path:
    return paths.cache_dir / stage_id


def artifact_tables_dir(definition: AnalysisDefinition) -> Path:
    return repo_root() / "backend" / "data" / "processed" / "analysis" / definition.output_dirname


def default_stage_progress(ctx: LaunchContext, spec: AnalysisStageSpec) -> dict[str, Any]:
    tasks = ctx.stage_task_values(spec.stage_id)
    return {
        "stage_id": spec.stage_id,
        "description": spec.description,
        "compute_class": spec.compute_class,
        "resource_profile": spec.resource_profile,
        "dependencies": list(spec.dependencies),
        "status": "pending",
        "task_total": max(1, len(tasks)),
        "completed_tasks": 0,
        "failed_tasks": 0,
        "last_update_utc": utc_now_iso(),
    }


def create_launch(
    *,
    definition: AnalysisDefinition,
    config: BackendConfig,
    launch_label: str,
    profile: str,
    input_manifest_path: Path,
) -> dict[str, Any]:
    paths = AnalysisLaunchPaths.create(artifact_tables_dir(definition), launch_label)
    if paths.launch_dir.exists() and any(paths.launch_dir.iterdir()):
        raise FileExistsError(f"launch dir already exists and is not empty: {paths.launch_dir}")
    paths.ensure_dirs()

    base_snapshot_manifest_path = snapshot_base_bundle(
        source_manifest_path=input_manifest_path.resolve(),
        destination_dir=paths.base_snapshot_dir,
    )
    specs_manifest = snapshot_specs(definition.docs_to_snapshot, paths.specs_dir)
    source_code_manifest = snapshot_analysis_code(definition)
    launch_manifest_path = paths.configs_dir / "launch_manifest.json"
    launch_manifest: dict[str, Any] = {
        "analysis_id": definition.analysis_id,
        "output_dirname": definition.output_dirname,
        "launch_label": launch_label,
        "profile": profile,
        "created_at_utc": utc_now_iso(),
        "config_path": str(config.config_path),
        "source_base_manifest_path": str(input_manifest_path.resolve()),
        "base_snapshot_manifest_path": str(base_snapshot_manifest_path.relative_to(paths.launch_dir)),
        "specs_manifest": specs_manifest,
        "source_code_manifest": source_code_manifest,
        "stage_specs": [
            {
                "stage_id": spec.stage_id,
                "description": spec.description,
                "compute_class": spec.compute_class,
                "resource_profile": spec.resource_profile,
                "dependencies": list(spec.dependencies),
            }
            for spec in definition.stage_specs
        ],
        "stage_tasks": {},
        "analysis": {},
    }
    ctx = LaunchContext(
        definition=definition,
        config=config,
        paths=paths,
        launch_manifest=launch_manifest,
        launch_manifest_path=launch_manifest_path,
    )
    analysis_payload = definition.initialize_launch(ctx, profile)
    launch_manifest["analysis"] = dict(analysis_payload.get("analysis", {}))
    launch_manifest["stage_tasks"] = {
        stage_id: [str(value) for value in values]
        for stage_id, values in dict(analysis_payload.get("stage_tasks", {})).items()
    }
    write_json(launch_manifest_path, launch_manifest)
    initialize_progress(ctx)
    return {
        "analysis_id": definition.analysis_id,
        "launch_dir": str(paths.launch_dir),
        "launch_manifest_path": str(launch_manifest_path),
        "base_snapshot_manifest_path": str(base_snapshot_manifest_path),
        "profile": profile,
        "selected_day_sessions": launch_manifest["analysis"].get("selected_day_session_ids", []),
    }


def load_launch_context(
    *,
    definition: AnalysisDefinition,
    config: BackendConfig,
    launch_dir: Path,
) -> LaunchContext:
    paths = AnalysisLaunchPaths.create(artifact_tables_dir(definition), launch_dir.name)
    if launch_dir.resolve() != paths.launch_dir.resolve():
        paths = AnalysisLaunchPaths(
            analysis_root=paths.analysis_root,
            launches_root=paths.launches_root,
            latest_path=paths.latest_path,
            launch_dir=launch_dir.resolve(),
            base_snapshot_dir=launch_dir.resolve() / "base_snapshot",
            tables_dir=launch_dir.resolve() / "tables",
            reports_dir=launch_dir.resolve() / "reports",
            cache_dir=launch_dir.resolve() / "cache",
            progress_dir=launch_dir.resolve() / "progress",
            stage_progress_dir=launch_dir.resolve() / "progress" / "stages",
            task_progress_dir=launch_dir.resolve() / "progress" / "tasks",
            specs_dir=launch_dir.resolve() / "specs",
            configs_dir=launch_dir.resolve() / "configs",
            hpc_bundle_dir=launch_dir.resolve() / "hpc_bundle",
        )
    launch_manifest_path = paths.configs_dir / "launch_manifest.json"
    launch_manifest = read_json(launch_manifest_path)
    if not launch_manifest:
        raise FileNotFoundError(f"launch manifest not found: {launch_manifest_path}")
    return LaunchContext(
        definition=definition,
        config=config,
        paths=paths,
        launch_manifest=launch_manifest,
        launch_manifest_path=launch_manifest_path,
    )


def initialize_progress(ctx: LaunchContext) -> None:
    for spec in ctx.definition.stage_specs:
        write_json(stage_progress_path(ctx.paths, spec.stage_id), default_stage_progress(ctx, spec))
    refresh_launch_progress(ctx)


def refresh_launch_progress(ctx: LaunchContext) -> dict[str, Any]:
    stage_rows: list[dict[str, Any]] = []
    for spec in ctx.definition.stage_specs:
        row = read_json(stage_progress_path(ctx.paths, spec.stage_id))
        if not row:
            row = default_stage_progress(ctx, spec)
        else:
            row = {**default_stage_progress(ctx, spec), **row}
        stage_rows.append(row)
    statuses = [str(row.get("status", "pending")) for row in stage_rows]
    if statuses and all(status == "completed" for status in statuses):
        overall = "completed"
    elif any(status == "failed" for status in statuses):
        overall = "failed"
    elif any(status == "running" for status in statuses):
        overall = "running"
    else:
        overall = "pending"
    payload = {
        "analysis_id": ctx.definition.analysis_id,
        "launch_label": ctx.launch_manifest["launch_label"],
        "launch_dir": str(ctx.paths.launch_dir),
        "stage_status": overall,
        "completed_stages": int(sum(status == "completed" for status in statuses)),
        "total_stages": len(stage_rows),
        "stages": {
            row["stage_id"]: {
                "status": row.get("status", "pending"),
                "completed_tasks": row.get("completed_tasks", 0),
                "task_total": row.get("task_total", 0),
            }
            for row in stage_rows
            if row
        },
        "last_update_utc": utc_now_iso(),
    }
    write_json(ctx.paths.progress_dir / "launch_progress.json", payload)
    return payload


def run_stage(
    *,
    ctx: LaunchContext,
    stage_id: str,
    task_value: str | None = None,
) -> dict[str, Any]:
    spec = ctx.stage_spec(stage_id)
    ensure_dependencies_completed(ctx, spec)
    stage_tasks = ctx.stage_task_values(stage_id)
    if stage_tasks:
        if task_value is not None:
            return run_stage_task(ctx, spec, task_value)
        results = [run_stage_task(ctx, spec, current) for current in stage_tasks]
        progress = read_json(stage_progress_path(ctx.paths, stage_id))
        progress["status"] = "completed"
        progress["completed_tasks"] = len(stage_tasks)
        progress["failed_tasks"] = 0
        progress["last_update_utc"] = utc_now_iso()
        write_json(stage_progress_path(ctx.paths, stage_id), progress)
        refresh_launch_progress(ctx)
        return {
            "analysis_id": ctx.definition.analysis_id,
            "launch_dir": str(ctx.paths.launch_dir),
            "stage_id": stage_id,
            "task_count": len(stage_tasks),
            "results": results,
        }

    progress_path = stage_progress_path(ctx.paths, stage_id)
    progress = read_json(progress_path) or default_stage_progress(ctx, spec)
    if progress.get("status") == "completed":
        if int(progress.get("failed_tasks", 0)) != 0:
            progress["failed_tasks"] = 0
            progress["last_update_utc"] = utc_now_iso()
            write_json(progress_path, progress)
            refresh_launch_progress(ctx)
        return {
            "analysis_id": ctx.definition.analysis_id,
            "launch_dir": str(ctx.paths.launch_dir),
            "stage_id": stage_id,
            "status": "skipped_cached",
        }
    progress["status"] = "running"
    progress["last_update_utc"] = utc_now_iso()
    write_json(progress_path, progress)
    refresh_launch_progress(ctx)
    try:
        result = ctx.definition.run_stage(ctx, spec, None)
    except Exception:
        progress["status"] = "failed"
        progress["failed_tasks"] = 1
        progress["last_update_utc"] = utc_now_iso()
        write_json(progress_path, progress)
        refresh_launch_progress(ctx)
        raise
    progress["status"] = "completed"
    progress["completed_tasks"] = 1
    progress["failed_tasks"] = 0
    progress["last_update_utc"] = utc_now_iso()
    progress["result"] = result
    write_json(progress_path, progress)
    refresh_launch_progress(ctx)
    return result


def run_stage_task(ctx: LaunchContext, spec: AnalysisStageSpec, task_value: str) -> dict[str, Any]:
    if task_value not in ctx.stage_task_values(spec.stage_id):
        raise ValueError(f"task value is not registered for stage {spec.stage_id}: {task_value}")
    task_progress = stage_task_progress_path(ctx.paths, spec.stage_id, task_value)
    cached = read_json(task_progress)
    if cached.get("status") == "completed":
        return cached

    progress_path = stage_progress_path(ctx.paths, spec.stage_id)
    progress = read_json(progress_path) or default_stage_progress(ctx, spec)
    progress["status"] = "running"
    progress["last_update_utc"] = utc_now_iso()
    write_json(progress_path, progress)
    refresh_launch_progress(ctx)
    try:
        result = ctx.definition.run_stage(ctx, spec, task_value)
    except Exception:
        write_json(
            task_progress,
            {
                "stage_id": spec.stage_id,
                "task_value": task_value,
                "status": "failed",
                "last_update_utc": utc_now_iso(),
            },
        )
        progress["status"] = "failed"
        progress["failed_tasks"] = int(progress.get("failed_tasks", 0)) + 1
        progress["last_update_utc"] = utc_now_iso()
        write_json(progress_path, progress)
        refresh_launch_progress(ctx)
        raise
    write_json(
        task_progress,
        {
            "stage_id": spec.stage_id,
            "task_value": task_value,
            "status": "completed",
            "last_update_utc": utc_now_iso(),
            "result": result,
        },
    )
    completed_tasks = count_completed_stage_tasks(ctx.paths, spec.stage_id)
    progress["completed_tasks"] = completed_tasks
    progress["status"] = "completed" if completed_tasks == len(ctx.stage_task_values(spec.stage_id)) else "running"
    if progress["status"] == "completed":
        progress["failed_tasks"] = 0
    progress["last_update_utc"] = utc_now_iso()
    write_json(progress_path, progress)
    refresh_launch_progress(ctx)
    return result


def count_completed_stage_tasks(paths: AnalysisLaunchPaths, stage_id: str) -> int:
    task_dir = paths.task_progress_dir / stage_id
    if not task_dir.is_dir():
        return 0
    completed = 0
    for path in task_dir.glob("*.json"):
        payload = read_json(path)
        if payload.get("status") == "completed":
            completed += 1
    return completed


def ensure_dependencies_completed(ctx: LaunchContext, spec: AnalysisStageSpec) -> None:
    for dependency in spec.dependencies:
        payload = repair_stage_completion_from_tasks(ctx, dependency)
        if payload.get("status") != "completed":
            raise RuntimeError(f"stage {spec.stage_id} requires completed dependency {dependency}")


def repair_stage_completion_from_tasks(ctx: LaunchContext, stage_id: str) -> dict[str, Any]:
    spec = ctx.stage_spec(stage_id)
    payload_path = stage_progress_path(ctx.paths, stage_id)
    payload = read_json(payload_path) or default_stage_progress(ctx, spec)
    tasks = ctx.stage_task_values(stage_id)
    if tasks:
        completed_tasks = count_completed_stage_tasks(ctx.paths, stage_id)
        if completed_tasks == len(tasks):
            payload["status"] = "completed"
            payload["completed_tasks"] = completed_tasks
            payload["failed_tasks"] = 0
            payload["last_update_utc"] = utc_now_iso()
            write_json(payload_path, payload)
    return payload


def prepare_hpc(ctx: LaunchContext) -> dict[str, Any]:
    prepared: list[dict[str, Any]] = []
    remote_launch_dir = remote_launch_dir_for_paths(ctx.config, ctx.paths.launch_dir)
    remote_config_path = remote_repo_path(ctx.config, ctx.config.config_path)
    for spec in ctx.definition.stage_specs:
        resources = resolve_resource_profile(ctx.config, spec.resource_profile)
        task_values = ctx.stage_task_values(spec.stage_id)
        param_values = task_values or [SINGLE_TASK_SENTINEL]
        bundle_dir = ctx.paths.hpc_bundle_dir / spec.stage_id
        logs_dir = bundle_dir / "logs"
        configs_dir = bundle_dir / "configs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        configs_dir.mkdir(parents=True, exist_ok=True)
        params_path = bundle_dir / "params.tsv"
        params_path.write_text("\n".join(param_values) + "\n", encoding="utf8")
        array_clause = f"0-{len(param_values) - 1}%{resources.array_concurrency}" if param_values else "0-0%1"
        remote_bundle_dir = str(PurePosixPath(remote_launch_dir) / "hpc_bundle" / spec.stage_id)
        job_name = f"{ctx.launch_manifest['launch_label']}-{spec.stage_id}"
        stage_command = [
            "python",
            "-m",
            "stv_analysis.cli",
            "--config",
            shlex.quote(remote_config_path),
            "launch",
            "run-stage",
            ctx.definition.analysis_id,
            "--launch-dir",
            shlex.quote(remote_launch_dir),
            "--stage",
            spec.stage_id,
        ]
        base_command = " ".join(stage_command)
        environment_lines: list[str] = [
            "if ! command -v module >/dev/null 2>&1; then",
            "  for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do",
            '    if [ -r "$module_init" ]; then',
            "      # shellcheck disable=SC1090",
            '      source "$module_init"',
            "      break",
            "    fi",
            "  done",
            "fi",
        ]
        if ctx.config.analysis_hpc.module_loads:
            module_args = " ".join(shlex.quote(value) for value in ctx.config.analysis_hpc.module_loads)
            environment_lines.append(
                f"if command -v module >/dev/null 2>&1; then module load {module_args}; "
                "else echo 'module command unavailable after init; continuing with ambient PATH' >&2; fi"
            )
        environment_lines.extend(
            f"export {key}={shlex.quote(value)}"
            for key, value in sorted(ctx.config.analysis_hpc.environment.items())
        )
        environment_lines.extend(
            [
                'if [[ -n "${R_LIBS_USER:-}" ]]; then mkdir -p "$R_LIBS_USER"; fi',
                'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
                'export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
                'export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
                "export PYTHONUNBUFFERED=1",
            ]
        )
        sbatch = "\n".join(
            [
                "#!/bin/bash",
                f"#SBATCH --job-name={job_name}",
                f"#SBATCH --partition={ctx.config.analysis_hpc.partition}",
                f"#SBATCH --account={ctx.config.analysis_hpc.account}",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
                f"#SBATCH --mem={resources.memory_gb}G",
                f"#SBATCH --time={resources.time_limit}",
                f"#SBATCH --array={array_clause}",
                "#SBATCH --output=logs/%x-%A_%a.out",
                "#SBATCH --error=logs/%x-%A_%a.err",
                "",
                "set -euo pipefail",
                *environment_lines,
                f"source {shlex.quote(ctx.config.analysis_hpc.remote_venv_path)}/bin/activate",
                "LINE=\"$(sed -n \"$((SLURM_ARRAY_TASK_ID + 1))p\" params.tsv)\"",
                f"cd {shlex.quote(str(PurePosixPath(ctx.config.analysis_hpc.remote_repo_root) / 'backend'))}",
                "export PYTHONPATH=\"$PWD/src${PYTHONPATH:+:$PYTHONPATH}\"",
                f"if [[ \"$LINE\" == \"{SINGLE_TASK_SENTINEL}\" ]]; then",
                f"  {base_command}",
                "else",
                f"  {base_command} --task-value \"$LINE\"",
                "fi",
                "",
            ]
        )
        (bundle_dir / "job-array.sbatch").write_text(sbatch, encoding="utf8")
        write_json(
            bundle_dir / "manifest.json",
            {
                "analysis_id": ctx.definition.analysis_id,
                "stage_id": spec.stage_id,
                "launch_dir": str(ctx.paths.launch_dir),
                "remote_launch_dir": remote_launch_dir,
                "remote_bundle_dir": remote_bundle_dir,
                "remote_config_path": remote_config_path,
                "partition": ctx.config.analysis_hpc.partition,
                "account": ctx.config.analysis_hpc.account,
                "params_row_count": len(param_values),
                "task_values": task_values,
                "compute_class": spec.compute_class,
                "resource_profile": spec.resource_profile,
                "resources": {
                    "cpus_per_task": resources.cpus_per_task,
                    "memory_gb": resources.memory_gb,
                    "time_limit": resources.time_limit,
                    "array_concurrency": resources.array_concurrency,
                },
                "module_loads": list(ctx.config.analysis_hpc.module_loads),
                "environment": dict(ctx.config.analysis_hpc.environment),
                "prepared_at_utc": utc_now_iso(),
            },
        )
        write_json(configs_dir / "launch_manifest.json", ctx.launch_manifest)
        prepared.append(
            {
                "stage_id": spec.stage_id,
                "bundle_dir": str(bundle_dir),
                "params_row_count": len(param_values),
                "compute_class": spec.compute_class,
            }
        )
    return {
        "analysis_id": ctx.definition.analysis_id,
        "launch_dir": str(ctx.paths.launch_dir),
        "remote_launch_dir": remote_launch_dir,
        "prepared_stages": prepared,
    }


def submit_hpc(ctx: LaunchContext, *, start_stage: str | None = None) -> dict[str, Any]:
    stage_specs = list(ctx.definition.stage_specs)
    if start_stage is not None:
        stage_ids = [spec.stage_id for spec in stage_specs]
        if start_stage not in stage_ids:
            raise ValueError(f"unknown start stage: {start_stage}")
        stage_specs = stage_specs[stage_ids.index(start_stage) :]

    submissions: list[dict[str, Any]] = []
    submitted_jobs: dict[str, tuple[str, int]] = {}
    for spec in stage_specs:
        bundle_manifest_path = ctx.paths.hpc_bundle_dir / spec.stage_id / "manifest.json"
        bundle_manifest = read_json(bundle_manifest_path)
        if not bundle_manifest:
            raise FileNotFoundError(
                f"HPC bundle for stage {spec.stage_id} is missing. Run `stv-analysis launch prepare-hpc` first."
            )
        remote_bundle_dir = str(bundle_manifest["remote_bundle_dir"])
        dependency_task_ids: list[str] = []
        for dependency_stage_id in spec.dependencies:
            if dependency_stage_id in submitted_jobs:
                dependency_job_id, dependency_task_count = submitted_jobs[dependency_stage_id]
                dependency_task_ids.extend(
                    concrete_array_task_ids(dependency_job_id, dependency_task_count)
                )
                continue
            dependency_progress = repair_stage_completion_from_tasks(ctx, dependency_stage_id)
            if dependency_progress.get("status") != "completed":
                raise RuntimeError(
                    f"cannot submit {spec.stage_id}: dependency {dependency_stage_id} "
                    "is neither completed nor part of this submission"
                )
        dependency = "afterok:" + ":".join(dependency_task_ids) if dependency_task_ids else None
        sbatch_parts = ["sbatch"]
        if dependency:
            sbatch_parts.append(f"--dependency={dependency}")
        sbatch_parts.append("job-array.sbatch")
        remote_command = (
            f"{remote_slurm_bootstrap()}; cd {shlex.quote(remote_bundle_dir)} && {' '.join(sbatch_parts)}"
        )
        completed = run_remote_bash(ctx.config.analysis_hpc.host, remote_command)
        stdout = completed.stdout.strip()
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if match is None:
            raise RuntimeError(f"could not parse sbatch output for stage {spec.stage_id}: {stdout}")
        job_id = match.group(1)
        submission = {
            "stage_id": spec.stage_id,
            "job_id": job_id,
            "dependency": dependency,
            "submitted_at_utc": utc_now_iso(),
            "raw_stdout": stdout,
            "remote_bundle_dir": remote_bundle_dir,
        }
        write_json(ctx.paths.hpc_bundle_dir / spec.stage_id / "submission.json", submission)
        submissions.append(submission)
        submitted_jobs[spec.stage_id] = (job_id, int(bundle_manifest["params_row_count"]))
    return {
        "analysis_id": ctx.definition.analysis_id,
        "launch_dir": str(ctx.paths.launch_dir),
        "submissions": submissions,
    }


def concrete_array_afterok_dependency(job_id: str, task_count: int) -> str:
    return "afterok:" + ":".join(concrete_array_task_ids(job_id, task_count))


def concrete_array_task_ids(job_id: str, task_count: int) -> list[str]:
    task_total = max(1, int(task_count))
    return [f"{job_id}_{task_index}" for task_index in range(task_total)]


def show_status(ctx: LaunchContext) -> dict[str, Any]:
    overall = refresh_launch_progress(ctx)
    stages: list[dict[str, Any]] = []
    for spec in ctx.definition.stage_specs:
        stage_payload = {
            **default_stage_progress(ctx, spec),
            **read_json(stage_progress_path(ctx.paths, spec.stage_id)),
        }
        submission = read_json(ctx.paths.hpc_bundle_dir / spec.stage_id / "submission.json")
        row = {
            "stage_id": spec.stage_id,
            "description": spec.description,
            "compute_class": spec.compute_class,
            "resource_profile": spec.resource_profile,
            "status": stage_payload.get("status", "pending"),
            "completed_tasks": stage_payload.get("completed_tasks", 0),
            "task_total": stage_payload.get("task_total", 0),
        }
        if submission.get("job_id"):
            row["job_id"] = submission["job_id"]
            if row["status"] != "completed":
                try:
                    row["squeue"] = read_remote_squeue(ctx.config.analysis_hpc.host, str(submission["job_id"]))
                    row["sacct"] = read_remote_sacct(ctx.config.analysis_hpc.host, str(submission["job_id"]))
                except subprocess.CalledProcessError as exc:
                    row["remote_status_error"] = (exc.stderr or exc.stdout or str(exc)).strip()
                except subprocess.TimeoutExpired as exc:
                    row["remote_status_error"] = f"remote status command timed out after {exc.timeout}s"
        stages.append(row)
    return {
        "analysis_id": ctx.definition.analysis_id,
        "launch_label": ctx.launch_manifest["launch_label"],
        "launch_dir": str(ctx.paths.launch_dir),
        "overall": overall,
        "stages": stages,
    }


def snapshot_base_bundle(*, source_manifest_path: Path, destination_dir: Path) -> Path:
    manifest = read_json(source_manifest_path)
    if not manifest:
        raise FileNotFoundError(f"analysis-base manifest not found: {source_manifest_path}")
    destination_dir.mkdir(parents=True, exist_ok=True)

    bundle_files = {
        "recorders": destination_dir / "recorders.parquet",
        "segments": destination_dir / "segments.parquet",
        "events": destination_dir / "events.parquet",
        "trace_points_filtered": destination_dir / "trace_points_filtered.parquet",
        "sidecar_index": destination_dir / "sidecar_index.parquet",
    }
    for table_name, destination_path in bundle_files.items():
        source_path = Path(manifest["tables"][table_name]["path"])
        destination_path.write_bytes(source_path.read_bytes())

    sidecar_path = bundle_files["sidecar_index"]
    sidecar_index = pd.read_parquet(sidecar_path)
    if "cached_csv_path" in sidecar_index.columns:
        rewritten: list[str | None] = []
        root = repo_root()
        for value in sidecar_index["cached_csv_path"].tolist():
            if value in (None, "") or pd.isna(value):
                rewritten.append(None)
                continue
            path = Path(str(value))
            if path.is_absolute():
                try:
                    rewritten.append(str(path.resolve().relative_to(root)))
                    continue
                except ValueError:
                    pass
            rewritten.append(str(path))
        sidecar_index["cached_csv_path"] = rewritten
        sidecar_index.to_parquet(sidecar_path, index=False)

    gps_blocker_rule_path = destination_dir / "gps_blocker_rule.json"
    gps_blocker_rule_path.write_text(Path(manifest["gps_blocker_rule_path"]).read_text(encoding="utf8"), encoding="utf8")
    snapshot_manifest = {
        "bundle_name": "analysis_base_snapshot",
        "bundle_version": int(manifest.get("bundle_version", 1)),
        "generated_at_utc": utc_now_iso(),
        "source_manifest_path": str(source_manifest_path),
        "dataset_name": str(manifest["dataset_name"]),
        "timezone": str(manifest["timezone"]),
        "artifact_dir": str(destination_dir),
        "gps_blocker_rule_path": str(gps_blocker_rule_path.relative_to(destination_dir)),
        "tables": {
            table_name: {
                "path": str(path.relative_to(destination_dir)),
                "rows": int(manifest["tables"][table_name].get("rows", 0)),
                "primary_key": list(manifest["tables"][table_name].get("primary_key", [])),
            }
            for table_name, path in bundle_files.items()
        },
    }
    snapshot_manifest_path = destination_dir / "manifest.json"
    write_json(snapshot_manifest_path, snapshot_manifest)
    return snapshot_manifest_path


def snapshot_specs(doc_paths: tuple[str, ...], destination_dir: Path) -> dict[str, Any]:
    root = repo_root()
    files: list[dict[str, Any]] = []
    for relative_doc in doc_paths:
        source_path = (root / relative_doc).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"spec doc not found: {source_path}")
        destination_path = destination_dir / relative_doc
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(source_path.read_bytes())
        files.append(
            {
                "source_relative_path": relative_doc,
                "snapshot_relative_path": str(destination_path.relative_to(destination_dir)),
                "sha256": sha256_file(destination_path),
            }
        )
    payload = {
        "snapshot_created_at_utc": utc_now_iso(),
        "files": files,
    }
    write_json(destination_dir / "manifest.json", payload)
    return payload


def snapshot_analysis_code(definition: AnalysisDefinition) -> dict[str, Any]:
    root = repo_root()
    source_file = inspect.getsourcefile(definition.initialize_launch)
    if source_file is None:
        raise RuntimeError(f"cannot resolve source package for analysis {definition.analysis_id}")
    package_dir = Path(source_file).resolve().parent
    source_paths = sorted(
        path
        for path in package_dir.iterdir()
        if path.is_file() and path.suffix in {".py", ".R"}
    )
    source_paths.extend(
        [
            Path(__file__).resolve(),
            (root / "backend/src/stv_backend/config.py").resolve(),
        ]
    )
    files = []
    for path in sorted(set(source_paths)):
        files.append(
            {
                "source_relative_path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
        )
    return {
        "captured_at_utc": utc_now_iso(),
        "files": files,
    }


def resolve_resource_profile(config: BackendConfig, profile_name: str) -> AnalysisHpcResourceProfile:
    try:
        return config.analysis_hpc.resources[profile_name]
    except KeyError as exc:
        raise KeyError(f"analysis_hpc.resources is missing profile {profile_name}") from exc


def remote_repo_path(config: BackendConfig, local_path: Path) -> str:
    try:
        relative = local_path.resolve().relative_to(repo_root())
    except ValueError:
        relative = default_backend_config_path().resolve().relative_to(repo_root())
    return str(PurePosixPath(config.analysis_hpc.remote_repo_root) / PurePosixPath(relative.as_posix()))


def remote_launch_dir_for_paths(config: BackendConfig, launch_dir: Path) -> str:
    relative = launch_dir.resolve().relative_to(repo_root())
    return str(PurePosixPath(config.analysis_hpc.remote_repo_root) / PurePosixPath(relative.as_posix()))


def refresh_latest_alias(paths: AnalysisLaunchPaths) -> None:
    target = Path("launches") / paths.launch_dir.name
    if paths.latest_path.exists() or paths.latest_path.is_symlink():
        if paths.latest_path.is_dir() and not paths.latest_path.is_symlink():
            raise RuntimeError(f"refusing to overwrite non-symlink latest dir: {paths.latest_path}")
        paths.latest_path.unlink()
    paths.latest_path.symlink_to(target)


def remote_slurm_bootstrap() -> str:
    return (
        'OS_MAJOR="$(. /etc/os-release && printf %s "${VERSION_ID%%.*}")"; '
        'SLURM_BIN="/opt/slurm/el${OS_MAJOR}/current/install/bin"; '
        'if [ ! -d "$SLURM_BIN" ]; then '
        'SLURM_BIN="$(ls -d /opt/slurm/el${OS_MAJOR}/*/install/bin 2>/dev/null | sort | tail -n 1)"; '
        "fi; "
        'if [ -n "$SLURM_BIN" ] && [ -d "$SLURM_BIN" ]; then export PATH="$SLURM_BIN:$PATH"; fi'
    )


def run_remote_bash(host: str, command: str, *, timeout_seconds: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-n",
            "-oBatchMode=yes",
            "-oConnectTimeout=15",
            host,
            f"bash -lc {shlex.quote(command)}",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout_seconds,
    )


def read_remote_squeue(host: str, job_id: str) -> list[dict[str, str]]:
    completed = run_remote_bash(
        host,
        f"{remote_slurm_bootstrap()}; squeue -h -j {shlex.quote(job_id)} -o '%i|%T|%M|%R' || true",
        timeout_seconds=10,
    )
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split("|", maxsplit=3)]
        if len(parts) != 4 or not any(parts):
            continue
        rows.append({"job_id": parts[0], "state": parts[1], "elapsed": parts[2], "reason": parts[3]})
    return rows


def read_remote_sacct(host: str, job_id: str) -> list[dict[str, str]]:
    completed = run_remote_bash(
        host,
        f"{remote_slurm_bootstrap()}; sacct -P -n -j {shlex.quote(job_id)} --format=JobID,State,Elapsed,ExitCode || true",
        timeout_seconds=10,
    )
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split("|", maxsplit=3)]
        if len(parts) != 4 or not any(parts):
            continue
        rows.append({"job_id": parts[0], "state": parts[1], "elapsed": parts[2], "exit_code": parts[3]})
    return rows

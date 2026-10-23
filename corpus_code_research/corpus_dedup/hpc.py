from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd

from .config import DedupConfig, HPCStageConfig
from .constants import HPC_STAGE_NAMES, PROGRESS_STAGE_NAMES
from .features import launch_paths_for_config
from .manifest_utils import write_json
from .paths import LaunchPaths
from .progress import read_progress


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf8"))


def _run_ssh_command(host: str, remote_command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-n",
            "-oBatchMode=yes",
            "-oConnectTimeout=15",
            host,
            remote_command,
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )


def _run_remote_bash(host: str, command: str) -> subprocess.CompletedProcess[str]:
    return _run_ssh_command(host, f"bash -lc {shlex.quote(command)}")


def _remote_launch_dir(config: DedupConfig, paths: LaunchPaths) -> str:
    repo_root = Path.cwd().resolve()
    if not config.hpc.remote_repo_root:
        raise ValueError("HPC config requires `remote_repo_root`.")
    try:
        relative_launch_path = paths.launch_dir.resolve().relative_to(repo_root)
        return str(PurePosixPath(config.hpc.remote_repo_root) / PurePosixPath(relative_launch_path.as_posix()))
    except ValueError:
        artifact_root = config.artifact_root.resolve()
        return str(
            PurePosixPath(config.hpc.remote_repo_root)
            / artifact_root.parent.name
            / artifact_root.name
            / paths.launch_dir.name
        )


def _stage_resources(config: DedupConfig, stage: str) -> HPCStageConfig:
    if stage == "features":
        return config.hpc.features
    if stage == "feature_merge":
        return config.hpc.feature_merge
    if stage == "grid":
        return config.hpc.grid
    if stage == "aggregate":
        return config.hpc.aggregate
    if stage == "exemplars":
        return config.hpc.exemplars
    raise ValueError(f"Unsupported stage: {stage}")


def _stage_bundle_dir(paths: LaunchPaths, stage: str) -> Path:
    return paths.hpc_bundle_dir / stage


def _remote_slurm_path_bootstrap() -> str:
    return (
        'OS_MAJOR="$(. /etc/os-release && printf %s "${VERSION_ID%%.*}")"; '
        'SLURM_BIN="/opt/slurm/el${OS_MAJOR}/current/install/bin"; '
        'if [ ! -d "$SLURM_BIN" ]; then '
        'SLURM_BIN="$(ls -d /opt/slurm/el${OS_MAJOR}/*/install/bin 2>/dev/null | sort | tail -n 1)"; '
        "fi; "
        'if [ -n "$SLURM_BIN" ] && [ -d "$SLURM_BIN" ]; then export PATH="$SLURM_BIN:$PATH"; fi'
    )


def _read_remote_text(host: str, path: str) -> str:
    result = _run_remote_bash(
        host,
        f"test -f {shlex.quote(path)} && cat {shlex.quote(path)} || true",
    )
    return result.stdout.strip()


def _read_remote_json(host: str, path: str) -> dict:
    raw = _read_remote_text(host, path)
    return json.loads(raw) if raw else {}


def _read_remote_logs(host: str, logs_dir: str, log_pattern: str) -> list[str]:
    result = _run_remote_bash(
        host,
        f"find {shlex.quote(logs_dir)} -maxdepth 1 -type f -name {shlex.quote(log_pattern)} -print | sort || true",
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _read_remote_squeue(host: str, job_id: str) -> list[dict[str, str]]:
    result = _run_remote_bash(
        host,
        f"{_remote_slurm_path_bootstrap()}; squeue -h -j {shlex.quote(job_id)} -o '%i|%T|%M|%R' || true",
    )
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split("|", maxsplit=3)]
        if len(parts) != 4 or not any(parts):
            continue
        rows.append(
            {
                "job_id": parts[0],
                "state": parts[1],
                "elapsed": parts[2],
                "reason": parts[3],
            }
        )
    return rows


def _read_remote_sacct(host: str, job_id: str) -> list[dict[str, str]]:
    result = _run_remote_bash(
        host,
        f"{_remote_slurm_path_bootstrap()}; sacct -P -n -j {shlex.quote(job_id)} --format=JobID,State,Elapsed,ExitCode || true",
    )
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split("|", maxsplit=3)]
        if len(parts) != 4 or not any(parts):
            continue
        rows.append(
            {
                "job_id": parts[0],
                "state": parts[1],
                "elapsed": parts[2],
                "exit_code": parts[3],
            }
        )
    return rows


def _stage_params(paths: LaunchPaths, stage: str) -> pd.DataFrame:
    if stage == "features":
        return pd.read_parquet(paths.manifests_dir / "feature_shards.parquet").loc[:, ["shard_id"]].copy()
    if stage == "feature_merge":
        return pd.DataFrame([{"feature_merge": "finalize"}])
    if stage == "grid":
        return pd.read_parquet(paths.manifests_dir / "grid_batches.parquet").loc[:, ["batch_id"]].copy()
    if stage == "aggregate":
        return pd.DataFrame([{"aggregate": "summarize"}])
    if stage == "exemplars":
        return pd.DataFrame([{"exemplars": "export"}])
    raise ValueError(f"Unsupported stage: {stage}")


def _stage_command(stage: str, remote_launch_dir: str) -> str:
    base = "python -m corpus_dedup.cli"
    if stage == "features":
        return base + f" run-features --launch-dir {shlex.quote(remote_launch_dir)} --shard-id \"$TASK_ARG\""
    if stage == "feature_merge":
        return base + f" finalize-features --launch-dir {shlex.quote(remote_launch_dir)}"
    if stage == "grid":
        return base + f" run-grid --launch-dir {shlex.quote(remote_launch_dir)} --batch-id \"$TASK_ARG\""
    if stage == "aggregate":
        return base + f" summarize-run --launch-dir {shlex.quote(remote_launch_dir)}"
    if stage == "exemplars":
        return base + f" export-top-exemplars --launch-dir {shlex.quote(remote_launch_dir)}"
    raise ValueError(f"Unsupported stage: {stage}")


def prepare_hpc(config: DedupConfig, *, stage: str) -> LaunchPaths:
    paths = launch_paths_for_config(config)
    bundle_dir = _stage_bundle_dir(paths, stage)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "configs").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "logs").mkdir(parents=True, exist_ok=True)

    remote_launch_dir = _remote_launch_dir(config, paths)
    resources = _stage_resources(config, stage)
    params_df = _stage_params(paths, stage)
    params_path = bundle_dir / "params.tsv"
    params_df.to_csv(params_path, sep="\t", header=False, index=False)

    task_count = len(params_df.index)
    array_clause = f"0-{task_count - 1}%{resources.array_concurrency}" if task_count > 0 else "0-0%1"
    job_name = f"{paths.launch_dir.name}-{stage}"
    command = _stage_command(stage, remote_launch_dir)
    if not config.hpc.remote_venv_path:
        raise ValueError("HPC config requires `remote_venv_path`.")
    sbatch = "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={config.hpc.partition}",
            f"#SBATCH --account={config.hpc.account}",
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
            "mkdir -p logs",
            "cd \"${SLURM_SUBMIT_DIR:-$(pwd)}\"",
            f"source {shlex.quote(config.hpc.remote_venv_path)}/bin/activate",
            "LINE=\"$(sed -n \"$((SLURM_ARRAY_TASK_ID + 1))p\" params.tsv)\"",
            "IFS=$'\\t' read -r TASK_ARG EXTRA <<< \"$LINE\"",
            f"cd {shlex.quote(config.hpc.remote_repo_root)}",
            command,
            "",
        ]
    )
    (bundle_dir / "job-array.sbatch").write_text(sbatch, encoding="utf8")
    (bundle_dir / "configs" / "launch_config.yaml").write_text(
        (paths.configs_dir / "launch_config.yaml").read_text(encoding="utf8"),
        encoding="utf8",
    )
    write_json(
        bundle_dir / "manifest.json",
        {
            "stage": stage,
            "remote_launch_dir": remote_launch_dir,
            "local_launch_dir": str(paths.launch_dir),
            "params_row_count": task_count,
            "resources": resources.__dict__,
            "job_name": job_name,
        },
    )
    return paths


def submit_hpc(
    config: DedupConfig,
    *,
    stage: str,
    dependency: str | None = None,
    dry_run: bool = False,
) -> str:
    paths = launch_paths_for_config(config)
    bundle_dir = _stage_bundle_dir(paths, stage)
    remote_bundle_dir = str(PurePosixPath(_remote_launch_dir(config, paths)) / "hpc_bundle" / stage)
    sbatch_parts = ["sbatch"]
    if dependency:
        sbatch_parts.append(f"--dependency={dependency}")
    sbatch_parts.append("job-array.sbatch")
    sbatch_command = " ".join(sbatch_parts)
    remote_command = f"bash -lc {shlex.quote(f'{_remote_slurm_path_bootstrap()}; cd {shlex.quote(remote_bundle_dir)} && {sbatch_command}')}"
    if dry_run:
        return remote_command

    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(3):
        try:
            result = _run_ssh_command(config.hpc.host, remote_command)
            stdout = result.stdout.strip()
            match = re.search(r"Submitted batch job (\d+)", stdout)
            job_id = match.group(1) if match else None
            write_json(
                bundle_dir / "submission.json",
                {
                    "stage": stage,
                    "job_id": job_id,
                    "dependency": dependency,
                    "remote_bundle_dir": remote_bundle_dir,
                    "submitted_at_utc": read_progress(paths.progress_dir / "launch_progress.json").get("last_update_utc"),
                    "raw_stdout": stdout,
                },
            )
            return stdout
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if "Socket timed out" not in ((exc.stderr or "") + (exc.stdout or "")) or attempt == 2:
                raise
    assert last_error is not None
    raise last_error


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "NONE"
    return ",".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _compress_paths(paths: list[str], *, limit: int = 3) -> str:
    if not paths:
        return "NONE"
    if len(paths) <= limit:
        return ", ".join(paths)
    remainder = len(paths) - limit
    return ", ".join(paths[:limit]) + f", ... (+{remainder} more)"


def _stage_progress_summary(stage: str, progress: dict) -> str:
    if not progress:
        return "progress=missing"
    parts = [f"progress={progress.get('stage_status', 'unknown')}"]
    if stage == "features":
        parts.append(
            f"completed={progress.get('completed_feature_shards', 0)}/{progress.get('total_feature_shards', 0)}"
        )
        parts.append(f"failed={progress.get('failed_feature_shards', 0)}")
    elif stage == "grid":
        parts.append(f"completed={progress.get('completed_grid_batches', 0)}/{progress.get('total_grid_batches', 0)}")
        parts.append(f"models={progress.get('completed_models', 0)}/{progress.get('total_models', 0)}")
        parts.append(f"failed={progress.get('failed_grid_batches', 0)}")
    elif stage == "feature_merge":
        merged_rows = progress.get("merged_feature_rows")
        if merged_rows is not None:
            parts.append(f"merged_rows={merged_rows}")
    elif stage == "aggregate":
        best_model_id = progress.get("best_model_id")
        if best_model_id:
            parts.append(f"best_model_id={best_model_id}")
    elif stage == "exemplars":
        parts.append(f"exported={progress.get('top_k_exported', 0)}/{progress.get('top_k_requested', 0)}")
    last_update = progress.get("last_update_utc")
    if last_update:
        parts.append(f"last_update_utc={last_update}")
    return " ".join(parts)


def _launch_progress_summary(launch_progress: dict, *, launch_name: str, remote_run_dir: str) -> str:
    if not launch_progress:
        return f"launch={launch_name} remote_run_dir={remote_run_dir} remote_launch_progress=missing"
    return " ".join(
        [
            f"launch={launch_progress.get('launch_id', launch_name)}",
            f"remote_run_dir={remote_run_dir}",
            f"current_stage={launch_progress.get('current_stage', 'unknown')}",
            f"status={launch_progress.get('stage_status', 'unknown')}",
            f"features={launch_progress.get('completed_feature_shards', 0)}/{launch_progress.get('total_feature_shards', 0)}",
            f"grid={launch_progress.get('completed_grid_batches', 0)}/{launch_progress.get('total_grid_batches', 0)}",
            f"models={launch_progress.get('completed_models', 0)}/{launch_progress.get('total_models', 0)}",
            f"top_k_exported={launch_progress.get('top_k_exported', 0)}",
            f"top_k_requested={launch_progress.get('top_k_requested', 0)}",
            f"last_update_utc={launch_progress.get('last_update_utc', 'unknown')}",
        ]
    )


def _top_level_sacct_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("job_id") and "." not in row["job_id"]]


def _scheduler_state_summary(squeue_rows: list[dict[str, str]], sacct_rows: list[dict[str, str]]) -> str:
    states: list[str] = []
    for row in squeue_rows:
        state = row.get("state")
        if state and state not in states:
            states.append(state)
    if states:
        return ",".join(states)
    for row in _top_level_sacct_rows(sacct_rows):
        state = row.get("state")
        if state and state not in states:
            states.append(state)
    return ",".join(states) if states else "UNKNOWN"


def show_hpc_status(
    config: DedupConfig,
    *,
    stage: str | None = None,
    job_id: str | None = None,
    remote_run_dir: str | None = None,
) -> str:
    paths = launch_paths_for_config(config)
    resolved_remote_run_dir = remote_run_dir or _remote_launch_dir(config, paths)
    remote_launch_progress_path = str(PurePosixPath(resolved_remote_run_dir) / "progress" / "launch_progress.json")
    remote_launch_progress = _read_remote_json(config.hpc.host, remote_launch_progress_path)
    stages = [stage] if stage else list(HPC_STAGE_NAMES)
    lines: list[str] = [
        _launch_progress_summary(
            remote_launch_progress,
            launch_name=paths.launch_dir.name,
            remote_run_dir=resolved_remote_run_dir,
        ),
        f"remote_launch_progress_path={remote_launch_progress_path}",
        "",
    ]
    for stage_name in stages:
        submission = _load_json(_stage_bundle_dir(paths, stage_name) / "submission.json")
        stage_job_id = job_id or submission.get("job_id")
        remote_bundle_dir = str(PurePosixPath(resolved_remote_run_dir) / "hpc_bundle" / stage_name)
        remote_progress_path = str(PurePosixPath(resolved_remote_run_dir) / "progress" / f"{stage_name}_progress.json")
        stage_progress = _read_remote_json(config.hpc.host, remote_progress_path)
        remote_logs_dir = str(PurePosixPath(remote_bundle_dir) / "logs")
        remote_log_pattern = f"*{stage_job_id or ''}*"
        squeue_rows: list[dict[str, str]] = []
        sacct_rows: list[dict[str, str]] = []
        log_paths: list[str] = []
        if stage_job_id:
            squeue_rows = _read_remote_squeue(config.hpc.host, str(stage_job_id))
            sacct_rows = _read_remote_sacct(config.hpc.host, str(stage_job_id))
            log_paths = _read_remote_logs(config.hpc.host, remote_logs_dir, remote_log_pattern)
        scheduler_summary = _scheduler_state_summary(squeue_rows, sacct_rows) if stage_job_id else "UNSUBMITTED"
        top_level_sacct = _top_level_sacct_rows(sacct_rows)
        state_counts = Counter(row["state"] for row in top_level_sacct if row.get("state"))
        failed_tasks = [
            row["job_id"]
            for row in top_level_sacct
            if row.get("state")
            and row["state"] not in {"COMPLETED", "PENDING", "RUNNING", "COMPLETING"}
        ]
        headline_parts = [f"[{stage_name}]"]
        if stage_job_id:
            headline_parts.append(f"job_id={stage_job_id}")
        headline_parts.append(f"scheduler={scheduler_summary}")
        headline_parts.append(_stage_progress_summary(stage_name, stage_progress))
        lines.append(" ".join(headline_parts))
        lines.append(f"remote_bundle_dir={remote_bundle_dir}")
        lines.append(f"remote_progress_path={remote_progress_path}")
        lines.append(f"remote_logs_dir={remote_logs_dir}")
        lines.append(f"sacct_states={_format_counter(state_counts)}")
        lines.append(f"failed_tasks={','.join(failed_tasks[:10]) if failed_tasks else 'NONE'}")
        lines.append(f"log_files={len(log_paths)}")
        lines.append(f"log_paths={_compress_paths(log_paths)}")
        lines.append("")
    return "\n".join(lines).strip()


def _feature_task_lines(paths: LaunchPaths, config: DedupConfig) -> list[str]:
    manifest_path = paths.manifests_dir / "feature_shards.parquet"
    if not manifest_path.is_file():
        return ["features: manifest missing"]
    shards_df = pd.read_parquet(manifest_path)
    feature_config_id = read_progress(paths.progress_dir / "features_progress.json").get(
        "feature_config_id",
        None,
    )
    if feature_config_id is None:
        feature_config_id = "unknown"
    cache_dir = paths.cache_dir / "features" / str(feature_config_id)
    lines: list[str] = []
    for row in shards_df.to_dict(orient="records"):
        shard_id = int(row["shard_id"])
        status = "completed" if (cache_dir / f"shard_{shard_id:04d}.parquet").is_file() else "missing"
        lines.append(f"feature_shard={shard_id} status={status} hums={int(row['hum_count'])} wavs={int(row['wav_count'])}")
    return lines


def _grid_task_lines(paths: LaunchPaths, config: DedupConfig) -> list[str]:
    batches_path = paths.manifests_dir / "grid_batches.parquet"
    models_path = paths.manifests_dir / "grid_models.parquet"
    if not batches_path.is_file() or not models_path.is_file():
        return ["grid: manifest missing"]
    batches_df = pd.read_parquet(batches_path)
    models_df = pd.read_parquet(models_path)
    model_lookup = {str(row["dedup_model_id"]): row for row in models_df.to_dict(orient="records")}
    lines: list[str] = []
    for batch in batches_df.to_dict(orient="records"):
        model_ids = batch["model_ids"]
        if isinstance(model_ids, str):
            model_ids = json.loads(model_ids)
        completed = 0
        for model_id in model_ids:
            summary_path = paths.cache_dir / "model_runs" / str(model_id) / "summary.json"
            if _load_json(summary_path):
                completed += 1
        lines.append(f"grid_batch={batch['batch_id']} status={completed}/{len(model_ids)}")
    return lines


def format_progress(paths: LaunchPaths, *, stage: str | None = None, detail: str = "summary") -> str:
    if detail == "json":
        target = paths.progress_dir / ("launch_progress.json" if stage is None else f"{stage}_progress.json")
        return target.read_text(encoding="utf8") if target.is_file() else "{}\n"

    launch_progress = read_progress(paths.progress_dir / "launch_progress.json")
    stages = list(PROGRESS_STAGE_NAMES) if stage is None else [stage]
    if detail == "summary":
        return (
            f"launch={launch_progress.get('launch_id', paths.launch_dir.name)} "
            f"stage={launch_progress.get('current_stage', 'unknown')} "
            f"status={launch_progress.get('stage_status', 'unknown')} "
            f"features={launch_progress.get('completed_feature_shards', 0)}/{launch_progress.get('total_feature_shards', 0)} "
            f"grid={launch_progress.get('completed_grid_batches', 0)}/{launch_progress.get('total_grid_batches', 0)} "
            f"models={launch_progress.get('completed_models', 0)}/{launch_progress.get('total_models', 0)} "
            f"top_k_exported={launch_progress.get('top_k_exported', 0)}"
        )

    if detail == "stage":
        lines = []
        for stage_name in stages:
            progress = read_progress(paths.progress_dir / f"{stage_name}_progress.json")
            lines.append(f"{stage_name}: {progress if progress else 'missing'}")
        return "\n".join(lines)

    lines: list[str] = []
    if stage in (None, "features"):
        lines.extend(_feature_task_lines(paths, load_config_from_launch(paths)))
    if stage in (None, "feature_merge", "aggregate", "exemplars"):
        for stage_name in [value for value in stages if value in {"feature_merge", "aggregate", "exemplars"}]:
            lines.append(f"{stage_name}: {read_progress(paths.progress_dir / f'{stage_name}_progress.json') or 'missing'}")
    if stage in (None, "grid"):
        lines.extend(_grid_task_lines(paths, load_config_from_launch(paths)))
    return "\n".join(lines)


def load_config_from_launch(paths: LaunchPaths) -> DedupConfig:
    from .config import load_config

    return load_config(paths.configs_dir / "launch_config.yaml")

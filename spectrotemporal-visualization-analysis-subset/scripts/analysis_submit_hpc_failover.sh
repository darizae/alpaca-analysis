#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_analysis_common.sh"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <analysis-id> <launch-dir> <target-stage-id> <primary-job-id> [downstream-job-id ...]" >&2
  exit 1
fi

ANALYSIS_ID="$1"
LAUNCH_DIR="$2"
TARGET_STAGE_ID="$3"
PRIMARY_JOB_ID="$4"
shift 4

DOWNSTREAM_JOB_IDS=("$@")
CONFIG_PATH="${CONFIG_PATH:-$DEFAULT_CONFIG_PATH}"

RESOLVED=()
while IFS= read -r line; do
  RESOLVED+=("$line")
done < <("$PYTHON_BIN" - "$CONFIG_PATH" "$LAUNCH_DIR" "$TARGET_STAGE_ID" <<'PY'
import sys
from pathlib import Path

from stv_analysis.framework import remote_launch_dir_for_paths, remote_repo_path
from stv_backend.config import load_config

config = load_config(Path(sys.argv[1]))
launch_dir = Path(sys.argv[2]).resolve()
target_stage_id = sys.argv[3]
manifest = __import__("json").loads(
    (launch_dir / "configs" / "launch_manifest.json").read_text(encoding="utf8")
)
stage_ids = [str(item["stage_id"]) for item in manifest["stage_specs"]]
if target_stage_id not in stage_ids:
    raise SystemExit(f"unknown failover target stage: {target_stage_id}")
target_spec = next(item for item in manifest["stage_specs"] if str(item["stage_id"]) == target_stage_id)
resource_profile = str(target_spec.get("resource_profile", "light"))
resources = config.analysis_hpc.resources[resource_profile]
print(config.analysis_hpc.host)
print(remote_launch_dir_for_paths(config, launch_dir))
print(config.analysis_hpc.remote_repo_root)
print(config.analysis_hpc.remote_venv_path)
print(remote_repo_path(config, Path(sys.argv[1])))
print(launch_dir)
print(f"PARTITION\t{config.analysis_hpc.partition}")
print(f"ACCOUNT\t{config.analysis_hpc.account}")
print(f"RESOURCES\t{resources.cpus_per_task}\t{resources.memory_gb}\t{resources.time_limit}")
for module_name in config.analysis_hpc.module_loads:
    print(f"MODULE\t{module_name}")
for key, value in sorted(config.analysis_hpc.environment.items()):
    print(f"ENV\t{key}\t{value}")
print("STAGES\t" + "\t".join(stage_ids[stage_ids.index(target_stage_id) + 1 :]))
PY
)

HOST="${HOST_OVERRIDE:-${RESOLVED[0]}}"
REMOTE_LAUNCH_DIR="${REMOTE_LAUNCH_DIR_OVERRIDE:-${RESOLVED[1]}}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT_OVERRIDE:-${RESOLVED[2]}}"
REMOTE_VENV_PATH="${REMOTE_VENV_PATH_OVERRIDE:-${RESOLVED[3]}}"
REMOTE_CONFIG_PATH="${REMOTE_CONFIG_PATH_OVERRIDE:-${RESOLVED[4]}}"
LOCAL_LAUNCH_DIR="${RESOLVED[5]}"

MODULE_LOADS=()
ENVIRONMENT_ROWS=()
DOWNSTREAM_STAGE_IDS=()
PARTITION="compute"
ACCOUNT="alpaca-project"
FAILOVER_CPUS_PER_TASK="2"
FAILOVER_MEMORY_GB="32"
FAILOVER_TIME_LIMIT="12:00:00"
for row in "${RESOLVED[@]:6}"; do
  if [[ "$row" == PARTITION$'\t'* ]]; then
    PARTITION="${row#*$'\t'}"
  elif [[ "$row" == ACCOUNT$'\t'* ]]; then
    ACCOUNT="${row#*$'\t'}"
  elif [[ "$row" == RESOURCES$'\t'* ]]; then
    IFS=$'\t' read -r FAILOVER_CPUS_PER_TASK FAILOVER_MEMORY_GB FAILOVER_TIME_LIMIT <<< "${row#*$'\t'}"
  elif [[ "$row" == MODULE$'\t'* ]]; then
    MODULE_LOADS+=("${row#*$'\t'}")
  elif [[ "$row" == ENV$'\t'* ]]; then
    ENVIRONMENT_ROWS+=("${row#*$'\t'}")
  elif [[ "$row" == STAGES$'\t'* ]]; then
    IFS=$'\t' read -r -a DOWNSTREAM_STAGE_IDS <<< "${row#*$'\t'}"
  fi
done

if [[ ${#DOWNSTREAM_STAGE_IDS[@]} -ne ${#DOWNSTREAM_JOB_IDS[@]} ]]; then
  echo "Failover downstream stage/job mismatch: ${#DOWNSTREAM_STAGE_IDS[@]} stages, ${#DOWNSTREAM_JOB_IDS[@]} jobs" >&2
  exit 1
fi

FAILOVER_STAGE_ID="failover_${TARGET_STAGE_ID}"
REMOTE_BUNDLE_DIR="${REMOTE_LAUNCH_DIR}/hpc_bundle/${FAILOVER_STAGE_ID}"
LOCAL_BUNDLE_DIR="${LOCAL_LAUNCH_DIR}/hpc_bundle/${FAILOVER_STAGE_ID}"
mkdir -p "$LOCAL_BUNDLE_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

read -r -d '' SLURM_BOOTSTRAP <<'EOF' || true
OS_MAJOR="$(. /etc/os-release && printf %s "${VERSION_ID%%.*}")"
SLURM_BIN="/opt/slurm/el${OS_MAJOR}/current/install/bin"
if [ ! -d "$SLURM_BIN" ]; then
  SLURM_BIN="$(ls -d /opt/slurm/el${OS_MAJOR}/*/install/bin 2>/dev/null | sort | tail -n 1)"
fi
if [ -n "$SLURM_BIN" ] && [ -d "$SLURM_BIN" ]; then
  export PATH="$SLURM_BIN:$PATH"
fi
EOF

DOWNSTREAM_DECL="DOWNSTREAM_JOB_IDS=()"
if [[ ${#DOWNSTREAM_JOB_IDS[@]} -gt 0 ]]; then
  printf -v DOWNSTREAM_DECL 'DOWNSTREAM_JOB_IDS=(%s)' "$(printf '%q ' "${DOWNSTREAM_JOB_IDS[@]}")"
fi

RUNTIME_BOOTSTRAP=""
if [[ ${#MODULE_LOADS[@]} -gt 0 ]]; then
  printf -v MODULE_ARGS '%q ' "${MODULE_LOADS[@]}"
  RUNTIME_BOOTSTRAP="module load ${MODULE_ARGS}"
fi
for row in "${ENVIRONMENT_ROWS[@]}"; do
  key="${row%%$'\t'*}"
  value="${row#*$'\t'}"
  printf -v value_quoted '%q' "$value"
  RUNTIME_BOOTSTRAP+=$'\n'"export ${key}=${value_quoted}"
done

STAGE_RUN_COMMANDS=""
for stage_id in "$TARGET_STAGE_ID" "${DOWNSTREAM_STAGE_IDS[@]}"; do
  printf -v stage_quoted '%q' "$stage_id"
  STAGE_RUN_COMMANDS+="python -m stv_analysis.cli --config $(printf '%q' "$REMOTE_CONFIG_PATH") launch run-stage $(printf '%q' "$ANALYSIS_ID") --launch-dir $(printf '%q' "$REMOTE_LAUNCH_DIR") --stage ${stage_quoted}"$'\n'
done

cat > "${TMP_DIR}/run_failover.sh" <<EOF
#!/bin/bash
set -euo pipefail

${SLURM_BOOTSTRAP}

PRIMARY_JOB_ID="${PRIMARY_JOB_ID}"
${DOWNSTREAM_DECL}

mapfile -t PRIMARY_TASK_STATES < <(
  sacct -X -P -n -j "\${PRIMARY_JOB_ID}" --format=JobID,State |
    awk -F'|' -v job="\${PRIMARY_JOB_ID}" '\$1 ~ ("^" job "_[0-9]+$") {print \$2}'
)
if [[ \${#PRIMARY_TASK_STATES[@]} -eq 0 ]]; then
  echo "No completed accounting records found for primary array \${PRIMARY_JOB_ID}" >&2
  exit 1
fi
PRIMARY_COMPLETED=true
for state in "\${PRIMARY_TASK_STATES[@]}"; do
  state="\${state%%+*}"
  if [[ "\${state}" != "COMPLETED" ]]; then
    PRIMARY_COMPLETED=false
    break
  fi
done

if [[ "\${PRIMARY_COMPLETED}" == "true" ]]; then
  exit 0
fi

if [[ \${#DOWNSTREAM_JOB_IDS[@]} -gt 0 ]]; then
  scancel "\${DOWNSTREAM_JOB_IDS[@]}" || true
fi

${RUNTIME_BOOTSTRAP}
source "${REMOTE_VENV_PATH}/bin/activate"
cd "${REMOTE_REPO_ROOT}/backend"
export PYTHONPATH="\$PWD/src\${PYTHONPATH:+:\$PYTHONPATH}"

${STAGE_RUN_COMMANDS}
EOF

cat > "${TMP_DIR}/failover.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=${ANALYSIS_ID}-failover
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${FAILOVER_CPUS_PER_TASK}
#SBATCH --mem=${FAILOVER_MEMORY_GB}G
#SBATCH --time=${FAILOVER_TIME_LIMIT}
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
bash run_failover.sh
EOF

chmod +x "${TMP_DIR}/run_failover.sh" "${TMP_DIR}/failover.sbatch"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "mkdir -p $(printf '%q' "${REMOTE_BUNDLE_DIR}/logs")"
rsync -avP "${TMP_DIR}/" "${HOST}:${REMOTE_BUNDLE_DIR}/" >/dev/null

SUBMIT_OUTPUT="$(
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
    "bash -lc $(printf '%q' "${SLURM_BOOTSTRAP}; cd ${REMOTE_BUNDLE_DIR} && sbatch --dependency=afterany:${PRIMARY_JOB_ID} failover.sbatch")"
)"

FAILOVER_JOB_ID="$("$PYTHON_BIN" - "$SUBMIT_OUTPUT" <<'PY'
import re
import sys

text = sys.argv[1].replace("\r", "")
matches = re.findall(r"Submitted batch job (\d+)", text)
print(matches[-1] if matches else "")
PY
)"
if [[ -z "$FAILOVER_JOB_ID" ]]; then
  echo "Could not parse failover sbatch output: $SUBMIT_OUTPUT" >&2
  exit 1
fi

"$PYTHON_BIN" - "$LOCAL_BUNDLE_DIR" "$TARGET_STAGE_ID" "$PRIMARY_JOB_ID" "$FAILOVER_JOB_ID" "$SUBMIT_OUTPUT" "${DOWNSTREAM_JOB_IDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

bundle_dir = Path(sys.argv[1])
target_stage_id = sys.argv[2]
primary_job_id = sys.argv[3]
failover_job_id = sys.argv[4]
raw_stdout = sys.argv[5]
downstream_job_ids = sys.argv[6:]
payload = {
    "stage_id": f"failover_{target_stage_id}",
    "target_stage_id": target_stage_id,
    "primary_job_id": primary_job_id,
    "job_id": failover_job_id,
    "dependency": f"afterany:{primary_job_id}",
    "downstream_job_ids": downstream_job_ids,
    "raw_stdout": raw_stdout,
}
path = bundle_dir / "submission.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
print(json.dumps(payload, indent=2))
PY

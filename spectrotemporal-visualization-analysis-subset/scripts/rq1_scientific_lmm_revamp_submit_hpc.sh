#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_analysis_common.sh"

CONFIG_PATH="${1:-$DEFAULT_CONFIG_PATH}"

ANALYSIS_REL="backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp"
THRESHOLD_ROOT_REL="${ANALYSIS_REL}/threshold_supplement"
PRIMARY_OUTPUT_ADD_REL="${ANALYSIS_REL}/h1_model_outputs"
THRESHOLD_OUTPUT_ADD_REL="${THRESHOLD_ROOT_REL}/r_outputs_threshold_global_standard_matching_standard_model_nloptwrap"
PRIMARY_SMOKE_ADD_REL="${PRIMARY_OUTPUT_ADD_REL}_smoke"
THRESHOLD_SMOKE_ADD_REL="${THRESHOLD_OUTPUT_ADD_REL}_smoke"
PRIMARY_INPUT_REL="${ANALYSIS_REL}/data/h1_contact_reinforcement_model_input.csv"
THRESHOLD_RUN_CONFIG_REL="${THRESHOLD_ROOT_REL}/tables/rq1_threshold_scientific_lmm_run_config.csv"
R_SCRIPT_REL="backend/src/stv_analysis/rq1_post_hum_cohesion/scientific_lmm_guide_workflow.R"
THRESHOLD_MATERIALIZE_REL="backend/src/stv_analysis/rq1_post_hum_cohesion/scientific_lmm_threshold_materialize.py"
BUNDLE_ID="rq1_predictor_threshold_revamp_$(date -u +%Y%m%dT%H%M%SZ)"
BRANCH_NAME="$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || true)"
KEEP_EXISTING_R_OUTPUTS="${KEEP_EXISTING_R_OUTPUTS:-1}"

RESOLVED=()
while IFS= read -r line; do
  RESOLVED+=("$line")
done < <("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH"
import sys
from pathlib import Path

from stv_backend.config import load_config

config = load_config(Path(sys.argv[1]))
print(config.analysis_hpc.host)
print(config.analysis_hpc.partition)
print(config.analysis_hpc.account)
print(config.analysis_hpc.remote_repo_root)
print(config.analysis_hpc.remote_venv_path)
for module_name in config.analysis_hpc.module_loads:
    print(f"MODULE\t{module_name}")
for key, value in sorted(config.analysis_hpc.environment.items()):
    print(f"ENV\t{key}\t{value}")
PY
)

HOST="${HOST_OVERRIDE:-${RESOLVED[0]}}"
PARTITION="${PARTITION_OVERRIDE:-${RESOLVED[1]}}"
ACCOUNT="${ACCOUNT_OVERRIDE:-${RESOLVED[2]}}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT_OVERRIDE:-${RESOLVED[3]}}"
REMOTE_VENV_PATH="${REMOTE_VENV_PATH_OVERRIDE:-${RESOLVED[4]}}"
REMOTE_ANALYSIS_DIR="${REMOTE_REPO_ROOT}/${ANALYSIS_REL}"
REMOTE_THRESHOLD_ROOT="${REMOTE_REPO_ROOT}/${THRESHOLD_ROOT_REL}"
REMOTE_BUNDLE_DIR="${REMOTE_ANALYSIS_DIR}/hpc_bundle/${BUNDLE_ID}"
LOCAL_ANALYSIS_DIR="${REPO_ROOT}/${ANALYSIS_REL}"
LOCAL_BUNDLE_DIR="${LOCAL_ANALYSIS_DIR}/hpc_bundle/${BUNDLE_ID}"

MODULE_LOADS=()
ENVIRONMENT_ROWS=()
for row in "${RESOLVED[@]:5}"; do
  if [[ "$row" == MODULE$'\t'* ]]; then
    MODULE_LOADS+=("${row#*$'\t'}")
  elif [[ "$row" == ENV$'\t'* ]]; then
    ENVIRONMENT_ROWS+=("${row#*$'\t'}")
  fi
done

mkdir -p "$LOCAL_BUNDLE_DIR"
"${SCRIPT_DIR}/analysis_sync_repo_to_hpc.sh" "$CONFIG_PATH"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
  "rm -f $(printf '%q' "${REMOTE_REPO_ROOT}/scripts/rq1_threshold_scientific_lmm_submit_hpc.sh")"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
  "mkdir -p $(printf '%q' "$REMOTE_ANALYSIS_DIR/data") $(printf '%q' "$REMOTE_ANALYSIS_DIR/tables") $(printf '%q' "$REMOTE_ANALYSIS_DIR/reports") $(printf '%q' "$REMOTE_THRESHOLD_ROOT") $(printf '%q' "$REMOTE_BUNDLE_DIR/logs") $(printf '%q' "${REMOTE_REPO_ROOT}/backend/data/processed/analysis/base")"

rsync -avP --delete "${REPO_ROOT}/backend/data/processed/analysis/base/" \
  "${HOST}:${REMOTE_REPO_ROOT}/backend/data/processed/analysis/base/" >/dev/null
if [[ ! -f "${REPO_ROOT}/${PRIMARY_INPUT_REL}" ]]; then
  echo "Missing retained primary model input: ${REPO_ROOT}/${PRIMARY_INPUT_REL}" >&2
  exit 1
fi
rsync -avP --delete "${LOCAL_ANALYSIS_DIR}/data/" \
  "${HOST}:${REMOTE_ANALYSIS_DIR}/data/" >/dev/null
if [[ -d "${LOCAL_ANALYSIS_DIR}/threshold_diagnostics" ]]; then
  rsync -avP --delete "${LOCAL_ANALYSIS_DIR}/threshold_diagnostics/" \
    "${HOST}:${REMOTE_ANALYSIS_DIR}/threshold_diagnostics/" >/dev/null
fi

if [[ "$KEEP_EXISTING_R_OUTPUTS" != "1" ]]; then
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
    "rm -rf $(printf '%q' "${REMOTE_REPO_ROOT}/${PRIMARY_OUTPUT_ADD_REL}") $(printf '%q' "${REMOTE_REPO_ROOT}/${THRESHOLD_OUTPUT_ADD_REL}") $(printf '%q' "${REMOTE_REPO_ROOT}/${PRIMARY_SMOKE_ADD_REL}") $(printf '%q' "${REMOTE_REPO_ROOT}/${THRESHOLD_SMOKE_ADD_REL}")"
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cat > "${TMP_DIR}/primary_additive_run_ids.txt" <<'EOF'
h1_30s
h1_60s
h1_120s
EOF

cat > "${TMP_DIR}/threshold_additive_run_ids.txt" <<'EOF'
threshold_49m_h1_30s
threshold_49m_h1_60s
threshold_49m_h1_120s
EOF

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

RUNTIME_BOOTSTRAP='
if ! command -v module >/dev/null 2>&1; then
  for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
    if [ -r "$module_init" ]; then
      # shellcheck disable=SC1090
      source "$module_init"
      break
    fi
  done
fi'
if [[ ${#MODULE_LOADS[@]} -gt 0 ]]; then
  printf -v MODULE_ARGS '%q ' "${MODULE_LOADS[@]}"
  RUNTIME_BOOTSTRAP+=$'\n'"if command -v module >/dev/null 2>&1; then module load ${MODULE_ARGS} || echo 'module load failed; continuing with ambient PATH/fallbacks' >&2; else echo 'module command unavailable after init; continuing with ambient PATH/fallbacks' >&2; fi"
fi
RUNTIME_BOOTSTRAP+=$'\n''if ! command -v Rscript >/dev/null 2>&1; then
  for rscript_candidate in /sw/rev/*/*/linux-rocky8-*/gcc-13.2.0/r-4.5.2-*/bin/Rscript /sw/rev/*/*/linux-rocky8-*/gcc-*/r-4.5.2-*/bin/Rscript; do
    if [ -x "$rscript_candidate" ]; then
      export PATH="$(dirname "$rscript_candidate"):$PATH"
      break
    fi
  done
fi
if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript is unavailable after module load/fallback lookup" >&2
  exit 127
fi'
for row in "${ENVIRONMENT_ROWS[@]}"; do
  key="${row%%$'\t'*}"
  value="${row#*$'\t'}"
  printf -v value_quoted '%q' "$value"
  RUNTIME_BOOTSTRAP+=$'\n'"export ${key}=${value_quoted}"
done
RUNTIME_BOOTSTRAP+=$'\n''mkdir -p "${R_LIBS_USER:-$HOME/R/library}"'
RUNTIME_BOOTSTRAP+=$'\n''if command -v module >/dev/null 2>&1; then module load python/3.11.9 || echo '\''python module load failed; continuing with ambient Python'\'' >&2; fi'
RUNTIME_BOOTSTRAP+=$'\n'"source $(printf '%q' "${REMOTE_VENV_PATH}/bin/activate")"
printf -v REMOTE_SITE_PACKAGES '%q' "${REMOTE_VENV_PATH}/lib/python3.11/site-packages"
printf -v REMOTE_BACKEND_SRC '%q' "${REMOTE_REPO_ROOT}/backend/src"
RUNTIME_BOOTSTRAP+=$'\n'"export PYTHONPATH=${REMOTE_BACKEND_SRC}:\${PYTHONPATH:-}"
RUNTIME_BOOTSTRAP+=$'\n'"if [ -d ${REMOTE_SITE_PACKAGES} ]; then export PYTHONPATH=${REMOTE_SITE_PACKAGES}:\${PYTHONPATH}; fi"
RUNTIME_BOOTSTRAP+=$'\n''export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"'
RUNTIME_BOOTSTRAP+=$'\n''export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"'
RUNTIME_BOOTSTRAP+=$'\n''export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"'

cat > "${TMP_DIR}/install_r_packages.sh" <<EOF
#!/bin/bash
set -euo pipefail

${SLURM_BOOTSTRAP}
${RUNTIME_BOOTSTRAP}

Rscript - <<'RSCRIPT'
ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = c(CRAN = "https://cloud.r-project.org"))
  }
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste("R package is required but unavailable:", pkg))
  }
}
for (pkg in c("lme4", "lmerTest", "ggplot2", "nloptr")) {
  ensure_package(pkg)
}
RSCRIPT
EOF

cat > "${TMP_DIR}/materialize_threshold.sh" <<EOF
#!/bin/bash
set -euo pipefail

${SLURM_BOOTSTRAP}
${RUNTIME_BOOTSTRAP}

cd "$(printf '%q' "$REMOTE_REPO_ROOT")"
python "$(printf '%q' "$THRESHOLD_MATERIALIZE_REL")" \\
  --repo-root "$(printf '%q' "$REMOTE_REPO_ROOT")" \\
  --primary-input "$(printf '%q' "$PRIMARY_INPUT_REL")" \\
  --output-dir "$(printf '%q' "$THRESHOLD_ROOT_REL")"

python - <<'PY'
from pathlib import Path
import pandas as pd

run_config = Path("$(printf '%q' "$THRESHOLD_RUN_CONFIG_REL")")
if not run_config.is_file():
    raise SystemExit(f"missing threshold run config: {run_config}")
df = pd.read_csv(run_config)
if len(df.index) != 3:
    raise SystemExit(f"expected 3 threshold run config rows, found {len(df.index)}")
thresholds = {round(float(value), 6) for value in df["threshold_m"].dropna().unique()}
if thresholds != {49.358079}:
    raise SystemExit(f"unexpected threshold values: {sorted(thresholds)}")
print(df[["model_id", "threshold_m", "window_s", "data_path"]].to_string(index=False))
PY
EOF

make_r_smoke() {
  local path="$1" variant="$2" output_rel="$3" only_run="$4" primary_model_id="$5" run_config_arg="$6"
  local run_config_append=""
  if [[ -n "$run_config_arg" ]]; then
    run_config_append="R_CMD+=(${run_config_arg})"
  fi
  cat > "$path" <<EOF
#!/bin/bash
set -euo pipefail

${SLURM_BOOTSTRAP}
${RUNTIME_BOOTSTRAP}

cd "$(printf '%q' "$REMOTE_REPO_ROOT")"
R_CMD=(
  Rscript "$(printf '%q' "$R_SCRIPT_REL")"
  --mode smoke
  --stability none
  --predictor-set rq1_predictor_revamp
  --fixed-effect-variant "$variant"
  --optimizer nloptwrap
  --optimizer-checkpoint-every 25
  --max-rows-per-run 3000
)
${run_config_append}
R_CMD+=(
  --primary-model-id "$primary_model_id"
  --only-run "$only_run"
  --skip-aggregate
  --output-dir "$(printf '%q' "$output_rel")"
)
"\${R_CMD[@]}"
EOF
}

make_r_run_one() {
  local path="$1" variant="$2" output_rel="$3" run_ids_file="$4" run_config_arg="$5" aic_arg="$6"
  local run_config_append=""
  local aic_append=""
  if [[ -n "$run_config_arg" ]]; then
    run_config_append="R_CMD+=(${run_config_arg})"
  fi
  if [[ -n "$aic_arg" ]]; then
    aic_append="R_CMD+=(${aic_arg})"
  fi
  cat > "$path" <<EOF
#!/bin/bash
set -euo pipefail

${SLURM_BOOTSTRAP}
${RUNTIME_BOOTSTRAP}

cd "$(printf '%q' "$REMOTE_REPO_ROOT")"
RUN_IDS_PATH="$(printf '%q' "${REMOTE_BUNDLE_DIR}/${run_ids_file}")"
RUN_ID="\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "\$RUN_IDS_PATH")"
if [[ -z "\$RUN_ID" ]]; then
  echo "No run id for task \${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

R_CMD=(
  Rscript "$(printf '%q' "$R_SCRIPT_REL")"
  --mode full
  --stability design
  --predictor-set rq1_predictor_revamp
  --fixed-effect-variant "$variant"
  --optimizer nloptwrap
  --optimizer-checkpoint-every 25
)
${run_config_append}
${aic_append}
R_CMD+=(
  --only-run "\$RUN_ID"
  --skip-aggregate
  --output-dir "$(printf '%q' "$output_rel")"
)
"\${R_CMD[@]}"
EOF
}

make_r_aggregate() {
  local path="$1" variant="$2" output_rel="$3" expected_file="$4" primary_model_id="$5" run_config_arg="$6" aic_arg="$7"
  local run_config_append=""
  local aic_append=""
  if [[ -n "$run_config_arg" ]]; then
    run_config_append="R_CMD+=(${run_config_arg})"
  fi
  if [[ -n "$aic_arg" ]]; then
    aic_append="R_CMD+=(${aic_arg})"
  fi
  cat > "$path" <<EOF
#!/bin/bash
set -euo pipefail

${SLURM_BOOTSTRAP}
${RUNTIME_BOOTSTRAP}

cd "$(printf '%q' "$REMOTE_REPO_ROOT")"
R_CMD=(
  Rscript "$(printf '%q' "$R_SCRIPT_REL")"
  --mode full
  --aggregate-only
  --predictor-set rq1_predictor_revamp
  --fixed-effect-variant "$variant"
  --optimizer nloptwrap
)
${run_config_append}
${aic_append}
R_CMD+=(
  --primary-model-id "$primary_model_id"
  --output-dir "$(printf '%q' "$output_rel")"
)
"\${R_CMD[@]}"

python - <<'PY'
from pathlib import Path
import pandas as pd

expected = {line.strip() for line in Path("$(printf '%q' "${REMOTE_BUNDLE_DIR}/${expected_file}")").read_text().splitlines() if line.strip()}
manifest = Path("$(printf '%q' "$output_rel")") / "tables" / "rq1_lmm_run_manifest.csv"
if not manifest.is_file():
    raise SystemExit(f"missing run manifest: {manifest}")
df = pd.read_csv(manifest)
found = set(df["model_id"].astype(str))
if found != expected:
    raise SystemExit(f"expected model ids {sorted(expected)}, found {sorted(found)}")
if set(df["predictor_set"]) != {"rq1_predictor_revamp"}:
    raise SystemExit(f"unexpected predictor_set values: {sorted(set(df['predictor_set']))}")
if set(df["fixed_effect_variant"]) != {"$variant"}:
    raise SystemExit(f"unexpected fixed_effect_variant values: {sorted(set(df['fixed_effect_variant']))}")
if set(df["matching_policy"]) != {"standard_matching"}:
    raise SystemExit(f"unexpected matching_policy values: {sorted(set(df['matching_policy']))}")
print(df[["model_id", "fixed_effect_variant", "optimizer", "n_rows", "n_hum_rows", "n_control_rows", "fit_status", "singular"]].to_string(index=False))
PY
EOF
}

RUN_CONFIG_ARG="--run-config-csv $(printf '%q' "$THRESHOLD_RUN_CONFIG_REL")"
make_r_smoke "${TMP_DIR}/primary_smoke_additive.sh" "additive" "$PRIMARY_SMOKE_ADD_REL" "h1_60s" "h1_60s" ""
make_r_run_one "${TMP_DIR}/primary_run_one_additive.sh" "additive" "$PRIMARY_OUTPUT_ADD_REL" "primary_additive_run_ids.txt" "" "--run-aic-predictor-scores"
make_r_aggregate "${TMP_DIR}/primary_aggregate_additive.sh" "additive" "$PRIMARY_OUTPUT_ADD_REL" "primary_additive_run_ids.txt" "h1_60s" "" "--run-aic-predictor-scores"

make_r_smoke "${TMP_DIR}/threshold_smoke_additive.sh" "additive" "$THRESHOLD_SMOKE_ADD_REL" "threshold_49m_h1_60s" "threshold_49m_h1_60s" "$RUN_CONFIG_ARG"
make_r_run_one "${TMP_DIR}/threshold_run_one_additive.sh" "additive" "$THRESHOLD_OUTPUT_ADD_REL" "threshold_additive_run_ids.txt" "$RUN_CONFIG_ARG" ""
make_r_aggregate "${TMP_DIR}/threshold_aggregate_additive.sh" "additive" "$THRESHOLD_OUTPUT_ADD_REL" "threshold_additive_run_ids.txt" "threshold_49m_h1_60s" "$RUN_CONFIG_ARG" ""

cat > "${TMP_DIR}/materialize_threshold.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=rq1-thr-data
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
bash materialize_threshold.sh
EOF

make_sbatch() {
  local path="$1" job_name="$2" script_name="$3" cpus="$4" mem="$5" time_limit="$6" array_spec="${7:-}"
  {
    echo '#!/bin/bash'
    echo "#SBATCH --job-name=${job_name}"
    echo "#SBATCH --partition=${PARTITION}"
    echo "#SBATCH --account=${ACCOUNT}"
    echo '#SBATCH --nodes=1'
    echo '#SBATCH --ntasks=1'
    echo "#SBATCH --cpus-per-task=${cpus}"
    echo "#SBATCH --mem=${mem}"
    echo "#SBATCH --time=${time_limit}"
    if [[ -n "$array_spec" ]]; then
      echo "#SBATCH --array=${array_spec}"
      echo '#SBATCH --output=logs/%x-%A_%a.out'
      echo '#SBATCH --error=logs/%x-%A_%a.err'
    else
      echo '#SBATCH --output=logs/%x-%j.out'
      echo '#SBATCH --error=logs/%x-%j.err'
    fi
    echo
    echo 'set -euo pipefail'
    echo "bash ${script_name}"
  } > "$path"
}

make_sbatch "${TMP_DIR}/primary_smoke_additive.sbatch" "rq1-add-smoke" "primary_smoke_additive.sh" 2 16G "00:30:00"
make_sbatch "${TMP_DIR}/primary_array_additive.sbatch" "rq1-add" "primary_run_one_additive.sh" 4 64G "48:00:00" "1-3%3"
make_sbatch "${TMP_DIR}/primary_failover_additive.sbatch" "rq1-add-fail" "primary_run_one_additive.sh" 4 64G "48:00:00" "1-3%3"
make_sbatch "${TMP_DIR}/primary_aggregate_additive.sbatch" "rq1-add-agg" "primary_aggregate_additive.sh" 2 16G "01:00:00"

make_sbatch "${TMP_DIR}/threshold_smoke_additive.sbatch" "rq1-thr-smoke" "threshold_smoke_additive.sh" 2 16G "00:30:00"
make_sbatch "${TMP_DIR}/threshold_array_additive.sbatch" "rq1-thr" "threshold_run_one_additive.sh" 4 64G "48:00:00" "1-3%3"
make_sbatch "${TMP_DIR}/threshold_failover_additive.sbatch" "rq1-thr-fail" "threshold_run_one_additive.sh" 4 64G "48:00:00" "1-3%3"
make_sbatch "${TMP_DIR}/threshold_aggregate_additive.sbatch" "rq1-thr-agg" "threshold_aggregate_additive.sh" 2 16G "01:00:00"

chmod +x "${TMP_DIR}"/*.sh
rsync -avP "${TMP_DIR}/" "${HOST}:${REMOTE_BUNDLE_DIR}/" >/dev/null

submit_remote() {
  local command="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
    "bash -lc $(printf '%q' "${SLURM_BOOTSTRAP}; cd ${REMOTE_BUNDLE_DIR} && ${command}")"
}

submit_remote "bash install_r_packages.sh" >/dev/null

PRIMARY_SMOKE_ADD_JOB="$(submit_remote "sbatch --parsable primary_smoke_additive.sbatch")"
PRIMARY_ARRAY_ADD_JOB="$(submit_remote "sbatch --parsable --dependency=afterok:${PRIMARY_SMOKE_ADD_JOB} primary_array_additive.sbatch")"
PRIMARY_FAIL_ADD_JOB="$(submit_remote "sbatch --parsable --dependency=afterany:${PRIMARY_ARRAY_ADD_JOB} primary_failover_additive.sbatch")"
PRIMARY_AGG_ADD_JOB="$(submit_remote "sbatch --parsable --dependency=afterany:${PRIMARY_FAIL_ADD_JOB} primary_aggregate_additive.sbatch")"

THRESHOLD_MATERIALIZE_JOB="$(submit_remote "sbatch --parsable --dependency=afterok:${PRIMARY_AGG_ADD_JOB} materialize_threshold.sbatch")"
THRESHOLD_SMOKE_ADD_JOB="$(submit_remote "sbatch --parsable --dependency=afterok:${THRESHOLD_MATERIALIZE_JOB} threshold_smoke_additive.sbatch")"
THRESHOLD_ARRAY_ADD_JOB="$(submit_remote "sbatch --parsable --dependency=afterok:${THRESHOLD_SMOKE_ADD_JOB} threshold_array_additive.sbatch")"
THRESHOLD_FAIL_ADD_JOB="$(submit_remote "sbatch --parsable --dependency=afterany:${THRESHOLD_ARRAY_ADD_JOB} threshold_failover_additive.sbatch")"
THRESHOLD_AGG_ADD_JOB="$(submit_remote "sbatch --parsable --dependency=afterany:${THRESHOLD_FAIL_ADD_JOB} threshold_aggregate_additive.sbatch")"

python - <<PY
import json
from pathlib import Path

payload = {
    "bundle_id": "$BUNDLE_ID",
    "branch": "$BRANCH_NAME",
    "host": "$HOST",
    "partition": "$PARTITION",
    "account": "$ACCOUNT",
    "remote_repo_root": "$REMOTE_REPO_ROOT",
    "remote_bundle_dir": "$REMOTE_BUNDLE_DIR",
    "primary_input_relative": "$PRIMARY_INPUT_REL",
    "threshold_run_config_relative": "$THRESHOLD_RUN_CONFIG_REL",
    "primary_additive_output": "$PRIMARY_OUTPUT_ADD_REL",
    "threshold_additive_output": "$THRESHOLD_OUTPUT_ADD_REL",
    "jobs": {
        "primary_additive_smoke": "$PRIMARY_SMOKE_ADD_JOB",
        "primary_additive_array": "$PRIMARY_ARRAY_ADD_JOB",
        "primary_additive_failover": "$PRIMARY_FAIL_ADD_JOB",
        "primary_additive_aggregate": "$PRIMARY_AGG_ADD_JOB",
        "threshold_materialize": "$THRESHOLD_MATERIALIZE_JOB",
        "threshold_additive_smoke": "$THRESHOLD_SMOKE_ADD_JOB",
        "threshold_additive_array": "$THRESHOLD_ARRAY_ADD_JOB",
        "threshold_additive_failover": "$THRESHOLD_FAIL_ADD_JOB",
        "threshold_additive_aggregate": "$THRESHOLD_AGG_ADD_JOB"
    },
    "predictor_set": "rq1_predictor_revamp",
    "retained_fixed_effects": [
        "condition_hum",
        "starting_centroid_distance_m",
        "mean_speed_pre_m_s",
        "pre_delta_centroid_distance_m",
        "observed_polygon_area_pre_m2",
        "recent_change_in_polygon_area_m2"
    ],
    "excluded_fixed_effects": [
        "time_of_day_minutes",
        "local_collared_density_per_1000m2",
        "n_collars_pre",
        "collar_availability",
        "minutes_since_session_start"
    ],
    "threshold_m": 49.358079,
    "sequence_policy": "primary additive models complete before the threshold sensitivity materialization and additive threshold fits",
    "materialization_time_limit": "03:00:00",
    "fit_time_limit": "48:00:00",
    "optimizer": "nloptwrap",
    "optimizer_checkpoint_every": 25
}
path = Path("$LOCAL_BUNDLE_DIR") / "submission.json"
path.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf8")
print(json.dumps(payload, indent=2))
PY

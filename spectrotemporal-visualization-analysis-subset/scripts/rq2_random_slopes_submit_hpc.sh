#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_analysis_common.sh"

CONFIG_PATH="${1:-$DEFAULT_CONFIG_PATH}"

RUN_ROOT_REL="${RUN_ROOT_REL:-backend/data/processed/analysis/rq2_hum_timing_group_dispersion/random_slopes_hpc}"
RETAINED_RANDOM_SLOPES_BUNDLE_ID="rq2_h2_attenuation_random_slopes_20260907T194255Z"
RETAINED_RANDOM_SLOPES_REL="${RUN_ROOT_REL}/${RETAINED_RANDOM_SLOPES_BUNDLE_ID}"
RUN_ROOT_ABS="${REPO_ROOT}/${RUN_ROOT_REL}"
ACTIVE_BUNDLE_FILE="${RUN_ROOT_ABS}/.active_random_slopes_bundle_id"
if [[ -z "${BUNDLE_ID:-}" ]]; then
  ACTIVE_BUNDLE_ID=""
  if [[ "${FORCE_NEW_BUNDLE:-0}" != "1" && -s "$ACTIVE_BUNDLE_FILE" ]]; then
    ACTIVE_BUNDLE_ID="$(tr -d '[:space:]' < "$ACTIVE_BUNDLE_FILE")"
  fi
  if [[ -n "$ACTIVE_BUNDLE_ID" && -d "${RUN_ROOT_ABS}/${ACTIVE_BUNDLE_ID}" ]]; then
    BUNDLE_ID="$ACTIVE_BUNDLE_ID"
  else
    LATEST_SUBMISSION=""
    if [[ "${FORCE_NEW_BUNDLE:-0}" != "1" && -d "$RUN_ROOT_ABS" ]]; then
      LATEST_SUBMISSION="$(find "$RUN_ROOT_ABS" -mindepth 3 -maxdepth 3 -path '*/hpc_bundle/submission.json' -print 2>/dev/null | sort | tail -n 1)"
    fi
    if [[ -n "$LATEST_SUBMISSION" ]]; then
      BUNDLE_ID="$(basename "$(dirname "$(dirname "$LATEST_SUBMISSION")")")"
    else
      BUNDLE_ID="rq2_h2_attenuation_random_slopes_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
  fi
fi
RUN_REL="${RUN_ROOT_REL}/${BUNDLE_ID}"
ATTENUATION_INPUT_SOURCE_REL="${ATTENUATION_INPUT_SOURCE_REL:-${RETAINED_RANDOM_SLOPES_REL}/inputs/rq2_attenuation_filter_model_input_60s.csv}"
SPEED_INPUT_SOURCE_REL="${SPEED_INPUT_SOURCE_REL:-${RETAINED_RANDOM_SLOPES_REL}/inputs/rq2_speed_sensitivity_model_input_60s.csv}"
ATTENUATION_R_SCRIPT_REL="backend/src/stv_analysis/rq2_hum_timing_group_dispersion/amplitude_threshold_glmmtmb_workflow.R"
SPEED_R_SCRIPT_REL="backend/src/stv_analysis/rq2_hum_timing_group_dispersion/speed_sensitivity_nb1_interaction_workflow.R"
RANDOM_EFFECTS_STRUCTURE_VERSION="rq2_session_maximal_random_slopes_v1"
ATTENUATION_THRESHOLDS="attenuation_lenient,attenuation_primary,attenuation_conservative"
SMOKE_NSIM="${SMOKE_NSIM:-20}"
FULL_NSIM="${FULL_NSIM:-1000}"
SMOKE_MAX_ROWS="${SMOKE_MAX_ROWS:-1200}"
SMOKE_TIME_LIMIT="${SMOKE_TIME_LIMIT:-01:00:00}"
SMOKE_MEM="${SMOKE_MEM:-16G}"
INSTALL_TIME_LIMIT="${INSTALL_TIME_LIMIT:-04:00:00}"
INSTALL_MEM="${INSTALL_MEM:-16G}"
INSTALL_CPUS="${INSTALL_CPUS:-2}"
FULL_TIME_LIMIT="${FULL_TIME_LIMIT:-48:00:00}"
FULL_MEM="${FULL_MEM:-64G}"
FULL_CPUS="${FULL_CPUS:-4}"
BRANCH_NAME="$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || true)"

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
resources = config.analysis_hpc.resources.get("model_fit")
if resources is not None:
    print(f"MODEL_FIT\t{resources.cpus_per_task}\t{resources.memory_gb}G\t{resources.time_limit}")
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
MODEL_FIT_ROW_INDEX=5
if [[ "${RESOLVED[5]:-}" == MODEL_FIT$'\t'* ]]; then
  IFS=$'\t' read -r _ FULL_CPUS_FROM_CONFIG FULL_MEM_FROM_CONFIG FULL_TIME_FROM_CONFIG <<<"${RESOLVED[5]}"
  FULL_CPUS="${FULL_CPUS_OVERRIDE:-${FULL_CPUS_FROM_CONFIG:-$FULL_CPUS}}"
  FULL_MEM="${FULL_MEM_OVERRIDE:-${FULL_MEM_FROM_CONFIG:-$FULL_MEM}}"
  FULL_TIME_LIMIT="${FULL_TIME_LIMIT_OVERRIDE:-${FULL_TIME_FROM_CONFIG:-$FULL_TIME_LIMIT}}"
  MODEL_FIT_ROW_INDEX=6
fi

MODULE_LOADS=()
ENVIRONMENT_ROWS=()
for row in "${RESOLVED[@]:$MODEL_FIT_ROW_INDEX}"; do
  if [[ "$row" == MODULE$'\t'* ]]; then
    MODULE_LOADS+=("${row#*$'\t'}")
  elif [[ "$row" == ENV$'\t'* ]]; then
    ENVIRONMENT_ROWS+=("${row#*$'\t'}")
  fi
done

LOCAL_RUN_DIR="${REPO_ROOT}/${RUN_REL}"
LOCAL_INPUT_DIR="${LOCAL_RUN_DIR}/inputs"
LOCAL_BUNDLE_DIR="${LOCAL_RUN_DIR}/hpc_bundle"
REMOTE_RUN_DIR="${REMOTE_REPO_ROOT}/${RUN_REL}"
REMOTE_BUNDLE_DIR="${REMOTE_RUN_DIR}/hpc_bundle"

mkdir -p "$RUN_ROOT_ABS"
printf '%s\n' "$BUNDLE_ID" > "$ACTIVE_BUNDLE_FILE"

print_submission_summary() {
  local job_ids
  job_ids="$("$PYTHON_BIN" - <<'PY' "${LOCAL_BUNDLE_DIR}/submission.json"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
print(",".join(str(value) for value in payload.get("jobs", {}).values() if value))
PY
)"
  cat "${LOCAL_BUNDLE_DIR}/submission.json"
  echo
  echo "Monitor:"
  echo "  ssh ${HOST} 'bash -lc '\\''OS_MAJOR=\"\$(. /etc/os-release && printf %s \"\${VERSION_ID%%.*}\")\"; SLURM_BIN=\"/opt/slurm/el\${OS_MAJOR}/current/install/bin\"; if [ ! -d \"\$SLURM_BIN\" ]; then SLURM_BIN=\"\$(ls -d /opt/slurm/el\${OS_MAJOR}/*/install/bin 2>/dev/null | sort | tail -n 1)\"; fi; export PATH=\"\$SLURM_BIN:\$PATH\"; squeue -j ${job_ids} -o \"%i %T %M %L %R %j\"'\\'''"
  echo "Pull later:"
  echo "  ${SCRIPT_DIR}/rq2_random_slopes_pull_hpc.sh ${LOCAL_RUN_DIR}"
}

if [[ -f "${LOCAL_BUNDLE_DIR}/submission.json" && "${RESUBMIT:-0}" != "1" ]]; then
  echo "Existing submission found for ${BUNDLE_ID}; set RESUBMIT=1 to submit a replacement graph or FORCE_NEW_BUNDLE=1 for a new run." >&2
  print_submission_summary
  exit 0
fi

mkdir -p "$LOCAL_INPUT_DIR" "$LOCAL_BUNDLE_DIR"

prepare_r_package_bundle() {
  local package_dir="${LOCAL_BUNDLE_DIR}/r_packages"
  mkdir -p "$package_dir"
  Rscript - <<'RSCRIPT' "$package_dir"
args <- commandArgs(trailingOnly = TRUE)
package_dir <- normalizePath(args[[1]], mustWork = FALSE)
dir.create(package_dir, recursive = TRUE, showWarnings = FALSE)

repos <- c(CRAN = "https://cloud.r-project.org")
roots <- c("glmmTMB", "DHARMa", "ggplot2")
base_packages <- c(
  "base", "compiler", "datasets", "graphics", "grDevices", "grid",
  "methods", "parallel", "splines", "stats", "stats4", "tcltk",
  "tools", "utils", "R"
)

available <- utils::available.packages(repos = repos, type = "source")
missing_roots <- setdiff(roots, rownames(available))
if (length(missing_roots)) {
  stop(paste("CRAN source metadata missing packages:", paste(missing_roots, collapse = ", ")))
}

direct_deps <- function(pkg) {
  deps <- tools::package_dependencies(pkg, db = available, which = c("Depends", "Imports", "LinkingTo"), recursive = FALSE)[[pkg]]
  deps <- setdiff(unique(deps), base_packages)
  intersect(deps, rownames(available))
}

needed <- setdiff(unique(c(
  roots,
  unlist(tools::package_dependencies(roots, db = available, which = c("Depends", "Imports", "LinkingTo"), recursive = TRUE), use.names = FALSE)
)), base_packages)
needed <- intersect(needed, rownames(available))

visited <- character()
order <- character()
visit <- function(pkg) {
  if (pkg %in% visited || !(pkg %in% needed)) return(invisible(NULL))
  visited <<- c(visited, pkg)
  for (dep in direct_deps(pkg)) visit(dep)
  order <<- c(order, pkg)
  invisible(NULL)
}
for (pkg in roots) visit(pkg)
order <- unique(order)

manifest <- data.frame(
  package = order,
  version = available[order, "Version"],
  tarball = paste0(order, "_", available[order, "Version"], ".tar.gz"),
  stringsAsFactors = FALSE
)
missing_tarballs <- manifest$package[!file.exists(file.path(package_dir, manifest$tarball))]
if (length(missing_tarballs)) {
  invisible(utils::download.packages(missing_tarballs, destdir = package_dir, repos = repos, type = "source", quiet = TRUE))
}

utils::write.csv(manifest, file.path(package_dir, "package_manifest.csv"), row.names = FALSE, na = "")
writeLines(manifest$package, file.path(package_dir, "install_order.txt"))
cat("Prepared R source package bundle:", package_dir, "\n")
cat("Package count:", nrow(manifest), "\n")
cat("Roots:", paste(roots, collapse = ", "), "\n")
RSCRIPT
}

prepare_r_package_bundle

copy_input() {
  local source_rel="$1"
  local dest_name="$2"
  local source_path="${REPO_ROOT}/${source_rel}"
  if [[ ! -f "$source_path" ]]; then
    echo "Missing required input: $source_path" >&2
    exit 1
  fi
  if [[ "$source_path" == "${LOCAL_INPUT_DIR}/${dest_name}" ]]; then
    return
  fi
  rsync -a "$source_path" "${LOCAL_INPUT_DIR}/${dest_name}"
}

copy_input "$ATTENUATION_INPUT_SOURCE_REL" "rq2_attenuation_filter_model_input_60s.csv"
copy_input "$SPEED_INPUT_SOURCE_REL" "rq2_speed_sensitivity_model_input_60s.csv"

cat > "${LOCAL_RUN_DIR}/input_manifest.json" <<EOF
{
  "bundle_id": "$BUNDLE_ID",
  "attenuation_input_source_rel": "$ATTENUATION_INPUT_SOURCE_REL",
  "speed_input_source_rel": "$SPEED_INPUT_SOURCE_REL",
  "attenuation_thresholds": "$ATTENUATION_THRESHOLDS",
  "random_effects_structure_version": "$RANDOM_EFFECTS_STRUCTURE_VERSION",
  "created_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

"${SCRIPT_DIR}/analysis_sync_repo_to_hpc.sh" "$CONFIG_PATH"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
  "mkdir -p $(printf '%q' "$REMOTE_RUN_DIR/inputs") $(printf '%q' "$REMOTE_BUNDLE_DIR/logs") $(printf '%q' "$REMOTE_BUNDLE_DIR/r_packages")"
rsync -aP --checksum "${LOCAL_INPUT_DIR}/" "${HOST}:${REMOTE_RUN_DIR}/inputs/" >/dev/null
rsync -avP "${LOCAL_RUN_DIR}/input_manifest.json" "${HOST}:${REMOTE_RUN_DIR}/input_manifest.json" >/dev/null
rsync -avP "$ACTIVE_BUNDLE_FILE" "${HOST}:${REMOTE_REPO_ROOT}/${RUN_ROOT_REL}/.active_random_slopes_bundle_id" >/dev/null
rsync -aP --checksum "${LOCAL_BUNDLE_DIR}/r_packages/" "${HOST}:${REMOTE_BUNDLE_DIR}/r_packages/" >/dev/null

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
RUNTIME_BOOTSTRAP+=$'\n''export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"'
RUNTIME_BOOTSTRAP+=$'\n''mkdir -p "$R_LIBS_USER"'
RUNTIME_BOOTSTRAP+=$'\n'"source $(printf '%q' "${REMOTE_VENV_PATH}/bin/activate")"
printf -v REMOTE_BACKEND_SRC '%q' "${REMOTE_REPO_ROOT}/backend/src"
RUNTIME_BOOTSTRAP+=$'\n'"export PYTHONPATH=${REMOTE_BACKEND_SRC}:\${PYTHONPATH:-}"
RUNTIME_BOOTSTRAP+=$'\n''export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"'
RUNTIME_BOOTSTRAP+=$'\n''export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"'
RUNTIME_BOOTSTRAP+=$'\n''export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"'

cat > "${TMP_DIR}/install_r_packages.sh" <<EOF
#!/bin/bash
set -euo pipefail

${SLURM_BOOTSTRAP}
${RUNTIME_BOOTSTRAP}

export STV_R_PACKAGE_BUNDLE="$(printf '%q' "$REMOTE_BUNDLE_DIR/r_packages")"

Rscript - <<'RSCRIPT'
bundle_dir <- Sys.getenv("STV_R_PACKAGE_BUNDLE")
install_order_path <- file.path(bundle_dir, "install_order.txt")
manifest_path <- file.path(bundle_dir, "package_manifest.csv")
if (!file.exists(install_order_path) || !file.exists(manifest_path)) {
  stop(paste("R package bundle is incomplete:", bundle_dir))
}
manifest <- utils::read.csv(manifest_path, check.names = FALSE, stringsAsFactors = FALSE)
install_order <- readLines(install_order_path, warn = FALSE)

install_from_bundle <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    row <- manifest[manifest\$package == pkg, , drop = FALSE]
    if (nrow(row) != 1L) {
      stop(paste("R package missing from source bundle:", pkg))
    }
    tarball <- file.path(bundle_dir, row\$tarball[[1]])
    if (!file.exists(tarball)) {
      stop(paste("R package tarball missing from source bundle:", tarball))
    }
    install.packages(tarball, repos = NULL, type = "source", dependencies = FALSE)
  }
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste("R package is required but unavailable:", pkg))
  }
  cat(pkg, as.character(utils::packageVersion(pkg)), "\n")
}

cat("R_LIBS_USER=", Sys.getenv("R_LIBS_USER"), "\n", sep = "")
cat("bundle_dir=", bundle_dir, "\n", sep = "")
for (pkg in install_order) {
  install_from_bundle(pkg)
}
for (pkg in c("glmmTMB", "DHARMa", "ggplot2")) {
  install_from_bundle(pkg)
}
cat("libPaths=", paste(.libPaths(), collapse = " | "), "\n", sep = "")
RSCRIPT
EOF

cat > "${TMP_DIR}/smoke.sh" <<EOF
#!/bin/bash
set -euo pipefail

${SLURM_BOOTSTRAP}
${RUNTIME_BOOTSTRAP}

cd "$(printf '%q' "$REMOTE_REPO_ROOT")"
export STV_ANALYSIS_RUN_CONTEXT=hpc_slurm_random_slopes_smoke

Rscript - <<'RSCRIPT'
for (pkg in c("glmmTMB", "DHARMa", "ggplot2")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste("missing required R package:", pkg))
  }
}
for (path in c(
  "$(printf '%q' "$ATTENUATION_R_SCRIPT_REL")",
  "$(printf '%q' "$SPEED_R_SCRIPT_REL")"
)) {
  parse(path)
}
RSCRIPT

python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("$(printf '%q' "$REMOTE_RUN_DIR")") / "inputs/rq2_attenuation_filter_model_input_60s.csv"
df = pd.read_csv(path)
expected = {
    "attenuation_lenient": 6371,
    "attenuation_primary": 4617,
    "attenuation_conservative": 3517,
}
actual = df.groupby("threshold_id")["hum_count"].sum().astype(int).to_dict()
if actual != expected:
    raise SystemExit(f"attenuation input hum totals mismatch: {actual} != {expected}")
print("attenuation input hum totals:", actual)
PY

Rscript "$(printf '%q' "$ATTENUATION_R_SCRIPT_REL")" \\
  --input "$(printf '%q' "$REMOTE_RUN_DIR/inputs/rq2_attenuation_filter_model_input_60s.csv")" \\
  --output-dir "$(printf '%q' "$REMOTE_RUN_DIR/smoke/amplitude_attenuation_filter")" \\
  --thresholds "$ATTENUATION_THRESHOLDS" \\
  --output-prefix rq2_attenuation \\
  --model-id-tag attenuation_amplitude \\
  --response-label attenuation_thresholded_hum_count \\
  --run-label attenuation-threshold \\
  --plot-title-prefix "Hum-rate fixed effects after attenuation-calibrated amplitude filtering" \\
  --mode smoke \\
  --max-rows "$(printf '%q' "$SMOKE_MAX_ROWS")" \\
  --nsim "$(printf '%q' "$SMOKE_NSIM")" \\
  --seed 20260902

Rscript "$(printf '%q' "$SPEED_R_SCRIPT_REL")" \\
  --input "$(printf '%q' "$REMOTE_RUN_DIR/inputs/rq2_speed_sensitivity_model_input_60s.csv")" \\
  --output-dir "$(printf '%q' "$REMOTE_RUN_DIR/smoke/speed_sensitivity")" \\
  --speed-intervals 1,5,10,20 \\
  --mode smoke \\
  --max-rows "$(printf '%q' "$SMOKE_MAX_ROWS")" \\
  --nsim "$(printf '%q' "$SMOKE_NSIM")" \\
  --seed 20260903

python - <<'PY'
from pathlib import Path
import pandas as pd

run_dir = Path("$(printf '%q' "$REMOTE_RUN_DIR")")
version = "$(printf '%q' "$RANDOM_EFFECTS_STRUCTURE_VERSION")"
checks = [
    ("amplitude_attenuation_filter", run_dir / "smoke/amplitude_attenuation_filter/tables/rq2_attenuation_model_run_manifest.csv", 12),
    ("speed", run_dir / "smoke/speed_sensitivity/tables/rq2_speed_sensitivity_model_run_manifest.csv", 4),
]
for label, path, expected_rows in checks:
    if not path.is_file():
        raise SystemExit(f"missing smoke manifest for {label}: {path}")
    df = pd.read_csv(path)
    if len(df.index) != expected_rows:
        raise SystemExit(f"{label} smoke manifest row count {len(df.index)} != {expected_rows}")
    if set(df["random_effects_structure_version"].dropna().astype(str)) != {version}:
        raise SystemExit(f"{label} smoke manifest does not use {version}")
    if not df["random_effects"].astype(str).str.contains("z_area").all():
        raise SystemExit(f"{label} smoke manifest is missing random slopes")
    print(label, df[["model_id", "fit_status", "cache_status"]].to_string(index=False))
PY
EOF

make_full_script() {
  local path="$1"
  local label="$2"
  local r_script_rel="$3"
  local input_name="$4"
  local output_subdir="$5"
  local extra_args="$6"
  local seed="$7"
  local extra_args_block=""
  if [[ -n "$extra_args" ]]; then
    extra_args_block="  ${extra_args} \\
"
  fi
  cat > "$path" <<EOF
#!/bin/bash
set -euo pipefail

${SLURM_BOOTSTRAP}
${RUNTIME_BOOTSTRAP}

cd "$(printf '%q' "$REMOTE_REPO_ROOT")"
export STV_ANALYSIS_RUN_CONTEXT=hpc_slurm_random_slopes_full

Rscript "$(printf '%q' "$r_script_rel")" \\
  --input "$(printf '%q' "$REMOTE_RUN_DIR/inputs/${input_name}")" \\
  --output-dir "$(printf '%q' "$REMOTE_RUN_DIR/full/${output_subdir}")" \\
${extra_args_block}  --mode full \\
  --nsim "$(printf '%q' "$FULL_NSIM")" \\
  --seed "$seed"
EOF
}

make_full_script "${TMP_DIR}/speed_full.sh" "speed" "$SPEED_R_SCRIPT_REL" "rq2_speed_sensitivity_model_input_60s.csv" "speed_sensitivity" "--speed-intervals 1,5,10,20" "20260913"

cat > "${TMP_DIR}/attenuation_full.sh" <<EOF
#!/bin/bash
set -euo pipefail

${SLURM_BOOTSTRAP}
${RUNTIME_BOOTSTRAP}

cd "$(printf '%q' "$REMOTE_REPO_ROOT")"
export STV_ANALYSIS_RUN_CONTEXT=hpc_slurm_random_slopes_full

Rscript "$(printf '%q' "$ATTENUATION_R_SCRIPT_REL")" \\
  --input "$(printf '%q' "$REMOTE_RUN_DIR/inputs/rq2_attenuation_filter_model_input_60s.csv")" \\
  --output-dir "$(printf '%q' "$REMOTE_RUN_DIR/full/amplitude_attenuation_filter")" \\
  --thresholds "$ATTENUATION_THRESHOLDS" \\
  --output-prefix rq2_attenuation \\
  --model-id-tag attenuation_amplitude \\
  --response-label attenuation_thresholded_hum_count \\
  --run-label attenuation-threshold \\
  --plot-title-prefix "Hum-rate fixed effects after attenuation-calibrated amplitude filtering" \\
  --mode full \\
  --nsim "$(printf '%q' "$FULL_NSIM")" \\
  --seed 20260912
EOF

cat > "${TMP_DIR}/summary.sh" <<EOF
#!/bin/bash
set -euo pipefail

${SLURM_BOOTSTRAP}
${RUNTIME_BOOTSTRAP}

python - <<'PY'
from pathlib import Path
import json
import pandas as pd

run_dir = Path("$(printf '%q' "$REMOTE_RUN_DIR")")
version = "$(printf '%q' "$RANDOM_EFFECTS_STRUCTURE_VERSION")"
cases = {
    "amplitude_attenuation_filter": run_dir / "full/amplitude_attenuation_filter/tables/rq2_attenuation_model_run_manifest.csv",
    "speed_sensitivity": run_dir / "full/speed_sensitivity/tables/rq2_speed_sensitivity_model_run_manifest.csv",
}
summary = {
    "bundle_id": "$(printf '%q' "$BUNDLE_ID")",
    "random_effects_structure_version": version,
    "remote_run_dir": str(run_dir),
    "outputs": {},
}
for label, path in cases.items():
    if not path.is_file():
        raise SystemExit(f"missing full manifest for {label}: {path}")
    df = pd.read_csv(path)
    if "random_effects_structure_version" not in df.columns:
        raise SystemExit(f"{label} manifest has no random_effects_structure_version")
    versions = sorted(set(df["random_effects_structure_version"].dropna().astype(str)))
    if versions != [version]:
        raise SystemExit(f"{label} manifest versions {versions} != {version}")
    summary["outputs"][label] = {
        "manifest": str(path.relative_to(run_dir)),
        "models": int(len(df.index)),
        "fit_statuses": sorted(set(df["fit_status"].dropna().astype(str))),
        "cache_statuses": sorted(set(df["cache_status"].dropna().astype(str))) if "cache_status" in df.columns else [],
        "random_effects": sorted(set(df["random_effects"].dropna().astype(str))),
    }
    if label == "amplitude_attenuation_filter":
        expected_hums = {
            "attenuation_lenient": 6371,
            "attenuation_primary": 4617,
            "attenuation_conservative": 3517,
        }
        actual_hums = df.groupby("threshold_id")["hum_count_total"].first().astype(int).to_dict()
        if actual_hums != expected_hums:
            raise SystemExit(f"attenuation full hum totals mismatch: {actual_hums} != {expected_hums}")
        summary["outputs"][label]["hum_count_totals"] = actual_hums
summary_path = run_dir / "run_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
EOF

make_sbatch() {
  local path="$1"
  local job_name="$2"
  local script_name="$3"
  local cpus="$4"
  local mem="$5"
  local time_limit="$6"
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
    echo '#SBATCH --output=logs/%x-%j.out'
    echo '#SBATCH --error=logs/%x-%j.err'
    echo
    echo 'set -euo pipefail'
    echo "bash ${script_name}"
  } > "$path"
}

make_sbatch "${TMP_DIR}/smoke.sbatch" "rq2-h2-rs-smoke" "smoke.sh" 2 "$SMOKE_MEM" "$SMOKE_TIME_LIMIT"
make_sbatch "${TMP_DIR}/install_r_packages.sbatch" "rq2-h2-rs-install" "install_r_packages.sh" "$INSTALL_CPUS" "$INSTALL_MEM" "$INSTALL_TIME_LIMIT"
make_sbatch "${TMP_DIR}/install_r_packages_retry.sbatch" "rq2-h2-rs-install-retry" "install_r_packages.sh" "$INSTALL_CPUS" "$INSTALL_MEM" "$INSTALL_TIME_LIMIT"
make_sbatch "${TMP_DIR}/attenuation_full.sbatch" "rq2-h2-rs-attenuation" "attenuation_full.sh" "$FULL_CPUS" "$FULL_MEM" "$FULL_TIME_LIMIT"
make_sbatch "${TMP_DIR}/attenuation_retry.sbatch" "rq2-h2-rs-attenuation-retry" "attenuation_full.sh" "$FULL_CPUS" "$FULL_MEM" "$FULL_TIME_LIMIT"
make_sbatch "${TMP_DIR}/speed_full.sbatch" "rq2-h2-rs-speed" "speed_full.sh" "$FULL_CPUS" "$FULL_MEM" "$FULL_TIME_LIMIT"
make_sbatch "${TMP_DIR}/speed_retry.sbatch" "rq2-h2-rs-speed-retry" "speed_full.sh" "$FULL_CPUS" "$FULL_MEM" "$FULL_TIME_LIMIT"
make_sbatch "${TMP_DIR}/summary.sbatch" "rq2-h2-rs-summary" "summary.sh" 2 16G "00:30:00"

chmod +x "${TMP_DIR}"/*.sh
rsync -avP "${TMP_DIR}/" "${HOST}:${REMOTE_BUNDLE_DIR}/" >/dev/null

submit_remote() {
  local command="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
    "bash -lc $(printf '%q' "${SLURM_BOOTSTRAP}; cd ${REMOTE_BUNDLE_DIR} && ${command}")"
}

INSTALL_JOB="$(submit_remote "sbatch --parsable install_r_packages.sbatch")"
INSTALL_RETRY_JOB="$(submit_remote "sbatch --parsable --dependency=afterany:${INSTALL_JOB} install_r_packages_retry.sbatch")"
SMOKE_JOB="$(submit_remote "sbatch --parsable --dependency=afterok:${INSTALL_RETRY_JOB} smoke.sbatch")"
ATTENUATION_JOB="$(submit_remote "sbatch --parsable --dependency=afterok:${SMOKE_JOB} attenuation_full.sbatch")"
SPEED_JOB="$(submit_remote "sbatch --parsable --dependency=afterok:${SMOKE_JOB} speed_full.sbatch")"
ATTENUATION_RETRY_JOB="$(submit_remote "sbatch --parsable --dependency=afterok:${SMOKE_JOB},afterany:${ATTENUATION_JOB} attenuation_retry.sbatch")"
SPEED_RETRY_JOB="$(submit_remote "sbatch --parsable --dependency=afterok:${SMOKE_JOB},afterany:${SPEED_JOB} speed_retry.sbatch")"
SUMMARY_JOB="$(submit_remote "sbatch --parsable --dependency=afterok:${SMOKE_JOB},afterany:${ATTENUATION_RETRY_JOB}:${SPEED_RETRY_JOB} summary.sbatch")"

cat > "${LOCAL_BUNDLE_DIR}/submission.json" <<EOF
{
  "bundle_id": "$BUNDLE_ID",
  "branch": "$BRANCH_NAME",
  "host": "$HOST",
  "partition": "$PARTITION",
  "account": "$ACCOUNT",
  "remote_repo_root": "$REMOTE_REPO_ROOT",
  "remote_run_dir": "$REMOTE_RUN_DIR",
  "remote_bundle_dir": "$REMOTE_BUNDLE_DIR",
  "local_run_dir": "$LOCAL_RUN_DIR",
  "random_effects_structure_version": "$RANDOM_EFFECTS_STRUCTURE_VERSION",
  "inputs": {
    "amplitude_attenuation_filter": "inputs/rq2_attenuation_filter_model_input_60s.csv",
    "speed_sensitivity": "inputs/rq2_speed_sensitivity_model_input_60s.csv"
  },
  "outputs": {
    "amplitude_attenuation_filter": "full/amplitude_attenuation_filter",
    "speed_sensitivity": "full/speed_sensitivity"
  },
  "jobs": {
    "install_r_packages": "$INSTALL_JOB",
    "install_r_packages_retry": "$INSTALL_RETRY_JOB",
    "smoke": "$SMOKE_JOB",
    "attenuation_full": "$ATTENUATION_JOB",
    "speed_full": "$SPEED_JOB",
    "attenuation_retry": "$ATTENUATION_RETRY_JOB",
    "speed_retry": "$SPEED_RETRY_JOB",
    "summary": "$SUMMARY_JOB"
  },
  "smoke": {
    "max_rows_per_model": "$SMOKE_MAX_ROWS",
    "dharma_nsim": "$SMOKE_NSIM",
    "time_limit": "$SMOKE_TIME_LIMIT",
    "memory": "$SMOKE_MEM"
  },
  "install": {
    "cpus_per_task": "$INSTALL_CPUS",
    "memory": "$INSTALL_MEM",
    "time_limit": "$INSTALL_TIME_LIMIT"
  },
  "full": {
    "dharma_nsim": "$FULL_NSIM",
    "cpus_per_task": "$FULL_CPUS",
    "memory": "$FULL_MEM",
    "time_limit": "$FULL_TIME_LIMIT"
  },
  "submitted_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
rsync -aP "${LOCAL_BUNDLE_DIR}/submission.json" "${HOST}:${REMOTE_BUNDLE_DIR}/submission.json" >/dev/null
print_submission_summary

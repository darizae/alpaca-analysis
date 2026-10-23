#!/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG_PATH="${1:-config/hpc_full_grid.yaml}"
shift || true

START_STAGE="features"
INITIAL_DEPENDENCY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-stage)
      START_STAGE="${2:-}"
      shift 2
      ;;
    --initial-dependency)
      INITIAL_DEPENDENCY="${2:-}"
      shift 2
      ;;
    *)
      echo "Usage: $0 <config-path> [--start-stage features|feature_merge|grid|aggregate|exemplars] [--initial-dependency <dependency>]" >&2
      exit 1
      ;;
  esac
done

submit_stage() {
  local stage="$1"
  local dependency="${2:-}"
  if [[ -n "$dependency" ]]; then
    "$PYTHON_BIN" -m corpus_dedup.cli submit-hpc --config "$CONFIG_PATH" --stage "$stage" --dependency "$dependency"
  else
    "$PYTHON_BIN" -m corpus_dedup.cli submit-hpc --config "$CONFIG_PATH" --stage "$stage"
  fi
}

extract_job_id() {
  awk '/Submitted batch job/ {print $4}'
}

STAGES=(features feature_merge grid aggregate exemplars)
start_index=-1
for i in "${!STAGES[@]}"; do
  if [[ "${STAGES[$i]}" == "$START_STAGE" ]]; then
    start_index="$i"
    break
  fi
done

if [[ "$start_index" -lt 0 ]]; then
  echo "Unsupported start stage: $START_STAGE" >&2
  exit 1
fi

previous_job_id=""
for i in "${!STAGES[@]}"; do
  if [[ "$i" -lt "$start_index" ]]; then
    continue
  fi
  stage="${STAGES[$i]}"
  dependency=""
  if [[ -n "$previous_job_id" ]]; then
    dependency="afterok:${previous_job_id}"
  elif [[ -n "$INITIAL_DEPENDENCY" ]]; then
    dependency="$INITIAL_DEPENDENCY"
  fi

  if [[ -n "$dependency" ]]; then
    stage_output="$(submit_stage "$stage" "$dependency")"
  else
    stage_output="$(submit_stage "$stage")"
  fi
  echo "$stage_output"
  previous_job_id="$(printf '%s\n' "$stage_output" | extract_job_id)"
  if [[ -z "$previous_job_id" ]]; then
    echo "Could not parse ${stage} job id" >&2
    exit 1
  fi
done

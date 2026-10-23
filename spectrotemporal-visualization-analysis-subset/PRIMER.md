# Retained Analysis Primer

Start here before touching `backend/src/stv_analysis`, processed analysis artifacts, or HPC wrappers.

## Current Scope

The repository now keeps only the backend data-preparation code and the retained thesis analyses:

- `stv_backend`: fetch recorder metadata files, build the processed dataset, verify it, and materialize the analysis-base bundle.
- `stv_core`: shared AudioMoth and geometry primitives.
- `stv_analysis`: retained RQ1 post-hum cohesion and RQ2 hum-timing/group-dispersion analyses.

Removed surfaces are intentional: the web app, behavior-derived paddock zones, the sibling behavior-zone scratch project, RQ1 dyadic experiments, exploratory amplitude filters derived from the main deployment dataset, and one-off draft/spec work.

No backwards compatibility is required. Prefer deleting obsolete paths and commands over preserving aliases.

## Retained Artifacts

The analysis artifacts to preserve are:

- `backend/data/processed/analysis/base`
- `backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp`
- `backend/data/processed/analysis/rq2_hum_timing_group_dispersion/random_slopes_hpc/rq2_h2_attenuation_random_slopes_20260907T194255Z`

The retained RQ2 bundle is the attenuation-calibrated random-slopes result and includes the conservative-threshold speed sensitivity analysis. Other amplitude filters are not part of the supported research surface.

## Backend Workflow

Local prerequisite:

```bash
cd backend
./.venv/bin/stv-backend build-dataset
./.venv/bin/stv-backend build-analysis-base
./.venv/bin/stv-backend verify-dataset
```

The backend manifest still writes the GPS-derived `paddock_hull_path`, because retained analyses need the deployment geometry. It does not build or verify behavior-derived paddock-zone layers.

## Launch Model

RQ1 and RQ2 still use immutable launch directories when upstream materialization must be rebuilt:

```text
backend/data/processed/analysis/<analysis_dir>/launches/<launch_label>/
```

Each launch contains a base copy, configs, copied docs, stage caches, progress JSON, tables, reports, and optional HPC bundle files. Launch directories are immutable; if a launch is wrong, create a new one.

## RQ1

Retained research output:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp
```

The kept Python/R surface is the primary post-hum cohesion materialization path, the threshold sensitivity materialization, the further-convergence summary, and the scientific LMM R workflow. Retired exploratory sweeps, dyadic-distance, bout, and local-threshold forks are out of scope.

The main artifact contract remains:

```text
docs/rq1_post_hum_cohesion/ARTIFACT_CONTRACT.md
```

## RQ2

Retained research output:

```text
backend/data/processed/analysis/rq2_hum_timing_group_dispersion/random_slopes_hpc/rq2_h2_attenuation_random_slopes_20260907T194255Z
```

The supported amplitude path is:

- Python calibration and data materialization: `amplitude_attenuation_filter.py`
- shared audio metric helpers: `amplitude_audio.py`
- shared thresholded-response GLMM workflow: `amplitude_threshold_glmmtmb_workflow.R`
- random-slopes submission wrapper: `scripts/rq2_random_slopes_submit_hpc.sh`
- random-slopes pull wrapper: `scripts/rq2_random_slopes_pull_hpc.sh`

The submitter defaults to the retained attenuation inputs and submits the attenuation-calibrated amplitude job with:

```text
--output-prefix rq2_attenuation
--model-id-tag attenuation_amplitude
--response-label attenuation_thresholded_hum_count
--run-label attenuation-threshold
```

It also validates the expected retained hum totals:

```text
attenuation_lenient: 6371
attenuation_primary: 4617
attenuation_conservative: 3517
```

The companion speed-sensitivity workflow is retained only as part of the final RQ2 random-slopes bundle.

## HPC Defaults

Remote target defaults live in `backend/config/default.toml` under `[analysis_hpc]`.

Wrapper scripts:

- `scripts/analysis_sync_repo_to_hpc.sh`
- `scripts/analysis_sync_launch_to_hpc.sh`
- `scripts/analysis_submit_hpc_pipeline.sh`
- `scripts/analysis_show_hpc_progress.sh`
- `scripts/analysis_pull_hpc_outputs.sh`
- `scripts/rq1_scientific_lmm_revamp_submit_hpc.sh`
- `scripts/rq2_random_slopes_submit_hpc.sh`
- `scripts/rq2_random_slopes_pull_hpc.sh`

Do not rely on keeping a laptop process alive for canonical RQ1 or RQ2 fits.

## Fail-Fast Rules

- If `build-analysis-base` has not been run, fail at launch creation.
- If a dependency stage is incomplete, downstream stages must fail immediately.
- If a required cached grid/task output is missing, fail instead of silently recomputing in another stage.
- If neither a local cached recorder-metadata path nor `csv_hpc_path` is readable, metadata-backed stages must fail immediately.
- If a launch directory already exists, fail instead of mutating it in place.

## Cleanup Policy

- Keep retained artifacts self-contained.
- Delete stale launches, old generated outputs, caches, draft specs, and abandoned analysis branches.
- Keep tests only where they protect retained behavior.
- Update docs and manifests when removing a surface; do not leave commands that point at deleted code.

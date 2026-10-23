# Alpaca Spatiotemporal Analysis Backend

Python 3.12 package for preparing the analysis-base dataset and running the
retained H1 post-hum cohesion and H2 hum-timing/group-dispersion workflows.

## Install and test

From the `alpaca-analysis` repository root:

```bash
uv sync --project spectrotemporal-visualization-analysis-subset/backend \
  --extra dev --locked
uv run --project spectrotemporal-visualization-analysis-subset/backend \
  pytest spectrotemporal-visualization-analysis-subset/backend/tests
```

## External inputs

No real research dataset is included. Copy `config/default.toml` to an
untracked local path and replace its template paths and, if needed, remote/HPC
settings. The backend requires the separately deposited cleaned corpus,
metadata-backfill tables, recorder sidecars, and GPS records. `fetch-sidecars`
also requires SSH access to the configured raw-recording store.

## Prepare analysis inputs

```bash
CONFIG=/absolute/path/to/local.toml

uv run --project spectrotemporal-visualization-analysis-subset/backend \
  stv-backend --config "$CONFIG" build-dataset
uv run --project spectrotemporal-visualization-analysis-subset/backend \
  stv-backend --config "$CONFIG" build-analysis-base
uv run --project spectrotemporal-visualization-analysis-subset/backend \
  stv-backend --config "$CONFIG" verify-dataset
```

## Scientific launches

Create a launch with analysis ID `rq1-post-hum-cohesion` or
`rq2-hum-timing-group-dispersion`, then execute its stages in manifest order:

```bash
uv run --project spectrotemporal-visualization-analysis-subset/backend \
  stv-analysis --config "$CONFIG" launch create rq1-post-hum-cohesion \
  --label release-run --profile full \
  --input-manifest /path/to/analysis/base/manifest.json

uv run --project spectrotemporal-visualization-analysis-subset/backend \
  stv-analysis --config "$CONFIG" launch run-stage rq1-post-hum-cohesion \
  --launch-dir /path/to/launch --stage units
```

The full stage order and compute classification are defined in the launch
manifest and documented in `../PRIMER.md` and `../docs/`. H2's
`scientific_glmm` stage and related final fits require R packages including
`glmmTMB` and `DHARMa`. Canonical full runs also require the configured SLURM
environment; unit and integration tests do not.

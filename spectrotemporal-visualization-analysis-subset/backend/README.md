# Thesis Analysis Code

This folder contains the code for the H1 and H2 analysis tables.

It is a trimmed code copy. It keeps the backend, scripts, tests, and analysis docs. It does not include the old web app.

## What This Code Reads

| Input | Path |
|---|---|
| Cleaned corpus | `Published\Data\Corpus construction` |
| Analysis base tables | `Published\Data\Analysis base` |
| H1 final outputs | `Published\Data\H1 Contact Reinforcement` |
| H2 final outputs | `Published\Data\H2 Contact Maintenance` |

## Install

From the `alpaca-analysis` repository root:

```bat
conda activate alpaca-thesis
python -m pip install -e "spectrotemporal-visualization-analysis-subset/backend[dev]"
```

## Test

```bat
python -m pytest spectrotemporal-visualization-analysis-subset\backend\tests
```

## Small Included-Data Check

The easiest check is the top-level helper script:

```bat
python Code\Tools\run_small_real_data_check.py --rows 8
```

It builds small H1 and H2 model-input tables from included real data. It stops before full statistical model fitting.

## Full Python Stage Order

Use this only for a serious rerun. It can take a long time.

H1 stages:

`units`, `filtered_trace_grids`, `filtered_window_inventory`, `blocker_scan`, `blocker_merge`, `raw_trace_grids`, `window_validity`, `centroid_distance`, `baseline_covariates`, `covariate_table`, `matched_controls`, `model_input`, `reports`

H2 stages:

`units`, `gps_filter_audit`, `trace_grids`, `window_grid`, `dispersion_features`, `hum_rate_features`, `model_input`, `scientific_glmm`, `diagnostics_outputs`, `reports`

Use the same `stv-analysis ... launch run-stage ... --stage STAGE_NAME` pattern shown in the smoke run.

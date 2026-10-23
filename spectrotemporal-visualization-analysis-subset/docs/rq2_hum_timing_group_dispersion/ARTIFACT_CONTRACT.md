# RQ2 Hum Timing / Group Dispersion Artifact Contract

RQ2 is a 60 s count-GLMM workflow for the association between alpaca hum counts and observed collared-group spread.

## Identity

- `analysis_id`: `rq2-hum-timing-group-dispersion`
- `analysis_version`: `5`
- `output_dirname`: `rq2_hum_timing_group_dispersion`
- Canonical root: `backend/data/processed/analysis/rq2_hum_timing_group_dispersion/`
- Launch root: `backend/data/processed/analysis/rq2_hum_timing_group_dispersion/launches/<launch_label>/`
- Local convenience alias: `backend/data/processed/analysis/rq2_hum_timing_group_dispersion/latest`

## Scope

- Window length: `60` seconds only.
- Response: `hum_count`.
- Offset: `offset(log(window_length_minutes))`.
- Fixed predictors: `area_m2`, `spread_change_m2_per_min`, `speed_mps`, `centroid_change_in_position_m`.
- Model families: Poisson and negative binomial NB1, both fitted with `glmmTMB`.
- Fixed-effect surfaces: interaction and additive spread/spread-change variants.
- Random effects: session-level random intercept plus maximal session-level random slopes matching each fixed-effect surface (`rq2_session_maximal_random_slopes_v1`).
- Diagnostics: DHARMa tests, plus residual-vs-window and residual ACF plots.

## Stage Graph

All RQ2 stages are `heavy_compute`:

```text
units -> gps_filter_audit -> trace_grids -> window_grid -> dispersion_features -> hum_rate_features -> model_input -> scientific_glmm -> diagnostics_outputs -> reports
```

`scientific_glmm` runs `backend/src/stv_analysis/rq2_hum_timing_group_dispersion/local_glmmtmb_60s_workflow.R`. The current fitted results were generated locally in the MacBook Air M2 environment, not as an HPC model fit.

## Model Formulas

Interaction surface:

```text
hum_count ~ z_area * z_spread_change + z_speed + z_centroid_change
  + offset(log(window_length_minutes))
  + (1 + z_area * z_spread_change + z_speed + z_centroid_change | session_id)
```

Additive surface:

```text
hum_count ~ z_area + z_spread_change + z_speed + z_centroid_change
  + offset(log(window_length_minutes))
  + (1 + z_area + z_spread_change + z_speed + z_centroid_change | session_id)
```

Both surfaces are fitted as Poisson and NB1. Coefficient p-values are the conditional-model Wald z tests from `glmmTMB` summaries.

## Required Tables

Canonical tables live in `tables/`.

- `rq2_units.parquet`
- `variable_feasibility_log.csv`
- `rq2_gps_residual_outlier_audit.csv`
- `rq2_gps_residual_outlier_points.parquet`
- `rq2_gps_qc_trace_point_summary.csv`
- `rq2_paddock_scale_sanity.csv`
- `rq2_window_grid.parquet`
- `rq2_window_coverage_summary.csv`
- `hard_gps_filter_summary.csv`
- `rq2_dispersion_features.parquet`
- `rq2_dispersion_feature_summary.csv`
- `rq2_dispersion_change_summary.csv`
- `rq2_hum_rate_features.parquet`
- `rq2_hum_rate_summary.csv`
- `rq2_model_input.parquet`
- `rq2_model_input.csv`
- `rq2_model_input_summary.csv`
- `rq2_model_formulas.csv`
- `rq2_predictor_standardization.csv`
- `rq2_model_run_manifest.csv`
- `rq2_fixed_effects.csv`
- `rq2_dharma_standard_tests.csv`
- `rq2_dharma_temporal_autocorrelation.csv`
- `rq2_scaled_residuals.csv.gz`
- `rq2_residual_acf.csv`
- `rq2_figure_index.csv`
- `rq2_environment.csv`
- `rq2_data_availability.csv`
- `rq2_filtering_attrition.csv`
- `rq2_model_diagnostics.csv`
- `rq2_diagnostic_figure_index.csv`
- `rq2_final_output_index.csv`

Required launch-local cache:

- `cache/gps_filter_audit/trace_points_rq2_gps_qc.parquet`
- `cache/gps_filter_audit/residual_gps_filter_manifest.json`
- `cache/trace_grids/<session-token>.npz`
- `cache/scientific_glmm/objects/rq2_fit_<model_id>.rds`
- `cache/scientific_glmm/objects/rq2_dharma_<model_id>.rds`
- `cache/scientific_glmm/rq2_glmmtmb_stdout.log`
- `cache/scientific_glmm/rq2_glmmtmb_stderr.log`

## Required Figures

Canonical figures live in `figures/`.

- `rq2_gps_qc_cloud.png`
- `rq2_gps_session_hull_audit.png`
- `rq2_hum_count_distribution.png`
- `rq2_area_distribution.png`
- `rq2_spread_change_distribution.png`
- `rq2_speed_distribution.png`
- `rq2_centroid_position_change_distribution.png`
- `rq2_dharma_<model_id>.png`
- `rq2_residuals_vs_window_<model_id>.png`
- `rq2_residual_acf_<model_id>.png`
- `rq2_fixed_effects_incidence_rate_ratios.png`

`<model_id>` is one of `poisson_interaction`, `poisson_additive`, `nb1_interaction`, or `nb1_additive`.

## Required Reports

Canonical runtime reports live in `reports/`.

- `subaim_1_units.md`
- `subaim_2_gps_audit.md`
- `subaim_3_window_grid.md`
- `subaim_4_dispersion_features.md`
- `subaim_5_hum_counts.md`
- `subaim_6_model_input.md`
- `rq2_methods_results.md`
- `rq2_session_info.txt`
- `subaim_8_diagnostics.md`
- `README.md`

The final launch root must also contain `manifest.json`.

## Runtime Manifest Fields

`manifest.json` records:

- `analysis_id`
- `analysis_version`
- `launch_label`
- `profile`
- `selected_sessions`
- `stage_status`
- `canonical_artifact_paths`
- `row_counts`
- `documented_paddock_dimensions_m`
- `documented_paddock_area_m2`
- `existing_hard_gps_filter_version`
- `residual_gps_filter_version`
- `model_input_path`
- `glmmtmb_model_runs`
- `predictor_standardization`
- `dharma_standard_tests`
- `input_manifest_path`
- `specs_manifest`

## Claim Boundary

RQ2 artifacts preserve these interpretation boundaries:

- observed collared subnetwork, not full herd
- recorder-level acoustic event proximity, not confirmed caller identity
- hum-count association, not causation
- temporal autocorrelation is acknowledged as a likely type-I-error inflation risk
- the NB1 fits are the overdispersion sensitivity check used to qualify Poisson output

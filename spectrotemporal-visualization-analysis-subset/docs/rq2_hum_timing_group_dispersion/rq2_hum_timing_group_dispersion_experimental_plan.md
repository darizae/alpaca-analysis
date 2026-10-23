# RQ2 Operational Spec: Hum Timing And Observed Group Spread

## Research Question

Do alpaca hum counts in 60 s windows vary with observed collared-group spread and recent change in spread, after accounting for group member speed and centroid-position change?

The claim boundary is observational and deliberately narrow: results describe the collared subnetwork and recorder-event hum detections. They do not identify callers, represent the full herd, or establish causality.

## Current Decisions

- Use 60 s windows only.
- Fit Poisson first as the count-model baseline.
- Fit negative binomial NB1 as the overdispersion sensitivity model.
- Fit the full interaction surface and the additive companion surface where spread and spread change do not interact.
- Report both Poisson and NB1, and both interaction and additive surfaces.
- Use DHARMa diagnostics.
- Plot DHARMa residuals against within-session window number.
- Plot within-session residual autocorrelation and refer to those plots as supplementary diagnostic material.
- Run the model fits locally in the MacBook Air M2 environment; the model stage is not an HPC model-fit job.

## Data Construction

RQ2 consumes a launch-local copy of the analysis-base files:

```text
base_snapshot/recorders.parquet
base_snapshot/segments.parquet
base_snapshot/events.parquet
base_snapshot/trace_points_filtered.parquet
base_snapshot/gps_blocker_rule.json
base_snapshot/manifest.json
```

The analysis-base hard GPS filter is already represented by `trace_points_filtered.parquet`. RQ2 then creates a launch-local residual GPS-QC table:

```text
cache/gps_filter_audit/trace_points_rq2_gps_qc.parquet
```

Only non-overlapping 60 s windows are materialized. Terminal incomplete windows are discarded.

## Event Policy

Canonical hum events are:

- `alpaca_hum`
- `None`
- unlabeled non-explicit rows

Explicit exclusions are:

- `rumble`
- `human_talking`

Event assignment uses the event midpoint and the matching `day_session_id`.

## Model Variables

Response:

```text
hum_count
```

Exposure offset:

```text
offset(log(window_length_minutes))
```

Predictors:

- `area_m2`: median observed convex-hull area of the collared animals within the window.
- `spread_change_m2_per_min`: signed change in observed convex-hull area from the previous 60 s window, divided by one minute.
- `speed_mps`: mean collar speed within the window after excluding biologically impossible speed samples.
- `centroid_change_in_position_m`: Euclidean movement of the observed group centroid from the previous 60 s window.

All predictors are left on the raw ecological scale and then standardized to one SD units for fitting. Standardization parameters are written to `tables/rq2_predictor_standardization.csv`.

## Fitted Models

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

Each surface is fitted twice:

- Poisson with log link.
- NB1 with log link via `glmmTMB::nbinom1`.

Random effects use `rq2_session_maximal_random_slopes_v1`: the interaction surface includes the session-level random intercept and random slopes for `z_area`, `z_spread_change`, `z_area:z_spread_change`, `z_speed`, and `z_centroid_change`; the additive surface includes the same terms except the interaction slope.

The interaction model is retained because it was part of the full model discussed with the supervisor. The additive model is required because the individual effects of spread and spread change cannot be interpreted directly from the interaction model except at the zero value of the other interacting predictor.

## Diagnostics And Reporting

Required diagnostics:

- `DHARMa::simulateResiduals`
- `DHARMa::testUniformity`
- `DHARMa::testDispersion`
- `DHARMa::testZeroInflation`
- `DHARMa::testOutliers`
- `DHARMa::testTemporalAutocorrelation` within session
- residuals versus within-session window number
- within-session residual ACF plots

The autocorrelation diagnostics are reported as a limitation and supporting-information plot set. The interpretation follows the supervisor discussion: temporal autocorrelation is present in at least some sessions, its pattern is not uniform, it likely inflates type-I error, and correcting it directly is difficult enough that transparent reporting is preferable here.

## Stage Graph

```text
units
gps_filter_audit
trace_grids
window_grid
dispersion_features
hum_rate_features
model_input
scientific_glmm
diagnostics_outputs
reports
```

Dependency chain:

```text
units -> gps_filter_audit -> trace_grids -> window_grid -> dispersion_features -> hum_rate_features -> model_input -> scientific_glmm -> diagnostics_outputs -> reports
```

## Required Output Surface

The canonical artifact list is maintained in:

```text
docs/rq2_hum_timing_group_dispersion/ARTIFACT_CONTRACT.md
```

The main handoff files are:

```text
tables/rq2_model_input.parquet
tables/rq2_model_input.csv
tables/rq2_model_run_manifest.csv
tables/rq2_fixed_effects.csv
tables/rq2_dharma_standard_tests.csv
tables/rq2_dharma_temporal_autocorrelation.csv
tables/rq2_scaled_residuals.csv.gz
tables/rq2_residual_acf.csv
tables/rq2_environment.csv
reports/rq2_methods_results.md
manifest.json
```

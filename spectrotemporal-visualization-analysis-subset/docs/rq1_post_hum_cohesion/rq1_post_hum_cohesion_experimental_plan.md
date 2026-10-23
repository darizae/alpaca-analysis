# RQ1 Post-Hum Cohesion Experimental Plan

This plan defines the current RQ1 scientific LMM analysis and supplementary
whole-subnetwork threshold analysis, implemented through
`backend/src/stv_analysis/rq1_post_hum_cohesion/scientific_lmm_guide_workflow.R`.

## Biological Question

Are hum windows followed by increased cohesion between the focal alpaca and the
other observed collared alpacas?

The response is the post-window change in the focal alpaca's distance to the
centroid of other observed collared alpacas:

```text
Delta_D_cm = Delta_D_w * 100
```

Negative hum-window estimates indicate that the focal alpaca is closer to the
observed centroid after hum windows than after matched no-hum control windows,
conditional on the retained pre-event covariates.

This is an observed-collared-subnetwork claim. It is not a full-herd claim and
does not identify the vocalizing animal unless caller identity is independently
known.

## Canonical Data

The primary scientific LMM uses one materialized model view:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/data/
  h1_contact_reinforcement_model_input.csv
  h1_contact_reinforcement_model_input.parquet
```

Only the standard H1 model input is materialized for fitting. `match_set_id`
remains a traceability column but is not fitted as a random effect.

Event-label policy:

```text
alpaca_hum and None are treated as alpaca hums.
rumble is excluded.
```

The hard GPS spatial filter is part of upstream materialization and is audited
through the retained materialized-view report:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/
  tables/rq1_lmm_hard_gps_filter_summary.csv
  reports/rq1_lmm_materialized_views.md
```

## Primary Model Set

The additive primary analysis fits exactly three window-length models:

```text
h1_30s
h1_60s
h1_120s
```

W60 is the primary biological timescale. W30 and W120 check shorter and broader
post-event response windows over the same scientific model surface.

## Fixed Effects

The additive fixed-effect surface is:

```text
condition_hum
starting_centroid_distance_m
mean_speed_pre_m_s
pre_delta_centroid_distance_m
observed_polygon_area_pre_m2
recent_change_in_polygon_area_m2
```

Excluded from materialized model views, fixed effects, and random slopes:

```text
time_of_day_minutes
local_collared_density_per_1000m2
n_collars_pre
collar_availability
minutes_since_session_start
```

The retained predictors can remain correlated. They are not selected by AIC and
are not pruned for collinearity unless the scientific interpretation requires a
documented post-fit caveat.

Reader-facing reporting should describe `observed_polygon_area_pre_m2` as
pre-event spread and `recent_change_in_polygon_area_m2` as recent spread change.
The implementation retains the existing code column names.

## Transformations

Use the same predictor transformation policy for W30, W60, and W120:

| Predictor | Transformation |
|---|---|
| `starting_centroid_distance_m` | `log1p`, then z-score |
| `mean_speed_pre_m_s` | `log1p`, then z-score |
| `pre_delta_centroid_distance_m` | z-score only |
| `observed_polygon_area_pre_m2` | z-score only |
| `recent_change_in_polygon_area_m2` | z-score only |

`observed_polygon_area_pre_m2` is never log-transformed under any window.

The response is expressed in centimeters for numerical scaling and is not
distributionally transformed. Transformation decisions and distribution plots
are written per output tree:

```text
tables/rq1_lmm_transform_decisions.csv
figures/<model_id>_model_distributions.pdf
```

## Random Effects

The base grouping factors are exactly:

```text
focal_id
session_id
```

The workflow uses maximized correlated random slopes selected by the scientific
guide's within-group variation rules. For the additive surface, each retained
fixed effect is screened for slope identifiability within `focal_id` and
`session_id`.

The implemented formula has this form:

```text
Delta_D_cm
  ~ condition_hum + transformed retained covariates
  + (1 + guide-selected correlated slopes | focal_id)
  + (1 + guide-selected correlated slopes | session_id)
```

Random-slope decisions are stored in:

```text
tables/rq1_lmm_random_slope_decisions.csv
```

## Model Tests

For each window:

```text
Full ML model: condition + retained covariates + random-effects structure
Null ML model: retained covariates + same random-effects structure
Test: likelihood-ratio test for the hum-window condition effect
Coefficient table: REML lmerTest model with Satterthwaite tests
```

Core post-fit outputs:

```text
tables/rq1_lmm_full_null_tests.csv
tables/rq1_lmm_fixed_effects_lmerTest_REML.csv
tables/rq1_lmm_vif.csv
tables/rq1_lmm_random_effect_variance.csv
tables/rq1_lmm_residual_summary.csv
tables/rq1_lmm_model_stability_summary.csv
tables/rq1_lmm_model_stability_detailed.csv
figures/<model_id>_residual_diagnostics.pdf
figures/<model_id>_random_effect_histograms.pdf
figures/<model_id>_random_effect_qq.pdf
figures/<model_id>_fixed_effect_stability.pdf
```

## Predictor AIC Diagnostics

For the primary analysis output trees only, the workflow computes AIC scores for
fixed-effect change diagnostics. Each candidate is fit by ML on the same rows as
the intended model. When a fixed effect is removed, its random slopes are also
removed from the otherwise retained random-effect structure.

The diagnostic table is:

```text
tables/rq1_lmm_predictor_aic_scores.csv
```

These AIC scores do not select the scientific model. The intended model design
is retained irrespective of the AIC ranking. Predictor AIC diagnostics are not
computed for the supplementary threshold analysis.

## Supplementary Whole-Subnetwork Threshold Analysis

The threshold supplement uses one frozen global threshold:

```text
T = 49.358079 m
```

For each primary hum or control row, candidate companions are scored from the
pre-event interval using their median leave-one-candidate-out distance to the
other non-focal collared companions. Companions with finite scores at or below
the frozen threshold are retained. The focal alpaca is never part of the
retained companion set.

Companion membership is selected only from the pre-event interval and is then
frozen across the previous, pre-event, and post-event intervals. The primary
hum/control rows, match-set identifiers, and control assignments are preserved;
controls are not regenerated, replaced, or rematched.

Threshold geometry is recalculated from the retained companions:

```text
centroid distances: focal alpaca to centroid of retained non-focal companions
spread: focal alpaca plus retained companions
pre-event companion minimum: 2 retained companions
```

Rows are excluded from the threshold model input for fewer than two retained
companions, missing recalculated centroid coverage, missing recalculated spread
coverage, or another non-finite required model variable. Exclusion reasons are
recorded for every excluded row.

The threshold materialized model input is:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/threshold_supplement/data/
  h1_threshold_filtered_model_input.csv
  h1_threshold_filtered_model_input.parquet
```

Threshold run configuration:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/threshold_supplement/tables/
  rq1_threshold_scientific_lmm_run_config.csv
```

Threshold model IDs:

```text
threshold_49m_h1_30s
threshold_49m_h1_60s
threshold_49m_h1_120s
```

## HPC Orchestration

Canonical submitter:

```text
scripts/rq1_scientific_lmm_revamp_submit_hpc.sh
```

The submitter materializes data and fits all models on HPC. It creates a single
bundle under:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/hpc_bundle/
```

Resource policy:

```text
materialization jobs: 4 CPUs, 32-64G memory, 03:00:00 wall time
smoke jobs: 2 CPUs, 16G memory, 00:30:00 wall time
fit arrays and failovers: 4 CPUs, 64G memory, 48:00:00 wall time
aggregate jobs: 2 CPUs, 16G memory, 01:00:00 wall time
optimizer: nloptwrap
checkpoint cadence: every 25 optimizer iterations
```

Execution sequence:

```text
1. materialize primary H1 model input
2. primary additive smoke
3. primary additive full fit array
4. primary additive failover array
5. primary additive aggregate with predictor AIC diagnostics
6. materialize threshold supplement
7. threshold additive smoke
8. threshold additive full fit array
9. threshold additive failover array
10. threshold additive aggregate
```

## Active HPC Submission

Partition probe on July 29, 2026:

```text
compute materialize/fit test-only start: 2026-07-31 02:04:49
compute:h100: invalid SLURM partition name
gpu materialize/fit test-only start: 2026-07-29 12:36:49
```

The active chain uses `gpu`:

```text
bundle: rq1_predictor_threshold_revamp_20260729T104008Z
remote bundle:
  /cluster/projects/alpaca/spectrotemporal-visualization/backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/hpc_bundle/rq1_predictor_threshold_revamp_20260729T104008Z
```

SLURM jobs:

```text
primary_additive_smoke:          15096117
primary_additive_array:          15096118_[1-3]
primary_additive_failover:       15096119_[1-3]
primary_additive_aggregate:      15096120
threshold_materialize:           15096121
threshold_additive_smoke:        15096122
threshold_additive_array:        15096123_[1-3]
threshold_additive_failover:     15096124_[1-3]
threshold_additive_aggregate:    15096125
```

Audit status at July 30, 2026 09:39 Europe/Berlin:

```text
retained primary model input was present and validated.
primary additive smoke, full array, failover, and aggregate completed.
threshold materialization, smoke, full array, and aggregate completed.
threshold additive failover 15096124 was cancelled after all threshold cache
  COMPLETE markers were verified; it was only blocking aggregate while queued.
```

## Reporting Language

Use:

```text
hum-window effect
observed collared subnetwork
centroid of the other observed collared alpacas
smaller post-window increase in centroid distance
maintained or increased cohesion with the observed centroid
whole-subnetwork threshold sensitivity analysis
```

Avoid:

```text
hums caused cohesion
the herd became more cohesive
the caller moved toward the group
the observed centroid was the full herd centroid
the lowest-AIC model was selected
the threshold identified a true social boundary
```

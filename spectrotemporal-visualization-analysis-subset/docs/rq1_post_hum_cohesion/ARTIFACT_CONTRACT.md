# RQ1 Post-Hum Cohesion Artifact Contract

This contract defines the current RQ1 scientific LMM artifacts for the primary
observed-centroid analysis and the supplementary whole-subnetwork threshold
analysis. The fitted R workflow is
`backend/src/stv_analysis/rq1_post_hum_cohesion/scientific_lmm_guide_workflow.R`.

## Materialized Data

Primary H1 model input:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/data/
  h1_contact_reinforcement_model_input.csv
  h1_contact_reinforcement_model_input.parquet
```

Primary materialization audit:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/
  materialization_manifest.json
  tables/rq1_lmm_materialized_view_summary.csv
  tables/rq1_lmm_event_label_audit.csv
  tables/rq1_lmm_hard_gps_filter_summary.csv
  reports/rq1_lmm_materialized_views.md
```

Threshold supplement model view:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/threshold_supplement/data/
  h1_threshold_filtered_model_input.csv
  h1_threshold_filtered_model_input.parquet
```

Threshold materialization audit:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/threshold_supplement/
  threshold_materialization_manifest.json
  tables/rq1_threshold_row_comparison.csv.gz
  tables/rq1_threshold_model_input_summary.csv
  tables/rq1_threshold_retained_companion_distribution.csv
  tables/rq1_threshold_geometry_exclusions.csv
  tables/rq1_threshold_matched_set_completeness.csv
  tables/rq1_threshold_pre_event_balance.csv
  tables/rq1_threshold_raw_change_summary.csv
  tables/rq1_threshold_scientific_lmm_run_config.csv
  reports/rq1_threshold_materialized_views.md
```

Further convergence breakdown supplement:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/further_convergence_breakdown/
  manifest.json
  tables/rq1_further_convergence_pair_level.csv
  tables/rq1_further_convergence_case_summary.csv
  tables/rq1_further_convergence_convergent_magnitude_summary.csv
  tables/rq1_further_convergence_matched_set_audit.csv
  tables/h1_directional_cases_w30.csv
  tables/h1_directional_cases_w60.csv
  tables/h1_directional_cases_w120.csv
  figures/cases_breakdown_w30.png
  figures/cases_breakdown_w60.png
  figures/cases_breakdown_w120.png
  figures/convergent_cases_breakdown_w30.png
  figures/convergent_cases_breakdown_w60.png
  figures/convergent_cases_breakdown_w120.png
  figures/cases_breakdown.png
  figures/convergent_cases_breakdown.png
  reports/rq1_further_convergence_breakdown.md
```

This supplement is produced from the frozen primary model input without
rerunning upstream materialization:

```bash
PYTHONPATH=backend/src python -m stv_analysis.rq1_post_hum_cohesion.further_convergence_breakdown
```

Its paired comparisons require exactly one hum row and one control row for a
given `window_s` x `match_set_id`. W60 is the default plot window because it is
the primary biological timescale.

The model-input views must not contain:

```text
time_of_day_minutes
local_collared_density_per_1000m2
n_collars_pre
collar_availability
minutes_since_session_start
```

`match_set_id` may be present for traceability. It is not used in any model
formula.

## Output Trees

Primary additive output:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/
  h1_model_outputs/
```

Threshold additive output:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/threshold_supplement/
  r_outputs_threshold_global_standard_matching_standard_model_nloptwrap/
```

The HPC submission bundle is written under:

```text
backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/hpc_bundle/
```

Each bundle must include `submission.json` with the host, partition, account,
remote bundle directory, model-output directories, frozen threshold, retained
and excluded predictors, dependency sequence, and SLURM job IDs.

Active submission bundle:

```text
bundle: rq1_predictor_threshold_revamp_20260729T104008Z
partition: gpu
remote bundle:
  /cluster/projects/alpaca/spectrotemporal-visualization/backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/hpc_bundle/rq1_predictor_threshold_revamp_20260729T104008Z
```

Active SLURM jobs:

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
primary additive path completed through aggregate.
threshold additive path completed through aggregate.
threshold additive failover 15096124 was cancelled after all threshold caches
  were verified complete, releasing the aggregate.
```

## Model IDs

Primary additive:

```text
h1_30s
h1_60s
h1_120s
```

Threshold additive:

```text
threshold_49m_h1_30s
threshold_49m_h1_60s
threshold_49m_h1_120s
```

## Model Surface

All fitted models use:

```text
response: Delta_D_cm = Delta_D_w * 100
family/link: Gaussian identity LMM
optimizer: nloptwrap
control matching: standard_matching
match-set random effect: omitted
random-effect grouping factors: focal_id and session_id
random slopes: maximized correlated slopes selected by guide within-group variation rules
predictor set: rq1_predictor_revamp
```

Additive fixed effects:

```text
condition_hum
starting_centroid_distance_m
mean_speed_pre_m_s
pre_delta_centroid_distance_m
observed_polygon_area_pre_m2
recent_change_in_polygon_area_m2
```

Reader-facing reports should describe `observed_polygon_area_pre_m2` as
pre-event spread and `recent_change_in_polygon_area_m2` as recent spread change.
The artifact column names remain unchanged.

Predictor transformations:

| Predictor | Transformation |
|---|---|
| `starting_centroid_distance_m` | `log1p`, then z-score |
| `mean_speed_pre_m_s` | `log1p`, then z-score |
| `pre_delta_centroid_distance_m` | z-score only |
| `observed_polygon_area_pre_m2` | z-score only |
| `recent_change_in_polygon_area_m2` | z-score only |

`observed_polygon_area_pre_m2` is never log-transformed.

## Required Fit Tables

Each output tree must contain:

```text
tables/rq1_lmm_run_manifest.csv
tables/rq1_lmm_full_null_tests.csv
tables/rq1_lmm_fixed_effects_lmerTest_REML.csv
tables/rq1_lmm_transform_decisions.csv
tables/rq1_lmm_random_slope_decisions.csv
tables/rq1_lmm_random_effect_variance.csv
tables/rq1_lmm_residual_summary.csv
tables/rq1_lmm_random_effect_summary.csv
tables/rq1_lmm_vif.csv
tables/rq1_lmm_model_stability_summary.csv
tables/rq1_lmm_model_stability_detailed.csv
tables/rq1_lmm_fit_checkpoint_audit.csv
```

Primary output trees must also contain:

```text
tables/rq1_lmm_predictor_aic_scores.csv
```

The AIC table is a predictor-diagnostic artifact only. It is not used to select
the scientific model. Threshold output trees must not be treated as requiring
predictor AIC diagnostics.

## Required Figures

Each model must have:

```text
figures/<model_id>_model_distributions.pdf
figures/<model_id>_residual_diagnostics.pdf
figures/<model_id>_random_effect_histograms.pdf
figures/<model_id>_random_effect_qq.pdf
figures/<model_id>_fixed_effect_stability.pdf
```

## Required Cache

Each model cache must contain:

```text
cache/<model_id>/COMPLETE
cache/<model_id>/result.rds
cache/<model_id>/full_ml_fit.rds
cache/<model_id>/null_ml_fit.rds
cache/<model_id>/full_reml_fit.rds
cache/<model_id>/*_optimizer_trace.rds
cache/<model_id>/*_optimizer_trace_progress.csv
cache/<model_id>/<model_id>_residuals.csv.gz
cache/<model_id>/<model_id>_random_effects.csv.gz
cache/<model_id>/stability_refits/
```

Caches are reusable only for the same materialized data, same fixed-effect
variant, same random-effect design, same optimizer settings, and same code
semantics.

## Threshold Contract

The supplementary threshold analysis uses exactly:

```text
threshold_m: 49.358079
threshold_role: global_outlier_filtered
pre-event retained companion minimum: 2
```

Threshold model rows preserve primary hum/control row identities and control
assignments. Controls are not regenerated, replaced, or rematched.

Companion selection is based on pre-event leave-one-candidate-out median
distance to other non-focal companions. Retained companion membership is frozen
across previous, pre-event, and post-event intervals, and centroid distance and
spread predictors are recalculated from the retained subnetwork.

Every excluded threshold row must have a non-empty
`threshold_exclusion_reason`.

## HPC Sequence

The canonical orchestration is:

```text
scripts/rq1_scientific_lmm_revamp_submit_hpc.sh
```

The dependency chain runs:

```text
retained primary model-input sync and validation
primary additive smoke, array, failover, aggregate
threshold materialization
threshold additive smoke, array, failover, aggregate
```

Threshold materialization runs on a CPU SLURM job with a three-hour limit. Fit
arrays and failover arrays use 48-hour limits and resumable caches.

## Reporting Boundary

Report the additive primary models as the planned RQ1 analysis. Report the
threshold supplement as a spatial-reference sensitivity analysis, not as a true
social-boundary estimator.

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stv_analysis.framework import create_launch, load_launch_context, prepare_hpc
from stv_analysis.registry import get_analysis_definition
from stv_analysis.rq1_post_hum_cohesion.builder import _segments_with_sidecars
from stv_analysis.rq1_post_hum_cohesion.contracts import SessionTraceGrid
from stv_analysis.rq1_post_hum_cohesion.contracts import STAGE_IDS
from stv_analysis.rq1_post_hum_cohesion.subaim1_units import build_analysis_units
from stv_analysis.rq1_post_hum_cohesion.subaim2_windows import distance_arrays
from stv_analysis.rq1_post_hum_cohesion.subaim3_validity import decorate_window_validity
from stv_analysis.rq1_post_hum_cohesion.subaim4_centroid import (
    directional_movement_components,
)
from stv_analysis.rq1_post_hum_cohesion.scientific_lmm_materialize import (
    EXCLUDED_PREDICTOR_COLUMNS as PRIMARY_EXCLUDED_PREDICTOR_COLUMNS,
    KEEP_COLUMNS as PRIMARY_KEEP_COLUMNS,
    normalize_model_input,
)
from stv_analysis.rq1_post_hum_cohesion.further_convergence_breakdown import (
    build_further_convergence_breakdown,
    build_pair_level_table,
    classify_case,
    summarize_convergent_magnitude,
)
from stv_analysis.rq1_post_hum_cohesion.whole_subnetwork_threshold import (
    GLOBAL_OUTLIER_THRESHOLD_M,
    EXCLUDED_PREDICTOR_COLUMNS as THRESHOLD_EXCLUDED_PREDICTOR_COLUMNS,
    _assert_no_excluded_predictors,
    companion_outlier_scores,
    threshold_features_for_primary_row,
)
from stv_backend.analysis_base import build_analysis_base
from stv_backend.config import BackendConfig
from stv_backend.pipeline import build_dataset

from test_pipeline_smoke import (
    _write_fixture_wav,
)


def _build_config(tmp_path: Path) -> BackendConfig:
    fixture_root = tmp_path / "fixtures"
    shutil.copytree(Path(__file__).parent / "fixtures", fixture_root)
    _write_fixture_wav(
        fixture_root
        / "raw_corpus_1"
        / "cleaned_alpaca_hum_corpus"
        / "review_concat.wav"
    )
    blocker_path = tmp_path / "gps_blocker.json"
    blocker_path.write_text(
        json.dumps(
            {
                "name": "fixture blocker",
                "description": "fixture",
                "selection_source": "test",
                "buffer_meters": 5,
                "points": [[138.689, -34.525]],
            }
        ),
        encoding="utf8",
    )
    return BackendConfig(
        dataset_name="fixture-dataset",
        timezone_name="Australia/Adelaide",
        raw_corpus_dir=fixture_root / "raw_corpus_1",
        sidecar_cache_dir=fixture_root / "sidecars",
        processed_dir=tmp_path / "processed",
        intermediate_dir=tmp_path / "processed" / "intermediate",
        gps_blocker_path=blocker_path,
        ssh_host="",
        canonical_raw_root="/projects/example",
        config_path=tmp_path / "config.toml",
    )


def test_build_analysis_units_uses_event_midpoint() -> None:
    events = pd.DataFrame(
        [
            {
                "review_row_key": "event-1",
                "recorder_id": "recorder-A",
                "recorder_display_name": "A",
                "device_id": "device-A",
                "reviewed_resolved_label_semantic": "alpaca_hum",
                "timestamp_start_local": pd.Timestamp("2025-02-03T14:00:00+10:30"),
                "timestamp_end_local": pd.Timestamp("2025-02-03T14:00:10+10:30"),
                "date_local": "2025-02-03",
                "session_id": "session-1",
                "day_session_id": "2025-02-03:session-001",
                "seconds_since_midnight_local": 50_400.0,
            }
        ]
    )
    recorders = pd.DataFrame(
        [
            {
                "recorder_id": "recorder-A",
                "recorder_display_name": "A",
                "device_id": "device-A",
            }
        ]
    )
    units, summary, feasibility = build_analysis_units(events, recorders)
    assert units.iloc[0]["event_midpoint_seconds_local"] == 50_405.0
    assert summary["missing_focal_mapping"] == 0
    assert not feasibility.empty


def test_distance_arrays_use_only_other_observed_collars() -> None:
    x_m = np.array([[0.0, 10.0, 20.0]])
    y_m = np.zeros_like(x_m)
    nonfocal_count, loo_distance = distance_arrays(
        x_m, y_m, np.ones_like(x_m, dtype=bool)
    )
    assert nonfocal_count.tolist() == [[2, 2, 2]]
    assert loo_distance.tolist() == [[15.0, 0.0, 15.0]]


def test_directional_movement_decomposition_categories() -> None:
    recorder_ids = ("A", "B", "C", "D")
    x = np.array([[0, 10, 10, 10], [1, 9, 9, 9]], dtype=float)
    y = np.zeros_like(x)
    valid = np.ones_like(x, dtype=bool)
    nonfocal, loo = distance_arrays(x, y, valid)
    grid = SessionTraceGrid(
        day_session_id="session",
        recorder_ids=recorder_ids,
        recorder_index={value: index for index, value in enumerate(recorder_ids)},
        second_bins=np.array([0, 2]),
        bin_times_s=np.array([0.0, 2.0]),
        valid=valid,
        x_m=x,
        y_m=y,
        nonfocal_count=nonfocal,
        loo_distance_m=loo,
    )
    result = directional_movement_components(
        grid=grid, focal_id="A", t0_seconds_local=1.0, window_s=1
    )
    assert result["directional_category"] == "both moved toward each other"
    assert result["focal_distance_reduction_m"] == 1.0
    assert result["centroid_distance_reduction_m"] == 1.0
    assert result["directional_geometry_valid"]


def test_global_threshold_uses_pre_event_leave_one_candidate_out_scores() -> None:
    recorder_ids = ("A", "B", "C", "D")
    bin_times = np.arange(10, dtype=float)
    x = np.zeros((10, 4), dtype=float)
    y = np.zeros_like(x)
    x[:, 1:3] = 4.0
    y[:, 1] = -1.0
    y[:, 2] = 1.0
    x[:, 3] = 100.0
    x[2:5, 1:3] = 5.0
    x[6:9, 1:3] = 6.0
    x[6:9, 3] = 0.0
    valid = np.ones_like(x, dtype=bool)
    grid = SessionTraceGrid(
        day_session_id="session",
        recorder_ids=recorder_ids,
        recorder_index={value: index for index, value in enumerate(recorder_ids)},
        second_bins=np.arange(10, dtype=np.int32),
        bin_times_s=bin_times,
        valid=valid,
        x_m=x,
        y_m=y,
        nonfocal_count=np.full_like(x, 3, dtype=np.int16),
        loo_distance_m=np.full_like(x, np.nan, dtype=float),
    )

    scores = companion_outlier_scores(
        grid, focal_index=0, pre_indices=np.array([2, 3, 4])
    )

    assert np.isclose(scores[1], np.hypot(47.5, 1.5))
    assert np.isclose(scores[2], np.hypot(47.5, 1.5))
    assert scores[3] == 95.0

    row = SimpleNamespace(
        analysis_window_id="event-1:w3",
        condition=1,
        match_set_id="event-1",
        focal_id="A",
        focal_alpaca_id="A",
        day_session_id="session",
        session_id="session",
        date_local="2025-02-03",
        window_s=3,
        window_center_seconds_local=5.0,
        window_center_timestamp="2025-02-03T14:00:05+10:30",
        Delta_D_w=0.0,
        mean_speed_pre_m_s=0.2,
        matching_rule_used="fixture",
        buffer_rule_used="fixture",
        buffer_seconds=0.0,
        match_rank=0,
        control_match_use_count=0,
        standardized_match_distance=0.0,
    )

    feature = threshold_features_for_primary_row(
        row=row,
        grid=grid,
        threshold_m=GLOBAL_OUTLIER_THRESHOLD_M,
        hard_gps_filter_version="fixture-filter",
    )

    assert feature["threshold_model_row"]
    assert feature["threshold_exclusion_reason"] == ""
    assert feature["retained_companion_count_pre"] == 2
    assert set(feature["retained_companion_ids"].split("|")) == {"B", "C"}
    assert "D" not in feature["retained_companion_ids"].split("|")
    assert feature["pre_distance_loo_centroid_m"] == 5.0
    assert feature["post_distance_loo_centroid_m"] == 6.0
    assert feature["pre_delta_centroid_distance_m"] == 1.0
    assert feature["observed_polygon_area_pre_m2"] == 5.0
    assert feature["recent_change_in_polygon_area_m2"] == 1.0
    assert feature["hard_gps_filter_version"] == "fixture-filter"


def test_primary_revamp_model_view_drops_excluded_predictors() -> None:
    row = {column: "fixture" for column in PRIMARY_KEEP_COLUMNS}
    row.update(
        {
            "condition": 1,
            "window_s": 60,
            "window_length_s": 60,
            "Delta_D_w": 0.12,
            "session_id": "session",
            "day_session_id": "session",
            "focal_id": "A",
            "match_set_id": "set-1",
            "analysis_window_id": "event-1:w60",
            "window_center_seconds_local": 10.0,
            "match_rank": 0,
            "control_match_use_count": 0,
            "standardized_match_distance": 0.0,
            "distance_sample_count_pre": 3,
            "distance_sample_count_post": 3,
            "starting_centroid_distance_m": 5.0,
            "mean_speed_pre_m_s": 0.2,
            "pre_delta_centroid_distance_m": 1.0,
            "observed_polygon_area_pre_m2": 5.0,
            "recent_change_in_polygon_area_m2": 1.0,
        }
    )
    for column in PRIMARY_EXCLUDED_PREDICTOR_COLUMNS:
        row[column] = 999.0

    view = normalize_model_input(pd.DataFrame([row]))

    assert not (set(PRIMARY_EXCLUDED_PREDICTOR_COLUMNS) & set(view.columns))
    assert view.iloc[0]["Delta_D_cm"] == 12.0


def test_threshold_model_contract_rejects_excluded_predictors() -> None:
    valid = pd.DataFrame(
        [
            {
                "condition_hum": 1,
                "starting_centroid_distance_m": 5.0,
                "mean_speed_pre_m_s": 0.2,
                "pre_delta_centroid_distance_m": 1.0,
                "observed_polygon_area_pre_m2": 5.0,
                "recent_change_in_polygon_area_m2": 1.0,
            }
        ]
    )
    _assert_no_excluded_predictors(valid)

    invalid = valid.assign(time_of_day_minutes=600.0)
    try:
        _assert_no_excluded_predictors(invalid)
    except AssertionError as exc:
        assert "time_of_day_minutes" in str(exc)
    else:
        raise AssertionError("threshold model input accepted an excluded predictor")

    assert {
        "time_of_day_minutes",
        "local_collared_density_per_1000m2",
        "n_collars_pre",
        "collar_availability",
        "minutes_since_session_start",
    } <= THRESHOLD_EXCLUDED_PREDICTOR_COLUMNS


def test_further_convergence_case_classifier_orients_h1_contrast() -> None:
    assert (
        classify_case(hum_delta_D_m=4.0, control_delta_D_m=2.0)
        == "both_expand_hum_expands_more"
    )
    assert (
        classify_case(hum_delta_D_m=2.0, control_delta_D_m=4.0)
        == "both_expand_hum_expands_less"
    )
    assert (
        classify_case(hum_delta_D_m=-4.0, control_delta_D_m=4.0)
        == "hum_contracts_control_expands"
    )
    assert (
        classify_case(hum_delta_D_m=-4.0, control_delta_D_m=-2.0)
        == "both_contract_hum_contracts_more"
    )
    assert (
        classify_case(hum_delta_D_m=-2.0, control_delta_D_m=-4.0)
        == "both_contract_control_contracts_more"
    )
    assert (
        classify_case(hum_delta_D_m=4.0, control_delta_D_m=-4.0)
        == "hum_expands_control_contracts"
    )


def test_further_convergence_pair_table_uses_complete_match_sets() -> None:
    rows = [
        _convergence_row("set-1", 1, "hum-1", 6.0, 8.0),
        _convergence_row("set-1", 0, "control-1", 6.0, 10.0),
        _convergence_row("set-2", 1, "hum-2", 6.0, 2.0),
        _convergence_row("set-2", 0, "control-2", 6.0, 4.0),
        _convergence_row("set-3", 1, "hum-only", 6.0, 1.0),
    ]

    pairs = build_pair_level_table(pd.DataFrame(rows))
    magnitude = summarize_convergent_magnitude(pairs)

    assert pairs["match_set_id"].tolist() == ["set-1", "set-2"]
    assert pairs["case_id"].tolist() == [
        "both_expand_hum_expands_less",
        "both_contract_hum_contracts_more",
    ]
    assert pairs["supports_h1"].tolist() == [True, True]
    assert magnitude["negative_contrast_magnitude_m"].sum() == 4.0
    problematic = magnitude.loc[
        magnitude["convergent_magnitude_group"].eq("both_expand_hum_expands_less")
    ].iloc[0]
    assert problematic["percent_of_total_negative_contrast_magnitude"] == 50.0


def test_further_convergence_artifact_writes_tables_and_plots(tmp_path: Path) -> None:
    input_path = tmp_path / "model_input.csv"
    pd.DataFrame(
        [
            _convergence_row("set-1", 1, "hum-1", 6.0, 8.0),
            _convergence_row("set-1", 0, "control-1", 6.0, 10.0),
            _convergence_row("set-2", 1, "hum-2", 6.0, 2.0),
            _convergence_row("set-2", 0, "control-2", 6.0, 10.0),
            _convergence_row("set-3", 1, "hum-3", 6.0, 2.0),
            _convergence_row("set-3", 0, "control-3", 6.0, 4.0),
            _convergence_row("set-4", 1, "hum-4", 6.0, 4.0),
            _convergence_row("set-4", 0, "control-4", 6.0, 2.0),
            _convergence_row("set-5", 1, "hum-5", 6.0, 10.0),
            _convergence_row("set-5", 0, "control-5", 6.0, 2.0),
            _convergence_row("set-6", 1, "hum-6", 6.0, 10.0),
            _convergence_row("set-6", 0, "control-6", 6.0, 8.0),
        ]
    ).to_csv(input_path, index=False)

    result = build_further_convergence_breakdown(
        input_path=input_path,
        output_dir=tmp_path / "artifact",
        plot_window_s=60,
    )

    output_dir = Path(result["output_dir"])
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "tables/rq1_further_convergence_pair_level.csv").is_file()
    assert (output_dir / "tables/rq1_further_convergence_case_summary.csv").is_file()
    h1_cases = pd.read_csv(output_dir / "tables/h1_directional_cases_w60.csv")
    assert h1_cases.columns.tolist() == [
        "Case",
        "Mean hum Delta D (m)",
        "Mean control Delta D (m)",
        "Hum-to-control mean ratio",
        "Proportion",
    ]
    assert (
        h1_cases.loc[
            h1_cases["Case"].eq("Hum expands less"), "Hum-to-control mean ratio"
        ].iloc[0]
        == 0.5
    )
    assert (output_dir / "figures/cases_breakdown.png").is_file()
    assert (output_dir / "figures/convergent_cases_breakdown.png").is_file()
    assert result["plot_window_paired_rows"] == 6


def _convergence_row(
    match_set_id: str, condition: int, analysis_window_id: str, pre: float, post: float
) -> dict[str, object]:
    return {
        "analysis_window_id": analysis_window_id,
        "condition": condition,
        "match_set_id": match_set_id,
        "window_s": 60,
        "Delta_D_w": post - pre,
        "pre_distance_loo_centroid_m": pre,
        "post_distance_loo_centroid_m": post,
        "match_rank": 0,
        "standardized_match_distance": 0.0,
    }


def test_decorate_window_validity_applies_fixed_three_other_collar_rule() -> None:
    row = {
        "analysis_window_id": "event-1:w30",
        "event_id": "event-1",
        "focal_id": "A",
        "day_session_id": "session",
        "date_local": "2025-02-03",
        "window_s": 30,
        "event_midpoint_local": pd.Timestamp("2025-02-03T14:00:05+10:30"),
        "event_midpoint_seconds_local": 50_405.0,
        "filtered_focal_pre_fix_count": 30,
        "filtered_focal_post_fix_count": 0,
        "filtered_nonfocal_pre_unique_count": 12,
        "filtered_nonfocal_post_unique_count": 2,
        "filtered_aligned_pre_sample_count": 30,
        "filtered_aligned_post_sample_count": 0,
        "raw_focal_pre_fix_count": 30,
        "raw_focal_post_fix_count": 30,
        "raw_nonfocal_pre_unique_count": 12,
        "raw_nonfocal_post_unique_count": 12,
        "raw_aligned_pre_sample_count": 30,
        "raw_aligned_post_sample_count": 30,
    }
    enriched = decorate_window_validity(pd.DataFrame([row])).iloc[0]
    assert bool(enriched["raw_gps_valid_min_nonfocal_3"])
    assert not bool(enriched["filtered_gps_valid_min_nonfocal_3"])


def test_segments_with_sidecars_preserves_hpc_path() -> None:
    bundle = SimpleNamespace(
        segments=pd.DataFrame(
            [
                {
                    "segment_id": "s1",
                    "day_session_id": "d1",
                    "csv_hpc_path": "/remote/s1.csv",
                }
            ]
        ),
        sidecar_index=pd.DataFrame(
            [
                {
                    "segment_id": "s1",
                    "day_session_id": "d1",
                    "cached_csv_path": "cache/s1.csv",
                    "csv_hpc_path": "/remote/s1.csv",
                }
            ]
        ),
    )
    merged = _segments_with_sidecars(bundle, ["d1"])
    assert merged.iloc[0]["csv_hpc_path"] == "/remote/s1.csv"


def test_rq1_launch_manifest_uses_new_stage_contract(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    build_dataset(config)
    base = build_analysis_base(config)
    definition = get_analysis_definition("rq1-post-hum-cohesion")
    label = f"launch_fixture_{tmp_path.name}"
    launch_root = (
        Path(__file__).resolve().parents[1]
        / "data/processed/analysis/rq1_post_hum_cohesion/launches"
        / label
    )
    if launch_root.exists():
        shutil.rmtree(launch_root)
    try:
        launch = create_launch(
            definition=definition,
            config=config,
            launch_label=label,
            profile="smoke",
            input_manifest_path=Path(base["manifest_path"]),
        )
        ctx = load_launch_context(
            definition=definition, config=config, launch_dir=Path(launch["launch_dir"])
        )
        prepared = prepare_hpc(ctx)
        stage_ids = [item["stage_id"] for item in prepared["prepared_stages"]]
        assert stage_ids == list(STAGE_IDS)
        model_input_bundle = json.loads(
            (Path(launch["launch_dir"]) / "hpc_bundle/model_input/manifest.json").read_text()
        )
        assert model_input_bundle["params_row_count"] == 1
        assert model_input_bundle["resources"]["time_limit"] == "00:20:00"
    finally:
        if launch_root.exists():
            shutil.rmtree(launch_root)

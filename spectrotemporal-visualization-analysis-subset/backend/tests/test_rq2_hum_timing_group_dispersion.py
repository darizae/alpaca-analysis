from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from stv_analysis.framework import concrete_array_afterok_dependency, create_launch, load_launch_context, prepare_hpc
from stv_analysis.registry import get_analysis_definition
from stv_analysis.rq2_hum_timing_group_dispersion.builder import (
    build_hum_rate_features,
    build_model_input,
    build_rq2_units,
    gps_residual_qc_tables,
    hum_event_frame,
)
from stv_analysis.rq2_hum_timing_group_dispersion.speed_sensitivity import aggregated_window_speed_metrics
from stv_analysis.rq1_post_hum_cohesion.contracts import SessionTraceGrid
from stv_analysis.rq2_hum_timing_group_dispersion.amplitude_audio import (
    PRIMARY_METRIC,
    amplitude_metrics,
    dbfs,
    frame_rms_values,
)
from stv_analysis.rq2_hum_timing_group_dispersion.amplitude_attenuation_filter import (
    THRESHOLD_IDS,
    attenuation_threshold_definitions,
)
from stv_analysis.rq2_hum_timing_group_dispersion.contracts import ANALYSIS_ID, HEAVY_COMPUTE_LABEL, MODEL_FIT_RESOURCE_PROFILE, STAGE_IDS
from stv_backend.analysis_base import build_analysis_base
from stv_backend.pipeline import build_dataset

from test_rq1_post_hum_cohesion import _build_config


def test_rq2_registry_stage_graph_and_compute_classes() -> None:
    definition = get_analysis_definition(ANALYSIS_ID)

    assert definition.output_dirname == "rq2_hum_timing_group_dispersion"
    assert tuple(spec.stage_id for spec in definition.stage_specs) == STAGE_IDS
    assert all(spec.compute_class == HEAVY_COMPUTE_LABEL for spec in definition.stage_specs)
    assert definition.stage_specs[-3].stage_id == "scientific_glmm"
    assert definition.stage_specs[-3].resource_profile == MODEL_FIT_RESOURCE_PROFILE
    assert definition.stage_specs[0].dependencies == ()
    for previous, current in zip(definition.stage_specs, definition.stage_specs[1:]):
        assert current.dependencies == (previous.stage_id,)


def test_hpc_dependencies_use_concrete_array_task_ids() -> None:
    assert concrete_array_afterok_dependency("123", 1) == "afterok:123_0"
    assert concrete_array_afterok_dependency("123", 3) == "afterok:123_0:123_1:123_2"


def test_rq2_artifact_contract_lists_required_outputs() -> None:
    contract = Path(__file__).resolve().parents[2] / "docs" / "rq2_hum_timing_group_dispersion" / "ARTIFACT_CONTRACT.md"
    text = contract.read_text(encoding="utf8")

    for required in (
        "rq2_units.parquet",
        "rq2_gps_residual_outlier_audit.csv",
        "trace_points_rq2_gps_qc.parquet",
        "rq2_window_grid.parquet",
        "rq2_dispersion_features.parquet",
        "rq2_dispersion_change_summary.csv",
        "rq2_hum_rate_features.parquet",
        "rq2_model_input.parquet",
        "rq2_model_input.csv",
        "rq2_model_run_manifest.csv",
        "rq2_fixed_effects.csv",
        "rq2_dharma_standard_tests.csv",
        "rq2_dharma_temporal_autocorrelation.csv",
        "rq2_scaled_residuals.csv.gz",
        "rq2_residual_acf.csv",
        "manifest.json",
    ):
        assert required in text
    assert "rq2_glmm_term_tests_car_anova.csv" not in text
    assert "NB2" not in text
    assert "120 s" not in text


def test_rq2_window_construction_is_deterministic() -> None:
    trace_points = pd.DataFrame(
        {
            "day_session_id": ["2025-02-03:session-001"] * 121,
            "date_local": ["2025-02-03"] * 121,
            "session_id": ["session-001"] * 121,
            "seconds_since_midnight_local": np.arange(100.0, 221.0),
            "latitude": np.linspace(-34.5, -34.49, 121),
            "recorder_id": ["A"] * 121,
        }
    )
    events = _events([105.0])
    segments = pd.DataFrame(
        [{"segment_id": "seg-A", "day_session_id": "2025-02-03:session-001", "recorder_id": "A"}]
    )
    recorders = pd.DataFrame([{"recorder_id": "A"}])

    first, _, first_summary = build_rq2_units(
        trace_points,
        events,
        segments,
        recorders,
        selected_sessions=["2025-02-03:session-001"],
        timezone_name="Australia/Adelaide",
        dataset_name="fixture",
    )
    second, _, second_summary = build_rq2_units(
        trace_points,
        events,
        segments,
        recorders,
        selected_sessions=["2025-02-03:session-001"],
        timezone_name="Australia/Adelaide",
        dataset_name="fixture",
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_summary == second_summary
    assert first.loc[first["window_length_s"] == 60, "window_start_seconds_local"].tolist() == [100.0, 160.0]
    assert set(first["window_length_s"]) == {60}


def test_rq2_hum_event_window_assignment_uses_midpoint_and_session() -> None:
    windows = pd.DataFrame(
        [
            _spread_row("w1", "2025-02-03:session-001", 100.0, 160.0),
            _spread_row("w2", "2025-02-03:session-001", 160.0, 220.0),
            _spread_row("w3", "2025-02-04:session-001", 100.0, 160.0),
        ]
    )
    events = pd.concat([_events([159.9, 160.0]), _events([120.0], day_session_id="2025-02-04:session-001")])

    assigned = build_hum_rate_features(windows, events)

    assert assigned.loc[assigned["rq2_window_id"] == "w1", "hum_count"].iloc[0] == 1
    assert assigned.loc[assigned["rq2_window_id"] == "w2", "hum_count"].iloc[0] == 1
    assert assigned.loc[assigned["rq2_window_id"] == "w3", "hum_count"].iloc[0] == 1
    assert assigned.loc[assigned["rq2_window_id"] == "w1", "hum_rate_per_min"].iloc[0] == 1.0


def test_rq2_hum_event_policy_keeps_none_and_excludes_rumble() -> None:
    events = _events([100.0, 101.0, 102.0, 103.0])
    events.loc[0, "reviewed_resolved_label_semantic"] = "alpaca_hum"
    events.loc[1, "reviewed_resolved_label_semantic"] = "None"
    events.loc[2, "reviewed_resolved_label_semantic"] = "rumble"
    events.loc[3, "reviewed_resolved_label_semantic"] = "human_talking"

    hums = hum_event_frame(events)

    assert hums["reviewed_resolved_label_semantic"].tolist() == ["alpaca_hum", "None"]


def test_rq2_residual_gps_outlier_fingerprint_creates_qc_trace() -> None:
    trace = pd.DataFrame(
        [
            _trace_point("s1", "A", -34.5000, 138.7000, 100.0),
            _trace_point("s1", "B", -34.5001, 138.7001, 100.0),
            _trace_point("s1", "C", -34.5002, 138.7002, 100.0),
            _trace_point("s1", "A", -34.5050, 138.6950, 101.0),
        ]
    )
    origin = {"origin_latitude": -34.5000, "origin_longitude": 138.7000}

    qc_trace, audit, outliers, point_summary, _paddock = gps_residual_qc_tables(trace, origin)

    assert len(qc_trace.index) == 3
    assert len(outliers.index) == 1
    assert int(audit["residual_outlier_points"].sum()) == 1
    assert int(point_summary["removed_points"].sum()) == 1


def test_rq2_model_input_keeps_poisson_response_offset_and_predictors() -> None:
    rows = [
        _spread_row(f"w{i}", "s1", float(i * 60), float((i + 1) * 60), hum_count=i % 2, polygon_area=area, spread_rate=rate)
        for i, (area, rate) in enumerate([(10, -5), (20, -1), (30, 0), (40, 1), (50, 5), (60, 9)])
    ]
    first, first_summary = build_model_input(pd.DataFrame(rows))
    second, second_summary = build_model_input(pd.DataFrame(rows))

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert {"hum_count", "window_length_minutes", "area_m2", "spread_change_m2_per_min", "speed_mps", "centroid_change_in_position_m"}.issubset(first.columns)
    assert first["spread_change_m2_per_min"].tolist() == [-5.0, -1.0, 0.0, 1.0, 5.0, 9.0]
    assert first["window_length_minutes"].eq(1.0).all()
    assert set(first["model_offset"]) == {"offset(log(window_length_minutes))"}
    assert first_summary.loc[first_summary["window_length_s"] == 60, "complete_model_rows"].iloc[0] == 6


def test_rq2_speed_sensitivity_bin_aggregation_preserves_linear_motion() -> None:
    seconds = np.arange(10, dtype=np.int32)
    x_m = np.column_stack([seconds.astype(float), seconds.astype(float) * 2.0])
    y_m = np.zeros_like(x_m, dtype=float)
    valid = np.ones_like(x_m, dtype=bool)
    grid = SessionTraceGrid(
        day_session_id="s1",
        recorder_ids=("A", "B"),
        recorder_index={"A": 0, "B": 1},
        second_bins=seconds,
        bin_times_s=seconds.astype(float),
        valid=valid,
        x_m=x_m,
        y_m=y_m,
        nonfocal_count=np.zeros_like(x_m, dtype=np.int16),
        loo_distance_m=np.full_like(x_m, np.nan, dtype=float),
    )

    one_second = aggregated_window_speed_metrics(grid, 0.0, 10.0, 1)
    five_second = aggregated_window_speed_metrics(grid, 0.0, 10.0, 5)

    assert one_second["speed_sample_count"] == 18
    assert one_second["speed_mps"] == 1.5
    assert five_second["speed_sample_count"] == 2
    assert five_second["speed_mps"] == 1.5


def test_rq2_amplitude_metrics() -> None:
    sample_rate = 48_000
    seconds = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = (0.5 * np.sin(2 * np.pi * 300.0 * seconds)).astype(np.float32)

    metrics = amplitude_metrics(samples, sample_rate, {})

    assert dbfs(1.0) == 0.0
    assert frame_rms_values(samples, sample_rate).shape[0] > 10
    assert -12.0 < metrics[PRIMARY_METRIC] < -6.0


def test_rq2_attenuation_thresholds_use_external_fence_quantiles() -> None:
    calibration = pd.DataFrame(
        {
            "calibration_class": ["self_collar"] * 4 + ["external_fence"] * 4,
            PRIMARY_METRIC: [-10.0, -20.0, -30.0, -40.0, -50.0, -40.0, -30.0, -20.0],
        }
    )

    thresholds = attenuation_threshold_definitions(calibration)

    assert tuple(thresholds["threshold_id"]) == THRESHOLD_IDS
    assert set(thresholds["threshold_scope"]) == {"attenuation_experiment_external_fence_quantile"}
    np.testing.assert_allclose(thresholds["cutoff_dbfs"].to_numpy(dtype=float), [-23.0, -21.5, -20.75])


def test_rq2_smoke_launch_creation_and_hpc_bundle(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    build_dataset(config)
    base_result = build_analysis_base(config)
    definition = get_analysis_definition(ANALYSIS_ID)
    launch_label = f"rq2_launch_fixture_smoke_{tmp_path.name}"
    launch_root = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "analysis"
        / "rq2_hum_timing_group_dispersion"
        / "launches"
        / launch_label
    )
    if launch_root.exists():
        shutil.rmtree(launch_root)
    latest_path = launch_root.parent.parent / "latest"
    previous_latest_target = latest_path.readlink() if latest_path.is_symlink() else None

    try:
        launch = create_launch(
            definition=definition,
            config=config,
            launch_label=launch_label,
            profile="smoke",
            input_manifest_path=Path(base_result["manifest_path"]),
        )
        launch_dir = Path(launch["launch_dir"])
        ctx = load_launch_context(definition=definition, config=config, launch_dir=launch_dir)
        assert json.loads((launch_dir / "configs" / "launch_manifest.json").read_text())["analysis_id"] == ANALYSIS_ID

        prepared = prepare_hpc(ctx)
        assert len(prepared["prepared_stages"]) == len(STAGE_IDS)
        trace_bundle = json.loads((launch_dir / "hpc_bundle" / "trace_grids" / "manifest.json").read_text())
        assert trace_bundle["compute_class"] == "heavy_compute"
        assert trace_bundle["params_row_count"] == 1
        model_bundle = json.loads((launch_dir / "hpc_bundle" / "scientific_glmm" / "manifest.json").read_text())
        assert model_bundle["resource_profile"] == MODEL_FIT_RESOURCE_PROFILE
        assert model_bundle["resources"]["time_limit"] == config.analysis_hpc.resources[MODEL_FIT_RESOURCE_PROFILE].time_limit
        assert (launch_dir / "hpc_bundle" / "gps_filter_audit" / "manifest.json").is_file()
    finally:
        if launch_root.exists():
            shutil.rmtree(launch_root)
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        if previous_latest_target is not None:
            latest_path.symlink_to(previous_latest_target)


def _events(seconds: list[float], *, day_session_id: str = "2025-02-03:session-001") -> pd.DataFrame:
    date = day_session_id.split(":", maxsplit=1)[0]
    rows = []
    for index, second in enumerate(seconds):
        start_second = second - 0.1
        start = pd.Timestamp(date, tz="Australia/Adelaide") + pd.to_timedelta(start_second, unit="s")
        rows.append(
            {
                "review_row_key": f"{day_session_id}-event-{index}-{second}",
                "recorder_id": "A",
                "recorder_display_name": "A",
                "device_id": "device-A",
                "reviewed_resolved_label_semantic": "alpaca_hum",
                "timestamp_start_local": start,
                "timestamp_end_local": start + pd.to_timedelta(0.2, unit="s"),
                "date_local": date,
                "session_id": day_session_id,
                "day_session_id": day_session_id,
                "seconds_since_midnight_local": start_second,
            }
        )
    return pd.DataFrame(rows)


def _spread_row(
    window_id: str,
    day_session_id: str,
    start_s: float,
    end_s: float,
    *,
    hum_count: int = 0,
    polygon_area: float | None = None,
    spread_rate: float | None = None,
) -> dict[str, object]:
    area = 100.0 + start_s if polygon_area is None else polygon_area
    rate = (float(start_s % 120) - 30.0) if spread_rate is None else spread_rate
    return {
        "rq2_window_id": window_id,
        "day_session_id": day_session_id,
        "date_local": day_session_id.split(":", maxsplit=1)[0] if ":" in day_session_id else "2025-02-03",
        "session_id": day_session_id,
        "window_length_s": int(end_s - start_s),
        "window_start_seconds_local": start_s,
        "window_end_seconds_local": end_s,
        "window_midpoint_seconds_local": (start_s + end_s) / 2,
        "time_of_day": "00:00:00",
        "minutes_since_session_start": start_s / 60,
        "hum_count": hum_count,
        "hum_presence": int(hum_count > 0),
        "hum_rate_per_min": hum_count / ((end_s - start_s) / 60),
        "observed_polygon_area_m2": area,
        "recent_change_in_polygon_area_m2": rate,
        "dispersion_change_m2_per_min": rate / ((end_s - start_s) / 60),
        "mean_distance_to_observed_centroid_m": 5.0 + (start_s / 100.0),
        "recent_change_in_centroid_distance_m": (start_s % 90) / 30.0,
        "window_centroid_x_m": start_s / 10.0,
        "window_centroid_y_m": end_s / 20.0,
        "centroid_change_in_position_m": (start_s % 120) / 10.0,
        "centroid_change_in_position_m_per_min": ((start_s % 120) / 10.0) / ((end_s - start_s) / 60),
        "mean_speed_mps": 0.5 + (start_s % 30) / 100.0,
        "n_collars_available": 4,
        "n_audio_recorders_active": 3,
        "audio_coverage_fraction": 1.0,
        "gps_coverage_fraction": 1.0,
        "hard_gps_filter_version": "fixture",
        "residual_gps_filter_version": "fixture_residual_qc",
        "median_distance_to_observed_centroid_m": 5.0,
        "spread_valid_seconds": int(end_s - start_s),
        "biologically_impossible_speed_samples": 0,
        "window_eligible": True,
        "n_valid_gps_seconds": int(end_s - start_s),
        "n_any_gps_seconds": int(end_s - start_s),
    }


def _trace_point(day_session_id: str, recorder_id: str, lat: float, lon: float, second: float) -> dict[str, object]:
    return {
        "segment_id": f"{recorder_id}-seg",
        "point_index_in_segment": int(second),
        "recorder_id": recorder_id,
        "recorder_display_name": recorder_id,
        "wav_basename": f"{recorder_id}.WAV",
        "timestamp_utc": pd.Timestamp("2025-02-03T00:00:00Z"),
        "timestamp_local": pd.Timestamp("2025-02-03", tz="Australia/Adelaide") + pd.to_timedelta(second, unit="s"),
        "date_local": "2025-02-03",
        "seconds_since_midnight_local": second,
        "latitude": lat,
        "longitude": lon,
        "gps_status": "A",
        "timer_count": 0,
        "buffers_filled": 0,
        "buffers_written": 0,
        "sample_rate_estimate": 48000.0,
        "session_id": day_session_id,
        "day_session_id": day_session_id,
    }

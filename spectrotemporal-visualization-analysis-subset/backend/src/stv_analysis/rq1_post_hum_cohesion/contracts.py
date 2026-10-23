from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..framework import COMPUTE_CLASS_HEAVY, COMPUTE_CLASS_LOCAL_OK


WINDOW_LENGTHS_SECONDS = (30, 60, 120)
PRIMARY_NONFOCAL_THRESHOLD = 3
OUTPUT_DIRNAME = "rq1_post_hum_cohesion"
SUMMARY_METHOD = "median_aligned_second_distance"
LATITUDE_METERS_PER_DEGREE = 111_320.0
ANALYSIS_VERSION = 6
ANALYSIS_ID = "rq1-post-hum-cohesion"
HEAVY_COMPUTE_LABEL = COMPUTE_CLASS_HEAVY
LOCAL_OK_LABEL = COMPUTE_CLASS_LOCAL_OK
STAGE_IDS = (
    "units",
    "filtered_trace_grids",
    "filtered_window_inventory",
    "blocker_scan",
    "blocker_merge",
    "raw_trace_grids",
    "window_validity",
    "centroid_distance",
    "baseline_covariates",
    "covariate_table",
    "matched_controls",
    "model_input",
    "reports",
)


@dataclass(frozen=True)
class SessionTraceGrid:
    day_session_id: str
    recorder_ids: tuple[str, ...]
    recorder_index: dict[str, int]
    second_bins: np.ndarray
    bin_times_s: np.ndarray
    valid: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    nonfocal_count: np.ndarray
    loo_distance_m: np.ndarray


@dataclass(frozen=True)
class BaseAnalysisBundle:
    manifest_path: Path
    manifest: dict[str, Any]
    timezone_name: str
    dataset_name: str
    recorders: pd.DataFrame
    segments: pd.DataFrame
    events: pd.DataFrame
    trace_points_filtered: pd.DataFrame
    sidecar_index: pd.DataFrame
    gps_blocker_rule: dict[str, Any]

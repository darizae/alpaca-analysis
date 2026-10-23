from __future__ import annotations

from ..framework import COMPUTE_CLASS_HEAVY


ANALYSIS_ID = "rq2-hum-timing-group-dispersion"
OUTPUT_DIRNAME = "rq2_hum_timing_group_dispersion"
ANALYSIS_VERSION = 6
HEAVY_COMPUTE_LABEL = COMPUTE_CLASS_HEAVY
MODEL_FIT_RESOURCE_PROFILE = "model_fit"

WINDOW_LENGTHS_SECONDS = (60,)
PRIMARY_WINDOW_LENGTH_S = 60
BIOLOGICALLY_IMPOSSIBLE_SPEED_MPS = 12.0

DOCUMENTED_PADDOCK_LENGTH_M = 140.0
DOCUMENTED_PADDOCK_WIDTH_M = 70.0
DOCUMENTED_PADDOCK_AREA_M2 = DOCUMENTED_PADDOCK_LENGTH_M * DOCUMENTED_PADDOCK_WIDTH_M
RESIDUAL_GPS_FILTER_VERSION = "rq2_residual_extreme_position_filter_v1"
STAGE_IDS = (
    "units",
    "gps_filter_audit",
    "trace_grids",
    "window_grid",
    "dispersion_features",
    "hum_rate_features",
    "model_input",
    "scientific_glmm",
    "diagnostics_outputs",
    "reports",
)

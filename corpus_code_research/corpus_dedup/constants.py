from __future__ import annotations

from pathlib import Path

DEFAULT_ARTIFACT_ROOT = Path("artifacts/dedup_full_corpus")

DEFAULT_ALGORITHM_VERSION = "overlap_temporal_v4"
DEFAULT_OVERLAP_FRACTION_SHORT_THRESHOLDS = [0.05, 0.10, 0.25, 0.50, 0.75]
DEFAULT_GPS_THRESHOLDS_M = [2.0, 3.0, 4.0, 5.0, 6.0]
DEFAULT_RELATIVE_DIFFERENCE_THRESHOLDS = [None, 0.10, 0.20, 0.30, 0.40]
DEFAULT_GROUP_ACOUSTIC_THRESHOLDS = [None, "q90", "q75", "q50"]

DEFAULT_TOP_K_MODELS = 5
DEFAULT_EXEMPLARS_PER_MODEL = 10

DEFAULT_FEATURE_COLUMNS = [
    "f0_hz",
    "duration90_s",
    "peak_frequency_hz",
    "frequency5_hz",
    "frequency95_hz",
    "bandwidth90_hz",
    "average_entropy",
]

HPC_STAGE_NAMES = ("features", "feature_merge", "grid", "aggregate", "exemplars")
PROGRESS_STAGE_NAMES = ("features", "feature_merge", "grid", "aggregate", "exemplars")

REPRESENTATIVE_SELECTION_RULE = (
    "max_snr_then_max_rms_then_max_classifier_score_then_internal_deterministic_tiebreak"
)
SOURCE_ASSIGNMENT_STATUS = "unknown"
FEATURE_METHOD_VERSION = "dedup_feature_extraction_v1"
DECISION_METHOD_VERSION = "overlap_temporal_v4"
ROBUSTNESS_METHOD_VERSION = "overlap_temporal_harmonic_v4"
ACOUSTIC_MARGIN_QUANTILE = 0.25
TARGET_ACOUSTIC_MARGIN = 0.10
MIN_FINAL_SELECTABLE_ACOUSTIC_COVERAGE = 0.95
CURRENT_SCHEMA_VERSION = 3

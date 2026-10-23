from __future__ import annotations

from .framework import AnalysisDefinition
from .rq1_post_hum_cohesion.builder import ANALYSIS_DEFINITION as RQ1_ANALYSIS_DEFINITION
from .rq2_hum_timing_group_dispersion.builder import ANALYSIS_DEFINITION as RQ2_ANALYSIS_DEFINITION


ANALYSIS_DEFINITIONS: dict[str, AnalysisDefinition] = {
    RQ1_ANALYSIS_DEFINITION.analysis_id: RQ1_ANALYSIS_DEFINITION,
    RQ2_ANALYSIS_DEFINITION.analysis_id: RQ2_ANALYSIS_DEFINITION,
}


def get_analysis_definition(analysis_id: str) -> AnalysisDefinition:
    try:
        return ANALYSIS_DEFINITIONS[analysis_id]
    except KeyError as exc:
        known = ", ".join(sorted(ANALYSIS_DEFINITIONS))
        raise KeyError(f"unknown analysis_id {analysis_id!r}; known analyses: {known}") from exc

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import ANALYSIS_ID, ANALYSIS_VERSION


def build_runtime_manifest(
    *,
    launch_label: str,
    launch_dir: Path,
    input_manifest_path: Path,
    specs_manifest: dict[str, Any],
    source_code_manifest: dict[str, Any],
    tables_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(tables_dir.iterdir()):
        if not path.is_file():
            continue
        row_count: int | None = None
        try:
            if path.suffix == ".csv":
                row_count = len(pd.read_csv(path))
            elif path.suffix == ".parquet":
                row_count = len(pd.read_parquet(path, columns=[]))
        except Exception:
            row_count = None
        artifacts[path.stem] = {
            "path": str(path.relative_to(launch_dir)),
            "rows": row_count,
            "canonical_status": "current-canonical",
        }
    return {
        "analysis_id": ANALYSIS_ID,
        "analysis_version": ANALYSIS_VERSION,
        "launch_label": launch_label,
        "launch_root": ".",
        "input_manifest_path": str(input_manifest_path.relative_to(launch_dir)),
        "specs_manifest": specs_manifest,
        "source_code_manifest": source_code_manifest,
        "reports_dir": str(reports_dir.relative_to(launch_dir)),
        "artifacts": artifacts,
        "model_boundary": (
            "This launch materializes the RQ1 matched-control model input. "
            "The retained LMM fits are run by scientific_lmm_guide_workflow.R."
        ),
    }


def write_runtime_reports(*, report_dir: Path, tables_dir: Path, context: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    model_summary = _read_table(tables_dir / "model_input_summary.csv")
    match_summary = _read_table(tables_dir / "matched_control_windows_summary.csv")
    directional = _read_table(tables_dir / "rq1_directional_movement_summary.csv")
    attrition = _read_table(tables_dir / "filtering_attrition.csv")

    lines = [
        "# RQ1 Model-Input Launch",
        "",
        "This launch materializes the upstream matched-control input for the retained RQ1 scientific LMM.",
        "It does not fit the retired exploratory sweep.",
        "",
        "## Scope",
        "",
        "- Fitted LMM workflow: `backend/src/stv_analysis/rq1_post_hum_cohesion/scientific_lmm_guide_workflow.R`.",
        "- Primary materialized view: `scientific_lmm_materialize.py`.",
        "- Sensitivity materialized view: `scientific_lmm_threshold_materialize.py`.",
        "- Sensitivity fit uses the same R workflow with the threshold run config.",
        "",
        "## Counts",
        "",
        f"- Analysis units: {context['analysis_units']:,}.",
        f"- Blocked raw GPS fixes restored for audit: {context['blocked_trace_points']:,}.",
        f"- Baseline covariate rows: {context['baseline_covariate_rows']:,}.",
        f"- Matched control rows: {context['matched_controls']:,}.",
        f"- Model-input rows: {context['model_input_rows']:,}.",
        "",
        "## Model-Input Summary",
        "",
        _frame_text(model_summary),
        "",
        "## Matching Summary",
        "",
        _frame_text(match_summary),
        "",
        "## Directional Movement Summary",
        "",
        _frame_text(directional),
        "",
        "## Filtering Attrition",
        "",
        _frame_text(attrition),
        "",
    ]
    (report_dir / "rq1_model_input_build_report.md").write_text("\n".join(lines), encoding="utf8")
    (report_dir / "README.md").write_text(
        "# RQ1 model-input launch reports\n\n"
        "This directory contains materialization reports for the RQ1 matched-control model input. "
        "LMM fit outputs are produced by the retained R workflow under `scientific_lmm_revamp`.\n",
        encoding="utf8",
    )
    _write_output_index(tables_dir, report_dir)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _frame_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    return "```text\n" + frame.to_string(index=False) + "\n```"


def _write_output_index(tables_dir: Path, report_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for root, role in ((tables_dir, "table"), (report_dir, "report")):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*")):
            if path.is_file():
                rows.append(
                    {
                        "artifact_name": path.name,
                        "artifact_role": role,
                        "path": str(path.relative_to(tables_dir.parent)),
                        "canonical_status": "current-canonical",
                    }
                )
    rows.append(
        {
            "artifact_name": "rq1_final_output_index.csv",
            "artifact_role": "table",
            "path": "tables/rq1_final_output_index.csv",
            "canonical_status": "current-canonical",
        }
    )
    pd.DataFrame(rows).to_csv(tables_dir / "rq1_final_output_index.csv", index=False)

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp")

RETAINED_FIXED_EFFECT_RAW_COLUMNS = [
    "starting_centroid_distance_m",
    "mean_speed_pre_m_s",
    "pre_delta_centroid_distance_m",
    "observed_polygon_area_pre_m2",
    "recent_change_in_polygon_area_m2",
]

EXCLUDED_PREDICTOR_COLUMNS = [
    "time_of_day_minutes",
    "minutes_since_session_start",
    "n_collars_pre",
    "local_collared_density_per_1000m2",
    "collar_availability",
    "time_of_day",
    "local_density",
]

KEEP_COLUMNS = [
    "analysis_window_id",
    "condition",
    "condition_label",
    "match_set_id",
    "event_id",
    "hum_event_id",
    "focal_id",
    "focal_alpaca_id",
    "day_session_id",
    "session_id",
    "date_local",
    "window_s",
    "window_length_s",
    "window_code",
    "window_center_seconds_local",
    "window_center_timestamp",
    "Delta_D_w",
    "change_in_centroid_distance_w",
    "change_in_centroid_distance_loo_m",
    "pre_distance_loo_centroid_m",
    "post_distance_loo_centroid_m",
    *RETAINED_FIXED_EFFECT_RAW_COLUMNS,
    "starting_distance",
    "pre_window_speed",
    "recent_pre_event_distance_change",
    "observed_polygon_area_pre",
    "recent_change_in_polygon_area",
    "directional_summary_method",
    "focal_distance_reduction_m",
    "centroid_distance_reduction_m",
    "position_distance_interaction_m",
    "distance_change_outcome",
    "directional_category",
    "directional_category_clean",
    "focal_contribution_weight",
    "centroid_contribution_weight",
    "directional_assignment_low_support",
    "directional_geometry_valid",
    "directional_geometry_invalid_reason",
    "distance_sample_count_pre",
    "distance_sample_count_post",
    "matching_rule_used",
    "buffer_rule_used",
    "buffer_seconds",
    "match_rank",
    "control_match_use_count",
    "standardized_match_distance",
    "hard_gps_filter_version",
    "source_confidence",
]


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    source_launch = resolve_path(repo_root, args.source_launch)
    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for directory in (data_dir, tables_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_input_path = source_launch / "tables" / "model_input.parquet"
    if not model_input_path.is_file():
        raise FileNotFoundError(f"missing source model input: {model_input_path}")

    model_input = pd.read_parquet(model_input_path)
    model_input = normalize_model_input(model_input)

    standard_matching = materialized_view(enforce_standard_matching(model_input), "standard_matching")

    write_view(standard_matching, data_dir / "h1_contact_reinforcement_model_input")

    view_summary = summarize_views(
        {
            "standard_matching": standard_matching,
        }
    )
    view_summary.to_csv(tables_dir / "rq1_lmm_materialized_view_summary.csv", index=False)

    event_label_audit = build_event_label_audit(repo_root, source_launch)
    event_label_audit.to_csv(tables_dir / "rq1_lmm_event_label_audit.csv", index=False)

    gps_summary = read_optional_table(source_launch / "tables" / "hard_gps_filter_summary.csv")
    if not gps_summary.empty:
        gps_summary.to_csv(tables_dir / "rq1_lmm_hard_gps_filter_summary.csv", index=False)

    model_input_summary = read_optional_table(source_launch / "tables" / "model_input_summary.csv")
    if not model_input_summary.empty:
        model_input_summary.to_csv(tables_dir / "rq1_lmm_source_model_input_summary.csv", index=False)

    write_materialization_report(
        reports_dir / "rq1_lmm_materialized_views.md",
        source_launch=source_launch,
        output_dir=output_dir,
        view_summary=view_summary,
        event_label_audit=event_label_audit,
        gps_summary=gps_summary,
    )

    manifest = {
        "source_launch": str(source_launch),
        "model_input_path": str(model_input_path),
        "output_dir": str(output_dir),
        "views": {
            "standard_matching": {
                "csv": str(data_dir / "h1_contact_reinforcement_model_input.csv"),
                "parquet": str(data_dir / "h1_contact_reinforcement_model_input.parquet"),
            },
        },
        "retained_fixed_effects": [
            "condition_hum",
            *RETAINED_FIXED_EFFECT_RAW_COLUMNS,
        ],
        "excluded_predictor_columns": EXCLUDED_PREDICTOR_COLUMNS,
        "standard_matching_rule": (
            "For controls selected multiple times by the upstream matched-control construction, "
            "retain the occurrence with the smallest standardized_match_distance, then lowest "
            "match_rank, then lexical match_set_id. Hum rows are retained as the source-of-truth "
            "event rows even when their originally matched control was assigned to another set."
        ),
        "event_label_policy": "canonical RQ1 hum events are alpaca_hum and None; rumble is excluded upstream.",
    }
    (output_dir / "materialization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"output_dir": str(output_dir), "view_summary_rows": len(view_summary.index)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize RQ1 scientific-LMM model views.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--source-launch", required=True, help="RQ1 model-input launch to reuse upstream tables from.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for scientific LMM inputs.")
    return parser.parse_args()


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def normalize_model_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in KEEP_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"source model input is missing required columns: {missing}")
    out = df[KEEP_COLUMNS].copy()
    out["condition"] = pd.to_numeric(out["condition"], errors="raise").astype("int8")
    out["condition_label"] = out["condition"].map({0: "control", 1: "hum"})
    out["Delta_D_w"] = pd.to_numeric(out["Delta_D_w"], errors="coerce")
    out["Delta_D_cm"] = out["Delta_D_w"] * 100.0
    out["session_id"] = out["session_id"].fillna(out["day_session_id"]).astype(str)
    out["day_session_id"] = out["day_session_id"].astype(str)
    out["focal_id"] = out["focal_id"].astype(str)
    out["match_set_id"] = out["match_set_id"].astype(str)
    out["analysis_window_id"] = out["analysis_window_id"].astype(str)
    for column in [
        "window_s",
        "window_length_s",
        "window_center_seconds_local",
        "match_rank",
        "control_match_use_count",
        "standardized_match_distance",
        "distance_sample_count_pre",
        "distance_sample_count_post",
        *RETAINED_FIXED_EFFECT_RAW_COLUMNS,
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["window_s"] = out["window_s"].astype("Int64")
    out["window_length_s"] = out["window_length_s"].astype("Int64")
    out["model_view_source"] = "launch_model_input"
    return out.sort_values(["window_s", "match_set_id", "condition", "match_rank", "analysis_window_id"]).reset_index(drop=True)


def enforce_standard_matching(df: pd.DataFrame) -> pd.DataFrame:
    hum = df.loc[df["condition"].eq(1)].copy()
    controls = df.loc[df["condition"].eq(0)].copy()
    controls = controls.sort_values(
        [
            "window_s",
            "analysis_window_id",
            "standardized_match_distance",
            "match_rank",
            "match_set_id",
        ],
        na_position="last",
    )
    controls = controls.drop_duplicates(subset=["window_s", "analysis_window_id"], keep="first")
    out = pd.concat([hum, controls], ignore_index=True, sort=False)
    out["control_match_use_count"] = np.where(out["condition"].eq(0), 1, 0)
    return out.sort_values(["window_s", "match_set_id", "condition", "match_rank", "analysis_window_id"]).reset_index(drop=True)


def materialized_view(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    out = df.copy()
    out["matching_policy"] = policy
    out["response_unit"] = "centimeters"
    out["response_transform"] = "Delta_D_cm = Delta_D_w * 100; no distributional transform"
    out["canonical_event_policy"] = "alpaca_hum and None are canonical; rumble is excluded"
    out["centroid_rule"] = "centroid of other observed collared alpacas, excluding focal"
    return out


def write_view(df: pd.DataFrame, stem: Path) -> None:
    df.to_csv(stem.with_suffix(".csv"), index=False)
    df.to_parquet(stem.with_suffix(".parquet"), index=False)


def summarize_views(views: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, df in views.items():
        for window_s, group in df.groupby("window_s", sort=True):
            controls = group.loc[group["condition"].eq(0)]
            hum = group.loc[group["condition"].eq(1)]
            hum_sets = set(hum["match_set_id"].astype(str))
            control_sets = set(controls["match_set_id"].astype(str))
            rows.append(
                {
                    "matching_policy": policy,
                    "window_s": int(window_s),
                    "n_rows": int(len(group.index)),
                    "n_hum_rows": int(len(hum.index)),
                    "n_control_rows": int(len(controls.index)),
                    "n_match_sets": int(group["match_set_id"].nunique()),
                    "n_hum_match_sets": int(len(hum_sets)),
                    "n_control_match_sets": int(len(control_sets)),
                    "n_match_sets_with_hum_and_control": int(len(hum_sets & control_sets)),
                    "n_hum_match_sets_without_control": int(len(hum_sets - control_sets)),
                    "n_control_match_sets_without_hum": int(len(control_sets - hum_sets)),
                    "n_unique_control_windows": int(controls["analysis_window_id"].nunique()),
                    "n_focal_ids": int(group["focal_id"].nunique()),
                    "n_sessions": int(group["session_id"].nunique()),
                    "median_delta_D_cm": float(group["Delta_D_cm"].median()),
                    "median_hum_delta_D_cm": float(hum["Delta_D_cm"].median()) if not hum.empty else np.nan,
                    "median_control_delta_D_cm": float(controls["Delta_D_cm"].median()) if not controls.empty else np.nan,
                    "max_control_match_use_count": int(controls["control_match_use_count"].max()) if not controls.empty else 0,
                }
            )
    return pd.DataFrame(rows).sort_values(["matching_policy", "window_s"]).reset_index(drop=True)


def build_event_label_audit(repo_root: Path, source_launch: Path) -> pd.DataFrame:
    candidates = [
        source_launch / "base_snapshot" / "events.parquet",
        repo_root / "backend/data/processed/analysis/base/events.parquet",
        repo_root / "data/raw_corpus_1/metadata_backfill/cleaned_enriched_detections.parquet",
    ]
    for path in candidates:
        if path.is_file():
            events = pd.read_parquet(path)
            break
    else:
        return pd.DataFrame(columns=["label", "rows", "canonical_for_rq1", "policy_note", "source_path"])
    label_col = "reviewed_resolved_label_semantic"
    counts = events[label_col].astype(object).where(events[label_col].notna(), "None").value_counts(dropna=False)
    rows = []
    for label, count in counts.items():
        label_text = str(label)
        canonical = label_text in {"alpaca_hum", "None"}
        rows.append(
            {
                "label": label_text,
                "rows": int(count),
                "canonical_for_rq1": canonical,
                "policy_note": "retained as alpaca hum" if canonical else "excluded from RQ1 hum-event units",
                "source_path": str(path),
            }
        )
    return pd.DataFrame(rows).sort_values(["canonical_for_rq1", "label"], ascending=[False, True]).reset_index(drop=True)


def read_optional_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_materialization_report(
    path: Path,
    *,
    source_launch: Path,
    output_dir: Path,
    view_summary: pd.DataFrame,
    event_label_audit: pd.DataFrame,
    gps_summary: pd.DataFrame,
) -> None:
    def frame_text(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No rows."
        return "```text\n" + frame.to_string(index=False) + "\n```"

    lines = [
        "# RQ1 Scientific LMM Materialized Views",
        "",
        f"- Source launch: `{source_launch}`",
        f"- Output directory: `{output_dir}`",
        "- Model view: `standard_matching` only.",
        "- Control rule: each control window is retained only once, using the closest existing match by standardized distance.",
        "- The standard H1 model input keeps all hum rows. It does not enforce one-to-one proportionality between hum rows and control rows; hum-only match sets can occur after control deduplication.",
        "- Canonical event-label policy: `alpaca_hum` and `None` are retained as hum events; `rumble` is excluded.",
        "- Response unit for the scientific LMM layer: centimeters (`Delta_D_cm = Delta_D_w * 100`).",
        "- Retained model predictors: `condition_hum`, `starting_centroid_distance_m`, `mean_speed_pre_m_s`, `pre_delta_centroid_distance_m`, `observed_polygon_area_pre_m2`, `recent_change_in_polygon_area_m2`.",
        "- Excluded predictor columns are not materialized into this model view: `time_of_day_minutes`, `local_collared_density_per_1000m2`, `n_collars_pre`, `collar_availability`, `minutes_since_session_start`.",
        "",
        "## View Summary",
        "",
        frame_text(view_summary),
        "",
        "## Event Label Audit",
        "",
        frame_text(event_label_audit) if not event_label_audit.empty else "Event label audit unavailable.",
        "",
        "## Hard GPS Filter",
        "",
        frame_text(gps_summary) if not gps_summary.empty else "Hard GPS filter summary unavailable.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


DEFAULT_MODEL_INPUT = Path(
    "backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/data/"
    "h1_contact_reinforcement_model_input.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/"
    "further_convergence_breakdown"
)
DEFAULT_PLOT_WINDOW_S = 60
DEFAULT_ZERO_TOLERANCE_M = 1e-9

REQUIRED_COLUMNS = {
    "analysis_window_id",
    "condition",
    "match_set_id",
    "window_s",
    "Delta_D_w",
    "pre_distance_loo_centroid_m",
    "post_distance_loo_centroid_m",
}


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    case_label: str
    plot_label: str
    h1_direction: str


CASE_DEFINITIONS = (
    CaseDefinition(
        case_id="both_expand_hum_expands_more",
        case_label="Both expand, but hum expands more",
        plot_label="Both expand,\nbut hum\nexpands more",
        h1_direction="against_h1",
    ),
    CaseDefinition(
        case_id="both_expand_hum_expands_less",
        case_label="* Both expand, but hum expands less",
        plot_label="* Both expand,\nbut hum\nexpands less",
        h1_direction="supports_h1",
    ),
    CaseDefinition(
        case_id="hum_contracts_control_expands",
        case_label="Hum contracts, control expands",
        plot_label="Hum contracts,\ncontrol\nexpands",
        h1_direction="supports_h1",
    ),
    CaseDefinition(
        case_id="both_contract_hum_contracts_more",
        case_label="Both contract, but hum contracts more",
        plot_label="Both contract,\nbut hum\ncontracts more",
        h1_direction="supports_h1",
    ),
    CaseDefinition(
        case_id="both_contract_control_contracts_more",
        case_label="Both contract, but control contracts more",
        plot_label="Both contract,\nbut control\ncontracts more",
        h1_direction="against_h1",
    ),
    CaseDefinition(
        case_id="hum_expands_control_contracts",
        case_label="Hum expands, control contracts",
        plot_label="Hum expands,\ncontrol\ncontracts",
        h1_direction="against_h1",
    ),
    CaseDefinition(
        case_id="boundary_or_tie",
        case_label="Boundary or tie",
        plot_label="Boundary\nor tie",
        h1_direction="boundary_or_tie",
    ),
)

CASE_BY_ID = {definition.case_id: definition for definition in CASE_DEFINITIONS}
PLOT_CASE_IDS = tuple(
    definition.case_id
    for definition in CASE_DEFINITIONS
    if definition.case_id != "boundary_or_tie"
)

CONVERGENT_MAGNITUDE_ORDER = (
    "hum_actually_contracts",
    "both_expand_hum_expands_less",
)
CONVERGENT_MAGNITUDE_LABELS = {
    "hum_actually_contracts": "Hum actually contracts",
    "both_expand_hum_expands_less": "* Both expand,\nbut hum expands less",
}
H1_DIRECTIONAL_CASE_LABELS = {
    "both_expand_hum_expands_more": "Hum expands more",
    "both_expand_hum_expands_less": "Hum expands less",
    "hum_contracts_control_expands": "Hum contracts, control expands",
    "both_contract_hum_contracts_more": "Both contract, hum contracts more",
    "both_contract_control_contracts_more": "Both contract, control contracts more",
    "hum_expands_control_contracts": "Hum expands, control contracts",
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    input_path = resolve_path(repo_root, args.input)
    output_dir = resolve_path(repo_root, args.output_dir)
    result = build_further_convergence_breakdown(
        input_path=input_path,
        output_dir=output_dir,
        plot_window_s=int(args.window_s),
        zero_tolerance_m=float(args.zero_tolerance_m),
    )
    print(json.dumps(result, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the RQ1 supplemental hum-vs-control convergence breakdown "
            "from an already materialized model input."
        )
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_MODEL_INPUT),
        help="Materialized RQ1 model input CSV/parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the supplemental artifact.",
    )
    parser.add_argument(
        "--window-s",
        default=str(DEFAULT_PLOT_WINDOW_S),
        help="Window length to use for the two ChatGPT-style plots.",
    )
    parser.add_argument(
        "--zero-tolerance-m",
        default=str(DEFAULT_ZERO_TOLERANCE_M),
        help="Absolute Delta D tolerance treated as zero when assigning expansion/contraction cases.",
    )
    return parser.parse_args(argv)


def build_further_convergence_breakdown(
    *,
    input_path: Path,
    output_dir: Path,
    plot_window_s: int = DEFAULT_PLOT_WINDOW_S,
    zero_tolerance_m: float = DEFAULT_ZERO_TOLERANCE_M,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    for directory in (tables_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_input = read_model_input(input_path)
    pair_level = build_pair_level_table(model_input, zero_tolerance_m=zero_tolerance_m)
    matched_set_audit = build_matched_set_audit(model_input)
    case_summary = summarize_cases(pair_level)
    convergent_magnitude = summarize_convergent_magnitude(pair_level)
    h1_directional_cases_by_window = build_h1_directional_cases_by_window(pair_level)
    h1_directional_cases = h1_directional_cases_by_window[int(plot_window_s)]

    if not pair_level["window_s"].eq(int(plot_window_s)).any():
        available = sorted(
            pair_level["window_s"].dropna().astype(int).unique().tolist()
        )
        raise ValueError(
            f"no complete paired rows are available for window_s={plot_window_s}; available windows: {available}"
        )

    pair_level.to_csv(
        tables_dir / "rq1_further_convergence_pair_level.csv", index=False
    )
    case_summary.to_csv(
        tables_dir / "rq1_further_convergence_case_summary.csv", index=False
    )
    convergent_magnitude.to_csv(
        tables_dir / "rq1_further_convergence_convergent_magnitude_summary.csv",
        index=False,
    )
    matched_set_audit.to_csv(
        tables_dir / "rq1_further_convergence_matched_set_audit.csv", index=False
    )
    for window_s, window_cases in h1_directional_cases_by_window.items():
        window_cases.to_csv(
            tables_dir / f"h1_directional_cases_w{window_s}.csv", index=False
        )
    h1_directional_cases.to_csv(tables_dir / "h1_directional_cases.csv", index=False)

    case_plot_paths: dict[int, Path] = {}
    convergent_plot_paths: dict[int, Path] = {}
    for window_s in sorted(h1_directional_cases_by_window):
        case_plot_paths[window_s] = figures_dir / f"cases_breakdown_w{window_s}.png"
        convergent_plot_paths[window_s] = (
            figures_dir / f"convergent_cases_breakdown_w{window_s}.png"
        )
        plot_case_breakdown(
            case_summary, window_s=window_s, path=case_plot_paths[window_s]
        )
        plot_convergent_magnitude(
            convergent_magnitude,
            window_s=window_s,
            path=convergent_plot_paths[window_s],
        )

    cases_plot = figures_dir / "cases_breakdown.png"
    convergent_plot = figures_dir / "convergent_cases_breakdown.png"
    plot_case_breakdown(case_summary, window_s=int(plot_window_s), path=cases_plot)
    plot_convergent_magnitude(
        convergent_magnitude, window_s=int(plot_window_s), path=convergent_plot
    )

    report_path = reports_dir / "rq1_further_convergence_breakdown.md"
    write_report(
        path=report_path,
        input_path=input_path,
        output_dir=output_dir,
        plot_window_s=int(plot_window_s),
        zero_tolerance_m=float(zero_tolerance_m),
        matched_set_audit=matched_set_audit,
        case_summary=case_summary,
        convergent_magnitude=convergent_magnitude,
    )

    manifest = {
        "source_input": str(input_path),
        "output_dir": str(output_dir),
        "plot_window_s": int(plot_window_s),
        "zero_tolerance_m": float(zero_tolerance_m),
        "pairing_rule": (
            "One complete comparison is one window_s x match_set_id group with exactly one hum row "
            "and exactly one control row in the materialized model input."
        ),
        "response": "Delta_D_w = post_distance_loo_centroid_m - pre_distance_loo_centroid_m; negative means contraction/convergence.",
        "h1_rule": "A paired row supports H1 when hum Delta_D_w is less than control Delta_D_w.",
        "h1_directional_cases_ratio": (
            "Hum-to-control mean ratio = mean hum Delta_D_w / mean control Delta_D_w "
            "within each case and window. This is a signed ratio because Delta_D_w can be "
            "positive for expansion or negative for contraction."
        ),
        "problematic_case": "* Both expand, but hum expands less.",
        "outputs": {
            "pair_level_csv": str(
                tables_dir / "rq1_further_convergence_pair_level.csv"
            ),
            "case_summary_csv": str(
                tables_dir / "rq1_further_convergence_case_summary.csv"
            ),
            "convergent_magnitude_summary_csv": str(
                tables_dir / "rq1_further_convergence_convergent_magnitude_summary.csv"
            ),
            "matched_set_audit_csv": str(
                tables_dir / "rq1_further_convergence_matched_set_audit.csv"
            ),
            "h1_directional_cases_csv": str(tables_dir / "h1_directional_cases.csv"),
            "h1_directional_cases_by_window_csv": {
                f"W{window_s}": str(
                    tables_dir / f"h1_directional_cases_w{window_s}.csv"
                )
                for window_s in sorted(h1_directional_cases_by_window)
            },
            "cases_breakdown_png": str(cases_plot),
            "convergent_cases_breakdown_png": str(convergent_plot),
            "cases_breakdown_by_window_png": {
                f"W{window_s}": str(path)
                for window_s, path in sorted(case_plot_paths.items())
            },
            "convergent_cases_breakdown_by_window_png": {
                f"W{window_s}": str(path)
                for window_s, path in sorted(convergent_plot_paths.items())
            },
            "report_md": str(report_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )
    return {
        "output_dir": str(output_dir),
        "plot_window_s": int(plot_window_s),
        "paired_rows": int(len(pair_level.index)),
        "plot_window_paired_rows": int(
            pair_level["window_s"].eq(int(plot_window_s)).sum()
        ),
        "manifest": str(output_dir / "manifest.json"),
    }


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def read_model_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing materialized RQ1 model input: {path}")
    if ".parquet" in path.suffixes:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"model input is missing required columns: {missing}")
    out = frame.copy()
    out["condition"] = pd.to_numeric(out["condition"], errors="coerce").astype("Int64")
    out["window_s"] = pd.to_numeric(out["window_s"], errors="coerce").astype("Int64")
    for column in [
        "Delta_D_w",
        "pre_distance_loo_centroid_m",
        "post_distance_loo_centroid_m",
        "match_rank",
        "standardized_match_distance",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["analysis_window_id", "match_set_id"]:
        out[column] = out[column].astype(str)
    return out


def build_pair_level_table(
    model_input: pd.DataFrame, *, zero_tolerance_m: float = DEFAULT_ZERO_TOLERANCE_M
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sort_columns = [
        column
        for column in [
            "standardized_match_distance",
            "match_rank",
            "analysis_window_id",
        ]
        if column in model_input.columns
    ]
    for (window_s, match_set_id), group in model_input.groupby(
        ["window_s", "match_set_id"], sort=True
    ):
        hum_rows = group.loc[group["condition"].eq(1)].copy()
        control_rows = group.loc[group["condition"].eq(0)].copy()
        if len(hum_rows.index) != 1 or len(control_rows.index) != 1:
            continue
        if sort_columns:
            hum_rows = hum_rows.sort_values(sort_columns, na_position="last")
            control_rows = control_rows.sort_values(sort_columns, na_position="last")
        hum = hum_rows.iloc[0]
        control = control_rows.iloc[0]
        hum_delta = float(hum["Delta_D_w"])
        control_delta = float(control["Delta_D_w"])
        if not (np.isfinite(hum_delta) and np.isfinite(control_delta)):
            continue
        case_id = classify_case(
            hum_delta_D_m=hum_delta,
            control_delta_D_m=control_delta,
            zero_tolerance_m=zero_tolerance_m,
        )
        contrast_m = hum_delta - control_delta
        rows.append(
            {
                "window_s": int(window_s),
                "match_set_id": str(match_set_id),
                "hum_analysis_window_id": str(hum["analysis_window_id"]),
                "control_analysis_window_id": str(control["analysis_window_id"]),
                "hum_pre_distance_m": float(hum["pre_distance_loo_centroid_m"]),
                "hum_post_distance_m": float(hum["post_distance_loo_centroid_m"]),
                "hum_delta_D_m": hum_delta,
                "hum_delta_D_cm": hum_delta * 100.0,
                "control_pre_distance_m": float(control["pre_distance_loo_centroid_m"]),
                "control_post_distance_m": float(
                    control["post_distance_loo_centroid_m"]
                ),
                "control_delta_D_m": control_delta,
                "control_delta_D_cm": control_delta * 100.0,
                "hum_minus_control_contrast_m": contrast_m,
                "control_minus_hum_contrast_m": -contrast_m,
                "h1_direction": CASE_BY_ID[case_id].h1_direction,
                "supports_h1": bool(contrast_m < -zero_tolerance_m),
                "case_id": case_id,
                "case_label": CASE_BY_ID[case_id].case_label,
                "hum_change_direction": signed_change_label(
                    hum_delta, zero_tolerance_m
                ),
                "control_change_direction": signed_change_label(
                    control_delta, zero_tolerance_m
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["window_s", "match_set_id"])
        .reset_index(drop=True)
    )


def classify_case(
    *,
    hum_delta_D_m: float,
    control_delta_D_m: float,
    zero_tolerance_m: float = DEFAULT_ZERO_TOLERANCE_M,
) -> str:
    hum_sign = signed_change_label(hum_delta_D_m, zero_tolerance_m)
    control_sign = signed_change_label(control_delta_D_m, zero_tolerance_m)
    hum_less = hum_delta_D_m < control_delta_D_m - zero_tolerance_m
    hum_greater = hum_delta_D_m > control_delta_D_m + zero_tolerance_m

    if hum_sign == "expands" and control_sign == "expands":
        if hum_less:
            return "both_expand_hum_expands_less"
        if hum_greater:
            return "both_expand_hum_expands_more"
        return "boundary_or_tie"
    if hum_sign == "contracts" and control_sign == "expands":
        return "hum_contracts_control_expands"
    if hum_sign == "contracts" and control_sign == "contracts":
        if hum_less:
            return "both_contract_hum_contracts_more"
        if hum_greater:
            return "both_contract_control_contracts_more"
        return "boundary_or_tie"
    if hum_sign == "expands" and control_sign == "contracts":
        return "hum_expands_control_contracts"
    return "boundary_or_tie"


def signed_change_label(
    delta_m: float, zero_tolerance_m: float = DEFAULT_ZERO_TOLERANCE_M
) -> str:
    if delta_m > zero_tolerance_m:
        return "expands"
    if delta_m < -zero_tolerance_m:
        return "contracts"
    return "stable"


def build_matched_set_audit(model_input: pd.DataFrame) -> pd.DataFrame:
    counts = (
        model_input.groupby(["window_s", "match_set_id"], sort=True)["condition"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: "control_rows", 1: "hum_rows"})
        .reset_index()
    )
    for column in ["hum_rows", "control_rows"]:
        if column not in counts:
            counts[column] = 0
    counts["matched_set_status"] = np.select(
        [
            counts["hum_rows"].eq(1) & counts["control_rows"].eq(1),
            counts["hum_rows"].gt(0) & counts["control_rows"].eq(0),
            counts["hum_rows"].eq(0) & counts["control_rows"].gt(0),
            counts["hum_rows"].gt(1) | counts["control_rows"].gt(1),
        ],
        [
            "complete_one_hum_one_control",
            "hum_without_control",
            "control_without_hum",
            "non_one_to_one_group",
        ],
        default="empty_group",
    )
    summary = (
        counts.groupby(["window_s", "matched_set_status"], observed=True)
        .agg(
            n_match_sets=("match_set_id", "size"),
            hum_rows=("hum_rows", "sum"),
            control_rows=("control_rows", "sum"),
        )
        .reset_index()
        .sort_values(["window_s", "matched_set_status"])
        .reset_index(drop=True)
    )
    return summary


def summarize_cases(pair_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_s, group in pair_level.groupby("window_s", sort=True):
        total_pairs = int(len(group.index))
        for definition in CASE_DEFINITIONS:
            current = group.loc[group["case_id"].eq(definition.case_id)]
            n_pairs = int(len(current.index))
            rows.append(
                {
                    "window_s": int(window_s),
                    "case_id": definition.case_id,
                    "case_label": definition.case_label,
                    "h1_direction": definition.h1_direction,
                    "n_pairs": n_pairs,
                    "total_pairs": total_pairs,
                    "percent_of_pairs": (n_pairs / total_pairs * 100.0)
                    if total_pairs
                    else np.nan,
                    "median_hum_delta_D_m": float(current["hum_delta_D_m"].median())
                    if n_pairs
                    else np.nan,
                    "median_control_delta_D_m": float(
                        current["control_delta_D_m"].median()
                    )
                    if n_pairs
                    else np.nan,
                    "median_hum_minus_control_contrast_m": (
                        float(current["hum_minus_control_contrast_m"].median())
                        if n_pairs
                        else np.nan
                    ),
                    "sum_abs_hum_minus_control_contrast_m": (
                        float(current["hum_minus_control_contrast_m"].abs().sum())
                        if n_pairs
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_convergent_magnitude(pair_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_s, group in pair_level.groupby("window_s", sort=True):
        supporting = group.loc[group["h1_direction"].eq("supports_h1")].copy()
        supporting["negative_contrast_magnitude_m"] = supporting[
            "control_minus_hum_contrast_m"
        ].clip(lower=0.0)
        total_magnitude = float(supporting["negative_contrast_magnitude_m"].sum())
        total_pairs = int(len(supporting.index))
        supporting["convergent_magnitude_group"] = np.where(
            supporting["case_id"].eq("both_expand_hum_expands_less"),
            "both_expand_hum_expands_less",
            "hum_actually_contracts",
        )
        for group_id in CONVERGENT_MAGNITUDE_ORDER:
            current = supporting.loc[
                supporting["convergent_magnitude_group"].eq(group_id)
            ]
            magnitude = (
                float(current["negative_contrast_magnitude_m"].sum())
                if not current.empty
                else 0.0
            )
            n_pairs = int(len(current.index))
            rows.append(
                {
                    "window_s": int(window_s),
                    "convergent_magnitude_group": group_id,
                    "plot_label": CONVERGENT_MAGNITUDE_LABELS[group_id],
                    "n_pairs": n_pairs,
                    "supporting_pairs": total_pairs,
                    "percent_of_supporting_pairs": (n_pairs / total_pairs * 100.0)
                    if total_pairs
                    else np.nan,
                    "negative_contrast_magnitude_m": magnitude,
                    "total_negative_contrast_magnitude_m": total_magnitude,
                    "percent_of_total_negative_contrast_magnitude": (
                        magnitude / total_magnitude * 100.0
                        if total_magnitude
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_h1_directional_cases_csv(
    pair_level: pd.DataFrame, *, window_s: int
) -> pd.DataFrame:
    current_window = pair_level.loc[pair_level["window_s"].eq(int(window_s))]
    total_pairs = len(current_window.index)
    rows: list[dict[str, Any]] = []
    for case_id in PLOT_CASE_IDS:
        current = current_window.loc[current_window["case_id"].eq(case_id)]
        mean_hum_delta_m = (
            float(current["hum_delta_D_m"].mean()) if not current.empty else np.nan
        )
        mean_control_delta_m = (
            float(current["control_delta_D_m"].mean()) if not current.empty else np.nan
        )
        rows.append(
            {
                "Case": H1_DIRECTIONAL_CASE_LABELS[case_id],
                "Mean hum Delta D (m)": round(mean_hum_delta_m, 2)
                if np.isfinite(mean_hum_delta_m)
                else np.nan,
                "Mean control Delta D (m)": round(mean_control_delta_m, 2)
                if np.isfinite(mean_control_delta_m)
                else np.nan,
                "Hum-to-control mean ratio": rounded_signed_ratio(
                    mean_hum_delta_m, mean_control_delta_m
                ),
                "Proportion": round(len(current.index) / total_pairs, 4)
                if total_pairs
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rounded_signed_ratio(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return np.nan
    if np.isclose(denominator, 0.0, atol=DEFAULT_ZERO_TOLERANCE_M):
        return np.nan
    return round(numerator / denominator, 4)


def build_h1_directional_cases_by_window(
    pair_level: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    return {
        int(window_s): build_h1_directional_cases_csv(
            pair_level, window_s=int(window_s)
        )
        for window_s in sorted(
            pair_level["window_s"].dropna().astype(int).unique().tolist()
        )
    }


def plot_case_breakdown(
    case_summary: pd.DataFrame, *, window_s: int, path: Path
) -> None:
    current = case_summary.loc[
        case_summary["window_s"].eq(int(window_s))
        & case_summary["case_id"].isin(PLOT_CASE_IDS)
    ].set_index("case_id")
    definitions = [CASE_BY_ID[case_id] for case_id in PLOT_CASE_IDS]
    labels = [definition.plot_label for definition in definitions]
    values = np.array(
        [
            float(current.loc[definition.case_id, "percent_of_pairs"])
            for definition in definitions
        ]
    )
    colors = [
        "#1b9e77" if definition.h1_direction == "supports_h1" else "#d95f02"
        for definition in definitions
    ]

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.55)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("% of comparisons", fontsize=18)
    ax.set_xlabel("Case", fontsize=18, labelpad=18)
    ax.tick_params(axis="both", labelsize=16)
    upper = max(30.0, float(np.nanmax(values)) + 3.0)
    ax.set_ylim(0, upper)
    ax.legend(
        handles=[
            Patch(facecolor="#1b9e77", label="Supports H1"),
            Patch(facecolor="#d95f02", label="Against H1"),
        ],
        frameon=False,
        fontsize=18,
        loc="upper right",
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + upper * 0.02,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=16,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_convergent_magnitude(
    convergent_magnitude: pd.DataFrame, *, window_s: int, path: Path
) -> None:
    current = convergent_magnitude.loc[
        convergent_magnitude["window_s"].eq(int(window_s))
    ].set_index("convergent_magnitude_group")
    values = np.array(
        [
            float(current.loc[group_id, "percent_of_total_negative_contrast_magnitude"])
            for group_id in CONVERGENT_MAGNITUDE_ORDER
        ]
    )
    labels = [
        CONVERGENT_MAGNITUDE_LABELS[group_id] for group_id in CONVERGENT_MAGNITUDE_ORDER
    ]

    fig, ax = plt.subplots(figsize=(6.58, 4.5))
    bars = ax.bar(
        np.arange(len(values)), values, color=["#1f77b4", "#ff7f0e"], width=0.8
    )
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of total negative contrast magnitude")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(
    *,
    path: Path,
    input_path: Path,
    output_dir: Path,
    plot_window_s: int,
    zero_tolerance_m: float,
    matched_set_audit: pd.DataFrame,
    case_summary: pd.DataFrame,
    convergent_magnitude: pd.DataFrame,
) -> None:
    selected_cases = case_summary.loc[
        case_summary["window_s"].eq(int(plot_window_s))
    ].copy()
    selected_magnitude = convergent_magnitude.loc[
        convergent_magnitude["window_s"].eq(int(plot_window_s))
    ].copy()
    complete_pairs = (
        int(selected_cases["total_pairs"].max()) if not selected_cases.empty else 0
    )
    support_pairs = int(
        selected_cases.loc[
            selected_cases["h1_direction"].eq("supports_h1"), "n_pairs"
        ].sum()
    )
    against_pairs = int(
        selected_cases.loc[
            selected_cases["h1_direction"].eq("against_h1"), "n_pairs"
        ].sum()
    )
    problematic = selected_cases.loc[
        selected_cases["case_id"].eq("both_expand_hum_expands_less")
    ]
    problematic_percent = (
        float(problematic["percent_of_pairs"].iloc[0])
        if not problematic.empty
        else np.nan
    )
    problematic_magnitude = selected_magnitude.loc[
        selected_magnitude["convergent_magnitude_group"].eq(
            "both_expand_hum_expands_less"
        )
    ]
    problematic_magnitude_percent = (
        float(
            problematic_magnitude["percent_of_total_negative_contrast_magnitude"].iloc[
                0
            ]
        )
        if not problematic_magnitude.empty
        else np.nan
    )

    lines = [
        "# RQ1 Further Convergence Breakdown",
        "",
        f"- Source input: `{input_path}`",
        f"- Output directory: `{output_dir}`",
        f"- Plot window: W{plot_window_s}",
        f"- Zero/tie tolerance: {zero_tolerance_m:g} m",
        "",
        "## Method",
        "",
        "- One comparison is one `window_s` x `match_set_id` group with exactly one hum row and exactly one control row.",
        "- `Delta_D_w` is post minus pre focal-to-centroid distance; negative values mean contraction/convergence.",
        "- A pair supports H1 when the hum delta is lower than the control delta.",
        "- The starred case is diagnostic because it supports the contrast only by expanding less than the matched control, not by actual contraction.",
        "",
        "## Primary W60 Readout",
        "",
        f"- Complete W{plot_window_s} comparisons: {complete_pairs:,}",
        f"- Supports H1: {support_pairs:,} ({support_pairs / complete_pairs * 100.0:.2f}%)"
        if complete_pairs
        else "- Supports H1: unavailable",
        f"- Against H1: {against_pairs:,} ({against_pairs / complete_pairs * 100.0:.2f}%)"
        if complete_pairs
        else "- Against H1: unavailable",
        (
            f"- Starred both-expanding case: {problematic_percent:.2f}% of complete comparisons and "
            f"{problematic_magnitude_percent:.1f}% of the total H1-supporting contrast magnitude."
            if np.isfinite(problematic_percent)
            and np.isfinite(problematic_magnitude_percent)
            else "- Starred both-expanding case: unavailable"
        ),
        "",
        "## Case Summary",
        "",
        frame_text(
            selected_cases[
                [
                    "case_label",
                    "h1_direction",
                    "n_pairs",
                    "percent_of_pairs",
                    "median_hum_delta_D_m",
                    "median_control_delta_D_m",
                ]
            ]
        ),
        "",
        "## H1-Supporting Contrast Magnitude",
        "",
        frame_text(
            selected_magnitude[
                [
                    "plot_label",
                    "n_pairs",
                    "percent_of_supporting_pairs",
                    "negative_contrast_magnitude_m",
                    "percent_of_total_negative_contrast_magnitude",
                ]
            ]
        ),
        "",
        "## Matched-Set Audit",
        "",
        frame_text(
            matched_set_audit.loc[matched_set_audit["window_s"].eq(int(plot_window_s))]
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf8")


def frame_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    return (
        "```text\n"
        + frame.to_string(index=False, float_format=lambda value: f"{value:.6g}")
        + "\n```"
    )


if __name__ == "__main__":
    raise SystemExit(main())

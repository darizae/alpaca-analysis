from __future__ import annotations

import json
import math
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import wavfile

from .audio import apply_path_rewrites, extract_snippet, load_wav
from .cache import atomic_replace, file_lock
from .config import DedupConfig
from .constants import (
    CURRENT_SCHEMA_VERSION,
    DECISION_METHOD_VERSION,
    DEFAULT_FEATURE_COLUMNS,
    REPRESENTATIVE_SELECTION_RULE,
    ROBUSTNESS_METHOD_VERSION,
)
from .features import dedup_config_hash, feature_config_hash, launch_paths_for_config
from .manifest_utils import runtime_context, stable_json_hash, write_json
from .paths import LaunchPaths
from .progress import read_progress, utc_now_iso, write_launch_progress, write_stage_progress


def _overlap_short_setting_label(value: float) -> str:
    return f"ovshort{int(round(value * 100)):03d}"


def _gps_setting_label(value: float) -> str:
    return f"g{int(round(value)):02d}"


def _relative_setting_label(value: float | None) -> str:
    if value is None:
        return "relnone"
    return f"rel{int(round(value * 100)):02d}"


def _group_acoustic_setting_label(value: str | None) -> str:
    if value is None:
        return "acnone"
    return f"ac{value}"


def _model_id(
    overlap_fraction_short_threshold: float,
    gps_threshold_m: float,
    relative_difference_threshold: float | None,
    group_acoustic_threshold: str | None,
) -> str:
    return (
        f"dedup_{_overlap_short_setting_label(overlap_fraction_short_threshold)}"
        f"_{_gps_setting_label(gps_threshold_m)}"
        f"_{_relative_setting_label(relative_difference_threshold)}"
        f"_{_group_acoustic_setting_label(group_acoustic_threshold)}"
    )


def _batch_id(overlap_fraction_short_threshold: float, gps_threshold_m: float) -> str:
    return (
        f"batch_{_overlap_short_setting_label(overlap_fraction_short_threshold)}"
        f"_{_gps_setting_label(gps_threshold_m)}"
    )


def _group_acoustic_quantile(label: str | None) -> float | None:
    if label is None:
        return None
    if label == "q90":
        return 0.90
    if label == "q75":
        return 0.75
    if label == "q50":
        return 0.50
    raise ValueError(f"Unsupported group acoustic threshold: {label}")


def _duplicate_pair_key(left_hum_id: str, right_hum_id: str) -> str:
    left, right = sorted((str(left_hum_id), str(right_hum_id)))
    return f"{left}\t{right}"


def _cluster_duplicate_pair_keys(member_ids: list[str]) -> list[str]:
    ordered_members = sorted(str(member_id) for member_id in member_ids)
    pair_keys: list[str] = []
    for left_index in range(len(ordered_members)):
        for right_index in range(left_index + 1, len(ordered_members)):
            pair_keys.append(_duplicate_pair_key(ordered_members[left_index], ordered_members[right_index]))
    return pair_keys


def _dice_overlap(left_pairs: set[str], right_pairs: set[str]) -> float:
    if not left_pairs and not right_pairs:
        return 1.0
    denominator = len(left_pairs) + len(right_pairs)
    if denominator <= 0:
        return 0.0
    return float((2 * len(left_pairs & right_pairs)) / denominator)


def _acoustic_margin_label(config: DedupConfig) -> str:
    return f"acoustic_margin_q{int(round(config.robustness.acoustic_margin_quantile * 100)):02d}"


def prepare_grid(config: DedupConfig) -> LaunchPaths:
    paths = launch_paths_for_config(config)
    model_rows: list[dict[str, Any]] = []
    for overlap_index, overlap_fraction_short_threshold in enumerate(config.overlap_fraction_short_thresholds):
        for gps_index, gps_threshold_m in enumerate(config.gps_thresholds_m):
            for relative_index, relative_difference_threshold in enumerate(config.relative_difference_thresholds):
                for acoustic_index, group_acoustic_threshold in enumerate(config.group_acoustic_thresholds):
                    model_rows.append(
                        {
                            "dedup_model_id": _model_id(
                                float(overlap_fraction_short_threshold),
                                float(gps_threshold_m),
                                relative_difference_threshold,
                                group_acoustic_threshold,
                            ),
                            "T_overlap_short": float(overlap_fraction_short_threshold),
                            "T_gps_m": float(gps_threshold_m),
                            "relative_difference_setting": _relative_setting_label(relative_difference_threshold),
                            "relative_difference_threshold": relative_difference_threshold,
                            "group_acoustic_setting": _group_acoustic_setting_label(group_acoustic_threshold),
                            "group_acoustic_threshold": group_acoustic_threshold,
                            "overlap_index": int(overlap_index),
                            "gps_index": int(gps_index),
                            "relative_index": int(relative_index),
                            "acoustic_index": int(acoustic_index),
                            "batch_id": _batch_id(float(overlap_fraction_short_threshold), float(gps_threshold_m)),
                        }
                    )
    models_df = pd.DataFrame(model_rows).sort_values(
        ["overlap_index", "gps_index", "relative_index", "acoustic_index"],
        kind="stable",
    ).reset_index(drop=True)
    models_df.to_parquet(paths.manifests_dir / "grid_models.parquet", index=False)
    models_df.to_csv(paths.manifests_dir / "grid_models.csv", index=False)

    batches_df = (
        models_df.groupby(["T_overlap_short", "T_gps_m", "batch_id"], sort=True)["dedup_model_id"]
        .agg(list)
        .reset_index(name="model_ids")
        .sort_values(["T_overlap_short", "T_gps_m"], kind="stable")
        .reset_index(drop=True)
    )
    batches_df["model_count"] = batches_df["model_ids"].apply(len)
    batches_df["model_ids_json"] = batches_df["model_ids"].apply(json.dumps)
    batches_df.to_parquet(paths.manifests_dir / "grid_batches.parquet", index=False)
    batches_df.to_csv(paths.manifests_dir / "grid_batches.csv", index=False)

    adjacency_rows: list[dict[str, Any]] = []
    by_key = {
        (int(row.overlap_index), int(row.gps_index), int(row.relative_index), int(row.acoustic_index)): row.dedup_model_id
        for row in models_df.itertuples(index=False)
    }
    axes = [
        ("overlap_index", len(config.overlap_fraction_short_thresholds)),
        ("gps_index", len(config.gps_thresholds_m)),
        ("relative_index", len(config.relative_difference_thresholds)),
        ("acoustic_index", len(config.group_acoustic_thresholds)),
    ]
    for row in models_df.itertuples(index=False):
        base_key = [int(row.overlap_index), int(row.gps_index), int(row.relative_index), int(row.acoustic_index)]
        for axis_index, (axis_name, axis_size) in enumerate(axes):
            for delta in (-1, 1):
                neighbor_key = list(base_key)
                neighbor_key[axis_index] += delta
                if 0 <= neighbor_key[axis_index] < axis_size:
                    neighbor_id = by_key.get(tuple(neighbor_key))
                    if neighbor_id is None:
                        continue
                    adjacency_rows.append(
                        {
                            "dedup_model_id": row.dedup_model_id,
                            "neighbor_model_id": neighbor_id,
                            "axis": axis_name,
                        }
                    )
    adjacency_df = pd.DataFrame(adjacency_rows).drop_duplicates().sort_values(
        ["dedup_model_id", "neighbor_model_id", "axis"],
        kind="stable",
    )
    adjacency_df.to_parquet(paths.manifests_dir / "model_adjacency.parquet", index=False)
    adjacency_df.to_csv(paths.manifests_dir / "model_adjacency.csv", index=False)

    write_json(
        paths.manifests_dir / "ranking_manifest.json",
        {
            "decision_method_version": DECISION_METHOD_VERSION,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
            "model_count": int(len(models_df.index)),
            "batch_count": int(len(batches_df.index)),
            "neighbor_axes": ["T_overlap_short", "T_gps_m", "relative_difference_threshold", "group_acoustic_threshold"],
            "ranking_formula": {
                "local_decision_stability": "median(Dice overlap between the model duplicate-pair decision set and each neighbor duplicate-pair decision set)",
                "removal_count_stability": "1 - abs(N_hums_removed(model) - median(N_hums_removed(neighbors))) / max(N_hums_removed(model), median_neighbor_removed, 1)",
                _acoustic_margin_label(config): "configured quantile of clipped edge acoustic clearance over accepted edges with real acoustic thresholds",
                "acoustic_threshold_coverage": "N_accepted_edges_with_real_acoustic_threshold / max(N_accepted_edges, 1)",
                "final_selectable": (
                    "T_rel != none AND T_acoustic != none AND "
                    f"acoustic_threshold_coverage >= {config.robustness.min_final_selectable_acoustic_coverage:.2f}"
                ),
                "acoustic_strength": (
                    "acoustic_threshold_coverage * "
                    f"min({_acoustic_margin_label(config)} / {config.robustness.target_acoustic_margin:.2f}, 1.0)"
                ),
                "decision_acoustic_core": "harmonic mean of local_decision_stability and acoustic_strength, with 0 when the denominator is 0",
                "robustness_score": (
                    f"{config.robustness.decision_acoustic_core_weight:.2f} * decision_acoustic_core + "
                    f"{config.robustness.removal_count_stability_weight:.2f} * removal_count_stability "
                    "for final-selectable models only"
                ),
            },
            "top_k_models": int(config.top_k_models),
        },
    )

    write_stage_progress(
        paths,
        "grid",
        {
            "stage_status": "prepared",
            "stage_started_at": utc_now_iso(),
            "stage_finished_at": None,
            "total_grid_batches": int(len(batches_df.index)),
            "completed_grid_batches": 0,
            "failed_grid_batches": 0,
            "total_models": int(len(models_df.index)),
            "completed_models": 0,
            "failed_models": 0,
            "cached_models_reused": 0,
        },
    )
    launch_progress = read_progress(paths.progress_dir / "launch_progress.json")
    write_launch_progress(
        paths,
        {
            **launch_progress,
            "launch_id": paths.launch_dir.name,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
            "current_stage": "grid",
            "stage_status": "prepared",
            "total_grid_batches": int(len(batches_df.index)),
            "completed_grid_batches": 0,
            "failed_grid_batches": 0,
            "total_models": int(len(models_df.index)),
            "completed_models": 0,
            "failed_models": 0,
            "cached_models_reused": 0,
            "top_k_requested": int(config.top_k_models),
        },
    )
    return paths


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    return 2.0 * radius_m * math.asin(math.sqrt(a))


def _interval_overlap_s(start_i: float, end_i: float, start_j: float, end_j: float) -> float:
    return max(0.0, min(end_i, end_j) - max(start_i, start_j))


def _overlap_fraction_short(start_i: float, end_i: float, start_j: float, end_j: float) -> float:
    duration_i = max(end_i - start_i, 0.0)
    duration_j = max(end_j - start_j, 0.0)
    shorter_duration = min(duration_i, duration_j)
    if shorter_duration <= 0.0:
        return 0.0
    return _interval_overlap_s(start_i, end_i, start_j, end_j) / shorter_duration


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def _load_feature_manifest_hashes(paths: LaunchPaths) -> dict[str, Any]:
    manifest_path = paths.tables_dir / "hum_features_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing feature manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf8"))


def _candidate_cache_metadata(config: DedupConfig, paths: LaunchPaths, hum_features_df: pd.DataFrame) -> dict[str, Any]:
    feature_manifest = _load_feature_manifest_hashes(paths)
    input_table_hash = feature_manifest["input_table"]["sha256"]
    source_index_hash = feature_manifest["source_wav_index"]["sha256"]
    source_audio_manifest_hash = stable_json_hash(
        {
            "feature_source_audio_files": hum_features_df["feature_source_audio_file"].astype(str).sort_values().tolist()
        }
    )
    cache_key = stable_json_hash(
        {
            "algorithm_version": config.algorithm_version,
            "min_overlap_fraction_short": min(config.overlap_fraction_short_thresholds),
            "max_gps_m": max(config.gps_thresholds_m),
            "feature_config_hash": feature_config_hash(config),
            "dedup_config_hash": dedup_config_hash(config),
            "input_table_hash": input_table_hash,
            "source_index_hash": source_index_hash,
            "schema_version": CURRENT_SCHEMA_VERSION,
            "decision_method_version": DECISION_METHOD_VERSION,
        }
    )[:16]
    return {
        "cache_key": cache_key,
        "input_manifest_hash": stable_json_hash(
            {
                "algorithm_version": config.algorithm_version,
                "min_overlap_fraction_short": min(config.overlap_fraction_short_thresholds),
                "max_gps_m": max(config.gps_thresholds_m),
                "hum_rows": int(len(hum_features_df.index)),
            }
        ),
        "source_table_hash": input_table_hash,
        "source_audio_manifest_hash": source_audio_manifest_hash,
        "feature_config_hash": feature_config_hash(config),
        "dedup_config_hash": dedup_config_hash(config),
        "algorithm_version": config.algorithm_version,
        "runtime_context": runtime_context(),
        "schema_version": CURRENT_SCHEMA_VERSION,
        "decision_method_version": DECISION_METHOD_VERSION,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf8"))


def _metadata_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    existing = _load_json(path)
    for key, value in expected.items():
        if existing.get(key) != value:
            return False
    return True


def build_candidate_pair_cache(config: DedupConfig, paths: LaunchPaths) -> pd.DataFrame:
    hum_features_df = pd.read_parquet(paths.tables_dir / "hum_features.parquet")
    expected_metadata = _candidate_cache_metadata(config, paths, hum_features_df)
    cache_dir = paths.cache_dir / "candidate_pairs" / expected_metadata["cache_key"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / "candidate_pairs.parquet"
    metadata_path = cache_dir / "candidate_pairs.json"
    lock_path = cache_dir / ".lock"

    if final_path.is_file() and _metadata_matches(metadata_path, expected_metadata):
        return pd.read_parquet(final_path)

    with file_lock(lock_path):
        if final_path.is_file() and _metadata_matches(metadata_path, expected_metadata):
            return pd.read_parquet(final_path)

        working_df = hum_features_df.loc[hum_features_df["has_any_gps"]].copy()
        working_df = working_df.sort_values(["start_time_epoch_s", "hum_id"], kind="stable").reset_index(drop=True)
        min_overlap_short = float(min(config.overlap_fraction_short_thresholds))
        max_gps_m = float(max(config.gps_thresholds_m))
        pair_rows: list[dict[str, Any]] = []

        hum_ids = working_df["hum_id"].astype(str).tolist()
        collars = working_df["collar_id"].astype(str).tolist()
        start_times = working_df["start_time_epoch_s"].astype(float).tolist()
        end_times = working_df["end_time_epoch_s"].astype(float).tolist()
        latitudes = working_df["latitude"].astype(float).tolist()
        longitudes = working_df["longitude"].astype(float).tolist()

        right_index = 0
        for left_index in range(len(working_df.index)):
            if right_index < left_index + 1:
                right_index = left_index + 1
            while right_index < len(working_df.index) and start_times[right_index] < end_times[left_index]:
                right_index += 1
            for candidate_index in range(left_index + 1, right_index):
                if collars[left_index] == collars[candidate_index]:
                    continue
                overlap_s = _interval_overlap_s(
                    start_times[left_index],
                    end_times[left_index],
                    start_times[candidate_index],
                    end_times[candidate_index],
                )
                if overlap_s <= 0.0:
                    continue
                overlap_fraction_short = _overlap_fraction_short(
                    start_times[left_index],
                    end_times[left_index],
                    start_times[candidate_index],
                    end_times[candidate_index],
                )
                if overlap_fraction_short < min_overlap_short:
                    continue
                gps_distance_m = _haversine_m(
                    latitudes[left_index],
                    longitudes[left_index],
                    latitudes[candidate_index],
                    longitudes[candidate_index],
                )
                if gps_distance_m > max_gps_m:
                    continue
                pair_rows.append(
                    {
                        "hum_id_i": hum_ids[left_index],
                        "hum_id_j": hum_ids[candidate_index],
                        "collar_id_i": collars[left_index],
                        "collar_id_j": collars[candidate_index],
                        "overlap_s": float(overlap_s),
                        "overlap_fraction_short": float(overlap_fraction_short),
                        "gps_distance_m": float(gps_distance_m),
                    }
                )
        candidate_df = (
            pd.DataFrame(pair_rows)
            .sort_values(
                ["overlap_fraction_short", "gps_distance_m", "hum_id_i", "hum_id_j"],
                ascending=[False, True, True, True],
                kind="stable",
            )
            .reset_index(drop=True)
            if pair_rows
            else pd.DataFrame(
                columns=[
                    "hum_id_i",
                    "hum_id_j",
                    "collar_id_i",
                    "collar_id_j",
                    "overlap_s",
                    "overlap_fraction_short",
                    "gps_distance_m",
                ]
            )
        )
        temp_path = final_path.with_suffix(".tmp.parquet")
        candidate_df.to_parquet(temp_path, index=False)
        atomic_replace(temp_path, final_path)
        write_json(
            metadata_path,
            {
                **expected_metadata,
                "creation_timestamp_utc": utc_now_iso(),
                "row_count": int(len(candidate_df.index)),
            },
        )
        return candidate_df


def _relative_incompatibility(left: pd.Series, right: pd.Series) -> float:
    values: list[float] = []
    for column in DEFAULT_FEATURE_COLUMNS:
        left_value = float(left[column])
        right_value = float(right[column])
        denominator = max(abs(left_value), abs(right_value), 1e-9)
        values.append(abs(left_value - right_value) / denominator)
    return float(max(values)) if values else 0.0


def _pairwise_distance_lookup(feature_matrix: np.ndarray, member_ids: list[str]) -> dict[tuple[str, str], float]:
    if len(member_ids) <= 1:
        return {}
    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0)
    stds = np.where(stds <= 1e-12, 1.0, stds)
    z_matrix = (feature_matrix - means) / stds
    distances: dict[tuple[str, str], float] = {}
    for left_index in range(len(member_ids)):
        for right_index in range(left_index + 1, len(member_ids)):
            distance = float(np.linalg.norm(z_matrix[left_index] - z_matrix[right_index]))
            distances[(member_ids[left_index], member_ids[right_index])] = distance
    return distances


def _build_batch_context(
    hum_features_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    batch_row: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap_threshold = float(batch_row["T_overlap_short"])
    gps_threshold_m = float(batch_row["T_gps_m"])
    batch_candidates_df = candidate_df.loc[
        (candidate_df["overlap_fraction_short"] >= overlap_threshold)
        & (candidate_df["gps_distance_m"] <= gps_threshold_m)
    ].copy()
    batch_candidates_df = batch_candidates_df.sort_values(
        ["overlap_fraction_short", "gps_distance_m", "hum_id_i", "hum_id_j"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    hum_lookup = hum_features_df.set_index("hum_id")
    hum_ids = hum_features_df["hum_id"].astype(str).tolist()
    union_find = UnionFind(hum_ids)
    for pair in batch_candidates_df.itertuples(index=False):
        union_find.union(str(pair.hum_id_i), str(pair.hum_id_j))

    provisional_groups: dict[str, list[str]] = {}
    for hum_id in hum_ids:
        provisional_groups.setdefault(union_find.find(hum_id), []).append(hum_id)

    sorted_groups = sorted(
        provisional_groups.values(),
        key=lambda members: (
            float(hum_lookup.loc[members]["start_time_epoch_s"].min()),
            min(str(member) for member in members),
        ),
    )

    annotated_groups: list[dict[str, Any]] = []
    annotated_edges: list[pd.DataFrame] = []
    for group_index, members in enumerate(sorted_groups, start=1):
        ordered_members = (
            hum_lookup.loc[members]
            .reset_index()
            .sort_values(["start_time_epoch_s", "hum_id"], kind="stable")
            ["hum_id"]
            .astype(str)
            .tolist()
        )
        group_id = f"{batch_row['batch_id']}__prov_{group_index:05d}"
        group_size = int(len(ordered_members))
        group_size_class = "size_1" if group_size == 1 else ("size_2" if group_size == 2 else "size_3plus")
        group_candidate_edges = batch_candidates_df.loc[
            batch_candidates_df["hum_id_i"].isin(ordered_members) & batch_candidates_df["hum_id_j"].isin(ordered_members)
        ].copy()

        annotated_groups.append(
            {
                "provisional_group_id": group_id,
                "member_ids": ordered_members,
                "member_ids_json": json.dumps(ordered_members),
                "provisional_group_size": group_size,
                "group_size_class": group_size_class,
                "group_start_time_epoch_s": float(hum_lookup.loc[ordered_members]["start_time_epoch_s"].min()),
            }
        )
        if group_candidate_edges.empty:
            continue

        group_candidate_edges["provisional_group_id"] = group_id
        group_candidate_edges["provisional_group_size"] = group_size
        group_candidate_edges["group_size_class"] = group_size_class
        group_candidate_edges["size2_relative_incompatibility"] = np.nan
        group_candidate_edges["group_acoustic_distance"] = np.nan
        group_candidate_edges["group_threshold_q50"] = np.nan
        group_candidate_edges["group_threshold_q75"] = np.nan
        group_candidate_edges["group_threshold_q90"] = np.nan
        group_candidate_edges["reference_incompatibility"] = np.nan

        if group_size == 2:
            left = hum_lookup.loc[str(group_candidate_edges.iloc[0]["hum_id_i"])]
            right = hum_lookup.loc[str(group_candidate_edges.iloc[0]["hum_id_j"])]
            incompatibility = _relative_incompatibility(left, right)
            group_candidate_edges.loc[:, "size2_relative_incompatibility"] = incompatibility
            group_candidate_edges.loc[:, "reference_incompatibility"] = incompatibility
        else:
            group_features = hum_lookup.loc[ordered_members, DEFAULT_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            pair_distances = _pairwise_distance_lookup(group_features, ordered_members)
            all_distances = np.asarray(list(pair_distances.values()), dtype=np.float64)
            q50 = float(np.quantile(all_distances, 0.50)) if len(all_distances) else 0.0
            q75 = float(np.quantile(all_distances, 0.75)) if len(all_distances) else 0.0
            q90 = float(np.quantile(all_distances, 0.90)) if len(all_distances) else 0.0
            group_candidate_edges.loc[:, "group_threshold_q50"] = q50
            group_candidate_edges.loc[:, "group_threshold_q75"] = q75
            group_candidate_edges.loc[:, "group_threshold_q90"] = q90
            group_candidate_edges.loc[:, "group_acoustic_distance"] = [
                pair_distances[(str(row.hum_id_i), str(row.hum_id_j))]
                for row in group_candidate_edges.itertuples(index=False)
            ]
            group_candidate_edges.loc[:, "reference_incompatibility"] = group_candidate_edges[
                "group_acoustic_distance"
            ].astype(float)
        annotated_edges.append(group_candidate_edges)

    groups_df = pd.DataFrame(annotated_groups).sort_values(
        ["group_start_time_epoch_s", "provisional_group_id"],
        kind="stable",
    )
    if annotated_edges:
        annotated_candidates_df = pd.concat(annotated_edges, ignore_index=True)
        annotated_candidates_df = annotated_candidates_df.sort_values(
            ["overlap_fraction_short", "gps_distance_m", "hum_id_i", "hum_id_j"],
            ascending=[False, True, True, True],
            kind="stable",
        ).reset_index(drop=True)
    else:
        annotated_candidates_df = pd.DataFrame(
            columns=[
                "hum_id_i",
                "hum_id_j",
                "collar_id_i",
                "collar_id_j",
                "overlap_s",
                "overlap_fraction_short",
                "gps_distance_m",
                "provisional_group_id",
                "provisional_group_size",
                "group_size_class",
                "size2_relative_incompatibility",
                "group_acoustic_distance",
                "group_threshold_q50",
                "group_threshold_q75",
                "group_threshold_q90",
                "reference_incompatibility",
            ]
        )
    return annotated_candidates_df, groups_df


def _representative_row(cluster_df: pd.DataFrame) -> pd.Series:
    return (
        cluster_df.sort_values(
            ["snippet_snr_db", "snippet_rms", "model_positive_score", "start_time_utc", "hum_id"],
            ascending=[False, False, False, True, True],
            kind="stable",
        )
        .iloc[0]
    )


def _component_sizes(hum_ids: list[str], accepted_edges_df: pd.DataFrame) -> dict[str, list[str]]:
    union_find = UnionFind(hum_ids)
    for pair in accepted_edges_df.itertuples(index=False):
        union_find.union(str(pair.hum_id_i), str(pair.hum_id_j))
    clusters: dict[str, list[str]] = {}
    for hum_id in hum_ids:
        clusters.setdefault(union_find.find(hum_id), []).append(hum_id)
    return clusters


def _evaluate_model_from_batch(
    hum_features_df: pd.DataFrame,
    batch_candidates_df: pd.DataFrame,
    provisional_groups_df: pd.DataFrame,
    model_row: dict[str, Any],
    *,
    materialize_details: bool = False,
) -> dict[str, Any]:
    hum_lookup = hum_features_df.set_index("hum_id")
    hum_ids = hum_features_df["hum_id"].astype(str).tolist()
    accepted_edges_df = batch_candidates_df.copy()
    if accepted_edges_df.empty:
        accepted_edges_df = accepted_edges_df.assign(
            passed_acoustic_filter=pd.Series(dtype=bool),
            threshold_type=pd.Series(dtype="object"),
            threshold_setting=pd.Series(dtype="object"),
            threshold_value=pd.Series(dtype=float),
            acoustic_incompatibility=pd.Series(dtype=float),
        )

    size2_mask = accepted_edges_df["group_size_class"] == "size_2"
    size3_mask = accepted_edges_df["group_size_class"] == "size_3plus"
    if model_row["relative_difference_threshold"] is None:
        accepted_edges_df.loc[size2_mask, "passed_acoustic_filter"] = True
        accepted_edges_df.loc[size2_mask, "threshold_type"] = "relative_difference"
        accepted_edges_df.loc[size2_mask, "threshold_setting"] = "none"
        accepted_edges_df.loc[size2_mask, "threshold_value"] = np.nan
        accepted_edges_df.loc[size2_mask, "acoustic_incompatibility"] = accepted_edges_df.loc[
            size2_mask,
            "size2_relative_incompatibility",
        ].astype(float)
    else:
        relative_threshold = float(model_row["relative_difference_threshold"])
        accepted_edges_df.loc[size2_mask, "passed_acoustic_filter"] = (
            accepted_edges_df.loc[size2_mask, "size2_relative_incompatibility"].astype(float) <= relative_threshold
        )
        accepted_edges_df.loc[size2_mask, "threshold_type"] = "relative_difference"
        accepted_edges_df.loc[size2_mask, "threshold_setting"] = model_row["relative_difference_setting"]
        accepted_edges_df.loc[size2_mask, "threshold_value"] = relative_threshold
        accepted_edges_df.loc[size2_mask, "acoustic_incompatibility"] = accepted_edges_df.loc[
            size2_mask,
            "size2_relative_incompatibility",
        ].astype(float)

    acoustic_threshold = model_row["group_acoustic_threshold"]
    if acoustic_threshold is None:
        accepted_edges_df.loc[size3_mask, "passed_acoustic_filter"] = True
        accepted_edges_df.loc[size3_mask, "threshold_type"] = "group_acoustic_distance"
        accepted_edges_df.loc[size3_mask, "threshold_setting"] = "none"
        accepted_edges_df.loc[size3_mask, "threshold_value"] = np.nan
        accepted_edges_df.loc[size3_mask, "acoustic_incompatibility"] = accepted_edges_df.loc[
            size3_mask,
            "group_acoustic_distance",
        ].astype(float)
    else:
        threshold_column = {
            "q50": "group_threshold_q50",
            "q75": "group_threshold_q75",
            "q90": "group_threshold_q90",
        }[str(acoustic_threshold)]
        threshold_values = accepted_edges_df.loc[size3_mask, threshold_column].astype(float)
        accepted_edges_df.loc[size3_mask, "passed_acoustic_filter"] = (
            accepted_edges_df.loc[size3_mask, "group_acoustic_distance"].astype(float) <= threshold_values
        )
        accepted_edges_df.loc[size3_mask, "threshold_type"] = "group_acoustic_distance"
        accepted_edges_df.loc[size3_mask, "threshold_setting"] = model_row["group_acoustic_setting"]
        accepted_edges_df.loc[size3_mask, "threshold_value"] = threshold_values
        accepted_edges_df.loc[size3_mask, "acoustic_incompatibility"] = accepted_edges_df.loc[
            size3_mask,
            "group_acoustic_distance",
        ].astype(float)

    if not accepted_edges_df.empty:
        accepted_edges_df = accepted_edges_df.loc[accepted_edges_df["passed_acoustic_filter"]].copy()
    clusters = _component_sizes(hum_ids, accepted_edges_df)

    sorted_clusters = sorted(
        clusters.values(),
        key=lambda members: (
            float(hum_lookup.loc[members]["start_time_epoch_s"].min()),
            min(str(member) for member in members),
        ),
    )
    duplicate_cluster_sizes = [len(members) for members in sorted_clusters if len(members) > 1]
    duplicate_pair_keys = sorted(
        pair_key
        for members in sorted_clusters
        if len(members) > 1
        for pair_key in _cluster_duplicate_pair_keys(members)
    )
    removed_as_duplicates = int(len(hum_ids) - len(sorted_clusters))
    reference_values = batch_candidates_df["reference_incompatibility"].dropna().astype(float)
    accepted_values = accepted_edges_df["acoustic_incompatibility"].dropna().astype(float)
    accepted_incompatibility_median = float(accepted_values.median()) if not accepted_values.empty else None
    reference_incompatibility_median = float(reference_values.median()) if not reference_values.empty else None
    clearance_values: list[float] = []
    if not accepted_edges_df.empty:
        for row in accepted_edges_df.to_dict(orient="records"):
            threshold_value = row.get("threshold_value")
            if threshold_value in (None, "") or pd.isna(threshold_value):
                continue
            threshold_float = float(threshold_value)
            incompatibility = float(row["acoustic_incompatibility"])
            if threshold_float <= 0.0:
                clearance = 1.0 if incompatibility <= 0.0 else 0.0
            else:
                clearance = 1.0 - (incompatibility / threshold_float)
            clearance_values.append(float(np.clip(clearance, 0.0, 1.0)))
    accepted_edge_count_with_real_acoustic_threshold = int(len(clearance_values))
    summary = {
        "dedup_model_id": model_row["dedup_model_id"],
        "batch_id": model_row["batch_id"],
        "T_overlap_short": float(model_row["T_overlap_short"]),
        "T_gps_m": float(model_row["T_gps_m"]),
        "relative_difference_setting": model_row["relative_difference_setting"],
        "relative_difference_threshold": model_row["relative_difference_threshold"],
        "group_acoustic_setting": model_row["group_acoustic_setting"],
        "group_acoustic_threshold": model_row["group_acoustic_threshold"],
        "N_raw_hums": int(len(hum_ids)),
        "N_candidate_edges": int(len(batch_candidates_df.index)),
        "N_accepted_edges": int(len(accepted_edges_df.index)),
        "N_provisional_groups": int(len(provisional_groups_df.index)),
        "N_provisional_groups_size_2": int((provisional_groups_df["group_size_class"] == "size_2").sum())
        if not provisional_groups_df.empty
        else 0,
        "N_provisional_groups_size_3plus": int((provisional_groups_df["group_size_class"] == "size_3plus").sum())
        if not provisional_groups_df.empty
        else 0,
        "N_final_events": int(len(sorted_clusters)),
        "N_hums_removed": removed_as_duplicates,
        "percent_removed": float(removed_as_duplicates / max(len(hum_ids), 1)),
        "N_duplicate_events": int(sum(1 for members in sorted_clusters if len(members) > 1)),
        "mean_cluster_size": float(np.mean(duplicate_cluster_sizes)) if duplicate_cluster_sizes else 1.0,
        "median_cluster_size": float(np.median(duplicate_cluster_sizes)) if duplicate_cluster_sizes else 1.0,
        "max_cluster_size": int(max(duplicate_cluster_sizes)) if duplicate_cluster_sizes else 1,
        "accepted_edge_count_size_2": int((accepted_edges_df["group_size_class"] == "size_2").sum())
        if not accepted_edges_df.empty
        else 0,
        "accepted_edge_count_size_3plus": int((accepted_edges_df["group_size_class"] == "size_3plus").sum())
        if not accepted_edges_df.empty
        else 0,
        "N_accepted_edges_with_real_acoustic_threshold": accepted_edge_count_with_real_acoustic_threshold,
        "duplicate_pair_count": int(len(duplicate_pair_keys)),
        "duplicate_pair_decision_keys": duplicate_pair_keys,
        "accepted_edge_acoustic_clearances": clearance_values,
        "reference_incompatibility_median": reference_incompatibility_median,
        "accepted_incompatibility_median": accepted_incompatibility_median,
        "representative_selection_rule": REPRESENTATIVE_SELECTION_RULE,
    }
    if not materialize_details:
        return {"summary": summary}

    event_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    event_lookup: dict[str, str] = {}
    for event_index, members in enumerate(sorted_clusters, start=1):
        cluster_df = hum_lookup.loc[members].reset_index().sort_values(
            ["start_time_epoch_s", "hum_id"],
            kind="stable",
        )
        event_id = f"{model_row['dedup_model_id']}__event_{event_index:05d}"
        primary = _representative_row(cluster_df)
        primary_hum_id = str(primary["hum_id"])
        for hum_id in cluster_df["hum_id"].astype(str).tolist():
            event_lookup[hum_id] = event_id

        cluster_edges_df = accepted_edges_df.loc[
            accepted_edges_df["hum_id_i"].isin(cluster_df["hum_id"])
            & accepted_edges_df["hum_id_j"].isin(cluster_df["hum_id"])
        ].copy()
        adjacency: dict[str, set[str]] = {str(hum_id): set() for hum_id in cluster_df["hum_id"].astype(str).tolist()}
        for pair in cluster_edges_df.itertuples(index=False):
            adjacency[str(pair.hum_id_i)].add(str(pair.hum_id_j))
            adjacency[str(pair.hum_id_j)].add(str(pair.hum_id_i))

        parent: dict[str, str | None] = {primary_hum_id: None}
        depth: dict[str, int] = {primary_hum_id: 0}
        queue: deque[str] = deque([primary_hum_id])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency.get(current, [])):
                if neighbor in depth:
                    continue
                parent[neighbor] = current
                depth[neighbor] = depth[current] + 1
                queue.append(neighbor)
        tree_edges: dict[tuple[str, str], str] = {}
        for hum_id, parent_hum_id in parent.items():
            if parent_hum_id is None:
                continue
            edge_key = tuple(sorted((hum_id, parent_hum_id)))
            tree_edges[edge_key] = "direct" if parent_hum_id == primary_hum_id else "indirect"

        member_table = cluster_df.copy()
        member_table["time_delta_to_primary_s"] = (
            member_table["start_time_epoch_s"].astype(float) - float(primary["start_time_epoch_s"])
        ).abs()
        member_table["gps_distance_to_primary_m"] = member_table.apply(
            lambda row: _haversine_m(
                float(primary["latitude"]),
                float(primary["longitude"]),
                float(row["latitude"]),
                float(row["longitude"]),
            )
            if row["hum_id"] != primary_hum_id and not (pd.isna(row["latitude"]) or pd.isna(row["longitude"]))
            else 0.0,
            axis=1,
        )
        member_table["is_primary"] = member_table["hum_id"].astype(str) == primary_hum_id
        member_table = member_table.sort_values(
            ["is_primary", "time_delta_to_primary_s", "gps_distance_to_primary_m", "snippet_snr_db", "hum_id"],
            ascending=[False, True, True, False, True],
            kind="stable",
        ).reset_index(drop=True)

        for member_rank, row in enumerate(member_table.to_dict(orient="records"), start=1):
            hum_id = str(row["hum_id"])
            parent_hum_id = parent.get(hum_id)
            member_rows.append(
                {
                    "dedup_event_id": event_id,
                    "dedup_model_id": model_row["dedup_model_id"],
                    "hum_id": hum_id,
                    "primary_hum_id": primary_hum_id,
                    "cluster_size": int(len(cluster_df.index)),
                    "is_primary": hum_id == primary_hum_id,
                    "cluster_member_rank": int(member_rank),
                    "path_depth_to_primary": int(depth.get(hum_id, 0)),
                    "connected_via_hum_id": parent_hum_id,
                    "path_to_primary_type": "primary"
                    if hum_id == primary_hum_id
                    else ("direct" if parent_hum_id == primary_hum_id else "indirect"),
                }
            )

        event_rows.append(
            {
                "dedup_event_id": event_id,
                "dedup_model_id": model_row["dedup_model_id"],
                "primary_hum_id": primary_hum_id,
                "primary_collar_id": primary["collar_id"],
                "primary_recording_id": primary["recording_id"],
                "primary_start_time_utc": primary["start_time_utc"],
                "primary_end_time_utc": primary["end_time_utc"],
                "primary_latitude": primary["latitude"],
                "primary_longitude": primary["longitude"],
                "primary_snr_db": primary["snippet_snr_db"],
                "primary_rms": primary["snippet_rms"],
                "primary_classifier_score": primary["model_positive_score"],
                "cluster_size": int(len(cluster_df.index)),
                "cluster_member_hum_ids_json": json.dumps(cluster_df["hum_id"].astype(str).tolist()),
                "duplicate_hum_ids_json": json.dumps(
                    [hum_id for hum_id in cluster_df["hum_id"].astype(str).tolist() if hum_id != primary_hum_id]
                ),
                "cluster_start_time_min_utc": cluster_df["start_time_utc"].min(),
                "cluster_start_time_max_utc": cluster_df["start_time_utc"].max(),
                "cluster_time_span_s": float(
                    cluster_df["end_time_epoch_s"].max() - cluster_df["start_time_epoch_s"].min()
                ),
                "accepted_edge_count": int(len(cluster_edges_df.index)),
                "representative_selection_rule": REPRESENTATIVE_SELECTION_RULE,
            }
        )

        for row in cluster_edges_df.to_dict(orient="records"):
            edge_key = tuple(sorted((str(row["hum_id_i"]), str(row["hum_id_j"]))))
            tree_type = tree_edges.get(edge_key)
            link_rows.append(
                {
                    "dedup_event_id": event_id,
                    "dedup_model_id": model_row["dedup_model_id"],
                    "primary_hum_id": primary_hum_id,
                    "hum_id_i": str(row["hum_id_i"]),
                    "hum_id_j": str(row["hum_id_j"]),
                    "collar_id_i": str(row["collar_id_i"]),
                    "collar_id_j": str(row["collar_id_j"]),
                    "overlap_s": float(row["overlap_s"]),
                    "overlap_fraction_short": float(row["overlap_fraction_short"]),
                    "gps_distance_m": float(row["gps_distance_m"]),
                    "provisional_group_id": str(row["provisional_group_id"]),
                    "provisional_group_size": int(row["provisional_group_size"]),
                    "provisional_group_size_class": str(row["group_size_class"]),
                    "acoustic_incompatibility": float(row["acoustic_incompatibility"]),
                    "threshold_type": str(row["threshold_type"]),
                    "threshold_setting": str(row["threshold_setting"]),
                    "threshold_value": float(row["threshold_value"]) if pd.notna(row["threshold_value"]) else None,
                    "is_component_tree_edge": edge_key in tree_edges,
                    "path_to_primary_type": tree_type if tree_type is not None else "redundant",
                }
            )

    return {
        "summary": summary,
        "events_df": (
            pd.DataFrame(
                event_rows,
                columns=[
                    "dedup_event_id",
                    "dedup_model_id",
                    "primary_hum_id",
                    "primary_collar_id",
                    "primary_recording_id",
                    "primary_start_time_utc",
                    "primary_end_time_utc",
                    "primary_latitude",
                    "primary_longitude",
                    "primary_snr_db",
                    "primary_rms",
                    "primary_classifier_score",
                    "cluster_size",
                    "cluster_member_hum_ids_json",
                    "duplicate_hum_ids_json",
                    "cluster_start_time_min_utc",
                    "cluster_start_time_max_utc",
                    "cluster_time_span_s",
                    "accepted_edge_count",
                    "representative_selection_rule",
                ],
            )
            .sort_values("dedup_event_id", kind="stable")
            .reset_index(drop=True)
        ),
        "members_df": (
            pd.DataFrame(
                member_rows,
                columns=[
                    "dedup_event_id",
                    "dedup_model_id",
                    "hum_id",
                    "primary_hum_id",
                    "cluster_size",
                    "is_primary",
                    "cluster_member_rank",
                    "path_depth_to_primary",
                    "connected_via_hum_id",
                    "path_to_primary_type",
                ],
            )
            .sort_values(["dedup_event_id", "cluster_member_rank", "hum_id"], kind="stable")
            .reset_index(drop=True)
        ),
        "links_df": (
            pd.DataFrame(
                link_rows,
                columns=[
                    "dedup_event_id",
                    "dedup_model_id",
                    "primary_hum_id",
                    "hum_id_i",
                    "hum_id_j",
                    "collar_id_i",
                    "collar_id_j",
                    "overlap_s",
                    "overlap_fraction_short",
                    "gps_distance_m",
                    "provisional_group_id",
                    "provisional_group_size",
                    "provisional_group_size_class",
                    "acoustic_incompatibility",
                    "threshold_type",
                    "threshold_setting",
                    "threshold_value",
                    "is_component_tree_edge",
                    "path_to_primary_type",
                ],
            )
            .sort_values(["dedup_event_id", "hum_id_i", "hum_id_j"], kind="stable")
            .reset_index(drop=True)
        ),
        "accepted_edges_df": accepted_edges_df.reset_index(drop=True),
        "event_lookup": event_lookup,
    }


def _model_summary_metadata(config: DedupConfig, model_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_key": stable_json_hash(
            {
                "algorithm_version": config.algorithm_version,
                "model_id": model_row["dedup_model_id"],
                "feature_config_hash": feature_config_hash(config),
                "dedup_config_hash": dedup_config_hash(config),
                "schema_version": CURRENT_SCHEMA_VERSION,
                "decision_method_version": DECISION_METHOD_VERSION,
            }
        )[:16],
        "algorithm_version": config.algorithm_version,
        "feature_config_hash": feature_config_hash(config),
        "dedup_config_hash": dedup_config_hash(config),
        "runtime_context": runtime_context(),
        "schema_version": CURRENT_SCHEMA_VERSION,
        "decision_method_version": DECISION_METHOD_VERSION,
    }


def _model_summary_is_valid(path: Path, expected_metadata: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    payload = _load_json(path)
    metadata = payload.get("_cache_metadata", {})
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            return False
    return True


def _collect_grid_completion(paths: LaunchPaths, config: DedupConfig) -> dict[str, int]:
    models_df = pd.read_parquet(paths.manifests_dir / "grid_models.parquet")
    batches_df = pd.read_parquet(paths.manifests_dir / "grid_batches.parquet")
    completed_model_ids: set[str] = set()
    for model_row in models_df.to_dict(orient="records"):
        summary_path = paths.cache_dir / "model_runs" / str(model_row["dedup_model_id"]) / "summary.json"
        if _model_summary_is_valid(summary_path, _model_summary_metadata(config, model_row)):
            completed_model_ids.add(str(model_row["dedup_model_id"]))
    completed_batches = 0
    for batch in batches_df.to_dict(orient="records"):
        model_ids = batch["model_ids"]
        if isinstance(model_ids, str):
            model_ids = json.loads(model_ids)
        if all(str(model_id) in completed_model_ids for model_id in model_ids):
            completed_batches += 1
    return {
        "completed_grid_batches": int(completed_batches),
        "completed_models": int(len(completed_model_ids)),
        "cached_models_reused": int(len(completed_model_ids)),
    }


def seed_model_artifacts(config: DedupConfig, *, source_launch_dir: Path, force: bool = False) -> LaunchPaths:
    paths = launch_paths_for_config(config)
    source_paths = LaunchPaths.create(source_launch_dir.resolve())
    models_df = pd.read_parquet(paths.manifests_dir / "grid_models.parquet")
    batches_df = pd.read_parquet(paths.manifests_dir / "grid_batches.parquet")
    target_model_runs_dir = paths.cache_dir / "model_runs"
    if not force and target_model_runs_dir.exists() and any(target_model_runs_dir.iterdir()):
        raise FileExistsError(f"Target launch already has model-run cache artifacts: {target_model_runs_dir}")

    missing_or_incompatible: list[str] = []
    source_summary_paths: list[tuple[Path, Path]] = []
    for model_row in models_df.to_dict(orient="records"):
        model_id = str(model_row["dedup_model_id"])
        source_summary_path = source_paths.cache_dir / "model_runs" / model_id / "summary.json"
        if not _model_summary_is_valid(source_summary_path, _model_summary_metadata(config, model_row)):
            missing_or_incompatible.append(model_id)
            continue
        source_summary_paths.append((source_summary_path.parent, target_model_runs_dir / model_id))

    if missing_or_incompatible:
        preview = ", ".join(missing_or_incompatible[:10])
        suffix = "" if len(missing_or_incompatible) <= 10 else f" ... (+{len(missing_or_incompatible) - 10} more)"
        raise FileNotFoundError(
            "Missing or incompatible reusable model-run caches for target launch: "
            f"{preview}{suffix}"
        )

    target_model_runs_dir.mkdir(parents=True, exist_ok=True)
    for source_model_dir, target_model_dir in source_summary_paths:
        shutil.copytree(source_model_dir, target_model_dir, dirs_exist_ok=True)

    total_grid_batches = int(len(batches_df.index))
    total_models = int(len(models_df.index))
    write_stage_progress(
        paths,
        "grid",
        {
            "stage_status": "completed",
            "stage_started_at": utc_now_iso(),
            "stage_finished_at": utc_now_iso(),
            "total_grid_batches": total_grid_batches,
            "completed_grid_batches": total_grid_batches,
            "failed_grid_batches": 0,
            "total_models": total_models,
            "completed_models": total_models,
            "failed_models": 0,
            "cached_models_reused": total_models,
            "seeded_from_launch_dir": str(source_paths.launch_dir),
        },
    )
    launch_progress = read_progress(paths.progress_dir / "launch_progress.json")
    write_launch_progress(
        paths,
        {
            **launch_progress,
            "launch_id": paths.launch_dir.name,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
            "current_stage": "grid",
            "stage_status": "completed",
            "total_grid_batches": total_grid_batches,
            "completed_grid_batches": total_grid_batches,
            "failed_grid_batches": 0,
            "total_models": total_models,
            "completed_models": total_models,
            "failed_models": 0,
            "cached_models_reused": total_models,
            "top_k_requested": int(config.top_k_models),
        },
    )
    return paths


def run_grid(config: DedupConfig, *, batch_ids: list[str] | None = None) -> LaunchPaths:
    paths = launch_paths_for_config(config)
    hum_features_df = pd.read_parquet(paths.tables_dir / "hum_features.parquet")
    candidate_df = build_candidate_pair_cache(config, paths)
    models_df = pd.read_parquet(paths.manifests_dir / "grid_models.parquet")
    batches_df = pd.read_parquet(paths.manifests_dir / "grid_batches.parquet")
    if batch_ids is not None:
        batches_df = batches_df.loc[batches_df["batch_id"].isin(batch_ids)].copy()

    all_batches_df = pd.read_parquet(paths.manifests_dir / "grid_batches.parquet")
    model_lookup = {str(row["dedup_model_id"]): row for row in models_df.to_dict(orient="records")}

    for batch in batches_df.to_dict(orient="records"):
        batch_candidate_df, provisional_groups_df = _build_batch_context(hum_features_df, candidate_df, batch)
        model_ids = batch["model_ids"]
        if isinstance(model_ids, str):
            model_ids = json.loads(model_ids)
        for model_id in model_ids:
            model_row = dict(model_lookup[str(model_id)])
            summary_path = paths.cache_dir / "model_runs" / str(model_id) / "summary.json"
            model_dir = summary_path.parent
            model_dir.mkdir(parents=True, exist_ok=True)
            metadata = _model_summary_metadata(config, model_row)
            lock_path = model_dir / ".lock"
            if _model_summary_is_valid(summary_path, metadata):
                continue
            with file_lock(lock_path):
                if _model_summary_is_valid(summary_path, metadata):
                    continue
                result = _evaluate_model_from_batch(
                    hum_features_df,
                    batch_candidate_df,
                    provisional_groups_df,
                    model_row,
                    materialize_details=False,
                )
                write_json(
                    summary_path,
                    {
                        **result["summary"],
                        "_cache_metadata": metadata,
                        "created_at_utc": utc_now_iso(),
                    },
                )
        progress_counts = _collect_grid_completion(paths, config)
        total_grid_batches = int(len(all_batches_df.index))
        total_models = int(len(models_df.index))
        write_stage_progress(
            paths,
            "grid",
            {
                "stage_status": "running" if progress_counts["completed_grid_batches"] < total_grid_batches else "completed",
                "stage_started_at": read_progress(paths.progress_dir / "grid_progress.json").get(
                    "stage_started_at",
                    utc_now_iso(),
                ),
                "stage_finished_at": utc_now_iso() if progress_counts["completed_grid_batches"] == total_grid_batches else None,
                "total_grid_batches": total_grid_batches,
                "completed_grid_batches": progress_counts["completed_grid_batches"],
                "failed_grid_batches": 0,
                "total_models": total_models,
                "completed_models": progress_counts["completed_models"],
                "failed_models": 0,
                "cached_models_reused": progress_counts["cached_models_reused"],
            },
        )

    launch_progress = read_progress(paths.progress_dir / "launch_progress.json")
    progress_counts = _collect_grid_completion(paths, config)
    write_launch_progress(
        paths,
        {
            **launch_progress,
            "launch_id": paths.launch_dir.name,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
            "current_stage": "grid",
            "stage_status": "completed" if progress_counts["completed_grid_batches"] == int(len(all_batches_df.index)) else "running",
            "total_grid_batches": int(len(all_batches_df.index)),
            "completed_grid_batches": progress_counts["completed_grid_batches"],
            "failed_grid_batches": 0,
            "total_models": int(len(models_df.index)),
            "completed_models": progress_counts["completed_models"],
            "failed_models": 0,
            "cached_models_reused": progress_counts["cached_models_reused"],
            "top_k_requested": int(config.top_k_models),
        },
    )
    return paths


def _write_ranking_report(paths: LaunchPaths, summary_df: pd.DataFrame, config: DedupConfig) -> None:
    acoustic_margin_label = _acoustic_margin_label(config)
    top_rows = summary_df.loc[summary_df["final_selectable"]].head(config.top_k_models).copy()
    table_columns = [
        "rank",
        "dedup_model_id",
        "robustness_score",
        "final_selectable",
        "ineligibility_reason",
        "local_decision_stability",
        "removal_count_stability",
        "decision_acoustic_core",
        "acoustic_strength",
        "acoustic_threshold_coverage",
        acoustic_margin_label,
        "N_final_events",
        "N_hums_removed",
        "N_accepted_edges",
    ]
    header = "| " + " | ".join(table_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table_columns)) + " |"
    table_lines = [header, separator]
    for record in top_rows.loc[:, table_columns].to_dict(orient="records"):
        table_lines.append(
            "| "
            + " | ".join(
                f"{record[column]:.6f}" if isinstance(record[column], float) else str(record[column])
                for column in table_columns
            )
            + " |"
        )

    lines = [
        f"# Model Ranking Report: {paths.launch_dir.name}",
        "",
        "## Robustness Score",
        f"`eligible = T_rel != none AND T_acoustic != none AND acoustic_threshold_coverage >= {config.robustness.min_final_selectable_acoustic_coverage:.2f}`.",
        f"`acoustic_strength = acoustic_threshold_coverage * min({acoustic_margin_label} / {config.robustness.target_acoustic_margin:.2f}, 1.0)`.",
        "`decision_acoustic_core = 2 * local_decision_stability * acoustic_strength / (local_decision_stability + acoustic_strength)` with `0` when the denominator is `0`.",
        f"`robustness_score = {config.robustness.decision_acoustic_core_weight:.2f} * decision_acoustic_core + {config.robustness.removal_count_stability_weight:.2f} * removal_count_stability` for final-selectable models only.",
        "",
        "## Local Duplicate-Decision Stability",
        "`median(Dice overlap(model duplicate-pair set, neighbor duplicate-pair set))` across one-step neighbors.",
        "",
        "## Removal-Count Stability",
        "`1 - abs(N_hums_removed(model) - median(N_hums_removed(neighbors))) / max(N_hums_removed(model), median_neighbor_removed, 1)` clipped to `[0, 1]`.",
        "",
        "## Acoustic Threshold Margin And Coverage",
        f"`q{int(round(config.robustness.acoustic_margin_quantile * 100)):02d}(1 - acoustic_incompatibility / edge_threshold)` over accepted edges with real acoustic thresholds, clipped to `[0, 1]`.",
        f"`acoustic_threshold_coverage = N_accepted_edges_with_real_acoustic_threshold / max(N_accepted_edges, 1)`.",
        f"`{acoustic_margin_label} = 0` when a model has no accepted edge with a real acoustic threshold.",
        "",
        "## Notes",
        "- Neighbors differ by one step in exactly one of the four grid axes.",
        "- Duplicate-pair sets are derived from final connected components, not just accepted direct edges.",
        "- Models stay in the diagnostic summary even when they are not final-selectable.",
        "- Edges accepted under `none` acoustic settings count toward accepted edges but not toward acoustic-threshold coverage.",
        "",
        "## Top Final-Selectable Models",
        *table_lines,
        "",
    ]
    (paths.reports_dir / "model_ranking_report.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_launch_readme(paths: LaunchPaths, config: DedupConfig) -> None:
    acoustic_margin_label = _acoustic_margin_label(config)
    lines = [
        f"# Launch README: {paths.launch_dir.name}",
        "",
        "## What This Launch Computes",
        "- Canonical merged hum features from the cleaned full-corpus artifact.",
        "- A 500-model dedup grid spanning overlap-fraction-short, GPS, relative-difference, and group-acoustic thresholds.",
        "- Numeric model summaries and rankings for every model.",
        "- Full final outputs only for the selected best model.",
        "- Qualitative exemplar bundles only for the top-ranked final-selectable models.",
        "",
        "## Feature Extraction",
        "- Worker stage writes shard caches only.",
        "- `finalize-features` validates and merges shard caches into `tables/hum_features.parquet`.",
        "- Acoustic vector columns are the seven interpretable revamp features.",
        "",
        "## Grid Generation",
        "- Models are batched by `(T_overlap_short, T_gps)`.",
        "- Each batch evaluates 20 models across the `T_rel` and `T_acoustic` axes.",
        "- Candidate pairs are built once at the minimum overlap threshold and maximum GPS threshold, then reused.",
        "",
        "## Ranking",
        "- Local duplicate-decision stability is the median Dice overlap against one-step neighboring models.",
        "- Removal-count stability compares `N_hums_removed` against the median neighboring removal count.",
        f"- Acoustic margin uses the q{int(round(config.robustness.acoustic_margin_quantile * 100)):02d} of accepted-edge threshold clearance for real acoustic thresholds.",
        f"- A model is final-selectable only when `T_rel` and `T_acoustic` are both real thresholds and acoustic-threshold coverage is at least `{config.robustness.min_final_selectable_acoustic_coverage:.2f}`.",
        f"- `acoustic_strength = acoustic_threshold_coverage * min({acoustic_margin_label} / {config.robustness.target_acoustic_margin:.2f}, 1.0)`.",
        "- `decision_acoustic_core` is the harmonic mean of local decision stability and acoustic strength.",
        f"- `robustness_score = {config.robustness.decision_acoustic_core_weight:.2f} * decision_acoustic_core + {config.robustness.removal_count_stability_weight:.2f} * removal_count_stability` for final-selectable models only.",
        "",
        "## HPC Flow",
        "- Use `prepare-hpc` to materialize `hpc_bundle/<stage>/` inside this launch root.",
        "- Submit stages in order: features -> feature_merge -> grid -> aggregate -> exemplars.",
        "- Shell wrappers should only sync, submit, monitor, or pull prepared artifacts.",
        "",
        "## Caching",
        "- Feature shard cache",
        "- Merged feature table cache",
        "- Superset candidate-pair cache",
        "- Per-model numeric result cache",
        "- Exemplar export cache",
        "",
        "## Outputs",
        "- Final selected-model tables live in `tables/`.",
        "- Ranking report lives in `reports/model_ranking_report.md`.",
        "- Launch documentation lives in `readme/`.",
        "- Pulled remote artifacts should stay under the same canonical launch root locally.",
        "",
        "## Reproduction",
        f"- Config: `{paths.configs_dir / 'launch_config.yaml'}`",
        "- Local: `prepare-features -> run-features -> finalize-features -> prepare-grid -> run-grid -> summarize-run -> export-top-exemplars`",
        "- HPC: `prepare-hpc`, sync the launch root, then submit stage bundles with dependencies.",
        "",
        f"- Top models requested: `{config.top_k_models}`",
        f"- Exemplars per model: `{config.exemplars_per_model}`",
        "",
    ]
    (paths.readme_dir / "launch_README.md").write_text("\n".join(lines), encoding="utf8")


def _evaluate_model_details(config: DedupConfig, paths: LaunchPaths, model_id: str) -> dict[str, Any]:
    hum_features_df = pd.read_parquet(paths.tables_dir / "hum_features.parquet")
    candidate_df = build_candidate_pair_cache(config, paths)
    models_df = pd.read_parquet(paths.manifests_dir / "grid_models.parquet")
    model_row = models_df.loc[models_df["dedup_model_id"] == model_id].iloc[0].to_dict()
    batch_row = pd.read_parquet(paths.manifests_dir / "grid_batches.parquet").loc[
        lambda df: df["batch_id"] == model_row["batch_id"]
    ].iloc[0].to_dict()
    batch_candidate_df, provisional_groups_df = _build_batch_context(hum_features_df, candidate_df, batch_row)
    return _evaluate_model_from_batch(
        hum_features_df,
        batch_candidate_df,
        provisional_groups_df,
        model_row,
        materialize_details=True,
    )


def materialize_final_outputs(config: DedupConfig, paths: LaunchPaths, best_model_id: str) -> None:
    result = _evaluate_model_details(config, paths, best_model_id)
    events_df = result["events_df"]
    members_df = result["members_df"]
    links_df = result["links_df"]

    events_df.to_parquet(paths.tables_dir / "deduplicated_hums.parquet", index=False)
    events_df.to_csv(paths.tables_dir / "deduplicated_hums.csv", index=False)
    members_df.to_parquet(paths.tables_dir / "dedup_event_members.parquet", index=False)
    members_df.to_csv(paths.tables_dir / "dedup_event_members.csv", index=False)
    links_df.to_parquet(paths.tables_dir / "dedup_links.parquet", index=False)
    links_df.to_csv(paths.tables_dir / "dedup_links.csv", index=False)
    write_json(
        paths.tables_dir / "final_model_selection.json",
        {
            "best_model_id": best_model_id,
            "selection_status": "selected",
            "selected_at_utc": utc_now_iso(),
            "schema_version": CURRENT_SCHEMA_VERSION,
            "decision_method_version": DECISION_METHOD_VERSION,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
        },
    )


def _clear_final_output_tables(paths: LaunchPaths) -> None:
    for filename in [
        "deduplicated_hums.parquet",
        "deduplicated_hums.csv",
        "dedup_event_members.parquet",
        "dedup_event_members.csv",
        "dedup_links.parquet",
        "dedup_links.csv",
        "final_model_selection.json",
    ]:
        path = paths.tables_dir / filename
        if path.exists():
            path.unlink()


def summarize_run(config: DedupConfig) -> LaunchPaths:
    paths = launch_paths_for_config(config)
    acoustic_margin_label = _acoustic_margin_label(config)
    models_df = pd.read_parquet(paths.manifests_dir / "grid_models.parquet")
    adjacency_df = pd.read_parquet(paths.manifests_dir / "model_adjacency.parquet")

    summary_rows: list[dict[str, Any]] = []
    for model_row in models_df.to_dict(orient="records"):
        summary_path = paths.cache_dir / "model_runs" / str(model_row["dedup_model_id"]) / "summary.json"
        if not _model_summary_is_valid(summary_path, _model_summary_metadata(config, model_row)):
            raise FileNotFoundError(f"Missing or stale model summary: {summary_path}")
        payload = _load_json(summary_path)
        payload.pop("_cache_metadata", None)
        summary_rows.append(payload)

    summary_df = pd.DataFrame(summary_rows).sort_values("dedup_model_id", kind="stable").reset_index(drop=True)
    removed_counts = summary_df.set_index("dedup_model_id")["N_hums_removed"].to_dict()
    duplicate_pair_sets = {
        str(row["dedup_model_id"]): set(str(pair_key) for pair_key in row.get("duplicate_pair_decision_keys", []))
        for row in summary_df.to_dict(orient="records")
    }

    local_decision_stability: list[float] = []
    removal_count_stability: list[float] = []
    acoustic_margin_values: list[float] = []
    acoustic_threshold_coverage_values: list[float] = []
    acoustic_strength_values: list[float] = []
    decision_acoustic_core_values: list[float] = []
    final_selectable_values: list[bool] = []
    ineligibility_reasons: list[str | None] = []
    for row in summary_df.to_dict(orient="records"):
        model_id = row["dedup_model_id"]
        neighbors = adjacency_df.loc[adjacency_df["dedup_model_id"] == model_id, "neighbor_model_id"].tolist()
        neighbor_sets = [duplicate_pair_sets[neighbor_id] for neighbor_id in neighbors if neighbor_id in duplicate_pair_sets]
        if neighbor_sets:
            local_decision_stability.append(
                float(
                    np.median(
                        [_dice_overlap(duplicate_pair_sets[str(model_id)], neighbor_set) for neighbor_set in neighbor_sets]
                    )
                )
            )
        else:
            local_decision_stability.append(0.0)

        neighbor_removed_counts = [float(removed_counts[neighbor_id]) for neighbor_id in neighbors if neighbor_id in removed_counts]
        if neighbor_removed_counts:
            current_removed = float(row["N_hums_removed"])
            median_neighbor_removed = float(np.median(neighbor_removed_counts))
            denominator = max(current_removed, median_neighbor_removed, 1.0)
            stability = 1.0 - abs(current_removed - median_neighbor_removed) / denominator
            removal_count_stability.append(float(np.clip(stability, 0.0, 1.0)))
        else:
            removal_count_stability.append(0.0)

        edge_clearances = [float(value) for value in (row.get("accepted_edge_acoustic_clearances") or [])]
        accepted_edges_with_real_threshold = row.get("N_accepted_edges_with_real_acoustic_threshold")
        if accepted_edges_with_real_threshold is None:
            accepted_edges_with_real_threshold = len(edge_clearances)
        coverage = float(accepted_edges_with_real_threshold) / max(float(row.get("N_accepted_edges", 0)), 1.0)
        coverage = float(np.clip(coverage, 0.0, 1.0))
        acoustic_threshold_coverage_values.append(coverage)
        if edge_clearances:
            acoustic_margin = float(np.quantile(edge_clearances, config.robustness.acoustic_margin_quantile))
        else:
            acoustic_margin = 0.0
        acoustic_margin = float(np.clip(acoustic_margin, 0.0, 1.0))
        acoustic_margin_values.append(acoustic_margin)
        acoustic_strength = coverage * min(acoustic_margin / config.robustness.target_acoustic_margin, 1.0)
        acoustic_strength_values.append(float(np.clip(acoustic_strength, 0.0, 1.0)))
        denominator = local_decision_stability[-1] + acoustic_strength_values[-1]
        if denominator <= 0.0:
            decision_acoustic_core_values.append(0.0)
        else:
            decision_acoustic_core_values.append(
                float((2.0 * local_decision_stability[-1] * acoustic_strength_values[-1]) / denominator)
            )

        reasons: list[str] = []
        relative_difference_threshold = row.get("relative_difference_threshold")
        if relative_difference_threshold is None or pd.isna(relative_difference_threshold):
            reasons.append("relative_difference_threshold_none")
        group_acoustic_threshold = row.get("group_acoustic_threshold")
        if group_acoustic_threshold is None or pd.isna(group_acoustic_threshold):
            reasons.append("group_acoustic_threshold_none")
        if coverage < config.robustness.min_final_selectable_acoustic_coverage:
            reasons.append(
                "acoustic_threshold_coverage_below_"
                f"{int(round(config.robustness.min_final_selectable_acoustic_coverage * 100)):02d}"
            )
        final_selectable = len(reasons) == 0
        final_selectable_values.append(final_selectable)
        ineligibility_reasons.append(None if final_selectable else ";".join(reasons))

    summary_df["local_decision_stability"] = local_decision_stability
    summary_df["removal_count_stability"] = removal_count_stability
    summary_df[acoustic_margin_label] = acoustic_margin_values
    summary_df["acoustic_threshold_coverage"] = acoustic_threshold_coverage_values
    summary_df["acoustic_strength"] = acoustic_strength_values
    summary_df["decision_acoustic_core"] = decision_acoustic_core_values
    summary_df["final_selectable"] = final_selectable_values
    summary_df["ineligibility_reason"] = ineligibility_reasons
    summary_df["robustness_method_version"] = ROBUSTNESS_METHOD_VERSION
    summary_df["robustness_score"] = np.where(
        summary_df["final_selectable"],
        (config.robustness.decision_acoustic_core_weight * summary_df["decision_acoustic_core"])
        + (config.robustness.removal_count_stability_weight * summary_df["removal_count_stability"]),
        np.nan,
    )
    summary_df = summary_df.sort_values(
        ["final_selectable", "robustness_score", "dedup_model_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    summary_df["rank"] = range(1, len(summary_df.index) + 1)

    export_df = summary_df.drop(
        columns=["duplicate_pair_decision_keys", "accepted_edge_acoustic_clearances"],
        errors="ignore",
    ).copy()

    export_df.to_parquet(paths.tables_dir / "model_sensitivity_summary.parquet", index=False)
    export_df.to_csv(paths.tables_dir / "model_sensitivity_summary.csv", index=False)
    top_selectable_df = export_df.loc[export_df["final_selectable"]].head(config.top_k_models).copy()
    top_selectable_df.to_csv(paths.tables_dir / "top_model_rankings.csv", index=False)
    _write_ranking_report(paths, export_df, config)
    _write_launch_readme(paths, config)

    selectable_summary_df = summary_df.loc[summary_df["final_selectable"]].copy()
    best_model_id: str | None = None
    if selectable_summary_df.empty:
        _clear_final_output_tables(paths)
        write_json(
            paths.tables_dir / "final_model_selection.json",
            {
                "best_model_id": None,
                "selection_status": "no_final_selectable_model",
                "reason": "No model satisfied the overlap-temporal v4 final-selectable criteria.",
                "selected_at_utc": utc_now_iso(),
                "schema_version": CURRENT_SCHEMA_VERSION,
                "decision_method_version": DECISION_METHOD_VERSION,
                "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
            },
        )
    else:
        best_model_id = str(selectable_summary_df.iloc[0]["dedup_model_id"])
        materialize_final_outputs(config, paths, best_model_id)

    write_stage_progress(
        paths,
        "aggregate",
        {
            "stage_status": "completed",
            "stage_started_at": read_progress(paths.progress_dir / "aggregate_progress.json").get(
                "stage_started_at",
                utc_now_iso(),
            ),
            "stage_finished_at": utc_now_iso(),
            "best_model_id": best_model_id,
            "top_k_requested": int(config.top_k_models),
        },
    )
    launch_progress = read_progress(paths.progress_dir / "launch_progress.json")
    progress_counts = _collect_grid_completion(paths, config)
    write_launch_progress(
        paths,
        {
            **launch_progress,
            "launch_id": paths.launch_dir.name,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
            "current_stage": "aggregate",
            "stage_status": "completed",
            "total_grid_batches": int(len(pd.read_parquet(paths.manifests_dir / "grid_batches.parquet").index)),
            "completed_grid_batches": progress_counts["completed_grid_batches"],
            "failed_grid_batches": 0,
            "total_models": int(len(models_df.index)),
            "completed_models": progress_counts["completed_models"],
            "failed_models": 0,
            "cached_models_reused": progress_counts["cached_models_reused"],
            "top_k_requested": int(config.top_k_models),
        },
    )
    return paths


def _cluster_threshold_boundary_score(
    cluster_edges_df: pd.DataFrame,
    model_row: dict[str, Any],
) -> float:
    scores: list[float] = []
    overlap_threshold = max(float(model_row["T_overlap_short"]), 1e-9)
    gps_threshold = max(float(model_row["T_gps_m"]), 1e-9)
    for row in cluster_edges_df.to_dict(orient="records"):
        overlap_fraction = float(row["overlap_fraction_short"])
        scores.append(
            float(np.clip(overlap_threshold / max(overlap_fraction, overlap_threshold), 0.0, 1.0))
        )
        scores.append(float(row["gps_distance_m"]) / gps_threshold)
        threshold_value = row.get("threshold_value")
        if threshold_value not in (None, "") and pd.notna(threshold_value) and float(threshold_value) > 0:
            scores.append(float(row["acoustic_incompatibility"]) / float(threshold_value))
    return float(max(scores)) if scores else 0.0


def _spread_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    raw = np.linspace(0, length - 1, num=count)
    indices: list[int] = []
    for value in raw:
        index = int(round(float(value)))
        if index not in indices:
            indices.append(index)
    while len(indices) < count:
        for candidate in range(length):
            if candidate not in indices:
                indices.append(candidate)
            if len(indices) == count:
                break
    return indices[:count]


def export_top_exemplars(config: DedupConfig, *, top_k: int | None = None) -> LaunchPaths:
    paths = launch_paths_for_config(config)
    summary_df = pd.read_parquet(paths.tables_dir / "model_sensitivity_summary.parquet")
    hum_features_df = pd.read_parquet(paths.tables_dir / "hum_features.parquet")
    if top_k is None:
        top_k = config.top_k_models
    selected_models = (
        summary_df.loc[summary_df["final_selectable"].fillna(False)]
        .head(top_k)["dedup_model_id"]
        .astype(str)
        .tolist()
    )
    hum_lookup = hum_features_df.set_index("hum_id")

    for root_dir in [paths.exemplar_concats_dir, paths.exemplar_manifests_dir, paths.cache_dir / "exemplars"]:
        if not root_dir.exists():
            continue
        for child in root_dir.iterdir():
            if child.name not in selected_models:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    for model_id in selected_models:
        cache_metadata = {
            "cache_key": stable_json_hash(
                {
                    "model_id": model_id,
                    "exemplars_per_model": int(config.exemplars_per_model),
                    "top_k": int(top_k),
                    "feature_config_hash": feature_config_hash(config),
                    "dedup_config_hash": dedup_config_hash(config),
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "decision_method_version": DECISION_METHOD_VERSION,
                    "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
                }
            )[:16],
            "feature_config_hash": feature_config_hash(config),
            "dedup_config_hash": dedup_config_hash(config),
            "schema_version": CURRENT_SCHEMA_VERSION,
            "decision_method_version": DECISION_METHOD_VERSION,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
        }
        cache_dir = paths.cache_dir / "exemplars" / model_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_manifest_path = cache_dir / "manifest.json"
        if _metadata_matches(cache_manifest_path, cache_metadata):
            continue

        result = _evaluate_model_details(config, paths, model_id)
        model_row = pd.read_parquet(paths.manifests_dir / "grid_models.parquet").loc[
            lambda df: df["dedup_model_id"] == model_id
        ].iloc[0].to_dict()
        events_df = result["events_df"].copy()
        links_df = result["links_df"].copy()
        members_df = result["members_df"].copy()
        duplicate_events_df = events_df.loc[events_df["cluster_size"] > 1].copy()

        boundary_rows: list[dict[str, Any]] = []
        for event_id in duplicate_events_df["dedup_event_id"].astype(str).tolist():
            cluster_edges_df = links_df.loc[links_df["dedup_event_id"] == event_id].copy()
            boundary_rows.append(
                {
                    "dedup_event_id": event_id,
                    "boundary_score": _cluster_threshold_boundary_score(cluster_edges_df, model_row),
                }
            )
        boundary_df = pd.DataFrame(boundary_rows)
        if not boundary_df.empty:
            duplicate_events_df = duplicate_events_df.merge(boundary_df, on="dedup_event_id", how="left")
        else:
            duplicate_events_df["boundary_score"] = 0.0

        boundary_selected = (
            duplicate_events_df.sort_values(
                ["boundary_score", "cluster_size", "cluster_start_time_min_utc", "dedup_event_id"],
                ascending=[False, False, True, True],
                kind="stable",
            )
            .head(min(5, int(config.exemplars_per_model)))
            .copy()
        )
        seen_event_ids = set(boundary_selected["dedup_event_id"].astype(str).tolist())
        remaining_slots = max(int(config.exemplars_per_model) - len(seen_event_ids), 0)
        spread_selected = pd.DataFrame(columns=duplicate_events_df.columns)
        if remaining_slots > 0:
            remaining_df = duplicate_events_df.loc[
                ~duplicate_events_df["dedup_event_id"].astype(str).isin(seen_event_ids)
            ].copy()
            remaining_df = remaining_df.sort_values(
                ["cluster_size", "cluster_start_time_min_utc", "dedup_event_id"],
                ascending=[True, True, True],
                kind="stable",
            ).reset_index(drop=True)
            spread_indices = _spread_indices(len(remaining_df.index), min(5, remaining_slots))
            if spread_indices:
                spread_selected = remaining_df.iloc[spread_indices].copy()

        selected_events_df = pd.concat([boundary_selected, spread_selected], ignore_index=True)
        selected_events_df = selected_events_df.drop_duplicates("dedup_event_id").head(int(config.exemplars_per_model))

        model_audio_dir = paths.exemplar_concats_dir / model_id
        model_manifest_dir = paths.exemplar_manifests_dir / model_id
        if model_audio_dir.exists():
            shutil.rmtree(model_audio_dir)
        if model_manifest_dir.exists():
            shutil.rmtree(model_manifest_dir)
        model_audio_dir.mkdir(parents=True, exist_ok=True)
        model_manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, Any]] = []

        for exemplar_index, event_row in enumerate(selected_events_df.to_dict(orient="records"), start=1):
            event_id = str(event_row["dedup_event_id"])
            event_members_df = members_df.loc[members_df["dedup_event_id"] == event_id].copy()
            event_members_df = event_members_df.merge(
                hum_features_df,
                on="hum_id",
                how="left",
                validate="one_to_one",
            )
            primary_hum_id = str(event_row["primary_hum_id"])
            primary = hum_lookup.loc[primary_hum_id]
            event_members_df["time_delta_to_primary_s"] = (
                event_members_df["start_time_epoch_s"].astype(float) - float(primary["start_time_epoch_s"])
            ).abs()
            event_members_df["gps_distance_to_primary_m"] = event_members_df.apply(
                lambda row: _haversine_m(
                    float(primary["latitude"]),
                    float(primary["longitude"]),
                    float(row["latitude"]),
                    float(row["longitude"]),
                )
                if row["hum_id"] != primary_hum_id and not (pd.isna(row["latitude"]) or pd.isna(row["longitude"]))
                else 0.0,
                axis=1,
            )
            event_members_df = event_members_df.sort_values(
                ["is_primary", "time_delta_to_primary_s", "gps_distance_to_primary_m", "snippet_snr_db", "hum_id"],
                ascending=[False, True, True, False, True],
                kind="stable",
            ).reset_index(drop=True)

            concat_segments = []
            selection_rows = []
            current_time_s = 0.0
            output_sample_rate = None
            for selection_index, row in enumerate(event_members_df.to_dict(orient="records"), start=1):
                source_path = apply_path_rewrites(str(row["feature_source_audio_file"]), config.path_rewrites)
                sample_rate, samples = load_wav(source_path, mmap=True)
                snippet = extract_snippet(samples, sample_rate, float(row["source_start_s"]), float(row["source_end_s"]))
                if output_sample_rate is None:
                    output_sample_rate = sample_rate
                if sample_rate != output_sample_rate:
                    raise ValueError(f"Mismatched sample rate in exemplar cluster {event_id}: {sample_rate} != {output_sample_rate}")
                concat_segments.append(snippet)
                begin_time_s = current_time_s
                end_time_s = current_time_s + (len(snippet) / sample_rate)
                selection_rows.append(
                    {
                        "Selection": selection_index,
                        "View": 1,
                        "Channel": 1,
                        "Begin Time (s)": begin_time_s,
                        "End Time (s)": end_time_s,
                        "Low Freq (Hz)": 0.0,
                        "High Freq (Hz)": sample_rate / 2.0,
                        "review_label": event_id,
                        "hum_id": row["hum_id"],
                        "is_primary": bool(row["is_primary"]),
                        "source_audio_file": row["feature_source_audio_file"],
                        "source_start_s": row["source_start_s"],
                        "source_end_s": row["source_end_s"],
                        "snippet_snr_db": row["snippet_snr_db"],
                        "gps_distance_to_primary_m": row["gps_distance_to_primary_m"],
                    }
                )
                current_time_s = end_time_s
                if selection_index < len(event_members_df.index):
                    gap = np.zeros(int(sample_rate * 0.1), dtype=np.float32)
                    concat_segments.append(gap)
                    current_time_s += len(gap) / sample_rate
            output_samples = np.concatenate(concat_segments) if concat_segments else np.array([], dtype=np.float32)
            wav_path = model_audio_dir / f"{exemplar_index:02d}_{event_id}.wav"
            wavfile.write(wav_path, output_sample_rate or 48_000, np.asarray(output_samples * 32767.0, dtype=np.int16))
            selection_table_path = model_manifest_dir / f"{exemplar_index:02d}_{event_id}_selection_table.tsv"
            pd.DataFrame(selection_rows).to_csv(selection_table_path, sep="\t", index=False)
            manifest_row = {
                "dedup_model_id": model_id,
                "dedup_event_id": event_id,
                "primary_hum_id": primary_hum_id,
                "cluster_size": int(event_row["cluster_size"]),
                "boundary_score": float(event_row.get("boundary_score", 0.0)),
                "wav_path": str(wav_path.resolve()),
                "selection_table_path": str(selection_table_path.resolve()),
            }
            manifest_rows.append(manifest_row)
            write_json(model_manifest_dir / f"{exemplar_index:02d}_{event_id}.json", manifest_row)

        pd.DataFrame(manifest_rows).to_csv(model_manifest_dir / "cluster_manifest.csv", index=False)
        write_json(
            cache_manifest_path,
            {
                **cache_metadata,
                "creation_timestamp_utc": utc_now_iso(),
                "exported_cluster_count": int(len(manifest_rows)),
            },
        )

    write_stage_progress(
        paths,
        "exemplars",
        {
            "stage_status": "completed",
            "stage_started_at": read_progress(paths.progress_dir / "exemplars_progress.json").get(
                "stage_started_at",
                utc_now_iso(),
            ),
            "stage_finished_at": utc_now_iso(),
            "top_k_requested": int(top_k),
            "top_k_exported": int(len(selected_models)),
        },
    )
    launch_progress = read_progress(paths.progress_dir / "launch_progress.json")
    write_launch_progress(
        paths,
        {
            **launch_progress,
            "launch_id": paths.launch_dir.name,
            "robustness_method_version": ROBUSTNESS_METHOD_VERSION,
            "current_stage": "exemplars",
            "stage_status": "completed",
            "top_k_requested": int(top_k),
            "top_k_exported": int(len(selected_models)),
        },
    )
    return paths

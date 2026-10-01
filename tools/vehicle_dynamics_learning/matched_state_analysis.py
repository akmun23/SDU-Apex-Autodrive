#!/usr/bin/env python3
"""Cross-run matched-state response dispersion with increasing history."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET = (
    REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "plant_teacher_race_domain_v1/cooldown_2s"
)
HISTORY_MS = (0, 100, 250, 500, 1000, 2000)
MATCH_RMS_RADII = (0.35, 0.50, 0.75)
MIN_NEIGHBORS = 9
MIN_OTHER_RUNS = 3
NEIGHBOR_CANDIDATES = 128
MAX_NEIGHBORS_PER_RUN = 4
MAX_NEIGHBOR_RUNS = 8
MAX_QUERIES_PER_RUN = 160
BOOTSTRAP_REPEATS = 1000
DT_S = 0.025


def _current_features(data: dict[str, np.ndarray]) -> np.ndarray:
    f = data["frames"].astype(np.float64, copy=False)
    r = data["simulator_rigid_state"].astype(np.float64, copy=False)
    return np.column_stack((
        r[:, 7], r[:, 8], r[:, 12], f[:, 3], f[:, 4], f[:, 5], f[:, 6],
        f[:, 7], f[:, 8],
    ))


def _history_features(current: np.ndarray, sample_indices: np.ndarray,
                      horizon_steps: int) -> np.ndarray:
    """Current state + lagged state + four causal summaries over the window."""
    if horizon_steps <= 0:
        return current[sample_indices]
    rows = np.empty((len(sample_indices), current.shape[1] * 6),
                    dtype=np.float64)
    for row_index, sample_index in enumerate(sample_indices):
        start = int(sample_index) - horizon_steps
        end = int(sample_index)
        if start < 0:
            raise ValueError("history window starts before the sequence")
        rows[row_index, :9] = current[sample_index]
        rows[row_index, 9:18] = current[start]
        for band in range(4):
            left = start + (horizon_steps * band) // 4
            right = start + (horizon_steps * (band + 1)) // 4
            right = max(left + 1, right)
            rows[row_index, 18 + band * 9:27 + band * 9] = np.mean(
                current[left:right], axis=0)
    return rows


def _balanced_query_indices(data: dict[str, np.ndarray],
                            current: np.ndarray, seed: int
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    bounds = data["sequence_bounds"]
    sequence_run = data["sequence_run_index"]
    run_splits = data["run_splits"].astype(str)
    frames = data["frames"]
    rigid = data["simulator_rigid_state"]
    acceleration = data["simulator_linear_acceleration"]
    candidates: dict[int, list[np.ndarray]] = defaultdict(list)
    for start_value, end_value, run_value in zip(
            bounds[:, 0], bounds[:, 1], sequence_run):
        start, end, run = int(start_value), int(end_value), int(run_value)
        # Keep blind whole-run splits out of both the query set and its
        # cross-run neighbour pool. This report informs model selection.
        if run_splits[run] not in {"train", "validation"}:
            continue
        first = start + int(2000 / 1000 / DT_S)
        last = end - 1
        if last <= first:
            continue
        valid = (np.isfinite(current[first:last]).all(axis=1)
                 & np.isfinite(rigid[first:last]).all(axis=1)
                 & np.isfinite(acceleration[first:last]).all(axis=1)
                 & np.isfinite(frames[first:last]).all(axis=1))
        local = np.flatnonzero(valid) + first
        if len(local):
            candidates[run].append(local)
    selected: list[np.ndarray] = []
    for run, blocks in sorted(candidates.items()):
        rows = np.concatenate(blocks)
        if len(rows) > MAX_QUERIES_PER_RUN:
            rows = np.sort(rng.choice(rows, MAX_QUERIES_PER_RUN, replace=False))
        selected.append(rows)
    if not selected:
        raise ValueError("no samples have complete 2-second history and labels")
    sample_indices = np.concatenate(selected)
    sample_run = data["frame_run_index"][sample_indices].astype(np.int32)
    sample_speed = np.hypot(current[sample_indices, 0], current[sample_indices, 1])
    return sample_indices, sample_run, sample_speed


def _scale_features(features: np.ndarray, train_mask: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = features[train_mask]
    if not len(train):
        raise ValueError("balanced query set contains no training runs")
    median = np.median(train, axis=0)
    q25, q75 = np.quantile(train, (0.25, 0.75), axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
    return (features - median) / scale, median, scale


def _cluster_ci(values: np.ndarray, run_index: np.ndarray, repeats: int,
                rng: np.random.Generator) -> tuple[float | None, float | None,
                                                   int]:
    valid = np.isfinite(values)
    if not np.any(valid):
        return None, None, 0
    runs = np.unique(run_index[valid])
    run_values = np.asarray([
        np.median(values[valid & (run_index == run)]) for run in runs
    ], dtype=np.float64)
    if len(run_values) < 2:
        return None, None, int(len(runs))
    draw = rng.integers(0, len(run_values), size=(repeats, len(run_values)))
    bootstrap = np.median(run_values[draw], axis=1)
    return (float(np.quantile(bootstrap, .025)),
            float(np.quantile(bootstrap, .975)), int(len(runs)))


def _nearest_response(features: np.ndarray, sample_indices: np.ndarray,
                      sample_runs: np.ndarray, target: np.ndarray,
                      radii: tuple[float, ...], seed: int
                      ) -> dict[float, dict[str, np.ndarray]]:
    del seed  # tree construction and query are deterministic for a fixed input
    tree = cKDTree(features)
    k = min(NEIGHBOR_CANDIDATES, len(sample_indices))
    distances, neighbours = tree.query(features, k=k, workers=-1)
    if k == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]
    rms_distance = distances / np.sqrt(features.shape[1])
    result: dict[float, dict[str, np.ndarray]] = {}
    dims = target.shape[1]
    for radius in radii:
        local_std = np.full((len(sample_indices), dims), np.nan)
        local_count = np.zeros(len(sample_indices), dtype=np.int16)
        local_run_count = np.zeros(len(sample_indices), dtype=np.int16)
        nearest_distance = np.full(len(sample_indices), np.nan)
        for row in range(len(sample_indices)):
            candidates = neighbours[row]
            chosen_list: list[int] = []
            chosen_distances: list[float] = []
            counts_by_run: dict[int, int] = {}
            selected_runs: set[int] = set()
            for candidate, distance in zip(candidates, rms_distance[row]):
                if (sample_runs[candidate] == sample_runs[row]
                        or distance > radius):
                    continue
                source_run = int(sample_runs[candidate])
                if source_run not in selected_runs:
                    if len(selected_runs) >= MAX_NEIGHBOR_RUNS:
                        continue
                    selected_runs.add(source_run)
                if counts_by_run.get(source_run, 0) >= MAX_NEIGHBORS_PER_RUN:
                    continue
                chosen_list.append(int(candidate))
                chosen_distances.append(float(distance))
                counts_by_run[source_run] = counts_by_run.get(source_run, 0) + 1
            chosen = np.asarray(chosen_list, dtype=np.int64)
            if not len(chosen):
                continue
            other_runs = np.unique(sample_runs[chosen])
            if len(chosen) < MIN_NEIGHBORS or len(other_runs) < MIN_OTHER_RUNS:
                continue
            response = target[chosen]
            valid_response = np.isfinite(response).all(axis=1)
            response = response[valid_response]
            response_runs = sample_runs[chosen][valid_response]
            run_responses = np.asarray([
                np.mean(response[response_runs == source_run], axis=0)
                for source_run in np.unique(response_runs)
            ])
            if len(response) < MIN_NEIGHBORS or len(run_responses) < MIN_OTHER_RUNS:
                continue
            local_std[row] = np.std(run_responses, axis=0)
            local_count[row] = len(response)
            local_run_count[row] = len(np.unique(response_runs))
            nearest_distance[row] = float(np.median(chosen_distances))
        result[radius] = {
            "local_std": local_std,
            "neighbor_count": local_count,
            "other_run_count": local_run_count,
            "median_rms_distance": nearest_distance,
        }
    return result


def _history_feature_sets(data: dict[str, np.ndarray], current: np.ndarray,
                          indices: np.ndarray) -> dict[int, np.ndarray]:
    features = {0: current[indices]}
    for milliseconds in HISTORY_MS[1:]:
        steps = int(round(milliseconds / 1000 / DT_S))
        features[milliseconds] = _history_features(current, indices, steps)
    return features


def _target_responses(data: dict[str, np.ndarray],
                      indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
    acceleration = data["simulator_linear_acceleration"]
    rigid = data["simulator_rigid_state"]
    r = rigid[:, 12]
    yaw_accel = np.full(len(r), np.nan, dtype=np.float64)
    for start_value, end_value in data["sequence_bounds"]:
        start, end = int(start_value), int(end_value)
        if end - start > 1:
            yaw_accel[start:end - 1] = np.diff(r[start:end]) / DT_S
    target = np.column_stack((acceleration[:, 0], acceleration[:, 1], yaw_accel))
    return target[indices], ["ax_sim_body_mps2", "ay_sim_body_mps2",
                             "yaw_acceleration_rps2"]


def _summarize(data: dict[str, np.ndarray], sample_indices: np.ndarray,
               sample_runs: np.ndarray, sample_speed: np.ndarray,
               features_by_history: dict[int, np.ndarray], target: np.ndarray,
               target_names: list[str], seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    run_ids = data["run_ids"].astype(str)
    splits = data["run_splits"].astype(str)
    train_queries = splits[sample_runs] == "train"
    nearest_by_history = {}
    for milliseconds, features in features_by_history.items():
        # Fit scale only on sampled training runs; retain fixed feature scale
        # for all query/neighbor points on this same set.
        scaled, _, _ = _scale_features(features, train_queries)
        nearest_by_history[milliseconds] = _nearest_response(
            scaled, sample_indices, sample_runs, target,
            MATCH_RMS_RADII, seed + milliseconds)

    report_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    query_steering = np.abs(data["frames"][sample_indices, 3])
    regime_masks = {
        "speed_0_to_3": (sample_speed >= 0) & (sample_speed < 3),
        "speed_3_to_5": (sample_speed >= 3) & (sample_speed < 5),
        "speed_5_to_7": (sample_speed >= 5) & (sample_speed < 7),
        "speed_7_to_9": (sample_speed >= 7) & (sample_speed < 9),
        "speed_9_to_10": (sample_speed >= 9) & (sample_speed < 10),
        "speed_10_to_11": (sample_speed >= 10) & (sample_speed < 11),
        "speed_11_to_12": (sample_speed >= 11) & (sample_speed <= 12),
        "steering_abs_0_to_0_10": query_steering < .10,
        "steering_abs_0_10_to_0_20": (query_steering >= .10) & (query_steering < .20),
        "steering_abs_0_20_to_0_30": (query_steering >= .20) & (query_steering < .30),
        "steering_abs_0_30_to_0_40": (query_steering >= .30) & (query_steering < .40),
        "steering_abs_0_40_to_0_524": (query_steering >= .40) & (query_steering <= .524),
    }
    current_support = nearest_by_history[0]
    for milliseconds in HISTORY_MS:
        for radius in MATCH_RMS_RADII:
            result = nearest_by_history[milliseconds][radius]
            base_result = current_support[radius]
            supported = np.isfinite(result["median_rms_distance"])
            row = {
                "history_ms": milliseconds,
                "match_rms_radius": radius,
                "sampled_query_count": int(len(sample_indices)),
                "supported_query_count": int(np.count_nonzero(supported)),
                "support_fraction": float(np.mean(supported)),
                "supported_query_run_count": int(
                    len(np.unique(sample_runs[supported]))),
                "median_neighbor_count": (float(np.median(
                    result["neighbor_count"][supported])) if np.any(supported)
                    else None),
                "median_other_run_count": (float(np.median(
                    result["other_run_count"][supported])) if np.any(supported)
                    else None),
                "median_match_rms_distance": (float(np.median(
                    result["median_rms_distance"][supported]))
                    if np.any(supported) else None),
                "targets": {},
                "paired_to_current_state_only": {},
                "by_regime": {},
            }
            for target_index, target_name in enumerate(target_names):
                local = result["local_std"][:, target_index]
                ci_low, ci_high, clustered_runs = _cluster_ci(
                    local, sample_runs, BOOTSTRAP_REPEATS, rng)
                row["targets"][target_name] = {
                    "median_cross_run_local_std": (
                        float(np.nanmedian(local)) if np.isfinite(local).any()
                        else None),
                    "p90_cross_run_local_std": (
                        float(np.nanquantile(local, .90))
                        if np.isfinite(local).any() else None),
                    "run_cluster_bootstrap_95pct_ci": [ci_low, ci_high],
                    "query_runs_with_supported_match": clustered_runs,
                }
                common = (np.isfinite(local)
                          & np.isfinite(base_result["local_std"][:, target_index]))
                base_local = base_result["local_std"][common, target_index]
                history_local = local[common]
                ratios = history_local / np.maximum(base_local, 1.0e-8)
                ratio_low, ratio_high, ratio_runs = _cluster_ci(
                    ratios, sample_runs[common], BOOTSTRAP_REPEATS, rng)
                row["paired_to_current_state_only"][target_name] = {
                    "common_supported_query_count": int(np.count_nonzero(common)),
                    "median_local_std_ratio": (
                        float(np.median(ratios)) if len(ratios) else None),
                    "run_cluster_bootstrap_95pct_ci": [ratio_low, ratio_high],
                    "paired_query_run_count": ratio_runs,
                }
                for regime_name, regime_mask in regime_masks.items():
                    selected = local[regime_mask & np.isfinite(local)]
                    row["by_regime"].setdefault(regime_name, {})[target_name] = {
                        "matched_query_count": int(len(selected)),
                        "median_local_std": (float(np.median(selected))
                                             if len(selected) else None),
                        "query_run_count": int(len(np.unique(
                            sample_runs[regime_mask & np.isfinite(local)]))),
                    }
                for query_index in np.flatnonzero(np.isfinite(local)):
                    query_rows.append({
                        "query_index": int(query_index),
                        "sample_index": int(sample_indices[query_index]),
                        "run_id": run_ids[sample_runs[query_index]],
                        "split": splits[sample_runs[query_index]],
                        "speed_mps": float(sample_speed[query_index]),
                        "steering_rad": float(data["frames"][
                            sample_indices[query_index], 3]),
                        "history_ms": milliseconds,
                        "match_rms_radius": radius,
                        "target": target_name,
                        "local_response_std": float(local[query_index]),
                        "neighbor_count": int(result["neighbor_count"][query_index]),
                        "other_run_count": int(result["other_run_count"][query_index]),
                        "median_match_rms_distance": float(
                            result["median_rms_distance"][query_index]),
                    })
            report_rows.append(row)
    return {"comparisons": report_rows}, query_rows


def _write_query_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(list(rows[0]) if rows else []))
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def build(dataset_dir: Path, output_dir: Path, seed: int = 20261001
          ) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(dataset_dir / "openplane_dynamics.npz", allow_pickle=False)
    if (int(z["schema_version"][0]) != 8
            or str(z["dataset_role"].astype(str)[0]) != "race_domain"
            or float(z["domain_speed_cap_mps"][0]) != 12.0):
        raise ValueError("matched-state analysis requires race-domain schema 8")
    required = ("frames", "simulator_rigid_state",
                "simulator_linear_acceleration", "sequence_bounds",
                "sequence_run_index", "run_ids", "run_splits",
                "frame_run_index", "frame_domain_speed_mps")
    missing = [key for key in required if key not in z.files]
    if missing:
        raise ValueError("reset-safe dataset missing arrays: " + ", ".join(missing))
    data = {key: z[key] for key in required}
    z.close()
    current = _current_features(data)
    if (not np.isfinite(data["frame_domain_speed_mps"]).all()
            or np.any(data["frame_domain_speed_mps"] > 12.0)):
        raise ValueError("matched-state dataset violates the 12 m/s race cap")
    indices, runs, speeds = _balanced_query_indices(data, current, seed)
    targets, target_names = _target_responses(data, indices)
    features = _history_feature_sets(data, current, indices)
    compare, query_rows = _summarize(
        data, indices, runs, speeds, features, targets, target_names, seed)
    query_path = output_dir / "matched_state_query_metrics.csv.gz"
    _write_query_csv(query_path, query_rows)
    run_ids = data["run_ids"].astype(str)
    selected_run_ids = sorted({run_ids[index] for index in np.unique(runs)})
    report = {
        "schema_version": 1,
        "dataset": str(dataset_dir / "openplane_dynamics.npz"),
        "fixed_dt_s": DT_S,
        "dataset_role": "race_domain",
        "speed_cap_mps": 12.0,
        "matching_method": (
            "Nearest neighbors in robust-IQR-scaled observed state/input space, "
            "with all neighbors from the query's source run excluded. History "
            "adds the state/input snapshot at t-H and four means over equal "
            "subwindows of the preceding H interval."),
        "current_state_input_features": [
            "u_com", "v_com", "yaw_rate", "steering_feedback",
            "throttle_feedback", "rear_left_surface_speed",
            "rear_right_surface_speed", "steering_command",
            "throttle_command"],
        "history_ms": list(HISTORY_MS),
        "match_rms_radii_iqr_units": list(MATCH_RMS_RADII),
        "neighbor_policy": {
            "candidate_count": NEIGHBOR_CANDIDATES,
            "minimum_neighbors": MIN_NEIGHBORS,
            "minimum_independent_other_runs": MIN_OTHER_RUNS,
            "maximum_neighbors_per_other_run": MAX_NEIGHBORS_PER_RUN,
            "maximum_other_runs_per_query": MAX_NEIGHBOR_RUNS,
            "max_queries_per_run": MAX_QUERIES_PER_RUN,
            "bootstrap_unit": "query source run",
            "bootstrap_replicates": BOOTSTRAP_REPEATS,
        },
        "seed": seed,
        "balanced_query_count": int(len(indices)),
        "query_run_count": int(len(selected_run_ids)),
        "query_run_ids": selected_run_ids,
        "query_splits_included": ["train", "validation"],
        "targets": target_names,
        "results": compare,
        "query_metrics_csv_gz": str(query_path),
        "interpretation_limits": [
            "This estimates observed response dispersion under cross-run local matching; it is not a causal experiment.",
            "History summaries are fixed low-dimensional descriptors, not a learned latent-state estimate.",
            "Match support decreases as history dimensions grow; paired comparisons use only queries supported at both horizons.",
            "Neighbors and query samples are correlated within a run; confidence intervals resample query runs, not frames.",
        ],
    }
    output_path = output_dir / "matched_state_analysis.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    try:
        report = build(args.dataset_dir, args.output_dir, args.seed)
    except (OSError, ValueError, KeyError) as exc:
        print(f"matched-state analysis failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    print(json.dumps({
        "query_count": report["balanced_query_count"],
        "query_runs": report["query_run_count"],
        "history_ms": report["history_ms"],
        "comparisons": len(report["results"]["comparisons"]),
    }, indent=2))
    print(f"wrote {args.output_dir / 'matched_state_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

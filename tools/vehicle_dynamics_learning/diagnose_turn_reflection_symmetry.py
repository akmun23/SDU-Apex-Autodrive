#!/usr/bin/env python3
"""Test measured simulator transition symmetry under left/right reflection.

Held-out transitions are paired only within the same whole run, at similar
speed, body motion, actuator state, wheel speeds and current/delayed commands.
Mirroring changes lateral/yaw/steering signs and swaps rear wheels. The report
tests the data; it does not assume symmetry or augment training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from tools.vehicle_dynamics_learning.diagnose_effective_race_first_step import (
    _transition_indices,
)
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    physical_state_from_dataset,
)
from tools.vehicle_dynamics_learning.turn_reflection import (
    reflect_commands,
    reflect_state_channels,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/" \
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/" \
    "encoder_raw_state_teacher_v1"
DEFAULT_DYNAMIC = DATA_ROOT / "openplane_dynamics_raw_wheels.npz"
DEFAULT_PRACTICE = DATA_ROOT / "practice_dynamics_raw_wheels.npz"
DEFAULT_OUTPUT = DATA_ROOT / "turn_reflection_symmetry_validation_v1.json"

# Feature order: state u/v/r/steer/throttle/wheel-L/wheel-R, then the delayed
# and current steering/throttle command. Limits are deliberately explicit so
# a reported pair has nearby states and commands, not just a small aggregate
# Euclidean distance.
FEATURE_LIMITS = np.asarray((0.10, 0.10, 0.10, 0.025, 0.03,
                             0.25, 0.25, 0.025, 0.03, 0.025, 0.03))
FEATURE_SCALES = FEATURE_LIMITS.copy()
HARD_STEERING_RAD = 0.30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reflect_features(features: np.ndarray) -> np.ndarray:
    reflected = np.asarray(features, dtype=np.float64).copy()
    if reflected.ndim != 2 or reflected.shape[1] != 11:
        raise ValueError("reflection features must have shape (N, 11)")
    reflected[:, :7] = reflect_state_channels(reflected[:, :7])
    reflected[:, 7:9] = reflect_commands(reflected[:, 7:9])
    reflected[:, 9:11] = reflect_commands(reflected[:, 9:11])
    return reflected


def _reflect_state_delta(delta: np.ndarray) -> np.ndarray:
    return reflect_state_channels(np.asarray(delta, dtype=np.float64))


def _pair_run(state: np.ndarray, frames: np.ndarray, rows: np.ndarray,
              max_neighbors: int = 32) -> tuple[list[tuple[int, int]], np.ndarray]:
    if len(rows) < 2:
        return [], np.empty((0, len(FEATURE_LIMITS)))
    # Transition i receives command i+1; actuator state at i depends on the
    # preceding command. Both are represented when looking for a mirror pair.
    features = np.column_stack((
        state[rows, :7], frames[rows - 1, 7:9], frames[rows + 1, 7:9]))
    reflected_query = _reflect_features(features)
    tree = cKDTree(features / FEATURE_SCALES)
    k = min(max_neighbors, len(rows))
    _, candidates = tree.query(reflected_query / FEATURE_SCALES, k=k)
    if k == 1:
        candidates = candidates[:, None]
    pairs: set[tuple[int, int]] = set()
    feature_differences: dict[tuple[int, int], np.ndarray] = {}
    for local_i, candidate_row in enumerate(candidates):
        global_i = int(rows[local_i])
        for local_j in np.atleast_1d(candidate_row):
            global_j = int(rows[int(local_j)])
            if abs(global_i - global_j) < 80:
                continue
            observed_j = features[int(local_j)]
            mismatch = np.abs(features[local_i] - _reflect_features(
                observed_j[None, :])[0])
            if np.any(mismatch > FEATURE_LIMITS):
                continue
            pair = tuple(sorted((global_i, global_j)))
            if global_i == global_j:
                continue
            pairs.add(pair)
            feature_differences[pair] = mismatch
            break
    ordered = sorted(pairs)
    differences = (np.stack([feature_differences[pair] for pair in ordered])
                   if ordered else np.empty((0, len(FEATURE_LIMITS))))
    return ordered, differences


def _rmse(values: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2))) \
        if np.size(values) else None


def _summarize_pairs(pairs: list[tuple[int, int]], state: np.ndarray,
                     run_id: str) -> dict[str, Any]:
    if not pairs:
        return {"pair_count": 0, "independent_whole_runs": int(bool(run_id)),
                "increment_error_rmse": {}}
    left = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    right = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    delta = state[:, :7][1:] - state[:, :7][:-1]
    left_change = delta[left]
    right_change = _reflect_state_delta(delta[right])
    difference = left_change - right_change
    metrics = {
        "forward_speed_increment_mps": _rmse(difference[:, 0]),
        "lateral_speed_increment_mps": _rmse(difference[:, 1]),
        "yaw_rate_increment_rps": _rmse(difference[:, 2]),
        "steering_increment_rad": _rmse(difference[:, 3]),
        "throttle_feedback_increment_norm": _rmse(difference[:, 4]),
        "rear_wheel_increment_pair_mps": _rmse(
            np.sqrt(np.mean(difference[:, 5:7] ** 2, axis=1))),
    }
    signed_bias = np.mean(difference, axis=0)
    return {
        "pair_count": len(pairs),
        "independent_whole_runs": 1,
        "run_id": run_id,
        "increment_error_rmse": metrics,
        "increment_error_bias": {
            "forward_speed_increment_mps": float(signed_bias[0]),
            "lateral_speed_increment_mps": float(signed_bias[1]),
            "yaw_rate_increment_rps": float(signed_bias[2]),
            "steering_increment_rad": float(signed_bias[3]),
            "throttle_feedback_increment_norm": float(signed_bias[4]),
        },
    }


def _diagnose_dataset(path: Path, split: str) -> dict[str, Any]:
    data = _load_dataset(path)
    if not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("turn-reflection analysis requires 25 ms samples")
    state = physical_state_from_dataset(data)
    frames = np.asarray(data["frames"], dtype=np.float64)
    rows, run_indices = _transition_indices(data, {split})
    run_ids = np.asarray(data["run_ids"]).astype(str)
    per_run: dict[str, Any] = {}
    for run in sorted(set(run_indices.tolist())):
        run_rows = rows[run_indices == run]
        pairs, feature_differences = _pair_run(state, frames, run_rows)
        run_name = str(run_ids[run])
        all_summary = _summarize_pairs(pairs, state, run_name)
        high_pairs = [pair for pair in pairs
                      if min(abs(state[pair[0], 3]),
                             abs(state[pair[1], 3])) >= HARD_STEERING_RAD]
        high_summary = _summarize_pairs(high_pairs, state, run_name)
        per_run[run_name] = {
            "eligible_transition_count": int(len(run_rows)),
            "matched_pairs": all_summary,
            "high_steering_matched_pairs": high_summary,
        }
        if len(feature_differences):
            per_run[run_name]["match_feature_difference_p95"] = {
                "u_mps": float(np.percentile(feature_differences[:, 0], 95)),
                "v_mps": float(np.percentile(feature_differences[:, 1], 95)),
                "yaw_rate_rps": float(np.percentile(feature_differences[:, 2], 95)),
                "steering_rad": float(np.percentile(feature_differences[:, 3], 95)),
            }
    return {
        "dataset": str(path),
        "dataset_sha256": _sha256(path),
        "split": split,
        "whole_run_count": len(per_run),
        "pairing_definition": {
            "same_whole_run": True,
            "minimum_sample_separation": 80,
            "feature_absolute_limits": FEATURE_LIMITS.tolist(),
            "reflection": "u/throttle unchanged; lateral/yaw/steering signs reversed; rear wheels swapped",
        },
        "per_run": per_run,
    }


def diagnose(dynamic_dataset: Path, practice_dataset: Path,
             output: Path) -> dict[str, Any]:
    dynamic_dataset, practice_dataset, output = (
        path.resolve() for path in (dynamic_dataset, practice_dataset, output))
    result = {
        "schema_version": 1,
        "purpose": "test whether held-out simulator response supports left-right reflection as a structural prior",
        "test_and_final_test_used": False,
        "model_or_training_data_modified": False,
        "domains": {
            "dynamic_validation": _diagnose_dataset(dynamic_dataset, "validation"),
            "practice_unseen": _diagnose_dataset(practice_dataset, "unseen_practice"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-dataset", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = diagnose(args.dynamic_dataset, args.practice_dataset, args.output)
    concise = {}
    for domain, report in result["domains"].items():
        concise[domain] = {run: {
            "pairs": value["matched_pairs"]["pair_count"],
            "high_steering_pairs": value["high_steering_matched_pairs"]["pair_count"],
            "yaw_rate_increment_reflection_rmse": value["matched_pairs"][
                "increment_error_rmse"].get("yaw_rate_increment_rps"),
            "high_steering_yaw_rate_increment_reflection_rmse": value[
                "high_steering_matched_pairs"]["increment_error_rmse"].get(
                    "yaw_rate_increment_rps"),
        } for run, value in report["per_run"].items()}
    print(json.dumps({"output": str(args.output.resolve()), "summary": concise},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

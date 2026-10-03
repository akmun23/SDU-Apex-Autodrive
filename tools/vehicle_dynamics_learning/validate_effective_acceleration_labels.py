#!/usr/bin/env python3
"""Validate simulator acceleration frame/time alignment against 25 ms truth.

This is WP8 for the replacement offline plant. Labels are finite differences
of the rigid body's body-frame COM velocity. Simulator acceleration is used
only to diagnose its coordinate convention and timestamp alignment; this tool
does not modify the dataset or fit a vehicle model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    generalized_acceleration_targets,
    physical_state_from_dataset,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
    "replacement_teacher_dataset_v1/openplane_dynamics.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
    "acceleration_label_validation_20261002/report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotate_xyzw(quaternion: np.ndarray, vectors: np.ndarray,
                 inverse: bool = False) -> np.ndarray:
    """Rotate vectors by normalized xyzw quaternions, without scipy."""
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vectors, dtype=np.float64)
    if q.shape != (len(v), 4) or v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("quaternion/vector arrays must have shapes (N,4)/(N,3)")
    norm = np.linalg.norm(q, axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm < 1e-8):
        raise ValueError("invalid simulator orientation quaternion")
    q = q / norm[:, None]
    xyz = q[:, :3].copy()
    if inverse:
        xyz *= -1.0
    twice_cross = 2.0 * np.cross(xyz, v)
    return v + q[:, 3, None] * twice_cross + np.cross(xyz, twice_cross)


def _nlerp_xyzw(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Normalized shortest-arc midpoint orientation for adjacent 25 ms rows."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 4:
        raise ValueError("quaternion endpoints must align as (N,4)")
    flip = np.sum(left * right, axis=1) < 0.0
    right = right.copy()
    right[flip] *= -1.0
    middle = left + right
    norm = np.linalg.norm(middle, axis=1)
    if np.any(norm < 1e-8):
        raise ValueError("cannot interpolate antipodal quaternion endpoints")
    return middle / norm[:, None]


def _correlation(target: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(target) < 2 or np.std(target) < 1e-10 or np.std(prediction) < 1e-10:
        return None
    return float(np.corrcoef(target, prediction)[0, 1])


def _axis_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    return {
        "samples": int(len(target)),
        "rmse_mps2": float(np.sqrt(np.mean(error ** 2))),
        "mae_mps2": float(np.mean(np.abs(error))),
        "bias_mps2": float(np.mean(error)),
        "correlation": _correlation(target, prediction),
    }


def _pair_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return {
        "ax": _axis_metrics(target[:, 0], prediction[:, 0]),
        "ay": _axis_metrics(target[:, 1], prediction[:, 1]),
        "combined_rmse_mps2": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }


def _scalar_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "absolute_error_p95": float(np.quantile(np.abs(error), 0.95)),
        "absolute_error_max": float(np.max(np.abs(error))),
    }


def _make_candidates(rigid: np.ndarray, simulator_acceleration: np.ndarray,
                     transition_indices: np.ndarray) -> dict[str, np.ndarray]:
    index = transition_indices
    acceleration_now = simulator_acceleration[index, :3].astype(np.float64)
    acceleration_next = simulator_acceleration[index + 1, :3].astype(np.float64)
    quaternion_now = rigid[index, 3:7].astype(np.float64)
    quaternion_next = rigid[index + 1, 3:7].astype(np.float64)
    quaternion_mid = _nlerp_xyzw(quaternion_now, quaternion_next)

    direct_now = acceleration_now[:, :2]
    direct_next = acceleration_next[:, :2]
    body_now = _rotate_xyzw(quaternion_now, acceleration_now, inverse=True)[:, :2]
    body_next = _rotate_xyzw(quaternion_next, acceleration_next, inverse=True)[:, :2]
    body_mid_now = _rotate_xyzw(quaternion_mid, acceleration_now, inverse=True)[:, :2]
    body_mid_next = _rotate_xyzw(quaternion_mid, acceleration_next, inverse=True)[:, :2]
    return {
        "unrotated_start_sample": direct_now,
        "unrotated_end_sample": direct_next,
        "unrotated_interval_mean": 0.5 * (direct_now + direct_next),
        "quaternion_inverse_start_sample": body_now,
        "quaternion_inverse_end_sample": body_next,
        "quaternion_inverse_interval_mean": 0.5 * (body_now + body_next),
        "quaternion_inverse_midpoint_orientation_mean":
            0.5 * (body_mid_now + body_mid_next),
    }


def _frame_speed_mismatch(data: dict[str, Any]) -> np.ndarray:
    frames = np.asarray(data["frames"], dtype=np.float64)
    # Existing wheel/body mismatch proxy uses bridge rear-axle longitudinal
    # speed, not COM speed. It is a regime tag, not ground-truth tire slip.
    return np.abs(0.5 * (frames[:, 5] + frames[:, 6]) - frames[:, 0])


def _regime_masks(data: dict[str, Any], transitions: np.ndarray,
                  targets: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frames = np.asarray(data["frames"], dtype=np.float64)
    state = physical_state_from_dataset(data).astype(np.float64)
    speed = np.hypot(state[:, 0], state[:, 1])
    mismatch = _frame_speed_mismatch(data)
    run_by_frame = np.empty(len(frames), dtype=np.int32)
    for (start, end), run in zip(data["bounds"], data["seq_run"]):
        run_by_frame[int(start):int(end)] = int(run)
    split = np.asarray(data["splits"]).astype(str)[run_by_frame[transitions]]
    train_transitions = transitions[split == "train"]
    train_speed = speed[train_transitions]
    train_throttle = frames[train_transitions, 8]
    racing_support = (train_speed <= 12.0) & (train_throttle <= 0.50)
    train_mismatch = mismatch[train_transitions[racing_support]]
    if len(train_mismatch) < 1000:
        raise ValueError("insufficient training rows define in-envelope mismatch")
    p90 = float(np.quantile(train_mismatch, 0.90))
    p95 = float(np.quantile(train_mismatch, 0.95))
    local_speed = speed[transitions]
    local_steering = np.abs(state[transitions, 3])
    local_throttle = frames[transitions, 8]
    local_mismatch = mismatch[transitions]
    in_racing_support = (local_speed <= 12.0) & (local_throttle <= 0.50)
    masks = {
        "ordinary_low_steering_low_wheel_mismatch": (
            (local_speed >= 1.0) & (local_speed <= 8.0)
            & (local_steering < 0.20) & (local_throttle > 0.05)
            & (local_mismatch <= p90)),
        "high_steering": in_racing_support & (local_steering >= 0.40),
        "zero_throttle_deceleration": (
            (local_speed >= 1.0) & (local_throttle <= 1e-3)
            & (targets[:, 0] < 0.0)),
        "high_wheel_body_mismatch_proxy": (
            in_racing_support & (local_mismatch >= p95)),
    }
    thresholds = {
        "wheel_body_mismatch_proxy_train_p90_mps": p90,
        "wheel_body_mismatch_proxy_train_p95_mps": p95,
        "wheel_body_mismatch_threshold_training_domain": {
            "ground_truth_body_speed_mps_max": 12.0,
            "throttle_command_norm_max": 0.50,
            "training_rows": int(len(train_mismatch)),
        },
        "wheel_body_mismatch_proxy_definition":
            "abs(mean(rear wheel surface speed) - bridge rear-axle u); proxy only, not tire-slip truth",
    }
    return masks, thresholds


def validate(dataset_path: Path) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    data = _load_dataset(dataset_path)
    if int(data["schema_version"]) != 9:
        raise ValueError("WP8 requires the immutable schema-9 teacher dataset")
    if data["simulator_rigid_state"] is None or data[
            "simulator_linear_acceleration"] is None:
        raise ValueError("dataset lacks simulator rigid-body state/acceleration")

    state = physical_state_from_dataset(data).astype(np.float64)
    targets_all = generalized_acceleration_targets(
        state, data["bounds"], data["dt_s"]).astype(np.float64)
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    simulator_acceleration = np.asarray(
        data["simulator_linear_acceleration"], dtype=np.float64)
    frame_run = np.empty(len(state), dtype=np.int32)
    sequence_id = np.empty(len(state), dtype=np.int32)
    for seq, ((start_raw, end_raw), run_raw) in enumerate(
            zip(data["bounds"], data["seq_run"])):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        frame_run[start:end] = run
        sequence_id[start:end] = seq
        if not np.allclose(data["dt_s"][start:end], DT_S, rtol=0, atol=1e-7):
            raise ValueError("acceleration-label validation requires exact 25 ms dt")

    all_indices = np.flatnonzero(np.isfinite(targets_all[:, :2]).all(axis=1))
    run_split = np.asarray(data["splits"]).astype(str)[frame_run[all_indices]]
    selected = np.isin(run_split, ("train", "validation"))
    indices = all_indices[selected]
    targets = targets_all[indices, :2]
    if not np.isfinite(simulator_acceleration[indices]).all():
        raise ValueError("non-finite simulator acceleration in scored transitions")
    candidates = _make_candidates(rigid, simulator_acceleration, indices)
    regimes, thresholds = _regime_masks(data, indices, targets_all[indices])
    frames = np.asarray(data["frames"], dtype=np.float64)
    convention_predictions = {
        "u_rear_from_com_longitudinal": rigid[indices, 7],
        "v_rear_from_com_lateral": (
            rigid[indices, 8] - 0.15532 * rigid[indices, 12]),
        "yaw_rate": rigid[indices, 12],
    }
    convention_targets = {
        "u_rear_from_com_longitudinal": frames[indices, 0],
        "v_rear_from_com_lateral": frames[indices, 1],
        "yaw_rate": frames[indices, 2],
    }

    per_split: dict[str, Any] = {}
    run_ids = np.asarray(data["run_ids"]).astype(str)
    split_by_transition = np.asarray(data["splits"]).astype(str)[frame_run[indices]]
    for split_name in ("train", "validation"):
        split_mask = split_by_transition == split_name
        split_indices = indices[split_mask]
        split_targets = targets[split_mask]
        split_report: dict[str, Any] = {
            "transitions": int(split_mask.sum()),
            "independent_runs": int(len(np.unique(frame_run[split_indices]))),
            "per_candidate": {},
            "per_regime": {},
            "rigid_state_convention_crosscheck": {
                name: _scalar_metrics(
                    convention_targets[name][split_mask],
                    convention_predictions[name][split_mask])
                for name in convention_targets
            },
        }
        for name, prediction_all in candidates.items():
            prediction = prediction_all[split_mask]
            result = _pair_metrics(split_targets, prediction)
            by_run = {}
            local_runs = frame_run[split_indices]
            for run in sorted(np.unique(local_runs)):
                mask = local_runs == run
                by_run[str(run_ids[run])] = _pair_metrics(
                    split_targets[mask], prediction[mask])
            result["run_metrics"] = by_run
            result["macro_run_combined_rmse_mps2"] = float(np.mean([
                item["combined_rmse_mps2"] for item in by_run.values()]))
            split_report["per_candidate"][name] = result

        for regime_name, regime_all in regimes.items():
            local_mask = split_mask & regime_all
            if not np.any(local_mask):
                split_report["per_regime"][regime_name] = {
                    "transitions": 0, "independent_runs": 0,
                    "per_candidate": {},
                }
                continue
            regime_report: dict[str, Any] = {
                "transitions": int(local_mask.sum()),
                "independent_runs": int(len(np.unique(frame_run[indices[local_mask]]))),
                "per_candidate": {},
            }
            for name, prediction_all in candidates.items():
                local_target = targets[local_mask]
                local_prediction = prediction_all[local_mask]
                metric = _pair_metrics(local_target, local_prediction)
                local_runs = frame_run[indices[local_mask]]
                metric["run_metrics"] = {
                    str(run_ids[run]): _pair_metrics(
                        local_target[local_runs == run],
                        local_prediction[local_runs == run])
                    for run in sorted(np.unique(local_runs))
                }
                regime_report["per_candidate"][name] = metric
            split_report["per_regime"][regime_name] = regime_report
        per_split[split_name] = split_report

    best = min(
        per_split["validation"]["per_candidate"],
        key=lambda name: per_split["validation"]["per_candidate"][name][
            "macro_run_combined_rmse_mps2"],
    )
    train_best = min(
        per_split["train"]["per_candidate"],
        key=lambda name: per_split["train"]["per_candidate"][name][
            "macro_run_combined_rmse_mps2"],
    )
    return {
        "schema_version": 1,
        "work_package": "WP8_acceleration_transform_validation",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_role": "replacement_teacher_dataset_v1",
        "scored_splits": ["train", "validation"],
        "test_and_final_test_used": False,
        "timebase_s": DT_S,
        "transition_alignment":
            "finite difference from row k to k+1; simulator acceleration tested at start, end and interval-mean alignments",
        "body_equations": {
            "du_dt": "ax + yaw_rate * v",
            "dv_dt": "ay - yaw_rate * u",
            "ax_target": "du_dt - yaw_rate * v",
            "ay_target": "dv_dt + yaw_rate * u",
            "yaw_accel_target": "d(yaw_rate)/dt",
            "rigid_velocity_convention":
                "body-frame COM velocity; independently checked against same-packet odometry/rear-axle twist",
        },
        "rigid_state_convention": {
            "longitudinal": "simulator rigid-state body COM u equals rear-axle odometry u for this rear-offset geometry",
            "lateral": "rear-axle odometry v = simulator body COM v - 0.15532 m * yaw_rate",
            "yaw_rate": "simulator rigid-state body yaw rate equals aligned odometry yaw rate",
            "crosscheck_source": "same-packet dataset frame channels; split-separated metrics below",
        },
        "quaternion_convention": "simulator rigid-state quaternion is xyzw",
        "regime_thresholds_train_only": thresholds,
        "best_validation_candidate_by_macro_run_rmse": best,
        "best_training_candidate_by_macro_run_rmse": train_best,
        "acceleration_labels_for_training":
            "finite differences of simulator body-frame COM u/v/yaw-rate at exact 25 ms; simulator linear acceleration retained as frame/alignment diagnostic only",
        "splits": per_split,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        report = validate(args.dataset)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if (existing.get("work_package")
                    != "WP8_acceleration_transform_validation"
                    or existing.get("dataset_sha256")
                    != report.get("dataset_sha256")):
                raise FileExistsError(
                    f"refusing to overwrite unrelated WP8 report: {output}")
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"acceleration label validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "validation_rows": report["splits"]["validation"]["transitions"],
        "validation_runs": report["splits"]["validation"]["independent_runs"],
        "best_train_alignment": report["best_training_candidate_by_macro_run_rmse"],
        "best_validation_alignment": report["best_validation_candidate_by_macro_run_rmse"],
        "regimes": {
            split: {
                name: {
                    "transitions": details["transitions"],
                    "runs": details["independent_runs"],
                }
                for name, details in split_report["per_regime"].items()
            }
            for split, split_report in report["splits"].items()
        },
        "report": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

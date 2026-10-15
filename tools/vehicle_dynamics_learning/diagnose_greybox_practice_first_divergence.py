#!/usr/bin/env python3
"""Find which command-only grey-box state first diverges on held-out runs.

Ground truth initializes each sequence once and is used only to score the
resulting recursive rollout. After that initialization, the plant receives
only the recorded steering/throttle command sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    DT_S,
    WHEEL_RADIUS_M,
    _initial_state,
    _integrate_pose,
    _physical_model,
    _targets,
)
from tools.vehicle_dynamics_learning.score_four_wheel_greybox_free_run import (
    HISTORY_STEPS,
    _sequence_rows,
    _wrap_angle,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch


THRESHOLDS = {
    "position_xy_m": 0.05,
    "heading_abs_rad": 0.10,
    "u_abs_mps": 0.10,
    "v_abs_mps": 0.10,
    "yaw_rate_abs_rps": 0.10,
    "rear_left_wheel_abs_mps": 0.10,
    "rear_right_wheel_abs_mps": 0.10,
    "steering_feedback_abs_rad": 0.02,
    "throttle_feedback_abs": 0.05,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rollout(model, data: dict[str, Any], physical_state: np.ndarray,
                  start: int, end: int, run_id: str, condition: str,
                  torch, device) -> dict[str, np.ndarray | str | float]:
    context_end = start + HISTORY_STEPS - 1
    initial = _initial_state(data, context_end, physical_state, start)
    commands_np = data["frames"][context_end:end - 1, 7:9]
    with torch.no_grad():
        predicted_internal, _ = model.rollout(
            torch.as_tensor(initial[None], dtype=torch.float32, device=device),
            torch.as_tensor(commands_np[None], dtype=torch.float32,
                            device=device))
    internal = predicted_internal[0].cpu().numpy()
    predicted = np.column_stack((
        internal[:, :3], internal[:, 5:7] * WHEEL_RADIUS_M,
        internal[:, 17:19],
    )).astype(np.float64)
    truth = np.column_stack((
        physical_state[context_end + 1:end],
        data["frames"][context_end + 1:end, 3:5],
    )).astype(np.float64)
    pose0 = data["simulator_pose_xyyaw"][context_end].astype(np.float64)
    pose_truth = data["simulator_pose_xyyaw"][context_end + 1:end].astype(
        np.float64)
    predicted_pose = _integrate_pose(predicted, physical_state[context_end],
                                     pose0)
    pose_error = predicted_pose - pose_truth
    pose_error[:, 2] = _wrap_angle(pose_error[:, 2])
    state_error = predicted - truth
    speed_error = (np.linalg.norm(predicted[:, :2], axis=1)
                   - np.linalg.norm(truth[:, :2], axis=1))
    errors = {
        "position_xy_m": np.linalg.norm(pose_error[:, :2], axis=1),
        "heading_abs_rad": np.abs(pose_error[:, 2]),
        "u_abs_mps": state_error[:, 0],
        "v_abs_mps": state_error[:, 1],
        "yaw_rate_abs_rps": state_error[:, 2],
        "rear_left_wheel_abs_mps": state_error[:, 3],
        "rear_right_wheel_abs_mps": state_error[:, 4],
        "steering_feedback_abs_rad": state_error[:, 5],
        "throttle_feedback_abs": state_error[:, 6],
        "speed_abs_mps": speed_error,
    }
    return {
        "run_id": run_id,
        "condition": condition,
        "initial_speed_mps": float(np.linalg.norm(physical_state[context_end, :2])),
        "initial_steering_feedback_rad": float(data["frames"][context_end, 3]),
        "initial_throttle_feedback": float(data["frames"][context_end, 4]),
        "command_steering_abs_p95_rad": float(np.quantile(
            np.abs(commands_np[:, 0]), 0.95)),
        "command_throttle_p05_p95": np.quantile(
            commands_np[:, 1], (0.05, 0.95)).tolist(),
        "errors": errors,
    }


def diagnose(checkpoint_path: Path, dataset_path: Path,
             run_ids: list[str], output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("provide unique held-out run IDs")
    torch, nn = _torch()
    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    metadata = checkpoint["metadata"]
    data = _load_dataset(dataset_path)
    if (data["schema_version"] < 7
            or data["simulator_pose_xyyaw"] is None
            or data["simulator_rigid_state"] is None
            or not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1e-7)):
        raise ValueError("requires schema-7+ rigid state, pose and 25 ms data")
    if data["feature_names"] != metadata["feature_names"]:
        raise ValueError("checkpoint and dataset feature layouts differ")
    lookup = {str(run_id): index for index, run_id in enumerate(data["run_ids"])}
    missing = sorted(set(run_ids) - set(lookup))
    if missing:
        raise ValueError(f"selected run IDs are missing: {missing}")
    overlap = sorted(set(run_ids).intersection(map(str, metadata["training_runs"])))
    if overlap:
        raise ValueError(f"refusing to inspect checkpoint training runs: {overlap}")
    invalid_split = {
        run_id: data["splits"][lookup[run_id]] for run_id in run_ids
        if data["splits"][lookup[run_id]] not in (
            "validation", "test", "final_test", "unseen_practice")
    }
    if invalid_split:
        raise ValueError(f"selected captures are not held out: {invalid_split}")

    model_type = _physical_model(
        torch, nn, bool(metadata["tire_relaxation_state_enabled"]))
    model = model_type()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    physical_state, _ = _targets(data)
    rollouts = []
    run_summaries = {}
    for run_id in run_ids:
        run_index = lookup[run_id]
        rows = _sequence_rows(data, run_index, round(0.25 / DT_S))
        if not rows:
            raise ValueError(f"{run_id}: no contiguous sequences to score")
        selected = [
            _load_rollout(model, data, physical_state, start, end, run_id,
                          condition, torch, torch.device("cpu"))
            for start, end, condition in rows
        ]
        rollouts.extend(selected)
        shared_steps = min(len(row["errors"]["u_abs_mps"])
                           for row in selected)
        metrics = sorted(set.intersection(*[
            set(row["errors"]) for row in selected
        ]))
        curves = {}
        for name in metrics:
            matrix = np.stack([
                np.asarray(row["errors"][name][:shared_steps], dtype=np.float64)
                for row in selected
            ])
            # State metrics other than position/heading are signed residuals.
            abs_matrix = np.abs(matrix)
            curve = np.sqrt(np.mean(matrix * matrix, axis=0))
            first_rmse = float(curve[0])
            threshold = THRESHOLDS.get(name)
            absolute_crossing = (None if threshold is None else
                                 next((int(index + 1) for index, value in
                                       enumerate(curve) if value >= threshold),
                                      None))
            doubling_crossing = next((int(index + 1) for index, value in
                                      enumerate(curve)
                                      if value >= max(2.0 * first_rmse, 1e-9)),
                                     None)
            curves[name] = {
                "macro_sequence_rmse_first_step": first_rmse,
                "macro_sequence_rmse_at_100ms": float(curve[min(3, shared_steps - 1)]),
                "macro_sequence_rmse_at_500ms": float(curve[min(19, shared_steps - 1)]),
                "macro_sequence_rmse_at_1s": float(curve[min(39, shared_steps - 1)]),
                "macro_sequence_rmse_at_2s": float(curve[min(79, shared_steps - 1)]),
                "macro_sequence_rmse_at_5s": (float(curve[199])
                                               if shared_steps >= 200 else None),
                "macro_sequence_rmse_at_shared_end": float(curve[-1]),
                "shared_end_s": shared_steps * DT_S,
                "first_crossing_absolute_diagnostic_threshold_step": absolute_crossing,
                "first_crossing_absolute_diagnostic_threshold_s": (
                    None if absolute_crossing is None
                    else absolute_crossing * DT_S),
                "diagnostic_threshold": threshold,
                "first_crossing_2x_first_step_rmse_s": (
                    None if doubling_crossing is None
                    else doubling_crossing * DT_S),
                "signed_macro_bias_at_first_100ms_500ms_1s_2s": {
                    f"{time:g}s": float(np.mean(matrix[:, step - 1]))
                    for time, step in ((0.025, 1), (0.100, 4),
                                       (0.500, 20), (1.0, 40), (2.0, 80))
                    if step <= shared_steps
                },
                "macro_sequence_absolute_error_at_first_100ms_500ms_1s_2s": {
                    f"{time:g}s": float(np.mean(abs_matrix[:, step - 1]))
                    for time, step in ((0.025, 1), (0.100, 4),
                                       (0.500, 20), (1.0, 40), (2.0, 80))
                    if step <= shared_steps
                },
            }
        run_summaries[run_id] = {
            "split": data["splits"][run_index],
            "sequence_count": len(selected),
            "shared_curve_steps": shared_steps,
            "shared_curve_seconds": shared_steps * DT_S,
            "error_curves": curves,
            "sequences": [{
                "condition": row["condition"],
                "initial_speed_mps": row["initial_speed_mps"],
                "initial_steering_feedback_rad": row[
                    "initial_steering_feedback_rad"],
                "initial_throttle_feedback": row["initial_throttle_feedback"],
                "command_steering_abs_p95_rad": row[
                    "command_steering_abs_p95_rad"],
                "command_throttle_p05_p95": row["command_throttle_p05_p95"],
            } for row in selected],
        }

    report = {
        "schema_version": 1,
        "purpose": "locate first command-only recursive plant divergence by state on untouched practice runs",
        "future_truth_or_sensor_inputs": False,
        "initialization": "simulator rigid state and pose plus causal sensors at one 1.975 s context endpoint; simulator truth thereafter is score-only",
        "context_seconds": (HISTORY_STEPS - 1) * DT_S,
        "prediction_inputs_after_initialization": [
            "steering_command_rad", "throttle_command_norm"],
        "statistical_unit": "two independent whole practice captures; lap/condition sequences are repeated measures within each capture",
        "diagnostic_thresholds_are_acceptance_gates": False,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_training_runs": metadata["training_runs"],
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "run_level_summaries": run_summaries,
        "sequence_conditions": [{
            "run_id": row["run_id"],
            "condition": row["condition"],
            "initial_speed_mps": row["initial_speed_mps"],
            "initial_steering_feedback_rad": row[
                "initial_steering_feedback_rad"],
            "initial_throttle_feedback": row["initial_throttle_feedback"],
            "command_steering_abs_p95_rad": row[
                "command_steering_abs_p95_rad"],
            "command_throttle_p05_p95": row["command_throttle_p05_p95"],
        } for row in rollouts],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run-id", action="append", dest="run_ids",
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.checkpoint, args.dataset, args.run_ids, args.output)
    print(json.dumps({
        "output": str(args.output),
        "runs": {
            run_id: {
                "sequence_count": row["sequence_count"],
                "errors": {
                    name: {
                        "rmse_25ms": curve["macro_sequence_rmse_first_step"],
                        "rmse_100ms": curve["macro_sequence_rmse_at_100ms"],
                        "rmse_500ms": curve["macro_sequence_rmse_at_500ms"],
                        "rmse_1s": curve["macro_sequence_rmse_at_1s"],
                        "rmse_2s": curve["macro_sequence_rmse_at_2s"],
                        "first_threshold_crossing_s": curve[
                            "first_crossing_absolute_diagnostic_threshold_s"],
                    } for name, curve in row["error_curves"].items()
                },
            } for run_id, row in report["run_level_summaries"].items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

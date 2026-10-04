#!/usr/bin/env python3
"""Compare held-out one-step errors by operating region for frozen plants.

This diagnostic uses every eligible 25 ms start in the frozen validation runs.
The measured current history/state initialize each one-step prediction; labels
are used only for scoring. It does not train, launch the simulator, or change
production code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    PoseIntegrator,
)
from tools.vehicle_dynamics_learning.evaluate_rigid_acceleration_highsteer_3run_transfer import (
    DEFAULT_CANDIDATE,
    DEFAULT_CONTROL,
    _training_context,
)
from tools.vehicle_dynamics_learning.operating_regions import region_masks
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    _batch_arrays,
    _torch_norm,
)
from tools.vehicle_dynamics_learning.run_rigid_acceleration_highsteer_transfer import (
    _load_acceleration_model,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import (
    FROZEN_EVAL_STARTS,
    ROOT,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _write_json,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    _collect_horizon_windows,
    sha256_file,
)


TASK_ROOT = DEFAULT_CANDIDATE.parent.parent
DEFAULT_OUTPUT = TASK_ROOT / "rigid_acceleration_one_step_regions_20261004.json"
CHUNK_SIZE = 512
BOOTSTRAP_REPLICATES = 5000
SEED = 20261046
REGION_MASK_NAMES = (
    "7_to_9mps_high_steering",
    "high_speed_near_straight",
    "high_speed_moderate_steering",
    "simultaneous_steering_throttle_transition",
    "large_wheel_body_mismatch",
    "steering_turn_in",
    "steering_unwind",
    "throttle_pickup",
    "braking_release",
    "negative_command_braking",
    "near_zero_throttle_command",
    "low_positive_throttle_command",
    "ordinary_racing_throttle_command",
    "high_throttle_command",
    "R0",
    "R1",
    "R2",
    "R3",
    "T0",
    "T1",
    "T2",
    "T3",
    "M0",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
)
ERROR_COLUMNS = (
    "u_mps",
    "v_mps",
    "yaw_rate_rps",
    "steering_feedback_rad",
    "throttle_feedback_norm",
    "position_x_m",
    "position_y_m",
    "heading_rad",
    "acceleration_x_mps2",
    "acceleration_y_mps2",
    "yaw_acceleration_rps2",
)
SCORED_METRICS = (
    "position_radial_rmse_m",
    "heading_rmse_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
    "steering_feedback_rmse_rad",
    "throttle_feedback_rmse",
    "acceleration_x_rmse_mps2",
    "acceleration_y_rmse_mps2",
    "yaw_acceleration_rmse_rps2",
)
BIAS_COLUMNS = {
    "u_bias_mps": 0,
    "v_bias_mps": 1,
    "yaw_rate_bias_rps": 2,
    "position_x_bias_m": 5,
    "position_y_bias_m": 6,
    "heading_bias_rad": 7,
    "acceleration_x_bias_mps2": 8,
    "acceleration_y_bias_mps2": 9,
    "yaw_acceleration_bias_rps2": 10,
}


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _regions_for_capture(capture) -> dict[str, np.ndarray]:
    reset_rows = np.full(len(capture.frames), -1, dtype=np.int32)
    for sequence_index, (begin, end) in enumerate(capture.bounds):
        reset_rows[int(begin):int(end)] = capture.sequence_reset[sequence_index]
    valid = (
        np.isfinite(capture.frames).all(axis=1)
        & np.isfinite(capture.body).all(axis=1)
        & np.isclose(capture.dt_s, DT_S, rtol=0.0, atol=1e-7)
    )
    masks = region_masks(capture.frames, reset_rows, capture.packet, valid, DT_S)
    speed = np.hypot(capture.body[:, 0], capture.body[:, 1])
    steering = capture.frames[:, 3]
    masks.update({
        "speed_0_to_4mps": (speed >= 0.0) & (speed <= 4.0),
        "speed_4_to_6mps": (speed > 4.0) & (speed <= 6.0),
        "speed_6_to_8mps": (speed > 6.0) & (speed <= 8.0),
        "speed_8_to_10mps": (speed > 8.0) & (speed <= 10.0),
        "speed_10_to_12mps": (speed > 10.0) & (speed <= 12.0),
        "steering_abs_0_to_0p10rad": (np.abs(steering) <= 0.10),
        "steering_abs_0p10_to_0p30rad": (
            (np.abs(steering) > 0.10) & (np.abs(steering) <= 0.30)),
        "steering_abs_0p30_to_0p40rad": (
            (np.abs(steering) > 0.30) & (np.abs(steering) <= 0.40)),
        "steering_abs_0p40_to_0p524rad": (
            (np.abs(steering) > 0.40) & (np.abs(steering) <= 0.524)),
        "steering_left": steering > 0.0,
        "steering_right": steering < 0.0,
        "steering_zero": steering == 0.0,
    })
    return masks


def _predict_one_step_errors(data, refs, model, norm_np, config,
                             device: torch.device) -> np.ndarray:
    norm = _torch_norm(norm_np, config, device)
    integrator = PoseIntegrator(DT_S).to(device)
    chunks = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(refs), CHUNK_SIZE):
            batch_refs = refs[offset:offset + CHUNK_SIZE]
            arrays = _batch_arrays(
                data, batch_refs, 1, CONTEXT_STEPS["2.0s"],
                norm_np, config, device)
            history, mask, state, pose, commands, target, target_pose = arrays
            delta, predicted_acceleration = model.forward_with_acceleration(
                history, mask, state, commands[:, 0])
            predicted_state = state + delta
            current_body = (state[:, :3] * norm["state_scale"][:3]
                            + norm["state_mean"][:3])
            following_body = (
                predicted_state[:, :3] * norm["state_scale"][:3]
                + norm["state_mean"][:3])
            predicted_pose = integrator(
                pose, 0.5 * (current_body + following_body))
            true_pose = target_pose[:, 0]
            true_acceleration = model.midpoint_acceleration_label(
                state, target[:, 0])
            state_error = (predicted_state - target[:, 0]) \
                * norm["state_scale"]
            pose_error = predicted_pose - true_pose
            pose_error = torch.cat((pose_error[:, :2], torch.atan2(
                torch.sin(pose_error[:, 2:3]),
                torch.cos(pose_error[:, 2:3]))), dim=-1)
            acceleration_error = predicted_acceleration - true_acceleration
            errors = torch.cat((
                state_error,
                pose_error,
                acceleration_error,
            ), dim=-1)
            chunks.append(errors.cpu().numpy())
    result = np.concatenate(chunks, axis=0)
    if result.shape != (len(refs), len(ERROR_COLUMNS)) \
            or not np.isfinite(result).all():
        raise FloatingPointError("one-step error output is invalid")
    return result


def _metric_rmse(errors: np.ndarray) -> dict[str, float]:
    mean_square = np.mean(np.square(errors), axis=0)
    return {
        "position_radial_rmse_m": float(np.sqrt(mean_square[5] + mean_square[6])),
        "heading_rmse_rad": float(np.sqrt(mean_square[7])),
        "u_rmse_mps": float(np.sqrt(mean_square[0])),
        "v_rmse_mps": float(np.sqrt(mean_square[1])),
        "yaw_rate_rmse_rps": float(np.sqrt(mean_square[2])),
        "steering_feedback_rmse_rad": float(np.sqrt(mean_square[3])),
        "throttle_feedback_rmse": float(np.sqrt(mean_square[4])),
        "acceleration_x_rmse_mps2": float(np.sqrt(mean_square[8])),
        "acceleration_y_rmse_mps2": float(np.sqrt(mean_square[9])),
        "yaw_acceleration_rmse_rps2": float(np.sqrt(mean_square[10])),
    }


def _metric_bias(errors: np.ndarray) -> dict[str, float]:
    return {name: float(errors[:, index].mean())
            for name, index in BIAS_COLUMNS.items()}


def _paired_run_delta(candidate: dict[str, dict[str, float]],
                      control: dict[str, dict[str, float]],
                      metric: str, seed: int) -> dict[str, Any]:
    common = sorted(set(candidate) & set(control))
    deltas = np.asarray(
        [candidate[run][metric] - control[run][metric] for run in common],
        dtype=np.float64)
    interval = None
    if len(deltas) >= 2:
        rng = np.random.default_rng(seed)
        indexes = rng.integers(0, len(deltas),
                               size=(BOOTSTRAP_REPLICATES, len(deltas)))
        interval = [float(value) for value in np.quantile(
            deltas[indexes].mean(axis=1), [0.025, 0.975])]
    return {
        "candidate_minus_control_macro_run_delta": float(deltas.mean()),
        "run_cluster_bootstrap_95pct_ci": interval,
        "independent_runs": len(common),
        "per_run_delta": dict(zip(common, deltas.tolist())),
    }


def run(device_name: str = "cpu", output_path: Path = DEFAULT_OUTPUT,
        candidate_path: Path = DEFAULT_CANDIDATE,
        control_path: Path = DEFAULT_CONTROL) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)

    control_data, control_config, control_norm = _training_context(False, device)
    candidate_data, candidate_config, candidate_norm = _training_context(True, device)
    candidate, candidate_sha = _load_acceleration_model(
        candidate_path, candidate_norm, device)
    control, control_sha = _load_acceleration_model(
        control_path, control_norm, device)

    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    expected_runs = set(frozen["split_roles"]["development_validation"])
    all_refs = _collect_horizon_windows(
        control_data.captures, control_data.raw_sources,
        control_data.poses, {"validation"}, 1, {0})
    if not expected_runs.issubset(all_refs):
        raise RuntimeError(
            "one-step validation lacks frozen runs: "
            f"{sorted(expected_runs - set(all_refs))}")
    all_refs = {run_id: all_refs[run_id] for run_id in sorted(expected_runs)}
    if (expected_runs & set(control_data.training_runs)
            or expected_runs & set(candidate_data.training_runs)):
        raise RuntimeError("one-step validation runs overlap a training split")

    capture = control_data.captures[0]
    masks = _regions_for_capture(capture)
    selected_regions = tuple(REGION_MASK_NAMES) + tuple(
        name for name in masks
        if name.startswith("speed_") or name.startswith("steering_abs_"))
    candidate_per_run: dict[str, dict[str, dict[str, float]]] = {}
    control_per_run: dict[str, dict[str, dict[str, float]]] = {}
    candidate_bias_per_run: dict[str, dict[str, dict[str, float]]] = {}
    control_bias_per_run: dict[str, dict[str, dict[str, float]]] = {}
    region_counts: dict[str, dict[str, int]] = {}
    start_counts: dict[str, int] = {}

    for run_index, run_id in enumerate(sorted(expected_runs)):
        refs = all_refs[run_id]
        indexes = np.asarray([
            int(capture.bounds[sequence_index, 0]) + int(source_row)
            for _, sequence_index, source_row in refs], dtype=np.int64)
        candidate_errors = _predict_one_step_errors(
            candidate_data, refs, candidate, candidate_norm,
            candidate_config, device)
        control_errors = _predict_one_step_errors(
            control_data, refs, control, control_norm,
            control_config, device)
        candidate_per_run[run_id] = {}
        control_per_run[run_id] = {}
        candidate_bias_per_run[run_id] = {}
        control_bias_per_run[run_id] = {}
        region_counts[run_id] = {}
        start_counts[run_id] = len(refs)
        for region in selected_regions:
            if region not in masks:
                continue
            selected = np.asarray(masks[region][indexes], dtype=bool)
            count = int(selected.sum())
            region_counts[run_id][region] = count
            if not count:
                continue
            candidate_per_run[run_id][region] = _metric_rmse(
                candidate_errors[selected])
            control_per_run[run_id][region] = _metric_rmse(
                control_errors[selected])
            candidate_bias_per_run[run_id][region] = _metric_bias(
                candidate_errors[selected])
            control_bias_per_run[run_id][region] = _metric_bias(
                control_errors[selected])

        candidate_per_run[run_id]["all_validation_starts"] = _metric_rmse(
            candidate_errors)
        control_per_run[run_id]["all_validation_starts"] = _metric_rmse(
            control_errors)
        candidate_bias_per_run[run_id]["all_validation_starts"] = _metric_bias(
            candidate_errors)
        control_bias_per_run[run_id]["all_validation_starts"] = _metric_bias(
            control_errors)
        if run_index % 2 == 1:
            print(f"Scored all one-step starts for {run_index + 1}/"
                  f"{len(expected_runs)} validation runs", flush=True)

    paired: dict[str, dict[str, Any]] = {}
    paired_bias: dict[str, dict[str, Any]] = {}
    region_names = sorted({
        region for run_metrics in candidate_per_run.values()
        for region in run_metrics})
    for region_index, region in enumerate(region_names):
        paired[region] = {
            metric: _paired_run_delta(
                {run_id: values[region] for run_id, values in candidate_per_run.items()
                 if region in values},
                {run_id: values[region] for run_id, values in control_per_run.items()
                 if region in values},
                metric, SEED + region_index * len(SCORED_METRICS) + metric_index)
            for metric_index, metric in enumerate(SCORED_METRICS)}
        paired_bias[region] = {
            metric: _paired_run_delta(
                {run_id: values[region] for run_id, values in candidate_bias_per_run.items()
                 if region in values},
                {run_id: values[region] for run_id, values in control_bias_per_run.items()
                 if region in values},
                metric, SEED + 1000 + region_index * len(BIAS_COLUMNS) + metric_index)
            for metric_index, metric in enumerate(BIAS_COLUMNS)}

    report = {
        "study": "all eligible teacher-forced 25 ms one-step errors by operating region",
        "candidate_checkpoint_sha256": candidate_sha,
        "matched_control_checkpoint_sha256": control_sha,
        "candidate_training_dataset_sha256": sha256_file(
            Path(__file__).resolve().parents[2]
            / "live_runs/derived_dynamics_learning_20260928"
            / "full_modeling_reset_20261001/highsteer_75_train_expanded_20261004"
            / "openplane_dynamics.npz"),
        "horizon_seconds": DT_S,
        "context_seconds": CONTEXT_STEPS["2.0s"] * DT_S,
        "independent_unit": "whole capture run; all rows within a run are summarized before bootstrap",
        "validation_runs": sorted(expected_runs),
        "start_count_by_run": start_counts,
        "region_start_count_by_run": region_counts,
        "candidate_per_run": candidate_per_run,
        "matched_control_per_run": control_per_run,
        "candidate_signed_error_bias_per_run": candidate_bias_per_run,
        "matched_control_signed_error_bias_per_run": control_bias_per_run,
        "paired_candidate_minus_control_run_cluster_ci": paired,
        "paired_candidate_minus_control_signed_bias_run_cluster_ci": paired_bias,
        "future_measurements_or_truth_used_as_inputs": False,
        "test_or_final_test_opened": False,
        "simulator_launched": False,
        "production_or_mpc_integration": False,
        "limits": [
            "This is a teacher-forced local transition diagnostic, not recursive simulation accuracy.",
            "All eligible starts are correlated rows; uncertainty is estimated across six whole validation runs.",
            "The high-steering training addition has only two independent training runs, so regional transfer still needs unseen-run confirmation.",
        ],
    }
    _write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    args = parser.parse_args()
    paths = tuple(path if path.is_absolute() else ROOT / path
                  for path in (args.output, args.candidate, args.control))
    result = run(args.device, *paths)
    print(json.dumps({
        "output": paths[0].relative_to(ROOT).as_posix(),
        "starts_per_run": result["start_count_by_run"],
        "all_region_start_counts": result["region_start_count_by_run"],
        "all_validation_delta": result[
            "paired_candidate_minus_control_run_cluster_ci"
                ]["all_validation_starts"],
        "transition_delta": result[
            "paired_candidate_minus_control_run_cluster_ci"].get(
                "simultaneous_steering_throttle_transition"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

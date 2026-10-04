#!/usr/bin/env python3
"""Evaluate the rigid-acceleration teacher on complete held-out practice runs.

The initial measured state, pose, and 2 s history are used once per continuous
run. Every later transition receives only its recorded command and recursively
predicted state/history. A separate isolated-lap score reinitializes only at a
lap boundary using that boundary's measured state and prior history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    PoseIntegrator,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    FixedCapacityHistoryTransition,
    _normalization,
    _rollout,
    _torch_norm,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _state,
    _write_json,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    _load_data,
    _training_windows_and_stats,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
            / "full_modeling_reset_20261001"
            / "replacement_offline_sim_raceline_20261003"
            / "full_throttle_domain_v1/next_phase_after_2129427"
            / "history_context_sufficiency_v1")
CANDIDATE_CHECKPOINT = (
    TASK_ROOT / "rigid_acceleration_history_direct_supervision_10s_v1"
    / "checkpoint.pt")
FIVE_SECOND_CHECKPOINT = (
    TASK_ROOT / "rigid_acceleration_history_direct_supervision_5s_v1"
    / "checkpoint.pt")
REFERENCE_CHECKPOINT = (
    TASK_ROOT / "long_rollout_5s_selected_context_v1/checkpoint.pt")
OUTPUT = (CANDIDATE_CHECKPOINT.parent
          / "full_practice_5s_vs_10s_recursive_comparison_20261004_v1.json")
EXPECTED_CANDIDATE_SHA256 = (
    "8f5dfb88c84ba554fac90e9d66a13dc9507eca184aca7695ab8592ab9229ea7d")
EXPECTED_FIVE_SECOND_SHA256 = (
    "474324d8f556c3189947f6ae307105b259f1b3122cf4660d2a0355c5ec165c1b")
EXPECTED_RUNS = (
    "practice_unseen_model_validation_20261001_r02",
    "practice_unseen_model_validation_20261001_r03",
)
CONTEXT_STEPS_USED = CONTEXT_STEPS["2.0s"]
POSITION_THRESHOLDS_M = (0.25, 0.5, 1.0)


def _continuous_run(capture: Any, run_id: str) -> tuple[int, int, int]:
    matches = np.flatnonzero(capture.run_ids.astype(str) == run_id)
    if matches.size != 1:
        raise ValueError(f"expected one practice run id, got {run_id}")
    run_index = int(matches[0])
    sequences = np.flatnonzero(capture.sequence_run == run_index)
    if not sequences.size:
        raise ValueError(f"practice run has no sequence bounds: {run_id}")
    ordered = sorted((int(capture.bounds[i, 0]), int(capture.bounds[i, 1]),
                      int(i)) for i in sequences)
    if any(left[1] != right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"practice run has a sequence gap: {run_id}")
    sequence_ids = [item[2] for item in ordered]
    if len({int(capture.sequence_reset[i]) for i in sequence_ids}) != 1:
        raise ValueError(f"practice run crosses a simulator reset: {run_id}")
    begin, end = ordered[0][0], ordered[-1][1]
    packet = np.asarray(capture.packet[begin:end])
    if (end - begin <= CONTEXT_STEPS_USED + 1
            or packet.size != end - begin
            or not np.all(np.diff(packet) == 1)
            or not np.allclose(capture.dt_s[begin:end], DT_S,
                               rtol=0.0, atol=1e-7)):
        raise ValueError(f"practice run is not fixed-cadence/continuous: {run_id}")
    return run_index, begin, end


def _arrays(data: Any, capture_index: int, run_begin: int, start: int,
            end: int, norm_np: dict[str, np.ndarray], config: Any,
            device: torch.device) -> tuple[torch.Tensor, ...]:
    capture = data.captures[capture_index]
    context = CONTEXT_STEPS_USED
    horizon = end - start - 1
    history_start = start - context + 1
    if (history_start < run_begin or horizon < 1
            or end > len(capture.frames)):
        raise ValueError("replay segment lacks in-run context or future")
    raw_history = np.asarray(
        capture.input_features[history_start:start + 1, :7], dtype=np.float32)
    if raw_history.shape != (context, 7) or not np.isfinite(raw_history).all():
        raise ValueError("initial history is incomplete or non-finite")
    history = np.zeros((1, 160, 7), dtype=np.float32)
    history[0, -context:] = (
        raw_history - config.history_mean[:7]
    ) / config.history_scale[:7]
    mask = np.zeros((1, 160), dtype=np.float32)
    mask[0, -context:] = 1.0

    states = np.asarray([_state(capture, row)
                         for row in range(start, end)], dtype=np.float32)
    commands = np.asarray(capture.frames[start:end - 1, 7:9],
                          dtype=np.float32)
    target_pose = np.asarray(data.poses[capture_index][start + 1:end],
                             dtype=np.float32)
    initial_pose = np.asarray(data.poses[capture_index][start:start + 1],
                              dtype=np.float32)
    if (not np.isfinite(states).all() or not np.isfinite(commands).all()
            or not np.isfinite(target_pose).all()
            or not np.isfinite(initial_pose).all()):
        raise ValueError("practice replay contains non-finite state/command/pose")
    state = ((states[:1] - norm_np["state_mean"])
             / norm_np["state_scale"])
    command = ((commands - norm_np["command_mean"])
               / norm_np["command_scale"])[None]
    target = ((states[1:] - norm_np["state_mean"])
              / norm_np["state_scale"])[None]
    target_pose = target_pose[None]
    values = (history, mask, state.astype(np.float32), initial_pose,
              command.astype(np.float32), target.astype(np.float32),
              target_pose)
    return tuple(torch.as_tensor(value, dtype=torch.float32, device=device)
                 for value in values)


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _metrics(pred_state: np.ndarray, pred_pose: np.ndarray,
             truth_state: np.ndarray, truth_pose: np.ndarray) -> dict[str, Any]:
    position_error = pred_pose[:, :2] - truth_pose[:, :2]
    radial = np.linalg.norm(position_error, axis=1)
    heading = _wrap(pred_pose[:, 2] - truth_pose[:, 2])
    state_error = pred_state - truth_state
    speed_error = (np.hypot(pred_state[:, 0], pred_state[:, 1])
                   - np.hypot(truth_state[:, 0], truth_state[:, 1]))
    return {
        "sample_count": int(len(radial)),
        "duration_s": float(len(radial) * DT_S),
        "position_radial_rmse_m": float(np.sqrt(np.mean(radial ** 2))),
        "position_endpoint_m": float(radial[-1]),
        "position_p95_m": float(np.quantile(radial, 0.95)),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading ** 2))),
        "heading_endpoint_abs_rad": float(abs(heading[-1])),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_error ** 2))),
        "speed_bias_mps": float(np.mean(speed_error)),
        "forward_speed_rmse_mps": float(np.sqrt(np.mean(state_error[:, 0] ** 2))),
        "forward_speed_bias_mps": float(np.mean(state_error[:, 0])),
        "lateral_speed_rmse_mps": float(np.sqrt(np.mean(state_error[:, 1] ** 2))),
        "lateral_speed_bias_mps": float(np.mean(state_error[:, 1])),
        "yaw_rate_rmse_rps": float(np.sqrt(np.mean(state_error[:, 2] ** 2))),
        "yaw_rate_bias_rps": float(np.mean(state_error[:, 2])),
        "steering_feedback_rmse_rad": float(np.sqrt(np.mean(state_error[:, 3] ** 2))),
        "throttle_feedback_rmse": float(np.sqrt(np.mean(state_error[:, 4] ** 2))),
        "wheel_speed_metrics": None,
        "wheel_speed_note": (
            "This five-state candidate and its WP28 comparator do not predict "
            "wheel speeds; no wheel accuracy is claimed."),
    }


def _first_threshold_crossings(pred_pose: np.ndarray,
                               truth_pose: np.ndarray) -> dict[str, Any]:
    error = np.linalg.norm(pred_pose[:, :2] - truth_pose[:, :2], axis=1)
    result = {}
    for threshold in POSITION_THRESHOLDS_M:
        indices = np.flatnonzero(error >= threshold)
        result[f"position_error_ge_{threshold:g}m"] = (
            None if not indices.size else {
                "step": int(indices[0] + 1),
                "time_s": float((indices[0] + 1) * DT_S),
                "error_m": float(error[indices[0]]),
            })
    return result


def _regime_metrics(capture: Any, rows: np.ndarray,
                    pred_state: np.ndarray, pred_pose: np.ndarray,
                    truth_state: np.ndarray, truth_pose: np.ndarray
                    ) -> dict[str, Any]:
    """Attribute rollout residuals to recorded operating regions (labels only)."""
    rows = np.asarray(rows, dtype=np.int64)
    steering = capture.frames[rows, 3].astype(np.float64)
    throttle_command = capture.frames[rows, 8].astype(np.float64)
    steering_rate = np.zeros(len(rows), dtype=np.float64)
    throttle_slew = np.zeros(len(rows), dtype=np.float64)
    previous_rows = rows - 1
    contiguous = previous_rows >= 0
    steering_rate[contiguous] = (
        steering[contiguous] - capture.frames[previous_rows[contiguous], 3]
    ) / DT_S
    throttle_slew[contiguous] = (
        throttle_command[contiguous]
        - capture.frames[previous_rows[contiguous], 8]
    ) / DT_S
    true_speed = np.hypot(truth_state[:, 0], truth_state[:, 1])
    wheel_values = np.asarray(capture.encoder_rate[rows], dtype=np.float64)
    wheel_valid = (np.asarray(capture.encoder_valid[rows], dtype=bool)
                   & np.isfinite(wheel_values).all(axis=1))
    wheel_body_mismatch = np.full(len(rows), np.nan, dtype=np.float64)
    wheel_body_mismatch[wheel_valid] = np.abs(
        np.mean(np.abs(wheel_values[wheel_valid]), axis=1)
        - np.abs(truth_state[wheel_valid, 0]))

    masks: dict[str, np.ndarray] = {}

    def bands(name: str, values: np.ndarray,
              definitions: tuple[tuple[str, float, float], ...]) -> None:
        for label, lower, upper in definitions:
            masks[f"{name}:{label}"] = (
                np.isfinite(values) & (values >= lower) & (values < upper))

    inf = float("inf")
    bands("speed_mps", true_speed,
          (("0_to_3", 0.0, 3.0), ("3_to_6", 3.0, 6.0),
           ("6_to_9", 6.0, 9.0), ("9_to_12", 9.0, 12.0),
           ("12_plus", 12.0, inf)))
    bands("abs_steering_rad", np.abs(steering),
          (("below_0p1", 0.0, 0.1), ("0p1_to_0p2", 0.1, 0.2),
           ("0p2_to_0p3", 0.2, 0.3), ("0p3_plus", 0.3, inf)))
    masks["steering_direction:left_positive"] = steering > 0.02
    masks["steering_direction:right_negative"] = steering < -0.02
    masks["steering_direction:near_zero"] = np.abs(steering) <= 0.02
    bands("abs_steering_rate_rps", np.abs(steering_rate),
          (("below_0p1", 0.0, 0.1), ("0p1_to_0p5", 0.1, 0.5),
           ("0p5_plus", 0.5, inf)))
    bands("throttle_command", throttle_command,
          (("below_0p2", -inf, 0.2), ("0p2_to_0p5", 0.2, 0.5),
           ("0p5_to_0p8", 0.5, 0.8), ("0p8_plus", 0.8, inf)))
    bands("abs_throttle_slew_per_s", np.abs(throttle_slew),
          (("below_0p5", 0.0, 0.5), ("0p5_to_5", 0.5, 5.0),
           ("5_plus", 5.0, inf)))
    bands("wheel_body_speed_mismatch_mps", wheel_body_mismatch,
          (("below_0p25", 0.0, 0.25), ("0p25_to_0p75", 0.25, 0.75),
           ("0p75_plus", 0.75, inf)))

    position_error = pred_pose[:, :2] - truth_pose[:, :2]
    heading_error = _wrap(pred_pose[:, 2] - truth_pose[:, 2])
    state_error = pred_state - truth_state
    result = {}
    for name, mask in masks.items():
        selected = np.flatnonzero(mask)
        if len(selected) < 20:
            continue
        error = state_error[selected]
        radial = np.linalg.norm(position_error[selected], axis=1)
        result[name] = {
            "sample_count": int(len(selected)),
            "position_radial_rmse_m": float(np.sqrt(np.mean(radial ** 2))),
            "heading_rmse_rad": float(np.sqrt(np.mean(heading_error[selected] ** 2))),
            "forward_speed_bias_mps": float(np.mean(error[:, 0])),
            "forward_speed_rmse_mps": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "lateral_speed_bias_mps": float(np.mean(error[:, 1])),
            "lateral_speed_rmse_mps": float(np.sqrt(np.mean(error[:, 1] ** 2))),
            "yaw_rate_bias_rps": float(np.mean(error[:, 2])),
            "yaw_rate_rmse_rps": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        }
    result["conditioning_note"] = (
        "Recorded truth/feedback/encoder values select offline scoring bins only; "
        "none are supplied to the free-running model. Wheel/body mismatch uses "
        "valid fixed-25-ms encoder surface-rate magnitude and truth forward speed.")
    return result


def _predict(model, arrays: tuple[torch.Tensor, ...], norm: dict[str, torch.Tensor],
             horizon: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        output = _rollout(model, arrays, norm, horizon, CONTEXT_STEPS_USED,
                          PoseIntegrator(DT_S).to(device))
    state = (output["state"].cpu().numpy()[0]
             * norm["state_scale"].cpu().numpy()
             + norm["state_mean"].cpu().numpy())
    pose = output["pose"].cpu().numpy()[0]
    if not np.isfinite(state).all() or not np.isfinite(pose).all():
        raise FloatingPointError("candidate produced a non-finite full-run rollout")
    return state.astype(np.float64), pose.astype(np.float64)


def _one_step_teacher_forced(data: Any, run_begin: int, run_end: int,
                             norm_np: dict[str, np.ndarray], config: Any,
                             norm: dict[str, torch.Tensor], models: dict,
                             device: torch.device) -> dict[str, Any]:
    """Score local transitions from measured past/current state, never future input."""
    capture = data.captures[1]
    source_rows = np.arange(run_begin + CONTEXT_STEPS_USED - 1,
                            run_end - 1, dtype=np.int64)
    target_rows = source_rows + 1
    context = CONTEXT_STEPS_USED
    predictions = {name: {"state": [], "pose": []} for name in models}
    truth_state_parts, truth_pose_parts = [], []
    batch_size = 128
    for offset in range(0, len(source_rows), batch_size):
        current_rows = source_rows[offset:offset + batch_size]
        next_rows = current_rows + 1
        raw_history = np.stack([
            capture.input_features[row - context + 1:row + 1, :7]
            for row in current_rows])
        if not np.isfinite(raw_history).all():
            raise ValueError("one-step practice history is non-finite")
        history = np.zeros((len(current_rows), 160, 7), dtype=np.float32)
        history[:, -context:] = (
            raw_history - config.history_mean[:7]
        ) / config.history_scale[:7]
        mask = np.zeros((len(current_rows), 160), dtype=np.float32)
        mask[:, -context:] = 1.0
        current_state = np.asarray([_state(capture, row)
                                    for row in current_rows], dtype=np.float32)
        next_state = np.asarray([_state(capture, row)
                                 for row in next_rows], dtype=np.float32)
        commands = capture.frames[current_rows, 7:9].astype(np.float32)
        poses = data.poses[1][current_rows].astype(np.float32)
        next_poses = data.poses[1][next_rows].astype(np.float32)
        arrays_np = (
            history, mask,
            ((current_state - norm_np["state_mean"])
             / norm_np["state_scale"]).astype(np.float32),
            poses,
            ((commands - norm_np["command_mean"])
             / norm_np["command_scale"])[:, None, :].astype(np.float32),
            ((next_state - norm_np["state_mean"])
             / norm_np["state_scale"])[:, None, :].astype(np.float32),
            next_poses[:, None, :],
        )
        arrays = tuple(torch.as_tensor(value, dtype=torch.float32,
                                       device=device) for value in arrays_np)
        truth_state_parts.append(next_state.astype(np.float64))
        truth_pose_parts.append(next_poses.astype(np.float64))
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                output = _rollout(
                    model, arrays, norm, 1, CONTEXT_STEPS_USED,
                    PoseIntegrator(DT_S).to(device))
            predicted_state = (
                output["state"].cpu().numpy()[:, 0]
                * norm_np["state_scale"] + norm_np["state_mean"])
            predicted_pose = output["pose"].cpu().numpy()[:, 0]
            if (not np.isfinite(predicted_state).all()
                    or not np.isfinite(predicted_pose).all()):
                raise FloatingPointError("one-step plant prediction is non-finite")
            predictions[name]["state"].append(predicted_state.astype(np.float64))
            predictions[name]["pose"].append(predicted_pose.astype(np.float64))

    truth_state = np.concatenate(truth_state_parts)
    truth_pose = np.concatenate(truth_pose_parts)
    target_rows = target_rows[:len(truth_state)]
    result = {}
    for name, values in predictions.items():
        state = np.concatenate(values["state"])
        pose = np.concatenate(values["pose"])
        result[name] = {
            "teacher_forced_one_step_metrics": _metrics(
                state, pose, truth_state, truth_pose),
            "error_by_recorded_operating_region": _regime_metrics(
                capture, target_rows, state, pose, truth_state, truth_pose),
            "input_contract": (
                "Each transition uses that row's measured past history, measured "
                "current state/pose, and current command. The next state/pose "
                "are labels only; no future observation enters model input."),
        }
    return result


def _load_models(norm_np: dict[str, np.ndarray], device: torch.device
                 ) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    def acceleration_model(path: Path):
        saved = torch.load(path, map_location=device, weights_only=True)
        metadata = saved["metadata"]
        model = RigidAccelerationHistoryTransition(
            norm_np["state_mean"], norm_np["state_scale"],
            np.asarray(metadata["acceleration_mean_train_only"],
                       dtype=np.float32),
            np.asarray(metadata["acceleration_scale_train_only"],
                       dtype=np.float32),
            norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
            dt_s=float(metadata["dt_s"]),
            rear_axle_to_com_x_m=float(metadata["rear_axle_to_com_x_m"]),
        ).to(device)
        model.load_state_dict(saved["state_dict"], strict=True)
        return model, metadata

    five_second, five_metadata = acceleration_model(FIVE_SECOND_CHECKPOINT)
    ten_second, ten_metadata = acceleration_model(CANDIDATE_CHECKPOINT)
    reference_saved = torch.load(REFERENCE_CHECKPOINT, map_location=device,
                                weights_only=True)
    reference = FixedCapacityHistoryTransition(
        norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    reference.load_state_dict(reference_saved["state_dict"], strict=True)
    return {"rigid_acceleration_5s": five_second,
            "rigid_acceleration_10s": ten_second,
            "WP28_selected_5s": reference}, {
                "rigid_acceleration_5s": five_metadata,
                "rigid_acceleration_10s": ten_metadata,
            }


def _run_report(data: Any, run_id: str, norm_np: dict[str, np.ndarray],
                config: Any, norm: dict[str, torch.Tensor], models: dict,
                device: torch.device, lap_count: np.ndarray
                ) -> dict[str, Any]:
    capture_index = 1
    capture = data.captures[capture_index]
    run_index, run_begin, run_end = _continuous_run(capture, run_id)
    if str(capture.splits[run_index]) != "unseen_practice":
        raise ValueError(f"run is not frozen unseen practice: {run_id}")
    initialization = run_begin + CONTEXT_STEPS_USED - 1
    full_arrays = _arrays(data, capture_index, run_begin, initialization,
                          run_end, norm_np, config, device)
    horizon = run_end - initialization - 1
    truth_state = full_arrays[5].cpu().numpy()[0] * norm_np["state_scale"] \
        + norm_np["state_mean"]
    truth_pose = full_arrays[6].cpu().numpy()[0]
    future_rows = np.arange(initialization + 1, run_end, dtype=np.int64)
    lap_targets = lap_count[future_rows]
    model_results: dict[str, Any] = {}
    for name, model in models.items():
        state, pose = _predict(model, full_arrays, norm, horizon, device)
        result = {
            "continuous_capture_metrics": _metrics(
                state, pose, truth_state, truth_pose),
            "first_position_divergence": _first_threshold_crossings(
                pose, truth_pose),
            "error_by_recorded_operating_region": _regime_metrics(
                capture, future_rows, state, pose, truth_state, truth_pose),
            "per_recorded_lap": {},
        }
        for lap_id in sorted(int(value) for value in np.unique(lap_targets)):
            indexes = np.flatnonzero(lap_targets == lap_id)
            if len(indexes) < 20:
                continue
            result["per_recorded_lap"][str(lap_id)] = _metrics(
                state[indexes], pose[indexes], truth_state[indexes],
                truth_pose[indexes])
        isolated_laps = []
        transitions = [row for row in range(run_begin + 1, run_end)
                       if lap_count[row] == lap_count[row - 1] + 1]
        for start, next_crossing in zip(transitions, transitions[1:]):
            if start - run_begin < CONTEXT_STEPS_USED - 1:
                continue
            isolated_arrays = _arrays(
                data, capture_index, run_begin, int(start),
                int(next_crossing) + 1, norm_np, config, device)
            lap_horizon = int(next_crossing) - int(start)
            lap_state, lap_pose = _predict(
                model, isolated_arrays, norm, lap_horizon, device)
            lap_truth_state = (isolated_arrays[5].cpu().numpy()[0]
                               * norm_np["state_scale"]
                               + norm_np["state_mean"])
            lap_truth_pose = isolated_arrays[6].cpu().numpy()[0]
            isolated_laps.append({
                "initial_lap_counter": int(lap_count[start]),
                "initial_frame": int(start),
                **_metrics(lap_state, lap_pose,
                           lap_truth_state, lap_truth_pose),
            })
        result["isolated_lap_replays"] = isolated_laps
        if isolated_laps:
            metric_names = ("position_radial_rmse_m", "position_endpoint_m",
                            "heading_rmse_rad", "forward_speed_rmse_mps",
                            "lateral_speed_rmse_mps", "yaw_rate_rmse_rps")
            result["isolated_lap_macro"] = {
                key: float(np.mean([lap[key] for lap in isolated_laps]))
                for key in metric_names}
        model_results[name] = result

    one_step_results = _one_step_teacher_forced(
        data, run_begin, run_end, norm_np, config, norm, models, device)

    paired_metrics = (
        "position_radial_rmse_m", "position_endpoint_m", "position_p95_m",
        "heading_rmse_rad", "forward_speed_rmse_mps",
        "lateral_speed_rmse_mps", "yaw_rate_rmse_rps",
    )
    reference_name = "WP28_selected_5s"
    paired_run_delta = {}
    for candidate_name in ("rigid_acceleration_5s", "rigid_acceleration_10s"):
        paired_run_delta[candidate_name] = {}
        for metric in paired_metrics:
            paired_run_delta[candidate_name][metric] = float(
                model_results[candidate_name]["continuous_capture_metrics"][metric]
                - model_results[reference_name]["continuous_capture_metrics"][metric])
    return {
        "run_id": run_id,
        "split": str(capture.splits[run_index]),
        "run_duration_s": float((run_end - run_begin - 1) * DT_S),
        "sample_count": int(run_end - run_begin),
        "joined_sequence_count": int(np.count_nonzero(
            capture.sequence_run == run_index)),
        "initialization_frame": int(initialization),
        "initialization_contract": (
            "Measured current state/pose and prior 80 samples initialize once; "
            "afterward only recorded commands and recursively predicted state/history."),
        "future_truth_or_sensor_feedback_used": False,
        "paired_candidate_minus_WP28_continuous_run_delta": paired_run_delta,
        "one_step_teacher_forced_diagnostics": one_step_results,
        "models": model_results,
    }


def evaluate(output: Path = OUTPUT, device_name: str = "cpu") -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    if sha256_file(CANDIDATE_CHECKPOINT) != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("candidate checkpoint hash differs from frozen report")
    if sha256_file(FIVE_SECOND_CHECKPOINT) != EXPECTED_FIVE_SECOND_SHA256:
        raise ValueError("5 s parent checkpoint hash differs from frozen report")
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    models, metadata = _load_models(norm_np, device)
    norm = _torch_norm(norm_np, config, device)
    capture = data.captures[1]
    with np.load(DEFAULT_PRACTICE, allow_pickle=False) as archive:
        if not np.array_equal(archive["packet_sequence"], capture.packet):
            raise ValueError("lap labels are not aligned with practice plant data")
        lap_count = np.asarray(archive["lap_count"], dtype=np.int32)
    reports = {}
    for run_id in EXPECTED_RUNS:
        reports[run_id] = _run_report(
            data, run_id, norm_np, config, norm, models, device, lap_count)
    for candidate_name, candidate_metadata in metadata.items():
        candidate_runs = set(candidate_metadata["training_run_ids"])
        if candidate_runs & set(EXPECTED_RUNS):
            raise ValueError(
                f"practice evaluation overlaps {candidate_name} training")
    report = {
        "study": "recursive complete-practice-run replay of explicit rigid-acceleration plant",
        "candidate_checkpoint_sha256": sha256_file(CANDIDATE_CHECKPOINT),
        "five_second_checkpoint_sha256": sha256_file(FIVE_SECOND_CHECKPOINT),
        "reference_checkpoint_sha256": sha256_file(REFERENCE_CHECKPOINT),
        "dataset_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "dynamic_fixed40hz": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed40hz": sha256_file(DEFAULT_PRACTICE_FIXED),
        },
        "sample_rate_hz": 40.0,
        "dt_s": DT_S,
        "context_steps": CONTEXT_STEPS_USED,
        "command_alignment": (
            "For a transition from state row k to k+1, use the command stored "
            "on source row k; this is offset -1 relative to the target-state row."),
        "candidate_training_runs": {
            name: values["training_run_ids"] for name, values in metadata.items()},
        "candidate_validation_runs": {
            name: values["validation_run_ids"] for name, values in metadata.items()},
        "practice_runs_are_held_out_from_candidate_training": True,
        "future_truth_or_sensor_feedback_used": False,
        "wheel_state_supported": False,
        "simulator_launched": False,
        "production_integration": False,
        "interpretation_limit": (
            "The two practice captures are held out from candidate fitting but are "
            "only two independent runs. Per-lap rows are repeated observations, "
            "not independent experimental units. Wheel dynamics are not predicted "
            "by either compared five-state model."),
        "results": reports,
    }
    _write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    started = time.monotonic()
    report = evaluate(output, args.device)
    compact = {
        run_id: {
            "duration_s": run["run_duration_s"],
            "lap_count": len(run["models"]["rigid_acceleration_10s"][
                "per_recorded_lap"]),
            "models": {
                name: {
                    "continuous": values["continuous_capture_metrics"],
                    "isolated_lap_macro": values.get("isolated_lap_macro"),
                }
                for name, values in run["models"].items()},
            "paired_deltas_minus_WP28": run[
                "paired_candidate_minus_WP28_continuous_run_delta"],
        }
        for run_id, run in report["results"].items()}
    print(json.dumps({
        "report": output.relative_to(ROOT).as_posix(),
        "elapsed_s": time.monotonic() - started,
        "results": compact,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

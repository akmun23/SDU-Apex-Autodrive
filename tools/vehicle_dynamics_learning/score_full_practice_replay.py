#!/usr/bin/env python3
"""Replay recorded practice commands through an EDSSM for complete captures.

Only the initial 80-frame measured history/pose initializes the plant. Every
later step receives the recorded command and the model's own predicted state;
truth is used strictly for scoring and for identifying lap/turn intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    HISTORY_STEPS,
    append_roll_state,
    integrate_pose,
    physical_state_from_dataset,
    raw_encoder_history_features,
    wheel_innovation_history_features,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = (ROOT / "live_runs/derived_dynamics_learning_20260928/"
                     "full_modeling_reset_20261001/practice_transfer_benchmark_v1.json")
DEFAULT_OUTPUT = (ROOT / "live_runs/derived_dynamics_learning_20260928/"
                  "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
                  "full_practice_replay_v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_indices(target_indices: np.ndarray,
                     command_offset_frames: int) -> np.ndarray:
    if command_offset_frames not in (-1, 0):
        raise ValueError("command alignment offset must be -1 or 0 frames")
    return np.asarray(target_indices, dtype=np.int64) + command_offset_frames


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _contiguous_run_indices(data: dict[str, Any], run_index: int) -> np.ndarray:
    sequence_ids = np.flatnonzero(data["seq_run"] == run_index)
    if sequence_ids.size == 0:
        raise ValueError("practice run has no sequence bounds")
    ordered = sorted((tuple(map(int, data["bounds"][index])),
                      int(data["sequence_reset_index"][index]))
                     for index in sequence_ids)
    run_start = ordered[0][0][0]
    run_end = ordered[-1][0][1]
    if (run_end - run_start < HISTORY_STEPS + 1
            or any(left[0][1] != right[0][0]
                   for left, right in zip(ordered, ordered[1:]))):
        raise ValueError("practice run must be one contiguous capture long enough to replay")
    if len({reset for _, reset in ordered}) != 1:
        raise ValueError("full-capture replay cannot span a simulator reset")
    return np.arange(run_start, run_end, dtype=np.int64)


def _rollout(checkpoint: Path, data: dict[str, Any], run_index: int,
             device: str, start_frame: int | None = None,
             end_frame: int | None = None,
             command_offset_frames: int = 0) -> dict[str, Any]:
    torch, model, metadata = _load_model(checkpoint, device)
    state = physical_state_from_dataset(
        data, wheel_state_source=str(metadata.get(
            "wheel_state_source", "filtered_odometry"))).astype(np.float32)
    if model.include_roll_state:
        state = append_roll_state(data, state)
    rows = _contiguous_run_indices(data, run_index)
    run_start, run_end = int(rows[0]), int(rows[-1] + 1)
    start = (run_start + HISTORY_STEPS - 1 if start_frame is None
             else int(start_frame))
    segment_end = run_end if end_frame is None else int(end_frame)
    if (start < run_start + HISTORY_STEPS - 1 or start >= run_end
            or segment_end <= start + 1 or segment_end > run_end):
        raise ValueError("practice replay segment lacks a complete in-run history/future")
    history_rows = np.arange(start - HISTORY_STEPS + 1, start + 1)
    future_rows = np.arange(start + 1, segment_end)
    if future_rows.size == 0:
        raise ValueError("practice capture has no future commands after initialization")
    history_state_size = int(model.history_state_size)
    history = np.column_stack((state[history_rows, :history_state_size],
                               data["frames"][history_rows, 7:9]))
    extra_history = (
        raw_encoder_history_features(data) if model.include_raw_encoder_history else
        wheel_innovation_history_features(data)
        if model.include_wheel_innovation_history else None)
    if extra_history is not None:
        history = np.column_stack((history, extra_history[history_rows]))
    command_rows = _command_indices(future_rows, command_offset_frames)
    if command_rows[0] < run_start or command_rows[-1] >= run_end:
        raise ValueError("command alignment leaves the contiguous capture")
    commands = data["frames"][command_rows, 7:9][None].astype(np.float32)
    initial = state[start:start + 1]
    delayed = data["frames"][start - 1:start, 7:9]
    initial_pose = data["simulator_pose_xyyaw"][start:start + 1]
    with torch.no_grad():
        predicted_state, _, _, _ = model.rollout(
            torch.as_tensor(initial, dtype=torch.float32, device=device),
            torch.as_tensor(delayed, dtype=torch.float32, device=device),
            torch.as_tensor(history[None], dtype=torch.float32, device=device),
            torch.as_tensor(commands, dtype=torch.float32, device=device))
        predicted_pose = integrate_pose(
            torch, predicted_state,
            torch.as_tensor(initial_pose, dtype=torch.float32, device=device),
            torch.as_tensor(initial, dtype=torch.float32, device=device))
    state_out = predicted_state[0].cpu().numpy().astype(np.float64)
    pose_out = predicted_pose[0].cpu().numpy().astype(np.float64)
    if not np.isfinite(state_out).all() or not np.isfinite(pose_out).all():
        raise FloatingPointError(f"{checkpoint} produced non-finite full-run state")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "metadata": metadata,
        "run_start": run_start,
        "run_end": run_end,
        "initial_frame": start,
        "future_frames": future_rows,
        "state": state_out,
        "pose": pose_out,
        "initial_pose": initial_pose[0].astype(np.float64),
    }


def _lap_transition_indices(lap_count: np.ndarray, start: int,
                            end: int) -> list[int]:
    return [index for index in range(start + 1, end)
            if lap_count[index] == lap_count[index - 1] + 1]


def _isolated_lap_replays(data: dict[str, Any], checkpoint: Path,
                          run_index: int, device: str,
                          gate_xy: np.ndarray, tangent: np.ndarray,
                          crossing_times: list[float],
                          command_offset_frames: int = 0
                          ) -> list[dict[str, Any]]:
    rows = _contiguous_run_indices(data, run_index)
    run_start, run_end = int(rows[0]), int(rows[-1] + 1)
    transitions = _lap_transition_indices(
        np.asarray(data["lap_count"], dtype=np.int32), run_start, run_end)
    if len(transitions) != len(crossing_times):
        raise ValueError("practice lap transition times and frame indices disagree")
    episodes = []
    for lap_index, (start, next_crossing) in enumerate(
            zip(transitions, transitions[1:])):
        result = _rollout(checkpoint, data, run_index, device,
                          start_frame=start, end_frame=next_crossing + 1,
                          command_offset_frames=command_offset_frames)
        metrics = _score_one_run(data, result)
        predicted_crossings = _predicted_crossings(
            result["initial_pose"], result["pose"], gate_xy, tangent)
        # The rollout initializes on the first recorded sample after the prior
        # gate crossing; compare predicted time from that sample to the next
        # interpolated recorded crossing, without using future truth as input.
        actual_duration = (crossing_times[lap_index + 1]
                           - start * DT_S)
        predicted_duration = _match_isolated_lap_crossing(
            predicted_crossings, actual_duration)
        episodes.append({
            "lap_index": lap_index,
            "lap_counter_at_initialization": int(data["lap_count"][start]),
            "initial_frame": start,
            "future_sample_count": int(len(result["future_frames"])),
            "episode_duration_s": float((next_crossing - start) * DT_S),
            "full_lap_metrics": metrics["full_recursive_metrics"],
            "predicted_gate_crossing_time_s": predicted_duration,
            "actual_gate_crossing_time_s": float(actual_duration),
            "gate_crossing_abs_error_s": (
                float(abs(predicted_duration - actual_duration))
                if predicted_duration is not None else None),
            "first_divergence_thresholds": metrics[
                "first_divergence_thresholds"],
            "initialization_contract": (
                "measured state, pose, and prior 80-frame history only; all future "
                "state is recursively predicted from recorded commands"),
        })
    return episodes


def _metrics(pred_state: np.ndarray, pred_pose: np.ndarray,
             truth_state: np.ndarray, truth_pose: np.ndarray,
             wheel_valid: np.ndarray | None = None) -> dict[str, Any]:
    position = pred_pose[:, :2] - truth_pose[:, :2]
    heading = _wrap(pred_pose[:, 2] - truth_pose[:, 2])
    truth_cos = np.cos(truth_pose[:, 2])
    truth_sin = np.sin(truth_pose[:, 2])
    along_track_error = position[:, 0] * truth_cos + position[:, 1] * truth_sin
    cross_track_error = -position[:, 0] * truth_sin + position[:, 1] * truth_cos
    speed_error = (np.hypot(pred_state[:, 0], pred_state[:, 1])
                   - np.hypot(truth_state[:, 0], truth_state[:, 1]))
    state_error = pred_state[:, :7] - truth_state[:, :7]
    wheel_valid = (np.ones(len(state_error), dtype=bool) if wheel_valid is None
                   else np.asarray(wheel_valid, dtype=bool))
    if wheel_valid.shape != (len(state_error),):
        raise ValueError("wheel validity mask does not match replay samples")
    result = {
        "position_radial_rmse_m": float(np.sqrt(np.mean(np.sum(position ** 2, axis=1)))),
        "position_endpoint_m": float(np.linalg.norm(position[-1])),
        "position_p95_m": float(np.quantile(np.linalg.norm(position, axis=1), 0.95)),
        "along_track_rmse_m": float(np.sqrt(np.mean(along_track_error ** 2))),
        "along_track_bias_m": float(np.mean(along_track_error)),
        "cross_track_rmse_m": float(np.sqrt(np.mean(cross_track_error ** 2))),
        "cross_track_bias_m": float(np.mean(cross_track_error)),
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
        "max_predicted_speed_mps": float(np.max(np.hypot(pred_state[:, 0], pred_state[:, 1]))),
    }
    if np.any(wheel_valid):
        result.update({
            "rear_left_wheel_bias_mps": float(np.mean(
                state_error[wheel_valid, 5])),
            "rear_right_wheel_bias_mps": float(np.mean(
                state_error[wheel_valid, 6])),
            "rear_left_wheel_rmse_mps": float(np.sqrt(np.mean(
                state_error[wheel_valid, 5] ** 2))),
            "rear_right_wheel_rmse_mps": float(np.sqrt(np.mean(
                state_error[wheel_valid, 6] ** 2))),
            "rear_wheel_pair_rmse_mps": float(np.sqrt(np.mean(
                state_error[wheel_valid, 5:7] ** 2))),
            "wheel_valid_sample_count": int(wheel_valid.sum()),
        })
    else:
        result.update({
            "rear_left_wheel_bias_mps": None,
            "rear_right_wheel_bias_mps": None,
            "rear_left_wheel_rmse_mps": None,
            "rear_right_wheel_rmse_mps": None,
            "rear_wheel_pair_rmse_mps": None,
            "wheel_valid_sample_count": 0,
        })
    if pred_state.shape[1] >= 9 and truth_state.shape[1] >= 9:
        result["roll_angle_rmse_rad"] = float(np.sqrt(np.mean(
            (pred_state[:, 7] - truth_state[:, 7]) ** 2)))
        result["roll_rate_rmse_rps"] = float(np.sqrt(np.mean(
            (pred_state[:, 8] - truth_state[:, 8]) ** 2)))
    return result


def _divergence_summary(pred_state: np.ndarray, pred_pose: np.ndarray,
                        truth_state: np.ndarray, truth_pose: np.ndarray,
                        lap_count: np.ndarray, future_indices: np.ndarray
                        ) -> dict[str, Any]:
    position_error = np.linalg.norm(pred_pose[:, :2] - truth_pose[:, :2], axis=1)
    speed_error = np.abs(np.hypot(pred_state[:, 0], pred_state[:, 1])
                         - np.hypot(truth_state[:, 0], truth_state[:, 1]))
    signals = {
        "position_error_over_0p5m": (position_error, 0.5, "m"),
        "position_error_over_1m": (position_error, 1.0, "m"),
        "forward_speed_error_over_0p5mps": (
            np.abs(pred_state[:, 0] - truth_state[:, 0]), 0.5, "m/s"),
        "speed_error_over_1mps": (speed_error, 1.0, "m/s"),
        "lateral_speed_error_over_0p25mps": (
            np.abs(pred_state[:, 1] - truth_state[:, 1]), 0.25, "m/s"),
        "yaw_rate_error_over_0p5rps": (
            np.abs(pred_state[:, 2] - truth_state[:, 2]), 0.5, "rad/s"),
        "rear_wheel_pair_error_over_1mps": (
            np.sqrt(np.mean((pred_state[:, 5:7] - truth_state[:, 5:7]) ** 2,
                            axis=1)), 1.0, "m/s"),
    }
    if pred_state.shape[1] >= 9 and truth_state.shape[1] >= 9:
        signals["roll_angle_error_over_0p1rad"] = (
            np.abs(pred_state[:, 7] - truth_state[:, 7]), 0.1, "rad")
        signals["roll_rate_error_over_0p5rps"] = (
            np.abs(pred_state[:, 8] - truth_state[:, 8]), 0.5, "rad/s")
    result = {}
    for name, (error, threshold, unit) in signals.items():
        matches = np.flatnonzero(error > threshold)
        if matches.size == 0:
            result[name] = {"threshold": threshold, "unit": unit,
                            "first_time_s": None}
            continue
        index = int(matches[0])
        result[name] = {
            "threshold": threshold,
            "unit": unit,
            "first_time_s": float((index + 1) * DT_S),
            "frame_index": int(future_indices[index]),
            "lap_count": int(lap_count[future_indices[index]]),
            "error_at_first_crossing": float(error[index]),
        }
    return result


def _horizon_summary(pred_state: np.ndarray, pred_pose: np.ndarray,
                     truth_state: np.ndarray, truth_pose: np.ndarray,
                     wheel_valid: np.ndarray
                     ) -> dict[str, Any]:
    result = {}
    for horizon_s in (0.75, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0):
        count = min(len(pred_state), int(round(horizon_s / DT_S)))
        if count < 1:
            continue
        result[f"{horizon_s:g}s"] = _metrics(
            pred_state[:count], pred_pose[:count],
            truth_state[:count], truth_pose[:count], wheel_valid[:count])
    return result


def _crossing_times(lap_count: np.ndarray, start: int, end: int) -> list[float]:
    transitions = [index for index in range(start + 1, end)
                   if lap_count[index] == lap_count[index - 1] + 1]
    # The timing contract is the validated 40 Hz frame period, not packet
    # receipt jitter. Interpolate a counter transition within its 25 ms step.
    result = []
    for index in transitions:
        before, after = int(lap_count[index - 1]), int(lap_count[index])
        fraction = (before + 1.0 - before) / (after - before)
        result.append(((index - 1) + fraction) * DT_S)
    return result


def _finish_gate(lap_count: np.ndarray, pose: np.ndarray,
                 run_start: int, run_end: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    transition_indices = [index for index in range(run_start + 1, run_end)
                          if lap_count[index] == lap_count[index - 1] + 1]
    if len(transition_indices) < 2:
        raise ValueError("practice capture needs at least two lap-count transitions")
    crossing_xy, headings, crossing_times = [], [], []
    for index in transition_indices:
        crossing_xy.append(0.5 * (pose[index - 1, :2] + pose[index, :2]))
        dyaw = _wrap(np.asarray([pose[index, 2] - pose[index - 1, 2]]))[0]
        headings.append(pose[index - 1, 2] + 0.5 * dyaw)
        crossing_times.append((index - 0.5) * DT_S)
    heading = float(np.arctan2(np.mean(np.sin(headings)),
                               np.mean(np.cos(headings))))
    tangent = np.asarray((np.cos(heading), np.sin(heading)))
    return np.mean(crossing_xy, axis=0), tangent, crossing_times


def _predicted_crossings(initial_pose: np.ndarray, predicted_pose: np.ndarray,
                         gate_xy: np.ndarray, tangent: np.ndarray) -> list[float]:
    path = np.vstack((initial_pose[None], predicted_pose))
    signed = (path[:, :2] - gate_xy) @ tangent
    result = []
    for index in range(1, len(path)):
        if signed[index - 1] < 0.0 <= signed[index]:
            denominator = signed[index] - signed[index - 1]
            if denominator > 1e-9:
                fraction = -signed[index - 1] / denominator
                result.append(((index - 1) + fraction) * DT_S)
    return result


def _match_isolated_lap_crossing(predicted_crossings: list[float],
                                 actual_duration_s: float
                                 ) -> float | None:
    """Associate the predicted next-lap crossing, not the initialized gate.

    Isolated replay starts at a measured lap-counter transition. Small pose/gate
    interpolation offsets can therefore make the first predicted sample cross
    that same gate again at t~=0. Select only a crossing at least halfway
    through the recorded lap interval, then choose the one nearest its endpoint.
    Recorded duration is used only to associate scored events, never as model
    input.
    """
    minimum_elapsed = 0.5 * float(actual_duration_s)
    candidates = [float(value) for value in predicted_crossings
                  if value >= minimum_elapsed]
    if not candidates:
        return None
    return min(candidates, key=lambda value: abs(value - actual_duration_s))


def _turn_intervals(yaw: np.ndarray, start: int, end: int,
                    threshold_rps: float = 0.18,
                    gap_fill_steps: int = 8,
                    minimum_steps: int = 12) -> list[tuple[int, int, int]]:
    unwrapped = np.unwrap(np.asarray(yaw[start:end], dtype=np.float64))
    rate = np.gradient(unwrapped, DT_S)
    smooth = np.convolve(rate, np.ones(9, dtype=np.float64) / 9.0,
                         mode="same")
    direction = np.where(smooth >= threshold_rps, 1,
                         np.where(smooth <= -threshold_rps, -1, 0))
    # Bridge brief near-zero gaps only when the turn direction is unchanged.
    index = 0
    while index < len(direction):
        if direction[index] != 0:
            index += 1
            continue
        gap_start = index
        while index < len(direction) and direction[index] == 0:
            index += 1
        if (gap_start > 0 and index < len(direction)
                and index - gap_start <= gap_fill_steps
                and direction[gap_start - 1] == direction[index]):
            direction[gap_start:index] = direction[gap_start - 1]
    spans: list[tuple[int, int, int]] = []
    active = np.flatnonzero(direction)
    if active.size == 0:
        return spans
    begin = previous = int(active[0])
    sign = int(direction[begin])
    for value_raw in active[1:]:
        value = int(value_raw)
        value_sign = int(direction[value])
        if (value - previous > gap_fill_steps + 1
                or value_sign != sign):
            if previous + 1 - begin >= minimum_steps:
                spans.append((start + begin, start + previous + 1, sign))
            begin, sign = value, value_sign
        previous = value
    if previous + 1 - begin >= minimum_steps:
        spans.append((start + begin, start + previous + 1, sign))
    return spans


def _score_one_run(data: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    start = int(result["initial_frame"])
    future = np.asarray(result["future_frames"], dtype=np.int64)
    end = int(future[-1] + 1)
    state = np.asarray(result["state"])
    pose = np.asarray(result["pose"])
    metadata = result["metadata"]
    wheel_source = str(metadata.get("wheel_state_source", "filtered_odometry"))
    truth_state_all = physical_state_from_dataset(
        data, wheel_state_source=wheel_source).astype(np.float64)
    if wheel_source == "raw_encoder":
        raw_valid = data.get("encoder_raw_valid")
        if raw_valid is None:
            raise ValueError("raw-wheel full replay requires encoder validity labels")
        wheel_valid_all = np.asarray(raw_valid, dtype=bool)
    else:
        wheel_valid_all = np.ones(len(truth_state_all), dtype=bool)
    truth_state = truth_state_all[future]
    wheel_valid = wheel_valid_all[future]
    roll_valid = np.asarray(data["imu_attitude_valid"], dtype=bool)[future]
    if state.shape[1] == 9:
        if not np.all(roll_valid):
            raise ValueError("predicted-roll full replay requires complete held-out roll labels")
        attitude = np.asarray(data["imu_attitude_frames"], dtype=np.float64)
        truth_state = np.column_stack((
            truth_state,
            attitude[future, 0], attitude[future, 2]))
    truth_pose = data["simulator_pose_xyyaw"][future].astype(np.float64)
    lap_count = np.asarray(data["lap_count"], dtype=np.int32)
    lap_values = lap_count[future]
    per_lap = {}
    for lap in np.unique(lap_values):
        mask = lap_values == lap
        if np.count_nonzero(mask) < 6:
            continue
        per_lap[str(int(lap))] = _metrics(
            state[mask], pose[mask], truth_state[mask], truth_pose[mask],
            wheel_valid[mask])
    turns = []
    for index, (left, right, direction) in enumerate(_turn_intervals(
            data["simulator_pose_xyyaw"][:, 2], start + 1, end), start=1):
        mask = (future >= left) & (future < right)
        if np.count_nonzero(mask) < 6:
            continue
        turn = _metrics(state[mask], pose[mask], truth_state[mask],
                        truth_pose[mask], wheel_valid[mask])
        turn.update({
            "turn_index_in_capture": index,
            "signed_yaw_direction": "positive" if direction > 0 else "negative",
            "lap_count": int(lap_count[left]),
            "start_after_initialization_s": (left - start) * DT_S,
            "duration_s": (right - left) * DT_S,
            "reference_peak_abs_yaw_rate_rps": float(np.max(np.abs(
                np.gradient(np.unwrap(data["simulator_pose_xyyaw"][left:right, 2]),
                            DT_S)))),
        })
        turns.append(turn)
    return {
        "initialization_at_capture_time_s": (start - int(result["run_start"])) * DT_S,
        "recursive_duration_s": len(future) * DT_S,
        "recursive_sample_count": len(future),
        "lap_count_values": sorted(int(value) for value in np.unique(lap_values)),
        "full_recursive_metrics": _metrics(
            state, pose, truth_state, truth_pose, wheel_valid),
        "recursive_horizon_metrics": _horizon_summary(
            state, pose, truth_state, truth_pose, wheel_valid),
        "first_divergence_thresholds": _divergence_summary(
            state, pose, truth_state, truth_pose, lap_count, future),
        "per_lap_metrics": per_lap,
        "turn_segmentation": {
            "method": "smoothed ground-truth pose yaw-rate sign segments; scoring only",
            "absolute_yaw_rate_threshold_rps": 0.18,
            "smoothing_window_steps": 9,
            "gap_fill_steps": 8,
            "minimum_turn_duration_s": 0.30,
            "turn_intervals": turns,
        },
    }


def score(dataset_path: Path, benchmark_path: Path,
          checkpoints: dict[str, Path], output_path: Path,
          device: str, isolated_laps: bool = False,
          command_offset_frames: int = 0) -> dict[str, Any]:
    if command_offset_frames not in (-1, 0):
        raise ValueError("command alignment offset must be -1 or 0 frames")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (benchmark.get("benchmark_id") != "practice_transfer_benchmark_v1"
            or benchmark.get("frozen") is not True
            or benchmark.get("training_or_checkpoint_selection_use") is not False):
        raise ValueError("practice dataset selector must be the frozen validation benchmark")
    dataset_path = dataset_path.resolve()
    if _sha256(dataset_path) != benchmark["dataset_sha256"]:
        raise ValueError("practice replay dataset hash differs from frozen benchmark")
    data = _load_dataset(dataset_path)
    if data.get("lap_count") is None:
        raise ValueError("practice dataset does not contain lap-counter labels")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite full practice replay report: {output_path}")
    run_ids = np.asarray(data["run_ids"]).astype(str)
    run_splits = np.asarray(data["splits"]).astype(str)
    run_indices = [index for index, split in enumerate(run_splits)
                   if split == "unseen_practice"]
    if not run_indices:
        raise ValueError("dataset has no unseen practice runs")
    run_reports: dict[str, Any] = {}
    for run_index in run_indices:
        run_name = str(run_ids[run_index])
        rows = _contiguous_run_indices(data, run_index)
        run_start, run_end = int(rows[0]), int(rows[-1] + 1)
        truth_pose = data["simulator_pose_xyyaw"]
        gate_xy, tangent, actual_crossings_all = _finish_gate(
            data["lap_count"], truth_pose, run_start, run_end)
        actual_transition_indices = _lap_transition_indices(
            np.asarray(data["lap_count"], dtype=np.int32), run_start, run_end)
        if len(actual_transition_indices) != len(actual_crossings_all):
            raise ValueError("practice lap transition geometry is inconsistent")
        report_by_model = {}
        for name, checkpoint in checkpoints.items():
            result = _rollout(
                checkpoint.resolve(), data, run_index, device,
                command_offset_frames=command_offset_frames)
            score_report = _score_one_run(data, result)
            if isolated_laps:
                score_report["isolated_lap_replays"] = _isolated_lap_replays(
                    data, checkpoint.resolve(), run_index, device, gate_xy,
                    tangent, actual_crossings_all, command_offset_frames)
            predicted_crossings = _predicted_crossings(
                result["initial_pose"], result["pose"], gate_xy, tangent)
            start_time_s = (result["initial_frame"] * DT_S)
            actual_crossings = [time for time in actual_crossings_all
                                if time > start_time_s]
            actual_laps = np.diff(actual_crossings).tolist()
            predicted_laps = np.diff(predicted_crossings).tolist()
            paired_count = min(len(actual_laps), len(predicted_laps))
            score_report["lap_times"] = {
                "actual_lap_crossing_count_after_initialization": len(actual_crossings),
                "predicted_gate_crossing_count": len(predicted_crossings),
                "actual_lap_times_s": actual_laps,
                "predicted_lap_times_s": predicted_laps,
                "paired_lap_time_abs_error_s": [
                    abs(actual_laps[index] - predicted_laps[index])
                    for index in range(paired_count)],
                "paired_lap_time_mae_s": (
                    float(np.mean([abs(actual_laps[index] - predicted_laps[index])
                                   for index in range(paired_count)]))
                    if paired_count else None),
                "comparison_note": (
                    "predicted crossing times use a finish gate estimated only "
                    "from the recorded ground-truth lap transitions; command "
                    "replay does not consume future pose or lap-count labels"),
            }
            report_by_model[name] = score_report
        run_reports[run_name] = {
            "capture_frame_count": int(run_end - run_start),
            "capture_duration_s": float((run_end - run_start - 1) * DT_S),
            "recorded_lap_counter_transitions": int(len(actual_crossings_all)),
            "finish_gate_xy_m": gate_xy.tolist(),
            "finish_gate_forward_tangent": tangent.tolist(),
            "models": report_by_model,
        }
    report = {
        "schema_version": 2 if isolated_laps else 1,
        "purpose": "full-capture command-only practice replay after paired model improvement",
        "timebase_s": DT_S,
        "command_offset_frames_from_target_state_row": int(
            command_offset_frames),
        "command_alignment_note": (
            "-1 uses the command stored with the preceding frame, consistent "
            "with the train-only fitted one-frame actuator delay; 0 uses the "
            "command stored with the target-state frame"),
        "initialization": f"one measured {HISTORY_STEPS}-frame history and pose; all later plant state is recursive prediction",
        "future_truth_or_sensor_feedback_used": False,
        "isolated_lap_replays_enabled": bool(isolated_laps),
        "test_and_final_test_used": False,
        "practice_checkpoint_selection_use": False,
        "benchmark": str(benchmark_path.resolve()),
        "benchmark_sha256": _sha256(benchmark_path.resolve()),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "models": {name: {"checkpoint": str(path.resolve()),
                           "checkpoint_sha256": _sha256(path.resolve())}
                   for name, path in checkpoints.items()},
        "lap_time_method": "crossings of a gate fit from recorded lap-counter transitions and simulator pose",
        "runs": run_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--dataset", type=Path,
                        help="must match the frozen practice benchmark's dataset hash")
    parser.add_argument("--checkpoint", action="append", required=True,
                        metavar="NAME=PATH", help="one or more EDSSM checkpoints")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "full_replay.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--isolated-laps", action="store_true",
                        help="add full-lap rollouts reinitialized only from measured start-of-lap state and prior history")
    parser.add_argument("--command-offset-frames", type=int, choices=(-1, 0),
                        default=0,
                        help="align each transition's command to target-state row (0) or preceding row (-1)")
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    dataset = args.dataset or Path(str(benchmark["dataset"]).replace(
        "/workspace/", str(ROOT) + "/"))
    checkpoints = {}
    for value in args.checkpoint:
        if "=" not in value:
            parser.error("--checkpoint must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or name in checkpoints:
            parser.error("checkpoint names must be non-empty and unique")
        checkpoints[name] = Path(path)
    report = score(dataset, args.benchmark, checkpoints, args.output,
                   args.device, args.isolated_laps,
                   args.command_offset_frames)
    compact = {
        run: {model: {
            "full_recursive_metrics": detail["full_recursive_metrics"],
            "lap_times": detail["lap_times"],
        } for model, detail in value["models"].items()}
        for run, value in report["runs"].items()
    }
    print(json.dumps({"report": str(args.output.resolve()), "runs": compact}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

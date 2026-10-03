#!/usr/bin/env python3
"""Score deterministic EDSSM free rollouts on whole-run validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    ACCELERATION_NAMES,
    COMMAND_NAMES,
    DT_S,
    HISTORY_STEPS,
    LATENT_SIZE,
    PHYSICAL_STATE_NAMES,
    ROLL_STATE_NAMES,
    ActuatorChannel,
    ActuatorFit,
    append_roll_state,
    effective_model_type,
    fit_actuator_dynamics,
    generalized_acceleration_targets,
    integrate_pose,
    physical_state_from_dataset,
    raw_encoder_history_features,
    wheel_innovation_history_features,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
    "replacement_teacher_dataset_v1/openplane_dynamics.npz"
)
HORIZON_STEPS = {"0.75s": 30, "2s": 80, "5s": 200}
MIN_SCALE = np.asarray((0.10, 0.10, 0.10, 0.02, 0.02,
                        0.20, 0.20, 0.02, 0.02), dtype=np.float64)


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required to score the EDSSM") from exc
    return torch, nn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _robust_normalizers(data: dict[str, Any], state: np.ndarray,
                        acceleration: np.ndarray,
                        history_state: np.ndarray | None = None,
                        history_extra: np.ndarray | None = None,
                        require_raw_encoder_valid: bool = False,
                        max_throttle_command: float = 0.50
                        ) -> dict[str, np.ndarray]:
    if (not np.isfinite(max_throttle_command)
            or not 0.0 <= max_throttle_command <= 1.0):
        raise ValueError("maximum throttle command must be in [0, 1]")
    frames = np.asarray(data["frames"], dtype=np.float64)
    run_by_frame = np.empty(len(frames), dtype=np.int32)
    for (start, end), run in zip(data["bounds"], data["seq_run"]):
        run_by_frame[int(start):int(end)] = int(run)
    train = np.asarray(data["splits"]).astype(str)[run_by_frame] == "train"
    speed = np.hypot(state[:, 0], state[:, 1])
    domain = (train & (speed <= 12.0)
              & (frames[:, 8] <= max_throttle_command)
              & np.isfinite(acceleration).all(axis=1))
    raw_wheel_valid = data.get("encoder_raw_valid")
    if require_raw_encoder_valid and raw_wheel_valid is None:
        raise ValueError("raw encoder target normalization needs its validity sidecar")
    if require_raw_encoder_valid and raw_wheel_valid is not None:
        domain &= np.asarray(raw_wheel_valid, dtype=bool)
    if np.count_nonzero(domain) < 1000:
        raise ValueError("too few training rows in the declared plant domain")
    history_values = state if history_state is None else history_state
    if len(history_values) != len(state):
        raise ValueError("EDSSM history/state arrays do not align")
    history = np.column_stack((history_values, frames[:, 7:9]))
    if history_extra is not None:
        history_extra = np.asarray(history_extra, dtype=np.float64)
        if (history_extra.ndim != 2 or len(history_extra) != len(state)
                or not np.isfinite(history_extra).all()):
            raise ValueError("EDSSM extra history features do not align")
        history = np.column_stack((history, history_extra))

    def center_scale(values: np.ndarray, floor: np.ndarray):
        selected = values[domain]
        center = np.median(selected, axis=0)
        q25, q75 = np.quantile(selected, (0.25, 0.75), axis=0)
        scale = np.maximum(q75 - q25, floor)
        return center.astype(np.float32), scale.astype(np.float32)

    if state.shape[1] == 7:
        state_floor = MIN_SCALE[:7]
    elif state.shape[1] == 9:
        state_floor = np.concatenate((MIN_SCALE[:7], (0.01, 0.02)))
    else:
        raise ValueError("EDSSM state must contain seven or nine channels")
    history_state_floor = MIN_SCALE[:history_values.shape[1]]
    history_floor = np.concatenate((history_state_floor, MIN_SCALE[7:9]))
    if history_extra is not None:
        history_floor = np.concatenate((
            history_floor, np.asarray((0.20, 0.20, 0.05), dtype=np.float64)))
    history_mean, history_scale = center_scale(history, history_floor)
    state_mean, state_scale = center_scale(state, state_floor)
    command_mean, command_scale = center_scale(
        frames[:, 7:9], MIN_SCALE[7:9])
    absolute = np.abs(acceleration[domain])
    acceleration_bounds = np.quantile(absolute, 0.999, axis=0) * 1.15
    acceleration_bounds = np.maximum(
        acceleration_bounds, np.asarray((1.0, 1.0, 0.5, 1.0, 1.0)))
    return {
        "history_mean": history_mean,
        "history_scale": history_scale,
        "state_mean": state_mean,
        "state_scale": state_scale,
        "command_mean": command_mean,
        "command_scale": command_scale,
        "acceleration_bounds": acceleration_bounds.astype(np.float32),
        "training_domain_rows": np.asarray([np.count_nonzero(domain)], dtype=np.int64),
    }


def _load_model(checkpoint_path: Path, device_name: str):
    torch, nn = _torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    actuator_fit = ActuatorFit(
        ActuatorChannel(**metadata["actuator_fit"]["steering"]),
        ActuatorChannel(**metadata["actuator_fit"]["throttle"]),
        metadata["actuator_fit"].get("diagnostics", {}),
    )
    arrays = {name: np.asarray(metadata[name], dtype=np.float32) for name in (
        "history_mean", "history_scale", "state_mean", "state_scale",
        "command_mean", "command_scale", "acceleration_bounds")}
    model_class = effective_model_type(
        torch, nn, arrays["history_mean"], arrays["history_scale"],
        arrays["state_mean"], arrays["state_scale"],
        arrays["command_mean"], arrays["command_scale"],
        arrays["acceleration_bounds"], actuator_fit,
        encoder=str(metadata["encoder"]),
        latent_size=int(metadata["latent_size"]),
        expert_count=int(metadata.get("expert_count", 1)),
        include_roll_state=bool(metadata.get("include_roll_state", False)),
        couple_roll_acceleration=bool(
            metadata.get("couple_roll_acceleration", False)),
        roll_residual_mode=bool(metadata.get("roll_residual_mode", False)),
        roll_oscillator_coefficients=(
            (metadata.get("roll_oscillator_fit") or {}).get("coefficients")),
        include_raw_encoder_history=bool(
            metadata.get("include_raw_encoder_history", False)),
        include_wheel_innovation_history=bool(
            metadata.get("include_wheel_innovation_history", False)),
        wheel_dynamics_mode=str(metadata.get(
            "wheel_dynamics_mode", "surface_acceleration")))
    model = model_class()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device_name)
    model.eval()
    return torch, model, metadata


def _frame_run_index(data: dict[str, Any]) -> np.ndarray:
    result = np.full(len(data["frames"]), -1, dtype=np.int32)
    for (start_raw, end_raw), run_raw in zip(data["bounds"], data["seq_run"]):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if np.any(result[start:end] >= 0):
            raise ValueError("overlapping sequence bounds in validation dataset")
        result[start:end] = run
    if np.any(result < 0):
        raise ValueError("sequence bounds do not cover the validation dataset")
    return result


def _support_limits(data: dict[str, Any], state: np.ndarray,
                    max_throttle_command: float = 0.50) -> tuple[float, float]:
    frames = np.asarray(data["frames"], dtype=np.float64)
    run_index = _frame_run_index(data)
    train = np.asarray(data["splits"]).astype(str)[run_index] == "train"
    support = train & (np.hypot(state[:, 0], state[:, 1]) <= 12.0) & (
        frames[:, 8] <= max_throttle_command)
    mismatch = np.abs(0.5 * (state[:, 5] + state[:, 6]) - state[:, 0])
    return (float(np.quantile(mismatch[support], 0.90)),
            float(np.quantile(mismatch[support], 0.95)))


def _window_regimes(data: dict[str, Any], state: np.ndarray, start: int,
                    mismatch_p90: float, mismatch_p95: float,
                    max_throttle_command: float = 0.50) -> list[str]:
    frames = data["frames"]
    speed = float(np.hypot(state[start, 0], state[start, 1]))
    steering = abs(float(state[start, 3]))
    throttle = float(frames[start, 8])
    mismatch = abs(float(0.5 * (state[start, 5] + state[start, 6])
                         - state[start, 0]))
    future_steering = np.abs(state[start + 1:start + 31, 3])
    future_speed = np.hypot(state[start + 1:start + 81, 0],
                            state[start + 1:start + 81, 1])
    future_throttle = frames[start + 1:start + 81, 8]
    future_mismatch = np.abs(
        0.5 * (state[start + 1:start + 81, 5]
               + state[start + 1:start + 81, 6])
        - state[start + 1:start + 81, 0])
    tags = []
    if (1.0 <= speed <= 8.0 and steering < 0.20 and throttle > 0.05
            and mismatch <= mismatch_p90):
        tags.append("ordinary")
    if ((speed <= 12.0 and throttle <= max_throttle_command
            and steering >= 0.40)
            or (future_steering.size
                and np.mean(future_steering) >= 0.40
                and np.max(future_throttle[:len(future_steering)])
                    <= max_throttle_command)):
        tags.append("high_steering")
    if (speed >= 1.0 and throttle <= 1e-3) or np.any(
            (future_speed >= 1.0) & (future_throttle <= 1e-3)):
        tags.append("zero_throttle")
    if (3.0 <= speed <= 8.0 and throttle <= 0.05) or np.any(
            (future_speed >= 3.0) & (future_speed <= 8.0)
            & (future_throttle <= 0.05)):
        tags.append("low_throttle_moving")
    if ((speed <= 12.0 and throttle <= max_throttle_command
            and mismatch >= mismatch_p95)
            or np.any((future_speed <= 12.0)
                      & (future_throttle <= max_throttle_command)
                      & (future_mismatch >= mismatch_p95))):
        tags.append("wheel_mismatch_proxy")
    return tags


def _fixed_validation_windows(data: dict[str, Any], state: np.ndarray,
                              max_windows_per_run: int = 24,
                              horizon_steps: int = 200,
                              max_throttle_command: float = 0.50,
                              split: str = "validation"
                              ) -> list[dict[str, Any]]:
    if max_windows_per_run <= 0 or horizon_steps < 200:
        raise ValueError("validation windows need a positive budget and >=5 s horizon")
    if (not np.isfinite(max_throttle_command)
            or not 0.0 <= max_throttle_command <= 1.0):
        raise ValueError("maximum throttle command must be in [0, 1]")
    if split not in ("validation", "unseen_practice"):
        raise ValueError("scoring split must be validation or unseen_practice")
    run_index = _frame_run_index(data)
    splits = np.asarray(data["splits"]).astype(str)
    mismatch_p90, mismatch_p95 = _support_limits(
        data, state, max_throttle_command)
    raw_wheel_valid = data.get("encoder_raw_valid")
    if raw_wheel_valid is not None:
        raw_wheel_valid = np.asarray(raw_wheel_valid, dtype=bool)
        if raw_wheel_valid.shape != (len(state),):
            raise ValueError("raw encoder validity mask does not align with state")
    candidates: dict[int, list[dict[str, Any]]] = {}
    for sequence_id, ((start_raw, end_raw), run_raw) in enumerate(
            zip(data["bounds"], data["seq_run"])):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if (splits[run] != split
                or end - start < HISTORY_STEPS + horizon_steps):
            continue
        low = start + HISTORY_STEPS - 1
        high = end - horizon_steps - 1
        step = 40
        starts = list(range(low, high + 1, step))
        if high not in starts:
            starts.append(high)
        for row_start in starts:
            future = np.arange(row_start + 1,
                               row_start + horizon_steps + 1)
            if (np.max(np.hypot(state[future, 0], state[future, 1])) > 12.0
                    or np.max(data["frames"][future, 8])
                        > max_throttle_command
                    or not np.isfinite(data["simulator_pose_xyyaw"][
                        row_start:row_start + horizon_steps + 1]).all()):
                continue
            candidates.setdefault(run, []).append({
                "sequence_id": sequence_id,
                "start": row_start,
                "run": run,
                "regimes": _window_regimes(
                    data, state, row_start, mismatch_p90, mismatch_p95,
                    max_throttle_command),
            })
    selected: list[dict[str, Any]] = []
    for run, rows in sorted(candidates.items()):
        # First preserve one representative for every available hard regime,
        # then fill the remaining fixed budget by evenly spaced run positions.
        chosen: dict[tuple[int, int], dict[str, Any]] = {}
        for regime in ("high_steering", "zero_throttle",
                       "low_throttle_moving", "wheel_mismatch_proxy",
                       "ordinary"):
            matches = [row for row in rows if regime in row["regimes"]]
            if matches:
                chosen[(matches[0]["sequence_id"], matches[0]["start"])] = matches[0]
                if len(matches) > 1:
                    chosen[(matches[-1]["sequence_id"], matches[-1]["start"])] = matches[-1]
        remaining = max(0, max_windows_per_run - len(chosen))
        if remaining and rows:
            ordered = sorted(rows, key=lambda row: (row["sequence_id"], row["start"]))
            positions = np.linspace(0, len(ordered) - 1,
                                    min(remaining, len(ordered)), dtype=np.int64)
            for position in positions:
                row = ordered[int(position)]
                chosen[(row["sequence_id"], row["start"])] = row
        selected.extend(sorted(chosen.values(),
                               key=lambda row: (row["run"], row["sequence_id"], row["start"])))
    if not selected:
        raise ValueError("validation split has no complete EDSSM windows")
    return selected


def _window_batch(data: dict[str, Any], state: np.ndarray,
                  windows: list[dict[str, Any]],
                  history_state_size: int,
                  raw_history_features: np.ndarray | None = None,
                  horizon_steps: int = 200,
                  command_offset_frames: int = 0
                  ) -> tuple[np.ndarray, ...]:
    if command_offset_frames not in (-1, 0):
        raise ValueError("command alignment offset must be -1 or 0 frames")
    histories, initial, delayed, commands, target_states, poses = [], [], [], [], [], []
    for row in windows:
        start = int(row["start"])
        hist_indices = np.arange(start - HISTORY_STEPS + 1, start + 1)
        future_indices = np.arange(start + 1, start + horizon_steps + 1)
        history = np.column_stack((state[hist_indices, :history_state_size],
                                   data["frames"][hist_indices, 7:9]))
        if raw_history_features is not None:
            history = np.column_stack((
                history, raw_history_features[hist_indices]))
        histories.append(history)
        initial.append(state[start])
        delayed.append(data["frames"][hist_indices[-2], 7:9])
        command_indices = future_indices + command_offset_frames
        commands.append(data["frames"][command_indices, 7:9])
        target_states.append(state[future_indices])
        poses.append(data["simulator_pose_xyyaw"][
            np.concatenate(([start], future_indices))])
    return tuple(np.asarray(values, dtype=np.float32) for values in (
        histories, initial, delayed, commands, target_states, poses))


def _rmse(values: np.ndarray, axis=None) -> np.ndarray | float:
    result = np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2, axis=axis))
    return float(result) if np.ndim(result) == 0 else result


def _cluster_summary(per_run: dict[str, float], seed: int = 20261002
                     ) -> dict[str, Any]:
    values = np.asarray(list(per_run.values()), dtype=np.float64)
    if not len(values):
        return {"independent_runs": 0, "macro_run_mean": None,
                "run_cluster_bootstrap_95pct_ci": None, "per_run": {}}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(5000, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "independent_runs": int(len(values)),
        "macro_run_mean": float(np.mean(values)),
        "run_cluster_bootstrap_95pct_ci": np.quantile(
            means, (0.025, 0.975)).tolist(),
        "per_run": dict(sorted(per_run.items())),
    }


def _gate_usage_summary(per_run_values: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    if not per_run_values:
        return {"independent_runs": 0, "macro_run_mean_probability": []}
    per_run = {
        run: np.mean(np.stack(rows), axis=0)
        for run, rows in per_run_values.items()
    }
    mean = np.mean(np.stack(list(per_run.values())), axis=0)
    entropy = -float(np.sum(mean * np.log(np.maximum(mean, 1e-12))))
    return {
        "independent_runs": len(per_run),
        "macro_run_mean_probability": mean.tolist(),
        "macro_run_mean_entropy_nats": entropy,
        "per_run_mean_probability": {
            run: values.tolist() for run, values in sorted(per_run.items())},
    }


def evaluate_model(model, data: dict[str, Any], state: np.ndarray,
                   state_scale: np.ndarray, device_name: str,
                   windows: list[dict[str, Any]] | None = None,
                   batch_size: int = 32,
                   wheel_state_source: str = "filtered_odometry",
                   horizon_steps: dict[str, int] | None = None,
                   command_offset_frames: int = 0
                   ) -> dict[str, Any]:
    torch, _ = _torch()
    if not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("EDSSM validation requires exact 25 ms dt")
    horizon_steps = HORIZON_STEPS if horizon_steps is None else horizon_steps
    if (not horizon_steps or min(horizon_steps.values()) < 1
            or len(set(horizon_steps.values())) != len(horizon_steps)):
        raise ValueError("validation horizon map must contain unique positive step counts")
    windows = windows or _fixed_validation_windows(
        data, state, horizon_steps=max(horizon_steps.values()))
    run_ids = np.asarray(data["run_ids"]).astype(str)
    scale = np.asarray(state_scale, dtype=np.float64)
    if wheel_state_source not in ("filtered_odometry", "raw_encoder"):
        raise ValueError("unknown wheel-state evaluation target")
    raw_wheel_valid = data.get("encoder_raw_valid")
    if wheel_state_source == "raw_encoder":
        if raw_wheel_valid is None:
            raise ValueError("raw wheel evaluation requires its validity sidecar")
        raw_wheel_valid = np.asarray(raw_wheel_valid, dtype=bool)
    per_run_mse: dict[str, dict[str, list[float]]] = {}
    per_regime_mse: dict[str, dict[str, dict[str, list[float]]]] = {}
    per_run_gates: dict[str, list[np.ndarray]] = {}
    per_regime_gates: dict[str, dict[str, list[np.ndarray]]] = {}
    speed_violation_count = 0
    prediction_count = 0
    state_names = (PHYSICAL_STATE_NAMES + ROLL_STATE_NAMES
                   if state.shape[1] == 9 else PHYSICAL_STATE_NAMES)
    raw_history_features = (
        raw_encoder_history_features(data) if model.include_raw_encoder_history else
        wheel_innovation_history_features(data)
        if model.include_wheel_innovation_history else None)
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start:batch_start + batch_size]
        history, initial, delayed, commands, truth_states, truth_poses = (
            _window_batch(data, state, batch_windows,
                          model.history_state_size,
                          raw_history_features,
                          max(horizon_steps.values()),
                          command_offset_frames))
        device = torch.device(device_name)
        with torch.no_grad():
            predicted_states, _, _, gate_weights = model.rollout(
                torch.as_tensor(initial, dtype=torch.float32, device=device),
                torch.as_tensor(delayed, dtype=torch.float32, device=device),
                torch.as_tensor(history, dtype=torch.float32, device=device),
                torch.as_tensor(commands, dtype=torch.float32, device=device))
            predicted_poses = integrate_pose(
                torch, predicted_states,
                torch.as_tensor(truth_poses[:, 0], dtype=torch.float32,
                                device=device),
                torch.as_tensor(initial, dtype=torch.float32, device=device))
        pred = predicted_states.detach().cpu().numpy().astype(np.float64)
        pred_pose = predicted_poses.detach().cpu().numpy().astype(np.float64)
        gates = gate_weights.detach().cpu().numpy().astype(np.float64)
        speed_violation_count += int(np.count_nonzero(
            np.hypot(pred[:, :, 0], pred[:, :, 1]) > 12.0))
        prediction_count += int(pred.shape[0] * pred.shape[1])
        for local_index, row in enumerate(batch_windows):
            run_name = str(run_ids[int(row["run"])])
            run_bucket = per_run_mse.setdefault(run_name, {})
            mean_gate = np.mean(gates[local_index], axis=0)
            per_run_gates.setdefault(run_name, []).append(mean_gate)
            truth = truth_states[local_index].astype(np.float64)
            target_pose = truth_poses[local_index].astype(np.float64)[1:]
            position_error = pred_pose[local_index, :, :2] - target_pose[:, :2]
            heading_error = _wrap_angle(
                pred_pose[local_index, :, 2] - target_pose[:, 2])
            for name, steps in horizon_steps.items():
                state_error = (pred[local_index, :steps] - truth[:steps]) / scale
                body_rmse = float(_rmse(state_error[:, :5]))
                wheel_mask = np.ones(steps, dtype=bool)
                if wheel_state_source == "raw_encoder":
                    start = int(row["start"])
                    wheel_mask = raw_wheel_valid[start + 1:start + steps + 1]
                    if not np.any(wheel_mask):
                        raise ValueError("validation horizon has no raw wheel labels")
                wheel_rmse = float(_rmse(state_error[wheel_mask, 5:7]))
                roll_rmse = (float(_rmse(state_error[:, 7:9]))
                             if state.shape[1] == 9 else None)
                position_rmse = float(_rmse(position_error[:steps]))
                heading_rmse = float(_rmse(heading_error[:steps]))
                run_bucket.setdefault(f"{name}/body", []).append(body_rmse ** 2)
                run_bucket.setdefault(f"{name}/wheel", []).append(wheel_rmse ** 2)
                if roll_rmse is not None:
                    run_bucket.setdefault(f"{name}/roll_state", []).append(
                        roll_rmse ** 2)
                run_bucket.setdefault(f"{name}/position_m", []).append(position_rmse ** 2)
                radial_rmse = float(_rmse(
                    np.linalg.norm(position_error[:steps], axis=1)))
                run_bucket.setdefault(f"{name}/position_radial_m", []).append(
                    radial_rmse ** 2)
                run_bucket.setdefault(f"{name}/heading_rad", []).append(heading_rmse ** 2)
                raw_error = pred[local_index, :steps] - truth[:steps]
                for channel_index, channel_name in enumerate(state_names):
                    channel_mask = (wheel_mask if channel_index in (5, 6)
                                    and wheel_state_source == "raw_encoder"
                                    else slice(None))
                    run_bucket.setdefault(
                        f"{name}/raw/{channel_name}", []).append(
                            float(np.mean(
                                raw_error[channel_mask, channel_index] ** 2)))
                predicted_speed = np.hypot(
                    pred[local_index, :steps, 0], pred[local_index, :steps, 1])
                truth_speed = np.hypot(truth[:steps, 0], truth[:steps, 1])
                run_bucket.setdefault(f"{name}/speed_mps", []).append(
                    float(np.mean((predicted_speed - truth_speed) ** 2)))
                for regime in row["regimes"]:
                    per_regime_gates.setdefault(regime, {}).setdefault(
                        run_name, []).append(mean_gate)
                    regime_bucket = per_regime_mse.setdefault(regime, {}).setdefault(
                        run_name, {})
                    regime_bucket.setdefault(f"{name}/body", []).append(body_rmse ** 2)
                    regime_bucket.setdefault(f"{name}/wheel", []).append(wheel_rmse ** 2)
                    regime_bucket.setdefault(f"{name}/position_m", []).append(position_rmse ** 2)
                    regime_bucket.setdefault(f"{name}/position_radial_m", []).append(
                        radial_rmse ** 2)
                    regime_bucket.setdefault(f"{name}/heading_rad", []).append(heading_rmse ** 2)
                    for channel_index, channel_name in enumerate(state_names):
                        channel_mask = (wheel_mask if channel_index in (5, 6)
                                        and wheel_state_source == "raw_encoder"
                                        else slice(None))
                        regime_bucket.setdefault(
                            f"{name}/raw/{channel_name}", []).append(
                                float(np.mean(
                                    raw_error[channel_mask, channel_index] ** 2)))
                    regime_bucket.setdefault(f"{name}/speed_mps", []).append(
                        float(np.mean((predicted_speed - truth_speed) ** 2)))
                    if roll_rmse is not None:
                        regime_bucket.setdefault(f"{name}/roll_state", []).append(
                            roll_rmse ** 2)

    horizons: dict[str, Any] = {}
    for name in horizon_steps:
        metrics = {}
        channel_metrics = (PHYSICAL_STATE_NAMES + ROLL_STATE_NAMES
                           if state.shape[1] == 9 else PHYSICAL_STATE_NAMES)
        fields = ["body", "wheel", "position_m", "position_radial_m",
                  "heading_rad", "speed_mps"]
        if state.shape[1] == 9:
            fields.append("roll_state")
        for field in (*fields,
                      *(f"raw/{channel}" for channel in channel_metrics)):
            metrics[field] = _cluster_summary({
                run: float(np.sqrt(np.mean(values[f"{name}/{field}"])))
                for run, values in per_run_mse.items()
                if values.get(f"{name}/{field}")
            })
        horizons[name] = metrics
    regime_report: dict[str, Any] = {}
    for regime, run_values in per_regime_mse.items():
        regime_report[regime] = {"horizons": {}}
        for name in horizon_steps:
            regime_report[regime]["horizons"][name] = {}
            fields = ["body", "wheel", "position_m", "position_radial_m",
                      "heading_rad", "speed_mps"]
            if state.shape[1] == 9:
                fields.append("roll_state")
            for field in (*fields,
                          *(f"raw/{channel}" for channel in channel_metrics)):
                regime_report[regime]["horizons"][name][field] = _cluster_summary({
                    run: float(np.sqrt(np.mean(values[f"{name}/{field}"])))
                    for run, values in run_values.items()
                    if values.get(f"{name}/{field}")
                })
    return {
        "window_count": int(len(windows)),
        "independent_run_count": int(len(per_run_mse)),
        "validation_run_ids": sorted(per_run_mse),
        "future_truth_or_feedback_used": False,
        "fixed_context_seconds": (HISTORY_STEPS - 1) * DT_S,
        "future_inputs": list(COMMAND_NAMES),
        "horizons": horizons,
        "hard_regimes": regime_report,
        "gate_usage": {
            "all_validation_windows": _gate_usage_summary(per_run_gates),
            "by_hard_regime": {
                regime: _gate_usage_summary(run_values)
                for regime, run_values in per_regime_gates.items()
            },
        },
        "predicted_speed_over_12mps_fraction": (
            speed_violation_count / prediction_count if prediction_count else 0.0),
        "window_manifest": [
            {"run_id": str(run_ids[int(row["run"])]),
             "sequence_id": int(row["sequence_id"]),
             "start_frame": int(row["start"]),
             "regimes": list(row["regimes"])}
            for row in windows
        ],
    }


def score(checkpoint_path: Path, dataset_path: Path, output_path: Path,
          device_name: str = "cpu", max_windows_per_run: int = 24,
          max_rollout_steps: int = 200,
          max_throttle_command_override: float | None = None,
          score_split: str = "validation",
          allow_external_dataset: bool = False,
          command_offset_frames: int = 0
          ) -> dict[str, Any]:
    checkpoint_path, dataset_path, output_path = map(
        lambda value: value.resolve(), (checkpoint_path, dataset_path, output_path))
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (existing.get("checkpoint_sha256") != _sha256(checkpoint_path)
                or existing.get("dataset_sha256") != _sha256(dataset_path)):
            raise FileExistsError(
                f"refusing to overwrite unrelated EDSSM score: {output_path}")
    torch, model, metadata = _load_model(checkpoint_path, device_name)
    del torch
    data = _load_dataset(dataset_path)
    same_dataset = metadata["dataset_sha256"] == _sha256(dataset_path)
    if same_dataset:
        if int(data["schema_version"]) != 9:
            raise ValueError("the training dataset must use frozen schema 9")
    elif not allow_external_dataset:
        raise ValueError(
            "checkpoint and dataset hash differ; external evaluation requires "
            "--allow-external-dataset")
    elif (int(metadata["dataset_schema_version"]) != 9
          or int(data["schema_version"]) not in (8, 9)
          or data["frames"].shape[1] != 9
          or data["feature_names"] != [
              "u_rear_mps", "v_rear_mps", "yaw_rate_rps",
              "steering_feedback_rad", "throttle_feedback_norm",
              "rear_left_surface_mps", "rear_right_surface_mps",
              "steering_command_rad", "throttle_command_norm"]):
        raise ValueError(
            "external EDSSM data must preserve the schema-9 nine-channel "
            "feature layout")
    if score_split not in ("validation", "unseen_practice"):
        raise ValueError("scoring split must be validation or unseen_practice")
    selected_run_ids = np.asarray(data["run_ids"])[
        np.asarray(data["splits"]).astype(str) == score_split].astype(str)
    if set(metadata["training_runs"]).intersection(selected_run_ids):
        raise ValueError(f"checkpoint training runs overlap {score_split}")
    if max_rollout_steps not in (200, 400):
        raise ValueError("EDSSM scoring supports five- or ten-second horizons")
    if command_offset_frames not in (-1, 0):
        raise ValueError("command alignment offset must be -1 or 0 frames")
    state = physical_state_from_dataset(
        data, wheel_state_source=str(metadata.get(
            "wheel_state_source", "filtered_odometry"))).astype(np.float64)
    if bool(metadata.get("include_roll_state", False)):
        state = append_roll_state(data, state).astype(np.float64)
    max_throttle_command = float(
        metadata.get("max_throttle_command_norm", 0.50)
        if max_throttle_command_override is None
        else max_throttle_command_override)
    windows = _fixed_validation_windows(
        data, state, max_windows_per_run,
        horizon_steps=max_rollout_steps,
        max_throttle_command=max_throttle_command,
        split=score_split)
    horizons = dict(HORIZON_STEPS)
    if max_rollout_steps == 400:
        horizons["10s"] = 400
    evaluation = evaluate_model(
        model, data, state, np.asarray(metadata["state_scale"]), device_name,
        windows=windows,
        wheel_state_source=str(metadata.get(
            "wheel_state_source", "filtered_odometry")),
        horizon_steps=horizons,
        command_offset_frames=command_offset_frames)
    report = {
        "schema_version": 1,
        "model": "deterministic_effective_race_teacher",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "external_dataset_evaluation": not same_dataset,
        "split_scored": score_split,
        "test_and_final_test_used": False,
        "primary_statistical_unit": "whole run; windows nested within run",
        "max_rollout_steps": max_rollout_steps,
        "max_throttle_command_norm": max_throttle_command,
        "command_offset_frames_from_target_state_row": int(
            command_offset_frames),
        "metrics": evaluation,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available()
                        else "cpu")
    parser.add_argument("--max-windows-per-run", type=int, default=24)
    parser.add_argument("--max-rollout-steps", type=int, choices=(200, 400),
                        default=200,
                        help="score at up to five or ten seconds using identical recursive starts")
    parser.add_argument("--max-throttle-command", type=float,
                        help="override checkpoint throttle cap to score both models on the same validation windows")
    parser.add_argument("--split", choices=("validation", "unseen_practice"),
                        default="validation",
                        help="whole-run split to score; test/final-test are intentionally unavailable")
    parser.add_argument("--allow-external-dataset", action="store_true",
                        help="score a separate schema-8/9 capture with identical feature layout; checkpoint normalizers stay fixed")
    parser.add_argument("--command-offset-frames", type=int, choices=(-1, 0),
                        default=0,
                        help="align each transition's command to target-state row (0) or preceding row (-1)")
    args = parser.parse_args()
    try:
        report = score(args.checkpoint, args.dataset, args.output,
                       args.device, args.max_windows_per_run,
                       args.max_rollout_steps,
                       args.max_throttle_command,
                       args.split,
                       args.allow_external_dataset,
                       args.command_offset_frames)
    except (OSError, ValueError, KeyError, IndexError, TypeError,
            FloatingPointError) as exc:
        print(f"EDSSM score failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "validation_runs": report["metrics"]["independent_run_count"],
        "windows": report["metrics"]["window_count"],
        "horizons": {
            horizon: {
                field: result["macro_run_mean"]
                for field, result in metrics.items()
            }
            for horizon, metrics in report["metrics"]["horizons"].items()
        },
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

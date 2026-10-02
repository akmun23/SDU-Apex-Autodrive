#!/usr/bin/env python3
"""Fit a command-driven 3D rigid-body plant teacher from Explore captures.

The recurrent residual predicts effective body-frame linear acceleration,
body angular acceleration, actuator rates, and rear-wheel speed rates. Rigid
body transport, quaternion attitude, and world-position integration remain
explicit. After the initial history, the rollout consumes only its own state,
the integrated attitude, and the recorded command trace. Simulator telemetry
is used only for training labels and scoring; this is an offline teacher, not
a competition-runtime component.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.vehicle_dynamics_learning.train_nssm import (
    SIMULATOR_DT_S,
    _load_dataset,
    _torch,
)
from tools.vehicle_dynamics_learning.structured_body_models import (
    REAR_AXLE_TO_COM_X_M,
)
from tools.vehicle_dynamics_learning.family_condition_sampler import (
    build_sequence_sampler,
)
from tools.vehicle_dynamics_learning.race_domain_objectives import (
    RACE_SPEED_BIN_EDGES_MPS,
    race_speed_mismatch_weights,
)


STATE_NAMES = (
    "body_vx_mps", "body_vy_mps", "body_vz_mps",
    "body_wx_rps", "body_wy_rps", "body_wz_rps",
    "steering_feedback_rad", "throttle_feedback_norm",
    "rear_left_surface_mps", "rear_right_surface_mps",
)
COMMAND_NAMES = ("steering_command_rad", "throttle_command_norm")
HISTORY_STEPS = 16
DEFAULT_HORIZONS = (1, 4, 10, 20, 40, 80)


def _torch_model(torch, nn, hidden_size: int, state_mean: np.ndarray,
                 state_scale: np.ndarray, command_mean: np.ndarray,
                 command_scale: np.ndarray, rate_mean: np.ndarray,
                 rate_scale: np.ndarray):
    class RigidBodyTeacher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("state_mean", torch.as_tensor(state_mean))
            self.register_buffer("state_scale", torch.as_tensor(state_scale))
            self.register_buffer("command_mean", torch.as_tensor(command_mean))
            self.register_buffer("command_scale", torch.as_tensor(command_scale))
            self.register_buffer("rate_mean", torch.as_tensor(rate_mean))
            self.register_buffer("rate_scale", torch.as_tensor(rate_scale))
            self.cell = nn.GRUCell(15, hidden_size)
            self.residual = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 10),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

        @staticmethod
        def _cross(left, right):
            return torch.linalg.cross(left, right, dim=-1)

        @classmethod
        def _quat_multiply(cls, left, right):
            lv, lw = left[..., :3], left[..., 3:4]
            rv, rw = right[..., :3], right[..., 3:4]
            vector = lw * rv + rw * lv + cls._cross(lv, rv)
            scalar = lw * rw - torch.sum(lv * rv, dim=-1, keepdim=True)
            return torch.cat((vector, scalar), dim=-1)

        @staticmethod
        def _rotation_vector_quaternion(rotation_vector):
            angle = torch.linalg.vector_norm(rotation_vector, dim=-1,
                                             keepdim=True)
            half = 0.5 * angle
            small = angle < 1.0e-6
            ratio = torch.where(
                small,
                0.5 - angle * angle / 48.0,
                torch.sin(half) / torch.clamp(angle, min=1.0e-12),
            )
            return torch.cat((rotation_vector * ratio, torch.cos(half)),
                             dim=-1)

        @classmethod
        def _rotate(cls, quaternion, vector):
            qv = quaternion[..., :3]
            twice_cross = 2.0 * cls._cross(qv, vector)
            return vector + quaternion[..., 3:4] * twice_cross + cls._cross(
                qv, twice_cross)

        def _features(self, state, quaternion, command):
            down_world = torch.zeros_like(state[..., :3])
            down_world[..., 2] = -1.0
            gravity_body = self._rotate(
                torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1),
                down_world,
            )
            normalized_state = (state - self.state_mean) / self.state_scale
            normalized_command = ((command - self.command_mean)
                                  / self.command_scale)
            return torch.cat((normalized_state, gravity_body,
                              normalized_command), dim=-1)

        def update_hidden(self, state, quaternion, command, hidden):
            return self.cell(self._features(state, quaternion, command), hidden)

        def transition(self, state, quaternion, position, command, hidden,
                       dt):
            hidden = self.update_hidden(state, quaternion, command, hidden)
            normalized_rate = self.residual(hidden)
            rates = normalized_rate * self.rate_scale + self.rate_mean

            velocity = state[..., :3]
            omega = state[..., 3:6]
            alpha = rates[..., 3:6]
            omega_next = omega + dt[..., None] * alpha
            omega_mid = 0.5 * (omega + omega_next)

            # Solve the implicit midpoint body-frame transport equation:
            # dv/dt = a_eff - omega x v. This preserves the exact rigid-body
            # coupling while the recurrent head learns only effective forces.
            wx, wy, wz = omega_mid.unbind(dim=-1)
            zero = torch.zeros_like(wx)
            skew = torch.stack((
                zero, -wz, wy,
                wz, zero, -wx,
                -wy, wx, zero,
            ), dim=-1).reshape(*omega_mid.shape[:-1], 3, 3)
            identity = torch.eye(3, dtype=state.dtype,
                                 device=state.device).expand_as(skew)
            half_dt_skew = 0.5 * dt[..., None, None] * skew
            rhs = torch.matmul(identity - half_dt_skew,
                               velocity.unsqueeze(-1)).squeeze(-1)
            rhs = rhs + dt[..., None] * rates[..., :3]
            velocity_next = torch.linalg.solve(
                identity + half_dt_skew, rhs.unsqueeze(-1)).squeeze(-1)

            auxiliary_next = (state[..., 6:]
                              + dt[..., None] * rates[..., 6:])
            next_state = torch.cat((velocity_next, omega_next,
                                    auxiliary_next), dim=-1)

            q_mid_delta = self._rotation_vector_quaternion(
                omega_mid * (0.5 * dt[..., None]))
            q_mid = self._quat_multiply(quaternion, q_mid_delta)
            q_delta = self._rotation_vector_quaternion(omega_mid * dt[..., None])
            next_quaternion = self._quat_multiply(quaternion, q_delta)
            next_quaternion = next_quaternion / torch.clamp(
                torch.linalg.vector_norm(next_quaternion, dim=-1, keepdim=True),
                min=1.0e-8,
            )

            velocity_mid = 0.5 * (velocity + velocity_next)
            rear_offset_body = torch.zeros_like(velocity_mid)
            rear_offset_body[..., 0] = -REAR_AXLE_TO_COM_X_M
            rear_velocity_mid = velocity_mid + self._cross(
                omega_mid, rear_offset_body)
            next_position = (position
                             + self._rotate(q_mid, rear_velocity_mid)
                             * dt[..., None])
            return next_state, next_quaternion, next_position, hidden, rates

    return RigidBodyTeacher


def _torch_arrays(torch, batch: dict[str, np.ndarray], device):
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in batch.items()
    }


def _state_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rigid = data["simulator_rigid_state"]
    if rigid is None or data["schema_version"] < 6:
        raise ValueError("the 3D rigid-body teacher requires schema-6 state labels")
    state = np.column_stack((
        rigid[:, 7:13], data["frames"][:, 3:7],
    )).astype(np.float32, copy=False)
    quaternion = rigid[:, 3:7].astype(np.float32, copy=False)
    position = rigid[:, :3].astype(np.float32, copy=False)
    if not (np.isfinite(state).all() and np.isfinite(quaternion).all()
            and np.isfinite(position).all()):
        raise ValueError("all selected 3D teacher rows must have complete finite labels")
    return state, quaternion, position


def _normalizers(data, state, train_run_indices):
    seq_mask = np.isin(data["seq_run"], np.asarray(train_run_indices,
                                                    dtype=np.int32))
    row_mask = np.zeros(len(state), dtype=bool)
    for start, end in data["bounds"][seq_mask]:
        row_mask[int(start):int(end)] = True
    train_state = state[row_mask].astype(np.float64)
    train_commands = data["frames"][row_mask, 7:9].astype(np.float64)
    state_mean = train_state.mean(axis=0).astype(np.float32)
    state_scale = np.maximum(train_state.std(axis=0), np.asarray(
        [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.02, 0.02,
         0.10, 0.10], dtype=np.float64)).astype(np.float32)
    command_mean = train_commands.mean(axis=0).astype(np.float32)
    command_scale = np.maximum(train_commands.std(axis=0),
                               np.asarray([0.02, 0.02])).astype(np.float32)

    rate_rows = []
    for seq_id, (start_raw, end_raw) in enumerate(data["bounds"]):
        if not seq_mask[seq_id]:
            continue
        start, end = int(start_raw), int(end_raw)
        current = state[start:end - 1].astype(np.float64)
        following = state[start + 1:end].astype(np.float64)
        dt = data["dt_s"][start + 1:end].astype(np.float64)
        rates = (following - current) / dt[:, None]
        velocity_mid = 0.5 * (current[:, :3] + following[:, :3])
        omega_mid = 0.5 * (current[:, 3:6] + following[:, 3:6])
        rates[:, :3] += np.cross(omega_mid, velocity_mid)
        rate_rows.append(rates)
    if not rate_rows:
        raise ValueError("no training transitions available for rate scaling")
    train_rates = np.concatenate(rate_rows, axis=0)
    rate_mean = train_rates.mean(axis=0).astype(np.float32)
    rate_scale = np.maximum(train_rates.std(axis=0), np.asarray(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.10, 0.10, 0.5, 0.5],
        dtype=np.float64)).astype(np.float32)
    return state_mean, state_scale, command_mean, command_scale, rate_mean, rate_scale


def _groups(data, split, history_steps, rollout_steps):
    groups: dict[int, list[int]] = {}
    for seq_id, (start, end) in enumerate(data["bounds"]):
        run = int(data["seq_run"][seq_id])
        if data["splits"][run] != split:
            continue
        if int(end - start) >= history_steps + rollout_steps:
            groups.setdefault(run, []).append(seq_id)
    return groups


def _sample_indices(data, groups, batch_size, history_steps, rollout_steps,
                    rng, family_sampler=None,
                    mismatch_thresholds_mps=None):
    run_ids = np.asarray(sorted(groups), dtype=np.int32)
    batch = {key: [] for key in ("history_state", "history_quaternion",
                                  "history_command", "initial_state",
                                  "initial_quaternion", "initial_position",
                                  "commands", "target_state",
                                  "target_quaternion", "target_position",
                                  "target_acceleration",
                                  "target_rate", "sample_weights")}
    state_all, quaternion_all, position_all = _SAMPLE_ARRAYS
    for _ in range(batch_size):
        if family_sampler is None:
            run = int(rng.choice(run_ids))
            seq_id = int(rng.choice(groups[run]))
        else:
            run, seq_id, _, _ = family_sampler.sample(rng)
            seq_id = int(seq_id)
        seq_start, seq_end = map(int, data["bounds"][seq_id])
        low = seq_start + history_steps - 1
        high = seq_end - rollout_steps - 1
        current = int(rng.integers(low, high + 1))
        history_begin = current - history_steps + 1
        target_indices = np.arange(current + 1,
                                   current + rollout_steps + 1,
                                   dtype=np.int64)
        batch["history_state"].append(state_all[history_begin:current + 1])
        batch["history_quaternion"].append(
            quaternion_all[history_begin:current + 1])
        batch["history_command"].append(
            data["frames"][history_begin:current + 1, 7:9])
        batch["initial_state"].append(state_all[current])
        batch["initial_quaternion"].append(quaternion_all[current])
        batch["initial_position"].append(position_all[current])
        batch["commands"].append(data["frames"][current:current + rollout_steps,
                                                   7:9])
        batch["target_state"].append(state_all[target_indices])
        batch["target_quaternion"].append(quaternion_all[target_indices])
        batch["target_position"].append(position_all[target_indices])
        acceleration = data["simulator_linear_acceleration"]
        batch["target_acceleration"].append(0.5 * (
            acceleration[current:current + rollout_steps]
            + acceleration[target_indices]))
        previous = np.concatenate((
            state_all[current:current + 1],
            state_all[target_indices[:-1]],
        ), axis=0).astype(np.float64)
        following = state_all[target_indices].astype(np.float64)
        rates = (following - previous) / SIMULATOR_DT_S
        velocity_mid = 0.5 * (previous[:, :3] + following[:, :3])
        omega_mid = 0.5 * (previous[:, 3:6] + following[:, 3:6])
        rates[:, :3] += np.cross(omega_mid, velocity_mid)
        batch["target_rate"].append(rates.astype(np.float32))
        if mismatch_thresholds_mps is not None:
            target_speed = np.hypot(state_all[target_indices, 0],
                                    state_all[target_indices, 1])
            target_mismatch = np.abs(
                0.5 * (data["frames"][target_indices, 5]
                       + data["frames"][target_indices, 6])
                - data["frames"][target_indices, 0])
            batch["sample_weights"].append(race_speed_mismatch_weights(
                target_speed, target_mismatch, mismatch_thresholds_mps))
        else:
            batch["sample_weights"].append(
                np.ones(rollout_steps, dtype=np.float32))
    return {key: np.asarray(value, dtype=np.float32)
            for key, value in batch.items()}


def _prepare_tensor_batch(torch, batch, device):
    return _torch_arrays(torch, batch, device)


def _training_mismatch_thresholds(data, groups) -> tuple[float, float]:
    values = []
    for sequences in groups.values():
        for sequence_id in sequences:
            start, end = map(int, data["bounds"][sequence_id])
            frames = data["frames"][start:end]
            values.append(np.abs(0.5 * (frames[:, 5] + frames[:, 6])
                                 - frames[:, 0]))
    if not values:
        raise ValueError("no training-run wheel/body mismatch proxy values")
    thresholds = np.quantile(np.concatenate(values), (0.50, 0.90))
    if thresholds[1] <= thresholds[0]:
        raise ValueError("training wheel/body mismatch proxy has no spread")
    return float(thresholds[0]), float(thresholds[1])


def _rollout(model, batch, dt_s: float, history_steps: int,
             state_scale, position_weight: float, attitude_weight: float,
             rate_weight: float, acceleration_weight: float):
    torch, _ = _torch()
    history_state = batch["history_state"]
    hidden = torch.zeros(history_state.shape[0], model.cell.hidden_size,
                         dtype=history_state.dtype,
                         device=history_state.device)
    for index in range(history_steps - 1):
        hidden = model.update_hidden(
            history_state[:, index], batch["history_quaternion"][:, index],
            batch["history_command"][:, index], hidden)
    state = batch["initial_state"]
    quaternion = batch["initial_quaternion"]
    position = batch["initial_position"]
    dt = torch.full((state.shape[0],), dt_s, dtype=state.dtype,
                    device=state.device)
    predicted_state = []
    predicted_quaternion = []
    predicted_position = []
    predicted_rate = []
    for index in range(batch["commands"].shape[1]):
        state, quaternion, position, hidden, rates = model.transition(
            state, quaternion, position, batch["commands"][:, index], hidden,
            dt)
        predicted_state.append(state)
        predicted_quaternion.append(quaternion)
        predicted_position.append(position)
        predicted_rate.append(rates)
    predicted_state = torch.stack(predicted_state, dim=1)
    predicted_quaternion = torch.stack(predicted_quaternion, dim=1)
    predicted_position = torch.stack(predicted_position, dim=1)
    predicted_rate = torch.stack(predicted_rate, dim=1)

    state_error = ((predicted_state - batch["target_state"])
                   / state_scale[None, None, :])
    sample_weights = batch.get("sample_weights")

    def weighted_time_mean(values):
        if values.ndim == 3:
            values = values.mean(dim=-1)
        if sample_weights is None:
            return torch.mean(values)
        weights = sample_weights.to(dtype=values.dtype, device=values.device)
        return torch.sum(values * weights) / torch.clamp(
            torch.sum(weights), min=1.0e-12)

    state_loss = weighted_time_mean(torch.nn.functional.smooth_l1_loss(
        state_error, torch.zeros_like(state_error), beta=0.5,
        reduction="none"))
    position_error_m = (predicted_position - batch["target_position"])
    position_loss = weighted_time_mean(torch.nn.functional.smooth_l1_loss(
        position_error_m / 0.15, torch.zeros_like(position_error_m), beta=1.0,
        reduction="none"))
    relative_q = model._quat_multiply(
        torch.cat((-predicted_quaternion[..., :3],
                   predicted_quaternion[..., 3:4]), dim=-1),
        batch["target_quaternion"],
    )
    relative_vector_norm = torch.linalg.vector_norm(relative_q[..., :3], dim=-1)
    attitude_error = 2.0 * torch.atan2(relative_vector_norm,
                                      torch.abs(relative_q[..., 3]))
    attitude_loss = weighted_time_mean(torch.nn.functional.smooth_l1_loss(
        attitude_error / 0.05, torch.zeros_like(attitude_error), beta=1.0,
        reduction="none"))
    normalized_rate_error = ((predicted_rate - batch["target_rate"])
                             / model.rate_scale[None, None, :])
    rate_loss = weighted_time_mean(torch.nn.functional.smooth_l1_loss(
        normalized_rate_error, torch.zeros_like(normalized_rate_error),
        beta=0.5, reduction="none"))
    normalized_acceleration_error = (
        (predicted_rate[..., :3] - batch["target_acceleration"])
        / model.rate_scale[None, None, :3])
    acceleration_loss = weighted_time_mean(torch.nn.functional.smooth_l1_loss(
        normalized_acceleration_error,
        torch.zeros_like(normalized_acceleration_error), beta=0.5,
        reduction="none"))
    total = (state_loss + position_weight * position_loss
             + attitude_weight * attitude_loss
             + rate_weight * rate_loss
             + acceleration_weight * acceleration_loss)
    return (total, state_loss, position_loss, attitude_loss, rate_loss,
            acceleration_loss,
            predicted_state, predicted_quaternion, predicted_position)


def _make_validation_batch(data, groups, history_steps, rollout_steps,
                           max_windows=48):
    state_all, quaternion_all, position_all = _SAMPLE_ARRAYS
    rows = []
    for run in sorted(groups):
        run_rows = []
        for seq_id in groups[run]:
            start, end = map(int, data["bounds"][seq_id])
            low = start + history_steps - 1
            high = end - rollout_steps - 1
            if high < low:
                continue
            count = min(16, high - low + 1)
            starts = np.linspace(low, high, count, dtype=np.int64)
            run_rows.extend((seq_id, int(value)) for value in starts)
        if len(run_rows) > max_windows // max(1, len(groups)):
            selected = np.linspace(0, len(run_rows) - 1,
                                   max_windows // len(groups), dtype=np.int64)
            run_rows = [run_rows[index] for index in selected]
        rows.extend(run_rows)
    output = {key: [] for key in ("history_state", "history_quaternion",
                                   "history_command", "initial_state",
                                   "initial_quaternion", "initial_position",
                                   "commands", "target_state",
                                   "target_quaternion", "target_position",
                                   "target_acceleration",
                                   "target_rate")}
    for seq_id, current in rows:
        seq_start, _ = map(int, data["bounds"][seq_id])
        history_start = current - history_steps + 1
        target_idx = np.arange(current + 1,
                               current + rollout_steps + 1, dtype=np.int64)
        output["history_state"].append(state_all[history_start:current + 1])
        output["history_quaternion"].append(
            quaternion_all[history_start:current + 1])
        output["history_command"].append(
            data["frames"][history_start:current + 1, 7:9])
        output["initial_state"].append(state_all[current])
        output["initial_quaternion"].append(quaternion_all[current])
        output["initial_position"].append(position_all[current])
        output["commands"].append(data["frames"][current:current + rollout_steps,
                                                    7:9])
        output["target_state"].append(state_all[target_idx])
        output["target_quaternion"].append(quaternion_all[target_idx])
        output["target_position"].append(position_all[target_idx])
        acceleration = data["simulator_linear_acceleration"]
        output["target_acceleration"].append(0.5 * (
            acceleration[current:current + rollout_steps]
            + acceleration[target_idx]))
        previous = np.concatenate((
            state_all[current:current + 1], state_all[target_idx[:-1]],
        ), axis=0).astype(np.float64)
        following = state_all[target_idx].astype(np.float64)
        rates = (following - previous) / SIMULATOR_DT_S
        velocity_mid = 0.5 * (previous[:, :3] + following[:, :3])
        omega_mid = 0.5 * (previous[:, 3:6] + following[:, 3:6])
        rates[:, :3] += np.cross(omega_mid, velocity_mid)
        output["target_rate"].append(rates.astype(np.float32))
    if not rows:
        raise ValueError("validation split has no sequence long enough for rollout")
    return {key: np.asarray(value, dtype=np.float32)
            for key, value in output.items()}


def _metrics(state_error, quaternion_error, position_error):
    state_rmse = np.sqrt(np.mean(state_error ** 2, axis=(0, 1)))
    position_radial = np.linalg.norm(position_error, axis=-1)
    return {
        "state_rmse": dict(zip(STATE_NAMES, map(float, state_rmse))),
        "position_radial_rmse_m": float(np.sqrt(np.mean(position_radial ** 2))),
        "position_radial_p95_m": float(np.quantile(position_radial, 0.95)),
        "orientation_rmse_rad": float(np.sqrt(np.mean(quaternion_error ** 2))),
        "orientation_p95_rad": float(np.quantile(quaternion_error, 0.95)),
    }


def _score_validation(torch, model, batch, device, args):
    tensor_batch = _prepare_tensor_batch(torch, batch, device)
    model.eval()
    with torch.no_grad():
        result = _rollout(
            model, tensor_batch, SIMULATOR_DT_S, args.history_steps,
            model.state_scale, args.position_loss_weight,
            args.attitude_loss_weight, args.rate_label_loss_weight,
            args.acceleration_label_loss_weight)
    (total, state_loss, pos_loss, att_loss, rate_loss, acceleration_loss,
     pred_state, pred_q, pred_pos) = result
    state_error = (pred_state - tensor_batch["target_state"]).cpu().numpy()
    qrel = model._quat_multiply(
        torch.cat((-pred_q[..., :3], pred_q[..., 3:4]), dim=-1),
        tensor_batch["target_quaternion"])
    qerr = (2.0 * torch.atan2(torch.linalg.vector_norm(qrel[..., :3], dim=-1),
                              torch.abs(qrel[..., 3]))).cpu().numpy()
    pos_error = (pred_pos - tensor_batch["target_position"]).cpu().numpy()
    report = _metrics(state_error, qerr, pos_error)
    report.update({"loss": float(total.cpu()),
                   "normalized_state_loss": float(state_loss.cpu()),
                   "position_loss": float(pos_loss.cpu()),
                   "attitude_loss": float(att_loss.cpu()),
                   "rate_label_loss": float(rate_loss.cpu()),
                   "acceleration_label_loss": float(acceleration_loss.cpu())})
    return report


def _full_sequence_scores(torch, model, data, state_all, quaternion_all,
                          position_all, split, device, history_steps):
    reports = []
    model.eval()
    for seq_id, (start_raw, end_raw) in enumerate(data["bounds"]):
        run_index = int(data["seq_run"][seq_id])
        if data["splits"][run_index] != split:
            continue
        start, end = int(start_raw), int(end_raw)
        if end - start <= history_steps:
            continue
        states = state_all[start:end]
        quaternions = quaternion_all[start:end]
        positions = position_all[start:end]
        commands = data["frames"][start:end, 7:9]
        with torch.no_grad():
            state = torch.as_tensor(states[history_steps - 1],
                                    dtype=torch.float32, device=device)[None]
            quaternion = torch.as_tensor(quaternions[history_steps - 1],
                                         dtype=torch.float32, device=device)[None]
            position = torch.as_tensor(positions[history_steps - 1],
                                       dtype=torch.float32, device=device)[None]
            hidden = torch.zeros(1, model.cell.hidden_size, device=device)
            for local in range(history_steps - 1):
                h_state = torch.as_tensor(states[local:local + 1],
                                          dtype=torch.float32, device=device)
                h_quat = torch.as_tensor(quaternions[local:local + 1],
                                         dtype=torch.float32, device=device)
                h_cmd = torch.as_tensor(commands[local:local + 1],
                                        dtype=torch.float32, device=device)
                hidden = model.update_hidden(h_state, h_quat, h_cmd, hidden)
            pred_states=[]; pred_quats=[]; pred_positions=[]
            for local in range(history_steps - 1, len(states) - 1):
                cmd = torch.as_tensor(commands[local:local + 1],
                                      dtype=torch.float32, device=device)
                dt = torch.full((1,), SIMULATOR_DT_S, device=device)
                state, quaternion, position, hidden, _ = model.transition(
                    state, quaternion, position, cmd, hidden, dt)
                pred_states.append(state[0].cpu().numpy())
                pred_quats.append(quaternion[0].cpu().numpy())
                pred_positions.append(position[0].cpu().numpy())
        pred_states = np.asarray(pred_states)
        pred_quats = np.asarray(pred_quats)
        pred_positions = np.asarray(pred_positions)
        truth_states = states[history_steps:]
        truth_quats = quaternions[history_steps:]
        truth_positions = positions[history_steps:]
        rel = (Rotation.from_quat(truth_quats).inv()
               * Rotation.from_quat(pred_quats)).magnitude()
        position_error = pred_positions - truth_positions
        radial = np.linalg.norm(position_error[:, :2], axis=1)
        duration = (len(pred_states) * SIMULATOR_DT_S)
        horizons = {}
        for seconds in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0):
            index = int(round(seconds / SIMULATOR_DT_S)) - 1
            if index < len(pred_states):
                horizons[f"{seconds:g}s"] = {
                    "xy_error_m": float(np.linalg.norm(position_error[index, :2])),
                    "orientation_error_rad": float(rel[index]),
                    "body_velocity_error_mps": float(np.linalg.norm(
                        pred_states[index, :3] - truth_states[index, :3])),
                }
        reports.append({
            "run_id": str(data["run_ids"][run_index]),
            "sequence_index": int(seq_id),
            "samples": int(len(pred_states)),
            "free_run_duration_s": duration,
            "xy_position_rmse_m": float(np.sqrt(np.mean(radial ** 2))),
            "xy_position_p95_m": float(np.quantile(radial, 0.95)),
            "xy_position_endpoint_m": float(radial[-1]),
            "orientation_rmse_rad": float(np.sqrt(np.mean(rel ** 2))),
            "state_rmse": dict(zip(STATE_NAMES, map(float, np.sqrt(
                np.mean((pred_states - truth_states) ** 2, axis=0))))),
            "horizons": horizons,
        })
    return reports


def train(args):
    torch, nn = _torch()
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        torch.set_num_threads(1)
    device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
              else "cpu" if args.device == "auto" else args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this PyTorch environment")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = _load_dataset(args.dataset)
    if data["schema_version"] < 6:
        raise ValueError("3D teacher requires schema-6 data")
    if (data["simulator_linear_acceleration"] is None
            or not np.isfinite(data["simulator_linear_acceleration"]).all()):
        raise ValueError("3D teacher requires complete simulator acceleration labels")
    if not np.allclose(data["dt_s"], SIMULATOR_DT_S, rtol=0.0, atol=1.0e-7):
        raise ValueError("training requires exact simulator 25 ms time steps")
    state_all, quaternion_all, position_all = _state_arrays(data)
    global _SAMPLE_ARRAYS
    _SAMPLE_ARRAYS = state_all, quaternion_all, position_all

    train_groups = _groups(data, "train", args.history_steps,
                           args.rollout_steps)
    validation_groups = _groups(data, "validation", args.history_steps,
                                args.rollout_steps)
    if not train_groups or not validation_groups:
        raise ValueError("whole-run train and validation splits must both have usable sequences")
    family_sampler = build_sequence_sampler(data, train_groups)
    mismatch_thresholds_mps = (
        _training_mismatch_thresholds(data, train_groups)
        if data["schema_version"] >= 8 else None)
    # Keep feature/rate scales identical when comparing short- and long-
    # rollout fits. The 80-step floor is the shortest horizon in the model
    # evaluation suite; shorter training-only runs never define the scales.
    normalization_groups = _groups(
        data, "train", args.history_steps, max(DEFAULT_HORIZONS))
    if not normalization_groups:
        raise ValueError("no training runs support the normalization horizon")
    normalization_run_indices = np.asarray(
        sorted(normalization_groups), dtype=np.int32)
    state_mean, state_scale, command_mean, command_scale, rate_mean, rate_scale = (
        _normalizers(data, state_all, normalization_run_indices))
    validation_batch = _make_validation_batch(
        data, validation_groups, args.history_steps, args.rollout_steps,
        args.validation_windows)
    model_type = _torch_model(torch, nn, args.hidden_size, state_mean,
                              state_scale, command_mean, command_scale,
                              rate_mean, rate_scale)
    model = model_type().to(device)
    if args.initialize_from is not None:
        initial = torch.load(args.initialize_from, map_location=device,
                             weights_only=False)
        metadata = initial.get("metadata", {})
        if metadata.get("architecture") != "causal_gru_effective_3d_rigid_body":
            raise ValueError("initial checkpoint architecture is incompatible")
        for key, current in (("state_mean", state_mean),
                             ("state_scale", state_scale),
                             ("command_mean", command_mean),
                             ("command_scale", command_scale),
                             ("rate_mean", rate_mean),
                             ("rate_scale", rate_scale)):
            if key not in metadata or not np.allclose(
                    np.asarray(metadata[key]), current, rtol=0.0, atol=1.0e-6):
                raise ValueError(
                    f"initial checkpoint {key} does not match this training split")
        model.load_state_dict(initial["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best_score = math.inf
    best_step = 0
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        batch = _prepare_tensor_batch(torch, _sample_indices(
            data, train_groups, args.batch_size, args.history_steps,
            args.rollout_steps, rng, family_sampler,
            mismatch_thresholds_mps), device)
        losses = _rollout(
            model, batch, SIMULATOR_DT_S, args.history_steps,
            model.state_scale, args.position_loss_weight,
            args.attitude_loss_weight, args.rate_label_loss_weight,
            args.acceleration_label_loss_weight)
        (loss, state_loss, position_loss, attitude_loss, rate_loss,
         acceleration_loss) = losses[:6]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            validation = _score_validation(torch, model, validation_batch,
                                           device, args)
            long_validation_rows = _full_sequence_scores(
                torch, model, data, state_all, quaternion_all, position_all,
                "validation", device, args.history_steps)
            long_validation = [
                row for row in long_validation_rows
                if row["free_run_duration_s"] >= 10.0]
            if not long_validation:
                raise ValueError(
                    "validation split needs at least one complete run for "
                    "free-running checkpoint selection")
            long_position_rmse = float(np.mean([
                row["xy_position_rmse_m"] for row in long_validation]))
            long_attitude_rmse = float(np.mean([
                row["orientation_rmse_rad"] for row in long_validation]))
            long_validation_summary = {
                "long_runs": long_validation,
                "mean_xy_position_rmse_m": long_position_rmse,
                "mean_orientation_rmse_rad": long_attitude_rmse,
            }
            score = (validation["loss"]
                     + args.full_run_position_loss_weight * long_position_rmse
                     + args.full_run_attitude_loss_weight * long_attitude_rmse)
            validation["full_run_validation"] = long_validation_summary
            validation["model_selection_score"] = score
            row = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "train_state_loss": float(state_loss.detach().cpu()),
                "train_position_loss": float(position_loss.detach().cpu()),
                "train_attitude_loss": float(attitude_loss.detach().cpu()),
                "train_rate_label_loss": float(rate_loss.detach().cpu()),
                "train_acceleration_label_loss": float(
                    acceleration_loss.detach().cpu()),
                "validation": validation,
            }
            history.append(row)
            print(
                f"step={step}/{args.steps} train={row['train_loss']:.6f} "
                f"val={score:.6f} window_xy={validation['position_radial_rmse_m']:.3f}m "
                f"lap_xy={long_position_rmse:.2f}m "
                f"lap_att={long_attitude_rmse:.3f}rad "
                f"elapsed={time.monotonic() - started:.1f}s device={device}",
                flush=True,
            )
            if score < best_score:
                best_score, best_step = score, step
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "architecture": "causal_gru_effective_3d_rigid_body",
                        "state_names": list(STATE_NAMES),
                        "command_names": list(COMMAND_NAMES),
                        "history_steps": args.history_steps,
                        "rollout_steps": args.rollout_steps,
                        "hidden_size": args.hidden_size,
                        "simulator_dt_s": SIMULATOR_DT_S,
                        "state_mean": state_mean,
                        "state_scale": state_scale,
                        "command_mean": command_mean,
                        "command_scale": command_scale,
                        "rate_mean": rate_mean,
                        "rate_scale": rate_scale,
                        "training_run_ids": [str(data["run_ids"][i])
                                             for i in sorted(train_groups)],
                        "training_sampler": family_sampler.metadata,
                        "race_domain_training_loss_weighting": ({
                            "speed_bin_edges_mps": list(
                                RACE_SPEED_BIN_EDGES_MPS),
                            "wheel_body_mismatch_proxy_thresholds_mps":
                                mismatch_thresholds_mps,
                            "wheel_body_mismatch_proxy_definition": (
                                "abs(mean(rear wheel surface speed) - rear axle u); "
                                "not a tire-slip measurement"),
                            "method": (
                                "multiply equal-population speed-bin and "
                                "train-quantile mismatch-bin weights; renormalize "
                                "to mean one per sampled rollout"),
                        } if mismatch_thresholds_mps is not None else None),
                        "normalization_run_ids": [
                            str(data["run_ids"][i])
                            for i in normalization_run_indices],
                        "validation_run_ids": [
                            str(data["run_ids"][i])
                            for i in sorted(validation_groups)],
                        "frame_contract": (
                            "body-frame simulator linear velocity at COM and "
                            "angular velocity; world-frame quaternion and "
                            "rear-axle position. Pose uses v_rear=v_com+"
                            "omega x [-L,0,0]; feedback and wheel speeds plus "
                            "command-only future inputs"),
                        "direct_truth_input_after_initial_history": False,
                        "full_run_position_loss_weight":
                            args.full_run_position_loss_weight,
                        "full_run_attitude_loss_weight":
                            args.full_run_attitude_loss_weight,
                        "rate_label_loss_weight": args.rate_label_loss_weight,
                        "acceleration_label_loss_weight":
                            args.acceleration_label_loss_weight,
                    },
                }
                torch.save(checkpoint, args.output_dir / "best_teacher.pt")

    checkpoint = torch.load(args.output_dir / "best_teacher.pt",
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    holdout_reports = {}
    for split in args.score_splits:
        holdout_reports[split] = _full_sequence_scores(
            torch, model, data, state_all, quaternion_all, position_all,
            split, device, args.history_steps)
    report = {
        "architecture": checkpoint["metadata"]["architecture"],
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "initialized_from": (str(args.initialize_from)
                             if args.initialize_from is not None else None),
        "schema_version": int(data["schema_version"]),
        "sample_count": int(len(data["frames"])),
        "run_split_counts": {
            split: int(np.count_nonzero(data["splits"] == split))
            for split in np.unique(data["splits"])},
        "train_runs": checkpoint["metadata"]["training_run_ids"],
        "training_sampler": family_sampler.metadata,
        "race_domain_training_loss_weighting": checkpoint["metadata"][
            "race_domain_training_loss_weighting"],
        "normalization_runs": checkpoint["metadata"]["normalization_run_ids"],
        "validation_runs": checkpoint["metadata"]["validation_run_ids"],
        "best_validation_step": int(best_step),
        "best_validation_score": float(best_score),
        "history": history,
        "free_running_holdouts": holdout_reports,
        "holdout_policy": (
            "holdout scores are diagnostic only and were not used for "
            "checkpoint selection; some source runs were inspected during "
            "earlier 2D-model work, so a new capture is still required for "
            "a blind confirmation"),
        "training_seconds": float(time.monotonic() - started),
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "best_validation_step": best_step,
        "best_validation_score": best_score,
        "report": str(args.output_dir / "training_report.json"),
        "holdout_runs_scored": {
            split: len(rows) for split, rows in holdout_reports.items()},
    }, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--history-steps", type=int, default=HISTORY_STEPS)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-windows", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--position-loss-weight", type=float, default=0.25)
    parser.add_argument("--attitude-loss-weight", type=float, default=0.20)
    parser.add_argument("--full-run-position-loss-weight", type=float,
                        default=2.0)
    parser.add_argument("--full-run-attitude-loss-weight", type=float,
                        default=1.0)
    parser.add_argument("--rate-label-loss-weight", type=float, default=0.15)
    parser.add_argument("--acceleration-label-loss-weight", type=float,
                        default=1.0)
    parser.add_argument("--score-splits", nargs="+", default=["final_test"])
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.rollout_steps < max(DEFAULT_HORIZONS):
        parser.error("steps/batch-size must be positive and rollout must cover 80 steps")
    train(args)


if __name__ == "__main__":
    main()

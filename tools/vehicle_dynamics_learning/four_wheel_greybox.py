#!/usr/bin/env python3
"""Differentiable, guide-structured four-wheel grey-box plant experiment.

The model is offline-only. Geometry, mass, tire slip definitions, and the
piecewise tire-force landmarks come from AutoDRIVE's published vehicle guide.
Unreported tire spline tangents, wheel inertia, effective load-transfer
height, drivetrain scale, and drag are shared trainable parameters. Rear
wheel speed is supervised; front wheel speed is a latent propagated state.
Actuator feedback is supplied as a *measured conditioning input* in this
identification experiment, so this is not yet a command-only plant.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch
from tools.vehicle_dynamics_learning.experiment_artifacts import (
    write_standard_artifacts,
)


DT_S = 0.025
MASS_KG = 3.906
GRAVITY_MPS2 = 9.80665
WHEELBASE_M = 0.324
TRACK_M = 0.236
WHEEL_RADIUS_M = 0.059
COM_X_M = 0.15532
COM_Z_M = 0.01434
FRONT_X_M = WHEELBASE_M - COM_X_M
REAR_X_M = -COM_X_M
HALF_TRACK_M = TRACK_M / 2.0
WHEEL_X = (FRONT_X_M, FRONT_X_M, REAR_X_M, REAR_X_M)
WHEEL_Y = (HALF_TRACK_M, -HALF_TRACK_M,
           HALF_TRACK_M, -HALF_TRACK_M)
LONGITUDINAL_KNOTS = (0.15, 0.72, 0.25, 0.464)
LATERAL_KNOTS = (0.01, 1.00, 0.10, 0.500)
OUTPUT_NAMES = (
    "u_com_mps", "v_com_mps", "yaw_rate_rps",
    "rear_left_surface_mps", "rear_right_surface_mps",
    "steering_feedback_rad", "throttle_feedback_norm",
)
ACCELERATION_NAMES = ("ax_body_mps2", "ay_body_mps2", "yaw_acceleration_rps2")


def _logit(value: float) -> float:
    value = min(max(value, 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


def _physical_model(torch, nn, use_tire_relaxation: bool = True):
    class FourWheelGreyBox(nn.Module):
        """Planar rigid body plus four driven wheel-rotation states."""

        PARAMETER_BOUNDS = {
            "yaw_inertia_kgm2": (0.005, 0.080),
            "wheel_inertia_kgm2": (0.00001, 0.005),
            "motor_torque_scale": (0.0, 428.0),
            "idle_brake_torque_nm": (0.0, 50.0),
            "wheel_damping_nms": (0.0, 0.01),
            "rolling_resistance_n": (0.0, 10.0),
            "quadratic_drag_n_per_mps2": (0.0, 0.10),
            "longitudinal_force_scale": (0.2, 2.0),
            "lateral_force_scale": (0.2, 2.0),
            "longitudinal_zero_slip_tangent": (0.0, 3.0),
            "lateral_zero_slip_tangent": (0.0, 3.0),
            "effective_load_transfer_height_m": (0.0, 0.20),
            "front_drive_fraction": (0.05, 0.95),
            "load_transfer_time_constant_s": (0.01, 0.30),
            "steering_lag_time_constant_s": (0.001, 0.30),
            "steering_rate_limit_radps": (0.5, 8.0),
            "throttle_lag_time_constant_s": (0.001, 1.0),
            "throttle_rate_limit_normps": (0.2, 20.0),
            "longitudinal_tire_relaxation_time_s": (0.005, 0.25),
            "lateral_tire_relaxation_time_s": (0.005, 0.25),
        }
        INITIAL = {
            "yaw_inertia_kgm2": 0.030,
            "wheel_inertia_kgm2": 0.00025,
            # These are effective wheel-side values. The guide publishes a
            # motor torque but not its Unity-unit/gearbox mapping to each tire.
            "motor_torque_scale": 0.5,
            "idle_brake_torque_nm": 0.05,
            "wheel_damping_nms": 0.001,
            "rolling_resistance_n": 1.5,
            "quadratic_drag_n_per_mps2": 0.01,
            "longitudinal_force_scale": 1.0,
            "lateral_force_scale": 1.0,
            "longitudinal_zero_slip_tangent": 1.0,
            "lateral_zero_slip_tangent": 1.5,
            "effective_load_transfer_height_m": COM_Z_M,
            "front_drive_fraction": 0.5,
            "load_transfer_time_constant_s": 0.05,
            "steering_lag_time_constant_s": 0.025,
            "steering_rate_limit_radps": 3.2,
            "throttle_lag_time_constant_s": 0.05,
            "throttle_rate_limit_normps": 10.0,
            "longitudinal_tire_relaxation_time_s": 0.05,
            "lateral_tire_relaxation_time_s": 0.05,
        }

        def __init__(self) -> None:
            super().__init__()
            self.use_tire_relaxation = use_tire_relaxation
            initial = []
            for name, (low, high) in self.PARAMETER_BOUNDS.items():
                fraction = (self.INITIAL[name] - low) / (high - low)
                initial.append(_logit(fraction))
            self.raw_parameters = nn.Parameter(
                torch.tensor(initial, dtype=torch.float32))
            self.register_buffer("wheel_x", torch.tensor(WHEEL_X))
            self.register_buffer("wheel_y", torch.tensor(WHEEL_Y))
            static_front = MASS_KG * GRAVITY_MPS2 * COM_X_M / WHEELBASE_M / 2.0
            static_rear = MASS_KG * GRAVITY_MPS2 * (
                WHEELBASE_M - COM_X_M) / WHEELBASE_M / 2.0
            self.register_buffer("static_loads", torch.tensor(
                [static_front, static_front, static_rear, static_rear]))
            self.register_buffer("side_sign", torch.tensor(
                [-1.0, 1.0, -1.0, 1.0]))
            self.register_buffer("axle_sign", torch.tensor(
                [-1.0, -1.0, 1.0, 1.0]))

        def physical_parameters(self) -> dict[str, Any]:
            result = {}
            for index, (name, (low, high)) in enumerate(
                    self.PARAMETER_BOUNDS.items()):
                result[name] = low + (high - low) * torch.sigmoid(
                    self.raw_parameters[index])
            return result

        @staticmethod
        def _curve(abs_slip, extremum_slip: float, extremum_force: float,
                   asymptote_slip: float, asymptote_force: float,
                   zero_slip_tangent):
            first_t = torch.clamp(abs_slip / extremum_slip, 0.0, 1.0)
            first_h10 = first_t**3 - 2.0 * first_t**2 + first_t
            first_h01 = -2.0 * first_t**3 + 3.0 * first_t**2
            first = (zero_slip_tangent * extremum_slip * first_h10
                     + extremum_force * first_h01)
            second_t = torch.clamp(
                (abs_slip - extremum_slip)
                / (asymptote_slip - extremum_slip), 0.0, 1.0)
            second = (extremum_force
                      + (asymptote_force - extremum_force)
                      * (3.0 * second_t**2 - 2.0 * second_t**3))
            return torch.where(abs_slip < extremum_slip, first,
                               torch.where(abs_slip < asymptote_slip,
                                           second, asymptote_force))

        @staticmethod
        def _ackermann(delta):
            tangent = torch.tan(delta)
            numerator = 2.0 * WHEELBASE_M * tangent
            denominator_left = 2.0 * WHEELBASE_M + TRACK_M * tangent
            denominator_right = 2.0 * WHEELBASE_M - TRACK_M * tangent
            formula_left = torch.atan2(numerator, denominator_left)
            formula_right = torch.atan2(numerator, denominator_right)
            # x-forward/y-left: positive left steering makes the left wheel
            # the inside wheel. This is the coordinate-correct assignment
            # confirmed by the existing bridge/static-transform audit.
            return torch.stack((formula_right, formula_left), dim=-1)

        def _wheel_loads(self, load_ax, load_ay, params):
            transfer_long = (MASS_KG * load_ax
                             * params["effective_load_transfer_height_m"]
                             / (2.0 * WHEELBASE_M))
            transfer_lat = (MASS_KG * load_ay
                            * params["effective_load_transfer_height_m"]
                            / (2.0 * TRACK_M))
            return torch.clamp(
                self.static_loads + self.axle_sign * transfer_long[..., None]
                + self.side_sign * transfer_lat[..., None], min=0.0)

        def _forces_and_derivatives(self, state, steering, throttle):
            p = self.physical_parameters()
            u, v, yaw_rate = state[:, 0], state[:, 1], state[:, 2]
            omega = state[:, 3:7]
            load_ax, load_ay = state[:, 15], state[:, 16]
            wheel_x = self.wheel_x.to(dtype=state.dtype, device=state.device)
            wheel_y = self.wheel_y.to(dtype=state.dtype, device=state.device)
            wheel_vx_body = u[:, None] - yaw_rate[:, None] * wheel_y
            wheel_vy_body = v[:, None] + yaw_rate[:, None] * wheel_x
            front_angles = self._ackermann(steering)
            zero = torch.zeros_like(steering)
            wheel_angles = torch.stack(
                (front_angles[:, 0], front_angles[:, 1], zero, zero), dim=-1)
            cosine, sine = torch.cos(wheel_angles), torch.sin(wheel_angles)
            wheel_vx = cosine * wheel_vx_body + sine * wheel_vy_body
            wheel_vy = -sine * wheel_vx_body + cosine * wheel_vy_body
            surface = WHEEL_RADIUS_M * omega
            slip_x_target = ((surface - wheel_vx)
                             / torch.clamp(torch.abs(wheel_vx), min=0.5))
            slip_y_target = (wheel_vy
                             / torch.clamp(torch.abs(wheel_vx), min=0.5))
            if self.use_tire_relaxation:
                slip_x, slip_y = state[:, 7:11], state[:, 11:15]
                slip_x_dot = ((slip_x_target - slip_x)
                              / p["longitudinal_tire_relaxation_time_s"])
                slip_y_dot = ((slip_y_target - slip_y)
                              / p["lateral_tire_relaxation_time_s"])
            else:
                slip_x, slip_y = slip_x_target, slip_y_target
                slip_x_dot = torch.zeros_like(slip_x)
                slip_y_dot = torch.zeros_like(slip_y)
            loads = self._wheel_loads(load_ax, load_ay, p)
            fx_local = (torch.sign(slip_x) * loads
                        * p["longitudinal_force_scale"]
                        * self._curve(torch.abs(slip_x), *LONGITUDINAL_KNOTS[:2],
                                      *LONGITUDINAL_KNOTS[2:],
                                      p["longitudinal_zero_slip_tangent"]))
            fy_local = (-torch.sign(slip_y) * loads
                        * p["lateral_force_scale"]
                        * self._curve(torch.abs(slip_y), *LATERAL_KNOTS[:2],
                                      *LATERAL_KNOTS[2:],
                                      p["lateral_zero_slip_tangent"]))
            fx_body_wheel = cosine * fx_local - sine * fy_local
            fy_body_wheel = sine * fx_local + cosine * fy_local
            fx_tire = fx_body_wheel.sum(dim=-1)
            fy_tire = fy_body_wheel.sum(dim=-1)
            drive_front = (p["motor_torque_scale"] * throttle
                           * p["front_drive_fraction"] / 2.0)
            drive_rear = (p["motor_torque_scale"] * throttle
                          * (1.0 - p["front_drive_fraction"]) / 2.0)
            drive_torque = torch.stack((drive_front, drive_front,
                                        drive_rear, drive_rear), dim=-1)
            idle_torque = (p["idle_brake_torque_nm"]
                           * (1.0 - torch.clamp(
                               torch.abs(throttle), 0.0, 1.0))[:, None]
                           * torch.tanh(omega / 0.5))
            omega_dot = (drive_torque - WHEEL_RADIUS_M * fx_local
                         - idle_torque - p["wheel_damping_nms"] * omega)
            omega_dot = omega_dot / p["wheel_inertia_kgm2"]
            fx_drag = (p["rolling_resistance_n"] * torch.tanh(u / 0.1)
                       + p["quadratic_drag_n_per_mps2"] * u * torch.abs(u))
            ax = (fx_tire - fx_drag) / MASS_KG
            ay = fy_tire / MASS_KG
            moment = (wheel_x * fy_body_wheel
                      - wheel_y * fx_body_wheel).sum(dim=-1)
            yaw_accel = moment / p["yaw_inertia_kgm2"]
            tau = p["load_transfer_time_constant_s"]
            derivative = torch.cat((
                (ax + yaw_rate * v)[:, None],
                (ay - yaw_rate * u)[:, None],
                yaw_accel[:, None], omega_dot, slip_x_dot, slip_y_dot,
                ((ax - load_ax) / tau)[:, None],
                ((ay - load_ay) / tau)[:, None],
            ), dim=-1)
            acceleration = torch.stack((ax, ay, yaw_accel), dim=-1)
            return derivative, acceleration

        def _advance_actuator(self, current, command, dt):
            p = self.physical_parameters()
            tau = torch.stack((p["steering_lag_time_constant_s"],
                               p["throttle_lag_time_constant_s"]))
            rate = torch.stack((p["steering_rate_limit_radps"],
                                p["throttle_rate_limit_normps"]))
            lagged_change = (1.0 - torch.exp(-dt / tau)) * (command - current)
            maximum_change = rate * dt
            change = torch.maximum(
                torch.minimum(lagged_change, maximum_change), -maximum_change)
            following = current + change
            limits = torch.tensor((0.5236, 1.0), dtype=current.dtype,
                                  device=current.device)
            return torch.maximum(torch.minimum(following, limits), -limits)

        def step(self, state, command, dt: float = DT_S):
            """Advance from commands; actuator state remains internal."""
            actuator = state[:, 17:19]
            delayed_command = state[:, 19:21]
            actuator_next = self._advance_actuator(
                actuator, delayed_command, dt)
            actuator_mid = 0.5 * (actuator + actuator_next)
            derivative0, _ = self._forces_and_derivatives(
                state[:, :17], actuator[:, 0], actuator[:, 1])
            physical_mid = state[:, :17] + (0.5 * dt) * derivative0
            derivative_mid, acceleration_mid = self._forces_and_derivatives(
                physical_mid, actuator_mid[:, 0], actuator_mid[:, 1])
            following_physical = state[:, :17] + dt * derivative_mid
            following = torch.cat((following_physical, actuator_next, command),
                                  dim=-1)
            return following, acceleration_mid

        def rollout(self, initial_state, commands):
            """Roll out commands causally; no future feedback is supplied."""
            state = initial_state
            states, accelerations = [], []
            for index in range(commands.shape[1]):
                following, acceleration = self.step(state, commands[:, index])
                states.append(following)
                accelerations.append(acceleration)
                state = following
            return torch.stack(states, dim=1), torch.stack(accelerations, dim=1)

        def export_parameters(self) -> dict[str, float]:
            return {name: float(value.detach().cpu())
                    for name, value in self.physical_parameters().items()}

    return FourWheelGreyBox


def _targets(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rigid = data["simulator_rigid_state"]
    acceleration = data["simulator_linear_acceleration"]
    if rigid is None or acceleration is None:
        raise ValueError("four-wheel fit needs simulator rigid/acceleration labels")
    yaw_accel = np.full(len(rigid), np.nan, dtype=np.float32)
    for start_value, end_value in data["bounds"]:
        start, end = int(start_value), int(end_value)
        yaw_accel[start:end - 1] = np.diff(rigid[start:end, 12]) / DT_S
    physical_state = np.column_stack((
        rigid[:, 7], rigid[:, 8], rigid[:, 12],
        data["frames"][:, 5], data["frames"][:, 6],
    )).astype(np.float32)
    accelerations = np.column_stack((acceleration[:, :2], yaw_accel)).astype(np.float32)
    return physical_state, accelerations


def _groups(data: dict[str, Any], physical_state: np.ndarray,
            acceleration: np.ndarray, split: str,
            horizon_steps: int) -> dict[int, list[tuple[int, int]]]:
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for seq_id, (start_value, end_value) in enumerate(data["bounds"]):
        run = int(data["seq_run"][seq_id])
        if data["splits"][run] != split:
            continue
        start, end = int(start_value), int(end_value)
        if end - start <= horizon_steps:
            continue
        if (not np.isfinite(physical_state[start:end]).all()
                or not np.isfinite(acceleration[start:end - 1]).all()
                or not np.isfinite(data["frames"][start:end, 3:7]).all()
                or (data.get("sensor_frames") is not None
                    and (not np.isfinite(
                        data["sensor_frames"][start:end, 4:6]).all()
                         or not data["sensor_valid"][start:end].all()))):
            continue
        grouped[run].append((start, end))
    return dict(grouped)


def _initial_state(data: dict[str, Any], start: int,
                   physical_state: np.ndarray,
                   sequence_start: int) -> np.ndarray:
    u, v, yaw_rate = map(float, physical_state[start, :3])
    steering = float(data["frames"][start, 3])
    tangent = math.tan(steering)
    numerator = 2.0 * WHEELBASE_M * tangent
    left_formula = math.atan2(numerator, 2.0 * WHEELBASE_M + TRACK_M * tangent)
    right_formula = math.atan2(numerator, 2.0 * WHEELBASE_M - TRACK_M * tangent)
    front_angles = (right_formula, left_formula)
    wheel_omega = []
    for index, (x, y) in enumerate(zip(WHEEL_X, WHEEL_Y)):
        vx_body = u - yaw_rate * y
        vy_body = v + yaw_rate * x
        angle = front_angles[index] if index < 2 else 0.0
        vx_wheel = math.cos(angle) * vx_body + math.sin(angle) * vy_body
        if index < 2:
            wheel_omega.append(vx_wheel / WHEEL_RADIUS_M)
        else:
            rear_col = 3 + index - 2
            wheel_omega.append(float(physical_state[start, rear_col])
                               / WHEEL_RADIUS_M)
    steering_values = np.asarray((front_angles[0], front_angles[1], 0.0, 0.0))
    omega_values = np.asarray(wheel_omega)
    wheel_vx_body = u - yaw_rate * np.asarray(WHEEL_Y)
    wheel_vy_body = v + yaw_rate * np.asarray(WHEEL_X)
    wheel_vx = (np.cos(steering_values) * wheel_vx_body
                + np.sin(steering_values) * wheel_vy_body)
    wheel_vy = (-np.sin(steering_values) * wheel_vx_body
                + np.cos(steering_values) * wheel_vy_body)
    slip_denominator = np.maximum(np.abs(wheel_vx), 0.5)
    initial_slip_x = (WHEEL_RADIUS_M * omega_values - wheel_vx) / slip_denominator
    initial_slip_y = wheel_vy / slip_denominator
    if data.get("sensor_frames") is not None:
        # IMU is a legal causal initial signal for the latent load-transfer
        # state. It is not read after the rollout starts.
        load_accel = data["sensor_frames"][start, 4:6]
    else:
        load_accel = np.zeros(2, dtype=np.float32)
    steering_feedback, throttle_feedback = map(
        float, data["frames"][start, 3:5])
    if start > sequence_start:
        delayed_command = data["frames"][start - 1, 7:9]
    else:
        # No pre-reset command is part of this sequence. Hold the measured
        # actuator state through the first 25 ms, matching the observed
        # one-packet command/feedback latency without leaking prior-run data.
        delayed_command = data["frames"][start, 3:5]
    return np.asarray([u, v, yaw_rate, *wheel_omega,
                       *initial_slip_x, *initial_slip_y,
                       float(load_accel[0]), float(load_accel[1]),
                       steering_feedback, throttle_feedback,
                       *delayed_command],
                      dtype=np.float32)


def _sample_batch(data, physical_state, acceleration, groups,
                  batch_size, horizon_steps, rng):
    run_ids = np.asarray(sorted(groups), dtype=np.int32)
    initials, commands, state_targets, accel_targets = [], [], [], []
    for run_value in rng.choice(run_ids, size=batch_size, replace=True):
        sequences = groups[int(run_value)]
        sequence_start, end = sequences[int(rng.integers(0, len(sequences)))]
        start = sequence_start
        index = int(rng.integers(sequence_start, end - horizon_steps))
        initials.append(_initial_state(
            data, index, physical_state, sequence_start))
        commands.append(data["frames"][index:index + horizon_steps, 7:9])
        next_actuators = data["frames"][index + 1:index + horizon_steps + 1, 3:5]
        state_targets.append(np.column_stack((
            physical_state[index + 1:index + horizon_steps + 1],
            next_actuators)))
        accel_targets.append(acceleration[index:index + horizon_steps])
    return tuple(np.stack(rows).astype(np.float32) for rows in
                 (initials, commands, state_targets, accel_targets))


def _integrate_pose(predicted: np.ndarray, initial_state: np.ndarray,
                    initial_pose: np.ndarray):
    pose = np.empty((len(predicted), 3), dtype=np.float64)
    x, y, yaw = map(float, initial_pose)
    previous = initial_state[:3].astype(np.float64)
    for index, following in enumerate(predicted[:, :3]):
        u = 0.5 * (previous[0] + following[0])
        r = 0.5 * (previous[2] + following[2])
        v_rear = 0.5 * ((previous[1] - COM_X_M * previous[2])
                        + (following[1] - COM_X_M * following[2]))
        yaw_mid = yaw + 0.5 * r * DT_S
        x += (u * math.cos(yaw_mid) - v_rear * math.sin(yaw_mid)) * DT_S
        y += (u * math.sin(yaw_mid) + v_rear * math.cos(yaw_mid)) * DT_S
        yaw += r * DT_S
        pose[index] = (x, y, yaw)
        previous = following
    return pose


def _evaluate(torch, model, data, physical_state, acceleration, groups,
              body_scale, device, horizon_steps, seed,
              max_windows_per_run):
    rng = np.random.default_rng(seed)
    per_run = {}
    model.eval()
    horizons = [step for step in (10, 30, 40, 80)
                if step <= horizon_steps]
    with torch.no_grad():
        for run_id, sequences in sorted(groups.items()):
            states_at_horizon: dict[int, list[np.ndarray]] = defaultdict(list)
            poses_at_horizon: dict[int, list[np.ndarray]] = defaultdict(list)
            truth_at_horizon: dict[int, list[np.ndarray]] = defaultdict(list)
            for _ in range(max_windows_per_run):
                start, end = sequences[int(rng.integers(0, len(sequences)))]
                index = int(rng.integers(start, end - horizon_steps))
                initial = _initial_state(
                    data, index, physical_state, start)
                initial_t = torch.as_tensor(initial[None], dtype=torch.float32,
                                            device=device)
                commands = torch.as_tensor(
                    data["frames"][index:index + horizon_steps, 7:9][None],
                    dtype=torch.float32, device=device)
                prediction, _ = model.rollout(initial_t, commands)
                model_state = prediction[0].cpu().numpy()
                states = np.column_stack((
                    model_state[:, :3],
                    model_state[:, 5:7] * WHEEL_RADIUS_M,
                    model_state[:, 17:19],
                ))
                for step in horizons:
                    states_at_horizon[step].append(states[step - 1])
                    truth_at_horizon[step].append(np.concatenate((
                        physical_state[index + step],
                        data["frames"][index + step, 3:5])))
                if "simulator_pose_xyyaw" in data:
                    pose0 = data["simulator_pose_xyyaw"][index]
                    true_pose = data["simulator_pose_xyyaw"][
                        index + 1:index + horizon_steps + 1]
                    if np.isfinite(pose0).all() and np.isfinite(true_pose).all():
                        predicted_pose = _integrate_pose(
                            states[:, :3], physical_state[index], pose0)
                        for step in horizons:
                            poses_at_horizon[step].append(
                                predicted_pose[step - 1] - true_pose[step - 1])
            horizon_result = {}
            for step in horizons:
                predicted = np.stack(states_at_horizon[step])
                truth = np.stack(truth_at_horizon[step])
                error = predicted - truth
                row = {
                    "state_rmse": dict(zip(OUTPUT_NAMES,
                                            np.sqrt(np.mean(error**2, axis=0)).tolist())),
                    "normalized_body_state_rmse": float(np.sqrt(np.mean(
                        (error[:, :3] / body_scale[None, :]) ** 2))),
                    "window_count": len(error),
                }
                if poses_at_horizon[step]:
                    pose_error = np.stack(poses_at_horizon[step])
                    row["pose_xy_rmse_m"] = np.sqrt(
                        np.mean(pose_error[:, :2]**2, axis=0)).tolist()
                    row["heading_rmse_rad"] = float(np.sqrt(
                        np.mean(pose_error[:, 2]**2)))
                horizon_result[f"{step * DT_S:g}s"] = row
            per_run[str(data["run_ids"][run_id])] = {
                "window_count": max_windows_per_run,
                "horizons": horizon_result,
            }
    score_keys = [key for key in ("0.25s", "0.75s", "1s", "2s")
                  if all(key in row["horizons"] for row in per_run.values())]
    aggregate = {}
    for key in score_keys:
        values = []
        for row in per_run.values():
            values.append(row["horizons"][key][
                "normalized_body_state_rmse"])
        aggregate[key] = {
            "independent_run_count": len(values),
            "macro_run_body_state_rmse": float(np.mean(values)),
            "run_min": float(np.min(values)), "run_max": float(np.max(values)),
        }
    score = float(np.mean([aggregate[key]["macro_run_body_state_rmse"]
                           for key in score_keys])) if score_keys else None
    return {"eligible_runs": len(groups), "per_run": per_run,
            "macro_run_horizons": aggregate,
            "checkpoint_selection_score": score}


def train(dataset_path: Path, output_dir: Path, *, device_name: str,
          horizon_seconds: float, shooting_seconds: float,
          use_tire_relaxation: bool,
          batch_size: int, max_steps: int,
          eval_every: int, patience: int, max_eval_windows_per_run: int,
          seed: int, score_test: bool) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, nn = _torch()
    torch.set_num_threads(1)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    data = _load_dataset(dataset_path)
    physical_state, acceleration = _targets(data)
    horizon_steps = round(horizon_seconds / DT_S)
    shooting_steps = round(shooting_seconds / DT_S)
    if shooting_steps < 1 or shooting_steps > horizon_steps:
        raise ValueError("shooting horizon must be in (0, evaluation horizon]")
    groups = {split: _groups(data, physical_state, acceleration, split,
                             horizon_steps)
              for split in ("train", "validation", "test", "final_test")}
    if not groups["train"] or not groups["validation"]:
        raise ValueError("need eligible whole-run train and validation data")
    train_indices = np.concatenate([
        np.arange(start, end - 1)
        for sequences in groups["train"].values()
        for start, end in sequences
    ])
    train_feedback = data["frames"][train_indices, 3:5]
    state_scale = np.maximum(
        np.std(np.column_stack((physical_state[train_indices], train_feedback)),
               axis=0),
        [0.5, 0.25, 0.25, 0.5, 0.5, 0.1, 0.2]).astype(np.float32)
    acceleration_scale = np.maximum(
        np.nanstd(acceleration[train_indices], axis=0),
        [0.5, 0.5, 0.5]).astype(np.float32)
    model_type = _physical_model(torch, nn, use_tire_relaxation)
    model = model_type().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3,
                                  weight_decay=1.0e-5)
    rng = np.random.default_rng(seed)
    best_score, best_step, stale = math.inf, 0, 0
    history = []
    start_time = time.perf_counter()
    for step in range(1, max_steps + 1):
        initial, commands, target, accel_target = _sample_batch(
            data, physical_state, acceleration, groups["train"], batch_size,
            shooting_steps, rng)
        initial_t = torch.as_tensor(initial, dtype=torch.float32, device=device)
        commands_t = torch.as_tensor(commands, dtype=torch.float32,
                                     device=device)
        target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
        accel_target_t = torch.as_tensor(
            accel_target, dtype=torch.float32, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, predicted_acceleration = model.rollout(initial_t, commands_t)
        rear_wheel_surface = prediction[:, :, 5:7] * WHEEL_RADIUS_M
        output = torch.cat((prediction[:, :, :3], rear_wheel_surface,
                            prediction[:, :, 17:19]), dim=-1)
        state_loss = nn.functional.smooth_l1_loss(
            (output - target_t) / torch.as_tensor(
                state_scale, dtype=torch.float32, device=device),
            torch.zeros_like(target_t), beta=0.5)
        acceleration_loss = nn.functional.smooth_l1_loss(
            (predicted_acceleration - accel_target_t) / torch.as_tensor(
                acceleration_scale, dtype=torch.float32, device=device),
            torch.zeros_like(accel_target_t), beta=0.5)
        loss = state_loss + 0.15 * acceleration_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite grey-box loss at step {step}")
        loss.backward()
        if any(parameter.grad is not None and
               not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise FloatingPointError(
                f"non-finite grey-box gradient at step {step}; "
                "reduce the shooting window or revise the physical structure")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % eval_every == 0 or step == max_steps:
            validation = _evaluate(
                torch, model, data, physical_state, acceleration,
                groups["validation"], state_scale[:3], device,
                horizon_steps, seed + 2001,
                max_eval_windows_per_run)
            score = validation["checkpoint_selection_score"]
            history.append({
                "step": step, "training_loss": float(loss.detach().cpu()),
                "state_loss": float(state_loss.detach().cpu()),
                "acceleration_loss": float(acceleration_loss.detach().cpu()),
                "validation_score": score, "validation": validation,
                "parameters": model.export_parameters(),
            })
            print(json.dumps({"step": step,
                              "loss": float(loss.detach().cpu()),
                              "validation_score": score}, sort_keys=True),
                  flush=True)
            if score is not None and score < best_score:
                best_score, best_step, stale = score, step, 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "architecture": "four_wheel_guide_greybox",
                        "feature_names": data["feature_names"],
                        "training_runs": [str(data["run_ids"][i])
                                          for i in sorted(groups["train"])],
                        "horizon_steps": horizon_steps,
                        "actuator_feedback_is_conditioned_input": False,
                        "actuator_response_is_fitted_and_command_driven": True,
                        "front_wheel_speeds_are_latent": True,
                        "tire_relaxation_state_enabled": use_tire_relaxation,
                        "physics_source": (
                            "AutoDRIVE vehicle dynamics guide section 1.3.2"),
                    },
                }, output_dir / "best_four_wheel_greybox.pt")
            else:
                stale += 1
            (output_dir / "validation_history.json").write_text(
                json.dumps(history, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            if stale >= patience:
                break

    best = torch.load(output_dir / "best_four_wheel_greybox.pt",
                      map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    model.eval()
    validation = _evaluate(
        torch, model, data, physical_state, acceleration,
        groups["validation"], state_scale[:3], device, horizon_steps,
        seed + 2001,
        max_eval_windows_per_run)
    test = None
    if score_test and groups["test"]:
        test = _evaluate(torch, model, data, physical_state, acceleration,
                         groups["test"], state_scale[:3], device,
                         horizon_steps, seed + 2002,
                         max_eval_windows_per_run)
    report = {
        "schema_version": 1,
        "architecture": "four_wheel_guide_greybox",
        "dataset": str(dataset_path.resolve()),
        "device": str(device),
        "horizon_seconds": horizon_steps * DT_S,
        "multiple_shooting_window_seconds": shooting_steps * DT_S,
        "tire_relaxation_state_enabled": use_tire_relaxation,
        "optimizer_steps": step,
        "optimizer_steps_per_second": step / max(
            time.perf_counter() - start_time, 1.0e-9),
        "eligible_runs": {
            key: [str(data["run_ids"][i]) for i in sorted(value)]
            for key, value in groups.items()},
        "best_step": best_step,
        "best_validation_score": best_score,
        "parameters": model.export_parameters(),
        "validation": validation,
        "test_scored_once": bool(score_test),
        "test": test,
        "checkpoint": str(output_dir / "best_four_wheel_greybox.pt"),
        "limitations": [
            "front wheel speeds have no direct labels",
            "per-wheel normal loads and tire forces are unobserved",
            "front wheel speed and tire/load/drivetrain parameters may be non-identifiable",
        ],
        "history": history,
    }
    write_standard_artifacts(
        output_dir, dataset_path, data,
        {"architecture": "four_wheel_guide_greybox",
         "horizon_seconds": horizon_steps * DT_S,
         "shooting_seconds": shooting_steps * DT_S,
         "tire_relaxation_state_enabled": use_tire_relaxation,
         "batch_size": batch_size, "max_steps": max_steps,
         "eval_every": eval_every, "patience": patience,
         "max_eval_windows_per_run": max_eval_windows_per_run,
         "score_test": score_test}, seed, report,
        ("tools/vehicle_dynamics_learning/four_wheel_greybox.py",
         "tools/vehicle_dynamics_learning/train_nssm.py",
         "tools/vehicle_dynamics_learning/experiment_artifacts.py"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--horizon-seconds", type=float, default=2.0)
    parser.add_argument("--shooting-seconds", type=float, default=0.5)
    parser.add_argument("--no-tire-relaxation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-eval-windows-per-run", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--score-test", action="store_true")
    args = parser.parse_args()
    report = train(
        args.dataset, args.output_dir, device_name=args.device,
        horizon_seconds=args.horizon_seconds,
        shooting_seconds=args.shooting_seconds,
        use_tire_relaxation=not args.no_tire_relaxation,
        batch_size=args.batch_size,
        max_steps=args.max_steps, eval_every=args.eval_every,
        patience=args.patience,
        max_eval_windows_per_run=args.max_eval_windows_per_run,
        seed=args.seed, score_test=args.score_test)
    print(f"wrote {args.output_dir / 'training_report.json'}; "
          f"best validation={report['best_validation_score']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

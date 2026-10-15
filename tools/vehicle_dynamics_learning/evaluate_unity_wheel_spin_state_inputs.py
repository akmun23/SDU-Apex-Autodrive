#!/usr/bin/env python3
"""Score the wheel-spin model using only reconstructible current-state inputs.

Training is on r07 and r11 is untouched whole-capture transfer. WheelHit
slip/load/contact labels are used only for diagnostics and for the frozen
training tire-curve fit; the candidate predictor reconstructs contact kinematics
and suspension load from current body/wheel state, source geometry and command.
This is still a one-step wheel-speed test, not a full vehicle rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from tools.vehicle_dynamics_learning.analyze_unity_3d_moment_balance import (
    DEFAULT_CAPTURES,
    WHEELS,
    _curve,
    _balance_data,
    _predict_balance,
)
from tools.vehicle_dynamics_learning.analyze_unity_suspension_load import (
    WHEEL_TRANSFORM,
)
from tools.vehicle_dynamics_learning.evaluate_unity_wheel_spin_1ms import (
    ROOT,
    SPEED_BINS,
    STEERING_BINS,
    WHEEL_RADIUS_SOLID_CYLINDER_INERTIA,
    _evaluate_capture,
    _metrics,
    _robust_shared_fit,
    _step_arrays,
)


DEFAULT_TIRE_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "tire_curve_force_balance_20261012.json")
DEFAULT_SLIP_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "longitudinal_slip_law_fit_v3_20261012.json")
DEFAULT_SUSPENSION_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "suspension_load_geometry_calibration_v2_20261012.json")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "wheel_spin_state_inputs_v10_20261013.json")
MOTOR_TORQUE_NM = 428.0
GRAVITY_MPS2 = 9.80665
WHEEL_ORDER = ("fl", "fr", "rl", "rr")
PREFIX = {"fl": "front_left", "fr": "front_right",
          "rl": "rear_left", "rr": "rear_right"}
TIRE_CURVE_METADATA = {
    "fl": "front_left_forward_curve",
    "fr": "front_right_forward_curve",
    "rl": "rear_left_forward_curve",
    "rr": "rear_right_forward_curve",
}


def _plain_metrics(error: np.ndarray) -> dict[str, Any]:
    error = error[np.isfinite(error)]
    if error.size == 0:
        return {"samples": 0, "rmse": None, "abs_p95": None,
                "abs_p99": None, "abs_max": None}
    absolute = np.abs(error)
    return {
        "samples": int(error.size),
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "abs_p95": float(np.quantile(absolute, 0.95)),
        "abs_p99": float(np.quantile(absolute, 0.99)),
        "abs_max": float(np.max(absolute)),
    }


def _rotate_rows(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    xyz = quaternion[:, :3]
    twice_cross = 2.0 * np.cross(xyz, vectors)
    return vectors + quaternion[:, 3, None] * twice_cross + np.cross(
        xyz, twice_cross)


def _rotate_fixed(quaternion: tuple[float, float, float, float],
                  vectors: np.ndarray) -> np.ndarray:
    xyz = np.asarray(quaternion[:3], dtype=np.float64)
    twice_cross = 2.0 * np.cross(np.broadcast_to(xyz, vectors.shape), vectors)
    return vectors + quaternion[3] * twice_cross + np.cross(
        np.broadcast_to(xyz, vectors.shape), twice_cross)


def _state_reconstructed_data(
    data: dict[str, Any], slip_fit: dict[str, Any],
    suspension_fit: dict[str, Any],
    contact_offsets: dict[str, list[float]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct current wheel contact inputs from body state and commands."""
    count = len(data["speed"])
    quaternion = data["rotation_world_xyzw"]
    inverse = quaternion.copy()
    inverse[:, :3] *= -1.0
    up_world = np.broadcast_to((0.0, 1.0, 0.0), (count, 3))
    normal_body = _rotate_rows(inverse, up_world)
    position = data["position_world"]
    com = np.asarray([float(v) for v in
                      data["metadata"]["center_of_mass_body"].split(",")])
    velocity = data["velocity_body"]
    angular_velocity = data["omega_body"]
    throttle = data["throttle_command"]
    motor_per_wheel = MOTOR_TORQUE_NM * throttle / 4.0
    brake_per_wheel = np.where(throttle == 0.0, MOTOR_TORQUE_NM, 0.0)
    slip_parameters = slip_fit["shared_max_speed_plus_regularizer_candidate"]
    radius_scale = float(slip_parameters["fitted_radius_scale"])
    regularizer = float(slip_parameters["fitted_speed_regularizer_mps"])
    geometry_calibration = suspension_fit["geometry_only_load_model"]
    rebuilt_wheels: list[dict[str, np.ndarray]] = []
    diagnostics: dict[str, Any] = {"per_wheel": {}}

    for index, wheel_name in enumerate(WHEELS):
        wheel_key = WHEEL_ORDER[index]
        wheel = data["wheels"][index]
        transform_position, transform_rotation = WHEEL_TRANSFORM[wheel_key]
        mount = np.asarray(transform_position, dtype=np.float64)
        collider_center = np.asarray((0.0, 0.025, 0.0))
        center_local = mount + _rotate_fixed(transform_rotation,
                                              collider_center[None, :])[0]
        down_local = _rotate_fixed(
            transform_rotation, np.asarray(((0.0, -1.0, 0.0),)))[0]
        center_world = position + _rotate_rows(
            quaternion, np.broadcast_to(center_local, (count, 3)))
        down_world = _rotate_rows(
            quaternion, np.broadcast_to(down_local, (count, 3)))
        radius = float(wheel["radius"])
        down_world_y = down_world[:, 1]
        if np.any(np.abs(down_world_y) < 1.0e-6):
            raise ValueError("suspension ray became parallel to the flat road")
        travel = (radius - center_world[:, 1]) / down_world_y

        # Wheel travel rate follows the derivative of the geometric plane
        # constraint. Rigidbody velocity is at its COM, so use the complete
        # rigid-body point velocity at the moving suspension origin.
        center_from_com_with_travel = (
            center_local[None, :] + travel[:, None] * down_local[None, :]
            - com[None, :])
        point_velocity = velocity + np.cross(
            angular_velocity, center_from_com_with_travel)
        denominator = normal_body @ down_local
        travel_rate = -np.einsum("ij,ij->i", normal_body, point_velocity) \
            / denominator

        calibration = geometry_calibration[wheel_key]
        target_travel = float(
            calibration["training_calibrated_target_travel_m"])
        sprung_mass = float(data["metadata"][
            f"{PREFIX[wheel_key]}_sprung_mass_kg"])
        spring = float(data["metadata"][
            f"{PREFIX[wheel_key]}_spring_n_per_m"])
        damper = float(data["metadata"][
            f"{PREFIX[wheel_key]}_damper_ns_per_m"])
        load = np.maximum(
            0.0, sprung_mass * GRAVITY_MPS2
            + spring * (target_travel - travel) - damper * travel_rate)

        # The WheelCollider steering angle is applied around its local up
        # axis after the prefab transform (including the front-wheel camber).
        steer = wheel["steering_rad"]
        forward_local = np.column_stack((np.sin(steer), np.zeros(count),
                                         np.cos(steer)))
        forward_body = _rotate_fixed(transform_rotation, forward_local)
        forward_body -= normal_body * np.einsum(
            "ij,ij->i", forward_body, normal_body)[:, None]
        forward_norm = np.linalg.norm(forward_body, axis=1)
        forward_body /= np.maximum(forward_norm[:, None], 1.0e-12)
        sideways_body = np.cross(normal_body, forward_body)
        sideways_norm = np.linalg.norm(sideways_body, axis=1)
        sideways_body /= np.maximum(sideways_norm[:, None], 1.0e-12)

        raw_contact_body = (center_local[None, :] + travel[:, None]
                            * down_local[None, :]
                            - radius * normal_body)
        target_position = float(data["metadata"][
            f"{PREFIX[wheel_key]}_target_position"])
        suspension_distance = float(data["metadata"][
            f"{PREFIX[wheel_key]}_suspension_distance_m"])
        force_app_point_distance = float(data["metadata"][
            f"{PREFIX[wheel_key]}_force_app_point_distance_m"])
        rest_travel = (1.0 - target_position) * suspension_distance
        force_application_point = (
            center_local + down_local * (rest_travel + radius)
            + np.asarray((0.0, force_app_point_distance, 0.0)))
        force_application_lever = force_application_point - com
        contact_offset = np.asarray(
            (contact_offsets or {}).get(wheel_key, (0.0, 0.0, 0.0)),
            dtype=np.float64)
        contact_body = raw_contact_body + contact_offset[None, :]
        lever = contact_body - com[None, :]
        contact_velocity = velocity + np.cross(angular_velocity, lever)
        contact_speed = np.einsum("ij,ij->i", contact_velocity,
                                  forward_body)
        lateral_contact_speed = np.einsum(
            "ij,ij->i", contact_velocity, sideways_body)
        surface_speed = wheel["omega"] * radius_scale * radius
        slip_x = ((surface_speed - contact_speed)
                  / (np.maximum(np.abs(surface_speed),
                                np.abs(contact_speed)) + regularizer))
        slip_y = np.arctan2(lateral_contact_speed,
                            np.abs(contact_speed) + 1.0)

        # VehicleController.cs maps the captured command to four equal motor
        # torques, with exact zero selecting the all-wheel brake branch.
        rebuilt_wheels.append({
            **wheel,
            "load": load,
            "slip_x": slip_x,
            "slip_y": slip_y,
            "lever": lever,
            "forward": forward_body,
            "sideways": sideways_body,
            "normal": normal_body,
            "motor": motor_per_wheel,
            "brake": brake_per_wheel,
            "reconstructed_contact_speed_mps": contact_speed,
            "reconstructed_lateral_contact_speed_mps": lateral_contact_speed,
            "reconstructed_contact_body": contact_body,
            "force_application_point_body": np.broadcast_to(
                force_application_point, (count, 3)),
            "force_application_lever": np.broadcast_to(
                force_application_lever, (count, 3)),
            "reconstructed_travel_m": travel,
            "reconstructed_travel_rate_mps": travel_rate,
        })

        actual_load = wheel["load"]
        actual_contact = wheel["lever"] + com[None, :]
        actual_forward = wheel["forward"]
        actual_sideways = wheel["sideways"]
        contact_error = contact_body - actual_contact
        raw_contact_error = raw_contact_body - actual_contact
        direction_dot = np.clip(np.einsum(
            "ij,ij->i", forward_body, actual_forward), -1.0, 1.0)
        direction_error = np.arccos(direction_dot)
        sideways_dot = np.clip(np.einsum(
            "ij,ij->i", sideways_body, actual_sideways), -1.0, 1.0)
        sideways_direction_error = np.arccos(sideways_dot)
        slip_error = slip_x - wheel["slip_x"]
        lateral_slip_error = slip_y - wheel["slip_y"]
        valid = np.isfinite(actual_load) & np.isfinite(slip_error)
        diagnostics["per_wheel"][wheel_key] = {
            "load_error_n": _plain_metrics((load - actual_load)[valid]),
            "contact_point_error_m": {
                "samples": int(np.count_nonzero(valid)),
                "xyz_rmse": np.sqrt(np.mean(
                    contact_error[valid]**2, axis=0)).tolist(),
                "norm_rmse": float(np.sqrt(np.mean(
                    np.sum(contact_error[valid]**2, axis=1)))),
                "norm_p95": float(np.quantile(
                    np.linalg.norm(contact_error[valid], axis=1), 0.95)),
            },
            "raw_contact_point_error_m": {
                "norm_rmse": float(np.sqrt(np.mean(
                    np.sum(raw_contact_error[valid]**2, axis=1)))),
                "norm_p95": float(np.quantile(
                    np.linalg.norm(raw_contact_error[valid], axis=1), 0.95)),
            },
            "tire_forward_direction_error_rad": _plain_metrics(
                direction_error[valid]),
            "tire_sideways_direction_error_rad": _plain_metrics(
                sideways_direction_error[valid]),
            "wheelhit_slip_diagnostic": _plain_metrics(slip_error[valid]),
            "wheelhit_lateral_slip_diagnostic": _plain_metrics(
                lateral_slip_error[valid]),
            "predicted_load_range_n": [
                float(np.quantile(load[valid], 0.01)),
                float(np.quantile(load[valid], 0.99))],
        }

    return {
        **data,
        "wheels": rebuilt_wheels,
        "slip_radius_scale": radius_scale,
        "slip_regularizer_mps": regularizer,
    }, diagnostics


def _fit_training_contact_offsets(
    data: dict[str, Any], reconstructed: dict[str, Any],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Fit a constant body-frame contact-point correction on training labels."""
    com = np.asarray([float(v) for v in
                      data["metadata"]["center_of_mass_body"].split(",")])
    offsets: dict[str, list[float]] = {}
    diagnostics: dict[str, Any] = {}
    for index, wheel_key in enumerate(WHEEL_ORDER):
        wheel = data["wheels"][index]
        predicted = reconstructed["wheels"][index][
            "reconstructed_contact_body"]
        actual = wheel["lever"] + com[None, :]
        valid = (wheel["grounded"] & np.isfinite(predicted).all(axis=1)
                 & np.isfinite(actual).all(axis=1))
        if np.count_nonzero(valid) < 100:
            raise ValueError(
                f"insufficient grounded training contact samples for {wheel_key}")
        residual = actual[valid] - predicted[valid]
        # The mean is the least-squares constant offset. Quantiles are kept as
        # diagnostics, not used to trim away difficult maneuvers.
        offset = np.mean(residual, axis=0)
        offsets[wheel_key] = offset.tolist()
        centered = residual - offset
        diagnostics[wheel_key] = {
            "training_grounded_samples": int(residual.shape[0]),
            "fitted_body_frame_offset_m": offset.tolist(),
            "training_residual_rmse_by_axis_m": np.sqrt(
                np.mean(centered**2, axis=0)).tolist(),
            "training_residual_norm_p95_m": float(np.quantile(
                np.linalg.norm(centered, axis=1), 0.95)),
        }
    return offsets, diagnostics


def _command_torque_diagnostic(data: dict[str, Any]) -> dict[str, Any]:
    throttle = data["throttle_command"]
    expected_motor = MOTOR_TORQUE_NM * throttle / 4.0
    expected_brake = np.where(throttle == 0.0, MOTOR_TORQUE_NM, 0.0)
    motor_errors = []
    brake_errors = []
    for wheel in data["wheels"]:
        motor_errors.append(wheel["motor"] - expected_motor)
        brake_errors.append(wheel["brake"] - expected_brake)
    motor_error = np.concatenate(motor_errors)
    brake_error = np.concatenate(brake_errors)
    return {
        "motor_torque_error_nm": _metrics(motor_error),
        "brake_torque_error_nm": _metrics(brake_error),
        "exact_zero_command_fraction": float(np.mean(throttle == 0.0)),
        "interpretation": "source command mapping compared with captured WheelCollider torques",
    }


def _implicit_wheel_step_arrays(
    data: dict[str, Any], explicit_force: np.ndarray,
    tire_parameters: np.ndarray, inertia: float, damping: float, dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve wheel torque balance with tire force at the unknown next RPM."""
    predictions, targets = _step_arrays(
        data, explicit_force, inertia, damping, dt_s)
    scale_x, tangent_x = tire_parameters[:2]
    for index, wheel in enumerate(data["wheels"]):
        active = wheel["brake"] == 0.0
        if not np.any(active):
            continue
        omega = wheel["omega"][active]
        contact_speed = wheel["reconstructed_contact_speed_mps"][active]
        radius = wheel["radius"]
        load = wheel["load"][active]
        motor = wheel["motor"][active]
        metadata_key = TIRE_CURVE_METADATA[WHEEL_ORDER[index]]
        landmarks = np.asarray([float(value) for value in
                                data["metadata"][metadata_key].split(",")])
        guess = predictions[active, index].copy()

        def torque_residual(candidate: np.ndarray) -> np.ndarray:
            surface = candidate * radius * float(
                data["slip_radius_scale"])
            slip = ((surface - contact_speed)
                    / (np.maximum(np.abs(surface), np.abs(contact_speed))
                       + float(data["slip_regularizer_mps"])))
            tire_force = scale_x * load * _curve(
                slip, landmarks, tangent_x)
            return (inertia * (candidate - omega) / dt_s
                    - motor + radius * tire_force
                    + damping * candidate)

        # Newton solves the scalar implicit tire/wheel coupling independently
        # for each sample. This is a state solve, not a fit to future signals.
        for _ in range(8):
            residual = torque_residual(guess)
            step = 0.25
            slope = (torque_residual(guess + step)
                     - torque_residual(guess - step)) / (2.0 * step)
            safe_slope = np.where(np.abs(slope) > 1.0e-9, slope,
                                  np.sign(slope) * 1.0e-9 + 1.0e-9)
            correction = residual / safe_slope
            guess -= np.clip(correction, -500.0, 500.0)
        predictions[active, index] = guess
    # Report the implicit equation's residual in angular-speed units so
    # convergence can be distinguished from agreement with recorded RPM.
    equation_errors = np.zeros_like(predictions)
    for index, wheel in enumerate(data["wheels"]):
        active = wheel["brake"] == 0.0
        if not np.any(active):
            continue
        candidate = predictions[:, index]
        surface = (candidate * wheel["radius"]
                   * float(data["slip_radius_scale"]))
        slip = ((surface - wheel["reconstructed_contact_speed_mps"])
                / (np.maximum(np.abs(surface),
                              np.abs(wheel["reconstructed_contact_speed_mps"]))
                   + float(data["slip_regularizer_mps"])))
        landmarks = np.asarray([float(value) for value in
                                data["metadata"][
                                    TIRE_CURVE_METADATA[WHEEL_ORDER[index]]]
                                .split(",")])
        tire_force = (scale_x * wheel["load"]
                      * _curve(slip, landmarks, tangent_x))
        residual = (inertia * (candidate - wheel["omega"]) / dt_s
                    - wheel["motor"] + wheel["radius"] * tire_force
                    + damping * candidate)
        equation_errors[active, index] = residual[active] * dt_s / inertia
    return predictions, equation_errors


def _score_predictions(context: dict[str, Any], predictions: np.ndarray,
                       equation_errors: np.ndarray | None = None
                       ) -> dict[str, Any]:
    data = context["reconstructed_data"]
    targets = np.column_stack([wheel["omega"] for wheel in data["wheels"]])
    errors = predictions[:-1] - targets[1:]
    dt_valid = np.isclose(context["dt"], context["fixed_dt"], rtol=0.0,
                          atol=1.0e-9)
    valid = np.broadcast_to(dt_valid[:, None], errors.shape).copy()
    valid &= np.isfinite(errors)
    speed = data["speed"][:-1]
    result: dict[str, Any] = {"all": _metrics(errors[valid])}
    result["speed_mps"] = {}
    for low, high in SPEED_BINS:
        mask = (speed >= low) & (speed < high)
        result["speed_mps"][f"{low:g}-{high:g}"] = _metrics(
            errors[np.broadcast_to(mask[:, None], errors.shape) & valid])
    throttle = data["throttle_command"]
    command_step = np.abs(np.diff(throttle)) > 1.0e-6
    release = (throttle[1:] > 0.0) & (throttle[:-1] == 0.0)
    result["command_transitions"] = {
        "any_throttle_change": _metrics(errors[
            np.broadcast_to(command_step[:, None], errors.shape) & valid]),
        "zero_brake_to_positive_throttle": _metrics(errors[
            np.broadcast_to(release[:, None], errors.shape) & valid]),
        "steady_command": _metrics(errors[
            np.broadcast_to((~command_step)[:, None], errors.shape) & valid]),
    }
    if equation_errors is not None:
        result["implicit_equation_residual_radps"] = _metrics(
            equation_errors[:-1][valid])
    return result


def _recursive_wheel_only_score(
    context: dict[str, Any], tire_parameters: np.ndarray,
    inertia: float, damping: float, start_stride: int = 25,
    horizon_steps: tuple[int, ...] = (1, 5, 10, 25, 50, 100, 250),
) -> dict[str, Any]:
    """Recursively propagate wheel RPM while conditioning on current body truth.

    Body pose, velocity, yaw/roll rates, actuator command and steering are
    refreshed from the capture at each simulated step. Wheel RPM alone is
    recursively predicted; no future wheel RPM or WheelHit tire values enter
    the rollout. This isolates the wheel subsystem and is not a vehicle rollout.
    """
    data = context["reconstructed_data"]
    wheel_count = len(data["wheels"])
    sample_count = len(data["speed"])
    max_horizon = max(horizon_steps)
    starts = np.arange(0, sample_count - max_horizon, start_stride,
                       dtype=np.int64)
    omega_target = np.column_stack(
        [wheel["omega"] for wheel in data["wheels"]])
    grounded = np.column_stack(
        [wheel["grounded"] for wheel in data["wheels"]])
    wheel_valid = grounded.all(axis=1) & np.isfinite(omega_target).all(axis=1)
    for wheel in data["wheels"]:
        wheel_valid &= np.isfinite(wheel["load"])
        wheel_valid &= np.isfinite(wheel["reconstructed_contact_speed_mps"])
    invalid_prefix = np.r_[0, np.cumsum(~wheel_valid, dtype=np.int64)]
    window_invalid = (invalid_prefix[starts + max_horizon + 1]
                      - invalid_prefix[starts])
    starts = starts[window_invalid == 0]
    if starts.size == 0:
        raise ValueError("no fully grounded wheel-only rollout windows")

    state = omega_target[starts].copy()
    horizon_results: dict[str, Any] = {}
    horizon_set = set(horizon_steps)
    scale_x, tangent_x = tire_parameters[:2]
    radius_scale = float(data["slip_radius_scale"])
    epsilon = float(data["slip_regularizer_mps"])
    curve_landmarks = [np.asarray([float(v) for v in
                                  data["metadata"][
                                      TIRE_CURVE_METADATA[WHEEL_ORDER[i]]
                                  ].split(",")], dtype=np.float64)
                       for i in range(wheel_count)]
    score_errors: dict[int, np.ndarray] = {}

    for step in range(1, max_horizon + 1):
        rows = starts + step - 1
        next_state = np.empty_like(state)
        for index, wheel in enumerate(data["wheels"]):
            radius = wheel["radius"]
            contact_speed = wheel["reconstructed_contact_speed_mps"][rows]
            load = wheel["load"][rows]
            surface = state[:, index] * radius_scale * radius
            slip = ((surface - contact_speed)
                    / (np.maximum(np.abs(surface), np.abs(contact_speed))
                       + epsilon))
            tire_force = (scale_x * load
                          * _curve(slip, curve_landmarks[index], tangent_x))
            motor = wheel["motor"][rows]
            brake = wheel["brake"][rows]
            free = state[:, index] + context["fixed_dt"] * (
                motor - radius * tire_force - damping * state[:, index]
            ) / inertia
            brake_speed = brake * context["fixed_dt"] / inertia
            next_state[:, index] = np.where(
                free > 0.0, np.maximum(0.0, free - brake_speed),
                np.where(free < 0.0, np.minimum(0.0, free + brake_speed),
                         0.0))
        state = next_state
        if step in horizon_set:
            score_errors[step] = state - omega_target[starts + step]

    speed_at_start = data["speed"][starts]
    steer_at_start = np.max(np.abs(np.column_stack(
        [wheel["steering_rad"][starts] for wheel in data["wheels"]])), axis=1)
    initial_slip = np.column_stack(
        [wheel["slip_x"][starts] for wheel in data["wheels"]])
    result: dict[str, Any] = {
        "start_samples": int(starts.size),
        "start_stride_ms": int(round(start_stride * 1000.0
                                      * context["fixed_dt"])),
        "per_horizon": {},
        "conditioning": (
            "At every 1 ms step the captured body pose/velocity/angular rate, "
            "steering angle, and throttle command are supplied. Wheel RPM is "
            "the only recursively predicted state. This is a wheel-subsystem "
            "test, not a no-truth full-vehicle prediction."),
    }
    for step, error in score_errors.items():
        horizon_ms = int(round(step * context["fixed_dt"] * 1000.0))
        entry: dict[str, Any] = {
            "horizon_ms": horizon_ms,
            "all_wheels": _metrics(error.ravel()),
            "by_wheel": {
                WHEEL_ORDER[index]: _metrics(error[:, index])
                for index in range(wheel_count)},
            "initial_speed_mps": {},
            "initial_abs_steering_rad": {},
            "initial_abs_slip": {},
        }
        for low, high in SPEED_BINS:
            mask = (speed_at_start >= low) & (speed_at_start < high)
            entry["initial_speed_mps"][f"{low:g}-{high:g}"] = _metrics(
                error[np.broadcast_to(mask[:, None], error.shape)].ravel())
        for low, high in STEERING_BINS:
            mask = (steer_at_start >= low) & (steer_at_start < high)
            entry["initial_abs_steering_rad"][f"{low:g}-{high:g}"] = _metrics(
                error[np.broadcast_to(mask[:, None], error.shape)].ravel())
        for low, high in ((0.0, 0.05), (0.05, 0.15), (0.15, 0.5),
                          (0.5, 1.01)):
            mask = ((np.abs(initial_slip) >= low)
                    & (np.abs(initial_slip) < high))
            entry["initial_abs_slip"][f"{low:g}-{high:g}"] = _metrics(
                error[mask])
        result["per_horizon"][f"{horizon_ms}ms"] = entry
    return result


def _vector_metrics(error: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(error).all(axis=1)
    selected = error[finite]
    if selected.size == 0:
        return {"samples": 0, "rmse_by_axis": None,
                "vector_norm_rmse": None, "vector_norm_p95": None}
    norm = np.linalg.norm(selected, axis=1)
    return {
        "samples": int(selected.shape[0]),
        "bias_by_axis": np.mean(selected, axis=0).tolist(),
        "rmse_by_axis": np.sqrt(np.mean(selected**2, axis=0)).tolist(),
        "abs_p95_by_axis": np.quantile(np.abs(selected), 0.95,
                                        axis=0).tolist(),
        "vector_norm_rmse": float(np.sqrt(np.mean(norm**2))),
        "vector_norm_p95": float(np.quantile(norm, 0.95)),
    }


def _total_contact_balance(data: dict[str, Any],
                           tire_parameters: np.ndarray,
                           axle_lateral_parameters: np.ndarray | None = None,
                           use_force_application_point: bool = False,
                           ) -> tuple[np.ndarray, np.ndarray]:
    if (axle_lateral_parameters is None
            and not use_force_application_point):
        tire_force, tire_torque, _ = _predict_balance(data, tire_parameters)
        force = tire_force.copy()
        torque = tire_torque.copy()
        for wheel in data["wheels"]:
            normal_force = wheel["load"][:, None] * wheel["normal"]
            force += normal_force
            lever = (wheel["force_application_lever"]
                     if use_force_application_point else wheel["lever"])
            torque += np.cross(lever, normal_force)
        return force, torque

    force = np.zeros_like(data["force_target_total"])
    torque = np.zeros_like(data["torque_target_total"])
    for index, wheel_key in enumerate(WHEEL_ORDER):
        wheel = data["wheels"][index]
        if axle_lateral_parameters is None:
            lateral_scale, lateral_tangent = tire_parameters[2:4]
        else:
            axle_index = 0 if index < 2 else 1
            lateral_scale, lateral_tangent = axle_lateral_parameters[
                2 * axle_index:2 * axle_index + 2]
        landmarks_x = np.asarray([float(value) for value in
                                  data["metadata"][
                                      TIRE_CURVE_METADATA[wheel_key]]
                                  .split(",")])
        metadata_key = PREFIX[wheel_key]
        landmarks_y = np.asarray([float(value) for value in
                                  data["metadata"][
                                      f"{metadata_key}_sideways_curve"]
                                  .split(",")])
        longitudinal = (tire_parameters[0] * wheel["load"]
                        * _curve(wheel["slip_x"], landmarks_x,
                                 tire_parameters[1]))
        lateral = (-lateral_scale * wheel["load"]
                   * _curve(wheel["slip_y"], landmarks_y,
                            lateral_tangent))
        tire_force = (longitudinal[:, None] * wheel["forward"]
                      + lateral[:, None] * wheel["sideways"])
        normal_force = wheel["load"][:, None] * wheel["normal"]
        total_force = tire_force + normal_force
        force += total_force
        lever = (wheel["force_application_lever"]
                 if use_force_application_point else wheel["lever"])
        torque += np.cross(lever, total_force)
    return force, torque


def _fit_state_reconstructed_tire_curve(
    training_data: dict[str, Any], initial: np.ndarray,
    sample_stride: int = 25,
) -> dict[str, Any]:
    """Fit the existing tire-curve structure on r07 reconstructed inputs."""
    source_rows = training_data["rows"][::sample_stride]
    count = len(training_data["speed"])
    fit_data = {
        **training_data,
        "rows": np.arange(source_rows.size, dtype=np.int64),
        "speed": training_data["speed"][source_rows],
        "force_target": training_data["force_target"][source_rows],
        "torque_target": training_data["torque_target"][source_rows],
        "force_target_total": training_data[
            "force_target_total"][source_rows],
        "torque_target_total": training_data[
            "torque_target_total"][source_rows],
        "wheels": [{
            key: (value[source_rows] if isinstance(value, np.ndarray)
                  and value.ndim > 0 and value.shape[0] == count else value)
            for key, value in wheel.items()
        } for wheel in training_data["wheels"]],
    }
    rows = fit_data["rows"]
    force_scale = np.maximum(np.sqrt(np.mean(
        fit_data["force_target_total"]**2, axis=0)), 1.0)
    torque_scale = np.maximum(np.sqrt(np.mean(
        fit_data["torque_target_total"]**2, axis=0)), 0.02)

    def residual(parameters: np.ndarray) -> np.ndarray:
        force, torque = _total_contact_balance(fit_data, parameters)
        return np.concatenate((
            ((force[rows] - fit_data["force_target_total"][rows])
             / force_scale).ravel(),
            ((torque[rows] - fit_data["torque_target_total"][rows])
             / torque_scale).ravel(),
        ))

    result = least_squares(
        residual, initial,
        bounds=([0.1, 0.01, 0.1, 0.01], [4.0, 80.0, 4.0, 250.0]),
        loss="soft_l1", f_scale=1.0, max_nfev=200)
    return {
        "parameters": result.x.tolist(),
        "fit_samples": int(rows.size),
        "fit_stride": sample_stride,
        "normalized_cost": float(result.cost),
        "optimizer_status": int(result.status),
        "optimizer_message": result.message,
        "force_axis_scale_n": force_scale.tolist(),
        "torque_axis_scale_nm": torque_scale.tolist(),
        "training_only": True,
    }


def _fit_lateral_response_to_training_lateral_force(
    training_data: dict[str, Any], initial: np.ndarray,
    sample_stride: int = 25,
) -> dict[str, Any]:
    """Fit only the shared lateral curve parameters to training Fx balance."""
    rows = training_data["rows"][::sample_stride]
    target_scale = max(float(np.sqrt(np.mean(
        training_data["force_target_total"][rows, 0]**2))), 1.0)

    def residual(parameters: np.ndarray) -> np.ndarray:
        candidate = initial.copy()
        candidate[2:] = parameters
        force, _ = _total_contact_balance(training_data, candidate)
        return ((force[rows, 0]
                 - training_data["force_target_total"][rows, 0])
                / target_scale)

    result = least_squares(
        residual, initial[2:], bounds=([0.1, 0.01], [4.0, 250.0]),
        loss="soft_l1", f_scale=1.0, max_nfev=200)
    return {
        "parameters": result.x.tolist(),
        "parameter_order": ["lateral_scale", "lateral_zero_slip_tangent"],
        "fit_samples": int(rows.size),
        "fit_stride": sample_stride,
        "target_force_axis": "Unity body x (lateral)",
        "normalized_target_scale_n": target_scale,
        "normalized_cost": float(result.cost),
        "optimizer_status": int(result.status),
        "optimizer_message": result.message,
        "training_only": True,
    }


def _fit_lateral_response_with_source_force_point(
    training_data: dict[str, Any], initial: np.ndarray,
    sample_stride: int = 25,
) -> dict[str, Any]:
    """Fit lateral curve parameters with serialized Unity force lever arms."""
    rows = training_data["rows"][::sample_stride]
    force_scale = np.maximum(np.sqrt(np.mean(
        training_data["force_target_total"][rows]**2, axis=0)), 1.0)
    torque_scale = np.maximum(np.sqrt(np.mean(
        training_data["torque_target_total"][rows]**2, axis=0)), 0.02)

    def residual(parameters: np.ndarray) -> np.ndarray:
        candidate = initial.copy()
        candidate[2:] = parameters
        force, torque = _total_contact_balance(
            training_data, candidate, use_force_application_point=True)
        return np.concatenate((
            ((force[rows] - training_data["force_target_total"][rows])
             / force_scale).ravel(),
            ((torque[rows] - training_data["torque_target_total"][rows])
             / torque_scale).ravel(),
        ))

    result = least_squares(
        residual, initial[2:], bounds=([0.1, 0.01], [4.0, 250.0]),
        loss="soft_l1", f_scale=1.0, max_nfev=200)
    return {
        "parameters": result.x.tolist(),
        "parameter_order": ["lateral_scale", "lateral_zero_slip_tangent"],
        "fit_samples": int(rows.size),
        "fit_stride": sample_stride,
        "normalized_force_axis_scale_n": force_scale.tolist(),
        "normalized_torque_axis_scale_nm": torque_scale.tolist(),
        "normalized_cost": float(result.cost),
        "optimizer_status": int(result.status),
        "optimizer_message": result.message,
        "training_only": True,
    }


def _fit_front_rear_lateral_response(
    training_data: dict[str, Any], tire_parameters: np.ndarray,
    sample_stride: int = 25,
) -> dict[str, Any]:
    """Fit independent front/rear lateral curves to training force/moments."""
    rows = training_data["rows"][::sample_stride]
    force_scale = np.maximum(np.sqrt(np.mean(
        training_data["force_target_total"][rows]**2, axis=0)), 1.0)
    torque_scale = np.maximum(np.sqrt(np.mean(
        training_data["torque_target_total"][rows]**2, axis=0)), 0.02)
    initial = np.asarray((tire_parameters[2], tire_parameters[3],
                          tire_parameters[2], tire_parameters[3]))

    def residual(parameters: np.ndarray) -> np.ndarray:
        force, torque = _total_contact_balance(
            training_data, tire_parameters, parameters)
        return np.concatenate((
            ((force[rows] - training_data["force_target_total"][rows])
             / force_scale).ravel(),
            ((torque[rows] - training_data["torque_target_total"][rows])
             / torque_scale).ravel(),
        ))

    result = least_squares(
        residual, initial,
        bounds=([0.1, 0.01, 0.1, 0.01], [4.0, 250.0, 4.0, 250.0]),
        loss="soft_l1", f_scale=1.0, max_nfev=200)
    return {
        "parameters": result.x.tolist(),
        "parameter_order": ["front_scale", "front_zero_slip_tangent",
                            "rear_scale", "rear_zero_slip_tangent"],
        "fit_samples": int(rows.size),
        "fit_stride": sample_stride,
        "normalized_force_axis_scale_n": force_scale.tolist(),
        "normalized_torque_axis_scale_nm": torque_scale.tolist(),
        "normalized_cost": float(result.cost),
        "optimizer_status": int(result.status),
        "optimizer_message": result.message,
        "training_only": True,
    }


def _contact_balance_one(data: dict[str, Any],
                         tire_parameters: np.ndarray,
                         axle_lateral_parameters: np.ndarray | None = None,
                         use_force_application_point: bool = False
                         ) -> dict[str, Any]:
    predicted_force, predicted_torque = _total_contact_balance(
        data, tire_parameters, axle_lateral_parameters,
        use_force_application_point)
    rows = data["rows"]
    force_error = predicted_force[rows] - data["force_target_total"][rows]
    torque_error = predicted_torque[rows] - data["torque_target_total"][rows]
    angular_acceleration_error = torque_error @ np.linalg.inv(
        data["inertia"]).T
    result: dict[str, Any] = {
        "force_residual_n": _vector_metrics(force_error),
        "torque_residual_nm": _vector_metrics(torque_error),
        "angular_acceleration_residual_radps2": _vector_metrics(
            angular_acceleration_error),
        "yaw_moment_residual_rmse_nm": float(np.sqrt(np.mean(
            torque_error[:, 1]**2))),
        "speed_bins": {},
    }
    speed = data["speed"][rows]
    for low, high in SPEED_BINS:
        mask = (speed >= low) & (speed < high)
        result["speed_bins"][f"{low:g}-{high:g}"] = {
            "samples": int(np.count_nonzero(mask)),
            "force_vector_norm_rmse_n": (
                float(np.sqrt(np.mean(np.sum(force_error[mask]**2, axis=1))))
                if np.any(mask) else None),
            "yaw_moment_rmse_nm": (
                float(np.sqrt(np.mean(torque_error[mask, 1]**2)))
                if np.any(mask) else None),
        }
    return result


def _contact_balance_scores(context: dict[str, Any],
                             tire_parameters: np.ndarray
                             ) -> dict[str, Any]:
    return {
        "wheelhit_oracle": _contact_balance_one(
            context["data"], tire_parameters),
        "state_reconstructed": _contact_balance_one(
            context["reconstructed_data"], tire_parameters),
    }


def _context(path: Path, tire_parameters: np.ndarray,
             derivative_window: int, data: dict[str, Any],
             reconstructed: dict[str, Any]) -> dict[str, Any]:
    _, _, force_oracle = _predict_balance(data, tire_parameters)
    _, _, force_reconstructed = _predict_balance(reconstructed,
                                                 tire_parameters)
    frame = pd.read_csv(path, comment="#", usecols=[
        "sim_time_s", "steering_command"])
    dt = np.diff(frame["sim_time_s"].to_numpy(dtype=np.float64))
    fixed_dt = float(data["metadata"]["fixed_delta_time_s"])
    return {"path": path, "data": data,
            "reconstructed_data": reconstructed,
            "wheel_force_oracle": force_oracle,
            "wheel_force_reconstructed": force_reconstructed,
            "frame": frame, "dt": dt, "fixed_dt": fixed_dt}


def evaluate(train_path: Path, heldout_path: Path, tire_fit_path: Path,
             slip_fit_path: Path, suspension_fit_path: Path,
             derivative_window: int) -> dict[str, Any]:
    force_fit = json.loads(tire_fit_path.read_text(encoding="utf-8"))[
        "curve_model"]["training_fit"]
    tire = np.asarray((force_fit["longitudinal_scale"],
                       force_fit["longitudinal_zero_slip_tangent"],
                       force_fit["lateral_scale"],
                       force_fit["lateral_zero_slip_tangent"]),
                      dtype=np.float64)
    slip_fit = json.loads(slip_fit_path.read_text(encoding="utf-8"))
    suspension_fit = json.loads(
        suspension_fit_path.read_text(encoding="utf-8"))

    contexts = {}
    diagnostics = {}
    command_diagnostics = {}
    for key, path in (("train", train_path), ("heldout", heldout_path)):
        data = _balance_data(path, derivative_window, 1)
        reconstructed, state_diagnostics = _state_reconstructed_data(
            data, slip_fit, suspension_fit)
        # Both parameter fits use command-derived torque, whose exact match to
        # the captured torque is independently reported below.
        command_wheels = [{
            **wheel,
            "motor": MOTOR_TORQUE_NM * data["throttle_command"] / 4.0,
            "brake": np.where(data["throttle_command"] == 0.0,
                              MOTOR_TORQUE_NM, 0.0),
        } for wheel in data["wheels"]]
        oracle_command = {**data, "wheels": command_wheels}
        contexts[key] = {
            "oracle": _context(path, tire, derivative_window,
                               oracle_command, oracle_command),
            "state": _context(path, tire, derivative_window,
                              oracle_command, reconstructed),
        }
        diagnostics[key] = state_diagnostics
        command_diagnostics[key] = _command_torque_diagnostic(data)

    fit_oracle = _robust_shared_fit(
        contexts["train"]["oracle"]["data"],
        contexts["train"]["oracle"]["wheel_force_oracle"],
        contexts["train"]["oracle"]["fixed_dt"])
    fit_state = _robust_shared_fit(
        contexts["train"]["state"]["reconstructed_data"],
        contexts["train"]["state"]["wheel_force_reconstructed"],
        contexts["train"]["state"]["fixed_dt"])
    tire_state_fit = _fit_state_reconstructed_tire_curve(
        contexts["train"]["state"]["reconstructed_data"], tire)
    tire_state_parameters = np.asarray(tire_state_fit["parameters"],
                                       dtype=np.float64)

    scores = {}
    for model_name, fit, context_key, slip_source in (
        ("wheelhit_oracle_contact_inputs", fit_oracle, "oracle",
         "oracle_wheelhit"),
        ("state_reconstructed_contact_inputs", fit_state, "state",
         "state_reconstructed"),
    ):
        scores[model_name] = {}
        for split in ("train", "heldout"):
            context = contexts[split][context_key]
            scores[model_name][split] = _evaluate_capture(
                context, fit["effective_inertia_kgm2"],
                fit["wheel_damping_nms"], slip_source)
    implicit_scores = {}
    recursive_scores = {}
    contact_balance_scores = {}
    tire_curve_candidate_scores = {}
    for split in ("train", "heldout"):
        context = contexts[split]["state"]
        reconstructed = context["reconstructed_data"]
        predicted, equation_error = _implicit_wheel_step_arrays(
            reconstructed, context["wheel_force_reconstructed"], tire,
            fit_state["effective_inertia_kgm2"],
            fit_state["wheel_damping_nms"], context["fixed_dt"])
        implicit_scores[split] = _score_predictions(
            context, predicted, equation_error)
        recursive_scores[split] = _recursive_wheel_only_score(
            context, tire, fit_state["effective_inertia_kgm2"],
            fit_state["wheel_damping_nms"])
        contact_balance_scores[split] = _contact_balance_scores(context, tire)
        tire_curve_candidate_scores[split] = {
            "frozen_source_fit_on_state_inputs": _contact_balance_one(
                context["reconstructed_data"], tire),
            "r07_state_refit_on_state_inputs": _contact_balance_one(
                context["reconstructed_data"], tire_state_parameters),
        }

    # A single geometric offset per wheel is estimated from r07 WheelHit
    # contact points, then frozen for r11. This tests whether the several-mm
    # contact reconstruction bias is a stable geometry correction or merely
    # run-specific residual. It is not used as a runtime truth input.
    contact_offsets, contact_offset_fit = _fit_training_contact_offsets(
        contexts["train"]["state"]["data"],
        contexts["train"]["state"]["reconstructed_data"])
    contact_contexts: dict[str, dict[str, Any]] = {}
    contact_diagnostics: dict[str, Any] = {}
    for split, path in (("train", train_path), ("heldout", heldout_path)):
        base = contexts[split]["state"]["data"]
        reconstructed, contact_diagnostic = _state_reconstructed_data(
            base, slip_fit, suspension_fit, contact_offsets)
        contact_contexts[split] = _context(
            path, tire, derivative_window, base, reconstructed)
        contact_diagnostics[split] = contact_diagnostic
    contact_fit = _robust_shared_fit(
        contact_contexts["train"]["reconstructed_data"],
        contact_contexts["train"]["wheel_force_reconstructed"],
        contact_contexts["train"]["fixed_dt"])
    contact_offset_scores: dict[str, Any] = {}
    for split in ("train", "heldout"):
        context = contact_contexts[split]
        predicted, equation_error = _implicit_wheel_step_arrays(
            context["reconstructed_data"],
            context["wheel_force_reconstructed"], tire,
            contact_fit["effective_inertia_kgm2"],
            contact_fit["wheel_damping_nms"], context["fixed_dt"])
        contact_offset_scores[split] = {
            "one_step_wheel_spin": _evaluate_capture(
                context, contact_fit["effective_inertia_kgm2"],
                contact_fit["wheel_damping_nms"], "state_reconstructed"),
            "implicit_wheel_step": _score_predictions(
                context, predicted, equation_error),
            "frozen_tire_current_state_force_balance": _contact_balance_one(
                context["reconstructed_data"], tire),
            "state_reconstruction_diagnostics": contact_diagnostics[split],
        }

    # The dominant state-only force residual is lateral. These controlled
    # refits freeze longitudinal response and compare a shared lateral change
    # with a front/rear split. Neither is promoted unless every held-out force
    # and moment component is considered.
    training_state = contexts["train"]["state"]["reconstructed_data"]
    lateral_force_fit = _fit_lateral_response_to_training_lateral_force(
        training_state, tire)
    lateral_force_parameters = tire.copy()
    lateral_force_parameters[2:] = lateral_force_fit["parameters"]
    axle_lateral_fit = _fit_front_rear_lateral_response(
        training_state, tire)
    lateral_response_scores: dict[str, Any] = {}
    for split in ("train", "heldout"):
        state_data = contexts[split]["state"]["reconstructed_data"]
        lateral_response_scores[split] = {
            "frozen_shared_curve": _contact_balance_one(state_data, tire),
            "shared_lateral_fit_to_training_body_x_force":
                _contact_balance_one(state_data, lateral_force_parameters),
            "front_rear_lateral_fit_to_training_force_and_moments":
                _contact_balance_one(
                    state_data, tire,
                    np.asarray(axle_lateral_fit["parameters"],
                               dtype=np.float64)),
        }

    source_force_point_fit = _fit_lateral_response_with_source_force_point(
        training_state, tire)
    source_force_point_parameters = tire.copy()
    source_force_point_parameters[2:] = source_force_point_fit["parameters"]
    source_force_point_scores: dict[str, Any] = {}
    for split in ("train", "heldout"):
        state_data = contexts[split]["state"]["reconstructed_data"]
        source_force_point_scores[split] = {
            "WheelHit_contact_point_with_frozen_curve":
                _contact_balance_one(state_data, tire),
            "serialized_force_point_with_frozen_curve":
                _contact_balance_one(
                    state_data, tire, use_force_application_point=True),
            "serialized_force_point_with_r07_lateral_curve_fit":
                _contact_balance_one(
                    state_data, source_force_point_parameters,
                    use_force_application_point=True),
        }
    derivative_window_sensitivity: dict[str, Any] = {}
    for window in sorted(set((derivative_window, 101))):
        derivative_window_sensitivity[str(window)] = {}
        for split, path in (("train", train_path), ("heldout", heldout_path)):
            if window == derivative_window:
                state_data = contexts[split]["state"]["reconstructed_data"]
            else:
                raw = _balance_data(path, window, 1)
                state_data, _ = _state_reconstructed_data(
                    raw, slip_fit, suspension_fit)
            derivative_window_sensitivity[str(window)][split] = {
                "frozen_contact_point_source_curve": _contact_balance_one(
                    state_data, tire),
                "frozen_source_application_point_and_lateral_candidate":
                    _contact_balance_one(
                        state_data, source_force_point_parameters,
                        use_force_application_point=True),
            }

    return {
        "schema_version": 1,
        "status": (
            "state-reconstructed one-step wheel-spin model; full-body recursive "
            "accuracy not established"),
        "training_capture": str(train_path.relative_to(ROOT)),
        "heldout_capture": str(heldout_path.relative_to(ROOT)),
        "frozen_source_tire_fit": str(tire_fit_path.relative_to(ROOT)),
        "frozen_training_slip_fit": str(slip_fit_path.relative_to(ROOT)),
        "frozen_training_suspension_calibration": str(
            suspension_fit_path.relative_to(ROOT)),
        "training_fit_oracle_contact_inputs": fit_oracle,
        "training_fit_state_reconstructed_inputs": fit_state,
        "training_tire_curve_refit_state_inputs": tire_state_fit,
        "evaluations": scores,
        "implicit_tire_wheel_coupling_test": {
            "decision": "not promoted; its held-out recursive/one-step score must beat the explicit baseline",
            "method": (
                "solve the source torque-balance equation implicitly for next "
                "wheel speed, evaluating the fitted tire curve at predicted "
                "next slip; hold current body contact speed/load/steering fixed"),
            "parameters_frozen_from_r07_explicit_fit": {
                "effective_inertia_kgm2": fit_state[
                    "effective_inertia_kgm2"],
                "wheel_damping_nms": fit_state["wheel_damping_nms"],
                "tire_curve_fit": str(tire_fit_path.relative_to(ROOT)),
            },
            "newton_iterations": 8,
            "scores": implicit_scores,
        },
        "recursive_wheel_only_conditional_test": {
            "training_fit_parameters": {
                "effective_inertia_kgm2": fit_state[
                    "effective_inertia_kgm2"],
                "wheel_damping_nms": fit_state["wheel_damping_nms"],
                "tire_force_parameters": tire.tolist(),
            },
            "scores": recursive_scores,
        },
        "current_state_full_contact_force_balance": {
            "description": (
                "Use current body truth, wheel RPM, steering, source command, "
                "state-reconstructed slip and suspension load to predict all "
                "four tire and normal contact forces/moments; compare against "
                "Newton-Euler force/torque targets from body acceleration. "
                "This is a current-state balance test, not a recursive rollout."),
            "scores": contact_balance_scores,
        },
        "training_only_tire_curve_refit_transfer": {
            "parameters": tire_state_fit["parameters"],
            "scores": tire_curve_candidate_scores,
        },
        "training_fitted_contact_point_offset_transfer": {
            "decision": (
                "diagnostic candidate only; accept only if held-out contact, "
                "slip and force/moment errors improve together"),
            "method": (
                "least-squares constant body-frame contact-point offset per "
                "wheel, fitted from grounded r07 WheelHit contact points; "
                "offset frozen on independent r11 capture"),
            "fit": contact_offset_fit,
            "wheel_spin_parameter_fit_on_corrected_training_geometry":
                contact_fit,
            "scores": contact_offset_scores,
        },
        "lateral_tire_response_diagnosis": {
            "decision": (
                "neither fit is promoted automatically; the force-only fit "
                "must not conceal moment regression, and the axle fit must "
                "not conceal translational-force regression"),
            "fixed_parameters": {
                "longitudinal_scale": float(tire[0]),
                "longitudinal_zero_slip_tangent": float(tire[1]),
            },
            "shared_lateral_force_fit": lateral_force_fit,
            "front_rear_lateral_force_and_moment_fit": axle_lateral_fit,
            "scores": lateral_response_scores,
        },
        "source_force_application_point_lateral_fit": {
            "decision": (
                "one-step held-out candidate only; requires recursive "
                "command-only rollout before it can be called an offline "
                "plant improvement"),
            "source_basis": (
                "WheelCollider forceAppPointDistance and targetPosition from "
                "the captured Unity source metadata and serialized prefab"),
            "fit": source_force_point_fit,
            "scores": source_force_point_scores,
            "derivative_window_sensitivity_samples":
                derivative_window_sensitivity,
        },
        "state_reconstruction_diagnostics": diagnostics,
        "source_actuator_mapping_check": command_diagnostics,
        "baselines": {
            "solid_cylinder_inertia_kgm2":
                WHEEL_RADIUS_SOLID_CYLINDER_INERTIA,
        },
        "state_input_equations": {
            "contact_point_velocity": "v_contact = v_com + omega x (p_contact - p_com)",
            "suspension_travel": "intersection of the transformed WheelCollider suspension ray with the known flat road plane, offset by the serialized tire radius",
            "suspension_travel_rate": "differentiate the plane constraint using current rigid-body velocity and angular velocity; no future samples",
            "normal_load": "max(0, sprung_mass*g + spring*(calibrated_target_travel-travel) - damper*travel_rate)",
            "tire_forward": "prefab wheel frame followed by current WheelCollider steer angle, projected onto the road plane",
            "longitudinal_slip": "(radius*wheel_omega-contact_speed)/(max(abs(radius*wheel_omega),abs(contact_speed))+epsilon); epsilon and radius scale frozen from r07",
            "lateral_slip": "atan2(lateral_contact_speed, abs(longitudinal_contact_speed)+1 m/s), using the projected wheel frame; the 1 m/s regularizer is empirically identified",
            "wheel_spin": "I_eff*omega_dot = commanded_motor_torque - radius*tire_forward_force - damping*omega, with source's zero-throttle brake branch as a unilateral stop",
            "implicit_variant": "solve the same wheel equation with tire force evaluated at predicted next wheel speed instead of current measured slip",
        },
        "limitations": [
            "Body pose, body velocity, body angular velocity, wheel RPM, and per-wheel steering angle are current-state inputs from Unity truth; this does not yet show they can be maintained accurately by production odometry.",
            "The flat road plane is known. General road height/normal must come from a permitted map/state estimate.",
            "The slip regularizer, per-wheel target travel, tire force curve parameters and effective wheel inertia are fitted from r07 training labels only.",
            "The tire force curve fit was identified from WheelHit slip/load and rigid-body force balance; the reconstructed candidate does not consume held-out WheelHit labels.",
            "The predictor advances wheel speed one 1 ms step only; it does not recursively advance body speed, lateral velocity, yaw, pose, suspension, or actuator state.",
            "r11 has no high center-steering coverage comparable to r07; steering-envelope transfer remains incomplete.",
        ],
        "next_step": (
            "Use the held-out contact-offset transfer result to decide whether "
            "the residual is a stable geometric bias. If it is not, isolate "
            "the lateral slip/force discrepancy by wheel and maneuver before "
            "adding any body dynamics; only then compose explicit 3-D "
            "Newton-Euler motion and recursive actuator/suspension state."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_CAPTURES[0])
    parser.add_argument("--heldout", type=Path, default=DEFAULT_CAPTURES[1])
    parser.add_argument("--tire-fit", type=Path, default=DEFAULT_TIRE_FIT)
    parser.add_argument("--slip-fit", type=Path, default=DEFAULT_SLIP_FIT)
    parser.add_argument("--suspension-fit", type=Path,
                        default=DEFAULT_SUSPENSION_FIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--derivative-window", type=int, default=31)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = evaluate(args.train, args.heldout, args.tire_fit, args.slip_fit,
                      args.suspension_fit, args.derivative_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for name, splits in report["evaluations"].items():
        fit = report["training_fit_oracle_contact_inputs" if name.startswith(
            "wheelhit") else "training_fit_state_reconstructed_inputs"]
        print(name, "Ieff", fit["effective_inertia_kgm2"],
              "damping", fit["wheel_damping_nms"],
              "r11 RMSE", splits["heldout"]["state_step_error"]["all"][
                  "rmse_radps"],
              "p95", splits["heldout"]["state_step_error"]["all"][
                  "abs_p95_radps"])
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

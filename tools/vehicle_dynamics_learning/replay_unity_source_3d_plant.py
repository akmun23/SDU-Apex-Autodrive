#!/usr/bin/env python3
"""Command-only 1 kHz 3-D plant using Unity source geometry and state equations.

This research model propagates Rigidbody position/orientation, body linear and
angular velocity, four wheel speeds, steering, and geometry-derived suspension
loads. Simulator WheelHit and future truth are scoring-only. Unity's native
WheelCollider force solver is private, so the tire curve response is fitted on
r07 and the complete recursive model is assessed on independent r11.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

from tools.vehicle_dynamics_learning.analyze_unity_suspension_load import (
    WHEEL_TRANSFORM,
)
from tools.vehicle_dynamics_learning.replay_unity_source_mechanics import (
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    DT_S,
    ROOT,
    _load_capture,
    _metadata,
    _wrap,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    GRAVITY_MPS2,
    MOTOR_TORQUE_NM,
    RIGIDBODY_DRAG_PER_S,
    RIGIDBODY_INERTIA_DIAGONAL_KGM2,
    RIGIDBODY_INERTIA_ROTATION_XYZW,
    RIGIDBODY_MASS_KG,
    STEERING_LIMIT_RAD,
    STEERING_RATE_RADPS,
    WHEEL_DAMPING_RATE_NMS,
    WHEEL_RADIUS_M,
    ackermann_center_from_wheel_angles,
    inertia_tensor_body,
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
DEFAULT_WHEEL_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "wheel_spin_state_inputs_v10_20261013.json")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "source_mechanics_3d_plant_rollout_20261013.json")
WHEELS = ("fl", "fr", "rl", "rr")
WHEEL_KEYS = {"fl": "front_left", "fr": "front_right",
              "rl": "rear_left", "rr": "rear_right"}
HORIZON_MILLISECONDS = (25, 100, 250, 500, 1000, 2000, 3000, 4000, 5000)


@njit(cache=False)
def _cross(a, b):
    return np.asarray((a[1] * b[2] - a[2] * b[1],
                       a[2] * b[0] - a[0] * b[2],
                       a[0] * b[1] - a[1] * b[0]))


@njit(cache=False)
def _qrotate(q, vector):
    qv = q[:3]
    twice_cross = 2.0 * _cross(qv, vector)
    return vector + q[3] * twice_cross + _cross(qv, twice_cross)


@njit(cache=False)
def _qmul(a, b):
    return np.asarray((
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ))


@njit(cache=False)
def _qnormalized(q):
    norm = math.sqrt(np.dot(q, q))
    return q / norm


@njit(cache=False)
def _qstep(q, omega_body, dt):
    speed = math.sqrt(np.dot(omega_body, omega_body))
    if speed < 1.0e-12:
        return q.copy()
    half_angle = 0.5 * speed * dt
    scale = math.sin(half_angle) / speed
    delta = np.asarray((omega_body[0] * scale,
                        omega_body[1] * scale,
                        omega_body[2] * scale,
                        math.cos(half_angle)))
    return _qnormalized(_qmul(q, delta))


@njit(cache=False)
def _curve(abs_slip, curve, tangent):
    extremum_slip, extremum_value, asymptote_slip, asymptote_value, stiffness = curve
    if abs_slip < extremum_slip:
        t = abs_slip / extremum_slip
        value = (tangent * extremum_slip * (t**3 - 2.0*t*t + t)
                 + extremum_value * (-2.0*t**3 + 3.0*t*t))
    elif abs_slip < asymptote_slip:
        t = (abs_slip - extremum_slip) / (asymptote_slip - extremum_slip)
        value = extremum_value + (asymptote_value - extremum_value) * (
            3.0*t*t - 2.0*t**3)
    else:
        value = asymptote_value
    return value * stiffness


@njit(cache=False)
def _plant_derivative(state, center_steering, throttle, constants,
                      yaw_acceleration_residual=0.0,
                      body_x_acceleration_residual=0.0,
                      body_z_acceleration_residual=0.0,
                      relaxed_tire_forces=None,
                      documented_tire_slips=False):
    # Packed constants: wheel mounts, wheel down axes, wheel static quaternions,
    # wheel force-app levers, springs, dampers, sprung masses, calibrated target
    # travels, per-wheel curves, tire parameters, inertia tensor and inverse,
    # center of mass, effective wheel inertia, slip regularizer, radius scale.
    (mounts, down_local, wheel_q, app_levers, springs, dampers, sprung,
     target_travel, curves_x, curves_y, tire_parameters, inertia, inverse_i,
     com, wheel_inertia, slip_epsilon, radius_scale) = constants

    q = state[3:7]
    position = state[:3]
    velocity = state[7:10]
    omega_body = state[10:13]
    wheel_omega = state[13:17]
    q_inv = q.copy()
    q_inv[:3] *= -1.0
    ground_normal_body = _qrotate(q_inv, np.asarray((0.0, 1.0, 0.0)))
    body_force = np.zeros(3)
    body_torque = np.zeros(3)
    wheel_alpha = np.zeros(4)

    tangent = math.tan(center_steering)
    numerator = 2.0 * 0.324 * tangent
    left = math.atan2(numerator, 2.0 * 0.324 - 0.236 * tangent)
    right = math.atan2(numerator, 2.0 * 0.324 + 0.236 * tangent)
    physical_steers = np.asarray((-left, -right, 0.0, 0.0))

    for index in range(4):
        body_mount = mounts[index]
        down = down_local[index]
        center_world = position + _qrotate(q, body_mount)
        down_world = _qrotate(q, down)
        if abs(down_world[1]) < 1.0e-6:
            return np.full(18, np.nan)
        travel = (WHEEL_RADIUS_M - center_world[1]) / down_world[1]
        center_relative_com = (body_mount + travel * down - com)
        velocity_center = velocity + _cross(omega_body, center_relative_com)
        travel_rate = -np.dot(ground_normal_body, velocity_center) / np.dot(
            ground_normal_body, down)
        load = max(0.0, sprung[index] * GRAVITY_MPS2
                   + springs[index] * (target_travel[index] - travel)
                   - dampers[index] * travel_rate)

        local_forward = np.asarray((math.sin(physical_steers[index]),
                                    0.0,
                                    math.cos(physical_steers[index])))
        wheel_forward = _qrotate(wheel_q[index], local_forward)
        wheel_forward -= ground_normal_body * np.dot(
            wheel_forward, ground_normal_body)
        forward_norm = math.sqrt(np.dot(wheel_forward, wheel_forward))
        wheel_forward /= max(forward_norm, 1.0e-12)
        wheel_side = _cross(ground_normal_body, wheel_forward)
        side_norm = math.sqrt(np.dot(wheel_side, wheel_side))
        wheel_side /= max(side_norm, 1.0e-12)

        contact = body_mount + travel * down - WHEEL_RADIUS_M * ground_normal_body
        contact_lever = contact - com
        contact_velocity = velocity + _cross(omega_body, contact_lever)
        forward_speed = np.dot(contact_velocity, wheel_forward)
        lateral_speed = np.dot(contact_velocity, wheel_side)
        surface_speed = wheel_omega[index] * WHEEL_RADIUS_M * radius_scale
        if documented_tire_slips:
            # AutoDRIVE's documented definitions are Sx=(r*w-vx)/vx and
            # Sy=vy/|vx|. The equations are singular at standstill; use the
            # independently fitted speed regularizer only in that limit.
            denominator = max(abs(forward_speed), slip_epsilon)
            slip_x = (surface_speed - forward_speed) / denominator
            slip_y = lateral_speed / denominator
        else:
            slip_x = ((surface_speed - forward_speed)
                      / (max(abs(surface_speed), abs(forward_speed))
                         + slip_epsilon))
            slip_y = math.atan2(lateral_speed, abs(forward_speed) + 1.0)
        fx = (math.copysign(1.0, slip_x) * load
              * tire_parameters[0]
              * _curve(abs(slip_x), curves_x[index], tire_parameters[1]))
        fy = (-math.copysign(1.0, slip_y) * load
              * tire_parameters[2]
              * _curve(abs(slip_y), curves_y[index], tire_parameters[3]))
        # Optional research hypothesis: rear lateral force falls after the
        # serialized longitudinal tire curve passes its peak. A zero gain is
        # an exact no-op, preserving the baseline plant.
        if index >= 2 and tire_parameters[4] > 0.0:
            excess_longitudinal_slip = max(
                abs(slip_x) - tire_parameters[5], 0.0)
            fy *= math.exp(-tire_parameters[4] * excess_longitudinal_slip)
        if relaxed_tire_forces is not None:
            fx = relaxed_tire_forces[index, 0]
            fy = relaxed_tire_forces[index, 1]
        tire_force = fx * wheel_forward + fy * wheel_side
        normal_force = load * ground_normal_body
        total_force = tire_force + normal_force
        body_force += total_force
        body_torque += _cross(app_levers[index], total_force)

        if throttle == 0.0:
            motor = 0.0
        else:
            motor = MOTOR_TORQUE_NM * throttle / 4.0
        wheel_alpha[index] = (
            motor - WHEEL_RADIUS_M * fx
            - WHEEL_DAMPING_RATE_NMS * wheel_omega[index]) / wheel_inertia

    gravity_body = _qrotate(q_inv, np.asarray((0.0, -GRAVITY_MPS2, 0.0)))
    body_force -= RIGIDBODY_MASS_KG * RIGIDBODY_DRAG_PER_S * velocity
    linear_acceleration = (body_force / RIGIDBODY_MASS_KG + gravity_body
                           - _cross(omega_body, velocity))
    linear_acceleration[0] += body_x_acceleration_residual
    linear_acceleration[2] += body_z_acceleration_residual
    angular_momentum = inertia @ omega_body
    angular_rhs = (body_torque - _cross(omega_body, angular_momentum)
                   - 0.1 * angular_momentum)
    angular_acceleration = inverse_i @ angular_rhs
    angular_acceleration[1] += yaw_acceleration_residual

    derivative = np.zeros(18)
    # Rigidbody.position is the body-transform origin, while Rigidbody.velocity
    # is the COM velocity. Keep those distinct when the configured COM is offset.
    derivative[:3] = _qrotate(
        q, velocity - _cross(omega_body, com))
    derivative[3:7] = 0.5 * _qmul(
        q, np.asarray((omega_body[0], omega_body[1], omega_body[2], 0.0)))
    derivative[7:10] = linear_acceleration
    derivative[10:13] = angular_acceleration
    derivative[13:17] = wheel_alpha
    derivative[17] = 0.0
    return derivative


@njit(cache=False)
def _advance(state, steering_command, throttle, constants, dt,
             yaw_acceleration_residual=0.0,
             body_x_acceleration_residual=0.0,
             body_z_acceleration_residual=0.0,
             unity_autonomous_steer=True,
             relaxed_tire_forces=None,
             documented_tire_slips=False):
    result = state.copy()
    if unity_autonomous_steer:
        # Mirror VehicleController.Steer() exactly, including its comparison
        # between normalized command and last physical steering angle. State
        # 17 uses the opposite sign of the controller's SteeringAngle.
        command = np.float32(min(1.0, max(-1.0, steering_command)))
        limit_degrees = np.float32(30.0)
        rate_degrees_per_second = np.float32(183.346)
        steering_angle_degrees = np.float32(
            -state[17] * (180.0 / math.pi))
        target_degrees = np.float32(-command * limit_degrees)
        increment = np.float32(rate_degrees_per_second * np.float32(dt))
        if command > np.float32(steering_angle_degrees / limit_degrees):
            steering_angle_degrees = np.float32(
                steering_angle_degrees - increment)
            if steering_angle_degrees <= target_degrees:
                steering_angle_degrees = target_degrees
        else:
            steering_angle_degrees = np.float32(
                steering_angle_degrees + increment)
            if steering_angle_degrees >= target_degrees:
                steering_angle_degrees = target_degrees
        center_steering = -float(steering_angle_degrees) * (math.pi / 180.0)
    else:
        target = min(STEERING_LIMIT_RAD,
                     max(-STEERING_LIMIT_RAD,
                         steering_command * STEERING_LIMIT_RAD))
        maximum_step = STEERING_RATE_RADPS * dt
        delta = target - state[17]
        center_steering = state[17] + min(maximum_step,
                                          max(-maximum_step, delta))
    # The capture is taken inside FixedUpdate after VehicleController.Steer()
    # has set the WheelCollider angle, but before Unity simulates this physics
    # step. Thus state[17] is the wheel angle used for this interval; the
    # computed center_steering is the next boundary's actuator state.
    force_steering = state[17] if unity_autonomous_steer else center_steering
    derivative = _plant_derivative(
        state, force_steering, throttle, constants,
        yaw_acceleration_residual, body_x_acceleration_residual,
        body_z_acceleration_residual, relaxed_tire_forces,
        documented_tire_slips)
    if not np.isfinite(derivative).all():
        return np.full(18, np.nan)
    result[:3] += derivative[:3] * dt
    result[3:7] = _qnormalized(state[3:7] + derivative[3:7] * dt)
    result[7:13] += derivative[7:13] * dt
    result[13:17] += derivative[13:17] * dt
    if throttle == 0.0:
        impulse = MOTOR_TORQUE_NM * dt / constants[14]
        for index in range(4):
            if result[13 + index] > 0.0:
                result[13 + index] = max(0.0, result[13 + index] - impulse)
            elif result[13 + index] < 0.0:
                result[13 + index] = min(0.0, result[13 + index] + impulse)
    result[17] = center_steering
    return result


def _rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = q[:3]
    twice = 2.0 * np.cross(xyz, vector)
    return vector + q[3] * twice + np.cross(xyz, twice)


def _prepare_constants(metadata: dict[str, str], tire_fit_path: Path,
                       slip_fit_path: Path, suspension_fit_path: Path,
                       wheel_fit_path: Path,
                       rear_lateral_spin_attenuation: float = 0.0,
                       rear_lateral_spin_threshold: float = 0.15,
                       use_tire_fit_lateral: bool = False
                       ) -> tuple[Any, dict[str, Any]]:
    if (not math.isfinite(rear_lateral_spin_attenuation)
            or rear_lateral_spin_attenuation < 0.0):
        raise ValueError("rear lateral spin attenuation must be finite and nonnegative")
    if (not math.isfinite(rear_lateral_spin_threshold)
            or rear_lateral_spin_threshold < 0.0):
        raise ValueError("rear lateral spin threshold must be finite and nonnegative")
    tire_report = json.loads(tire_fit_path.read_text(encoding="utf-8"))
    tire = tire_report["curve_model"]["training_fit"]
    wheel_report = json.loads(wheel_fit_path.read_text(encoding="utf-8"))
    lateral = (np.asarray((tire["lateral_scale"],
                           tire["lateral_zero_slip_tangent"]))
               if use_tire_fit_lateral else
               wheel_report["source_force_application_point_lateral_fit"][
                   "fit"]["parameters"])
    wheel_balance = wheel_report["training_fit_state_reconstructed_inputs"]
    slip_report = json.loads(slip_fit_path.read_text(encoding="utf-8"))
    slip = slip_report["shared_max_speed_plus_regularizer_candidate"]
    suspension_report = json.loads(
        suspension_fit_path.read_text(encoding="utf-8"))
    suspension = suspension_report["geometry_only_load_model"]

    center_of_mass = np.asarray([
        float(value) for value in metadata["center_of_mass_body"].split(",")],
        dtype=np.float64)
    mounts = np.zeros((4, 3), dtype=np.float64)
    down_local = np.zeros((4, 3), dtype=np.float64)
    wheel_q = np.zeros((4, 4), dtype=np.float64)
    app_levers = np.zeros((4, 3), dtype=np.float64)
    springs = np.zeros(4)
    dampers = np.zeros(4)
    sprung = np.zeros(4)
    target_travel = np.zeros(4)
    curves_x = np.zeros((4, 5))
    curves_y = np.zeros((4, 5))
    for index, wheel in enumerate(WHEELS):
        wheel_key = WHEEL_KEYS[wheel]
        mount_tuple, quaternion_tuple = WHEEL_TRANSFORM[wheel]
        q_static = np.asarray(quaternion_tuple, dtype=np.float64)
        collider_center = _rotate(q_static, np.asarray((0.0, 0.025, 0.0)))
        mounts[index] = np.asarray(mount_tuple) + collider_center
        down_local[index] = _rotate(q_static, np.asarray((0.0, -1.0, 0.0)))
        wheel_q[index] = q_static
        radius = float(metadata[f"{wheel_key}_radius_m"])
        target_position = float(metadata[f"{wheel_key}_target_position"])
        distance = float(metadata[f"{wheel_key}_suspension_distance_m"])
        app_distance = float(metadata[f"{wheel_key}_force_app_point_distance_m"])
        rest_travel = (1.0 - target_position) * distance
        app_point = (mounts[index] + down_local[index] * (rest_travel + radius)
                     + np.asarray((0.0, app_distance, 0.0)))
        app_levers[index] = app_point - center_of_mass
        springs[index] = float(metadata[f"{wheel_key}_spring_n_per_m"])
        dampers[index] = float(metadata[f"{wheel_key}_damper_ns_per_m"])
        sprung[index] = float(metadata[f"{wheel_key}_sprung_mass_kg"])
        target_travel[index] = float(
            suspension[wheel]["training_calibrated_target_travel_m"])
        curves_x[index] = np.asarray([float(value) for value in metadata[
            f"{wheel_key}_forward_curve"].split(",")])
        curves_y[index] = np.asarray([float(value) for value in metadata[
            f"{wheel_key}_sideways_curve"].split(",")])

    inertia = np.asarray(inertia_tensor_body(
        RIGIDBODY_INERTIA_DIAGONAL_KGM2,
        RIGIDBODY_INERTIA_ROTATION_XYZW), dtype=np.float64)
    constants = (
        mounts, down_local, wheel_q, app_levers, springs, dampers, sprung,
        target_travel, curves_x, curves_y,
        np.asarray((tire["longitudinal_scale"],
                    tire["longitudinal_zero_slip_tangent"],
                    lateral[0], lateral[1],
                    rear_lateral_spin_attenuation,
                    rear_lateral_spin_threshold), dtype=np.float64),
        inertia, np.linalg.inv(inertia), center_of_mass,
        float(wheel_balance["effective_inertia_kgm2"]),
        float(slip["fitted_speed_regularizer_mps"]),
        float(slip["fitted_radius_scale"]),
    )
    description = {
        "lateral_curve_parameter_source": (
            "tire force-balance fit" if use_tire_fit_lateral else
            "wheel-spin state input fit"),
        "tire_parameters": constants[10].tolist(),
        "rear_lateral_spin_attenuation_per_slip": (
            rear_lateral_spin_attenuation),
        "rear_lateral_spin_threshold": rear_lateral_spin_threshold,
        "wheel_effective_inertia_kgm2": constants[14],
        "wheel_damping_nms": WHEEL_DAMPING_RATE_NMS,
        "longitudinal_slip_regularizer_mps": constants[15],
        "radius_scale": constants[16],
        "calibrated_suspension_target_travel_m": target_travel.tolist(),
        "force_application_point": "serialized WheelCollider point; normal and tire force moment use that lever arm",
        "braking": "zero-throttle torque impulse with unilateral zero-speed stop",
        "integrator": "explicit 1 ms rigid-body update; quaternion normalized each step",
    }
    return constants, description


def _initial_state(data: dict[str, np.ndarray], index: int) -> np.ndarray:
    state = np.zeros(18, dtype=np.float64)
    for axis, field in enumerate(("pos_world_x_m", "pos_world_y_m",
                                  "pos_world_z_m")):
        state[axis] = data[field][index]
    state[3:7] = np.asarray((data["rot_world_x"][index],
                             data["rot_world_y"][index],
                             data["rot_world_z"][index],
                             data["rot_world_w"][index]))
    state[3:7] /= np.linalg.norm(state[3:7])
    state[7:10] = np.asarray((data["vel_body_x_mps"][index],
                              data["vel_body_y_mps"][index],
                              data["vel_body_z_mps"][index]))
    state[10:13] = np.asarray((data["omega_body_x_rps"][index],
                               data["omega_body_y_rps"][index],
                               data["omega_body_z_rps"][index]))
    for wheel_index, wheel in enumerate(WHEELS):
        state[13 + wheel_index] = (
            data[f"{wheel}_rpm"][index] * 2.0 * math.pi / 60.0)
    state[17] = ackermann_center_from_wheel_angles(
        -math.radians(data["fl_steer_deg"][index]),
        -math.radians(data["fr_steer_deg"][index]))
    return state


def _truth_state(data: dict[str, np.ndarray], index: int) -> np.ndarray:
    return _initial_state(data, index)


def _heading(q: np.ndarray) -> float:
    forward = _rotate(q, np.asarray((0.0, 0.0, 1.0)))
    return math.atan2(-float(forward[0]), float(forward[2]))


def _state_error(predicted: np.ndarray,
                 truth: np.ndarray) -> dict[str, Any]:
    position = predicted[:3] - truth[:3]
    qdot = abs(float(np.dot(predicted[3:7], truth[3:7])))
    orientation = 2.0 * math.acos(min(1.0, max(-1.0, qdot)))
    return {
        "position_horizontal_m": float(np.hypot(position[0], position[2])),
        "position_vertical_m": float(position[1]),
        "orientation_rad": orientation,
        "heading_rad": _wrap(_heading(predicted[3:7])
                              - _heading(truth[3:7])),
        "body_velocity_xyz_mps": (predicted[7:10] - truth[7:10]).tolist(),
        "body_omega_xyz_radps": (predicted[10:13] - truth[10:13]).tolist(),
        "wheel_omega_radps": (predicted[13:17] - truth[13:17]).tolist(),
        "steering_rad": float(predicted[17] - truth[17]),
    }


def _evaluate_capture(path: Path, constants: Any, window_stride_s: float,
                      horizon_s: float,
                      documented_tire_slips: bool = False) -> dict[str, Any]:
    data, metadata = _load_capture(path)
    dt = float(metadata["fixed_delta_time_s"])
    if not math.isclose(dt, DT_S, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"expected 1 kHz source data, got dt={dt}")
    steps = round(horizon_s / dt)
    checkpoints = tuple(
        round(milliseconds / 1000.0 / dt)
        for milliseconds in HORIZON_MILLISECONDS
        if milliseconds / 1000.0 <= horizon_s)
    stride = round(window_stride_s / DT_S)
    starts = list(range(100, len(data["speed_mps"]) - steps - 1, stride))
    starts = [start for start in starts if data["speed_mps"][start] >= 2.0]
    starts = [start for start in starts if all(
        data[f"{wheel}_grounded"][start:start + steps + 1].min() > 0.5
        for wheel in WHEELS)]
    positions = np.stack((data["pos_world_x_m"], data["pos_world_y_m"],
                           data["pos_world_z_m"]), axis=1)
    position_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed_steps = np.maximum(data["speed_mps"][:-1],
                             data["speed_mps"][1:])
    # Rigidbody motion over 1 ms cannot traverse three times the observed
    # speed-based displacement plus 1 cm. The large outliers in these source
    # captures are simulator resets; a rollout spanning one is not meaningful.
    discontinuity = position_steps > (3.0 * speed_steps * dt + 0.01)
    discontinuity_prefix = np.concatenate((
        np.zeros(1, dtype=np.int64), np.cumsum(discontinuity, dtype=np.int64)))
    pre_reset_window_count = len(starts)
    starts = [start for start in starts
              if discontinuity_prefix[start + steps + 1]
              == discontinuity_prefix[start + 1]]
    if not starts:
        raise ValueError(f"no fully grounded windows in {path}")
    errors: dict[int, list[dict[str, Any]]] = {step: [] for step in checkpoints}
    window_diagnostics: list[dict[str, Any]] = []
    for start in starts:
        state = _initial_state(data, start)
        window_end = start + steps
        steering_window = data["steering_command"][start:window_end]
        throttle_window = data["throttle_command"][start:window_end]
        window_row: dict[str, Any] = {
            "start_sample": start,
            "initial_speed_mps": float(data["speed_mps"][start]),
            "initial_abs_steering_rad": abs(float(state[17])),
            "command_summary": {
                "abs_steering_mean_normalized": float(np.mean(
                    np.abs(steering_window))),
                "abs_steering_max_normalized": float(np.max(
                    np.abs(steering_window))),
                "throttle_mean": float(np.mean(throttle_window)),
                "abs_throttle_mean": float(np.mean(np.abs(throttle_window))),
                "zero_throttle_fraction": float(np.mean(
                    throttle_window == 0.0)),
            },
            "errors": {},
        }
        for offset in range(steps):
            index = start + offset
            state = _advance(
                state, data["steering_command"][index],
                data["throttle_command"][index], constants, DT_S,
                documented_tire_slips=documented_tire_slips)
            if not np.isfinite(state).all():
                raise FloatingPointError(
                    f"non-finite 3-D state at {path}:{index}")
            elapsed = offset + 1
            if elapsed in errors:
                error = _state_error(state, _truth_state(data, index + 1))
                errors[elapsed].append(error)
                window_row["errors"][f"{elapsed * dt:.3f}s"] = error
        window_diagnostics.append(window_row)

    summary = {}
    for step, rows in errors.items():
        vel = np.asarray([row["body_velocity_xyz_mps"] for row in rows])
        omega = np.asarray([row["body_omega_xyz_radps"] for row in rows])
        wheels = np.asarray([row["wheel_omega_radps"] for row in rows])
        heading = np.asarray([row["heading_rad"] for row in rows])
        summary[f"{step * dt:.3f}s"] = {
            "windows": len(rows),
            "position_horizontal_rmse_m": float(np.sqrt(np.mean(
                np.asarray([row["position_horizontal_m"] for row in rows])**2))),
            "position_horizontal_p95_m": float(np.quantile(
                [row["position_horizontal_m"] for row in rows], 0.95)),
            "position_vertical_rmse_m": float(np.sqrt(np.mean(
                np.asarray([row["position_vertical_m"] for row in rows])**2))),
            "orientation_rmse_rad": float(np.sqrt(np.mean(
                np.asarray([row["orientation_rad"] for row in rows])**2))),
            "heading_rmse_rad": float(np.sqrt(np.mean(heading**2))),
            "heading_bias_rad": float(np.mean(heading)),
            "body_velocity_xyz_rmse_mps": np.sqrt(np.mean(vel**2, axis=0)).tolist(),
            "body_omega_xyz_rmse_radps": np.sqrt(np.mean(omega**2, axis=0)).tolist(),
            "body_omega_xyz_bias_radps": np.mean(omega, axis=0).tolist(),
            "wheel_omega_each_rmse_radps": np.sqrt(np.mean(wheels**2, axis=0)).tolist(),
            "steering_rmse_rad": float(np.sqrt(np.mean(
                np.asarray([row["steering_rad"] for row in rows])**2))),
        }
    return {
        "capture": str(path.relative_to(ROOT)),
        "rows": len(data["speed_mps"]),
        "window_count": len(starts),
        "excluded_reset_crossing_windows": pre_reset_window_count - len(starts),
        "detected_position_discontinuities": np.flatnonzero(
            discontinuity).astype(int).tolist(),
        "window_stride_s": stride * DT_S,
        "horizon_s": steps * dt,
        "initial_speed_p05_p50_p95_mps": np.quantile(
            [data["speed_mps"][start] for start in starts],
            (0.05, 0.50, 0.95)).tolist(),
        "rollout_errors": summary,
        "window_diagnostics": window_diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--tire-fit", type=Path, default=DEFAULT_TIRE_FIT)
    parser.add_argument("--slip-fit", type=Path, default=DEFAULT_SLIP_FIT)
    parser.add_argument("--suspension-fit", type=Path,
                        default=DEFAULT_SUSPENSION_FIT)
    parser.add_argument("--wheel-fit", type=Path, default=DEFAULT_WHEEL_FIT)
    parser.add_argument("--slip-definition",
                        choices=("existing", "documented"),
                        default="existing")
    parser.add_argument("--use-tire-fit-lateral", action="store_true",
                        help="use lateral coefficients from --tire-fit instead "
                        "of the independently fitted wheel-state report")
    parser.add_argument("--window-stride-seconds", type=float, default=5.0)
    parser.add_argument("--horizon-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 0.025 <= args.horizon_seconds <= 5.0:
        parser.error("horizon must be between 0.025 and 5 seconds")
    if args.window_stride_seconds <= 0.0:
        parser.error("window stride must be positive")
    metadata = _metadata(args.train)
    constants, model = _prepare_constants(
        metadata, args.tire_fit, args.slip_fit,
        args.suspension_fit, args.wheel_fit,
        use_tire_fit_lateral=args.use_tire_fit_lateral)
    report = {
        "schema_version": 1,
        "status": "research-only 3-D source mechanics plant; not runtime odometry or an exact native WheelCollider clone",
        "training_capture": str(args.train.relative_to(ROOT)),
        "heldout_capture": str(args.heldout.relative_to(ROOT)),
        "training_only_parameter_sources": {
            "tire_force_curve": str(args.tire_fit.relative_to(ROOT)),
            "slip_regularizer": str(args.slip_fit.relative_to(ROOT)),
            "suspension_target_travel": str(args.suspension_fit.relative_to(ROOT)),
            "wheel_inertia_and_lateral_curve": str(args.wheel_fit.relative_to(ROOT)),
        },
        "model": model,
        "parameter_selection": {
            "longitudinal_curve": "--tire-fit training_fit",
            "lateral_curve": ("--tire-fit training_fit" if
                              args.use_tire_fit_lateral else
                              "--wheel-fit source force application point fit"),
        },
        "tire_slip_definition": {
            "selected": args.slip_definition,
            "equations": (
                "Sx=(r*wheelOmega-vx)/max(|vx|, fitted_regularizer); "
                "Sy=vy/max(|vx|, fitted_regularizer)"
                if args.slip_definition == "documented" else
                "existing fitted bounded longitudinal slip ratio and "
                "atan2 lateral slip angle"),
            "low_speed_handling": (
                "denominator uses the independently fitted 0.1005 m/s "
                "regularizer near standstill; the published equations are "
                "singular at vx=0" if args.slip_definition == "documented"
                else "unchanged existing formulation"),
        },
        "state": ["world_position_xyz", "world_quaternion_xyzw",
                  "body_velocity_xyz", "body_angular_velocity_xyz",
                  "four_wheel_angular_velocities", "center_steering"],
        "future_inputs": "recorded steering/throttle command at each 1 ms source step; no future simulator truth or WheelHit input",
        "evaluation_horizon_s": args.horizon_seconds,
        "window_stride_s": args.window_stride_seconds,
        "training": _evaluate_capture(
            args.train, constants, args.window_stride_seconds,
            args.horizon_seconds,
            documented_tire_slips=args.slip_definition == "documented"),
        "independent_heldout": _evaluate_capture(
            args.heldout, constants, args.window_stride_seconds,
            args.horizon_seconds,
            documented_tire_slips=args.slip_definition == "documented"),
        "caveats": [
            "flat ground plane is assumed; the selected source captures are open-plane runs",
            "per-wheel tire forces use the documented curve shape with fitted shared coefficients",
            "wheel normal load is reconstructed from a calibrated flat-plane spring/damper geometry",
            "force application point uses serialized WheelCollider geometry",
            "native WheelCollider solver details are private; full recursive accuracy is not established",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    for split in ("training", "independent_heldout"):
        print(split, "windows", report[split]["window_count"])
        for horizon in report[split]["rollout_errors"]:
            row = report[split]["rollout_errors"][horizon]
            print(horizon, "position", round(
                row["position_horizontal_rmse_m"], 4),
                "vxyz", [round(value, 3) for value in
                         row["body_velocity_xyz_rmse_mps"]],
                "omega", [round(value, 3) for value in
                          row["body_omega_xyz_rmse_radps"]],
                "wheel", [round(value, 3) for value in
                          row["wheel_omega_each_rmse_radps"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

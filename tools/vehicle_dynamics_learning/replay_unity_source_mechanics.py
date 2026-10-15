#!/usr/bin/env python3
"""Run a command-only planar plant from the captured Unity vehicle mechanics.

This is an offline model experiment. It initializes each prediction window
from one captured state, then consumes only captured actuator commands. Unity
WheelHit, future pose, and future body/wheel truth are used only for scoring.
The rigid body and wheel torque equations are explicit; tire force is the
source-captured two-piece curve with force-balance-fit parameters.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import savgol_coeffs

from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    CAPTURED_COM_HEIGHT_ABOVE_ROAD_M,
    MOTOR_TORQUE_NM,
    RIGIDBODY_DRAG_PER_S,
    RIGIDBODY_MASS_KG,
    RIGIDBODY_YAW_INERTIA_KGM2,
    STEERING_LIMIT_RAD,
    STEERING_RATE_RADPS,
    TRACK_M,
    WHEEL_DAMPING_RATE_NMS,
    WHEEL_RADIUS_M,
    WHEEL_SPRUNG_MASSES_KG,
    WHEEL_X_M,
    WHEEL_Y_M,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
DEFAULT_VALIDATION = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_exact_replay_20261012_r11/"
    "wheel_dynamics_unity_vehicle_model_parity_source_exact_replay_20261012_r11.csv")
DEFAULT_FORCE_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "tire_curve_force_balance_20261012.json")
WHEELS = ("fl", "fr", "rl", "rr")
DT_S = 0.001
ACKERMANN_WHEELBASE_M = 0.324
AXLE_SPAN_M = WHEEL_X_M[0] - WHEEL_X_M[2]
GRAVITY_MPS2 = 9.80665
HORIZONS_STEPS = (5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250,
                  500, 750, 1000, 2000, 5000)
WINDOW_STRIDE_STEPS = 5000


@njit(cache=False)
def _curve(abs_slip, extremum_slip, extremum_value,
           asymptote_slip, asymptote_value, stiffness, tangent):
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
def _derivative(state, steering, throttle, slip_floor,
                longitudinal_scale, longitudinal_tangent,
                lateral_scale, lateral_tangent,
                wheel_inertia, longitudinal_curve, lateral_curve,
                static_loads, brake_impulse_mode, slip_law=0):
    u, v, yaw_rate = state[0], state[1], state[2]
    omega = state[3:7]
    ax_load, ay_load = state[8], state[9]
    tangent = math.tan(steering)
    ackermann_num = 2.0 * ACKERMANN_WHEELBASE_M * tangent
    left_angle = math.atan2(
        ackermann_num, 2.0 * ACKERMANN_WHEELBASE_M - TRACK_M * tangent)
    right_angle = math.atan2(
        ackermann_num, 2.0 * ACKERMANN_WHEELBASE_M + TRACK_M * tangent)
    angles = (left_angle, right_angle, 0.0, 0.0)

    fx_body = np.zeros(4)
    fy_body = np.zeros(4)
    wheel_alpha = np.zeros(4)
    for index in range(4):
        x = WHEEL_X_M[index]
        y = WHEEL_Y_M[index]
        wheel_u_body = u - yaw_rate * y
        wheel_v_body = v + yaw_rate * x
        delta = angles[index]
        c, s = math.cos(delta), math.sin(delta)
        vx = c * wheel_u_body + s * wheel_v_body
        vy = -s * wheel_u_body + c * wheel_v_body
        if slip_law == 0:
            vx_denominator = max(abs(vx), slip_floor)
            slip_x = (WHEEL_RADIUS_M * omega[index] - vx) / vx_denominator
            slip_y = vy / vx_denominator
        else:
            wheel_surface_speed = WHEEL_RADIUS_M * omega[index]
            slip_x = ((wheel_surface_speed - vx)
                      / (max(abs(wheel_surface_speed), abs(vx))
                         + 0.10054783962080052))
            slip_y = math.atan2(vy, abs(vx) + 1.0)

        load_long = (RIGIDBODY_MASS_KG * ax_load
                     * CAPTURED_COM_HEIGHT_ABOVE_ROAD_M
                     / (2.0 * AXLE_SPAN_M))
        load_lat = (RIGIDBODY_MASS_KG * ay_load
                    * CAPTURED_COM_HEIGHT_ABOVE_ROAD_M
                    / (2.0 * TRACK_M))
        side_sign = -1.0 if index % 2 == 0 else 1.0
        axle_sign = -1.0 if index < 2 else 1.0
        load = max(0.0, static_loads[index]
                   + axle_sign * load_long + side_sign * load_lat)

        fx_local = (math.copysign(1.0, slip_x) * load * longitudinal_scale
                    * _curve(abs(slip_x), longitudinal_curve[0],
                             longitudinal_curve[1], longitudinal_curve[2],
                             longitudinal_curve[3], longitudinal_curve[4],
                             longitudinal_tangent))
        fy_local = (-math.copysign(1.0, slip_y) * load * lateral_scale
                    * _curve(abs(slip_y), lateral_curve[0],
                             lateral_curve[1], lateral_curve[2],
                             lateral_curve[3], lateral_curve[4],
                             lateral_tangent))
        fx_body[index] = c * fx_local - s * fy_local
        fy_body[index] = s * fx_local + c * fy_local

        if throttle == 0.0:
            motor_torque = 0.0
            brake_reaction = (0.0 if brake_impulse_mode else
                              MOTOR_TORQUE_NM * math.copysign(
                                  1.0, omega[index])
                              if omega[index] != 0.0 else 0.0)
        else:
            motor_torque = MOTOR_TORQUE_NM * throttle / 4.0
            brake_reaction = 0.0
        wheel_alpha[index] = (
            motor_torque - brake_reaction - WHEEL_RADIUS_M * fx_local
            - WHEEL_DAMPING_RATE_NMS * omega[index]) / wheel_inertia

    tire_fx = np.sum(fx_body)
    tire_fy = np.sum(fy_body)
    drag_fx = RIGIDBODY_MASS_KG * RIGIDBODY_DRAG_PER_S * u
    drag_fy = RIGIDBODY_MASS_KG * RIGIDBODY_DRAG_PER_S * v
    force_ax = (tire_fx - drag_fx) / RIGIDBODY_MASS_KG
    force_ay = (tire_fy - drag_fy) / RIGIDBODY_MASS_KG
    moment = 0.0
    for index in range(4):
        moment += WHEEL_X_M[index] * fy_body[index] \
            - WHEEL_Y_M[index] * fx_body[index]
    yaw_accel = moment / RIGIDBODY_YAW_INERTIA_KGM2

    derivative = np.empty(10)
    derivative[0] = force_ax + yaw_rate * v
    derivative[1] = force_ay - yaw_rate * u
    derivative[2] = yaw_accel
    derivative[3:7] = wheel_alpha
    derivative[7] = 0.0
    derivative[8] = force_ax
    derivative[9] = force_ay
    return derivative


@njit(cache=False)
def _apply_brake_impulse(state, throttle, dt_s, wheel_inertia):
    if throttle != 0.0:
        return
    impulse_speed = MOTOR_TORQUE_NM * dt_s / wheel_inertia
    for index in range(4):
        speed = state[3 + index]
        if speed > 0.0:
            state[3 + index] = max(0.0, speed - impulse_speed)
        elif speed < 0.0:
            state[3 + index] = min(0.0, speed + impulse_speed)


@njit(cache=False)
def _advance(state, steering_command, throttle, slip_floor,
             longitudinal_scale, longitudinal_tangent,
             lateral_scale, lateral_tangent, wheel_inertia,
             longitudinal_curve, lateral_curve, static_loads,
             brake_impulse_mode, slip_law=0):
    next_steering = state[7]
    target = max(-STEERING_LIMIT_RAD,
                 min(STEERING_LIMIT_RAD,
                     steering_command * STEERING_LIMIT_RAD))
    step = STEERING_RATE_RADPS * DT_S
    next_steering += max(-step, min(step, target - next_steering))

    # Midpoint integration at the source player's 1 kHz physics interval.
    d0 = _derivative(state, state[7], throttle, slip_floor,
                     longitudinal_scale, longitudinal_tangent,
                     lateral_scale, lateral_tangent, wheel_inertia,
                     longitudinal_curve, lateral_curve, static_loads,
                     brake_impulse_mode, slip_law)
    midpoint = state + 0.5 * DT_S * d0
    # State slots 8/9 carry current body acceleration for load transfer. The
    # derivative returns acceleration, not jerk, so these slots are updated
    # algebraically rather than integrated as ordinary dynamic states.
    midpoint[8] = d0[8]
    midpoint[9] = d0[9]
    midpoint[7] = 0.5 * (state[7] + next_steering)
    if brake_impulse_mode:
        _apply_brake_impulse(midpoint, throttle, 0.5 * DT_S,
                             wheel_inertia)
    dmid = _derivative(midpoint, midpoint[7], throttle, slip_floor,
                       longitudinal_scale, longitudinal_tangent,
                       lateral_scale, lateral_tangent, wheel_inertia,
                       longitudinal_curve, lateral_curve, static_loads,
                       brake_impulse_mode, slip_law)
    result = state + DT_S * dmid
    result[8] = dmid[8]
    result[9] = dmid[9]
    if brake_impulse_mode:
        _apply_brake_impulse(result, throttle, DT_S, wheel_inertia)
    result[7] = next_steering
    return result


def _metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, value = line[2:].strip().split("=", 1)
            result[key] = value
    return result


def _load_capture(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    metadata = _metadata(path)
    usecols = ["capture_realtime_s", "pos_world_x_m", "pos_world_y_m",
               "pos_world_z_m",
               "rot_world_x", "rot_world_y", "rot_world_z", "rot_world_w",
               "vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
               "omega_body_x_rps", "omega_body_y_rps", "omega_body_z_rps",
               "steering_command", "throttle_command"]
    for wheel in WHEELS:
        usecols += [f"{wheel}_{name}" for name in (
            "rpm", "steer_deg", "grounded", "forward_slip", "sideways_slip")]
    frame = pd.read_csv(path, comment="#", usecols=usecols)
    data = {name: frame[name].to_numpy(dtype=np.float64)
            for name in frame.columns}
    data["speed_mps"] = np.linalg.norm(frame[[
        "vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps"]].to_numpy(), axis=1)
    return data, metadata


def _truth_state(data: dict[str, np.ndarray], index: int,
                 ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
    state = np.zeros(10, dtype=np.float64)
    state[0] = data["vel_body_z_mps"][index]
    state[1] = -data["vel_body_x_mps"][index]
    state[2] = -data["omega_body_y_rps"][index]
    for wheel_index, wheel in enumerate(WHEELS):
        state[3 + wheel_index] = (
            data[f"{wheel}_rpm"][index] * 2.0 * math.pi / 60.0)
    state[7] = -math.radians(0.5 * (
        data["fl_steer_deg"][index] + data["fr_steer_deg"][index]))
    state[8] = ax[index]
    state[9] = ay[index]
    return state


def _pose_truth(data: dict[str, np.ndarray], index: int) -> np.ndarray:
    qx = data["rot_world_x"][index]
    qy = data["rot_world_y"][index]
    qz = data["rot_world_z"][index]
    qw = data["rot_world_w"][index]
    forward_x = 2.0 * (qx * qz + qw * qy)
    forward_z = 1.0 - 2.0 * (qx*qx + qy*qy)
    heading = math.atan2(-forward_x, forward_z)
    return np.asarray((-data["pos_world_x_m"][index],
                       data["pos_world_z_m"][index], heading), dtype=np.float64)


def _integrate_pose(pose: np.ndarray, previous: np.ndarray,
                    following: np.ndarray) -> np.ndarray:
    u = 0.5 * (previous[0] + following[0])
    v = 0.5 * (previous[1] + following[1])
    yaw_rate = 0.5 * (previous[2] + following[2])
    heading_mid = pose[2] + 0.5 * yaw_rate * DT_S
    result = pose.copy()
    result[0] += (u * math.sin(heading_mid) + v * math.cos(heading_mid)) * DT_S
    result[1] += (u * math.cos(heading_mid) - v * math.sin(heading_mid)) * DT_S
    result[2] += yaw_rate * DT_S
    return result


def _initial_acceleration(data: dict[str, np.ndarray], dt: float) -> tuple[np.ndarray, np.ndarray]:
    u = data["vel_body_z_mps"]
    v = -data["vel_body_x_mps"]
    r = -data["omega_body_y_rps"]
    window = max(5, int(round(0.031 / dt)))
    if window % 2 == 0:
        window += 1
    coefficients = savgol_coeffs(
        window, 3, deriv=1, delta=dt, pos=window - 1, use="dot")
    u_dot = np.zeros_like(u)
    v_dot = np.zeros_like(v)
    for index in range(window - 1, len(u)):
        u_dot[index] = np.dot(u[index - window + 1:index + 1], coefficients)
        v_dot[index] = np.dot(v[index - window + 1:index + 1], coefficients)
    ax = u_dot - r * v
    ay = v_dot + r * u
    return ax, ay


def _run_window(data: dict[str, np.ndarray], start: int, steps: int,
                floor: float, params: dict[str, float], curves: tuple[np.ndarray, np.ndarray],
                static_loads: np.ndarray, initial_ax: np.ndarray,
                initial_ay: np.ndarray, brake_impulse_mode: bool,
                slip_law: int = 0
                ) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], int]:
    state = _truth_state(data, start, initial_ax, initial_ay)
    pose = _pose_truth(data, start)
    previous_state = state.copy()
    target_steps = set(step for step in HORIZONS_STEPS if step <= steps)
    scores = []
    completed = 0
    long_curve, lat_curve = curves
    for offset in range(steps):
        i = start + offset
        state_next = _advance(
            state, data["steering_command"][i], data["throttle_command"][i], floor,
            params["longitudinal_force_scale"],
            params["longitudinal_zero_slip_tangent"],
            params["lateral_force_scale"],
            params["lateral_zero_slip_tangent"],
            params["wheel_inertia_kgm2"], long_curve, lat_curve, static_loads,
            brake_impulse_mode, slip_law)
        pose_next = _integrate_pose(pose, state[:3], state_next[:3])
        if not np.isfinite(state_next).all() or not np.isfinite(pose_next).all():
            break
        state = state_next
        pose = pose_next
        completed = offset + 1
        if completed in target_steps:
            scores.append((completed, state.copy(), pose.copy()))
    return scores, completed


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _evaluate_capture(path: Path, floors: tuple[float, ...], window_seconds: float,
                      stride_steps: int, sample_limit: int) -> dict[str, Any]:
    data, metadata = _load_capture(path)
    dt = float(metadata["fixed_delta_time_s"])
    if not math.isclose(dt, DT_S, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"expected 1 kHz source capture, got dt={dt}")
    ax, ay = _initial_acceleration(data, dt)
    steps = round(window_seconds / dt)
    static_loads = np.asarray(WHEEL_SPRUNG_MASSES_KG) * GRAVITY_MPS2
    fit_info = json.loads(DEFAULT_FORCE_FIT.read_text(encoding="utf-8"))
    source_curves = fit_info["curve_model"]["training_fit"]
    fitted_curves = (
        np.asarray((0.15, 0.9, 0.25, 0.58, 0.8), dtype=np.float64),
        np.asarray((0.01, 1.0, 0.10, 0.50, 1.0), dtype=np.float64),
    )
    report_fits = {
        "r07_force_balance_fit_frozen": {
            "longitudinal_force_scale": source_curves["longitudinal_scale"],
            "longitudinal_zero_slip_tangent": source_curves[
                "longitudinal_zero_slip_tangent"],
            "lateral_force_scale": source_curves["lateral_scale"],
            "lateral_zero_slip_tangent": source_curves[
                "lateral_zero_slip_tangent"],
            "wheel_inertia_kgm2": fit_info["wheel_spin_balance"][
                "training_capture_fit"]["fitted_shared_wheel_inertia_kgm2"],
        },
    }
    lateral_report = json.loads((ROOT / (
        "live_runs/unity_exact_binary_audit_20261012/"
        "wheel_spin_state_inputs_v10_20261013.json")).read_text(
            encoding="utf-8"))
    lateral_fit = lateral_report[
        "source_force_application_point_lateral_fit"]["fit"]["parameters"]
    recursive_lateral = dict(report_fits["r07_force_balance_fit_frozen"])
    recursive_lateral["lateral_force_scale"] = lateral_fit[0]
    recursive_lateral["lateral_zero_slip_tangent"] = lateral_fit[1]
    report_fits["r07_joint_fit_lateral_curve_recursive_isolation"] = (
        recursive_lateral)
    results = {}
    for fit_name, fit_params in report_fits.items():
        fit_params = dict(fit_params)
        fit_params["wheel_damping_nms"] = WHEEL_DAMPING_RATE_NMS
        fit_results = {}
        for brake_mode_name, brake_impulse_mode in (
                ("continuous_coulomb", False),
                ("unilateral_impulse_lock", True)):
            brake_results = {}
            slip_cases = [
                (f"legacy_floor_{floor:g}mps", 0, floor)
                for floor in floors]
            slip_cases.append(("wheelhit_reconstructed", 1, 0.5))
            for slip_case, slip_law, floor in slip_cases:
                horizon_rows: dict[int, list[dict[str, Any]]] = {
                    horizon: [] for horizon in HORIZONS_STEPS if horizon <= steps}
                windows = 0
                nonfinite_windows = 0
                starts = list(range(100, len(data["speed_mps"]) - steps - 1,
                                    stride_steps))
                starts = [start for start in starts
                          if data["speed_mps"][start] >= 2.0]
                if sample_limit > 0:
                    starts = starts[:sample_limit]
                for start in starts:
                    wheel_grounded = all(
                        data[f"{wheel}_grounded"][start:start + steps].min() > 0.5
                        for wheel in WHEELS)
                    if not wheel_grounded:
                        continue
                    scores, completed = _run_window(
                        data, start, steps, floor, fit_params, fitted_curves,
                        static_loads, ax, ay, brake_impulse_mode, slip_law)
                    windows += 1
                    if completed < steps:
                        nonfinite_windows += 1
                    for horizon, predicted, predicted_pose in scores:
                        index = start + horizon
                        truth = np.asarray((
                            data["vel_body_z_mps"][index],
                            -data["vel_body_x_mps"][index],
                            -data["omega_body_y_rps"][index],
                            *[data[f"{wheel}_rpm"][index] * 2.0 * math.pi / 60.0
                              for wheel in WHEELS]), dtype=np.float64)
                        truth_pose = _pose_truth(data, index)
                        state_error = predicted[:7] - truth
                        truth_steering = -math.radians(0.5 * (
                            data["fl_steer_deg"][index]
                            + data["fr_steer_deg"][index]))
                        pose_error = predicted_pose - truth_pose
                        pose_error[2] = _wrap(pose_error[2])
                        horizon_rows[horizon].append({
                            "u": state_error[0], "v": state_error[1],
                            "yaw_rate": state_error[2],
                            "wheel_omega": state_error[3:7].tolist(),
                            "steering_rad": predicted[7] - truth_steering,
                            "position_m": float(np.linalg.norm(pose_error[:2])),
                            "heading_rad": float(pose_error[2]),
                            "start_speed_mps": float(data["speed_mps"][start]),
                        })
                horizon_summary = {}
                for horizon, rows in horizon_rows.items():
                    if not rows:
                        continue
                    u_err = np.asarray([r["u"] for r in rows])
                    v_err = np.asarray([r["v"] for r in rows])
                    yaw_err = np.asarray([r["yaw_rate"] for r in rows])
                    wheel_err = np.asarray([r["wheel_omega"] for r in rows])
                    steering_err = np.asarray([r["steering_rad"] for r in rows])
                    pos_err = np.asarray([r["position_m"] for r in rows])
                    heading_err = np.asarray([r["heading_rad"] for r in rows])
                    horizon_summary[f"{horizon * dt:g}s"] = {
                        "windows": len(rows),
                        "u_rmse_mps": float(np.sqrt(np.mean(u_err**2))),
                        "v_rmse_mps": float(np.sqrt(np.mean(v_err**2))),
                        "yaw_rate_rmse_radps": float(np.sqrt(np.mean(yaw_err**2))),
                        "steering_rmse_rad": float(
                            np.sqrt(np.mean(steering_err**2))),
                        "steering_abs_p95_rad": float(
                            np.quantile(np.abs(steering_err), 0.95)),
                        "per_wheel_omega_rmse_radps": np.sqrt(
                            np.mean(wheel_err**2, axis=0)).tolist(),
                        "position_radial_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
                        "position_radial_p95_m": float(np.quantile(pos_err, 0.95)),
                        "heading_rmse_rad": float(np.sqrt(np.mean(heading_err**2))),
                        "start_speed_p05_p50_p95_mps": np.quantile(
                            [r["start_speed_mps"] for r in rows],
                            (0.05, 0.5, 0.95)).tolist(),
                    }
                brake_results[slip_case] = {
                    "windows": windows,
                    "nonfinite_windows": nonfinite_windows,
                    "horizons": horizon_summary,
                }
            fit_results[brake_mode_name] = brake_results
        results[fit_name] = fit_results
    return {
        "capture": str(path.relative_to(ROOT)),
        "capture_samples": len(data["speed_mps"]),
        "capture_overwritten_samples": int(metadata["overwritten_samples"]),
        "coverage_speed_p05_p50_p95_max_mps": np.quantile(
            data["speed_mps"], (0.05, 0.5, 0.95)).tolist()
            + [float(np.max(data["speed_mps"]))],
        "window_seconds": window_seconds,
        "window_stride_seconds": stride_steps * dt,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--window-stride-seconds", type=float, default=5.0)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / (
        "live_runs/unity_exact_binary_audit_20261012/"
        "source_mechanics_command_rollout_20261012.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    stride_steps = round(args.window_stride_seconds / DT_S)
    if stride_steps < 1 or args.window_seconds <= 0.0:
        parser.error("window duration and stride must be positive")
    floors = (0.5,)
    report = {
        "schema_version": 1,
        "status": "command-only planar plant experiment; not a runtime odometry model",
        "model": {
            "rigid_body_mass_kg": RIGIDBODY_MASS_KG,
            "rigid_body_yaw_inertia_kgm2": RIGIDBODY_YAW_INERTIA_KGM2,
            "rigid_body_drag_per_s": RIGIDBODY_DRAG_PER_S,
            "wheelbase_ackermann_m": ACKERMANN_WHEELBASE_M,
            "axle_span_m": AXLE_SPAN_M,
            "wheel_x_relative_to_com_m": list(WHEEL_X_M),
            "wheel_y_relative_to_com_m": list(WHEEL_Y_M),
            "wheel_contact_radius_m": WHEEL_RADIUS_M,
            "wheel_damping_rate": WHEEL_DAMPING_RATE_NMS,
            "load_transfer_height_empirical_m": CAPTURED_COM_HEIGHT_ABOVE_ROAD_M,
            "tire_slip": {
                "legacy": "(R*omega-vx)/max(abs(vx),floor), vy/max(abs(vx),floor)",
                "wheelhit_reconstructed_candidate": "(R*omega-vx)/(max(abs(R*omega),abs(vx))+0.10054783962080052), atan2(vy,abs(vx)+1 m/s)",
                "candidate_basis": "r07-fit/r11-held-out WheelHit slip reconstruction; candidate uses state-derived contact speed and wheel speed only",
            },
            "tire_force": "serialized curve landmarks and unit-tangent baseline; compared with r07-only force-balance fit",
            "wheel_inertia": "serialized solid-cylinder value compared with r07-only spin fit",
            "brake_comparison": {
                "continuous_coulomb": "legacy model applies full opposing brake torque as a derivative and can reverse wheel speed within a 1 ms step",
                "unilateral_impulse_lock": "same source brake torque applied as an opposing angular impulse with a zero-speed stop constraint",
            },
            "integrator": "midpoint/RK2, 1 ms per internal step",
            "load_transfer_acceleration_state": "acceleration-valued state is carried algebraically from the current force balance; it is not integrated as jerk",
            "initialization": "captured body u/v/yaw rate, four RPMs, center steer, and prior-only smoothed acceleration",
            "future_inputs": "captured steering/throttle commands only",
            "future_simulator_truth_used": False,
        },
        "training_capture": str(args.train.relative_to(ROOT)),
        "held_out_capture": str(args.validation.relative_to(ROOT)),
        "training": _evaluate_capture(
            args.train, floors, args.window_seconds, stride_steps,
            args.sample_limit),
        "held_out": _evaluate_capture(
            args.validation, floors, args.window_seconds, stride_steps,
            args.sample_limit),
        "caveats": [
            "the model is planar and does not propagate roll, pitch, or suspension states",
            "the fitted force curves use privileged WheelHit/load labels during identification but not during rollout",
            "the r11 capture is an independent Unity-source trajectory, but its replayed public commands did not match the r02 request boundary",
            "a successful short-window score would not establish full-lap accuracy",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    for split in ("training", "held_out"):
        print(split, report[split]["capture"])
        for fit_name, fit_scores in report[split]["results"].items():
            for brake_mode, floor_scores in fit_scores.items():
                for floor, result in floor_scores.items():
                    print(f"  {fit_name}/{brake_mode}/{floor}: "
                          f"windows={result['windows']} "
                          f"nonfinite={result['nonfinite_windows']}")
                    for horizon, score in result["horizons"].items():
                        print(f"    {horizon}: pos="
                              f"{score['position_radial_rmse_m']:.4f}m "
                              f"u={score['u_rmse_mps']:.4f} "
                              f"v={score['v_rmse_mps']:.4f} "
                              f"r={score['yaw_rate_rmse_radps']:.4f} "
                              f"steer={score['steering_rmse_rad']:.5f}rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

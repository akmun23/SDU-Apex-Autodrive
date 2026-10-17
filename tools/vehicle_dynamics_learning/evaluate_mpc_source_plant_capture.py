#!/usr/bin/env python3
"""Validate the source-derived fast MPC plant on a 1 kHz Unity capture.

Ground truth initializes each evaluation window and is used only for scoring.
Inside every rollout, the plant receives only the recorded 40 Hz steering and
throttle commands (held on the captured 1 ms grid); future sensor/truth states
are never fed back. The evaluator reports both teacher-forced 25 ms steps and
recursive command-only windows so those two error modes cannot be conflated.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools.vehicle_dynamics_learning import (
    evaluate_physx_no_drive_plant_one_step as source_replay,
)
from tools.vehicle_dynamics_learning.physx_vehicle_source import (
    physx_jounce_from_wheel_pose_actor_body,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE = ROOT / (
    "live_runs/unity_mechanics_dynamic_validation_20261013_r12g/"
    "unity_source_capture/"
    "wheel_dynamics_unity_mechanics_dynamic_validation_r12g.csv")
DT_S = 0.001
CONTROL_DT_S = 0.025
HORIZONS_MS = (25, 100, 250, 500, 750, 1000)
MAX_STEERING_RAD = 0.5235987755982988
WHEEL_RADIUS_M = 0.059
WHEELBASE_M = 0.324
TRACK_M = 0.236
WHEELS = ("fl", "fr", "rl", "rr")


class PlantInput(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_double * 18),
        ("steering_target_rad", ctypes.c_double),
        ("throttle_normalized", ctypes.c_double),
        ("suspension_history_initialized", ctypes.c_int),
    ]


class PlantOutput(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_double * 18),
        ("suspension_history_initialized", ctypes.c_int),
        ("longitudinal_slip", ctypes.c_double * 4),
        ("lateral_slip", ctypes.c_double * 4),
        ("normal_load_n", ctypes.c_double * 4),
        ("suspension_force_n", ctypes.c_double * 4),
        ("suspension_jounce_rate_mps", ctypes.c_double * 4),
        ("tire_longitudinal_force_n", ctypes.c_double * 4),
        ("tire_lateral_force_n", ctypes.c_double * 4),
        ("longitudinal_acceleration_mps2", ctypes.c_double),
        ("lateral_acceleration_mps2", ctypes.c_double),
        ("yaw_acceleration_radps2", ctypes.c_double),
    ]


def _library(path: Path):
    lib = ctypes.CDLL(str(path.resolve()))
    lib.unity_source_plant_step_c.argtypes = [
        ctypes.POINTER(PlantInput), ctypes.c_double,
        ctypes.POINTER(PlantOutput)]
    lib.unity_source_plant_step_c.restype = ctypes.c_int
    return lib


def _rotation_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


# Unity's right/up/forward frame maps to forward/left/up with one reflection.
_UNITY_FROM_MODEL = np.asarray([[0.0, -1.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0]])


def _model_euler_and_yaw(row) -> tuple[float, float, float]:
    q = np.asarray([row.rot_world_x, row.rot_world_y,
                    row.rot_world_z, row.rot_world_w], dtype=float)
    unity_rotation = _rotation_matrix_xyzw(q)
    model_rotation = _UNITY_FROM_MODEL.T @ unity_rotation @ _UNITY_FROM_MODEL
    roll = math.atan2(model_rotation[2, 1], model_rotation[2, 2])
    pitch = math.asin(float(np.clip(-model_rotation[2, 0], -1.0, 1.0)))
    yaw = math.atan2(model_rotation[1, 0], model_rotation[0, 0])
    return roll, pitch, yaw


def _center_steering(row) -> float:
    # Unity reports the WheelCollider front angles in the opposite lateral
    # sign to this plant. Invert the source Ackermann equations using both
    # wheels, then average to suppress float rounding at the max lock.
    fl = -math.radians(float(row.fl_steer_deg))
    fr = -math.radians(float(row.fr_steer_deg))
    tan_fl, tan_fr = math.tan(fl), math.tan(fr)
    denom_fl = 2.0 * WHEELBASE_M + TRACK_M * tan_fl
    denom_fr = 2.0 * WHEELBASE_M - TRACK_M * tan_fr
    values = []
    if abs(denom_fl) > 1.0e-9:
        values.append(2.0 * WHEELBASE_M * tan_fl / denom_fl)
    if abs(denom_fr) > 1.0e-9:
        values.append(2.0 * WHEELBASE_M * tan_fr / denom_fr)
    if not values:
        return 0.0
    return math.atan(float(np.mean(values)))


def _truth_state(row, metadata: dict) -> np.ndarray:
    roll, pitch, _ = _model_euler_and_yaw(row)
    rotation = _rotation_matrix_xyzw(np.asarray([
        row.rot_world_x, row.rot_world_y, row.rot_world_z, row.rot_world_w],
        dtype=float))
    body_velocity_world_y = float((rotation @ np.asarray([
        row.vel_body_x_mps, row.vel_body_y_mps, row.vel_body_z_mps],
        dtype=float))[1])
    # WheelCollider.GetWorldPose in the 1 kHz capture exposes the prior-query
    # wheel pose. Reconstruct its cached jounce so the next PhysX damper step
    # starts from the actual previous jounce instead of an artificial zero.
    state = np.zeros(19, dtype=np.float64)
    state[0] = float(row.vel_body_z_mps)
    state[1] = -float(row.vel_body_x_mps)
    state[2] = -float(row.omega_body_y_rps)
    for index, wheel in enumerate(WHEELS):
        state[3 + index] = float(getattr(row, f"{wheel}_rpm")) * (2.0 * math.pi / 60.0)
    state[7] = float(row.pos_world_y_m)
    # The plant's heave state is world-vertical, while Unity logs velocity in
    # the body frame. Directly copying body Y injects a large false heave rate
    # when the chassis is pitched/rolled and moving quickly.
    state[8] = body_velocity_world_y
    state[9] = roll
    state[10] = -float(row.omega_body_z_rps)
    state[11] = pitch
    state[12] = float(row.omega_body_x_rps)
    state[13] = _center_steering(row)
    for index, wheel in enumerate(WHEELS):
        configuration = source_replay._wheel_configuration(
            metadata, index, math.radians(float(getattr(row, f"{wheel}_steer_deg"))),
            0.0, 0.0)
        wheel_position = tuple(
            float(getattr(row, f"{wheel}_wheel_pose_body_{axis}"))
            for axis in "xyz")
        state[14 + index] = physx_jounce_from_wheel_pose_actor_body(
            wheel_position,
            source_replay.RIGIDBODY_CENTER_OF_MASS_BODY_M,
            configuration.wheel_center_offset_com_body_m,
            configuration.suspension_direction_body)
    state[18] = 1.0
    return state


def _truth_position(row) -> np.ndarray:
    return np.asarray([row.pos_world_z_m, -row.pos_world_x_m], dtype=float)


def _advance(lib, state: np.ndarray, steering: float, throttle: float,
             with_aux: bool = False):
    inp = PlantInput()
    for i, value in enumerate(state[:18]):
        inp.state[i] = float(value)
    inp.steering_target_rad = float(steering) * MAX_STEERING_RAD
    inp.throttle_normalized = float(throttle)
    inp.suspension_history_initialized = int(round(state[18]))
    out = PlantOutput()
    if not lib.unity_source_plant_step_c(ctypes.byref(inp), DT_S,
                                         ctypes.byref(out)):
        raise RuntimeError("source plant rejected a finite capture sample")
    next_state = np.zeros(19, dtype=np.float64)
    next_state[:18] = np.ctypeslib.as_array(out.state)
    next_state[18] = float(out.suspension_history_initialized)
    if not with_aux:
        return next_state
    aux = {
        "longitudinal_slip": np.ctypeslib.as_array(out.longitudinal_slip).copy(),
        "lateral_slip": np.ctypeslib.as_array(out.lateral_slip).copy(),
        "normal_load_n": np.ctypeslib.as_array(out.normal_load_n).copy(),
        "suspension_force_n": np.ctypeslib.as_array(
            out.suspension_force_n).copy(),
        "suspension_jounce_rate_mps": np.ctypeslib.as_array(
            out.suspension_jounce_rate_mps).copy(),
        "tire_longitudinal_force_n": np.ctypeslib.as_array(
            out.tire_longitudinal_force_n).copy(),
        "tire_lateral_force_n": np.ctypeslib.as_array(
            out.tire_lateral_force_n).copy(),
        "ax": float(out.longitudinal_acceleration_mps2),
        "ay": float(out.lateral_acceleration_mps2),
        "yaw_accel": float(out.yaw_acceleration_radps2),
    }
    return next_state, aux


def _integrate_pose(position: np.ndarray, yaw: float,
                    old: np.ndarray, new: np.ndarray) -> tuple[np.ndarray, float]:
    next_yaw = yaw + 0.5 * (old[2] + new[2]) * DT_S
    u = 0.5 * (old[0] + new[0])
    v = 0.5 * (old[1] + new[1])
    mid_yaw = 0.5 * (yaw + next_yaw)
    position = position + DT_S * np.asarray([
        u * math.cos(mid_yaw) - v * math.sin(mid_yaw),
        u * math.sin(mid_yaw) + v * math.cos(mid_yaw)])
    return position, next_yaw


def _error_accumulator() -> dict[str, list[float]]:
    return {name: [] for name in (
        "u_mps", "v_mps", "yaw_rate_radps", "yaw_rad",
        "body_height_m", "roll_rad", "roll_rate_radps",
        "pitch_rad", "pitch_rate_radps",
        "position_m", "wheel_fl_radps", "wheel_fr_radps",
        "wheel_rl_radps", "wheel_rr_radps")}


def _record_errors(acc: dict[str, list[float]], predicted: np.ndarray,
                   predicted_position: np.ndarray, predicted_yaw: float,
                   row) -> None:
    gt_roll, gt_pitch, gt_yaw = _model_euler_and_yaw(row)
    acc["u_mps"].append(float(predicted[0] - row.vel_body_z_mps))
    acc["v_mps"].append(float(predicted[1] + row.vel_body_x_mps))
    acc["yaw_rate_radps"].append(
        float(predicted[2] + row.omega_body_y_rps))
    acc["body_height_m"].append(float(predicted[7] - row.pos_world_y_m))
    acc["roll_rad"].append(float(predicted[9] - gt_roll))
    acc["roll_rate_radps"].append(
        float(predicted[10] + row.omega_body_z_rps))
    acc["pitch_rad"].append(float(predicted[11] - gt_pitch))
    acc["pitch_rate_radps"].append(
        float(predicted[12] - row.omega_body_x_rps))
    acc["yaw_rad"].append(math.atan2(math.sin(predicted_yaw - gt_yaw),
                                     math.cos(predicted_yaw - gt_yaw)))
    acc["position_m"].append(float(np.linalg.norm(
        predicted_position - _truth_position(row))))
    for index, wheel in enumerate(WHEELS):
        truth = float(getattr(row, f"{wheel}_rpm")) * (2.0 * math.pi / 60.0)
        acc[f"wheel_{wheel}_radps"].append(float(predicted[3 + index] - truth))


def _metrics(acc: dict[str, list[float]]) -> dict:
    result = {}
    for key, values in acc.items():
        error = np.asarray(values, dtype=float)
        if not len(error):
            result[key] = {"n": 0}
            continue
        absolute = np.abs(error)
        result[key] = {
            "n": int(len(error)),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "p95_abs": float(np.quantile(absolute, 0.95)),
            "max_abs": float(np.max(absolute)),
        }
    return result


def _segments(data: pd.DataFrame) -> list[tuple[int, int]]:
    x = data.pos_world_x_m.to_numpy(dtype=float)
    z = data.pos_world_z_m.to_numpy(dtype=float)
    t = data.sim_time_s.to_numpy(dtype=float)
    dt = np.diff(t)
    jumps = np.hypot(np.diff(x), np.diff(z)) > 0.10
    broken_time = (dt < 0.0007) | (dt > 0.0013)
    cuts = np.flatnonzero(jumps | broken_time) + 1
    edges = np.concatenate(([0], cuts, [len(data)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])
            if b - a > 2]


def _capture_coverage(data: pd.DataFrame) -> dict:
    velocity = data[["vel_body_x_mps", "vel_body_y_mps",
                     "vel_body_z_mps"]].to_numpy(dtype=float)
    speed = np.linalg.norm(velocity, axis=1)
    fl = -np.deg2rad(data.fl_steer_deg.to_numpy(dtype=float))
    fr = -np.deg2rad(data.fr_steer_deg.to_numpy(dtype=float))
    tan_fl, tan_fr = np.tan(fl), np.tan(fr)
    denom_fl = 2.0 * WHEELBASE_M + TRACK_M * tan_fl
    denom_fr = 2.0 * WHEELBASE_M - TRACK_M * tan_fr
    center_tan = 0.5 * (
        2.0 * WHEELBASE_M * tan_fl / denom_fl +
        2.0 * WHEELBASE_M * tan_fr / denom_fr)
    steering = np.arctan(center_tan)
    throttle = data.throttle_command.to_numpy(dtype=float)

    def summary(values: np.ndarray) -> dict:
        values = values[np.isfinite(values)]
        if not len(values):
            return {"n": 0}
        return {
            "n": int(len(values)),
            "min": float(np.min(values)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }

    return {
        "speed_mps": summary(speed),
        "abs_physical_steering_rad": summary(np.abs(steering)),
        "throttle_normalized": summary(throttle),
        "moving_fraction_speed_over_1mps": float(np.mean(speed > 1.0)),
    }


def evaluate(capture: Path, library: Path) -> dict:
    columns = ["sim_time_s", "pos_world_x_m", "pos_world_y_m", "pos_world_z_m",
               "rot_world_x", "rot_world_y", "rot_world_z", "rot_world_w",
               "vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
               "omega_body_x_rps", "omega_body_y_rps", "omega_body_z_rps",
               "throttle_command", "steering_command"]
    for wheel in WHEELS:
        columns.extend([f"{wheel}_rpm", f"{wheel}_steer_deg",
                        f"{wheel}_forward_slip", f"{wheel}_sideways_slip"])
        columns.extend([f"{wheel}_wheel_pose_body_{axis}" for axis in "xyz"])
    data = pd.read_csv(capture, comment="#", usecols=columns)
    if len(data) < 1002:
        raise ValueError("capture is too short for recursive horizons")
    metadata = source_replay._metadata(capture)
    lib = _library(library)

    # One held-command, 25 ms step, initialized from the measured state.
    one_step = _error_accumulator()
    one_ms = _error_accumulator()
    slip_errors = {f"{wheel}_{axis}": []
                   for wheel in WHEELS for axis in ("longitudinal", "lateral")}
    acceleration_errors = {name: [] for name in ("ax_mps2", "ay_mps2",
                                                   "yaw_accel_radps2")}
    stable_intervals = 0
    for start, end in _segments(data):
        for row_index in range(start, end - 25, 25):
            command_window = data.iloc[row_index:row_index + 25]
            if (np.ptp(command_window.steering_command.to_numpy(dtype=float)) > 1.0e-9
                    or np.ptp(command_window.throttle_command.to_numpy(dtype=float)) > 1.0e-9):
                continue
            state = _truth_state(data.iloc[row_index], metadata)
            position = _truth_position(data.iloc[row_index])
            _, _, yaw = _model_euler_and_yaw(data.iloc[row_index])
            steering = float(data.steering_command.iloc[row_index])
            throttle = float(data.throttle_command.iloc[row_index])
            for j in range(row_index, row_index + 25):
                if j == row_index:
                    next_state, aux = _advance(
                        lib, state, steering, throttle, with_aux=True)
                    measured_next = data.iloc[j + 1]
                    next_pos, next_yaw = _integrate_pose(
                        position.copy(), yaw, state, next_state)
                    _record_errors(one_ms, next_state, next_pos, next_yaw,
                                   measured_next)
                    acceleration_errors["ax_mps2"].append(
                        aux["ax"] - (measured_next.vel_body_z_mps
                                     - data.vel_body_z_mps.iloc[j]) / DT_S)
                    acceleration_errors["ay_mps2"].append(
                        aux["ay"] + (measured_next.vel_body_x_mps
                                     - data.vel_body_x_mps.iloc[j]) / DT_S)
                    acceleration_errors["yaw_accel_radps2"].append(
                        aux["yaw_accel"] +
                        (measured_next.omega_body_y_rps
                         - data.omega_body_y_rps.iloc[j]) / DT_S)
                    for index, wheel in enumerate(WHEELS):
                        slip_errors[f"{wheel}_longitudinal"].append(
                            float(aux["longitudinal_slip"][index]
                                  - getattr(data.iloc[j],
                                            f"{wheel}_forward_slip")))
                        slip_errors[f"{wheel}_lateral"].append(
                            # The candidate's body frame is left-positive;
                            # Unity WheelHit.sidewaysSlip uses right-positive.
                            float(-aux["lateral_slip"][index]
                                  - getattr(data.iloc[j],
                                            f"{wheel}_sideways_slip")))
                else:
                    next_state = _advance(lib, state, steering, throttle)
                position, yaw = _integrate_pose(position, yaw, state, next_state)
                state = next_state
            _record_errors(one_step, state, position, yaw,
                           data.iloc[row_index + 25])
            stable_intervals += 1

    # Recursive command-only windows, each initialized once from truth. The
    # window starts are spaced at 1 s; metrics are aggregated by horizon.
    recursive: dict[str, dict[str, list[float]]] = {
        f"{horizon}ms": _error_accumulator() for horizon in HORIZONS_MS}
    speed_edges = (0.0, 3.0, 6.0, 9.0, 12.0, math.inf)
    steering_edges = (0.0, 0.1, 0.3, math.inf)
    by_speed = {
        f"{low:g}-{high:g}mps" if math.isfinite(high) else f">={low:g}mps":
            {f"{h}ms": _error_accumulator() for h in HORIZONS_MS}
        for low, high in zip(speed_edges[:-1], speed_edges[1:])}
    by_steering = {
        f"{low:g}-{high:g}rad" if math.isfinite(high) else f">={low:g}rad":
            {f"{h}ms": _error_accumulator() for h in HORIZONS_MS}
        for low, high in zip(steering_edges[:-1], steering_edges[1:])}
    window_count = 0
    max_steps = max(HORIZONS_MS)
    one_second_position_windows: list[dict] = []
    for start, end in _segments(data):
        anchors = range(start, end - max_steps - 1, max_steps)
        for anchor in anchors:
            row = data.iloc[anchor]
            state = _truth_state(row, metadata)
            position = _truth_position(row)
            _, _, yaw = _model_euler_and_yaw(row)
            speed = math.sqrt(
                float(row.vel_body_x_mps) ** 2 +
                float(row.vel_body_y_mps) ** 2 +
                float(row.vel_body_z_mps) ** 2)
            abs_steering = abs(state[13])
            speed_key = next(
                key for (low, high), key in zip(
                    zip(speed_edges[:-1], speed_edges[1:]), by_speed)
                if low <= speed < high)
            steering_key = next(
                key for (low, high), key in zip(
                    zip(steering_edges[:-1], steering_edges[1:]), by_steering)
                if low <= abs_steering < high)
            wanted = {horizon: horizon for horizon in HORIZONS_MS}
            for offset in range(1, max_steps + 1):
                row_index = anchor + offset - 1
                old_state = state
                state = _advance(
                    lib, old_state,
                    float(data.steering_command.iloc[row_index]),
                    float(data.throttle_command.iloc[row_index]))
                position, yaw = _integrate_pose(position, yaw, old_state, state)
                if offset in wanted.values():
                    key = f"{offset}ms"
                    _record_errors(recursive[key], state, position, yaw,
                                   data.iloc[anchor + offset])
                    _record_errors(by_speed[speed_key][key], state, position,
                                   yaw, data.iloc[anchor + offset])
                    _record_errors(by_steering[steering_key][key], state,
                                   position, yaw, data.iloc[anchor + offset])
                if offset == max_steps:
                    truth = data.iloc[anchor + offset]
                    _, _, truth_yaw = _model_euler_and_yaw(truth)
                    one_second_position_windows.append({
                        "start_sim_time_s": float(row.sim_time_s),
                        "start_speed_mps": speed,
                        "start_abs_physical_steering_rad": abs_steering,
                        "start_throttle": float(row.throttle_command),
                        "start_steering_command": float(row.steering_command),
                        "position_error_m": float(np.linalg.norm(
                            position - _truth_position(truth))),
                        "yaw_error_rad": float(math.atan2(
                            math.sin(yaw - truth_yaw),
                            math.cos(yaw - truth_yaw))),
                        "u_error_mps": float(
                            state[0] - truth.vel_body_z_mps),
                        "v_error_mps": float(
                            state[1] + truth.vel_body_x_mps),
                        "yaw_rate_error_radps": float(
                            state[2] + truth.omega_body_y_rps),
                    })
            window_count += 1

    return {
        "capture": str(capture),
        "model": "source-derived candidate; not the native Unity/PhysX solver",
        "capture_rows": int(len(data)),
        "continuous_segments": [b - a for a, b in _segments(data)],
        "input_coverage": _capture_coverage(data),
        "one_step_25ms": {
            "held_command_intervals": stable_intervals,
            "errors": _metrics(one_step),
        },
        "one_step_1ms_at_interval_start": {
            "errors": _metrics(one_ms),
            "source_acceleration_error": _metrics(acceleration_errors),
            "captured_slip_error": _metrics(slip_errors),
        },
        "recursive_command_only": {
            "ground_truth_initialized_windows": window_count,
            "window_s": max_steps * DT_S,
            "errors_by_horizon": {
                key: _metrics(value) for key, value in recursive.items()},
            "errors_by_start_speed": {
                regime: {key: _metrics(value) for key, value in horizons.items()}
                for regime, horizons in by_speed.items()},
            "errors_by_start_abs_steering": {
                regime: {key: _metrics(value) for key, value in horizons.items()}
                for regime, horizons in by_steering.items()},
            "largest_1000ms_position_errors": sorted(
                one_second_position_windows,
                key=lambda item: item["position_error_m"], reverse=True)[:10],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--library", type=Path, required=True,
                        help="Shared library exporting unity_source_plant_step_c")
    parser.add_argument("--output", type=Path,
                        help="Optional path for a persistent JSON report")
    args = parser.parse_args()
    report = evaluate(args.capture, args.library)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

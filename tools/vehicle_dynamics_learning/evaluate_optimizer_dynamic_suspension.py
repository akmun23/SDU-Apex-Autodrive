#!/usr/bin/env python3
"""Test a Unity-parameterized heave/roll/pitch load model on frozen captures.

This isolates the vertical/suspension subsystem. It recursively predicts
body heave, roll, pitch, and four wheel loads at the capture's 1 kHz interval,
while measured in-plane acceleration is supplied as an external diagnostic
forcing. No future suspension pose or WheelHit load is fed into the rollout.
Training-calibrated spring target positions are frozen from r07 and scored on
the independent r11 capture.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from tools.vehicle_dynamics_learning.analyze_unity_suspension_load import (
    WHEEL_TRANSFORM,
    _metadata,
)
from tools.vehicle_dynamics_learning.compare_optimizer_load_transfer import (
    ROOT,
    TRAIN,
    VALIDATION,
    WHEELS,
    _rotate_rows,
    _score,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    CAPTURED_COM_HEIGHT_ABOVE_ROAD_M,
    GRAVITY_MPS2,
    RIGIDBODY_DRAG_PER_S,
    WHEELBASE_ACKERMANN_M,
)


CALIBRATION = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "suspension_load_geometry_calibration_v2_20261012.json")
OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "optimizer_dynamic_suspension_rollout_20261015.json")
ACCELERATION_WINDOW = 21
SCORE_WINDOWS = (11, 21, 41)
COM_BODY_UNITY = np.asarray((-0.00008, 0.06434, -0.00468))


def _capture_arrays(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = _metadata(path)
    columns = [
        "pos_world_y_m", "rot_world_x", "rot_world_y", "rot_world_z",
        "rot_world_w", "vel_body_x_mps", "vel_body_y_mps",
        "vel_body_z_mps", "omega_body_x_rps", "omega_body_y_rps",
        "omega_body_z_rps",
    ]
    for wheel in WHEELS:
        columns += [f"{wheel}_grounded", f"{wheel}_contact_force_n"]
    frame = pd.read_csv(path, comment="#", usecols=columns)
    quaternion = frame[["rot_world_x", "rot_world_y", "rot_world_z",
                        "rot_world_w"]].to_numpy(dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=1)[:, None]
    velocity_body = frame[["vel_body_x_mps", "vel_body_y_mps",
                           "vel_body_z_mps"]].to_numpy(dtype=np.float64)
    velocity_world = _rotate_rows(quaternion, velocity_body)
    dt = float(metadata["fixed_delta_time_s"])
    acceleration_world = np.column_stack([
        savgol_filter(velocity_world[:, axis], ACCELERATION_WINDOW, 3,
                      deriv=1, delta=dt, mode="interp")
        for axis in range(3)])
    inverse = quaternion.copy()
    inverse[:, :3] *= -1.0
    acceleration_body = _rotate_rows(inverse, acceleration_world)
    # Map Unity body axes (right, up, forward) to the optimizer's
    # (forward, left, up) planar convention.
    accel_forward = acceleration_body[:, 2]
    accel_left = -acceleration_body[:, 0]
    rotation = Rotation.from_quat(quaternion).as_matrix()
    # Unity Euler order is Y-X-Z: yaw, then pitch, then roll. The pitch and
    # roll expressions below isolate the tilt from the rotation-matrix row
    # that is invariant to yaw.
    roll = np.arctan2(rotation[:, 1, 0], rotation[:, 1, 1])
    pitch = np.arcsin(np.clip(-rotation[:, 1, 2], -1.0, 1.0))
    arrays: dict[str, np.ndarray] = {
        "position_y": frame["pos_world_y_m"].to_numpy(dtype=np.float64),
        "quaternion": quaternion,
        "velocity_body": velocity_body,
        "velocity_world": velocity_world,
        "omega_body": frame[["omega_body_x_rps", "omega_body_y_rps",
                              "omega_body_z_rps"]].to_numpy(dtype=np.float64),
        "accel_forward": accel_forward,
        "accel_left": accel_left,
        "roll": roll,
        "pitch": pitch,
        "grounded": np.column_stack([
            frame[f"{wheel}_grounded"].to_numpy(dtype=np.float64) > 0.5
            for wheel in WHEELS]),
        "actual_load": np.column_stack([
            frame[f"{wheel}_contact_force_n"].to_numpy(dtype=np.float64)
            for wheel in WHEELS]),
    }
    del frame
    return metadata, arrays


def _geometry_constants(metadata: dict[str, str], calibration: dict[str, Any]
                        ) -> dict[str, Any]:
    wheel_data = {}
    for wheel in WHEELS:
        prefix = {
            "fl": "front_left", "fr": "front_right",
            "rl": "rear_left", "rr": "rear_right",
        }[wheel]
        position, quaternion = WHEEL_TRANSFORM[wheel]
        q = np.asarray(quaternion, dtype=np.float64)
        local_rotation = Rotation.from_quat(q)
        center = local_rotation.apply((0.0, 0.025, 0.0))
        down = local_rotation.apply((0.0, -1.0, 0.0))
        base = np.asarray(position, dtype=np.float64) + center
        camber = math.atan2(float(down[0]), float(-down[1]))
        wheel_data[wheel] = {
            "base_xyz_m": base,
            "camber_rad": camber,
            "radius_m": float(metadata[f"{prefix}_radius_m"]),
            "sprung_mass_kg": float(metadata[f"{prefix}_sprung_mass_kg"]),
            "spring_n_per_m": float(metadata[f"{prefix}_spring_n_per_m"]),
            "damper_ns_per_m": float(metadata[f"{prefix}_damper_ns_per_m"]),
            "target_travel_m": float(calibration[wheel]),
            "x_right_from_com_m": float(-{
                "fl": 0.118, "fr": -0.118,
                "rl": 0.118, "rr": -0.118,
            }[wheel]),
            "z_forward_from_com_m": float({
                "fl": 0.17468, "fr": 0.17468,
                "rl": -0.15532, "rr": -0.15532,
            }[wheel]),
        }
    inertia_diagonal = np.asarray([
        float(value) for value in metadata["inertia_tensor_body"].split(",")])
    inertia_rotation = np.asarray([
        float(value) for value in metadata[
            "inertia_tensor_rotation_body"].split(",")])
    inertia_body = (Rotation.from_quat(inertia_rotation).as_matrix()
                    @ np.diag(inertia_diagonal)
                    @ Rotation.from_quat(inertia_rotation).as_matrix().T)
    return {
        "wheel": wheel_data,
        "mass_kg": float(metadata["mass_kg"]),
        "inertia_body_kgm2": inertia_body,
        "inertia_pitch_kgm2": float(inertia_body[0, 0]),
        "inertia_roll_kgm2": float(inertia_body[2, 2]),
        "com_height_m": CAPTURED_COM_HEIGHT_ABOVE_ROAD_M,
        "ground_y_m": 0.0,
        "linear_drag_per_s": RIGIDBODY_DRAG_PER_S,
        "angular_drag_per_s": float(metadata.get("angular_drag_per_s", "0.1")),
    }


def _travel_and_rate(state: np.ndarray, state_rate: np.ndarray,
                     wheel: dict[str, Any], ground_y: float
                     ) -> tuple[float, float]:
    root_y, root_rate, roll, roll_rate, pitch, pitch_rate = state
    base_x, base_y, base_z = wheel["base_xyz_m"]
    camber = wheel["camber_rad"]
    sin_roll, cos_roll = math.sin(roll), math.cos(roll)
    sin_pitch, cos_pitch = math.sin(pitch), math.cos(pitch)
    tilted_base_y = (cos_pitch * (sin_roll * base_x + cos_roll * base_y)
                     - sin_pitch * base_z)
    denominator = cos_pitch * math.cos(roll + camber)
    numerator = root_y + tilted_base_y - ground_y - wheel["radius_m"]
    travel = numerator / denominator
    base_roll = (cos_pitch * (cos_roll * base_x - sin_roll * base_y))
    base_pitch = (-sin_pitch * (sin_roll * base_x + cos_roll * base_y)
                  - cos_pitch * base_z)
    denominator_roll = -cos_pitch * math.sin(roll + camber)
    denominator_pitch = -sin_pitch * math.cos(roll + camber)
    travel_roll = ((base_roll * denominator
                   - numerator * denominator_roll) / denominator**2)
    travel_pitch = ((base_pitch * denominator
                    - numerator * denominator_pitch) / denominator**2)
    travel_root = 1.0 / denominator
    travel_rate = (travel_root * root_rate + travel_roll * roll_rate
                   + travel_pitch * pitch_rate)
    return travel, travel_rate


def _loads_and_acceleration(state: np.ndarray, accel_forward: float,
                            accel_left: float, velocity_body: np.ndarray,
                            model: dict[str, Any]
                            ) -> tuple[np.ndarray, np.ndarray]:
    loads = []
    roll_moment = 0.0
    pitch_moment = 0.0
    for wheel_name in WHEELS:
        wheel = model["wheel"][wheel_name]
        travel, travel_rate = _travel_and_rate(
            state, state[1::2], wheel, model["ground_y_m"])
        compression = wheel["target_travel_m"] - travel
        compression_rate = -travel_rate
        load = (wheel["sprung_mass_kg"] * GRAVITY_MPS2
                + wheel["spring_n_per_m"] * compression
                + wheel["damper_ns_per_m"] * compression_rate)
        load = max(0.0, load)
        loads.append(load)
        roll_moment += wheel["x_right_from_com_m"] * load
        pitch_moment -= wheel["z_forward_from_com_m"] * load

    loads_array = np.asarray(loads, dtype=np.float64)
    tire_forward_force = model["mass_kg"] * (
        accel_forward + model["linear_drag_per_s"] * velocity_body[2])
    lateral_velocity = -velocity_body[0]
    tire_left_force = model["mass_kg"] * (
        accel_left + model["linear_drag_per_s"] * lateral_velocity)
    # VehicleController's WheelCollider force application distance is zero,
    # so the wheel-normal and tire forces act at the ground-level wheel base.
    roll_moment -= model["com_height_m"] * tire_left_force
    pitch_moment -= model["com_height_m"] * tire_forward_force
    derivative = np.asarray((
        state[1],
        (float(np.sum(loads_array)) / model["mass_kg"] - GRAVITY_MPS2
         - model["linear_drag_per_s"] * state[1]),
        state[3],
        roll_moment / model["inertia_roll_kgm2"]
        - model["angular_drag_per_s"] * state[3],
        state[5],
        pitch_moment / model["inertia_pitch_kgm2"]
        - model["angular_drag_per_s"] * state[5],
    ), dtype=np.float64)
    return loads_array, derivative


def _predict_capture(path: Path, calibrated_targets: dict[str, Any]
                     ) -> dict[str, Any]:
    metadata, arrays = _capture_arrays(path)
    model = _geometry_constants(metadata, calibrated_targets)
    n = len(arrays["position_y"])
    dt = float(metadata["fixed_delta_time_s"])
    initial_yaw_rate_unity = arrays["omega_body"][0, 1]
    initial_state = np.asarray((
        arrays["position_y"][0],
        arrays["velocity_world"][0, 1],
        arrays["roll"][0],
        arrays["omega_body"][0, 2],
        arrays["pitch"][0],
        arrays["omega_body"][0, 0],
    ), dtype=np.float64)
    # Keep the unused measured yaw component explicit in the report: this
    # reduced-order test intentionally omits yaw-roll inertia coupling.
    del initial_yaw_rate_unity
    states = np.empty((n, 6), dtype=np.float64)
    predicted_loads = np.empty((n, 4), dtype=np.float64)
    states[0] = initial_state
    for index in range(n):
        state = states[index]
        loads, k1 = _loads_and_acceleration(
            state, arrays["accel_forward"][index],
            arrays["accel_left"][index], arrays["velocity_body"][index],
            model)
        predicted_loads[index] = loads
        if index + 1 >= n:
            break
        _, k2 = _loads_and_acceleration(
            state + 0.5 * dt * k1, arrays["accel_forward"][index],
            arrays["accel_left"][index], arrays["velocity_body"][index],
            model)
        _, k3 = _loads_and_acceleration(
            state + 0.5 * dt * k2, arrays["accel_forward"][index],
            arrays["accel_left"][index], arrays["velocity_body"][index],
            model)
        _, k4 = _loads_and_acceleration(
            state + dt * k3, arrays["accel_forward"][index],
            arrays["accel_left"][index], arrays["velocity_body"][index],
            model)
        states[index + 1] = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if not np.isfinite(states[index + 1]).all():
            raise FloatingPointError(
                f"non-finite suspension state at {index * dt:.3f} s")

    result: dict[str, Any] = {
        "capture": str(path.relative_to(ROOT)),
        "run_id": metadata.get("run_id", path.stem),
        "duration_s": (n - 1) * dt,
        "samples": n,
        "prediction_inputs": "measured in-plane acceleration from filtered 1 kHz body velocity; no measured heave/roll/pitch or WheelHit loads after initialization",
        "initialize_from_truth": True,
        "scores": {},
        "state_error": {},
        "diagnostics": {
            "predicted_root_height_range_m": [
                float(np.min(states[:, 0])), float(np.max(states[:, 0]))],
            "predicted_roll_range_deg": np.rad2deg(
                [float(np.min(states[:, 2])), float(np.max(states[:, 2]))]).tolist(),
            "predicted_pitch_range_deg": np.rad2deg(
                [float(np.min(states[:, 4])), float(np.max(states[:, 4]))]).tolist(),
            "negative_load_clamped_count": int(np.count_nonzero(
                predicted_loads <= 0.0)),
        },
    }
    for window in SCORE_WINDOWS:
        actual_smoothed = np.column_stack([
            savgol_filter(arrays["actual_load"][:, wheel], window, 3,
                          mode="interp")
            for wheel in range(4)])
        predicted_smoothed = np.column_stack([
            savgol_filter(predicted_loads[:, wheel], window, 3,
                          mode="interp")
            for wheel in range(4)])
        valid = arrays["grounded"].copy()
        valid[:window] = False
        valid[-window:] = False
        result["scores"][str(window)] = {
            "load_pooled": _score(
                (predicted_smoothed - actual_smoothed)[valid]),
            "load_per_wheel": {
                WHEELS[i]: _score(
                    (predicted_smoothed - actual_smoothed)[valid[:, i], i])
                for i in range(4)},
        }
        roll_error = states[:, 2] - arrays["roll"]
        pitch_error = states[:, 4] - arrays["pitch"]
        valid_rows = np.all(arrays["grounded"], axis=1)
        result["state_error"][str(window)] = {
            "roll_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(
                roll_error[valid_rows]**2)))),
            "roll_p95_abs_deg": float(np.rad2deg(np.quantile(
                np.abs(roll_error[valid_rows]), 0.95))),
            "pitch_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(
                pitch_error[valid_rows]**2)))),
            "pitch_p95_abs_deg": float(np.rad2deg(np.quantile(
                np.abs(pitch_error[valid_rows]), 0.95))),
            "heave_rmse_m": float(np.sqrt(np.mean(
                (states[valid_rows, 0] - arrays["position_y"][valid_rows])**2))),
            "heave_p95_abs_m": float(np.quantile(np.abs(
                states[valid_rows, 0] - arrays["position_y"][valid_rows]), 0.95)),
        }
    del arrays
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=TRAIN)
    parser.add_argument("--validation", type=Path, default=VALIDATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not CALIBRATION.is_file():
        raise FileNotFoundError(CALIBRATION)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    calibration_report = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    calibrated_targets = {
        wheel: calibration_report["geometry_only_load_model"][wheel][
            "training_calibrated_target_travel_m"]
        for wheel in WHEELS
    }
    report = {
        "schema_version": 1,
        "status": "candidate diagnostic; no OCP/runtime integration",
        "source_parameters": {
            "spring_n_per_m": 500.0,
            "damper_ns_per_m": 100.0,
            "wheel_force_application_point_distance_m": 0.0,
            "body_height_and_inertia": "Unity capture metadata, r07 parameters frozen",
            "spring_target_travel_m": calibrated_targets,
            "target_calibration_capture": str(TRAIN.relative_to(ROOT)),
        },
        "model": "one hertz-integrated heave/roll/pitch body state with four source WheelCollider spring/damper loads; vertical force balance plus inertial lateral/longitudinal load moments; angular and linear drag included",
        "sample_interval_s": 0.001,
        "acceleration_filter_window_ms": ACCELERATION_WINDOW,
        "captures": {
            "training": _predict_capture(args.training, calibrated_targets),
            "independent_validation": _predict_capture(
                args.validation, calibrated_targets),
        },
        "limitations": [
            "Measured in-plane acceleration is supplied as a forcing input, so this is a subsystem test, not a command-only full-plant rollout.",
            "The reduced model uses uncoupled roll/pitch inertias and omits the Unity inertia tensor's yaw-roll coupling.",
            "The final optimizer model must use its predicted tire forces, integrate all coupled body states, and pass whole-lap practice validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    for name, capture in report["captures"].items():
        print(name, capture["run_id"], f"duration={capture['duration_s']:.1f}s")
        for window, score in capture["scores"].items():
            load = score["load_pooled"]
            state = capture["state_error"][window]
            print(f"  window={window}: load RMSE={load['rmse_n']:.3f} N, "
                  f"p95={load['p95_abs_n']:.3f} N; "
                  f"roll RMSE={state['roll_rmse_deg']:.3f} deg, "
                  f"pitch RMSE={state['pitch_rmse_deg']:.3f} deg, "
                  f"heave RMSE={state['heave_rmse_m']*1000:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

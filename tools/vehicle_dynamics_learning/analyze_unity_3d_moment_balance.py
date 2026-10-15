#!/usr/bin/env python3
"""Test full 3-D rigid-body moment balance from Unity WheelHit captures.

This closes a specific gap exposed by the planar recursive plant: compare
wheel-contact moments from source tire curves with the torque required by the
captured 3-D angular motion, including Unity's rotated inertia tensor,
gyroscopic term, and serialized angular drag. WheelHit/load data are privileged
offline identification labels and are never runtime observer inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    RIGIDBODY_INERTIA_DIAGONAL_KGM2,
    RIGIDBODY_INERTIA_ROTATION_XYZW,
    inertia_tensor_body,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURES = (
    ROOT / "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv",
    ROOT / "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_exact_replay_20261012_r11/"
    "wheel_dynamics_unity_vehicle_model_parity_source_exact_replay_20261012_r11.csv",
)
DEFAULT_FORCE_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "tire_curve_force_balance_20261012.json")
WHEELS = ("fl", "fr", "rl", "rr")
PREFIX = {"fl": "front_left", "fr": "front_right",
          "rl": "rear_left", "rr": "rear_right"}
SPEED_BINS = ((1.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.1))
DERIVATIVE_WINDOWS = (31, 101)
ANGULAR_DRAG_PER_S = 0.1
MASS_KG = 3.47
LINEAR_DRAG_PER_S = 0.273
GRAVITY_MPS2 = 9.80665
DEFAULT_SAMPLE_STRIDE = 25


def _metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, value = line[2:].strip().split("=", 1)
            result[key] = value
    return result


def _curve(slip: np.ndarray, landmarks: np.ndarray,
           tangent: float) -> np.ndarray:
    extremum_slip, extremum_value, asymptote_slip, asymptote_value, stiffness = landmarks
    magnitude = np.abs(slip)
    result = np.empty_like(magnitude)
    first = magnitude < extremum_slip
    t = magnitude[first] / extremum_slip
    result[first] = (
        tangent * extremum_slip * (t**3 - 2.0*t**2 + t)
        + extremum_value * (-2.0*t**3 + 3.0*t**2))
    second = (magnitude >= extremum_slip) & (magnitude < asymptote_slip)
    t = ((magnitude[second] - extremum_slip)
         / (asymptote_slip - extremum_slip))
    result[second] = (extremum_value + (asymptote_value - extremum_value)
                      * (3.0*t**2 - 2.0*t**3))
    result[magnitude >= asymptote_slip] = asymptote_value
    return np.sign(slip) * result * stiffness


def _score(error: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = error[mask]
    if selected.size == 0:
        return {"samples": 0, "rmse": None, "bias": None, "p95_abs": None}
    return {
        "samples": int(selected.shape[0]),
        "rmse": np.sqrt(np.mean(selected**2, axis=0)).tolist(),
        "bias": np.mean(selected, axis=0).tolist(),
        "p95_abs": np.quantile(np.abs(selected), 0.95, axis=0).tolist(),
    }


def _rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[:, :3]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[:, 3, None] * twice_cross + np.cross(
        xyz, twice_cross)


def _balance_data(path: Path, derivative_window: int,
                  sample_stride: int) -> dict[str, Any]:
    """Build measured force/torque targets and wheel-force features."""
    metadata = _metadata(path)
    dt = float(metadata["fixed_delta_time_s"])
    usecols = ["pos_world_x_m", "pos_world_y_m", "pos_world_z_m",
               "rot_world_x", "rot_world_y", "rot_world_z", "rot_world_w",
               "vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
               "omega_body_x_rps", "omega_body_y_rps", "omega_body_z_rps",
               "throttle_command", "steering_command"]
    for wheel in WHEELS:
        usecols += [f"{wheel}_{field}" for field in (
            "grounded", "forward_slip", "sideways_slip", "contact_force_n",
            "rpm", "steer_deg", "motor_torque_nm", "brake_torque_nm",
            "contact_body_x", "contact_body_y", "contact_body_z",
            "normal_body_x", "normal_body_y", "normal_body_z",
            "contact_forward_body_x", "contact_forward_body_y",
            "contact_forward_body_z", "contact_sideways_body_x",
            "contact_sideways_body_y", "contact_sideways_body_z")]
    frame = pd.read_csv(path, comment="#", usecols=usecols)
    position_world = frame[["pos_world_x_m", "pos_world_y_m",
                            "pos_world_z_m"]].to_numpy(dtype=np.float64)
    reset_transition = (np.linalg.norm(np.diff(position_world, axis=0), axis=1)
                        > 0.05)
    reset_rows = np.zeros(len(frame), dtype=bool)
    half_window = derivative_window // 2
    for transition in np.flatnonzero(reset_transition):
        # The finite-difference filters span both sides of this reset edge.
        reset_rows[max(0, transition - half_window):
                   min(len(frame), transition + half_window + 2)] = True
    quaternion = frame[["rot_world_x", "rot_world_y", "rot_world_z",
                        "rot_world_w"]].to_numpy(dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=1)[:, None]
    inverse = quaternion.copy()
    inverse[:, :3] *= -1.0
    velocity_body = frame[["vel_body_x_mps", "vel_body_y_mps",
                           "vel_body_z_mps"]].to_numpy(dtype=np.float64)
    velocity_world = _rotate(quaternion, velocity_body)
    velocity_world_filtered = savgol_filter(
        velocity_world, derivative_window, 3, axis=0, mode="interp")
    acceleration_world = np.gradient(velocity_world_filtered, dt, axis=0)
    acceleration_body = _rotate(inverse, acceleration_world)
    gravity_world = np.zeros_like(acceleration_world)
    gravity_world[:, 1] = -GRAVITY_MPS2
    gravity_body = _rotate(inverse, gravity_world)

    omega = frame[["omega_body_x_rps", "omega_body_y_rps",
                   "omega_body_z_rps"]].to_numpy(dtype=np.float64)
    throttle_command = frame["throttle_command"].to_numpy(dtype=np.float64)
    alpha = savgol_filter(omega, derivative_window, 3, deriv=1, delta=dt,
                          axis=0, mode="interp")
    inertia = np.asarray(inertia_tensor_body(
        RIGIDBODY_INERTIA_DIAGONAL_KGM2,
        RIGIDBODY_INERTIA_ROTATION_XYZW), dtype=np.float64)
    angular_momentum = omega @ inertia.T
    torque_target = (alpha @ inertia.T + np.cross(omega, angular_momentum)
                     + ANGULAR_DRAG_PER_S * angular_momentum)

    force_target = (MASS_KG * (acceleration_body - gravity_body)
                    + MASS_KG * LINEAR_DRAG_PER_S * velocity_body)
    force_target_total = force_target.copy()
    torque_target_total = torque_target.copy()
    com = np.asarray([float(value) for value in
                      metadata["center_of_mass_body"].split(",")])
    grounded = np.ones(len(frame), dtype=bool)
    wheels: list[dict[str, np.ndarray]] = []
    for wheel in WHEELS:
        grounded &= frame[f"{wheel}_grounded"].to_numpy() > 0.5
        load = frame[f"{wheel}_contact_force_n"].to_numpy(dtype=np.float64)
        slip_x = frame[f"{wheel}_forward_slip"].to_numpy(dtype=np.float64)
        slip_y = frame[f"{wheel}_sideways_slip"].to_numpy(dtype=np.float64)
        point = frame[[f"{wheel}_contact_body_{axis}"
                        for axis in "xyz"]].to_numpy(dtype=np.float64)
        normal = frame[[f"{wheel}_normal_body_{axis}"
                        for axis in "xyz"]].to_numpy(dtype=np.float64)
        forward = frame[[f"{wheel}_contact_forward_body_{axis}"
                         for axis in "xyz"]].to_numpy(dtype=np.float64)
        sideways = frame[[f"{wheel}_contact_sideways_body_{axis}"
                          for axis in "xyz"]].to_numpy(dtype=np.float64)
        normal_force = load[:, None] * normal
        force_target -= normal_force
        torque_target -= np.cross(point - com, normal_force)
        wheel_omega = (frame[f"{wheel}_rpm"].to_numpy(dtype=np.float64)
                       * (2.0 * np.pi / 60.0))
        wheel_alpha = savgol_filter(
            wheel_omega, derivative_window, 3, deriv=1, delta=dt,
            mode="interp")
        motor = frame[f"{wheel}_motor_torque_nm"].to_numpy(dtype=np.float64)
        brake = frame[f"{wheel}_brake_torque_nm"].to_numpy(dtype=np.float64)
        steering = np.deg2rad(frame[f"{wheel}_steer_deg"].to_numpy(
            dtype=np.float64))
        prefix = PREFIX[wheel]
        radius = float(metadata[f"{prefix}_radius_m"])
        wheels.append({"grounded": frame[f"{wheel}_grounded"].to_numpy() > 0.5,
                       "load": load, "slip_x": slip_x, "slip_y": slip_y,
                       "lever": point - com, "forward": forward,
                       "sideways": sideways, "normal": normal,
                       "omega": wheel_omega,
                       "alpha": wheel_alpha, "motor": motor,
                       "brake": brake, "radius": radius,
                       "steering_rad": steering})

    speed = np.linalg.norm(velocity_body, axis=1)
    valid = grounded & (speed > 1.0)
    valid[:derivative_window] = False
    valid[-derivative_window:] = False
    valid &= ~reset_rows
    valid &= (np.isfinite(force_target).all(axis=1)
              & np.isfinite(torque_target).all(axis=1))
    for wheel in wheels:
        valid &= np.isfinite(wheel["load"])
        valid &= np.isfinite(wheel["slip_x"]) & np.isfinite(wheel["slip_y"])
        valid &= np.isfinite(wheel["omega"]) & np.isfinite(wheel["alpha"])
        valid &= np.isfinite(wheel["motor"]) & np.isfinite(wheel["brake"])
        valid &= np.isfinite(wheel["forward"]).all(axis=1)
        valid &= np.isfinite(wheel["sideways"]).all(axis=1)
    rows = np.flatnonzero(valid)[::sample_stride]
    return {"metadata": metadata, "wheels": wheels, "force_target": force_target,
            "torque_target": torque_target,
            "force_target_total": force_target_total,
            "torque_target_total": torque_target_total,
            "rows": rows, "inertia": inertia,
            "speed": speed, "velocity_body": velocity_body,
            "omega_body": omega, "position_world": position_world,
            "rotation_world_xyzw": quaternion,
            "throttle_command": throttle_command,
            "steering_command": frame["steering_command"].to_numpy(
                dtype=np.float64),
            "reset_discontinuity_count": int(np.count_nonzero(reset_transition)),
            "reset_excluded_rows": int(np.count_nonzero(reset_rows)),
            "derivative_window_s": derivative_window * dt}


def _predict_balance(data: dict[str, Any], parameters: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale_x, tangent_x, scale_y, tangent_y = parameters
    force = np.zeros_like(data["force_target"])
    torque = np.zeros_like(data["torque_target"])
    wheel_forward_force = np.zeros((len(force), len(WHEELS)), dtype=np.float64)
    for index, wheel in enumerate(WHEELS):
        features = data["wheels"][index]
        prefix = PREFIX[wheel]
        fx_curve = np.asarray([float(value) for value in data["metadata"][
            f"{prefix}_forward_curve"].split(",")])
        fy_curve = np.asarray([float(value) for value in data["metadata"][
            f"{prefix}_sideways_curve"].split(",")])
        fx = (scale_x * features["load"]
              * _curve(features["slip_x"], fx_curve, tangent_x))
        fy = (-scale_y * features["load"]
              * _curve(features["slip_y"], fy_curve, tangent_y))
        wheel_force = (fx[:, None] * features["forward"]
                       + fy[:, None] * features["sideways"])
        wheel_forward_force[:, index] = fx
        force += wheel_force
        torque += np.cross(features["lever"], wheel_force)
    return force, torque, wheel_forward_force


def _fit_joint_balance(train: dict[str, Any], initial: np.ndarray
                       ) -> dict[str, Any]:
    rows = train["rows"]
    force_scale = np.maximum(np.sqrt(np.mean(
        train["force_target"][rows]**2, axis=0)), 1.0)
    torque_scale = np.maximum(np.sqrt(np.mean(
        train["torque_target"][rows]**2, axis=0)), 0.02)

    def residual(parameters: np.ndarray) -> np.ndarray:
        force, torque, _ = _predict_balance(train, parameters)
        return np.concatenate((
            ((force[rows] - train["force_target"][rows]) / force_scale).ravel(),
            ((torque[rows] - train["torque_target"][rows]) / torque_scale).ravel()))

    result = least_squares(
        residual, initial,
        bounds=([0.1, 0.01, 0.1, 0.01], [4.0, 80.0, 4.0, 250.0]),
        loss="soft_l1", f_scale=1.0, max_nfev=250)
    return {"parameters": result.x,
            "optimizer": {"status": int(result.status),
                          "message": result.message,
                          "nfev": int(result.nfev),
                          "fit_samples": int(rows.size),
                          "normalized_cost": float(result.cost),
                          "force_axis_scale_n": force_scale.tolist(),
                          "torque_axis_scale_nm": torque_scale.tolist()}}


def _wheel_spin_prediction(data: dict[str, Any], tire_parameters: np.ndarray,
                           wheel_inertia: float, wheel_damping: float
                           ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    _, _, wheel_force = _predict_balance(data, tire_parameters)
    predicted: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    for index, features in enumerate(data["wheels"]):
        omega = features["omega"]
        brake_sign = np.sign(omega)
        angular_acceleration = (
            features["motor"] - features["brake"] * brake_sign
            - features["radius"] * wheel_force[:, index]
            - wheel_damping * omega) / wheel_inertia
        predicted.append(angular_acceleration)
        valid.append(np.abs(omega) > 15.0)
    return predicted, valid


def _fit_joint_balance_with_spin(train: dict[str, Any], initial: np.ndarray
                                 ) -> dict[str, Any]:
    """Fit tire curves and wheel spin inertia/damping to one shared state model."""
    rows = train["rows"]
    force_scale = np.maximum(np.sqrt(np.mean(
        train["force_target"][rows]**2, axis=0)), 1.0)
    torque_scale = np.maximum(np.sqrt(np.mean(
        train["torque_target"][rows]**2, axis=0)), 0.02)
    wheel_alpha_scale = []
    for features in train["wheels"]:
        valid = np.abs(features["omega"][rows]) > 15.0
        if not np.any(valid):
            raise ValueError("no rotating wheel samples in training rows")
        wheel_alpha_scale.append(max(
            float(np.sqrt(np.mean(features["alpha"][rows][valid]**2))), 20.0))
    wheel_alpha_scale_array = np.asarray(wheel_alpha_scale)

    def residual(parameters: np.ndarray) -> np.ndarray:
        tire = parameters[:4]
        inertia, damping = parameters[4:]
        force, torque, wheel_force = _predict_balance(train, tire)
        parts = [
            ((force[rows] - train["force_target"][rows]) / force_scale).ravel(),
            ((torque[rows] - train["torque_target"][rows]) / torque_scale).ravel(),
        ]
        for index, features in enumerate(train["wheels"]):
            selected = rows[np.abs(features["omega"][rows]) > 15.0]
            omega = features["omega"][selected]
            alpha_predicted = (
                features["motor"][selected]
                - features["brake"][selected] * np.sign(omega)
                - features["radius"] * wheel_force[selected, index]
                - damping * omega) / inertia
            parts.append((alpha_predicted - features["alpha"][selected])
                         / wheel_alpha_scale_array[index])
        return np.concatenate(parts)

    result = least_squares(
        residual, initial,
        bounds=([0.1, 0.01, 0.1, 0.01, 1.0e-5, 0.0],
                [4.0, 80.0, 4.0, 250.0, 0.005, 1.0]),
        loss="soft_l1", f_scale=1.0, max_nfev=300)
    return {"parameters": result.x,
            "optimizer": {"status": int(result.status),
                          "message": result.message,
                          "nfev": int(result.nfev),
                          "fit_samples": int(rows.size),
                          "normalized_cost": float(result.cost),
                          "force_axis_scale_n": force_scale.tolist(),
                          "body_torque_axis_scale_nm": torque_scale.tolist(),
                          "wheel_angular_acceleration_scale_radps2":
                              wheel_alpha_scale_array.tolist()}}


def _wheel_spin_metrics(data: dict[str, Any], tire_parameters: np.ndarray,
                        wheel_inertia: float, wheel_damping: float
                        ) -> dict[str, Any]:
    predicted, rotating = _wheel_spin_prediction(
        data, tire_parameters, wheel_inertia, wheel_damping)
    result: dict[str, Any] = {"by_wheel": {}, "by_speed": {}}
    rows = data["rows"]
    for index, wheel in enumerate(WHEELS):
        features = data["wheels"][index]
        valid_rows = rows[rotating[index][rows]]
        error = predicted[index][valid_rows] - features["alpha"][valid_rows]
        result["by_wheel"][wheel] = {
            "samples": int(valid_rows.size),
            "angular_acceleration_rmse_radps2": float(
                np.sqrt(np.mean(error**2))),
            "angular_acceleration_bias_radps2": float(np.mean(error)),
            "angular_acceleration_p95_abs_radps2": float(
                np.quantile(np.abs(error), 0.95)),
            "wheel_torque_balance_rmse_nm": float(
                wheel_inertia * np.sqrt(np.mean(error**2))),
        }
    for low, high in SPEED_BINS:
        selected = rows[(data["speed"][rows] >= low)
                        & (data["speed"][rows] < high)]
        all_errors = []
        for index, features in enumerate(data["wheels"]):
            wheel_rows = selected[rotating[index][selected]]
            all_errors.extend((predicted[index][wheel_rows]
                               - features["alpha"][wheel_rows]).tolist())
        errors = np.asarray(all_errors, dtype=np.float64)
        result["by_speed"][f"{low:g}-{high:g}mps"] = {
            "samples": int(errors.size),
            "angular_acceleration_rmse_radps2": (
                float(np.sqrt(np.mean(errors**2))) if errors.size else None),
            "angular_acceleration_p95_abs_radps2": (
                float(np.quantile(np.abs(errors), 0.95)) if errors.size else None),
        }
    return result


def _balance_metrics(data: dict[str, Any], parameters: np.ndarray,
                     rows: np.ndarray | None = None
                     ) -> dict[str, Any]:
    if rows is None:
        rows = data["rows"]
    force, torque, _ = _predict_balance(data, parameters)
    force_error = force[rows] - data["force_target"][rows]
    torque_error = torque[rows] - data["torque_target"][rows]
    angular_accel_error = torque_error @ np.linalg.inv(data["inertia"]).T
    return {
        "samples": int(rows.size),
        "force_error_n": _score(force_error, np.ones(rows.size, dtype=bool)),
        "torque_error_nm": _score(torque_error, np.ones(rows.size, dtype=bool)),
        "angular_acceleration_error_radps2": _score(
            angular_accel_error, np.ones(rows.size, dtype=bool)),
    }


def _metrics_by_speed(data: dict[str, Any], parameters: np.ndarray
                      ) -> dict[str, Any]:
    result = _balance_metrics(data, parameters)
    rows = data["rows"]
    result["by_speed"] = {}
    for low, high in SPEED_BINS:
        selected = rows[(data["speed"][rows] >= low)
                        & (data["speed"][rows] < high)]
        result["by_speed"][f"{low:g}-{high:g}mps"] = _balance_metrics(
            data, parameters, selected)
    return result


def _capture(path: Path, params: dict[str, float],
             derivative_window: int, sample_stride: int) -> dict[str, Any]:
    metadata = _metadata(path)
    dt = float(metadata["fixed_delta_time_s"])
    if derivative_window < 5 or derivative_window % 2 == 0:
        raise ValueError("derivative window must be odd and at least five")
    if sample_stride < 1:
        raise ValueError("sample stride must be positive")
    usecols = ["vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
               "omega_body_x_rps", "omega_body_y_rps", "omega_body_z_rps"]
    for wheel in WHEELS:
        usecols += [f"{wheel}_{field}" for field in (
            "grounded", "forward_slip", "sideways_slip", "contact_force_n",
            "contact_body_x", "contact_body_y", "contact_body_z",
            "normal_body_x", "normal_body_y", "normal_body_z",
            "contact_forward_body_x", "contact_forward_body_y",
            "contact_forward_body_z", "contact_sideways_body_x",
            "contact_sideways_body_y", "contact_sideways_body_z")]
    frame = pd.read_csv(path, comment="#", usecols=usecols)
    omega = frame[["omega_body_x_rps", "omega_body_y_rps",
                   "omega_body_z_rps"]].to_numpy(dtype=np.float64)
    alpha = savgol_filter(omega, derivative_window, 3, deriv=1, delta=dt,
                          axis=0, mode="interp")
    inertia = np.asarray(inertia_tensor_body(
        RIGIDBODY_INERTIA_DIAGONAL_KGM2,
        RIGIDBODY_INERTIA_ROTATION_XYZW), dtype=np.float64)
    angular_momentum = omega @ inertia.T
    gyroscopic = np.cross(omega, angular_momentum)
    required_torque = alpha @ inertia.T + gyroscopic
    required_torque_with_drag = required_torque + ANGULAR_DRAG_PER_S * angular_momentum

    com = np.asarray([float(value) for value in
                      metadata["center_of_mass_body"].split(",")])
    predicted_torque = np.zeros_like(required_torque)
    predicted_torque_baseline = np.zeros_like(required_torque)
    grounded_all = np.ones(len(frame), dtype=bool)
    speed = np.linalg.norm(frame[["vel_body_x_mps", "vel_body_y_mps",
                                  "vel_body_z_mps"]].to_numpy(dtype=np.float64), axis=1)
    def add_wheel_torque(target: np.ndarray, scale_x: float, tangent_x: float,
                         scale_y: float, tangent_y: float) -> None:
        for wheel in WHEELS:
            grounded = frame[f"{wheel}_grounded"].to_numpy() > 0.5
            grounded_all[:] &= grounded
            point = frame[[f"{wheel}_contact_body_{axis}"
                           for axis in ("x", "y", "z")]].to_numpy(dtype=np.float64)
            lever = point - com
            normal = frame[[f"{wheel}_normal_body_{axis}"
                            for axis in ("x", "y", "z")]].to_numpy(dtype=np.float64)
            forward = frame[[f"{wheel}_contact_forward_body_{axis}"
                             for axis in ("x", "y", "z")]].to_numpy(dtype=np.float64)
            sideways = frame[[f"{wheel}_contact_sideways_body_{axis}"
                              for axis in ("x", "y", "z")]].to_numpy(dtype=np.float64)
            normal_load = frame[f"{wheel}_contact_force_n"].to_numpy(dtype=np.float64)
            slip_x = frame[f"{wheel}_forward_slip"].to_numpy(dtype=np.float64)
            slip_y = frame[f"{wheel}_sideways_slip"].to_numpy(dtype=np.float64)
            fx_curve = np.asarray([float(value) for value in metadata[
                f"{PREFIX[wheel]}_forward_curve"].split(",")])
            fy_curve = np.asarray([float(value) for value in metadata[
                f"{PREFIX[wheel]}_sideways_curve"].split(",")])
            fx = (scale_x * normal_load
                  * _curve(slip_x, fx_curve, tangent_x))
            fy = (-scale_y * normal_load
                  * _curve(slip_y, fy_curve, tangent_y))
            force = (fx[:, None] * forward + fy[:, None] * sideways
                     + normal_load[:, None] * normal)
            target += np.cross(lever, force)

    add_wheel_torque(
        predicted_torque, params["longitudinal_scale"],
        params["longitudinal_zero_slip_tangent"],
        params["lateral_scale"], params["lateral_zero_slip_tangent"])
    add_wheel_torque(predicted_torque_baseline, 1.0, 1.0, 1.0, 1.0)
    edge = derivative_window
    valid = grounded_all & (speed > 1.0) & np.isfinite(required_torque).all(axis=1)
    valid[:edge] = False
    valid[-edge:] = False
    rows = np.flatnonzero(valid)[::sample_stride]
    result: dict[str, Any] = {
        "capture": path.name,
        "samples": len(frame),
        "scored_samples": len(rows),
        "derivative_window_samples": derivative_window,
        "derivative_window_seconds": derivative_window * dt,
        "body_inertia_tensor_kgm2": inertia.tolist(),
        "scalar_yaw_inertia_only_kgm2": float(inertia[1, 1]),
        "comparison": {},
        "by_speed": {},
    }
    for name, predicted in (("serialized_landmark_unit_tangent", predicted_torque_baseline),
                            ("r07_fit_frozen", predicted_torque)):
        for drag_mode, required in (("without_angular_drag", required_torque),
                                    ("with_serialized_angular_drag", required_torque_with_drag)):
            residual = predicted - required
            result["comparison"][f"{name}/{drag_mode}"] = _score(residual, np.isin(
                np.arange(len(frame)), rows))
        residual_full = predicted - required_torque_with_drag
        bins: dict[str, Any] = {}
        for low, high in SPEED_BINS:
            mask = np.zeros(len(frame), dtype=bool)
            mask[rows] = True
            mask &= (speed >= low) & (speed < high)
            bins[f"{low:g}-{high:g}mps"] = _score(
                residual_full, mask)
        result["by_speed"][name] = bins

    # Compare the common planar yaw-only equation with full 3-D required yaw
    # torque to quantify off-diagonal and gyroscopic contributions.
    scalar_yaw = inertia[1, 1] * alpha[:, 1]
    result["yaw_equation_ablation"] = {
        "full_required_yaw_torque_rmse_vs_contact_fit_Nm": float(np.sqrt(np.mean(
            (predicted_torque[rows, 1] - required_torque_with_drag[rows, 1])**2))),
        "scalar_Iyy_alpha_yaw_torque_rmse_vs_contact_fit_Nm": float(np.sqrt(np.mean(
            (predicted_torque[rows, 1] - scalar_yaw[rows])**2))),
        "gyroscopic_and_off_diagonal_correction_rms_Nm": float(np.sqrt(np.mean(
            (required_torque_with_drag[rows, 1] - scalar_yaw[rows])**2))),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", type=Path,
                        default=DEFAULT_CAPTURES)
    parser.add_argument("--force-fit", type=Path, default=DEFAULT_FORCE_FIT)
    parser.add_argument("--output", type=Path, default=ROOT / (
        "live_runs/unity_exact_binary_audit_20261012/"
        "full3d_moment_balance_20261012.json"))
    parser.add_argument("--sample-stride", type=int, default=25)
    parser.add_argument("--joint-fit-only", action="store_true",
                        help="fit/score one shared tire curve against force and moment balance")
    parser.add_argument("--joint-fit-output", type=Path, default=ROOT / (
        "live_runs/unity_exact_binary_audit_20261012/"
        "joint_tire_force_moment_balance_20261012.json"))
    parser.add_argument("--wheel-spin-joint-only", action="store_true",
                        help="jointly fit tire curves and wheel spin dynamics")
    parser.add_argument("--wheel-spin-joint-output", type=Path, default=ROOT / (
        "live_runs/unity_exact_binary_audit_20261012/"
        "joint_tire_body_wheel_spin_balance_20261012.json"))
    args = parser.parse_args()
    fit = json.loads(args.force_fit.read_text(encoding="utf-8"))
    fitted = fit["curve_model"]["training_fit"]
    parameters = {
        "longitudinal_scale": fitted["longitudinal_scale"],
        "longitudinal_zero_slip_tangent": fitted[
            "longitudinal_zero_slip_tangent"],
        "lateral_scale": fitted["lateral_scale"],
        "lateral_zero_slip_tangent": fitted["lateral_zero_slip_tangent"],
    }
    if args.wheel_spin_joint_only:
        if len(args.captures) != 2:
            raise ValueError("joint wheel fit requires training and independent captures")
        if args.wheel_spin_joint_output.exists():
            raise FileExistsError(
                f"refusing to overwrite {args.wheel_spin_joint_output}")
        training = _balance_data(args.captures[0], 31, args.sample_stride)
        independent = _balance_data(args.captures[1], 31, args.sample_stride)
        spin_fit = fit["wheel_spin_balance"]["training_capture_fit"]
        tire_initial = np.asarray([parameters[key] for key in (
            "longitudinal_scale", "longitudinal_zero_slip_tangent",
            "lateral_scale", "lateral_zero_slip_tangent")])
        initial = np.concatenate((tire_initial, np.asarray([
            spin_fit["fitted_shared_wheel_inertia_kgm2"],
            spin_fit["fitted_shared_damping_nms"]])))
        joint = _fit_joint_balance_with_spin(training, initial)
        joint_parameters = joint["parameters"]
        models = {
            "force_only_tire_fit_plus_r07_wheel_fit": {
                "tire": tire_initial,
                "wheel_inertia": spin_fit[
                    "fitted_shared_wheel_inertia_kgm2"],
                "wheel_damping": spin_fit["fitted_shared_damping_nms"],
            },
            "force_moment_and_wheel_spin_joint_fit": {
                "tire": joint_parameters[:4],
                "wheel_inertia": float(joint_parameters[4]),
                "wheel_damping": float(joint_parameters[5]),
            },
        }
        scores: dict[str, Any] = {}
        for name, model in models.items():
            scores[name] = {}
            for split, data in (("training", training),
                                ("independent_run", independent)):
                scores[name][split] = {
                    "body_force_and_moment": _metrics_by_speed(
                        data, model["tire"]),
                    "wheel_spin": _wheel_spin_metrics(
                        data, model["tire"], model["wheel_inertia"],
                        model["wheel_damping"]),
                }
        payload = {
            "schema_version": 1,
            "status": "diagnostic joint fit; no runtime model promoted",
            "training_capture": str(args.captures[0].relative_to(ROOT)),
            "independent_capture": str(args.captures[1].relative_to(ROOT)),
            "method": "one frozen two-piece tire model plus shared wheel inertia/damping, jointly fit against 3-axis body force, 3-axis body moment, and per-wheel angular acceleration",
            "initial_parameters": initial.tolist(),
            "joint_parameters": joint_parameters.tolist(),
            "optimizer": joint["optimizer"],
            "models": scores,
            "limitations": [
                "WheelHit force/slip/load and body truth are offline labels; the evaluation is not sensor-only.",
                "Wheel angular acceleration is smoothed over 31 ms and only rows with wheel speed above 15 rad/s are scored.",
                "r11 is a distinct source trajectory, not exact packet-command parity with r02.",
                "This is one-step force/torque/spin closure; it does not establish recursive plant accuracy.",
            ],
        }
        args.wheel_spin_joint_output.parent.mkdir(parents=True, exist_ok=True)
        args.wheel_spin_joint_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(args.wheel_spin_joint_output)
        for model_name, model_scores in scores.items():
            for split, score in model_scores.items():
                spin = score["wheel_spin"]["by_wheel"]
                spin_rmse = float(np.sqrt(np.mean([
                    wheel["angular_acceleration_rmse_radps2"]**2
                    for wheel in spin.values()])))
                body = score["body_force_and_moment"]
                print(model_name, split,
                      "force_axis_rmse_n", body["force_error_n"]["rmse"],
                      "yaw_alpha_rmse_radps2",
                      body["angular_acceleration_error_radps2"]["rmse"][1],
                      "wheel_alpha_rmse_radps2", spin_rmse)
        return 0
    if args.joint_fit_only:
        if len(args.captures) != 2:
            raise ValueError("joint fit requires exactly training and independent captures")
        if args.joint_fit_output.exists():
            raise FileExistsError(f"refusing to overwrite {args.joint_fit_output}")
        training = _balance_data(args.captures[0], 31, args.sample_stride)
        independent = _balance_data(args.captures[1], 31, args.sample_stride)
        initial = np.asarray([parameters[key] for key in (
            "longitudinal_scale", "longitudinal_zero_slip_tangent",
            "lateral_scale", "lateral_zero_slip_tangent")])
        joint = _fit_joint_balance(training, initial)
        model_parameters = {
            "serialized_landmark_unit_tangent": np.asarray([1.0, 1.0, 1.0, 1.0]),
            "force_only_r07_fit_frozen": initial,
            "force_and_moment_joint_r07_fit_frozen": joint["parameters"],
        }
        results = {}
        for name, candidate in model_parameters.items():
            results[name] = {
                "training": _metrics_by_speed(training, candidate),
                "independent_run": _metrics_by_speed(independent, candidate),
            }
        payload = {
            "schema_version": 1,
            "status": "diagnostic candidate; no runtime model promoted",
            "fit_objective": "joint 3-axis tire-force and 3-axis rigid-body torque balance, normalized by training-target RMS per axis",
            "training_capture": str(args.captures[0].relative_to(ROOT)),
            "independent_capture": str(args.captures[1].relative_to(ROOT)),
            "training_fit_initial_parameters": initial.tolist(),
            "joint_fit_parameters": joint["parameters"].tolist(),
            "optimizer": joint["optimizer"],
            "models": results,
            "interpretation": [
                "The joint fit uses Unity WheelHit slip/load/direction and motion truth as offline labels.",
                "A transfer gain would validate force/torque closure only, not command-only recursive prediction.",
                "r11 is a distinct source trajectory; its public commands are not considered an exact replay of r02.",
            ],
        }
        args.joint_fit_output.parent.mkdir(parents=True, exist_ok=True)
        args.joint_fit_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(args.joint_fit_output)
        for model, scores in results.items():
            for split, score in scores.items():
                print(model, split,
                      "force_vector_rmse_n",
                      score["force_error_n"]["rmse"],
                      "torque_vector_rmse_nm",
                      score["torque_error_nm"]["rmse"],
                      "yaw_accel_rmse_radps2",
                      score["angular_acceleration_error_radps2"]["rmse"][1])
        return 0
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    captures = [_capture(path, parameters, window, args.sample_stride)
                for path in args.captures
                for window in DERIVATIVE_WINDOWS]
    payload = {
        "schema_version": 1,
        "status": "full 3-D torque-balance diagnosis using r07-fitted tire parameters, unchanged on r11",
        "training_fit": parameters,
        "training_run": str(args.captures[0].relative_to(ROOT)),
        "independent_run": str(args.captures[1].relative_to(ROOT)),
        "method": {
            "required_torque": "I_body * angular_acceleration_body + omega_body x (I_body * omega_body)",
            "inertia_tensor": "Unity Rigidbody principal diagonal rotated into body axes from the captured inertiaTensorRotation",
            "contact_moment": "sum((WheelHit.point - COM) x (normal_load*normal + fitted longitudinal tire force*forwardDir + fitted lateral tire force*sidewaysDir))",
            "angular_drag": "compare with and without serialized Rigidbody.angularDrag=0.1/s",
            "identification_inputs": "Unity-only WheelHit, contact load/directions, body angular velocity and capture truth",
        },
        "captures": captures,
        "limitations": [
            "WheelHit slip/load and truth angular velocity are privileged labels, not runtime odometry inputs",
            "angular acceleration uses a filtered derivative; both 31 ms and 101 ms windows are reported",
            "this validates torque closure, not a command-only recursive yaw model",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    for entry in captures:
        print(entry["capture"], entry["derivative_window_seconds"])
        for key, value in entry["comparison"].items():
            print(" ", key, "torque_rmse_Nm", value["rmse"])
        print("  yaw ablation", entry["yaw_equation_ablation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

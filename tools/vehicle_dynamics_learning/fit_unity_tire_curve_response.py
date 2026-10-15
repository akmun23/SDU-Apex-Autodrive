#!/usr/bin/env python3
"""Fit guide-shaped tire curves to pinned Unity force-balance captures.

The fit uses WheelHit slip/load/directions and simulator body truth only as
offline identification labels. It predicts the aggregate tangential tire
force needed by the measured rigid-body acceleration; it is not a sensor-only
observer or a free-running plant validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
DEFAULT_VALIDATION = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_exact_replay_20261012_r11/"
    "wheel_dynamics_unity_vehicle_model_parity_source_exact_replay_20261012_r11.csv")
WHEELS = ("fl", "fr", "rl", "rr")
WHEEL_PREFIX = {"fl": "front_left", "fr": "front_right",
                "rl": "rear_left", "rr": "rear_right"}
GRAVITY_MPS2 = 9.80665
RIGIDBODY_DRAG = 0.273
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


def _rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = q[:, :3]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + q[:, 3, None] * twice_cross + np.cross(xyz, twice_cross)


def _curve(slip: np.ndarray, landmarks: np.ndarray,
           zero_tangent: float) -> np.ndarray:
    extremum_slip, extremum_value, asymptote_slip, asymptote_value, stiffness = landmarks
    magnitude = np.abs(slip)
    result = np.empty_like(magnitude)
    first = magnitude < extremum_slip
    t = magnitude[first] / extremum_slip
    result[first] = (
        zero_tangent * extremum_slip * (t**3 - 2.0 * t**2 + t)
        + extremum_value * (-2.0 * t**3 + 3.0 * t**2))
    second = (magnitude >= extremum_slip) & (magnitude < asymptote_slip)
    t = ((magnitude[second] - extremum_slip)
         / (asymptote_slip - extremum_slip))
    result[second] = (extremum_value + (asymptote_value - extremum_value)
                      * (3.0 * t**2 - 2.0 * t**3))
    result[magnitude >= asymptote_slip] = asymptote_value
    return np.sign(slip) * result * stiffness


def _prepare(path: Path, derivative_window_samples: int,
             slip_definition: str = "wheel_hit") -> dict[str, Any]:
    metadata = _metadata(path)
    frame = pd.read_csv(path, comment="#")
    dt = float(metadata["fixed_delta_time_s"])
    mass = float(metadata["mass_kg"])
    if derivative_window_samples < 5 or derivative_window_samples % 2 == 0:
        raise ValueError("derivative window must be odd and at least five samples")

    velocity_body = frame[["vel_body_x_mps", "vel_body_y_mps",
                           "vel_body_z_mps"]].to_numpy(dtype=np.float64)
    quaternion = frame[["rot_world_x", "rot_world_y", "rot_world_z",
                        "rot_world_w"]].to_numpy(dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=1)[:, None]
    velocity_world = _rotate(quaternion, velocity_body)
    filtered_world_velocity = savgol_filter(
        velocity_world, derivative_window_samples, 3, axis=0, mode="interp")
    acceleration_world = np.gradient(filtered_world_velocity, dt, axis=0)
    inverse = quaternion.copy()
    inverse[:, :3] *= -1.0
    acceleration_body = _rotate(inverse, acceleration_world)
    gravity_world = np.zeros_like(acceleration_world)
    gravity_world[:, 1] = -GRAVITY_MPS2
    gravity_body = _rotate(inverse, gravity_world)

    # Rigidbody equation: m(a-g) = tire + contact-normal + drag.
    # Unity drag is represented by its local linear damping in this balance.
    force_target = mass * (acceleration_body - gravity_body)
    force_target += mass * RIGIDBODY_DRAG * velocity_body
    if slip_definition not in ("wheel_hit", "guide_equations"):
        raise ValueError("slip_definition must be wheel_hit or guide_equations")
    wheel_directions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    wheel_slips: list[tuple[np.ndarray, np.ndarray]] = []
    grounded = np.ones(len(frame), dtype=bool)
    guide_speed_valid = np.ones(len(frame), dtype=bool)
    velocity_body = frame[["vel_body_x_mps", "vel_body_y_mps",
                           "vel_body_z_mps"]].to_numpy(dtype=np.float64)
    omega_body = frame[["omega_body_x_rps", "omega_body_y_rps",
                        "omega_body_z_rps"]].to_numpy(dtype=np.float64)
    center_of_mass = np.asarray([
        float(value) for value in metadata["center_of_mass_body"].split(",")],
        dtype=np.float64)
    for wheel in WHEELS:
        grounded &= frame[f"{wheel}_grounded"].to_numpy() > 0.5
        normal = frame[[f"{wheel}_normal_body_x", f"{wheel}_normal_body_y",
                        f"{wheel}_normal_body_z"]].to_numpy(dtype=np.float64)
        load = frame[f"{wheel}_contact_force_n"].to_numpy(dtype=np.float64)
        force_target -= normal * load[:, None]
        forward = frame[[f"{wheel}_contact_forward_body_x",
                         f"{wheel}_contact_forward_body_y",
                         f"{wheel}_contact_forward_body_z"]].to_numpy(dtype=np.float64)
        sideways = frame[[f"{wheel}_contact_sideways_body_x",
                          f"{wheel}_contact_sideways_body_y",
                          f"{wheel}_contact_sideways_body_z"]].to_numpy(dtype=np.float64)
        wheel_directions.append((forward, sideways, load))
        if slip_definition == "wheel_hit":
            longitudinal_slip = frame[
                f"{wheel}_forward_slip"].to_numpy(dtype=np.float64)
            lateral_slip = frame[
                f"{wheel}_sideways_slip"].to_numpy(dtype=np.float64)
        else:
            point = frame[[f"{wheel}_contact_body_{axis}"
                           for axis in "xyz"]].to_numpy(dtype=np.float64)
            contact_velocity = velocity_body + np.cross(
                omega_body, point - center_of_mass)
            longitudinal_speed = np.einsum(
                "ij,ij->i", contact_velocity, forward)
            lateral_speed = np.einsum(
                "ij,ij->i", contact_velocity, sideways)
            wheel_name = WHEEL_PREFIX[wheel]
            radius = float(metadata[f"{wheel_name}_radius_m"])
            wheel_omega = (frame[f"{wheel}_rpm"].to_numpy(dtype=np.float64)
                           * (2.0 * np.pi / 60.0))
            surface_speed = radius * wheel_omega
            denominator = np.abs(longitudinal_speed)
            guide_speed_valid &= denominator > 1.0
            longitudinal_slip = np.zeros_like(longitudinal_speed)
            lateral_slip = np.zeros_like(lateral_speed)
            np.divide(surface_speed - longitudinal_speed,
                      longitudinal_speed, out=longitudinal_slip,
                      where=denominator > 1.0)
            np.divide(lateral_speed, denominator, out=lateral_slip,
                      where=denominator > 1.0)
        wheel_slips.append((longitudinal_slip, lateral_slip))

    speed = np.linalg.norm(velocity_body, axis=1)
    valid = grounded & (speed > 1.0)
    if slip_definition == "guide_equations":
        valid &= guide_speed_valid
    edge = derivative_window_samples
    valid[:edge] = False
    valid[-edge:] = False
    valid &= np.isfinite(force_target).all(axis=1)
    for forward, sideways, load in wheel_directions:
        valid &= np.isfinite(forward).all(axis=1) & np.isfinite(sideways).all(axis=1)
        valid &= np.isfinite(load)
    for longitudinal, lateral in wheel_slips:
        valid &= np.isfinite(longitudinal) & np.isfinite(lateral)

    return {
        "path": path,
        "metadata": metadata,
        "frame": frame,
        "force_target": force_target,
        "wheel_directions": wheel_directions,
        "wheel_slips": wheel_slips,
        "valid": valid,
        "speed": speed,
        "sample_count": len(frame),
        "dt_s": dt,
        "derivative_window_s": derivative_window_samples * dt,
    }


def _predict(data: dict[str, Any], parameters: np.ndarray) -> np.ndarray:
    scale_x, tangent_x, scale_y, tangent_y = parameters
    prediction = np.zeros_like(data["force_target"])
    for index, wheel in enumerate(WHEELS):
        prefix = WHEEL_PREFIX[wheel]
        metadata = data["metadata"]
        forward_landmarks = np.asarray([
            float(value) for value in metadata[f"{prefix}_forward_curve"].split(",")])
        lateral_landmarks = np.asarray([
            float(value) for value in metadata[f"{prefix}_sideways_curve"].split(",")])
        forward, sideways, load = data["wheel_directions"][index]
        slip_x, slip_y = data["wheel_slips"][index]
        force_x = (scale_x * load
                   * _curve(slip_x, forward_landmarks, tangent_x))
        # Positive WheelHit sideways slip follows positive contact velocity;
        # the tire force opposes it.
        force_y = (-scale_y * load
                   * _curve(slip_y, lateral_landmarks, tangent_y))
        prediction += force_x[:, None] * forward + force_y[:, None] * sideways
    return prediction


def _wheel_longitudinal_force(data: dict[str, Any],
                              parameters: np.ndarray) -> np.ndarray:
    scale_x, tangent_x = parameters[:2]
    force = np.zeros((data["sample_count"], len(WHEELS)), dtype=np.float64)
    metadata = data["metadata"]
    for index, wheel in enumerate(WHEELS):
        curve = np.asarray([
            float(value) for value in metadata[
                f"{WHEEL_PREFIX[wheel]}_forward_curve"].split(",")])
        load = data["wheel_directions"][index][2]
        slip = data["wheel_slips"][index][0]
        force[:, index] = scale_x * load * _curve(slip, curve, tangent_x)
    return force


def _wheel_spin_balance(data: dict[str, Any], curve_parameters: np.ndarray,
                        fitted_balance: tuple[float, float] | None = None
                        ) -> dict[str, Any]:
    frame = data["frame"]
    dt = data["dt_s"]
    window = max(5, round(data["derivative_window_s"] / dt))
    if window % 2 == 0:
        window += 1
    n = data["sample_count"]
    force_x = _wheel_longitudinal_force(data, curve_parameters)
    angular_speed = np.column_stack([
        frame[f"{wheel}_rpm"].to_numpy(dtype=np.float64) * (2.0 * np.pi / 60.0)
        for wheel in WHEELS])
    angular_acceleration = savgol_filter(
        angular_speed, window, 3, deriv=1, delta=dt, axis=0, mode="interp")
    motor = np.column_stack([
        frame[f"{wheel}_motor_torque_nm"].to_numpy(dtype=np.float64)
        for wheel in WHEELS])
    brake = np.column_stack([
        frame[f"{wheel}_brake_torque_nm"].to_numpy(dtype=np.float64)
        for wheel in WHEELS])
    radius = np.asarray([
        float(data["metadata"][f"{WHEEL_PREFIX[wheel]}_radius_m"])
        for wheel in WHEELS])
    torque_available = (motor - brake * np.sign(angular_speed)
                        - force_x * radius[None, :])
    valid = data["valid"] & (np.abs(angular_speed).max(axis=1) > 15.0)
    valid[:window] = False
    valid[-window:] = False
    sample_rows = np.flatnonzero(valid)[::DEFAULT_SAMPLE_STRIDE]
    alpha = angular_acceleration[sample_rows].ravel()
    omega = angular_speed[sample_rows].ravel()
    target = torque_available[sample_rows].ravel()
    design = np.column_stack((alpha, omega))
    if fitted_balance is None:
        fit = lsq_linear(design, target,
                         bounds=([1.0e-6, 0.0], [0.005, 1.0]))
        balance = tuple(float(value) for value in fit.x)
        fit_samples = int(len(target))
    else:
        balance = fitted_balance
        fit_samples = 0
    inertia, damping = balance
    residual = target - design @ np.asarray(balance)
    alpha_prediction = ((target - damping * omega) / inertia)
    alpha_error = alpha_prediction - alpha
    source_inertia = np.asarray([
        0.5 * float(data["metadata"][f"{WHEEL_PREFIX[wheel]}_wheel_mass_kg"])
        * float(data["metadata"][f"{WHEEL_PREFIX[wheel]}_radius_m"])**2
        for wheel in WHEELS])
    return {
        "fit_samples": fit_samples,
        "scored_wheel_samples": int(len(target)),
        "fitted_shared_wheel_inertia_kgm2": inertia,
        "fitted_shared_damping_nms": damping,
        "serialized_wheel_damping_rate": float(
            data["metadata"]["front_left_wheel_damping_rate"]),
        "solid_cylinder_inertia_from_serialized_mass_kgm2": float(
            np.mean(source_inertia)),
        "torque_balance_rmse_nm": float(np.sqrt(np.mean(residual**2))),
        "wheel_angular_acceleration_rmse_radps2": float(
            np.sqrt(np.mean(alpha_error**2))),
    }


def _metrics(data: dict[str, Any], prediction: np.ndarray) -> dict[str, Any]:
    valid = data["valid"]
    error = prediction - data["force_target"]
    speed = data["speed"]
    rows: dict[str, Any] = {}
    for label, mask in (("all", valid), ("0-3mps", valid & (speed < 3.0)),
                        ("3-6mps", valid & (speed >= 3.0) & (speed < 6.0)),
                        ("6-9mps", valid & (speed >= 6.0) & (speed < 9.0)),
                        ("9-12mps", valid & (speed >= 9.0) & (speed < 12.1))):
        residual = error[mask]
        rows[label] = {
            "samples": int(np.count_nonzero(mask)),
            "axis_rmse_n_unity_xyz": (
                np.sqrt(np.mean(residual**2, axis=0)).tolist()
                if len(residual) else None),
            "vector_rmse_n": (
                float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
                if len(residual) else None),
            "target_vector_rms_n": (
                float(np.sqrt(np.mean(np.sum(data["force_target"][mask]**2, axis=1))))
                if len(residual) else None),
        }
    return rows


def _fit(train: dict[str, Any], stride: int) -> dict[str, Any]:
    valid_rows = np.flatnonzero(train["valid"])
    selected = valid_rows[::stride]
    baseline_parameters = np.asarray([1.0, 1.0, 1.0, 1.0])

    def residual(parameters: np.ndarray) -> np.ndarray:
        prediction = _predict(train, parameters)
        return (prediction[selected] - train["force_target"][selected]).ravel()

    result = least_squares(
        residual, baseline_parameters,
        bounds=([0.1, 0.01, 0.1, 0.01], [4.0, 80.0, 4.0, 250.0]),
        loss="soft_l1", f_scale=5.0, max_nfev=250)
    return {
        "parameters": result.x,
        "optimizer": {
            "status": int(result.status),
            "message": result.message,
            "nfev": int(result.nfev),
            "sample_stride": stride,
            "fit_samples": int(len(selected)),
            "cost": float(result.cost),
            "active_bounds": [int(value) for value in result.active_mask],
        },
        "baseline_parameters": baseline_parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "live_runs/unity_exact_binary_audit_20261012/"
                        "tire_curve_force_balance_20261012.json")
    parser.add_argument("--derivative-window", type=int, default=31)
    parser.add_argument("--sample-stride", type=int, default=DEFAULT_SAMPLE_STRIDE)
    parser.add_argument("--slip-definition",
                        choices=("wheel_hit", "guide_equations"),
                        default="wheel_hit")
    args = parser.parse_args()
    train = _prepare(args.train, args.derivative_window,
                     args.slip_definition)
    validation = _prepare(args.validation, args.derivative_window,
                          args.slip_definition)
    fit = _fit(train, args.sample_stride)
    fitted = np.asarray(fit["parameters"])
    baseline = np.asarray(fit["baseline_parameters"])
    spin_fit = _wheel_spin_balance(train, fitted)
    fitted_spin = (spin_fit["fitted_shared_wheel_inertia_kgm2"],
                   spin_fit["fitted_shared_damping_nms"])
    output = {
        "status": "offline force-balance identification; not a recursive plant or runtime observer",
        "training_capture": str(args.train.relative_to(ROOT)),
        "validation_capture": str(args.validation.relative_to(ROOT)),
        "training_rows": train["sample_count"],
        "validation_rows": validation["sample_count"],
        "inputs": ("WheelHit slips, contact loads and directions, plus rigid-body state truth"
                   if args.slip_definition == "wheel_hit" else
                   "guide slip equations from truth rigid-body velocity, angular velocity, wheel RPM, contact points and tire directions; WheelHit slips are not used"),
        "slip_definition": {
            "source": args.slip_definition,
            "equations": ("captured Unity WheelHit.forwardSlip and sidewaysSlip"
                          if args.slip_definition == "wheel_hit" else
                          "Sx=(r*wheelOmega-vx)/vx; Sy=vy/abs(vx), with contact-point velocity vCOM + omega x (contact-COM)"),
            "guide_equation_domain": ("all valid rows have abs(contact longitudinal speed) > 1 m/s; lower-speed rows are excluded because the documented slip ratios are singular at zero" if args.slip_definition == "guide_equations" else None),
        },
        "label": "rigid-body force balance after subtracting gravity, captured normal forces, and Rigidbody linear drag",
        "derivative": {
            "method": "Savitzky-Golay filtered world COM velocity followed by time derivative and inverse body rotation",
            "window_samples": args.derivative_window,
            "window_seconds": train["derivative_window_s"],
            "sampling_interval_s": train["dt_s"],
        },
        "curve_model": {
            "form": "two cubic segments: zero-to-extremum Hermite with fitted origin tangent and zero extremum slope; extremum-to-asymptote smoothstep; constant after asymptote",
            "wheel_scales_and_landmarks": "exact captured WheelCollider curves; only longitudinal/lateral force scale and origin tangent are fitted",
            "parameter_names": ["longitudinal_scale", "longitudinal_zero_slip_tangent",
                                "lateral_scale", "lateral_zero_slip_tangent"],
            "training_fit": {
                key: float(value) for key, value in zip(
                    ("longitudinal_scale", "longitudinal_zero_slip_tangent",
                     "lateral_scale", "lateral_zero_slip_tangent"), fitted)
            },
            "optimizer": fit["optimizer"],
        },
        "training": {
            "baseline_serialized_curve": _metrics(train, _predict(train, baseline)),
            "fitted_curve": _metrics(train, _predict(train, fitted)),
        },
        "independent_run_validation": {
            "baseline_serialized_curve": _metrics(
                validation, _predict(validation, baseline)),
            "training_fit_transferred_unchanged": _metrics(
                validation, _predict(validation, fitted)),
        },
        "wheel_spin_balance": {
            "training_capture_fit": spin_fit,
            "independent_run_validation_with_frozen_parameters":
                _wheel_spin_balance(validation, fitted, fitted_spin),
        },
        "limitations": [
            "WheelHit slip and normal load are Unity-only labels, not permitted runtime inputs",
            "body acceleration is a smoothed offline derivative, not the live odometry signal",
            "this tests aggregate tire force balance, not wheel-wise force identifiability",
            "r11 private telemetry is truncated at the end and its public command replay had a request-boundary phase mismatch",
            "no command-only recursive rollout has been evaluated",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print("fit", output["curve_model"]["training_fit"])
    for split in ("training", "independent_run_validation"):
        print(split)
        for model, score in output[split].items():
            print(" ", model, "vector_rmse_N",
                  score["all"]["vector_rmse_n"], "9-12mps",
                  score["9-12mps"]["vector_rmse_n"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit and test the Unity wheel-spin equation on native 1 ms state steps.

The fit uses r07 only. r11 stays a whole-capture transfer check. WheelHit
slip/load and RPM are Unity-only identification labels, so this validates the
wheel-spin submodel with privileged tire inputs, not a sensor-only observer or
a closed-loop vehicle rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.vehicle_dynamics_learning.analyze_unity_3d_moment_balance import (
    DEFAULT_CAPTURES,
    WHEELS,
    _balance_data,
    _predict_balance,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIRE_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "tire_curve_force_balance_20261012.json")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "wheel_spin_discrete_1ms_v2_20261012.json")
SPEED_BINS = ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0),
              (6.0, 9.0), (9.0, 12.1))
STEERING_BINS = ((0.0, 0.15), (0.15, 0.30), (0.30, 0.45),
                 (0.45, 0.524))
WHEEL_RADIUS_SOLID_CYLINDER_INERTIA = 0.5 * 0.109 * 0.059**2


def _metrics(error: np.ndarray) -> dict[str, Any]:
    error = error[np.isfinite(error)]
    if error.size == 0:
        return {"samples": 0, "rmse_radps": None, "abs_p95_radps": None}
    absolute = np.abs(error)
    return {
        "samples": int(error.size),
        "bias_radps": float(np.mean(error)),
        "rmse_radps": float(np.sqrt(np.mean(error**2))),
        "abs_p95_radps": float(np.quantile(absolute, 0.95)),
        "abs_p99_radps": float(np.quantile(absolute, 0.99)),
        "abs_max_radps": float(np.max(absolute)),
    }


def _robust_shared_fit(data: dict[str, Any], wheel_force: np.ndarray,
                       dt_s: float) -> dict[str, Any]:
    inertia_columns = []
    damping_columns = []
    torque_targets = []
    fit_counts = {}
    for index, wheel in enumerate(WHEELS):
        state = data["wheels"][index]
        omega = state["omega"]
        dt = np.full(omega.size - 1, dt_s)
        next_omega = omega[1:]
        alpha = (next_omega - omega[:-1]) / dt
        motor = state["motor"][:-1]
        brake = state["brake"][:-1]
        omega_now = omega[:-1]
        target_torque = (motor - state["radius"] * wheel_force[:-1, index])
        valid = (np.isfinite(alpha) & np.isfinite(omega_now)
                 & np.isfinite(target_torque) & (brake == 0.0)
                 & ((np.abs(omega_now) > 15.0) | (np.abs(motor) > 0.0)))
        inertia_columns.append(alpha[valid])
        damping_columns.append(omega_now[valid])
        torque_targets.append(target_torque[valid])
        fit_counts[wheel] = int(np.sum(valid))

    design = np.column_stack((np.concatenate(inertia_columns),
                              np.concatenate(damping_columns)))
    target = np.concatenate(torque_targets)
    weights = np.ones(target.size, dtype=np.float64)
    for _ in range(12):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = target * np.sqrt(weights)
        parameters = np.linalg.lstsq(weighted_design, weighted_target,
                                     rcond=None)[0]
        residual = target - design @ parameters
        center = np.median(residual)
        scale = max(1.4826 * np.median(np.abs(residual - center)), 0.02)
        weights = np.minimum(1.0, 1.5 * scale
                             / np.maximum(np.abs(residual), 1.0e-12))
    if parameters[0] <= 0.0 or parameters[1] < 0.0:
        raise ValueError("wheel-spin fit returned nonphysical parameters")
    residual = target - design @ parameters
    return {
        "effective_inertia_kgm2": float(parameters[0]),
        "wheel_damping_nms": float(parameters[1]),
        "fit_samples_by_wheel": fit_counts,
        "fit_samples": int(target.size),
        "torque_residual_rmse_nm": float(np.sqrt(np.mean(residual**2))),
        "torque_residual_abs_p95_nm": float(
            np.quantile(np.abs(residual), 0.95)),
        "robust_scale_nm": float(scale),
    }


def _step_arrays(data: dict[str, Any], wheel_force: np.ndarray,
                 inertia: float, damping: float, dt_s: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.empty((len(data["speed"]), len(WHEELS)), dtype=np.float64)
    targets = np.empty_like(predictions)
    for index, wheel in enumerate(data["wheels"]):
        omega = wheel["omega"]
        free = omega + dt_s * (
            wheel["motor"] - wheel["radius"] * wheel_force[:, index]
            - damping * omega) / inertia
        brake_speed = wheel["brake"] * dt_s / inertia
        predictions[:, index] = np.where(
            free > 0.0, np.maximum(0.0, free - brake_speed),
            np.where(free < 0.0, np.minimum(0.0, free + brake_speed), 0.0))
        targets[:, index] = np.r_[omega[1:], np.nan]
    return predictions, targets


def _data_with_reconstructed_slip(data: dict[str, Any]
                                  ) -> dict[str, Any]:
    """Recompute current longitudinal slip from current body/wheel state."""
    reconstructed_wheels = []
    for wheel in data["wheels"]:
        contact_velocity = (data["velocity_body"]
                            + np.cross(data["omega_body"], wheel["lever"]))
        contact_speed = np.einsum("ij,ij->i", contact_velocity,
                                  wheel["forward"])
        surface_speed = wheel["omega"] * wheel["radius"]
        denominator = (np.maximum(np.abs(surface_speed),
                                  np.abs(contact_speed))
                       + 0.1)
        slip = (surface_speed - contact_speed) / denominator
        reconstructed_wheels.append({**wheel, "slip_x": slip})
    return {**data, "wheels": reconstructed_wheels}


def _group_metrics(errors: np.ndarray, speed: np.ndarray,
                   steering_rad: np.ndarray, slip: np.ndarray,
                   brake_active: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"all": _metrics(errors)}
    for name, bins, values in (
        ("speed_mps", SPEED_BINS, speed),
        ("abs_center_steering_rad", STEERING_BINS, steering_rad),
        ("abs_forward_slip", ((0.0, 0.05), (0.05, 0.15), (0.15, 0.5),
                               (0.5, 1.01)), np.abs(slip)),
    ):
        groups = {}
        for low, high in bins:
            mask = (values >= low) & (values < high)
            groups[f"{low:g}-{high:g}"] = _metrics(errors[mask])
        result[name] = groups
    result["actuation"] = {
        "motor_or_coast": _metrics(errors[~brake_active]),
        "brake": _metrics(errors[brake_active]),
    }
    return result


def _capture_context(path: Path, tire_parameters: np.ndarray,
                     derivative_window: int) -> dict[str, Any]:
    data = _balance_data(path, derivative_window, 1)
    reconstructed_data = _data_with_reconstructed_slip(data)
    _, _, wheel_force_oracle = _predict_balance(data, tire_parameters)
    _, _, wheel_force_reconstructed = _predict_balance(
        reconstructed_data, tire_parameters)
    columns = ["sim_time_s", "steering_command"]
    for wheel in WHEELS:
        columns.extend((f"{wheel}_rpm", f"{wheel}_forward_slip",
                        f"{wheel}_brake_torque_nm"))
    frame = pd.read_csv(path, comment="#", usecols=columns)
    dt = np.diff(frame["sim_time_s"].to_numpy(dtype=np.float64))
    fixed_dt = float(data["metadata"]["fixed_delta_time_s"])
    return {"path": path, "data": data,
            "reconstructed_data": reconstructed_data,
            "wheel_force_oracle": wheel_force_oracle,
            "wheel_force_reconstructed": wheel_force_reconstructed,
            "frame": frame, "dt": dt, "fixed_dt": fixed_dt}


def _evaluate_capture(context: dict[str, Any], inertia: float,
                      damping: float, slip_source: str) -> dict[str, Any]:
    path = context["path"]
    if slip_source == "oracle_wheelhit":
        data = context["data"]
        wheel_force = context["wheel_force_oracle"]
    elif slip_source == "state_reconstructed":
        data = context["reconstructed_data"]
        wheel_force = context["wheel_force_reconstructed"]
    else:
        raise ValueError(f"unknown slip source: {slip_source}")
    frame = context["frame"]
    dt = context["dt"]
    fixed_dt = context["fixed_dt"]
    predictions, targets = _step_arrays(data, wheel_force, inertia, damping,
                                         fixed_dt)
    errors = predictions[:-1] - targets[:-1]
    valid_dt = np.isclose(dt, fixed_dt, rtol=0.0, atol=1.0e-9)
    valid = np.broadcast_to(valid_dt[:, None], errors.shape).copy()
    valid &= np.isfinite(errors) & np.isfinite(targets[:-1])

    speed = data["speed"][:-1]
    steering = np.abs(frame["steering_command"].to_numpy(dtype=np.float64)[:-1])
    steering *= np.deg2rad(30.0)
    wheel_results = {}
    flattened_error = []
    flattened_speed = []
    flattened_steering = []
    flattened_slip = []
    flattened_brake = []
    for index, wheel in enumerate(WHEELS):
        mask = valid[:, index]
        wheel_error = errors[:, index]
        wheel_data = data["wheels"][index]
        brake_active = wheel_data["brake"][:-1] > 0.0
        wheel_results[wheel] = _group_metrics(
            wheel_error[mask], speed[mask], steering[mask],
            wheel_data["slip_x"][:-1][mask],
            brake_active[mask])
        flattened_error.append(wheel_error[mask])
        flattened_speed.append(speed[mask])
        flattened_steering.append(steering[mask])
        flattened_slip.append(wheel_data["slip_x"][:-1][mask])
        flattened_brake.append(brake_active[mask])

    aggregate = _group_metrics(
        np.concatenate(flattened_error), np.concatenate(flattened_speed),
        np.concatenate(flattened_steering), np.concatenate(flattened_slip),
        np.concatenate(flattened_brake))
    dt_error = np.abs(dt - fixed_dt)
    return {
        "capture": str(path.relative_to(ROOT)),
        "samples": int(len(frame)),
        "evaluated_steps": int(np.sum(valid_dt)),
        "fixed_dt_s": fixed_dt,
        "max_abs_dt_deviation_s": float(np.max(dt_error)),
        "state_step_error": aggregate,
        "slip_source": slip_source,
        "per_wheel": wheel_results,
        "provenance": (
            "WheelHit contact load and body truth are still used. For the "
            "state_reconstructed case, current WheelHit longitudinal slip is "
            "replaced by the fitted formula using current body state, wheel "
            "RPM and captured tire direction. This remains a one-step test, "
            "not a command-only recursive rollout."),
    }


def evaluate(train_path: Path, heldout_path: Path, tire_fit_path: Path,
             derivative_window: int) -> dict[str, Any]:
    force_fit = json.loads(tire_fit_path.read_text(encoding="utf-8"))[
        "curve_model"]["training_fit"]
    tire = np.asarray((force_fit["longitudinal_scale"],
                       force_fit["longitudinal_zero_slip_tangent"],
                       force_fit["lateral_scale"],
                       force_fit["lateral_zero_slip_tangent"]),
                      dtype=np.float64)
    train_context = _capture_context(train_path, tire, derivative_window)
    heldout_context = _capture_context(heldout_path, tire, derivative_window)
    fit_oracle = _robust_shared_fit(
        train_context["data"], train_context["wheel_force_oracle"],
        train_context["fixed_dt"])
    fit_reconstructed = _robust_shared_fit(
        train_context["reconstructed_data"],
        train_context["wheel_force_reconstructed"],
        train_context["fixed_dt"])
    heldout_fit_reconstructed = _robust_shared_fit(
        heldout_context["reconstructed_data"],
        heldout_context["wheel_force_reconstructed"],
        heldout_context["fixed_dt"])

    evaluations = {
        "oracle_slip_parameters": {
            "training_oracle_slip": _evaluate_capture(
                train_context, fit_oracle["effective_inertia_kgm2"],
                fit_oracle["wheel_damping_nms"], "oracle_wheelhit"),
            "heldout_oracle_slip": _evaluate_capture(
                heldout_context, fit_oracle["effective_inertia_kgm2"],
                fit_oracle["wheel_damping_nms"], "oracle_wheelhit"),
            "heldout_reconstructed_slip": _evaluate_capture(
                heldout_context, fit_oracle["effective_inertia_kgm2"],
                fit_oracle["wheel_damping_nms"], "state_reconstructed"),
        },
        "reconstructed_slip_parameters": {
            "training_reconstructed_slip": _evaluate_capture(
                train_context, fit_reconstructed["effective_inertia_kgm2"],
                fit_reconstructed["wheel_damping_nms"],
                "state_reconstructed"),
            "heldout_reconstructed_slip": _evaluate_capture(
                heldout_context, fit_reconstructed["effective_inertia_kgm2"],
                fit_reconstructed["wheel_damping_nms"],
                "state_reconstructed"),
            "heldout_oracle_slip": _evaluate_capture(
                heldout_context, fit_reconstructed["effective_inertia_kgm2"],
                fit_reconstructed["wheel_damping_nms"], "oracle_wheelhit"),
        },
    }

    baselines = {}
    for label, inertia in (
        ("solid_cylinder_from_serialized_mass_and_radius",
         WHEEL_RADIUS_SOLID_CYLINDER_INERTIA),
        ("earlier_smoothed_derivative_fit", 0.00027728625969045454),
    ):
        baselines[label] = {
            "r07_training": _evaluate_capture(
                train_context, inertia, 0.25,
                "state_reconstructed")["state_step_error"]["all"],
            "r11_heldout": _evaluate_capture(
                heldout_context, inertia, 0.25,
                "state_reconstructed")["state_step_error"]["all"],
        }
    return {
        "schema_version": 1,
        "status": "validated one-step wheel-spin submodel with reconstructed slip; not a full vehicle model",
        "training_capture": str(train_path.relative_to(ROOT)),
        "heldout_capture": str(heldout_path.relative_to(ROOT)),
        "tire_force_fit": str(tire_fit_path.relative_to(ROOT)),
        "tire_force_parameters_frozen_from_training": tire.tolist(),
        "fit_method": (
            "robust fit of motor torque - radius * fitted tire force = "
            "effective inertia * measured 1 ms wheel angular acceleration "
            "+ serialized wheel damping * angular speed; r07-only fit is "
            "reported separately for oracle and state-reconstructed slip"),
        "training_fit_oracle_slip": fit_oracle,
        "training_fit_reconstructed_slip": fit_reconstructed,
        "heldout_independent_fit_reconstructed_slip_diagnostic":
            heldout_fit_reconstructed,
        "evaluations": evaluations,
        "baseline_comparison": baselines,
        "limitations": [
            "WheelHit normal load remains a Unity-only label; the predictor does not yet calculate it recursively.",
            "Current body truth and captured tire basis direction are used to reconstruct slip; they must be supplied from odometry state and steering geometry at runtime.",
            "This tests one 1 ms wheel-speed step only, not body velocity, yaw, pose, or a long-horizon rollout.",
            "r11 has little coverage below 3 m/s and does not reach the highest steering bin; transfer there remains unverified.",
            "The fitted inertia is an effective WheelCollider parameter and is not the solid-cylinder inertia implied by serialized mass/radius.",
        ],
        "next_required_model_step": (
            "Replace oracle WheelHit load with the validated suspension-geometry "
            "estimate, construct tire directions from steering geometry, and "
            "validate recursive wheel/body state on held-out capture."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_CAPTURES[0])
    parser.add_argument("--heldout", type=Path, default=DEFAULT_CAPTURES[1])
    parser.add_argument("--tire-fit", type=Path, default=DEFAULT_TIRE_FIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--derivative-window", type=int, default=31)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = evaluate(args.train, args.heldout, args.tire_fit,
                      args.derivative_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for name in ("training_fit_oracle_slip",
                 "training_fit_reconstructed_slip"):
        fit = report[name]
        print(name, "effective inertia", fit["effective_inertia_kgm2"],
              "damping", fit["wheel_damping_nms"])
    for name in ("oracle_slip_parameters", "reconstructed_slip_parameters"):
        score = report["evaluations"][name]
        heldout_key = ("heldout_oracle_slip" if name == "oracle_slip_parameters"
                       else "heldout_reconstructed_slip")
        metrics = score[heldout_key]["state_step_error"]["all"]
        print(name, "r11 RMSE", metrics["rmse_radps"],
              "p95", metrics["abs_p95_radps"])
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

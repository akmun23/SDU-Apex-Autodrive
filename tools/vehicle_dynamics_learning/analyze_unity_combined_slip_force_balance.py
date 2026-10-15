#!/usr/bin/env python3
"""Test a PhysX-style combined-slip constraint against Unity force balance.

This is an offline mechanics hypothesis, not an assertion that Unity's native
WheelCollider uses the PhysX vehicle tire shader. The independent training and
held-out captures decide whether a shared friction-ellipse cap is useful.
WheelHit slip/load/contact frames are privileged scoring inputs only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from tools.vehicle_dynamics_learning.analyze_unity_3d_moment_balance import (
    DEFAULT_CAPTURES,
    DEFAULT_SAMPLE_STRIDE,
    WHEELS,
    _balance_data,
    _curve,
)
from tools.vehicle_dynamics_learning.evaluate_unity_wheel_spin_state_inputs import (
    _total_contact_balance,
)
from tools.vehicle_dynamics_learning.replay_unity_source_mechanics import ROOT


DEFAULT_FORCE_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "tire_curve_force_balance_20261012.json")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "combined_slip_force_balance_20261013.json")
PREFIX = {"fl": "front_left", "fr": "front_right",
          "rl": "rear_left", "rr": "rear_right"}


def _tire_forces(data: dict[str, Any], parameters: np.ndarray,
                 cap_scale: float | None) -> tuple[np.ndarray, np.ndarray]:
    scale_x, tangent_x, scale_y, tangent_y = parameters
    count = len(data["speed"])
    force = np.zeros_like(data["force_target_total"])
    torque = np.zeros_like(data["torque_target_total"])
    for index, wheel_name in enumerate(WHEELS):
        wheel = data["wheels"][index]
        prefix = PREFIX[wheel_name]
        curve_x = np.asarray([float(value) for value in data["metadata"][
            f"{prefix}_forward_curve"].split(",")])
        curve_y = np.asarray([float(value) for value in data["metadata"][
            f"{prefix}_sideways_curve"].split(",")])
        fx = (scale_x * wheel["load"]
              * _curve(wheel["slip_x"], curve_x, tangent_x))
        fy = (-scale_y * wheel["load"]
              * _curve(wheel["slip_y"], curve_y, tangent_y))
        if cap_scale is not None:
            mu_x = curve_x[1] * curve_x[4]
            mu_y = curve_y[1] * curve_y[4]
            safe_load = np.maximum(wheel["load"], 1.0e-9)
            utilization = np.sqrt(
                (fx / (safe_load * mu_x))**2
                + (fy / (safe_load * mu_y))**2)
            factor = np.minimum(
                1.0, cap_scale / np.maximum(utilization, 1.0e-12))
            fx *= factor
            fy *= factor
        tire_force = (fx[:, None] * wheel["forward"]
                      + fy[:, None] * wheel["sideways"])
        normal_force = wheel["load"][:, None] * wheel["normal"]
        total = tire_force + normal_force
        force += total
        torque += np.cross(wheel["lever"], total)
    return force, torque


def _metrics(data: dict[str, Any], force: np.ndarray,
             torque: np.ndarray) -> dict[str, Any]:
    rows = data["rows"]
    force_error = force[rows] - data["force_target_total"][rows]
    torque_error = torque[rows] - data["torque_target_total"][rows]
    speed = data["speed"][rows]
    steering = np.max(np.abs(np.column_stack([
        wheel["steering_rad"][rows] for wheel in data["wheels"]])), axis=1)

    def score(error: np.ndarray) -> dict[str, Any]:
        norm = np.linalg.norm(error, axis=1)
        return {
            "axis_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "vector_rmse": float(np.sqrt(np.mean(norm**2))),
            "vector_abs_p95": float(np.quantile(norm, 0.95)),
        }

    strata = {}
    for name, mask in (
            ("speed_1_3", (speed >= 1.0) & (speed < 3.0)),
            ("speed_3_6", (speed >= 3.0) & (speed < 6.0)),
            ("speed_6_9", (speed >= 6.0) & (speed < 9.0)),
            ("speed_9_12", (speed >= 9.0) & (speed <= 12.1)),
            ("steer_lt_0p15", steering < 0.15),
            ("steer_0p15_0p30", (steering >= 0.15) & (steering < 0.30)),
            ("steer_ge_0p30", steering >= 0.30)):
        strata[name] = {
            "samples": int(np.count_nonzero(mask)),
            "force_vector_rmse_n": (
                score(force_error[mask])["vector_rmse"]
                if np.any(mask) else None),
            "yaw_moment_rmse_nm": (
                float(np.sqrt(np.mean(torque_error[mask, 1]**2)))
                if np.any(mask) else None),
        }
    return {"samples": int(len(rows)),
            "force_n": score(force_error),
            "moment_nm": score(torque_error),
            "yaw_moment_rmse_nm": float(np.sqrt(
                np.mean(torque_error[:, 1]**2))),
            "strata": strata}


def _fit_cap(training: dict[str, Any], parameters: np.ndarray
             ) -> dict[str, Any]:
    rows = training["rows"][::4]
    force_scale = np.maximum(np.sqrt(np.mean(
        training["force_target_total"][rows]**2, axis=0)), 1.0)
    torque_scale = np.maximum(np.sqrt(np.mean(
        training["torque_target_total"][rows]**2, axis=0)), 0.02)

    def residual(candidate: np.ndarray) -> np.ndarray:
        force, torque = _tire_forces(training, parameters, float(candidate[0]))
        return np.concatenate((
            ((force[rows] - training["force_target_total"][rows])
             / force_scale).ravel(),
            ((torque[rows] - training["torque_target_total"][rows])
             / torque_scale).ravel(),
        ))

    result = least_squares(residual, np.asarray([1.0]),
                           bounds=([0.1], [2.0]), loss="soft_l1",
                           f_scale=1.0, max_nfev=40)
    return {
        "cap_scale": float(result.x[0]),
        "training_only": True,
        "fit_samples": int(rows.size),
        "fit_sample_stride_from_physics_valid_rows": 4,
        "normalized_cost": float(result.cost),
        "optimizer_status": int(result.status),
        "optimizer_message": result.message,
    }


def _load_parameters(path: Path) -> np.ndarray:
    report = json.loads(path.read_text(encoding="utf-8"))
    fit = report["curve_model"]["training_fit"]
    return np.asarray((fit["longitudinal_scale"],
                       fit["longitudinal_zero_slip_tangent"],
                       fit["lateral_scale"],
                       fit["lateral_zero_slip_tangent"]), dtype=np.float64)


def evaluate(train_path: Path, heldout_path: Path, force_fit_path: Path,
             output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    parameters = _load_parameters(force_fit_path)
    train = _balance_data(train_path, derivative_window=31,
                          sample_stride=DEFAULT_SAMPLE_STRIDE)
    heldout = _balance_data(heldout_path, derivative_window=31,
                            sample_stride=DEFAULT_SAMPLE_STRIDE)
    fitted_cap = _fit_cap(train, parameters)
    models = {
        "independent_source_curves": None,
        "fixed_friction_ellipse_cap_1": 1.0,
        "r07_fitted_shared_ellipse_cap": fitted_cap["cap_scale"],
    }
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "offline physics hypothesis; no runtime, simulator, or MPC changes",
        "training_capture": str(train_path.resolve().relative_to(ROOT)),
        "heldout_capture": str(heldout_path.resolve().relative_to(ROOT)),
        "capture_coverage": {
            "training_valid_force_balance_rows": int(train["rows"].size),
            "training_reset_discontinuities": int(
                train["reset_discontinuity_count"]),
            "training_reset_excluded_derivative_rows": int(
                train["reset_excluded_rows"]),
            "heldout_valid_force_balance_rows": int(heldout["rows"].size),
            "heldout_reset_discontinuities": int(
                heldout["reset_discontinuity_count"]),
            "heldout_reset_excluded_derivative_rows": int(
                heldout["reset_excluded_rows"]),
            "derivative_window_s": float(heldout["derivative_window_s"]),
        },
        "force_curve_parameters_frozen_from_r07": parameters.tolist(),
        "model_hypothesis": (
            "Scale each wheel's independent source-curve forces by "
            "min(1, cap_scale/sqrt((Fx/(mu_x*Fz))^2 + "
            "(Fy/(mu_y*Fz))^2)); mu uses the serialized curve peak. "
            "This is a testable friction-ellipse hypothesis, not claimed "
            "to be Unity WheelCollider's implementation."),
        "r07_training_only_fitted_cap": fitted_cap,
        "results": {},
    }
    for split, data in (("training", train), ("heldout", heldout)):
        output["results"][split] = {}
        for name, cap in models.items():
            force, torque = _tire_forces(data, parameters, cap)
            output["results"][split][name] = _metrics(data, force, torque)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_CAPTURES[0])
    parser.add_argument("--heldout", type=Path, default=DEFAULT_CAPTURES[1])
    parser.add_argument("--force-fit", type=Path, default=DEFAULT_FORCE_FIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate(args.train, args.heldout, args.force_fit, args.output)
    print("r07-fitted cap", report["r07_training_only_fitted_cap"])
    for split, models in report["results"].items():
        print(split)
        for name, metrics in models.items():
            print(" ", name,
                  "force_N", round(metrics["force_n"]["vector_rmse"], 4),
                  "yaw_Nm", round(metrics["yaw_moment_rmse_nm"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

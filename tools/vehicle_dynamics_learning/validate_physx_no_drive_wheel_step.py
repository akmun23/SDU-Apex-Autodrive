#!/usr/bin/env python3
"""One-step, no-fit PhysX wheel-speed replay from Unity source captures.

Recorded slip and WheelHit normal force are used only as current-step inputs
to validate the source tire callback and wheel integrator. This is not a
future-state predictor because those measurements are not available to an
open-loop rollout. Below the 5 m/s substep threshold, recorded slip/load are
held across the five internal substeps; that subset is diagnostic only. The
at-or-above-5 m/s subset uses the configured single substep per 1 ms capture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3.f1tenth_planning.autodrive_mintime.unity_wheel_model import (
    UnityWheelModel,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    GRAVITY_MPS2,
    WHEEL_MASS_KG,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
WHEELS = ("fl", "fr", "rl", "rr")
FIXED_DT_S = 0.001
TIRE_LOAD_FILTER = (0.0, 0.2308, 3.0, 3.0)


def _metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, separator, value = line[2:].strip().partition("=")
            if separator:
                metadata[key] = value
    return metadata


def _curve_response(slip: np.ndarray, curve: tuple[float, ...]) -> np.ndarray:
    extremum_slip, extremum_value, asymptote_slip, asymptote_value, stiffness = curve
    magnitude = np.abs(slip)
    t = magnitude / extremum_slip
    first = extremum_value * (t + t**2 - t**3)
    decay_t = (magnitude - extremum_slip) / (asymptote_slip - extremum_slip)
    smooth = 3.0 * decay_t**2 - 2.0 * decay_t**3
    second = extremum_value + (asymptote_value - extremum_value) * smooth
    response = np.where(
        magnitude < extremum_slip, first,
        np.where(magnitude < asymptote_slip, second, asymptote_value))
    return stiffness * np.sign(slip) * response


def _filtered_tire_load(suspension_force_n: np.ndarray,
                        sprung_mass_kg: float) -> np.ndarray:
    x_min, y_min, x_max, y_max = TIRE_LOAD_FILTER
    rest_load = GRAVITY_MPS2 * (sprung_mass_kg + WHEEL_MASS_KG)
    normalized = suspension_force_n / rest_load
    filtered = np.where(
        normalized <= x_min, y_min,
        np.where(normalized >= x_max, y_max,
                 y_min + (normalized - x_min) * (y_max - y_min)
                 / (x_max - x_min)))
    return filtered * rest_load


def _source_longitudinal_force(long_slip: np.ndarray,
                               lateral_slip: np.ndarray,
                               tire_load_n: np.ndarray,
                               forward_curve: tuple[float, ...],
                               lateral_curve: tuple[float, ...]) -> np.ndarray:
    long_response = _curve_response(long_slip, forward_curve)
    lateral_response = _curve_response(
        lateral_slip, lateral_curve)
    utilization_sq = (
        (long_response / forward_curve[1])**2
        + (lateral_response / lateral_curve[1])**2)
    combined_scale = 1.0 / np.sqrt(np.maximum(1.0, utilization_sq))
    return tire_load_n * long_response * combined_scale


def validate_capture(path: Path) -> dict[str, object]:
    metadata = _metadata(path)
    fixed_dt = float(metadata.get("fixed_delta_time_s", "nan"))
    if not np.isclose(fixed_dt, FIXED_DT_S, rtol=0.0, atol=1e-8):
        raise ValueError(f"expected 1 ms Unity capture, got {fixed_dt}: {path}")
    model = UnityWheelModel.from_source_capture(path)
    columns = ["sim_time_s", "vel_body_z_mps", "throttle_command",
               "steering_command"]
    for wheel in WHEELS:
        columns.extend((f"{wheel}_rpm", f"{wheel}_motor_torque_nm",
                        f"{wheel}_brake_torque_nm", f"{wheel}_grounded",
                        f"{wheel}_forward_slip", f"{wheel}_sideways_slip",
                        f"{wheel}_contact_force_n"))
    frame = pd.read_csv(path, comment="#", usecols=columns)
    time = frame["sim_time_s"].to_numpy(dtype=np.float64)
    forward_speed = frame["vel_body_z_mps"].to_numpy(dtype=np.float64)
    dt = np.diff(time)
    valid_dt = np.isclose(dt, FIXED_DT_S, rtol=0.0, atol=2e-7)
    substeps = np.where(np.abs(forward_speed[:-1]) < 5.0, 5, 1)
    speed_bands = {
        "below_5mps_five_substeps": substeps == 5,
        "at_or_above_5mps_one_substep": substeps == 1,
    }
    predicted_by_wheel: dict[str, np.ndarray] = {}
    actual_by_wheel: dict[str, np.ndarray] = {}
    valid_by_wheel: dict[str, np.ndarray] = {}
    residual_by_wheel: dict[str, np.ndarray] = {}

    for index, wheel in enumerate(WHEELS):
        rpm = frame[f"{wheel}_rpm"].to_numpy(dtype=np.float64)
        omega = rpm * (2.0 * np.pi / 60.0)
        drive = frame[f"{wheel}_motor_torque_nm"].to_numpy(dtype=np.float64)
        brake = frame[f"{wheel}_brake_torque_nm"].to_numpy(dtype=np.float64)
        grounded = frame[f"{wheel}_grounded"].to_numpy(dtype=np.float64) > 0.5
        long_slip = frame[f"{wheel}_forward_slip"].to_numpy(dtype=np.float64)
        lateral_slip = frame[f"{wheel}_sideways_slip"].to_numpy(dtype=np.float64)
        contact_force = frame[f"{wheel}_contact_force_n"].to_numpy(
            dtype=np.float64)
        tire_load = _filtered_tire_load(contact_force, model.sprung_mass_kg[index])
        force_long = _source_longitudinal_force(
            long_slip, lateral_slip, tire_load, model.forward_curve[index],
            model.lateral_curve[index])
        tire_torque = -force_long * model.wheel_radius_m
        predicted = omega[:-1].astype(np.float32)
        drive_step = drive[:-1].astype(np.float32)
        brake_step = brake[:-1].astype(np.float32)
        tire_step = tire_torque[:-1].astype(np.float32)
        inverse_inertia = np.float32(
            1.0 / model.wheel_inertia_kgm2)
        damping = np.float32(model.wheel_damping_nms)
        brake_applied = brake_step != np.float32(0.0)
        for substep in range(5):
            active = substeps > substep
            sub_dt = (np.float32(FIXED_DT_S) /
                      substeps.astype(np.float32))
            dt_over_inertia = sub_dt * inverse_inertia
            signed_brake = -np.sign(predicted).astype(np.float32) * brake_step
            numerator = predicted + dt_over_inertia * (
                tire_step + drive_step + signed_brake)
            denominator = np.float32(1.0) + damping * dt_over_inertia
            candidate = (numerator / denominator).astype(np.float32)
            crossed = brake_applied & (predicted * candidate <= np.float32(0.0))
            candidate = np.where(crossed, np.float32(0.0), candidate)
            predicted = np.where(active, candidate, predicted).astype(np.float32)
        predicted = predicted.astype(np.float64)
        predicted_by_wheel[wheel] = predicted
        actual_by_wheel[wheel] = omega[1:]
        valid_by_wheel[wheel] = (
            valid_dt & grounded[:-1] & grounded[1:]
            & np.isfinite(tire_torque[:-1]))
        residual_by_wheel[wheel] = predicted - omega[1:]

    results: dict[str, object] = {
        "capture": str(path.resolve().relative_to(ROOT)),
        "rows": len(frame),
        "invalid_time_steps": int(len(dt) - np.count_nonzero(valid_dt)),
        "substep_counts": {str(count): int(np.count_nonzero(substeps == count))
                           for count in (1, 5)},
        "method_limit": (
            "Captured slip and suspension force are current-step validation "
            "inputs; held constant across internal substeps. Only the >=5 "
            "m/s one-substep band avoids within-step force holding. The "
            "UnityPlayer VehicleConfig allocates/configures one tire and one "
            "surface type; PhysX initializes its sole friction-pair value "
            "to 1.0."),
        "bands": {},
    }
    bands: dict[str, object] = {}
    for band_name, band_mask in speed_bands.items():
        errors = []
        per_wheel: dict[str, object] = {}
        largest: tuple[float, str, int] | None = None
        for wheel in WHEELS:
            mask = valid_by_wheel[wheel] & band_mask
            wheel_error = residual_by_wheel[wheel][mask]
            errors.extend(wheel_error.tolist())
            if len(wheel_error):
                per_wheel[wheel] = {
                    "samples": int(len(wheel_error)),
                    "omega_rmse_radps": float(np.sqrt(np.mean(wheel_error**2))),
                    "omega_p95_abs_radps": float(
                        np.quantile(np.abs(wheel_error), 0.95)),
                    "omega_max_abs_radps": float(np.max(np.abs(wheel_error))),
                }
                valid_indices = np.flatnonzero(valid_by_wheel[wheel] & band_mask)
                local = int(np.argmax(np.abs(wheel_error)))
                candidate = (float(abs(wheel_error[local]),), wheel,
                             int(valid_indices[local]))
                if largest is None or candidate[0] > largest[0]:
                    largest = candidate
        array = np.asarray(errors, dtype=np.float64)
        if len(array):
            band_report: dict[str, object] = {
                "wheel_samples": int(len(array)),
                "omega_rmse_radps": float(np.sqrt(np.mean(array**2))),
                "omega_p95_abs_radps": float(np.quantile(np.abs(array), 0.95)),
                "omega_median_bias_radps": float(np.median(array)),
                "omega_max_abs_radps": float(np.max(np.abs(array))),
                "rpm_rmse": float(np.sqrt(np.mean(array**2)) * 60.0 / (2.0 * np.pi)),
                "by_wheel": per_wheel,
            }
            if largest is not None:
                abs_error, wheel, row = largest
                band_report["largest_error_sample"] = {
                    "wheel": wheel,
                    "row": row,
                    "time_s": float(time[row]),
                    "forward_speed_mps": float(forward_speed[row]),
                    "throttle_command": float(
                        frame["throttle_command"].iloc[row])
                        if "throttle_command" in frame else None,
                    "steering_command": float(
                        frame["steering_command"].iloc[row])
                        if "steering_command" in frame else None,
                    "motor_torque_nm": float(frame[f"{wheel}_motor_torque_nm"].iloc[row]),
                    "brake_torque_nm": float(frame[f"{wheel}_brake_torque_nm"].iloc[row]),
                    "wheel_omega_start_radps": float(
                        frame[f"{wheel}_rpm"].iloc[row] * 2.0 * np.pi / 60.0),
                    "predicted_wheel_omega_radps": float(
                        predicted_by_wheel[wheel][row]),
                    "observed_wheel_omega_next_radps": float(
                        actual_by_wheel[wheel][row]),
                    "forward_slip": float(
                        frame[f"{wheel}_forward_slip"].iloc[row]),
                    "sideways_slip": float(
                        frame[f"{wheel}_sideways_slip"].iloc[row]),
                    "absolute_error_radps": abs_error,
                }
            bands[band_name] = band_report
        else:
            bands[band_name] = {"wheel_samples": 0}
    results["bands"] = bands
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", type=Path,
                        default=(DEFAULT_CAPTURE,))
    args = parser.parse_args()
    for path in args.captures:
        print(validate_capture(path))


if __name__ == "__main__":
    main()

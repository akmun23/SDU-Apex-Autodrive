#!/usr/bin/env python3
"""Audit Unity wheel rotation and forward-slip observability.

The report separates settled plateaus from torque/brake transients. It uses
the WheelCollider radius and the recorded RPM/``WheelHit.forwardSlip``; it
does not fit a tire peak or hide wheel inertia inside a force coefficient.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from structured_vehicle_plant import (
    GRAVITY_MPS2,
    UNITY_FORWARD_ASYMPTOTE_SLIP,
    UNITY_FORWARD_ASYMPTOTE_VALUE,
    UNITY_FORWARD_EXTREMUM_SLIP,
    UNITY_FORWARD_EXTREMUM_VALUE,
    UNITY_FORWARD_STIFFNESS,
    UNITY_WHEEL_DAMPING_RATE,
    UNITY_WHEEL_MASS_KG,
    unity_forward_slip,
    unity_wheel_friction_value,
)


def _float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _stats(values: np.ndarray | list[float]) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "p05": None, "p50": None,
                "p95": None, "min": None, "max": None}
    p05, p50, p95 = np.percentile(finite, [5.0, 50.0, 95.0])
    return {"count": int(finite.size), "mean": float(np.mean(finite)),
            "p05": float(p05), "p50": float(p50), "p95": float(p95),
            "min": float(np.min(finite)), "max": float(np.max(finite))}


def _read_rows(traces: list[Path]) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    for path in traces:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            current = list(reader.fieldnames or [])
            if not fields:
                fields = current
            elif current != fields:
                raise ValueError(f"trace header mismatch: {path}")
            rows.extend(reader)
    if not rows:
        raise ValueError("trace contains no rows")
    return rows, fields


def _derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    positive_dt = np.diff(times)
    median_dt = float(np.median(positive_dt[positive_dt > 0.0]))
    if len(values) >= 21:
        return savgol_filter(values, 21, 2, deriv=1,
                             delta=median_dt, mode="interp")
    return np.gradient(values, times)


def _fit_wheel_rotational_transition_screen(
        times: np.ndarray, body_speed: np.ndarray, throttle: np.ndarray,
        steering: np.ndarray,
        motor_torque: np.ndarray, brake_torque: np.ndarray,
        rpm: np.ndarray, normal_load: np.ndarray,
        forward_slip: np.ndarray, radius: float,
        include_steering: bool = False) -> dict[str, Any]:
    """Fit the explicit WheelCollider rotational balance around drive steps.

    The balance is only an offline observability screen:

      I * domega + c * omega = Tmotor - sign(omega) * Tbrake
                              - radius * Fforward

    ``Fforward`` uses the recorded WheelHit force and the serialized forward
    curve.  Positive-drive steps are fit separately from throttle-down steps;
    zero throttle is an active CAWB brake command and is therefore not
    treated as a continuous wheel-inertia experiment.  The result is never
    promoted to the eight-state plant by this audit.
    """
    if len(times) < 3:
        return {"status": "insufficient_rows", "runtime_promoted": False}
    dt = np.diff(times)
    valid_dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if valid_dt.size == 0:
        return {"status": "invalid_timestamps", "runtime_promoted": False}
    fixed_dt = float(np.median(valid_dt))
    omega = rpm * (2.0 * math.pi / 60.0)
    domega = np.diff(omega) / dt
    sign = np.sign(omega[:-1])
    sign[sign == 0.0] = 1.0
    curve_force = np.zeros(len(times), dtype=float)
    for index, value in enumerate(forward_slip):
        if math.isfinite(value) and math.isfinite(normal_load[index]):
            curve_force[index] = normal_load[index] * unity_wheel_friction_value(
                value, UNITY_FORWARD_EXTREMUM_SLIP,
                UNITY_FORWARD_EXTREMUM_VALUE, UNITY_FORWARD_ASYMPTOTE_SLIP,
                UNITY_FORWARD_ASYMPTOTE_VALUE, UNITY_FORWARD_STIFFNESS)
    rhs = (motor_torque[:-1] - sign * brake_torque[:-1] -
           radius * curve_force[:-1])
    steering_condition = (np.ones(len(domega), dtype=bool)
                          if include_steering else
                          (np.abs(steering[:-1]) < 1.0e-6))
    finite = (np.isfinite(domega) & np.isfinite(rhs) &
              np.isfinite(omega[:-1]) & np.isfinite(throttle[:-1]) &
              np.isfinite(steering[:-1]) &
              np.isfinite(normal_load[:-1]) & (normal_load[:-1] > 1.0) &
              (brake_torque[:-1] < 1.0) &
              steering_condition &
              (np.abs(domega) > 100.0))

    changes = np.flatnonzero(np.abs(np.diff(throttle)) > 1.0e-8) + 1
    events: list[dict[str, Any]] = []
    drive_events: list[tuple[float, np.ndarray]] = []
    event_windows: list[tuple[dict[str, Any], np.ndarray]] = []
    training_indices: list[int] = []
    holdout_indices: list[int] = []
    for change_index in changes:
        old = float(throttle[change_index - 1])
        new = float(throttle[change_index])
        # The first sample after a command change is the first causal sample
        # in this trace. Keep a short, fixed window and identify it by the
        # applied command, not by a fitted residual.
        event_mask = (np.arange(len(domega)) >= change_index) & (
            np.arange(len(domega)) < min(change_index + 20, len(domega)))
        selected = np.flatnonzero(finite & event_mask)
        if selected.size < 3:
            continue
        is_positive_drive = new > old and new > 0.0 and old >= 0.0
        if is_positive_drive:
            # The final positive-drive step is held out. This leaves three
            # distinct throttle/speed transitions for identification and one
            # separate transition for validation.
            if len([event for event in events if event["kind"] ==
                    "positive_drive_train"]) < 3:
                kind = "positive_drive_train"
                training_indices.extend(selected.tolist())
            else:
                kind = "positive_drive_holdout"
                holdout_indices.extend(selected.tolist())
        if is_positive_drive or (new > 0.0 and old > 0.0):
            drive_events.append((float(body_speed[change_index]), selected))
        elif new > 0.0 and old > 0.0:
            kind = "positive_drive_downstep_diagnostic"
        else:
            kind = "active_brake_or_start_diagnostic"
        event = {
            "time_s": float(times[change_index]),
            "body_speed_mps": float(body_speed[change_index]),
            "old_throttle": old,
            "new_throttle": new,
            "kind": kind,
            "samples": int(selected.size),
            "used_for_fit": kind == "positive_drive_train",
        }
        events.append(event)
        event_windows.append((event, selected))

    def fit(indices: list[int]) -> dict[str, Any]:
        if len(indices) < 3:
            return {"samples": len(indices), "identified": False}
        unique = np.asarray(sorted(set(indices)), dtype=int)
        design = np.column_stack((domega[unique], omega[:-1][unique]))
        target = rhs[unique]
        coefficients, _, rank, singular = np.linalg.lstsq(
            design, target, rcond=None)
        residual = target - design @ coefficients
        return {
            "samples": int(target.size),
            "identified": bool(rank == design.shape[1]),
            "effective_inertia_kgm2": float(coefficients[0]),
            "rotational_damping_nms": float(coefficients[1]),
            "rmse_torque_balance_nm": float(np.sqrt(np.mean(residual * residual))),
            "mae_torque_balance_nm": float(np.mean(np.abs(residual))),
            "residual_p05_p50_p95_nm": [
                float(value) for value in np.percentile(residual, [5.0, 50.0, 95.0])
            ],
            "design_singular_values": [float(value) for value in singular],
        }

    for event, indices in event_windows:
        if event["kind"] in {
                "positive_drive_train", "positive_drive_holdout",
                "positive_drive_downstep_diagnostic"}:
            event["local_balance_fit"] = fit(indices.tolist())

    fitted = fit(training_indices)
    holdout = fit(holdout_indices)
    validation: dict[str, Any] = {
        "samples": len(holdout_indices),
        "identified_from_training": bool(fitted.get("identified", False)),
        "predicted_holdout_rmse_torque_balance_nm": None,
        "predicted_holdout_mae_torque_balance_nm": None,
    }
    if fitted.get("identified") and holdout_indices:
        unique = np.asarray(sorted(set(holdout_indices)), dtype=int)
        prediction = (float(fitted["effective_inertia_kgm2"]) * domega[unique] +
                      float(fitted["rotational_damping_nms"]) * omega[:-1][unique])
        residual = rhs[unique] - prediction
        validation["predicted_holdout_rmse_torque_balance_nm"] = float(
            np.sqrt(np.mean(residual * residual)))
        validation["predicted_holdout_mae_torque_balance_nm"] = float(
            np.mean(np.abs(residual)))

    def fit_speed_regime(lower: float, upper: float) -> dict[str, Any]:
        candidates = [(speed_value, indices) for speed_value, indices
                      in drive_events if lower <= speed_value < upper]
        if not candidates:
            return {"events": 0, "identified": False}
        # Hold out the last event in each regime. A regime fit is only
        # considered identifiable when at least two independent transitions
        # remain for training.
        train = [index for _, indices in candidates[:-1]
                 for index in indices.tolist()]
        holdout_event = candidates[-1]
        holdout = holdout_event[1].tolist()
        train_fit = fit(train)
        prediction: dict[str, Any] = {
            "samples": len(holdout),
            "event_body_speed_mps": holdout_event[0],
            "predicted_rmse_torque_balance_nm": None,
            "predicted_mae_torque_balance_nm": None,
        }
        if train_fit.get("identified") and holdout:
            unique = np.asarray(sorted(set(holdout)), dtype=int)
            predicted = (float(train_fit["effective_inertia_kgm2"]) *
                         domega[unique] +
                         float(train_fit["rotational_damping_nms"]) *
                         omega[:-1][unique])
            residual = rhs[unique] - predicted
            prediction["predicted_rmse_torque_balance_nm"] = float(
                np.sqrt(np.mean(residual * residual)))
            prediction["predicted_mae_torque_balance_nm"] = float(
                np.mean(np.abs(residual)))
        return {
            "speed_interval_mps": [lower, upper],
            "events": len(candidates),
            "training_event_speeds_mps": [value for value, _ in candidates[:-1]],
            "holdout_event_speed_mps": holdout_event[0],
            "training_fit": train_fit,
            "holdout_prediction": prediction,
        }

    speed_regimes = [
        fit_speed_regime(0.0, 6.0),
        fit_speed_regime(6.0, 12.0),
        fit_speed_regime(12.0, 16.000001),
    ]

    return {
        "status": "offline_wheel_rotational_balance_screen",
        "fixed_dt_s": fixed_dt,
        "equation": (
            "I * d(angular_velocity)/dt + c * angular_velocity = "
            "motor_torque - sign(angular_velocity)*brake_torque - "
            "collider_radius * WheelHit.forward_curve_force"
        ),
        "forward_force_screen": {
            "normal_load_source": "recorded WheelHit.force",
            "slip_source": "recorded WheelHit.forwardSlip",
            "serialized_curve_used": True,
            "not_a_fitted_tire_parameter": True,
        },
        "steering_selection": (
            "included_in_transition_screen" if include_steering else
            "near_zero_applied_steering_only"),
        "configured_reference": {
            "wheel_mass_kg": UNITY_WHEEL_MASS_KG,
            "wheel_radius_m": radius,
            "mass_times_radius_squared_kgm2": UNITY_WHEEL_MASS_KG * radius * radius,
            "solid_disk_inertia_kgm2": 0.5 * UNITY_WHEEL_MASS_KG * radius * radius,
            "wheel_damping_rate": UNITY_WHEEL_DAMPING_RATE,
        },
        "events": events,
        "training_fit": fitted,
        "holdout_window_fit_diagnostic": holdout,
        "holdout_prediction_using_training_fit": validation,
        "speed_regime_fits": speed_regimes,
        "interpretation": (
            "positive-drive transitions identify a mechanical wheel-state "
            "screen; zero-throttle CAWB braking and positive throttle "
            "downsteps are retained as hybrid/downstep diagnostics. "
            "No value is promoted to the runtime plant."
        ),
        "runtime_promoted": False,
    }


def analyze(traces: list[Path], static: Path, output: Path,
            include_steering: bool = False) -> dict[str, Any]:
    rows, fields = _read_rows(traces)
    metadata = json.loads(static.read_text(encoding="utf-8"))
    vehicle = metadata["vehicle"]
    rigid_body = vehicle["rigidBody"]
    wheels = vehicle["wheels"]
    times = np.asarray([_float(row, "fixed_time_s") or 0.0 for row in rows])
    speed = np.asarray([_float(row, "body_velocity_z_mps") or 0.0
                        for row in rows])
    lateral_speed = np.asarray([_float(row, "body_velocity_x_mps") or 0.0
                                for row in rows])
    steering = np.asarray([_float(row, "applied_steering_norm") or 0.0
                           for row in rows])
    throttle = np.asarray([_float(row, "applied_throttle_norm") or 0.0
                           for row in rows])
    acceleration = _derivative(speed, times)
    mass = float(rigid_body["mass"])
    drag = float(rigid_body["drag"])
    total_static_load = sum(float(wheel["sprungMass"]) for wheel in wheels)
    total_static_load *= GRAVITY_MPS2
    straight_steady = ((speed > 0.5) & (speed <= 16.0) &
                       (np.abs(lateral_speed) < 0.01) &
                       (np.abs(steering) < 1.0e-4) &
                       (throttle > 0.0) & (np.abs(acceleration) < 0.02))
    inferred_force = mass * acceleration + mass * drag * speed
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "offline_unity_wheel_drive_observability_audit",
        "simulator_modified": False,
        "production_mpc_modified": False,
        "ground_truth_use": "diagnostic_trace_only",
        "source": {
            "traces": [str(path) for path in traces],
            "static_snapshot": str(static),
            "trace_rows": len(rows),
            "trace_columns": len(fields),
            "scene": metadata.get("sceneName"),
            "trace_build_tag": metadata.get("simulatorBuildTag"),
            "steering_selection": (
                "included_in_transition_screen" if include_steering else
                "near_zero_applied_steering_only"),
        },
        "unity_required_values": {
            "wheel_collider_radius_m": [float(wheel["radius"])
                                         for wheel in wheels],
            "controller_wheel_radius_m": float(vehicle["wheelRadiusControllerM"]),
            "wheel_mass_kg": [float(wheel["mass"]) for wheel in wheels],
            "wheel_damping_rate": [float(wheel["wheelDampingRate"])
                                    for wheel in wheels],
            "forward_curve": {
                "extremum_slip": UNITY_FORWARD_EXTREMUM_SLIP,
                "extremum_value": UNITY_FORWARD_EXTREMUM_VALUE,
                "asymptote_slip": UNITY_FORWARD_ASYMPTOTE_SLIP,
                "asymptote_value": UNITY_FORWARD_ASYMPTOTE_VALUE,
                "stiffness": UNITY_FORWARD_STIFFNESS,
            },
            "total_static_sprung_load_n": total_static_load,
            "body_mass_kg": mass,
            "body_drag": drag,
        },
        "selection": {
            "straight_steady_rows": int(np.count_nonzero(straight_steady)),
            "conditions": [
                "0.5 < body_velocity_z <= 16 m/s",
                "abs(body_velocity_x) < 0.01 m/s",
                "zero applied steering",
                "positive throttle and zero-brake drive plateaus",
                "abs(smoothed longitudinal acceleration) < 0.02 m/s^2",
            ],
        },
        "per_wheel": [],
        "body_force_screen": {
            "inferred_longitudinal_contact_force_n": _stats(
                inferred_force[straight_steady]),
            "method": "mass * longitudinal_acceleration + mass * Unity drag * speed",
            "tire_curve_comparison_is_diagnostic_only": True,
        },
        "acceptance": {
            "runtime_use": False,
            "wheel_inertia_identified": False,
            "forward_curve_refit": False,
            "requires_transient_holdout": True,
        },
    }
    for wheel in range(4):
        rpm = np.asarray([_float(row, f"wheel{wheel}_rpm") or 0.0
                          for row in rows])
        recorded_values = [
            _float(row, f"wheel{wheel}_forward_slip") for row in rows]
        recorded = np.asarray([
            math.nan if value is None else value for value in recorded_values])
        motor = np.asarray([
            _float(row, f"wheel{wheel}_motor_torque_nm") or 0.0
            for row in rows])
        brake = np.asarray([_float(row, f"wheel{wheel}_brake_torque_nm") or 0.0
                            for row in rows])
        normal_values = [
            _float(row, f"wheel{wheel}_contact_force_n") for row in rows]
        normal_load = np.asarray([
            math.nan if value is None else value for value in normal_values])
        radius = float(wheels[wheel]["radius"])
        surface_speed = rpm * (2.0 * math.pi / 60.0) * radius
        reconstructed = np.asarray([
            unity_forward_slip(surface_speed[index], speed[index])
            for index in range(len(rows))
        ])
        valid_steady = straight_steady & np.isfinite(recorded)
        error = recorded - reconstructed
        curve_fraction = np.asarray([
            unity_wheel_friction_value(
                value, UNITY_FORWARD_EXTREMUM_SLIP,
                UNITY_FORWARD_EXTREMUM_VALUE, UNITY_FORWARD_ASYMPTOTE_SLIP,
                UNITY_FORWARD_ASYMPTOTE_VALUE, UNITY_FORWARD_STIFFNESS)
            if math.isfinite(value) else math.nan for value in recorded])
        curve_force = curve_fraction * total_static_load
        result["per_wheel"].append({
            "wheel": wheel,
            "steady_forward_slip_recorded": _stats(recorded[valid_steady]),
            "steady_forward_slip_from_collider_radius_and_rpm": _stats(
                reconstructed[valid_steady]),
            "steady_recorded_minus_reconstructed_slip": _stats(
                error[valid_steady]),
            "steady_recorded_curve_force_screen_n": _stats(
                curve_force[valid_steady]),
            "steady_inferred_body_force_n": _stats(
                inferred_force[valid_steady]),
            "all_rows_forward_slip_transient_screen": {
                "recorded": _stats(recorded),
                "reconstructed": _stats(reconstructed),
                "error": _stats(error),
            },
            "wheel_rpm": _stats(rpm),
            "motor_torque_nm": _stats(np.asarray([
                _float(row, f"wheel{wheel}_motor_torque_nm") or 0.0
                for row in rows])),
            "brake_torque_nm": _stats(brake),
            "wheel_rotational_balance_screen":
                _fit_wheel_rotational_transition_screen(
                    times, speed, throttle, steering, motor, brake, rpm,
                    normal_load, recorded, radius,
                    include_steering=include_steering),
            "interpretation": (
                "steady plateaus test collider-radius slip mapping; "
                "transient mismatch is retained as wheel rotational-state "
                "evidence, not absorbed into a force or tire parameter"),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-steering", action="store_true",
        help="include drive transitions while the car is steering",
    )
    args = parser.parse_args()
    result = analyze(args.trace, args.static, args.output,
                     include_steering=args.include_steering)
    print(json.dumps({
        "output": str(args.output),
        "source": result["source"],
        "selection": result["selection"],
        "per_wheel": result["per_wheel"],
        "acceptance": result["acceptance"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Identify diagnostic wheel-drive dynamics without hiding brake events.

This is an offline-only identification tool. It uses source timestamps from
the Unity diagnostic trace and keeps two behaviours separate:

* powered motion is fitted as a continuous wheel-state model;
* the trace's zero-throttle brake application is reported as a hybrid event.

The powered model is deliberately small and provisional:

    domega = km * motor_torque
           - kx * radius * Fz * phi_x(Sx)
           - kd * omega

It is not a physical wheel-inertia claim. ``WheelHit.force`` is only a load
proxy and the trace does not expose a separate tire Fx, so the contact term is
an effective identification basis. The result must not be consumed by a
runtime controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter

try:
    from wheel_friction_curve import GUIDE_LONGITUDINAL_CURVE, friction_value
except ModuleNotFoundError:
    from tools.model_id.wheel_friction_curve import (  # type: ignore
        GUIDE_LONGITUDINAL_CURVE,
        friction_value,
    )


WHEEL_COUNT = 4
WHEEL_RADIUS_M = 0.059
TRACE_DT_S = 0.001
DERIVATIVE_OUTLIER_LIMIT_RADPS2 = 500.0
POWERED_MOTOR_MIN_NM = 0.05
POWERED_OMEGA_MIN_RADPS = 1.0
POWERED_MAX_ABS_SLIP = 0.40
BRAKE_THRESHOLD_NM = 0.01
MAX_BRAKE_EVENT_S = 1.0

# The drive excitation has four powered blocks. The first part of every
# powered block identifies the model and the remainder is a temporal holdout.
DEFAULT_TRAIN_WINDOWS = (
    (0.15, 6.0),
    (12.15, 16.0),
    (22.15, 26.0),
    (36.15, 40.0),
)
DEFAULT_VALIDATION_WINDOWS = (
    (6.0, 8.0),
    (16.0, 18.0),
    (26.0, 32.0),
    (40.0, 46.0),
)


def _float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("trace contains no rows")
    required = {"fixed_time_s", "body_velocity_z_mps",
                "applied_throttle_norm"}
    for wheel in range(WHEEL_COUNT):
        required.update({
            f"wheel{wheel}_rpm", f"wheel{wheel}_motor_torque_nm",
            f"wheel{wheel}_brake_torque_nm", f"wheel{wheel}_forward_slip",
            f"wheel{wheel}_contact_force_n", f"wheel{wheel}_grounded",
        })
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"trace is missing fields: {missing}")
    return rows


def _in_windows(time_s: float, windows: Iterable[tuple[float, float]]) -> bool:
    return any(start <= time_s < end for start, end in windows)


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)],
                        dtype=float)
    if finite.size == 0:
        return {"count": 0, "mae": None, "rmse": None, "p50": None,
                "p95": None, "max": None, "bias": None}
    return {
        "count": int(finite.size),
        "mae": float(np.mean(np.abs(finite))),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
        "p50": float(np.percentile(np.abs(finite), 50)),
        "p95": float(np.percentile(np.abs(finite), 95)),
        "max": float(np.max(np.abs(finite))),
        "bias": float(np.mean(finite)),
    }


def _wheel_series(rows: list[dict[str, str]], wheel: int) -> list[dict[str, float]]:
    """Parse one wheel and estimate a smoothed source-time derivative.

    The derivative is calculated on the complete contiguous source series
    before powered-mode filtering. This avoids applying a smoothing window
    across the gaps introduced when brake/airborne samples are excluded.
    """
    series: list[dict[str, float]] = []
    for row in rows:
        time_s = _float(row, "fixed_time_s")
        rpm = _float(row, f"wheel{wheel}_rpm")
        motor = _float(row, f"wheel{wheel}_motor_torque_nm")
        brake = _float(row, f"wheel{wheel}_brake_torque_nm")
        slip = _float(row, f"wheel{wheel}_forward_slip")
        load = _float(row, f"wheel{wheel}_contact_force_n")
        grounded = _float(row, f"wheel{wheel}_grounded")
        if None in (time_s, rpm, motor, brake, slip, load, grounded):
            continue
        omega = float(rpm) * 2.0 * math.pi / 60.0
        contact_torque = WHEEL_RADIUS_M * max(0.0, float(load)) * \
            friction_value(float(slip), GUIDE_LONGITUDINAL_CURVE)
        series.append({
            "time_s": float(time_s),
            "omega_radps": omega,
            "motor_torque_nm": float(motor),
            "brake_torque_nm": float(brake),
            "contact_torque_nm": contact_torque,
            "body_u_mps": _float(row, "body_velocity_z_mps") or 0.0,
            "throttle_norm": _float(row, "applied_throttle_norm") or 0.0,
            "forward_slip": float(slip),
            "load_proxy_n": max(0.0, float(load)),
            "grounded": float(grounded),
        })

    if len(series) < 3:
        return []

    times = np.asarray([item["time_s"] for item in series], dtype=float)
    omega = np.asarray([item["omega_radps"] for item in series], dtype=float)
    dts = np.diff(times)
    positive_dts = dts[dts > 0.0]
    median_dt = float(np.median(positive_dts)) if positive_dts.size else 0.0
    derivative = np.empty_like(omega)
    derivative[0] = (omega[1] - omega[0]) / max(times[1] - times[0], 1e-9)
    derivative[-1] = (omega[-1] - omega[-2]) / max(times[-1] - times[-2], 1e-9)
    derivative[1:-1] = (omega[2:] - omega[:-2]) / np.maximum(
        times[2:] - times[:-2], 1e-9)

    # The current trace is contiguous at approximately 1 kHz. If a future
    # trace contains a gap, retain the source-time finite difference instead
    # of smoothing across it.
    if (median_dt > 0.0 and len(series) >= 21 and
            float(np.max(dts)) <= 3.0 * median_dt):
        derivative = savgol_filter(
            omega, 21, 2, deriv=1, delta=median_dt, mode="interp")
    for item, value in zip(series, derivative):
        item["omega_dot_radps2"] = float(value)
    return series


def _powered_records(series: list[dict[str, float]],
                     windows: Iterable[tuple[float, float]]) -> list[dict[str, float]]:
    return [item for item in series
            if _in_windows(item["time_s"], windows)
            and item["grounded"] >= 0.5
            and item["motor_torque_nm"] > POWERED_MOTOR_MIN_NM
            and item["brake_torque_nm"] <= BRAKE_THRESHOLD_NM
            and item["omega_radps"] > POWERED_OMEGA_MIN_RADPS
            and abs(item["forward_slip"]) <= POWERED_MAX_ABS_SLIP
            and abs(item["omega_dot_radps2"]) <= DERIVATIVE_OUTLIER_LIMIT_RADPS2]


def _design(records: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    # The signs are explicit: a positive forward slip produces a positive
    # contact torque that opposes positive wheel acceleration.
    features = np.asarray([
        [item["motor_torque_nm"],
         -item["contact_torque_nm"],
         -item["omega_radps"]]
        for item in records], dtype=float)
    targets = np.asarray([item["omega_dot_radps2"] for item in records],
                         dtype=float)
    return features, targets


def _fit(records: list[dict[str, float]]) -> dict[str, Any]:
    if len(records) < 20:
        raise ValueError("too few powered grounded records for drive fit")
    features, targets = _design(records)
    result = lsq_linear(features, targets, bounds=(0.0, np.inf),
                        lsmr_tol="auto")
    residual = features @ result.x - targets
    condition = float(np.linalg.cond(features)) if len(features) >= 3 else None
    return {
        "coefficients": {
            "motor_torque_to_omega_dot_per_kgm2": float(result.x[0]),
            "contact_torque_to_omega_dot_per_kgm2": float(result.x[1]),
            "wheel_damping_per_kgm2": float(result.x[2]),
        },
        "derivative_fit": _stats(residual),
        "samples_total": int(len(records)),
        "derivative_outlier_limit_radps2": DERIVATIVE_OUTLIER_LIMIT_RADPS2,
        "design_condition_number": condition,
        "optimizer": {
            "status": int(result.status),
            "message": result.message,
            "optimality": float(result.optimality),
            "cost": float(0.5 * np.dot(residual, residual)),
            "active_mask": result.active_mask.tolist(),
        },
    }


def _predict_derivative(omega: float, item: dict[str, float],
                        coefficients: dict[str, float]) -> float:
    return (
        coefficients["motor_torque_to_omega_dot_per_kgm2"] *
        item["motor_torque_nm"] -
        coefficients["contact_torque_to_omega_dot_per_kgm2"] *
        item["contact_torque_nm"] -
        coefficients["wheel_damping_per_kgm2"] * omega)


def _contiguous_segments(records: list[dict[str, float]],
                         max_gap_s: float = 0.02) -> list[list[dict[str, float]]]:
    segments: list[list[dict[str, float]]] = []
    for item in records:
        if not segments or item["time_s"] - segments[-1][-1]["time_s"] > max_gap_s:
            segments.append([item])
        else:
            segments[-1].append(item)
    return [segment for segment in segments if len(segment) >= 3]


def _replay_segment(records: list[dict[str, float]],
                    coefficients: dict[str, float],
                    sample_period_s: float) -> list[float]:
    if len(records) < 3:
        return []
    source_times = np.asarray([item["time_s"] for item in records], dtype=float)
    source_omega = np.asarray([item["omega_radps"] for item in records], dtype=float)
    start = float(source_times[0])
    end = float(source_times[-1])
    target_times = np.arange(start + sample_period_s, end + 1e-9,
                             sample_period_s)
    if target_times.size == 0:
        return []

    omega = float(source_omega[0])
    input_index = 0
    predicted_error: list[float] = []
    for target_time in target_times:
        # Hold the recorded applied input from the beginning of this model
        # step. The state is recursive; no future measured omega is used.
        while (input_index + 1 < len(records) and
               source_times[input_index + 1] <= target_time - sample_period_s + 1e-9):
            input_index += 1
        omega = max(0.0, omega + sample_period_s * _predict_derivative(
            omega, records[input_index], coefficients))
        predicted_error.append(omega - float(np.interp(
            target_time, source_times, source_omega)))
    return predicted_error


def _replay(records: list[dict[str, float]], coefficients: dict[str, float],
            sample_period_s: float) -> dict[str, Any]:
    errors: list[float] = []
    segments = _contiguous_segments(records)
    for segment in segments:
        errors.extend(_replay_segment(segment, coefficients, sample_period_s))
    return {
        "samples": len(errors),
        "sample_period_s": sample_period_s,
        "segments": len(segments),
        "error_radps": _stats(errors),
    }


def _brake_events(rows: list[dict[str, str]], wheel: int) -> dict[str, Any]:
    """Summarize zero-throttle braking as a separate hybrid event."""
    series = _wheel_series(rows, wheel)
    events: list[dict[str, float]] = []
    for index, item in enumerate(series):
        previous = series[index - 1] if index else None
        if (previous is None or previous["brake_torque_nm"] > BRAKE_THRESHOLD_NM or
                item["brake_torque_nm"] <= BRAKE_THRESHOLD_NM or
                previous["omega_radps"] <= POWERED_OMEGA_MIN_RADPS or
                item["motor_torque_nm"] > POWERED_MOTOR_MIN_NM):
            continue
        stop_time: float | None = None
        minimum_derivative = item["omega_dot_radps2"]
        for later in series[index:]:
            if later["time_s"] - item["time_s"] > MAX_BRAKE_EVENT_S:
                break
            minimum_derivative = min(minimum_derivative,
                                     later["omega_dot_radps2"])
            if later["omega_radps"] <= POWERED_OMEGA_MIN_RADPS:
                stop_time = later["time_s"] - item["time_s"]
                break
            if later["brake_torque_nm"] <= BRAKE_THRESHOLD_NM:
                break
        if stop_time is not None:
            events.append({
                "start_time_s": item["time_s"],
                "initial_omega_radps": item["omega_radps"],
                "initial_body_u_mps": item["body_u_mps"],
                "brake_torque_nm": item["brake_torque_nm"],
                "stop_time_s": stop_time,
                "minimum_smoothed_omega_dot_radps2": minimum_derivative,
            })
    return {
        "event_count": len(events),
        "events": events,
        "initial_omega_radps": _stats(
            event["initial_omega_radps"] for event in events),
        "stop_time_s": _stats(event["stop_time_s"] for event in events),
        "brake_torque_nm": _stats(event["brake_torque_nm"] for event in events),
        "interpretation": (
            "This trace applies a large brake command when throttle is zero. "
            "It is a hybrid actuator event, not passive coast data and not a "
            "continuous damping coefficient."),
    }


def _window_list(windows: Iterable[tuple[float, float]]) -> list[list[float]]:
    return [[float(start), float(end)] for start, end in windows]


def fit(trace: Path, output: Path,
        train_windows: tuple[tuple[float, float], ...] = DEFAULT_TRAIN_WINDOWS,
        validation_windows: tuple[tuple[float, float], ...] =
        DEFAULT_VALIDATION_WINDOWS) -> dict[str, Any]:
    rows = _read_trace(trace)
    wheels: dict[str, Any] = {}
    brake_reports: dict[str, Any] = {}
    for wheel in range(WHEEL_COUNT):
        series = _wheel_series(rows, wheel)
        train = _powered_records(series, train_windows)
        validation = _powered_records(series, validation_windows)
        fit_report = _fit(train)
        coefficients = fit_report["coefficients"]
        features, targets = _design(validation)
        validation_residual = features @ np.asarray([
            coefficients["motor_torque_to_omega_dot_per_kgm2"],
            coefficients["contact_torque_to_omega_dot_per_kgm2"],
            coefficients["wheel_damping_per_kgm2"],
        ]) - targets
        wheels[f"wheel_{wheel}"] = {
            "train": {
                "windows_s": _window_list(train_windows),
                **fit_report,
            },
            "validation": {
                "windows_s": _window_list(validation_windows),
                "samples_total_after_powered_filter": len(validation),
                "derivative_fit": _stats(validation_residual),
            },
            "arbitrary_dt_replay": {
                str(period): _replay(validation, coefficients, period)
                for period in (0.001, 0.005, 0.010, 0.025)
            },
        }
        brake_reports[f"wheel_{wheel}"] = _brake_events(rows, wheel)

    report = {
        "schema_version": 2,
        "status": "continuous_drive_identification_v2_offline_only",
        "acceptance": "not_accepted",
        "ground_truth_use": "offline_identification_and_scoring_only",
        "recursive_prediction_uses_future_gt": False,
        "trace": str(trace),
        "trace_timing": {
            "nominal_dt_s": TRACE_DT_S,
            "state_source": "Unity diagnostic wheel rpm",
            "derivatives_use_source_fixed_time_s": True,
        },
        "physical_parameters": {
            "wheel_radius_m": WHEEL_RADIUS_M,
            "longitudinal_curve": "unity_prefab:F1TENTH.prefab surrogate",
            "wheel_load": "WheelHit.force scalar proxy, clipped nonnegative",
        },
        "actuator_semantics": {
            "zero_throttle_is_passive_coast": False,
            "observed_zero_throttle_behavior": (
                "VehicleController applies MotorTorque as brakeTorque to all "
                "wheels when DriveTorque is zero"),
            "brake_event_reports": brake_reports,
            "brake_is_fitted_as_continuous_drive_term": False,
        },
        "model": {
            "equation": (
                "domega = km*motor_torque - "
                "kx*radius*Fz*phi_x(Sx) - kd*omega"),
            "separate_tire_force_components_observed": False,
            "coefficients_are_physical_inertia_claims": False,
            "powered_filter": {
                "motor_torque_min_nm": POWERED_MOTOR_MIN_NM,
                "brake_torque_max_nm": BRAKE_THRESHOLD_NM,
                "omega_min_radps": POWERED_OMEGA_MIN_RADPS,
                "max_abs_forward_slip": POWERED_MAX_ABS_SLIP,
                "grounded_required": True,
            },
            "continuous_derivative_smoothing": {
                "method": "savgol",
                "window_samples": 21,
                "polynomial_order": 2,
                "purpose": "reduce 1 kHz contact-solver differentiation noise",
            },
        },
        "split": {
            "train_windows_s": _window_list(train_windows),
            "validation_windows_s": _window_list(validation_windows),
            "reason": (
                "powered blocks are split into within-block temporal holdouts; "
                "braking blocks are not treated as powered validation"),
        },
        "wheels": wheels,
        "limitations": [
            "The trace contains no passive-coast phase because zero throttle applies brake torque.",
            "WheelHit.force is a load proxy; tire Fx is not directly observed.",
            "The excitation is straight-line and does not identify combined-slip drive effects.",
            "The four-wheel fit is an effective offline basis, not a promoted MPC plant.",
        ],
        "next_action": (
            "Collect repeated E5 throttle sequences with source-time logging, "
            "then use E3/E4 combined-slip and causal-load data before reducing "
            "the four wheel states for MPC."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = fit(args.trace, args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "acceptance": report["acceptance"],
        "wheels": list(report["wheels"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit a causal load-transfer state from the offline Unity contact trace."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


GRAVITY_MPS2 = 9.81
MIN_SPEED_MPS = 1.0
MAX_ABS_ACCELERATION_MPS2 = 30.0
BRAKE_THRESHOLD_NM = 0.01
MAX_GAP_S = 0.02


def _float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _stats(values: Iterable[float]) -> dict[str, Any]:
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


def _derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    dts = np.diff(times)
    positive = dts[dts > 0.0]
    median_dt = float(np.median(positive)) if positive.size else 0.0
    if (len(values) >= 21 and median_dt > 0.0 and
            float(np.max(dts)) <= 3.0 * median_dt):
        return savgol_filter(values, 21, 2, deriv=1,
                             delta=median_dt, mode="interp")
    return np.gradient(values, times)


def _load(trace: Path, dump: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with trace.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("trace contains no rows")
    return rows, json.loads(dump.read_text(encoding="utf-8"))


def _measurement(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    times = np.asarray([_float(row, "fixed_time_s") or 0.0 for row in rows])
    u = np.asarray([_float(row, "body_velocity_z_mps") or 0.0 for row in rows])
    v = np.asarray([_float(row, "body_velocity_x_mps") or 0.0 for row in rows])
    yaw = np.asarray([
        _float(row, "body_angular_velocity_y_radps") or 0.0 for row in rows
    ])
    u_dot = _derivative(u, times)
    v_dot = _derivative(v, times)
    ax = u_dot - yaw * v
    ay = v_dot + yaw * u
    loads = np.zeros((len(rows), 4), dtype=float)
    grounded = np.ones(len(rows), dtype=bool)
    brake_free = np.ones(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        for wheel in range(4):
            prefix = f"wheel{wheel}_"
            force = _float(row, prefix + "contact_force_n")
            normal_y = _float(row, prefix + "contact_normal_y")
            if force is None or normal_y is None:
                grounded[index] = False
                loads[index, wheel] = np.nan
            else:
                loads[index, wheel] = max(0.0, force * normal_y)
            if ((_float(row, prefix + "grounded") or 0.0) < 0.5):
                grounded[index] = False
            if ((_float(row, prefix + "brake_torque_nm") or 0.0) >
                    BRAKE_THRESHOLD_NM):
                brake_free[index] = False
    left_right = loads[:, 0] + loads[:, 2] - loads[:, 1] - loads[:, 3]
    front_rear = loads[:, 0] + loads[:, 1] - loads[:, 2] - loads[:, 3]
    valid = (grounded & brake_free & (np.abs(u) >= MIN_SPEED_MPS) &
             np.isfinite(loads).all(axis=1) &
             (loads >= 0.0).all(axis=1) & (loads <= 30.0).all(axis=1) &
             (np.abs(ax) <= MAX_ABS_ACCELERATION_MPS2) &
             (np.abs(ay) <= MAX_ABS_ACCELERATION_MPS2))
    return {
        "times": times,
        "u": u,
        "ax": ax,
        "ay": ay,
        "loads": loads,
        "left_right": left_right,
        "front_rear": front_rear,
        "valid": valid,
    }


def _linear_screen(acceleration: np.ndarray, transfer: np.ndarray,
                   valid: np.ndarray) -> dict[str, Any]:
    mask = valid & np.isfinite(acceleration) & np.isfinite(transfer)
    x = acceleration[mask]
    y = transfer[mask]
    if len(x) < 3:
        return {"samples": len(x), "slope_N_per_mps2": None,
                "intercept_N": None, "residual": _stats(())}
    design = np.column_stack((np.ones(len(x)), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "samples": len(x),
        "slope_N_per_mps2": float(slope),
        "intercept_N": float(intercept),
        "residual": _stats(design @ np.asarray((intercept, slope)) - y),
    }


def _valid_segments(times: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    segments: list[list[int]] = []
    for index in indices:
        if (not segments or
                times[index] - times[segments[-1][-1]] > MAX_GAP_S):
            segments.append([int(index)])
        else:
            segments[-1].append(int(index))
    return [np.asarray(segment, dtype=int)
            for segment in segments if len(segment) >= 3]


def _fit_causal_state(times: np.ndarray, acceleration: np.ndarray,
                      measured: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    segments = _valid_segments(times, valid)
    if not segments:
        raise ValueError("no valid contiguous load-transfer segments")
    screen = _linear_screen(acceleration, measured, valid)
    initial_bias = float(screen["intercept_N"] or 0.0)
    initial_gain = float(screen["slope_N_per_mps2"] or 0.0)

    def simulate(parameters: np.ndarray, return_errors: bool = False) -> np.ndarray:
        bias = float(parameters[0])
        gain = float(parameters[1])
        tau = float(parameters[2])
        errors: list[float] = []
        for segment in segments:
            state = float(measured[segment[0]])
            for previous_index, current_index in zip(segment, segment[1:]):
                dt = times[current_index] - times[previous_index]
                alpha = min(1.0, max(0.0, dt / max(tau, 1.0e-6)))
                target = bias + gain * acceleration[previous_index]
                state += alpha * (target - state)
                errors.append(state - measured[current_index])
        return np.asarray(errors, dtype=float)

    result = least_squares(
        lambda parameters: simulate(parameters, True),
        x0=np.asarray((initial_bias, initial_gain, 0.05), dtype=float),
        bounds=(np.asarray((-30.0, -100.0, 0.001)),
                np.asarray((30.0, 100.0, 2.0))),
        max_nfev=200,
    )
    errors = simulate(result.x)
    return {
        "static_linear_screen": screen,
        "parameters": {
            "steady_bias_N": float(result.x[0]),
            "transfer_gain_N_per_mps2": float(result.x[1]),
            "time_constant_s": float(result.x[2]),
        },
        "causal_replay": {
            "segments": len(segments),
            "samples": len(errors),
            "residual": _stats(errors),
        },
        "optimizer": {
            "status": int(result.status),
            "message": result.message,
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
        },
    }


def fit(trace: Path, dump: Path, output: Path) -> dict[str, Any]:
    rows, payload = _load(trace, dump)
    measurement = _measurement(rows)
    valid = measurement["valid"]
    rigid_body = payload["vehicle"]["rigidBody"]
    wheel_snapshots = payload["vehicle"]["wheels"]
    static_dump_loads = [
        float(wheel.get("sprungMass", 0.0)) * GRAVITY_MPS2
        for wheel in wheel_snapshots
    ]
    report = {
        "schema_version": 1,
        "status": "causal_load_transfer_identification_offline_only",
        "acceptance": "not_accepted",
        "offline_only": True,
        "future_ground_truth_used": False,
        "input": {"trace": str(trace), "static_dump": str(dump)},
        "source_time": {
            "derivative_field": "fixed_time_s",
            "callback_time_used": False,
            "derivative_smoothing": "Savitzky-Golay 21 samples, offline only",
        },
        "structural_parameters": {
            "rigidbody_mass_kg": float(rigid_body["mass"]),
            "static_sprung_loads_N_from_dump": static_dump_loads,
            "static_sprung_load_sum_N": sum(static_dump_loads),
        },
        "selection": {
            "all_four_wheels_grounded": True,
            "zero_brake_event_rows_only": True,
            "speed_min_mps": MIN_SPEED_MPS,
            "max_abs_acceleration_mps2": MAX_ABS_ACCELERATION_MPS2,
            "valid_rows": int(np.count_nonzero(valid)),
            "total_rows": len(rows),
        },
        "measured_loads": {
            "wheel_loads_N": [
                _stats(measurement["loads"][valid, wheel])
                for wheel in range(4)],
            "left_right_transfer_N": _stats(measurement["left_right"][valid]),
            "front_rear_transfer_N": _stats(measurement["front_rear"][valid]),
            "lateral_acceleration_mps2": _stats(measurement["ay"][valid]),
            "longitudinal_acceleration_mps2": _stats(measurement["ax"][valid]),
        },
        "linear_screens": {
            "left_right_from_lateral_acceleration": _linear_screen(
                measurement["ay"], measurement["left_right"], valid),
            "front_rear_from_longitudinal_acceleration": _linear_screen(
                measurement["ax"], measurement["front_rear"], valid),
        },
        "causal_states": {
            "left_right": _fit_causal_state(
                measurement["times"], measurement["ay"],
                measurement["left_right"], valid),
            "front_rear": _fit_causal_state(
                measurement["times"], measurement["ax"],
                measurement["front_rear"], valid),
        },
        "model_form": {
            "left_right": "z_lr_dot = (k_lr * ay - z_lr) / tau_lr",
            "front_rear": "z_fr_dot = (k_fr * ax - z_fr) / tau_fr",
            "distribution": (
                "add z_lr/4 to each left wheel and subtract from each right; "
                "add z_fr/4 to each front wheel and subtract from each rear"),
            "total_supported_load_conserved": True,
        },
        "limitations": [
            "WheelHit.force*normal_y is a normal-load proxy.",
            "Automatic zero-throttle brake phases are excluded from fit selection.",
            "The fitted states are effective simulator-load states, not a mechanical CG-height estimate.",
        ],
        "next_action": (
            "Re-run body force/moment inversion with the causal load state and "
            "compare force-distribution residuals by regime."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = fit(args.trace, args.dump, args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "selection": report["selection"],
        "causal_states": report["causal_states"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

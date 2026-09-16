#!/usr/bin/env python3
"""Audit Unity suspension pose/load response without promoting a gain.

This tool uses only the read-only exact-scene diagnostic trace.  It compares
the serialized WheelCollider spring/damper law with the measured per-wheel
``WheelHit`` load and body pose, then reports a chronological holdout.  Any
fitted intercept or coefficient is an observability result only; it is not
added to the vehicle plant, odometry, or production MPC by this script.
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
from scipy.spatial.transform import Rotation


GRAVITY_MPS2 = 9.81
MIN_SPEED_MPS = 1.0
BRAKE_THRESHOLD_NM = 0.01


def _float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mae": None, "rmse": None, "p50": None,
                "p95": None, "bias": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mae": float(np.mean(np.abs(finite))),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
        "p50": float(np.percentile(np.abs(finite), 50.0)),
        "p95": float(np.percentile(np.abs(finite), 95.0)),
        "bias": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _derivative_segment(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    dt = np.diff(times)
    positive = dt[dt > 0.0]
    if not positive.size:
        return np.gradient(values, times)
    median_dt = float(np.median(positive))
    if len(values) >= 21 and np.max(dt) <= 3.0 * median_dt:
        return savgol_filter(values, 21, 2, deriv=1, delta=median_dt,
                             mode="interp")
    return np.gradient(values, times)


def _derivative(values: np.ndarray, times: np.ndarray,
                breakpoints: np.ndarray | None = None) -> np.ndarray:
    """Differentiate trace segments without crossing a simulator reset."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if breakpoints is None or not np.any(breakpoints):
        return _derivative_segment(values, times)
    starts = [0] + [int(index) for index in np.flatnonzero(breakpoints)]
    ends = starts[1:] + [len(values)]
    output = np.full_like(values, math.nan, dtype=float)
    for start, end in zip(starts, ends):
        if end > start:
            output[start:end] = _derivative_segment(
                values[start:end], times[start:end])
    # The first sample after a reset has no causal derivative across the
    # discontinuity. The selector removes a short neighbourhood as well.
    output[breakpoints] = math.nan
    return output


def _reset_boundaries(rows: list[dict[str, str]]) -> np.ndarray:
    """Detect diagnostic-trace teleports caused by an experiment reset.

    The Unity experiment resetter preserves the fixed-step/time counters, so
    a timestamp-gap test is insufficient. A reset is identified as a large
    root-position jump that lands at near-zero body speed. These thresholds
    are far above one 1 kHz physical transition and are used only to segment
    offline diagnostics.
    """
    position = np.asarray([
        [_float(row, f"root_position_{axis}_m") or 0.0
         for axis in "xyz"] for row in rows], dtype=float)
    velocity = np.asarray([
        [_float(row, f"body_velocity_{axis}_mps") or 0.0
         for axis in "xyz"] for row in rows], dtype=float)
    if len(rows) < 2:
        return np.zeros(len(rows), dtype=bool)
    position_jump = np.linalg.norm(np.diff(position, axis=0), axis=1)
    previous_speed = np.linalg.norm(velocity[:-1], axis=1)
    current_speed = np.linalg.norm(velocity[1:], axis=1)
    return np.r_[False, (position_jump > 0.2) &
                 (current_speed < 0.5) &
                 ((previous_speed - current_speed) > 1.0)]


def _reset_exclusion_mask(boundaries: np.ndarray,
                           radius: int = 10) -> np.ndarray:
    """Remove derivative-contaminated rows around reset edges."""
    excluded = np.zeros(len(boundaries), dtype=bool)
    for boundary in np.flatnonzero(boundaries):
        excluded[max(0, int(boundary) - radius):
                 min(len(boundaries), int(boundary) + radius + 1)] = True
    return excluded


def _read(traces: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    fields: list[str] | None = None
    for trace in traces:
        with trace.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            current = list(reader.fieldnames or [])
            if fields is None:
                fields = current
            elif fields != current:
                raise ValueError(f"trace header mismatch: {trace}")
            rows.extend(reader)
    if not rows:
        raise ValueError("traces contain no rows")
    required = {
        "fixed_time_s", "root_position_y_m", "body_velocity_z_mps",
        "root_rotation_x", "root_rotation_y", "root_rotation_z",
        "root_rotation_w",
    }
    required.update(
        field for wheel in range(4)
        for field in (
            f"wheel{wheel}_grounded", f"wheel{wheel}_contact_force_n",
            f"wheel{wheel}_contact_normal_y",
            f"wheel{wheel}_brake_torque_nm",
            f"wheel{wheel}_positionVehicleFrame_x_m",
            f"wheel{wheel}_positionVehicleFrame_z_m"))
    # The diagnostic exporter names the two static wheel coordinates in the
    # JSON, not in each trace row.  Exclude those two dynamically below.
    required -= {
        f"wheel{wheel}_positionVehicleFrame_x_m" for wheel in range(4)}
    required -= {
        f"wheel{wheel}_positionVehicleFrame_z_m" for wheel in range(4)}
    missing = sorted(required.difference(fields or ()))
    if missing:
        raise ValueError(f"trace is missing required fields: {missing}")
    return rows


def _pose_terms(rows: list[dict[str, str]],
                reset_boundaries: np.ndarray | None = None
                ) -> dict[str, np.ndarray]:
    times = np.asarray([_float(row, "fixed_time_s") or 0.0 for row in rows])
    root_y = np.asarray([_float(row, "root_position_y_m") or 0.0
                         for row in rows])
    quaternions = np.asarray([
        [_float(row, f"root_rotation_{axis}") or 0.0
         for axis in ("x", "y", "z", "w")]
        for row in rows
    ], dtype=float)
    rotations = Rotation.from_quat(quaternions)
    # Do not use a fixed Euler sequence here.  The vehicle can accumulate a
    # large yaw angle on the raceline, and extracting local pitch/roll from
    # ``relative.as_euler("xyz")`` then aliases yaw into the suspension
    # coordinates.  Remove the yaw from the body-up vector first; this keeps
    # pitch/roll small and frame-correct even after several turns.
    initial = rotations[0]
    relative_up = initial.inv().apply(rotations.apply([0.0, 1.0, 0.0]))
    relative_forward = initial.inv().apply(rotations.apply([0.0, 0.0, 1.0]))
    yaw = np.unwrap(np.arctan2(relative_forward[:, 0],
                               relative_forward[:, 2]))
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    up_x_without_yaw = (cosine * relative_up[:, 0] -
                        sine * relative_up[:, 2])
    up_z_without_yaw = (sine * relative_up[:, 0] +
                        cosine * relative_up[:, 2])
    return {
        "times": times,
        "root_y": root_y,
        "pitch": np.arctan2(up_z_without_yaw, relative_up[:, 1]),
        "roll": -np.arcsin(np.clip(up_x_without_yaw, -1.0, 1.0)),
        "upright_cosine": relative_up[:, 1],
    }


def _wheel_suspension_travel(
        rows: list[dict[str, str]], metadata: dict[str, Any], wheel: int
        ) -> np.ndarray:
    """Reconstruct Unity's normalized WheelCollider travel from ``WheelHit``.

    This is the same geometry used by the source repository's diagnostic
    ``AntiRollBar`` helper: the hit point is transformed into the wheel
    collider frame and the wheel radius is removed.  Unlike a body-pose-only
    approximation, this includes the actual wheel-collider local pose and is
    therefore valid during turning on the flat identification plane.
    """
    vehicle = metadata["vehicle"]
    source = vehicle["wheels"][wheel]
    root = np.asarray([
        [float(_float(row, f"root_position_{axis}_m") or 0.0)
         for axis in "xyz"]
        for row in rows
    ], dtype=float)
    root_quaternions = np.asarray([
        [_float(row, f"root_rotation_{axis}") or 0.0
         for axis in ("x", "y", "z", "w")]
        for row in rows
    ], dtype=float)
    root_rotation = Rotation.from_quat(root_quaternions)
    local_position = np.asarray([
        float(source["localPosition"][axis]) for axis in "xyz"], dtype=float)
    local_quaternion = np.asarray([
        float(source["localRotation"][axis]) for axis in "xyzw"], dtype=float)
    collider_position = root + root_rotation.apply(local_position)
    collider_rotation = root_rotation * Rotation.from_quat(
        np.tile(local_quaternion, (len(rows), 1)))
    contact_points = np.asarray([
        [float(value) if value is not None else math.nan
         for axis in "xyz"
         for value in (_float(row, f"wheel{wheel}_contact_point_{axis}_m"),)]
        if _float(row, f"wheel{wheel}_contact_point_x_m") is not None
        else [math.nan, math.nan, math.nan]
        for row in rows
    ], dtype=float)
    local_hit = collider_rotation.inv().apply(contact_points - collider_position)
    return ((-local_hit[:, 1] - float(source["radius"])) /
            float(source["suspensionDistance"]))


def _finite_derivative(values: np.ndarray, times: np.ndarray,
                       reset_boundaries: np.ndarray | None = None
                       ) -> np.ndarray:
    """Differentiate a trace with ungrounded rows without contaminating it."""
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return np.full_like(values, math.nan, dtype=float)
    filled = np.interp(times, times[finite], values[finite])
    derivative = _derivative(filled, times, reset_boundaries)
    derivative[~finite] = math.nan
    return derivative


def analyze(traces: list[Path], static: Path, output: Path) -> dict[str, Any]:
    rows = _read(traces)
    metadata = json.loads(static.read_text(encoding="utf-8"))
    vehicle = metadata["vehicle"]
    rigid_body = vehicle["rigidBody"]
    wheels = vehicle["wheels"]
    reset_boundaries = _reset_boundaries(rows)
    reset_excluded = _reset_exclusion_mask(reset_boundaries)
    pose = _pose_terms(rows, reset_boundaries)
    times = pose["times"]
    speed = np.asarray([_float(row, "body_velocity_z_mps") or 0.0
                        for row in rows])
    dt = np.diff(times)
    median_dt = float(np.median(dt[dt > 0.0]))
    pitch_rate = _derivative(pose["pitch"], times, reset_boundaries)
    roll_rate = _derivative(pose["roll"], times, reset_boundaries)
    valid_common = speed > MIN_SPEED_MPS
    wheel_reports: list[dict[str, Any]] = []
    loads_matrix: list[np.ndarray] = []
    for wheel in range(4):
        source = wheels[wheel]
        # Vehicle-frame Unity coordinates are x=lateral and z=forward.  The
        # API plant uses y=left and x=forward, so the signed y coordinate is
        # the negated source x coordinate.
        lateral_y = -float(source["positionVehicleFrame"]["x"])
        forward_x = (float(source["positionVehicleFrame"]["z"]) -
                     float(rigid_body["centerOfMass"]["z"]))
        pose_displacement = (pose["root_y"] -
                        pose["pitch"] * forward_x +
                        pose["roll"] * lateral_y)
        pose_displacement -= float(np.median(
            pose_displacement[valid_common]))
        pose_displacement_rate = _derivative(
            pose_displacement, times, reset_boundaries)
        # The primary suspension coordinate is the actual WheelCollider
        # travel reconstructed from the contact point.  Body pose is kept as
        # a secondary observability screen, but it omits wheel travel and
        # therefore must not be used as the force law during turning.
        travel = _wheel_suspension_travel(rows, metadata, wheel)
        travel_rate = _finite_derivative(
            travel, times, reset_boundaries)
        load_values: list[float] = []
        for row in rows:
            force = _float(row, f"wheel{wheel}_contact_force_n")
            normal = _float(row, f"wheel{wheel}_contact_normal_y")
            load_values.append(
                force * normal if force is not None and normal is not None
                else math.nan)
        load = np.asarray(load_values, dtype=float)
        loads_matrix.append(load)
        grounded = np.asarray([
            (_float(row, f"wheel{wheel}_grounded") or 0.0) > 0.5
            for row in rows], dtype=bool)
        brake_free = np.asarray([
            (_float(row, f"wheel{wheel}_brake_torque_nm") or 0.0) <
            BRAKE_THRESHOLD_NM
            for row in rows], dtype=bool)
        normal_y = np.asarray([
            _float(row, f"wheel{wheel}_contact_normal_y") or math.nan
            for row in rows], dtype=float)
        valid = (valid_common & ~reset_excluded & grounded & brake_free &
                 (normal_y > 0.98) &
                 (pose["upright_cosine"] > 0.90) &
                 (np.abs(pose["pitch"]) < 0.35) &
                 (np.abs(pose["roll"]) < 0.35) &
                 np.isfinite(load) & np.isfinite(travel_rate))
        spring = float(source["suspensionSpring"]["spring"])
        damper = float(source["suspensionSpring"]["damper"])
        base = float(source["sprungMass"]) * GRAVITY_MPS2
        suspension_distance = float(source["suspensionDistance"])
        known_prediction = (base - spring * suspension_distance * travel -
                            damper * suspension_distance * travel_rate)
        known_error = load - known_prediction
        pose_prediction = (base - spring * pose_displacement -
                           damper * pose_displacement_rate)
        pose_error = load - pose_prediction
        indices = np.flatnonzero(valid)
        split = int(len(indices) * 0.75)

        def fit(selected: np.ndarray) -> dict[str, Any]:
            if len(selected) < 3:
                return {"samples": int(len(selected)), "identified": False}
            design = np.column_stack((
                np.ones(len(selected)), travel[selected],
                travel_rate[selected]))
            coefficients, _, rank, _ = np.linalg.lstsq(
                design, load[selected], rcond=None)
            residual = load[selected] - design @ coefficients
            return {
                "samples": int(len(selected)),
                "identified": bool(rank == 3),
                "intercept_n": float(coefficients[0]),
                "travel_coefficient_n_per_normalized_travel": float(
                    coefficients[1]),
                "travel_rate_coefficient_ns_per_normalized_travel": float(
                    coefficients[2]),
                "residual": _stats(residual),
            }

        training = indices[:split]
        holdout = indices[split:]
        fitted = fit(training)
        holdout_prediction: dict[str, Any] = {
            "samples": int(len(holdout)),
            "predicted_residual": None,
        }
        if fitted.get("identified") and len(holdout):
            design = np.column_stack((
                np.ones(len(holdout)), travel[holdout],
                travel_rate[holdout]))
            prediction = design @ np.asarray((
                fitted["intercept_n"],
                fitted["travel_coefficient_n_per_normalized_travel"],
                fitted["travel_rate_coefficient_ns_per_normalized_travel"]),
                dtype=float)
            holdout_prediction["predicted_residual"] = _stats(
                load[holdout] - prediction)
        wheel_reports.append({
            "wheel": wheel,
            "source_values": {
                "sprung_mass_kg": float(source["sprungMass"]),
                "static_load_n": base,
                "spring_n_per_m": spring,
                "damper_ns_per_m": damper,
                "suspension_distance_m": float(source["suspensionDistance"]),
                "target_position": float(
                    source["suspensionSpring"]["targetPosition"]),
            },
            "pose_reconstruction": {
                "forward_x_relative_to_com_m": forward_x,
                "lateral_y_relative_to_com_m": lateral_y,
                "displacement": _stats(pose_displacement[valid]),
                "velocity": _stats(pose_displacement_rate[valid]),
            },
            "wheel_pose_travel_reconstruction": {
                "travel": _stats(travel[valid]),
                "travel_rate": _stats(travel_rate[valid]),
                "law_coordinates": (
                    "WheelCollider travel from Transform.InverseTransformPoint"
                    "(WheelHit.point), radius, and suspensionDistance"),
            },
            "measured_load": _stats(load[valid]),
            "known_serialized_spring_damper": {
                "prediction": _stats(known_prediction[valid]),
                "residual": _stats(known_error[valid]),
                "load_prediction_correlation": (
                    float(np.corrcoef(load[valid], known_prediction[valid])[0, 1])
                    if np.count_nonzero(valid) > 2 else None),
            },
            "body_pose_only_load_screen": {
                "prediction": _stats(pose_prediction[valid]),
                "residual": _stats(pose_error[valid]),
            },
            "identified_travel_load_screen": {
                "training": fit(training),
                "chronological_holdout": holdout_prediction,
                "not_a_runtime_parameter": True,
            },
            "selection": {
                "valid_rows": int(np.count_nonzero(valid)),
                "total_rows": len(rows),
                "speed_min_mps": MIN_SPEED_MPS,
                "zero_brake_only": True,
                "flat_contact_normal_y_min": 0.98,
                "upright_cosine_min": 0.90,
            },
        })

    loads = np.column_stack(loads_matrix)
    valid_total = (valid_common & ~reset_excluded &
                   np.isfinite(loads).all(axis=1))
    static_sum = sum(float(wheel["sprungMass"]) for wheel in wheels) * GRAVITY_MPS2
    total_deviation = loads.sum(axis=1) - static_sum
    front_rear = loads[:, 0] + loads[:, 1] - loads[:, 2] - loads[:, 3]
    left_right = loads[:, 0] + loads[:, 2] - loads[:, 1] - loads[:, 3]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "offline_unity_suspension_pose_load_audit",
        "simulator_modified": False,
        "production_mpc_modified": False,
        "ground_truth_use": "diagnostic_trace_only",
        "source": {
            "traces": [str(path) for path in traces],
            "static_snapshot": str(static),
            "trace_rows": len(rows),
            "trace_columns": len(rows[0]),
            "scene": metadata.get("sceneName"),
            "trace_build_tag": metadata.get("simulatorBuildTag"),
            "fixed_dt_median_s": median_dt,
            "reset_segmentation": {
                "detected_boundaries": int(np.count_nonzero(
                    reset_boundaries)),
                "boundary_times_s": [
                    float(times[index])
                    for index in np.flatnonzero(reset_boundaries)],
                "excluded_rows": int(np.count_nonzero(reset_excluded)),
                "exclusion_radius_rows": 10,
                "method": (
                    "root-position jump > 0.2 m, landing body speed < 0.5 "
                    "m/s, and speed drop > 1 m/s"),
            },
        },
        "known_model": {
            "body_mass_kg": float(rigid_body["mass"]),
            "body_com_local_y_m": float(rigid_body["centerOfMass"]["y"]),
            "body_inertia_pitch_kgm2": float(rigid_body["bodyInertiaX"]),
            "body_inertia_roll_kgm2": float(rigid_body["bodyInertiaZ"]),
            "pose_coordinates": (
                "q_i = root_position_y - pitch * wheel_forward_x "
                "+ roll * wheel_lateral_y; force = static_load - k*q - c*q_dot"),
        },
        "total_load_screen": {
            "static_sprung_load_n": static_sum,
            "measured_total_load": _stats(total_deviation[valid_total]),
            "front_minus_rear_load": _stats(front_rear[valid_total]),
            "left_minus_right_load": _stats(left_right[valid_total]),
            "total_load_deviation_definition": "sum(WheelHit.force * contact_normal_y) - static sprung load",
        },
        "per_wheel": wheel_reports,
        "acceptance": {
            "runtime_use": False,
            "suspension_parameters_promoted": False,
            "requires_cross_schedule_holdout": True,
            "requires_initial_suspension_state": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace, args.static, args.output)
    print(json.dumps({
        "output": str(args.output),
        "source": result["source"],
        "total_load_screen": result["total_load_screen"],
        "per_wheel": result["per_wheel"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

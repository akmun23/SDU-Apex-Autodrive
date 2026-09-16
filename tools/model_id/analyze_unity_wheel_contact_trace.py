#!/usr/bin/env python3
"""Audit Unity WheelHit slip/load observability without fitting a tire peak.

The trace is from the disposable diagnostic component in the external Unity
checkout.  This script only reads the exported trace and its static snapshot.
It checks whether the planar state reconstructs Unity's recorded
``sidewaysSlip`` and reports how much ``contact_force_n`` varies around the
serialized sprung-mass load.  It deliberately does not turn either result
into a runtime correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TRACE = Path(
    "sdu_apex_autodrive/artifacts/model_id_work/diagnostics_20260914_v2/"
    "combined_slip_matrix_v1_20260914/wheel_contact_trace.csv")
DEFAULT_STATIC = DEFAULT_TRACE.with_name("simulator_parameters.json")
GRAVITY_MPS2 = 9.81
WHEEL_POSITIONS = (
    (0.17, 0.118), (0.17, -0.118),
    (-0.16, 0.118), (-0.16, -0.118),
)


def _stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "mae": float(np.mean(np.abs(array))),
        "p50": float(np.percentile(np.abs(array), 50)),
        "p95": float(np.percentile(np.abs(array), 95)),
        "max": float(np.max(np.abs(array))),
    }


def _percentiles(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"count": 0}
    p05, p50, p95 = np.percentile(array, [5, 50, 95])
    return {
        "count": int(array.size),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "mean": float(np.mean(array)),
    }


def _wheel_forward_slip_candidates(
        row: dict[str, str], wheel: int, radius_m: float
        ) -> tuple[float, dict[str, float]]:
    """Reconstruct candidate forward-slip definitions for one wheel sample.

    Unity records wheel RPM, WheelHit.forwardSlip, the WheelCollider pose,
    WheelHit point/direction, and Rigidbody world twist.  This compares
    plausible reference points, denominators, and signs directly against that
    recorded slip rather than assuming a conventional tire definition.
    """
    world_velocity = np.asarray([
        float(row[f"world_velocity_{axis}_mps"]) for axis in "xyz"],
        dtype=float)
    world_angular_velocity = np.asarray([
        float(row[f"world_angular_velocity_{axis}_radps"])
        for axis in "xyz"], dtype=float)
    world_com = np.asarray([
        float(row[f"world_com_{axis}_m"]) for axis in "xyz"], dtype=float)
    wheel_center = np.asarray([
        float(row[f"wheel{wheel}_world_pose_{axis}_m"])
        for axis in "xyz"], dtype=float)
    contact_point = np.asarray([
        float(row[f"wheel{wheel}_contact_point_{axis}_m"])
        for axis in "xyz"], dtype=float)
    forward_direction = np.asarray([
        float(row[f"wheel{wheel}_forward_dir_{axis}"])
        for axis in "xyz"], dtype=float)
    direction_norm = float(np.linalg.norm(forward_direction))
    if direction_norm <= 1.0e-12:
        raise ValueError("recorded WheelHit forward direction is degenerate")
    forward_direction /= direction_norm
    wheel_surface_speed = (
        float(row[f"wheel{wheel}_rpm"]) * (2.0 * math.pi / 60.0) * radius_m)
    recorded = float(row[f"wheel{wheel}_forward_slip"])

    candidates: dict[str, float] = {}
    for point_name, point in (("wheel_center", wheel_center),
                              ("contact_point", contact_point)):
        point_velocity = (world_velocity + np.cross(
            world_angular_velocity, point - world_com))
        ground_speed = float(point_velocity @ forward_direction)
        difference = wheel_surface_speed - ground_speed
        denominators = {
            "max_abs_wheel_ground_floor_0p25": max(
                abs(wheel_surface_speed), abs(ground_speed), 0.25),
            "abs_ground_floor_0p25": max(abs(ground_speed), 0.25),
            "abs_wheel_floor_0p25": max(abs(wheel_surface_speed), 0.25),
        }
        for denominator_name, denominator in denominators.items():
            for sign_name, numerator in (
                    ("wheel_minus_ground", difference),
                    ("ground_minus_wheel", -difference)):
                name = f"{point_name}_{sign_name}_{denominator_name}"
                ratio = numerator / denominator
                candidates[name] = ratio
                candidates[f"clip_pm1_{name}"] = max(
                    -1.0, min(1.0, ratio))
        candidates[f"{point_name}_wheel_minus_ground_raw_mps"] = difference
        candidates[f"{point_name}_ground_minus_wheel_raw_mps"] = -difference
    return recorded, candidates


def _forward_slip_error_stats(
        values: list[float]) -> dict[str, float | int]:
    errors = np.asarray(values, dtype=float)
    if errors.size == 0:
        return {"count": 0}
    return {
        "count": int(errors.size),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "p95_abs": float(np.percentile(np.abs(errors), 95)),
        "max_abs": float(np.max(np.abs(errors))),
    }


def analyze(trace: Path, static: Path, output: Path) -> dict[str, Any]:
    metadata = json.loads(static.read_text(encoding="utf-8"))
    sprung_masses = [float(wheel["sprungMass"])
                     for wheel in metadata["vehicle"]["wheels"]]
    rows: list[dict[str, str]] = []
    with trace.open(newline="", encoding="utf-8") as stream:
        rows.extend(csv.DictReader(stream))

    slip_errors: list[list[float]] = [[] for _ in range(4)]
    slip_ratios: list[list[float]] = [[] for _ in range(4)]
    forward_slip_errors: dict[str, list[list[float]]] = {}
    loads: list[list[float]] = [[] for _ in range(4)]
    front_rear_transfer: list[float] = []
    left_right_transfer: list[float] = []
    selected_rows = 0
    for row in rows:
        try:
            u = float(row["body_velocity_z_mps"])
            v = -float(row["body_velocity_x_mps"])
            yaw_rate = -float(row["body_angular_velocity_y_radps"])
        except (KeyError, TypeError, ValueError):
            continue
        if u < 2.0:
            continue
        wheel_slips: list[float] = []
        wheel_loads: list[float] = []
        forward_candidates_by_wheel: list[dict[str, float]] = []
        forward_recorded_by_wheel: list[float] = []
        valid = True
        for wheel, (x_position, y_position) in enumerate(WHEEL_POSITIONS):
            if row[f"wheel{wheel}_grounded"] != "1":
                valid = False
                break
            recorded = row[f"wheel{wheel}_sideways_slip"]
            force = row[f"wheel{wheel}_contact_force_n"]
            if not recorded or not force:
                valid = False
                break
            try:
                wheel_angle = -math.radians(
                    float(row[f"wheel{wheel}_steer_angle_deg"]))
                hit_slip = float(recorded)
                contact_force = float(force)
            except (TypeError, ValueError):
                valid = False
                break
            wheel_vx = u - yaw_rate * y_position
            wheel_vy = v + yaw_rate * x_position
            cosine = math.cos(wheel_angle)
            sine = math.sin(wheel_angle)
            forward = wheel_vx * cosine + wheel_vy * sine
            sideways = -wheel_vx * sine + wheel_vy * cosine
            reconstructed = -sideways / max(abs(forward), 0.5)
            slip_errors[wheel].append(reconstructed - hit_slip)
            if abs(hit_slip) > 1.0e-4:
                slip_ratios[wheel].append(reconstructed / hit_slip)
            try:
                recorded_forward, forward_candidates = (
                    _wheel_forward_slip_candidates(
                        row, wheel,
                        float(metadata["vehicle"]["wheels"][wheel]["radius"])))
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            forward_recorded_by_wheel.append(recorded_forward)
            forward_candidates_by_wheel.append(forward_candidates)
            wheel_slips.append(hit_slip)
            wheel_loads.append(contact_force)
        if not valid:
            continue
        selected_rows += 1
        for wheel, (recorded_forward, candidates) in enumerate(zip(
                forward_recorded_by_wheel, forward_candidates_by_wheel)):
            for name, prediction in candidates.items():
                if not math.isfinite(prediction):
                    continue
                if name not in forward_slip_errors:
                    forward_slip_errors[name] = [[] for _ in range(4)]
                forward_slip_errors[name][wheel].append(
                    prediction - recorded_forward)
        for index in range(4):
            loads[index].append(wheel_loads[index])
        front_rear_transfer.append(
            wheel_loads[0] + wheel_loads[1] - wheel_loads[2] - wheel_loads[3])
        left_right_transfer.append(
            wheel_loads[0] + wheel_loads[2] - wheel_loads[1] - wheel_loads[3])

    static_loads = [mass * GRAVITY_MPS2 for mass in sprung_masses]
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "offline_unity_wheel_contact_observability_audit",
        "simulator_modified": False,
        "production_mpc_modified": False,
        "ground_truth_use": "diagnostic_trace_only",
        "source": {
            "trace": str(trace),
            "static_snapshot": str(static),
            "scene": metadata["sceneName"],
            "unity_version": metadata["unityVersion"],
            "trace_build_tag": metadata["simulatorBuildTag"],
            "warning": ("exact competition-scene runtime snapshot is confirmed; "
                        "dynamic contact/load behavior remains diagnostic-only"),
        },
        "selection": {
            "minimum_body_forward_speed_mps": 2.0,
            "all_four_wheels_grounded": True,
            "rows": selected_rows,
        },
        "sideways_slip_reconstruction": {
            "state_mapping": {
                "unity_forward_axis": "body_velocity_z",
                "model_lateral_axis": "-body_velocity_x",
                "model_yaw_rate": "-body_angular_velocity_y",
                "slip_definition": "minus wheel-frame sideways velocity divided by wheel-frame forward speed",
            },
            "per_wheel_error": [_stats(values) for values in slip_errors],
            "per_wheel_reconstructed_to_recorded_ratio": [
                _percentiles(values) for values in slip_ratios],
        },
        "forward_slip_reconstruction": {
            "kinematics": (
                "world Rigidbody COM velocity transported to the recorded "
                "WheelCollider center or WheelHit point using omega cross r, "
                "then projected on recorded WheelHit.forwardDir"),
            "wheel_surface_speed": (
                "recorded WheelCollider RPM times the exact dumped physical "
                "WheelCollider radius"),
            "per_candidate_per_wheel_error": {
                name: [_forward_slip_error_stats(values) for values in per_wheel]
                for name, per_wheel in sorted(forward_slip_errors.items())
            },
            "interpretation": (
                "candidate definitions are diagnostic comparisons only; the "
                "best trace reconstruction must be independently validated "
                "before changing the offline plant law"),
        },
        "contact_load_variation": {
            "static_sprung_loads_n": static_loads,
            "per_wheel_contact_force_n": [_percentiles(values) for values in loads],
            "front_minus_rear_contact_force_n": _percentiles(front_rear_transfer),
            "left_minus_right_contact_force_n": _percentiles(left_right_transfer),
            "interpretation": "WheelHit contact load varies materially around sprung-mass load; a dynamic load state must be identified explicitly before using the direct curve for track prediction.",
        },
        "acceptance": {
            "runtime_use": False,
            "tire_peak_fitted": False,
            "load_transfer_promoted": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace, args.static, args.output)
    print(json.dumps({
        "output": str(args.output),
        "selection": result["selection"],
        "slip": result["sideways_slip_reconstruction"],
        "forward_slip": result["forward_slip_reconstruction"],
        "loads": result["contact_load_variation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

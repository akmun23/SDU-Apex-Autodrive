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


def analyze(trace: Path, static: Path, output: Path) -> dict[str, Any]:
    metadata = json.loads(static.read_text(encoding="utf-8"))
    sprung_masses = [float(wheel["sprungMass"])
                     for wheel in metadata["vehicle"]["wheels"]]
    rows: list[dict[str, str]] = []
    with trace.open(newline="", encoding="utf-8") as stream:
        rows.extend(csv.DictReader(stream))

    slip_errors: list[list[float]] = [[] for _ in range(4)]
    slip_ratios: list[list[float]] = [[] for _ in range(4)]
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
            wheel_slips.append(hit_slip)
            wheel_loads.append(contact_force)
        if not valid:
            continue
        selected_rows += 1
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
        "loads": result["contact_load_variation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

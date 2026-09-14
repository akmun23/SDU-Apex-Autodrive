#!/usr/bin/env python3
"""Report coverage and validity of the diagnostic combined-slip matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


SX_BANDS = (
    ("abs_sx<0.02", 0.0, 0.02),
    ("0.02<=abs_sx<0.10", 0.02, 0.10),
    ("0.10<=abs_sx<0.25", 0.10, 0.25),
    ("abs_sx>=0.25", 0.25, math.inf),
)
SY_BANDS = (
    ("abs_sy<0.01", 0.0, 0.01),
    ("0.01<=abs_sy<0.05", 0.01, 0.05),
    ("0.05<=abs_sy<0.10", 0.05, 0.10),
    ("0.10<=abs_sy<0.15", 0.10, 0.15),
    ("abs_sy>=0.15", 0.15, math.inf),
)


def _float(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "")
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _stats(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return {"count": 0, "min": None, "p50": None, "p95": None,
                "max": None, "mean": None}
    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _band(value: float, bands: tuple[tuple[str, float, float], ...]) -> str:
    magnitude = abs(value)
    for name, lower, upper in bands:
        if lower <= magnitude < upper:
            return name
    return bands[-1][0]


def analyze(trace_path: Path) -> dict[str, Any]:
    with trace_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("trace contains no rows")

    times = [_float(row, "fixed_time_s") for row in rows]
    valid_times = [value for value in times if value is not None]
    dts = [b - a for a, b in zip(valid_times, valid_times[1:])]
    command_groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        throttle = _float(row, "applied_throttle_norm") or 0.0
        steering = _float(row, "applied_steering_rad") or 0.0
        key = f"throttle={throttle:.6f},steering={steering:.6f}"
        command_groups.setdefault(key, []).append(row)

    phases: list[dict[str, Any]] = []
    for key, group in command_groups.items():
        throttle = _float(group[0], "applied_throttle_norm") or 0.0
        steering = _float(group[0], "applied_steering_rad") or 0.0
        phase_times = [_float(row, "fixed_time_s") for row in group]
        phase_times = [value for value in phase_times if value is not None]
        speed = [_float(row, "body_velocity_z_mps") or 0.0 for row in group]
        grounded = [
            all((_float(row, f"wheel{wheel}_grounded") or 0.0) >= 0.5
                for wheel in range(4))
            for row in group
        ]
        sx = [
            abs(_float(row, f"wheel{wheel}_forward_slip") or 0.0)
            for row in group for wheel in range(4)
        ]
        sy = [
            abs(_float(row, f"wheel{wheel}_sideways_slip") or 0.0)
            for row in group for wheel in range(4)
        ]
        phases.append({
            "command": {"throttle_norm": throttle,
                        "steering_rad": steering},
            "rows": len(group),
            "time_s": [min(phase_times), max(phase_times)],
            "body_speed_z_mps": _stats(speed),
            "all_four_wheels_grounded_fraction": (
                sum(grounded) / len(grounded) if grounded else 0.0),
            "abs_forward_slip": _stats(sx),
            "abs_sideways_slip": _stats(sy),
        })
    phases.sort(key=lambda phase: phase["time_s"][0])

    joint_counts = {
        sx_name: {sy_name: 0 for sy_name, _, _ in SY_BANDS}
        for sx_name, _, _ in SX_BANDS
    }
    for row in rows:
        for wheel in range(4):
            sx = _float(row, f"wheel{wheel}_forward_slip")
            sy = _float(row, f"wheel{wheel}_sideways_slip")
            if sx is None or sy is None:
                continue
            joint_counts[_band(sx, SX_BANDS)][_band(sy, SY_BANDS)] += 1

    return {
        "schema_version": 1,
        "status": "combined_slip_matrix_coverage_analyzed",
        "offline_only": True,
        "input": str(trace_path),
        "trace": {
            "rows": len(rows),
            "columns": len(rows[0]),
            "first_time_s": valid_times[0],
            "last_time_s": valid_times[-1],
            "dt_s": _stats(dts),
            "nonpositive_dt_rows": sum(dt <= 0.0 for dt in dts),
        },
        "command_phases": phases,
        "slip_coverage_all_wheels": joint_counts,
        "coverage_interpretation": {
            "targeted_longitudinal_band": "0.10<=abs_sx<0.25",
            "targeted_lateral_band": "abs_sy>=0.15",
            "lateral_band_observed": any(
                joint_counts[sx]["abs_sy>=0.15"] > 0 for sx, _, _ in SX_BANDS),
            "combined_slip_force_components_observed": False,
            "recommendation": (
                "Use this trace for repeatability, load-transfer and slip-domain "
                "screening. It is not sufficient to identify separate Fx/Fy; "
                "perform vehicle-level force/moment inversion or add a "
                "diagnostic force export."),
        },
        "provenance": {
            "future_ground_truth_used": False,
            "runtime_controller_consumption": False,
            "simulator_physics_modified": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "rows": report["trace"]["rows"],
        "phases": len(report["command_phases"]),
        "joint_coverage": report["slip_coverage_all_wheels"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

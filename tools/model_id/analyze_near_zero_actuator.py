#!/usr/bin/env python3
"""Summarize near-zero actuator traces without fitting a vehicle model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def number(row: dict, key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def percentile(values: list[float], fraction: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = fraction * (len(finite) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def trace_summary(directory: Path) -> dict:
    dump_path = directory / "simulator_parameters.json"
    trace_path = directory / "wheel_contact_trace.csv"
    with dump_path.open(encoding="utf-8") as stream:
        dump = json.load(stream)
    with trace_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    measurement = [
        row for row in rows
        if number(row, "experiment_measurement_active", 0.0) >= 0.5
    ]
    if len(measurement) < 3:
        raise ValueError(f"not enough measurement rows in {trace_path}")
    times = [number(row, "fixed_time_s") for row in measurement]
    speeds = [number(row, "body_velocity_z_mps") for row in measurement]
    accelerations = []
    for previous, current in zip(measurement, measurement[1:]):
        dt = number(current, "fixed_time_s") - number(previous, "fixed_time_s")
        du = number(current, "body_velocity_z_mps") - number(previous, "body_velocity_z_mps")
        if dt > 0.0 and math.isfinite(du):
            accelerations.append(du / dt)

    motor = []
    brake = []
    rpm = []
    slips = []
    loads = []
    wheel_radius = float(dump["vehicle"]["wheels"][0]["radius"])
    for row in measurement:
        for wheel in range(4):
            motor.append(number(row, f"wheel{wheel}_motor_torque_nm"))
            brake.append(number(row, f"wheel{wheel}_brake_torque_nm"))
            rpm_value = number(row, f"wheel{wheel}_rpm")
            rpm.append(rpm_value)
            slips.append(number(row, f"wheel{wheel}_forward_slip"))
            loads.append(number(row, f"wheel{wheel}_contact_force_n"))
    wheel_speed = [value * 2.0 * math.pi * wheel_radius / 60.0 for value in rpm]
    mismatch = []
    for row in measurement:
        body_speed = number(row, "body_velocity_z_mps")
        wheel_values = [number(row, f"wheel{wheel}_rpm")
                        for wheel in range(4)]
        for value in wheel_values:
            if math.isfinite(value) and math.isfinite(body_speed):
                mismatch.append(value * 2.0 * math.pi * wheel_radius / 60.0 - body_speed)

    metadata = dump["vehicle"]
    requested = number(measurement[0], "experiment_requested_throttle_norm")
    return {
        "directory": str(directory),
        "build_tag": dump.get("simulatorBuildTag"),
        "speed_command_mps": dump.get("experimentInitialSpeedMps"),
        "command_norm": requested,
        "warmup_seconds": dump.get("experimentWarmupSeconds"),
        "warmup_throttle_norm": dump.get("experimentWarmupThrottleNorm"),
        "measurement_seconds": dump.get("experimentMeasurementSeconds"),
        "measurement_rows": len(measurement),
        "measurement_time_s": [times[0], times[-1]],
        "dt_s": {
            "median": percentile([b - a for a, b in zip(times, times[1:])], 0.5),
            "p95": percentile([b - a for a, b in zip(times, times[1:])], 0.95),
        },
        "body_u_mps": {
            "start": speeds[0],
            "end": speeds[-1],
            "median": percentile(speeds, 0.5),
        },
        "body_ax_mps2": {
            "median": percentile(accelerations, 0.5),
            "p05": percentile(accelerations, 0.05),
            "p95": percentile(accelerations, 0.95),
        },
        "actuator": {
            "motor_torque_nm": {
                "mean": mean(motor),
                "min": min((value for value in motor if math.isfinite(value)), default=None),
                "max": max((value for value in motor if math.isfinite(value)), default=None),
            },
            "brake_torque_nm": {
                "mean": mean(brake),
                "min": min((value for value in brake if math.isfinite(value)), default=None),
                "max": max((value for value in brake if math.isfinite(value)), default=None),
            },
            "mode": (
                "brake" if max((value for value in brake if math.isfinite(value)), default=0.0) > 1.0
                else "drive" if max((value for value in motor if math.isfinite(value)), default=0.0) > 1.0e-6
                else "zero_torque"
            ),
        },
        "wheel": {
            "rpm_median": percentile(rpm, 0.5),
            "rpm_p95_abs": percentile([abs(value) for value in rpm], 0.95),
            "wheel_speed_minus_body_u_mps_median": percentile(mismatch, 0.5),
            "forward_slip_median": percentile(slips, 0.5),
            "contact_force_n_median": percentile(loads, 0.5),
        },
        "vehicle_mass_kg": metadata["rigidBody"]["mass"],
        "wheel_radius_m": wheel_radius,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    results = []
    failures = []
    for trace in sorted(root.glob("speed_*/*")):
        if trace.name != "wheel_contact_trace.csv":
            continue
        try:
            results.append(trace_summary(trace.parent))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            failures.append({"directory": str(trace.parent), "error": str(error)})

    conditions = defaultdict(list)
    for result in results:
        key = (result["speed_command_mps"], result["command_norm"])
        conditions[key].append(result)
    condition_rows = []
    for (speed, command), entries in sorted(conditions.items()):
        modes = sorted({entry["actuator"]["mode"] for entry in entries})
        condition_rows.append({
            "speed_command_mps": speed,
            "command_norm": command,
            "repeats": len(entries),
            "modes": modes,
            "body_u_end_mps_mean": mean([
                entry["body_u_mps"]["end"] for entry in entries]),
            "body_ax_mps2_median_mean": mean([
                entry["body_ax_mps2"]["median"] for entry in entries]),
            "motor_torque_max_nm_mean": mean([
                entry["actuator"]["motor_torque_nm"]["max"] for entry in entries]),
            "brake_torque_max_nm_mean": mean([
                entry["actuator"]["brake_torque_nm"]["max"] for entry in entries]),
            "wheel_rpm_median_mean": mean([
                entry["wheel"]["rpm_median"] for entry in entries]),
        })

    result = {
        "schema": "autodrive.near_zero_actuator_analysis.v1",
        "campaign_root": str(root),
        "run_count": len(results),
        "failure_count": len(failures),
        "runs": results,
        "conditions": condition_rows,
        "failures": failures,
        "interpretation": {
            "fit_scope": "actuator_semantics_only; no vehicle-model promotion",
            "measurement_rows_exclude_warmup": True,
            "hard_brake_threshold_nm": 1.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "runs": len(results),
        "failures": len(failures),
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

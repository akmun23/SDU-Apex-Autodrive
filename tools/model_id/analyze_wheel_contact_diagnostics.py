#!/usr/bin/env python3
"""Analyze an offline Unity WheelHit/contact diagnostic trace.

The current Unity stream exposes WheelHit.force (a contact-load magnitude),
contact directions, and slip scalars. It does not expose separate longitudinal
and lateral tire-force magnitudes. This tool therefore reports a clearly
labelled normal-load proxy and refuses to manufacture Fx/Fy or friction curves
that the stream cannot identify.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable


WHEEL_FIELDS = (
    "steer_angle_deg", "rpm", "motor_torque_nm", "brake_torque_nm",
    "grounded", "forward_slip", "sideways_slip", "contact_force_n",
    "forward_dir_x", "forward_dir_y", "forward_dir_z",
    "sideways_dir_x", "sideways_dir_y", "sideways_dir_z",
    "contact_point_x_m", "contact_point_y_m", "contact_point_z_m",
    "contact_normal_x", "contact_normal_y", "contact_normal_z",
)
BASE_FIELDS = (
    "fixed_time_s", "fixed_step", "render_frame",
    "root_position_x_m", "root_position_y_m", "root_position_z_m",
    "world_com_x_m", "world_com_y_m", "world_com_z_m",
    "root_rotation_x", "root_rotation_y", "root_rotation_z",
    "root_rotation_w", "world_velocity_x_mps", "world_velocity_y_mps",
    "world_velocity_z_mps", "body_velocity_x_mps", "body_velocity_y_mps",
    "body_velocity_z_mps", "world_angular_velocity_x_radps",
    "world_angular_velocity_y_radps", "world_angular_velocity_z_radps",
    "body_angular_velocity_x_radps", "body_angular_velocity_y_radps",
    "body_angular_velocity_z_radps", "controller_physics_step",
    "applied_command_sequence", "applied_throttle_norm",
    "applied_steering_norm", "applied_steering_rad",
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


def _stats(values: Iterable[float]) -> dict[str, Any]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "rms": None, "p50": None, "p95": None, "p99": None}

    def quantile(fraction: float) -> float:
        index = fraction * (len(finite) - 1)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return finite[lower]
        weight = index - lower
        return finite[lower] * (1.0 - weight) + finite[upper] * weight

    return {
        "count": len(finite),
        "min": finite[0],
        "max": finite[-1],
        "mean": sum(finite) / len(finite),
        "rms": math.sqrt(sum(value * value for value in finite) / len(finite)),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
    }


def _correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 3:
        return None
    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)
    numerator = sum((a - mean_first) * (b - mean_second)
                    for a, b in zip(first, second))
    first_norm = math.sqrt(sum((a - mean_first) ** 2 for a in first))
    second_norm = math.sqrt(sum((b - mean_second) ** 2 for b in second))
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        return None
    return numerator / (first_norm * second_norm)


def _derivative(values: list[float], dt_s: float) -> list[float]:
    if len(values) < 2 or dt_s <= 0.0:
        return []
    return [(b - a) / dt_s for a, b in zip(values, values[1:])]


def _linear_fit(first: list[float], second: list[float]) -> dict[str, Any]:
    if len(first) != len(second) or len(first) < 3:
        return {"samples": len(first), "intercept": None, "slope": None}
    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)
    denominator = sum((value - mean_first) ** 2 for value in first)
    if denominator <= 1.0e-12:
        return {"samples": len(first), "intercept": mean_second, "slope": None}
    slope = sum((a - mean_first) * (b - mean_second)
                for a, b in zip(first, second)) / denominator
    return {
        "samples": len(first),
        "intercept": mean_second - slope * mean_first,
        "slope": slope,
    }


def _lagged_peak(first: list[float], second: list[float], dt_s: float,
                 max_lag_s: float = 0.10) -> dict[str, Any]:
    if len(first) != len(second) or len(first) < 10 or dt_s <= 0.0:
        return {"lag_s": None, "correlation": None, "tested_lag_s": max_lag_s}
    # The diagnostic stream is normally 1 kHz, while a 1 ms lag grid is not
    # useful for this correlation screen. Downsample to a 5 ms grid so the
    # report remains quick and the reported lag precision matches the signal.
    sample_stride = max(1, int(round(0.005 / dt_s)))
    first = first[::sample_stride]
    second = second[::sample_stride]
    sampled_dt_s = dt_s * sample_stride
    max_lag = min(int(round(max_lag_s / sampled_dt_s)), len(first) // 4)
    candidates: list[tuple[float, int]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = first[-lag:], second[:lag]
        elif lag > 0:
            a, b = first[:-lag], second[lag:]
        else:
            a, b = first, second
        correlation = _correlation(a, b)
        if correlation is not None:
            candidates.append((abs(correlation), lag))
    if not candidates:
        return {"lag_s": None, "correlation": None, "tested_lag_s": max_lag_s}
    _, best_lag = max(candidates)
    if best_lag < 0:
        a, b = first[-best_lag:], second[:best_lag]
    elif best_lag > 0:
        a, b = first[:-best_lag], second[best_lag:]
    else:
        a, b = first, second
    return {
        "lag_s": best_lag * sampled_dt_s,
        "correlation": _correlation(a, b),
        "tested_lag_s": max_lag_s,
        "sample_interval_s": sampled_dt_s,
        "interpretation": (
            "positive lag means second signal follows first; negative lag "
            "means second signal leads first"),
    }


def _speed_bin(speed_mps: float) -> str:
    speed = abs(speed_mps)
    if speed < 1.0:
        return "u<1"
    if speed < 3.0:
        return "1<=u<3"
    if speed < 6.0:
        return "3<=u<6"
    if speed < 10.0:
        return "6<=u<10"
    return "u>=10"


def _paired_report(rows: list[dict[str, str]], first: int, second: int,
                   field: str) -> dict[str, Any]:
    left: list[float] = []
    right: list[float] = []
    differences: list[float] = []
    for row in rows:
        a = _float(row, f"wheel{first}_{field}")
        b = _float(row, f"wheel{second}_{field}")
        if a is None or b is None:
            continue
        left.append(a)
        right.append(b)
        differences.append(a - b)
    return {
        "samples": len(differences),
        "first_minus_second": _stats(differences),
        "correlation": _correlation(left, right),
    }


def _load_transfer_report(rows: list[dict[str, str]], dt_s: float,
                          lateral_acceleration: list[float]) -> dict[str, Any]:
    """Screen measured contact-load transfer without fitting a tire law.

    WheelHit.force is only a load proxy here. Startup suspension spikes and
    airborne rows are excluded so the correlation cannot be dominated by
    invalid contact values. The result is evidence for/against measuring load
    transfer, not a vehicle-model parameter identification.
    """
    longitudinal_speed = [
        _float(row, "body_velocity_z_mps") or 0.0 for row in rows]
    lateral_speed = [
        _float(row, "body_velocity_x_mps") or 0.0 for row in rows]
    yaw_rate = [
        _float(row, "body_angular_velocity_y_radps") or 0.0 for row in rows]
    longitudinal_acceleration = [0.0] * len(rows)
    if dt_s > 0.0:
        dvz = _derivative(longitudinal_speed, dt_s)
        longitudinal_acceleration = [0.0] + [
            acceleration - yaw * lateral
            for acceleration, yaw, lateral in zip(
                dvz, yaw_rate[1:], lateral_speed[1:])]

    left_right_difference: list[float] = []
    front_rear_difference: list[float] = []
    lateral_values: list[float] = []
    longitudinal_values: list[float] = []
    for index, row in enumerate(rows):
        loads: list[float] = []
        valid = True
        for wheel in range(4):
            prefix = f"wheel{wheel}_"
            force = _float(row, prefix + "contact_force_n")
            normal_y = _float(row, prefix + "contact_normal_y")
            if (_float(row, prefix + "grounded") != 1.0 or
                    force is None or normal_y is None):
                valid = False
                break
            load = force * normal_y
            if not math.isfinite(load) or load < 0.0 or load > 30.0:
                valid = False
                break
            loads.append(load)
        if not valid:
            continue
        left_right_difference.append((loads[0] + loads[2]) -
                                     (loads[1] + loads[3]))
        front_rear_difference.append((loads[0] + loads[1]) -
                                     (loads[2] + loads[3]))
        lateral_values.append(lateral_acceleration[index])
        longitudinal_values.append(longitudinal_acceleration[index])
    return {
        "samples": len(left_right_difference),
        "filter": {
            "all_four_wheels_grounded": True,
            "normal_load_proxy_range_n": [0.0, 30.0],
            "startup_and_contact_spikes_excluded": True,
        },
        "left_right_load_difference_n": _stats(left_right_difference),
        "front_rear_load_difference_n": _stats(front_rear_difference),
        "body_lateral_acceleration_mps2": _stats(lateral_values),
        "body_longitudinal_acceleration_mps2": _stats(longitudinal_values),
        "correlation": {
            "left_right_load_difference_vs_lateral_acceleration": _correlation(
                left_right_difference, lateral_values),
            "front_rear_load_difference_vs_longitudinal_acceleration": _correlation(
                front_rear_difference, longitudinal_values),
        },
        "linear_fit": {
            "left_right_load_difference_from_lateral_acceleration": _linear_fit(
                lateral_values, left_right_difference),
            "front_rear_load_difference_from_longitudinal_acceleration": _linear_fit(
                longitudinal_values, front_rear_difference),
        },
        "interpretation": (
            "Contact-load transfer is measurable in this diagnostic trace, but "
            "WheelHit.force is a load proxy and these regressions do not identify "
            "a tire-force law or a reduced-model height."),
    }


def _ackermann_report(rows: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sign_name, predicate in (("positive", lambda value: value > 0.02),
                                 ("negative", lambda value: value < -0.02)):
        front_left: list[float] = []
        front_right: list[float] = []
        for row in rows:
            command = _float(row, "applied_steering_rad")
            left = _float(row, "wheel0_steer_angle_deg")
            right = _float(row, "wheel1_steer_angle_deg")
            if command is None or left is None or right is None:
                continue
            if predicate(command):
                front_left.append(left)
                front_right.append(right)
        result[sign_name] = {
            "samples": len(front_left),
            "front_left_deg": _stats(front_left),
            "front_right_deg": _stats(front_right),
            "left_minus_right_deg": _stats(
                [left - right for left, right in zip(front_left, front_right)]),
            "larger_magnitude_wheel": (
                "front_left" if sum(abs(v) for v in front_left) >
                sum(abs(v) for v in front_right) else "front_right"),
        }
    return result


def analyze(trace_path: Path, dump_path: Path | None = None) -> dict[str, Any]:
    with trace_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("trace has no CSV header")
        expected = list(BASE_FIELDS)
        for wheel in range(4):
            expected.extend(f"wheel{wheel}_{field}" for field in WHEEL_FIELDS)
        missing = sorted(set(expected) - set(reader.fieldnames))
        extra = sorted(set(reader.fieldnames) - set(expected))
        if missing:
            raise ValueError(f"trace is missing required columns: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError("trace contains no data rows")
    times = [_float(row, "fixed_time_s") for row in rows]
    times = [value for value in times if value is not None]
    if len(times) != len(rows):
        raise ValueError("trace contains non-finite fixed_time_s")
    dts = [b - a for a, b in zip(times, times[1:])]
    positive_dts = [dt for dt in dts if dt > 0.0]
    dt_s = median(positive_dts) if positive_dts else 0.0
    steps = [_float(row, "fixed_step") for row in rows]
    step_gaps = [b - a for a, b in zip(steps, steps[1:])
                 if a is not None and b is not None]
    render_frames = [_float(row, "render_frame") for row in rows]
    speed = [_float(row, "body_velocity_z_mps") or 0.0 for row in rows]
    yaw_rate = [_float(row, "body_angular_velocity_y_radps") or 0.0
                for row in rows]
    lateral_velocity = [_float(row, "body_velocity_x_mps") or 0.0
                        for row in rows]
    lateral_acceleration = [0.0] * len(rows)
    if dt_s > 0.0:
        dvx = _derivative(lateral_velocity, dt_s)
        lateral_acceleration = [0.0] + [
            acceleration + yaw * longitudinal
            for acceleration, yaw, longitudinal in zip(
                dvx, yaw_rate[1:], speed[1:])]
    yaw_acceleration = [0.0] * len(rows)
    if dt_s > 0.0:
        dry = _derivative(yaw_rate, dt_s)
        yaw_acceleration = [0.0] + dry

    wheel_reports: dict[str, Any] = {}
    for wheel in range(4):
        prefix = f"wheel{wheel}_"
        grounded_rows = [row for row in rows
                         if _float(row, prefix + "grounded") == 1.0]
        load: list[float] = []
        raw_force: list[float] = []
        forward_slip: list[float] = []
        sideways_slip: list[float] = []
        rpm: list[float] = []
        load_for_forward_slip: list[float] = []
        load_for_sideways_slip: list[float] = []
        for row in grounded_rows:
            force = _float(row, prefix + "contact_force_n")
            normal_y = _float(row, prefix + "contact_normal_y")
            if force is not None:
                raw_force.append(force)
                if normal_y is not None:
                    load_value = max(0.0, force * normal_y)
                    load.append(load_value)
                    forward_value = _float(row, prefix + "forward_slip")
                    sideways_value = _float(row, prefix + "sideways_slip")
                    if forward_value is not None:
                        load_for_forward_slip.append(load_value)
                    if sideways_value is not None:
                        load_for_sideways_slip.append(load_value)
            for target, field in ((forward_slip, "forward_slip"),
                                  (sideways_slip, "sideways_slip"),
                                  (rpm, "rpm")):
                value = _float(row, prefix + field)
                if value is not None:
                    target.append(value)
        by_speed: dict[str, dict[str, list[float]]] = {}
        for row in grounded_rows:
            bin_name = _speed_bin(_float(row, "body_velocity_z_mps") or 0.0)
            bucket = by_speed.setdefault(bin_name, {
                "load_n": [], "forward_slip": [], "sideways_slip": []})
            force = _float(row, prefix + "contact_force_n")
            normal_y = _float(row, prefix + "contact_normal_y")
            if force is not None and normal_y is not None:
                bucket["load_n"].append(max(0.0, force * normal_y))
            for target, field in ((bucket["forward_slip"], "forward_slip"),
                                  (bucket["sideways_slip"], "sideways_slip")):
                value = _float(row, prefix + field)
                if value is not None:
                    target.append(value)
        wheel_reports[f"wheel_{wheel}"] = {
            "rows": len(rows),
            "grounded_rows": len(grounded_rows),
            "grounded_fraction": len(grounded_rows) / len(rows),
            "contact_force_n": _stats(raw_force),
            "normal_load_proxy_n": _stats(load),
            "forward_slip": _stats(forward_slip),
            "sideways_slip": _stats(sideways_slip),
            "rpm": _stats(rpm),
            "by_speed_bin": {
                name: {
                    "normal_load_proxy_n": _stats(values["load_n"]),
                    "forward_slip": _stats(values["forward_slip"]),
                    "sideways_slip": _stats(values["sideways_slip"]),
                }
                for name, values in sorted(by_speed.items())
            },
            "load_slip_correlation": {
                "normal_load_vs_abs_forward_slip": _correlation(
                    [abs(value) for value in load_for_forward_slip],
                    [abs(value) for value in forward_slip]),
                "normal_load_vs_abs_sideways_slip": _correlation(
                    [abs(value) for value in load_for_sideways_slip],
                    [abs(value) for value in sideways_slip]),
            },
        }

    lag_reports: dict[str, Any] = {}
    steering = [_float(row, "applied_steering_rad") or 0.0 for row in rows]
    steering_rate = _derivative(steering, dt_s)
    for wheel in range(4):
        prefix = f"wheel{wheel}_"
        wheel_slip = [_float(row, prefix + "sideways_slip") or 0.0
                      for row in rows]
        wheel_load = []
        for row in rows:
            force = _float(row, prefix + "contact_force_n") or 0.0
            normal_y = _float(row, prefix + "contact_normal_y") or 0.0
            wheel_load.append(max(0.0, force * normal_y))
        slip_change = _derivative(wheel_slip, dt_s)
        load_change = _derivative(wheel_load, dt_s)
        lag_reports[f"wheel_{wheel}"] = {
            "steering_rate_vs_load_change": _lagged_peak(
                steering_rate, load_change, dt_s),
            "sideways_slip_vs_lateral_acceleration": _lagged_peak(
                wheel_slip, lateral_acceleration, dt_s),
            "sideways_slip_vs_yaw_acceleration": _lagged_peak(
                wheel_slip, yaw_acceleration, dt_s),
            "note": (
                "These are diagnostic correlations, not proof of tire-force "
                "lag. Grounding transitions and load-proxy noise can dominate."),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "wheel_contact_diagnostics_analyzed",
        "input": str(trace_path),
        "static_dump": str(dump_path) if dump_path else None,
        "trace_schema": {
            "expected_columns": len(expected),
            "actual_columns": len(reader.fieldnames),
            "missing_columns": missing,
            "extra_columns": extra,
            "rows": len(rows),
        },
        "timing": {
            "first_fixed_time_s": times[0],
            "last_fixed_time_s": times[-1],
            "median_dt_s": dt_s,
            "dt": _stats(dts),
            "positive_dt_rows": len(positive_dts),
            "nonpositive_dt_rows": sum(dt <= 0.0 for dt in dts),
            "fixed_step_gap": _stats(step_gaps),
            "render_frame": _stats(render_frames),
        },
        "vehicle": {
            "forward_speed_body_z_mps": _stats(speed),
            "lateral_speed_body_x_mps": _stats(lateral_velocity),
            "yaw_rate_body_y_radps": _stats(yaw_rate),
            "lateral_acceleration_body_x_mps2": _stats(lateral_acceleration),
            "yaw_acceleration_body_y_radps2": _stats(yaw_acceleration),
        },
        "inputs": {
            "throttle": _stats([_float(row, "applied_throttle_norm") or 0.0
                                 for row in rows]),
            "steering": _stats(steering),
            "ackermann_side_test": _ackermann_report(rows),
        },
        "wheel_reports": wheel_reports,
        "contact_load_transfer_diagnostics": _load_transfer_report(
            rows, dt_s, lateral_acceleration),
        "wheel_state_independence": {
            "front_left_vs_rear_left_rpm": _paired_report(rows, 0, 2, "rpm"),
            "front_right_vs_rear_right_rpm": _paired_report(rows, 1, 3, "rpm"),
            "left_vs_right_front_sideways_slip": _paired_report(
                rows, 0, 1, "sideways_slip"),
            "left_vs_right_rear_sideways_slip": _paired_report(
                rows, 2, 3, "sideways_slip"),
        },
        "force_identification": {
            "separate_fx_fy_available": False,
            "available_contact_force": "WheelHit.force scalar",
            "normal_load_proxy": "max(0, WheelHit.force * contact_normal_y)",
            "friction_curve_outputs": False,
            "reason": (
                "The Unity trace has slip scalars and WheelHit.force, but not "
                "separate longitudinal/lateral tire-force magnitudes. A force "
                "vector cannot be reconstructed from WheelHit.force alone."),
            "required_next_export": (
                "Log per-wheel tire-force components if the simulator exposes "
                "them; otherwise use vehicle acceleration/moment inversion "
                "with explicit assumptions and label it effective."),
        },
        "force_lag_diagnostics": lag_reports,
        "provenance": {
            "offline_only": True,
            "runtime_controller_consumption": False,
            "simulator_modified": False,
            "future_ground_truth_used": False,
        },
        "next_action": (
            "Use this trace to validate timing, steering-side behavior, wheel "
            "state independence, and load/lag diagnostics. Do not fit Fx/Fy "
            "curve parameters until separate tire-force observations exist."),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.trace, args.dump)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()

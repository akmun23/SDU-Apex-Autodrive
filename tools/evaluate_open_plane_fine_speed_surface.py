#!/usr/bin/env python3
"""Cross-speed held-out test of the measured 4--5 m/s steering transition.

The 4.0 and 5.0 m/s captures train a signed yaw-rate / lateral-acceleration
response surface using repetitions 1--2. All 4.5 m/s repetitions are unseen.
This is a development-data model check, not a driving controller or force
identification test.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as analysis
from tools import evaluate_open_plane_highspeed_crossfactor as crossfactor
from tools import evaluate_open_plane_yaw_spline as yaw_spline
from tools.fit_open_plane_tire_model import _load_samples


SOURCE_SPEEDS_MPS = (4.0, 4.25, 4.75, 5.0)
HOLDOUT_SPEED_MPS = 4.5
EXPECTED_ANGLE_COUNTS = {4.0: 23, 4.25: 24, 4.5: 23,
                         4.75: 24, 5.0: 23}
SUPPORT_ANGLES_RAD = (0.125, 0.275, 0.30, 0.35, 0.42, 0.46, 0.50)
CONFIGURED_GAIN_PER_M = 2.95
MAX_YAW_RATE_RMSE_RPS = 0.05
MIN_CONFIGURED_RMSE_REDUCTION = 0.50
MIN_JACOBIAN_MATCH_FRACTION = 0.90
MIN_RESOLVED_SLOPE_RPS_PER_RAD = 0.10


def _validated_blocks(path: Path, speed: float) -> tuple[list[dict], tuple[float, ...]]:
    rows, run_end = crossfactor.load_blocks(path)
    selected = [row for row in rows
                if row["mode"] == "speed_hold"
                and abs(row["speed_target"] - speed) < 1e-6]
    angles = tuple(sorted({round(abs(row["steering_command"]), 4)
                           for row in selected}))
    expected = {(repetition, sign, angle)
                for repetition in (1, 2, 3)
                for sign in (-1, 1)
                for angle in angles}
    indexed = {
        (row["repetition"],
         1 if row["steering_command"] > 0.0 else -1,
         round(abs(row["steering_command"]), 4)): row
        for row in selected
    }
    if (len(angles) != EXPECTED_ANGLE_COUNTS[speed]
            or len(indexed) != len(selected)
            or set(indexed) != expected):
        raise ValueError(f"{speed:.1f} m/s capture is incomplete: "
                         f"{len(indexed)}/{len(expected)} matched blocks")
    if (run_end.get("aborted", True)
            or run_end.get("quality_failures", []) not in ([], 0, None)
            or any(row["valid"] is not True for row in selected)):
        raise ValueError(f"{speed:.1f} m/s capture reports an abort or invalid block")
    for row in selected:
        if (not math.isfinite(row["start_speed"])
                or abs(row["start_speed"] - speed) > 0.20
                or abs(row["speed"] - speed) > 0.20):
            raise ValueError(f"unmatched speed in {row['label']}")
    return selected, angles


def _validated_support_blocks(path: Path, speed: float) -> list[dict]:
    with contextlib.redirect_stdout(io.StringIO()):
        quality_status = analysis.analyze(path)
    if quality_status != 0:
        raise ValueError(f"support capture failed data-quality analysis: {path}")
    rows, run_end = crossfactor.load_blocks(path)
    selected = [row for row in rows
                if row["mode"] == "speed_hold"
                and abs(row["speed_target"] - speed) < 1e-6
                and "_support_steer_" in row["label"]]
    expected = {(repetition, sign, angle)
                for repetition in (1, 2, 3)
                for sign in (-1, 1)
                for angle in SUPPORT_ANGLES_RAD}
    indexed = {
        (row["repetition"],
         1 if row["steering_command"] > 0.0 else -1,
         round(abs(row["steering_command"]), 4)): row
        for row in selected
    }
    if (len(indexed) != len(selected) or set(indexed) != expected
            or run_end.get("aborted", True)
            or run_end.get("quality_failures", []) not in ([], 0, None)
            or any(row["valid"] is not True for row in selected)):
        raise ValueError(f"invalid steering-demand support capture: {path}")
    for row in selected:
        if (not math.isfinite(row["start_speed"])
                or abs(row["start_speed"] - speed) > 0.20
                or abs(row["speed"] - speed) > 0.20):
            raise ValueError(f"unmatched support speed in {row['label']}")
    return selected


def _source_curves(rows: list[dict], speed: float, sign: int,
                   response: str) -> tuple[np.ndarray, np.ndarray]:
    values: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if (row["repetition"] not in (1, 2)
                or (1 if row["steering_command"] > 0.0 else -1) != sign
                or abs(row["speed_target"] - speed) >= 1e-6):
            continue
        angle = round(abs(row["steering_command"]), 4)
        signed_rate = sign * row["yaw_rate"]
        if response == "yaw_rate":
            value = signed_rate
        elif response == "lateral_acceleration":
            value = row["forward_speed"] * signed_rate
        else:
            raise ValueError(f"unknown response: {response}")
        values[angle].append(value)

    angles = np.asarray(sorted(values), dtype=float)
    response_values = np.asarray(
        [statistics.mean(values[angle]) for angle in angles], dtype=float)
    if (len(angles) < 10 or not np.all(np.isfinite(response_values))
            or any(not left < right for left, right in zip(angles, angles[1:]))):
        raise ValueError(f"invalid {response} source curve at {speed:.1f} m/s")
    return angles, response_values


def _surface_value(curves: dict[tuple[float, int, str], tuple[np.ndarray, np.ndarray]],
                   speed: float, sign: int, steering: float,
                   response: str) -> tuple[float, float]:
    angles, values = curves[(speed, sign, response)]
    return yaw_spline._pchip_value_and_slope(angles, values, abs(steering))


def _blend(curves: dict[tuple[float, int, str], tuple[np.ndarray, np.ndarray]],
           speed: float, sign: int,
           steering: float, response: str) -> tuple[float, float]:
    if not SOURCE_SPEEDS_MPS[0] <= speed <= SOURCE_SPEEDS_MPS[-1]:
        raise ValueError(f"holdout speed {speed:.3f} outside "
                         f"[{SOURCE_SPEEDS_MPS[0]}, {SOURCE_SPEEDS_MPS[-1]}]")
    values, steering_slopes = [], []
    for source_speed in SOURCE_SPEEDS_MPS:
        value, slope = _surface_value(
            curves, source_speed, sign, steering, response)
        values.append(value)
        steering_slopes.append(slope)
    speed_knots = np.asarray(SOURCE_SPEEDS_MPS, dtype=float)
    value, _ = yaw_spline._pchip_value_and_slope(
        speed_knots, np.asarray(values), speed)
    slope, _ = yaw_spline._pchip_value_and_slope(
        speed_knots, np.asarray(steering_slopes), speed)
    return value, slope


def _rmse(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(statistics.mean((a - p) ** 2
                                     for a, p in zip(actual, predicted)))


def _demand_curve(rows_by_speed: dict[float, list[dict]], sign: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Fit lateral acceleration against speed-scaled steering demand."""
    values: dict[float, list[float]] = defaultdict(list)
    for speed in SOURCE_SPEEDS_MPS:
        for row in rows_by_speed[speed]:
            if (row["repetition"] not in (1, 2)
                    or (1 if row["steering_command"] > 0.0 else -1) != sign):
                continue
            forward_speed = row["forward_speed"]
            steering = abs(row["steering_command"])
            demand = forward_speed * math.tan(steering)
            lateral_acceleration = sign * forward_speed * row["yaw_rate"]
            if not all(math.isfinite(value) for value in
                       (demand, lateral_acceleration)):
                continue
            knot = round(demand / 0.01) * 0.01
            values[knot].append(lateral_acceleration)
    knots = np.asarray(sorted(values), dtype=float)
    response = np.asarray(
        [statistics.mean(values[knot]) for knot in knots], dtype=float)
    if len(knots) < 10 or not np.all(np.isfinite(response)):
        raise ValueError(f"insufficient steering-demand support for sign {sign:+d}")
    return knots, response


def _demand_speed_curves(rows_by_speed: dict[float, list[dict]], sign: int
                         ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Keep speed dependence while expressing steering as lateral demand."""
    curves = {}
    for speed in SOURCE_SPEEDS_MPS:
        values: dict[float, list[float]] = defaultdict(list)
        for row in rows_by_speed[speed]:
            if (row["repetition"] not in (1, 2)
                    or (1 if row["steering_command"] > 0.0 else -1) != sign):
                continue
            demand = row["forward_speed"] * math.tan(
                abs(row["steering_command"]))
            response = sign * row["forward_speed"] * row["yaw_rate"]
            if math.isfinite(demand) and math.isfinite(response):
                knot = round(demand / 0.005) * 0.005
                values[knot].append(response)
        knots = np.asarray(sorted(values), dtype=float)
        response = np.asarray(
            [statistics.mean(values[knot]) for knot in knots], dtype=float)
        if len(knots) < 10 or not np.all(np.isfinite(response)):
            raise ValueError(f"insufficient demand-curve support at "
                             f"{speed:.2f} m/s, sign {sign:+d}")
        curves[speed] = (knots, response)
    return curves


def _demand_surface_value(
    curves: dict[float, tuple[np.ndarray, np.ndarray]],
    speed: float, demand: float,
) -> tuple[float, float]:
    responses, demand_slopes = [], []
    for source_speed in SOURCE_SPEEDS_MPS:
        knots, values = curves[source_speed]
        response, slope = yaw_spline._pchip_value_and_slope(
            knots, values, demand)
        responses.append(response)
        demand_slopes.append(slope)
    response, _ = yaw_spline._pchip_value_and_slope(
        np.asarray(SOURCE_SPEEDS_MPS), np.asarray(responses), speed)
    slope, _ = yaw_spline._pchip_value_and_slope(
        np.asarray(SOURCE_SPEEDS_MPS), np.asarray(demand_slopes), speed)
    return response, slope


def _hybrid_yaw_rate(
    speed: float, steering: float, sign: int,
    steering_curves: dict[tuple[float, int, str], tuple[np.ndarray, np.ndarray]],
    demand_curves: dict[int, dict[float, tuple[np.ndarray, np.ndarray]]],
    demand_low: float, demand_high: float,
) -> float:
    angle = abs(steering)
    demand = speed * math.tan(angle)
    demand = min(max(demand, demand_low), demand_high)
    demand_acceleration, _ = _demand_surface_value(
        demand_curves[sign], speed, demand)
    angle_acceleration, _ = _blend(
        steering_curves, speed, sign, angle, "lateral_acceleration")
    if angle <= 0.25:
        return demand_acceleration / speed
    if angle >= 0.275:
        return angle_acceleration / speed
    blend = (angle - 0.25) / 0.025
    blend = blend * blend * (3.0 - 2.0 * blend)
    return ((1.0 - blend) * demand_acceleration
            + blend * angle_acceleration) / speed


def _slip_response_by_phase(path: Path) -> dict[str, tuple[int, float, float, float]]:
    """Return repetition, front |Sy|, |u*r|, and speed per measured phase."""
    grouped: dict[str, list] = defaultdict(list)
    for sample in _load_samples(path):
        grouped[sample.phase].append(sample)
    result = {}
    for phase, samples in grouped.items():
        slip = statistics.median(
            0.5 * (abs(sample.slips[0]) + abs(sample.slips[1]))
            for sample in samples)
        lateral_acceleration = statistics.median(
            abs(sample.u_mps * sample.yaw_rate_rps) for sample in samples)
        speed = statistics.median(abs(sample.u_mps) for sample in samples)
        if all(math.isfinite(value) for value in
               (slip, lateral_acceleration, speed)) and speed > 1.0:
            result[phase] = (samples[0].repetition, slip,
                             lateral_acceleration, speed)
    return result


def evaluate(source_4mps: Path, source_425mps: Path,
             source_475mps: Path, source_5mps: Path,
             holdout_45mps: Path,
             support_paths: dict[float, Path] | None = None,
             holdout_support_path: Path | None = None) -> bool:
    source_paths = {4.0: source_4mps, 4.25: source_425mps,
                    4.75: source_475mps, 5.0: source_5mps}
    captures = tuple((source_paths[speed], speed)
                     for speed in SOURCE_SPEEDS_MPS) + (
                         (holdout_45mps, HOLDOUT_SPEED_MPS),)
    for path, speed in captures:
        with contextlib.redirect_stdout(io.StringIO()):
            quality_status = analysis.analyze(path)
        if quality_status != 0:
            raise ValueError(f"data-quality analysis failed for {path}")
    rows_by_speed = {}
    for path, speed in captures[:-1]:
        rows_by_speed[speed], _ = _validated_blocks(path, speed)
    holdout, _ = _validated_blocks(holdout_45mps, HOLDOUT_SPEED_MPS)
    if holdout_support_path is not None:
        holdout.extend(
            row for row in _validated_support_blocks(
                holdout_support_path, HOLDOUT_SPEED_MPS)
            if abs(row["steering_command"]) >= 0.275
    )
    if support_paths:
        if set(support_paths) != set(SOURCE_SPEEDS_MPS):
            raise ValueError("provide one support capture for every source speed")
        for speed, path in support_paths.items():
            rows_by_speed[speed].extend(_validated_support_blocks(path, speed))
    curves = {
        (speed, sign, response): _source_curves(
            rows_by_speed[speed], speed, sign, response)
        for speed in SOURCE_SPEEDS_MPS
        for sign in (-1, 1)
        for response in ("yaw_rate", "lateral_acceleration")
    }

    actual_rate: list[float] = []
    configured_rate: list[float] = []
    direct_rate: list[float] = []
    acceleration_rate: list[float] = []
    for row in holdout:
        sign = 1 if row["steering_command"] > 0.0 else -1
        steering = abs(row["steering_command"])
        speed = row["forward_speed"]
        rate = sign * row["yaw_rate"]
        direct, _ = _blend(curves, speed, sign, steering, "yaw_rate")
        lateral_acceleration, _ = _blend(
            curves, speed, sign, steering, "lateral_acceleration")
        actual_rate.append(rate)
        direct_rate.append(direct)
        acceleration_rate.append(lateral_acceleration / speed)
        configured_rate.append(speed * math.tan(steering) * CONFIGURED_GAIN_PER_M)

    direct_rmse = _rmse(actual_rate, direct_rate)
    acceleration_rmse = _rmse(actual_rate, acceleration_rate)
    configured_rmse = _rmse(actual_rate, configured_rate)
    reduction = 1.0 - min(direct_rmse, acceleration_rmse) / configured_rmse

    demand_curves = {
        sign: _demand_curve(rows_by_speed, sign) for sign in (-1, 1)
    }
    demand_actual: list[float] = []
    demand_predicted: list[float] = []
    for row in holdout:
        sign = 1 if row["steering_command"] > 0.0 else -1
        speed = row["forward_speed"]
        demand = speed * math.tan(abs(row["steering_command"]))
        knots, values = demand_curves[sign]
        if not knots[0] <= demand <= knots[-1]:
            raise ValueError(f"steering-demand holdout requires extrapolation: "
                             f"{demand:.4f} outside "
                             f"[{knots[0]:.4f}, {knots[-1]:.4f}]")
        lateral_acceleration, _ = yaw_spline._pchip_value_and_slope(
            knots, values, demand)
        demand_actual.append(sign * row["yaw_rate"])
        demand_predicted.append(lateral_acceleration / speed)
    demand_rmse = _rmse(demand_actual, demand_predicted)

    demand_slope_matches = 0
    demand_slope_count = 0
    for sign in (-1, 1):
        by_angle: dict[float, list[dict]] = defaultdict(list)
        for row in holdout:
            if (1 if row["steering_command"] > 0.0 else -1) == sign:
                by_angle[round(abs(row["steering_command"]), 4)].append(row)
        angles = sorted(by_angle)
        knots, values = demand_curves[sign]
        for left, right in zip(angles, angles[1:]):
            left_rows, right_rows = by_angle[left], by_angle[right]
            measured_left = statistics.mean(sign * row["yaw_rate"]
                                            for row in left_rows)
            measured_right = statistics.mean(sign * row["yaw_rate"]
                                             for row in right_rows)
            measured_slope = (measured_right - measured_left) / (right - left)
            if abs(measured_slope) < MIN_RESOLVED_SLOPE_RPS_PER_RAD:
                continue
            midpoint = 0.5 * (left + right)
            speed = statistics.mean(
                row["forward_speed"] for row in left_rows + right_rows)
            demand = speed * math.tan(midpoint)
            _acceleration, slope = yaw_spline._pchip_value_and_slope(
                knots, values, demand)
            predicted_slope = slope / (math.cos(midpoint) ** 2)
            demand_slope_count += 1
            demand_slope_matches += measured_slope * predicted_slope > 0.0

    slip_by_path = {
        path: _slip_response_by_phase(path)
        for path, _ in captures
    }
    training_slips: dict[float, list[float]] = defaultdict(list)
    training_accelerations: dict[float, list[float]] = defaultdict(list)
    for speed in SOURCE_SPEEDS_MPS:
        source_path = source_paths[speed]
        slip_rows = slip_by_path[source_path]
        for row in rows_by_speed[speed]:
            if (row["repetition"] not in (1, 2)
                    or "_support_steer_" in row["label"]):
                continue
            phase = slip_rows.get(row["label"])
            if phase is None:
                raise ValueError(f"front-slip features are missing for {row['label']}")
            _, slip, lateral_acceleration, _ = phase
            knot = round(slip, 3)
            training_slips[knot].append(slip)
            training_accelerations[knot].append(lateral_acceleration)
    slip_knots = np.asarray(sorted(training_slips), dtype=float)
    acceleration_knots = np.asarray([
        statistics.mean(training_accelerations[knot])
        for knot in sorted(training_slips)
    ], dtype=float)
    if len(slip_knots) < 10:
        raise ValueError("insufficient front-slip support for the tire-response fit")

    holdout_slip: list[tuple[dict, float, float, float]] = []
    for row in holdout:
        if "_support_steer_" in row["label"]:
            continue
        phase = slip_by_path[holdout_45mps].get(row["label"])
        if phase is None:
            raise ValueError(f"front-slip features are missing for {row['label']}")
        _, slip, lateral_acceleration, speed = phase
        if not slip_knots[0] <= slip <= slip_knots[-1]:
            raise ValueError(f"front-slip holdout requires extrapolation: {slip:.5f} "
                             f"outside [{slip_knots[0]:.5f}, {slip_knots[-1]:.5f}]")
        predicted_acceleration, _ = yaw_spline._pchip_value_and_slope(
            slip_knots, acceleration_knots, slip)
        holdout_slip.append((row, lateral_acceleration, predicted_acceleration,
                             speed))
    slip_actual_rate = [row["forward_speed"] * abs(row["yaw_rate"])
                        / row["forward_speed"] for row, _, _, _ in holdout_slip]
    slip_predicted_rate = [predicted_acceleration / speed
                           for _, _, predicted_acceleration, speed in holdout_slip]
    slip_rmse = _rmse(slip_actual_rate, slip_predicted_rate)
    slip_baseline_rmse = _rmse(
        slip_actual_rate,
        [row["forward_speed"] * math.tan(abs(row["steering_command"]))
         * CONFIGURED_GAIN_PER_M for row, _, _, _ in holdout_slip])

    slip_slope_matches = 0
    slip_slope_count = 0
    for sign in (-1, 1):
        by_angle: dict[float, list[tuple[float, float]]] = defaultdict(list)
        for row, _, predicted_acceleration, speed in holdout_slip:
            if (1 if row["steering_command"] > 0.0 else -1) == sign:
                angle = round(abs(row["steering_command"]), 4)
                by_angle[angle].append((abs(row["yaw_rate"]),
                                        predicted_acceleration / speed))
        angles = sorted(by_angle)
        for left, right in zip(angles, angles[1:]):
            measured_left = statistics.mean(value[0] for value in by_angle[left])
            measured_right = statistics.mean(value[0] for value in by_angle[right])
            measured_slope = (measured_right - measured_left) / (right - left)
            if abs(measured_slope) < MIN_RESOLVED_SLOPE_RPS_PER_RAD:
                continue
            predicted_left = statistics.mean(value[1] for value in by_angle[left])
            predicted_right = statistics.mean(value[1] for value in by_angle[right])
            predicted_slope = (predicted_right - predicted_left) / (right - left)
            slip_slope_count += 1
            slip_slope_matches += measured_slope * predicted_slope > 0.0

    # Compare local slope signs on the unseen 4.5 m/s curves. Repetitions are
    # averaged by sign and knot; slopes smaller than the declared noise floor
    # are not assigned a sign.
    slopes: dict[str, list[tuple[float, float]]] = {
        "direct yaw rate": [], "lateral acceleration": []}
    for sign in (-1, 1):
        by_angle: dict[float, list[dict]] = defaultdict(list)
        for row in holdout:
            if (1 if row["steering_command"] > 0.0 else -1) == sign:
                by_angle[round(abs(row["steering_command"]), 4)].append(row)
        angles = sorted(by_angle)
        for left, right in zip(angles, angles[1:]):
            left_rows, right_rows = by_angle[left], by_angle[right]
            measured_left = statistics.mean(sign * row["yaw_rate"]
                                            for row in left_rows)
            measured_right = statistics.mean(sign * row["yaw_rate"]
                                             for row in right_rows)
            measured_slope = (measured_right - measured_left) / (right - left)
            if abs(measured_slope) < MIN_RESOLVED_SLOPE_RPS_PER_RAD:
                continue
            midpoint = 0.5 * (left + right)
            midpoint_speed = statistics.mean(
                row["forward_speed"] for row in left_rows + right_rows)
            _, direct_slope = _blend(
                curves, midpoint_speed, sign, midpoint, "yaw_rate")
            _, accel_slope = _blend(
                curves, midpoint_speed, sign, midpoint,
                "lateral_acceleration")
            slopes["direct yaw rate"].append((measured_slope, direct_slope))
            slopes["lateral acceleration"].append(
                (measured_slope, accel_slope / midpoint_speed))

    matches = {
        name: sum(actual_slope * predicted_slope > 0.0
                  for actual_slope, predicted_slope in values)
        for name, values in slopes.items()
    }
    sign_fractions = {
        name: matches[name] / len(values) if values else 0.0
        for name, values in slopes.items()
    }
    yaw_gate = min(direct_rmse, acceleration_rmse) <= MAX_YAW_RATE_RMSE_RPS
    jacobian_gate = max(sign_fractions.values()) >= MIN_JACOBIAN_MATCH_FRACTION

    print("training: four local speed knots (4.00, 4.25, 4.75, 5.00 m/s); "
          "only repetitions 1--2")
    print(f"sources: {', '.join(str(source_paths[speed]) for speed in SOURCE_SPEEDS_MPS)}")
    print(f"unseen speed holdout: {holdout_45mps}; 4.5 m/s, all three repetitions")
    print(f"held-out blocks: {len(holdout)}; yaw-rate RMSE (direct / ay blend / current)="
          f"{direct_rmse:.5f} / {acceleration_rmse:.5f} / {configured_rmse:.5f} rad/s")
    print(f"current-model yaw-rate error reduction: {reduction:+.1%}; "
          f"max RMSE threshold={MAX_YAW_RATE_RMSE_RPS:.2f} rad/s")
    for name, values in slopes.items():
        print(f"{name} local Jacobian signs: {matches[name]}/{len(values)} "
              f"({sign_fractions[name]:.1%}; threshold="
              f"{MIN_JACOBIAN_MATCH_FRACTION:.0%})")
    demand_slope_fraction = (demand_slope_matches / demand_slope_count
                             if demand_slope_count else 0.0)
    print("speed-scaled steering-demand model (train speeds only; "
          "q=u*tan(|steer|), response=|u*r|): "
          f"yaw-rate RMSE={demand_rmse:.5f} rad/s; local Jacobian signs="
          f"{demand_slope_matches}/{demand_slope_count} "
          f"({demand_slope_fraction:.1%})")

    demand_speed_curves = {
        sign: _demand_speed_curves(rows_by_speed, sign) for sign in (-1, 1)
    }
    support_low = max(curves[speed][0][0]
                      for curves in demand_speed_curves.values()
                      for speed in SOURCE_SPEEDS_MPS)
    support_high = min(curves[speed][0][-1]
                       for curves in demand_speed_curves.values()
                       for speed in SOURCE_SPEEDS_MPS)
    support_by_speed = {
        speed: (
            max(demand_speed_curves[sign][speed][0][0] for sign in (-1, 1)),
            min(demand_speed_curves[sign][speed][0][-1] for sign in (-1, 1)),
        )
        for speed in SOURCE_SPEEDS_MPS
    }
    demand_test_rows = [row for row in holdout
                        if 0.15 <= abs(row["steering_command"]) <= 0.25]
    demand_surface_actual: list[float] = []
    demand_surface_predicted: list[float] = []
    supported_holdout = []
    for row in demand_test_rows:
        sign = 1 if row["steering_command"] > 0.0 else -1
        speed = row["forward_speed"]
        demand = speed * math.tan(abs(row["steering_command"]))
        if support_low <= demand <= support_high:
            acceleration, _ = _demand_surface_value(
                demand_speed_curves[sign], speed, demand)
            demand_surface_actual.append(sign * row["yaw_rate"])
            demand_surface_predicted.append(acceleration / speed)
            supported_holdout.append(row)
    demand_surface_rmse = (_rmse(demand_surface_actual,
                                 demand_surface_predicted)
                           if supported_holdout else math.inf)
    surface_slope_matches = 0
    surface_slope_count = 0
    for sign in (-1, 1):
        by_angle: dict[float, list[dict]] = defaultdict(list)
        for row in supported_holdout:
            if (1 if row["steering_command"] > 0.0 else -1) == sign:
                by_angle[round(abs(row["steering_command"]), 4)].append(row)
        angles = sorted(by_angle)
        for left, right in zip(angles, angles[1:]):
            left_rows, right_rows = by_angle[left], by_angle[right]
            measured_left = statistics.mean(sign * row["yaw_rate"]
                                            for row in left_rows)
            measured_right = statistics.mean(sign * row["yaw_rate"]
                                             for row in right_rows)
            measured_slope = (measured_right - measured_left) / (right - left)
            if abs(measured_slope) < MIN_RESOLVED_SLOPE_RPS_PER_RAD:
                continue
            midpoint = 0.5 * (left + right)
            speed = statistics.mean(
                row["forward_speed"] for row in left_rows + right_rows)
            demand = speed * math.tan(midpoint)
            _acceleration, slope = _demand_surface_value(
                demand_speed_curves[sign], speed, demand)
            predicted_slope = slope / (math.cos(midpoint) ** 2)
            surface_slope_count += 1
            surface_slope_matches += measured_slope * predicted_slope > 0.0
    surface_slope_fraction = (surface_slope_matches / surface_slope_count
                              if surface_slope_count else 0.0)
    unsupported_angles = sorted({
        round(abs(row["steering_command"]), 4)
        for row in demand_test_rows
        if not support_low <= row["forward_speed"] * math.tan(
                abs(row["steering_command"])) <= support_high
    })
    demand_surface_pass = (
        len(supported_holdout) == len(demand_test_rows)
        and demand_surface_rmse <= MAX_YAW_RATE_RMSE_RPS
        and surface_slope_count >= 10
        and surface_slope_fraction >= MIN_JACOBIAN_MATCH_FRACTION
    )
    print("2-D speed x steering-demand transition model "
          "(|steer|=0.15..0.25 rad): "
          f"yaw-rate RMSE={demand_surface_rmse:.5f} rad/s; "
          f"coverage={len(supported_holdout)}/{len(demand_test_rows)}; "
          f"q=[{support_low:.3f},{support_high:.3f}]; local Jacobian signs="
          f"{surface_slope_matches}/{surface_slope_count} "
          f"({surface_slope_fraction:.1%})")
    print(f"  speed-knot demand support: {support_by_speed}; "
          f"unsupported holdout steering magnitudes={unsupported_angles}")

    high_holdout = [row for row in holdout
                    if 0.275 <= abs(row["steering_command"]) <= 0.50]
    high_by_sign_angle: dict[tuple[int, float], list[tuple[float, float]]] = defaultdict(list)
    high_actual, high_predicted = [], []
    for row in high_holdout:
        sign = 1 if row["steering_command"] > 0.0 else -1
        angle = abs(row["steering_command"])
        speed = row["forward_speed"]
        acceleration, _ = _blend(
            curves, speed, sign, angle, "lateral_acceleration")
        measured_rate = sign * row["yaw_rate"]
        predicted_rate = acceleration / speed
        high_actual.append(measured_rate)
        high_predicted.append(predicted_rate)
        high_by_sign_angle[(sign, round(angle, 4))].append(
            (measured_rate, predicted_rate))
    high_rmse = (_rmse(high_actual, high_predicted)
                 if high_actual else math.inf)
    high_slope_matches = 0
    high_slope_count = 0
    for sign in (-1, 1):
        angles = sorted(angle for row_sign, angle in high_by_sign_angle
                        if row_sign == sign)
        for left, right in zip(angles, angles[1:]):
            left_actual, left_predicted = zip(*high_by_sign_angle[(sign, left)])
            right_actual, right_predicted = zip(*high_by_sign_angle[(sign, right)])
            measured_slope = (statistics.mean(right_actual)
                              - statistics.mean(left_actual)) / (right - left)
            predicted_slope = (statistics.mean(right_predicted)
                               - statistics.mean(left_predicted)) / (right - left)
            if abs(measured_slope) < MIN_RESOLVED_SLOPE_RPS_PER_RAD:
                continue
            high_slope_count += 1
            high_slope_matches += measured_slope * predicted_slope > 0.0
    high_slope_fraction = (high_slope_matches / high_slope_count
                           if high_slope_count else 0.0)
    high_surface_pass = (
        len(high_holdout) == 36
        and high_rmse <= MAX_YAW_RATE_RMSE_RPS
        and high_slope_count >= 8
        and high_slope_fraction >= MIN_JACOBIAN_MATCH_FRACTION
    )
    print("direct steering-angle high-turn surface "
          "(|steer|=0.275..0.50 rad): "
          f"yaw-rate RMSE={high_rmse:.5f} rad/s; "
          f"coverage={len(high_holdout)}/36; local Jacobian signs="
          f"{high_slope_matches}/{high_slope_count} "
          f"({high_slope_fraction:.1%})")

    hybrid_by_sign_angle: dict[tuple[int, float], list[tuple[float, float]]] = defaultdict(list)
    hybrid_actual, hybrid_predicted = [], []
    for row in holdout:
        sign = 1 if row["steering_command"] > 0.0 else -1
        predicted_rate = _hybrid_yaw_rate(
            row["forward_speed"], row["steering_command"], sign,
            curves, demand_speed_curves, support_low, support_high)
        measured_rate = sign * row["yaw_rate"]
        hybrid_actual.append(measured_rate)
        hybrid_predicted.append(predicted_rate)
        hybrid_by_sign_angle[(sign, round(abs(row["steering_command"]), 4))].append(
            (measured_rate, predicted_rate))
    hybrid_rmse = (_rmse(hybrid_actual, hybrid_predicted)
                   if hybrid_actual else math.inf)
    hybrid_slope_matches = 0
    hybrid_slope_count = 0
    for sign in (-1, 1):
        angles = sorted(angle for row_sign, angle in hybrid_by_sign_angle
                        if row_sign == sign)
        for left, right in zip(angles, angles[1:]):
            left_actual, left_predicted = zip(*hybrid_by_sign_angle[(sign, left)])
            right_actual, right_predicted = zip(*hybrid_by_sign_angle[(sign, right)])
            measured_slope = (statistics.mean(right_actual)
                              - statistics.mean(left_actual)) / (right - left)
            predicted_slope = (statistics.mean(right_predicted)
                               - statistics.mean(left_predicted)) / (right - left)
            if abs(measured_slope) < MIN_RESOLVED_SLOPE_RPS_PER_RAD:
                continue
            hybrid_slope_count += 1
            hybrid_slope_matches += measured_slope * predicted_slope > 0.0
    hybrid_slope_fraction = (hybrid_slope_matches / hybrid_slope_count
                             if hybrid_slope_count else 0.0)
    hybrid_pass = (
        len(hybrid_actual) == 174
        and hybrid_rmse <= MAX_YAW_RATE_RMSE_RPS
        and hybrid_slope_count >= 30
        and hybrid_slope_fraction >= MIN_JACOBIAN_MATCH_FRACTION
    )
    print("piecewise speed-demand/high-turn surface "
          "(|steer|=0.15..0.50 rad): "
          f"yaw-rate RMSE={hybrid_rmse:.5f} rad/s; "
          f"coverage={len(hybrid_actual)}/174; local Jacobian signs="
          f"{hybrid_slope_matches}/{hybrid_slope_count} "
          f"({hybrid_slope_fraction:.1%})")
    passed = demand_surface_pass and high_surface_pass and hybrid_pass
    slip_reduction = (1.0 - slip_rmse / slip_baseline_rmse
                      if slip_baseline_rmse > 0.0 else 0.0)
    slip_slope_fraction = (slip_slope_matches / slip_slope_count
                           if slip_slope_count else 0.0)
    print(f"front-slip-conditioned lateral-acceleration PCHIP: "
          f"yaw-rate RMSE={slip_rmse:.5f} rad/s vs current="
          f"{slip_baseline_rmse:.5f}; reduction={slip_reduction:+.1%}; "
          f"Sy support=[{slip_knots[0]:.3f},{slip_knots[-1]:.3f}]")
    print(f"  held-out local Jacobian signs: {slip_slope_matches}/"
          f"{slip_slope_count} ({slip_slope_fraction:.1%})")
    print("  interpretation: current measured-state feature only; this is not a "
          "recursive prediction or controller acceptance")
    decision = ("PASS piecewise 0.15..0.50 rad surface for recursive rollout "
                "validation only" if passed else
                "REJECT full-range cross-speed surface for rollout validation")
    print("decision: " + decision)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_4mps", type=Path)
    parser.add_argument("source_425mps", type=Path)
    parser.add_argument("source_475mps", type=Path)
    parser.add_argument("source_5mps", type=Path)
    parser.add_argument("holdout_45mps", type=Path)
    parser.add_argument("--support-bags", nargs=4, type=Path,
                        metavar=("SUPPORT_4", "SUPPORT_425",
                                 "SUPPORT_475", "SUPPORT_5"),
                        help="optional endpoint-support captures in source-speed order")
    parser.add_argument("--holdout-support-bag", type=Path,
                        help="optional 4.5 m/s high-steering holdout capture")
    args = parser.parse_args()
    support_paths = (dict(zip(SOURCE_SPEEDS_MPS, args.support_bags))
                     if args.support_bags else None)
    return 0 if evaluate(args.source_4mps, args.source_425mps,
                         args.source_475mps, args.source_5mps,
                         args.holdout_45mps, support_paths,
                         args.holdout_support_bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())

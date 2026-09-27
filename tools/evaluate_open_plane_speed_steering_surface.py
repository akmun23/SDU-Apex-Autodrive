#!/usr/bin/env python3
"""Cross-validate measured-speed × steering steady-state yaw response maps.

The legacy isolated-bag path tests 2.2–5.0 m/s. ``--grid-speed-holdout`` tests
grid captures at a deliberately withheld speed, including optimizer-relevant
local steering-derivative signs.
"""

from __future__ import annotations

import argparse
import bisect
import math
import re
import statistics
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as analysis
from tools import evaluate_open_plane_rear_slip_effect as rear_slip
from tools import evaluate_open_plane_yaw_spline as yaw_spline


HELDOUT_ANGLES_RAD = (0.42, 0.46)
MAX_HELDOUT_RMSE_PER_M = 0.10
MIN_CONFIGURED_RMSE_REDUCTION = 0.50
PARTIAL_GROUP_ANGLES_RAD = (0.15, 0.20, 0.21, 0.22, 0.23,
                            0.25, 0.30, 0.35, 0.42, 0.50)
GRID_ANGLES_RAD = (0.15, 0.25, 0.30, 0.35, 0.42)


@dataclass(frozen=True)
class SpeedBlock:
    phase: str
    repetition: int
    target_speed_mps: float
    steering_rad: float
    yaw_gain_per_m: float
    measured_forward_speed_mps: float
    front_lateral_slip_proxy: float | None = None
    rear_lateral_slip_proxy: float | None = None
    lateral_acceleration_mps2: float | None = None


def _load_complete_speed_group_from_partial_run(
        path: Path, target_speed_mps: float,
        repetitions: tuple[int, ...] = (1, 2, 3)) -> list[SpeedBlock]:
    """Load one fully completed matched speed group from a later-aborted bag.

    The whole run is never promoted: the requested speed group must cover the
    exact requested repetition/sign/angle schedule, each phase must be
    valid/matched, and its time interval must meet stream/collision/timing gates.
    """
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        required = (analysis.ODOM, analysis.STEERING, analysis.COLLISIONS,
                    analysis.TIMING_FAULT, analysis.PACKET_TIMING, analysis.IMU,
                    analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("partial-run bag is missing: " + ", ".join(missing))
        phases, experiment_end = analysis._phase_events(connection, topics)
        collisions = [int(message.data) for _, message in analysis._messages(
            connection, topics, analysis.COLLISIONS)]
        faults = [bool(message.data) for _, message in analysis._messages(
            connection, topics, analysis.TIMING_FAULT)]
        selected = [phase for phase in phases
                    if phase.label.startswith("isolated_r")
                    and abs(phase.target_speed_mps - target_speed_mps) < 1e-6
                    and int(phase.label.split("_", 2)[1][1:]) in repetitions]
        expected = {
            f"isolated_r{repetition}_{sign * angle:+.2f}rad_"
            f"{target_speed_mps:.1f}mps"
            for repetition in repetitions
            for angle in PARTIAL_GROUP_ANGLES_RAD
            for sign in (-1.0, 1.0)
        }
        if (not repetitions or len(set(repetitions)) != len(repetitions)
                or len(selected) != len(expected)
                or {phase.label for phase in selected} != expected):
            raise ValueError(f"speed group {target_speed_mps:.1f} m/s is incomplete: "
                             f"found {len(selected)}/{len(expected)} expected probes")
        if any(phase.valid is not True for phase in selected):
            raise ValueError("speed group contains an invalid probe")
        if experiment_end.get("quality_failures"):
            raise ValueError("partial capture has experiment quality failures")
        if not collisions or max(collisions) != 0 or any(faults):
            raise ValueError("partial capture has a collision or timing fault")
        for phase in selected:
            start = (phase.initial_window_speed_mps,
                     phase.initial_window_abs_vy_mps,
                     phase.initial_window_abs_yaw_rate_rps,
                     phase.initial_steering_rad)
            if (any(value is None for value in start)
                    or abs(start[0] - target_speed_mps) > 0.20
                    or start[1] > 0.08 or start[2] > 0.12
                    or abs(start[3]) > 0.02):
                raise ValueError(f"unmatched speed-group start: {phase.label}: {start}")

        first_ns = min(phase.start_ns for phase in selected)
        last_ns = max(phase.end_ns for phase in selected)
        stream_rates = {}
        for name in (analysis.ODOM, analysis.STEERING, analysis.LEFT_ENCODER,
                     analysis.RIGHT_ENCODER, analysis.IMU, analysis.PACKET_TIMING):
            topic_id = topics[name][0]
            receipts = [int(row[0]) for row in connection.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? "
                "AND timestamp>=? AND timestamp<=? ORDER BY timestamp, id",
                (topic_id, first_ns, last_ns),
            )]
            rate, p95_ms, max_ms = analysis._rate(receipts)
            stream_rates[name] = (rate, p95_ms, max_ms)
            if (rate is None or rate < 38.0 or p95_ms is None or p95_ms > 35.0
                    or max_ms is None or max_ms > 60.0):
                raise ValueError(f"speed-group stream gate failed for {name}: "
                                 f"{stream_rates[name]}")

        odometry = []
        for receipt_ns, message in analysis._messages(connection, topics, analysis.ODOM):
            twist = message.twist.twist
            odometry.append((receipt_ns, float(twist.linear.x),
                             float(twist.linear.y), float(twist.angular.z)))
        steering = [
            analysis.ScalarRow(receipt_ns, receipt_ns, float(message.data))
            for receipt_ns, message in analysis._messages(
                connection, topics, analysis.STEERING)
        ]
    finally:
        connection.close()

    odom_times = [row[0] for row in odometry]
    steering_times = [row.receipt_ns for row in steering]
    result = []
    half_track = analysis.TRACK_WIDTH_M / 2.0
    for phase in selected:
        begin = int(np.searchsorted(odom_times,
                                    phase.start_ns + analysis.PHASE_SETTLE_NS,
                                    side="left"))
        end = int(np.searchsorted(odom_times, phase.end_ns, side="left"))
        gains = []
        forward_speeds = []
        front_slips = []
        rear_slips = []
        lateral_accelerations = []
        for receipt_ns, forward_speed, lateral_speed, yaw_rate in odometry[begin:end]:
            actual_steering = analysis._nearest_scalar(
                steering, steering_times, receipt_ns)
            if actual_steering is None or abs(forward_speed) < 0.5:
                continue
            denominator = forward_speed * math.tan(actual_steering.value)
            if abs(denominator) < 0.1:
                continue
            gains.append(yaw_rate / denominator)
            forward_speeds.append(abs(forward_speed))
            rear_vx, rear_vy = analysis._rear_axle_velocity(
                forward_speed, lateral_speed, yaw_rate)
            delta_left, delta_right = analysis._ackermann_angles(actual_steering.value)
            front_left = analysis._wheel_slip(
                rear_vx, rear_vy, yaw_rate,
                analysis.WHEELBASE_M, half_track, delta_left)
            front_right = analysis._wheel_slip(
                rear_vx, rear_vy, yaw_rate,
                analysis.WHEELBASE_M, -half_track, delta_right)
            rear_left = analysis._wheel_slip(
                rear_vx, rear_vy, yaw_rate, 0.0, half_track, 0.0)
            rear_right = analysis._wheel_slip(
                rear_vx, rear_vy, yaw_rate, 0.0, -half_track, 0.0)
            if front_left is not None and front_right is not None:
                front_slips.append(0.5 * (abs(front_left) + abs(front_right)))
            if rear_left is not None and rear_right is not None:
                rear_slips.append(0.5 * (abs(rear_left) + abs(rear_right)))
            lateral_accelerations.append(forward_speed * yaw_rate)
        if len(gains) < 30:
            raise ValueError(f"too few settled samples in {phase.label}: {len(gains)}")
        result.append(SpeedBlock(
            phase=phase.label,
            repetition=int(phase.label.split("_", 2)[1][1:]),
            target_speed_mps=phase.target_speed_mps,
            steering_rad=phase.commanded_steering_rad,
            yaw_gain_per_m=float(np.median(gains)),
            measured_forward_speed_mps=float(np.median(forward_speeds)),
            front_lateral_slip_proxy=(float(np.median(front_slips))
                                      if front_slips else None),
            rear_lateral_slip_proxy=(float(np.median(rear_slips))
                                     if rear_slips else None),
            lateral_acceleration_mps2=float(np.median(lateral_accelerations)),
        ))
    return result


def _load_grid_speed_group(path: Path, target_speed_mps: float,
                           capture_id: int) -> list[SpeedBlock]:
    """Load one valid speed band from a grid capture, even if later phases aborted."""
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        required = (analysis.ODOM, analysis.STEERING, analysis.COLLISIONS,
                    analysis.TIMING_FAULT, analysis.PACKET_TIMING, analysis.IMU,
                    analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("grid bag is missing: " + ", ".join(missing))
        phases, experiment_end = analysis._phase_events(connection, topics)
        selected = [phase for phase in phases
                    if phase.label.startswith("steer_")
                    and abs(phase.target_speed_mps - target_speed_mps) < 1e-6]
        expected = {
            (round(target_speed_mps, 1), round(sign * angle, 2))
            for angle in GRID_ANGLES_RAD for sign in (-1.0, 1.0)
        }
        observed = set()
        for phase in selected:
            match = re.fullmatch(
                r"steer_\d+_(\d+\.\d+)mps_([+-]\d+\.\d+)rad", phase.label)
            if match:
                observed.add((round(float(match.group(1)), 1),
                              round(float(match.group(2)), 2)))
        if observed != expected or len(selected) != len(expected):
            raise ValueError(f"grid speed {target_speed_mps:.1f} m/s is incomplete: "
                             f"found {len(selected)}/{len(expected)} steering probes")
        if any(phase.valid is not True for phase in selected):
            raise ValueError("grid speed group contains an invalid probe")
        if experiment_end.get("quality_failures"):
            raise ValueError("grid capture reports experiment quality failures")
        collisions = [int(msg.data) for _, msg in analysis._messages(
            connection, topics, analysis.COLLISIONS)]
        faults = [bool(msg.data) for _, msg in analysis._messages(
            connection, topics, analysis.TIMING_FAULT)]
        if not collisions or max(collisions) != 0 or any(faults):
            raise ValueError("grid capture contains a collision or timing fault")

        start_ns = min(phase.start_ns for phase in selected)
        end_ns = max(phase.end_ns for phase in selected)
        for name in (analysis.ODOM, analysis.STEERING, analysis.LEFT_ENCODER,
                     analysis.RIGHT_ENCODER, analysis.IMU, analysis.PACKET_TIMING):
            topic_id = topics[name][0]
            receipts = [int(row[0]) for row in connection.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? "
                "AND timestamp>=? AND timestamp<=? ORDER BY timestamp, id",
                (topic_id, start_ns, end_ns),
            )]
            rate, p95_ms, max_ms = analysis._rate(receipts)
            if (rate is None or rate < 38.0 or p95_ms is None or p95_ms > 35.0
                    or max_ms is None or max_ms > 60.0):
                raise ValueError(f"grid cadence gate failed for {name}: "
                                 f"{rate}, {p95_ms}, {max_ms}")

        odometry = []
        for receipt_ns, message in analysis._messages(connection, topics, analysis.ODOM):
            twist = message.twist.twist
            odometry.append((receipt_ns, float(twist.linear.x),
                             float(twist.angular.z)))
        steering = [
            analysis.ScalarRow(receipt_ns, receipt_ns, float(message.data))
            for receipt_ns, message in analysis._messages(
                connection, topics, analysis.STEERING)
        ]
    finally:
        connection.close()

    odom_times = [row[0] for row in odometry]
    steering_times = [row.receipt_ns for row in steering]
    result = []
    for phase in selected:
        begin = bisect.bisect_left(
            odom_times, phase.start_ns + analysis.PHASE_SETTLE_NS)
        end = bisect.bisect_left(odom_times, phase.end_ns)
        gains = []
        speeds = []
        for receipt_ns, forward_speed, yaw_rate in odometry[begin:end]:
            actual_steering = analysis._nearest_scalar(
                steering, steering_times, receipt_ns)
            if actual_steering is None or abs(forward_speed) < 0.5:
                continue
            denominator = forward_speed * math.tan(actual_steering.value)
            if abs(denominator) < 0.1:
                continue
            gains.append(yaw_rate / denominator)
            speeds.append(abs(forward_speed))
        if len(gains) < 30:
            raise ValueError(f"too few settled samples in {phase.label}: {len(gains)}")
        result.append(SpeedBlock(
            phase=phase.label,
            repetition=capture_id,
            target_speed_mps=target_speed_mps,
            steering_rad=phase.commanded_steering_rad,
            yaw_gain_per_m=float(np.median(gains)),
            measured_forward_speed_mps=float(np.median(speeds)),
        ))
    return result


def _configured_gain(steering_rad: float) -> float:
    return yaw_spline._configured_gain(steering_rad)


def _source_curve(blocks: list[rear_slip.Block], speed: float,
                  sign: int, heldout_repetition: int | None,
                  heldout_angle: float) -> tuple[np.ndarray, np.ndarray]:
    at_speed = [block for block in blocks
                if abs(block.target_speed_mps - speed) < 1e-6
                and (1 if block.steering_rad > 0.0 else -1) == sign
                and (heldout_repetition is None
                     or block.repetition != heldout_repetition)
                and abs(abs(block.steering_rad) - heldout_angle) >= 1e-6]
    by_angle: dict[float, list[float]] = {}
    for block in at_speed:
        angle = round(abs(block.steering_rad), 2)
        by_angle.setdefault(angle, []).append(block.yaw_gain_per_m)
    angles = np.asarray(sorted(by_angle), dtype=float)
    gains = np.asarray([float(np.mean(by_angle[angle])) for angle in angles])
    if len(angles) < 3:
        raise ValueError(f"expected at least 3 steering knots at {speed:.1f} m/s, "
                         f"sign={sign:+d}, held-out angle={heldout_angle:.2f}; "
                         f"got {angles.tolist()}")
    return angles, gains


def _predict(speed: float, steering: float,
             source_speeds: tuple[float, float],
             source_curves: dict[tuple[float, int, float], tuple[np.ndarray, np.ndarray]]) \
        -> tuple[float, float]:
    """Return blended yaw gain and its derivative w.r.t. signed steering."""
    magnitude = abs(steering)
    sign = 1 if steering > 0.0 else -1
    source_predictions = []
    for source_speed in source_speeds:
        key = (source_speed, sign, magnitude)
        angles, gains = source_curves[key]
        value, derivative_magnitude = yaw_spline._pchip_value_and_slope(
            angles, gains, magnitude)
        source_predictions.append((value, derivative_magnitude * sign))
    low_weight = ((source_speeds[1] - speed)
                  / (source_speeds[1] - source_speeds[0]))
    high_weight = 1.0 - low_weight
    value = low_weight * source_predictions[0][0] + high_weight * source_predictions[1][0]
    derivative = (low_weight * source_predictions[0][1]
                  + high_weight * source_predictions[1][1])
    return float(value), float(derivative)


def _predict_lateral_acceleration(
        speed: float, steering: float, source_speeds: tuple[float, float],
        source_curves: dict[tuple[float, int, float], tuple[np.ndarray, np.ndarray]]) \
        -> tuple[float, float]:
    """Blend signed lateral acceleration over speed, then recover yaw gain."""
    magnitude = abs(steering)
    sign = 1 if steering > 0.0 else -1
    source_accelerations = []
    for source_speed in source_speeds:
        angles, gains = source_curves[(source_speed, sign, magnitude)]
        gain, derivative_magnitude = yaw_spline._pchip_value_and_slope(
            angles, gains, magnitude)
        derivative = derivative_magnitude * sign
        tangent = math.tan(steering)
        accel = source_speed**2 * tangent * gain
        accel_derivative = source_speed**2 * (
            gain / math.cos(steering)**2 + tangent * derivative)
        source_accelerations.append((accel, accel_derivative))

    low_weight = ((source_speeds[1] - speed)
                  / (source_speeds[1] - source_speeds[0]))
    high_weight = 1.0 - low_weight
    accel = (low_weight * source_accelerations[0][0]
             + high_weight * source_accelerations[1][0])
    accel_derivative = (low_weight * source_accelerations[0][1]
                        + high_weight * source_accelerations[1][1])
    gain = accel / (speed**2 * math.tan(steering))
    yaw_rate_sensitivity = accel_derivative / speed
    return float(gain), float(yaw_rate_sensitivity)


def _dense_acceleration_curves(
        blocks: list[SpeedBlock], steering_knots: tuple[float, ...],
        ) -> tuple[tuple[float, ...], dict[tuple[float, int], tuple[np.ndarray, np.ndarray]]]:
    """Build per-speed signed-slip-independent a_y(steer) curves from blocks."""
    grouped: dict[tuple[float, int, float], list[float]] = defaultdict(list)
    speed_rows: dict[float, list[float]] = defaultdict(list)
    for block in blocks:
        speed_rows[block.target_speed_mps].append(block.measured_forward_speed_mps)
        sign = 1 if block.steering_rad > 0.0 else -1
        angle = round(abs(block.steering_rad), 2)
        if angle in steering_knots and block.lateral_acceleration_mps2 is not None:
            grouped[(block.target_speed_mps, sign, angle)].append(
                sign * block.lateral_acceleration_mps2)

    target_speeds = tuple(sorted(speed_rows))
    if len(target_speeds) < 3:
        raise ValueError("dense surface requires at least three measured source speeds")
    curves: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
    for target_speed in target_speeds:
        for sign in (-1, 1):
            values = []
            for angle in steering_knots:
                observations = grouped[(target_speed, sign, angle)]
                if len(observations) < 2:
                    raise ValueError("source speed lacks repeated signed steering knot: "
                                     f"v={target_speed:.2f}, sign={sign}, "
                                     f"steer={angle:.2f}, n={len(observations)}")
                values.append(float(np.mean(observations)))
            if not np.all(np.isfinite(values)):
                raise ValueError("source lateral-acceleration curve is non-finite")
            curves[(target_speed, sign)] = (
                np.asarray(steering_knots, dtype=float),
                np.asarray(values, dtype=float))
    measured_speed_knots = tuple(
        float(np.mean(speed_rows[target_speed])) for target_speed in target_speeds)
    if not all(left < right for left, right in
               zip(measured_speed_knots, measured_speed_knots[1:])):
        raise ValueError(f"measured source speeds are not ordered: {measured_speed_knots}")
    return measured_speed_knots, curves


def _predict_dense_acceleration(
        speed: float, steering: float, speed_knots: tuple[float, ...],
        curves: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]],
        ) -> tuple[float, float]:
    """PCHIP lateral acceleration in steering and then speed; also dA/dsteer."""
    if not speed_knots[0] <= speed <= speed_knots[-1]:
        raise ValueError(f"speed {speed:.3f} outside source range {speed_knots}")
    magnitude = abs(steering)
    if not GRID_ANGLES_RAD[0] <= magnitude <= 0.50:
        raise ValueError(f"steering {magnitude:.4f} outside dense measured range")
    sign = 1 if steering >= 0.0 else -1
    accelerations = []
    steering_slopes = []
    for target_speed in sorted({key[0] for key in curves}):
        angles, values = curves[(target_speed, sign)]
        acceleration, slope = yaw_spline._pchip_value_and_slope(
            angles, values, magnitude)
        accelerations.append(acceleration)
        steering_slopes.append(slope)
    speed_array = np.asarray(speed_knots, dtype=float)
    acceleration, _ = yaw_spline._pchip_value_and_slope(
        speed_array, np.asarray(accelerations, dtype=float), speed)
    derivative, _ = yaw_spline._pchip_value_and_slope(
        speed_array, np.asarray(steering_slopes, dtype=float), speed)
    return float(acceleration), float(derivative)


def evaluate_dense_speed_pchip(source_path: Path, holdout_path: Path) -> bool:
    """Train 3/4/5 m/s response knots; score the independent 4.5 m/s group."""
    source_blocks = []
    for speed in (3.0, 4.0, 5.0):
        source_blocks.extend(_load_complete_speed_group_from_partial_run(
            source_path, speed, repetitions=(1, 2)))
    holdout = _load_complete_speed_group_from_partial_run(holdout_path, 4.5)
    steering_knots = PARTIAL_GROUP_ANGLES_RAD
    speed_knots, curves = _dense_acceleration_curves(
        source_blocks, steering_knots)
    if len(source_blocks) != 3 * 2 * len(steering_knots) * 2:
        raise ValueError(f"incomplete source surface: {len(source_blocks)} blocks")
    if len(holdout) != 3 * 2 * len(steering_knots):
        raise ValueError(f"incomplete 4.5 m/s holdout: {len(holdout)} blocks")

    actual_gain = []
    predicted_gain = []
    configured_gain = []
    yaw_errors = []
    actual_by_condition: dict[tuple[int, float], list[tuple[float, float, float]]] = defaultdict(list)
    predicted_by_angle: dict[float, list[float]] = defaultdict(list)
    for block in holdout:
        speed = block.measured_forward_speed_mps
        angle = abs(block.steering_rad)
        sign = 1 if block.steering_rad > 0.0 else -1
        predicted_acceleration, _ = _predict_dense_acceleration(
            speed, block.steering_rad, speed_knots, curves)
        actual_acceleration = sign * block.lateral_acceleration_mps2
        denominator = speed * speed * math.tan(angle)
        actual_gain.append(actual_acceleration / denominator)
        predicted_gain.append(predicted_acceleration / denominator)
        configured_gain.append(_configured_gain(block.steering_rad))
        yaw_errors.append((predicted_acceleration - actual_acceleration) / speed)
        actual_yaw = block.lateral_acceleration_mps2 / speed
        actual_by_condition[(sign, round(angle, 2))].append(
            (block.steering_rad, actual_yaw, speed))
        predicted_by_angle[round(angle, 2)].append(
            predicted_acceleration / speed)

    actual = np.asarray(actual_gain)
    predicted = np.asarray(predicted_gain)
    configured = np.asarray(configured_gain)
    gain_rmse = float(np.sqrt(np.mean(np.square(actual - predicted))))
    configured_rmse = float(np.sqrt(np.mean(np.square(actual - configured))))
    yaw_rmse = float(np.sqrt(np.mean(np.square(yaw_errors))))
    max_gain_error = float(np.max(np.abs(actual - predicted)))
    reduction = 1.0 - gain_rmse / configured_rmse if configured_rmse > 0 else 0.0

    derivative_checks = []
    actual_abs_by_angle: dict[float, list[float]] = defaultdict(list)
    for (sign, angle), rows in actual_by_condition.items():
        actual_abs_by_angle[angle].extend(abs(row[1]) for row in rows)
    for sign in (-1, 1):
        points = []
        for angle in steering_knots:
            rows = actual_by_condition[(sign, angle)]
            points.append((statistics.median(row[0] for row in rows),
                           statistics.median(row[1] for row in rows),
                           statistics.mean(row[2] for row in rows)))
        points.sort(key=lambda row: row[0])
        for left, right in zip(points, points[1:]):
            observed_slope = (right[1] - left[1]) / (right[0] - left[0])
            midpoint_steering = 0.5 * (left[0] + right[0])
            midpoint_speed = 0.5 * (left[2] + right[2])
            _, accel_slope = _predict_dense_acceleration(
                midpoint_speed, midpoint_steering, speed_knots, curves)
            predicted_slope = accel_slope / midpoint_speed
            if abs(observed_slope) >= 0.05:
                derivative_checks.append((observed_slope, predicted_slope,
                                          observed_slope * predicted_slope > 0.0))

    matched = sum(item[2] for item in derivative_checks)
    slope_pass = bool(derivative_checks) and matched == len(derivative_checks)
    passed = (gain_rmse <= MAX_HELDOUT_RMSE_PER_M
              and reduction >= MIN_CONFIGURED_RMSE_REDUCTION and slope_pass)
    print(f"training: {source_path}; speeds=3/4/5 m/s, repetitions 1--2")
    print(f"independent 4.5 m/s holdout: {holdout_path}; {len(holdout)} blocks, "
          "all three repetitions, 10 steering knots and both turn signs")
    print(f"measured source speed knots: {', '.join(f'{value:.3f}' for value in speed_knots)} m/s")
    print(f"speed PCHIP of lateral acceleration + steering PCHIP: yaw-gain RMSE="
          f"{gain_rmse:.5f} 1/m, configured RMSE={configured_rmse:.5f} 1/m, "
          f"gain={reduction:+.1%}, max error={max_gain_error:.5f} 1/m, "
          f"yaw-rate RMSE={yaw_rmse:.5f} rad/s")
    print(f"local yaw-rate derivative signs: {matched}/{len(derivative_checks)}; "
          f"predicted range={min(item[1] for item in derivative_checks):.3f}.."
          f"{max(item[1] for item in derivative_checks):.3f} 1/s")
    print("  steering  actual |yaw|  predicted |yaw| (rad/s)")
    for angle in steering_knots:
        measured = statistics.mean(actual_abs_by_angle[angle])
        prediction = statistics.mean(
            abs(value) for value in predicted_by_angle[angle])
        print(f"  {angle:.2f}       {measured:.4f}       {prediction:.4f}")
    print("decision: " + ("PASS steady cross-speed surface only" if passed else
                          "REJECT; do not promote this interpolator to MPC"))
    return passed


def evaluate(source_low: Path, heldout_middle: Path, source_high: Path,
             partial_middle_speed: float | None = None) -> bool:
    low_blocks = rear_slip._load_blocks(source_low)
    heldout_blocks = (
        _load_complete_speed_group_from_partial_run(heldout_middle,
                                                    partial_middle_speed)
        if partial_middle_speed is not None
        else rear_slip._load_blocks(heldout_middle)
    )
    high_blocks = rear_slip._load_blocks(source_high)
    low_speeds = {round(block.target_speed_mps, 1) for block in low_blocks}
    middle_speeds = {round(block.target_speed_mps, 1) for block in heldout_blocks}
    high_speeds = {round(block.target_speed_mps, 1) for block in high_blocks}
    if len(low_speeds) != 1 or len(middle_speeds) != 1 or len(high_speeds) != 1:
        raise ValueError("each input bag must contain only one target speed")
    low_speed = low_speeds.pop()
    heldout_speed = middle_speeds.pop()
    high_speed = high_speeds.pop()
    source_speeds = (low_speed, high_speed)
    if not low_speed < heldout_speed < high_speed:
        raise ValueError("the held-out target speed must lie between source speeds")
    source_blocks = low_blocks + high_blocks
    heldout_angles = ((0.35, 0.42) if partial_middle_speed is not None
                      else HELDOUT_ANGLES_RAD)
    if (partial_middle_speed is None
            and sorted({block.repetition for block in heldout_blocks}) != [1, 2, 3]):
        raise ValueError("the intermediate-speed holdout must contain repetitions 1, 2, and 3")

    predictions: list[float] = []
    accel_blend_predictions: list[float] = []
    actual: list[float] = []
    configured: list[float] = []
    yaw_rate_errors: list[float] = []
    accel_blend_yaw_rate_errors: list[float] = []
    sensitivities: list[float] = []
    for block in heldout_blocks:
        angle = abs(block.steering_rad)
        if not any(abs(angle - value) < 1e-6 for value in heldout_angles):
            continue
        sign = 1 if block.steering_rad > 0.0 else -1
        heldout_repetition = None if partial_middle_speed is not None else block.repetition
        curves = {
            (speed, sign, angle): _source_curve(
                source_blocks, speed, sign, heldout_repetition, angle)
            for speed in source_speeds
        }
        query_speed = (block.measured_forward_speed_mps
                       if partial_middle_speed is not None
                       else heldout_speed)
        predicted, _ = _predict(
            query_speed, block.steering_rad, source_speeds, curves)
        accel_blend_gain, accel_blend_sensitivity = _predict_lateral_acceleration(
            query_speed, block.steering_rad, source_speeds, curves)
        steering = block.steering_rad
        actual.append(block.yaw_gain_per_m)
        predictions.append(predicted)
        accel_blend_predictions.append(accel_blend_gain)
        configured.append(_configured_gain(steering))
        yaw_rate_errors.append(
            query_speed * math.tan(steering)
            * (predicted - block.yaw_gain_per_m))
        accel_blend_yaw_rate_errors.append(
            query_speed * math.tan(steering)
            * (accel_blend_gain - block.yaw_gain_per_m))
        sensitivities.append(accel_blend_sensitivity)

    expected_predictions = 12
    if len(actual) != expected_predictions:
        raise ValueError(f"expected {expected_predictions} held-out high-angle blocks, "
                         f"found {len(actual)}")
    actual_array = np.asarray(actual)
    prediction_array = np.asarray(predictions)
    accel_blend_array = np.asarray(accel_blend_predictions)
    configured_array = np.asarray(configured)
    rmse = float(np.sqrt(np.mean(np.square(actual_array - prediction_array))))
    accel_blend_rmse = float(np.sqrt(np.mean(
        np.square(actual_array - accel_blend_array))))
    configured_rmse = float(np.sqrt(np.mean(np.square(actual_array - configured_array))))
    max_error = float(np.max(np.abs(actual_array - prediction_array)))
    accel_blend_max_error = float(np.max(np.abs(actual_array - accel_blend_array)))
    yaw_rate_rmse = float(np.sqrt(np.mean(np.square(yaw_rate_errors))))
    accel_blend_yaw_rate_rmse = float(np.sqrt(
        np.mean(np.square(accel_blend_yaw_rate_errors))))
    positive_sensitivity = all(value > 0.0 and math.isfinite(value)
                               for value in sensitivities)
    reduction = (1.0 - accel_blend_rmse / configured_rmse
                 if configured_rmse > 1e-12 else 0.0)
    passed = (accel_blend_rmse <= MAX_HELDOUT_RMSE_PER_M
              and reduction >= MIN_CONFIGURED_RMSE_REDUCTION
              and positive_sensitivity)

    print("surface: K_yaw(speed, |steer|, turn-sign), PCHIP in steering and "
          f"linear speed blend {low_speed:.1f}→{high_speed:.1f} m/s")
    if partial_middle_speed is None:
        holdout_description = (f"{heldout_middle} at {heldout_speed:.1f} m/s; "
                               "leave-one-repetition-out intermediate-angle predictions")
    else:
        holdout_description = (
            f"{heldout_middle} complete {heldout_speed:.1f} m/s group extracted from "
            "later-aborted capture; independent-speed intermediate-angle predictions")
    print(f"held-out: {holdout_description}; {len(actual)} predictions")
    print(f"  linear-K blend (rejected): RMSE={rmse:.5f} 1/m; "
          f"max absolute error={max_error:.5f} 1/m; "
          f"yaw-rate RMSE={yaw_rate_rmse:.5f} rad/s")
    print(f"  linear lateral-acceleration blend: RMSE={accel_blend_rmse:.5f} 1/m; "
          f"configured RMSE={configured_rmse:.5f} 1/m; "
          f"reduction={reduction * 100:+.1f}%; "
          f"max absolute error={accel_blend_max_error:.5f} 1/m")
    print(f"  yaw-rate RMSE={accel_blend_yaw_rate_rmse:.5f} rad/s; "
          f"d(r_ss)/d(steer) range={min(sensitivities):.4f}.."
          f"{max(sensitivities):.4f} 1/s; positive finite={positive_sensitivity}")
    print("  held-out blocks: rep steer measured-K linear-K blend accel-blend-K")
    for block, measured, linear_gain, accel_gain in zip(
            [block for block in heldout_blocks
             if any(abs(abs(block.steering_rad) - angle) < 1e-6
                    for angle in heldout_angles)],
            actual, prediction_array, accel_blend_array):
        print(f"    r{block.repetition} {block.steering_rad:+.2f}rad "
              f"{block.yaw_gain_per_m:.3f} {linear_gain:.3f} {accel_gain:.3f} 1/m")
    print(f"  decision: " + (f"ACCEPT for steady-state interpolation inside "
                          f"{low_speed:.1f}–{high_speed:.1f} m/s"
                          if passed else
                          "REJECT; do not use this speed interpolation in MPC"))
    return passed


def evaluate_grid_speed_holdout(source_low: Path, heldout_middle: Path,
                                source_high: Path,
                                speeds: tuple[float, float, float]) -> bool:
    """Test speed interpolation and optimizer-relevant Jacobian on grid bags."""
    low_speed, heldout_speed, high_speed = speeds
    if not low_speed < heldout_speed < high_speed:
        raise ValueError("grid holdout speed must lie between source speeds")
    source_blocks = (
        _load_grid_speed_group(source_low, low_speed, capture_id=1)
        + _load_grid_speed_group(source_high, high_speed, capture_id=2)
    )
    heldout = _load_grid_speed_group(
        heldout_middle, heldout_speed, capture_id=3)
    angles = tuple(GRID_ANGLES_RAD)
    query_angles = set(angles)
    query_angles.update(
        0.5 * (left + right) for left, right in zip(angles, angles[1:]))
    curves: dict[tuple[float, int, float], tuple[np.ndarray, np.ndarray]] = {}
    for speed in (low_speed, high_speed):
        for sign in (-1, 1):
            knots, gains = _source_curve(
                source_blocks, speed, sign, heldout_repetition=None,
                heldout_angle=0.0)
            for angle in query_angles:
                curves[(speed, sign, angle)] = (knots, gains)

    actual: list[float] = []
    direct_predictions: list[float] = []
    accel_predictions: list[float] = []
    configured: list[float] = []
    predicted_sensitivities: list[float] = []
    for block in heldout:
        query_speed = block.measured_forward_speed_mps
        direct, _ = _predict(
            query_speed, block.steering_rad, (low_speed, high_speed), curves)
        accel_gain, sensitivity = _predict_lateral_acceleration(
            query_speed, block.steering_rad, (low_speed, high_speed), curves)
        actual.append(block.yaw_gain_per_m)
        direct_predictions.append(direct)
        accel_predictions.append(accel_gain)
        configured.append(_configured_gain(block.steering_rad))
        predicted_sensitivities.append(sensitivity)

    def rmse(predictions: list[float]) -> float:
        return float(np.sqrt(np.mean(np.square(
            np.asarray(actual) - np.asarray(predictions)))))

    direct_rmse = rmse(direct_predictions)
    accel_rmse = rmse(accel_predictions)
    configured_rmse = rmse(configured)
    reduction = (1.0 - accel_rmse / configured_rmse
                 if configured_rmse > 1e-12 else 0.0)

    measured_rates: dict[tuple[int, float], tuple[float, float]] = {}
    for block in heldout:
        sign = 1 if block.steering_rad > 0.0 else -1
        rate = (block.yaw_gain_per_m * block.measured_forward_speed_mps
                * math.tan(block.steering_rad))
        measured_rates[(sign, abs(block.steering_rad))] = (
            block.steering_rad, rate)
    derivative_checks: list[tuple[float, float, bool]] = []
    for sign in (-1, 1):
        points = sorted(
            (angle, measured_rates[(sign, angle)]) for angle in angles)
        for (left_angle, (_, left_rate)), (right_angle, (_, right_rate)) in zip(
                points, points[1:]):
            left_block = next(block for block in heldout
                              if round(abs(block.steering_rad), 2) == left_angle
                              and (1 if block.steering_rad > 0.0 else -1) == sign)
            right_block = next(block for block in heldout
                               if round(abs(block.steering_rad), 2) == right_angle
                               and (1 if block.steering_rad > 0.0 else -1) == sign)
            left_delta = left_block.steering_rad
            right_delta = right_block.steering_rad
            observed_slope = (right_rate - left_rate) / (right_delta - left_delta)
            midpoint = 0.5 * (left_delta + right_delta)
            midpoint_speed = 0.5 * (
                left_block.measured_forward_speed_mps
                + right_block.measured_forward_speed_mps)
            _, predicted_slope = _predict_lateral_acceleration(
                midpoint_speed, midpoint, (low_speed, high_speed), curves)
            derivative_checks.append((observed_slope, predicted_slope,
                                      observed_slope * predicted_slope > 0.0))

    sensitivities_finite = all(math.isfinite(value)
                               for value in predicted_sensitivities)
    derivative_signs_match = all(item[2] for item in derivative_checks)
    passed = (accel_rmse <= MAX_HELDOUT_RMSE_PER_M
              and reduction >= MIN_CONFIGURED_RMSE_REDUCTION
              and sensitivities_finite and derivative_signs_match)
    print(f"grid speed holdout: train={low_speed:.1f}/{high_speed:.1f} m/s, "
          f"held out={heldout_speed:.1f} m/s; 10 signed steering probes")
    print(f"  direct K blend RMSE={direct_rmse:.4f} 1/m")
    print(f"  lateral-acceleration blend RMSE={accel_rmse:.4f} 1/m; "
          f"configured-law RMSE={configured_rmse:.4f} 1/m; "
          f"reduction={100.0 * reduction:.1f}%")
    print(f"  predicted d(yaw rate)/d(steering) range="
          f"{min(predicted_sensitivities):.3f}.."
          f"{max(predicted_sensitivities):.3f} 1/s; finite={sensitivities_finite}")
    print(f"  finite-difference derivative signs matched="
          f"{sum(item[2] for item in derivative_checks)}/{len(derivative_checks)}; "
          f"measured slope range={min(item[0] for item in derivative_checks):.3f}.."
          f"{max(item[0] for item in derivative_checks):.3f} 1/s")
    for angle, measured, direct, accel in zip(
            angles,
            [statistics.mean(block.yaw_gain_per_m for block in heldout
                             if round(abs(block.steering_rad), 2) == angle)
             for angle in angles],
            [statistics.mean(direct_predictions[i] for i, block in enumerate(heldout)
                             if round(abs(block.steering_rad), 2) == angle)
             for angle in angles],
            [statistics.mean(accel_predictions[i] for i, block in enumerate(heldout)
                             if round(abs(block.steering_rad), 2) == angle)
             for angle in angles]):
        print(f"  |steer|={angle:.2f}: measured K={measured:.3f}, "
              f"direct-K={direct:.3f}, accel-blend={accel:.3f} 1/m")
    print("  decision: " + ("ACCEPT for this measured-speed band"
                           if passed else
                           "REJECT for MPC promotion; output fit alone is insufficient"))
    return passed


def _self_test() -> None:
    source_speeds = (2.2, 5.0)
    heldout_speed = 3.0
    low, high = 0.8, 0.4
    weight_high = ((source_speeds[1] - heldout_speed)
                   / (source_speeds[1] - source_speeds[0]))
    blended = (1.0 - weight_high) * low + weight_high * high
    assert 0.4 < blended < 0.8
    angles = np.asarray((0.30, 0.46, 0.50))
    gains = 1.2 - 0.7 * angles
    value, slope = yaw_spline._pchip_value_and_slope(angles, gains, 0.42)
    assert abs(value - (1.2 - 0.7 * 0.42)) < 1e-12
    assert abs(slope + 0.7) < 1e-12
    curves = {
        (2.2, 1, 0.42): (angles, np.full(3, 1.5)),
        (5.0, 1, 0.42): (angles, np.full(3, 0.5)),
    }
    accel_gain, accel_sensitivity = _predict_lateral_acceleration(
        heldout_speed, 0.42, source_speeds, curves)
    assert 0.0 < accel_gain < 1.5 and accel_sensitivity > 0.0
    print("speed-steering surface math: PASS (interpolation, signed sensitivity)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="*", type=Path,
                        help="low-speed source, intermediate-speed holdout, high-speed source")
    parser.add_argument("--partial-middle-speed", type=float,
                        help="score this complete target-speed group from the middle bag, "
                             "even if later probes in that capture were aborted")
    parser.add_argument("--grid-speed-holdout", nargs=3, type=Path,
                        metavar=("LOW_SOURCE", "HOLDOUT", "HIGH_SOURCE"),
                        help="score a grid bag at the middle speed using low/high grid bags")
    parser.add_argument("--dense-speed-pchip", nargs=2, type=Path,
                        metavar=("THREE_SPEED_SOURCE", "INDEPENDENT_HOLDOUT"),
                        help="train steering/lateral-acceleration PCHIPs from the "
                             "3/4/5 m/s dense capture and score its 4.5 m/s group")
    parser.add_argument("--grid-speeds", nargs=3, type=float,
                        default=(4.5, 6.5, 7.5),
                        metavar=("LOW", "MIDDLE", "HIGH"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.dense_speed_pchip:
        if args.bags or args.grid_speed_holdout:
            parser.error("use --dense-speed-pchip by itself")
        return 0 if evaluate_dense_speed_pchip(
            args.dense_speed_pchip[0], args.dense_speed_pchip[1]) else 1
    if args.grid_speed_holdout:
        if args.bags:
            parser.error("use positional bags or --grid-speed-holdout, not both")
        return 0 if evaluate_grid_speed_holdout(
            args.grid_speed_holdout[0], args.grid_speed_holdout[1],
            args.grid_speed_holdout[2], tuple(args.grid_speeds)) else 1
    if len(args.bags) != 3:
        parser.error("provide exactly three bags in increasing target-speed order")
    return 0 if evaluate(*args.bags, partial_middle_speed=args.partial_middle_speed) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Held-out transient test for empirical open-plane yaw and lateral-velocity maps.

Fits on repetitions 1 and 2 of the matched 3 m/s steering steps, then scores
only repetition 3 at 25--750 ms. ``--external-test`` instead trains from one
complete fixed-speed capture and scores every probe in an independent capture
at the same speed, including a direct-curvature yaw response. Inputs are
recorded steering feedback and
odometry speed; these are conditional plant-response tests, not closed-loop
MPC tests. They do not estimate per-wheel tire forces.

Acceptance fixed before scoring: the fitted yaw model and the fitted lateral
velocity model must each improve RMSE by >=20% at three or more horizons and
must not regress by >10% at any horizon versus the configured yaw law and
held-velocity baseline, respectively. Every prediction must remain finite.
"""

from __future__ import annotations

import argparse
import bisect
import math
import re
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import analyze_open_plane_dynamics as common
    import evaluate_open_plane_yaw_spline as yaw_spline
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - requires ROS 2 runtime
    raise SystemExit(f"ROS 2 Python modules are required: {exc}") from exc


ODOM = common.ODOM
STEERING = common.STEERING
COLLISIONS = common.COLLISIONS
TIMING_FAULT = common.TIMING_FAULT
PHASE = common.PHASE
CADENCE_TOPICS = (ODOM, STEERING, common.LEFT_ENCODER,
                  common.RIGHT_ENCODER, common.IMU, common.PACKET_TIMING)
YAW_KNOTS_RAD = (0.30, 0.42, 0.46, 0.50)
DENSE_SURFACE_KNOTS_RAD = (0.15, 0.20, 0.21, 0.22, 0.23,
                           0.25, 0.30, 0.35, 0.42, 0.50)
HORIZONS_S = (0.025, 0.125, 0.250, 0.500, 0.750)
TRAIN_REPETITIONS = (1, 2)
HOLDOUT_REPETITION = 3
FIT_WINDOW_S = 0.750
TAU_GRID_S = tuple(i / 1000.0 for i in range(5, 301, 1))
SHAPE_GRID = tuple(i / 20.0 for i in range(10, 101, 1))
PHASE_RE = re.compile(r"isolated_r(?P<rep>[123])_(?P<steer>[+-]\d+\.\d+)rad")


@dataclass(frozen=True)
class Sample:
    time_s: float
    speed_mps: float
    vy_mps: float
    yaw_rate_rps: float
    steering_rad: float


@dataclass(frozen=True)
class Probe:
    repetition: int
    commanded_steering_rad: float
    samples: tuple[Sample, ...]
    target_speed_mps: float


def _decode(connection: sqlite3.Connection, topic_id: int, msg_type: str):
    message_class = get_message(msg_type)
    for timestamp, payload in connection.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp, id",
            (topic_id,)):
        yield int(timestamp), deserialize_message(bytes(payload), message_class)


def _nearest(rows: list[tuple[int, float]], times: list[int], target: int) -> float | None:
    index = bisect.bisect_left(times, target)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    if not candidates:
        return None
    selected = min(candidates, key=lambda i: abs(times[i] - target))
    return rows[selected][1] if abs(times[selected] - target) <= 30_000_000 else None


def load_probes(path: Path) -> tuple[list[Probe], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"bag database does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = common._topic_map(connection)
        required = (*CADENCE_TOPICS, COLLISIONS, TIMING_FAULT, PHASE)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("missing topic(s): " + ", ".join(missing))

        phase_rows, experiment_end = common._phase_events(connection, topics)
        odom_topic_id, odom_type = topics[ODOM]
        steering_topic_id, steering_type = topics[STEERING]
        odom: list[tuple[int, float, float, float]] = []
        for receipt_ns, message in _decode(connection, odom_topic_id, odom_type):
            twist = message.twist.twist
            vx_com, vy_com, yaw_rate = (
                float(twist.linear.x), float(twist.linear.y),
                float(twist.angular.z))
            vx, vy = common._rear_axle_velocity(vx_com, vy_com, yaw_rate)
            values = (vx, vy, yaw_rate)
            if all(math.isfinite(value) for value in values):
                odom.append((receipt_ns, math.hypot(vx, vy), vy, yaw_rate))

        steering = [(receipt_ns, float(message.data))
                    for receipt_ns, message in _decode(
                        connection, steering_topic_id, steering_type)
                    if math.isfinite(float(message.data))]
        steering_times = [row[0] for row in steering]

        collision_id, collision_type = topics[COLLISIONS]
        collisions = [int(message.data) for _, message in _decode(
            connection, collision_id, collision_type)]
        fault_id, fault_type = topics[TIMING_FAULT]
        timing_faults = [bool(message.data) for _, message in _decode(
            connection, fault_id, fault_type)]

        probes: list[Probe] = []
        for phase in phase_rows:
            match = PHASE_RE.search(phase.label)
            if not match or phase.valid is not True:
                continue
            start = phase.start_ns
            end = phase.end_ns
            block: list[Sample] = []
            for receipt_ns, speed, vy, yaw_rate in odom:
                if receipt_ns < start or receipt_ns > end:
                    continue
                steering_rad = _nearest(steering, steering_times, receipt_ns)
                if steering_rad is None:
                    continue
                block.append(Sample((receipt_ns - start) / 1e9,
                                    speed, vy, yaw_rate, steering_rad))
            if len(block) < 40:
                continue
            probes.append(Probe(int(match.group("rep")),
                                float(match.group("steer")), tuple(block),
                                phase.target_speed_mps))
    finally:
        connection.close()

    if not probes:
        raise ValueError("no valid matched steering probes found")
    identified_phases = [phase for phase in phase_rows
                         if PHASE_RE.search(phase.label)]
    matched_starts = sum(
        phase.valid is True
        and phase.initial_window_speed_mps is not None
        and abs(phase.initial_window_speed_mps - phase.target_speed_mps) <= 0.20
        and phase.initial_window_abs_vy_mps is not None
        and phase.initial_window_abs_vy_mps <= 0.08
        and phase.initial_window_abs_yaw_rate_rps is not None
        and phase.initial_window_abs_yaw_rate_rps <= 0.12
        and phase.initial_steering_rad is not None
        and abs(phase.initial_steering_rad) <= 0.02
        for phase in identified_phases)
    quality = {
        "probe_counts": {str(rep): sum(p.repetition == rep for p in probes)
                         for rep in (1, 2, 3)},
        "valid_phase_count": sum(phase.valid is True
                                  for phase in identified_phases),
        "matched_starts": matched_starts,
        "collision_count_start": collisions[0] if collisions else None,
        "collision_count_end": collisions[-1] if collisions else None,
        "timing_fault_samples": sum(timing_faults),
        "aborted": bool(experiment_end.get("aborted", True)),
        "quality_failures": experiment_end.get("quality_failures", []),
        "target_speeds_mps": sorted({p.target_speed_mps for p in probes}),
        "stream_cadence": {},
        "odom_rate_hz": ((len(odom) - 1) / ((odom[-1][0] - odom[0][0]) / 1e9)
                         if len(odom) > 1 else None),
        "steering_alignment_fraction": sum(
            1 for p in probes for sample in p.samples
            if math.isfinite(sample.steering_rad)
        ) / sum(len(p.samples) for p in probes),
    }
    cadence_connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for name in CADENCE_TOPICS:
            topic_id = topics[name][0]
            receipts = [int(row[0]) for row in cadence_connection.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? "
                "ORDER BY timestamp, id", (topic_id,))]
            rate, p95_ms, max_ms = common._rate(receipts)
            quality["stream_cadence"][name] = {
                "rate_hz": rate,
                "gap_p95_ms": p95_ms,
                "gap_max_ms": max_ms,
            }
    finally:
        cadence_connection.close()
    return probes, quality


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _training_maps(
        probes: list[Probe],
        training_repetitions: tuple[int, ...] = TRAIN_REPETITIONS,
        ) -> tuple[dict[float, float], dict[float, float]]:
    yaw_by_angle: dict[float, list[float]] = {a: [] for a in YAW_KNOTS_RAD}
    vy_by_angle: dict[float, list[float]] = {a: [] for a in YAW_KNOTS_RAD}
    for probe in probes:
        if probe.repetition not in training_repetitions:
            continue
        angle = round(abs(probe.commanded_steering_rad), 2)
        if angle not in yaw_by_angle:
            continue
        settled = [sample for sample in probe.samples if sample.time_s >= 0.50]
        yaw_gain = [sample.yaw_rate_rps /
                    (sample.speed_mps * math.tan(sample.steering_rad))
                    for sample in settled
                    if abs(sample.steering_rad) > 0.1 and sample.speed_mps > 1.0]
        signed_vy = [sample.vy_mps * (1.0 if probe.commanded_steering_rad > 0 else -1.0)
                     for sample in settled]
        if len(yaw_gain) < 20 or len(signed_vy) < 20:
            raise ValueError(f"insufficient settled samples at {angle:.2f} rad")
        yaw_by_angle[angle].append(_median(yaw_gain))
        vy_by_angle[angle].append(_median(signed_vy))
    expected_conditions = 2 * len(training_repetitions)
    if any(len(values) != expected_conditions for values in yaw_by_angle.values()):
        raise ValueError("training data do not cover both signs at every yaw knot")
    if any(len(values) != expected_conditions for values in vy_by_angle.values()):
        raise ValueError("training data do not cover both signs at every lateral-velocity knot")
    return ({angle: _median(values) for angle, values in yaw_by_angle.items()},
            {angle: _median(values) for angle, values in vy_by_angle.items()})


def _curvature_maps(
        probes: list[Probe],
        training_repetitions: tuple[int, ...] = TRAIN_REPETITIONS,
        ) -> dict[int, dict[float, float]]:
    """Fit sign-specific |r|/u knots from settled training samples."""
    samples = {
        (sign, angle): []
        for sign in (-1, 1)
        for angle in YAW_KNOTS_RAD
    }
    for probe in probes:
        if probe.repetition not in training_repetitions:
            continue
        angle = round(abs(probe.commanded_steering_rad), 2)
        sign = 1 if probe.commanded_steering_rad > 0.0 else -1
        if angle not in YAW_KNOTS_RAD:
            continue
        for sample in probe.samples:
            if sample.time_s < 0.50 or sample.speed_mps <= 1.0:
                continue
            if abs(sample.steering_rad) < 0.1:
                continue
            samples[(sign, angle)].append(
                abs(sample.yaw_rate_rps) / sample.speed_mps)
    if any(len(values) < 30 for values in samples.values()):
        raise ValueError("insufficient settled samples for direct curvature knots")
    return {
        sign: {angle: _median(samples[(sign, angle)]) for angle in YAW_KNOTS_RAD}
        for sign in (-1, 1)
    }


def _lateral_acceleration_maps(
        probes: list[Probe], target_speed: float,
        steering_knots: tuple[float, ...] = YAW_KNOTS_RAD,
        training_repetitions: tuple[int, ...] = (1, 2, 3),
        ) -> dict[int, dict[float, float]]:
    """Estimate steady signed lateral acceleration at one source speed."""
    block_values = {
        (sign, angle): []
        for sign in (-1, 1)
        for angle in steering_knots
    }
    for probe in probes:
        if (abs(probe.target_speed_mps - target_speed) > 1e-6
                or probe.repetition not in training_repetitions):
            continue
        angle = round(abs(probe.commanded_steering_rad), 2)
        sign = 1 if probe.commanded_steering_rad > 0.0 else -1
        if angle not in steering_knots:
            continue
        settled = [sign * sample.speed_mps * sample.yaw_rate_rps
                   for sample in probe.samples
                   if sample.time_s >= 0.50 and sample.speed_mps > 1.0
                   and abs(sample.steering_rad) > 0.1]
        if len(settled) < 20:
            raise ValueError("insufficient settled lateral-acceleration samples "
                             f"at {target_speed:.1f} m/s, {angle:.2f} rad")
        value = _median(settled)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError("measured lateral acceleration has an inconsistent "
                             f"turn sign at {target_speed:.1f} m/s, {angle:.2f} rad")
        block_values[(sign, angle)].append(value)
    if any(len(values) != len(training_repetitions)
           for values in block_values.values()):
        raise ValueError(f"{target_speed:.1f} m/s source must have "
                         f"{len(training_repetitions)} repeats per angle and turn sign")
    return {
        sign: {angle: _median(block_values[(sign, angle)])
               for angle in steering_knots}
        for sign in (-1, 1)
    }


def _surface_yaw_target(
        speed: float, steering: float,
        acceleration_maps: dict[float, dict[int, dict[float, float]]],
        steering_knots: tuple[float, ...] = YAW_KNOTS_RAD,
        ) -> float:
    """Blend measured lateral acceleration in speed, then derive yaw rate."""
    low_speed, high_speed = min(acceleration_maps), max(acceleration_maps)
    if speed <= 0.5 or speed < low_speed - 0.60 or speed > high_speed + 0.35:
        raise ValueError(f"speed {speed:.3f} m/s is outside validated surface")
    blend_speed = min(max(speed, low_speed), high_speed)
    magnitude = abs(steering)
    if magnitude > steering_knots[-1] + 0.002:
        raise ValueError(f"steering outside validated surface: {magnitude:.4f} rad")
    turn_sign = 1 if steering >= 0.0 else -1

    def source_acceleration(source_speed: float) -> float:
        knots = acceleration_maps[source_speed][turn_sign]
        values = tuple(knots[angle] for angle in steering_knots)
        if magnitude <= steering_knots[0]:
            # Enforce the zero-steer equilibrium and keep the measured first
            # knot continuous; this short ramp is only used during probe entry.
            return values[0] * magnitude / steering_knots[0]
        query = min(magnitude, steering_knots[-1])
        interpolated, _ = yaw_spline._pchip_value_and_slope(
            np.asarray(steering_knots, dtype=float),
            np.asarray(values, dtype=float), query)
        return float(interpolated)

    low_acceleration = source_acceleration(low_speed)
    high_acceleration = source_acceleration(high_speed)
    high_weight = (blend_speed - low_speed) / (high_speed - low_speed)
    acceleration = ((1.0 - high_weight) * low_acceleration
                    + high_weight * high_acceleration)
    return turn_sign * acceleration / speed


def _surface_yaw_sensitivity(
        speed: float, steering: float,
        acceleration_maps: dict[float, dict[int, dict[float, float]]],
        steering_knots: tuple[float, ...],
        ) -> float:
    """Derivative d(yaw-rate target)/d(signed steering), in 1/s."""
    low_speed, high_speed = min(acceleration_maps), max(acceleration_maps)
    if speed <= 0.0 or speed < low_speed - 0.60 or speed > high_speed + 0.35:
        raise ValueError(f"speed {speed:.3f} m/s is outside validated surface")
    magnitude = abs(steering)
    sign = 1 if steering >= 0.0 else -1
    source_slopes = []
    for source_speed in (low_speed, high_speed):
        values = tuple(acceleration_maps[source_speed][sign][angle]
                       for angle in steering_knots)
        if magnitude <= steering_knots[0]:
            slope = values[0] / steering_knots[0]
        else:
            _, slope = yaw_spline._pchip_value_and_slope(
                np.asarray(steering_knots, dtype=float),
                np.asarray(values, dtype=float),
                min(magnitude, steering_knots[-1]))
        source_slopes.append(slope)
    high_weight = (min(max(speed, low_speed), high_speed) - low_speed) / (
        high_speed - low_speed)
    acceleration_slope = ((1.0 - high_weight) * source_slopes[0]
                          + high_weight * source_slopes[1])
    return float(acceleration_slope / speed)


def _linear(x: float, knots: tuple[float, ...], values: tuple[float, ...]) -> float:
    if x <= knots[0]:
        return values[0]
    if x >= knots[-1]:
        return values[-1]
    for index in range(len(knots) - 1):
        if knots[index] <= x <= knots[index + 1]:
            weight = (x - knots[index]) / (knots[index + 1] - knots[index])
            return values[index] + weight * (values[index + 1] - values[index])
    raise AssertionError("unreachable interpolation interval")


def _curvature_target(speed: float, steering: float,
                      curvature_knots: dict[int, dict[float, float]]) -> float:
    """Directly interpolate signed yaw curvature q=|r|/u versus steering."""
    magnitude = abs(steering)
    if magnitude > YAW_KNOTS_RAD[-1] + 0.002:
        raise ValueError(f"steering outside measured curvature knots: {magnitude:.4f}")
    sign = 1 if steering >= 0.0 else -1
    values = tuple(curvature_knots[sign][angle] for angle in YAW_KNOTS_RAD)
    if magnitude <= YAW_KNOTS_RAD[0]:
        curvature = values[0] * magnitude / YAW_KNOTS_RAD[0]
    else:
        curvature = _linear(min(magnitude, YAW_KNOTS_RAD[-1]),
                            YAW_KNOTS_RAD, values)
    return sign * speed * curvature


def _yaw_target(speed: float, steering: float, tau_shape: float,
                yaw_knots: dict[float, float]) -> float:
    magnitude = abs(steering)
    if magnitude <= YAW_KNOTS_RAD[0]:
        base = 2.95
        endpoint = yaw_knots[YAW_KNOTS_RAD[0]]
        gain = base + (endpoint - base) * (magnitude / YAW_KNOTS_RAD[0]) ** tau_shape
    else:
        gains = tuple(yaw_knots[knot] for knot in YAW_KNOTS_RAD)
        gain = _linear(magnitude, YAW_KNOTS_RAD, gains)
    return speed * math.tan(steering) * gain


def _vy_target(steering: float, shape: float,
               vy_knots: dict[float, float]) -> float:
    magnitude = abs(steering)
    sign = 1.0 if steering >= 0 else -1.0
    if magnitude <= YAW_KNOTS_RAD[0]:
        return sign * vy_knots[YAW_KNOTS_RAD[0]] * (
            magnitude / YAW_KNOTS_RAD[0]) ** shape
    values = tuple(vy_knots[knot] for knot in YAW_KNOTS_RAD)
    return sign * _linear(magnitude, YAW_KNOTS_RAD, values)


def _configured_yaw_target(speed: float, steering: float) -> float:
    gain = 2.95 - 35.6 * min(max(abs(steering) - 0.41, 0.0), 0.05)
    return speed * math.tan(steering) * gain


def _step_first_order(state: float, target: float, dt: float, tau: float) -> float:
    retention = math.exp(-dt / tau)
    return retention * state + (1.0 - retention) * target


def _rollout(probe: Probe, start_index: int, end_index: int, *,
             yaw_knots: dict[float, float], vy_knots: dict[float, float],
             yaw_tau: float, yaw_shape: float, vy_tau: float | None,
             vy_shape: float, configured: bool = False,
             curvature_knots: dict[int, dict[float, float]] | None = None
             ) -> tuple[float, float]:
    first = probe.samples[start_index]
    yaw_rate = first.yaw_rate_rps
    vy = first.vy_mps
    for index in range(start_index + 1, end_index + 1):
        previous = probe.samples[index - 1]
        sample = probe.samples[index]
        dt = sample.time_s - previous.time_s
        if dt <= 0.0 or dt > 0.075:
            raise ValueError(f"bad sample interval {dt:.4f}s")
        if configured:
            yaw_target = _configured_yaw_target(
                sample.speed_mps, sample.steering_rad)
        elif curvature_knots is not None:
            yaw_target = _curvature_target(
                sample.speed_mps, sample.steering_rad, curvature_knots)
        else:
            yaw_target = _yaw_target(
                sample.speed_mps, sample.steering_rad, yaw_shape, yaw_knots)
        yaw_rate = _step_first_order(yaw_rate, yaw_target, dt, yaw_tau)
        if vy_tau is not None:
            vy_target = _vy_target(sample.steering_rad, vy_shape, vy_knots)
            vy = _step_first_order(vy, vy_target, dt, vy_tau)
    return yaw_rate, vy


def _best_parameters(
        probes: list[Probe], yaw_knots: dict[float, float],
        vy_knots: dict[float, float],
        training_repetitions: tuple[int, ...] = TRAIN_REPETITIONS,
        ) -> tuple[float, float, float, float]:
    training = [probe for probe in probes
                if probe.repetition in training_repetitions]
    best_yaw = (math.inf, 0.015, 1.0)
    for tau in TAU_GRID_S:
        for shape in SHAPE_GRID:
            squared = 0.0
            count = 0
            for probe in training:
                indexes = [i for i, sample in enumerate(probe.samples)
                           if sample.time_s <= FIT_WINDOW_S]
                if not indexes:
                    continue
                prediction = probe.samples[indexes[0]].yaw_rate_rps
                for previous_index, index in zip(indexes, indexes[1:]):
                    previous, sample = probe.samples[previous_index], probe.samples[index]
                    dt = sample.time_s - previous.time_s
                    target = _yaw_target(sample.speed_mps, sample.steering_rad,
                                         shape, yaw_knots)
                    prediction = _step_first_order(prediction, target, dt, tau)
                    squared += (prediction - sample.yaw_rate_rps) ** 2
                    count += 1
            score = squared / count if count else math.inf
            if score < best_yaw[0]:
                best_yaw = (score, tau, shape)

    best_vy = (math.inf, 0.015, 1.0)
    for tau in TAU_GRID_S:
        for shape in SHAPE_GRID:
            squared = 0.0
            count = 0
            for probe in training:
                indexes = [i for i, sample in enumerate(probe.samples)
                           if sample.time_s <= FIT_WINDOW_S]
                if not indexes:
                    continue
                prediction = probe.samples[indexes[0]].vy_mps
                for previous_index, index in zip(indexes, indexes[1:]):
                    previous, sample = probe.samples[previous_index], probe.samples[index]
                    dt = sample.time_s - previous.time_s
                    target = _vy_target(sample.steering_rad, shape, vy_knots)
                    prediction = _step_first_order(prediction, target, dt, tau)
                    squared += (prediction - sample.vy_mps) ** 2
                    count += 1
            score = squared / count if count else math.inf
            if score < best_vy[0]:
                best_vy = (score, tau, shape)
    return best_yaw[1], best_yaw[2], best_vy[1], best_vy[2]


def _fit_curvature_tau(probes: list[Probe],
                       curvature_knots: dict[int, dict[float, float]],
                       training_repetitions: tuple[int, ...]
                       ) -> float:
    """Fit only the first-order yaw time constant on the training capture."""
    best_tau = math.nan
    best_error = math.inf
    training = [probe for probe in probes
                if probe.repetition in training_repetitions]
    for tau in TAU_GRID_S:
        squared_error = 0.0
        count = 0
        for probe in training:
            samples = [sample for sample in probe.samples
                       if sample.time_s <= FIT_WINDOW_S]
            if len(samples) < 2:
                continue
            estimate = samples[0].yaw_rate_rps
            for previous, sample in zip(samples, samples[1:]):
                dt = sample.time_s - previous.time_s
                target = _curvature_target(
                    sample.speed_mps, sample.steering_rad, curvature_knots)
                estimate = _step_first_order(estimate, target, dt, tau)
                squared_error += (estimate - sample.yaw_rate_rps) ** 2
                count += 1
        score = squared_error / count if count else math.inf
        if score < best_error:
            best_error, best_tau = score, tau
    if not math.isfinite(best_tau):
        raise ValueError("could not fit a finite direct-curvature yaw time constant")
    return best_tau


def _horizon_errors(probes: list[Probe], horizons: tuple[float, ...], *,
                    yaw_knots: dict[float, float], vy_knots: dict[float, float],
                    yaw_tau: float, yaw_shape: float, vy_tau: float,
                    vy_shape: float, model: str,
                    curvature_knots: dict[int, dict[float, float]] | None = None
                    ) -> dict[float, tuple[float, float, int]]:
    result: dict[float, tuple[float, float, int]] = {}
    holdout = [probe for probe in probes if probe.repetition == HOLDOUT_REPETITION]
    for horizon in horizons:
        yaw_errors: list[float] = []
        vy_errors: list[float] = []
        for probe in holdout:
            target_index = min(range(len(probe.samples)),
                               key=lambda i: abs(probe.samples[i].time_s - horizon))
            if abs(probe.samples[target_index].time_s - horizon) > 0.035:
                continue
            predicted_yaw, predicted_vy = _rollout(
                probe, 0, target_index, yaw_knots=yaw_knots, vy_knots=vy_knots,
                yaw_tau=yaw_tau, yaw_shape=yaw_shape,
                vy_tau=vy_tau if model in ("candidate", "direct_curvature_full") else None,
                vy_shape=vy_shape,
                configured=model == "baseline",
                curvature_knots=(curvature_knots
                                 if model.startswith("direct_curvature") else None))
            actual = probe.samples[target_index]
            yaw_errors.append(predicted_yaw - actual.yaw_rate_rps)
            vy_errors.append(predicted_vy - actual.vy_mps)
        yaw_rmse = math.sqrt(sum(error * error for error in yaw_errors) / len(yaw_errors))
        vy_rmse = math.sqrt(sum(error * error for error in vy_errors) / len(vy_errors))
        result[horizon] = yaw_rmse, vy_rmse, len(yaw_errors)
    return result


def evaluate(path: Path) -> bool:
    probes, quality = load_probes(path)
    if len(quality["target_speeds_mps"]) != 1:
        raise ValueError("within-capture evaluation requires one fixed target speed")
    _validate_external_capture(
        "capture", probes, quality, quality["target_speeds_mps"][0])

    yaw_knots, vy_knots = _training_maps(probes)
    yaw_tau, yaw_shape, vy_tau, vy_shape = _best_parameters(
        probes, yaw_knots, vy_knots)
    baseline = _horizon_errors(
        probes, HORIZONS_S, yaw_knots=yaw_knots, vy_knots=vy_knots,
        yaw_tau=0.015, yaw_shape=1.0, vy_tau=0.015, vy_shape=1.0,
        model="baseline")
    candidate = _horizon_errors(
        probes, HORIZONS_S, yaw_knots=yaw_knots, vy_knots=vy_knots,
        yaw_tau=yaw_tau, yaw_shape=yaw_shape, vy_tau=vy_tau,
        vy_shape=vy_shape, model="candidate")

    yaw_improvements: list[float] = []
    vy_improvements: list[float] = []
    for horizon in HORIZONS_S:
        base_yaw, base_vy, n = baseline[horizon]
        fit_yaw, fit_vy, fit_n = candidate[horizon]
        if n != 8 or fit_n != 8:
            raise ValueError(f"incomplete holdout at {horizon:.3f}s: n={n}/{fit_n}")
        yaw_improvement = 1.0 - fit_yaw / base_yaw if base_yaw > 0 else 0.0
        vy_improvement = 1.0 - fit_vy / base_vy if base_vy > 0 else 0.0
        yaw_improvements.append(yaw_improvement)
        vy_improvements.append(vy_improvement)

    yaw_pass = sum(value >= 0.20 for value in yaw_improvements) >= 3 and min(
        yaw_improvements) >= -0.10
    vy_pass = sum(value >= 0.20 for value in vy_improvements) >= 3 and min(
        vy_improvements) >= -0.10
    finite = all(math.isfinite(value)
                 for row in (*baseline.values(), *candidate.values())
                 for value in row[:2])

    print(f"bag: {path}")
    print("data quality:", quality)
    print("training-only equilibrium knots:")
    for angle in YAW_KNOTS_RAD:
        print(f"  |steer|={angle:.2f}: K_yaw={yaw_knots[angle]:.4f} 1/m, "
              f"signed v_y={vy_knots[angle]:.4f} m/s")
    print(f"fitted yaw: tau={yaw_tau:.3f}s, low-angle gain-shape={yaw_shape:.2f}")
    print(f"fitted lateral velocity: tau={vy_tau:.3f}s, low-angle shape={vy_shape:.2f}")
    print("holdout repetition 3, one rollout from each matched start (n=8 per horizon):")
    print("horizon_ms  yaw_RMSE_base/candidate  yaw_gain  vy_RMSE_hold/candidate  vy_gain")
    for horizon, yaw_gain, vy_gain in zip(HORIZONS_S, yaw_improvements, vy_improvements):
        base_yaw, base_vy, _ = baseline[horizon]
        fit_yaw, fit_vy, _ = candidate[horizon]
        print(f"{horizon * 1000:10.0f}  {base_yaw:.5f}/{fit_yaw:.5f} rad/s "
              f"{yaw_gain:+.1%}  {base_vy:.5f}/{fit_vy:.5f} m/s {vy_gain:+.1%}")
    print(f"acceptance: yaw={'PASS' if yaw_pass else 'FAIL'}, "
          f"lateral_velocity={'PASS' if vy_pass else 'FAIL'}, finite={finite}")
    return yaw_pass and vy_pass and finite


def _validate_external_capture(label: str, probes: list[Probe],
                               quality: dict[str, Any],
                               expected_speed_mps: float) -> None:
    if quality["aborted"] or quality["quality_failures"]:
        raise ValueError(f"{label} capture did not finish cleanly: {quality}")
    if quality["collision_count_start"] != 0 or quality["collision_count_end"] != 0:
        raise ValueError(f"{label} capture contains a collision")
    if quality["timing_fault_samples"] != 0:
        raise ValueError(f"{label} capture contains a bridge timing fault")
    if quality["probe_counts"] != {"1": 8, "2": 8, "3": 8}:
        raise ValueError(f"{label} capture needs 8 probes in each repetition: {quality}")
    if quality["valid_phase_count"] != 24 or quality["matched_starts"] != 24:
        raise ValueError(f"{label} capture needs 24 valid matched starts: {quality}")
    if (len(quality["target_speeds_mps"]) != 1
            or abs(quality["target_speeds_mps"][0] - expected_speed_mps) > 1e-6
            or any(abs(probe.target_speed_mps - expected_speed_mps) > 1e-6
                   for probe in probes)):
        raise ValueError(f"{label} capture must contain only "
                         f"{expected_speed_mps:.1f} m/s probes")
    if quality["odom_rate_hz"] is None or quality["odom_rate_hz"] < 38.0:
        raise ValueError(f"{label} odometry rate is below 38 Hz: {quality}")
    if quality["steering_alignment_fraction"] != 1.0:
        raise ValueError(f"{label} steering/odometry alignment is incomplete")
    cadence = quality.get("stream_cadence", {})
    if set(cadence) != set(CADENCE_TOPICS):
        raise ValueError(f"{label} capture is missing a measured source cadence")
    for topic, values in cadence.items():
        if (values["rate_hz"] is None or values["rate_hz"] < 38.0
                or values["gap_p95_ms"] is None
                or values["gap_p95_ms"] > 35.0
                or values["gap_max_ms"] is None
                or values["gap_max_ms"] > 60.0):
            raise ValueError(f"{label} source cadence failed for {topic}: {values}")


def evaluate_external(training_path: Path, test_path: Path) -> bool:
    """Train and score complete, independent captures at the same speed."""
    if training_path.resolve() == test_path.resolve():
        raise ValueError("training and independent test bags must differ")
    training_probes, training_quality = load_probes(training_path)
    test_probes, test_quality = load_probes(test_path)
    if len(training_quality["target_speeds_mps"]) != 1:
        raise ValueError("training capture must contain exactly one target speed")
    training_speed = training_quality["target_speeds_mps"][0]
    _validate_external_capture("training", training_probes, training_quality,
                               training_speed)
    _validate_external_capture("independent test", test_probes, test_quality,
                               training_speed)

    training_repetitions = (1, 2, 3)
    yaw_knots, vy_knots = _training_maps(
        training_probes, training_repetitions)
    curvature_knots = _curvature_maps(training_probes, training_repetitions)
    yaw_tau, yaw_shape, vy_tau, vy_shape = _best_parameters(
        training_probes, yaw_knots, vy_knots, training_repetitions)
    direct_tau = _fit_curvature_tau(
        training_probes, curvature_knots, training_repetitions)

    # Re-label only for the existing horizon scorer; every test repetition is
    # independent of training and all 24 phases remain in the score.
    holdout = [Probe(HOLDOUT_REPETITION, probe.commanded_steering_rad,
                     probe.samples, probe.target_speed_mps)
               for probe in test_probes]
    baseline = _horizon_errors(
        holdout, HORIZONS_S, yaw_knots=yaw_knots, vy_knots=vy_knots,
        yaw_tau=0.015, yaw_shape=1.0, vy_tau=0.015, vy_shape=1.0,
        model="baseline")
    direct_yaw = _horizon_errors(
        holdout, HORIZONS_S, yaw_knots=yaw_knots, vy_knots=vy_knots,
        yaw_tau=direct_tau, yaw_shape=1.0, vy_tau=vy_tau,
        vy_shape=vy_shape, model="direct_curvature",
        curvature_knots=curvature_knots)
    direct_full = _horizon_errors(
        holdout, HORIZONS_S, yaw_knots=yaw_knots, vy_knots=vy_knots,
        yaw_tau=direct_tau, yaw_shape=1.0, vy_tau=vy_tau,
        vy_shape=vy_shape, model="direct_curvature_full",
        curvature_knots=curvature_knots)

    yaw_improvements = []
    vy_improvements = []
    for horizon in HORIZONS_S:
        base_yaw, base_vy, base_n = baseline[horizon]
        direct_yaw_rmse, direct_vy_rmse, yaw_n = direct_yaw[horizon]
        _, full_vy_rmse, full_n = direct_full[horizon]
        if base_n != 24 or yaw_n != 24 or full_n != 24:
            raise ValueError(f"external holdout incomplete at {horizon:.3f}s: "
                             f"{base_n}/{yaw_n}/{full_n}")
        yaw_improvements.append(
            1.0 - direct_yaw_rmse / base_yaw if base_yaw > 0.0 else 0.0)
        vy_improvements.append(
            1.0 - full_vy_rmse / base_vy if base_vy > 0.0 else 0.0)

    yaw_pass = (sum(value >= 0.20 for value in yaw_improvements) >= 3
                and min(yaw_improvements) >= -0.10)
    vy_pass = (sum(value >= 0.20 for value in vy_improvements) >= 3
               and min(vy_improvements) >= -0.10)
    finite = all(math.isfinite(value)
                 for table in (baseline, direct_yaw, direct_full)
                 for row in table.values() for value in row[:2])

    print(f"training capture: {training_path}")
    print(f"independent full-run holdout: {test_path}")
    print(f"training quality: {training_quality}")
    print(f"holdout quality: {test_quality}")
    print("training response knots:")
    for angle in YAW_KNOTS_RAD:
        print(f"  |steer|={angle:.2f}: K_yaw={yaw_knots[angle]:.4f} 1/m, "
              f"signed v_y={vy_knots[angle]:.4f} m/s, "
              f"q(-/+)={curvature_knots[-1][angle]:.4f}/"
              f"{curvature_knots[1][angle]:.4f}")
    print(f"fitted gain-model tau/shape={yaw_tau:.3f}s/{yaw_shape:.2f}; "
          f"direct-curvature tau={direct_tau:.3f}s; "
          f"lateral-velocity tau/shape={vy_tau:.3f}s/{vy_shape:.2f}")
    if direct_tau <= TAU_GRID_S[0] + 1e-9:
        print("  direct-curvature tau reached the 5 ms lower search bound; "
              "its physical time constant is unresolved")
    print("horizon_ms  yaw_RMSE production/direct-q  gain  vy_RMSE hold/direct-vy  gain")
    for horizon, yaw_gain, vy_gain in zip(
            HORIZONS_S, yaw_improvements, vy_improvements):
        base_yaw, base_vy, _ = baseline[horizon]
        fitted_yaw, _, _ = direct_yaw[horizon]
        _, fitted_vy, _ = direct_full[horizon]
        print(f"{horizon * 1000:10.0f}  {base_yaw:.5f}/{fitted_yaw:.5f} rad/s "
              f"{yaw_gain:+.1%}  {base_vy:.5f}/{fitted_vy:.5f} m/s {vy_gain:+.1%}")
    print(f"external recursive gate: yaw={'PASS' if yaw_pass else 'FAIL'}, "
          f"lateral_velocity={'PASS' if vy_pass else 'FAIL'}, finite={finite}")
    print("scope: conditional yaw/lateral-velocity rollout using recorded speed "
          "and steering; longitudinal speed is not predicted")
    return yaw_pass and vy_pass and finite


def evaluate_speed_surface_external(low_path: Path, high_path: Path,
                                    test_path: Path) -> bool:
    """Recursively test a speed-blended lateral-acceleration surface."""
    resolved = {path.resolve() for path in (low_path, high_path, test_path)}
    if len(resolved) != 3:
        raise ValueError("low, high, and independent test captures must differ")
    low_probes, low_quality = load_probes(low_path)
    high_probes, high_quality = load_probes(high_path)
    test_probes, test_quality = load_probes(test_path)
    if (len(low_quality["target_speeds_mps"]) != 1
            or len(high_quality["target_speeds_mps"]) != 1
            or len(test_quality["target_speeds_mps"]) != 1):
        raise ValueError("each speed-surface capture must contain one target speed")
    low_speed = low_quality["target_speeds_mps"][0]
    high_speed = high_quality["target_speeds_mps"][0]
    test_speed = test_quality["target_speeds_mps"][0]
    if not low_speed < test_speed < high_speed:
        raise ValueError("independent test speed must lie strictly between sources")
    _validate_external_capture("low-speed source", low_probes, low_quality,
                               low_speed)
    _validate_external_capture("high-speed source", high_probes, high_quality,
                               high_speed)
    _validate_external_capture("independent test", test_probes, test_quality,
                               test_speed)

    acceleration_maps = {
        low_speed: _lateral_acceleration_maps(low_probes, low_speed),
        high_speed: _lateral_acceleration_maps(high_probes, high_speed),
    }

    # Estimate one common response time constant from source captures only.
    training = [probe for source_probes in (low_probes, high_probes)
                for probe in source_probes]
    best_tau, best_error = math.nan, math.inf
    for tau in TAU_GRID_S:
        squared_error, count = 0.0, 0
        for probe in training:
            samples = [sample for sample in probe.samples
                       if sample.time_s <= FIT_WINDOW_S]
            if len(samples) < 2:
                continue
            estimate = samples[0].yaw_rate_rps
            for previous, sample in zip(samples, samples[1:]):
                dt = sample.time_s - previous.time_s
                target = _surface_yaw_target(
                    sample.speed_mps, sample.steering_rad, acceleration_maps)
                estimate = _step_first_order(estimate, target, dt, tau)
                squared_error += (estimate - sample.yaw_rate_rps) ** 2
                count += 1
        error = squared_error / count if count else math.inf
        if error < best_error:
            best_error, best_tau = error, tau
    if not math.isfinite(best_tau):
        raise ValueError("could not fit speed-surface response time constant")

    base_errors: dict[float, list[float]] = {horizon: [] for horizon in HORIZONS_S}
    fitted_tau_errors: dict[float, list[float]] = {
        horizon: [] for horizon in HORIZONS_S}
    production_tau_errors: dict[float, list[float]] = {
        horizon: [] for horizon in HORIZONS_S}
    for probe in test_probes:
        for horizon in HORIZONS_S:
            target_index = min(range(len(probe.samples)),
                               key=lambda i: abs(probe.samples[i].time_s - horizon))
            if abs(probe.samples[target_index].time_s - horizon) > 0.035:
                continue
            models = ((base_errors, "production", 0.015),
                      (fitted_tau_errors, "surface", best_tau),
                      (production_tau_errors, "surface", 0.015))
            for errors, model, tau in models:
                estimate = probe.samples[0].yaw_rate_rps
                for index in range(1, target_index + 1):
                    previous, sample = probe.samples[index - 1:index + 1]
                    dt = sample.time_s - previous.time_s
                    target = (_surface_yaw_target(
                        sample.speed_mps, sample.steering_rad, acceleration_maps)
                        if model == "surface" else _configured_yaw_target(
                            sample.speed_mps, sample.steering_rad))
                    estimate = _step_first_order(estimate, target, dt, tau)
                errors[horizon].append(
                    estimate - probe.samples[target_index].yaw_rate_rps)

    fitted_improvements: list[float] = []
    production_tau_improvements: list[float] = []
    finite = True
    print(f"low-speed source: {low_path} ({low_speed:.1f} m/s)")
    print(f"high-speed source: {high_path} ({high_speed:.1f} m/s)")
    print(f"independent recursive holdout: {test_path} ({test_speed:.1f} m/s)")
    print(f"fitted yaw response tau={best_tau:.3f}s; source-only MSE={best_error:.8g}")
    if best_tau <= TAU_GRID_S[0] + 1e-9:
        print("  tau reached the 5 ms lower search bound; faster dynamics are "
              "unresolved at 40 Hz")
    print("horizon_ms  yaw_RMSE production/fitted-tau/prior-15ms  gains")
    for horizon in HORIZONS_S:
        base = base_errors[horizon]
        fitted = fitted_tau_errors[horizon]
        prior = production_tau_errors[horizon]
        if len(base) != 24 or len(fitted) != 24 or len(prior) != 24:
            raise ValueError(f"incomplete recursive holdout at {horizon:.3f}s: "
                             f"{len(base)}/{len(fitted)}/{len(prior)}")
        base_rmse = math.sqrt(sum(error * error for error in base) / len(base))
        fitted_rmse = math.sqrt(sum(error * error for error in fitted) / len(fitted))
        prior_rmse = math.sqrt(sum(error * error for error in prior) / len(prior))
        fitted_gain = 1.0 - fitted_rmse / base_rmse if base_rmse > 0.0 else 0.0
        prior_gain = 1.0 - prior_rmse / base_rmse if base_rmse > 0.0 else 0.0
        fitted_improvements.append(fitted_gain)
        production_tau_improvements.append(prior_gain)
        finite = finite and all(math.isfinite(value) for value in
                                (base_rmse, fitted_rmse, prior_rmse))
        print(f"{horizon * 1000:10.0f}  {base_rmse:.6f}/"
              f"{fitted_rmse:.6f}/{prior_rmse:.6f} rad/s  "
              f"{fitted_gain:+.1%}/{prior_gain:+.1%}")
    fitted_pass = (sum(value >= 0.20 for value in fitted_improvements) >= 3
                   and min(fitted_improvements) >= -0.10 and finite)
    prior_pass = (sum(value >= 0.20 for value in production_tau_improvements) >= 3
                  and min(production_tau_improvements) >= -0.10 and finite)
    print(f"recursive surface gate: fitted-tau={'PASS' if fitted_pass else 'FAIL'}, "
          f"fixed prior 15 ms={'PASS' if prior_pass else 'FAIL'}; finite={finite}")
    print("scope: yaw response only; speed/steering are recorded conditional inputs; "
          "lateral-velocity dynamics are not fitted")
    return fitted_pass or prior_pass


def evaluate_dense_three_speed_surface(path: Path) -> bool:
    """Fit 3/5 m/s repetitions 1-2; hold out all 4 m/s probes and repeats."""
    probes, quality = load_probes(path)
    expected_per_speed = 2 * len(DENSE_SURFACE_KNOTS_RAD) * 3
    expected_repetitions = {"1": expected_per_speed,
                            "2": expected_per_speed,
                            "3": expected_per_speed}
    rates_ok = all(
        values["rate_hz"] is not None and values["rate_hz"] >= 38.0
        and values["gap_p95_ms"] is not None and values["gap_p95_ms"] <= 35.0
        and values["gap_max_ms"] is not None and values["gap_max_ms"] <= 60.0
        for values in quality["stream_cadence"].values())
    speed_counts = {
        speed: sum(abs(probe.target_speed_mps - speed) < 1e-6 for probe in probes)
        for speed in (3.0, 4.0, 5.0)
    }
    if (quality["target_speeds_mps"] != [3.0, 4.0, 5.0]
            or quality["valid_phase_count"] != 180
            or quality["matched_starts"] != 180
            or quality["probe_counts"] != expected_repetitions
            or any(count != expected_per_speed for count in speed_counts.values())
            or quality["collision_count_start"] != 0
            or quality["collision_count_end"] != 0
            or quality["timing_fault_samples"] != 0
            or quality["aborted"] or quality["quality_failures"]
            or not rates_ok):
        raise ValueError(f"dense speed-surface capture failed its quality gate: {quality}; "
                         f"per-speed probe counts={speed_counts}")

    # The 4 m/s speed is excluded from map fitting in its entirety. Repetitions
    # 1-2 at 3 and 5 m/s train the surface; all 4 m/s repetitions are holdout.
    training = [probe for probe in probes
                if probe.target_speed_mps in (3.0, 5.0)
                and probe.repetition in TRAIN_REPETITIONS]
    holdout = [probe for probe in probes
               if abs(probe.target_speed_mps - 4.0) < 1e-6]
    acceleration_maps = {
        speed: _lateral_acceleration_maps(
            training, speed, DENSE_SURFACE_KNOTS_RAD, TRAIN_REPETITIONS)
        for speed in (3.0, 5.0)
    }

    baseline_errors = {horizon: [] for horizon in HORIZONS_S}
    surface_errors = {horizon: [] for horizon in HORIZONS_S}
    for probe in holdout:
        for horizon in HORIZONS_S:
            target_index = min(
                range(len(probe.samples)),
                key=lambda index: abs(probe.samples[index].time_s - horizon))
            if abs(probe.samples[target_index].time_s - horizon) > 0.035:
                continue
            baseline = probe.samples[0].yaw_rate_rps
            candidate = probe.samples[0].yaw_rate_rps
            for index in range(1, target_index + 1):
                previous, sample = probe.samples[index - 1:index + 1]
                dt = sample.time_s - previous.time_s
                if dt <= 0.0 or dt > 0.075:
                    raise ValueError(f"invalid holdout sample interval {dt:.4f}s")
                baseline = _step_first_order(
                    baseline,
                    _configured_yaw_target(sample.speed_mps,
                                           sample.steering_rad),
                    dt, 0.015)
                candidate = _step_first_order(
                    candidate,
                    _surface_yaw_target(sample.speed_mps,
                                        sample.steering_rad,
                                        acceleration_maps,
                                        DENSE_SURFACE_KNOTS_RAD),
                    dt, 0.015)
            measured = probe.samples[target_index].yaw_rate_rps
            baseline_errors[horizon].append(baseline - measured)
            surface_errors[horizon].append(candidate - measured)

    improvements = []
    print(f"bag: {path}")
    print("capture PASS: 180/180 matched probes, 3 speeds x 10 signed steering "
          "knots x 3 repeats; zero collisions/timing faults; all streams >=38 Hz")
    print("training: 3 and 5 m/s repetitions 1-2; untouched speed holdout: "
          "all 60 probes at 4 m/s; fixed existing 15 ms yaw time constant")
    print("horizon_ms  configured/surface yaw RMSE (rad/s)  surface gain")
    finite = True
    for horizon in HORIZONS_S:
        baseline = baseline_errors[horizon]
        candidate = surface_errors[horizon]
        if len(baseline) != expected_per_speed or len(candidate) != expected_per_speed:
            raise ValueError(f"incomplete 4 m/s recursive holdout at {horizon:.3f}s: "
                             f"{len(baseline)}/{len(candidate)}")
        base_rmse = math.sqrt(sum(error * error for error in baseline) / len(baseline))
        surface_rmse = math.sqrt(
            sum(error * error for error in candidate) / len(candidate))
        gain = 1.0 - surface_rmse / base_rmse if base_rmse > 0.0 else 0.0
        improvements.append(gain)
        finite = finite and math.isfinite(base_rmse) and math.isfinite(surface_rmse)
        print(f"{horizon * 1000:10.0f}  {base_rmse:.6f}/{surface_rmse:.6f}  "
              f"{gain:+.1%}")

    actual_steady: dict[tuple[int, float], list[tuple[float, float, float]]] = {}
    predicted_abs_by_angle: dict[float, list[float]] = {}
    for probe in holdout:
        angle = round(abs(probe.commanded_steering_rad), 2)
        sign = 1 if probe.commanded_steering_rad > 0.0 else -1
        settled = [sample for sample in probe.samples
                   if sample.time_s >= 0.50
                   and abs(sample.steering_rad) >= DENSE_SURFACE_KNOTS_RAD[0] - 0.002]
        if len(settled) < 20:
            raise ValueError(f"insufficient steady holdout samples at {angle:.2f} rad")
        measured_rate = _median([sample.yaw_rate_rps for sample in settled])
        measured_speed = _median([sample.speed_mps for sample in settled])
        key = (sign, angle)
        actual_steady.setdefault(key, []).append((probe.commanded_steering_rad,
                                                  measured_rate, measured_speed))
        predicted_rate = _surface_yaw_target(
            measured_speed, probe.commanded_steering_rad,
            acceleration_maps, DENSE_SURFACE_KNOTS_RAD)
        predicted_abs_by_angle.setdefault(angle, []).append(abs(predicted_rate))

    derivative_checks = []
    actual_abs_by_angle: dict[float, list[float]] = {}
    for (sign, angle), rows in actual_steady.items():
        actual_abs_by_angle.setdefault(angle, []).extend(
            abs(rate) for _, rate, _ in rows)
    for sign in (-1, 1):
        signed_points = []
        for angle in DENSE_SURFACE_KNOTS_RAD:
            rows = actual_steady[(sign, angle)]
            signed_points.append((statistics.median(value[0] for value in rows),
                                  statistics.median(value[1] for value in rows),
                                  statistics.mean(value[2] for value in rows)))
        signed_points.sort(key=lambda item: item[0])
        for left, right in zip(signed_points, signed_points[1:]):
            observed_slope = (right[1] - left[1]) / (right[0] - left[0])
            midpoint_steering = 0.5 * (left[0] + right[0])
            midpoint_speed = 0.5 * (left[2] + right[2])
            predicted_slope = _surface_yaw_sensitivity(
                midpoint_speed, midpoint_steering,
                acceleration_maps, DENSE_SURFACE_KNOTS_RAD)
            if abs(observed_slope) >= 0.05:
                derivative_checks.append((observed_slope, predicted_slope,
                                          observed_slope * predicted_slope > 0.0))

    matched = sum(check[2] for check in derivative_checks)
    derivative_pass = bool(derivative_checks) and matched == len(derivative_checks)
    print("steady yaw response at 4 m/s (absolute rad/s), measured/surface:")
    for angle in DENSE_SURFACE_KNOTS_RAD:
        measured = statistics.mean(actual_abs_by_angle[angle])
        predicted = statistics.mean(predicted_abs_by_angle[angle])
        print(f"  {angle:.2f} rad: {measured:.4f}/{predicted:.4f}")
    if derivative_checks:
        print(f"local d(yaw rate)/d(steer) signs: {matched}/"
              f"{len(derivative_checks)} resolvable intervals; predicted range="
              f"{min(item[1] for item in derivative_checks):.3f}.."
              f"{max(item[1] for item in derivative_checks):.3f} 1/s")
    recursive_pass = (sum(value >= 0.20 for value in improvements) >= 3
                      and min(improvements) >= -0.10 and finite)
    passed = recursive_pass and derivative_pass
    print(f"decision: {'PASS for yaw-only MPC model-integration work' if passed else 'REJECT for production MPC; retain model as offline candidate'}; "
          f"recursive={recursive_pass}, Jacobian={derivative_pass}")
    print("scope: steady and recursive yaw only; speed/steering are recorded "
          "conditional inputs; lateral-velocity dynamics and closed-loop MPC "
          "remain unvalidated")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="path to run_0.db3")
    parser.add_argument("--external-test", type=Path,
                        help="independent fixed-speed run_0.db3; bag is training capture")
    parser.add_argument("--speed-surface-test", nargs=2, type=Path,
                        metavar=("HIGH_SOURCE", "INDEPENDENT_TEST"),
                        help="use positional bag as low-speed source and recursively "
                             "test a speed-interpolated yaw surface")
    parser.add_argument("--dense-three-speed-surface", type=Path,
                        metavar="BAG",
                        help="train on 3/5 m/s repetitions 1-2 and recursively "
                             "hold out the full 4 m/s dense steering surface")
    args = parser.parse_args()
    if args.dense_three_speed_surface is not None:
        if args.external_test is not None or args.speed_surface_test is not None:
            parser.error("choose only one external/speed-surface evaluation mode")
        return 0 if evaluate_dense_three_speed_surface(
            args.dense_three_speed_surface) else 1
    if args.speed_surface_test is not None:
        if args.external_test is not None:
            parser.error("choose --external-test or --speed-surface-test")
        return 0 if evaluate_speed_surface_external(
            args.bag, args.speed_surface_test[0],
            args.speed_surface_test[1]) else 1
    if args.external_test is not None:
        return 0 if evaluate_external(args.bag, args.external_test) else 1
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())

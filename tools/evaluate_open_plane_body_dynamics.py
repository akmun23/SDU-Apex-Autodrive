#!/usr/bin/env python3
"""Identify and whole-run test a nonlinear open-plane body dynamics model.

This development evaluator fits an empirical, differentiable piecewise-cubic
surface to measured rear-axle [u, v, yaw-rate] derivatives. Its tensor-product
terms explicitly allow speed/steering/sideslip/throttle interactions. It is
not a simulator-physics patch and does not consume these signals in the
competition runtime. Encoder and IMU streams are audited for coverage and
timing; kinematic lateral-slip values are proxies, not WheelCollider truth.
"""

from __future__ import annotations

import argparse
import bisect
import math
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as analysis
from tools.fit_open_plane_tire_model import WHEEL_X_M, WHEEL_Y_M, _ackermann


STEERING = analysis.STEERING
STEERING_COMMAND = "/autodrive/roboracer_1/steering_command"
THROTTLE = analysis.THROTTLE_FEEDBACK
THROTTLE_COMMAND = analysis.THROTTLE_COMMAND
COM_X_M = analysis.COM_X_M
STEERING_LIMIT_RAD = 0.5236
MIN_DT_S = 0.015
MAX_DT_S = 0.075
SPLINE_BASIS_COUNT = 8
SPLINE_DEGREE = 3
TENSOR_PAIRS = ((0, 3), (1, 3), (0, 1), (0, 4),
                (3, 4), (2, 3), (5, 3), (6, 4))
WHEEL_TENSOR_PAIRS = TENSOR_PAIRS + ((0, 7), (0, 8), (4, 7),
                                    (4, 8), (7, 8))
HORIZONS_S = (0.025, 0.125, 0.250, 0.500, 0.750)
SENSOR_STREAM_TOPICS = (analysis.ODOM, STEERING, THROTTLE,
                        analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER,
                        analysis.IMU, analysis.PACKET_TIMING)
COMMAND_STREAM_TOPICS = (STEERING_COMMAND, THROTTLE_COMMAND)
STREAM_TOPICS = SENSOR_STREAM_TOPICS + COMMAND_STREAM_TOPICS


@dataclass(frozen=True)
class MotionSample:
    time_s: float
    state: np.ndarray  # rear-axle [u, v, r]
    actuators: np.ndarray  # measured steering, measured throttle, then commands
    rear_wheel_surface_mps: np.ndarray | None = None


@dataclass(frozen=True)
class Capture:
    path: Path
    sequences: tuple[tuple[MotionSample, ...], ...]
    domain_samples: np.ndarray  # speed, steer, throttle, v, yaw-rate
    phase_count: int
    valid_phase_count: int
    collision_count_start: int
    collision_count_end: int
    timing_faults: int
    aborted: bool
    reason: str
    stream_stats: dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class Model:
    name: str
    bounds: np.ndarray
    knots: tuple[np.ndarray, ...]
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    coefficients: np.ndarray
    tensor_pairs: tuple[tuple[int, int], ...]
    nonlinear: bool
    use_rear_wheel_speeds: bool


def _read_scalar(connection: sqlite3.Connection, topics: dict,
                 topic: str) -> tuple[list[analysis.ScalarRow], list[int]]:
    rows = [analysis.ScalarRow(receipt, receipt, float(message.data))
            for receipt, message in analysis._messages(connection, topics, topic)
            if math.isfinite(float(message.data))]
    return rows, [row.receipt_ns for row in rows]


def _continuous_segments(rows: list[MotionSample]) -> list[tuple[MotionSample, ...]]:
    segments: list[tuple[MotionSample, ...]] = []
    start = 0
    for index in range(1, len(rows)):
        dt = rows[index].time_s - rows[index - 1].time_s
        if MIN_DT_S <= dt <= MAX_DT_S:
            continue
        if index - start >= 10:
            segments.append(tuple(rows[start:index]))
        start = index
    if len(rows) - start >= 10:
        segments.append(tuple(rows[start:]))
    return segments


def _encoder_surface_speed(rows: list[analysis.EncoderRow], times: list[int],
                           target_ns: int) -> float | None:
    """Causal rear-wheel surface speed from a 100 ms cumulative-angle window."""
    if len(rows) < 3:
        return None
    current_index = bisect.bisect_right(times, target_ns) - 1
    old_target = target_ns - 100_000_000
    old_index = bisect.bisect_left(times, old_target)
    if current_index < 0 or old_index >= len(rows):
        return None
    if old_index > 0 and abs(times[old_index - 1] - old_target) < abs(
            times[old_index] - old_target):
        old_index -= 1
    current, older = rows[current_index], rows[old_index]
    dt_s = (current.source_ns - older.source_ns) / 1e9
    if not 0.06 <= dt_s <= 0.15:
        return None
    return analysis.WHEEL_RADIUS_M * (current.angle_rad - older.angle_rad) / dt_s


def load_capture(path: Path) -> Capture:
    if not path.is_file():
        raise ValueError(f"bag does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        required = (analysis.ODOM, STEERING, STEERING_COMMAND, THROTTLE,
                    THROTTLE_COMMAND, analysis.COLLISIONS,
                    analysis.TIMING_FAULT, analysis.PHASE, *STREAM_TOPICS)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("missing topic(s): " + ", ".join(sorted(set(missing))))

        steering, steering_times = _read_scalar(connection, topics, STEERING)
        steering_command, steering_command_times = _read_scalar(
            connection, topics, STEERING_COMMAND)
        throttle, throttle_times = _read_scalar(connection, topics, THROTTLE)
        throttle_command, throttle_command_times = _read_scalar(
            connection, topics, THROTTLE_COMMAND)

        def read_encoder(topic: str) -> list[analysis.EncoderRow]:
            rows = []
            for receipt_ns, message in analysis._messages(connection, topics, topic):
                if message.position and math.isfinite(float(message.position[0])):
                    rows.append(analysis.EncoderRow(
                        receipt_ns, analysis._stamp_ns(message.header.stamp),
                        float(message.position[0])))
            return rows

        left_encoder = read_encoder(analysis.LEFT_ENCODER)
        right_encoder = read_encoder(analysis.RIGHT_ENCODER)
        left_encoder_times = [row.receipt_ns for row in left_encoder]
        right_encoder_times = [row.receipt_ns for row in right_encoder]

        odometry: list[tuple[int, np.ndarray]] = []
        domain_rows: list[tuple[float, float, float, float, float]] = []
        for receipt_ns, message in analysis._messages(
                connection, topics, analysis.ODOM):
            twist = message.twist.twist
            u = float(twist.linear.x)
            v_com = float(twist.linear.y)
            yaw_rate = float(twist.angular.z)
            if not all(math.isfinite(value) for value in (u, v_com, yaw_rate)):
                continue
            v_rear = v_com - COM_X_M * yaw_rate
            state = np.asarray((u, v_rear, yaw_rate), dtype=float)
            odometry.append((receipt_ns, state))
            aligned = (
                analysis._nearest_scalar(steering, steering_times, receipt_ns),
                analysis._nearest_scalar(throttle, throttle_times, receipt_ns),
            )
            if all(row is not None for row in aligned):
                domain_rows.append((math.hypot(u, v_rear), aligned[0].value,
                                    aligned[1].value, v_rear, yaw_rate))

        phases, experiment_end = analysis._phase_events(connection, topics)
        collision_values = [int(message.data) for _, message in
                            analysis._messages(connection, topics,
                                               analysis.COLLISIONS)]
        fault_values = [bool(message.data) for _, message in
                        analysis._messages(connection, topics,
                                           analysis.TIMING_FAULT)]
        stream_stats = {}
        for topic in STREAM_TOPICS:
            topic_id = topics[topic][0]
            receipts = [int(row[0]) for row in connection.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp, id",
                (topic_id,),
            )]
            rate, p95_gap, max_gap = analysis._rate(receipts)
            stream_stats[topic] = (
                float(rate or 0.0), float(p95_gap or math.inf),
                float(max_gap or math.inf),
            )
    finally:
        connection.close()

    sequences: list[tuple[MotionSample, ...]] = []
    for phase in phases:
        if phase.valid is not True:
            continue
        # Include the prior 250 ms so input/state history at the phase boundary
        # remains available for transient diagnostics and future model variants.
        rows: list[MotionSample] = []
        lower_ns = phase.start_ns - 250_000_000
        for receipt_ns, state in odometry:
            if receipt_ns < lower_ns or receipt_ns > phase.end_ns:
                continue
            aligned = (
                analysis._nearest_scalar(steering, steering_times, receipt_ns),
                analysis._nearest_scalar(throttle, throttle_times, receipt_ns),
                analysis._nearest_scalar(throttle_command,
                                         throttle_command_times, receipt_ns),
                analysis._nearest_scalar(steering_command,
                                         steering_command_times, receipt_ns),
            )
            if any(row is None for row in aligned):
                continue
            if max(abs(row.receipt_ns - receipt_ns) for row in aligned) > 30_000_000:
                continue
            wheel_speeds = (
                _encoder_surface_speed(left_encoder, left_encoder_times, receipt_ns),
                _encoder_surface_speed(right_encoder, right_encoder_times, receipt_ns),
            )
            rear_wheel_surface_mps = (
                np.asarray(wheel_speeds, dtype=float)
                if all(value is not None for value in wheel_speeds) else None
            )
            rows.append(MotionSample(
                (receipt_ns - phase.start_ns) / 1e9,
                state.copy(),
                np.asarray([row.value for row in aligned], dtype=float),
                rear_wheel_surface_mps,
            ))
        rows.sort(key=lambda row: row.time_s)
        sequences.extend(_continuous_segments(rows))

    odom_receipts = [row[0] for row in odometry]
    rate, _, gap_max = analysis._rate(odom_receipts)
    collision_start = collision_values[0] if collision_values else 0
    collision_end = collision_values[-1] if collision_values else 0
    return Capture(
        path=path,
        sequences=tuple(sequences),
        domain_samples=np.asarray(domain_rows, dtype=float),
        phase_count=len(phases),
        valid_phase_count=sum(phase.valid is True for phase in phases),
        collision_count_start=collision_start,
        collision_count_end=collision_end,
        timing_faults=sum(fault_values),
        aborted=bool(experiment_end.get("aborted", True)),
        reason=str(experiment_end.get("reason", "missing experiment end")),
        stream_stats=stream_stats,
    )


def _validate_capture(capture: Capture, role: str) -> None:
    if (capture.aborted or capture.valid_phase_count != capture.phase_count
            or capture.collision_count_start != 0
            or capture.collision_count_end != capture.collision_count_start
            or capture.timing_faults):
        raise ValueError(
            f"{role} is not a clean complete capture: aborted={capture.aborted}, "
            f"phases={capture.valid_phase_count}/{capture.phase_count}, collisions="
            f"{capture.collision_count_start}->{capture.collision_count_end}, "
            f"timing_faults={capture.timing_faults}, reason={capture.reason!r}")
    bad_streams = []
    for name, (rate, p95, gap) in capture.stream_stats.items():
        max_gap = 120.0 if name in COMMAND_STREAM_TOPICS else 60.0
        if rate < 38.0 or p95 > 35.0 or gap > max_gap:
            bad_streams.append(name)
    if bad_streams:
        raise ValueError(f"{role} streams outside 40 Hz limits: {bad_streams}")
    if not capture.sequences:
        raise ValueError(f"{role} has no valid, aligned motion phases")


def _wheel_slip_angles(state: np.ndarray, steering: float
                       ) -> tuple[tuple[float, float, float, float], int]:
    """Return finite wheel-local slip-angle proxies and low-forward-speed count.

    tan(alpha) is singular when a wheel's longitudinal velocity approaches
    zero. alpha=atan2(v_lateral, |v_longitudinal|) remains a meaningful,
    bounded kinematic angle through those states instead of dropping them.
    """
    u, v, yaw_rate = state
    left, right = _ackermann(steering)
    angles = []
    low_forward_speed = 0
    for x, y, steer in zip(WHEEL_X_M, WHEEL_Y_M, (left, right, 0.0, 0.0)):
        vx_body = u - yaw_rate * y
        vy_body = v + yaw_rate * x
        cosine, sine = math.cos(steer), math.sin(steer)
        vx_wheel = cosine * vx_body + sine * vy_body
        vy_wheel = -sine * vx_body + cosine * vy_body
        low_forward_speed += int(abs(vx_wheel) < 0.5)
        angles.append(math.atan2(vy_wheel, abs(vx_wheel)))
    return tuple(angles), low_forward_speed  # type: ignore[return-value]


def _raw_features(state: np.ndarray, actuators: np.ndarray,
                  rear_wheel_surface_mps: np.ndarray | None = None,
                  low_forward_speed_counter: list[int] | None = None) -> np.ndarray:
    u, v, yaw_rate = state
    steering, throttle_feedback = actuators[:2]
    slip_angles, low_forward_speed = _wheel_slip_angles(state, steering)
    if low_forward_speed_counter is not None:
        low_forward_speed_counter[0] += low_forward_speed
    front = 0.5 * (abs(slip_angles[0]) + abs(slip_angles[1]))
    rear = 0.5 * (abs(slip_angles[2]) + abs(slip_angles[3]))
    features = [u, v, yaw_rate, steering, throttle_feedback, front, rear]
    if rear_wheel_surface_mps is not None:
        half_track = analysis.TRACK_WIDTH_M / 2.0
        rear_left_ground_mps = u - yaw_rate * half_track
        rear_right_ground_mps = u + yaw_rate * half_track
        features.extend((
            float(rear_wheel_surface_mps[0]) - rear_left_ground_mps,
            float(rear_wheel_surface_mps[1]) - rear_right_ground_mps,
        ))
    return np.asarray(features, dtype=float)


def _basis(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    degree = SPLINE_DEGREE
    basis = np.zeros((len(values), len(knots) - 1), dtype=float)
    for index in range(len(knots) - 1):
        basis[:, index] = ((values >= knots[index]) & (values < knots[index + 1]))
    for order in range(1, degree + 1):
        next_basis = np.zeros((len(values), basis.shape[1] - 1), dtype=float)
        for index in range(next_basis.shape[1]):
            left_den = knots[index + order] - knots[index]
            right_den = knots[index + order + 1] - knots[index + 1]
            if left_den > 0.0:
                next_basis[:, index] += (
                    (values - knots[index]) / left_den * basis[:, index])
            if right_den > 0.0:
                next_basis[:, index] += (
                    (knots[index + order + 1] - values) / right_den
                    * basis[:, index + 1])
        basis = next_basis
    basis[values == knots[-1], :] = 0.0
    basis[values == knots[-1], -1] = 1.0
    return basis


def _design(raw: np.ndarray, model: Model | None = None,
            bounds: np.ndarray | None = None,
            knots: tuple[np.ndarray, ...] | None = None,
            tensor_pairs: tuple[tuple[int, int], ...] = TENSOR_PAIRS,
            nonlinear: bool = True) -> tuple[np.ndarray, np.ndarray,
                                              tuple[np.ndarray, ...], np.ndarray]:
    if model is not None:
        bounds, knots, tensor_pairs = model.bounds, model.knots, model.tensor_pairs
    if nonlinear:
        if bounds is None or knots is None:
            bounds = np.column_stack((raw.min(axis=0), raw.max(axis=0)))
            expanded = []
            for low, high in bounds:
                if high - low < 1e-8:
                    low -= 1e-4
                    high += 1e-4
                interior = np.linspace(low, high, SPLINE_BASIS_COUNT
                                       - SPLINE_DEGREE + 1)[1:-1]
                expanded.append(np.concatenate((np.repeat(low, SPLINE_DEGREE + 1),
                                                interior,
                                                np.repeat(high, SPLINE_DEGREE + 1))))
            knots = tuple(expanded)
        ood = np.any((raw < bounds[:, 0]) | (raw > bounds[:, 1]), axis=1)
        bases = [_basis(raw[:, index], knots[index])
                 for index in range(raw.shape[1])]
        pieces = [np.ones((len(raw), 1), dtype=float), *bases]
        pieces.extend((bases[left][:, :, None] * bases[right][:, None, :])
                      .reshape(len(raw), -1)
                      for left, right in tensor_pairs)
        return np.concatenate(pieces, axis=1), ood, knots, bounds
    return np.column_stack((np.ones(len(raw)), raw)), np.zeros(len(raw), dtype=bool), (), np.empty((0, 2))


def fit(captures: list[Capture], ridge: float, nonlinear: bool,
        use_rear_wheel_speeds: bool = False) -> tuple[Model, int, int]:
    raw_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    low_forward_speed_wheels = [0]
    for capture in captures:
        for sequence in capture.sequences:
            for index in range(1, len(sequence) - 1, 2):
                previous, current, following = sequence[index - 1:index + 2]
                if current.time_s < 0.0:
                    continue
                dt = following.time_s - previous.time_s
                if not 2.0 * MIN_DT_S <= dt <= 2.0 * MAX_DT_S:
                    continue
                if use_rear_wheel_speeds and current.rear_wheel_surface_mps is None:
                    continue
                raw_rows.append(_raw_features(
                    current.state, current.actuators,
                    current.rear_wheel_surface_mps if use_rear_wheel_speeds else None,
                    low_forward_speed_wheels))
                targets.append((following.state - previous.state) / dt)
    if len(raw_rows) < 5000:
        raise ValueError(f"too few training transitions: {len(raw_rows)}")
    raw = np.asarray(raw_rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    if len(raw) > 40000:
        rng = np.random.default_rng(20260927)
        selected = np.sort(rng.choice(len(raw), 40000, replace=False))
        raw, target = raw[selected], target[selected]
    tensor_pairs = (WHEEL_TENSOR_PAIRS if nonlinear and use_rear_wheel_speeds
                    else TENSOR_PAIRS if nonlinear else ())
    design, _, knots, bounds = _design(
        raw, tensor_pairs=tensor_pairs, nonlinear=nonlinear)
    x_mean = design.mean(axis=0)
    x_scale = design.std(axis=0)
    x_scale[x_scale < 1e-9] = 1.0
    x_scale[0] = 1.0
    x_mean[0] = 0.0
    y_mean = target.mean(axis=0)
    y_scale = target.std(axis=0)
    y_scale[y_scale < 1e-9] = 1.0
    x_scaled = (design - x_mean) / x_scale
    y_scaled = (target - y_mean) / y_scale
    penalty = ridge * len(raw) * np.eye(x_scaled.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        x_scaled.T @ x_scaled + penalty, x_scaled.T @ y_scaled)
    name = ("nonlinear_spline_rear_wheels" if nonlinear and use_rear_wheel_speeds
            else "nonlinear_spline" if nonlinear else "affine")
    return Model(name,
                 bounds, knots, x_mean, x_scale, y_mean, y_scale,
                 coefficients, tensor_pairs, nonlinear, use_rear_wheel_speeds), \
        len(raw), low_forward_speed_wheels[0]


def _derivative(model: Model, state: np.ndarray,
                actuators: np.ndarray,
                rear_wheel_surface_mps: np.ndarray | None = None
                ) -> tuple[np.ndarray, bool]:
    raw = _raw_features(
        state, actuators,
        rear_wheel_surface_mps if model.use_rear_wheel_speeds else None,
    ).reshape(1, -1)
    design, ood, _, _ = _design(raw, model=model,
                                nonlinear=model.nonlinear)
    if bool(ood[0]):
        return np.zeros(3, dtype=float), True
    normalized = (design - model.x_mean) / model.x_scale
    result = (normalized @ model.coefficients)[0] * model.y_scale + model.y_mean
    return result, False


def _rmse(values: list[float]) -> float:
    return math.sqrt(statistics.mean(value * value for value in values)) if values else math.inf


def score(model: Model, capture: Capture) -> dict:
    one_step = [[], [], []]
    recursive = {horizon: [[], [], []] for horizon in HORIZONS_S}
    total = 0
    out_of_domain = 0
    valid_rollouts = {horizon: 0 for horizon in HORIZONS_S}
    rollout_counts = {horizon: 0 for horizon in HORIZONS_S}

    for sequence in capture.sequences:
        for index in range(1, len(sequence) - 1):
            previous, current, following = sequence[index - 1:index + 2]
            if current.time_s < 0.0:
                continue
            dt = following.time_s - current.time_s
            if not MIN_DT_S <= dt <= MAX_DT_S:
                continue
            if (model.use_rear_wheel_speeds
                    and current.rear_wheel_surface_mps is None):
                continue
            derivative, ood = _derivative(model, current.state,
                                          current.actuators,
                                          current.rear_wheel_surface_mps)
            total += 1
            out_of_domain += int(ood)
            if ood:
                continue
            predicted = current.state + dt * derivative
            for axis in range(3):
                one_step[axis].append(predicted[axis] - following.state[axis])

        times = [row.time_s for row in sequence]
        start = next((index for index, value in enumerate(times) if value >= 0.0), None)
        if start is None:
            continue
        for horizon in HORIZONS_S:
            target_index = min(range(start, len(sequence)),
                               key=lambda index: abs(times[index] - horizon))
            if abs(times[target_index] - horizon) > 0.04:
                continue
            rollout_counts[horizon] += 1
            state = sequence[start].state.copy()
            failed_support = False
            for index in range(start, target_index):
                current, following = sequence[index:index + 2]
                dt = following.time_s - current.time_s
                if not MIN_DT_S <= dt <= MAX_DT_S:
                    failed_support = True
                    break
                if (model.use_rear_wheel_speeds
                        and current.rear_wheel_surface_mps is None):
                    failed_support = True
                    break
                derivative, ood = _derivative(
                    model, state, current.actuators,
                    current.rear_wheel_surface_mps)
                if ood:
                    failed_support = True
                    out_of_domain += 1
                    break
                state = state + dt * derivative
            if failed_support:
                continue
            valid_rollouts[horizon] += 1
            actual = sequence[target_index].state
            for axis in range(3):
                recursive[horizon][axis].append(state[axis] - actual[axis])

    axes = ("u_mps", "v_rear_mps", "yaw_rate_rps")
    return {
        "sequences": len(capture.sequences),
        "one_step_samples": [len(values) for values in one_step],
        "one_step_rmse": {name: _rmse(one_step[i]) for i, name in enumerate(axes)},
        "recursive_rmse": {
            horizon: {name: _rmse(recursive[horizon][i])
                      for i, name in enumerate(axes)}
            for horizon in HORIZONS_S
        },
        "support_fraction": 1.0 - out_of_domain / max(total, 1),
        "valid_rollouts": valid_rollouts,
        "rollout_counts": rollout_counts,
    }


def _print_capture(capture: Capture, role: str) -> None:
    speeds = capture.domain_samples[:, 0]
    print(f"{role}: {capture.path}; phases={capture.valid_phase_count}/"
          f"{capture.phase_count}; speed={np.percentile(speeds, 1):.2f}.."
          f"{np.percentile(speeds, 99):.2f} m/s (p01..p99); "
          f"steering={capture.domain_samples[:, 1].min():+.4f}.."
          f"{capture.domain_samples[:, 1].max():+.4f} rad; "
          f"throttle feedback={capture.domain_samples[:, 2].min():.3f}.."
          f"{capture.domain_samples[:, 2].max():.3f}")
    for topic, (rate, p95, gap) in capture.stream_stats.items():
        print(f"  {topic}: {rate:.3f} Hz, gap p95/max={p95:.2f}/{gap:.2f} ms")


def _print_nonlinear_observations(capture: Capture) -> None:
    rows = capture.domain_samples
    speed = rows[:, 0]
    steering = rows[:, 1]
    throttle = rows[:, 2]
    yaw = rows[:, 4]
    print("measured nonlinear response slices (observational; not causal force fits):")
    speed_bands = ((0.5, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 9.0))
    steer_bands = ((0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
                   (0.40, 0.50), (0.50, 0.524))
    for speed_low, speed_high in speed_bands:
        slices = []
        for steer_low, steer_high in steer_bands:
            select = ((speed >= speed_low) & (speed < speed_high)
                      & (np.abs(steering) >= steer_low)
                      & (np.abs(steering) < steer_high)
                      & (np.abs(steering) > 1e-6))
            count = int(np.count_nonzero(select))
            if count < 50:
                continue
            yaw_gain = yaw[select] / (speed[select] * np.tan(steering[select]))
            lateral_accel = np.abs(speed[select] * yaw[select])
            slices.append(
                f"|d|{steer_low:.2f}-{steer_high:.2f}: n={count}, "
                f"K_yaw p50={np.median(yaw_gain):.3f}/m, "
                f"|a_y| p50={np.median(lateral_accel):.2f}m/s2, "
                f"throttle p50={np.median(throttle[select]):.3f}")
        if slices:
            print(f"  speed {speed_low:.1f}-{speed_high:.1f} m/s: "
                  + " | ".join(slices))


def evaluate(source_paths: list[Path], holdout_path: Path,
             ridge: float, output_model: Path | None) -> bool:
    if not source_paths:
        raise ValueError("at least one --source-bag is required")
    if ridge <= 0.0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and positive")
    sources = [load_capture(path) for path in source_paths]
    holdout = load_capture(holdout_path)
    for index, capture in enumerate(sources, 1):
        _validate_capture(capture, f"source {index}")
        _print_capture(capture, f"source {index}")
    _validate_capture(holdout, "whole-run holdout")
    _print_capture(holdout, "whole-run holdout")
    if any(capture.path.resolve() == holdout.path.resolve() for capture in sources):
        raise ValueError("whole-run holdout cannot also be a training source")

    affine, affine_count, _ = fit(sources, ridge, nonlinear=False)
    spline, spline_count, low_forward_speed_wheels = fit(
        sources, ridge, nonlinear=True)
    wheel_spline, wheel_spline_count, wheel_low_forward_speed_wheels = fit(
        sources, ridge, nonlinear=True, use_rear_wheel_speeds=True)
    affine_result = score(affine, holdout)
    spline_result = score(spline, holdout)
    wheel_spline_result = score(wheel_spline, holdout)
    print(f"training transitions: affine={affine_count}, nonlinear={spline_count}, "
          f"rear-wheel nonlinear={wheel_spline_count}; "
          f"cubic tensor splines, ridge={ridge:g}")
    if low_forward_speed_wheels:
        print(f"wheel-local proxy: {low_forward_speed_wheels} wheel-samples had "
              "|v_longitudinal| < 0.5 m/s; retained with finite atan2 slip "
              "angles rather than discarding singular tan(alpha) rows")
    if wheel_low_forward_speed_wheels:
        print(f"rear-wheel model: {wheel_low_forward_speed_wheels} low-forward-"
              "speed wheel samples retained")
    _print_nonlinear_observations(holdout)
    print("whole-run held-out one-step state RMSE, affine / nonlinear / "
          "rear-wheel nonlinear:")
    for axis in ("u_mps", "v_rear_mps", "yaw_rate_rps"):
        print(f"  {axis}: {affine_result['one_step_rmse'][axis]:.5f} / "
              f"{spline_result['one_step_rmse'][axis]:.5f} / "
              f"{wheel_spline_result['one_step_rmse'][axis]:.5f}")
    print(f"nonlinear held-out one-step feature support: "
          f"{spline_result['support_fraction']:.1%}")
    print(f"rear-wheel nonlinear held-out one-step feature support: "
          f"{wheel_spline_result['support_fraction']:.1%}")
    print("recursive held-out RMSE, affine / nonlinear / rear-wheel nonlinear:")
    for horizon in HORIZONS_S:
        affine_row = affine_result["recursive_rmse"][horizon]
        spline_row = spline_result["recursive_rmse"][horizon]
        wheel_row = wheel_spline_result["recursive_rmse"][horizon]
        valid = wheel_spline_result["valid_rollouts"][horizon]
        total = wheel_spline_result["rollout_counts"][horizon]
        print(f"  {horizon * 1000:3.0f} ms ({valid}/{total} wheel-supported): "
              + ", ".join(
                  f"{axis}={affine_row[axis]:.5f}/{spline_row[axis]:.5f}/"
                  f"{wheel_row[axis]:.5f}"
                  for axis in ("u_mps", "v_rear_mps", "yaw_rate_rps")))
    if output_model is not None:
        output_model.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_model,
            bounds=spline.bounds,
            knots=np.stack(spline.knots),
            x_mean=spline.x_mean,
            x_scale=spline.x_scale,
            y_mean=spline.y_mean,
            y_scale=spline.y_scale,
            coefficients=spline.coefficients,
            tensor_pairs=np.asarray(spline.tensor_pairs, dtype=np.int32),
            nonlinear=np.asarray(spline.nonlinear),
            use_rear_wheel_speeds=np.asarray(spline.use_rear_wheel_speeds),
            feature_names=np.asarray(("u_rear", "v_rear", "yaw_rate", "steering",
                                      "throttle_feedback", "front_abs_slip_angle",
                                      "rear_abs_slip_angle")),
        )
        print(f"saved offline nonlinear candidate: {output_model}")
    print("Scope: state-only spline is conditioned on measured actuator feedback; "
          "rear-wheel comparison additionally uses 100 ms encoder-derived "
          "wheel-surface/slip-velocity inputs and is therefore conditional, "
          "not a standalone simulator. Slip angles are kinematic proxies, not "
          "tire forces. Actuator and wheel-speed dynamics remain unidentified. "
          "No competition runtime changed.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bag", action="append", type=Path, default=[],
                        help="clean training bag; may be repeated")
    parser.add_argument("--holdout-bag", type=Path, required=True,
                        help="independent clean run withheld in full")
    parser.add_argument("--ridge", type=float, default=0.01,
                        help="dimensionless ridge factor scaled by sample count")
    parser.add_argument("--output-model", type=Path,
                        help="optional path for the fitted offline .npz candidate")
    args = parser.parse_args()
    try:
        evaluate(args.source_bag, args.holdout_bag,
                 args.ridge, args.output_model)
    except (ValueError, sqlite3.Error, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"body model evaluation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import json
import math
import sqlite3
import statistics
from dataclasses import dataclass, field, replace
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
MAX_COMMAND_AGE_NS = 120_000_000
SPLINE_BASIS_COUNT = 8
SPLINE_DEGREE = 3
TENSOR_PAIRS = ((0, 3), (1, 3), (0, 1), (0, 4),
                (3, 4), (2, 3), (5, 3), (6, 4))
WHEEL_TENSOR_PAIRS = TENSOR_PAIRS + ((0, 7), (0, 8), (4, 7),
                                    (4, 8), (7, 8))
FEATURE_NAMES = ("u_mps", "v_rear_mps", "yaw_rate_rps", "steering_rad",
                 "throttle_feedback", "front_abs_slip_angle_rad",
                 "rear_abs_slip_angle_rad", "left_rear_slip_velocity_mps",
                 "right_rear_slip_velocity_mps")
ACTUATOR_HISTORY_NAMES = ("steering_rate_radps", "throttle_rate_per_s",
                          "steering_command_rad", "throttle_command")
HORIZONS_S = (0.025, 0.125, 0.250, 0.500, 0.750)
UNPHASED_ROLLOUT_SPACING_S = 0.500
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
    source_stamp_ns: int
    rear_wheel_surface_mps: np.ndarray | None = None
    feature_state: np.ndarray | None = None  # optional legal observer state
    lap_count: int | None = None  # whole-run statistical cluster, when available
    actuator_history: np.ndarray | None = None  # rates and current commands
    receipt_ns: int = 0  # odometry receipt-time coordinate for raw-stream joins
    imu_acceleration_mps2: np.ndarray | None = None  # causal body-frame [ax, ay]
    imu_yaw_rate_rps: float | None = None  # causal body-frame gyro z
    imu_roll_pitch_rad: np.ndarray | None = None  # measured roll, pitch
    imu_roll_pitch_rate_rps: np.ndarray | None = None  # body gyro x, y


@dataclass(frozen=True)
class Capture:
    path: Path
    sequences: tuple[tuple[MotionSample, ...], ...]
    sequence_labels: tuple[str, ...]
    domain_samples: np.ndarray  # speed, steer, throttle, v, yaw-rate
    has_phase_markers: bool
    final_lap_count: int | None
    phase_count: int
    valid_phase_count: int
    invalid_phase_count: int
    unscored_phase_count: int
    collision_count_start: int
    collision_count_end: int
    timing_faults: int
    aborted: bool
    reason: str
    stream_stats: dict[str, tuple[float, float, float]]
    phase_stream_stats: dict[str, tuple[float, float, float]] = field(
        default_factory=dict)
    actuator_streams: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict)
    phase_start_times_ns: tuple[int, ...] = ()


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
    body_acceleration_coordinates: bool = False
    include_actuator_history: bool = False


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


def _causal_scalar(rows: list[analysis.ScalarRow], times: list[int],
                   target_ns: int, max_age_ns: int) -> analysis.ScalarRow | None:
    index = bisect.bisect_right(times, target_ns) - 1
    if index < 0:
        return None
    row = rows[index]
    return row if target_ns - row.receipt_ns <= max_age_ns else None


def _roll_pitch_from_quaternion(x: float, y: float, z: float,
                                w: float) -> tuple[float, float] | None:
    """Return finite roll/pitch from a normalized IMU quaternion."""
    values = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(values).all() or not math.isfinite(norm) or norm < 0.5:
        return None
    x, y, z, w = values / norm
    roll = math.atan2(2.0 * (w * x + y * z),
                      1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    return (roll, pitch) if math.isfinite(roll) and math.isfinite(pitch) else None


def _command_at_receipt(capture: Capture, channel: str,
                        target_ns: int) -> float | None:
    command_times, command_values = capture.actuator_streams[
        f"{channel}_command"]
    index = bisect.bisect_right(command_times, target_ns) - 1
    if index < 0:
        return None
    if target_ns - int(command_times[index]) > MAX_COMMAND_AGE_NS:
        return None
    value = float(command_values[index])
    return (float(np.float32(value * STEERING_LIMIT_RAD))
            if channel == "steering" else value)


def _attach_actuator_history(rows: list[MotionSample]) -> list[MotionSample]:
    """Attach causal measured actuator rates and current command values."""
    updated: list[MotionSample] = []
    for index, sample in enumerate(rows):
        rates = np.zeros(2, dtype=float)
        if index:
            previous = rows[index - 1]
            dt_s = sample.time_s - previous.time_s
            if MIN_DT_S <= dt_s <= MAX_DT_S:
                rates = (sample.actuators[:2] - previous.actuators[:2]) / dt_s
        commands = np.asarray((
            float(np.float32(sample.actuators[3] * STEERING_LIMIT_RAD)),
            float(sample.actuators[2])), dtype=float)
        updated.append(replace(
            sample, actuator_history=np.concatenate((rates, commands))))
    return updated


def _nearest_time_index(times: list[float], target: float,
                       first_index: int = 0) -> int:
    right = bisect.bisect_left(times, target, lo=first_index)
    if right >= len(times):
        return len(times) - 1
    if right > first_index and target - times[right - 1] <= times[right] - target:
        return right - 1
    return right


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
                    analysis.TIMING_FAULT, *STREAM_TOPICS)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("missing topic(s): " + ", ".join(sorted(set(missing))))
        has_phase_markers = analysis.PHASE in topics
        lap_count_topic = "/autodrive/roboracer_1/lap_count"
        if not has_phase_markers and lap_count_topic not in topics:
            raise ValueError(
                "unphased whole-run bag must contain " + lap_count_topic)

        steering, steering_times = _read_scalar(connection, topics, STEERING)
        steering_command, steering_command_times = _read_scalar(
            connection, topics, STEERING_COMMAND)
        throttle, throttle_times = _read_scalar(connection, topics, THROTTLE)
        throttle_command, throttle_command_times = _read_scalar(
            connection, topics, THROTTLE_COMMAND)
        imu_ax: list[analysis.ScalarRow] = []
        imu_ay: list[analysis.ScalarRow] = []
        imu_wz: list[analysis.ScalarRow] = []
        imu_roll: list[analysis.ScalarRow] = []
        imu_pitch: list[analysis.ScalarRow] = []
        imu_wx: list[analysis.ScalarRow] = []
        imu_wy: list[analysis.ScalarRow] = []
        for receipt_ns, message in analysis._messages(
                connection, topics, analysis.IMU):
            ax = float(message.linear_acceleration.x)
            ay = float(message.linear_acceleration.y)
            wx = float(message.angular_velocity.x)
            wy = float(message.angular_velocity.y)
            wz = float(message.angular_velocity.z)
            source_ns = analysis._stamp_ns(message.header.stamp)
            roll_pitch = _roll_pitch_from_quaternion(
                float(message.orientation.x), float(message.orientation.y),
                float(message.orientation.z), float(message.orientation.w))
            if math.isfinite(ax):
                imu_ax.append(analysis.ScalarRow(receipt_ns, source_ns, ax))
            if math.isfinite(ay):
                imu_ay.append(analysis.ScalarRow(receipt_ns, source_ns, ay))
            if roll_pitch is not None:
                imu_roll.append(analysis.ScalarRow(receipt_ns, source_ns,
                                                   roll_pitch[0]))
                imu_pitch.append(analysis.ScalarRow(receipt_ns, source_ns,
                                                    roll_pitch[1]))
            if math.isfinite(wx):
                imu_wx.append(analysis.ScalarRow(receipt_ns, source_ns, wx))
            if math.isfinite(wy):
                imu_wy.append(analysis.ScalarRow(receipt_ns, source_ns, wy))
            if math.isfinite(wz):
                imu_wz.append(analysis.ScalarRow(receipt_ns, source_ns, wz))
        imu_ax_times = [row.receipt_ns for row in imu_ax]
        imu_ay_times = [row.receipt_ns for row in imu_ay]
        imu_wz_times = [row.receipt_ns for row in imu_wz]
        imu_roll_times = [row.receipt_ns for row in imu_roll]
        imu_pitch_times = [row.receipt_ns for row in imu_pitch]
        imu_wx_times = [row.receipt_ns for row in imu_wx]
        imu_wy_times = [row.receipt_ns for row in imu_wy]

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

        odometry: list[tuple[int, int, np.ndarray]] = []
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
            source_stamp_ns = analysis._stamp_ns(message.header.stamp)
            if source_stamp_ns <= 0:
                source_stamp_ns = receipt_ns
            odometry.append((receipt_ns, source_stamp_ns, state))
            aligned = (
                _causal_scalar(steering, steering_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(throttle, throttle_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
            )
            if all(row is not None for row in aligned):
                domain_rows.append((math.hypot(u, v_rear), aligned[0].value,
                                    aligned[1].value, v_rear, yaw_rate))

        if has_phase_markers:
            phases, experiment_end = analysis._phase_events(connection, topics)
        else:
            phases, experiment_end = [], {}
        lap_count_events = ([(int(receipt_ns), int(message.data))
                             for receipt_ns, message in analysis._messages(
                                 connection, topics, lap_count_topic)]
                            if lap_count_topic in topics else [])
        collision_values = [int(message.data) for _, message in
                            analysis._messages(connection, topics,
                                               analysis.COLLISIONS)]
        fault_values = [bool(message.data) for _, message in
                        analysis._messages(connection, topics,
                                           analysis.TIMING_FAULT)]
        stream_stats = {}
        phase_stream_stats = {}
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
            if phases:
                active_gaps_ms: list[float] = []
                active_intervals = 0
                active_duration_s = 0.0
                for phase in phases:
                    first = bisect.bisect_left(receipts, phase.start_ns)
                    last = bisect.bisect_right(receipts, phase.end_ns)
                    selected = receipts[first:last]
                    if len(selected) < 2:
                        continue
                    active_intervals += len(selected) - 1
                    active_duration_s += (selected[-1] - selected[0]) / 1e9
                    active_gaps_ms.extend(
                        (right - left) / 1e6
                        for left, right in zip(selected, selected[1:]))
                active_rate = (active_intervals / active_duration_s
                               if active_duration_s > 0.0 else 0.0)
                phase_stream_stats[topic] = (
                    active_rate,
                    float(np.quantile(active_gaps_ms, 0.95))
                    if active_gaps_ms else math.inf,
                    max(active_gaps_ms) if active_gaps_ms else math.inf,
                )
    finally:
        connection.close()

    sequences: list[tuple[MotionSample, ...]] = []
    sequence_labels: list[str] = []
    lap_receipts = [receipt_ns for receipt_ns, _ in lap_count_events]
    lap_values = [count for _, count in lap_count_events]

    def lap_at(receipt_ns: int) -> int | None:
        index = bisect.bisect_right(lap_receipts, receipt_ns) - 1
        return lap_values[index] if index >= 0 else None

    for phase in phases:
        if phase.valid is not True:
            continue
        # Include 500 ms of causal context for history-state diagnostics. The
        # model-fitting loop ignores negative phase times, so this does not add
        # pre-phase transitions to training.
        rows: list[MotionSample] = []
        lower_ns = phase.start_ns - 500_000_000
        for receipt_ns, source_stamp_ns, state in odometry:
            if receipt_ns < lower_ns or receipt_ns > phase.end_ns:
                continue
            aligned = (
                _causal_scalar(steering, steering_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(throttle, throttle_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(throttle_command, throttle_command_times,
                               receipt_ns, MAX_COMMAND_AGE_NS),
                _causal_scalar(steering_command, steering_command_times,
                               receipt_ns, MAX_COMMAND_AGE_NS),
            )
            if any(row is None for row in aligned):
                continue
            aligned_imu = (
                _causal_scalar(imu_ax, imu_ax_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(imu_ay, imu_ay_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(imu_wz, imu_wz_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
            )
            imu_acceleration = (
                np.asarray([row.value for row in aligned_imu[:2]], dtype=float)
                if all(row is not None for row in aligned_imu[:2]) else None)
            aligned_attitude = (
                _causal_scalar(imu_roll, imu_roll_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(imu_pitch, imu_pitch_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(imu_wx, imu_wx_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(imu_wy, imu_wy_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
            )
            imu_body_attitude = (
                np.asarray([row.value for row in aligned_attitude[:2]], dtype=float)
                if all(row is not None for row in aligned_attitude[:2]) else None)
            imu_body_attitude_rate = (
                np.asarray([row.value for row in aligned_attitude[2:]], dtype=float)
                if all(row is not None for row in aligned_attitude[2:]) else None)
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
                source_stamp_ns,
                rear_wheel_surface_mps,
                receipt_ns=receipt_ns,
                imu_acceleration_mps2=imu_acceleration,
                imu_yaw_rate_rps=(aligned_imu[2].value
                                  if aligned_imu[2] is not None else None),
                imu_roll_pitch_rad=imu_body_attitude,
                imu_roll_pitch_rate_rps=imu_body_attitude_rate,
            ))
        rows.sort(key=lambda row: row.time_s)
        rows = _attach_actuator_history(rows)
        segments = _continuous_segments(rows)
        sequences.extend(segments)
        sequence_labels.extend([phase.label] * len(segments))

    if not has_phase_markers and odometry:
        rows = []
        time_origin_ns = odometry[0][0]
        for receipt_ns, source_stamp_ns, state in odometry:
            aligned = (
                _causal_scalar(steering, steering_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(throttle, throttle_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(throttle_command, throttle_command_times,
                               receipt_ns, MAX_COMMAND_AGE_NS),
                _causal_scalar(steering_command, steering_command_times,
                               receipt_ns, MAX_COMMAND_AGE_NS),
            )
            if any(row is None for row in aligned):
                continue
            aligned_imu = (
                _causal_scalar(imu_ax, imu_ax_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(imu_ay, imu_ay_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(imu_wz, imu_wz_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
            )
            imu_acceleration = (
                np.asarray([row.value for row in aligned_imu[:2]], dtype=float)
                if all(row is not None for row in aligned_imu[:2]) else None)
            aligned_attitude = (
                _causal_scalar(imu_roll, imu_roll_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(imu_pitch, imu_pitch_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(imu_wx, imu_wx_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
                _causal_scalar(imu_wy, imu_wy_times, receipt_ns,
                               analysis.ALIGNMENT_LIMIT_NS),
            )
            imu_body_attitude = (
                np.asarray([row.value for row in aligned_attitude[:2]], dtype=float)
                if all(row is not None for row in aligned_attitude[:2]) else None)
            imu_body_attitude_rate = (
                np.asarray([row.value for row in aligned_attitude[2:]], dtype=float)
                if all(row is not None for row in aligned_attitude[2:]) else None)
            wheel_speeds = (
                _encoder_surface_speed(left_encoder, left_encoder_times, receipt_ns),
                _encoder_surface_speed(right_encoder, right_encoder_times, receipt_ns),
            )
            rear_wheel_surface_mps = (
                np.asarray(wheel_speeds, dtype=float)
                if all(value is not None for value in wheel_speeds) else None
            )
            rows.append(MotionSample(
                (receipt_ns - time_origin_ns) / 1e9,
                state.copy(),
                np.asarray([row.value for row in aligned], dtype=float),
                source_stamp_ns,
                rear_wheel_surface_mps,
                lap_count=lap_at(receipt_ns),
                receipt_ns=receipt_ns,
                imu_acceleration_mps2=imu_acceleration,
                imu_yaw_rate_rps=(aligned_imu[2].value
                                  if aligned_imu[2] is not None else None),
                imu_roll_pitch_rad=imu_body_attitude,
                imu_roll_pitch_rate_rps=imu_body_attitude_rate,
            ))
        rows.sort(key=lambda row: row.time_s)
        rows = _attach_actuator_history(rows)
        segments = _continuous_segments(rows)
        for segment in segments:
            lap_segment: list[MotionSample] = []
            for sample in segment:
                if (lap_segment
                        and sample.lap_count != lap_segment[-1].lap_count):
                    sequences.append(tuple(lap_segment))
                    label = lap_segment[-1].lap_count
                    sequence_labels.append(
                        f"lap_count={label}" if label is not None
                        else "lap_count=unknown")
                    lap_segment = []
                lap_segment.append(sample)
            if lap_segment:
                sequences.append(tuple(lap_segment))
                label = lap_segment[-1].lap_count
                sequence_labels.append(
                    f"lap_count={label}" if label is not None
                    else "lap_count=unknown")

    odom_receipts = [row[0] for row in odometry]
    rate, _, gap_max = analysis._rate(odom_receipts)
    collision_start = collision_values[0] if collision_values else 0
    collision_end = collision_values[-1] if collision_values else 0
    final_lap_count = lap_values[-1] if lap_values else None
    unphased_complete = (not has_phase_markers and final_lap_count is not None
                         and final_lap_count >= 12)
    return Capture(
        path=path,
        sequences=tuple(sequences),
        sequence_labels=tuple(sequence_labels),
        domain_samples=np.asarray(domain_rows, dtype=float),
        has_phase_markers=has_phase_markers,
        final_lap_count=final_lap_count,
        phase_count=len(phases),
        valid_phase_count=sum(phase.valid is True for phase in phases),
        invalid_phase_count=sum(phase.valid is False for phase in phases),
        unscored_phase_count=sum(phase.valid is None for phase in phases),
        collision_count_start=collision_start,
        collision_count_end=collision_end,
        timing_faults=sum(fault_values),
        aborted=bool(experiment_end.get("aborted", not unphased_complete)),
        reason=str(experiment_end.get(
            "reason", f"unphased whole run; final lap count={final_lap_count}")),
        stream_stats=stream_stats,
        phase_stream_stats=phase_stream_stats,
        actuator_streams={
            "steering_feedback": (
                np.asarray(steering_times, dtype=np.int64),
                np.asarray([row.value for row in steering], dtype=float)),
            "steering_command": (
                np.asarray(steering_command_times, dtype=np.int64),
                np.asarray([row.value for row in steering_command], dtype=float)),
            "throttle_feedback": (
                np.asarray(throttle_times, dtype=np.int64),
                np.asarray([row.value for row in throttle], dtype=float)),
            "throttle_command": (
                np.asarray(throttle_command_times, dtype=np.int64),
                np.asarray([row.value for row in throttle_command], dtype=float)),
        },
        phase_start_times_ns=tuple(
            phase.start_ns for phase in phases if phase.valid is True),
    )


def _read_replayed_states(path: Path,
                          topic: str = "/replayed_odom") -> dict[int, np.ndarray]:
    """Read timestamped body-twist state features from a ROS bag topic."""
    if not path.is_file():
        raise ValueError(f"state feature bag does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        if topic not in topics:
            raise ValueError(f"state feature bag lacks {topic}: {path}")
        states: dict[int, np.ndarray] = {}
        for _, message in analysis._messages(connection, topics, topic):
            source_ns = analysis._stamp_ns(message.header.stamp)
            if source_ns <= 0:
                continue
            twist = message.twist.twist
            state = np.asarray((twist.linear.x, twist.linear.y,
                                twist.angular.z), dtype=float)
            if np.isfinite(state).all():
                states[source_ns] = state
    finally:
        connection.close()
    if not states:
        raise ValueError(f"state feature bag has no stamped states on {topic}: {path}")
    return states


def _attach_feature_states(capture: Capture, path: Path,
                           topic: str = "/replayed_odom",
                           min_coverage: float = 0.98) -> tuple[Capture, float]:
    """Attach observer states without replacing simulator-truth target labels."""
    states = _read_replayed_states(path, topic)
    total = matched = 0
    sequences: list[tuple[MotionSample, ...]] = []
    for sequence in capture.sequences:
        updated = []
        for sample in sequence:
            feature_state = states.get(sample.source_stamp_ns)
            total += 1
            if feature_state is None:
                feature_state = np.full(3, np.nan, dtype=float)
            else:
                matched += 1
            updated.append(replace(sample, feature_state=feature_state))
        sequences.append(tuple(updated))
    coverage = matched / total if total else 0.0
    if coverage < min_coverage:
        raise ValueError(
            f"state feature topic {topic} covers only {coverage:.2%} of "
            f"{capture.path} samples; "
            f"need at least {min_coverage:.0%}")
    return replace(capture, sequences=tuple(sequences)), coverage


def _feature_state(sample: MotionSample) -> np.ndarray:
    return sample.state if sample.feature_state is None else sample.feature_state


def _validate_capture(capture: Capture, role: str) -> None:
    if (capture.aborted or capture.invalid_phase_count
            or capture.collision_count_start != 0
            or capture.collision_count_end != capture.collision_count_start
            or capture.timing_faults):
        raise ValueError(
            f"{role} is not a clean complete capture: aborted={capture.aborted}, "
            f"valid/invalid/unscored phases={capture.valid_phase_count}/"
            f"{capture.invalid_phase_count}/{capture.unscored_phase_count}, collisions="
            f"{capture.collision_count_start}->{capture.collision_count_end}, "
            f"timing_faults={capture.timing_faults}, reason={capture.reason!r}")
    if not capture.has_phase_markers and (
            capture.final_lap_count is None or capture.final_lap_count < 12):
        raise ValueError(
            f"{role} lacks phase markers and does not prove a complete 12-lap run: "
            f"final_lap_count={capture.final_lap_count}")
    bad_streams = []
    quality_stats = capture.stream_stats
    if capture.has_phase_markers and capture.phase_stream_stats:
        quality_stats = capture.phase_stream_stats
    for name, (rate, p95, gap) in quality_stats.items():
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
                  low_forward_speed_counter: list[int] | None = None,
                  actuator_history: np.ndarray | None = None) -> np.ndarray:
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
    if actuator_history is not None:
        if actuator_history.shape != (len(ACTUATOR_HISTORY_NAMES),):
            raise ValueError("actuator history must contain four causal features")
        features.extend(actuator_history.tolist())
    return np.asarray(features, dtype=float)


def _model_feature_names(model: Model) -> tuple[str, ...]:
    names = (FEATURE_NAMES if model.use_rear_wheel_speeds
             else FEATURE_NAMES[:7])
    if model.include_actuator_history:
        names += ACTUATOR_HISTORY_NAMES
    return names


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
        use_rear_wheel_speeds: bool = False,
        balance_runs: bool = False,
        body_acceleration_coordinates: bool = False,
        include_actuator_history: bool = False
        ) -> tuple[Model, int, int]:
    raw_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    run_ids: list[int] = []
    low_forward_speed_wheels = [0]
    for run_index, capture in enumerate(captures):
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
                state = _feature_state(current)
                if not np.isfinite(state).all():
                    continue
                raw_rows.append(_raw_features(
                    state, current.actuators,
                    current.rear_wheel_surface_mps if use_rear_wheel_speeds else None,
                    low_forward_speed_wheels,
                    current.actuator_history if include_actuator_history else None))
                target = (following.state - previous.state) / dt
                if body_acceleration_coordinates:
                    # Resolve the exact rotating-body-frame transport terms;
                    # fit physical-axis acceleration, then restore these
                    # known kinematics during recursive prediction.
                    truth_u, truth_v, truth_r = current.state
                    target[0] -= truth_v * truth_r
                    target[1] += truth_u * truth_r
                targets.append(target)
                run_ids.append(run_index)
    if len(raw_rows) < 5000:
        raise ValueError(f"too few training transitions: {len(raw_rows)}")
    raw = np.asarray(raw_rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    run_ids_array = np.asarray(run_ids, dtype=np.int32)
    if balance_runs:
        # Bound dense spline-fit cost without letting the longest bag dominate
        # training. Keep at most 10k unique rows from each independent run;
        # smaller runs stay complete and receive equal total regression weight.
        rng = np.random.default_rng(20260927)
        selected_parts = []
        for run_index in np.unique(run_ids_array):
            indices = np.flatnonzero(run_ids_array == run_index)
            if len(indices) > 10000:
                indices = np.sort(rng.choice(indices, 10000, replace=False))
            selected_parts.append(indices)
        selected = np.concatenate(selected_parts)
        raw = raw[selected]
        target = target[selected]
        run_ids_array = run_ids_array[selected]
    elif len(raw) > 40000:
        rng = np.random.default_rng(20260927)
        selected = np.sort(rng.choice(len(raw), 40000, replace=False))
        raw, target = raw[selected], target[selected]
        run_ids_array = run_ids_array[selected]
    tensor_pairs = (WHEEL_TENSOR_PAIRS if nonlinear and use_rear_wheel_speeds
                    else TENSOR_PAIRS if nonlinear else ())
    if nonlinear and include_actuator_history:
        history_offset = 9 if use_rear_wheel_speeds else 7
        tensor_pairs += (
            (0, history_offset + 1), (3, history_offset + 1),
            (3, history_offset + 3), (4, history_offset + 3),
        )
    design, _, knots, bounds = _design(
        raw, tensor_pairs=tensor_pairs, nonlinear=nonlinear)
    if balance_runs:
        run_counts = np.bincount(run_ids_array)
        active_runs = run_counts > 0
        run_weights = len(raw) / (np.count_nonzero(active_runs) * run_counts[run_ids_array])
    else:
        run_weights = np.ones(len(raw), dtype=float)
    x_mean = np.average(design, axis=0, weights=run_weights)
    x_scale = np.sqrt(np.average(
        (design - x_mean) ** 2, axis=0, weights=run_weights))
    x_scale[x_scale < 1e-9] = 1.0
    x_scale[0] = 1.0
    x_mean[0] = 0.0
    y_mean = np.average(target, axis=0, weights=run_weights)
    y_scale = np.sqrt(np.average(
        (target - y_mean) ** 2, axis=0, weights=run_weights))
    y_scale[y_scale < 1e-9] = 1.0
    x_scaled = (design - x_mean) / x_scale
    y_scaled = (target - y_mean) / y_scale
    sqrt_weights = np.sqrt(run_weights)
    weighted_x = x_scaled * sqrt_weights[:, None]
    weighted_y = y_scaled * sqrt_weights[:, None]
    penalty = ridge * float(np.sum(run_weights)) * np.eye(x_scaled.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_x.T @ weighted_x + penalty, weighted_x.T @ weighted_y)
    name = ("nonlinear_spline_rear_wheels" if nonlinear and use_rear_wheel_speeds
            else "nonlinear_spline" if nonlinear else "affine")
    if body_acceleration_coordinates:
        name += "_body_acceleration"
    if balance_runs:
        name += "_run_balanced"
    if include_actuator_history:
        name += "_actuator_history"
    return Model(name,
                 bounds, knots, x_mean, x_scale, y_mean, y_scale,
                 coefficients, tensor_pairs, nonlinear, use_rear_wheel_speeds,
                 body_acceleration_coordinates, include_actuator_history), \
        len(raw), low_forward_speed_wheels[0]


def _derivative(model: Model, state: np.ndarray,
                actuators: np.ndarray,
                rear_wheel_surface_mps: np.ndarray | None = None,
                actuator_history: np.ndarray | None = None
                ) -> tuple[np.ndarray, bool]:
    raw = _raw_features(
        state, actuators,
        rear_wheel_surface_mps if model.use_rear_wheel_speeds else None,
        actuator_history=actuator_history
        if model.include_actuator_history else None,
    ).reshape(1, -1)
    design, ood, _, _ = _design(raw, model=model,
                                nonlinear=model.nonlinear)
    if bool(ood[0]):
        return np.zeros(3, dtype=float), True
    normalized = (design - model.x_mean) / model.x_scale
    result = (normalized @ model.coefficients)[0] * model.y_scale + model.y_mean
    if model.body_acceleration_coordinates:
        u, v, yaw_rate = state
        result = np.asarray((result[0] + v * yaw_rate,
                             result[1] - u * yaw_rate,
                             result[2]), dtype=float)
    return result, False


def _rmse(values: list[float]) -> float:
    return math.sqrt(statistics.mean(value * value for value in values)) if values else math.inf


def _prediction_actuators(sequence: tuple[MotionSample, ...], times: list[float],
                          index: int, start: int,
                          command_alignment_delay_s: float | None,
                          capture: Capture
                          ) -> np.ndarray | None:
    """Use measured initial actuation, then optionally replay shifted commands.

    The delay is a command-to-feedback timeline hypothesis for offline
    validation, not a physical steering/throttle actuator model.
    """
    current = sequence[index]
    if command_alignment_delay_s is None or index == start:
        return current.actuators[:2]
    command_time_ns = (sequence[index].receipt_ns
                       - int(round(command_alignment_delay_s * 1e9)))
    values = (
        _command_at_receipt(capture, "steering", command_time_ns),
        _command_at_receipt(capture, "throttle", command_time_ns),
    )
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=float)


def _advance_actuator(value: float, command: float, dt_s: float,
                      model: dict) -> float:
    tau_s = float(model["time_constant_s"])
    if tau_s <= 0.0:
        change = command - value
    else:
        change = -math.expm1(-dt_s / tau_s) * (command - value)
    rate_limit = model.get("rate_limit_per_s")
    if rate_limit is not None:
        maximum_change = float(rate_limit) * dt_s
        change = min(maximum_change, max(-maximum_change, change))
    lower, upper = (float(item) for item in model["output_bounds"])
    return min(upper, max(lower, value + change))


def score(model: Model, capture: Capture,
          command_alignment_delay_s: float | None = None,
          integration_substeps: int = 1,
          lateral_model: Model | None = None,
          actuator_model: dict | None = None) -> dict:
    if integration_substeps < 1:
        raise ValueError("integration_substeps must be a positive integer")
    if actuator_model is not None and command_alignment_delay_s is not None:
        raise ValueError("use either an actuator model or a direct command delay")
    predicted_actuator_channels = (
        set(actuator_model.get("predict_channels", ("steering", "throttle")))
        if actuator_model is not None else set())
    if not predicted_actuator_channels.issubset({"steering", "throttle"}):
        raise ValueError("actuator model predict_channels must contain only "
                         "steering and/or throttle")
    one_step = [[], [], []]
    recursive = {horizon: [[], [], []] for horizon in HORIZONS_S}
    recursive_predictions = {horizon: [] for horizon in HORIZONS_S}
    total = 0
    out_of_domain = 0
    valid_rollouts = {horizon: 0 for horizon in HORIZONS_S}
    rollout_counts = {horizon: 0 for horizon in HORIZONS_S}
    missing_command_rollouts = {horizon: 0 for horizon in HORIZONS_S}
    recursive_ood_features = {name: 0 for name in _model_feature_names(model)}
    recursive_ood_context: list[tuple[float, float, float, float, float]] = []
    recursive_ood_phases: list[str] = []

    for sequence_index, sequence in enumerate(capture.sequences):
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
            feature_state = _feature_state(current)
            if not np.isfinite(feature_state).all():
                continue
            derivative, ood = _derivative(model, feature_state,
                                          current.actuators,
                                          current.rear_wheel_surface_mps,
                                          current.actuator_history)
            total += 1
            out_of_domain += int(ood)
            if ood:
                continue
            if lateral_model is not None:
                lateral, lateral_ood = _derivative(
                    lateral_model, feature_state, current.actuators)
                if lateral_ood:
                    continue
                derivative[1] = lateral[1]
            predicted = feature_state + dt * derivative
            for axis in range(3):
                one_step[axis].append(predicted[axis] - following.state[axis])

        times = [row.time_s for row in sequence]
        if capture.has_phase_markers:
            start = next((index for index, value in enumerate(times)
                          if value >= 0.0), None)
            if start is None:
                continue
            rollout_starts = [start]
        else:
            # An unphased whole-run bag can contain several continuous
            # segments after sensor gaps. Score throughout every segment,
            # rather than only near the first bag timestamp; otherwise most
            # of a practice run is never included in recursive validation.
            rollout_starts = []
            next_start_time = times[0]
            while next_start_time <= times[-1] - HORIZONS_S[-1]:
                start = _nearest_time_index(times, next_start_time)
                if not rollout_starts or start != rollout_starts[-1]:
                    rollout_starts.append(start)
                next_start_time += UNPHASED_ROLLOUT_SPACING_S

        for start in rollout_starts:
            initial_state = _feature_state(sequence[start])
            if not np.isfinite(initial_state).all():
                continue
            rollout_start_time = times[start]
            for horizon in HORIZONS_S:
                horizon_time = rollout_start_time + horizon
                target_index = _nearest_time_index(times, horizon_time, start)
                if abs(times[target_index] - horizon_time) > 0.04:
                    continue
                rollout_counts[horizon] += 1
                state = initial_state.copy()
                actuator_state = sequence[start].actuators[:2].copy()
                predicted_actuator_rates = (
                    sequence[start].actuator_history[:2].copy()
                    if sequence[start].actuator_history is not None
                    else np.zeros(2, dtype=float))
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
                    if actuator_model is None:
                        actuators = _prediction_actuators(
                            sequence, times, index, start,
                            command_alignment_delay_s, capture)
                        if actuators is None:
                            failed_support = True
                            missing_command_rollouts[horizon] += 1
                            break
                    for substep in range(integration_substeps):
                        step_dt = dt / integration_substeps
                        if actuator_model is not None:
                            command_time_ns = (current.receipt_ns
                                               + int(round(substep * step_dt * 1e9)))
                            channel_models = actuator_model["channel_models"]
                            actuators = actuator_state.copy()
                            commands = {}
                            for channel_index, channel in (
                                    (0, "steering"), (1, "throttle")):
                                if channel not in predicted_actuator_channels:
                                    actuators[channel_index] = (
                                        current.actuators[channel_index])
                                    continue
                                channel_model = channel_models[channel]
                                command = _command_at_receipt(
                                    capture, channel,
                                    command_time_ns - int(round(
                                        float(channel_model["delay_s"]) * 1e9)))
                                if command is None:
                                    failed_support = True
                                    missing_command_rollouts[horizon] += 1
                                    break
                                commands[channel] = command
                            if failed_support:
                                break
                        actuator_context = current.actuator_history
                        if (model.include_actuator_history
                                and actuator_model is not None):
                            actuator_context = current.actuator_history.copy()
                            for channel_index, channel in enumerate(
                                    ("steering", "throttle")):
                                if channel in predicted_actuator_channels:
                                    actuator_context[channel_index] = (
                                        predicted_actuator_rates[channel_index])
                        derivative, ood = _derivative(
                            model, state, actuators,
                            current.rear_wheel_surface_mps,
                            actuator_context)
                        if ood:
                            failed_support = True
                            out_of_domain += 1
                            if horizon == HORIZONS_S[-1]:
                                raw = _raw_features(
                                    state, actuators,
                                    current.rear_wheel_surface_mps
                                    if model.use_rear_wheel_speeds else None,
                                    actuator_history=actuator_context
                                    if model.include_actuator_history else None)
                                for name, value, bounds in zip(
                                        _model_feature_names(model), raw,
                                        model.bounds):
                                    if value < bounds[0] or value > bounds[1]:
                                        recursive_ood_features[name] += 1
                                recursive_ood_context.append((
                                    state[0], state[1], state[2],
                                    actuators[0], actuators[1]))
                                if sequence_index < len(capture.sequence_labels):
                                    recursive_ood_phases.append(
                                        capture.sequence_labels[sequence_index])
                            break
                        if lateral_model is not None:
                            lateral, lateral_ood = _derivative(
                                lateral_model, state, actuators)
                            if lateral_ood:
                                failed_support = True
                                break
                            derivative[1] = lateral[1]
                        state = state + step_dt * derivative
                        if actuator_model is not None:
                            for channel_index, channel in enumerate(
                                    ("steering", "throttle")):
                                if channel in predicted_actuator_channels:
                                    old_value = actuator_state[channel_index]
                                    actuator_state[channel_index] = (
                                        _advance_actuator(
                                            old_value, commands[channel],
                                            step_dt, channel_models[channel]))
                                    predicted_actuator_rates[channel_index] = (
                                        (actuator_state[channel_index] - old_value)
                                        / step_dt)
                    if failed_support:
                        break
                if failed_support:
                    continue
                valid_rollouts[horizon] += 1
                actual = sequence[target_index].state
                label = (capture.sequence_labels[sequence_index]
                         if sequence_index < len(capture.sequence_labels) else "")
                recursive_predictions[horizon].append(
                    (label, times[start], initial_state.copy(),
                     sequence[start].actuators[:2].copy(),
                     state.copy(), actual.copy()))
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
        "recursive_predictions": recursive_predictions,
        "support_fraction": 1.0 - out_of_domain / max(total, 1),
        "valid_rollouts": valid_rollouts,
        "rollout_counts": rollout_counts,
        "missing_command_rollouts": missing_command_rollouts,
        "recursive_ood_features": {
            name: count for name, count in recursive_ood_features.items() if count
        },
        "recursive_ood_context": {
            "count": len(recursive_ood_context),
            "median_u_mps": statistics.median(row[0] for row in recursive_ood_context)
            if recursive_ood_context else math.nan,
            "median_abs_v_mps": statistics.median(abs(row[1]) for row in recursive_ood_context)
            if recursive_ood_context else math.nan,
            "median_abs_yaw_rate_rps": statistics.median(abs(row[2]) for row in recursive_ood_context)
            if recursive_ood_context else math.nan,
            "median_abs_steering_rad": statistics.median(abs(row[3]) for row in recursive_ood_context)
            if recursive_ood_context else math.nan,
        },
        "recursive_ood_phases": recursive_ood_phases,
    }


def _rollout_map(result: dict, horizon: float) -> dict:
    return {
        (label, round(start_time, 6)): (initial, actuators, predicted, actual)
        for label, start_time, initial, actuators, predicted, actual
        in result["recursive_predictions"][horizon]
    }


def _matched_rollout_residuals(results: list[dict], horizon: float
                               ) -> tuple[list[tuple], list[np.ndarray]]:
    maps = [_rollout_map(result, horizon) for result in results]
    common_keys = sorted(set.intersection(*(set(records) for records in maps)))
    residuals = [np.asarray([records[key][2] - records[key][3]
                             for key in common_keys], dtype=float).reshape(-1, 3)
                 for records in maps]
    return common_keys, residuals


def _pairwise_rollout_line(horizon: float, before_name: str, before: dict,
                           after_name: str, after: dict) -> str:
    keys, residuals = _matched_rollout_residuals([before, after], horizon)
    if not len(keys):
        return f"{horizon * 1000:3.0f} ms {before_name}->{after_name}: no matched starts"
    rmse = [np.sqrt(np.mean(values ** 2, axis=0)) for values in residuals]
    line = (f"{horizon * 1000:3.0f} ms {before_name}->{after_name} "
            f"n={len(keys)}; "
            + ", ".join(
                f"{axis}={left:.4f}->{right:.4f}"
                for axis, left, right in zip(
                    ("u", "v", "r"), rmse[0], rmse[1])))
    if keys and all(str(key[0]).startswith("lap_count=") for key in keys):
        observed = {str(key[0]) for key in keys}
        numeric_clusters = sorted(
            (label for label in observed
             if label.removeprefix("lap_count=").isdigit()),
            key=lambda label: int(label.removeprefix("lap_count=")))
        # The initial counter value includes run-up/first-lap context and the
        # final counter value is a partial lap. Use only interior counter
        # values as whole-lap clusters for uncertainty estimates.
        clusters = numeric_clusters[1:-1]
        cluster_deltas = []
        for cluster in clusters:
            selected = np.asarray([key[0] == cluster for key in keys], dtype=bool)
            if not np.any(selected):
                continue
            before_cluster = np.sqrt(np.mean(residuals[0][selected] ** 2, axis=0))
            after_cluster = np.sqrt(np.mean(residuals[1][selected] ** 2, axis=0))
            cluster_deltas.append(after_cluster - before_cluster)
        if len(cluster_deltas) >= 2:
            deltas = np.asarray(cluster_deltas, dtype=float)
            rng = np.random.default_rng(20260928)
            indices = rng.integers(0, len(deltas), size=(10000, len(deltas)))
            boot_mean = np.mean(deltas[indices], axis=1)
            ci = np.quantile(boot_mean, (0.025, 0.975), axis=0)
            wins = np.count_nonzero(deltas < 0.0, axis=0)
            line += (f"; complete-lap clusters={len(deltas)}, "
                     "mean ΔRMSE(candidate−base) "
                     + ", ".join(
                         f"{axis}={mean:+.4f} [{low:+.4f},{high:+.4f}], "
                         f"wins={int(win)}/{len(deltas)}"
                         for axis, mean, low, high, win in zip(
                             ("u", "v", "r"), np.mean(deltas, axis=0),
                             ci[0], ci[1], wins)))
    return line


def _print_high_steering_rollouts(role: str, affine_result: dict,
                                  spline_result: dict,
                                  hybrid_result: dict | None = None,
                                  actuator_hybrid_result: dict | None = None) -> None:
    horizon = HORIZONS_S[-1]
    comparisons = [("affine", affine_result, "nonlinear", spline_result)]
    if hybrid_result is not None:
        comparisons.append(("affine", affine_result, "hybrid", hybrid_result))
    if actuator_hybrid_result is not None and hybrid_result is not None:
        comparisons.append(("hybrid-feedback", hybrid_result,
                            "hybrid-command-actuator", actuator_hybrid_result))
    print(f"{role}: 750 ms error by initial speed and steering "
          "(high-angle whole-run check; pairwise matched starts):")
    speed_bands = ((0.0, 3.0), (3.0, 6.0), (6.0, 10.0))
    steering_bands = ((0.30, 0.40), (0.40, STEERING_LIMIT_RAD + 1e-6))
    for speed_low, speed_high in speed_bands:
        for steer_low, steer_high in steering_bands:
            pair_reports = []
            for before_name, before, after_name, after in comparisons:
                common_keys, residuals = _matched_rollout_residuals(
                    [before, after], horizon)
                before_map = _rollout_map(before, horizon)
                selected = []
                for index, key in enumerate(common_keys):
                    initial, actuators, _, _ = before_map[key]
                    speed = math.hypot(initial[0], initial[1])
                    magnitude = abs(actuators[0])
                    if (speed_low <= speed < speed_high
                            and steer_low <= magnitude < steer_high):
                        selected.append(index)
                if len(selected) < 8:
                    continue
                errors = [np.sqrt(np.mean(values[selected] ** 2, axis=0))
                          for values in residuals]
                pair_reports.append(
                    f"{before_name}->{after_name} n={len(selected)} "
                    + ", ".join(
                        f"{axis}={left:.3f}->{right:.3f}"
                        for axis, left, right in zip(
                            ("u", "v", "r"), errors[0], errors[1])))
            if pair_reports:
                print(f"  speed {speed_low:.0f}-{speed_high:.0f}, "
                      f"|steer| {steer_low:.2f}-{steer_high:.2f} rad; "
                      + " | ".join(pair_reports))


def _print_capture(capture: Capture, role: str) -> None:
    speeds = capture.domain_samples[:, 0]
    mode = ("phase-labeled" if capture.has_phase_markers else
            f"unphased whole run; final laps={capture.final_lap_count}")
    print(f"{role}: {capture.path}; {mode}; valid/invalid/unscored phases="
          f"{capture.valid_phase_count}/{capture.invalid_phase_count}/"
          f"{capture.unscored_phase_count}; speed={np.percentile(speeds, 1):.2f}.."
          f"{np.percentile(speeds, 99):.2f} m/s (p01..p99); "
          f"steering={capture.domain_samples[:, 1].min():+.4f}.."
          f"{capture.domain_samples[:, 1].max():+.4f} rad; "
          f"throttle feedback={capture.domain_samples[:, 2].min():.3f}.."
          f"{capture.domain_samples[:, 2].max():.3f}")
    for topic, (rate, p95, gap) in capture.stream_stats.items():
        print(f"  {topic}: {rate:.3f} Hz, gap p95/max={p95:.2f}/{gap:.2f} ms")
    if capture.has_phase_markers and capture.phase_stream_stats:
        print("  active-phase cadence (quality gate; excludes unassigned warm-up):")
        for topic, (rate, p95, gap) in capture.phase_stream_stats.items():
            print(f"    {topic}: {rate:.3f} Hz, gap p95/max="
                  f"{p95:.2f}/{gap:.2f} ms")


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
             ridge: float, output_model: Path | None,
             command_alignment_delays_ms: list[float] | None = None,
             integration_substeps: list[int] | None = None,
             source_state_paths: list[Path] | None = None,
             holdout_state_path: Path | None = None,
             transfer_paths: list[Path] | None = None,
             transfer_state_paths: list[Path] | None = None,
             balance_runs: bool = False,
             include_rear_wheel_ablation: bool = False,
             body_acceleration_coordinates: bool = False,
             state_feature_topic: str = "/replayed_odom",
             actuator_model: dict | None = None,
             include_actuator_history: bool = False) -> bool:
    if not source_paths:
        raise ValueError("at least one --source-bag is required")
    if ridge <= 0.0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and positive")
    if command_alignment_delays_ms and any(
            not math.isfinite(delay) or delay < 0.0
            for delay in command_alignment_delays_ms):
        raise ValueError("command alignment delays must be finite and nonnegative")
    if actuator_model is not None and command_alignment_delays_ms:
        raise ValueError("do not combine a fitted actuator model with direct command delay")
    if integration_substeps and any(steps < 1 for steps in integration_substeps):
        raise ValueError("integration substeps must be positive integers")
    source_state_paths = source_state_paths or []
    transfer_paths = transfer_paths or []
    transfer_state_paths = transfer_state_paths or []
    use_observer_states = bool(source_state_paths or holdout_state_path
                               or transfer_state_paths)
    if use_observer_states:
        if (len(source_state_paths) != len(source_paths)
                or holdout_state_path is None
                or len(transfer_state_paths) != len(transfer_paths)):
            raise ValueError(
                "observer-state mode requires matching state bags for every source, "
                "the primary holdout, and every transfer holdout")
    sources = [load_capture(path) for path in source_paths]
    evaluations = [("whole-run holdout", load_capture(holdout_path))]
    evaluations.extend(("transfer holdout", load_capture(path))
                       for path in transfer_paths)
    state_feature_source = (
        "simulator truth /autodrive/roboracer_1/odom (development oracle)"
        if not use_observer_states else
        f"production body-twist state topic {state_feature_topic}")
    if use_observer_states:
        aligned_sources = []
        for capture, state_path in zip(sources, source_state_paths):
            capture, coverage = _attach_feature_states(
                capture, state_path, state_feature_topic)
            print(f"observer-state coverage {capture.path.name}: {coverage:.2%}")
            aligned_sources.append(capture)
        sources = aligned_sources
        evaluation_state_paths = [holdout_state_path, *transfer_state_paths]
        aligned_evaluations = []
        for (role, capture), state_path in zip(evaluations, evaluation_state_paths):
            capture, coverage = _attach_feature_states(
                capture, state_path, state_feature_topic)
            print(f"observer-state coverage {capture.path.name}: {coverage:.2%}")
            aligned_evaluations.append((role, capture))
        evaluations = aligned_evaluations
    for index, capture in enumerate(sources, 1):
        _validate_capture(capture, f"source {index}")
        _print_capture(capture, f"source {index}")
    seen_paths = {capture.path.resolve() for capture in sources}
    for role, capture in evaluations:
        _validate_capture(capture, role)
        _print_capture(capture, role)
        if capture.path.resolve() in seen_paths:
            raise ValueError(f"{role} must be independent of all training and holdout bags")
        seen_paths.add(capture.path.resolve())

    print(f"state features: {state_feature_source}")
    print("transition targets and held-out references: simulator-truth "
          "/autodrive/roboracer_1/odom")

    affine, affine_count, _ = fit(
        sources, ridge, nonlinear=False, balance_runs=balance_runs)
    spline, spline_count, low_forward_speed_wheels = fit(
        sources, ridge, nonlinear=True, balance_runs=balance_runs,
        body_acceleration_coordinates=body_acceleration_coordinates,
        include_actuator_history=include_actuator_history)
    wheel_spline = None
    wheel_low_forward_speed_wheels = 0
    if include_rear_wheel_ablation:
        wheel_spline, wheel_spline_count, wheel_low_forward_speed_wheels = fit(
            sources, ridge, nonlinear=True, use_rear_wheel_speeds=True,
            balance_runs=balance_runs)
    print(f"training transitions: affine={affine_count}, nonlinear={spline_count}; "
          f"cubic tensor splines, ridge={ridge:g}")
    print(f"run-balanced fitting: {'yes' if balance_runs else 'no'}")
    print(f"causal actuator rates/current commands: "
          f"{'included' if include_actuator_history else 'not included'}")
    print("nonlinear target coordinates: "
          + ("body-frame accelerations with exact rotating-frame terms restored"
             if body_acceleration_coordinates else
             "direct body-frame velocity derivatives"))
    if low_forward_speed_wheels:
        print(f"wheel-local proxy: {low_forward_speed_wheels} wheel-samples had "
              "|v_longitudinal| < 0.5 m/s; retained with finite atan2 slip "
              "angles rather than discarding singular tan(alpha) rows")
    if wheel_spline is not None and wheel_low_forward_speed_wheels:
        print(f"rear-wheel model: {wheel_low_forward_speed_wheels} low-forward-"
              "speed wheel samples retained")
    for role, capture in evaluations:
        _print_nonlinear_observations(capture)
        affine_result = score(affine, capture)
        spline_result = score(spline, capture)
        hybrid_result = score(spline, capture, lateral_model=affine)
        actuator_hybrid_result = (
            score(spline, capture, lateral_model=affine,
                  actuator_model=actuator_model)
            if actuator_model is not None else None)
        wheel_result = score(wheel_spline, capture) if wheel_spline else None
        model_results = [("affine", affine_result),
                         ("nonlinear", spline_result),
                         ("nonlinear-u/r + affine-v", hybrid_result)]
        if wheel_result:
            model_results.append(("rear-wheel nonlinear", wheel_result))
        print(f"{role}: one-step state RMSE:")
        for axis in ("u_mps", "v_rear_mps", "yaw_rate_rps"):
            print(f"  {axis}: " + " | ".join(
                f"{name}={result['one_step_rmse'][axis]:.5f}"
                for name, result in model_results))
        print(f"  nonlinear one-step feature support: "
              f"{spline_result['support_fraction']:.1%}")
        print(f"{role}: recursive state RMSE:")
        rollout_pairs = [("affine", affine_result,
                          "nonlinear", spline_result),
                         ("affine", affine_result,
                          "hybrid", hybrid_result),
                         ("nonlinear", spline_result,
                          "hybrid", hybrid_result)]
        for horizon in HORIZONS_S:
            counts = ", ".join(
                f"{name}={result['valid_rollouts'][horizon]}/"
                f"{result['rollout_counts'][horizon]}"
                for name, result in model_results)
            print(f"  {horizon * 1000:3.0f} ms support[{counts}]")
            for before_name, before, after_name, after in rollout_pairs:
                print("    " + _pairwise_rollout_line(
                    horizon, before_name, before, after_name, after))
        if actuator_hybrid_result is not None:
            print(f"{role}: recursive hybrid rollout with fitted command-to-"
                  "feedback actuator states (initialized from current feedback):")
            for horizon in HORIZONS_S:
                print("  " + _pairwise_rollout_line(
                    horizon, "hybrid-measured-feedback", hybrid_result,
                    "hybrid-predicted-actuators", actuator_hybrid_result))
            missing_commands = actuator_hybrid_result[
                "missing_command_rollouts"]
            if any(missing_commands.values()):
                print("  excluded rollouts without a fresh causal command: "
                      + ", ".join(
                          f"{horizon * 1000:.0f}ms="
                          f"{missing_commands[horizon]}"
                          for horizon in HORIZONS_S))
        _print_high_steering_rollouts(
            role, affine_result, spline_result, hybrid_result,
            actuator_hybrid_result)
        unsupported = spline_result["recursive_ood_features"]
        if unsupported:
            context = spline_result["recursive_ood_context"]
            print("  unsupported feature counts at 750 ms: "
                  + ", ".join(f"{name}={count}"
                               for name, count in unsupported.items()))
            print("  unsupported-state medians: "
                  f"u={context['median_u_mps']:.3f} m/s, "
                  f"|v|={context['median_abs_v_mps']:.3f} m/s, "
                  f"|r|={context['median_abs_yaw_rate_rps']:.3f} rad/s, "
                  f"|steer|={context['median_abs_steering_rad']:.3f} rad")
    requested_delays: list[float | None] = list(command_alignment_delays_ms or ())
    if integration_substeps and not requested_delays:
        requested_delays = [None]
    for delay_ms in requested_delays:
        delay_s = delay_ms / 1000.0 if delay_ms is not None else None
        delay_label = (f"commands shifted by {delay_ms:g} ms, measured initial actuation"
                       if delay_ms is not None else "measured actuator feedback")
        for role, capture in evaluations:
            for steps in integration_substeps or [1]:
                command_affine = score(affine, capture, delay_s, steps)
                command_spline = score(spline, capture, delay_s, steps)
                command_hybrid = score(
                    spline, capture, delay_s, steps, lateral_model=affine)
                command_results = [("affine", command_affine),
                                   ("nonlinear", command_spline),
                                   ("nonlinear-u/r + affine-v", command_hybrid)]
                command_pairs = [
                    ("affine", command_affine,
                     "nonlinear", command_spline),
                    ("affine", command_affine,
                     "hybrid", command_hybrid),
                    ("nonlinear", command_spline,
                     "hybrid", command_hybrid),
                ]
                print(f"{role}: recursive state RMSE with {delay_label}; "
                      f"{steps} integration substep(s) per sample "
                      "pairwise matched starts:")
                for horizon in HORIZONS_S:
                    for before_name, before, after_name, after in command_pairs:
                        print("  " + _pairwise_rollout_line(
                            horizon, before_name, before, after_name, after))
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
            body_acceleration_coordinates=np.asarray(
                spline.body_acceleration_coordinates),
            include_actuator_history=np.asarray(
                spline.include_actuator_history),
            model_name=np.asarray(spline.name),
            run_balanced=np.asarray(balance_runs),
            training_captures=np.asarray([str(path) for path in source_paths]),
            state_feature_source=np.asarray(state_feature_source),
            target_source=np.asarray(
                "simulator-truth /autodrive/roboracer_1/odom; development labels"),
            feature_names=np.asarray(_model_feature_names(spline)),
        )
        print(f"saved offline nonlinear candidate: {output_model}")
    print("Scope: state-only spline is fitted with measured actuator feedback; "
          + ("its target is body-frame acceleration with exact planar "
             "rotating-frame kinematics restored. "
             if body_acceleration_coordinates else "")
          + "rear-wheel comparison, when requested, additionally uses "
          "100 ms encoder-derived wheel-surface/slip-velocity inputs and is "
          "conditional, not a standalone simulator. Slip angles are kinematic "
          "proxies, not tire forces. "
          + ("The supplied actuator candidate predicts command-to-feedback "
             "state only; rear wheel-speed dynamics remain unmodelled. "
             if actuator_model is not None else
             "Actuator and wheel-speed dynamics remain unmodelled. ")
          + "No competition runtime changed.")
    return True


def _load_actuator_model(path: Path) -> dict:
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read actuator model {path}: {exc}") from exc
    if model.get("schema_version") != 1:
        raise ValueError("unsupported actuator-model schema")
    channels = model.get("channel_models")
    if not isinstance(channels, dict):
        raise ValueError("actuator model is missing channel_models")
    predicted_channels = model.get("predict_channels", ("steering", "throttle"))
    if (not isinstance(predicted_channels, (list, tuple))
            or not predicted_channels
            or not set(predicted_channels).issubset({"steering", "throttle"})):
        raise ValueError("predict_channels must contain steering and/or throttle")
    for channel in ("steering", "throttle"):
        config = channels.get(channel)
        if not isinstance(config, dict):
            raise ValueError(f"actuator model is missing {channel}")
        delay = float(config["delay_s"])
        tau = float(config["time_constant_s"])
        lower, upper = (float(item) for item in config["output_bounds"])
        rate = config.get("rate_limit_per_s")
        values = (delay, tau, lower, upper)
        if (not all(math.isfinite(value) for value in values)
                or delay < 0.0 or tau < 0.0 or lower >= upper
                or (rate is not None and
                    (not math.isfinite(float(rate)) or float(rate) <= 0.0))):
            raise ValueError(f"invalid actuator model parameters for {channel}")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bag", action="append", type=Path, default=[],
                        help="clean training bag; may be repeated")
    parser.add_argument("--holdout-bag", type=Path, required=True,
                        help="independent clean run withheld in full")
    parser.add_argument("--transfer-bag", type=Path, action="append", default=[],
                        help="additional independent evaluation run; may be repeated")
    parser.add_argument("--ridge", type=float, default=0.01,
                        help="dimensionless ridge factor scaled by sample count")
    parser.add_argument("--output-model", type=Path,
                        help="optional path for the fitted offline .npz candidate")
    parser.add_argument("--command-alignment-delay-ms", type=float, action="append",
                        help="also score recursive rollouts using past command values "
                             "shifted by this many ms (repeatable; not an actuator model)")
    parser.add_argument("--actuator-model-json", type=Path,
                        help="simulate command-to-feedback actuator states from a "
                             "separately whole-run-validated model")
    parser.add_argument("--integration-substeps", type=int, action="append",
                        help="recursive rollout substeps per 40 Hz interval (repeatable)")
    parser.add_argument("--source-state-bag", type=Path, action="append", default=[],
                        help="matching state-feature bag for each source bag; may repeat")
    parser.add_argument("--holdout-state-bag", type=Path,
                        help="matching state-feature bag for holdout bag")
    parser.add_argument("--transfer-state-bag", type=Path, action="append", default=[],
                        help="matching state-feature bag per transfer bag; may repeat")
    parser.add_argument("--state-feature-topic", default="/replayed_odom",
                        help="body-twist input topic within all state bags; /odom uses "
                             "the recorded production observer instead of a replay")
    parser.add_argument("--balance-runs", action="store_true",
                        help="give each independent source bag equal total fit weight")
    parser.add_argument("--include-rear-wheel-ablation", action="store_true",
                        help="also fit the encoder-derived rear-wheel feature ablation")
    parser.add_argument("--body-acceleration-coordinates", action="store_true",
                        help="fit accelerations and explicitly restore exact rotating-frame terms")
    parser.add_argument("--include-actuator-history", action="store_true",
                        help="include measured steering/throttle rates and current commands")
    args = parser.parse_args()
    try:
        actuator_model = (_load_actuator_model(args.actuator_model_json)
                          if args.actuator_model_json is not None else None)
        evaluate(args.source_bag, args.holdout_bag,
                 args.ridge, args.output_model,
                 args.command_alignment_delay_ms,
                 args.integration_substeps,
                 args.source_state_bag,
                 args.holdout_state_bag,
                 args.transfer_bag,
                 args.transfer_state_bag,
                 args.balance_runs,
                 args.include_rear_wheel_ablation,
                 args.body_acceleration_coordinates,
                 args.state_feature_topic,
                 actuator_model,
                 args.include_actuator_history)
    except (ValueError, sqlite3.Error, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"body model evaluation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

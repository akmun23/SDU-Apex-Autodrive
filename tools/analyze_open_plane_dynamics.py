#!/usr/bin/env python3
"""Analyze phase-labeled open-plane data using the published vehicle equations.

The reported tire-slip quantities are kinematic proxies. This data does not
contain per-wheel contact forces, so this tool does not claim to identify the
simulator's tire-force polynomial from net vehicle acceleration alone.
Repeated 40 Hz samples are reduced to phase-level medians before confidence
intervals are calculated.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - depends on ROS installation
    raise SystemExit(
        "ROS 2 Python modules are required (rclpy and rosidl_runtime_py): "
        f"{exc}"
    ) from exc


ODOM = "/autodrive/roboracer_1/odom"
STEERING = "/autodrive/roboracer_1/steering"
THROTTLE_COMMAND = "/autodrive/roboracer_1/throttle_command"
THROTTLE_FEEDBACK = "/autodrive/roboracer_1/throttle"
LEFT_ENCODER = "/autodrive/roboracer_1/left_encoder"
RIGHT_ENCODER = "/autodrive/roboracer_1/right_encoder"
IMU = "/autodrive/roboracer_1/imu"
IPS = "/autodrive/roboracer_1/ips"
COLLISIONS = "/autodrive/roboracer_1/collision_count"
PACKET_TIMING = "/autodrive/roboracer_1/bridge_packet_timing"
TIMING_FAULT = "/autodrive/roboracer_1/bridge_timing_fault"
PHASE = "/open_plane_experiment/phase"

# Values published in the AutoDRIVE technical guide; do not change simulator physics.
WHEELBASE_M = 0.324
TRACK_WIDTH_M = 0.236
WHEEL_RADIUS_M = 0.059
TOTAL_MASS_KG = 3.906
COM_X_M = 0.15532
GRAVITY_MPS2 = 9.80665
PHASE_SETTLE_NS = 350_000_000
ALIGNMENT_LIMIT_NS = 30_000_000
MIN_PHASE_SAMPLES = 30
CI95_T_CRITICAL = {1: 12.706, 2: 4.303}
MIN_STREAM_RATE_HZ = 38.0
MAX_STREAM_P95_GAP_MS = 35.0
MAX_STREAM_GAP_MS = 60.0
MIN_FIXED_THROTTLE_COMMANDS = 5
MAX_FIXED_THROTTLE_P95_ERROR = 0.02
PROBE_START_MAX_VY_MPS = 0.08
PROBE_START_MAX_YAW_RATE_RPS = 0.12
PROBE_START_MAX_SPEED_ERROR_MPS = 0.20
PROBE_START_MAX_STEERING_RAD = 0.02
PROBE_START_SETTLE_SEC = 0.50


@dataclass(frozen=True)
class OdomRow:
    receipt_ns: int
    source_ns: int
    vx_mps: float
    vy_mps: float
    yaw_rate_rps: float
    frame_id: str
    child_frame_id: str


@dataclass(frozen=True)
class ScalarRow:
    receipt_ns: int
    source_ns: int
    value: float


@dataclass(frozen=True)
class EncoderRow:
    receipt_ns: int
    source_ns: int
    angle_rad: float


@dataclass(frozen=True)
class ImuRow:
    receipt_ns: int
    source_ns: int
    yaw_rate_rps: float
    lateral_accel_mps2: float


@dataclass(frozen=True)
class Phase:
    index: int
    label: str
    start_ns: int
    end_ns: int
    target_speed_mps: float
    commanded_steering_rad: float
    valid: bool | None
    quality_failures: tuple[str, ...]
    initial_vx_mps: float | None
    initial_vy_mps: float | None
    initial_yaw_rate_rps: float | None
    initial_steering_rad: float | None
    initial_window_speed_mps: float | None
    initial_window_abs_vy_mps: float | None
    initial_window_abs_yaw_rate_rps: float | None
    throttle_mode: str | None
    fixed_throttle_command_norm: float | None


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _topic_map(connection: sqlite3.Connection) -> dict[str, tuple[int, str]]:
    return {
        name: (int(topic_id), msg_type)
        for topic_id, name, msg_type in connection.execute(
            "SELECT id, name, type FROM topics"
        )
    }


def _messages(connection: sqlite3.Connection, topics: dict[str, tuple[int, str]], name: str):
    topic_id, msg_type = topics[name]
    message_class = get_message(msg_type)
    cursor = connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp, id",
        (topic_id,),
    )
    for timestamp, payload in cursor:
        yield int(timestamp), deserialize_message(bytes(payload), message_class)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _rate(receipts_ns: list[int]) -> tuple[float | None, float | None, float | None]:
    if len(receipts_ns) < 2:
        return None, None, None
    gaps_ms = [
        (right - left) / 1e6
        for left, right in zip(receipts_ns, receipts_ns[1:])
        if right > left
    ]
    duration_s = (receipts_ns[-1] - receipts_ns[0]) / 1e9
    rate_hz = (len(receipts_ns) - 1) / duration_s if duration_s > 0.0 else None
    return rate_hz, _percentile(gaps_ms, 0.95), max(gaps_ms) if gaps_ms else None


def _nearest_scalar(rows: list[Any], times: list[int], target_ns: int,
                    max_offset_ns: int = ALIGNMENT_LIMIT_NS) -> Any | None:
    index = bisect.bisect_left(times, target_ns)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    if not candidates:
        return None
    row = min(candidates, key=lambda i: abs(times[i] - target_ns))
    return rows[row] if abs(times[row] - target_ns) <= max_offset_ns else None


def _ackermann_angles(steering_rad: float) -> tuple[float, float]:
    """Return Ackermann angles for x-forward/y-left wheel locations.

    Positive steering/yaw is a left turn: the left inside wheel therefore has
    the larger angle. This ordering matches the official bridge's static wheel
    transforms and is used only by development-side slip reconstruction.
    """
    tangent = math.tan(steering_rad)
    numerator = 2.0 * WHEELBASE_M * tangent
    left = math.atan2(numerator, 2.0 * WHEELBASE_M - TRACK_WIDTH_M * tangent)
    right = math.atan2(numerator, 2.0 * WHEELBASE_M + TRACK_WIDTH_M * tangent)
    return left, right


def _rear_axle_velocity(vx_com: float, vy_com: float,
                        yaw_rate: float) -> tuple[float, float]:
    """Shift AutoDRIVE's copied COM velocity to its rear-axle base frame."""
    return vx_com, vy_com - yaw_rate * COM_X_M


def _wheel_slip(vx: float, vy: float, yaw_rate: float,
                x_m: float, y_m: float, steering_rad: float) -> float | None:
    """Return tan(alpha)=v_y/|v_x| after rigid-body and wheel-frame transforms."""
    wheel_vx_body = vx - yaw_rate * y_m
    wheel_vy_body = vy + yaw_rate * x_m
    cosine = math.cos(steering_rad)
    sine = math.sin(steering_rad)
    wheel_vx = cosine * wheel_vx_body + sine * wheel_vy_body
    wheel_vy = -sine * wheel_vx_body + cosine * wheel_vy_body
    if abs(wheel_vx) < 0.5:
        return None
    return wheel_vy / abs(wheel_vx)


def _encoder_slip(encoder_rows: list[EncoderRow], times: list[int],
                  target_ns: int, wheel_vx_mps: float,
                  angle_scale: float) -> float | None:
    """Estimate rear longitudinal slip with a 0.1 s encoder-angle window."""
    if abs(wheel_vx_mps) < 0.5 or len(encoder_rows) < 3:
        return None
    current_index = bisect.bisect_right(times, target_ns) - 1
    old_target = target_ns - 100_000_000
    old_index = bisect.bisect_left(times, old_target)
    if current_index < 0 or old_index >= len(encoder_rows):
        return None
    if old_index > 0 and abs(times[old_index - 1] - old_target) < abs(times[old_index] - old_target):
        old_index -= 1
    current = encoder_rows[current_index]
    older = encoder_rows[old_index]
    dt_s = (current.source_ns - older.source_ns) / 1e9
    if dt_s < 0.06 or dt_s > 0.15:
        return None
    omega_wheel = ((current.angle_rad - older.angle_rad) / dt_s) * angle_scale
    wheel_speed = WHEEL_RADIUS_M * omega_wheel
    return (wheel_speed - wheel_vx_mps) / wheel_vx_mps


def _phase_events(connection: sqlite3.Connection,
                  topics: dict[str, tuple[int, str]]) -> tuple[list[Phase], dict[str, Any]]:
    starts: dict[int, tuple[int, dict[str, Any]]] = {}
    ends: dict[int, tuple[int, dict[str, Any]]] = {}
    experiment_end: dict[str, Any] = {}
    for receipt_ns, message in _messages(connection, topics, PHASE):
        try:
            event = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            continue
        kind = event.get("event")
        if kind == "phase_start":
            starts[int(event["phase_index"])] = (receipt_ns, event)
        elif kind == "phase_end":
            ends[int(event["phase_index"])] = (receipt_ns, event)
        elif kind == "experiment_end":
            experiment_end = event

    phases: list[Phase] = []
    for index, (start_ns, start) in sorted(starts.items()):
        end = ends.get(index)
        if end is None:
            continue
        end_ns, result = end
        valid = result.get("valid")
        def optional_finite(name: str) -> float | None:
            value = start.get(name)
            return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

        phases.append(Phase(
            index=index,
            label=str(start.get("label", "")),
            start_ns=start_ns,
            end_ns=end_ns,
            target_speed_mps=float(start.get("target_speed_mps", 0.0)),
            commanded_steering_rad=float(start.get("steering_command_rad", 0.0)),
            valid=valid if isinstance(valid, bool) else None,
            quality_failures=tuple(result.get("quality_failures", ())),
            initial_vx_mps=optional_finite("initial_vx_mps"),
            initial_vy_mps=optional_finite("initial_vy_mps"),
            initial_yaw_rate_rps=optional_finite("initial_yaw_rate_rps"),
            initial_steering_rad=optional_finite("initial_steering_rad"),
            initial_window_speed_mps=optional_finite("initial_window_speed_mps"),
            initial_window_abs_vy_mps=optional_finite("initial_window_abs_vy_mps"),
            initial_window_abs_yaw_rate_rps=optional_finite(
                "initial_window_abs_yaw_rate_rps"),
            throttle_mode=(str(start["throttle_mode"])
                           if start.get("throttle_mode") is not None else None),
            fixed_throttle_command_norm=optional_finite(
                "fixed_throttle_command_norm"),
        ))
    return phases, experiment_end


def analyze(path: Path) -> int:
    if not path.is_file():
        raise ValueError(f"bag database does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = _topic_map(connection)
        required = (ODOM, STEERING, THROTTLE_COMMAND, LEFT_ENCODER, RIGHT_ENCODER, IMU,
                    THROTTLE_FEEDBACK, IPS, COLLISIONS, PACKET_TIMING, PHASE)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("missing required topic(s): " + ", ".join(missing))

        odometry: list[OdomRow] = []
        for receipt, msg in _messages(connection, topics, ODOM):
            vx_com = float(msg.twist.twist.linear.x)
            vy_com = float(msg.twist.twist.linear.y)
            yaw_rate = float(msg.twist.twist.angular.z)
            if not all(map(math.isfinite, (vx_com, vy_com, yaw_rate))):
                continue
            vx, vy = _rear_axle_velocity(vx_com, vy_com, yaw_rate)
            odometry.append(OdomRow(
                receipt, _stamp_ns(msg.header.stamp), vx, vy, yaw_rate,
                str(msg.header.frame_id), str(msg.child_frame_id),
            ))

        steering = [
            ScalarRow(receipt, receipt, float(msg.data))
            for receipt, msg in _messages(connection, topics, STEERING)
            if math.isfinite(float(msg.data))
        ]
        steering_rows = steering
        steering_times = [row.receipt_ns for row in steering_rows]
        throttle_commands = [
            ScalarRow(receipt, receipt, float(msg.data))
            for receipt, msg in _messages(connection, topics, THROTTLE_COMMAND)
            if math.isfinite(float(msg.data))
        ]
        throttle_command_times = [row.receipt_ns for row in throttle_commands]
        throttle_feedback = [
            ScalarRow(receipt, receipt, float(msg.data))
            for receipt, msg in _messages(connection, topics, THROTTLE_FEEDBACK)
            if math.isfinite(float(msg.data))
        ]
        throttle_feedback_times = [row.receipt_ns for row in throttle_feedback]

        ips_x = [
            ScalarRow(receipt, receipt, float(msg.x))
            for receipt, msg in _messages(connection, topics, IPS)
            if math.isfinite(float(msg.x))
        ]
        ips_y = [
            ScalarRow(receipt, receipt, float(msg.y))
            for receipt, msg in _messages(connection, topics, IPS)
            if math.isfinite(float(msg.y))
        ]
        ips_times = [row.receipt_ns for row in ips_x]

        orientation_yaw: list[ScalarRow] = []
        for receipt, msg in _messages(connection, topics, ODOM):
            q = msg.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            if math.isfinite(yaw):
                orientation_yaw.append(ScalarRow(receipt, receipt, yaw))
        orientation_yaw_times = [row.receipt_ns for row in orientation_yaw]

        def load_encoder(topic: str) -> list[EncoderRow]:
            rows = []
            for receipt, msg in _messages(connection, topics, topic):
                if msg.position and math.isfinite(float(msg.position[0])):
                    rows.append(EncoderRow(
                        receipt, _stamp_ns(msg.header.stamp), float(msg.position[0])))
            return rows

        left_encoder = load_encoder(LEFT_ENCODER)
        right_encoder = load_encoder(RIGHT_ENCODER)
        left_times = [row.receipt_ns for row in left_encoder]
        right_times = [row.receipt_ns for row in right_encoder]

        imu_rows = []
        for receipt, msg in _messages(connection, topics, IMU):
            yaw_rate = float(msg.angular_velocity.z)
            accel_y = float(msg.linear_acceleration.y)
            if math.isfinite(yaw_rate) and math.isfinite(accel_y):
                imu_rows.append(ImuRow(
                    receipt, _stamp_ns(msg.header.stamp), yaw_rate, accel_y))
        imu_times = [row.receipt_ns for row in imu_rows]

        collision_values = [int(msg.data)
                            for _, msg in _messages(connection, topics, COLLISIONS)]
        fault_values = [bool(msg.data)
                        for _, msg in _messages(connection, topics, TIMING_FAULT)]
        phases, experiment_end = _phase_events(connection, topics)
        topic_receipts: dict[str, list[int]] = {}
        for name in (ODOM, STEERING, LEFT_ENCODER, RIGHT_ENCODER, IMU, PACKET_TIMING):
            topic_id = topics[name][0]
            topic_receipts[name] = [int(row[0]) for row in connection.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp, id",
                (topic_id,),
            )]
    finally:
        connection.close()

    if len(odometry) < 100 or not phases:
        raise ValueError("bag is too short or has no completed phase markers")
    odom_times = [row.receipt_ns for row in odometry]
    lateral_accel_mps2: dict[int, float] = {}
    for index in range(1, len(odometry) - 1):
        previous, current, following = odometry[index - 1:index + 2]
        dt_s = (following.source_ns - previous.source_ns) / 1e9
        if not 0.03 <= dt_s <= 0.10:
            continue
        previous_vy_com = previous.vy_mps + previous.yaw_rate_rps * COM_X_M
        following_vy_com = following.vy_mps + following.yaw_rate_rps * COM_X_M
        lateral_accel_mps2[current.receipt_ns] = (
            (following_vy_com - previous_vy_com) / dt_s
            + current.yaw_rate_rps * current.vx_mps
        )

    print(f"bag: {path}")
    print(f"vehicle constants: L={WHEELBASE_M:.3f} m, W={TRACK_WIDTH_M:.3f} m, "
          f"r={WHEEL_RADIUS_M:.3f} m, m={TOTAL_MASS_KG:.3f} kg, "
          f"x_COM={COM_X_M:.5f} m")
    print("stream rates (receipt-time):")
    stream_cadence_ok = True
    for name, values in topic_receipts.items():
        rate, p95_gap, max_gap = _rate(values)
        print(f"  {name}: n={len(values)}, rate={rate:.3f} Hz, "
              f"gap_p95={p95_gap:.2f} ms, gap_max={max_gap:.2f} ms")
        stream_cadence_ok &= (
            rate is not None and rate >= MIN_STREAM_RATE_HZ
            and p95_gap is not None and p95_gap <= MAX_STREAM_P95_GAP_MS
            and max_gap is not None and max_gap <= MAX_STREAM_GAP_MS
        )
    print(f"collision_count: min={min(collision_values)}, max={max(collision_values)}, "
          f"changes={sum(a != b for a, b in zip(collision_values, collision_values[1:]))}")
    print(f"bridge_timing_fault: true_samples={sum(fault_values)}")
    print(f"run_end: aborted={experiment_end.get('aborted')}, "
          f"reason={experiment_end.get('reason')!r}, "
          f"quality_failures={len(experiment_end.get('quality_failures', []))}")
    print(f"phase coverage: completed={len(phases)}, "
          f"valid={sum(phase.valid is True for phase in phases)}, "
          f"invalid={sum(phase.valid is False for phase in phases)}")

    fixed_throttle_phases = [
        phase for phase in phases
        if phase.throttle_mode == "fixed"
        and phase.fixed_throttle_command_norm is not None
    ]
    fixed_throttle_failures = []
    for phase in fixed_throttle_phases:
        first = bisect.bisect_left(throttle_command_times, phase.start_ns)
        last = bisect.bisect_left(throttle_command_times, phase.end_ns)
        values = throttle_commands[first:last]
        errors = [abs(row.value - phase.fixed_throttle_command_norm)
                  for row in values]
        p95_error = _percentile(errors, 0.95)
        if (len(values) < MIN_FIXED_THROTTLE_COMMANDS
                or p95_error is None
                or p95_error > MAX_FIXED_THROTTLE_P95_ERROR):
            fixed_throttle_failures.append(phase.label)
        print(f"  fixed throttle {phase.label}: requested="
              f"{phase.fixed_throttle_command_norm:.3f}, commands={len(values)}, "
              f"p95_error={p95_error if p95_error is not None else math.nan:.4f}")
    fixed_throttle_ok = not fixed_throttle_failures
    if fixed_throttle_phases:
        print(f"fixed-throttle fidelity: valid="
              f"{len(fixed_throttle_phases) - len(fixed_throttle_failures)}/"
              f"{len(fixed_throttle_phases)}")
    else:
        print("fixed-throttle fidelity: not encoded in this bag's phase metadata")

    if "failed to recover matched probe state" in str(experiment_end.get("reason", "")):
        recovery_start = phases[-1].end_ns if phases else odom_times[0]
        recovery = [row for row in odometry if row.receipt_ns >= recovery_start]
        if recovery:
            print("failed-reset telemetry by 0.5 s block "
                  "(speed / vy / yaw-rate / steering):")
            matched_run: list[OdomRow] = []
            matched_runs: list[list[OdomRow]] = []
            speed_errors: list[float] = []
            reset_failures = {"speed": 0, "vy": 0, "yaw": 0, "steering": 0}
            encoder_speed_pairs: list[tuple[float, float]] = []
            for offset in range(0, len(recovery), 20):
                block = recovery[offset:offset + 20]
                if not block:
                    continue
                start_s = (block[0].receipt_ns - recovery_start) / 1e9
                end_s = (block[-1].receipt_ns - recovery_start) / 1e9
                steer_values = [
                    steer.value for row in block
                    if (steer := _nearest_scalar(
                        steering_rows, steering_times, row.receipt_ns)) is not None
                ]
                throttle_values = [
                    throttle.value for row in block
                    if (throttle := _nearest_scalar(
                        throttle_commands, throttle_command_times, row.receipt_ns)) is not None
                ]
                for row in block:
                    steer = _nearest_scalar(steering_rows, steering_times, row.receipt_ns)
                    speed_error = abs(math.hypot(row.vx_mps, row.vy_mps) - 2.2)
                    speed_errors.append(speed_error)
                    reset_failures["speed"] += speed_error > PROBE_START_MAX_SPEED_ERROR_MPS
                    reset_failures["vy"] += abs(row.vy_mps) > PROBE_START_MAX_VY_MPS
                    reset_failures["yaw"] += abs(row.yaw_rate_rps) > PROBE_START_MAX_YAW_RATE_RPS
                    reset_failures["steering"] += (
                        steer is None or abs(steer.value) > PROBE_START_MAX_STEERING_RAD
                    )
                    matched = (
                        speed_error <= PROBE_START_MAX_SPEED_ERROR_MPS
                        and abs(row.vy_mps) <= PROBE_START_MAX_VY_MPS
                        and abs(row.yaw_rate_rps) <= PROBE_START_MAX_YAW_RATE_RPS
                        and steer is not None
                        and abs(steer.value) <= PROBE_START_MAX_STEERING_RAD
                    )
                    if matched:
                        matched_run.append(row)
                    else:
                        if matched_run:
                            matched_runs.append(matched_run)
                            matched_run = []
                    left_slip = _encoder_slip(
                        left_encoder, left_times, row.receipt_ns, 1.0, 1.0)
                    right_slip = _encoder_slip(
                        right_encoder, right_times, row.receipt_ns, 1.0, 1.0)
                    if left_slip is not None and right_slip is not None:
                        encoder_speed_pairs.append((
                            row.vx_mps, 1.0 + 0.5 * (left_slip + right_slip)))
                print(f"  {start_s:.2f}-{end_s:.2f}s: "
                      f"{statistics.median(math.hypot(r.vx_mps, r.vy_mps) for r in block):.3f} m/s / "
                      f"{statistics.median(r.vy_mps for r in block):+.3f} m/s / "
                      f"{statistics.median(r.yaw_rate_rps for r in block):+.3f} rad/s / "
                      f"{statistics.median(steer_values) if steer_values else math.nan:+.3f} rad / "
                      f"throttle={statistics.median(throttle_values) if throttle_values else math.nan:.3f}")
            if matched_run:
                matched_runs.append(matched_run)
            longest_ready_s = max(
                ((run[-1].receipt_ns - run[0].receipt_ns) / 1e9 for run in matched_runs),
                default=0.0,
            )
            print(f"  samples inside reset limits: {sum(map(len, matched_runs))}/{len(recovery)}, "
                  f"longest continuous interval={longest_ready_s:.3f} s (required "
                  f"{PROBE_START_SETTLE_SEC:.2f} s), "
                  f"speed-error max={max(speed_errors):.3f} m/s, "
                  f"failed thresholds={reset_failures}")
            if encoder_speed_pairs:
                odom_encoder_errors = [abs(odom_vx - encoder_v)
                                       for odom_vx, encoder_v in encoder_speed_pairs]
                print(f"  rear-encoder vs odom longitudinal speed: "
                      f"n={len(encoder_speed_pairs)}, "
                      f"median encoder={statistics.median(v for _, v in encoder_speed_pairs):.3f} m/s, "
                      f"|difference| p50/p95={statistics.median(odom_encoder_errors):.3f}/"
                      f"{_percentile(odom_encoder_errors, 0.95):.3f} m/s")

    probe_phases = [phase for phase in phases
                    if phase.label.startswith(("boundary_", "isolated_", "sweep_",
                                               "steer_"))]
    isolated_phases = [phase for phase in probe_phases
                       if phase.label.startswith("isolated_")]
    matched_starts = 0
    for phase in isolated_phases:
        state = (phase.initial_vx_mps, phase.initial_vy_mps,
                 phase.initial_yaw_rate_rps, phase.initial_steering_rad)
        if any(value is None for value in state):
            continue
        speed = (phase.initial_window_speed_mps
                 if phase.initial_window_speed_mps is not None
                 else math.hypot(phase.initial_vx_mps, phase.initial_vy_mps))
        abs_vy = (phase.initial_window_abs_vy_mps
                  if phase.initial_window_abs_vy_mps is not None
                  else abs(phase.initial_vy_mps))
        abs_yaw = (phase.initial_window_abs_yaw_rate_rps
                   if phase.initial_window_abs_yaw_rate_rps is not None
                   else abs(phase.initial_yaw_rate_rps))
        if (
            abs(speed - phase.target_speed_mps) <= PROBE_START_MAX_SPEED_ERROR_MPS
            and abs_vy <= PROBE_START_MAX_VY_MPS
            and abs_yaw <= PROBE_START_MAX_YAW_RATE_RPS
            and abs(phase.initial_steering_rad) <= PROBE_START_MAX_STEERING_RAD
        ):
            matched_starts += 1
    if isolated_phases:
        print(f"matched isolated probe starts: {matched_starts}/{len(isolated_phases)} "
              f"(speed error<={PROBE_START_MAX_SPEED_ERROR_MPS:.2f} m/s, "
              f"|vy|<={PROBE_START_MAX_VY_MPS:.2f} m/s, "
              f"|yaw rate|<={PROBE_START_MAX_YAW_RATE_RPS:.2f} rad/s, "
              f"|steering|<={PROBE_START_MAX_STEERING_RAD:.2f} rad)")

    rows_by_condition: dict[tuple[int, int, str, int], list[dict[str, float]]] = defaultdict(list)
    steering_errors_all: list[float] = []
    slip_observations: list[tuple[float, float]] = []
    high_angle_predictors: list[tuple[float, ...]] = []
    paired_sweeps: dict[tuple[int, int, int, int], dict[str, dict[str, float]]] = defaultdict(dict)
    probe_phases = [phase for phase in phases
                    if phase.label.startswith(("boundary_", "isolated_", "sweep_",
                                               "steer_"))]
    for phase in phases:
        if not phase.label.startswith(("boundary_", "isolated_", "sweep_",
                                      "steer_")):
            continue
        start = phase.start_ns + PHASE_SETTLE_NS
        low = bisect.bisect_left(odom_times, start)
        high = bisect.bisect_left(odom_times, phase.end_ns)
        selected = odometry[low:high]
        if not selected:
            continue

        speed_errors: list[float] = []
        yaw_ratios: list[float] = []
        yaw_gains: list[float] = []
        yaw_gains_early: list[float] = []
        yaw_gains_late: list[float] = []
        rear_slips: list[float] = []
        rear_signed_slips: list[float] = []
        front_slips: list[float] = []
        front_signed_slips: list[float] = []
        left_longitudinal_slips: list[float] = []
        right_longitudinal_slips: list[float] = []
        lateral_accels: list[float] = []
        imu_yaw_errors: list[float] = []
        imu_lateral_accels: list[float] = []
        encoder_angle_scales: list[float] = []
        ackermann_deltas: list[tuple[float, float]] = []

        for row in selected:
            actual_steering = _nearest_scalar(
                steering_rows, steering_times, row.receipt_ns)
            if actual_steering is None:
                continue
            delta = actual_steering.value
            steering_errors_all.append(abs(delta - phase.commanded_steering_rad))
            speed = math.hypot(row.vx_mps, row.vy_mps)
            speed_errors.append(abs(speed - phase.target_speed_mps))
            if row.receipt_ns in lateral_accel_mps2:
                lateral_accels.append(lateral_accel_mps2[row.receipt_ns])
            if abs(delta) < 0.08 or abs(row.vx_mps) < 0.5:
                continue

            predicted_yaw = row.vx_mps * math.tan(delta) / WHEELBASE_M
            if abs(predicted_yaw) > 0.1:
                yaw_ratios.append(row.yaw_rate_rps / predicted_yaw)
                yaw_gain = row.yaw_rate_rps / (row.vx_mps * math.tan(delta))
                yaw_gains.append(yaw_gain)
                if row.receipt_ns < (phase.start_ns + phase.end_ns) // 2:
                    yaw_gains_early.append(yaw_gain)
                else:
                    yaw_gains_late.append(yaw_gain)

            half_track = TRACK_WIDTH_M * 0.5
            rear_left = _wheel_slip(
                row.vx_mps, row.vy_mps, row.yaw_rate_rps, 0.0, half_track, 0.0)
            rear_right = _wheel_slip(
                row.vx_mps, row.vy_mps, row.yaw_rate_rps, 0.0, -half_track, 0.0)
            if rear_left is not None and rear_right is not None:
                rear_slips.append(0.5 * (abs(rear_left) + abs(rear_right)))
                rear_signed_slips.append(0.5 * (rear_left + rear_right))

            delta_left, delta_right = _ackermann_angles(delta)
            ackermann_deltas.append((delta_left, delta_right))
            front_left = _wheel_slip(
                row.vx_mps, row.vy_mps, row.yaw_rate_rps,
                WHEELBASE_M, half_track, delta_left)
            front_right = _wheel_slip(
                row.vx_mps, row.vy_mps, row.yaw_rate_rps,
                WHEELBASE_M, -half_track, delta_right)
            if front_left is not None and front_right is not None:
                front_slips.append(0.5 * (abs(front_left) + abs(front_right)))
                front_signed_slips.append(0.5 * (front_left + front_right))

            imu_yaw = _nearest_scalar(imu_rows, imu_times, row.receipt_ns)
            if imu_yaw is not None:
                imu_yaw_errors.append(abs(imu_yaw.yaw_rate_rps - row.yaw_rate_rps))
                odom_index = bisect.bisect_left(odom_times, row.receipt_ns)
                if 0 < odom_index < len(odometry) - 1:
                    previous, following = odometry[odom_index - 1], odometry[odom_index + 1]
                    dt_s = (following.source_ns - previous.source_ns) / 1e9
                    if 0.03 <= dt_s <= 0.10:
                        yaw_accel = (following.yaw_rate_rps - previous.yaw_rate_rps) / dt_s
                        imu_lateral_accels.append(
                            imu_yaw.lateral_accel_mps2
                            + yaw_accel * (COM_X_M - 0.08)
                        )

            left = _encoder_slip(
                left_encoder, left_times, row.receipt_ns,
                row.vx_mps - row.yaw_rate_rps * half_track, 1.0)
            right = _encoder_slip(
                right_encoder, right_times, row.receipt_ns,
                row.vx_mps + row.yaw_rate_rps * half_track, 1.0)
            if left is not None and right is not None:
                left_longitudinal_slips.append(left)
                right_longitudinal_slips.append(right)
                for slip_ratio in (left, right):
                    wheel_to_body_ratio = 1.0 + slip_ratio
                    if 0.05 < wheel_to_body_ratio < 10_000.0:
                        encoder_angle_scales.append(1.0 / wheel_to_body_ratio)

        if len(speed_errors) < MIN_PHASE_SAMPLES:
            continue
        phase_steering = phase.commanded_steering_rad
        repetition = int(phase.label.split("_", 2)[1][1:])
        sweep_direction = (
            phase.label.split("_")[3] if phase.label.startswith("sweep_")
            else (f"fixed_{phase.fixed_throttle_command_norm:.3f}"
                  if phase.throttle_mode == "fixed"
                  and phase.fixed_throttle_command_norm is not None
                  else phase.throttle_mode or "unknown")
        )
        key = (round(abs(phase_steering) * 100),
               1 if phase_steering > 0 else -1,
               sweep_direction,
               round(phase.target_speed_mps * 10))
        steering_errors = []
        for row in selected:
            actual = _nearest_scalar(steering_rows, steering_times, row.receipt_ns)
            if actual is not None:
                steering_errors.append(abs(actual.value - phase_steering))
        lateral_force_n = [TOTAL_MASS_KG * acceleration for acceleration in lateral_accels]
        odom_ay = statistics.median(lateral_accels) if lateral_accels else math.nan
        imu_ay = statistics.median(imu_lateral_accels) if imu_lateral_accels else math.nan
        initial_body_speed = phase.initial_window_speed_mps
        if (initial_body_speed is None and phase.initial_vx_mps is not None
                and phase.initial_vy_mps is not None):
            initial_body_speed = math.hypot(phase.initial_vx_mps, phase.initial_vy_mps)
        initial_left_slip = _encoder_slip(
            left_encoder, left_times, phase.start_ns, 1.0, 1.0)
        initial_right_slip = _encoder_slip(
            right_encoder, right_times, phase.start_ns, 1.0, 1.0)
        initial_encoder_scale = math.nan
        if (initial_body_speed is not None
                and initial_left_slip is not None
                and initial_right_slip is not None):
            initial_encoder_speed = 1.0 + 0.5 * (initial_left_slip + initial_right_slip)
            if initial_encoder_speed > 0.5:
                initial_encoder_scale = initial_body_speed / initial_encoder_speed
        initial_time = phase.start_ns + PHASE_SETTLE_NS
        initial_x_row = _nearest_scalar(ips_x, ips_times, initial_time)
        initial_y_row = _nearest_scalar(ips_y, ips_times, initial_time)
        initial_yaw_row = _nearest_scalar(
            orientation_yaw, orientation_yaw_times, initial_time)
        initial_throttle_row = _nearest_scalar(
            throttle_feedback, throttle_feedback_times, initial_time)
        initial_x = initial_x_row.value if initial_x_row is not None else math.nan
        initial_y = initial_y_row.value if initial_y_row is not None else math.nan
        initial_yaw = initial_yaw_row.value if initial_yaw_row is not None else math.nan
        initial_throttle = (
            initial_throttle_row.value if initial_throttle_row is not None else math.nan)
        previous_probes = [probe for probe in probe_phases if probe.index < phase.index]
        previous_probe = previous_probes[-1] if previous_probes else None
        previous_steering = (
            previous_probe.commanded_steering_rad if previous_probe is not None else math.nan)
        previous_probe_gap_s = (
            (phase.start_ns - previous_probe.end_ns) / 1e9
            if previous_probe is not None else math.nan)
        steady_throttles = [
            value.value for row in selected
            if (value := _nearest_scalar(
                throttle_feedback, throttle_feedback_times, row.receipt_ns)) is not None
        ]
        metric = {
            "yaw_ratio": statistics.median(yaw_ratios) if yaw_ratios else math.nan,
            "yaw_gain": statistics.median(yaw_gains) if yaw_gains else math.nan,
            "yaw_gain_early": (
                statistics.median(yaw_gains_early) if yaw_gains_early else math.nan),
            "yaw_gain_late": (
                statistics.median(yaw_gains_late) if yaw_gains_late else math.nan),
            "rear_slip": statistics.median(rear_slips) if rear_slips else math.nan,
            "rear_signed_slip": (
                statistics.median(rear_signed_slips) if rear_signed_slips else math.nan),
            "front_slip": statistics.median(front_slips) if front_slips else math.nan,
            "front_signed_slip": (
                statistics.median(front_signed_slips) if front_signed_slips else math.nan),
            "left_longitudinal_slip": (
                statistics.median(left_longitudinal_slips)
                if left_longitudinal_slips else math.nan),
            "right_longitudinal_slip": (
                statistics.median(right_longitudinal_slips)
                if right_longitudinal_slips else math.nan),
            "lateral_accel_g": (
                statistics.median(lateral_accels) / GRAVITY_MPS2
                if lateral_accels else math.nan
            ),
            "lateral_force_n": (
                statistics.median(lateral_force_n) if lateral_force_n else math.nan
            ),
            "imu_lateral_accel_g": (
                imu_ay / GRAVITY_MPS2 if math.isfinite(imu_ay) else math.nan
            ),
            "odom_imu_lateral_difference": (
                abs(odom_ay - imu_ay) if math.isfinite(odom_ay) and math.isfinite(imu_ay)
                else math.nan
            ),
            "imu_yaw_error": statistics.median(imu_yaw_errors) if imu_yaw_errors else math.nan,
            "encoder_angle_scale": (
                statistics.median(encoder_angle_scales)
                if encoder_angle_scales else math.nan
            ),
            "speed_error_p95": _percentile(speed_errors, 0.95) or 0.0,
            "steering_error_p95": _percentile(steering_errors, 0.95) or 0.0,
            "initial_encoder_scale": initial_encoder_scale,
            "initial_throttle_feedback": initial_throttle,
            "steady_throttle_feedback": (
                statistics.median(steady_throttles) if steady_throttles else math.nan),
            "initial_x": initial_x,
            "initial_y": initial_y,
            "initial_yaw": initial_yaw,
            "previous_probe_steering": previous_steering,
            "previous_probe_gap_s": previous_probe_gap_s,
        }
        if phase.initial_vx_mps is not None and phase.initial_vy_mps is not None:
            metric["initial_speed"] = math.hypot(
                phase.initial_vx_mps, phase.initial_vy_mps)
        if phase.initial_vy_mps is not None:
            metric["initial_vy"] = phase.initial_vy_mps
        if phase.initial_yaw_rate_rps is not None:
            metric["initial_yaw_rate"] = phase.initial_yaw_rate_rps
        rows_by_condition[key].append(metric | {"repetition": float(repetition)})
        if phase.label.startswith("sweep_"):
            pair_key = (
                round(phase.target_speed_mps * 10),
                round(abs(phase_steering) * 100),
                1 if phase_steering > 0.0 else -1,
                repetition,
            )
            paired_sweeps[pair_key][sweep_direction] = metric
        if 0.41 <= abs(phase_steering) <= 0.47 and math.isfinite(initial_encoder_scale):
            high_angle_predictors.append((
                initial_encoder_scale,
                initial_body_speed if initial_body_speed is not None else math.nan,
                metric["encoder_angle_scale"],
                initial_throttle,
                initial_x,
                initial_y,
                initial_yaw,
                previous_steering,
                previous_probe_gap_s,
                metric["yaw_gain"],
            ))
        if rear_slips and yaw_gains:
            slip_observations.append((statistics.median(rear_slips),
                                      statistics.median(yaw_gains)))

        left_med = (statistics.median(value[0] for value in ackermann_deltas)
                    if ackermann_deltas else math.nan)
        right_med = (statistics.median(value[1] for value in ackermann_deltas)
                     if ackermann_deltas else math.nan)
        print(
            f"  block {phase.label}: valid={phase.valid}, n={len(speed_errors)}, "
            f"speed_err_p95={metric['speed_error_p95']:.3f} m/s, "
            f"steer_err_p95={metric['steering_error_p95']:.4f} rad, "
            f"yaw_ratio={metric['yaw_ratio']:.3f}, "
            f"yaw_gain={metric['yaw_gain']:.3f} 1/m "
            f"(early/late={metric['yaw_gain_early']:.3f}/{metric['yaw_gain_late']:.3f}), "
            f"tire_Sy_abs[front/rear]={metric['front_slip']:.4f}/"
            f"{metric['rear_slip']:.4f}; signed={metric['front_signed_slip']:+.4f}/"
            f"{metric['rear_signed_slip']:+.4f}, "
            f"rear_Sx[left/right]={metric['left_longitudinal_slip']:+.4f}/"
            f"{metric['right_longitudinal_slip']:+.4f}, "
            f"net_Fy_CG={metric['lateral_force_n']:.2f} N "
            f"({metric['lateral_accel_g']:.3f} g), "
            f"IMU_ay_CG={metric['imu_lateral_accel_g']:.3f} g, "
            f"|odom-IMU ay|={metric['odom_imu_lateral_difference']:.3f} m/s^2, "
            f"encoder_angle_scale~{metric['encoder_angle_scale']:.4g}, "
            f"|IMU-odom yaw|={metric['imu_yaw_error']:.4f} rad/s, "
            f"start(vy/yaw)={metric.get('initial_vy', math.nan):+.3f} m/s/"
            f"{metric.get('initial_yaw_rate', math.nan):+.3f} rad/s, "
            f"start_speed={phase.initial_window_speed_mps if phase.initial_window_speed_mps is not None else math.nan:.3f} m/s, "
            f"start_encoder_scale={metric['initial_encoder_scale']:.4f}, "
            f"start_xy/yaw=({metric['initial_x']:.2f},{metric['initial_y']:.2f}) m/"
            f"{math.degrees(metric['initial_yaw']):+.1f} deg, "
            f"start/mean_throttle={metric['initial_throttle_feedback']:.3f}/"
            f"{metric['steady_throttle_feedback']:.3f}, "
            f"previous_probe={metric['previous_probe_steering']:+.2f}rad/"
            f"{metric['previous_probe_gap_s']:.2f}s, "
            f"Ackermann_left/right={left_med:.4f}/{right_med:.4f} rad")

    print("phase-block summary (means and t-based 95% CI across repetitions):")
    for (angle_centi, sign, direction, speed_deci), blocks in sorted(rows_by_condition.items()):
        yaw_values = [row["yaw_gain"] for row in blocks if math.isfinite(row["yaw_gain"])]
        rear_values = [row["rear_slip"] for row in blocks if math.isfinite(row["rear_slip"])]
        if len(yaw_values) >= 2:
            mean = statistics.mean(yaw_values)
            sem = statistics.stdev(yaw_values) / math.sqrt(len(yaw_values))
            critical = CI95_T_CRITICAL.get(len(yaw_values) - 1)
            if critical is None:
                yaw_ci = f"{mean:.3f} 1/m (n={len(yaw_values)}; CI unavailable)"
            else:
                half_width = critical * sem
                yaw_ci = f"{mean:.3f} +/- {half_width:.3f} 1/m (n={len(yaw_values)})"
        elif yaw_values:
            yaw_ci = f"{yaw_values[0]:.3f} 1/m (n=1; no CI)"
        else:
            yaw_ci = "n/a"
        rear_mean = statistics.mean(rear_values) if rear_values else math.nan
        print(f"  steer={sign * angle_centi / 100.0:+.2f} rad, "
              f"sweep={direction}, target={speed_deci / 10.0:.1f} m/s: "
              f"yaw_gain={yaw_ci}, rear_|Sy|_median={rear_mean:.4f}")

    if paired_sweeps:
        paired_groups: dict[tuple[int, int], list[tuple[float, float, float]]] = defaultdict(list)
        for (speed_deci, angle_centi, sign, _repetition), pair in paired_sweeps.items():
            if "up" not in pair or "down" not in pair:
                continue
            up, down = pair["up"], pair["down"]
            paired_groups[(speed_deci, angle_centi)].append((
                up["yaw_gain"] - down["yaw_gain"],
                up["rear_slip"] - down["rear_slip"],
                sign * (up["lateral_accel_g"] - down["lateral_accel_g"]),
            ))
        print("paired sweep hysteresis (ascending minus descending; four sign/repetition pairs):")
        for (speed_deci, angle_centi), differences in sorted(paired_groups.items()):
            yaw_delta = [row[0] for row in differences if math.isfinite(row[0])]
            rear_delta = [row[1] for row in differences if math.isfinite(row[1])]
            ay_delta = [row[2] for row in differences if math.isfinite(row[2])]
            if not yaw_delta:
                continue
            sign_consistency = max(
                sum(value > 0.0 for value in yaw_delta),
                sum(value < 0.0 for value in yaw_delta),
            )
            print(f"  target={speed_deci / 10.0:.1f} m/s steer={angle_centi / 100.0:.2f} rad: "
                  f"delta_yaw_gain median/range="
                  f"{statistics.median(yaw_delta):+.3f}/"
                  f"[{min(yaw_delta):+.3f},{max(yaw_delta):+.3f}] 1/m, "
                  f"same-sign={sign_consistency}/{len(yaw_delta)}, "
                  f"delta_rear_|Sy|={statistics.median(rear_delta):+.4f}, "
                  f"delta_sign_normalized_ay={statistics.median(ay_delta):+.3f} g")

    if len(high_angle_predictors) >= 3:
        predictor_labels = (
            "initial encoder/body speed ratio",
            "initial body speed",
            "within-probe encoder/body speed ratio",
            "initial throttle feedback",
            "initial IPS x",
            "initial IPS y",
            "initial world yaw",
            "previous probe steering",
            "previous probe settle gap",
        )
        for column, label in enumerate(predictor_labels):
            pairs = [(row[column], row[9]) for row in high_angle_predictors
                     if math.isfinite(row[column]) and math.isfinite(row[9])]
            if len(pairs) >= 3 and statistics.stdev(x for x, _ in pairs) > 0.0:
                correlation = statistics.correlation(
                    [x for x, _ in pairs], [y for _, y in pairs])
                print(f"high-angle block correlation, {label} vs yaw gain: "
                      f"r={correlation:+.3f} (n={len(pairs)}; exploratory)")

    expected_steering_max = 0.5236
    actual_max = max(abs(row.value) for row in steering_rows)
    print(f"steering feedback range: +/-{actual_max:.4f} rad; "
          f"published limit: +/-{expected_steering_max:.4f} rad")
    if slip_observations:
        x_values = [item[0] for item in slip_observations]
        y_values = [item[1] for item in slip_observations]
        print(f"observed rear tire-slip proxy |Sy| range: "
              f"{min(x_values):.4f}..{max(x_values):.4f}; "
              f"block yaw-gain range={min(y_values):.3f}..{max(y_values):.3f} 1/m; "
              f"guide lateral tire-force landmarks: peak at |Sy|=0.010, "
              "asymptote at |Sy|=0.100")
    print("interpretation: phase statistics characterize yaw authority and slip; "
          "they do not identify per-wheel force coefficients without contact-force data.")

    good = (
        experiment_end.get("aborted") is False
        and not experiment_end.get("quality_failures")
        and len(phases) == int(experiment_end.get("phase_count", len(phases)))
        and probe_phases
        and all(phase.valid is True for phase in probe_phases)
        and (not isolated_phases or matched_starts == len(isolated_phases))
        and stream_cadence_ok
        and fixed_throttle_ok
        and collision_values
        and max(collision_values) == 0
        and not any(fault_values)
    )
    return 0 if good else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase/block analysis of open-plane vehicle-dynamics captures.")
    parser.add_argument("bag", type=Path, help="run_0.db3 from the experiment recorder")
    args = parser.parse_args()
    try:
        return analyze(args.bag)
    except (OSError, sqlite3.Error, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export reset-isolated throttle experiments on the bridge request clock.

The accepted-condition list comes from the closed-bag throttle audit.  Bridge
response timestamps can bunch even though requests are issued regularly, so
the plant timebase is reconstructed from each packet's request monotonic time.
Simulator packet fields stay in separate debug/label arrays; IPS is an offline
position label and never becomes a plant input.

Run inside the repository's ROS 2 development image because rosbag CDR message
deserialization is provided by ROS 2 Python packages.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_open_plane_dynamics as common  # noqa: E402
from tools.evaluate_open_plane_body_dynamics import (  # noqa: E402
    COM_X_M,
    STEERING_LIMIT_RAD,
)


ODOM = common.ODOM
IPS = common.IPS
IMU = common.IMU
LEFT = common.LEFT_ENCODER
RIGHT = common.RIGHT_ENCODER
STEERING = common.STEERING
THROTTLE = common.THROTTLE_FEEDBACK
STEERING_COMMAND = "/autodrive/roboracer_1/steering_command"
THROTTLE_COMMAND = common.THROTTLE_COMMAND
PHASE = common.PHASE
PACKET_TIMING = common.PACKET_TIMING
ALIGNMENT_NS = 30_000_000
COMMAND_ALIGNMENT_NS = 120_000_000
IPS_NOMINAL_FORWARD_OFFSET_M = 0.08  # diagnostic hypothesis, not applied
SIMULATOR_DT_S = 0.025
ENCODER_DIFFERENCE_STEPS = 4

STREAM_NAMES = (
    "steering_feedback", "throttle_feedback", "steering_command",
    "throttle_command", "left_encoder", "right_encoder", "imu", "ips",
)


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def _roll_pitch(x: float, y: float, z: float,
                w: float) -> tuple[float, float]:
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or norm < 0.5:
        return math.nan, math.nan
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = math.atan2(2.0 * (w * x + y * z),
                      1.0 - 2.0 * (x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w*y - z*x))))
    return roll, pitch


def _read_series(connection: sqlite3.Connection, topic_id: int,
                 message_class: type, extract) -> tuple[np.ndarray, np.ndarray,
                                                       np.ndarray]:
    from rclpy.serialization import deserialize_message

    times: list[int] = []
    source_times: list[int] = []
    values: list[list[float]] = []
    cursor = connection.execute(
        "SELECT timestamp, data FROM messages "
        "WHERE topic_id=? ORDER BY timestamp, id", (topic_id,))
    for receipt_ns, payload in cursor:
        message = deserialize_message(bytes(payload), message_class)
        value, source_ns = extract(message, int(receipt_ns))
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.isfinite(vector).all():
            continue
        times.append(int(receipt_ns))
        source_times.append(int(source_ns))
        values.append(vector.tolist())
    return (np.asarray(times, dtype=np.int64),
            np.asarray(source_times, dtype=np.int64),
            np.asarray(values, dtype=np.float64))


def _topic_types(connection: sqlite3.Connection) -> dict[str, tuple[int, str]]:
    return {str(name): (int(topic_id), str(msg_type))
            for topic_id, name, msg_type in connection.execute(
                "SELECT id, name, type FROM topics")}


def _load_events(connection: sqlite3.Connection, topic_id: int) -> dict[str, Any]:
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String

    starts: dict[int, tuple[int, dict[str, Any]]] = {}
    stimuli: dict[int, tuple[int, dict[str, Any]]] = {}
    ends: dict[int, tuple[int, dict[str, Any]]] = {}
    recoveries: list[tuple[int, dict[str, Any]]] = []
    cursor = connection.execute(
        "SELECT timestamp, data FROM messages "
        "WHERE topic_id=? ORDER BY timestamp, id", (topic_id,))
    for receipt_ns, payload in cursor:
        event_message = deserialize_message(bytes(payload), String)
        try:
            event = json.loads(event_message.data)
        except (TypeError, json.JSONDecodeError):
            continue
        kind = event.get("event")
        index = event.get("phase_index")
        if kind == "phase_start" and index is not None:
            starts[int(index)] = (int(receipt_ns), event)
        elif kind == "throttle_slew_stimulus" and index is not None:
            stimuli[int(index)] = (int(receipt_ns), event)
        elif kind == "phase_end" and index is not None:
            ends[int(index)] = (int(receipt_ns), event)
        elif kind == "sim_reset_recovered":
            recoveries.append((int(receipt_ns), event))
    return {"starts": starts, "stimuli": stimuli, "ends": ends,
            "recoveries": recoveries}


PACKET_DEBUG_NAMES = (
    "simulator_position_x_m", "simulator_position_y_m",
    "simulator_position_z_m", "simulator_orientation_qx",
    "simulator_orientation_qy", "simulator_orientation_qz",
    "simulator_orientation_qw", "simulator_euler_x_rad",
    "simulator_euler_y_rad", "simulator_euler_z_rad",
    "simulator_velocity_x_mps", "simulator_velocity_y_mps",
    "simulator_velocity_z_mps", "simulator_angular_x_rps",
    "simulator_angular_y_rps", "simulator_angular_z_rps",
    "simulator_acceleration_x_mps2", "simulator_acceleration_y_mps2",
    "simulator_acceleration_z_mps2", "simulator_encoder_angle_left_rad",
    "simulator_encoder_angle_right_rad",
    "simulator_feedback_throttle_norm",
    "simulator_feedback_steering_rad", "simulator_collision_count",
)


def _load_packet_timing(connection: sqlite3.Connection,
                        topic_id: int) -> tuple[dict[int, dict[str, Any]],
                                                dict[str, Any]]:
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String

    packets: dict[int, dict[str, Any]] = {}
    receive_ns: list[int] = []
    arrival_ns: list[int] = []
    request_ns: list[int] = []
    packet_sequence: list[int] = []
    request_sequence: list[int] = []
    unmatched_response_count = 0
    unmatched_response_examples: list[dict[str, Any]] = []
    message_count = 0
    for _, payload in connection.execute(
            "SELECT timestamp, data FROM messages "
            "WHERE topic_id=? ORDER BY timestamp, id", (topic_id,)):
        message = deserialize_message(bytes(payload), String)
        try:
            row = json.loads(message.data)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid bridge_packet_timing JSON") from exc
        receive = row.get("bridge_receive_ros_stamp_ns")
        arrival = row.get("bridge_arrival_monotonic_ns")
        request = row.get("request_monotonic_ns")
        sequence = row.get("packet_sequence")
        request_id = row.get("request_sequence")
        message_count += 1
        if not all(isinstance(value, int) for value in
                   (receive, arrival, sequence)):
            raise ValueError("bridge packet lacks receive/arrival sequence fields")
        if (not isinstance(request, int) or not isinstance(request_id, int)
                or request_id < 0):
            # The bridge intentionally publishes a response even if no request
            # is pending. Such a response has no defensible control-time key;
            # retain it in the audit count but do not invent a timestamp.
            unmatched_response_count += 1
            if len(unmatched_response_examples) < 5:
                unmatched_response_examples.append({
                    "receive_ros_stamp_ns": receive,
                    "packet_sequence": sequence,
                    "request_sequence": request_id,
                    "request_monotonic_ns": request,
                })
            continue
        if receive in packets:
            raise ValueError(f"duplicate bridge receive stamp: {receive}")
        packets[receive] = row
        receive_ns.append(receive)
        arrival_ns.append(arrival)
        request_ns.append(request)
        packet_sequence.append(sequence)
        request_sequence.append(request_id)
    if len(packets) < 1000:
        raise ValueError("bridge_packet_timing has too few decoded packets")

    def intervals(values: list[int]) -> np.ndarray:
        return np.diff(np.asarray(values, dtype=np.int64)).astype(np.float64) / 1e6

    receive_dt = intervals(receive_ns)
    arrival_dt = intervals(arrival_ns)
    request_dt = intervals(request_ns)
    packet_step = np.diff(np.asarray(packet_sequence, dtype=np.int64))
    request_step = np.diff(np.asarray(request_sequence, dtype=np.int64))
    latency = (np.asarray(arrival_ns, dtype=np.int64)
               - np.asarray(request_ns, dtype=np.int64)).astype(np.float64) / 1e6
    ros_monotonic_offset_ms = (
        np.asarray(receive_ns, dtype=np.int64)
        - np.asarray(arrival_ns, dtype=np.int64)).astype(np.float64) / 1e6
    report = {
        "packet_count": len(packets),
        "bridge_timing_message_count": message_count,
        "unmatched_response_count": unmatched_response_count,
        "unmatched_response_examples": unmatched_response_examples,
        "packet_sequence_step_histogram": {
            str(int(value)): int(count) for value, count in
            zip(*np.unique(packet_step, return_counts=True))},
        "request_sequence_step_histogram": {
            str(int(value)): int(count) for value, count in
            zip(*np.unique(request_step, return_counts=True))},
        "request_interval_ms_p01_p50_p99_min_max": [
            float(np.quantile(request_dt, q)) for q in (0.01, 0.50, 0.99)]
            + [float(np.min(request_dt)), float(np.max(request_dt))],
        "response_receive_interval_ms_p01_p50_p99_min_max": [
            float(np.quantile(receive_dt, q)) for q in (0.01, 0.50, 0.99)]
            + [float(np.min(receive_dt)), float(np.max(receive_dt))],
        "bridge_arrival_interval_ms_p01_p50_p99_min_max": [
            float(np.quantile(arrival_dt, q)) for q in (0.01, 0.50, 0.99)]
            + [float(np.min(arrival_dt)), float(np.max(arrival_dt))],
        "request_to_arrival_latency_ms_p01_p50_p99": [
            float(np.quantile(latency, q)) for q in (0.01, 0.50, 0.99)],
        "ros_minus_monotonic_clock_offset_ms_p01_p50_p99": [
            float(np.quantile(ros_monotonic_offset_ms, q))
            for q in (0.01, 0.50, 0.99)],
        "nonincreasing_request_clock_count": int(np.count_nonzero(request_dt <= 0.0)),
        "nonunit_packet_sequence_step_count": int(np.count_nonzero(packet_step != 1)),
        "nonunit_request_sequence_step_count": int(np.count_nonzero(request_step != 1)),
    }
    return packets, report


def _causal_indices(times: np.ndarray, query: np.ndarray,
                    max_age_ns: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(times, query, side="right") - 1
    safe = np.maximum(indices, 0)
    age = query - times[safe]
    invalid = (indices < 0) | (age < 0) | (age > max_age_ns)
    age = age.astype(np.float64) / 1e6
    age[invalid] = np.nan
    return safe, age


def _nearest_indices(times: np.ndarray, query: np.ndarray,
                     max_offset_ns: int) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(times, query, side="left")
    left = np.clip(right - 1, 0, max(0, len(times) - 1))
    right = np.clip(right, 0, max(0, len(times) - 1))
    choose_right = np.abs(times[right] - query) < np.abs(times[left] - query)
    indices = np.where(choose_right, right, left)
    signed_ms = (times[indices] - query).astype(np.float64) / 1e6
    invalid = np.abs(signed_ms) > max_offset_ns / 1e6
    signed_ms[invalid] = np.nan
    return indices, signed_ms


def _exact_source_indices(source_times: np.ndarray,
                          query_source_times: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Join stamped sensors to their matching simulator packet.

    Receipt-time order can put the encoder response after its corresponding
    odometry response. A causal receipt lookup then borrows the previous
    wheel angle. Source stamps are used only as packet identity here, not as
    physical integration intervals; unmatched rows stay invalid.
    """
    if not len(source_times):
        return (np.zeros(len(query_source_times), dtype=np.int64),
                np.zeros(len(query_source_times), dtype=bool))
    order = np.argsort(source_times, kind="stable")
    ordered_stamps = source_times[order]
    position = np.searchsorted(ordered_stamps, query_source_times,
                               side="left")
    safe = np.minimum(position, len(ordered_stamps) - 1)
    matched = ((position < len(ordered_stamps))
               & (ordered_stamps[safe] == query_source_times))
    return order[safe], matched


def _finite_stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "median": None,
                "p95_abs": None, "rmse": None}
    return {"count": int(len(values)), "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95_abs": float(np.quantile(np.abs(values), 0.95)),
            "rmse": float(np.sqrt(np.mean(values**2)))}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _load_capture(bag: Path, audit_path: Path) -> tuple[dict[str, np.ndarray],
                                                       dict[str, Any]]:
    from rosidl_runtime_py.utilities import get_message

    audit = json.loads(audit_path.read_text())
    usable = {int(row["phase_index"]): row for row in audit["conditions"]
              if row.get("usable_for_response_fit") is True}
    if not usable:
        raise ValueError(f"audit has no fit-usable phases: {audit_path}")
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = _topic_types(connection)
        required = (ODOM, IPS, IMU, LEFT, RIGHT, STEERING, THROTTLE,
                    STEERING_COMMAND, THROTTLE_COMMAND, PHASE, PACKET_TIMING)
        missing = [topic for topic in required if topic not in topics]
        if missing:
            raise ValueError(f"{bag}: missing topic(s): {missing}")

        # Resolve classes from the bag's declared types, avoiding assumptions
        # about generated message paths beyond each standard type's fields.
        def read(topic: str, extract):
            topic_id, type_name = topics[topic]
            return _read_series(connection, topic_id, get_message(type_name),
                                extract)

        odom = read(ODOM, lambda m, t: (
            [m.pose.pose.position.x, m.pose.pose.position.y,
             m.pose.pose.position.z,
             _yaw_from_quaternion(m.pose.pose.orientation.x,
                                  m.pose.pose.orientation.y,
                                  m.pose.pose.orientation.z,
                                  m.pose.pose.orientation.w),
             m.twist.twist.linear.x, m.twist.twist.linear.y,
             m.twist.twist.linear.z, m.twist.twist.angular.x,
             m.twist.twist.angular.y, m.twist.twist.angular.z],
            _stamp_ns(m.header.stamp) or t))
        ips = read(IPS, lambda m, t: ([m.x, m.y, m.z], t))
        imu = read(IMU, lambda m, t: (
            [m.linear_acceleration.x, m.linear_acceleration.y,
             m.linear_acceleration.z, m.angular_velocity.x,
             m.angular_velocity.y, m.angular_velocity.z,
             *_roll_pitch(m.orientation.x, m.orientation.y,
                          m.orientation.z, m.orientation.w)],
            _stamp_ns(m.header.stamp) or t))

        def encoder(topic: str):
            def extract(m, receipt):
                position = float(m.position[0]) if m.position else math.nan
                velocity = float(m.velocity[0]) if m.velocity else math.nan
                if not math.isfinite(velocity):
                    velocity = 0.0  # validity is recorded separately below
                return [position, velocity, float(bool(m.position)),
                        float(bool(m.velocity))], _stamp_ns(m.header.stamp) or receipt
            return read(topic, extract)

        enc_l = encoder(LEFT)
        enc_r = encoder(RIGHT)

        def scalar(topic: str):
            return read(topic, lambda m, t: ([float(m.data)], t))

        steering = scalar(STEERING)
        throttle = scalar(THROTTLE)
        steering_command = scalar(STEERING_COMMAND)
        throttle_command = scalar(THROTTLE_COMMAND)
        events = _load_events(connection, topics[PHASE][0])
        packets, packet_clock_report = _load_packet_timing(
            connection, topics[PACKET_TIMING][0])
    finally:
        connection.close()

    odom_t, odom_src, odom_v = odom
    if len(odom_t) < 1000:
        raise ValueError(f"too few odometry samples in {bag}: {len(odom_t)}")
    run_id = bag.parents[1].name

    # ODOM headers carry bridge receive time. Join each response to its request
    # using the embedded receive ROS stamp, then map request monotonic time back
    # into the ROS clock with the per-bag monotonic/ROS offset.
    packet_rows: list[dict[str, Any] | None] = [None] * len(odom_src)
    clock_offsets_ns = np.asarray([
        int(row["bridge_receive_ros_stamp_ns"])
        - int(row["bridge_arrival_monotonic_ns"])
        for row in packets.values()], dtype=np.int64)
    clock_offset_ns = int(np.median(clock_offsets_ns))
    for index, stamp in enumerate(odom_src):
        packet_rows[index] = packets.get(int(stamp))
    packet_valid = np.asarray([row is not None for row in packet_rows], dtype=bool)
    packet_match_fraction = float(np.mean(packet_valid))
    if packet_match_fraction < 0.999:
        raise ValueError(f"{run_id}: only {packet_match_fraction:.3%} of /odom "
                         "samples match bridge response packets")

    packet_sequence = np.full(len(odom_src), -1, dtype=np.int64)
    packet_request_sequence = np.full(len(odom_src), -1, dtype=np.int64)
    packet_request_monotonic_ns = np.full(len(odom_src), -1, dtype=np.int64)
    packet_arrival_monotonic_ns = np.full(len(odom_src), -1, dtype=np.int64)
    packet_request_ros_ns = np.full(len(odom_src), -1, dtype=np.int64)
    packet_sent_commands = np.full((len(odom_src), 2), np.nan, dtype=np.float64)
    packet_debug = np.full((len(odom_src), len(PACKET_DEBUG_NAMES)),
                           np.nan, dtype=np.float64)
    packet_debug_keys = (
        "simulator_position_x", "simulator_position_y", "simulator_position_z",
        "simulator_orientation_quaternion_x", "simulator_orientation_quaternion_y",
        "simulator_orientation_quaternion_z", "simulator_orientation_quaternion_w",
        "simulator_orientation_euler_x", "simulator_orientation_euler_y",
        "simulator_orientation_euler_z", "simulator_linear_velocity_x",
        "simulator_linear_velocity_y", "simulator_linear_velocity_z",
        "simulator_angular_velocity_x", "simulator_angular_velocity_y",
        "simulator_angular_velocity_z", "simulator_linear_acceleration_x",
        "simulator_linear_acceleration_y", "simulator_linear_acceleration_z",
        "simulator_encoder_angles_left", "simulator_encoder_angles_right",
        "simulator_feedback_throttle_norm", "simulator_feedback_steering_norm",
        "simulator_collision_count",
    )
    for index, packet in enumerate(packet_rows):
        if packet is None:
            continue
        request_mono = int(packet["request_monotonic_ns"])
        arrival_mono = int(packet["bridge_arrival_monotonic_ns"])
        receive_ros = int(packet["bridge_receive_ros_stamp_ns"])
        packet_sequence[index] = int(packet["packet_sequence"])
        packet_request_sequence[index] = int(packet["request_sequence"])
        packet_request_monotonic_ns[index] = request_mono
        packet_arrival_monotonic_ns[index] = arrival_mono
        packet_request_ros_ns[index] = request_mono + clock_offset_ns
        sent_steer = packet.get("sent_steering_norm")
        sent_throttle = packet.get("sent_throttle_norm")
        if sent_steer is not None and sent_throttle is not None:
            packet_sent_commands[index] = (
                float(sent_steer) * STEERING_LIMIT_RAD,
                float(sent_throttle))
        for column, key in enumerate(packet_debug_keys):
            value = packet.get(key)
            if value is not None:
                packet_debug[index, column] = float(value)
        if abs(packet_request_ros_ns[index] - receive_ros) > 500_000_000:
            raise ValueError(f"{run_id}: request clock conversion out of range")

    matched = np.flatnonzero(packet_valid)
    request_dt_ms = np.diff(packet_request_monotonic_ns[matched]).astype(
        np.float64) / 1e6
    if np.any(request_dt_ms <= 0.0) or np.quantile(request_dt_ms, 0.99) > 75.0:
        raise ValueError(f"{run_id}: invalid bridge request cadence")
    phase_indices = sorted(set(events["starts"]) & set(events["stimuli"])
                           & set(events["ends"]) & set(usable))
    if set(phase_indices) != set(usable):
        missing_events = sorted(set(usable) - set(phase_indices))
        raise ValueError(f"audit/event phase mismatch for {run_id}: "
                         f"{missing_events[:10]}")

    streams = {
        "steering_feedback": steering,
        "throttle_feedback": throttle,
        "steering_command": steering_command,
        "throttle_command": throttle_command,
        "left_encoder": enc_l,
        "right_encoder": enc_r,
        "imu": imu,
        "ips": ips,
    }
    stream_values: dict[str, np.ndarray] = {}
    stream_age: dict[str, np.ndarray] = {}
    encoder_source_match_fraction: dict[str, float] = {}
    for name, (times, source_times, values) in streams.items():
        if not len(times):
            raise ValueError(f"{bag}: stream has no samples: {name}")
        query = odom_t
        if name in ("left_encoder", "right_encoder"):
            indices, matched = _exact_source_indices(source_times, odom_src)
            selected = values[indices].copy()
            selected[~matched] = np.nan
            encoder_source_match_fraction[name] = float(np.mean(matched))
            _, age = _causal_indices(times, query, ALIGNMENT_NS)
        elif name == "ips":
            indices, age = _nearest_indices(times, query, ALIGNMENT_NS)
            selected = values[indices].copy()
        else:
            max_age = (COMMAND_ALIGNMENT_NS if name.endswith("command")
                       else ALIGNMENT_NS)
            indices, age = _causal_indices(times, query, max_age)
            selected = values[indices].copy()
        if name not in ("left_encoder", "right_encoder"):
            selected[~np.isfinite(age)] = np.nan
        stream_values[name] = selected
        stream_age[f"{name}_alignment_ms"] = age

    # Preserve the existing rear-axle velocity convention used by the project.
    body_state = np.column_stack((
        odom_v[:, 4],
        odom_v[:, 5] - COM_X_M * odom_v[:, 9],
        odom_v[:, 9],
    ))
    steering_command_rad = (stream_values["steering_command"][:, 0]
                            * STEERING_LIMIT_RAD)
    recoveries = [stamp for stamp, _ in events["recoveries"]]
    blocks: dict[str, list[np.ndarray]] = {
        "body_state": [], "odom_pose": [], "ips_position": [],
        "plant_commands": [], "command_topics": [], "actuator_feedback": [],
        "encoder_position_rad": [], "encoder_velocity_rad_s": [],
        "encoder_velocity_valid": [], "encoder_surface_mps_100ms": [],
        "encoder_source_stamp_match": [],
        "imu_acceleration_mps2": [], "imu_angular_velocity_rps": [],
        "imu_roll_pitch_rad": [], "bridge_debug_telemetry": [],
        "packet_sequence": [], "request_sequence": [],
        "request_monotonic_ns": [], "request_time_ns": [],
        "dt_s": [],
        "sequence_time_s": [], "phase_receipt_relative_s": [],
        "time_from_stimulus_s": [], "receipt_time_ns": [],
        "source_stamp_offset_ms": [],
        "alignment_ages_ms": [], "sequence_index": [],
    }
    sequence_rows: list[dict[str, Any]] = []
    cursor = 0
    for sequence_index, phase_index in enumerate(phase_indices):
        start_ns, start = events["starts"][phase_index]
        stimulus_ns, _ = events["stimuli"][phase_index]
        end_ns, end = events["ends"][phase_index]
        indices = np.flatnonzero(
            packet_valid & (packet_request_ros_ns >= start_ns)
            & (packet_request_ros_ns <= end_ns))
        if len(indices) < 300:
            raise ValueError(f"short usable sequence {run_id}:{phase_index}: "
                             f"{len(indices)} odometry rows")
        n = len(indices)
        # The simulator advances on a nominal 40 Hz physics cadence. Host
        # request/receive clocks are retained for association and diagnostics,
        # but network jitter must not become simulated plant time.
        local_model_dt = np.full(n, SIMULATOR_DT_S, dtype=np.float64)
        local_encoder_position = np.column_stack((
            stream_values["left_encoder"][indices, 0],
            stream_values["right_encoder"][indices, 0]))
        local_packet_sequence = packet_sequence[indices]
        local_encoder_surface = np.full((n, 2), np.nan, dtype=np.float64)
        for local_index in range(ENCODER_DIFFERENCE_STEPS, n):
            window_sequences = local_packet_sequence[
                local_index - ENCODER_DIFFERENCE_STEPS:local_index + 1]
            window_angles = local_encoder_position[
                local_index - ENCODER_DIFFERENCE_STEPS:local_index + 1]
            if (not np.all(np.diff(window_sequences) == 1)
                    or not np.isfinite(window_angles).all()):
                continue
            local_encoder_surface[local_index] = (
                common.WHEEL_RADIUS_M
                * (local_encoder_position[local_index]
                   - local_encoder_position[
                       local_index - ENCODER_DIFFERENCE_STEPS])
                / (ENCODER_DIFFERENCE_STEPS * SIMULATOR_DT_S))
        def append(key: str, array: np.ndarray) -> None:
            arr = np.asarray(array)
            if arr.ndim == 1:
                arr = arr[:, None]
            if len(arr) != n:
                raise ValueError(f"internal row mismatch for {key}")
            blocks[key].append(arr)

        append("body_state", body_state[indices])
        append("odom_pose", odom_v[indices, :4])
        append("ips_position", stream_values["ips"][indices, :3])
        append("plant_commands", packet_sent_commands[indices])
        append("command_topics", np.column_stack((
            steering_command_rad[indices],
            stream_values["throttle_command"][indices, 0])))
        append("actuator_feedback", np.column_stack((
            stream_values["steering_feedback"][indices, 0],
            stream_values["throttle_feedback"][indices, 0])))
        append("encoder_position_rad", local_encoder_position)
        append("encoder_velocity_rad_s", np.column_stack((
            stream_values["left_encoder"][indices, 1],
            stream_values["right_encoder"][indices, 1])))
        append("encoder_velocity_valid", np.column_stack((
            stream_values["left_encoder"][indices, 3],
            stream_values["right_encoder"][indices, 3])))
        append("encoder_surface_mps_100ms", local_encoder_surface)
        append("encoder_source_stamp_match", np.isfinite(
            local_encoder_position).astype(np.float32))
        append("imu_acceleration_mps2", stream_values["imu"][indices, :3])
        append("imu_angular_velocity_rps", stream_values["imu"][indices, 3:6])
        append("imu_roll_pitch_rad", stream_values["imu"][indices, 6:8])
        append("bridge_debug_telemetry", packet_debug[indices])
        append("packet_sequence", packet_sequence[indices])
        append("request_sequence", packet_request_sequence[indices])
        append("request_monotonic_ns", packet_request_monotonic_ns[indices])
        append("request_time_ns", packet_request_ros_ns[indices])
        append("dt_s", local_model_dt)
        append("sequence_time_s", ((packet_request_ros_ns[indices]
                                    - start_ns) / 1e9))
        append("phase_receipt_relative_s", ((odom_t[indices] - start_ns) / 1e9))
        append("time_from_stimulus_s", ((packet_request_ros_ns[indices]
                                        - stimulus_ns) / 1e9))
        append("receipt_time_ns", odom_t[indices])
        append("source_stamp_offset_ms", ((odom_src[indices]
                                            - odom_t[indices]) / 1e6))
        append("alignment_ages_ms", np.column_stack(
            [stream_age[f"{name}_alignment_ms"][indices] for name in STREAM_NAMES]))
        append("sequence_index", np.full(n, sequence_index, dtype=np.int32))

        previous_recovery = bisect.bisect_right(recoveries, start_ns) - 1
        reset_id = (f"{run_id}:recovery_{previous_recovery:04d}"
                    if previous_recovery >= 0 else f"{run_id}:initial")
        condition = usable[phase_index]
        sequence_rows.append({
            "sequence_index": sequence_index,
            "start": cursor,
            "end": cursor + n,
            "run_id": run_id,
            "bag": str(bag.resolve().relative_to(REPO_ROOT)),
            "phase_index": phase_index,
            "label": condition.get("label"),
            "replicate_index": int(condition["replicate_index"]),
            "condition_key": [round(float(condition["steering_command_rad"]), 4),
                              round(float(condition["throttle_start_norm"]), 2),
                              round(float(condition["throttle_end_norm"]), 2)],
            "steering_command_rad": float(condition["steering_command_rad"]),
            "throttle_start_norm": float(condition["throttle_start_norm"]),
            "throttle_end_norm": float(condition["throttle_end_norm"]),
            "reset_id": reset_id,
            "phase_start_receipt_ns": start_ns,
            "stimulus_receipt_ns": stimulus_ns,
            "phase_end_receipt_ns": end_ns,
            "sample_count": n,
            "duration_s": float((n - 1) * SIMULATOR_DT_S),
            "encoder_source_stamp_match_fraction": [
                float(np.mean(np.isfinite(local_encoder_position[:, side])))
                for side in range(2)],
            "request_clock_span_s": float((packet_request_ros_ns[indices[-1]]
                                             - packet_request_ros_ns[indices[0]]) / 1e9),
            "response_latency_ms_median": float(np.median(
                (packet_arrival_monotonic_ns[indices]
                 - packet_request_monotonic_ns[indices]) / 1e6)),
            "packet_sequence_step_histogram": {
                str(int(value)): int(count) for value, count in zip(
                    *np.unique(np.diff(packet_sequence[indices]),
                               return_counts=True))},
            "quality": "audit_usable_for_response_fit",
            "baseline_state_from_audit": condition.get("baseline_state"),
            "end_state_from_audit": condition.get("end_state"),
            "end_event_valid": end.get("valid") is True,
        })
        cursor += n

    data = {key: np.concatenate(value, axis=0)
            for key, value in blocks.items() if value}
    data["sequence_bounds"] = np.asarray(
        [[row["start"], row["end"]] for row in sequence_rows], dtype=np.int64)
    data["sequence_run_id"] = np.asarray(
        [row["run_id"] for row in sequence_rows], dtype="U128")
    # Row-wise topics carry identical sensor ages; feature order is stable.
    ips_position = data["ips_position"]
    pose = data["odom_pose"]
    yaw = pose[:, 3]
    raw_delta_world = ips_position[:, :2] - pose[:, :2]
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    delta_body = np.column_stack((
        cos_yaw * raw_delta_world[:, 0] + sin_yaw * raw_delta_world[:, 1],
        -sin_yaw * raw_delta_world[:, 0] + cos_yaw * raw_delta_world[:, 1]))
    nominal_offset_body = delta_body - np.asarray(
        [IPS_NOMINAL_FORWARD_OFFSET_M, 0.0])[None, :]
    finite_ips = np.isfinite(ips_position[:, :2]).all(axis=1)
    # Estimate transform consistency from phase medians, not correlated 40 Hz
    # samples, and never feed the estimated offset into exported labels.
    phase_transform: list[list[float]] = []
    for row in sequence_rows:
        selected = slice(row["start"], row["end"])
        valid = finite_ips[selected]
        if np.count_nonzero(valid) < 20:
            continue
        local = delta_body[selected][valid]
        phase_transform.append(np.median(local, axis=0).tolist())
    transforms = np.asarray(phase_transform, dtype=float).reshape(-1, 2)
    pose_diagnostics = {
        "ips_nearest_alignment_abs_ms": _finite_stats(
            data["alignment_ages_ms"][:, STREAM_NAMES.index("ips")]),
        "ips_minus_odom_world_xy_m": {
            axis: _finite_stats(raw_delta_world[finite_ips, i])
            for i, axis in enumerate(("x", "y"))},
        "ips_minus_odom_rotated_into_body_m": {
            axis: _finite_stats(delta_body[finite_ips, i])
            for i, axis in enumerate(("forward", "left"))},
        "body_offset_phase_medians_m": {
            axis: _finite_stats(transforms[:, i]) if len(transforms) else None
            for i, axis in enumerate(("forward", "left"))},
        "residual_after_nominal_0p08m_forward_offset_m": {
            axis: _finite_stats(nominal_offset_body[finite_ips, i])
            for i, axis in enumerate(("forward", "left"))},
        "ips_valid_fraction": float(np.mean(finite_ips)),
        "nominal_sensor_offset_m": [IPS_NOMINAL_FORWARD_OFFSET_M, 0.0],
        "interpretation": (
            "Diagnostics only. Check frame, timestamp and reset consistency; "
            "do not treat IPS as truth or apply the nominal offset unless "
            "recorded residuals support it."),
    }
    encoder_valid = data["encoder_velocity_valid"] > 0.5
    encoder_speed_report = {}
    for side, name in enumerate(("left", "right")):
        angular = data["encoder_velocity_rad_s"][:, side]
        direct_surface = angular * common.WHEEL_RADIUS_M
        derived = data["encoder_surface_mps_100ms"][:, side]
        valid = encoder_valid[:, side] & np.isfinite(derived)
        encoder_speed_report[name] = {
            "valid_velocity_fraction": float(np.mean(encoder_valid[:, side])),
            "velocity_rad_s": _finite_stats(angular),
            "fixed_100ms_surface_speed_mps": _finite_stats(derived),
            "exact_source_stamp_match_fraction": float(np.mean(
                data["encoder_source_stamp_match"][:, side] > 0.5)),
            "direct_minus_100ms_derived_mps": _finite_stats(
                direct_surface[valid] - derived[valid]),
        }

    report = {
        "run_id": run_id,
        "bag": str(bag.resolve().relative_to(REPO_ROOT)),
        "bag_size_bytes": bag.stat().st_size,
        "bag_sha256": _sha256_file(bag),
        "audit": str(audit_path.resolve().relative_to(REPO_ROOT)),
        "audit_usable_conditions": len(usable),
        "odom_packet_match_fraction": packet_match_fraction,
        "request_clock_offset_ns_ros_minus_monotonic": clock_offset_ns,
        "packet_clock_audit": packet_clock_report,
        "request_clock_timebase": (
            "packet request_monotonic_ns mapped to ROS time using the median "
            "bridge_receive_ros_stamp_ns - bridge_arrival_monotonic_ns offset"),
        "exported_sequences": len(sequence_rows),
        "exported_samples": int(len(data["body_state"])),
        "sequence_duration_s_p01_p50_p99": [
            float(np.quantile([r["duration_s"] for r in sequence_rows], q))
            for q in (0.01, 0.50, 0.99)],
        "encoder_source_stamp_match_fraction": encoder_source_match_fraction,
        "pose_diagnostics": pose_diagnostics,
        "encoder_velocity_diagnostics": encoder_speed_report,
        "sequence_rows": sequence_rows,
    }
    return data, report


def export(captures: list[tuple[Path, Path]], output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data_blocks: dict[str, list[np.ndarray]] = {}
    reports = []
    all_rows = []
    sample_cursor = 0
    seq_cursor = 0
    for index, (bag, audit) in enumerate(captures, start=1):
        print(f"[{index}/{len(captures)}] exporting {bag.parent.parent.name}",
              flush=True)
        data, report = _load_capture(bag, audit)
        data["sequence_index"] = data["sequence_index"] + seq_cursor
        sample_cursor += len(data["body_state"])
        for key, value in data.items():
            data_blocks.setdefault(key, []).append(value)
        rows = report.pop("sequence_rows")
        for row in rows:
            row["start"] += sample_cursor - len(data["body_state"])
            row["end"] += sample_cursor - len(data["body_state"])
            row["sequence_index"] += seq_cursor
            all_rows.append(row)
        seq_cursor += len(rows)
        reports.append(report)
    arrays = {key: np.concatenate(value, axis=0)
              for key, value in data_blocks.items() if key != "sequence_bounds"}
    arrays["sequence_bounds"] = np.asarray(
        [[row["start"], row["end"]] for row in all_rows], dtype=np.int64)
    arrays["plant_command_topic_minus_packet_sent"] = (
        arrays["command_topics"] - arrays["plant_commands"])
    command_delta = arrays["plant_command_topic_minus_packet_sent"]
    debug = arrays["bridge_debug_telemetry"]
    debug_by_name = {name: index for index, name in enumerate(PACKET_DEBUG_NAMES)}
    simulator_velocity = debug[:, [debug_by_name[name] for name in (
        "simulator_velocity_x_mps", "simulator_velocity_y_mps",
        "simulator_angular_z_rps")]]
    odom_velocity = np.column_stack((arrays["body_state"][:, 0],
                                     arrays["body_state"][:, 1]
                                     + COM_X_M * arrays["body_state"][:, 2],
                                     arrays["body_state"][:, 2]))
    packet_alignment_summary = {
        "packet_sent_minus_recorded_command_topic": {
            "steering_rad": _finite_stats(command_delta[:, 0]),
            "throttle_norm": _finite_stats(command_delta[:, 1]),
        },
        "odom_twist_minus_packet_simulator_velocity": {
            "linear_x_mps": _finite_stats(
                odom_velocity[:, 0] - simulator_velocity[:, 0]),
            "linear_y_com_mps": _finite_stats(
                odom_velocity[:, 1] - simulator_velocity[:, 1]),
            "angular_z_rps": _finite_stats(
                odom_velocity[:, 2] - simulator_velocity[:, 2]),
        },
        "ips_minus_packet_simulator_position_xy_m": {
            axis: _finite_stats(arrays["ips_position"][:, i]
                                - debug[:, debug_by_name[f"simulator_position_{axis}_m"]])
            for i, axis in enumerate(("x", "y"))},
        "ros_feedback_minus_same_packet_feedback": {
            "steering_rad": _finite_stats(
                arrays["actuator_feedback"][:, 0]
                - debug[:, debug_by_name["simulator_feedback_steering_rad"]]),
            "throttle_norm": _finite_stats(
                arrays["actuator_feedback"][:, 1]
                - debug[:, debug_by_name["simulator_feedback_throttle_norm"]]),
        },
        "source_aligned_ros_encoder_minus_same_packet_debug_angle_rad": {
            side: _finite_stats(
                arrays["encoder_position_rad"][:, index]
                - debug[:, debug_by_name[
                    f"simulator_encoder_angle_{side}_rad"]])
            for index, side in enumerate(("left", "right"))},
        "feedback_semantics": (
            "Packet V1 Steering numerically matches the ROS steering feedback "
            "in radians; despite its diagnostic key suffix, it is not a "
            "normalized [-1,1] command. Sent steering command remains normalized."),
    }
    np.savez_compressed(output / "throttle_surface_sequences.npz", **arrays)
    manifest = {
        "schema_version": 1,
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "purpose": "full-rate, reset-isolated throttle response sequences",
        "input_policy": ["plant_commands only as plant controls",
                         "no future sensor values used as plant inputs",
                         "IPS retained as an offline position label only"],
        "time_policy": (
            "fixed simulator period dt=0.025 s per consecutive packet; request "
            "and receipt clocks are retained for event alignment and diagnostics "
            "only, never used as plant integration time"),
        "encoder_time_policy": (
            "Encoder angles are joined to /odom by exact source-stamp packet "
            "identity; source stamps identify packets only. The 100 ms surface "
            "speed uses exactly four consecutive packets and dt=0.025 s each. "
            "Receipt jitter and source-stamp deltas are never used as model dt."),
        "simulator_debug_policy": (
            "bridge_packet_timing simulator_* fields are offline diagnostic "
            "labels only, not plant controls or competition runtime inputs"),
        "body_state_names": ["u_rear_mps", "v_rear_mps", "yaw_rate_rps"],
        "odom_pose_names": ["x_m", "y_m", "z_m", "yaw_rad"],
        "plant_command_names": ["steering_command_rad",
                                "throttle_command_norm"],
        "actuator_feedback_names": ["steering_feedback_rad",
                                    "throttle_feedback_norm"],
        "encoder_names": ["left", "right"],
        "encoder_source_stamp_match_fraction": {
            row["run_id"]: row["encoder_source_stamp_match_fraction"]
            for row in reports},
        "imu_acceleration_names": ["ax_mps2", "ay_mps2", "az_mps2"],
        "imu_angular_velocity_names": ["wx_rps", "wy_rps", "wz_rps"],
        "imu_attitude_names": ["roll_rad", "pitch_rad"],
        "bridge_debug_telemetry_names": list(PACKET_DEBUG_NAMES),
        "packet_alignment_summary": packet_alignment_summary,
        "alignment_age_names_ms": list(STREAM_NAMES),
        "state_convention": ("bridge odometry twist, rear-axle lateral velocity "
                             "using project COM_X_M correction; not yet simulator truth"),
        "ips_semantics": ("nearest receipt-time point, <=30 ms; offline label only; "
                          "offset reported but not applied"),
        "sequence_count": len(all_rows),
        "sample_count": sample_cursor,
        "captures": reports,
        "sequences": all_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    summary = {
        "sequence_count": len(all_rows),
        "sample_count": sample_cursor,
        "run_count": len(captures),
        "capture_summaries": [
            {k: v for k, v in item.items() if k != "bag_sha256"}
            for item in reports],
        "condition_key_counts": {},
    }
    condition_counts: dict[str, int] = {}
    for row in all_rows:
        key = json.dumps(row["condition_key"], separators=(",", ":"))
        condition_counts[key] = condition_counts.get(key, 0) + 1
    summary["condition_key_counts"] = {
        "unique_conditions": len(condition_counts),
        "replicate_count_histogram": {
            str(count): sum(value == count for value in condition_counts.values())
            for count in sorted(set(condition_counts.values()))},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", nargs=2, action="append", metavar=("BAG", "AUDIT"),
                        required=True,
                        help="closed rosbag DB3 and its throttle-transition audit JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = [(Path(bag).resolve(), Path(audit).resolve())
             for bag, audit in args.capture]
    summary = export(pairs, args.output.resolve())
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

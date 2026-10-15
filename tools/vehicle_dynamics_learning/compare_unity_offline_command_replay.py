#!/usr/bin/env python3
"""Compare a 1 kHz source-Unity replay with its recorded 40 Hz command bag.

This is a runtime parity check, not a learned-model fit.  It requires the
captured Unity commands to match the recorded packet commands, then reports
state errors over plausible sub-packet sample offsets so the bridge/physics
observation phase is visible rather than silently tuned away.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np


PACKET_TOPIC = "/autodrive/roboracer_1/bridge_packet_timing"
LAP_COUNT_TOPIC = "/autodrive/roboracer_1/lap_count"
BASE_FIELDS = (
    "sim_time_s",
    "pos_world_x_m", "pos_world_y_m", "pos_world_z_m",
    "rot_world_x", "rot_world_y", "rot_world_z", "rot_world_w",
    "vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
    "omega_body_x_rps", "omega_body_y_rps", "omega_body_z_rps",
    "throttle_command", "steering_command",
)
WHEEL_FIELDS = tuple(
    f"{wheel}_{field}"
    for wheel in ("fl", "fr", "rl", "rr")
    for field in ("rpm", "steer_deg", "motor_torque_nm", "grounded",
                  "forward_slip", "sideways_slip", "contact_force_n")
)
# Serialized at Rear Axle Center.Transform.m_LocalPosition in the source
# F1TENTH prefab; its local rotation is identity and its scene instance has no
# child-transform override. This reconstructs the GPS.cs VehicleTransform
# origin from the captured F1TENTH Rigidbody transform, without fitting.
REAR_AXLE_CENTER_LOCAL_UNITY_M = np.asarray([0.0, 0.05, -0.16])
SCENE_MOTOR_TORQUE_NM = 428.0
STEERING_WHEELBASE_M = 0.324
STEERING_TRACK_WIDTH_M = 0.236


def read_packet_rows(bag_path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(bag_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topic = connection.execute(
            "SELECT id FROM topics WHERE name = ?", (PACKET_TOPIC,)
        ).fetchone()
        if topic is None:
            raise ValueError(f"bag has no {PACKET_TOPIC}")
        rows: list[dict[str, Any]] = []
        for (serialized,) in connection.execute(
                "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp, id",
                (int(topic[0]),)):
            raw = bytes(serialized)
            if len(raw) < 8:
                raise ValueError("invalid CDR packet-timing message")
            size = int.from_bytes(raw[4:8], "little", signed=False)
            if size < 2 or 8 + size > len(raw):
                raise ValueError("invalid CDR string length in packet-timing message")
            rows.append(json.loads(raw[8:8 + size].rstrip(b"\0")))
        rows.sort(key=lambda row: int(row["request_sequence"]))
        return rows
    finally:
        connection.close()


def read_lap_boundaries(bag_path: Path,
                        packets: list[dict[str, Any]]) -> list[dict[str, int]]:
    connection = sqlite3.connect(bag_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topic = connection.execute(
            "SELECT id FROM topics WHERE name = ?", (LAP_COUNT_TOPIC,)
        ).fetchone()
        if topic is None:
            return []
        packet_stamps = np.asarray([
            int(row["bridge_receive_ros_stamp_ns"]) for row in packets
        ], dtype=np.int64)
        events: list[dict[str, int]] = []
        previous_count: int | None = None
        for stamp_ns, serialized in connection.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp, id",
                (int(topic[0]),)):
            raw = bytes(serialized)
            if len(raw) < 8:
                raise ValueError("invalid CDR lap-count message")
            lap_count = int.from_bytes(raw[4:8], "little", signed=True)
            if previous_count is not None and lap_count > previous_count:
                insertion = int(np.searchsorted(packet_stamps, int(stamp_ns)))
                candidates = [index for index in (insertion - 1, insertion)
                              if 0 <= index < len(packet_stamps)]
                packet_index = min(candidates,
                                   key=lambda index: abs(int(packet_stamps[index]) -
                                                         int(stamp_ns)))
                events.append({"lap_count": lap_count,
                               "packet_index": packet_index})
            previous_count = lap_count
        unique = {event["lap_count"]: event for event in events}
        return [unique[key] for key in sorted(unique)]
    finally:
        connection.close()


def read_capture(path: Path) -> dict[str, np.ndarray]:
    names = BASE_FIELDS + WHEEL_FIELDS
    with path.open("r", newline="", encoding="utf-8") as stream:
        header_line = next((line for line in stream if not line.startswith("#")), None)
        if header_line is None:
            raise ValueError(f"capture has no CSV header: {path}")
        reader = csv.reader(chain((header_line,), stream))
        header = next(reader)
        index = {name: header.index(name) for name in names}
        columns = {name: [] for name in names}
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"capture row has {len(row)} fields; expected {len(header)}")
            for name in names:
                columns[name].append(float(row[index[name]]))
    result = {name: np.asarray(values, dtype=np.float64)
              for name, values in columns.items()}
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("capture contains non-finite state or wheel values")
    return result


def metric(error: np.ndarray) -> dict[str, Any]:
    magnitudes = np.linalg.norm(error, axis=1) if error.ndim > 1 else np.abs(error)
    return {
        "rmse_by_axis": (np.sqrt(np.mean(error ** 2, axis=0)).tolist()
                         if error.ndim > 1 else float(np.sqrt(np.mean(error ** 2)))),
        "vector_rmse": float(np.sqrt(np.mean(magnitudes ** 2))),
        "p95": float(np.quantile(magnitudes, 0.95)),
        "max": float(np.max(magnitudes)),
    }


def source_state_sample_gaps(packets: list[dict[str, Any]],
                             nominal_packet_period_s: float = 0.025
                             ) -> list[dict[str, Any]]:
    """Find delayed responses that also span multiple physical sample periods.

    The competition packet has no Unity physics-step counter.  A long host
    receive interval alone is therefore insufficient to call a physics sample
    missing.  Require the request cadence to remain nominal and use the
    measured position increment divided by mean body speed as an independent
    kinematic check that the Unity state itself advanced by multiple periods.
    This is an evaluation-time alignment diagnostic only; it is not a plant
    input or learned dynamics term.
    """
    if len(packets) < 2:
        return []
    positions = np.asarray([
        [row[f"simulator_position_{axis}"] for axis in "xy"]
        for row in packets
    ], dtype=np.float64)
    velocities = np.asarray([
        [row[f"simulator_linear_velocity_{axis}"] for axis in "xy"]
        for row in packets
    ], dtype=np.float64)
    receive_ns = np.asarray(
        [int(row["bridge_receive_ros_stamp_ns"]) for row in packets],
        dtype=np.int64)
    request_ns = np.asarray(
        [int(row["bridge_request_ros_stamp_ns"]) for row in packets],
        dtype=np.int64)
    receive_dt = np.diff(receive_ns) / 1.0e9
    request_dt = np.diff(request_ns) / 1.0e9
    distance = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    mean_speed = 0.5 * (
        np.linalg.norm(velocities[:-1], axis=1)
        + np.linalg.norm(velocities[1:], axis=1))
    implied_state_dt = np.divide(
        distance, mean_speed,
        out=np.full_like(distance, np.nan), where=mean_speed > 0.5)

    gaps = []
    for index, (recv_s, req_s, state_s) in enumerate(
            zip(receive_dt, request_dt, implied_state_dt)):
        request_is_nominal = abs(req_s - nominal_packet_period_s) <= 0.004
        receive_is_multiple = recv_s >= 1.6 * nominal_packet_period_s
        state_is_same_multiple = (
            np.isfinite(state_s)
            and state_s >= 1.6 * nominal_packet_period_s
            and abs(state_s - recv_s) <= 0.012)
        if request_is_nominal and receive_is_multiple and state_is_same_multiple:
            intervals = max(2, int(round(state_s / nominal_packet_period_s)))
            gaps.append({
                "after_packet_index": index,
                "before_packet_index": index + 1,
                "request_interval_ms": float(req_s * 1000.0),
                "host_receive_interval_ms": float(recv_s * 1000.0),
                "motion_implied_state_interval_ms": float(state_s * 1000.0),
                "estimated_missing_25ms_state_samples": intervals - 1,
                "position_increment_m": float(distance[index]),
                "mean_body_speed_mps": float(mean_speed[index]),
            })
    return gaps


def timing_aware_state_comparison(
        packets: list[dict[str, Any]],
        profile: dict[str, Any],
        capture: dict[str, np.ndarray],
        lap_boundaries: list[dict[str, int]],
        gaps: list[dict[str, Any]],
        initial_offset: int,
        ) -> dict[str, Any]:
    """Compare on a source-state sample clock, accounting for verified gaps.

    The initial phase is an explicit diagnostic input, selected from the
    separate phase sweep and never fitted to the metrics reported here.
    Thereafter, advance the sample phase only for intervals that pass
    ``source_state_sample_gaps``.  This alignment is for evaluating recorded
    output only; it never changes the command profile or plant.
    """
    offline_state = unity_capture_to_api_state(capture)
    offline_actuators = unity_capture_to_api_actuators(capture)
    n = len(packets)
    ticks_per_command = int(profile["physics_ticks_per_command"])
    source_throttle = np.asarray(
        [float(row["simulator_feedback_throttle_norm"]) for row in packets])
    source_steering = np.asarray(
        [float(row["simulator_feedback_steering_norm"]) for row in packets])

    gap_by_index = {
        int(gap["after_packet_index"]): int(
            gap["estimated_missing_25ms_state_samples"])
        for gap in gaps
    }
    capture_offsets = np.empty(n, dtype=np.int64)
    accumulated_missing = 0
    for packet_index in range(n):
        if packet_index > 0:
            accumulated_missing += gap_by_index.get(packet_index - 1, 0)
        capture_offsets[packet_index] = (
            initial_offset + accumulated_missing * ticks_per_command)

    source_position = np.asarray([
        [float(row[f"simulator_position_{axis}"]) for axis in "xyz"]
        for row in packets], dtype=np.float64)
    source_yaw = np.asarray([
        float(row["simulator_orientation_euler_z"]) for row in packets],
        dtype=np.float64)
    source_velocity = np.asarray([
        [float(row[f"simulator_linear_velocity_{axis}"]) for axis in "xyz"]
        for row in packets], dtype=np.float64)
    source_angular = np.asarray([
        [float(row[f"simulator_angular_velocity_{axis}"]) for axis in "xyz"]
        for row in packets
    ], dtype=np.float64)
    indices = ticks_per_command * np.arange(n) + capture_offsets
    valid = (indices >= 0) & (indices < len(capture["sim_time_s"]))
    packet_ids = np.flatnonzero(valid)
    sample_ids = indices[valid]
    if len(packet_ids) < 2:
        raise ValueError("timing-aware comparison has fewer than two samples")

    offline_position = offline_state["position"][sample_ids]
    offline_yaw = offline_state["yaw"][sample_ids]
    source_position = source_position[packet_ids]
    source_yaw = source_yaw[packet_ids]
    heading_offset = float(np.arctan2(
        np.sin(offline_yaw[0] - source_yaw[0]),
        np.cos(offline_yaw[0] - source_yaw[0])))
    rotation = np.asarray([
        [np.cos(heading_offset), -np.sin(heading_offset)],
        [np.sin(heading_offset), np.cos(heading_offset)],
    ])
    offline_relative = offline_position - offline_position[0]
    source_relative = source_position - source_position[0]
    position_error = np.column_stack((
        offline_relative[:, :2] - source_relative[:, :2] @ rotation.T,
        offline_relative[:, 2] - source_relative[:, 2],
    ))
    yaw_delta = (offline_yaw - offline_yaw[0]) - (source_yaw - source_yaw[0])
    yaw_error = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    velocity_error = (offline_state["velocity"][sample_ids]
                      - source_velocity[packet_ids])
    angular_error = (offline_state["angular_velocity"][sample_ids]
                     - source_angular[packet_ids])
    throttle_error = (offline_actuators["throttle_feedback"][sample_ids]
                      - source_throttle[packet_ids])
    steering_error = (offline_actuators["steering_feedback"][sample_ids]
                      - source_steering[packet_ids])

    per_lap = []
    previous_packet = 0
    for boundary in lap_boundaries:
        stop_packet = int(boundary["packet_index"]) + 1
        local = np.flatnonzero(
            (packet_ids >= previous_packet) & (packet_ids < stop_packet))
        if len(local) > 1:
            lap_heading_offset = float(np.arctan2(
                np.sin(offline_yaw[local[0]] - source_yaw[local[0]]),
                np.cos(offline_yaw[local[0]] - source_yaw[local[0]])))
            lap_rotation = np.asarray([
                [np.cos(lap_heading_offset), -np.sin(lap_heading_offset)],
                [np.sin(lap_heading_offset), np.cos(lap_heading_offset)],
            ])
            offline_lap_delta = (
                offline_position[local] - offline_position[local[0]])
            source_lap_delta = (
                source_position[local] - source_position[local[0]])
            local_position = np.column_stack((
                offline_lap_delta[:, :2]
                - source_lap_delta[:, :2] @ lap_rotation.T,
                offline_lap_delta[:, 2] - source_lap_delta[:, 2],
            ))
            local_yaw = yaw_error[local] - yaw_error[local[0]]
            local_yaw = np.arctan2(np.sin(local_yaw), np.cos(local_yaw))
            per_lap.append({
                "lap_count_at_end": int(boundary["lap_count"]),
                "sample_count": int(len(local)),
                "local_path_increment_rmse_m": metric(local_position),
                "lap_end_increment_error_m": float(
                    np.linalg.norm(local_position[-1])),
                "local_yaw_increment_rmse_rad": metric(local_yaw),
                "lap_end_yaw_increment_error_rad": float(local_yaw[-1]),
            })
        previous_packet = stop_packet

    return {
        "phase_source": "explicit initial offset from the separate capture-phase sweep; diagnostic alignment only, no state-model parameters are fitted",
        "initial_capture_offset_fixed_steps": initial_offset,
        "missing_sample_phase_corrections": gaps,
        "state_sample_count": int(len(packet_ids)),
        "capture_offset_range_fixed_steps": [
            int(np.min(capture_offsets[packet_ids])),
            int(np.max(capture_offsets[packet_ids]))],
        "relative_position_m": metric(position_error),
        "endpoint_relative_position_error_m": float(
            np.linalg.norm(position_error[-1])),
        "yaw_rad": metric(yaw_error),
        "body_linear_velocity_mps": metric(velocity_error),
        "body_angular_velocity_rps": metric(angular_error),
        "actual_throttle_feedback": metric(throttle_error),
        "actual_steering_feedback_rad": metric(steering_error),
        "lap_local_increments": per_lap,
    }


def unity_capture_to_api_state(capture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # GPS.cs / IMU.cs coordinate mapping: Unity x-right,y-up,z-forward to
    # simulator API x-forward,y-left,z-up; IMU velocities are body-local.
    qx, qy, qz, qw = (capture[name] for name in
                      ("rot_world_x", "rot_world_y", "rot_world_z", "rot_world_w"))
    qv = np.column_stack((qx, qy, qz))
    local_offset = np.broadcast_to(REAR_AXLE_CENTER_LOCAL_UNITY_M, qv.shape)
    twice_cross = 2.0 * np.cross(qv, local_offset)
    rotated_offset = (local_offset + qw[:, None] * twice_cross
                      + np.cross(qv, twice_cross))
    rear_axle_world = np.column_stack((capture["pos_world_x_m"],
                                       capture["pos_world_y_m"],
                                       capture["pos_world_z_m"])) + rotated_offset
    position = np.column_stack((rear_axle_world[:, 2],
                                -rear_axle_world[:, 0],
                                rear_axle_world[:, 1]))
    velocity = np.column_stack((capture["vel_body_z_mps"],
                                -capture["vel_body_x_mps"],
                                capture["vel_body_y_mps"]))
    angular = np.column_stack((-capture["omega_body_z_rps"],
                               capture["omega_body_x_rps"],
                               -capture["omega_body_y_rps"]))
    # Unity Y-axis yaw, sign-mapped by IMU.cs to API yaw around ROS z.
    yaw = -np.arctan2(2.0 * (qw * qy + qx * qz),
                      1.0 - 2.0 * (qy * qy + qz * qz))
    return {"position": position, "velocity": velocity,
            "angular_velocity": angular, "yaw": yaw}


def unity_capture_to_api_actuators(capture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Reconstruct the source controller's reported actual actuator values from
    # WheelCollider outputs; constants are the Explore scene serialized values.
    motor_torque = sum(capture[f"{wheel}_motor_torque_nm"]
                       for wheel in ("fl", "fr", "rl", "rr"))
    throttle = motor_torque / SCENE_MOTOR_TORQUE_NM
    wheel_angle = np.deg2rad(capture["fl_steer_deg"])
    tangent = np.tan(wheel_angle)
    center_steering = -np.arctan(
        (2.0 * STEERING_WHEELBASE_M * tangent) /
        (2.0 * STEERING_WHEELBASE_M - STEERING_TRACK_WIDTH_M * tangent))
    return {"throttle_feedback": throttle,
            "steering_feedback": center_steering}


def compare(bag_path: Path, profile_path: Path, capture_path: Path,
            state_capture_initial_offset: int | None = None
            ) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    commands = profile["commands"]
    packets = read_packet_rows(bag_path)
    capture = read_capture(capture_path)
    ticks_per_command = int(profile["physics_ticks_per_command"])
    if int(profile["physics_ticks_per_command"]) != 25:
        raise ValueError("comparison expects 25 fixed physics ticks per command")
    command_phase_steps = int(profile.get("command_phase_fixed_steps", 0))
    if command_phase_steps not in (0, 1):
        raise ValueError("command phase must be 0 or 1 fixed step")
    expected_steps = len(commands) * ticks_per_command + command_phase_steps
    if len(capture["sim_time_s"]) != expected_steps:
        raise ValueError(f"capture has {len(capture['sim_time_s'])} steps; expected {expected_steps}")
    requests = [int(row["request_sequence"]) for row in packets]
    expected_requests = list(range(int(profile["source_first_request_sequence"]),
                                   int(profile["source_last_request_sequence"]) + 1))
    if requests != expected_requests or len(packets) != len(commands):
        raise ValueError("closed bag packet sequence differs from the command profile")

    packet_steering = np.asarray([float(row["sent_steering_norm"]) for row in packets])
    packet_throttle = np.asarray([float(row["sent_throttle_norm"]) for row in packets])
    prof_steering = np.asarray([float(row["steering_norm"]) for row in commands])
    prof_throttle = np.asarray([float(row["throttle_norm"]) for row in commands])
    if not np.allclose(packet_steering, prof_steering, rtol=0.0, atol=1e-7):
        raise ValueError("profile steering differs from recorded sent commands")
    if not np.allclose(packet_throttle, prof_throttle, rtol=0.0, atol=1e-7):
        raise ValueError("profile throttle differs from recorded sent commands")

    held_steering = np.concatenate((
        np.repeat(prof_steering[:1], ticks_per_command + command_phase_steps),
        np.repeat(prof_steering[1:], ticks_per_command)))
    held_throttle = np.concatenate((
        np.repeat(prof_throttle[:1], ticks_per_command + command_phase_steps),
        np.repeat(prof_throttle[1:], ticks_per_command)))
    steering_error = capture["steering_command"] - held_steering
    throttle_error = capture["throttle_command"] - held_throttle
    sim_dt = np.diff(capture["sim_time_s"])

    source_state = {
        "position": np.asarray([[row[f"simulator_position_{axis}"] for axis in "xyz"]
                                 for row in packets], dtype=np.float64),
        "yaw": np.asarray([row["simulator_orientation_euler_z"] for row in packets],
                          dtype=np.float64),
        "velocity": np.asarray([[row[f"simulator_linear_velocity_{axis}"] for axis in "xyz"]
                                 for row in packets], dtype=np.float64),
        "angular_velocity": np.asarray([[row[f"simulator_angular_velocity_{axis}"]
                                         for axis in "xyz"] for row in packets],
                                        dtype=np.float64),
    }
    source_feedback = {
        "throttle": np.asarray([row["simulator_feedback_throttle_norm"]
                                for row in packets], dtype=np.float64),
        "steering": np.asarray([row["simulator_feedback_steering_norm"]
                                for row in packets], dtype=np.float64),
    }
    lap_boundaries = read_lap_boundaries(bag_path, packets)
    state_sample_gaps = source_state_sample_gaps(packets)
    timing_aware_comparison = (
        None if state_capture_initial_offset is None else
        timing_aware_state_comparison(
            packets, profile, capture, lap_boundaries, state_sample_gaps,
            state_capture_initial_offset))
    offline_state = unity_capture_to_api_state(capture)
    offline_actuators = unity_capture_to_api_actuators(capture)
    offsets = []
    packet_count = len(packets)
    for offset in range(-25, 26):
        indices = 25 * np.arange(packet_count, dtype=np.int64) + offset
        valid = (indices >= 0) & (indices < expected_steps)
        ids = np.flatnonzero(valid)
        selected = indices[valid]
        op = offline_state["position"][selected]
        sp = source_state["position"][ids]
        ov = offline_state["velocity"][selected]
        sv = source_state["velocity"][ids]
        ow = offline_state["angular_velocity"][selected]
        sw = source_state["angular_velocity"][ids]
        oyaw = offline_state["yaw"][selected]
        syaw = source_state["yaw"][ids]
        heading_frame_offset = float(np.arctan2(
            np.sin(oyaw[0] - syaw[0]), np.cos(oyaw[0] - syaw[0])))
        rotation = np.asarray([
            [np.cos(heading_frame_offset), -np.sin(heading_frame_offset)],
            [np.sin(heading_frame_offset), np.cos(heading_frame_offset)],
        ])
        offline_relative = op - op[0]
        source_relative = sp - sp[0]
        source_xy_aligned = source_relative[:, :2] @ rotation.T
        position_error = np.column_stack((
            offline_relative[:, :2] - source_xy_aligned,
            offline_relative[:, 2] - source_relative[:, 2],
        ))
        yaw_delta = (oyaw - oyaw[0]) - (syaw - syaw[0])
        yaw_error = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
        throttle_feedback_error = (offline_actuators["throttle_feedback"][selected]
                                   - source_feedback["throttle"][ids])
        steering_feedback_error = (offline_actuators["steering_feedback"][selected]
                                   - source_feedback["steering"][ids])
        offset_row = {
            "capture_offset_fixed_steps": offset,
            "sample_count": int(len(ids)),
            "relative_position_m": metric(position_error),
            "yaw_rad": metric(yaw_error),
            "body_linear_velocity_mps": metric(ov - sv),
            "body_angular_velocity_rps": metric(ow - sw),
            "actual_throttle_feedback": metric(throttle_feedback_error),
            "actual_steering_feedback_rad": metric(steering_feedback_error),
            "unregistered_world_position_delta_m": (op[0] - sp[0]).tolist(),
            "initial_heading_frame_offset_rad": float(heading_frame_offset),
            "endpoint_relative_position_delta_m": position_error[-1].tolist(),
            "endpoint_relative_position_error_m": float(
                np.linalg.norm(position_error[-1])),
        }
        if offset in (-1, 0, 1):
            windows = []
            for start_s in range(0, 270, 30):
                start = max(0, int(round(start_s * 40.0)))
                stop = min(len(ids), int(round((start_s + 30) * 40.0)))
                if stop - start < 2:
                    continue
                windows.append({
                    "start_s": start_s,
                    "stop_s": min(start_s + 30, 0.025 * int(ids[-1]) + 0.025),
                    "relative_position_m": metric(position_error[start:stop]),
                    "yaw_rad": metric(yaw_error[start:stop]),
                    "body_linear_velocity_mps": metric((ov - sv)[start:stop]),
                    "body_angular_velocity_rps": metric((ow - sw)[start:stop]),
                })
            offset_row["run_windows_30s"] = windows
            lap_metrics = []
            previous_index = 0
            for boundary in lap_boundaries:
                stop_index = boundary["packet_index"] + 1
                local = np.flatnonzero((ids >= previous_index) & (ids < stop_index))
                if len(local) > 1:
                    lap_metrics.append({
                        "lap_count_at_end": boundary["lap_count"],
                        "start_packet_index": previous_index,
                        "end_packet_index": boundary["packet_index"],
                        "source_elapsed_s": float((stop_index - previous_index) * 0.025),
                        "relative_position_m": metric(position_error[local]),
                        "end_relative_position_error_m": float(
                            np.linalg.norm(position_error[local[-1]])),
                        "yaw_change_rad": metric(yaw_error[local]),
                        "body_linear_velocity_mps": metric((ov - sv)[local]),
                        "body_angular_velocity_rps": metric((ow - sw)[local]),
                        "actual_throttle_feedback": metric(throttle_feedback_error[local]),
                        "actual_steering_feedback_rad": metric(steering_feedback_error[local]),
                    })
                previous_index = stop_index
            offset_row["lap_metrics"] = lap_metrics
        offsets.append(offset_row)

    wheel_summary = {}
    for wheel in ("fl", "fr", "rl", "rr"):
        wheel_summary[wheel] = {
            "grounded_fraction": float(np.mean(capture[f"{wheel}_grounded"] > 0.5)),
            "rpm_abs_p50_p95_p99": np.quantile(
                np.abs(capture[f"{wheel}_rpm"]), [0.5, 0.95, 0.99]).tolist(),
            "forward_slip_abs_p50_p95_p99": np.quantile(
                np.abs(capture[f"{wheel}_forward_slip"]), [0.5, 0.95, 0.99]).tolist(),
            "sideways_slip_abs_p50_p95_p99": np.quantile(
                np.abs(capture[f"{wheel}_sideways_slip"]), [0.5, 0.95, 0.99]).tolist(),
            "contact_force_n_p50_p95_p99": np.quantile(
                capture[f"{wheel}_contact_force_n"], [0.5, 0.95, 0.99]).tolist(),
        }

    return {
        "schema_version": 1,
        "reference_bag": str(bag_path),
        "command_profile": str(profile_path),
        "unity_capture": str(capture_path),
        "unity_fixed_delta_time_s": float(profile["fixed_delta_time_s"]),
        "physics_ticks_per_command": 25,
        "command_phase_fixed_steps": command_phase_steps,
        "packet_count": packet_count,
        "lap_count_boundaries": lap_boundaries,
        "capture_fixed_steps": len(capture["sim_time_s"]),
        "capture_sim_duration_s": float(capture["sim_time_s"][-1] - capture["sim_time_s"][0]),
        "position_reference": {
            "frame": "GPS.cs VehicleTransform / Rear Axle Center",
            "method": "Rigid transform of captured F1TENTH root pose using the serialized prefab child offset; not fitted.",
            "rear_axle_center_local_position_unity_m": REAR_AXLE_CENTER_LOCAL_UNITY_M.tolist(),
        },
        "trajectory_alignment": (
            "Subtract each run's initial GPS position and rotate the recorded "
            "source XY trajectory by the initial heading difference into the "
            "offline run's heading frame; no data-fitted transform."),
        "capture_step_dt_s_p50_p95_min_max": np.quantile(
            sim_dt, [0.5, 0.95]).tolist() + [float(np.min(sim_dt)), float(np.max(sim_dt))],
        "command_match": {
            "packet_sequence_exact": True,
            "unity_steering_max_abs_error": float(np.max(np.abs(steering_error))),
            "unity_throttle_max_abs_error": float(np.max(np.abs(throttle_error))),
            "unity_steering_rmse": float(np.sqrt(np.mean(steering_error ** 2))),
            "unity_throttle_rmse": float(np.sqrt(np.mean(throttle_error ** 2))),
        },
        "source_state_sample_gaps": state_sample_gaps,
        "timing_aware_state_comparison": timing_aware_comparison,
        "offset_sweep": offsets,
        "unity_wheel_dynamics_summary": wheel_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("closed_bag", type=Path)
    parser.add_argument("command_profile", type=Path)
    parser.add_argument("unity_capture_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--state-capture-initial-offset", type=int,
        help=("optional explicit 1 kHz sample phase for a timing diagnostic; "
              "does not change replay inputs or plant state"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = compare(args.closed_bag, args.command_profile,
                     args.unity_capture_csv,
                     args.state_capture_initial_offset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    offset_rows = result["offset_sweep"]
    key_offsets = []
    for row in offset_rows:
        if row["capture_offset_fixed_steps"] not in (-1, 0, 1):
            continue
        key_offsets.append({
            "capture_offset_fixed_steps": row["capture_offset_fixed_steps"],
            "relative_position_rmse_m": row["relative_position_m"]["vector_rmse"],
            "relative_position_p95_m": row["relative_position_m"]["p95"],
            "endpoint_position_error_m": row["endpoint_relative_position_error_m"],
            "yaw_rmse_rad": row["yaw_rad"]["vector_rmse"],
            "yaw_p95_rad": row["yaw_rad"]["p95"],
            "body_speed_rmse_mps": row["body_linear_velocity_mps"]["vector_rmse"],
            "body_angular_velocity_rmse_rps": row["body_angular_velocity_rps"]["vector_rmse"],
            "actual_throttle_feedback_rmse": row["actual_throttle_feedback"]["vector_rmse"],
            "actual_steering_feedback_rmse_rad": row["actual_steering_feedback_rad"]["vector_rmse"],
        })
    nominal = next((row for row in offset_rows
                    if row["capture_offset_fixed_steps"] == 0), None)
    timing_comparison = result["timing_aware_state_comparison"]
    timing_summary = None
    if timing_comparison is not None:
        timing_summary = {
            key: timing_comparison[key]
            for key in (
                "phase_source", "initial_capture_offset_fixed_steps",
                "missing_sample_phase_corrections", "state_sample_count",
                "capture_offset_range_fixed_steps", "relative_position_m",
                "endpoint_relative_position_error_m", "yaw_rad",
                "body_linear_velocity_mps", "body_angular_velocity_rps",
                "actual_throttle_feedback", "actual_steering_feedback_rad")
        }
        timing_summary["lap_local_increments"] = [{
            "lap": lap["lap_count_at_end"],
            "local_path_rmse_m": lap[
                "local_path_increment_rmse_m"]["vector_rmse"],
            "lap_end_increment_error_m": lap[
                "lap_end_increment_error_m"],
            "local_yaw_rmse_rad": lap[
                "local_yaw_increment_rmse_rad"]["vector_rmse"],
            "lap_end_yaw_increment_error_rad": lap[
                "lap_end_yaw_increment_error_rad"],
        } for lap in timing_comparison["lap_local_increments"]]
    lap_summary = []
    if nominal is not None:
        lap_summary = [{
            "lap": lap["lap_count_at_end"],
            "source_elapsed_s": lap["source_elapsed_s"],
            "relative_position_rmse_m": lap["relative_position_m"]["vector_rmse"],
            "position_error_at_lap_end_m": lap["end_relative_position_error_m"],
            "yaw_change_rmse_rad": lap["yaw_change_rad"]["vector_rmse"],
            "body_speed_rmse_mps": lap["body_linear_velocity_mps"]["vector_rmse"],
            "body_yaw_rate_rmse_rps": lap["body_angular_velocity_rps"]["rmse_by_axis"][2],
            "actual_throttle_feedback_rmse": lap["actual_throttle_feedback"]["vector_rmse"],
            "actual_steering_feedback_rmse_rad": lap["actual_steering_feedback_rad"]["vector_rmse"],
        } for lap in nominal.get("lap_metrics", [])]
    print(json.dumps({
        "output": str(args.output),
        "command_match": result["command_match"],
        "capture_fixed_steps": result["capture_fixed_steps"],
        "position_reference": result["position_reference"],
        "source_state_sample_gaps": result["source_state_sample_gaps"],
        "timing_aware_state_comparison": timing_summary,
        "selected_offsets": key_offsets,
        "nominal_phase_lap_metrics": lap_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

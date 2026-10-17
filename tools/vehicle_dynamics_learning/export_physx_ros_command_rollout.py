#!/usr/bin/env python3
"""Export a reset-initialized PhysX rollout from recorded ROS actuator data.

The source capture supplies the run clock and scoring labels only. Inputs are
sampled from permitted ROS bag topics using zero-order hold. Command mode
reconstructs Unity's controller equations; feedback mode uses the simulator's
reported physical steering angle and effective throttle directly. The
command/capture clock offset is a measured run-pairing input, not a model
parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tools.analyze_open_plane_dynamics import _topic_map
from tools.vehicle_dynamics_learning import evaluate_physx_no_drive_plant_one_step as replay
from tools.vehicle_dynamics_learning.export_physx_reset_initialized_rollout import (
    DEFAULT_CAPTURE,
    DEFAULT_SHAPES,
    export_reset_rollout,
)
from tools.vehicle_dynamics_learning.compare_unity_offline_command_replay import (
    PACKET_TOPIC,
    read_packet_rows,
)
from tools.vehicle_dynamics_learning.unity_vehicle_controller_source import (
    front_ackermann_angles_deg,
    step_autonomous_steering_deg,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import wheel_actuation


ROOT = Path(__file__).resolve().parents[2]
STEERING_TOPIC = "/autodrive/roboracer_1/steering_command"
THROTTLE_TOPIC = "/autodrive/roboracer_1/throttle_command"
STEERING_FEEDBACK_TOPIC = "/autodrive/roboracer_1/steering"
THROTTLE_FEEDBACK_TOPIC = "/autodrive/roboracer_1/throttle"


def _load_float_streams(
        bag: Path, topic_names: tuple[str, str]
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    db = bag if bag.is_file() else next(bag.glob("*.db3"))
    connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = _topic_map(connection)
        missing = [name for name in topic_names if name not in topics]
        if missing:
            raise ValueError(f"bag lacks actuator input topic(s): {missing}")
        bag_start_ns = int(connection.execute(
            "SELECT MIN(timestamp) FROM messages").fetchone()[0])
        streams = []
        for name in topic_names:
            topic_id, type_name = topics[name]
            message_type = get_message(type_name)
            times: list[float] = []
            values: list[float] = []
            for timestamp, payload in connection.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id=? "
                    "ORDER BY timestamp, id", (topic_id,)):
                message = deserialize_message(bytes(payload), message_type)
                times.append((int(timestamp) - bag_start_ns) * 1.0e-9)
                values.append(float(message.data))
            if len(times) < 2 or np.any(np.diff(times) <= 0.0):
                raise ValueError(f"invalid or insufficient {name} messages")
            streams.append((np.asarray(times), np.asarray(values)))
        return streams[0][0], streams[0][1], streams[1][0], streams[1][1]
    finally:
        connection.close()


def _load_command_streams(
        bag: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _load_float_streams(bag, (STEERING_TOPIC, THROTTLE_TOPIC))


def _load_feedback_streams(
        bag: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _load_float_streams(
        bag, (STEERING_FEEDBACK_TOPIC, THROTTLE_FEEDBACK_TOPIC))


def _load_packet_commands(
        bag: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    database = bag if bag.is_file() else next(bag.glob("*.db3"))
    packets = read_packet_rows(database)
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        bag_start_ns = int(connection.execute(
            "SELECT MIN(timestamp) FROM messages").fetchone()[0])
    finally:
        connection.close()
    packets = [row for row in packets
               if isinstance(row.get("bridge_request_ros_stamp_ns"), int)
               and isinstance(row.get("sent_steering_norm"), (int, float))
               and isinstance(row.get("sent_throttle_norm"), (int, float))]
    packets.sort(key=lambda row: int(row["bridge_request_ros_stamp_ns"]))
    times = np.asarray([
        (int(row["bridge_request_ros_stamp_ns"]) - bag_start_ns) * 1.0e-9
        for row in packets], dtype=np.float64)
    steering = np.asarray(
        [float(row["sent_steering_norm"]) for row in packets], dtype=np.float64)
    throttle = np.asarray(
        [float(row["sent_throttle_norm"]) for row in packets], dtype=np.float64)
    if (len(times) < 2 or np.any(np.diff(times) <= 0.0)
            or not np.all(np.isfinite(steering))
            or not np.all(np.isfinite(throttle))):
        raise ValueError(f"invalid {PACKET_TOPIC} command timeline")
    return times, steering, throttle


def export(capture: Path, bag: Path, count: int, start_row: int,
           source_realtime_minus_bag_s: float,
           shape_audit: Path = DEFAULT_SHAPES,
           input_source: str = "command",
           feedback_realtime_minus_bag_s: float | None = None) -> dict:
    if not np.isfinite(source_realtime_minus_bag_s):
        raise ValueError("capture-to-bag clock offset must be finite")
    if (feedback_realtime_minus_bag_s is not None
            and not np.isfinite(feedback_realtime_minus_bag_s)):
        raise ValueError("feedback clock offset must be finite")
    if input_source not in {
            "command", "feedback", "feedback_corrected", "packet_aligned",
            "packet_steer_feedback_throttle"}:
        raise ValueError(
            "input_source must be command, feedback, feedback_corrected, "
            "packet_aligned, or packet_steer_feedback_throttle")
    data = export_reset_rollout(capture, count, shape_audit, start_row=start_row)
    frame = pd.read_csv(capture, comment="#", usecols=["capture_realtime_s"],
                        nrows=start_row + count)
    capture_realtime = frame.capture_realtime_s.to_numpy(dtype=np.float64)[
        start_row:start_row + count]
    bag_times = capture_realtime - source_realtime_minus_bag_s
    command_streams = (_load_command_streams(bag)
                       if input_source != "feedback" else None)
    feedback_streams = (_load_feedback_streams(bag)
                        if input_source in {"feedback", "feedback_corrected",
                                            "packet_steer_feedback_throttle"}
                        else None)
    packet_command_count = 0
    packet_gap_fill_count = 0
    if input_source in {"packet_aligned", "packet_steer_feedback_throttle"}:
        packet_times, packet_steering, packet_throttle = _load_packet_commands(bag)
        steering_times, steering_values, throttle_times, throttle_values = (
            command_streams)
        steering_indexes = np.searchsorted(
            steering_times, bag_times, side="right") - 1
        throttle_indexes = np.searchsorted(
            throttle_times, bag_times, side="right") - 1
        if (np.any(steering_indexes < 0) or np.any(throttle_indexes < 0)):
            raise ValueError("capture interval precedes recorded command topics")
        # A response-side debug record is the exact command submitted at that
        # 40 Hz request. Use it only while the request timeline is contiguous;
        # when response records are missing, fill the interval from the ordinary
        # command topics rather than holding a stale debug command through a
        # reset or transport gap. This mode is offline analysis only.
        packet_indexes = np.searchsorted(packet_times, bag_times, side="right") - 1
        packet_valid = packet_indexes >= 0
        packet_age_s = np.full_like(bag_times, np.inf)
        packet_age_s[packet_valid] = (
            bag_times[packet_valid] - packet_times[packet_indexes[packet_valid]])
        packet_fresh = (packet_valid & (packet_age_s <= 0.040)
                        & (packet_times[np.maximum(packet_indexes, 0)]
                           >= bag_times[0]))
        steering = steering_values[steering_indexes].copy()
        throttle = throttle_values[throttle_indexes].copy()
        steering[packet_fresh] = packet_steering[packet_indexes[packet_fresh]]
        throttle[packet_fresh] = packet_throttle[packet_indexes[packet_fresh]]
        packet_command_count = int(np.count_nonzero(packet_fresh))
        packet_gap_fill_count = int(len(bag_times) - packet_command_count)
        if input_source == "packet_steer_feedback_throttle":
            feedback_steer_times, _, feedback_throttle_times, feedback_throttle_values = (
                feedback_streams)
            feedback_offset = (source_realtime_minus_bag_start_s
                               if feedback_realtime_minus_bag_s is None else
                               feedback_realtime_minus_bag_s)
            feedback_bag_times = capture_realtime - feedback_offset
            feedback_indexes = np.searchsorted(
                feedback_throttle_times, feedback_bag_times, side="right") - 1
            feedback_valid = feedback_indexes >= 0
            feedback_valid[feedback_valid] &= (
                feedback_throttle_times[feedback_indexes[feedback_valid]]
                >= feedback_bag_times[0])
            throttle = np.zeros_like(bag_times)
            throttle[feedback_valid] = feedback_throttle_values[
                feedback_indexes[feedback_valid]]
    elif input_source == "feedback":
        steering_times, steering_values, throttle_times, throttle_values = (
            feedback_streams)
        steering_indexes = np.searchsorted(
            steering_times, bag_times, side="right") - 1
        throttle_indexes = np.searchsorted(
            throttle_times, bag_times, side="right") - 1
        steering = np.zeros_like(bag_times)
        throttle = np.zeros_like(bag_times)
        steering_valid = steering_indexes >= 0
        throttle_valid = throttle_indexes >= 0
        steering_valid[steering_valid] &= (
            steering_times[steering_indexes[steering_valid]] >= bag_times[0])
        throttle_valid[throttle_valid] &= (
            throttle_times[throttle_indexes[throttle_valid]] >= bag_times[0])
        steering[steering_valid] = steering_values[
            steering_indexes[steering_valid]]
        throttle[throttle_valid] = throttle_values[
            throttle_indexes[throttle_valid]]
    else:
        steering_times, steering_values, throttle_times, throttle_values = (
            command_streams)
        steering_indexes = np.searchsorted(
            steering_times, bag_times, side="right") - 1
        throttle_indexes = np.searchsorted(
            throttle_times, bag_times, side="right") - 1
        if (bag_times[0] < steering_times[0]
                or bag_times[-1] > steering_times[-1]
                or bag_times[0] < throttle_times[0]
                or bag_times[-1] > throttle_times[-1]
                or np.any(steering_indexes < 0)
                or np.any(throttle_indexes < 0)):
            raise ValueError(
                "capture interval lies outside the recorded ROS commands")
        steering = steering_values[steering_indexes]
        throttle = throttle_values[throttle_indexes]
        if input_source == "feedback_corrected":
            # The segment begins at an explicit simulator reset. A command
            # timestamp just before that boundary is stale even if its value
            # remains latched in ROS; reset establishes neutral until the
            # first post-reset command arrives.
            steering[steering_times[steering_indexes] < bag_times[0]] = 0.0
            throttle[throttle_times[throttle_indexes] < bag_times[0]] = 0.0

    center_angle_deg = 0.0
    last_angle_deg = 0.0
    wheel_commands = []
    feedback_reinitialized = 0
    if input_source == "feedback_corrected":
        feedback_steer_times, feedback_steer_values, feedback_throttle_times, _ = (
            feedback_streams)
        feedback_offset = (source_realtime_minus_bag_s
                           if feedback_realtime_minus_bag_s is None else
                           feedback_realtime_minus_bag_s)
        feedback_bag_times = capture_realtime - feedback_offset
        feedback_steer_indexes = np.searchsorted(
            feedback_steer_times, feedback_bag_times, side="right") - 1
        feedback_steer_valid = feedback_steer_indexes >= 0
        feedback_steer_valid[feedback_steer_valid] &= (
            feedback_steer_times[
                feedback_steer_indexes[feedback_steer_valid]]
            >= feedback_bag_times[0])
        last_feedback_index = -1
    else:
        feedback_steer_times = feedback_steer_values = None
        feedback_steer_indexes = feedback_steer_valid = None
        feedback_throttle_times = None
        last_feedback_index = -1
    for steering_command, throttle_command in zip(steering, throttle):
        if input_source == "feedback":
            # CurrentSteeringAngle is already the measured physical actuator
            # output; hold it rather than applying the command slew a second time.
            center_angle_deg = -math.degrees(float(steering_command))
            last_angle_deg = center_angle_deg
        elif input_source == "feedback_corrected":
            # Re-anchor the source steering integrator whenever a new
            # feedback packet arrives, then propagate the known command at
            # 1 kHz until the next measurement.
            row_index = len(wheel_commands)
            if feedback_steer_valid[row_index]:
                feedback_index = int(feedback_steer_indexes[row_index])
                if feedback_index != last_feedback_index:
                    center_angle_deg = -math.degrees(float(
                        feedback_steer_values[feedback_index]))
                    last_angle_deg = center_angle_deg
                    last_feedback_index = feedback_index
                    feedback_reinitialized += 1
            center_angle_deg = step_autonomous_steering_deg(
                center_angle_deg, last_angle_deg, float(steering_command),
                replay.FIXED_DT_S)
            last_angle_deg = center_angle_deg
        else:
            center_angle_deg = step_autonomous_steering_deg(
                center_angle_deg, last_angle_deg, float(steering_command),
                replay.FIXED_DT_S)
            last_angle_deg = center_angle_deg
        left_deg, right_deg = front_ackermann_angles_deg(center_angle_deg)
        actuation = wheel_actuation(float(throttle_command))
        wheel_commands.append([
            {
                "steer_angle_rad": float(np.deg2rad(angle)),
                "drive_torque_nm": actuation.motor_torque_nm[index],
                "brake_torque_nm": actuation.brake_torque_nm[index],
            }
            for index, angle in enumerate((left_deg, right_deg, 0.0, 0.0))
        ])

    for index, command in enumerate(wheel_commands):
        data["steps"][index]["wheel_commands"] = command
    data["command_input"] = {
        "source": (
            "offline packet-aligned steering command plus allowed measured "
            "throttle feedback; command topics fill only missing packet-response "
            "intervals"
            if input_source == "packet_steer_feedback_throttle" else
            "offline bridge_packet_timing sent commands, with allowed ROS "
            "command topics used only across missing packet-response intervals"
            if input_source == "packet_aligned" else
            "ROS bag zero-order-held steering/throttle command topics"
            if input_source == "command" else
            "ROS bag actuator feedback topics"
            if input_source == "feedback_corrected" else
            "ROS bag zero-order-held simulator steering/throttle feedback topics"),
        "bag": str(bag.resolve().relative_to(ROOT)),
        "topics": (
            [PACKET_TOPIC, STEERING_TOPIC, THROTTLE_TOPIC,
             THROTTLE_FEEDBACK_TOPIC]
            if input_source == "packet_steer_feedback_throttle" else
            [PACKET_TOPIC, STEERING_TOPIC, THROTTLE_TOPIC]
            if input_source == "packet_aligned" else
            [STEERING_TOPIC, THROTTLE_TOPIC,
             STEERING_FEEDBACK_TOPIC, THROTTLE_FEEDBACK_TOPIC]
            if input_source == "feedback_corrected" else
            [STEERING_TOPIC, THROTTLE_TOPIC]
            if input_source == "command" else
            [STEERING_FEEDBACK_TOPIC, THROTTLE_FEEDBACK_TOPIC]),
        "feedback_interpretation": (
            None if input_source == "command" else {
                "steering": "VehicleController.CurrentSteeringAngle in radians; source getter is -SteeringAngle * pi/180",
                "throttle": "VehicleController.CurrentThrottle = DriveTorque / MotorTorque",
                "reset_gap": "neutral actuator state from the documented simulator reset",
            }),
        "source_realtime_minus_bag_start_s": source_realtime_minus_bag_s,
        "feedback_realtime_minus_bag_start_s": (
            feedback_realtime_minus_bag_s
            if input_source in {"feedback_corrected",
                                "packet_steer_feedback_throttle"} else None),
        "clock_alignment_role": "offline run pairing only; not a model parameter",
        "offline_only_restricted_packet_debug": input_source in {
            "packet_aligned", "packet_steer_feedback_throttle"},
        "packet_request_command_samples": packet_command_count,
        "packet_gap_command_topic_samples": packet_gap_fill_count,
        "throttle_feedback_used": (
            input_source == "packet_steer_feedback_throttle"),
        "controller_expansion": (
            "packet-aligned steering commands expanded through Unity steering "
            "equations; measured permitted throttle feedback held causally "
            "at its response rate"
            if input_source == "packet_steer_feedback_throttle" else
            "bridge request-sent commands held causally on the aligned 1 ms "
            "capture grid; ordinary command topics fill only missing response "
            "intervals; Unity steering equations at 1 ms"
            if input_source == "packet_aligned" else
            "Unity VehicleController equations at 1 ms"
            if input_source == "command" else
            "measured steering state reinitialized at each feedback packet; known command propagated with Unity steering equations at 1 ms"
            if input_source == "feedback_corrected" else
            "measured actuator feedback held causally at 1 ms; Ackermann wheel angles reconstructed from physical steering feedback"),
        "future_sensor_or_truth_inputs": False,
    }
    data["controller_boundary_reconstruction"] = {
        "simulation_step_s": replay.FIXED_DT_S,
        "input_update": (
            "40 Hz bridge steering command and measured throttle feedback "
            "zero-order holds; command topics fill missing steering packet records"
            if input_source == "packet_steer_feedback_throttle" else
            "40 Hz bridge request sent-command zero-order hold; command topics "
            "fill response-record gaps"
            if input_source == "packet_aligned" else
            "ROS command zero-order hold plus Unity controller state update"
            if input_source == "command" else
            "ROS command zero-order hold; measured steering-state correction at feedback packets"
            if input_source == "feedback_corrected" else
            "ROS actuator feedback zero-order hold; no future feedback used"),
        "held_input_intervals": int(np.count_nonzero(
            (np.diff(steering) != 0.0) | (np.diff(throttle) != 0.0))) + 1,
        "steering_feedback_reinitializations": feedback_reinitialized,
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--start-row", type=int, required=True)
    parser.add_argument("--count", type=int, required=True,
                        help="1 ms physics ticks; must be divisible by 25")
    parser.add_argument("--source-realtime-minus-bag-start-s", type=float,
                        required=True)
    parser.add_argument("--input-source",
                        choices=("command", "feedback", "feedback_corrected",
                                 "packet_aligned",
                                 "packet_steer_feedback_throttle"),
                        default="command")
    parser.add_argument("--feedback-realtime-minus-bag-start-s", type=float)
    parser.add_argument("--shape-audit", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.capture, args.bag, args.count, args.start_row,
                    args.source_realtime_minus_bag_start_s, args.shape_audit,
                    args.input_source,
                    args.feedback_realtime_minus_bag_start_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    print(f"wrote {args.count}-tick recursive rollout from ROS {args.input_source} data")


if __name__ == "__main__":
    main()

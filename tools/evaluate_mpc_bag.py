#!/usr/bin/env python3
"""Measure production odometry and MPC rollout errors against Unity bag truth.

Run in the ROS 2 environment that can deserialize the bag's message types:
  python3 tools/evaluate_mpc_bag.py live_runs/<run>/run [--trajectory path.csv]

The trajectory is optional; when supplied, diagnostic path progress and
cross-track error are converted back to a local XY/yaw rollout for comparison.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


GT = "/autodrive/roboracer_1/odom"
ODOM = "/odom"
EKF = "/ekf_odom"
PLANT = "/odom/unity_plant_state"
DIAGNOSTICS = "/mpc/diagnostics"
OPEN_LOOP_TEST = "/mpc/open_loop_test"
IMU = "/autodrive/roboracer_1/imu"
LAP = "/autodrive/roboracer_1/lap_count"
COLLISION = "/autodrive/roboracer_1/collision_count"
LAST_LAP = "/autodrive/roboracer_1/last_lap_time"
BEST_LAP = "/autodrive/roboracer_1/best_lap_time"
THROTTLE = "/autodrive/roboracer_1/throttle"
THROTTLE_COMMAND = "/autodrive/roboracer_1/throttle_command"
STEERING = "/autodrive/roboracer_1/steering"
STEERING_COMMAND = "/autodrive/roboracer_1/steering_command"
BRIDGE_FAULT = "/autodrive/roboracer_1/bridge_timing_fault_detail"
BRIDGE_TIMING = "/autodrive/roboracer_1/bridge_packet_timing"
DRIVE_COMMAND = "/mpc/drive_command"
LEFT_ENCODER = "/autodrive/roboracer_1/left_encoder"
RIGHT_ENCODER = "/autodrive/roboracer_1/right_encoder"
ODOM_DIAGNOSTICS = "/odom/diagnostics"
COM_TO_REAR_AXLE_M = 0.15532
WHEEL_RADIUS_M = 0.059
REAR_HALF_TRACK_M = 0.118
ENCODER_SAMPLE_PERIOD_S = 0.025
MAX_STEERING_RAD = math.pi / 6.0
UNITY_SOURCE_PLANT_STATE_SIZE = 26


class UnitySourcePlantInputC(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_double * UNITY_SOURCE_PLANT_STATE_SIZE),
        ("wheel_rotation_angle_rad", ctypes.c_double * 4),
        ("steering_target_rad", ctypes.c_double),
        ("throttle_normalized", ctypes.c_double),
        ("suspension_history_initialized", ctypes.c_int),
    ]


class UnitySourcePlantOutputC(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_double * UNITY_SOURCE_PLANT_STATE_SIZE),
        ("integrated_body_forward_displacement_m", ctypes.c_double),
        ("integrated_body_lateral_displacement_m", ctypes.c_double),
        ("integrated_yaw_delta_rad", ctypes.c_double),
        ("wheel_rotation_angle_rad", ctypes.c_double * 4),
        ("suspension_history_initialized", ctypes.c_int),
        ("longitudinal_slip", ctypes.c_double * 4),
        ("maximum_positive_longitudinal_slip", ctypes.c_double * 4),
        ("maximum_longitudinal_slip_excess_mps", ctypes.c_double * 4),
        ("lateral_slip", ctypes.c_double * 4),
        ("normal_load_n", ctypes.c_double * 4),
        ("suspension_force_n", ctypes.c_double * 4),
        ("suspension_jounce_rate_mps", ctypes.c_double * 4),
        ("tire_longitudinal_force_n", ctypes.c_double * 4),
        ("tire_lateral_force_n", ctypes.c_double * 4),
        ("longitudinal_acceleration_mps2", ctypes.c_double),
        ("lateral_acceleration_mps2", ctypes.c_double),
        ("yaw_acceleration_radps2", ctypes.c_double),
    ]


def validate_source_plant_abi() -> None:
    # Keep synchronized with unity_source_plant_c.h. A ctypes layout mismatch
    # can silently corrupt rollout inputs/outputs while still returning success.
    expected_input_offsets = {
        "wheel_rotation_angle_rad": 208,
        "steering_target_rad": 240,
        "throttle_normalized": 248,
        "suspension_history_initialized": 256,
    }
    expected_output_offsets = {
        "integrated_body_forward_displacement_m": 208,
        "integrated_body_lateral_displacement_m": 216,
        "integrated_yaw_delta_rad": 224,
        "wheel_rotation_angle_rad": 232,
        "suspension_history_initialized": 264,
        "longitudinal_slip": 272,
        "tire_longitudinal_force_n": 496,
        "longitudinal_acceleration_mps2": 560,
    }
    if ctypes.sizeof(UnitySourcePlantInputC) != 264:
        raise RuntimeError("UnitySourcePlantInputC size disagrees with C ABI")
    if ctypes.sizeof(UnitySourcePlantOutputC) != 584:
        raise RuntimeError("UnitySourcePlantOutputC size disagrees with C ABI")
    for structure, offsets in (
        (UnitySourcePlantInputC, expected_input_offsets),
        (UnitySourcePlantOutputC, expected_output_offsets),
    ):
        for field, expected in offsets.items():
            if getattr(structure, field).offset != expected:
                raise RuntimeError(
                    f"{structure.__name__}.{field} offset disagrees with C ABI")


validate_source_plant_abi()

TOPICS = {
    GT, ODOM, EKF, PLANT, DIAGNOSTICS, OPEN_LOOP_TEST, IMU, LAP, COLLISION, LAST_LAP,
    BEST_LAP, THROTTLE, THROTTLE_COMMAND, STEERING, STEERING_COMMAND,
    BRIDGE_FAULT, BRIDGE_TIMING, DRIVE_COMMAND, LEFT_ENCODER, RIGHT_ENCODER,
    ODOM_DIAGNOSTICS,
}


def stamp(message: Any) -> float:
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


def yaw_from_quaternion(q: Any) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def roll_pitch_from_quaternion(q: Any) -> tuple[float, float]:
    sin_roll = 2.0 * (q.w * q.x + q.y * q.z)
    cos_roll = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    sin_pitch = 2.0 * (q.w * q.y - q.z * q.x)
    return (math.atan2(sin_roll, cos_roll),
            math.asin(max(-1.0, min(1.0, sin_pitch))))


def normalized_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def read_bag(uri: Path) -> dict[str, list[dict[str, Any]]]:
    storage_id = "mcap" if any(uri.glob("*.mcap")) else "sqlite3"
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(uri), storage_id=storage_id),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    type_names = {
        topic.name: topic.type
        for topic in reader.get_all_topics_and_types()
        if topic.name in TOPICS
    }
    message_types = {}
    for name, kind in type_names.items():
        try:
            message_types[name] = get_message(kind)
        except (AttributeError, ModuleNotFoundError):
            # The plant-state message is diagnostic-only here; the 35-value
            # odom diagnostic topic already contains its raw pre-correction
            # plant state. This lets the evaluator run on a stock ROS install.
            if name not in (PLANT, DRIVE_COMMAND):
                raise
            print(f"note: skipping {name} (custom message package not sourced)")
    data = {topic: [] for topic in type_names}
    skipped_plant_schema = False

    while reader.has_next():
        topic, serialized, bag_time_ns = reader.read_next()
        if topic not in message_types:
            continue
        try:
            message = deserialize_message(serialized, message_types[topic])
        except Exception as error:
            # Older runtime captures used earlier UnityPlantState layouts.
            # That diagnostic topic is not needed for ground-truth odom or
            # MPC forecast scoring; keep the rest of the bag usable while
            # making the schema mismatch explicit.
            if topic != PLANT:
                raise
            # Schema 11 only appended (bool, float64) to schema 10. Fast-CDR
            # needs 16 bytes for that bool, alignment padding, and double.
            # Retry with zero defaults, then accept this compatibility path
            # only when the decoded payload explicitly identifies schema 10.
            try:
                legacy_message = deserialize_message(
                    serialized + bytes(16), message_types[topic])
            except Exception:
                legacy_message = None
            if (legacy_message is not None and
                    legacy_message.state_schema_version == 10):
                message = legacy_message
            else:
                if not skipped_plant_schema:
                    print(f"note: skipping incompatible {PLANT} payloads: {error}")
                    skipped_plant_schema = True
                continue
        row: dict[str, Any] = {"bag_time": bag_time_ns * 1e-9}
        if topic in (GT, ODOM, EKF):
            roll, pitch = roll_pitch_from_quaternion(
                message.pose.pose.orientation)
            row.update(
                stamp=stamp(message),
                frame=message.header.frame_id,
                child=message.child_frame_id,
                x=message.pose.pose.position.x,
                y=message.pose.pose.position.y,
                z=message.pose.pose.position.z,
                yaw=yaw_from_quaternion(message.pose.pose.orientation),
                roll=roll,
                pitch=pitch,
                u=message.twist.twist.linear.x,
                v=message.twist.twist.linear.y,
                r=message.twist.twist.angular.z,
            )
        elif topic == PLANT:
            row.update(
                stamp=stamp(message),
                schema=message.state_schema_version,
                u=message.longitudinal_velocity_mps,
                v_com=message.lateral_velocity_com_mps,
                v_rear=message.lateral_velocity_rear_axle_mps,
                r=message.yaw_rate_radps,
                gyro_r=message.gyro_yaw_rate_radps,
                wheel_omega=list(message.wheel_omega_radps),
                wheel_angle=list(message.wheel_rotation_angle_rad),
                body_height=message.body_height_m,
                body_height_rate=message.body_height_rate_mps,
                roll=message.roll_rad,
                roll_rate=message.roll_rate_radps,
                pitch=message.pitch_rad,
                pitch_rate=message.pitch_rate_radps,
                physical_steering=message.physical_steering_rad,
                suspension_jounce=list(message.suspension_jounce_m),
                suspension_history_initialized=message.suspension_history_initialized,
                low_forward_speed_time_s=list(message.low_forward_speed_time_s),
                low_side_speed_time_s=list(message.low_side_speed_time_s),
                delayed_steering_1=message.delayed_steering_command_1_normalized,
                delayed_throttle_1=message.delayed_throttle_command_1_normalized,
                delayed_steering_2=message.delayed_steering_command_2_normalized,
                delayed_throttle_2=message.delayed_throttle_command_2_normalized,
                long_slip=list(message.longitudinal_slip),
                lat_slip=list(message.lateral_slip),
                normal_load=list(message.normal_load_n),
                tire_fx=list(message.tire_longitudinal_force_n),
                tire_fy=list(message.tire_lateral_force_n),
                ax=message.longitudinal_acceleration_mps2,
                ay=message.lateral_acceleration_mps2,
                yaw_accel=message.yaw_acceleration_radps2,
            )
        elif topic in (DIAGNOSTICS, OPEN_LOOP_TEST, BRIDGE_TIMING):
            try:
                row["json"] = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                row["json"] = {}
        elif topic == IMU:
            row.update(
                stamp=stamp(message),
                ax=message.linear_acceleration.x,
                ay=message.linear_acceleration.y,
                az=message.linear_acceleration.z,
                roll_rate=message.angular_velocity.x,
                pitch_rate=message.angular_velocity.y,
                yaw_rate=message.angular_velocity.z,
            )
        elif topic == DRIVE_COMMAND:
            row.update(
                stamp=stamp(message),
                steering=message.steering_angle_rad,
                throttle=message.throttle_normalized,
            )
        elif topic in (LEFT_ENCODER, RIGHT_ENCODER):
            row.update(
                stamp=stamp(message),
                angle=message.position[0] if message.position else math.nan,
                velocity=message.velocity[0] if message.velocity else math.nan,
            )
        elif topic == ODOM_DIAGNOSTICS:
            row.update(label=message.layout.dim[0].label if message.layout.dim else "",
                       values=list(message.data))
        elif topic == BRIDGE_FAULT:
            row["value"] = message.data
        else:
            row["value"] = message.data
        data[topic].append(row)
    return data


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * (len(ordered) - 1)))]


def print_interval(topic: str, rows: list[dict[str, Any]]) -> None:
    times = [row.get("stamp", row["bag_time"]) for row in rows]
    intervals = [b - a for a, b in zip(times, times[1:]) if 0.0 < b - a < 0.2]
    if intervals:
        print(
            f"rate {topic}: n={len(rows)} mean={statistics.mean(intervals)*1000:.2f}ms "
            f"p95={percentile(intervals, 0.95)*1000:.2f}ms "
            f"span={times[-1]-times[0]:.2f}s"
        )


def matched_pairs(
    truth: list[dict[str, Any]], estimate: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    lookup = {round(row["stamp"], 6): row for row in estimate}
    return [
        (row, lookup[round(row["stamp"], 6)])
        for row in truth
        if round(row["stamp"], 6) in lookup
    ]


def report_state_errors(
    truth: list[dict[str, Any]], estimate: list[dict[str, Any]], label: str,
) -> None:
    pairs = matched_pairs(truth, estimate)
    print(f"{label}: matched={len(pairs)}")
    if not pairs:
        return
    for field in ("u", "v", "r"):
        # Simulator /odom twist is at the rigid-body COM; production odom is
        # at base_link (rear axle). Rotate the COM velocity to the rear axle
        # before scoring its lateral component.
        if field == "v" and label in ("production odom", "EKF odom"):
            errors = [estimated[field] - (actual["v"] - actual["r"] * COM_TO_REAR_AXLE_M)
                      for actual, estimated in pairs]
        else:
            errors = [estimated[field] - actual[field] for actual, estimated in pairs]
        absolute = [abs(error) for error in errors]
        print(
            f"  {field}: bias={statistics.mean(errors):.4f} "
            f"RMSE={math.sqrt(statistics.mean(e*e for e in errors)):.4f} "
            f"p95_abs={percentile(absolute, 0.95):.4f} "
            f"max_abs={max(absolute):.4f}"
        )

    actual0, estimated0 = pairs[0]
    yaw_alignment = estimated0["yaw"] - actual0["yaw"]
    position_errors = []
    yaw_errors = []
    c, s = math.cos(yaw_alignment), math.sin(yaw_alignment)
    for actual, estimated in pairs:
        dx, dy = actual["x"] - actual0["x"], actual["y"] - actual0["y"]
        actual_dx, actual_dy = c * dx - s * dy, s * dx + c * dy
        estimated_dx = estimated["x"] - estimated0["x"]
        estimated_dy = estimated["y"] - estimated0["y"]
        position_errors.append(math.hypot(
            estimated_dx - actual_dx, estimated_dy - actual_dy,
        ))
        yaw_errors.append(normalized_angle(
            (estimated["yaw"] - estimated0["yaw"])
            - (actual["yaw"] - actual0["yaw"])
        ))
    print(
        f"  relative_pose: xy_end={position_errors[-1]:.3f}m "
        f"xy_p95={percentile(position_errors, 0.95):.3f}m "
        f"xy_max={max(position_errors):.3f}m "
        f"yaw_end={yaw_errors[-1]:.3f}rad "
        f"yaw_p95_abs={percentile([abs(x) for x in yaw_errors], 0.95):.3f}rad"
    )


def report_open_loop_horizon_test(
    truth: list[dict[str, Any]], events: list[dict[str, Any]],
) -> None:
    payloads = [row.get("json", {}) for row in events if row.get("json")]
    plans = [row for row in payloads if row.get("event") == "plan"]
    samples = [row for row in payloads if row.get("event") == "sample"]
    outputs = [row for row in payloads if row.get("event") == "output"]
    completions = [row for row in payloads if row.get("event") == "complete"]
    aborts = [row for row in payloads if row.get("event") == "abort"]
    observation_gaps = [row for row in payloads
                        if row.get("event") == "observation_gap"]
    if not plans:
        print("MPC frozen-horizon test: no latched plan was recorded")
        return
    plan = plans[-1]
    plan_id = plan.get("plan_id")
    samples = [row for row in samples if row.get("plan_id") == plan_id]
    outputs = [row for row in outputs if row.get("plan_id") == plan_id]
    observation_gaps = [row for row in observation_gaps
                        if row.get("plan_id") == plan_id]
    controls = plan.get("controls", [])
    step_samples = sorted(samples, key=lambda row: row.get("stage", -1))
    sample_times = [row["control_stamp_ns"] * 1e-9
                    for row in step_samples if row.get("control_stamp_ns")]
    print("\nMPC frozen-plan execution:")
    print(
        f"  plan_id={plan_id} speed_at_latch="
        f"{plan.get('trigger_speed_mps', math.nan):.3f}m/s "
        f"steps={len(controls)} dt={plan.get('step_s', math.nan)*1000:.2f}ms "
        f"completed={bool(completions)} aborted={bool(aborts)}"
    )
    if observation_gaps:
        missing = sorted({int(row.get("stage", -1)) for row in observation_gaps})
        print(f"  missing-state observations={len(observation_gaps)} stages={missing}")
    completion = completions[-1] if completions else {}
    print(
        f"  emitted fixed commands={len(outputs)}/{len(controls)} "
        f"terminal_state_observed={completion.get('terminal_state_observed', False)}"
    )
    output_times = sorted(
        int(row["steady_stamp_ns"]) * 1.0e-9
        for row in outputs if row.get("steady_stamp_ns")
    )
    output_intervals = [b - a for a, b in zip(output_times, output_times[1:])
                        if b > a]
    if output_intervals:
        print(
            f"  output interval: median={statistics.median(output_intervals)*1000:.2f}ms "
            f"p95={percentile(output_intervals, 0.95)*1000:.2f}ms "
            f"max={max(output_intervals)*1000:.2f}ms"
        )
    if sample_times:
        intervals = [b - a for a, b in zip(sample_times, sample_times[1:])
                     if b > a]
        if intervals:
            print(
                f"  observed sample interval: median="
                f"{statistics.median(intervals)*1000:.2f}ms "
                f"p95={percentile(intervals, 0.95)*1000:.2f}ms "
                f"samples={len(sample_times)}"
            )

    truth = sorted((row for row in truth if math.isfinite(row.get("stamp", math.nan))),
                   key=lambda row: row["stamp"])
    truth_times = [row["stamp"] for row in truth]

    def truth_at(timestamp: float) -> dict[str, float] | None:
        index = bisect.bisect_left(truth_times, timestamp)
        if index < len(truth_times) and abs(truth_times[index] - timestamp) < 1e-6:
            return truth[index]
        if index == 0 or index >= len(truth_times):
            return None
        before, after = truth[index - 1], truth[index]
        gap = after["stamp"] - before["stamp"]
        if gap <= 0.0 or gap > 0.10:
            return None
        fraction = (timestamp - before["stamp"]) / gap
        result = {}
        for key in ("x", "y", "u", "v", "r"):
            result[key] = before[key] + fraction * (after[key] - before[key])
        result["yaw"] = before["yaw"] + fraction * normalized_angle(
            after["yaw"] - before["yaw"])
        return result

    errors: dict[str, list[float]] = {
        key: [] for key in ("position", "yaw", "u", "v_rear", "r")
    }
    hidden_errors = [[] for _ in range(4)]
    speeds: list[float] = []
    full_plan = [row for row in plan.get("predictions", []) if row]
    scored_rows: list[tuple[float, list[float], list[float], dict[str, Any] | None]] = []
    if full_plan and plan.get("stamp_ns"):
        plan_stamp = int(plan["stamp_ns"]) * 1e-9
        for prediction in full_plan:
            timestamp = plan_stamp + float(prediction.get("time_s", 0.0))
            scored_rows.append((timestamp, prediction.get("pose", []),
                                prediction.get("state", []), None))
        print("  accuracy source: complete frozen-plan states vs Unity ground truth")
    else:
        for row in step_samples:
            if row.get("control_stamp_ns"):
                scored_rows.append((row["control_stamp_ns"] * 1e-9,
                    row.get("predicted_pose", []), row.get("predicted_state", []), row))
        print("  accuracy source: sparse frozen-plan states vs Unity ground truth")

    for timestamp, predicted_pose, predicted_state, source_row in scored_rows:
        measured = truth_at(timestamp)
        if measured is None or len(predicted_pose or []) != 3 or len(predicted_state or []) < 6:
            continue
        px, py, pyaw = predicted_pose
        errors["position"].append(math.hypot(px - measured["x"], py - measured["y"]))
        errors["yaw"].append(abs(normalized_angle(pyaw - measured["yaw"])))
        errors["u"].append(predicted_state[2] - measured["u"])
        # The plant's MPC lateral velocity is at the rear axle; simulator
        # odometry reports the rigid-body/COM velocity.
        measured_v_rear = measured["v"] - measured["r"] * COM_TO_REAR_AXLE_M
        errors["v_rear"].append(predicted_state[3] - measured_v_rear)
        errors["r"].append(predicted_state[4] - measured["r"])
        speeds.append(math.hypot(measured["u"], measured_v_rear))

    for row in step_samples:
        predicted_state = row.get("predicted_state", [])
        actual_state = row.get("actual_state", [])
        if len(actual_state) >= 10 and len(predicted_state) >= 10:
            for wheel in range(4):
                hidden_errors[wheel].append(
                    predicted_state[6 + wheel] - actual_state[6 + wheel]
                )

    if not errors["position"]:
        print("  no samples could be aligned to simulator truth")
    else:
        for key, label, unit in (
            ("position", "position", "m"), ("yaw", "yaw", "rad"),
            ("u", "forward speed", "m/s"),
            ("v_rear", "rear lateral speed", "m/s"),
            ("r", "yaw rate", "rad/s"),
        ):
            values = errors[key]
            absolute = [abs(value) for value in values]
            print(
                f"  {label}: n={len(values)} RMSE="
                f"{math.sqrt(statistics.mean(value*value for value in values)):.5f}{unit} "
                f"p95_abs={percentile(absolute, 0.95):.5f}{unit} "
                f"max_abs={max(absolute):.5f}{unit}"
            )
        print(
            f"  truth speed during scored horizon: "
            f"{min(speeds):.3f}-{max(speeds):.3f}m/s"
        )
    if any(hidden_errors):
        print("  predicted-vs-reanchored odom wheel-rate RMSE (not independent wheel truth):")
        for wheel, label in enumerate(("FL", "FR", "RL", "RR")):
            values = hidden_errors[wheel]
            rmse = (math.sqrt(statistics.mean(v*v for v in values))
                    if values else math.nan)
            print(f"    {label}: n={len(values)} RMSE={rmse:.5f}rad/s")
    if aborts:
        print(f"  abort detail: {aborts[-1]}")


def nearest_by_stamp(rows: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    if not rows:
        return None
    times = [row.get("stamp", row["bag_time"]) for row in rows]
    index = bisect.bisect_left(times, target)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    return min((rows[i] for i in candidates), key=lambda row: abs(
        row.get("stamp", row["bag_time"]) - target))


def report_truth_pose_twist_phase(
    truth: list[dict[str, Any]], bridge_timing: list[dict[str, Any]],
) -> None:
    """Diagnose pose/twist epoch coherence without changing any model inputs.

    Unity's GPS-like pose is attached to the rear axle while its IMU linear
    velocity is measured at the Rigidbody. The existing rigid-body point
    correction is applied here. A small phase sweep asks whether the recorded
    pose increments agree better with the same packet's twist or with a
    shifted twist stream. The best lag is descriptive only; it is not applied
    to odometry or scored as a corrected result.
    """
    rows = sorted(
        (row for row in truth if math.isfinite(row.get("stamp", math.nan))),
        key=lambda row: row["stamp"],
    )
    if len(rows) < 3:
        print("truth pose/twist phase: needs at least 3 truth samples")
        return
    times = [row["stamp"] for row in rows]

    def sample_at(timestamp: float) -> tuple[float, float, float, float] | None:
        index = bisect.bisect_left(times, timestamp)
        if index < len(rows) and abs(times[index] - timestamp) < 1.0e-9:
            row = rows[index]
            return row["u"], row["v"], row["r"], row["yaw"]
        if index == 0 or index == len(rows):
            return None
        before, after = rows[index - 1], rows[index]
        gap = after["stamp"] - before["stamp"]
        if gap <= 0.0 or gap > 0.040:
            return None
        fraction = (timestamp - before["stamp"]) / gap
        yaw_delta = normalized_angle(after["yaw"] - before["yaw"])
        return (
            before["u"] + fraction * (after["u"] - before["u"]),
            before["v"] + fraction * (after["v"] - before["v"]),
            before["r"] + fraction * (after["r"] - before["r"]),
            before["yaw"] + fraction * yaw_delta,
        )

    def world_velocity(sample: tuple[float, float, float, float]) -> tuple[float, float]:
        u, v_com, yaw_rate, yaw = sample
        v_rear = v_com - yaw_rate * COM_TO_REAR_AXLE_M
        return (
            u * math.cos(yaw) - v_rear * math.sin(yaw),
            u * math.sin(yaw) + v_rear * math.cos(yaw),
        )

    intervals: list[tuple[int, float, float, float, float]] = []
    for index, (before, after) in enumerate(zip(rows, rows[1:])):
        dt = after["stamp"] - before["stamp"]
        if 0.010 <= dt <= 0.040:
            intervals.append((
                index, before["stamp"], dt,
                after["x"] - before["x"], after["y"] - before["y"],
            ))
    if len(intervals) < 3:
        print("truth pose/twist phase: insufficient regular intervals")
        return

    phase_results: list[tuple[float, list[float], list[tuple[float, float]]]] = []
    # Search ±10 ms in half-millisecond increments. This is a diagnostic
    # resolution only; source-sample timestamps are absent from these bags.
    for phase_index in range(-20, 21):
        phase_s = phase_index * 0.0005
        errors: list[float] = []
        residual_vectors: list[tuple[float, float]] = []
        for _, t0, dt, measured_x, measured_y in intervals:
            t1 = t0 + dt
            tmid = 0.5 * (t0 + t1)
            first = sample_at(t0 + phase_s)
            middle = sample_at(tmid + phase_s)
            last = sample_at(t1 + phase_s)
            if first is None or middle is None or last is None:
                continue
            vx0, vy0 = world_velocity(first)
            vxm, vym = world_velocity(middle)
            vx1, vy1 = world_velocity(last)
            predicted_x = dt * (vx0 + 4.0 * vxm + vx1) / 6.0
            predicted_y = dt * (vy0 + 4.0 * vym + vy1) / 6.0
            error_x = predicted_x - measured_x
            error_y = predicted_y - measured_y
            residual_vectors.append((error_x, error_y))
            errors.append(math.hypot(error_x, error_y))
        if errors:
            phase_results.append((phase_s, errors, residual_vectors))
    if not phase_results:
        print("truth pose/twist phase: no intervals could be aligned")
        return

    def summarize(label: str, errors: list[float]) -> str:
        return (
            f"{label}: n={len(errors)} RMSE={math.sqrt(statistics.mean(e*e for e in errors))*100:.3f}cm "
            f"p95={percentile(errors, 0.95)*100:.3f}cm max={max(errors)*100:.3f}cm"
        )

    zero = min(phase_results, key=lambda result: abs(result[0]))
    best = min(phase_results, key=lambda result: statistics.mean(e*e for e in result[1]))
    print("\nUnity truth pose/twist coherence (diagnostic only; no lag correction applied):")
    print("  " + summarize("zero phase", zero[1]))
    print(
        "  " + summarize(f"lowest-RMSE phase {best[0] * 1e3:+.1f}ms", best[1])
        + "; compare across independent runs before attributing a sensor epoch offset"
    )

    zero_residuals = zero[2]
    worst = sorted(
        range(len(zero_residuals)),
        key=lambda index: math.hypot(*zero_residuals[index]),
        reverse=True,
    )[:5]
    for rank, interval_index in enumerate(worst, start=1):
        source_index, timestamp, dt, _, _ = intervals[interval_index]
        before, after = rows[source_index], rows[source_index + 1]
        error_x, error_y = zero_residuals[interval_index]
        speed = math.hypot(before["u"], before["v"])
        print(
            f"    worst{rank}: t={timestamp:.6f}s dt={dt*1e3:.2f}ms "
            f"speed={speed:.2f}m/s r={before['r']:.3f}rad/s "
            f"pose/twist residual=({error_x*100:+.2f},{error_y*100:+.2f})cm "
            f"norm={math.hypot(error_x, error_y)*100:.2f}cm"
        )

    timing_events: list[dict[str, Any]] = []
    for row in bridge_timing:
        payload = row.get("json", {})
        request_monotonic_ns = payload.get("request_monotonic_ns")
        arrival_monotonic_ns = payload.get("bridge_arrival_monotonic_ns")
        request_ros_ns = payload.get("bridge_request_ros_stamp_ns")
        receive_ros_ns = payload.get("bridge_receive_ros_stamp_ns")
        if request_monotonic_ns is not None and arrival_monotonic_ns is not None:
            request_time = int(request_monotonic_ns) * 1.0e-9
            receive_time = int(arrival_monotonic_ns) * 1.0e-9
        elif request_ros_ns is not None and receive_ros_ns is not None:
            request_time = int(request_ros_ns) * 1.0e-9
            receive_time = int(receive_ros_ns) * 1.0e-9
        else:
            request_time = receive_time = None
        timing_events.append({
            "bag_time": row["bag_time"],
            "request_time": request_time,
            "receive_time": receive_time,
            "payload": payload,
        })
    timing_events.sort(key=lambda event: event["bag_time"])
    if not timing_events:
        print("  bridge timing: unavailable in this bag; source physics-step ID absent")
        return

    timing_bag_times = [event["bag_time"] for event in timing_events]
    used_timing_indices: set[int] = set()
    bag_match_offsets_ms: list[float] = []

    def timing_at(truth_row: dict[str, Any]) -> dict[str, Any] | None:
        timestamp = truth_row["bag_time"]
        index = bisect.bisect_left(timing_bag_times, timestamp)
        candidates = [
            i for i in (index - 1, index)
            if 0 <= i < len(timing_events) and i not in used_timing_indices
        ]
        if not candidates:
            return None
        event_index = min(
            candidates,
            key=lambda i: abs(timing_events[i]["bag_time"] - timestamp),
        )
        event = timing_events[event_index]
        offset_ms = abs(event["bag_time"] - timestamp) * 1.0e3
        if offset_ms > 10.0:
            return None
        used_timing_indices.add(event_index)
        bag_match_offsets_ms.append(offset_ms)
        return event

    timing_for_truth = [timing_at(row) for row in rows]
    closure_by_jitter: list[tuple[float, float]] = []
    latency_ms: list[float] = []
    physics_ids = 0
    matched_timing_intervals = 0
    for event in timing_events:
        if event["payload"].get("simulation_physics_step") is not None:
            physics_ids += 1
    for interval_index, (source_index, _, _, _, _) in enumerate(intervals):
        first = timing_for_truth[source_index]
        last = timing_for_truth[source_index + 1]
        if first is None or last is None:
            continue
        if (first["request_time"] is None or first["receive_time"] is None or
                last["request_time"] is None or last["receive_time"] is None):
            continue
        request_gap = last["request_time"] - first["request_time"]
        receive_gap = last["receive_time"] - first["receive_time"]
        jitter_s = receive_gap - request_gap
        closure = math.hypot(*zero_residuals[interval_index])
        closure_by_jitter.append((abs(jitter_s), closure))
        latency_ms.extend(((first["receive_time"] - first["request_time"]) * 1e3,
                           (last["receive_time"] - last["request_time"]) * 1e3))
        matched_timing_intervals += 1
    print(
        f"  bridge timing: events={len(timing_events)}/{len(bridge_timing)} "
        f"matched_intervals={matched_timing_intervals}/{len(intervals)} "
        f"physics-step IDs={physics_ids}/{len(timing_events)} "
        f"topic receipt match median/p95="
        f"{statistics.median(bag_match_offsets_ms) if bag_match_offsets_ms else math.nan:.2f}/"
        f"{percentile(bag_match_offsets_ms, 0.95) if bag_match_offsets_ms else math.nan:.2f}ms"
    )
    if latency_ms:
        print(
            f"    request→receive latency median={statistics.median(latency_ms):.2f}ms "
            f"p95={percentile(latency_ms, 0.95):.2f}ms "
            f"max={max(latency_ms):.2f}ms"
        )
    if closure_by_jitter:
        jitter_values = [pair[0] for pair in closure_by_jitter]
        closure_values = [pair[1] for pair in closure_by_jitter]
        mean_jitter = statistics.mean(jitter_values)
        mean_closure = statistics.mean(closure_values)
        covariance = statistics.mean(
            (jitter - mean_jitter) * (closure - mean_closure)
            for jitter, closure in closure_by_jitter)
        jitter_std = math.sqrt(statistics.mean(
            (value - mean_jitter) ** 2 for value in jitter_values))
        closure_std = math.sqrt(statistics.mean(
            (value - mean_closure) ** 2 for value in closure_values))
        correlation = (
            covariance / (jitter_std * closure_std)
            if jitter_std > 0.0 and closure_std > 0.0 else math.nan
        )
        ordered = sorted(closure_by_jitter, key=lambda pair: pair[0])
        split = max(1, len(ordered) // 4)
        low_jitter = [pair[1] for pair in ordered[:split]]
        high_jitter = [pair[1] for pair in ordered[-split:]]
        print(
            f"    |receive-gap minus request-gap| vs pose/twist residual: "
            f"corr={correlation:+.3f}; lowest-jitter-quartile RMSE="
            f"{math.sqrt(statistics.mean(x*x for x in low_jitter))*100:.3f}cm "
            f"highest-jitter-quartile RMSE="
            f"{math.sqrt(statistics.mean(x*x for x in high_jitter))*100:.3f}cm"
        )


def nearest_by_bag_time(
    rows: list[dict[str, Any]], target: float,
) -> dict[str, Any] | None:
    """Match receipt-time-only topics without mixing clock domains.

    In particular, Float64MultiArray odometry diagnostics have no ROS header.
    Their bag receipt time must not be compared with a sensor message's
    original simulator header stamp.
    """
    if not rows:
        return None
    times = [row["bag_time"] for row in rows]
    index = bisect.bisect_left(times, target)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    nearest = min(candidates, key=lambda i: abs(times[i] - target))
    return rows[nearest]


def report_odom_diagnostics(
    truth: list[dict[str, Any]], odom: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    steering_rows: list[dict[str, Any]], throttle_rows: list[dict[str, Any]],
) -> None:
    """Compare observer output and its uncorrected plant at the same epochs."""
    if not diagnostics:
        return
    print("\nOdometry source breakdown (Unity truth is evaluation-only):")
    print(f"  diagnostic label: {diagnostics[0].get('label', '<missing>')}")
    print("  fields: raw source-plant u/v/r, gyro-corrected yaw forecast when available, encoder speed")
    # Diagnostics and odom are emitted sequentially from the same callback,
    # but the recorder observes independent DDS subscriptions. Match them by
    # recorder receipt time, then match the resulting odom epoch to Unity by
    # the original message header stamp. Equal message counts do not guarantee
    # equal cross-topic order, and diag receipt-time is not in the simulator
    # header clock domain.
    samples: list[dict[str, float]] = []
    odom_receipt_times = [row["bag_time"] for row in odom]
    used_odom_indices: set[int] = set()
    aligned: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    diag_odom_offsets_s: list[float] = []
    truth_stamp_offsets_s: list[float] = []
    for diag in diagnostics:
        index = bisect.bisect_left(odom_receipt_times, diag["bag_time"])
        candidates = [
            i for i in (index - 1, index)
            if 0 <= i < len(odom_receipt_times) and i not in used_odom_indices
        ]
        if not candidates:
            continue
        odom_index = min(
            candidates,
            key=lambda i: abs(odom_receipt_times[i] - diag["bag_time"]))
        est = odom[odom_index]
        diag_odom_offset = est["bag_time"] - diag["bag_time"]
        if abs(diag_odom_offset) > 0.010:
            continue
        used_odom_indices.add(odom_index)
        gt = nearest_by_stamp(truth, est["stamp"])
        if gt is None:
            continue
        truth_stamp_offset = gt["stamp"] - est["stamp"]
        if abs(truth_stamp_offset) > 0.0125:
            continue
        diag_odom_offsets_s.append(diag_odom_offset)
        truth_stamp_offsets_s.append(truth_stamp_offset)
        aligned.append((gt, est, diag))
    if aligned:
        print(
            "  epoch alignment: diagnostics↔odom by bag receipt time; "
            "Unity truth↔odom by header stamp; "
            f"matched={len(aligned)}/{len(diagnostics)}, "
            f"diag/odom max Δ={max(map(abs, diag_odom_offsets_s))*1e3:.3f}ms, "
            f"truth/odom max Δ={max(map(abs, truth_stamp_offsets_s))*1e3:.3f}ms"
        )
    for sample_index, (gt, est, diag) in enumerate(aligned):
        values = diag.get("values", [])
        if len(values) < 35:
            continue
        gt_rear_v = gt["v"] - gt["r"] * COM_TO_REAR_AXLE_M
        speed = math.hypot(gt["u"], gt_rear_v)
        # Bridge actuator feedback are Float32 and headerless, so nearest bag
        # time is the correct temporal key for those streams.
        feedback_steer = nearest_by_bag_time(steering_rows, est["bag_time"])
        feedback_throttle = nearest_by_bag_time(throttle_rows, est["bag_time"])
        samples.append({
            "sample_index": float(sample_index),
            "speed": speed,
            "steering": abs(float(feedback_steer["value"])) if feedback_steer else 0.0,
            "throttle": float(feedback_throttle["value"]) if feedback_throttle else 0.0,
            "u_error": est["u"] - gt["u"],
            "raw_u_error": values[10] - gt["u"],
            "v_error": est["v"] - gt_rear_v,
            "raw_v_error": values[11] - (gt["v"]),
            "yaw_error": values[12] - gt["r"],
            "corrected_yaw_error": (
                values[39] - gt["r"] if len(values) >= 41 and
                any(version in diag.get("label", "") for version in (
                    "source_derived_odom_v2", "source_derived_odom_v3"))
                else math.nan
            ),
            "gt_yaw_rate": gt["r"],
            "raw_yaw_prediction": values[12],
            "corrected_yaw_prediction": (
                values[39] if len(values) >= 41 and
                any(version in diag.get("label", "") for version in (
                    "source_derived_odom_v2", "source_derived_odom_v3"))
                else math.nan
            ),
            "yaw_prediction_mode": values[41] if len(values) >= 42 else -1.0,
            "encoder_speed": values[7],
            "encoder_u_error": values[7] - gt["u"],
            "wheel_body_mismatch": 0.059 * 0.5 * (values[22] + values[23]) - values[10],
            "model_ax_error": values[32] - values[8],
            "model_ay_error": values[33] - values[9],
            "process_us": values[21],
        })
    if not samples:
        print("  no matched samples")
        return
    def show(label: str, selected: list[dict[str, float]]) -> None:
        if not selected:
            return
        print(f"  {label}: n={len(selected)}")
        for key, unit in (("u_error", "m/s"), ("raw_u_error", "m/s"),
                          ("encoder_u_error", "m/s"), ("v_error", "m/s"),
                          ("wheel_body_mismatch", "m/s")):
            errors = [sample[key] for sample in selected]
            print(f"    {key}: bias={statistics.mean(errors):+.4f} "
                  f"RMSE={math.sqrt(statistics.mean(x*x for x in errors)):.4f} {unit}")
        for label, key, prediction_key in (
            ("raw_yaw_forecast", "yaw_error", "raw_yaw_prediction"),
            ("corrected_yaw_forecast", "corrected_yaw_error",
             "corrected_yaw_prediction"),
        ):
            valid = [s for s in selected if math.isfinite(s[key])]
            errors = [s[key] for s in valid]
            if not errors:
                continue
            worst = max(valid, key=lambda s: abs(s[key]))
            absolute = [abs(error) for error in errors]
            print(f"    {label}: bias={statistics.mean(errors):+.4f} "
                  f"RMSE={math.sqrt(statistics.mean(x*x for x in errors)):.4f} "
                  f"p95={percentile(absolute, 0.95):.4f} "
                  f"p99={percentile(absolute, 0.99):.4f} "
                  f"max={abs(worst[key]):.4f} rad/s at sample "
                  f"{int(worst['sample_index'])} "
                  f"(GT={worst['gt_yaw_rate']:+.3f}, "
                  f"prediction={worst[prediction_key]:+.3f}, "
                  f"speed={worst['speed']:.2f}m/s, "
                  f"|steer|={worst['steering']:.3f}rad, "
                  f"throttle={worst['throttle']:.3f}, "
                  f"mode={int(worst['yaw_prediction_mode'])})")
        print(f"    raw_model_vs_IMU accel RMSE: ax={math.sqrt(statistics.mean(x['model_ax_error']**2 for x in selected)):.3f}, "
              f"ay={math.sqrt(statistics.mean(x['model_ay_error']**2 for x in selected)):.3f} m/s2")
    show("all", samples)
    for low, high in ((0, 2), (2, 4), (4, 6), (6, 8), (8, math.inf)):
        show(f"speed [{low},{high:g}) m/s", [s for s in samples if low <= s["speed"] < high])
    for low, high in ((0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, math.inf)):
        show(f"abs steering [{low:.2f},{high:g}) rad", [s for s in samples if low <= s["steering"] < high])
    for low, high in ((-math.inf, 0.01), (0.01, 0.25), (0.25, 0.75), (0.75, math.inf)):
        show(f"throttle [{low:g},{high:g})", [s for s in samples if low <= s["throttle"] < high])
    mismatch = [s for s in samples if abs(s["wheel_body_mismatch"]) > 0.5]
    show("wheel peripheral / raw-body mismatch >0.5 m/s", mismatch)


def report_encoder_phase(
    left_encoder: list[dict[str, Any]], right_encoder: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]], throttle_rows: list[dict[str, Any]],
) -> None:
    """Compare the 1 kHz pre-solve RPM integral with Unity encoder angles."""
    if not diagnostics or len(left_encoder) != len(right_encoder) or len(left_encoder) != len(diagnostics):
        return
    predicted = [row.get("values", [])[35:39] for row in diagnostics]
    if any(len(row) != 4 for row in predicted):
        return
    errors = [[], []]
    brake_transition_errors = [[], []]
    for index in range(1, len(diagnostics)):
        measured = (
            left_encoder[index]["angle"] - left_encoder[index - 1]["angle"],
            right_encoder[index]["angle"] - right_encoder[index - 1]["angle"],
        )
        for side in range(2):
            residual = measured[side] - (
                predicted[index][side + 2] - predicted[index - 1][side + 2]
            )
            errors[side].append(residual)
            q_now = nearest_by_stamp(throttle_rows, diagnostics[index]["bag_time"])
            q_prev = nearest_by_stamp(throttle_rows, diagnostics[index - 1]["bag_time"])
            if q_now and q_prev and float(q_now["value"]) <= 0.005 and float(q_prev["value"]) > 0.01:
                brake_transition_errors[side].append(residual)
    print("\nUnity encoder sample-phase check (observed encoder angle minus pre-solve RPM prediction):")
    for side, label in enumerate(("left rear", "right rear")):
        values = errors[side]
        print(f"  {label}: n={len(values)} RMSE={math.sqrt(statistics.mean(x*x for x in values)):.5f} rad")
        transition = brake_transition_errors[side]
        if transition:
            print(f"    drive-to-brake transitions: n={len(transition)} "
                  f"mean={statistics.mean(transition):+.5f} rad "
                  f"max_abs={max(abs(x) for x in transition):.5f} rad")


def report_encoder_slip(
    truth: list[dict[str, Any]], plant: list[dict[str, Any]],
    left_encoder: list[dict[str, Any]], right_encoder: list[dict[str, Any]],
    throttle_rows: list[dict[str, Any]],
) -> None:
    """Estimate rear longitudinal slip from measured wheel angle and GT body motion.

    This is an interval-average kinematic proxy, not Unity's internal contact
    slip: encoders are 40 Hz, body twist is at the COM, and each interval uses
    the known 25 ms simulator sampling period. Plant-reported slip is shown as
    a model-vs-proxy comparison, never labelled independent truth.
    """
    if min(len(truth), len(left_encoder), len(right_encoder)) < 3:
        return

    def keyed(rows: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
        return {round(row["stamp"], 6): row for row in rows}

    gt_by_t = keyed(truth)
    left_by_t, right_by_t = keyed(left_encoder), keyed(right_encoder)
    plant_by_t = keyed(plant)
    common = sorted(set(gt_by_t) & set(left_by_t) & set(right_by_t))
    rows: list[dict[str, float]] = []
    for previous_t, current_t in zip(common, common[1:]):
        if current_t - previous_t > 0.1:
            continue
        previous_gt, current_gt = gt_by_t[previous_t], gt_by_t[current_t]
        dt = ENCODER_SAMPLE_PERIOD_S
        enc_w = []
        for enc_by_t in (left_by_t, right_by_t):
            delta = enc_by_t[current_t]["angle"] - enc_by_t[previous_t]["angle"]
            enc_w.append(delta / dt)
        u_mid = 0.5 * (previous_gt["u"] + current_gt["u"])
        r_mid = 0.5 * (previous_gt["r"] + current_gt["r"])
        # Body coordinates: u forward, y left, positive r counter-clockwise.
        # The rear wheel centers are +/-0.118 m from the vehicle centerline.
        wheel_forward = (u_mid - r_mid * REAR_HALF_TRACK_M,
                         u_mid + r_mid * REAR_HALF_TRACK_M)
        interval: dict[str, float] = {
            "speed": abs(u_mid),
            "throttle": float((nearest_by_stamp(
                throttle_rows, current_gt["bag_time"]) or {}).get("value", math.nan)),
        }
        for side, index, vx in (("left", 2, wheel_forward[0]),
                                ("right", 3, wheel_forward[1])):
            surface = WHEEL_RADIUS_M * enc_w[0 if side == "left" else 1]
            slip_guide = ((surface - vx) / abs(vx)) if abs(vx) >= 0.5 else math.nan
            # PhysX's active-torque branch uses max(|v|, |R*omega|)+0.1m/s.
            slip_physx_active = (surface - vx) / (max(abs(vx), abs(surface)) + 0.1)
            slip_physx_coast = (surface - vx) / max(4.0, abs(vx), abs(surface))
            interval[f"{side}_surface_minus_body"] = surface - vx
            interval[f"{side}_slip_guide"] = slip_guide
            interval[f"{side}_slip_physx_active"] = slip_physx_active
            interval[f"{side}_slip_physx_coast"] = slip_physx_coast
            p = plant_by_t.get(current_t)
            if p:
                interval[f"{side}_reported_slip"] = p["long_slip"][index]
                interval[f"{side}_model_omega"] = p["wheel_omega"][index]
                interval[f"{side}_encoder_omega"] = enc_w[0 if side == "left" else 1]
                previous_plant = plant_by_t.get(previous_t)
                if previous_plant:
                    plant_delta = math.remainder(
                        p["wheel_angle"][index] - previous_plant["wheel_angle"][index],
                        10.0 * math.pi)
                    interval[f"{side}_model_interval_omega"] = plant_delta / dt
                    interval[f"{side}_encoder_minus_model_omega"] = (
                        enc_w[0 if side == "left" else 1] - plant_delta / dt)
                interval[f"{side}_model_slip_minus_active"] = (
                    p["long_slip"][index] - slip_physx_active)
        rows.append(interval)

    if not rows:
        return
    print("\nRear-wheel longitudinal-slip diagnostic (40 Hz encoder/GT proxy):")
    print("  This is not direct Unity contact-slip truth; wheel/body mismatch is interval-averaged.")
    print(f"  matched intervals={len(rows)}; encoder omega uses fixed {ENCODER_SAMPLE_PERIOD_S*1000:.0f} ms")

    def summarize(label: str, selected: list[dict[str, float]]) -> None:
        if not selected:
            return
        print(f"  {label}: n={len(selected)}")
        for side in ("left", "right"):
            prefix = f"{side}_"
            surface = [x[prefix + "surface_minus_body"] for x in selected]
            print(f"    {side}: Rω−v_wheel bias={statistics.mean(surface):+.3f} "
                  f"RMSE={math.sqrt(statistics.mean(x*x for x in surface)):.3f} m/s")
            for suffix in ("slip_guide", "slip_physx_active", "slip_physx_coast"):
                values = [x[prefix + suffix] for x in selected
                          if math.isfinite(x[prefix + suffix])]
                if values:
                    print(f"      {suffix}: median={statistics.median(values):+.3f} "
                          f"p95_abs={percentile([abs(v) for v in values], 0.95):.3f}")
            residual_key = prefix + "model_slip_minus_active"
            residuals = [x[residual_key] for x in selected if residual_key in x]
            if residuals:
                print(f"      source-plant slip minus encoder/GT active-slip proxy: "
                      f"bias={statistics.mean(residuals):+.3f} "
                      f"RMSE={math.sqrt(statistics.mean(x*x for x in residuals)):.3f}")
            omega_residual_key = prefix + "encoder_minus_model_omega"
            omega_residuals = [x[omega_residual_key] for x in selected
                               if omega_residual_key in x]
            if omega_residuals:
                print(f"      encoderΔθ/25ms minus source-plant Δθ/25ms: "
                      f"bias={statistics.mean(omega_residuals):+.2f} "
                      f"RMSE={math.sqrt(statistics.mean(x*x for x in omega_residuals)):.2f} rad/s")

    summarize("all", rows)
    summarize("zero-throttle/brake command", [x for x in rows if x["throttle"] < 0.01])
    summarize("positive throttle", [x for x in rows if x["throttle"] >= 0.01])
    summarize("wheel/body speed mismatch >0.5 m/s", [
        x for x in rows if max(abs(x["left_surface_minus_body"]),
                               abs(x["right_surface_minus_body"])) > 0.5])


def report_imu_velocity_propagation(
    truth: list[dict[str, Any]], odom: list[dict[str, Any]],
    imu: list[dict[str, Any]], odom_diagnostics: list[dict[str, Any]],
    throttle_feedback: list[dict[str, Any]],
) -> None:
    """Replay explicit Unity body-frame kinematics with sampled IMU accel."""
    gt_by_stamp = {round(row["stamp"], 6): row for row in truth}
    odom_by_stamp = {round(row["stamp"], 6): row for row in odom}
    imu_by_stamp = {round(row["stamp"], 6): row for row in imu}
    common = sorted(set(gt_by_stamp) & set(odom_by_stamp) & set(imu_by_stamp))
    if len(common) < 3:
        return
    # The body-frame velocity equations follow rotating-frame kinematics:
    # du/dt = a_x + r*v, dv/dt = a_y - r*u. IMU.cs publishes Rigidbody
    # acceleration rotated into the vehicle frame. Midpoint integration uses
    # adjacent 40 Hz samples; it uses no truth after initial alignment.
    first_odom = odom_by_stamp[common[0]]
    first_imu = imu_by_stamp[common[0]]
    u_hat = first_odom["u"]
    r0 = first_imu["yaw_rate"]
    v_com_hat = first_odom["v"] + r0 * COM_TO_REAR_AXLE_M
    estimates = [(u_hat, v_com_hat - r0 * COM_TO_REAR_AXLE_M)]
    one_step_errors: list[tuple[float, float]] = []
    plant_increment_errors: list[float] = []
    accel_increment_errors: list[float] = []
    fused_rollouts = {gain: [odom_by_stamp[common[0]]["u"]]
                      for gain in (0.0, 0.25, 0.5, 0.75, 1.0)}
    state_corrections = {gain: [odom_by_stamp[common[0]]["u"]]
                         for gain in (0.05, 0.10, 0.25, 0.50, 0.75, 1.0)}
    speed_conditioned_correction = [odom_by_stamp[common[0]]["u"]]
    gated_rollouts = {name: [odom_by_stamp[common[0]]["u"]]
                      for name in ("brake", "encoder_slip_0p5", "brake_or_slip")}
    gated_counts = Counter()
    for i in range(1, len(common)):
        previous, current = (imu_by_stamp[common[i - 1]], imu_by_stamp[common[i]])
        dt = (common[i] - common[i - 1])
        if not 0.0 < dt <= 0.1:
            estimates.append((u_hat, v_com_hat - current["yaw_rate"] * COM_TO_REAR_AXLE_M))
            continue
        gt_prev, gt_now = gt_by_stamp[common[i - 1]], gt_by_stamp[common[i]]
        odom_prev = odom_by_stamp[common[i - 1]]
        odom_now = odom_by_stamp[common[i]]
        r_mid = 0.5 * (previous["yaw_rate"] + current["yaw_rate"])
        ax_mid = 0.5 * (previous["ax"] + current["ax"])
        ay_mid = 0.5 * (previous["ay"] + current["ay"])
        u_stage = u_hat + 0.5 * dt * (previous["ax"] + previous["yaw_rate"] * v_com_hat)
        v_stage = v_com_hat + 0.5 * dt * (previous["ay"] - previous["yaw_rate"] * u_hat)
        u_hat += dt * (ax_mid + r_mid * v_stage)
        v_com_hat += dt * (ay_mid - r_mid * u_stage)
        rear_hat = v_com_hat - current["yaw_rate"] * COM_TO_REAR_AXLE_M
        estimates.append((u_hat, rear_hat))
        true_delta_u = gt_now["u"] - gt_prev["u"]
        imu_delta_u = dt * (ax_mid + r_mid * (
            odom_prev["v"] + previous["yaw_rate"] * COM_TO_REAR_AXLE_M))
        # Current raw source-plant output is stored in the diagnostic array,
        # sequence-paired with this 40 Hz packet. Its increment starts from
        # the previous sensor-corrected production state.
        current_diag = odom_diagnostics[i] if len(odom_diagnostics) == len(common) else {}
        has_raw_plant = (
            ("odom_v6" in current_diag.get("label", "") or
             "sensor_corrected_source_derived_odom_v" in current_diag.get("label", "")) and
            len(current_diag.get("values", [])) >= 35
        )
        raw_model_u = current_diag["values"][10] if has_raw_plant else odom_now["u"]
        model_delta_u = raw_model_u - odom_prev["u"]
        diag_values = current_diag.get("values", [])
        encoder_speed = diag_values[7] if has_raw_plant else math.nan
        feedback = nearest_by_stamp(throttle_feedback, gt_now["bag_time"])
        braking = feedback is not None and float(feedback["value"]) <= 0.005
        encoder_slip = math.isfinite(encoder_speed) and abs(encoder_speed - raw_model_u) > 0.5
        gates = {
            "brake": braking,
            "encoder_slip_0p5": encoder_slip,
            "brake_or_slip": braking or encoder_slip,
        }
        for name, active in gates.items():
            if active:
                gated_counts[name] += 1
            gated_rollouts[name].append(gated_rollouts[name][-1] +
                (imu_delta_u if active else model_delta_u))
        plant_increment_errors.append(model_delta_u - true_delta_u)
        accel_increment_errors.append(imu_delta_u - true_delta_u)
        for gain, rollout in fused_rollouts.items():
            rollout.append(rollout[-1] + (1.0 - gain) *
                           model_delta_u + gain * imu_delta_u)
        for gain, rollout in state_corrections.items():
            accel_prediction = rollout[-1] + imu_delta_u
            rollout.append(raw_model_u + gain * (accel_prediction - raw_model_u))
        # Candidate only: the model is trusted more at low speed, where the
        # current practice capture shows lower incremental model error; use a
        # smoothly increasing IMU correction as speed rises.  The two scale
        # constants are exposed here for paired-run evaluation, not hidden in
        # production code.
        speed_gain = 0.25 + 0.5 * min(abs(odom_prev["u"]) / 4.0, 1.0)
        accel_prediction = speed_conditioned_correction[-1] + imu_delta_u
        speed_conditioned_correction.append(
            raw_model_u + speed_gain * (accel_prediction - raw_model_u))
        # Oracle-initialized one-step error isolates sampled-acceleration
        # integration from accumulated state/initialization error.
        gt_u0, gt_v0 = gt_prev["u"], gt_prev["v"]
        gt_u_stage = gt_u0 + 0.5 * dt * (previous["ax"] + previous["yaw_rate"] * gt_v0)
        gt_v_stage = gt_v0 + 0.5 * dt * (previous["ay"] - previous["yaw_rate"] * gt_u0)
        pred_u = gt_u0 + dt * (ax_mid + r_mid * gt_v_stage)
        pred_v = gt_v0 + dt * (ay_mid - r_mid * gt_u_stage)
        one_step_errors.append((pred_u - gt_now["u"],
                                (pred_v - gt_now["v"]) - current["yaw_rate"] * COM_TO_REAR_AXLE_M))
    print("\nIMU-only velocity propagation candidate (midpoint, 25 ms samples):")
    for label, errors in (
        ("one-step, initialized from each prior GT state", one_step_errors),
        ("recursive rollout from first production odom state", [
            (estimate[0] - gt_by_stamp[key]["u"],
             estimate[1] - (gt_by_stamp[key]["v"] - gt_by_stamp[key]["r"] * COM_TO_REAR_AXLE_M))
            for key, estimate in zip(common[1:], estimates[1:])]),
    ):
        if not errors:
            continue
        print(f"  {label}: n={len(errors)}")
        for index, field in ((0, "u"), (1, "v_rear")):
            values = [error[index] for error in errors]
            print(f"    {field}: bias={statistics.mean(values):+.4f} "
                  f"RMSE={math.sqrt(statistics.mean(x*x for x in values)):.4f} "
                  f"p95_abs={percentile([abs(x) for x in values], 0.95):.4f} "
                  f"max_abs={max(abs(x) for x in values):.4f} m/s")
    print("\nOne-step forward-velocity increment comparison:")
    for label, errors in (("native PhysX increment", plant_increment_errors),
                          ("IMU kinematic increment", accel_increment_errors)):
        print(f"  {label}: bias={statistics.mean(errors):+.4f} "
              f"RMSE={math.sqrt(statistics.mean(x*x for x in errors)):.4f} "
              f"p95_abs={percentile([abs(x) for x in errors], 0.95):.4f} m/s")
    print("  recursive increment blend (diagnostic gain sweep, not a fitted model):")
    for gain, rollout in fused_rollouts.items():
        errors = [value - gt_by_stamp[key]["u"]
                  for value, key in zip(rollout[1:], common[1:])]
        print(f"    IMU delta weight={gain:.2f}: RMSE="
              f"{math.sqrt(statistics.mean(x*x for x in errors)):.4f} "
              f"end_error={errors[-1]:+.3f} m/s")
    print("  plant-state prediction corrected toward integrated IMU delta:")
    for gain, rollout in state_corrections.items():
        errors = [value - gt_by_stamp[key]["u"]
                  for value, key in zip(rollout[1:], common[1:])]
        print(f"    correction gain={gain:.2f}: RMSE="
              f"{math.sqrt(statistics.mean(x*x for x in errors)):.4f} "
              f"end_error={errors[-1]:+.3f} m/s")
    conditioned_errors = [value - gt_by_stamp[key]["u"]
                          for value, key in zip(
                              speed_conditioned_correction[1:], common[1:])]
    print("    speed-conditioned correction (0.25 + 0.50*min(|u|/4,1)): "
          f"RMSE={math.sqrt(statistics.mean(x*x for x in conditioned_errors)):.4f} "
          f"end_error={conditioned_errors[-1]:+.3f} m/s")
    print("  causal IMU delta substitution only in physically flagged intervals:")
    for name, rollout in gated_rollouts.items():
        errors = [value - gt_by_stamp[key]["u"]
                  for value, key in zip(rollout[1:], common[1:])]
        print(f"    {name}: flagged={gated_counts[name]} RMSE="
              f"{math.sqrt(statistics.mean(x*x for x in errors)):.4f} "
              f"end_error={errors[-1]:+.3f} m/s")


def load_trajectory(path: Path | None) -> tuple[list[dict[str, float]], float] | None:
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    first = rows[0]
    s_name = "# s_m" if "# s_m" in first else "s_m"
    points = [
        {
            "s": float(row[s_name]),
            "x": float(row["x_m"]),
            "y": float(row["y_m"]),
            "psi": float(row["psi_rad"]),
        }
        for row in rows
    ]
    return points, points[-1]["s"] + points[1]["s"] - points[0]["s"]


def trajectory_pose(
    trajectory: tuple[list[dict[str, float]], float], progress: float,
) -> tuple[float, float, float]:
    points, lap_length = trajectory
    s_value = progress % lap_length
    positions = [point["s"] for point in points]
    index = max(0, min(len(points) - 2, bisect.bisect_right(positions, s_value) - 1))
    a, b = points[index], points[index + 1]
    ratio = (s_value - a["s"]) / max(1e-9, b["s"] - a["s"])
    dpsi = normalized_angle(b["psi"] - a["psi"])
    psi = a["psi"] + ratio * dpsi
    x = a["x"] + ratio * (b["x"] - a["x"])
    y = a["y"] + ratio * (b["y"] - a["y"])
    return x, y, psi


def report_horizon_errors(
    truth: list[dict[str, Any]], diagnostics: list[dict[str, Any]],
    trajectory: tuple[list[dict[str, float]], float] | None,
) -> None:
    print(
        "horizon comparison note: closed-loop logs replan controls every 25 ms; "
        "beyond already-queued commands these errors are not fixed-input plant accuracy."
    )
    truth_times = [row["stamp"] for row in truth]
    errors: dict[int, list[dict[str, float]]] = {}
    for row in diagnostics:
        diag = row.get("json", {})
        control_ns = diag.get("control_ros_stamp_ns")
        if not isinstance(control_ns, int):
            continue
        start_time = control_ns * 1e-9
        control_prediction = diag.get("control_time_prediction", {})
        map_start = control_prediction.get("predicted_map_pose")
        start_index = bisect.bisect_left(truth_times, start_time)
        candidates = [i for i in (start_index - 1, start_index) if 0 <= i < len(truth)]
        if not candidates:
            continue
        start_truth = min(candidates, key=lambda i: abs(truth_times[i] - start_time))
        actual_start = truth[start_truth]
        for prediction in diag.get("predictions", []):
            n = prediction.get("n")
            state = prediction.get("state")
            if not isinstance(n, int) or not isinstance(state, list) or len(state) < 5:
                continue
            target_time = start_time + n * 0.025
            idx = bisect.bisect_left(truth_times, target_time)
            candidates = [i for i in (idx - 1, idx) if 0 <= i < len(truth)]
            if not candidates:
                continue
            target_index = min(candidates, key=lambda i: abs(truth_times[i] - target_time))
            if abs(truth_times[target_index] - target_time) > 0.018:
                continue
            actual = truth[target_index]
            item = {
                "dt": abs(truth_times[target_index] - target_time),
                "u": state[2] - actual["u"],
                "v": state[3] - actual["v"],
                "r": state[4] - actual["r"],
                "start_speed": math.hypot(actual_start["u"], actual_start["v"]),
            }
            if trajectory and isinstance(map_start, list) and len(map_start) == 3:
                rx, ry, rpsi = trajectory_pose(trajectory, prediction["s_m"])
                predicted_map_x = rx - state[0] * math.sin(rpsi)
                predicted_map_y = ry + state[0] * math.cos(rpsi)
                delta_yaw = normalized_angle(actual_start["yaw"] - map_start[2])
                dx, dy = predicted_map_x - map_start[0], predicted_map_y - map_start[1]
                predicted_x = actual_start["x"] + math.cos(delta_yaw) * dx - math.sin(delta_yaw) * dy
                predicted_y = actual_start["y"] + math.sin(delta_yaw) * dx + math.cos(delta_yaw) * dy
                item["xy"] = math.hypot(
                    predicted_x - actual["x"], predicted_y - actual["y"],
                )
                predicted_yaw = actual_start["yaw"] + normalized_angle(
                    rpsi + state[1] - map_start[2]
                )
                item["yaw"] = normalized_angle(predicted_yaw - actual["yaw"])
            errors.setdefault(n, []).append(item)

    for n, rows in sorted(errors.items()):
        print(
            f"horizon {n} ({n*25}ms): n={len(rows)} "
            f"time_align_p95={percentile([x['dt'] for x in rows], 0.95)*1000:.2f}ms"
        )
        for field in ("u", "v", "r", "xy", "yaw"):
            values = [row[field] for row in rows if field in row]
            if not values:
                continue
            absolute = [abs(value) for value in values]
            print(
                f"  {field}: RMSE={math.sqrt(statistics.mean(x*x for x in values)):.4f} "
                f"p95_abs={percentile(absolute, 0.95):.4f} max_abs={max(absolute):.4f}"
            )
        for lower, upper in ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, math.inf)):
            subset = [row["u"] for row in rows if lower <= row["start_speed"] < upper]
            if subset:
                rmse = math.sqrt(statistics.mean(value * value for value in subset))
                print(f"  speed_bin [{lower:g},{upper:g}): u_RMSE={rmse:.4f} n={len(subset)}")


def report_source_plant_rollout(
    truth: list[dict[str, Any]], plant: list[dict[str, Any]],
    odom_diagnostics: list[dict[str, Any]], library_path: Path,
    data_imu: list[dict[str, Any]],
    yaw_residual_gain: float = 0.0,
    yaw_residual_decay: float = 1.0,
    oracle_motion_initial_state: bool = False,
) -> None:
    """Replay only odom plant state and queued commands through the C++ MPC plant.

    Starts from a recorded UnityPlantState, then advances with the recorded
    slot-1 actuator command sequence. It never reads future measured states
    during a rollout; simulator truth is used only to score each endpoint.
    The optional oracle-motion diagnostic replaces only initial u/v/r with
    same-epoch Unity truth. Hidden plant states and all future inputs remain
    unchanged; this is diagnostic only and must never be used in production.
    """
    if not truth or not plant:
        print("source-plant rollout: requires both Unity truth and plant-state streams")
        return
    if not library_path.is_file():
        print(f"source-plant rollout: shared model not found: {library_path}")
        return
    try:
        model = ctypes.CDLL(str(library_path.resolve()))
    except OSError as error:
        print(f"source-plant rollout: cannot load model library: {error}")
        return
    step = model.unity_source_plant_step_c
    step.argtypes = [ctypes.POINTER(UnitySourcePlantInputC), ctypes.c_double,
                     ctypes.POINTER(UnitySourcePlantOutputC)]
    step.restype = ctypes.c_int

    truth_by_stamp = {round(row["stamp"], 6): row for row in truth}
    plant_by_stamp = {round(row["stamp"], 6): row for row in plant}
    # Float64MultiArray has no ROS header, so its first value is the local
    # observer clock (which starts at zero), not the bag/Unity epoch. Pair it
    # to the plant publication by recorder receive time; both are emitted in
    # sequence from the same sensor callback.
    raw_yaw_forecast_by_stamp: dict[float, float] = {}
    ordered_plant = sorted(plant, key=lambda row: row["bag_time"])
    ordered_diagnostics = sorted(odom_diagnostics, key=lambda row: row["bag_time"])
    for plant_row, diagnostic_row in zip(ordered_plant, ordered_diagnostics):
        if (abs(plant_row["bag_time"] - diagnostic_row["bag_time"]) <= 0.005 and
                len(diagnostic_row.get("values", [])) >= 13 and
                diagnostic_row.get("label") ==
                "sensor_corrected_source_derived_odom_v3"):
            raw_yaw_forecast_by_stamp[round(plant_row["stamp"], 6)] = \
                diagnostic_row["values"][12]
    times = sorted(set(truth_by_stamp) & set(plant_by_stamp))
    if len(times) < 31:
        print(f"source-plant rollout: only {len(times)} aligned epochs")
        return

    # Tire/contact forces depend on attitude and heave through wheel raycasts,
    # jounce, and per-wheel normal load. Check these online-estimated states
    # against the independent Unity pose/IMU channels before attributing yaw
    # misses solely to tire equations. Heave is relative to the first epoch.
    first_truth = truth_by_stamp[times[0]]
    first_plant = plant_by_stamp[times[0]]
    height_offset = first_truth["z"] - first_plant["body_height"]
    hidden_state_errors: dict[str, list[float]] = {
        "height_delta": [], "roll": [], "pitch": [],
        "roll_rate": [], "pitch_rate": [],
    }
    imu_by_stamp = {round(row["stamp"], 6): row for row in data_imu}
    for time in times:
        truth_row = truth_by_stamp[time]
        plant_row = plant_by_stamp[time]
        hidden_state_errors["height_delta"].append(
            plant_row["body_height"] + height_offset - truth_row["z"])
        hidden_state_errors["roll"].append(plant_row["roll"] - truth_row["roll"])
        hidden_state_errors["pitch"].append(plant_row["pitch"] - truth_row["pitch"])
        imu_row = imu_by_stamp.get(time)
        if imu_row is not None:
            hidden_state_errors["roll_rate"].append(
                plant_row["roll_rate"] - imu_row["roll_rate"])
            hidden_state_errors["pitch_rate"].append(
                plant_row["pitch_rate"] - imu_row["pitch_rate"])
    print("  source hidden-state agreement against offline Unity pose/IMU "
          "(heave is relative to first aligned epoch):")
    for name, values in hidden_state_errors.items():
        if not values:
            continue
        unit = "m" if name == "height_delta" else (
            "rad" if name in ("roll", "pitch") else "rad/s")
        print(f"    {name}: RMSE={math.sqrt(statistics.mean(v*v for v in values)):.5f}"
              f"{unit} p95_abs={percentile([abs(v) for v in values], 0.95):.5f}"
              f"{unit} max_abs={max(abs(v) for v in values):.5f}{unit}")

    def source_state(row: dict[str, Any]) -> list[float]:
        state = [0.0] * UNITY_SOURCE_PLANT_STATE_SIZE
        state[0] = row["u"]
        state[1] = row["v_com"]
        state[2] = row["r"]
        state[3:7] = row["wheel_omega"]
        state[7] = row["body_height"]
        state[8] = row["body_height_rate"]
        state[9] = row["roll"]
        state[10] = row["roll_rate"]
        state[11] = row["pitch"]
        state[12] = row["pitch_rate"]
        state[13] = row["physical_steering"]
        state[14:18] = row["suspension_jounce"]
        state[18:22] = row["low_forward_speed_time_s"]
        state[22:26] = row["low_side_speed_time_s"]
        return state

    def make_input(state: list[float], angles: list[float],
                   queue: dict[str, Any], history_initialized: bool,
                   command_slot: int = 1) -> UnitySourcePlantInputC:
        value = UnitySourcePlantInputC()
        value.state[:] = state
        value.wheel_rotation_angle_rad[:] = angles
        value.steering_target_rad = (
            queue[f"delayed_steering_{command_slot}"] * MAX_STEERING_RAD)
        value.throttle_normalized = queue[f"delayed_throttle_{command_slot}"]
        value.suspension_history_initialized = int(history_initialized)
        return value

    def add(errors: dict[str, list[float]], predicted: list[float],
            actual: dict[str, Any], predicted_yaw: float,
            predicted_pose_x: float, predicted_pose_y: float) -> None:
        errors["u"].append(predicted[0] - actual["u"])
        errors["v"].append(predicted[1] - actual["v"])
        errors["r"].append(predicted[2] - actual["r"])
        errors["yaw"].append(normalized_angle(predicted_yaw - actual["yaw"]))
        errors["xy"].append(math.hypot(
            predicted_pose_x - actual["x"],
            predicted_pose_y - actual["y"],
        ))

    def truth_pose_xy(row: dict[str, Any]) -> tuple[float, float]:
        return row["x"], row["y"]

    def integrate_interpolated_twist(
        x: float, y: float, u0: float, v0: float, yaw0: float,
        u1: float, v1: float, yaw1: float, dt: float,
    ) -> tuple[float, float]:
        # Diagnostic only: approximate pose propagation at 1 ms by linearly
        # interpolating the available 40 Hz endpoint state samples.
        substeps = max(1, round(dt / 0.001))
        yaw_delta = normalized_angle(yaw1 - yaw0)
        h = dt / substeps
        for substep in range(substeps):
            fraction = (substep + 0.5) / substeps
            u = u0 + fraction * (u1 - u0)
            v = v0 + fraction * (v1 - v0)
            yaw = yaw0 + fraction * yaw_delta
            x += h * (u * math.cos(yaw) - v * math.sin(yaw))
            y += h * (u * math.sin(yaw) + v * math.cos(yaw))
        return x, y

    # Check whether the simulator's published twist actually closes against
    # its published pose increments. This distinguishes plant rollout error
    # from a pose/twist frame or sampling-phase mismatch in the reference.
    closure_modes = {
        "body_twist_rotated_by_yaw": [],
        "body_twist_minus_yaw_rate_times_com_to_rear": [],
        "body_twist_plus_yaw_rate_times_com_to_rear": [],
        "twist_treated_as_world_xy": [],
        "body_lateral_sign_flipped": [],
    }
    closure_signed: dict[str, list[tuple[float, float, float]]] = {
        name: [] for name in closure_modes
    }
    for index in range(len(times) - 1):
        t0, t1 = times[index], times[index + 1]
        measured_dt = t1 - t0
        if abs(measured_dt - 0.025) > 0.004:
            continue
        dt = 0.025
        first, second = truth_by_stamp[t0], truth_by_stamp[t1]
        yaw_delta = normalized_angle(second["yaw"] - first["yaw"])
        yaw_mid = first["yaw"] + 0.5 * yaw_delta
        u_mid = 0.5 * (first["u"] + second["u"])
        v_mid = 0.5 * (first["v"] + second["v"])
        r_mid = 0.5 * (first["r"] + second["r"])
        x0, y0 = truth_pose_xy(first)
        x1, y1 = truth_pose_xy(second)
        dx, dy = x1 - x0, y1 - y0
        candidates = {
            "body_twist_rotated_by_yaw": (
                u_mid * math.cos(yaw_mid) - v_mid * math.sin(yaw_mid),
                u_mid * math.sin(yaw_mid) + v_mid * math.cos(yaw_mid)),
            "body_twist_minus_yaw_rate_times_com_to_rear": (
                u_mid * math.cos(yaw_mid) -
                (v_mid - r_mid * COM_TO_REAR_AXLE_M) * math.sin(yaw_mid),
                u_mid * math.sin(yaw_mid) +
                (v_mid - r_mid * COM_TO_REAR_AXLE_M) * math.cos(yaw_mid)),
            "body_twist_plus_yaw_rate_times_com_to_rear": (
                u_mid * math.cos(yaw_mid) -
                (v_mid + r_mid * COM_TO_REAR_AXLE_M) * math.sin(yaw_mid),
                u_mid * math.sin(yaw_mid) +
                (v_mid + r_mid * COM_TO_REAR_AXLE_M) * math.cos(yaw_mid)),
            "twist_treated_as_world_xy": (u_mid, v_mid),
            "body_lateral_sign_flipped": (
                u_mid * math.cos(yaw_mid) + v_mid * math.sin(yaw_mid),
                u_mid * math.sin(yaw_mid) - v_mid * math.cos(yaw_mid)),
        }
        for name, (vx, vy) in candidates.items():
            err_x, err_y = vx * dt - dx, vy * dt - dy
            closure_modes[name].append(math.hypot(err_x, err_y))
            closure_signed[name].append((
                err_x * math.cos(first["yaw"]) + err_y * math.sin(first["yaw"]),
                -err_x * math.sin(first["yaw"]) + err_y * math.cos(first["yaw"]),
                math.hypot(err_x, err_y)))

    horizons = (1, 4, 10, 20, 30)
    errors = {horizon: {field: [] for field in ("u", "v", "r", "yaw", "xy")}
              for horizon in horizons}
    grouped_yaw: dict[int, list[tuple[float, float, float]]] = {
        horizon: [] for horizon in horizons
    }
    position_decomposition: dict[int, list[dict[str, float]]] = {
        horizon: [] for horizon in horizons
    }
    fine_position_errors: dict[int, list[tuple[float, float, float]]] = {
        horizon: [] for horizon in horizons
    }
    oracle_position_errors: dict[int, list[tuple[float, float]]] = {
        horizon: [] for horizon in horizons
    }
    first_step_details: list[dict[str, float]] = []
    command_slot_errors = {
        slot: {field: [] for field in ("u", "v", "r", "steering")}
        for slot in (1, 2)
    }
    command_slot_regimes = {
        regime: {
            slot: {field: [] for field in ("r", "steering")}
            for slot in (1, 2)
        }
        for regime in ("high-steer", "high-steer-unwind", "high-steer-turn-in")
    }
    pathwise_position_maxima: list[float] = []
    skipped = 0
    residual_valid_windows = 0
    residual_enabled_windows = 0
    # Same-epoch one-step A/B: change only which already-logged queue slot is
    # supplied for the next 25 ms plant step. This directly tests command
    # phase against Unity without fitting dynamics or consulting future
    # truth as an input.
    for index in range(len(times) - 1):
        start_time, next_time = times[index], times[index + 1]
        if abs((next_time - start_time) - 0.025) > 0.004:
            continue
        start_plant = plant_by_stamp[start_time]
        start_truth = truth_by_stamp[start_time]
        next_truth = truth_by_stamp[next_time]
        next_plant = plant_by_stamp[next_time]
        current_steering = start_plant["physical_steering"]
        next_steering = next_plant["physical_steering"]
        steering_delta_magnitude = abs(next_steering) - abs(current_steering)
        regimes = []
        if abs(current_steering) >= 0.10:
            regimes.append("high-steer")
            if current_steering * next_steering > 0.0:
                if steering_delta_magnitude < -0.003:
                    regimes.append("high-steer-unwind")
                elif steering_delta_magnitude > 0.003:
                    regimes.append("high-steer-turn-in")
        initial_state = source_state(start_plant)
        if oracle_motion_initial_state:
            initial_state[0] = start_truth["u"]
            initial_state[1] = start_truth["v"]
            initial_state[2] = start_truth["r"]
        for slot in (1, 2):
            candidate_input = make_input(
                initial_state, list(start_plant["wheel_angle"]), start_plant,
                start_plant["suspension_history_initialized"], slot)
            candidate_output = UnitySourcePlantOutputC()
            if not step(ctypes.byref(candidate_input), 0.025,
                        ctypes.byref(candidate_output)):
                continue
            command_slot_errors[slot]["u"].append(
                candidate_output.state[0] - next_truth["u"])
            command_slot_errors[slot]["v"].append(
                candidate_output.state[1] - next_truth["v"])
            command_slot_errors[slot]["r"].append(
                candidate_output.state[2] - next_truth["r"])
            command_slot_errors[slot]["steering"].append(
                candidate_output.state[13] - next_steering)
            for regime in regimes:
                command_slot_regimes[regime][slot]["r"].append(
                    candidate_output.state[2] - next_truth["r"])
                command_slot_regimes[regime][slot]["steering"].append(
                    candidate_output.state[13] - next_steering)
    for start in range(0, len(times) - max(horizons)):
        # Only publish a window if its entire 750 ms input/truth interval is
        # available on the simulator's 40 Hz epoch grid.  Otherwise we'd keep
        # early-horizon scores from a window that failed at a later step.
        if any(abs((times[index + 1] - times[index]) - 0.025) > 0.004
               for index in range(start, start + max(horizons))):
            skipped += 1
            continue
        start_time = times[start]
        start_truth = truth_by_stamp[start_time]
        start_plant = plant_by_stamp[start_time]
        state = source_state(start_plant)
        if oracle_motion_initial_state:
            # Isolate whether a short-horizon yaw miss is caused by the
            # observer's initial motion-state error, without granting the
            # plant any future truth or changing its hidden PhysX states.
            state[0] = start_truth["u"]
            state[1] = start_truth["v"]
            state[2] = start_truth["r"]
        angles = list(start_plant["wheel_angle"])
        history_initialized = start_plant["suspension_history_initialized"]
        predicted_yaw = start_truth["yaw"]
        predicted_pose_x, predicted_pose_y = truth_pose_xy(start_truth)
        fine_predicted_pose_x, fine_predicted_pose_y = predicted_pose_x, predicted_pose_y
        gt_heading_predicted_motion_x = predicted_pose_x
        gt_heading_predicted_motion_y = predicted_pose_y
        predicted_heading_gt_motion_x = predicted_pose_x
        predicted_heading_gt_motion_y = predicted_pose_y
        truth_integrated_pose_x, truth_integrated_pose_y = truth_pose_xy(start_truth)
        fine_truth_pose_x, fine_truth_pose_y = truth_integrated_pose_x, truth_integrated_pose_y
        previous_truth = start_truth
        previous_r = state[2]
        start_speed = math.hypot(start_truth["u"], start_truth["v"])
        start_steering = abs(start_plant["physical_steering"])
        # advance() forecasts this callback's epoch before correct() applies
        # its same-epoch gyro sample. Thus the causal innovation is the raw
        # forecast paired to this publication minus this epoch's gyro. It
        # never reads future simulator truth.
        raw_forecast = raw_yaw_forecast_by_stamp.get(round(start_time, 6))
        yaw_residual = (
            raw_forecast - start_plant["gyro_r"]
            if raw_forecast is not None else 0.0)
        residual_valid_windows += raw_forecast is not None
        yaw_residual_enabled = (
            raw_forecast is not None and yaw_residual_gain != 0.0 and
            2.8 <= abs(state[0]) <= 5.0 and
            0.10 <= start_steering <= 0.31 and
            state[2] * yaw_residual < 0.0)
        residual_enabled_windows += yaw_residual_enabled
        window_max_abs_yaw_error = 0.0
        window_max_abs_steering = start_steering
        window_yaw_rate_errors: list[float] = []
        window_position_errors: list[float] = []
        window_pose_trace: list[dict[str, float]] = []
        window_max_jounce_m = max(state[14:18], default=-math.inf)
        window_suspension_stop_epochs = 0
        valid = True
        window_errors = {horizon: {field: [] for field in ("u", "v", "r", "yaw", "xy")}
                         for horizon in horizons}
        window_grouped_yaw: dict[int, list[tuple[float, float, float]]] = {
            horizon: [] for horizon in horizons
        }
        for horizon in range(1, max(horizons) + 1):
            current = plant_by_stamp[times[start + horizon - 1]]
            target_time = times[start + horizon]
            measured_dt = target_time - times[start + horizon - 1]
            if abs(measured_dt - 0.025) > 0.004:
                valid = False
                break
            dt = 0.025
            input_value = make_input(state, angles, current, history_initialized)
            output = UnitySourcePlantOutputC()
            if not step(ctypes.byref(input_value), 0.025, ctypes.byref(output)):
                valid = False
                break
            previous_state = state
            state = list(output.state)
            # PxVehicleUpdate activates its max-compression stop row when raw
            # ray jounce exceeds 25 mm. The carried public plant state is
            # clamped, so equality is a conservative indicator, not exact
            # contact telemetry.
            window_max_jounce_m = max(window_max_jounce_m, *state[14:18])
            if any(jounce >= 0.025 - 1.0e-6 for jounce in state[14:18]):
                window_suspension_stop_epochs += 1
            raw_r_next = state[2]
            predicted_speed = abs(state[0])
            predicted_steering = abs(state[13])
            residual_supported = (
                yaw_residual_enabled and
                2.8 <= predicted_speed <= 5.0 and
                0.10 <= predicted_steering <= 0.31 and
                state[13] * previous_state[13] > 0.0 and
                raw_r_next * yaw_residual < 0.0)
            residual_weight = yaw_residual_decay ** (horizon - 1)
            if residual_supported:
                state[2] = raw_r_next - (
                    yaw_residual_gain * residual_weight * yaw_residual)
            angles = list(output.wheel_rotation_angle_rad)
            history_initialized = bool(output.suspension_history_initialized)
            predicted_yaw_start = predicted_yaw
            yaw_delta = output.integrated_yaw_delta_rad
            body_forward_delta = output.integrated_body_forward_displacement_m
            body_lateral_delta = output.integrated_body_lateral_displacement_m
            if residual_supported:
                # Match the production rollout's causal endpoint correction
                # over this fixed 25 ms interval while retaining source-plant
                # intermediate integration for translation and heading.
                delta_r = state[2] - raw_r_next
                yaw_delta += delta_r * dt
                body_lateral_delta -= delta_r * COM_TO_REAR_AXLE_M * dt
            predicted_pose_x_start = predicted_pose_x
            predicted_pose_y_start = predicted_pose_y
            predicted_pose_x += (
                body_forward_delta * math.cos(predicted_yaw_start) -
                body_lateral_delta * math.sin(predicted_yaw_start))
            predicted_pose_y += (
                body_forward_delta * math.sin(predicted_yaw_start) +
                body_lateral_delta * math.cos(predicted_yaw_start))
            predicted_yaw_mid = predicted_yaw_start + 0.5 * yaw_delta
            predicted_yaw = predicted_yaw_start + yaw_delta
            fine_predicted_pose_x, fine_predicted_pose_y = integrate_interpolated_twist(
                fine_predicted_pose_x, fine_predicted_pose_y,
                previous_state[0],
                previous_state[1] - previous_r * COM_TO_REAR_AXLE_M,
                predicted_yaw_start, state[0],
                state[1] - state[2] * COM_TO_REAR_AXLE_M,
                predicted_yaw, dt)
            previous_r = state[2]
            actual = truth_by_stamp[target_time]
            position_error = math.hypot(
                predicted_pose_x - actual["x"],
                predicted_pose_y - actual["y"])
            window_position_errors.append(position_error)
            previous_truth_for_step = previous_truth
            step_error_world_x = (
                predicted_pose_x - predicted_pose_x_start -
                (actual["x"] - previous_truth_for_step["x"]))
            step_error_world_y = (
                predicted_pose_y - predicted_pose_y_start -
                (actual["y"] - previous_truth_for_step["y"]))
            start_yaw_cos = math.cos(start_truth["yaw"])
            start_yaw_sin = math.sin(start_truth["yaw"])
            truth_yaw_delta = normalized_angle(actual["yaw"] - previous_truth["yaw"])
            truth_yaw_mid = previous_truth["yaw"] + 0.5 * truth_yaw_delta
            truth_u_mid = 0.5 * (previous_truth["u"] + actual["u"])
            truth_rear_v_previous = (
                previous_truth["v"] - previous_truth["r"] * COM_TO_REAR_AXLE_M)
            truth_rear_v_current = actual["v"] - actual["r"] * COM_TO_REAR_AXLE_M
            truth_v_mid = 0.5 * (truth_rear_v_previous + truth_rear_v_current)
            gt_heading_predicted_motion_x += 0.025 * (
                truth_u_mid * math.cos(truth_yaw_mid) -
                truth_v_mid * math.sin(truth_yaw_mid))
            gt_heading_predicted_motion_y += 0.025 * (
                truth_u_mid * math.sin(truth_yaw_mid) +
                truth_v_mid * math.cos(truth_yaw_mid))
            predicted_heading_gt_motion_x += 0.025 * (
                truth_u_mid * math.cos(predicted_yaw_mid) -
                truth_v_mid * math.sin(predicted_yaw_mid))
            predicted_heading_gt_motion_y += 0.025 * (
                truth_u_mid * math.sin(predicted_yaw_mid) +
                truth_v_mid * math.cos(predicted_yaw_mid))
            truth_integrated_pose_x += dt * (
                truth_u_mid * math.cos(truth_yaw_mid) -
                truth_v_mid * math.sin(truth_yaw_mid))
            truth_integrated_pose_y += dt * (
                truth_u_mid * math.sin(truth_yaw_mid) +
                truth_v_mid * math.cos(truth_yaw_mid))
            truth_twist_step_x = dt * (
                truth_u_mid * math.cos(truth_yaw_mid) -
                truth_v_mid * math.sin(truth_yaw_mid))
            truth_twist_step_y = dt * (
                truth_u_mid * math.sin(truth_yaw_mid) +
                truth_v_mid * math.cos(truth_yaw_mid))
            fine_truth_pose_x, fine_truth_pose_y = integrate_interpolated_twist(
                fine_truth_pose_x, fine_truth_pose_y,
                previous_truth_for_step["u"], truth_rear_v_previous,
                previous_truth_for_step["yaw"], actual["u"],
                truth_rear_v_current, actual["yaw"], dt)
            previous_truth = actual
            window_yaw_rate_errors.append(state[2] - actual["r"])
            window_pose_trace.append({
                "step": float(horizon),
                "xy_error_m": position_error,
                "step_error_forward_m": (
                    step_error_world_x * start_yaw_cos +
                    step_error_world_y * start_yaw_sin),
                "step_error_lateral_m": (
                    -step_error_world_x * start_yaw_sin +
                    step_error_world_y * start_yaw_cos),
                "plant_vs_truth_twist_forward_m": (
                    (predicted_pose_x - predicted_pose_x_start - truth_twist_step_x) *
                    start_yaw_cos +
                    (predicted_pose_y - predicted_pose_y_start - truth_twist_step_y) *
                    start_yaw_sin),
                "plant_vs_truth_twist_lateral_m": (
                    -(predicted_pose_x - predicted_pose_x_start - truth_twist_step_x) *
                    start_yaw_sin +
                    (predicted_pose_y - predicted_pose_y_start - truth_twist_step_y) *
                    start_yaw_cos),
                "truth_twist_vs_pose_forward_m": (
                    (truth_twist_step_x - (actual["x"] - previous_truth_for_step["x"])) *
                    start_yaw_cos +
                    (truth_twist_step_y - (actual["y"] - previous_truth_for_step["y"])) *
                    start_yaw_sin),
                "truth_twist_vs_pose_lateral_m": (
                    -(truth_twist_step_x - (actual["x"] - previous_truth_for_step["x"])) *
                    start_yaw_sin +
                    (truth_twist_step_y - (actual["y"] - previous_truth_for_step["y"])) *
                    start_yaw_cos),
                "yaw_error_rad": normalized_angle(
                    predicted_yaw - actual["yaw"]),
                "yaw_rate_error_radps": state[2] - actual["r"],
                "speed_error_mps": state[0] - actual["u"],
                "lateral_speed_error_mps": state[1] - actual["v"],
                "physical_steering_rad": state[13],
                "queued_steering_rad":
                    current["delayed_steering_1"] * MAX_STEERING_RAD,
                "queued_throttle": current["delayed_throttle_1"],
            })
            window_max_abs_yaw_error = max(
                window_max_abs_yaw_error,
                abs(normalized_angle(predicted_yaw - actual["yaw"])))
            window_max_abs_steering = max(
                window_max_abs_steering,
                abs(current["physical_steering"]))
            if horizon in errors:
                add(window_errors[horizon], state, actual, predicted_yaw,
                    predicted_pose_x, predicted_pose_y)
                yaw0 = start_truth["yaw"]
                c0, s0 = math.cos(yaw0), math.sin(yaw0)
                pred_dx = predicted_pose_x - actual["x"]
                pred_dy = predicted_pose_y - actual["y"]
                truth_dx = truth_integrated_pose_x - actual["x"]
                truth_dy = truth_integrated_pose_y - actual["y"]
                model_dx = predicted_pose_x - truth_integrated_pose_x
                model_dy = predicted_pose_y - truth_integrated_pose_y
                position_decomposition[horizon].append({
                    "pred_world_x": pred_dx,
                    "pred_world_y": pred_dy,
                    "pred_forward": pred_dx * c0 + pred_dy * s0,
                    "pred_lateral": -pred_dx * s0 + pred_dy * c0,
                    "truth_twist_world_x": truth_dx,
                    "truth_twist_world_y": truth_dy,
                    "truth_twist_forward": truth_dx * c0 + truth_dy * s0,
                    "truth_twist_lateral": -truth_dx * s0 + truth_dy * c0,
                    "plant_vs_twist_world_x": model_dx,
                    "plant_vs_twist_world_y": model_dy,
                    "plant_vs_twist_forward": model_dx * c0 + model_dy * s0,
                    "plant_vs_twist_lateral": -model_dx * s0 + model_dy * c0,
                    "xy_m": math.hypot(pred_dx, pred_dy),
                    "start_speed_mps": start_speed,
                    "start_steering_rad": start_steering,
                    "max_steering_rad": window_max_abs_steering,
                    "max_abs_yaw_error_rad": window_max_abs_yaw_error,
                    "end_abs_yaw_error_rad": abs(normalized_angle(
                        predicted_yaw - actual["yaw"])),
                    "end_u_error_mps": state[0] - actual["u"],
                    "end_v_error_mps": state[1] - actual["v"],
                    "end_r_error_radps": state[2] - actual["r"],
                    "max_jounce_m": window_max_jounce_m,
                    "suspension_stop_epochs": float(
                        window_suspension_stop_epochs),
                    "yaw_rate_error_mean_radps": statistics.mean(
                        window_yaw_rate_errors),
                    "yaw_rate_error_rmse_radps": math.sqrt(statistics.mean(
                        error * error for error in window_yaw_rate_errors)),
                })
                fine_pred_error = math.hypot(
                    fine_predicted_pose_x - actual["x"],
                    fine_predicted_pose_y - actual["y"])
                fine_truth_error = math.hypot(
                    fine_truth_pose_x - actual["x"],
                    fine_truth_pose_y - actual["y"])
                fine_plant_vs_truth = math.hypot(
                    fine_predicted_pose_x - fine_truth_pose_x,
                    fine_predicted_pose_y - fine_truth_pose_y)
                fine_position_errors[horizon].append((
                    fine_pred_error, fine_truth_error, fine_plant_vs_truth))
                oracle_position_errors[horizon].append((
                    math.hypot(gt_heading_predicted_motion_x - actual["x"],
                               gt_heading_predicted_motion_y - actual["y"]),
                    math.hypot(predicted_heading_gt_motion_x - actual["x"],
                               predicted_heading_gt_motion_y - actual["y"])))
                window_grouped_yaw[horizon].append(
                    (start_speed, start_steering,
                     normalized_angle(predicted_yaw - actual["yaw"]))
                )
                if horizon == 1:
                    first_step_details.append({
                        "stamp": target_time,
                        "speed": start_speed,
                        "steering": start_steering,
                        "yaw_rate_start": start_truth["r"],
                        "yaw_rate_predicted": state[2],
                        "yaw_rate_actual": actual["r"],
                        "yaw_rate_error": state[2] - actual["r"],
                        "queued_throttle": current["delayed_throttle_1"],
                        "queued_steering": current["delayed_steering_1"] *
                            MAX_STEERING_RAD,
                        "wheel_body_mismatch": statistics.mean(
                            abs(omega * WHEEL_RADIUS_M - start_truth["u"])
                            for omega in start_plant["wheel_omega"]),
                    })
        if not valid:
            skipped += 1
            continue
        pathwise_position_maxima.append(max(window_position_errors))
        path_peak_index = max(
            range(len(window_position_errors)),
            key=window_position_errors.__getitem__)
        path_peak = window_position_errors[path_peak_index]
        path_row = position_decomposition[max(horizons)][-1]
        path_row["path_max_xy_m"] = path_peak
        path_row["path_max_step"] = float(path_peak_index + 1)
        path_row["start_yaw_residual_radps"] = yaw_residual
        if path_peak > 0.05:
            path_row["path_trace"] = window_pose_trace
        for horizon in horizons:
            for field in errors[horizon]:
                errors[horizon][field].extend(window_errors[horizon][field])
            grouped_yaw[horizon].extend(window_grouped_yaw[horizon])

    init_label = ("Unity-truth initial u/v/r + logged hidden UnityPlantState "
                  "and slot-1 commands (diagnostic only)" if oracle_motion_initial_state
                  else "logged UnityPlantState start + slot-1 commands")
    print(f"\nSource-plant open-loop rollout ({init_label}; no future state correction):")
    print(f"  causal yaw-residual correction gain={yaw_residual_gain:.3f}; "
          "residual is same-epoch raw forecast minus current gyro; "
          f"aligned={residual_valid_windows}, enabled={residual_enabled_windows}")
    print(f"  aligned_epochs={len(times)} start_windows={len(times)-max(horizons)} "
          f"skipped_windows={skipped} model_dt=25ms")
    print("  one-step command-queue phase A/B (same plant state; next Unity "
          "epoch used only for scoring):")
    for slot in (1, 2):
        fields = command_slot_errors[slot]
        if not fields["r"]:
            continue
        print(f"    queue slot {slot}: n={len(fields['r'])}")
        for field, unit in (("u", "m/s"), ("v", "m/s"),
                            ("r", "rad/s"), ("steering", "rad")):
            residuals = fields[field]
            absolute = [abs(value) for value in residuals]
            rmse = math.sqrt(statistics.mean(x*x for x in residuals))
            print(f"      {field}: RMSE={rmse:.5f} {unit} "
                  f"p95_abs={percentile(absolute, 0.95):.5f} "
                  f"max_abs={max(absolute):.5f}")
    for regime, slots in command_slot_regimes.items():
        if not slots[1]["r"]:
            continue
        print(f"    {regime}: n={len(slots[1]['r'])}")
        for slot in (1, 2):
            print(f"      slot {slot}:")
            for field, unit in (("r", "rad/s"), ("steering", "rad")):
                residuals = slots[slot][field]
                absolute = [abs(value) for value in residuals]
                rmse = math.sqrt(statistics.mean(x*x for x in residuals))
                print(f"        {field}: RMSE={rmse:.5f} {unit} "
                      f"p95_abs={percentile(absolute, 0.95):.5f} "
                      f"max_abs={max(absolute):.5f}")
    for horizon in horizons:
        values = errors[horizon]
        if not values["u"]:
            continue
        print(f"  horizon={horizon*25}ms n={len(values['u'])}")
        for field in ("u", "v", "r", "yaw", "xy"):
            unit = ("rad" if field == "yaw" else "rad/s" if field == "r" else
                    "m" if field == "xy" else "m/s")
            absolute = [abs(value) for value in values[field]]
            rmse = math.sqrt(statistics.mean(
                value * value for value in values[field]))
            print(f"    {field}: RMSE={rmse:.5f} {unit} "
                  f"p95_abs={percentile(absolute, 0.95):.5f} max={max(absolute):.5f}")
        for low, high in ((0, 2), (2, 4), (4, 6), (6, 8), (8, math.inf)):
            selected = [abs(yaw_error) for speed, _, yaw_error in grouped_yaw[horizon]
                        if low <= speed < high]
            if selected:
                print(f"    start_speed[{low:g},{high:g})m/s yaw_p95="
                      f"{percentile(selected, 0.95):.5f}rad n={len(selected)}")
        for low, high in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, math.inf)):
            selected = [abs(yaw_error) for _, steering, yaw_error
                        in grouped_yaw[horizon] if low <= steering < high]
            if selected:
                print(f"    start_abs_steer[{low:.2f},{high:g})rad yaw_p95="
                      f"{percentile(selected, 0.95):.5f}rad n={len(selected)}")

    print("  position-error decomposition at each scored horizon (rear-axle pose, "
          "axes rotated to each window's initial heading):")
    for horizon in horizons:
        rows = position_decomposition[horizon]
        if not rows:
            continue
        print(f"    {horizon*25}ms n={len(rows)}")
        for label, prefix in (("plant_vs_GT_pose", "pred"),
                              ("GT_twist_integral_vs_pose", "truth_twist"),
                              ("plant_vs_GT_twist_integral", "plant_vs_twist")):
            forward = [row[f"{prefix}_forward"] for row in rows]
            lateral = [row[f"{prefix}_lateral"] for row in rows]
            magnitude = [math.hypot(row[f"{prefix}_world_x"],
                                    row[f"{prefix}_world_y"]) for row in rows]
            rmse = math.sqrt(statistics.mean(value * value for value in magnitude))
            print(f"      {label}: forward_bias={statistics.mean(forward):+.4f}m "
                  f"lateral_bias={statistics.mean(lateral):+.4f}m "
                  f"xy_RMSE={rmse:.4f}m xy_p95={percentile(magnitude, 0.95):.4f}m")
    endpoint_rows = position_decomposition[max(horizons)]
    if endpoint_rows:
        over_5cm = sum(row["xy_m"] > 0.05 for row in endpoint_rows)
        print(f"  750ms 5cm endpoint gate: {len(endpoint_rows)-over_5cm}/"
              f"{len(endpoint_rows)} pass; {over_5cm} exceed 5cm")
        pathwise_over_5cm = sum(value > 0.05 for value in pathwise_position_maxima)
        pathwise_rmse = math.sqrt(statistics.mean(
            value * value for value in pathwise_position_maxima))
        print(f"  full 750ms path 5cm gate (max error at any 25ms step): "
              f"{len(pathwise_position_maxima)-pathwise_over_5cm}/"
              f"{len(pathwise_position_maxima)} pass; "
              f"{pathwise_over_5cm} exceed 5cm; "
              f"max-error RMSE={pathwise_rmse:.4f}m "
              f"p95={percentile(pathwise_position_maxima, 0.95):.4f}m "
              f"max={max(pathwise_position_maxima):.4f}m")
        for low, high in ((0.0, 0.10), (0.10, 0.20), (0.20, math.inf)):
            selected = [row for row in endpoint_rows
                        if low <= row["start_steering_rad"] < high]
            if not selected:
                continue
            position = [row["xy_m"] for row in selected]
            yaw = [row["end_abs_yaw_error_rad"] for row in selected]
            print(f"    start_abs_steer[{low:.2f},{high:g})rad: "
                  f">5cm={sum(value > 0.05 for value in position)}/{len(position)} "
                  f"xy_p95={percentile(position, 0.95):.4f}m "
                  f"xy_max={max(position):.4f}m yaw_abs_p95="
                  f"{percentile(yaw, 0.95):.4f}rad")
        print("  worst 750ms endpoint windows:")
        for row in sorted(endpoint_rows, key=lambda item: item["xy_m"],
                          reverse=True)[:5]:
            print(f"    xy={row['xy_m']:.4f}m start_speed="
                  f"{row['start_speed_mps']:.2f}m/s "
                  f"start_steer={row['start_steering_rad']:.3f}rad "
                  f"max_steer={row['max_steering_rad']:.3f}rad "
                  f"yaw_end={row['end_abs_yaw_error_rad']:.4f}rad "
                  f"yaw_peak={row['max_abs_yaw_error_rad']:.4f}rad "
                  f"du={row['end_u_error_mps']:+.3f}m/s "
                  f"dv={row['end_v_error_mps']:+.3f}m/s "
                  f"dr={row['end_r_error_radps']:+.3f}rad/s "
                  f"mean_dr={row['yaw_rate_error_mean_radps']:+.3f}rad/s "
                  f"rmse_dr={row['yaw_rate_error_rmse_radps']:.3f}rad/s")
        stop_windows = [row for row in endpoint_rows
                        if row["suspension_stop_epochs"] > 0.0]
        free_windows = [row for row in endpoint_rows
                        if row["suspension_stop_epochs"] == 0.0]
        for label, subset in (("stop-indicator", stop_windows),
                              ("no-stop-indicator", free_windows)):
            if not subset:
                continue
            xy = [row["xy_m"] for row in subset]
            yaw = [row["end_abs_yaw_error_rad"] for row in subset]
            print(f"  suspension {label}: windows={len(subset)} "
                  f"xy_p95={percentile(xy, 0.95):.4f}m "
                  f"xy_max={max(xy):.4f}m "
                  f"yaw_p95={percentile(yaw, 0.95):.4f}rad "
                  f"max_jounce={max(row['max_jounce_m'] for row in subset):.4f}m")
        if not stop_windows:
            print("  suspension compression-stop indicator: none in scored rollouts")
        failing_path_rows = [row for row in endpoint_rows
                             if row.get("path_max_xy_m", 0.0) > 0.05]
        if failing_path_rows:
            print("  full-path failures (>5cm at an intermediate or final 25ms step):")
            for row in sorted(failing_path_rows,
                              key=lambda item: item["path_max_xy_m"],
                              reverse=True)[:20]:
                print(f"    peak={row['path_max_xy_m']:.4f}m at="
                      f"{int(row['path_max_step'])*25}ms start_speed="
                      f"{row['start_speed_mps']:.2f}m/s start_steer="
                      f"{row['start_steering_rad']:.3f}rad "
                      f"yaw_innovation={row['start_yaw_residual_radps']:+.4f}rad/s")
                for step in row.get("path_trace", []):
                    if int(step["step"]) in (1, 4, 10, 20, 30) or \
                            abs(step["xy_error_m"] - row["path_max_xy_m"]) < 1.0e-9:
                        print(f"      {int(step['step'])*25:3d}ms xy="
                              f"{step['xy_error_m']:.4f}m yaw="
                              f"{step['yaw_error_rad']:+.4f}rad "
                              f"dXY=({step['step_error_forward_m']:+.4f},"
                              f"{step['step_error_lateral_m']:+.4f})m "
                              f"plant-truthTwist=({step['plant_vs_truth_twist_forward_m']:+.4f},"
                              f"{step['plant_vs_truth_twist_lateral_m']:+.4f})m "
                              f"truthTwist-pose=({step['truth_twist_vs_pose_forward_m']:+.4f},"
                              f"{step['truth_twist_vs_pose_lateral_m']:+.4f})m "
                              f"dr={step['yaw_rate_error_radps']:+.4f}rad/s "
                              f"du={step['speed_error_mps']:+.4f}/"
                              f"{step['lateral_speed_error_mps']:+.4f}m/s "
                              f"steer={step['physical_steering_rad']:+.3f}/"
                              f"{step['queued_steering_rad']:+.3f}rad "
                              f"q={step['queued_throttle']:.3f}")
        oracle_rows = oracle_position_errors[max(horizons)]
        for index, label in enumerate(("plant_translation_with_GT_yaw",
                                       "GT_translation_with_plant_yaw")):
            values = [row[index] for row in oracle_rows]
            print(f"    attribution {label}: xy_RMSE="
                  f"{math.sqrt(statistics.mean(value*value for value in values)):.4f}m "
                  f"xy_p95={percentile(values, 0.95):.4f}m "
                  f">5cm={sum(value > 0.05 for value in values)}/{len(values)}")
    print("  same trajectories with pose integrated at 1 ms from linearly "
          "interpolated 40 Hz endpoint states (diagnostic, not extra measurements):")
    for horizon in horizons:
        rows = fine_position_errors[horizon]
        if not rows:
            continue
        print(f"    {horizon*25}ms n={len(rows)}")
        for index, label in enumerate(("plant_vs_GT_pose", "GT_twist_integral_vs_pose",
                                       "plant_vs_GT_twist_integral")):
            values = [row[index] for row in rows]
            print(f"      {label}: xy_RMSE="
                  f"{math.sqrt(statistics.mean(value*value for value in values)):.4f}m "
                  f"xy_p95={percentile(values, 0.95):.4f}m")
    print("  truth pose/twist one-step closure hypotheses (per 25 ms interval):")
    for name, magnitudes in closure_modes.items():
        rows = closure_signed[name]
        if not magnitudes:
            continue
        forward = [row[0] for row in rows]
        lateral = [row[1] for row in rows]
        print(f"    {name}: displacement_RMSE="
              f"{math.sqrt(statistics.mean(value*value for value in magnitudes)):.5f}m "
              f"p95={percentile(magnitudes, 0.95):.5f}m "
              f"forward_bias={statistics.mean(forward):+.5f}m "
              f"lateral_bias={statistics.mean(lateral):+.5f}m n={len(rows)}")

    if first_step_details:
        print("  first-step yaw-rate attribution (overlapping 40 Hz windows):")
        partitions = (
            ("queued_hard_brake", lambda row: row["queued_throttle"] == 0.0),
            ("queued_drive", lambda row: row["queued_throttle"] > 0.0),
            ("queued_abs_steer<0.10", lambda row: abs(row["queued_steering"]) < 0.10),
            ("queued_abs_steer>=0.10", lambda row: abs(row["queued_steering"]) >= 0.10),
            ("wheel_body_mismatch<0.5", lambda row: row["wheel_body_mismatch"] < 0.5),
            ("wheel_body_mismatch>=0.5", lambda row: row["wheel_body_mismatch"] >= 0.5),
        )
        for label, predicate in partitions:
            selected = [row["yaw_rate_error"] for row in first_step_details
                        if predicate(row)]
            if selected:
                absolute = [abs(value) for value in selected]
                rmse = math.sqrt(statistics.mean(value * value for value in selected))
                print(f"    {label}: n={len(selected)} RMSE={rmse:.5f}rad/s "
                      f"p95_abs={percentile(absolute, 0.95):.5f} "
                      f"max_abs={max(absolute):.5f}")
        print("  largest first-step yaw-rate residuals:")
        for row in sorted(first_step_details,
                          key=lambda value: abs(value["yaw_rate_error"]),
                          reverse=True)[:5]:
            print(
                f"    t={row['stamp']:.3f} speed={row['speed']:.2f}m/s "
                f"steer={row['steering']:.3f}rad r0={row['yaw_rate_start']:.3f} "
                f"r_pred={row['yaw_rate_predicted']:.3f} "
                f"r_gt={row['yaw_rate_actual']:.3f} "
                f"q1=({row['queued_steering']:.3f}rad,"
                f"{row['queued_throttle']:.3f}) "
                f"wheel-body={row['wheel_body_mismatch']:.2f}m/s"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="rosbag directory")
    parser.add_argument("--trajectory", type=Path, help="raceline CSV used for the run")
    parser.add_argument(
        "--source-plant-rollout", action="store_true",
        help="score a no-reanchoring 25 ms command-sequence rollout using the logged "
             "UnityPlantState and the production C++ source plant",
    )
    parser.add_argument(
        "--source-plant-library", type=Path,
        default=Path("build/f1tenth_mpc/libunity_source_plant_eval.so"),
        help="optional evaluator library built from the production plant source",
    )
    parser.add_argument(
        "--yaw-residual-gain", type=float, default=0.0,
        help="apply the production gated causal gyro-residual yaw correction "
             "during fixed-input source-plant replay (default: disabled)",
    )
    parser.add_argument(
        "--yaw-residual-decay", type=float, default=1.0,
        help="per-25ms multiplicative decay of the starting causal yaw residual "
             "during fixed-input replay (default: constant residual)",
    )
    parser.add_argument(
        "--oracle-motion-initial-state", action="store_true",
        help="diagnostic only: initialize each source-plant rollout with same-epoch "
             "Unity-truth u/v/r while keeping logged hidden plant states and all "
             "future commands unchanged; never a production input",
    )
    parser.add_argument(
        "--truth-pose-twist-phase", action="store_true",
        help="diagnostic only: compare Unity truth pose increments with IMU twist "
             "under a ±10 ms phase sweep and correlate bridge jitter; never "
             "applies a correction",
    )
    parser.add_argument(
        "--truth-pose-twist-phase-only", action="store_true",
        help="run only the truth pose/twist phase diagnostic (implies "
             "--truth-pose-twist-phase)",
    )
    args = parser.parse_args()
    data = read_bag(args.bag)
    truth = data.get(GT, [])
    if not truth:
        parser.error(f"bag has no simulator truth stream: {GT}")

    print(f"truth: frame={truth[0]['frame']} child={truth[0]['child']} count={len(truth)}")
    if args.truth_pose_twist_phase or args.truth_pose_twist_phase_only:
        report_truth_pose_twist_phase(truth, data.get(BRIDGE_TIMING, []))
    if args.truth_pose_twist_phase_only:
        return 0
    for topic in (GT, ODOM, ODOM_DIAGNOSTICS, DIAGNOSTICS, THROTTLE_COMMAND):
        print_interval(topic, data.get(topic, []))
    for topic, label in ((ODOM, "production odom"), (EKF, "EKF odom")):
        if data.get(topic):
            report_state_errors(truth, data[topic], label)

    speeds = [math.hypot(row["u"], row["v"]) for row in truth]
    print(
        f"GT speed: median={statistics.median(speeds):.3f}m/s "
        f"p95={percentile(speeds, 0.95):.3f}m/s max={max(speeds):.3f}m/s"
    )
    for topic, label in ((LAP, "laps"), (COLLISION, "collisions"),
                         (LAST_LAP, "last_lap_s"), (BEST_LAP, "best_lap_s")):
        values = [row["value"] for row in data.get(topic, [])]
        if values:
            print(f"{label}: first={values[0]} last={values[-1]} max={max(values)}")
    for topic in (THROTTLE, THROTTLE_COMMAND, STEERING, STEERING_COMMAND):
        values = [row["value"] for row in data.get(topic, [])]
        if values:
            print(
                f"{topic}: min={min(values):.4f} max={max(values):.4f} "
                f"mean={statistics.mean(values):.4f}"
            )
    report_odom_diagnostics(
        truth, data.get(ODOM, []), data.get(ODOM_DIAGNOSTICS, []),
        data.get(STEERING, []), data.get(THROTTLE, []),
    )
    report_encoder_phase(
        data.get(LEFT_ENCODER, []), data.get(RIGHT_ENCODER, []),
        data.get(ODOM_DIAGNOSTICS, []), data.get(THROTTLE, []))
    if data.get(ODOM):
        report_imu_velocity_propagation(
            truth, data[ODOM], data.get(IMU, []), data.get(ODOM_DIAGNOSTICS, []),
            data.get(THROTTLE, []))
    plant_rows = data.get(PLANT, [])
    if plant_rows:
        print(f"plant schema versions: {sorted({row['schema'] for row in plant_rows})}")
        print("wheel omega has no independent Unity truth channel; do not score it as truth.")
        report_encoder_slip(
            truth, plant_rows, data.get(LEFT_ENCODER, []),
            data.get(RIGHT_ENCODER, []), data.get(THROTTLE, []))
        if args.source_plant_rollout:
            report_source_plant_rollout(
                truth, plant_rows, data.get(ODOM_DIAGNOSTICS, []),
                args.source_plant_library, data.get(IMU, []),
                args.yaw_residual_gain,
                args.yaw_residual_decay, args.oracle_motion_initial_state)
    elif args.source_plant_rollout:
        print(f"source-plant rollout: bag has no usable {PLANT} stream")
    diagnostics = [row for row in data.get(DIAGNOSTICS, []) if row.get("json")]
    if diagnostics:
        payloads = [row["json"] for row in diagnostics]
        print(f"MPC status counts: {dict(Counter(x.get('status') for x in payloads))}")
        print(f"MPC plant-state source counts: {dict(Counter(x.get('plant_state_source') for x in payloads))}")
        report_horizon_errors(truth, diagnostics, load_trajectory(args.trajectory))
    if data.get(OPEN_LOOP_TEST):
        report_open_loop_horizon_test(truth, data[OPEN_LOOP_TEST])
    for row in data.get(BRIDGE_FAULT, []):
        if row["value"]:
            print(f"bridge timing fault detail: {row['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

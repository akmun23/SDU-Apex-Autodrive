#!/usr/bin/env python3
"""Measure production odometry and MPC rollout errors against Unity bag truth.

Run in the ROS 2 environment that can deserialize the bag's message types:
  python3 tools/evaluate_mpc_bag.py live_runs/<run>/run [--trajectory path.csv]

The bag reader can decode the repository's UnityPlantState and DriveCommand
CDR layouts directly, so generated sdu_apex_msgs Python classes are optional.
The trajectory is optional; when supplied, diagnostic path progress and cross-
track error are converted back to a local XY/yaw rollout for comparison.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import json
import math
import struct
import statistics
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Any

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


GT = "/autodrive/roboracer_1/odom"
ODOM = "/odom"
EKF = "/ekf_odom"
EKF_POSE = "/ekf_pose"
CURRENT_MAP_POSE = "/current_map_pose"
AMCL_POSE = "/amcl_pose"
AMCL_HEALTH = "/amcl_localization_health"
AMCL_SCAN_ALIGNMENT = "/amcl_scan_alignment"
AMCL_PARTICLE_COUNT = "/amcl_particle_count"
AMCL_KLD_DIAGNOSTICS = "/amcl_kld_diagnostics"
AMCL_TIMING = "/amcl_timing"
AMCL_GPU_TIMING = "/amcl_gpu_timing"
PLANT = "/odom/unity_plant_state"
DIAGNOSTICS = "/mpc/diagnostics"
CANDIDATE_DIAGNOSTICS = "/mpc/candidate_diagnostics"
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
DEVELOPMENT_ACTUATOR_TIMING = (
    "/autodrive/roboracer_1/development_actuator_timing")
DRIVE_COMMAND = "/mpc/drive_command"
LEFT_ENCODER = "/autodrive/roboracer_1/left_encoder"
RIGHT_ENCODER = "/autodrive/roboracer_1/right_encoder"
ODOM_DIAGNOSTICS = "/odom/diagnostics"
CSV_REPLAY = "/csv_replay/diagnostics"
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
    GT, ODOM, EKF, EKF_POSE, CURRENT_MAP_POSE, AMCL_POSE,
    AMCL_HEALTH, AMCL_SCAN_ALIGNMENT, AMCL_PARTICLE_COUNT,
    AMCL_KLD_DIAGNOSTICS, AMCL_TIMING, AMCL_GPU_TIMING,
    PLANT, DIAGNOSTICS, CANDIDATE_DIAGNOSTICS,
    OPEN_LOOP_TEST, IMU, LAP, COLLISION, LAST_LAP,
    BEST_LAP, THROTTLE, THROTTLE_COMMAND, STEERING, STEERING_COMMAND,
    BRIDGE_FAULT, BRIDGE_TIMING, DEVELOPMENT_ACTUATOR_TIMING,
    DRIVE_COMMAND, LEFT_ENCODER, RIGHT_ENCODER,
    ODOM_DIAGNOSTICS, CSV_REPLAY,
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


class CdrReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 4
        self.origin = 4

    def read(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        alignment = min(size, 8)
        self.offset = self.origin + (
            (self.offset - self.origin + alignment - 1) & ~(alignment - 1)
        )
        if self.offset + size > len(self.data):
            raise ValueError("truncated CDR payload")
        values = struct.unpack_from("<" + fmt, self.data, self.offset)
        self.offset += size
        return values[0] if len(values) == 1 else values

    def read_string(self) -> str:
        length = self.read("I")
        if length == 0 or self.offset + length > len(self.data):
            raise ValueError("invalid CDR frame string")
        value = self.data[self.offset:self.offset + length]
        self.offset += length
        return value.rstrip(b"\x00").decode("utf-8")

    def doubles(self, count: int) -> list[float]:
        return list(self.read(f"{count}d"))


def decode_plant_cdr(
    serialized: bytes, bag_time_ns: int,
) -> dict[str, Any] | None:
    """Decode UnityPlantState schemas 9–11 without generated ROS classes.

    This is strictly an offline bag-reading path. It does not publish or feed
    the runtime observer/controller. Schema additions are version-gated below
    and exact payload consumption prevents silently misreading a new layout.
    """
    if len(serialized) < 4 or serialized[:2] != b"\x00\x01":
        return None
    cdr = CdrReader(serialized)
    sec = cdr.read("i")
    nanosec = cdr.read("I")
    frame_id = cdr.read_string()
    schema = cdr.read("I")
    if schema not in (9, 10, 11):
        return None

    state_origin = cdr.read("B")
    sensor_corrected_mask = cdr.read("I")
    history_initialized = cdr.read("?")
    if schema >= 10:
        low_forward_speed_time = cdr.doubles(4)
        low_side_speed_time = cdr.doubles(4)
    else:
        low_forward_speed_time = [0.0] * 4
        low_side_speed_time = [0.0] * 4
    u, v_com, v_rear, yaw_rate, gyro_yaw_rate = cdr.doubles(5)
    model_x, model_y = cdr.doubles(2)
    steering_command, throttle_command = cdr.doubles(2)
    delayed_steer_1, delayed_steer_2, delayed_throttle_1, delayed_throttle_2 = (
        cdr.doubles(4)
    )
    feedback_valid = cdr.read("?")
    steering_feedback, throttle_feedback = cdr.doubles(2)
    wheel_omega = cdr.doubles(4)
    wheel_angle = cdr.doubles(4)
    (body_height, body_height_rate, roll, roll_rate, pitch, pitch_rate,
     physical_steering) = cdr.doubles(7)
    suspension_jounce = cdr.doubles(4)
    long_slip = cdr.doubles(4)
    lateral_slip = cdr.doubles(4)
    normal_load = cdr.doubles(4)
    suspension_force = cdr.doubles(4)
    jounce_rate = cdr.doubles(4)
    tire_fx = cdr.doubles(4)
    tire_fy = cdr.doubles(4)
    ax, ay, yaw_accel = cdr.doubles(3)
    if schema >= 11:
        yaw_residual_valid = cdr.read("?")
        yaw_residual = cdr.read("d")
    else:
        yaw_residual_valid = False
        yaw_residual = 0.0
    if cdr.offset != len(serialized):
        raise ValueError(
            f"schema-{schema} CDR layout consumed "
            f"{cdr.offset}/{len(serialized)} bytes"
        )

    return {
        "bag_time": bag_time_ns * 1.0e-9,
        "stamp": sec + nanosec * 1.0e-9,
        "frame": frame_id,
        "schema": schema,
        "state_origin": state_origin,
        "sensor_corrected_mask": sensor_corrected_mask,
        "u": u,
        "v_com": v_com,
        "v_rear": v_rear,
        "r": yaw_rate,
        "gyro_r": gyro_yaw_rate,
        "yaw_rate_prediction_residual": yaw_residual,
        "yaw_rate_prediction_residual_valid": yaw_residual_valid,
        "wheel_omega": wheel_omega,
        "wheel_angle": wheel_angle,
        "body_height": body_height,
        "body_height_rate": body_height_rate,
        "roll": roll,
        "roll_rate": roll_rate,
        "pitch": pitch,
        "pitch_rate": pitch_rate,
        "physical_steering": physical_steering,
        "suspension_jounce": suspension_jounce,
        "suspension_history_initialized": history_initialized,
        "low_forward_speed_time_s": low_forward_speed_time,
        "low_side_speed_time_s": low_side_speed_time,
        "legacy_missing_sticky_timers": schema == 9,
        "delayed_steering_1": delayed_steer_1,
        "delayed_steering_2": delayed_steer_2,
        "delayed_throttle_1": delayed_throttle_1,
        "delayed_throttle_2": delayed_throttle_2,
        "actuator_feedback_valid": feedback_valid,
        "steering_feedback": steering_feedback,
        "throttle_feedback": throttle_feedback,
        "long_slip": long_slip,
        "lat_slip": lateral_slip,
        "normal_load": normal_load,
        "suspension_force": suspension_force,
        "jounce_rate": jounce_rate,
        "tire_fx": tire_fx,
        "tire_fy": tire_fy,
        "ax": ax,
        "ay": ay,
        "yaw_accel": yaw_accel,
        "steering_command": steering_command,
        "throttle_command": throttle_command,
        "physx_model_rear_axle_x_m": model_x,
        "physx_model_rear_axle_y_m": model_y,
    }


def decode_drive_command_cdr(
    serialized: bytes, bag_time_ns: int,
) -> dict[str, Any] | None:
    """Decode the current DriveCommand CDR layout for offline bag analysis."""
    if len(serialized) < 4 or serialized[:2] != b"\x00\x01":
        return None
    cdr = CdrReader(serialized)
    sec = cdr.read("i")
    nanosec = cdr.read("I")
    cdr.read_string()
    steering, throttle = cdr.doubles(2)
    if cdr.offset != len(serialized):
        raise ValueError(
            f"DriveCommand CDR layout consumed {cdr.offset}/{len(serialized)} bytes"
        )
    return {
        "bag_time": bag_time_ns * 1.0e-9,
        "stamp": sec + nanosec * 1.0e-9,
        "steering": steering,
        "throttle": throttle,
    }


def decode_json_payload(raw: str) -> dict[str, Any]:
    """Decode diagnostic JSON, repairing the known action_source typo only.

    One development capture contains a duplicated closing quote emitted by an
    earlier MPC diagnostic formatter. Preserve all records when reading it,
    but do not broadly rewrite arbitrary malformed JSON.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        repaired = raw
        for value in ("current_solver_candidate", "revalidated_prior_plan"):
            repaired = repaired.replace(f'"{value}""', f'"{value}"')
        if repaired == raw:
            return {"_decode_error": "invalid_json", "_raw": raw}
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError:
            return {"_decode_error": "invalid_json_after_known_repair",
                    "_raw": raw}
        if isinstance(payload, dict):
            payload["_bag_decoder_repaired_action_source_quote"] = True
    if not isinstance(payload, dict):
        return {"_decode_error": "json_root_not_object", "_raw": raw}
    return payload


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
            # Decode the two retained custom types directly from CDR below;
            # generated sdu_apex_msgs Python classes are optional.
            if name not in (PLANT, DRIVE_COMMAND):
                raise
            print(f"note: skipping {name} (custom message package not sourced)")
    data = {topic: [] for topic in type_names}
    skipped_plant_schema = False
    decoded_custom_topics: set[str] = set()

    while reader.has_next():
        topic, serialized, bag_time_ns = reader.read_next()
        if topic not in message_types:
            try:
                if topic == PLANT:
                    decoded = decode_plant_cdr(serialized, bag_time_ns)
                elif topic == DRIVE_COMMAND:
                    decoded = decode_drive_command_cdr(serialized, bag_time_ns)
                else:
                    decoded = None
            except (UnicodeDecodeError, ValueError, struct.error) as error:
                decoded = None
                if topic == PLANT and not skipped_plant_schema:
                    print(f"note: could not decode {PLANT} CDR: {error}")
                    skipped_plant_schema = True
            if decoded is not None:
                data[topic].append(decoded)
                if topic not in decoded_custom_topics:
                    if topic == PLANT:
                        print(
                            "note: decoded UnityPlantState CDR directly; "
                            "generated sdu_apex_msgs package is not required"
                        )
                    else:
                        print(
                            "note: decoded DriveCommand CDR directly; "
                            "generated sdu_apex_msgs package is not required"
                        )
                    decoded_custom_topics.add(topic)
            continue
        try:
            message = deserialize_message(serialized, message_types[topic])
        except Exception as error:
            # Older runtime captures used earlier UnityPlantState layouts.
            # That diagnostic topic is not needed for ground-truth odom or
            # MPC forecast scoring; keep the rest of the bag usable while
            # making the schema mismatch explicit.
            if topic not in (PLANT, DRIVE_COMMAND):
                raise
            try:
                if topic == PLANT:
                    legacy_row = decode_plant_cdr(serialized, bag_time_ns)
                else:
                    legacy_row = decode_drive_command_cdr(
                        serialized, bag_time_ns)
            except (UnicodeDecodeError, ValueError, struct.error):
                legacy_row = None
            if legacy_row is not None:
                data[topic].append(legacy_row)
                if topic not in decoded_custom_topics:
                    print(
                        f"note: decoded {topic} CDR directly after "
                        "generated-message deserialization failed"
                    )
                    decoded_custom_topics.add(topic)
                continue
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
        elif topic in (EKF_POSE, CURRENT_MAP_POSE, AMCL_POSE):
            pose = message.pose.pose
            row.update(
                stamp=stamp(message),
                frame=message.header.frame_id,
                x=pose.position.x,
                y=pose.position.y,
                z=pose.position.z,
                yaw=yaw_from_quaternion(pose.orientation),
                covariance_xy_max=max(message.pose.covariance[0],
                                      message.pose.covariance[7]),
                covariance_yaw=message.pose.covariance[35],
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
                yaw_rate_prediction_residual=(
                    message.yaw_rate_prediction_residual_radps),
                yaw_rate_prediction_residual_valid=(
                    message.yaw_rate_prediction_residual_valid),
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
                actuator_feedback_valid=message.actuator_feedback_valid,
                steering_feedback=message.steering_feedback_rad,
                throttle_feedback=message.throttle_feedback_normalized,
                long_slip=list(message.longitudinal_slip),
                lat_slip=list(message.lateral_slip),
                normal_load=list(message.normal_load_n),
                tire_fx=list(message.tire_longitudinal_force_n),
                tire_fy=list(message.tire_lateral_force_n),
                ax=message.longitudinal_acceleration_mps2,
                ay=message.lateral_acceleration_mps2,
                yaw_accel=message.yaw_acceleration_radps2,
            )
        elif topic in (
            DIAGNOSTICS, CANDIDATE_DIAGNOSTICS, OPEN_LOOP_TEST, BRIDGE_TIMING,
            DEVELOPMENT_ACTUATOR_TIMING, CSV_REPLAY,
        ):
            row["json"] = decode_json_payload(message.data)
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
        elif topic in (AMCL_HEALTH, AMCL_SCAN_ALIGNMENT,
                       AMCL_KLD_DIAGNOSTICS, AMCL_GPU_TIMING):
            row["values"] = list(message.data)
        elif topic == AMCL_PARTICLE_COUNT:
            row["value"] = message.data
        elif topic == AMCL_TIMING:
            row["value"] = message.data
        elif topic == BRIDGE_FAULT:
            row["value"] = message.data
        else:
            row["value"] = message.data
        data[topic].append(row)
    return data


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * (len(ordered) - 1)))]


def report_candidate_prefix_chains(
    diagnostics: list[dict[str, Any]],
    collision_count: int | None = None,
) -> None:
    """Summarize contiguous, exact-validated short-horizon publications.

    Keep the objective/slack semantics explicit: the objective is the solver's
    full-horizon candidate objective, while the prefix is separately validated
    over its short exact-rollout horizon. Older bags may omit these fields.
    """
    chains: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in diagnostics:
        if row.get("status") == "accepted_candidate_prefix":
            current.append(row)
        elif current:
            chains.append(current)
            current = []
    if current:
        chains.append(current)
    if not chains:
        return

    lengths = [len(chain) for chain in chains]
    print("MPC candidate-prefix chains:")
    print(f"  chains={len(chains)} total_prefix_actions={sum(lengths)} "
          f"lengths={lengths} max={max(lengths)} "
          f"run_collision_count={collision_count if collision_count is not None else 'unknown'}")
    for index, chain in enumerate(chains, 1):
        steer = [float(row["first_action"][0]) for row in chain
                 if isinstance(row.get("first_action"), list) and
                 len(row["first_action"]) >= 2 and
                 math.isfinite(float(row["first_action"][0]))]
        reversals = sum(left * right < 0.0
                        for left, right in zip(steer, steer[1:]))
        steer_curvature_pairs = [
            (float(row["first_action"][0]),
             float(row["path_curvature_per_m"]))
            for row in chain
            if isinstance(row.get("first_action"), list) and
            len(row["first_action"]) >= 2 and
            isinstance(row.get("path_curvature_per_m"), (int, float)) and
            math.isfinite(float(row["first_action"][0])) and
            math.isfinite(float(row["path_curvature_per_m"]))]
        same_sign = sum(value * curve > 0.0
                        for value, curve in steer_curvature_pairs)
        opposite_sign = sum(value * curve < 0.0
                            for value, curve in steer_curvature_pairs)
        solver = [row.get("solver", {}) for row in chain]
        residuals = [max(float(item.get("primal_residual", math.nan)),
                         float(item.get("dual_residual", math.nan)))
                     for item in solver]
        residuals = [value for value in residuals if math.isfinite(value)]
        objectives = []
        for row in chain:
            solver_diagnostics = row.get("solver", {})
            objective = solver_diagnostics.get(
                "objective", row.get("objective"))
            if (isinstance(objective, (int, float)) and
                    math.isfinite(float(objective))):
                objectives.append(float(objective))
        slacks = [float(row["minimum_predicted_corridor_slack_m"])
                  for row in chain
                  if isinstance(row.get("minimum_predicted_corridor_slack_m"),
                                (int, float)) and
                  math.isfinite(float(row["minimum_predicted_corridor_slack_m"]))]
        progress = [float(row["progress_m"]) for row in chain
                    if isinstance(row.get("progress_m"), (int, float)) and
                    math.isfinite(float(row["progress_m"]))]
        admm_statuses = Counter(
            str(item.get("admm_status", "missing")) for item in solver)
        residual_summary = (
            f"{percentile(residuals, 0.50):.4g}/"
            f"{percentile(residuals, 0.95):.4g}/{max(residuals):.4g}"
            if residuals else "unavailable")
        slack_summary = f"{min(slacks):.4g}m" if slacks else "unavailable"
        progress_summary = (
            f"{progress[0]:.3f}..{progress[-1]:.3f}m" if progress
            else "unavailable")
        print(
            f"  chain{index}: n={len(chain)} "
            f"progress={progress_summary} "
            f"steer(+/-/0)="
            f"{sum(value > 1e-4 for value in steer)}/"
            f"{sum(value < -1e-4 for value in steer)}/"
            f"{sum(abs(value) <= 1e-4 for value in steer)} "
            f"reversals={reversals} steer-curvature(same/opposite)="
            f"{same_sign}/{opposite_sign} ADMM={dict(admm_statuses)} "
            f"residual(p50/p95/max)={residual_summary} "
            f"prefix_objective_n={len(objectives)} "
            f"prefix_slack_n/min={len(slacks)}/{slack_summary}")
        if objectives:
            print("    solver full-horizon candidate objective "+
                  f"min/p50/p95/max={min(objectives):.6g}/"
                  f"{percentile(objectives, 0.50):.6g}/"
                  f"{percentile(objectives, 0.95):.6g}/"
                  f"{max(objectives):.6g}")
        if slacks:
            print("    validated-prefix corridor slack "+
                  f"min/p10/p50={min(slacks):.5f}/"
                  f"{percentile(slacks, 0.10):.5f}/"
                  f"{percentile(slacks, 0.50):.5f}m")

    print("  Steering×curvature is descriptive only; long-chain association is "
          "run-level and is not a causal threshold.")


def report_prefix_warm_start_consistency(
    diagnostics: list[dict[str, Any]],
) -> None:
    """Check that a published prefix seeds the next nominal after one shift."""
    comparisons: list[tuple[float, float]] = []
    skipped = 0
    seed_source_missing = 0
    seed_source_mismatch = 0
    ordered = sorted(diagnostics, key=lambda row: float(row.get("bag_time", 0.0)))
    for previous, current in zip(ordered, ordered[1:]):
        prev = previous.get("json", {})
        curr = current.get("json", {})
        if prev.get("status") != "accepted_candidate_prefix":
            continue
        dt = float(current.get("bag_time", 0.0)) - float(
            previous.get("bag_time", 0.0))
        if dt <= 0.0 or dt > 0.040:
            skipped += 1
            continue
        issued = prev.get("candidate_controls")
        nominal = curr.get("nominal_controls")
        current_solver = curr.get("solver", {})
        seed_source = current_solver.get(
            "nominal_control_seed_source",
            curr.get("nominal_control_seed_source"))
        if seed_source is None:
            seed_source_missing += 1
        elif seed_source != 2:
            seed_source_mismatch += 1
            skipped += 1
            continue
        if (not isinstance(issued, list) or len(issued) < 2 or
                not isinstance(nominal, list) or not nominal):
            skipped += 1
            continue
        try:
            steering_error = abs(float(nominal[0][0]) - float(issued[1][0]))
            throttle_error = abs(float(nominal[0][1]) - float(issued[1][1]))
        except (IndexError, TypeError, ValueError):
            skipped += 1
            continue
        comparisons.append((steering_error, throttle_error))

    if not comparisons and not skipped:
        return
    print("MPC candidate-prefix → next nominal warm-start consistency:")
    if not comparisons:
        print(f"  comparable adjacent pairs=0 skipped={skipped}; "
              "control arrays or an adjacent cycle are missing")
        return
    steering = [pair[0] for pair in comparisons]
    throttle = [pair[1] for pair in comparisons]
    print(f"  compared={len(comparisons)} skipped={skipped}; next nominal[0] "
          "vs published-prefix candidate[1] abs error (steering/throttle):")
    print(f"  steering max={max(steering):.3g}rad, "
          f"throttle max={max(throttle):.3g}; "
          f"exact-within-1e-5="
          f"{sum(a <= 1.0e-5 and b <= 1.0e-5 for a, b in comparisons)}"
          f"/{len(comparisons)}")
    if seed_source_missing or seed_source_mismatch:
        print(f"  seed-source code missing={seed_source_missing}, "
              f"explicitly not issued-sequence={seed_source_mismatch}; "
              "array equality is still measured independently")


def report_jacobian_worker_distribution(
    diagnostics: list[dict[str, Any]],
) -> None:
    rows = []
    for row in diagnostics:
        solver = row.get("solver", row)
        if (isinstance(solver.get("jacobian_worker_cpu_us"), list) and
                len(solver["jacobian_worker_cpu_us"]) == 8 and
                isinstance(solver.get("jacobian_worker_stage_wall_us"), list) and
                len(solver["jacobian_worker_stage_wall_us"]) == 8 and
                isinstance(solver.get("jacobian_worker_stage_count"), list) and
                len(solver["jacobian_worker_stage_count"]) == 8):
            rows.append({**row, **solver})
    if not rows:
        return
    print("MPC Jacobian worker load (new diagnostic schema):")
    for worker in range(8):
        active = [row for row in rows
                  if int(row["jacobian_worker_stage_count"][worker]) > 0]
        if not active:
            continue
        cpu = [float(row["jacobian_worker_cpu_us"][worker])
               for row in active]
        stage_wall = [float(row["jacobian_worker_stage_wall_us"][worker])
                      for row in active]
        stage_count = sum(int(row["jacobian_worker_stage_count"][worker])
                          for row in rows)
        worker_tids = [int(row["jacobian_worker_os_tid"][worker])
                       for row in active
                       if isinstance(row.get("jacobian_worker_os_tid"), list)
                       and len(row["jacobian_worker_os_tid"]) == 8
                       and int(row["jacobian_worker_os_tid"][worker]) > 0]
        tid = Counter(worker_tids).most_common(1)[0][0] if worker_tids else 0
        print(
            f"  worker{worker}(tid={tid or 'unrecorded'}): "
            f"active_cycles={len(active)} "
            f"stage_count={stage_count} "
            f"CPU_us(p50/p95/max)="
            f"{percentile(cpu, 0.50):.1f}/"
            f"{percentile(cpu, 0.95):.1f}/{max(cpu):.1f} "
            f"stage_wall_sum_us(p50/p95/max)="
            f"{percentile(stage_wall, 0.50):.1f}/"
            f"{percentile(stage_wall, 0.95):.1f}/{max(stage_wall):.1f}")
    print("  Per-worker stage-wall sums include task durations, not thread wait; "
          "use jacobian_wall_us for elapsed parallel wall time.")


def report_development_actuator_timing(rows: list[dict[str, Any]]) -> None:
    events = [row["json"] for row in rows
              if isinstance(row.get("json"), dict) and
              "_decode_error" not in row["json"]]
    if not events:
        return
    statuses = Counter(event.get("status", "missing") for event in events)
    forwarded = [event for event in events
                 if event.get("status") == "forwarded" and
                 isinstance(event.get("interface_receive_monotonic_ns"), int)
                 and isinstance(event.get("actuator_publish_monotonic_ns"), int)]
    elapsed_ms = [
        (event["actuator_publish_monotonic_ns"] -
         event["interface_receive_monotonic_ns"]) * 1.0e-6
        for event in forwarded
    ]
    sequences = sorted({event["command_sequence"] for event in forwarded
                        if isinstance(event.get("command_sequence"), int)})
    gaps = sum(right - left - 1 for left, right in zip(sequences, sequences[1:]))
    print("development actuator publication timing (diagnostic output, "
          "not a controller input):")
    print(f"  records={len(events)} statuses={dict(statuses)} "
          f"forwarded={len(forwarded)} sequence_gaps={gaps}")
    if elapsed_ms:
        print("  interface-receive→actuator-publish ms "
              f"median/p95/max={percentile(elapsed_ms, 0.50):.3f}/"
              f"{percentile(elapsed_ms, 0.95):.3f}/{max(elapsed_ms):.3f}")


def report_admm_convergence(
    diagnostics: list[dict[str, Any]],
    drive_commands: list[dict[str, Any]] | None = None,
) -> None:
    """Report the single MPC solve and exact nonlinear acceptance."""
    rows = [row["json"] for row in diagnostics if row.get("json")]

    # `solver.cycle_us` excludes work around the numerical solve. The timing
    # check records its own elapsed time before command publication; the later
    # diagnostic-emission timestamp includes ROS serialization and must not be
    # mistaken for the value which actually passed or failed the gate.
    gate_rows = [row for row in rows
                 if isinstance(row.get("publication_gate_elapsed_ms"),
                                (int, float))]
    has_gate_schema = any("publication_gate_elapsed_ms" in row for row in rows)
    emission_rows = [row for row in rows
                     if isinstance(row.get("end_to_end_control_ms"),
                                    (int, float))]
    if gate_rows:
        gate_ms = [float(row["publication_gate_elapsed_ms"])
                   for row in gate_rows]
        budgets = [float(row["end_to_end_cycle_budget_ms"])
                   for row in gate_rows
                   if isinstance(row.get("end_to_end_cycle_budget_ms"),
                                  (int, float))]
        budget = min(budgets) if budgets else math.nan
        missed_guard = sum(
            float(row["publication_gate_elapsed_ms"]) >=
            float(row["end_to_end_cycle_budget_ms"])
            for row in gate_rows
            if isinstance(row.get("end_to_end_cycle_budget_ms"), (int, float))
        )
        print(
            "  publication-gate elapsed ms p50/p95/max="
            f"{percentile(gate_ms, 0.50):.3f}/"
            f"{percentile(gate_ms, 0.95):.3f}/{max(gate_ms):.3f}; "
            f"budget={budget:.3f}ms; at/over={missed_guard}/{len(gate_rows)}"
        )
    elif has_gate_schema:
        print("  publication-gate timing: field present, but no valid candidate "
              "reached the publication gate in this capture")
    elif emission_rows:
        print("  publication-gate timing: not recorded in this legacy bag")
    if gate_rows:
        slowest_gate_rows = sorted(
            gate_rows,
            key=lambda row: float(row["publication_gate_elapsed_ms"]),
            reverse=True,
        )[:3]
        print("  slowest publication-gate samples (phase breakdown, ms):")
        for row in slowest_gate_rows:
            solver = row.get("solver")
            solver = solver if isinstance(solver, dict) else {}
            prediction = row.get("control_time_prediction")
            prediction = prediction if isinstance(prediction, dict) else {}
            prior_plan = row.get("prior_plan_revalidation")
            prior_plan = prior_plan if isinstance(prior_plan, dict) else {}

            def phase_ms(name: str) -> str:
                value = solver.get(name)
                if not isinstance(value, (int, float)):
                    return "n/a"
                return f"{float(value) * 1.0e-3:.3f}"

            node_solve_ms = solver.get("node_solve_us")
            outside_solver_ms = (
                float(row["publication_gate_elapsed_ms"]) -
                float(node_solve_ms) * 1.0e-3
                if isinstance(node_solve_ms, (int, float)) else math.nan
            )
            prediction_us = prediction.get("elapsed_us")
            prediction_ms = (
                float(prediction_us) * 1.0e-3
                if isinstance(prediction_us, (int, float)) else math.nan
            )
            print(
                "    gate="
                f"{float(row['publication_gate_elapsed_ms']):.3f} "
                f"status={row.get('status', 'unknown')} "
                f"action={row.get('action_source', 'unknown')} "
                f"published={row.get('command_published', 'n/a')} "
                f"node_solve={phase_ms('node_solve_us')} "
                f"QP={phase_ms('qp_build_us')} "
                f"Jacobian={phase_ms('jacobian_wall_us')} "
                f"ADMM={phase_ms('admm_us')} "
                f"control_prediction={prediction_ms:.3f} "
                "prior_plan=(attempted="
                f"{prior_plan.get('attempted', 'n/a')},"
                f"valid={prior_plan.get('exact_valid', 'n/a')},"
                f"deadline={prior_plan.get('timing_rejected', 'n/a')},"
                f"validate_us={prior_plan.get('validation_us', 'n/a')}) "
                f"callback_minus_node_solve={outside_solver_ms:.3f}"
            )
    if emission_rows:
        emission_ms = [float(row["end_to_end_control_ms"])
                       for row in emission_rows]
        print("  callback→diagnostic-emission ms (not gate time) "
              f"p50/p95/max={percentile(emission_ms, 0.50):.3f}/"
              f"{percentile(emission_ms, 0.95):.3f}/{max(emission_ms):.3f}")
    budget_rejections = [row for row in rows
                         if row.get("status") == "rejected" and
                         "budget" in str(row.get("reason", "")).lower()]
    if budget_rejections:
        print(f"  explicit publication-budget rejection events="
              f"{len(budget_rejections)}")

    solver_rows = [row for row in rows if isinstance(row.get("solver"), dict)]
    if not solver_rows:
        print("MPC solver status: not recorded in this bag schema")
        return
    termination_counts: Counter[str] = Counter()
    for row in solver_rows:
        solver = row["solver"]
        status = int(solver.get("admm_status", -1))
        if status == 0:
            termination_counts["converged"] += 1
        elif status == 1:
            budget_exhausted = solver.get("time_budget_exhausted")
            if budget_exhausted is True:
                termination_counts["wall_time_budget"] += 1
            elif budget_exhausted is False:
                termination_counts["iteration_cap"] += 1
            else:
                termination_counts["limit_cause_unrecorded"] += 1
        elif status == -1:
            termination_counts["not_run"] += 1
        else:
            termination_counts["solver_error"] += 1
    converged = [row for row in solver_rows
                 if int(row["solver"].get("admm_status", -1)) == 0]
    iterations = [int(row["solver"].get("iterations", 0)) for row in converged]
    residuals = [max(float(row["solver"].get("primal_residual", 0.0)),
                     float(row["solver"].get("dual_residual", 0.0)))
                 for row in converged]
    solve_ms = [float(row["solver"].get("cycle_us", 0.0)) * 1.0e-3
                for row in solver_rows]
    accepted = sum(row.get("status") in (
                       "accepted_optimal", "accepted_feasible",
                       "accepted_line_search", "accepted_prior_plan")
                   for row in solver_rows)
    print(
        f"single-solve MPC: exact-valid solver candidates="
        f"{accepted}/{len(solver_rows)}; "
        f"ADMM termination={dict(termination_counts)}"
    )
    if drive_commands is not None:
        print(f"  recorded MPC drive-command messages={len(drive_commands)} "
              "(published outputs; distinct from candidate validity)")
    if iterations:
        print(
            "  converged iterations p50/p95/max="
            f"{percentile(iterations, 0.50)}/{percentile(iterations, 0.95)}/"
            f"{max(iterations)}; max residual p50/p95/max="
            f"{percentile(residuals, 0.50):.4g}/"
            f"{percentile(residuals, 0.95):.4g}/{max(residuals):.4g}"
        )
    if solve_ms:
        print(
            "  solver-reported cycle ms p50/p95/max="
            f"{percentile(solve_ms, 0.50):.3f}/"
            f"{percentile(solve_ms, 0.95):.3f}/{max(solve_ms):.3f}"
        )
    if accepted:
        total_cycle_us = sum(float(row["solver"].get("cycle_us", 0.0))
                             for row in solver_rows)
        print(f"  valid-candidate-weighted cycle cost="
              f"{total_cycle_us / accepted * 1.0e-3:.3f} ms per exact-valid candidate")
    timing_fields = (
        ("nominal build", "nominal_build_us", 1.0e-3),
        ("cold-start command-queue seed", "cold_start_queue_seed_us", 1.0e-3),
        ("nominal rollout A", "nominal_rollout_a_us", 1.0e-3),
        ("progress/reference resampling", "progress_reference_resampling_us", 1.0e-3),
        ("nominal rollout B", "nominal_rollout_b_us", 1.0e-3),
        ("QP build", "qp_build_us", 1.0e-3),
        ("Jacobian wall", "jacobian_wall_us", 1.0e-3),
        ("Jacobian worker CPU", "jacobian_thread_cpu_us", 1.0e-3),
        ("Jacobian slowest stage", "jacobian_slowest_stage_us", 1.0e-3),
        ("ADMM total", "admm_us", 1.0e-3),
        ("ADMM cold initialization", "cold_start_initialization_time_ns", 1.0e-6),
        ("Riccati factorization", "factorization_time_ns", 1.0e-6),
        ("Riccati RHS passes", "riccati_rhs_time_ns", 1.0e-6),
        ("ADMM projection/residual", "admm_projection_time_ns", 1.0e-6),
        ("ADMM rho updates", "rho_update_time_ns", 1.0e-6),
        ("QP scaling transform", "scaling_transform_time_ns", 1.0e-6),
        ("QP output unscale", "output_unscale_time_ns", 1.0e-6),
        ("exact candidate rollout", "exact_rollout_us", 1.0e-3),
        ("nonlinear validation", "nonlinear_validation_us", 1.0e-3),
        ("RTI solve", "solve_us", 1.0e-3),
        ("node solve", "node_solve_us", 1.0e-3),
    )
    for label, field, to_ms in timing_fields:
        values = [float(row["solver"][field]) * to_ms
                  for row in solver_rows
                  if field in row["solver"] and
                  math.isfinite(float(row["solver"][field]))]
        if values:
            print(
                f"  {label} ms p50/p95/max="
                f"{percentile(values, 0.50):.3f}/"
                f"{percentile(values, 0.95):.3f}/{max(values):.3f}"
            )


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
    collision_events: list[dict[str, Any]] | None = None,
    steering_command_samples: list[dict[str, Any]] | None = None,
    throttle_command_samples: list[dict[str, Any]] | None = None,
) -> None:
    event_rows = [(row, row.get("json", {})) for row in events
                  if row.get("json")]
    plans = [(row, payload) for row, payload in event_rows
             if payload.get("event") == "plan"]
    if not plans:
        print("MPC frozen-horizon test: no latched plan was recorded")
        return
    plan_event, plan = plans[-1]
    plan_id = plan.get("plan_id")
    payloads = [payload for _, payload in event_rows
                if payload.get("plan_id") == plan_id]
    samples = [row for row in payloads if row.get("event") == "sample"]
    output_events = [(row, payload) for row, payload in event_rows
                     if payload.get("event") == "output" and
                     payload.get("plan_id") == plan_id]
    output_events.sort(key=lambda item: item[1].get("stage", -1))
    outputs = [payload for _, payload in output_events]
    completions = [row for row in payloads if row.get("event") == "complete"]
    aborts = [row for row in payloads if row.get("event") == "abort"]
    observation_gaps = [row for row in payloads
                        if row.get("event") == "observation_gap"]
    controls = plan.get("controls", [])
    expected_horizon = int(plan.get("horizon", len(controls)))
    output_rows = sorted(outputs, key=lambda row: row.get("stage", -1))
    expected_stages = list(range(min(len(output_rows), len(controls))))
    observed_stages = [row.get("stage") for row in output_rows]
    stage_sequence_ok = observed_stages == expected_stages
    control_deltas = []
    for row in output_rows:
        stage = row.get("stage", -1)
        actual = row.get("control", [])
        if (not isinstance(stage, int) or stage < 0 or stage >= len(controls) or
                len(actual) != 2 or len(controls[stage]) != 2):
            control_deltas.append(math.inf)
            continue
        control_deltas.extend(
            abs(float(actual[index]) - float(controls[stage][index]))
            for index in range(2)
        )
    max_control_delta = max(control_deltas, default=math.inf)
    plan_commands_match = (stage_sequence_ok and
                           max_control_delta <= 1.0e-6)
    full_plan_emitted = (plan_commands_match and
                         len(output_rows) == len(controls) == expected_horizon)
    step_samples = sorted(samples, key=lambda row: row.get("stage", -1))
    sample_times = [row["control_stamp_ns"] * 1e-9
                    for row in step_samples if row.get("control_stamp_ns")]
    print("\nMPC frozen-plan execution:")
    print(
        f"  plan_id={plan_id} speed_at_latch="
        f"{plan.get('trigger_speed_mps', math.nan):.3f}m/s "
        f"steps={len(controls)} declared_horizon={expected_horizon} "
        f"dt={plan.get('step_s', math.nan)*1000:.2f}ms "
        f"completed={bool(completions)} aborted={bool(aborts)}"
    )
    if observation_gaps:
        missing = sorted({int(row.get("stage", -1)) for row in observation_gaps})
        print(f"  missing-state observations={len(observation_gaps)} stages={missing}")
    completion = completions[-1] if completions else {}
    print(
        f"  emitted plan commands={len(output_rows)}/{len(controls)} "
        f"stages_contiguous={stage_sequence_ok} "
        f"controls_match_plan={plan_commands_match} "
        f"max_control_delta={max_control_delta:.3g} "
        f"full_plan_emitted={full_plan_emitted} "
        f"terminal_state_observed={completion.get('terminal_state_observed', False)}"
    )

    def compare_actuator_topic(
        label: str, topic_rows: list[dict[str, Any]] | None,
        control_index: int, scale: float,
    ) -> None:
        if topic_rows is None:
            return
        unused = set(range(len(topic_rows)))
        time_errors = []
        value_errors = []
        matched_stages = 0
        for event_row, output in output_events:
            stamp = event_row.get("bag_time")
            stage = output.get("stage", -1)
            if (stamp is None or not isinstance(stage, int) or
                    stage < 0 or stage >= len(controls)):
                continue
            nearby = [
                (abs(topic_rows[index].get("bag_time", math.inf) - stamp), index)
                for index in unused
                if abs(topic_rows[index].get("bag_time", math.inf) - stamp) <= 0.020
            ]
            if not nearby:
                continue
            delta_t, index = min(nearby)
            unused.remove(index)
            target = float(controls[stage][control_index]) * scale
            if control_index == 0:
                target = max(-1.0, min(1.0, target))
            else:
                target = max(0.0, min(1.0, target))
            time_errors.append(delta_t)
            value_errors.append(abs(float(topic_rows[index]["value"]) - target))
            matched_stages += 1
        all_matched = bool(output_events) and matched_stages == len(output_events)
        exact_values = bool(value_errors) and max(value_errors) <= 1.0e-5
        print(
            f"  {label} actuator-interface topic: matched={matched_stages}/"
            f"{len(output_events)} all_matched={all_matched} "
            f"values_match_plan={exact_values} "
            f"max_abs_error={max(value_errors, default=math.inf):.3g} "
            f"max_time_offset_ms={max(time_errors, default=math.inf)*1000:.2f}"
        )

    compare_actuator_topic(
        "steering", steering_command_samples, 0, 1.0 / MAX_STEERING_RAD)
    compare_actuator_topic("throttle", throttle_command_samples, 1, 1.0)
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

    plan_bag_time = float(plan_event.get("bag_time", math.nan))
    step_s = float(plan.get("step_s", math.nan))
    horizon_end_bag_time = plan_bag_time + expected_horizon * step_s
    collisions = collision_events or []
    prior_collision_counts = [
        float(row.get("value", 0.0)) for row in collisions
        if row.get("bag_time", math.inf) <= plan_bag_time
    ]
    collision_baseline = max(prior_collision_counts, default=0.0)
    collision_during_plan = any(
        float(row.get("value", 0.0)) > collision_baseline and
        plan_bag_time < row.get("bag_time", -math.inf) <= horizon_end_bag_time
        for row in collisions
    )
    if collisions:
        print(f"  new collision during frozen horizon={collision_during_plan}")
    else:
        print("  new collision during frozen horizon=unverified (no collision stream)")
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
    scored_times_s: list[float] = []
    scored_stages: list[int] = []
    scored_pose_pairs: list[
        tuple[tuple[float, float, float], dict[str, float]]
    ] = []
    scored_state_pairs: list[
        tuple[list[float], dict[str, float]]
    ] = []
    full_plan = [row for row in plan.get("predictions", []) if row]
    scored_rows: list[
        tuple[int, float, list[float], list[float], dict[str, Any] | None]
    ] = []
    if full_plan and plan.get("stamp_ns"):
        plan_stamp = int(plan["stamp_ns"]) * 1e-9
        for prediction in full_plan:
            timestamp = plan_stamp + float(prediction.get("time_s", 0.0))
            scored_rows.append((int(prediction.get("stage", -1)), timestamp,
                                prediction.get("pose", []),
                                prediction.get("state", []), None))
        print("  accuracy source: complete frozen-plan states vs Unity ground truth")
    else:
        for row in step_samples:
            if row.get("control_stamp_ns"):
                scored_rows.append((int(row.get("stage", -1)),
                    row["control_stamp_ns"] * 1e-9,
                    row.get("predicted_pose", []), row.get("predicted_state", []), row))
        print("  accuracy source: sparse frozen-plan states vs Unity ground truth")

    worst_yaw_rate_rows: list[tuple[float, int, float, float, float, float]] = []
    for stage, timestamp, predicted_pose, predicted_state, source_row in scored_rows:
        measured = truth_at(timestamp)
        if measured is None or len(predicted_pose or []) != 3 or len(predicted_state or []) < 6:
            continue
        scored_times_s.append(timestamp - int(plan["stamp_ns"]) * 1.0e-9)
        scored_stages.append(stage)
        px, py, pyaw = predicted_pose
        scored_pose_pairs.append(((px, py, pyaw), measured))
        scored_state_pairs.append((predicted_state, measured))
        errors["position"].append(math.hypot(px - measured["x"], py - measured["y"]))
        errors["yaw"].append(abs(normalized_angle(pyaw - measured["yaw"])))
        errors["u"].append(predicted_state[2] - measured["u"])
        # The plant's MPC lateral velocity is at the rear axle; simulator
        # odometry reports the rigid-body/COM velocity.
        measured_v_rear = measured["v"] - measured["r"] * COM_TO_REAR_AXLE_M
        errors["v_rear"].append(predicted_state[3] - measured_v_rear)
        errors["r"].append(predicted_state[4] - measured["r"])
        prior_control = controls[stage - 1] if 0 < stage <= len(controls) else None
        worst_yaw_rate_rows.append((
            abs(predicted_state[4] - measured["r"]), stage,
            predicted_state[4], measured["r"], measured["u"],
            math.nan if prior_control is None else float(prior_control[0]),
        ))
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
        available_prediction_count = len(full_plan) if full_plan else len(step_samples)
        print(
            f"  truth-aligned prediction coverage: n={len(scored_times_s)}/"
            f"{available_prediction_count} time_s="
            f"{min(scored_times_s):.3f}..{max(scored_times_s):.3f}"
        )
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
        predicted_start_pose, measured_start = scored_pose_pairs[0]
        predicted_start_state, _ = scored_state_pairs[0]
        start_position_error = math.hypot(
            predicted_start_pose[0] - measured_start["x"],
            predicted_start_pose[1] - measured_start["y"],
        )
        start_yaw_error = normalized_angle(
            predicted_start_pose[2] - measured_start["yaw"])
        start_v_rear = measured_start["v"] - measured_start["r"] * COM_TO_REAR_AXLE_M
        print(
            f"  initial estimate-vs-truth: xy={start_position_error:.5f}m "
            f"yaw={start_yaw_error:.5f}rad "
            f"u={predicted_start_state[2] - measured_start['u']:+.5f}m/s "
            f"v_rear={predicted_start_state[3] - start_v_rear:+.5f}m/s "
            f"r={predicted_start_state[4] - measured_start['r']:+.5f}rad/s"
        )

        # Remove only the initial rigid pose offset to separate rollout drift
        # from localization/coordinate-origin error. This is an offline score
        # decomposition, never a correction applied to the model or controller.
        yaw_alignment = normalized_angle(
            measured_start["yaw"] - predicted_start_pose[2])
        c_align, s_align = math.cos(yaw_alignment), math.sin(yaw_alignment)
        relative_position_errors = []
        relative_yaw_errors = []
        for predicted_pose, measured in scored_pose_pairs:
            dx = predicted_pose[0] - predicted_start_pose[0]
            dy = predicted_pose[1] - predicted_start_pose[1]
            aligned_x = measured_start["x"] + c_align * dx - s_align * dy
            aligned_y = measured_start["y"] + s_align * dx + c_align * dy
            relative_position_errors.append(math.hypot(
                aligned_x - measured["x"], aligned_y - measured["y"]))
            relative_yaw_errors.append(normalized_angle(
                (predicted_pose[2] - predicted_start_pose[2]) -
                (measured["yaw"] - measured_start["yaw"])))
        print(
            f"  rollout-only after initial pose alignment: XY RMSE="
            f"{math.sqrt(statistics.mean(x*x for x in relative_position_errors)):.5f}m "
            f"p95/max={percentile(relative_position_errors, 0.95):.5f}/"
            f"{max(relative_position_errors):.5f}m; relative yaw RMSE="
            f"{math.sqrt(statistics.mean(x*x for x in relative_yaw_errors)):.5f}rad "
            f"max_abs={max(abs(x) for x in relative_yaw_errors):.5f}rad"
        )
        yaw_rate_pose_diagnostics = []
        for index in range(1, len(scored_pose_pairs) - 1):
            elapsed = scored_times_s[index + 1] - scored_times_s[index - 1]
            if elapsed <= 0.0 or elapsed > 0.075:
                continue
            predicted_before = scored_pose_pairs[index - 1][0][2]
            predicted_after = scored_pose_pairs[index + 1][0][2]
            truth_before = scored_pose_pairs[index - 1][1]["yaw"]
            truth_after = scored_pose_pairs[index + 1][1]["yaw"]
            predicted_yaw_rate = normalized_angle(
                predicted_after - predicted_before) / elapsed
            truth_yaw_rate = normalized_angle(
                truth_after - truth_before) / elapsed
            predicted_state = scored_state_pairs[index][0]
            truth_rate = scored_pose_pairs[index][1]["r"]
            yaw_rate_pose_diagnostics.append((
                abs(predicted_state[4] - truth_rate), scored_stages[index],
                predicted_state[4], truth_rate, predicted_yaw_rate,
                truth_yaw_rate,
            ))
        if yaw_rate_pose_diagnostics:
            predicted_rate_residuals = [row[2] - row[4]
                                        for row in yaw_rate_pose_diagnostics]
            truth_rate_residuals = [row[3] - row[5]
                                    for row in yaw_rate_pose_diagnostics]
            print(
                "  centered pose-yaw-rate check (25 ms samples): "
                f"predicted r vs pose derivative RMSE="
                f"{math.sqrt(statistics.mean(x*x for x in predicted_rate_residuals)):.4f}rad/s; "
                f"Unity r vs pose derivative RMSE="
                f"{math.sqrt(statistics.mean(x*x for x in truth_rate_residuals)):.4f}rad/s"
            )
        print(
            f"  truth speed during scored horizon: "
            f"{min(speeds):.3f}-{max(speeds):.3f}m/s"
        )
        if controls:
            steering = [float(control[0]) for control in controls]
            throttle = [float(control[1]) for control in controls]
            max_steering_step = max(
                (abs(right - left) for left, right in zip(steering, steering[1:])),
                default=0.0,
            )
            print(
                f"  latched controls: steering={min(steering):+.4f}.."
                f"{max(steering):+.4f}rad max_step={max_steering_step:.4f}rad; "
                f"throttle={min(throttle):.4f}..{max(throttle):.4f} "
                f"all_drive={min(throttle) > 0.0}"
            )
        yaw_pose_by_stage = {row[1]: row for row in yaw_rate_pose_diagnostics}
        for absolute_error, stage, predicted_r, measured_r, measured_u, prior_steer in sorted(
                worst_yaw_rate_rows, reverse=True)[:5]:
            next_steer = (
                float(controls[stage][0])
                if 0 <= stage < len(controls) else math.nan
            )
            pose_rates = yaw_pose_by_stage.get(stage)
            pose_rate_text = (
                "" if pose_rates is None else
                f" pred/GT pose_yawdot={pose_rates[4]:+.3f}/{pose_rates[5]:+.3f}rad/s"
            )
            print(
                f"  yaw-rate tail: stage={stage} t={stage * plan.get('step_s', 0.0):.3f}s "
                f"pred={predicted_r:+.4f} truth={measured_r:+.4f} "
                f"abs_error={absolute_error:.4f}rad/s GT_u={measured_u:.3f}m/s "
                f"steer_prev/next={prior_steer:+.4f}/{next_steer:+.4f}rad"
                f"{pose_rate_text}"
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


def report_localization_diagnostics(
    truth: list[dict[str, Any]], data: dict[str, list[dict[str, Any]]],
) -> None:
    """Truth-score recorded localization outputs and summarize AMCL updates."""
    pose_topics = (
        (AMCL_POSE, "AMCL scan pose"),
        (CURRENT_MAP_POSE, "AMCL control map pose"),
        (EKF_POSE, "EKF pose"),
    )
    for topic, label in pose_topics:
        errors_xy = []
        errors_yaw = []
        matched = 0
        for estimate in data.get(topic, []):
            gt = nearest_by_stamp(truth, estimate["stamp"])
            if gt is None or abs(gt["stamp"] - estimate["stamp"]) > 0.0125:
                continue
            matched += 1
            errors_xy.append(math.hypot(estimate["x"] - gt["x"],
                                        estimate["y"] - gt["y"]))
            errors_yaw.append(abs(normalized_angle(
                estimate["yaw"] - gt["yaw"])))
        if errors_xy:
            print(
                f"{label} vs Unity truth: matched={matched}/"
                f"{len(data.get(topic, []))} xy_rmse="
                f"{math.sqrt(statistics.mean(e*e for e in errors_xy)):.3f}m "
                f"xy_p95/max={percentile(errors_xy, 0.95):.3f}/"
                f"{max(errors_xy):.3f}m yaw_p95/max="
                f"{percentile(errors_yaw, 0.95):.3f}/"
                f"{max(errors_yaw):.3f}rad"
            )

    collision_events = [
        row["bag_time"] for row in data.get(COLLISION, [])
        if row.get("value", 0) > 0
    ]
    if collision_events:
        first_collision_time = min(collision_events)
        pre_collision_truth = [
            row for row in truth
            if row.get("bag_time", math.inf) < first_collision_time
        ]
        for topic, label in pose_topics:
            errors_xy = []
            for estimate in data.get(topic, []):
                if estimate["bag_time"] >= first_collision_time:
                    continue
                gt = nearest_by_stamp(pre_collision_truth, estimate["stamp"])
                if gt is None or abs(gt["stamp"] - estimate["stamp"]) > 0.0125:
                    continue
                errors_xy.append(math.hypot(estimate["x"] - gt["x"],
                                            estimate["y"] - gt["y"]))
            if errors_xy:
                print(
                    f"  before collision, {label}: n={len(errors_xy)} "
                    f"xy_rmse={math.sqrt(statistics.mean(e*e for e in errors_xy)):.3f}m "
                    f"xy_p95/max={percentile(errors_xy, 0.95):.3f}/"
                    f"{max(errors_xy):.3f}m"
                )

    health_rows = [row for row in data.get(AMCL_HEALTH, [])
                   if len(row.get("values", [])) >= 10]
    scan_rows = [row for row in data.get(AMCL_SCAN_ALIGNMENT, [])
                 if len(row.get("values", [])) >= 14]
    if health_rows or scan_rows:
        applied_xy = [row["values"][8] for row in health_rows]
        applied_yaw = [abs(row["values"][9]) for row in health_rows]
        scan_accepted = sum(row["values"][3] >= 0.5 for row in scan_rows)
        print(
            "AMCL updates: "
            f"health={len(health_rows)}, scan-alignment={len(scan_rows)}, "
            f"scan accepted={scan_accepted}/{len(scan_rows)}, "
            f"applied xy correction p95/max="
            f"{percentile(applied_xy, 0.95) if applied_xy else math.nan:.3f}/"
            f"{max(applied_xy) if applied_xy else math.nan:.3f}m, "
            f"yaw max={max(applied_yaw) if applied_yaw else math.nan:.3f}rad"
        )


def report_csv_path_follow(
    diagnostics: list[dict[str, Any]],
    drive_commands: list[dict[str, Any]],
    csv_reference: list[dict[str, float]] | None = None,
) -> None:
    """Score continuous GT-indexed CSV path-following test telemetry."""
    events = [row.get("json", {}) for row in diagnostics if row.get("json")]
    tracking = [row for row in events if row.get("event") == "tracking"]
    completed = [row for row in events if row.get("event") == "complete"]
    aborted = [row for row in events if row.get("event") == "abort"]
    if not events:
        print("CSV path follow: no diagnostics recorded")
        return
    print("\nContinuous CSV path-follow test:")
    print(
        f"  tracking ticks={len(tracking)} commands={len(drive_commands)} "
        f"complete={bool(completed)} abort="
        f"{aborted[-1].get('reason') if aborted else 'none'}"
    )
    rows = tracking or completed
    if rows:
        progress = [float(row["csv_progress_m"]) for row in rows
                    if "csv_progress_m" in row]
        distances = [float(row["reference_distance_m"]) for row in tracking
                     if "reference_distance_m" in row]
        heading = [abs(float(row["reference_heading_error_rad"]))
                   for row in tracking
                   if "reference_heading_error_rad" in row]
        intervals = [float(row["command_timer_interval_ms"])
                     for row in tracking
                     if row.get("command_timer_interval_ms") is not None]
        if progress:
            total = float(rows[-1].get("csv_total_progress_m", math.nan))
            followed = max(0.0, progress[-1] - progress[0])
            available = max(0.0, total - progress[0])
            fraction = (followed / available if available > 0.0 else math.nan)
            print(
                f"  CSV progress={progress[0]:.3f}->{progress[-1]:.3f}m; "
                f"followed={followed:.3f}/{available:.3f}m "
                f"({100.0*fraction:.1f}% of available trace)"
            )
        if distances:
            print(
                f"  path distance RMSE="
                f"{math.sqrt(statistics.mean(x*x for x in distances)):.3f}m "
                f"p95/max={percentile(distances, 0.95):.3f}/"
                f"{max(distances):.3f}m; heading error p95/max="
                f"{percentile(heading, 0.95):.3f}/"
                f"{max(heading):.3f}rad"
            )
        if intervals:
            print(
                f"  controller timer interval median/p95/max="
                f"{statistics.median(intervals):.2f}/"
                f"{percentile(intervals, 0.95):.2f}/{max(intervals):.2f}ms"
            )
        for target, actual, label, unit in (
            ("csv_target_yaw_rate_radps", "ground_truth_yaw_rate_radps",
             "CSV target yaw rate", "rad/s"),
        ):
            errors = [float(row[actual]) - float(row[target])
                      for row in tracking
                      if row.get(target) is not None and
                      row.get(actual) is not None]
            if errors:
                absolute = [abs(value) for value in errors]
                print(
                    f"  {label} tracking error: RMSE="
                    f"{math.sqrt(statistics.mean(x*x for x in errors)):.3f}{unit} "
                    f"p95/max={percentile(absolute, 0.95):.3f}/"
                    f"{max(absolute):.3f}{unit}"
                )
        if csv_reference:
            speed_errors = []
            for row in tracking:
                logged_target = row.get("csv_target_speed_mps")
                if logged_target is not None:
                    speed_errors.append(
                        float(row["ground_truth_u_mps"]) -
                        float(logged_target))
                    continue
                index = int(row.get("csv_index", -1))
                if not 0 <= index < len(csv_reference):
                    continue
                fraction = max(
                    0.0, min(1.0, float(row.get("csv_fraction", 0.0))))
                reference_index = index % len(csv_reference)
                next_index = (reference_index + 1) % len(csv_reference)
                first_speed = csv_reference[reference_index]["target_speed_mps"]
                next_speed = csv_reference[next_index]["target_speed_mps"]
                target_speed = first_speed + fraction * (next_speed - first_speed)
                speed_errors.append(
                    float(row["ground_truth_u_mps"]) - target_speed)
            if speed_errors:
                absolute = [abs(value) for value in speed_errors]
                print(
                    f"  CSV target speed tracking error: RMSE="
                    f"{math.sqrt(statistics.mean(x*x for x in speed_errors)):.3f}m/s "
                    f"p95/max={percentile(absolute, 0.95):.3f}/"
                    f"{max(absolute):.3f}m/s"
                )
        else:
            print("  CSV target speed tracking error: reference CSV not supplied")
        throttle = [float(row["published_throttle_norm"])
                    for row in tracking if "published_throttle_norm" in row]
        steering = [float(row["published_steering_rad"])
                    for row in tracking if "published_steering_rad" in row]
        if throttle and steering:
            print(
                f"  output range: throttle={min(throttle):.3f}.."
                f"{max(throttle):.3f}; PP steering="
                f"{min(steering):+.3f}..{max(steering):+.3f}rad"
            )
    if completed:
        print(
            f"  completion endpoint: progress="
            f"{completed[-1].get('csv_progress_m', math.nan):.3f}m, "
            f"GT=({completed[-1].get('ground_truth_x_m', math.nan):.3f}, "
            f"{completed[-1].get('ground_truth_y_m', math.nan):.3f})m"
        )


def report_lap_splits(
    last_lap_rows: list[dict[str, Any]],
) -> None:
    """List distinct finite simulator-published completed-lap times."""
    splits = []
    previous = None
    for row in sorted(last_lap_rows, key=lambda item: item["bag_time"]):
        value = float(row.get("value", math.nan))
        if math.isfinite(value) and value != previous:
            splits.append(value)
        previous = value
    if not splits:
        print("  no finite completed-lap times recorded")
        return
    for index, split in enumerate(splits, start=1):
        print(f"  reported lap {index}: {split:.3f}s")
    print(
        f"  reported lap-time mean={statistics.mean(splits):.3f}s "
        f"range={min(splits):.3f}..{max(splits):.3f}s"
    )


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

    # The bridge stamps sensor headers at request time, although the values are
    # decoded from the later response. Compare the same pose/twist increments
    # using bag receipt times as a proxy for telemetry capture time. This is a
    # diagnostic of timestamp semantics, not a replacement for Unity physics
    # time: receipt time also contains transport and ROS scheduling delay.
    receipt_times = [row["bag_time"] for row in rows]

    def sample_at_receipt(timestamp: float) -> tuple[float, float, float, float] | None:
        index = bisect.bisect_left(receipt_times, timestamp)
        if index < len(rows) and abs(receipt_times[index] - timestamp) < 1.0e-9:
            row = rows[index]
            return row["u"], row["v"], row["r"], row["yaw"]
        if index == 0 or index == len(rows):
            return None
        before, after = rows[index - 1], rows[index]
        gap = receipt_times[index] - receipt_times[index - 1]
        if gap <= 0.0 or gap > 0.080:
            return None
        fraction = (timestamp - receipt_times[index - 1]) / gap
        yaw_delta = normalized_angle(after["yaw"] - before["yaw"])
        return (
            before["u"] + fraction * (after["u"] - before["u"]),
            before["v"] + fraction * (after["v"] - before["v"]),
            before["r"] + fraction * (after["r"] - before["r"]),
            before["yaw"] + fraction * yaw_delta,
        )

    receipt_errors: list[float] = []
    receipt_dts: list[float] = []
    request_errors_by_index = {
        source_index: error
        for (source_index, _, _, _, _), error in zip(intervals, zero[1])
    }
    request_errors_comparable: list[float] = []
    for index, (before, after) in enumerate(zip(rows, rows[1:])):
        dt = receipt_times[index + 1] - receipt_times[index]
        if dt < 0.010 or dt > 0.080:
            continue
        t0, t1 = receipt_times[index], receipt_times[index + 1]
        first = sample_at_receipt(t0)
        middle = sample_at_receipt(0.5 * (t0 + t1))
        last = sample_at_receipt(t1)
        if first is None or middle is None or last is None:
            continue
        vx0, vy0 = world_velocity(first)
        vxm, vym = world_velocity(middle)
        vx1, vy1 = world_velocity(last)
        predicted_x = dt * (vx0 + 4.0 * vxm + vx1) / 6.0
        predicted_y = dt * (vy0 + 4.0 * vym + vy1) / 6.0
        receipt_errors.append(math.hypot(
            predicted_x - (after["x"] - before["x"]),
            predicted_y - (after["y"] - before["y"]),
        ))
        receipt_dts.append(dt)
        if index in request_errors_by_index:
            request_errors_comparable.append(request_errors_by_index[index])

    if receipt_errors:
        print(
            "  timestamp diagnostic (not an applied correction): "
            f"request-header dt RMSE="
            f"{math.sqrt(statistics.mean(x*x for x in request_errors_comparable))*100:.3f}cm "
            f"(n={len(request_errors_comparable)}); bag-receipt dt RMSE="
            f"{math.sqrt(statistics.mean(x*x for x in receipt_errors))*100:.3f}cm "
            f"(n={len(receipt_errors)}), dt median/p95="
            f"{statistics.median(receipt_dts)*1e3:.2f}/"
            f"{percentile(receipt_dts, 0.95)*1e3:.2f}ms"
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
    a model-vs-proxy comparison, never labelled independent truth. This does
    not compare encoder angle with the plant's wheel-pose angle: Unity's encoder
    integrates pre-physics RPM while PxVehicle integrates wheel pose after its
    wheel-speed solve.
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
    summarize("all", rows)
    summarize("zero-throttle/brake command", [x for x in rows if x["throttle"] < 0.01])
    summarize("positive throttle", [x for x in rows if x["throttle"] >= 0.01])
    summarize("wheel/body speed mismatch >0.5 m/s", [
        x for x in rows if max(abs(x["left_surface_minus_body"]),
                               abs(x["right_surface_minus_body"])) > 0.5])


def report_actuator_feedback_packet_association(
    plant_rows: list[dict[str, Any]], bridge_rows: list[dict[str, Any]],
) -> None:
    """Check which received Unity telemetry packet supplied each odom feedback.

    Actuator feedback topics are headerless Float32 messages. Production odom
    snapshots their latest callback when a stamped sensor packet completes;
    this diagnostic compares that snapshot with the feedback fields decoded
    from the Bridge response associated with the same request epoch and its
    neighbors. It diagnoses callback/packet association only; it does not
    change the plant input or infer the command actually applied per physics
    substep.
    """
    if not plant_rows or not bridge_rows:
        return

    packets: list[dict[str, Any]] = []
    for row in bridge_rows:
        payload = row.get("json", {})
        request_ns = payload.get("bridge_request_ros_stamp_ns")
        if request_ns is None:
            continue
        try:
            request_time = round(int(request_ns) * 1.0e-9, 6)
        except (TypeError, ValueError):
            continue
        packets.append({"request_time": request_time, "payload": payload})
    packets.sort(key=lambda row: row["request_time"])
    if not packets:
        return

    by_request_time = [row["request_time"] for row in packets]
    feedback_fields = (
        ("steering", "steering_feedback", "simulator_feedback_steering_norm"),
        ("throttle", "throttle_feedback", "simulator_feedback_throttle_norm"),
    )
    errors: dict[str, dict[int, list[float]]] = {
        name: {offset: [] for offset in (-1, 0, 1)}
        for name, _, _ in feedback_fields
    }
    matched = 0
    valid = 0
    receive_latencies_ms: list[float] = []
    for plant in plant_rows:
        index = bisect.bisect_left(by_request_time, round(plant["stamp"], 6))
        candidates = [i for i in (index - 1, index)
                      if 0 <= i < len(packets)]
        if not candidates:
            continue
        packet_index = min(
            candidates,
            key=lambda i: abs(by_request_time[i] - plant["stamp"]))
        if abs(by_request_time[packet_index] - plant["stamp"]) > 1.0e-5:
            continue
        matched += 1
        payload = packets[packet_index]["payload"]
        receive_ns = payload.get("bridge_receive_ros_stamp_ns")
        request_ns = payload.get("bridge_request_ros_stamp_ns")
        if receive_ns is not None and request_ns is not None:
            try:
                receive_latencies_ms.append(
                    (int(receive_ns) - int(request_ns)) * 1.0e-6)
            except (TypeError, ValueError):
                pass
        if not plant.get("actuator_feedback_valid", False):
            continue
        valid += 1
        for name, plant_key, packet_key in feedback_fields:
            observed = plant.get(plant_key)
            if observed is None:
                continue
            for offset in (-1, 0, 1):
                neighbor = packet_index + offset
                if not 0 <= neighbor < len(packets):
                    continue
                value = packets[neighbor]["payload"].get(packet_key)
                if value is None:
                    continue
                try:
                    errors[name][offset].append(float(observed) - float(value))
                except (TypeError, ValueError):
                    continue

    print("\nActuator-feedback/Bridge-response association:")
    print("  UnityPlantState stores the callback snapshot chosen by production odom;")
    print("  Bridge fields are response telemetry, not per-substep applied-input truth.")
    print(f"  request-epoch matched={matched}/{len(plant_rows)}; "
          f"feedback-valid={valid}/{matched}")
    if receive_latencies_ms:
        print("  Bridge request→response: "
              f"median={statistics.median(receive_latencies_ms):.2f}ms "
              f"p95={percentile(receive_latencies_ms, 0.95):.2f}ms "
              f"max={max(receive_latencies_ms):.2f}ms")
    for name, _, _ in feedback_fields:
        print(f"  {name} callback snapshot minus decoded response value:")
        for offset in (-1, 0, 1):
            residuals = errors[name][offset]
            if not residuals:
                continue
            absolute = [abs(value) for value in residuals]
            print(
                f"    response epoch i{offset:+d}: n={len(residuals)} "
                f"RMSE={math.sqrt(statistics.mean(x*x for x in residuals)):.6g} "
                f"p95_abs={percentile(absolute, 0.95):.6g} "
                f"within_1e-4={sum(value <= 1.0e-4 for value in absolute)}/"
                f"{len(absolute)}")


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
    left_encoder: list[dict[str, Any]], right_encoder: list[dict[str, Any]],
    yaw_residual_gain: float = 0.0,
    yaw_residual_decay: float = 1.0,
    oracle_motion_initial_state: bool = False,
    diagnose_observed_state_reanchoring: bool = False,
    extra_steering_delay_packets: int = 0,
    extra_throttle_delay_packets: int = 0,
    oracle_steering_feedback: bool = False,
    oracle_throttle_feedback: bool = False,
    reconstruct_command_queue_from_stamps: bool = False,
    reconstruct_command_queue_from_bridge: bool = False,
    bridge_command_epoch_offset: int = 0,
    steering_command_samples: list[dict[str, Any]] | None = None,
    throttle_command_samples: list[dict[str, Any]] | None = None,
    bridge_timing_samples: list[dict[str, Any]] | None = None,
    reconcile_suspension_jounce_at_start: bool = False,
) -> None:
    """Replay only odom plant state and queued commands through the C++ MPC plant.

    Starts from a recorded UnityPlantState, then advances with the recorded
    slot-1 actuator command sequence. It never reads future measured states
    during a rollout; simulator truth is used only to score each endpoint.
    The optional oracle-motion diagnostic replaces only initial u/v/r with
    same-epoch Unity truth. Hidden plant states and all future inputs remain
    unchanged; this is diagnostic only and must never be used in production.
    """
    if min(extra_steering_delay_packets, extra_throttle_delay_packets) < 0:
        raise ValueError("extra command delays must be nonnegative")
    if (reconstruct_command_queue_from_stamps and
            reconstruct_command_queue_from_bridge):
        raise ValueError("select only one command-queue reconstruction source")
    if bridge_command_epoch_offset and not reconstruct_command_queue_from_bridge:
        raise ValueError("bridge command epoch offset requires bridge reconstruction")
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
    reconcile_jounce = model.unity_source_plant_reconcile_suspension_jounce_c
    reconcile_jounce.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    reconcile_jounce.restype = ctypes.c_int

    truth_by_stamp = {round(row["stamp"], 6): row for row in truth}
    plant_by_stamp = {round(row["stamp"], 6): row for row in plant}
    # Float64MultiArray has no ROS header, so its first value is the local
    # observer clock (which starts at zero), not the bag/Unity epoch. Pair it
    # to the plant publication by recorder receive time; both are emitted in
    # sequence from the same sensor callback.
    raw_yaw_forecast_by_stamp: dict[float, float] = {}
    raw_rear_wheel_omega_by_stamp: dict[float, tuple[float, float]] = {}
    ordered_plant = sorted(plant, key=lambda row: row["bag_time"])
    ordered_diagnostics = sorted(odom_diagnostics, key=lambda row: row["bag_time"])
    for plant_row, diagnostic_row in zip(ordered_plant, ordered_diagnostics):
        if (abs(plant_row["bag_time"] - diagnostic_row["bag_time"]) <= 0.005 and
                diagnostic_row.get("label") ==
                "sensor_corrected_source_derived_odom_v3"):
            values = diagnostic_row.get("values", [])
            stamp = round(plant_row["stamp"], 6)
            if len(values) >= 13:
                raw_yaw_forecast_by_stamp[stamp] = values[12]
            if len(values) >= 24:
                raw_rear_wheel_omega_by_stamp[stamp] = (
                    values[22], values[23])
    times = sorted(set(truth_by_stamp) & set(plant_by_stamp))
    if len(times) < 31:
        print(f"source-plant rollout: only {len(times)} aligned epochs")
        return

    def sample_commands_at_epochs(
        samples: list[dict[str, Any]] | None,
    ) -> list[float] | None:
        if not samples:
            return None
        ordered = sorted(samples, key=lambda row: row["bag_time"])
        sample_times = [row["bag_time"] for row in ordered]
        values = []
        for epoch in times:
            sample_index = bisect.bisect_right(sample_times, epoch) - 1
            values.append(
                ordered[sample_index]["value"] if sample_index >= 0 else 0.0)
        return values

    command_samples_from_bridge: tuple[list[dict[str, Any]],
                                       list[dict[str, Any]]] | None = None
    if reconstruct_command_queue_from_bridge:
        bridge_steering_samples: list[dict[str, Any]] = []
        bridge_throttle_samples: list[dict[str, Any]] = []
        for row in bridge_timing_samples or []:
            payload = row.get("json", {})
            request_stamp_ns = payload.get("bridge_request_ros_stamp_ns")
            steering_sent = payload.get("sent_steering_norm")
            throttle_sent = payload.get("sent_throttle_norm")
            if (request_stamp_ns is None or steering_sent is None or
                    throttle_sent is None):
                continue
            try:
                request_stamp = int(request_stamp_ns) * 1.0e-9
                steering_value = float(steering_sent)
                throttle_value = float(throttle_sent)
            except (TypeError, ValueError, OverflowError):
                continue
            if not (math.isfinite(request_stamp) and
                    math.isfinite(steering_value) and
                    math.isfinite(throttle_value)):
                continue
            bridge_steering_samples.append({
                "bag_time": request_stamp, "value": steering_value})
            bridge_throttle_samples.append({
                "bag_time": request_stamp, "value": throttle_value})
        command_samples_from_bridge = (
            bridge_steering_samples, bridge_throttle_samples)

    steering_samples = (command_samples_from_bridge[0]
                        if command_samples_from_bridge is not None
                        else steering_command_samples)
    throttle_samples = (command_samples_from_bridge[1]
                        if command_samples_from_bridge is not None
                        else throttle_command_samples)
    reconstruct_command_queue = (
        reconstruct_command_queue_from_stamps or
        reconstruct_command_queue_from_bridge)
    steering_command_by_epoch = sample_commands_at_epochs(steering_samples)
    throttle_command_by_epoch = sample_commands_at_epochs(throttle_samples)
    steering_feedback_phase_errors = {
        offset: [] for offset in range(-2, 4)
    }
    steering_feedback_best_phase: Counter[int] = Counter()
    steering_feedback_phase_latency_ms: dict[int, list[float]] = {
        offset: [] for offset in range(-2, 4)
    }
    steering_feedback_best_phase_latency_ms: dict[int, list[float]] = {
        offset: [] for offset in range(-2, 4)
    }
    if reconstruct_command_queue and (
        steering_command_by_epoch is None or throttle_command_by_epoch is None
    ):
        print("source-plant rollout: command reconstruction requires both "
              "steering and throttle inputs")
        return
    if reconstruct_command_queue:
        # A replayed bag can contain old source header stamps while the bag's
        # receive clock is current. Those are different clock domains, so
        # command-vs-request ordering cannot be reconstructed from bag_time.
        clock_offset = statistics.median(
            plant_by_stamp[timestamp]["bag_time"] - timestamp
            for timestamp in times)
        if abs(clock_offset) > 0.5:
            print(
                "source-plant rollout: timestamp command reconstruction is "
                f"not valid for this capture (bag/header clock offset "
                f"{clock_offset:+.3f}s); use a live request-stamped bag")
            return
        # Compare both serialized queue slots and actuator feedback against
        # nearby Bridge request epochs. The slot names are relative to the
        # observer's current publication, whereas the rollout offset is
        # relative to the preceding request; comparing only slot 1 to that
        # selected rollout epoch silently mixes those two conventions.
        # Keep this diagnostic descriptive: Bridge request stamps do not
        # expose Unity FixedUpdate/application sequence numbers.
        def report_bridge_epoch_alignment(
            label: str, observed_values: list[float],
            reference_values: list[float], valid: list[bool] | None = None,
        ) -> None:
            offsets = (-2, -1, 0, 1)
            print(f"  {label} vs Bridge sent command by request-epoch offset:")
            for offset in offsets:
                errors = [
                    observed_values[index] - reference_values[index + offset]
                    for index in range(len(times))
                    if (0 <= index + offset < len(times) and
                        (valid is None or valid[index]))
                ]
                if not errors:
                    continue
                absolute = [abs(error) for error in errors]
                print(
                    f"    C[i{offset:+d}]: n={len(errors)} "
                    f"RMSE={math.sqrt(statistics.fmean(e * e for e in errors)):.5f} "
                    f"p95_abs={percentile(absolute, 0.95):.5f} "
                    f"within_1e-3={sum(error <= 1.0e-3 for error in absolute)}"
                    f"/{len(errors)}")

        report_bridge_epoch_alignment(
            "published steering queue slot 1",
            [plant_by_stamp[timestamp]["delayed_steering_1"]
             for timestamp in times], steering_command_by_epoch)
        report_bridge_epoch_alignment(
            "published steering queue slot 2",
            [plant_by_stamp[timestamp]["delayed_steering_2"]
             for timestamp in times], steering_command_by_epoch)
        report_bridge_epoch_alignment(
            "published throttle queue slot 1",
            [plant_by_stamp[timestamp]["delayed_throttle_1"]
             for timestamp in times], throttle_command_by_epoch)
        report_bridge_epoch_alignment(
            "published throttle queue slot 2",
            [plant_by_stamp[timestamp]["delayed_throttle_2"]
             for timestamp in times], throttle_command_by_epoch)
        feedback_valid = [
            plant_by_stamp[timestamp]["actuator_feedback_valid"]
            for timestamp in times]
        report_bridge_epoch_alignment(
            "reported throttle actuator feedback",
            [plant_by_stamp[timestamp]["throttle_feedback"]
             for timestamp in times], throttle_command_by_epoch,
            feedback_valid)

    # Response receive time is not a Unity physics clock. Keep it only as an
    # A/B diagnostic: headers are request-stamped although telemetry is
    # decoded on response. This tests epoch mismatch without changing model
    # parameters or the production 25 ms input contract.
    bridge_by_request_stamp: dict[float, dict[str, Any]] = {}
    for event in bridge_timing_samples or []:
        payload = event.get("json", {})
        request_ns = payload.get("bridge_request_ros_stamp_ns")
        if request_ns is None:
            continue
        try:
            key = round(int(request_ns) * 1.0e-9, 6)
        except (TypeError, ValueError, OverflowError):
            continue
        bridge_by_request_stamp[key] = payload
    bridge_payload_by_epoch = [
        bridge_by_request_stamp.get(timestamp, {}) for timestamp in times
    ]
    bridge_receive_time_by_epoch: list[float | None] = []
    bridge_pipeline_depth_by_epoch: list[int | None] = []
    bridge_requests_monotonic: list[tuple[int, int]] = []
    for payload in bridge_payload_by_epoch:
        request_sequence = payload.get("request_sequence")
        request_ns = payload.get("request_monotonic_ns")
        if request_sequence is None or request_ns is None:
            continue
        try:
            bridge_requests_monotonic.append(
                (int(request_sequence), int(request_ns)))
        except (TypeError, ValueError, OverflowError):
            continue
    for payload in bridge_payload_by_epoch:
        receive_ns = payload.get("bridge_receive_ros_stamp_ns")
        arrival_monotonic_ns = payload.get("bridge_arrival_monotonic_ns")
        request_sequence = payload.get("request_sequence")
        try:
            bridge_receive_time_by_epoch.append(
                None if receive_ns is None else int(receive_ns) * 1.0e-9)
        except (TypeError, ValueError, OverflowError):
            bridge_receive_time_by_epoch.append(None)
        try:
            if arrival_monotonic_ns is None or request_sequence is None:
                bridge_pipeline_depth_by_epoch.append(None)
            else:
                seq = int(request_sequence)
                arrival = int(arrival_monotonic_ns)
                bridge_pipeline_depth_by_epoch.append(sum(
                    later_seq > seq and later_request_ns <= arrival
                    for later_seq, later_request_ns in bridge_requests_monotonic))
        except (TypeError, ValueError, OverflowError):
            bridge_pipeline_depth_by_epoch.append(None)

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
                   command_slot: int = 1,
                   steering_slot: int | None = None,
                   throttle_slot: int | None = None) -> UnitySourcePlantInputC:
        value = UnitySourcePlantInputC()
        value.state[:] = state
        value.wheel_rotation_angle_rad[:] = angles
        steering_slot = command_slot if steering_slot is None else steering_slot
        throttle_slot = command_slot if throttle_slot is None else throttle_slot
        value.steering_target_rad = (
            queue[f"delayed_steering_{steering_slot}"] * MAX_STEERING_RAD)
        value.throttle_normalized = queue[f"delayed_throttle_{throttle_slot}"]
        value.suspension_history_initialized = int(history_initialized)
        return value

    def reconcile_start_state(
        state: list[float], history_initialized: bool,
    ) -> tuple[list[float], bool]:
        if not reconcile_suspension_jounce_at_start:
            return state, history_initialized
        c_state = (ctypes.c_double * UNITY_SOURCE_PLANT_STATE_SIZE)(*state)
        c_history_initialized = ctypes.c_int(int(history_initialized))
        if not reconcile_jounce(c_state, ctypes.byref(c_history_initialized)):
            raise RuntimeError("source plant rejected suspension-jounce reconciliation")
        return list(c_state), bool(c_history_initialized.value)

    def report_encoder_rollout() -> None:
        """Compare real WheelEncoder.FixedUpdate with pre-solve plant RPM.

        The WheelEncoder reads Wheel.rpm before each 1 ms physics update,
        accumulates float revolutions, truncates to 1920 ticks/revolution, and
        publishes the quantized angle. Re-run those exact 25 samples from each
        logged plant state; do not compare against PxVehicle's separate
        post-solve wheel-pose angle.
        """
        encoder_left = {round(row["stamp"], 6): row for row in left_encoder}
        encoder_right = {round(row["stamp"], 6): row for row in right_encoder}
        common = sorted(set(times) & set(encoder_left) & set(encoder_right))
        f32 = lambda value: ctypes.c_float(value).value
        ticks_per_revolution = 16 * 120
        unity_pi = f32(math.pi)
        angle_scale = f32(f32(2.0) * unity_pi)
        tick_scale = f32(ticks_per_revolution)
        errors: dict[str, list[dict[str, float]]] = {
            name: [] for name in ("queue_slot_1", "queue_slot_2",
                                  "throttle_feedback_start",
                                  "throttle_feedback_end",
                                  "queue_slot_1_pre_observer_wheels")}
        # The plant topic is published after the observer corrects wheel
        # omega. Reconstruct the immediately-preceding raw endpoint from the
        # prior logged corrected state and the command applied over that
        # preceding 25 ms interval. This allows a direct causal comparison of
        # the corrected and uncorrected boundary states even when a bag did
        # not record /odom/diagnostics.
        raw_wheels_at: dict[float, tuple[float, float]] = {}
        for previous_time, current_time in zip(common, common[1:]):
            if abs((current_time - previous_time) - 0.025) > 0.004:
                continue
            previous = plant_by_stamp[previous_time]
            value = make_input(
                source_state(previous), list(previous["wheel_angle"]),
                previous, previous["suspension_history_initialized"])
            value.throttle_normalized = previous["delayed_throttle_1"]
            valid = True
            for _ in range(25):
                output = UnitySourcePlantOutputC()
                if not step(ctypes.byref(value), 0.001,
                            ctypes.byref(output)):
                    valid = False
                    break
                value.state[:] = output.state[:]
                value.wheel_rotation_angle_rad[:] = \
                    output.wheel_rotation_angle_rad[:]
                value.suspension_history_initialized = \
                    output.suspension_history_initialized
            if valid:
                raw_wheels_at[current_time] = (value.state[5], value.state[6])
        for start_time, end_time in zip(common, common[1:]):
            if abs((end_time - start_time) - 0.025) > 0.004:
                continue
            start = plant_by_stamp[start_time]
            end = plant_by_stamp[end_time]
            initial_angles = [encoder_left[start_time]["angle"],
                              encoder_right[start_time]["angle"]]
            initial_ticks = [int(round(angle * ticks_per_revolution /
                                       (2.0 * math.pi)))
                             for angle in initial_angles]
            measured_angles = [
                encoder_left[end_time]["angle"] - initial_angles[0],
                encoder_right[end_time]["angle"] - initial_angles[1],
            ]
            throttle_modes = {
                "queue_slot_1": (start["delayed_throttle_1"], False),
                "queue_slot_2": (start["delayed_throttle_2"], False),
                "throttle_feedback_start": (start["throttle_feedback"], False),
                # Offline-only phase diagnostic: this feedback is unavailable
                # at the interval start and must never be used by runtime.
                "throttle_feedback_end": (end["throttle_feedback"], False),
                # The bag's UnityPlantState is published after the observer
                # has adjusted rear wheel omega from this encoder interval.
                # Compare against the plant endpoint before that correction
                # to determine whether the correction itself harms the next
                # interval's initial condition.
                "queue_slot_1_pre_observer_wheels": (
                    start["delayed_throttle_1"], True),
            }
            measured_average_omega = [
                measured_angles[side] / (end_time - start_time)
                for side in range(2)]
            for mode, (throttle, use_pre_observer_wheels) in throttle_modes.items():
                revolutions = [f32(f32(ticks) / tick_scale)
                               for ticks in initial_ticks]
                predicted_angles = [
                    f32(f32(ticks * angle_scale) / tick_scale)
                    for ticks in initial_ticks]
                value = make_input(
                    source_state(start), list(start["wheel_angle"]), start,
                    start["suspension_history_initialized"])
                value.throttle_normalized = throttle
                if use_pre_observer_wheels:
                    raw_wheels = raw_rear_wheel_omega_by_stamp.get(
                        start_time, raw_wheels_at.get(start_time))
                    if raw_wheels is None:
                        continue
                    value.state[5] = raw_wheels[0]
                    value.state[6] = raw_wheels[1]
                start_omega = [value.state[5], value.state[6]]
                valid = True
                for _ in range(25):
                    # Wheel.rpm is read in FixedUpdate before the following
                    # PxVehicle update. Reproduce WheelEncoder.cs float math.
                    for side, wheel in enumerate((5, 6)):
                        omega = float(value.state[wheel])
                        rpm = f32(omega * (60.0 / (2.0 * math.pi)))
                        rps = f32(rpm / f32(60.0))
                        revolutions[side] = f32(
                            revolutions[side] + f32(rps * f32(0.001)))
                        ticks = int(f32(revolutions[side] * tick_scale))
                        predicted_angles[side] = f32(
                            f32(ticks * angle_scale) / tick_scale)
                    output = UnitySourcePlantOutputC()
                    if not step(ctypes.byref(value), 0.001,
                                ctypes.byref(output)):
                        valid = False
                        break
                    value.state[:] = output.state[:]
                    value.wheel_rotation_angle_rad[:] = \
                        output.wheel_rotation_angle_rad[:]
                    value.suspension_history_initialized = \
                        output.suspension_history_initialized
                if not valid:
                    continue
                predicted_delta = [
                    predicted_angles[side] - f32(
                        f32(initial_ticks[side] * angle_scale) / tick_scale)
                    for side in range(2)]
                for side in range(2):
                    predicted_average_omega = predicted_delta[side] / (
                        end_time - start_time)
                    errors[mode].append({
                        "q": start["delayed_throttle_1"],
                        "steer": start["physical_steering"],
                        "error": measured_angles[side] - predicted_delta[side],
                        "start_omega": start_omega[side],
                        "measured_average_omega": measured_average_omega[side],
                        "predicted_average_omega": predicted_average_omega,
                        "predicted_end_omega": value.state[5 + side],
                        "logged_end_omega": end["wheel_omega"][2 + side],
                        "speed": start["u"],
                        "end_q": end["delayed_throttle_1"],
                        "end_steer": end["physical_steering"],
                        "side": float(side),
                        "time": start_time,
                    })

        if not any(errors.values()):
            return

        def show(label: str, selected: list[dict[str, float]]) -> None:
            if not selected:
                return
            residuals = [row["error"] for row in selected]
            rmse = math.sqrt(statistics.mean(value * value
                                             for value in residuals))
            absolute = [abs(value) for value in residuals]
            print(f"    {label}: n={len(selected)} encoder-angle RMSE="
                  f"{rmse:.4f}rad (={rmse / 0.025:.2f}rad/s interval-equivalent), "
                  f"p95/max={percentile(absolute, 0.95):.4f}/"
                  f"{max(absolute):.4f}rad")

        print("  source-plant replay of Unity WheelEncoder pre-solve RPM sampling:")
        print("    command-phase alternatives are diagnostics; end feedback uses future data and is not a runtime input.")
        for mode, selected in errors.items():
            show(mode, selected)
            show(f"  {mode}, positive-throttle intervals", [
                row for row in selected if row["q"] > 0.005])
        for row in sorted(errors["queue_slot_1"],
                          key=lambda item: abs(item["error"]),
                          reverse=True)[:5]:
            print(f"      largest slot-1: t={row['time']:.3f} q1={row['q']:.3f} "
                  f"steer={row['steer']:+.3f} side={'R' if row['side'] else 'L'} "
                  f"u={row['speed']:.2f}m/s q1_end={row['end_q']:.3f} "
                  f"steer_end={row['end_steer']:+.3f} angle_residual="
                  f"{row['error']:+.4f}rad; omega_start="
                  f"{row['start_omega']:.1f}, avg_model/enc="
                  f"{row['predicted_average_omega']:.1f}/"
                  f"{row['measured_average_omega']:.1f}, endpoint_model/odom="
                  f"{row['predicted_end_omega']:.1f}/"
                  f"{row['logged_end_omega']:.1f}rad/s")

    report_encoder_rollout()

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
    command_phase_errors = {
        (steering_slot, throttle_slot): {
            field: [] for field in ("u", "v", "r", "steering")}
        for steering_slot, throttle_slot in ((1, 2), (2, 1))
    }
    observed_throttle_one_step_errors = {
        phase: {field: [] for field in ("u", "v", "r")}
        for phase in ("start_feedback", "end_feedback")
    }
    one_step_motion_details: list[dict[str, float]] = []
    response_dt_one_step_errors = {
        field: [] for field in ("u", "v", "r", "steering")
    }
    fixed_dt_one_step_errors = {
        field: [] for field in ("u", "v", "r", "steering")
    }
    response_dt_jitter_ms: list[float] = []
    reconstructed_one_step_errors = {
        field: [] for field in ("u", "v", "r", "steering")
    }
    command_timing_ab = {
        group: {
            source: {field: [] for field in ("u", "v", "r")}
            for source in ("published_head", "request_timestamp")
        }
        for group in ("queue_aligned", "queue_misaligned")
    }
    plant_state_names = (
        "u", "v_com", "yaw_rate", "wheel_omega_fl", "wheel_omega_fr",
        "wheel_omega_rl", "wheel_omega_rr", "body_height",
        "body_height_rate", "roll", "roll_rate", "pitch", "pitch_rate",
        "physical_steering", "jounce_fl", "jounce_fr", "jounce_rl",
        "jounce_rr", "low_forward_time_fl", "low_forward_time_fr",
        "low_forward_time_rl", "low_forward_time_rr", "low_side_time_fl",
        "low_side_time_fr", "low_side_time_rl", "low_side_time_rr",
    )
    one_step_plant_state_errors = {name: [] for name in plant_state_names}
    one_step_wheel_angle_errors = [[] for _ in range(4)]
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
        if (reconstruct_command_queue and
                index - 1 + bridge_command_epoch_offset < 0):
            # The request immediately before the capture is absent, so its
            # queued command cannot be reconstructed causally.
            continue
        start_plant = plant_by_stamp[start_time]
        start_truth = truth_by_stamp[start_time]
        next_truth = truth_by_stamp[next_time]
        next_plant = plant_by_stamp[next_time]
        if (reconstruct_command_queue_from_bridge and
                start_plant["actuator_feedback_valid"] and
                next_plant["actuator_feedback_valid"]):
            start_angle = start_plant["steering_feedback"]
            measured_next_angle = next_plant["steering_feedback"]
            max_steering_move = math.radians(183.345993) * 0.025
            phase_residuals: list[tuple[float, int]] = []
            for offset in steering_feedback_phase_errors:
                epoch = index + offset
                if not 0 <= epoch < len(steering_command_by_epoch):
                    continue
                target_angle = (
                    steering_command_by_epoch[epoch] * MAX_STEERING_RAD)
                delta = target_angle - start_angle
                predicted_angle = start_angle + max(
                    -max_steering_move, min(max_steering_move, delta))
                residual = predicted_angle - measured_next_angle
                steering_feedback_phase_errors[offset].append(residual)
                phase_residuals.append((abs(residual), offset))
                payload = bridge_payload_by_epoch[epoch]
                try:
                    steering_feedback_phase_latency_ms[offset].append(
                        (int(payload["bridge_receive_ros_stamp_ns"]) -
                         int(payload["bridge_request_ros_stamp_ns"])) * 1.0e-6)
                except (KeyError, TypeError, ValueError, OverflowError):
                    pass
            if phase_residuals:
                best_offset = min(phase_residuals)[1]
                steering_feedback_best_phase[best_offset] += 1
                try:
                    payload = bridge_payload_by_epoch[index + best_offset]
                    latency_ms = (
                        int(payload["bridge_receive_ros_stamp_ns"]) -
                        int(payload["bridge_request_ros_stamp_ns"])) * 1.0e-6
                    steering_feedback_best_phase_latency_ms[
                        best_offset].append(latency_ms)
                except (IndexError, KeyError, TypeError, ValueError,
                        OverflowError):
                    pass
        if reconstruct_command_queue:
            command_epoch = index - 1 + bridge_command_epoch_offset
            reconstructed_steering = steering_command_by_epoch[command_epoch]
            reconstructed_throttle = throttle_command_by_epoch[command_epoch]
            published_head_matches = (
                abs(start_plant["delayed_steering_1"] -
                    reconstructed_steering) <= 1.0e-3 and
                abs(start_plant["delayed_throttle_1"] -
                    reconstructed_throttle) <= 1.0e-3)
            group = ("queue_aligned" if published_head_matches else
                     "queue_misaligned")
            for source in ("published_head", "request_timestamp"):
                timing_input = make_input(
                    source_state(start_plant), list(start_plant["wheel_angle"]),
                    start_plant,
                    start_plant["suspension_history_initialized"])
                if source == "request_timestamp":
                    timing_input.steering_target_rad = (
                        reconstructed_steering * MAX_STEERING_RAD)
                    timing_input.throttle_normalized = reconstructed_throttle
                timing_output = UnitySourcePlantOutputC()
                if step(ctypes.byref(timing_input), 0.025, ctypes.byref(timing_output)):
                    command_timing_ab[group][source]["u"].append(
                        timing_output.state[0] - next_truth["u"])
                    command_timing_ab[group][source]["v"].append(
                        timing_output.state[1] - next_truth["v"])
                    command_timing_ab[group][source]["r"].append(
                        timing_output.state[2] - next_truth["r"])
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
        initial_state, initial_history_initialized = reconcile_start_state(
            initial_state, start_plant["suspension_history_initialized"])
        for slot in (1, 2):
            candidate_input = make_input(
                initial_state, list(start_plant["wheel_angle"]), start_plant,
                initial_history_initialized, slot)
            if reconstruct_command_queue and slot == 1:
                command_epoch = index - 1 + bridge_command_epoch_offset
                steering_value = steering_command_by_epoch[command_epoch]
                throttle_value = throttle_command_by_epoch[command_epoch]
                candidate_input.steering_target_rad = (
                    steering_value * MAX_STEERING_RAD)
                candidate_input.throttle_normalized = throttle_value
            candidate_output = UnitySourcePlantOutputC()
            if not step(ctypes.byref(candidate_input), 0.025,
                        ctypes.byref(candidate_output)):
                continue
            if reconstruct_command_queue and slot == 1:
                reconstructed_one_step_errors["u"].append(
                    candidate_output.state[0] - next_truth["u"])
                reconstructed_one_step_errors["v"].append(
                    candidate_output.state[1] - next_truth["v"])
                reconstructed_one_step_errors["r"].append(
                    candidate_output.state[2] - next_truth["r"])
                reconstructed_one_step_errors["steering"].append(
                    candidate_output.state[13] - next_steering)
                if (index + 1 < len(bridge_receive_time_by_epoch) and
                        bridge_receive_time_by_epoch[index] is not None and
                        bridge_receive_time_by_epoch[index + 1] is not None):
                    response_dt = (
                        bridge_receive_time_by_epoch[index + 1] -
                        bridge_receive_time_by_epoch[index])
                    fixed_steps = int(round(response_dt / 0.001))
                    if 10 <= fixed_steps <= 40:
                        response_dt_s = fixed_steps * 0.001
                        response_output = UnitySourcePlantOutputC()
                        if step(ctypes.byref(candidate_input), response_dt_s,
                                ctypes.byref(response_output)):
                            fixed_dt_one_step_errors["u"].append(
                                candidate_output.state[0] - next_truth["u"])
                            fixed_dt_one_step_errors["v"].append(
                                candidate_output.state[1] - next_truth["v"])
                            fixed_dt_one_step_errors["r"].append(
                                candidate_output.state[2] - next_truth["r"])
                            fixed_dt_one_step_errors["steering"].append(
                                candidate_output.state[13] - next_steering)
                            response_dt_one_step_errors["u"].append(
                                response_output.state[0] - next_truth["u"])
                            response_dt_one_step_errors["v"].append(
                                response_output.state[1] - next_truth["v"])
                            response_dt_one_step_errors["r"].append(
                                response_output.state[2] - next_truth["r"])
                            response_dt_one_step_errors["steering"].append(
                                response_output.state[13] - next_steering)
                            response_dt_jitter_ms.append(
                                response_dt * 1000.0 - 25.0)
            command_slot_errors[slot]["u"].append(
                candidate_output.state[0] - next_truth["u"])
            command_slot_errors[slot]["v"].append(
                candidate_output.state[1] - next_truth["v"])
            command_slot_errors[slot]["r"].append(
                candidate_output.state[2] - next_truth["r"])
            command_slot_errors[slot]["steering"].append(
                candidate_output.state[13] - next_steering)
            if slot == 1:
                next_state = source_state(next_plant)
                for state_index, name in enumerate(plant_state_names):
                    one_step_plant_state_errors[name].append(
                        candidate_output.state[state_index] -
                        next_state[state_index])
                for wheel in range(4):
                    one_step_wheel_angle_errors[wheel].append(
                        candidate_output.wheel_rotation_angle_rad[wheel] -
                        next_plant["wheel_angle"][wheel])
                observed_dt = next_time - start_time
                observed_mid_u = 0.5 * (start_truth["u"] + next_truth["u"])
                observed_mid_v = 0.5 * (start_truth["v"] + next_truth["v"])
                observed_mid_r = 0.5 * (start_truth["r"] + next_truth["r"])
                observed_ax = (
                    (next_truth["u"] - start_truth["u"]) / observed_dt -
                    observed_mid_r * observed_mid_v)
                observed_ay = (
                    (next_truth["v"] - start_truth["v"]) / observed_dt +
                    observed_mid_r * observed_mid_u)
                observed_yaw_accel = (
                    (next_truth["r"] - start_truth["r"]) / observed_dt)
                one_step_motion_details.append({
                    "stamp": start_time,
                    "speed": math.hypot(start_truth["u"], start_truth["v"]),
                    "steering": current_steering,
                    "steering_delta": next_steering - current_steering,
                    "commanded_steering": candidate_input.steering_target_rad,
                    "throttle": candidate_input.throttle_normalized,
                    "r_start": start_truth["r"],
                    "r_predicted": candidate_output.state[2],
                    "r_actual": next_truth["r"],
                    "predicted_ax": candidate_output.longitudinal_acceleration_mps2,
                    "observed_ax": observed_ax,
                    "predicted_ay": candidate_output.lateral_acceleration_mps2,
                    "observed_ay": observed_ay,
                    "predicted_yaw_accel": candidate_output.yaw_acceleration_radps2,
                    "observed_yaw_accel": observed_yaw_accel,
                    "physical_steering_error":
                        candidate_output.state[13] - next_steering,
                    "next_steering_feedback":
                        next_plant["steering_feedback"],
                    "next_throttle_feedback":
                        next_plant["throttle_feedback"],
                    "bridge_command_epoch": float(
                        command_epoch if reconstruct_command_queue else -1),
                    "bridge_steering_previous": (
                        steering_command_by_epoch[command_epoch - 1]
                        if reconstruct_command_queue and command_epoch > 0
                        else math.nan),
                    "bridge_steering_current": (
                        steering_command_by_epoch[command_epoch]
                        if reconstruct_command_queue else math.nan),
                    "bridge_steering_next": (
                        steering_command_by_epoch[command_epoch + 1]
                        if reconstruct_command_queue and
                        command_epoch + 1 < len(times) else math.nan),
                    "bridge_throttle_previous": (
                        throttle_command_by_epoch[command_epoch - 1]
                        if reconstruct_command_queue and command_epoch > 0
                        else math.nan),
                    "bridge_throttle_current": (
                        throttle_command_by_epoch[command_epoch]
                        if reconstruct_command_queue else math.nan),
                    "bridge_throttle_next": (
                        throttle_command_by_epoch[command_epoch + 1]
                        if reconstruct_command_queue and
                        command_epoch + 1 < len(times) else math.nan),
                    "gt_x": start_truth["x"],
                    "gt_y": start_truth["y"],
                    "gt_z": start_truth["z"],
                    "gt_z_delta": next_truth["z"] - start_truth["z"],
                    "gt_roll": start_truth["roll"],
                    "gt_pitch": start_truth["pitch"],
                    "response_dt_ms": (
                        response_dt * 1000.0
                        if reconstruct_command_queue and
                        index + 1 < len(bridge_receive_time_by_epoch) and
                        bridge_receive_time_by_epoch[index] is not None and
                        bridge_receive_time_by_epoch[index + 1] is not None
                        else math.nan),
                    "request_pipeline_depth": (
                        float(max(
                            bridge_pipeline_depth_by_epoch[index],
                            bridge_pipeline_depth_by_epoch[index + 1]))
                        if reconstruct_command_queue and
                        index + 1 < len(bridge_pipeline_depth_by_epoch) and
                        bridge_pipeline_depth_by_epoch[index] is not None and
                        bridge_pipeline_depth_by_epoch[index + 1] is not None
                        else math.nan),
                    "u_error": candidate_output.state[0] - next_truth["u"],
                    "v_error": candidate_output.state[1] - next_truth["v"],
                    "r_error": candidate_output.state[2] - next_truth["r"],
                    **{
                        f"fx_{wheel}":
                            candidate_output.tire_longitudinal_force_n[wheel]
                        for wheel in range(4)
                    },
                    **{
                        f"fy_{wheel}":
                            candidate_output.tire_lateral_force_n[wheel]
                        for wheel in range(4)
                    },
                    **{
                        f"slip_x_{wheel}":
                            candidate_output.longitudinal_slip[wheel]
                        for wheel in range(4)
                    },
                    **{
                        f"slip_y_{wheel}":
                            candidate_output.lateral_slip[wheel]
                        for wheel in range(4)
                    },
                    **{
                        f"load_{wheel}":
                            candidate_output.normal_load_n[wheel]
                        for wheel in range(4)
                    },
                })
            for regime in regimes:
                command_slot_regimes[regime][slot]["r"].append(
                    candidate_output.state[2] - next_truth["r"])
                command_slot_regimes[regime][slot]["steering"].append(
                    candidate_output.state[13] - next_steering)
        # Unity's CurrentThrottle is normalized DriveTorque already assigned
        # by FixedUpdate. Compare the command queue against both boundary
        # samples. End feedback is available to odometry when it processes the
        # completed interval, but is future leakage for a controller horizon;
        # therefore this is a one-step diagnostic only.
        for phase, throttle in (
            ("start_feedback", start_plant["throttle_feedback"]),
            ("end_feedback", next_plant["throttle_feedback"]),
        ):
            feedback_input = make_input(
                initial_state, list(start_plant["wheel_angle"]), start_plant,
                initial_history_initialized)
            feedback_input.throttle_normalized = throttle
            feedback_output = UnitySourcePlantOutputC()
            if step(ctypes.byref(feedback_input), 0.025,
                    ctypes.byref(feedback_output)):
                for field, state_index, actual in (
                    ("u", 0, next_truth["u"]),
                    ("v", 1, next_truth["v"]),
                    ("r", 2, next_truth["r"]),
                ):
                    observed_throttle_one_step_errors[phase][field].append(
                        feedback_output.state[state_index] - actual)
        for steering_slot, throttle_slot in ((1, 2), (2, 1)):
            phase_input = make_input(
                initial_state, list(start_plant["wheel_angle"]), start_plant,
                start_plant["suspension_history_initialized"],
                steering_slot=steering_slot, throttle_slot=throttle_slot)
            phase_output = UnitySourcePlantOutputC()
            if step(ctypes.byref(phase_input), 0.025, ctypes.byref(phase_output)):
                phase = command_phase_errors[(steering_slot, throttle_slot)]
                phase["u"].append(phase_output.state[0] - next_truth["u"])
                phase["v"].append(phase_output.state[1] - next_truth["v"])
                phase["r"].append(phase_output.state[2] - next_truth["r"])
                phase["steering"].append(
                    phase_output.state[13] - next_steering)
    maximum_extra_delay = max(
        (max(1 - bridge_command_epoch_offset, 0)
         if reconstruct_command_queue else 0),
        extra_steering_delay_packets, extra_throttle_delay_packets)
    for start in range(maximum_extra_delay,
                       len(times) - max(horizons)):
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
        state, history_initialized = reconcile_start_state(
            state, start_plant["suspension_history_initialized"])
        angles = list(start_plant["wheel_angle"])
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
            command_step = start + horizon - 1
            if reconstruct_command_queue:
                # Packet headers are stamped with the Bridge request epoch.
                # The command callbacks are asynchronous and can deliver the
                # next controller output before this response is processed.
                # Reconstruct the two-entry queue using only command samples
                # received by the request epoch; its head is the previous
                # request's command, which is what advance() applies next.
                command_epoch = command_step - 1 + bridge_command_epoch_offset
                steering_value = steering_command_by_epoch[command_epoch]
                throttle_value = throttle_command_by_epoch[command_epoch]
                steering_source = current
                throttle_source = current
            else:
                steering_source = plant_by_stamp[times[
                    command_step - extra_steering_delay_packets]]
                throttle_source = plant_by_stamp[times[
                    command_step - extra_throttle_delay_packets]]
            target_time = times[start + horizon]
            measured_dt = target_time - times[start + horizon - 1]
            if abs(measured_dt - 0.025) > 0.004:
                valid = False
                break
            dt = 0.025
            input_value = make_input(
                state, angles, steering_source, history_initialized)
            if reconstruct_command_queue:
                input_value.steering_target_rad = \
                    steering_value * MAX_STEERING_RAD
                input_value.throttle_normalized = throttle_value
            else:
                input_value.throttle_normalized = \
                    throttle_source["delayed_throttle_1"]
            if oracle_steering_feedback:
                # Diagnostic only: use the next recorded actuator feedback as
                # the steering target for this interval. This isolates the
                # yaw/plant error from command-to-feedback phase ambiguity; it
                # is future information and is never a runtime input.
                input_value.steering_target_rad = \
                    plant_by_stamp[target_time]["physical_steering"]
            if oracle_throttle_feedback:
                # Diagnostic only: CurrentThrottle is populated from the
                # torque already assigned by VehicleController.FixedUpdate.
                # Using the next feedback sample tests command/application
                # epoch ambiguity separately from the wheel/tire equations;
                # it is future information and must never enter production.
                input_value.throttle_normalized = \
                    plant_by_stamp[target_time]["throttle_feedback"]
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
                (steering_value if reconstruct_command_queue else
                     steering_source["delayed_steering_1"]) * MAX_STEERING_RAD,
                "queued_throttle": (throttle_value if reconstruct_command_queue else
                                    throttle_source["delayed_throttle_1"]),
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
                    "start_epoch_s": start_time,
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
    if reconcile_suspension_jounce_at_start:
        init_label += "; geometric jounce reconciled at each independent start"
    print(f"\nSource-plant open-loop rollout ({init_label}; no future state correction):")
    print(f"  causal yaw-residual correction gain={yaw_residual_gain:.3f}; "
          "residual is same-epoch raw forecast minus current gyro; "
          f"aligned={residual_valid_windows}, enabled={residual_enabled_windows}")
    print(f"  aligned_epochs={len(times)} start_windows={len(times)-max(horizons)} "
          f"skipped_windows={skipped} model_dt=25ms")
    print("  one-step command-queue phase A/B (same plant state; next Unity "
          "epoch used only for scoring; recorded queue-slot diagnostics):")
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
            print(f"      {field}: bias={statistics.mean(residuals):+.5f} "
                  f"RMSE={rmse:.5f} {unit} "
                  f"p95_abs={percentile(absolute, 0.95):.5f} "
                  f"max_abs={max(absolute):.5f}")
    if reconstruct_command_queue:
        print("  one-step actuator timing A/B from identical recorded initial states:")
        for group, sources in command_timing_ab.items():
            count = len(sources["request_timestamp"]["u"])
            if not count:
                continue
            print(f"    {group}: n={count}")
            for source in ("published_head", "request_timestamp"):
                print(f"      {source}:")
                for field, unit in (("u", "m/s"), ("v", "m/s"),
                                    ("r", "rad/s")):
                    values = sources[source][field]
                    rmse = math.sqrt(statistics.mean(value * value for value in values))
                    print(f"        {field}: RMSE={rmse:.5f}{unit} "
                          f"p95_abs={percentile([abs(v) for v in values], 0.95):.5f}")
        print("  timestamp-reconstructed command one-step errors:")
        for field, unit in (("u", "m/s"), ("v", "m/s"),
                            ("r", "rad/s"), ("steering", "rad")):
            values = reconstructed_one_step_errors[field]
            if values:
                absolute = [abs(value) for value in values]
                print(f"    {field}: bias={statistics.mean(values):+.5f} "
                      f"RMSE={math.sqrt(statistics.mean(v*v for v in values)):.5f}"
                      f" {unit} p95_abs={percentile(absolute, 0.95):.5f} "
                      f"max_abs={max(absolute):.5f}")
    print("  independent actuator queue-phase A/B (one-step only; "
          "published queue slots):")
    for (steering_slot, throttle_slot), fields in command_phase_errors.items():
        print(f"    steering slot {steering_slot}, throttle slot {throttle_slot}:")
        for field, unit in (("u", "m/s"), ("v", "m/s"),
                            ("r", "rad/s"), ("steering", "rad")):
            residuals = fields[field]
            if residuals:
                rmse = math.sqrt(statistics.mean(x*x for x in residuals))
                print(f"      {field}: RMSE={rmse:.5f} {unit}")
    for phase, label in (
        ("start_feedback", "starting-epoch feedback (causal for prediction)"),
        ("end_feedback", "completed-interval feedback (available to odom, "
         "not to a forward MPC horizon)"),
    ):
        fields = observed_throttle_one_step_errors[phase]
        if not fields["u"]:
            continue
        print(f"  one-step using {label}:")
        for field, unit in (("u", "m/s"), ("v", "m/s"),
                            ("r", "rad/s")):
            residuals = fields[field]
            absolute = [abs(value) for value in residuals]
            print(f"    {field}: bias={statistics.mean(residuals):+.5f} "
                  f"RMSE={math.sqrt(statistics.mean(value*value for value in residuals)):.5f} "
                  f"{unit} p95_abs={percentile(absolute, 0.95):.5f} "
                  f"max_abs={max(absolute):.5f}")
    closure_input_label = ("request-stamped reconstructed command" if
                           reconstruct_command_queue else
                           "published queue slot 1")
    print(f"  one-step carried-state closure using {closure_input_label}:")
    for name, residuals in one_step_plant_state_errors.items():
        if not residuals:
            continue
        absolute = [abs(value) for value in residuals]
        rmse = math.sqrt(statistics.mean(value * value for value in residuals))
        print(f"    {name}: RMSE={rmse:.6g} p95_abs="
              f"{percentile(absolute, 0.95):.6g} max_abs={max(absolute):.6g}")
    for wheel, residuals in enumerate(one_step_wheel_angle_errors):
        if residuals:
            absolute = [abs(value) for value in residuals]
            rmse = math.sqrt(statistics.mean(value * value for value in residuals))
            print(f"    wheel_angle_{wheel}: RMSE={rmse:.6g}rad "
                  f"p95_abs={percentile(absolute, 0.95):.6g}rad "
                  f"max_abs={max(absolute):.6g}rad")
    if one_step_motion_details:
        if reconstruct_command_queue_from_bridge:
            print("  steering-actuator command-epoch diagnostic (rate-limited "
                  "Unity steering equation; feedback-vs-feedback, not a "
                  "vehicle rollout):")
            for offset in sorted(steering_feedback_phase_errors):
                residuals = steering_feedback_phase_errors[offset]
                if not residuals:
                    continue
                absolute = [abs(value) for value in residuals]
                latencies = steering_feedback_phase_latency_ms[offset]
                latency_text = (
                    f" response-latency-median="
                    f"{statistics.median(latencies):.2f}ms"
                    if latencies else "")
                print(
                    f"    C[i{offset:+d}]: n={len(residuals)} "
                    f"steer-RMSE={math.sqrt(statistics.mean(x*x for x in residuals)):.5f}rad "
                    f"p95={percentile(absolute, 0.95):.5f}rad"
                    f"{latency_text}")
            best_total = sum(steering_feedback_best_phase.values())
            print(f"    best candidate epoch counts ({best_total} transitions):")
            for offset in sorted(steering_feedback_best_phase):
                count = steering_feedback_best_phase[offset]
                latencies = steering_feedback_best_phase_latency_ms[offset]
                latency_text = (
                    f" median-response-latency="
                    f"{statistics.median(latencies):.2f}ms"
                    if latencies else "")
                print(f"      C[i{offset:+d}]: {count}"
                      f"{latency_text}")
        motion_partitions = (
            ("steer_negative", lambda row: row["steering"] < -1.0e-4),
            ("steer_positive", lambda row: row["steering"] > 1.0e-4),
            ("steer_near_zero", lambda row: abs(row["steering"]) <= 1.0e-4),
            ("abs_steer_below_0.10", lambda row: abs(row["steering"]) < 0.10),
            ("abs_steer_at_least_0.10", lambda row: abs(row["steering"]) >= 0.10),
            ("turn_in", lambda row: row["steering"] * row["steering_delta"] > 1.0e-5),
            ("unwind", lambda row: row["steering"] * row["steering_delta"] < -1.0e-5),
            ("zero_throttle_brake", lambda row: row["throttle"] == 0.0),
            ("positive_throttle", lambda row: row["throttle"] > 0.0),
        )
        label = ("request-stamped reconstructed commands" if
                 reconstruct_command_queue else "queue slot 1")
        print(f"  one-step body-state residual partitions ({label}):")
        for label, predicate in motion_partitions:
            selected = [row for row in one_step_motion_details if predicate(row)]
            if not selected:
                continue
            print(f"    {label}: n={len(selected)}")
            for field, unit in (("u_error", "m/s"), ("v_error", "m/s"),
                                ("r_error", "rad/s")):
                values = [row[field] for row in selected]
                rmse = math.sqrt(statistics.mean(value * value for value in values))
                print(f"      {field}: bias={statistics.mean(values):+.5f} "
                      f"RMSE={rmse:.5f}{unit}")
        if reconstruct_command_queue:
            if response_dt_one_step_errors["u"]:
                print("  paired source-plant one-step A/B: fixed 25 ms vs "
                      "Bridge receive-gap rounded to 1 ms (diagnostic only):")
                print(
                    f"    usable intervals={len(response_dt_one_step_errors['u'])}; "
                    f"receive-minus-request dt median/p95="
                    f"{statistics.median(response_dt_jitter_ms):+.2f}/"
                    f"{percentile([abs(x) for x in response_dt_jitter_ms], 0.95):.2f}ms abs")
                for field, unit in (("u", "m/s"), ("v", "m/s"),
                                    ("r", "rad/s"), ("steering", "rad")):
                    fixed = fixed_dt_one_step_errors[field]
                    timed = response_dt_one_step_errors[field]
                    fixed_rmse = math.sqrt(statistics.mean(x*x for x in fixed))
                    timed_rmse = math.sqrt(statistics.mean(x*x for x in timed))
                    fixed_p95 = percentile([abs(x) for x in fixed], 0.95)
                    timed_p95 = percentile([abs(x) for x in timed], 0.95)
                    print(
                        f"    {field}: fixed RMSE/p95={fixed_rmse:.5f}/"
                        f"{fixed_p95:.5f}{unit}; response-dt="
                        f"{timed_rmse:.5f}/{timed_p95:.5f}{unit}")
            stable_commands = [
                row for row in one_step_motion_details
                if max(row["bridge_steering_previous"],
                       row["bridge_steering_current"],
                       row["bridge_steering_next"]) -
                min(row["bridge_steering_previous"],
                    row["bridge_steering_current"],
                    row["bridge_steering_next"]) <= 1.0e-3]
            changing_commands = [
                row for row in one_step_motion_details
                if max(row["bridge_steering_previous"],
                       row["bridge_steering_current"],
                       row["bridge_steering_next"]) -
                min(row["bridge_steering_previous"],
                    row["bridge_steering_current"],
                    row["bridge_steering_next"]) > 1.0e-3]
            vertical_motion = [
                row for row in one_step_motion_details
                if abs(row["gt_z_delta"]) > 0.001]
            elevated_attitude = [
                row for row in one_step_motion_details
                if max(abs(row["gt_roll"]), abs(row["gt_pitch"])) >
                math.radians(3.0)]
            pipeline_groups = (
                ("no newer request in flight", lambda row:
                    row["request_pipeline_depth"] == 0.0),
                ("one newer request in flight", lambda row:
                    row["request_pipeline_depth"] == 1.0),
                ("2+ newer requests in flight", lambda row:
                    row["request_pipeline_depth"] >= 2.0),
            )
            print("  yaw residual by recorded command/3-D motion regime:")
            for label, selected in (
                ("three request commands stable", stable_commands),
                ("steering command changes nearby", changing_commands),
                ("|GT Δz| > 1mm", vertical_motion),
                ("|GT roll| or |pitch| > 3deg", elevated_attitude),
            ):
                if not selected:
                    continue
                residuals = [row["r_error"] for row in selected]
                print(
                    f"    {label}: n={len(selected)} "
                    f"r_RMSE={math.sqrt(statistics.mean(x*x for x in residuals)):.5f} "
                    f"r_p95={percentile([abs(x) for x in residuals], 0.95):.5f}rad/s")
            print("  yaw residual by Bridge request/response overlap:")
            for label, predicate in pipeline_groups:
                selected = [row for row in one_step_motion_details
                            if math.isfinite(row["request_pipeline_depth"]) and
                            predicate(row)]
                if not selected:
                    continue
                residuals = [row["r_error"] for row in selected]
                print(
                    f"    {label}: n={len(selected)} "
                    f"r_RMSE={math.sqrt(statistics.mean(x*x for x in residuals)):.5f} "
                    f"r_p95={percentile([abs(x) for x in residuals], 0.95):.5f}rad/s")
        yaw_outliers = sorted(
            one_step_motion_details,
            key=lambda row: abs(row["r_error"]), reverse=True)[:12]
        if yaw_outliers:
            print("  largest one-step yaw-rate residuals (source plant vs next "
                  "Unity sample; per-wheel source force diagnostics):")
            for row in yaw_outliers:
                command_epoch = int(row["bridge_command_epoch"])
                command_window = "unavailable"
                command_age_ms = response_ms = math.nan
                if reconstruct_command_queue and 0 <= command_epoch < len(times):
                    first_epoch = max(0, command_epoch - 3)
                    last_epoch = min(len(times), command_epoch + 4)
                    command_window = ",".join(
                        f"{steering_command_by_epoch[k]:+.3f}"
                        for k in range(first_epoch, last_epoch))
                    timing_payload = bridge_payload_by_epoch[command_epoch]
                    try:
                        command_age_ms = float(
                            timing_payload["steering_command_update_age_ms"])
                        response_ms = (
                            int(timing_payload["bridge_receive_ros_stamp_ns"]) -
                            int(timing_payload["bridge_request_ros_stamp_ns"])) * 1.0e-6
                    except (KeyError, TypeError, ValueError, OverflowError):
                        pass
                print(
                    f"    t={row['stamp']:.6f}s speed={row['speed']:.3f}m/s "
                    f"steer={row['steering']:+.4f}->{row['steering'] + row['steering_delta']:+.4f}rad "
                    f"target={row['commanded_steering']:+.4f}rad "
                    f"throttle={row['throttle']:.3f} "
                    f"r={row['r_start']:+.4f}->{row['r_predicted']:+.4f}/"
                    f"{row['r_actual']:+.4f}rad/s "
                    f"err={row['r_error']:+.4f}rad/s "
                    f"next_feedback=(steer:{row['next_steering_feedback']:+.4f}rad,"
                    f"q:{row['next_throttle_feedback']:.3f}) "
                    f"Bridge_C[i-1,i,i+1]=steer:"
                    f"[{row['bridge_steering_previous']:+.3f},"
                    f"{row['bridge_steering_current']:+.3f},"
                    f"{row['bridge_steering_next']:+.3f}] throttle:"
                    f"[{row['bridge_throttle_previous']:.3f},"
                    f"{row['bridge_throttle_current']:.3f},"
                    f"{row['bridge_throttle_next']:.3f}] "
                    f"steer_C[i-3..i+3]=[{command_window}] "
                    f"cmd_age/response={command_age_ms:.1f}/{response_ms:.1f}ms "
                    f"GT=(x:{row['gt_x']:+.2f},y:{row['gt_y']:+.2f},"
                    f"z:{row['gt_z']:+.3f},roll:{math.degrees(row['gt_roll']):+.1f}deg,"
                    f"pitch:{math.degrees(row['gt_pitch']):+.1f}deg,"
                    f"dz:{row['gt_z_delta']*1000:+.1f}mm) "
                    f"yaw_accel={row['predicted_yaw_accel']:+.1f}/"
                    f"{row['observed_yaw_accel']:+.1f}rad/s2 "
                    f"ax={row['predicted_ax']:+.2f}/{row['observed_ax']:+.2f} "
                    f"ay={row['predicted_ay']:+.2f}/{row['observed_ay']:+.2f}m/s2 "
                    f"steer_err={row['physical_steering_error']:+.4f}rad "
                    f"Fx=[{row['fx_0']:+.2f},{row['fx_1']:+.2f},"
                    f"{row['fx_2']:+.2f},{row['fx_3']:+.2f}]N "
                    f"Fy=[{row['fy_0']:+.2f},{row['fy_1']:+.2f},"
                    f"{row['fy_2']:+.2f},{row['fy_3']:+.2f}]N "
                    f"slipX=[{row['slip_x_0']:+.3f},{row['slip_x_1']:+.3f},"
                    f"{row['slip_x_2']:+.3f},{row['slip_x_3']:+.3f}] "
                    f"slipY=[{row['slip_y_0']:+.3f},{row['slip_y_1']:+.3f},"
                    f"{row['slip_y_2']:+.3f},{row['slip_y_3']:+.3f}] "
                    f"load=[{row['load_0']:.1f},{row['load_1']:.1f},"
                    f"{row['load_2']:.1f},{row['load_3']:.1f}]N")
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
            print(f"    {field}: bias={statistics.mean(values[field]):+.5f} "
                  f"RMSE={rmse:.5f} {unit} "
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
            print(f"    start={row['start_epoch_s']:.6f}s "
                  f"xy={row['xy_m']:.4f}m start_speed="
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
            for failure_index, row in enumerate(sorted(
                    failing_path_rows,
                    key=lambda item: item["path_max_xy_m"],
                    reverse=True)[:20]):
                print(f"    start={row['start_epoch_s']:.6f}s "
                      f"peak={row['path_max_xy_m']:.4f}m at="
                      f"{int(row['path_max_step'])*25}ms start_speed="
                      f"{row['start_speed_mps']:.2f}m/s start_steer="
                      f"{row['start_steering_rad']:.3f}rad "
                      f"yaw_innovation={row['start_yaw_residual_radps']:+.4f}rad/s")
                for trace_step in row.get("path_trace", []):
                    if failure_index == 0 or \
                            int(trace_step["step"]) in (1, 4, 10, 20, 30) or \
                            abs(trace_step["xy_error_m"] - row["path_max_xy_m"]) < 1.0e-9:
                        print(f"      {int(trace_step['step'])*25:3d}ms xy="
                              f"{trace_step['xy_error_m']:.4f}m yaw="
                              f"{trace_step['yaw_error_rad']:+.4f}rad "
                              f"dXY=({trace_step['step_error_forward_m']:+.4f},"
                              f"{trace_step['step_error_lateral_m']:+.4f})m "
                              f"plant-truthTwist=({trace_step['plant_vs_truth_twist_forward_m']:+.4f},"
                              f"{trace_step['plant_vs_truth_twist_lateral_m']:+.4f})m "
                              f"truthTwist-pose=({trace_step['truth_twist_vs_pose_forward_m']:+.4f},"
                              f"{trace_step['truth_twist_vs_pose_lateral_m']:+.4f})m "
                              f"dr={trace_step['yaw_rate_error_radps']:+.4f}rad/s "
                              f"du={trace_step['speed_error_mps']:+.4f}/"
                              f"{trace_step['lateral_speed_error_mps']:+.4f}m/s "
                              f"steer={trace_step['physical_steering_rad']:+.3f}/"
                              f"{trace_step['queued_steering_rad']:+.3f}rad "
                              f"q={trace_step['queued_throttle']:.3f}")
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

    if maximum_extra_delay and not reconstruct_command_queue:
        print(
            "  fixed-input horizon diagnostic extra packet delay: "
            f"steering={extra_steering_delay_packets}, "
            f"throttle={extra_throttle_delay_packets}; this does not change "
            "the production delay model."
        )
    if oracle_steering_feedback:
        print(
            "  steering target used next recorded actuator feedback; this is "
            "an offline oracle diagnostic, not a deployable rollout input."
        )
    if reconstruct_command_queue:
        source_label = ("Bridge's recorded sent command per request" if
                        reconstruct_command_queue_from_bridge else
                        "command-topic receive times <= each Bridge request stamp")
        print(f"  fixed-input horizon reconstructed the command delay queue from "
              f"{source_label}; applied command epoch offset="
              f"{bridge_command_epoch_offset} relative to the preceding request.")

    if diagnose_observed_state_reanchoring:
        # Oracle values below are applied only at each 25 ms boundary to
        # identify which carried-state prediction drives recursive drift.
        # They are deliberately not valid controller inputs or model scores.
        modes = {
            "u_only": ("u",),
            "v_only": ("v",),
            "yaw_rate_only": ("yaw_rate",),
            "u_v_only": ("u", "v"),
            "u_yaw_rate": ("u", "yaw_rate"),
            "v_yaw_rate": ("v", "yaw_rate"),
            "u_v_r": ("motion",),
            "heading_only": ("heading",),
            "attitude_and_rates": ("attitude",),
            "encoder_wheel_speeds": ("wheels",),
            "u_v_r+attitude": ("motion", "attitude"),
            "u_v_r+wheels": ("motion", "wheels"),
            "u_v_r+heading": ("motion", "heading"),
            "all_observed_motion_attitude_wheels": (
                "motion", "attitude", "wheels"),
            "motion_attitude_heading_wheels": (
                "motion", "attitude", "heading", "wheels"),
        }
        oracle_xy: dict[str, list[float]] = {name: [] for name in modes}
        for start in range(0, len(times) - max(horizons)):
            if any(abs((times[index + 1] - times[index]) - 0.025) > 0.004
                   for index in range(start, start + max(horizons))):
                continue
            start_time = times[start]
            initial = source_state(plant_by_stamp[start_time])
            for name, corrected_groups in modes.items():
                state = initial.copy()
                angles = list(plant_by_stamp[start_time]["wheel_angle"])
                history_initialized = plant_by_stamp[start_time][
                    "suspension_history_initialized"]
                yaw = truth_by_stamp[start_time]["yaw"]
                x = truth_by_stamp[start_time]["x"]
                y = truth_by_stamp[start_time]["y"]
                valid = True
                for horizon in range(1, max(horizons) + 1):
                    current = plant_by_stamp[times[start + horizon - 1]]
                    target_time = times[start + horizon]
                    actual = truth_by_stamp[target_time]
                    output = UnitySourcePlantOutputC()
                    input_value = make_input(
                        state, angles, current, history_initialized)
                    if not step(ctypes.byref(input_value), 0.025,
                                ctypes.byref(output)):
                        valid = False
                        break
                    x += (output.integrated_body_forward_displacement_m *
                          math.cos(yaw) -
                          output.integrated_body_lateral_displacement_m *
                          math.sin(yaw))
                    y += (output.integrated_body_forward_displacement_m *
                          math.sin(yaw) +
                          output.integrated_body_lateral_displacement_m *
                          math.cos(yaw))
                    yaw += output.integrated_yaw_delta_rad
                    state = list(output.state)
                    angles = list(output.wheel_rotation_angle_rad)
                    history_initialized = bool(
                        output.suspension_history_initialized)
                    next_plant = plant_by_stamp[target_time]
                    if "motion" in corrected_groups or \
                            "u" in corrected_groups:
                        state[0] = actual["u"]
                    if "motion" in corrected_groups or \
                            "v" in corrected_groups:
                        state[1] = actual["v"]
                    if "motion" in corrected_groups or \
                            "yaw_rate" in corrected_groups:
                        state[2] = actual["r"]
                    if "attitude" in corrected_groups:
                        imu_row = imu_by_stamp.get(target_time)
                        state[9], state[11] = actual["roll"], actual["pitch"]
                        state[10] = (imu_row["roll_rate"] if imu_row else
                                     next_plant["roll_rate"])
                        state[12] = (imu_row["pitch_rate"] if imu_row else
                                     next_plant["pitch_rate"])
                    if "wheels" in corrected_groups:
                        state[3:7] = next_plant["wheel_omega"]
                    if "heading" in corrected_groups:
                        yaw = actual["yaw"]
                if valid:
                    oracle_xy[name].append(math.hypot(
                        x - truth_by_stamp[times[start + max(horizons)]]["x"],
                        y - truth_by_stamp[times[start + max(horizons)]]["y"]))
        print("  diagnostic-only 750ms state-oracle re-anchoring (future truth/sensors "
              "injected every 25ms; identifies error source, not a candidate):")
        for name, values in oracle_xy.items():
            if values:
                print(f"    {name}: endpoint XY RMSE="
                      f"{math.sqrt(statistics.mean(value*value for value in values)):.4f}m "
                      f"p95={percentile(values, 0.95):.4f}m "
                      f">5cm={sum(value > 0.05 for value in values)}/{len(values)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="rosbag directory")
    parser.add_argument("--trajectory", type=Path, help="raceline CSV used for the run")
    parser.add_argument(
        "--csv-follow-reference", type=Path,
        help="CSV command profile used by the development path follower",
    )
    parser.add_argument(
        "--source-plant-rollout", action="store_true",
        help="score a no-reanchoring 25 ms command-sequence rollout using the logged "
             "UnityPlantState and the production C++ source plant",
    )
    parser.add_argument(
        "--source-reconcile-suspension-jounce", action="store_true",
        help="diagnostic A/B: recompute geometric suspension jounce from each "
             "rollout's corrected initial body state, matching the production "
             "observer boundary update; no future measurements are used",
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
        "--source-extra-steering-delay-packets", type=int, default=0,
        help="diagnostic only: select the recorded steering queue-head command "
             "from this many earlier sensor epochs during replay",
    )
    parser.add_argument(
        "--source-extra-throttle-delay-packets", type=int, default=0,
        help="diagnostic only: select the recorded throttle queue-head command "
             "from this many earlier sensor epochs during replay",
    )
    parser.add_argument(
        "--source-oracle-steering-feedback", action="store_true",
        help="diagnostic only: use the next recorded steering feedback as the "
             "interval target to isolate steering-phase error; not deployable",
    )
    parser.add_argument(
        "--source-oracle-throttle-feedback", action="store_true",
        help="diagnostic only: use the next recorded Unity normalized torque "
             "feedback for each interval to isolate command/application phase; "
             "future information, never deployable",
    )
    parser.add_argument(
        "--source-reconstruct-command-queue-from-stamps", action="store_true",
        help="diagnostic: reconstruct the two-packet command queue from the "
             "recorded Float32 command topic arrival times and sensor request "
             "stamps, rather than the odom-published queue fields",
    )
    parser.add_argument(
        "--source-reconstruct-command-queue-from-bridge", action="store_true",
        help="offline diagnostic only: reconstruct the applied command sequence "
             "from bridge_packet_timing's recorded sent values keyed by each "
             "request stamp; this restricted debug stream is never a runtime input",
    )
    parser.add_argument(
        "--source-bridge-command-epoch-offset", type=int, default=0,
        help="offline diagnostic: choose command epoch (normally interval i uses "
             "request command i-1; offset +1 tests request i; negative tests older)",
    )
    parser.add_argument(
        "--diagnose-observed-state-reanchoring", action="store_true",
        help="diagnostic only: re-anchor selected predicted states to future "
             "captured truth/sensors every 25 ms to identify which state drives "
             "750 ms rollout drift; never a production input",
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
    csv_reference = None
    if args.csv_follow_reference:
        with args.csv_follow_reference.open(newline="", encoding="utf-8") as stream:
            first_line = stream.readline()
            if first_line.startswith("#"):
                first_line = first_line[1:].lstrip()
            reader = csv.DictReader(chain((first_line,), stream))
            fields = set(reader.fieldnames or ())
            if "target_speed_mps" in fields:
                speed_field = "target_speed_mps"
            elif {"s_m", "velocity_mps"}.issubset(fields):
                speed_field = "velocity_mps"
            else:
                parser.error(
                    "CSV follow reference needs target_speed_mps or the "
                    "native raceline fields s_m,velocity_mps")
            csv_reference = [
                {"target_speed_mps": float(row[speed_field])}
                for row in reader
            ]
    data = read_bag(args.bag)
    truth = data.get(GT, [])
    if not truth:
        parser.error(f"bag has no simulator truth stream: {GT}")

    print(f"truth: frame={truth[0]['frame']} child={truth[0]['child']} count={len(truth)}")
    report_development_actuator_timing(
        data.get(DEVELOPMENT_ACTUATOR_TIMING, []))
    report_localization_diagnostics(truth, data)
    if args.truth_pose_twist_phase or args.truth_pose_twist_phase_only:
        report_truth_pose_twist_phase(truth, data.get(BRIDGE_TIMING, []))
    if args.truth_pose_twist_phase_only:
        return 0
    for topic in (GT, ODOM, ODOM_DIAGNOSTICS, DIAGNOSTICS,
                  CANDIDATE_DIAGNOSTICS, THROTTLE_COMMAND, DRIVE_COMMAND,
                  CSV_REPLAY):
        print_interval(topic, data.get(topic, []))
    if data.get(CSV_REPLAY):
        report_csv_path_follow(
            data[CSV_REPLAY], data.get(DRIVE_COMMAND, []), csv_reference)
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
    if data.get(LAP):
        print("Completed-lap times (simulator last-lap topic):")
        report_lap_splits(data.get(LAST_LAP, []))
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
        report_actuator_feedback_packet_association(
            plant_rows, data.get(BRIDGE_TIMING, []))
        print("wheel omega has no independent Unity truth channel; do not score it as truth.")
        report_encoder_slip(
            truth, plant_rows, data.get(LEFT_ENCODER, []),
            data.get(RIGHT_ENCODER, []), data.get(THROTTLE, []))
        if args.source_plant_rollout:
            report_source_plant_rollout(
                truth, plant_rows, data.get(ODOM_DIAGNOSTICS, []),
                args.source_plant_library, data.get(IMU, []),
                data.get(LEFT_ENCODER, []), data.get(RIGHT_ENCODER, []),
                args.yaw_residual_gain,
                args.yaw_residual_decay, args.oracle_motion_initial_state,
                args.diagnose_observed_state_reanchoring,
                args.source_extra_steering_delay_packets,
                args.source_extra_throttle_delay_packets,
                args.source_oracle_steering_feedback,
                args.source_oracle_throttle_feedback,
                args.source_reconstruct_command_queue_from_stamps,
                args.source_reconstruct_command_queue_from_bridge,
                args.source_bridge_command_epoch_offset,
                data.get(STEERING_COMMAND, []),
                data.get(THROTTLE_COMMAND, []),
                data.get(BRIDGE_TIMING, []),
                args.source_reconcile_suspension_jounce)
    elif args.source_plant_rollout:
        print(f"source-plant rollout: bag has no usable {PLANT} stream")
    # A replay bag may contain both the source run's controller diagnostics
    # and a candidate controller's output. Score the candidate in isolation;
    # mixing both streams double-counts cycles and corrupts ADMM convergence
    # rates and rollout comparisons.
    diagnostic_topic = (
        CANDIDATE_DIAGNOSTICS
        if data.get(CANDIDATE_DIAGNOSTICS)
        else DIAGNOSTICS
    )
    diagnostics = [
        row for row in data.get(diagnostic_topic, []) if row.get("json")
    ]
    if diagnostics:
        payloads = [row["json"] for row in diagnostics]
        repaired = sum(bool(x.get("_bag_decoder_repaired_action_source_quote"))
                       for x in payloads)
        undecodable = sum("_decode_error" in x for x in payloads)
        if repaired or undecodable:
            print("MPC diagnostic JSON decode: "
                  f"repaired_known_quote={repaired} undecodable={undecodable}")
        print(f"MPC status counts: {dict(Counter(x.get('status') for x in payloads))}")
        collision_values = [int(row["value"])
                            for row in data.get(COLLISION, [])
                            if isinstance(row.get("value"), (int, float))]
        report_candidate_prefix_chains(
            payloads, max(collision_values) if collision_values else None)
        report_prefix_warm_start_consistency(diagnostics)
        report_jacobian_worker_distribution(payloads)
        report_admm_convergence(diagnostics, data.get(DRIVE_COMMAND, []))
        print(f"MPC plant-state source counts: {dict(Counter(x.get('plant_state_source') for x in payloads))}")
        report_horizon_errors(truth, diagnostics, load_trajectory(args.trajectory))
    if data.get(OPEN_LOOP_TEST):
        report_open_loop_horizon_test(
            truth, data[OPEN_LOOP_TEST], data.get(COLLISION, []),
            data.get(STEERING_COMMAND, []), data.get(THROTTLE_COMMAND, []))
    for row in data.get(BRIDGE_FAULT, []):
        if row["value"]:
            print(f"bridge timing fault detail: {row['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replay deployment sensor topics through the production C++ odometry path."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tools.racing.offline.localization.build_production_odom import build
from tools.racing.offline.localization.production_odom import ProductionOdometry


ROOT = Path(__file__).resolve().parents[4]
LEFT = "/autodrive/roboracer_1/left_encoder"
RIGHT = "/autodrive/roboracer_1/right_encoder"
IMU = "/autodrive/roboracer_1/imu"
STEERING = "/autodrive/roboracer_1/steering"
THROTTLE = "/autodrive/roboracer_1/throttle"
TRUTH = "/autodrive/roboracer_1/odom"
PRODUCTION = "/odom"
# Unity's raw bridge twist is measured at the Rigidbody COM, whereas the
# production observer publishes lateral velocity at the rear-axle base_link.
TRUTH_COM_TO_REAR_AXLE_M = 0.15532


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def yaw_from_quaternion(q: Any) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def stats(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    absolute = np.abs(finite)
    return {
        "count": int(len(finite)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
        "mae": float(np.mean(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def replay(bag: Path, library: Path, output: Path, config: Path,
           start_offset_s: float = 0.0,
           duration_s: float | None = None) -> dict[str, Any]:
    if not math.isfinite(start_offset_s) or start_offset_s < 0.0:
        raise ValueError("start offset must be finite and nonnegative")
    if duration_s is not None and (
            not math.isfinite(duration_s) or duration_s <= 0.0):
        raise ValueError("duration must be finite and positive")
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    message_types = {entry.name: get_message(entry.type)
                     for entry in reader.get_all_topics_and_types()}
    required = {LEFT, RIGHT, IMU, STEERING, THROTTLE, TRUTH}
    absent = sorted(required - set(message_types))
    if absent:
        raise ValueError(f"bag lacks required odometry channels: {absent}")
    has_recorded_production = PRODUCTION in message_types

    production_rows: dict[int, dict[str, float]] = {}
    truth_rows: dict[int, dict[str, float]] = {}
    replay_rows: dict[int, dict[str, float]] = {}
    last_replay_source_key: int | None = None
    latest_receive_ns = 0
    bag_start_ns: int | None = None
    window_start_ns: int | None = None
    window_end_ns: int | None = None
    with ProductionOdometry(library, config) as odom:
        while reader.has_next():
            topic, serialized, receive_ns = reader.read_next()
            receive_ns = int(receive_ns)
            if bag_start_ns is None:
                bag_start_ns = receive_ns
                window_start_ns = bag_start_ns + round(start_offset_s * 1.0e9)
                if duration_s is not None:
                    window_end_ns = window_start_ns + round(duration_s * 1.0e9)
            if receive_ns < window_start_ns:
                continue
            if window_end_ns is not None and receive_ns >= window_end_ns:
                break
            latest_receive_ns = max(latest_receive_ns, receive_ns)
            odom.advance_time(receive_ns)
            message = deserialize_message(serialized, message_types[topic])

            if topic == LEFT:
                odom.add_left_encoder(
                    stamp_ns(message.header.stamp), message.position[0], receive_ns)
            elif topic == RIGHT:
                odom.add_right_encoder(
                    stamp_ns(message.header.stamp), message.position[0], receive_ns)
            elif topic == IMU:
                odom.add_imu(
                    stamp_ns(message.header.stamp),
                    message.linear_acceleration.x,
                    message.linear_acceleration.y,
                    message.angular_velocity.z,
                    yaw_from_quaternion(message.orientation),
                    receive_ns,
                )
            elif topic == STEERING:
                odom.add_steering_feedback(receive_ns, float(message.data))
            elif topic == THROTTLE:
                odom.add_throttle_feedback(receive_ns, float(message.data))
            elif topic == PRODUCTION:
                header_ns = stamp_ns(message.header.stamp)
                q = message.pose.pose.orientation
                production_rows[header_ns] = {
                    "recorded_x": float(message.pose.pose.position.x),
                    "recorded_y": float(message.pose.pose.position.y),
                    "recorded_yaw": yaw_from_quaternion(q),
                    "recorded_u": float(message.twist.twist.linear.x),
                    "recorded_v": float(message.twist.twist.linear.y),
                    "recorded_yaw_rate": float(message.twist.twist.angular.z),
                }
            elif topic == TRUTH:
                q = message.pose.pose.orientation
                truth_yaw_rate = float(message.twist.twist.angular.z)
                truth_rows[stamp_ns(message.header.stamp)] = {
                    "truth_x": float(message.pose.pose.position.x),
                    "truth_y": float(message.pose.pose.position.y),
                    "truth_yaw": yaw_from_quaternion(q),
                    "truth_u": float(message.twist.twist.linear.x),
                    "truth_v_com": float(message.twist.twist.linear.y),
                    "truth_v_rear": float(message.twist.twist.linear.y)
                    - truth_yaw_rate * TRUTH_COM_TO_REAR_AXLE_M,
                    "truth_yaw_rate": truth_yaw_rate,
                }

            state = odom.snapshot()
            if state is not None:
                source_key = int(round(state.source_stamp_s * 1.0e9))
                if source_key != last_replay_source_key:
                    replay_rows[source_key] = {
                        f"replay_{name}": float(value)
                        for name, value in state.values.items()
                    }
                    last_replay_source_key = source_key

        odom.advance_time(latest_receive_ns + 20_000_000)
        state = odom.snapshot()
        if state is not None:
            source_key = int(round(state.source_stamp_s * 1.0e9))
            replay_rows[source_key] = {
                f"replay_{name}": float(value)
                for name, value in state.values.items()
            }

    truth_keys = sorted(truth_rows)
    aligned_replay: dict[int, dict[str, float]] = {}
    for source_key, row in replay_rows.items():
        index = int(np.searchsorted(truth_keys, source_key))
        nearest = min(truth_keys[max(index - 1, 0):min(index + 1, len(truth_keys))],
                      key=lambda candidate: abs(candidate - source_key))
        if abs(nearest - source_key) <= 1_000:
            aligned_replay[nearest] = row
    keys_set = set(truth_rows) & set(aligned_replay)
    if has_recorded_production:
        keys_set &= set(production_rows)
    keys = sorted(keys_set)
    if not keys:
        raise RuntimeError("no timestamp-aligned replay, production, and truth rows")
    rows: list[dict[str, float]] = []
    for key in keys:
        recorded = production_rows.get(key, {
            "recorded_x": math.nan,
            "recorded_y": math.nan,
            "recorded_yaw": math.nan,
            "recorded_u": math.nan,
            "recorded_v": math.nan,
            "recorded_yaw_rate": math.nan,
        })
        truth = truth_rows[key]
        replay = aligned_replay[key]
        replay_x = replay["replay_x_m"]
        replay_y = replay["replay_y_m"]
        replay_yaw = replay["replay_yaw_rad"]
        truth_yaw0 = truth_rows[keys[0]]["truth_yaw"]
        truth_x0 = truth_rows[keys[0]]["truth_x"]
        truth_y0 = truth_rows[keys[0]]["truth_y"]
        dx, dy = truth["truth_x"] - truth_x0, truth["truth_y"] - truth_y0
        c, s = math.cos(truth_yaw0), math.sin(truth_yaw0)
        truth_local_x = c * dx + s * dy
        truth_local_y = -s * dx + c * dy
        rows.append({
            "source_stamp_s": key * 1.0e-9,
            "replay_x_m": replay_x,
            "replay_y_m": replay_y,
            "recorded_x_m": recorded["recorded_x"],
            "recorded_y_m": recorded["recorded_y"],
            "truth_local_x_m": truth_local_x,
            "truth_local_y_m": truth_local_y,
            "replay_yaw_rad": replay_yaw,
            "recorded_yaw_rad": recorded["recorded_yaw"],
            "truth_relative_yaw_rad": float(wrap(np.array(
                [truth["truth_yaw"] - truth_yaw0]))[0]),
            "replay_u_mps": replay["replay_body_u_mps"],
            "recorded_u_mps": recorded["recorded_u"],
            "truth_u_mps": truth["truth_u"],
            "replay_v_mps": replay["replay_body_v_mps"],
            "recorded_v_mps": recorded["recorded_v"],
            "truth_v_com_mps": truth["truth_v_com"],
            "truth_v_mps": truth["truth_v_rear"],
            "replay_yaw_rate_radps": replay["replay_yaw_rate_radps"],
            "recorded_yaw_rate_radps": recorded["recorded_yaw_rate"],
            "truth_yaw_rate_radps": truth["truth_yaw_rate"],
            "actuator_feedback_valid": replay["replay_actuator_feedback_valid"],
            "steering_feedback_rad": replay["replay_steering_feedback_rad"],
            "throttle_feedback": replay["replay_throttle_feedback"],
            "wheel_raw_mps": replay["replay_wheel_raw_mps"],
            "wheel_mapped_mps": replay["replay_wheel_mapped_mps"],
            "wheel_packet_mps": replay["replay_wheel_packet_mps"],
            "turn_speed_bias_mps": replay["replay_turn_speed_bias_mps"],
            "ax_mps2": replay["replay_ax_mps2"],
            "ay_mps2": replay["replay_ay_mps2"],
            "wheel_update_used": replay["replay_wheel_update_used"],
            "wheel_burst_rejected": replay["replay_wheel_burst_rejected"],
            "turn_mode": replay["replay_turn_mode"],
            "reset_epoch": replay["replay_reset_epoch"],
            "timing_degraded": replay["replay_timing_degraded"],
            "sensor_outlier": replay["replay_sensor_outlier"],
        })

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "production_odom_source_replay.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    arrays = {name: np.asarray([row[name] for row in rows], dtype=np.float64)
              for name in rows[0]}
    reset_epoch_indices = np.flatnonzero(arrays["reset_epoch"] > 0.5)
    result = {
        "bag": str(bag),
        "config": str(config),
        "window_start_offset_s": start_offset_s,
        "window_duration_s": duration_s,
        "recorded_production_topic_available": has_recorded_production,
        "truth_twist_frame_conversion": {
            "raw_simulator_lateral_velocity_reference": "Rigidbody COM",
            "replay_and_production_lateral_velocity_reference": "rear axle / base_link",
            "com_to_rear_axle_forward_offset_m": TRUTH_COM_TO_REAR_AXLE_M,
            "equation": "v_rear = v_com - yaw_rate * 0.15532 m",
        },
        "aligned_samples": len(rows),
        "time_span_s": float(rows[-1]["source_stamp_s"] - rows[0]["source_stamp_s"]),
        "actuator_feedback_valid_fraction": float(np.mean(
            arrays["actuator_feedback_valid"])),
        "reset_epoch_events": {
            "count": int(len(reset_epoch_indices)),
            "events": [
                {
                    "source_stamp_s": float(rows[index]["source_stamp_s"]),
                    "wheel_raw_mps": float(rows[index]["wheel_raw_mps"]),
                    "body_u_mps": float(rows[index]["replay_u_mps"]),
                    "body_v_mps": float(rows[index]["replay_v_mps"]),
                }
                for index in reset_epoch_indices
            ],
        },
        "replay_vs_recorded_production": {
            "x_m": stats(arrays["replay_x_m"] - arrays["recorded_x_m"]),
            "y_m": stats(arrays["replay_y_m"] - arrays["recorded_y_m"]),
            "yaw_rad": stats(wrap(
                arrays["replay_yaw_rad"] - arrays["recorded_yaw_rad"])),
            "u_mps": stats(arrays["replay_u_mps"] - arrays["recorded_u_mps"]),
            "v_mps": stats(arrays["replay_v_mps"] - arrays["recorded_v_mps"]),
            "yaw_rate_radps": stats(
                arrays["replay_yaw_rate_radps"] - arrays["recorded_yaw_rate_radps"]),
        },
        "replay_vs_truth": {
            "planar_position_m": stats(np.hypot(
                arrays["replay_x_m"] - arrays["truth_local_x_m"],
                arrays["replay_y_m"] - arrays["truth_local_y_m"])),
            "yaw_rad": stats(wrap(
                arrays["replay_yaw_rad"] - arrays["truth_relative_yaw_rad"])),
            "u_mps": stats(arrays["replay_u_mps"] - arrays["truth_u_mps"]),
            "v_mps": stats(arrays["replay_v_mps"] - arrays["truth_v_mps"]),
            "yaw_rate_radps": stats(
                arrays["replay_yaw_rate_radps"] - arrays["truth_yaw_rate_radps"]),
        },
        "recorded_production_vs_truth": {
            "planar_position_m": stats(np.hypot(
                arrays["recorded_x_m"] - arrays["truth_local_x_m"],
                arrays["recorded_y_m"] - arrays["truth_local_y_m"])),
            "yaw_rad": stats(wrap(
                arrays["recorded_yaw_rad"] - arrays["truth_relative_yaw_rad"])),
            "u_mps": stats(arrays["recorded_u_mps"] - arrays["truth_u_mps"]),
            "v_mps": stats(arrays["recorded_v_mps"] - arrays["truth_v_mps"]),
            "yaw_rate_radps": stats(
                arrays["recorded_yaw_rate_radps"] - arrays["truth_yaw_rate_radps"]),
        },
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "f1tenth_localization/config/sensor_odometry.yaml")
    parser.add_argument("--start-offset-s", type=float, default=0.0,
                        help="restart the observer at this bag-relative time")
    parser.add_argument("--duration-s", type=float,
                        help="stop replay after this many seconds in the selected window")
    args = parser.parse_args()
    replay(args.bag.resolve(), args.library.resolve(), args.output.resolve(),
           args.config.resolve(), args.start_offset_s, args.duration_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

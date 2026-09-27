#!/usr/bin/env python3
"""Identify the reference point of simulator odometry velocity from a bag.

AutoDRIVE publishes vehicle pose at the rear-axle frame and copies the
simulator linear-velocity vector into Odometry.twist unchanged. This compares
that twist with the differentiated rear-axle pose under two hypotheses:
velocity already at the rear axle, or velocity at the documented COM.

The COM hypothesis uses the published 0.15532 m longitudinal COM offset. This
is an offline development diagnostic; it does not change runtime behavior.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - depends on ROS installation
    raise SystemExit(f"ROS 2 Python modules are required: {exc}") from exc


ODOM = "/autodrive/roboracer_1/odom"
COM_X_M = 0.15532
MIN_SPEED_MPS = 1.0
MIN_YAW_RATE_RPS = 0.20
MIN_SAMPLES = 100


@dataclass(frozen=True)
class Row:
    time_ns: int
    x_m: float
    y_m: float
    yaw_rad: float
    vx_mps: float
    vy_mps: float
    yaw_rate_rps: float


def _yaw(quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z))


def _angle_delta(right: float, left: float) -> float:
    return math.atan2(math.sin(right - left), math.cos(right - left))


def load_rows(path: Path) -> list[Row]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = {
            name: (int(topic_id), msg_type)
            for topic_id, name, msg_type in connection.execute(
                "SELECT id, name, type FROM topics")
        }
        if ODOM not in topics:
            raise ValueError(f"bag is missing {ODOM}")
        topic_id, msg_type = topics[ODOM]
        message_class = get_message(msg_type)
        rows = []
        for receipt_ns, payload in connection.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp, id", (topic_id,)):
            message = deserialize_message(bytes(payload), message_class)
            pose = message.pose.pose
            twist = message.twist.twist
            values = (float(pose.position.x), float(pose.position.y),
                      _yaw(pose.orientation), float(twist.linear.x),
                      float(twist.linear.y), float(twist.angular.z))
            if all(math.isfinite(value) for value in values):
                rows.append(Row(int(receipt_ns), *values))
        return rows
    finally:
        connection.close()


def _rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def evaluate(path: Path) -> bool:
    rows = load_rows(path)
    same_x, same_y, com_x, com_y = [], [], [], []
    yaw_rate_errors = []
    selected_turn_signs: set[int] = set()
    for previous, current, following in zip(rows, rows[1:], rows[2:]):
        dt_s = (following.time_ns - previous.time_ns) / 1e9
        if not 0.035 <= dt_s <= 0.065:
            continue
        world_vx = (following.x_m - previous.x_m) / dt_s
        world_vy = (following.y_m - previous.y_m) / dt_s
        cosine, sine = math.cos(current.yaw_rad), math.sin(current.yaw_rad)
        rear_vx = cosine * world_vx + sine * world_vy
        rear_vy = -sine * world_vx + cosine * world_vy
        pose_yaw_rate = _angle_delta(following.yaw_rad, previous.yaw_rad) / dt_s
        yaw_rate_errors.append(current.yaw_rate_rps - pose_yaw_rate)
        if (math.hypot(current.vx_mps, current.vy_mps) <= MIN_SPEED_MPS
                or abs(current.yaw_rate_rps) <= MIN_YAW_RATE_RPS):
            continue
        # A positive longitudinal lever arm gives v_COM_y = v_rear_y + r*x.
        same_x.append(current.vx_mps - rear_vx)
        same_y.append(current.vy_mps - rear_vy)
        com_x.append(current.vx_mps - rear_vx)
        com_y.append(current.vy_mps - (rear_vy + current.yaw_rate_rps * COM_X_M))
        selected_turn_signs.add(1 if current.yaw_rate_rps > 0.0 else -1)

    if len(same_x) < MIN_SAMPLES:
        raise ValueError(f"insufficient moving-turn data: {len(same_x)} samples")
    same_xy_rmse = math.hypot(_rmse(same_x), _rmse(same_y))
    com_xy_rmse = math.hypot(_rmse(com_x), _rmse(com_y))
    yaw_rmse = _rmse(yaw_rate_errors)
    ratio = same_xy_rmse / max(com_xy_rmse, 1.0e-12)
    com_supported = (
        len(selected_turn_signs) == 2
        and com_xy_rmse < same_xy_rmse
        and ratio >= 3.0
        and _rmse(com_x) <= 0.10
        and yaw_rmse <= 0.10)
    rear_supported = (
        len(selected_turn_signs) == 2
        and same_xy_rmse < com_xy_rmse
        and 1.0 / max(ratio, 1.0e-12) >= 3.0
        and _rmse(same_x) <= 0.10
        and yaw_rmse <= 0.10)
    print(f"bag: {path}")
    print(f"differentiated pose samples: {len(same_x)} moving-turn samples; "
          f"turn signs={sorted(selected_turn_signs)}")
    print(f"rear axle point vs copied twist: vx/vy RMSE="
          f"{_rmse(same_x):.5f}/{_rmse(same_y):.5f} m/s; "
          f"combined={same_xy_rmse:.5f} m/s")
    print(f"COM point vs copied twist: vx/vy RMSE="
          f"{_rmse(com_x):.5f}/{_rmse(com_y):.5f} m/s; "
          f"combined={com_xy_rmse:.5f} m/s; "
          f"rear/COM residual ratio={ratio:.2f}x")
    print(f"pose-derived vs copied yaw-rate RMSE={yaw_rmse:.5f} rad/s")
    if com_supported:
        print("decision: copied simulator velocity is COM-referenced; transform "
              "to rear axle before using it as the vehicle-frame twist")
    elif rear_supported:
        print("decision: copied simulator velocity is rear-axle-referenced; "
              "do not apply a COM offset")
    else:
        print("decision: inconclusive under preregistered separation/quality gates")
    return com_supported or rear_supported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())

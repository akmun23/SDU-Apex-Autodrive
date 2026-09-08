#!/usr/bin/env python3
"""Extract compact CSV diagnostics from an AutoDRIVE FTG rosbag.

Run inside the ROS 2 Humble workspace container so rosbag2_py is available:
  python3 /workspace/src/f1tenth_planning/analysis/extract_ftg_run.py BAG OUT
"""

from __future__ import annotations

import csv
import math
import os
import sys

import rosbag2_py
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float64MultiArray, Int32
from tf_transformations import euler_from_quaternion


def stamp(msg, bag_ns: int) -> float:
    header = getattr(msg, "header", None)
    if header is None:
        return bag_ns / 1.0e9
    return float(header.stamp.sec) + float(header.stamp.nanosec) / 1.0e9


def write_rows(path: str, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_ftg_run.py BAG OUT", file=sys.stderr)
        return 2
    bag_path, out_dir = sys.argv[1:]
    os.makedirs(out_dir, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    odom_rows: list[dict] = []
    diag_rows: list[dict] = []
    cmd_rows: list[dict] = []
    lidar_rows: list[dict] = []
    collision_rows: list[dict] = []
    lap_rows: list[dict] = []
    topic_classes = {
        "nav_msgs/msg/Odometry": Odometry,
        "std_msgs/msg/Float64MultiArray": Float64MultiArray,
        "ackermann_msgs/msg/AckermannDriveStamped": AckermannDriveStamped,
        "sensor_msgs/msg/LaserScan": LaserScan,
        "std_msgs/msg/Int32": Int32,
        "std_msgs/msg/Float32": Float32,
    }

    while reader.has_next():
        topic, raw, bag_ns = reader.read_next()
        msg_type = types[topic]
        cls = topic_classes.get(msg_type)
        if cls is None:
            continue
        msg = deserialize_message(raw, cls)
        t = stamp(msg, bag_ns)
        if topic.endswith("/odom"):
            q = msg.pose.pose.orientation
            yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            odom_rows.append({
                "bag_time": bag_ns / 1.0e9, "time": t,
                "x": msg.pose.pose.position.x, "y": msg.pose.pose.position.y,
                "yaw": yaw, "vx": msg.twist.twist.linear.x,
                "vy": msg.twist.twist.linear.y, "wz": msg.twist.twist.angular.z,
            })
        elif topic == "/ftg/diagnostics":
            row = {"bag_time": bag_ns / 1.0e9, "time": t}
            row.update({f"d{i}": value for i, value in enumerate(msg.data)})
            diag_rows.append(row)
        elif topic == "/cmd/speed":
            cmd_rows.append({
                "bag_time": bag_ns / 1.0e9, "time": t,
                "speed": msg.drive.speed, "steering": msg.drive.steering_angle,
            })
        elif topic.endswith("/lidar"):
            finite = [float(value) for value in msg.ranges if math.isfinite(value)]
            lidar_rows.append({
                "bag_time": bag_ns / 1.0e9, "time": t,
                "finite_count": len(finite),
                "min_finite": min(finite) if finite else "",
                "max_finite": max(finite) if finite else "",
                "angle_min": msg.angle_min, "angle_max": msg.angle_max,
            })
        elif topic.endswith("/collision_count"):
            collision_rows.append({"bag_time": bag_ns / 1.0e9, "time": t, "count": msg.data})
        elif topic.endswith("/lap_count"):
            lap_rows.append({"bag_time": bag_ns / 1.0e9, "time": t, "count": msg.data})

    write_rows(os.path.join(out_dir, "odom.csv"),
               ["bag_time", "time", "x", "y", "yaw", "vx", "vy", "wz"], odom_rows)
    max_diag_values = max(
        (len(row) - 2 for row in diag_rows), default=0)
    diag_fields = ["bag_time", "time"] + [
        f"d{i}" for i in range(max_diag_values)]
    write_rows(os.path.join(out_dir, "ftg_diagnostics.csv"), diag_fields, diag_rows)
    write_rows(os.path.join(out_dir, "cmd_speed.csv"),
               ["bag_time", "time", "speed", "steering"], cmd_rows)
    write_rows(os.path.join(out_dir, "lidar_summary.csv"),
               ["bag_time", "time", "finite_count", "min_finite", "max_finite",
                "angle_min", "angle_max"], lidar_rows)
    write_rows(os.path.join(out_dir, "collision_count.csv"),
               ["bag_time", "time", "count"], collision_rows)
    write_rows(os.path.join(out_dir, "lap_count.csv"),
               ["bag_time", "time", "count"], lap_rows)

    if odom_rows:
        start = odom_rows[0]
        end = odom_rows[-1]
        path_length = sum(
            math.hypot(float(cur["x"]) - float(prev["x"]),
                       float(cur["y"]) - float(prev["y"]))
            for prev, cur in zip(odom_rows, odom_rows[1:])
        )
        print(f"odom_rows={len(odom_rows)} path_length_m={path_length:.3f} "
              f"start=({start['x']:.3f},{start['y']:.3f}) "
              f"end=({end['x']:.3f},{end['y']:.3f})")
    if collision_rows:
        print(f"collision_final={collision_rows[-1]['count']}")
    if lidar_rows:
        finite = sum(row["finite_count"] > 0 for row in lidar_rows)
        print(f"lidar_scans={len(lidar_rows)} finite_scans={finite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

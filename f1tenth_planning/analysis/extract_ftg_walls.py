#!/usr/bin/env python3
"""Extract a complete FTG run and plot the ground-truth route and LiDAR walls.

Run inside the ROS 2 Humble workspace container, for example:

  python3 /workspace/src/f1tenth_planning/analysis/extract_ftg_walls.py \
      /tmp/ftg_fullrun_attempt2_20260908 \
      /workspace/src/f1tenth_planning/analysis/ftg_fullrun_attempt2_20260908

The rosbag remains the lossless source of record.  The generated wall CSV is
the raw LiDAR hit points transformed into the simulator world frame using the
recorded simulator odometry (ground truth for this mapping run).
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any

import rosbag2_py
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32, String
from tf_transformations import euler_from_quaternion


def message_time(message: Any, bag_ns: int) -> float:
    header = getattr(message, "header", None)
    if header is None:
        return bag_ns / 1.0e9
    return float(header.stamp.sec) + float(header.stamp.nanosec) / 1.0e9


def write_rows(path: str, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def yaw_from_odom(message: Odometry) -> float:
    q = message.pose.pose.orientation
    return euler_from_quaternion([q.x, q.y, q.z, q.w])[2]


def scalar_row(bag_ns: int, message: Any) -> dict[str, Any]:
    return {"bag_time": bag_ns / 1.0e9, "time": bag_ns / 1.0e9, "value": message.data}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_ftg_walls.py BAG OUT_DIR", file=sys.stderr)
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
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    message_classes = {
        "ackermann_msgs/msg/AckermannDriveStamped": AckermannDriveStamped,
        "nav_msgs/msg/Odometry": Odometry,
        "sensor_msgs/msg/LaserScan": LaserScan,
        "std_msgs/msg/Float32": Float32,
        "std_msgs/msg/Int32": Int32,
        "std_msgs/msg/String": String,
        "diagnostic_msgs/msg/DiagnosticArray": DiagnosticArray,
    }

    odom_rows: list[dict[str, Any]] = []
    scan_records: list[tuple[float, LaserScan]] = []
    cmd_rows: list[dict[str, Any]] = []
    steering_rows: list[dict[str, Any]] = []
    throttle_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    lap_rows: list[dict[str, Any]] = []
    bridge_timing_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    topic_counts: dict[str, int] = defaultdict(int)

    while reader.has_next():
        topic, raw, bag_ns = reader.read_next()
        topic_counts[topic] += 1
        message_class = message_classes.get(topic_types[topic])
        if message_class is None:
            continue
        message = deserialize_message(raw, message_class)
        time_s = message_time(message, bag_ns)

        if topic.endswith("/odom"):
            odom_rows.append({
                "bag_time": bag_ns / 1.0e9,
                "time": time_s,
                "x": message.pose.pose.position.x,
                "y": message.pose.pose.position.y,
                "z": message.pose.pose.position.z,
                "yaw": yaw_from_odom(message),
                "vx": message.twist.twist.linear.x,
                "vy": message.twist.twist.linear.y,
                "vz": message.twist.twist.linear.z,
                "wx": message.twist.twist.angular.x,
                "wy": message.twist.twist.angular.y,
                "wz": message.twist.twist.angular.z,
            })
        elif topic.endswith("/lidar"):
            scan_records.append((time_s, message))
        elif topic == "/cmd/speed":
            cmd_rows.append({
                "bag_time": bag_ns / 1.0e9,
                "time": time_s,
                "speed": message.drive.speed,
                "steering": message.drive.steering_angle,
                "acceleration": message.drive.acceleration,
                "jerk": message.drive.jerk,
                "steering_angle_velocity": message.drive.steering_angle_velocity,
            })
        elif topic.endswith("/steering_command"):
            steering_rows.append(scalar_row(bag_ns, message))
        elif topic.endswith("/throttle_command"):
            throttle_rows.append(scalar_row(bag_ns, message))
        elif topic.endswith("/collision_count"):
            collision_rows.append(scalar_row(bag_ns, message))
        elif topic.endswith("/lap_count"):
            lap_rows.append(scalar_row(bag_ns, message))
        elif topic.endswith("/bridge_packet_timing"):
            bridge_timing_rows.append({
                "bag_time": bag_ns / 1.0e9,
                "time": time_s,
                "value": message.data,
            })
        elif message_class is DiagnosticArray:
            for status in message.status:
                row: dict[str, Any] = {
                    "bag_time": bag_ns / 1.0e9,
                    "time": time_s,
                    "status": status.message,
                    "level": status.level,
                }
                for item in status.values:
                    row[item.key] = item.value
                diagnostic_rows.append(row)

    odom_rows.sort(key=lambda row: row["time"])
    scan_records.sort(key=lambda record: record[0])

    write_rows(
        os.path.join(out_dir, "ground_truth_odom.csv"),
        ["bag_time", "time", "x", "y", "z", "yaw", "vx", "vy", "vz", "wx", "wy", "wz"],
        odom_rows,
    )
    write_rows(
        os.path.join(out_dir, "cmd_speed.csv"),
        ["bag_time", "time", "speed", "steering", "acceleration", "jerk", "steering_angle_velocity"],
        cmd_rows,
    )
    write_rows(os.path.join(out_dir, "steering_command.csv"), ["bag_time", "time", "value"], steering_rows)
    write_rows(os.path.join(out_dir, "throttle_command.csv"), ["bag_time", "time", "value"], throttle_rows)
    write_rows(os.path.join(out_dir, "collision_count.csv"), ["bag_time", "time", "value"], collision_rows)
    write_rows(os.path.join(out_dir, "lap_count.csv"), ["bag_time", "time", "value"], lap_rows)
    write_rows(os.path.join(out_dir, "bridge_packet_timing.csv"), ["bag_time", "time", "value"], bridge_timing_rows)
    diagnostic_fields = ["bag_time", "time", "status", "level"]
    diagnostic_keys = sorted({
        key for row in diagnostic_rows for key in row
        if key not in diagnostic_fields
    })
    write_rows(
        os.path.join(out_dir, "ftg_diagnostics.csv"),
        diagnostic_fields + diagnostic_keys,
        diagnostic_rows,
    )

    odom_times = [row["time"] for row in odom_rows]
    lidar_summary: list[dict[str, Any]] = []
    wall_rows: list[dict[str, Any]] = []
    wall_x: list[float] = []
    wall_y: list[float] = []
    lidar_x_offset_m = 0.2733
    lidar_y_offset_m = 0.0

    for scan_index, (scan_time, scan) in enumerate(scan_records):
        finite_count = 0
        min_range = math.inf
        max_range = -math.inf
        odom_index = -1
        odom_dt = math.nan
        if odom_times:
            odom_index = min(
                max(0, bisect.bisect_left(odom_times, scan_time)),
                len(odom_times) - 1,
            )
            if odom_index > 0 and abs(odom_times[odom_index - 1] - scan_time) < abs(odom_times[odom_index] - scan_time):
                odom_index -= 1
            pose = odom_rows[odom_index]
            odom_dt = pose["time"] - scan_time
            cos_yaw = math.cos(pose["yaw"])
            sin_yaw = math.sin(pose["yaw"])
        else:
            pose = None
            cos_yaw = 1.0
            sin_yaw = 0.0

        for beam_index, value in enumerate(scan.ranges):
            range_m = float(value)
            if not math.isfinite(range_m) or range_m < scan.range_min or range_m > scan.range_max:
                continue
            finite_count += 1
            min_range = min(min_range, range_m)
            max_range = max(max_range, range_m)
            if pose is None:
                continue
            angle = scan.angle_min + beam_index * scan.angle_increment
            local_x = lidar_x_offset_m + range_m * math.cos(angle)
            local_y = lidar_y_offset_m + range_m * math.sin(angle)
            world_x = pose["x"] + cos_yaw * local_x - sin_yaw * local_y
            world_y = pose["y"] + sin_yaw * local_x + cos_yaw * local_y
            wall_x.append(world_x)
            wall_y.append(world_y)
            wall_rows.append({
                "scan_index": scan_index,
                "scan_time": scan_time,
                "odom_index": odom_index,
                "odom_time": pose["time"],
                "odom_dt": odom_dt,
                "beam_index": beam_index,
                "angle": angle,
                "range": range_m,
                "world_x": world_x,
                "world_y": world_y,
            })

        lidar_summary.append({
            "scan_index": scan_index,
            "bag_time": scan_time,
            "time": scan_time,
            "finite_count": finite_count,
            "min_range": "" if finite_count == 0 else min_range,
            "max_range": "" if finite_count == 0 else max_range,
            "angle_min": scan.angle_min,
            "angle_max": scan.angle_max,
            "angle_increment": scan.angle_increment,
            "range_min": scan.range_min,
            "range_max": scan.range_max,
            "odom_index": odom_index,
            "odom_dt": odom_dt,
        })

    write_rows(
        os.path.join(out_dir, "lidar_summary.csv"),
        ["scan_index", "bag_time", "time", "finite_count", "min_range", "max_range", "angle_min", "angle_max", "angle_increment", "range_min", "range_max", "odom_index", "odom_dt"],
        lidar_summary,
    )
    write_rows(
        os.path.join(out_dir, "lidar_wall_points.csv"),
        ["scan_index", "scan_time", "odom_index", "odom_time", "odom_dt", "beam_index", "angle", "range", "world_x", "world_y"],
        wall_rows,
    )

    path_length = sum(
        math.hypot(cur["x"] - prev["x"], cur["y"] - prev["y"])
        for prev, cur in zip(odom_rows, odom_rows[1:])
    )
    summary = {
        "bag": os.path.abspath(bag_path),
        "odom_rows": len(odom_rows),
        "lidar_scans": len(scan_records),
        "lidar_wall_points": len(wall_rows),
        "cmd_rows": len(cmd_rows),
        "steering_command_rows": len(steering_rows),
        "throttle_command_rows": len(throttle_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "path_length_m": path_length,
        "start": None if not odom_rows else [odom_rows[0]["x"], odom_rows[0]["y"]],
        "end": None if not odom_rows else [odom_rows[-1]["x"], odom_rows[-1]["y"]],
        "collision_final": None if not collision_rows else collision_rows[-1]["value"],
        "lap_final": None if not lap_rows else lap_rows[-1]["value"],
        "topic_counts": dict(topic_counts),
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w") as stream:
        json.dump(summary, stream, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 10))
        if wall_x:
            axis.scatter(wall_x, wall_y, s=0.12, c="0.45", alpha=0.28, linewidths=0, label="LiDAR wall hits")
        if odom_rows:
            route_x = [row["x"] for row in odom_rows]
            route_y = [row["y"] for row in odom_rows]
            axis.plot(route_x, route_y, color="tab:blue", linewidth=1.5, label="ground-truth route")
            axis.scatter(route_x[0], route_y[0], c="green", s=55, zorder=4, label="start")
            axis.scatter(route_x[-1], route_y[-1], c="red", marker="x", s=65, zorder=4, label="end")
        axis.set_title("Stripped FTG run: ground-truth route and LiDAR wall hits")
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")
        axis.axis("equal")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(os.path.join(out_dir, "route_and_lidar_walls.png"), dpi=220)
        plt.close(figure)
    except ImportError as error:
        print(f"plot skipped: {error}", file=sys.stderr)

    print(
        f"odom_rows={len(odom_rows)} lidar_scans={len(scan_records)} "
        f"wall_points={len(wall_rows)} path_length_m={path_length:.3f} "
        f"collision_final={summary['collision_final']} lap_final={summary['lap_final']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

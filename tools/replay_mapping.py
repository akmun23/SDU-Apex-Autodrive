#!/usr/bin/env python3
"""Rebuild a map from a sensor bag, replaying only LiDAR and odometry TF.

Run inside the mapping image with the bag, config and output paths mounted.
Recorded map->odom transforms are removed to prevent previous SLAM corrections
from entering a new mapping attempt. Actuator commands are never published.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import OccupancyGrid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=4.0)
    args = parser.parse_args()
    if args.rate <= 0:
        parser.error("--rate must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Node("sensor_mapping_replay")
    pubs = {
        "/tf": node.create_publisher(TFMessage, "/tf", 100),
        "/tf_static": node.create_publisher(TFMessage, "/tf_static", QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
        "/autodrive/roboracer_1/lidar": node.create_publisher(
            LaserScan, "/autodrive/roboracer_1/lidar", 10)}
    types = {"/tf": TFMessage, "/tf_static": TFMessage,
             "/autodrive/roboracer_1/lidar": LaserScan}
    clock_pub = node.create_publisher(Clock, "/clock", 10)
    latest_map = []
    def on_map(msg):
        latest_map[:] = [msg]
    node.create_subscription(OccupancyGrid, "/map", on_map,
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    logfile = (args.output / "slam.log").open("w")
    slam = subprocess.Popen(["ros2", "run", "slam_toolbox", "async_slam_toolbox_node",
        "--ros-args", "--params-file", str(args.config), "-p", "use_sim_time:=true"],
        stdout=logfile, stderr=subprocess.STDOUT)
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(args.bag), storage_id="sqlite3"), ConverterOptions("cdr", "cdr"))
    counts = {k: 0 for k in pubs}
    started = time.monotonic()
    while time.monotonic()-started < 2:
        rclpy.spin_once(node, timeout_sec=0.01)
    base = None
    epoch = time.monotonic()
    last_stamp = 0
    last_progress = -1
    try:
        while reader.has_next():
            topic, data, stamp = reader.read_next()
            if topic not in pubs:
                continue
            if base is None:
                base = stamp
            deadline = epoch+(stamp-base)*1.e-9/args.rate
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=max(0.0, min(0.01, deadline-time.monotonic())))
            clock = Clock()
            clock.clock.sec, clock.clock.nanosec = divmod(stamp, 1_000_000_000)
            clock_pub.publish(clock)
            last_stamp = stamp
            msg = deserialize_message(data, types[topic])
            if topic == "/tf":
                msg.transforms = [tf for tf in msg.transforms
                    if tf.header.frame_id == "odom" and tf.child_frame_id == "base_link"]
                if not msg.transforms:
                    continue
            pubs[topic].publish(msg)
            counts[topic] += 1
            rclpy.spin_once(node, timeout_sec=0)
            progress = int((stamp-base)*1.e-9/20)
            if progress != last_progress:
                print(f"REPLAY {(stamp-base)*1.e-9:.1f}s counts={counts}", flush=True)
                last_progress = progress
        # Let the last scans and occupancy publication finish, advancing the
        # replay clock without generating extra scans or vehicle commands.
        for i in range(100):
            clock = Clock()
            clock.clock.sec, clock.clock.nanosec = divmod(last_stamp+(i+1)*50_000_000, 1_000_000_000)
            clock_pub.publish(clock)
            rclpy.spin_once(node, timeout_sec=0.02)
        if not latest_map:
            raise RuntimeError("Mapper produced no occupancy grid")
        subprocess.run(["ros2", "run", "nav2_map_server", "map_saver_cli",
            "-f", str(args.output / "map"), "--ros-args", "-p", "save_map_timeout:=5.0"], check=True)
        grid = latest_map[0]
        report = dict(messages=counts, elapsed_s=time.monotonic()-epoch,
                      map_size=[grid.info.width,grid.info.height],
                      recorded_slam_tf_replayed=False, actuator_commands_published=0)
        (args.output / "replay.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
    finally:
        slam.terminate()
        try:
            slam.wait(timeout=5)
        except subprocess.TimeoutExpired:
            slam.kill()
            slam.wait()
        logfile.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

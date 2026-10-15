#!/usr/bin/env python3
"""Replay recorded open-plane commands on simulator packet boundaries.

Command values are read from a closed ROS 2 bag and published at the recorded
40 Hz sample count, phase-locked to bridge packet sequence. This is intended
for pinned-player/source-player parity checks, not controller runtime use.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import rosbag2_py
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


STEERING = "/autodrive/roboracer_1/steering_command"
THROTTLE = "/autodrive/roboracer_1/throttle_command"
PACKET_TIMING = "/autodrive/roboracer_1/bridge_packet_timing"
TOPICS = (STEERING, THROTTLE)
PACKET_DT_S = 0.025


def load_commands(path: Path) -> tuple[list[float], list[float]]:
    if not path.is_dir():
        raise ValueError(f"ROS bag directory not found: {path}")
    database = path / "run_0.db3"
    if not database.is_file():
        raise ValueError(f"closed SQLite ROS bag not found: {database}")
    with sqlite3.connect(database) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise ValueError(f"bag database is not closed/healthy: {result}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    message_type = get_message("std_msgs/msg/Float32")
    commands: dict[str, list[tuple[int, float]]] = {topic: [] for topic in TOPICS}
    while reader.has_next():
        topic, serialized, stamp_ns = reader.read_next()
        if topic in commands:
            message = deserialize_message(serialized, message_type)
            value = float(message.data)
            if not (-1.0 <= value <= 1.0):
                raise ValueError(f"out-of-range command {value} on {topic}")
            commands[topic].append((int(stamp_ns), value))
    for topic in TOPICS:
        commands[topic].sort(key=lambda row: row[0])
    steering = commands[STEERING]
    throttle = commands[THROTTLE]
    if not steering or len(steering) != len(throttle):
        raise ValueError("steering and throttle command streams must have equal nonzero lengths")
    if any(b[0] <= a[0] for a, b in zip(steering, steering[1:])):
        raise ValueError("steering command timestamps are duplicated or unordered")
    if any(b[0] <= a[0] for a, b in zip(throttle, throttle[1:])):
        raise ValueError("throttle command timestamps are duplicated or unordered")
    if len(steering) < 100:
        raise ValueError("command bag is too short for a parity run")
    return ([row[1] for row in steering], [row[1] for row in throttle])


class PacketLockedReplay(Node):
    def __init__(self, steering: list[float], throttle: list[float],
                 first_publish_sequence: int) -> None:
        super().__init__("open_plane_command_bag_replay")
        self.steering = steering
        self.throttle = throttle
        self.first_publish_sequence = first_publish_sequence
        self.index = 0
        self.active = False
        self.done = False
        self.last_sequence: int | None = None
        self.first_publish_monotonic_ns: int | None = None
        self.max_callback_late_s = 0.0
        self.steering_pub = self.create_publisher(
            get_message("std_msgs/msg/Float32"), STEERING, 10)
        self.throttle_pub = self.create_publisher(
            get_message("std_msgs/msg/Float32"), THROTTLE, 10)
        self.packet_sub = self.create_subscription(
            get_message("std_msgs/msg/String"), PACKET_TIMING,
            self.on_packet, 10)

    def on_packet(self, message) -> None:
        if self.done:
            return
        try:
            payload = json.loads(message.data)
            sequence = int(payload["packet_sequence"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid packet timing message: {exc}") from exc

        if self.last_sequence is not None and sequence <= self.last_sequence:
            return
        if not self.active:
            if sequence < self.first_publish_sequence:
                self.last_sequence = sequence
                return
            if sequence != self.first_publish_sequence:
                raise RuntimeError(
                    "missed command replay start packet: "
                    f"expected {self.first_publish_sequence}, got {sequence}")
            if self.steering_pub.get_subscription_count() < 1 or \
                    self.throttle_pub.get_subscription_count() < 1:
                raise RuntimeError("bridge command subscribers are not ready")
            self.active = True
            self.first_publish_monotonic_ns = time.monotonic_ns()
            self.get_logger().info(
                f"packet-locked replay starts at simulator sequence {sequence}; "
                f"{len(self.steering)} command pairs")
        else:
            if self.last_sequence is not None and sequence != self.last_sequence + 1:
                raise RuntimeError(
                    f"bridge packet sequence skipped {self.last_sequence}->{sequence} "
                    "during command replay")
            expected_s = self.first_publish_monotonic_ns + int(
                self.index * PACKET_DT_S * 1.0e9)
            late_s = max(0.0, (time.monotonic_ns() - expected_s) / 1.0e9)
            self.max_callback_late_s = max(self.max_callback_late_s, late_s)

        if self.index >= len(self.steering):
            self.done = True
            return
        from std_msgs.msg import Float32
        steering_message = Float32()
        throttle_message = Float32()
        steering_message.data = self.steering[self.index]
        throttle_message.data = self.throttle[self.index]
        self.steering_pub.publish(steering_message)
        self.throttle_pub.publish(throttle_message)
        self.index += 1
        self.last_sequence = sequence
        if self.index == len(self.steering):
            self.done = True
            self.get_logger().info(
                f"replayed {self.index} command pairs; max schedule lateness="
                f"{self.max_callback_late_s * 1000.0:.3f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="closed ROS 2 bag directory")
    parser.add_argument("--first-publish-sequence", type=int, default=86,
                        help="first simulator packet sequence that triggers command 0")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args()
    if args.first_publish_sequence < 0 or args.timeout_s <= 0.0:
        parser.error("sequence must be nonnegative and timeout must be positive")
    try:
        steering, throttle = load_commands(args.bag.resolve())
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"command replay input rejected: {exc}", file=sys.stderr)
        return 2

    rclpy.init()
    node = PacketLockedReplay(steering, throttle, args.first_publish_sequence)
    start_ns = time.monotonic_ns()
    try:
        while rclpy.ok() and not node.done:
            if (time.monotonic_ns() - start_ns) / 1.0e9 > args.timeout_s:
                raise TimeoutError("timed out waiting for command replay to complete")
            rclpy.spin_once(node, timeout_sec=0.05)
    except (RuntimeError, TimeoutError) as exc:
        node.get_logger().error(str(exc))
        return_code = 1
    else:
        return_code = 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

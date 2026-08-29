"""Run an armed controller for a bounded interval with independent guards."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import signal
import time

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, Int32


@dataclass
class Telemetry:
    x: float = math.nan
    y: float = math.nan
    speed_mps: float = math.nan
    front_clearance_m: float = math.nan
    nearest_clearance_m: float = math.nan
    collision_count: int = -1
    lap_count: int = -1
    target_speed_mps: float = math.nan
    throttle_command: float = math.nan
    odom_time: float = 0.0
    scan_time: float = 0.0
    collision_time: float = 0.0
    lap_time: float = 0.0


class GuardedRun(Node):
    """Enable the adapter while enforcing independent runtime stop limits."""

    def __init__(self) -> None:
        super().__init__('autodrive_guarded_run')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.telemetry = Telemetry()
        self._enable_pub = self.create_publisher(
            Bool, '/autodrive/adapter/enable', qos)
        self.create_subscription(
            Odometry, '/autodrive/roboracer_1/odom', self._on_odom, qos)
        self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self._on_scan, qos)
        self.create_subscription(
            Int32,
            '/autodrive/roboracer_1/collision_count',
            self._on_collision,
            qos,
        )
        self.create_subscription(
            Int32,
            '/autodrive/roboracer_1/lap_count',
            self._on_lap,
            qos,
        )
        self.create_subscription(
            AckermannDriveStamped,
            '/cmd/selected',
            self._on_command,
            qos,
        )
        self.create_subscription(
            Float32,
            '/autodrive/roboracer_1/throttle_command',
            self._on_throttle,
            qos,
        )

    def _on_odom(self, message: Odometry) -> None:
        linear = message.twist.twist.linear
        self.telemetry.x = float(message.pose.pose.position.x)
        self.telemetry.y = float(message.pose.pose.position.y)
        self.telemetry.speed_mps = math.hypot(
            float(linear.x), float(linear.y))
        self.telemetry.odom_time = time.monotonic()

    def _on_scan(self, message: LaserScan) -> None:
        front = []
        valid = []
        angle = float(message.angle_min)
        for value in message.ranges:
            value = float(value)
            if math.isfinite(value) and value >= float(message.range_min):
                valid.append(value)
            if (
                abs(angle) <= math.radians(25.0)
                and math.isfinite(value)
                and value >= float(message.range_min)
            ):
                front.append(value)
            angle += float(message.angle_increment)
        self.telemetry.front_clearance_m = min(front) if front else math.inf
        self.telemetry.nearest_clearance_m = min(valid) if valid else math.inf
        self.telemetry.scan_time = time.monotonic()

    def _on_collision(self, message: Int32) -> None:
        self.telemetry.collision_count = int(message.data)
        self.telemetry.collision_time = time.monotonic()

    def _on_lap(self, message: Int32) -> None:
        self.telemetry.lap_count = int(message.data)
        self.telemetry.lap_time = time.monotonic()

    def _on_command(self, message: AckermannDriveStamped) -> None:
        self.telemetry.target_speed_mps = float(message.drive.speed)

    def _on_throttle(self, message: Float32) -> None:
        self.telemetry.throttle_command = float(message.data)

    def set_enabled(self, enabled: bool, repeats: int = 3) -> None:
        for _ in range(repeats):
            self._enable_pub.publish(Bool(data=enabled))
            rclpy.spin_once(self, timeout_sec=0.05)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=15.0)
    parser.add_argument('--maximum-speed', type=float, default=2.75)
    parser.add_argument('--minimum-front-clearance', type=float, default=0.18)
    parser.add_argument('--minimum-any-clearance', type=float, default=0.16)
    parser.add_argument('--sensor-timeout', type=float, default=0.5)
    parser.add_argument('--laps', type=int, default=0)
    return parser.parse_args()


def _state_ready(state: Telemetry) -> bool:
    return (
        math.isfinite(state.x)
        and math.isfinite(state.y)
        and math.isfinite(state.speed_mps)
        and math.isfinite(state.front_clearance_m)
        and math.isfinite(state.nearest_clearance_m)
        and state.collision_count >= 0
        and state.lap_count >= 0
        and math.isfinite(state.target_speed_mps)
        and math.isfinite(state.throttle_command)
    )


def main(args=None) -> None:
    cli = _arguments()
    if cli.duration <= 0.0:
        raise SystemExit('--duration must be positive')

    rclpy.init(args=[] if args is None else args)
    node = GuardedRun()
    stop_requested = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        ready_deadline = time.monotonic() + 5.0
        while time.monotonic() < ready_deadline and not _state_ready(
            node.telemetry
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
        if not _state_ready(node.telemetry):
            raise RuntimeError(
                'missing odometry, LiDAR, or collision telemetry')

        initial = Telemetry(**asdict(node.telemetry))
        target_lap_count = initial.lap_count + cli.laps
        started = time.monotonic()
        next_report = started
        abort_reason = ''
        node.set_enabled(True)

        while not stop_requested and time.monotonic() - started < cli.duration:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            state = node.telemetry
            if state.collision_count > initial.collision_count:
                abort_reason = 'collision count increased'
                break
            if cli.laps > 0 and state.lap_count >= target_lap_count:
                abort_reason = 'target lap count reached'
                break
            if state.speed_mps > cli.maximum_speed:
                abort_reason = 'maximum speed exceeded'
                break
            if (
                state.speed_mps > 0.05
                and state.front_clearance_m < cli.minimum_front_clearance
            ):
                abort_reason = 'minimum front clearance crossed'
                break
            if (
                state.speed_mps > 0.05
                and state.nearest_clearance_m < cli.minimum_any_clearance
            ):
                abort_reason = 'minimum surrounding clearance crossed'
                break
            if (
                now - state.odom_time > cli.sensor_timeout
                or now - state.scan_time > cli.sensor_timeout
                or now - state.collision_time > cli.sensor_timeout
                or now - state.lap_time > cli.sensor_timeout
            ):
                abort_reason = 'telemetry timeout'
                break
            if now >= next_report:
                print(json.dumps({
                    'elapsed_s': round(now - started, 2),
                    'x': round(state.x, 4),
                    'y': round(state.y, 4),
                    'speed_mps': round(state.speed_mps, 4),
                    'target_speed_mps': round(
                        state.target_speed_mps, 4),
                    'throttle_command': round(
                        state.throttle_command, 4),
                    'front_clearance_m': round(
                        state.front_clearance_m, 4),
                    'nearest_clearance_m': round(
                        state.nearest_clearance_m, 4),
                    'collision_count': state.collision_count,
                    'lap_count': state.lap_count,
                }), flush=True)
                next_report = now + 1.0
    finally:
        node.set_enabled(False, repeats=5)
        end = node.telemetry
        distance = (
            math.hypot(end.x - initial.x, end.y - initial.y)
            if 'initial' in locals() else math.nan
        )
        print(json.dumps({
            'event': 'guarded_run_complete',
            'abort_reason': (
                abort_reason
                if 'abort_reason' in locals()
                else 'startup error'),
            'distance_from_start_m': round(distance, 4),
            'final': asdict(end),
        }), flush=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

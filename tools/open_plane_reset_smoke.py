#!/usr/bin/env python3
"""Real-simulator check of the development-only built-in Explore reset path."""

from __future__ import annotations

import json
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool, Float32, Int32, String


ODOM = "/autodrive/roboracer_1/odom"
THROTTLE_FEEDBACK = "/autodrive/roboracer_1/throttle"
STEERING_FEEDBACK = "/autodrive/roboracer_1/steering"
COLLISIONS = "/autodrive/roboracer_1/collision_count"
TIMING_FAULT = "/autodrive/roboracer_1/bridge_timing_fault"
THROTTLE_COMMAND = "/autodrive/roboracer_1/throttle_command"
STEERING_COMMAND = "/autodrive/roboracer_1/steering_command"
RESET_COMMAND = "/autodrive/reset_command"
PHASE = "/open_plane_experiment/phase"
RATE_HZ = 40.0
RESET_HOLD_S = 0.90
RESET_TIMEOUT_S = 5.0
ODOM_TIMEOUT_S = 0.25
COLLISION_TIMEOUT_S = 0.50
SPAWN_POSITION_TOLERANCE_M = 0.25


class ResetSmoke:
    def __init__(self, throttle: float, hold_s: float) -> None:
        self.node = rclpy.create_node("open_plane_reset_smoke")
        self.throttle_pub = self.node.create_publisher(Float32, THROTTLE_COMMAND, 1)
        self.steering_pub = self.node.create_publisher(Float32, STEERING_COMMAND, 1)
        self.reset_pub = self.node.create_publisher(Bool, RESET_COMMAND, 1)
        self.phase_pub = self.node.create_publisher(String, PHASE, 10)
        sensor_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.node.create_subscription(Odometry, ODOM, self._on_odom, sensor_qos)
        self.node.create_subscription(
            Float32, THROTTLE_FEEDBACK, self._on_throttle, sensor_qos)
        self.node.create_subscription(
            Float32, STEERING_FEEDBACK, self._on_steering, sensor_qos)
        self.node.create_subscription(Int32, COLLISIONS, self._on_collision, sensor_qos)
        self.node.create_subscription(Bool, TIMING_FAULT, self._on_timing_fault, 10)
        self.node.create_timer(1.0 / RATE_HZ, self._tick)
        self.started_at = time.monotonic()
        self.probe_throttle = throttle
        self.probe_hold_s = hold_s
        self.state = "waiting"
        self.state_started_at = self.started_at
        self.last_odom_at: float | None = None
        self.last_collision_at: float | None = None
        self.collision_count: int | None = None
        self.speed_mps: float | None = None
        self.position_xy: tuple[float, float] | None = None
        self.spawn_xy: tuple[float, float] | None = None
        self.pre_reset_xy: tuple[float, float] | None = None
        self.pre_reset_speed_mps: float | None = None
        self.max_speed_mps = 0.0
        self.throttle_feedback: float | None = None
        self.steering_feedback: float | None = None
        self.stable_since: float | None = None
        self.odom_count = 0
        self.odom_receipts: list[float] = []
        self.done = False
        self.aborted = False
        self.reason = ""
        self.reset_recovered = False

    def _publish_event(self, payload: dict[str, object]) -> None:
        payload.setdefault("wall_time_ns", time.time_ns())
        self.phase_pub.publish(String(data=json.dumps(
            payload, separators=(",", ":"), sort_keys=True)))

    def _command(self, throttle: float) -> None:
        self.throttle_pub.publish(Float32(data=throttle))
        self.steering_pub.publish(Float32(data=0.0))

    def _on_odom(self, msg: Odometry) -> None:
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        yaw_rate = float(msg.twist.twist.angular.z)
        speed = math.hypot(vx, vy)
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        if not all(map(math.isfinite, (vx, vy, yaw_rate, speed, x, y))):
            return
        now = time.monotonic()
        self.speed_mps = speed
        self.position_xy = (x, y)
        self.max_speed_mps = max(self.max_speed_mps, speed)
        self.last_odom_at = now
        self.odom_count += 1
        self.odom_receipts.append(now)

    def _on_throttle(self, msg: Float32) -> None:
        value = float(msg.data)
        if math.isfinite(value):
            self.throttle_feedback = value

    def _on_steering(self, msg: Float32) -> None:
        value = float(msg.data)
        if math.isfinite(value):
            self.steering_feedback = value

    def _on_collision(self, msg: Int32) -> None:
        self.last_collision_at = time.monotonic()
        self.collision_count = int(msg.data)
        if self.collision_count != 0:
            self._finish(f"collision_count={self.collision_count}", aborted=True)

    def _on_timing_fault(self, msg: Bool) -> None:
        if bool(msg.data):
            self._finish("bridge timing fault", aborted=True)

    def _finish(self, reason: str, *, aborted: bool) -> None:
        if self.done:
            return
        self.done = True
        self.aborted = aborted
        self.reason = reason
        self._command(0.0)
        self.reset_pub.publish(Bool(data=False))
        reset_distance = None
        if self.spawn_xy is not None and self.position_xy is not None:
            reset_distance = math.dist(self.spawn_xy, self.position_xy)
        active_gaps = [
            right - left
            for left, right in zip(self.odom_receipts, self.odom_receipts[1:])
            if 0.0 < right - left <= 0.075
        ]
        active_duration = sum(active_gaps)
        rate_hz = len(active_gaps) / active_duration if active_duration > 0.0 else None
        ordered_gaps = sorted(active_gaps)
        p95_gap_ms = (
            ordered_gaps[math.ceil(0.95 * len(ordered_gaps)) - 1] * 1000.0
            if ordered_gaps else None
        )
        pre_reset_displacement = (
            math.dist(self.spawn_xy, self.pre_reset_xy)
            if self.spawn_xy is not None and self.pre_reset_xy is not None else None
        )
        result = {
            "event": "experiment_end",
            "profile": "throttle_reset_smoke",
            "aborted": aborted,
            "reason": reason,
            "reset_recovered": self.reset_recovered,
            "pre_reset_speed_mps": self.pre_reset_speed_mps,
            "max_speed_mps": self.max_speed_mps,
            "pre_reset_displacement_m": pre_reset_displacement,
            "reset_position_error_m": reset_distance,
            "odom_samples": self.odom_count,
            "odom_active_interval_rate_hz": rate_hz,
            "odom_active_interval_gap_p95_ms": p95_gap_ms,
            "monotonic_ns": time.monotonic_ns(),
        }
        self._publish_event(result)
        self.node.get_logger().info(json.dumps(result, sort_keys=True))

    def _tick(self) -> None:
        if self.done:
            return
        now = time.monotonic()
        if self.state == "waiting" and now - self.started_at > 20.0:
            self._finish("preflight timeout", aborted=True)
            return
        if self.state == "waiting":
            ready = (
                self.last_odom_at is not None
                and self.last_collision_at is not None
                and self.throttle_feedback is not None
                and self.steering_feedback is not None
                and self.throttle_pub.get_subscription_count() > 0
                and self.steering_pub.get_subscription_count() > 0
                and self.reset_pub.get_subscription_count() > 0
            )
            if not ready:
                return
            self.spawn_xy = self.position_xy
            self.state = "drive"
            self.state_started_at = now
            self._publish_event({
                "event": "reset_smoke_start",
                "profile": "throttle_reset_smoke",
                "requested_throttle_norm": self.probe_throttle,
                "probe_hold_s": self.probe_hold_s,
                "speed_mps": self.speed_mps,
                "position_xy": self.spawn_xy,
                "monotonic_ns": time.monotonic_ns(),
            })

        if self.last_odom_at is None or now - self.last_odom_at > ODOM_TIMEOUT_S:
            if self.state not in ("reset_hold", "reset_wait"):
                self._finish("odometry timeout", aborted=True)
                return
        if (self.state not in ("reset_hold", "reset_wait")
                and (self.last_collision_at is None
                     or now - self.last_collision_at > COLLISION_TIMEOUT_S)):
            self._finish("collision telemetry timeout", aborted=True)
            return

        if self.state == "drive":
            if self.speed_mps is not None and self.speed_mps > 0.1:
                self.pre_reset_speed_mps = self.speed_mps
            self._command(self.probe_throttle)
            if now - self.state_started_at >= self.probe_hold_s:
                self.pre_reset_xy = self.position_xy
                self.state = "reset_hold"
                self.state_started_at = now
                self._command(0.0)
                self._publish_event({
                    "event": "sim_reset_start",
                    "profile": "throttle_reset_smoke",
                    "position_xy": self.position_xy,
                    "speed_mps": self.speed_mps,
                    "reset_command": True,
                    "monotonic_ns": time.monotonic_ns(),
                })

        if self.state == "reset_hold":
            self._command(0.0)
            self.reset_pub.publish(Bool(data=True))
            if now - self.state_started_at >= RESET_HOLD_S:
                self.reset_pub.publish(Bool(data=False))
                self.state = "reset_wait"
                self.state_started_at = now
                self.stable_since = None
                self._publish_event({
                    "event": "sim_reset_release",
                    "profile": "throttle_reset_smoke",
                    "reset_command": False,
                    "monotonic_ns": time.monotonic_ns(),
                })

        elif self.state == "reset_wait":
            self._command(0.0)
            if now - self.state_started_at > RESET_TIMEOUT_S:
                self._finish("simulator did not return to spawn after reset", aborted=True)
                return
            distance = (
                math.dist(self.spawn_xy, self.position_xy)
                if self.spawn_xy is not None and self.position_xy is not None else math.inf
            )
            recovered = (
                self.last_odom_at is not None
                and self.last_odom_at >= self.state_started_at
                and self.speed_mps is not None and self.speed_mps <= 0.20
                and distance <= SPAWN_POSITION_TOLERANCE_M
                and self.throttle_feedback is not None
                and abs(self.throttle_feedback) <= 0.02
                and self.steering_feedback is not None
                and abs(self.steering_feedback) <= 0.02
            )
            if recovered:
                if self.stable_since is None:
                    self.stable_since = now
                elif now - self.stable_since >= 0.50:
                    self.reset_recovered = True
                    self._finish("built-in reset returned to stable spawn", aborted=False)
            else:
                self.stable_since = None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throttle", type=float, default=0.30,
                        help="normalized throttle command for the reset probe")
    parser.add_argument("--hold-s", type=float, default=1.0,
                        help="duration before invoking the simulator reset")
    args = parser.parse_args()
    if (not math.isfinite(args.throttle) or not 0.0 <= args.throttle <= 1.0
            or not math.isfinite(args.hold_s) or args.hold_s <= 0.0):
        parser.error("throttle must be within [0,1] and hold duration positive")
    rclpy.init()
    probe = ResetSmoke(args.throttle, args.hold_s)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(probe.node)
    try:
        while rclpy.ok() and not probe.done:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        probe._finish("interrupted", aborted=True)
    finally:
        executor.shutdown()
        probe.node.destroy_node()
        rclpy.shutdown()
    return 1 if probe.aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())

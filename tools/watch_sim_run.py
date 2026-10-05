#!/usr/bin/env python3
"""Stop a development simulation screen on collision or at a lap limit."""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32


class SimRunWatcher(Node):
    def __init__(self) -> None:
        super().__init__("development_sim_run_watcher")
        self.lap_count: int | None = None
        self.collision_count: int | None = None
        self.progress_position: tuple[float, float] | None = None
        self.last_progress_time: float | None = None
        self.create_subscription(
            Int32,
            "/autodrive/roboracer_1/lap_count",
            self._lap_callback,
            10,
        )
        self.create_subscription(
            Int32,
            "/autodrive/roboracer_1/collision_count",
            self._collision_callback,
            10,
        )
        self.create_subscription(Odometry, "/odom", self._odom_callback, 10)

    def _lap_callback(self, message: Int32) -> None:
        if message.data != self.lap_count:
            self.lap_count = message.data
            print(f"lap_count={message.data}", flush=True)

    def _collision_callback(self, message: Int32) -> None:
        if message.data != self.collision_count:
            self.collision_count = message.data
            print(f"collision_count={message.data}", flush=True)

    def _odom_callback(self, message: Odometry) -> None:
        position = message.pose.pose.position
        current = (float(position.x), float(position.y))
        if self.progress_position is None:
            self.progress_position = current
            self.last_progress_time = time.monotonic()
            return
        dx = current[0] - self.progress_position[0]
        dy = current[1] - self.progress_position[1]
        if dx * dx + dy * dy >= 0.10 * 0.10:
            self.progress_position = current
            self.last_progress_time = time.monotonic()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-lap", type=int, default=0)
    parser.add_argument("--collision-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--stall-timeout-s", type=float, default=20.0)
    parser.add_argument("--post-target-lap-guard-s", type=float, default=0.5,
                        help="keep collision monitoring active after the target lap")
    args = parser.parse_args()
    if ((args.target_lap < 1 and not args.collision_only) or
            (args.collision_only and args.target_lap != 0) or
            args.timeout_s <= 0.0 or args.stall_timeout_s <= 0.0 or
            args.post_target_lap_guard_s < 0.0):
        parser.error(
            "set a positive target-lap or collision-only, and a positive timeout-s")

    rclpy.init()
    watcher = SimRunWatcher()
    deadline = time.monotonic() + args.timeout_s
    target_lap_reached_at: float | None = None
    result = 2
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(watcher, timeout_sec=0.02)
            if watcher.collision_count is not None and watcher.collision_count > 0:
                print("SCREEN_ABORT: collision", flush=True)
                result = 3
                break
            if (watcher.last_progress_time is not None and
                    time.monotonic() - watcher.last_progress_time >=
                    args.stall_timeout_s):
                print("SCREEN_ABORT: no odometry progress", flush=True)
                result = 4
                break
            if (args.target_lap > 0 and watcher.lap_count is not None and
                    watcher.lap_count >= args.target_lap):
                if target_lap_reached_at is None:
                    target_lap_reached_at = time.monotonic()
                    print("target lap reached; retaining collision watch", flush=True)
                elif (time.monotonic() - target_lap_reached_at >=
                      args.post_target_lap_guard_s):
                    print("SCREEN_COMPLETE: lap limit", flush=True)
                    result = 0
                    break
        else:
            print("SCREEN_ABORT: timeout", flush=True)
    finally:
        watcher.destroy_node()
        rclpy.shutdown()
    return result


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Stop a development simulation screen on collision or at a lap limit."""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32


class SimRunWatcher(Node):
    def __init__(self) -> None:
        super().__init__("development_sim_run_watcher")
        self.lap_count: int | None = None
        self.collision_count: int | None = None
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

    def _lap_callback(self, message: Int32) -> None:
        if message.data != self.lap_count:
            self.lap_count = message.data
            print(f"lap_count={message.data}", flush=True)

    def _collision_callback(self, message: Int32) -> None:
        if message.data != self.collision_count:
            self.collision_count = message.data
            print(f"collision_count={message.data}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-lap", type=int, default=0)
    parser.add_argument("--collision-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    if ((args.target_lap < 1 and not args.collision_only) or
            (args.collision_only and args.target_lap != 0) or
            args.timeout_s <= 0.0):
        parser.error(
            "set a positive target-lap or collision-only, and a positive timeout-s")

    rclpy.init()
    watcher = SimRunWatcher()
    deadline = time.monotonic() + args.timeout_s
    result = 2
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(watcher, timeout_sec=0.02)
            if watcher.collision_count is not None and watcher.collision_count > 0:
                print("SCREEN_ABORT: collision", flush=True)
                result = 3
                break
            if (args.target_lap > 0 and watcher.lap_count is not None and
                    watcher.lap_count >= args.target_lap):
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

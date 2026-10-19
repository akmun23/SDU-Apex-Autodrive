#!/usr/bin/env python3
"""Stop a development simulation screen on collision or at a lap limit."""

import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, String


class SimRunWatcher(Node):
    def __init__(self, watch_mpc_horizon_test: bool = False) -> None:
        super().__init__("development_sim_run_watcher")
        self.lap_count: int | None = None
        self.collision_count: int | None = None
        self.progress_position: tuple[float, float] | None = None
        self.last_progress_time: float | None = None
        self.truth_speed_mps: float | None = None
        self.mpc_horizon_event: str | None = None
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
        if watch_mpc_horizon_test:
            # This development-only watcher uses simulator truth strictly to
            # keep observing collisions until the open-loop vehicle has
            # physically stopped. It is not part of the controller stack.
            self.create_subscription(
                Odometry,
                "/autodrive/roboracer_1/odom",
                self._truth_odom_callback,
                10,
            )
            self.create_subscription(
                String,
                "/mpc/open_loop_test",
                self._mpc_horizon_callback,
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

    def _truth_odom_callback(self, message: Odometry) -> None:
        twist = message.twist.twist
        self.truth_speed_mps = (float(twist.linear.x) ** 2 +
                                float(twist.linear.y) ** 2) ** 0.5

    def _mpc_horizon_callback(self, message: String) -> None:
        try:
            event = json.loads(message.data).get("event")
        except (TypeError, json.JSONDecodeError):
            return
        if event in ("complete", "abort"):
            self.mpc_horizon_event = event
            print(f"MPC_HORIZON_TEST_{event.upper()}: {message.data}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-lap", type=int, default=0)
    parser.add_argument("--collision-only", action="store_true")
    parser.add_argument("--mpc-horizon-test", action="store_true",
                        help="keep collision watch active until one-shot horizon stops")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--stall-timeout-s", type=float, default=20.0)
    parser.add_argument("--post-target-lap-guard-s", type=float, default=0.5,
                        help="keep collision monitoring active after the target lap")
    args = parser.parse_args()
    if ((args.target_lap < 1 and not args.collision_only and
         not args.mpc_horizon_test) or
            (args.collision_only and args.target_lap != 0) or
            (args.mpc_horizon_test and
             (args.target_lap != 0 or args.collision_only)) or
            args.timeout_s <= 0.0 or args.stall_timeout_s <= 0.0 or
            args.post_target_lap_guard_s < 0.0):
        parser.error(
            "set a positive target-lap or collision-only, and a positive timeout-s")

    rclpy.init()
    watcher = SimRunWatcher(args.mpc_horizon_test)
    print("WATCHER_READY: collision, lap, odometry"
          + (", and fixed-horizon completion" if args.mpc_horizon_test else "")
          + " subscriptions created", flush=True)
    deadline = time.monotonic() + args.timeout_s
    target_lap_reached_at: float | None = None
    mpc_horizon_complete_at: float | None = None
    stationary_since: float | None = None
    result = 2
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(watcher, timeout_sec=0.02)
            if watcher.collision_count is not None and watcher.collision_count > 0:
                print("SCREEN_ABORT: collision", flush=True)
                result = 3
                break
            if watcher.mpc_horizon_event == "abort":
                print("SCREEN_ABORT: MPC fixed-horizon execution aborted", flush=True)
                result = 5
                break
            if (args.mpc_horizon_test and
                    watcher.mpc_horizon_event == "complete"):
                if mpc_horizon_complete_at is None:
                    mpc_horizon_complete_at = time.monotonic()
                    print("MPC horizon complete; retaining collision watch until "
                          "the vehicle stops",
                          flush=True)
                if watcher.truth_speed_mps is not None and \
                        watcher.truth_speed_mps <= 0.20:
                    if stationary_since is None:
                        stationary_since = time.monotonic()
                    elif time.monotonic() - stationary_since >= 0.5:
                        print("SCREEN_COMPLETE: MPC fixed-horizon test; "
                              "vehicle stopped with collision watch active",
                              flush=True)
                        result = 0
                        break
                else:
                    stationary_since = None
                if time.monotonic() - mpc_horizon_complete_at >= 6.0:
                    print("SCREEN_ABORT: vehicle did not stop after fixed-horizon "
                          "test", flush=True)
                    result = 5
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

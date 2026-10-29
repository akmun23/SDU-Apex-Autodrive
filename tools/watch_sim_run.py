#!/usr/bin/env python3
"""Stop a development simulation screen on collision or at a lap limit."""

import argparse
import json
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, String


def lap_target_satisfied(initial_lap_count: int | None,
                         current_lap_count: int | None,
                         target_laps: int,
                         odometry_distance_m: float,
                         minimum_lap_distance_m: float) -> bool:
    if (initial_lap_count is None or current_lap_count is None or
            current_lap_count < initial_lap_count + target_laps):
        return False
    return odometry_distance_m >= target_laps * minimum_lap_distance_m


def stall_watch_armed(watch_mpc_run: bool, mpc_active: bool) -> bool:
    """Gate motion-stall aborts on MPC startup, not localization warm-up."""
    return not watch_mpc_run or mpc_active


class SimRunWatcher(Node):
    def __init__(self, watch_mpc_run: bool = False,
                 watch_csv_command_test: bool = False,
                 watch_csv_path_follow: bool = False) -> None:
        super().__init__("development_sim_run_watcher")
        self.lap_count: int | None = None
        self.initial_lap_count: int | None = None
        self.collision_count: int | None = None
        self.progress_position: tuple[float, float] | None = None
        self.last_odometry_position: tuple[float, float] | None = None
        self.total_odometry_distance_m = 0.0
        self.last_progress_time: float | None = None
        self.mpc_fault_reason: str | None = None
        self.mpc_active = False
        self.csv_command_event: str | None = None
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
        if watch_mpc_run:
            self.create_subscription(
                String,
                "/mpc/diagnostics",
                self._mpc_diagnostics_callback,
                10,
            )
        if watch_csv_command_test or watch_csv_path_follow:
            self.create_subscription(
                String, "/csv_replay/diagnostics",
                self._csv_command_callback, 10)

    def _lap_callback(self, message: Int32) -> None:
        if self.initial_lap_count is None:
            self.initial_lap_count = message.data
            print(f"lap_count_start={message.data}", flush=True)
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
        if self.last_odometry_position is not None:
            step = ((current[0] - self.last_odometry_position[0]) ** 2 +
                    (current[1] - self.last_odometry_position[1]) ** 2) ** 0.5
            # Ignore impossible discontinuities rather than letting one bad
            # localization jump satisfy a multi-lap distance guard.
            if step <= 1.0:
                self.total_odometry_distance_m += step
        self.last_odometry_position = current
        if self.progress_position is None:
            self.progress_position = current
            self.last_progress_time = time.monotonic()
            return
        dx = current[0] - self.progress_position[0]
        dy = current[1] - self.progress_position[1]
        if dx * dx + dy * dy >= 0.10 * 0.10:
            self.progress_position = current
            self.last_progress_time = time.monotonic()

    def _mpc_diagnostics_callback(self, message: String) -> None:
        try:
            diagnostic = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.mpc_fault_reason = "invalid MPC diagnostics JSON"
            print("MPC_DIAGNOSTICS_INVALID_JSON: aborting supervised run",
                  flush=True)
            return
        status = diagnostic.get("status")
        if status in ("waiting_for_localization", "waiting_for_plant_state",
                      "waiting_for_actuator", "waiting_for_solver_budget",
                      "state_stale_wait"):
            return
        if status and status not in (
                "accepted_optimal", "accepted_feasible",
                "accepted_line_search",
                "accepted_candidate_prefix", "accepted_prior_plan",
                "deferred_deadline_recovery"):
            reason = diagnostic.get("nonlinear_failure_reason", "none")
            stage = diagnostic.get("nonlinear_failure_stage", -1)
            self.mpc_fault_reason = f"{status}; nonlinear={reason}@{stage}"
            print(f"MPC_CYCLE_REJECTED: {message.data}", flush=True)
            return
        if status in (
                "accepted_optimal", "accepted_feasible",
                "accepted_line_search", "accepted_candidate_prefix",
                "accepted_prior_plan", "deferred_deadline_recovery"):
            # Localization/plant warm-up may legitimately hold the car still.
            # Start the motion-stall timer when the controller first produces
            # a usable cycle, not when the first stationary odometry arrives.
            if not self.mpc_active:
                self.last_progress_time = time.monotonic()
            self.mpc_active = True

    def _csv_command_callback(self, message: String) -> None:
        try:
            event = json.loads(message.data).get("event")
        except (TypeError, json.JSONDecodeError):
            return
        if event in ("complete", "abort"):
            self.csv_command_event = event
            print(f"CSV_COMMAND_TEST_{event.upper()}: {message.data}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-lap", type=int, default=0)
    parser.add_argument("--collision-only", action="store_true")
    parser.add_argument("--mpc-run", action="store_true",
                        help="abort the simulator on the first rejected MPC cycle")
    parser.add_argument("--csv-command-test", action="store_true",
                        help="watch GT-indexed CSV replay completion or abort")
    parser.add_argument("--csv-path-follow", action="store_true",
                        help="watch follower abort while enforcing target laps")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--stall-timeout-s", type=float, default=20.0)
    parser.add_argument("--mpc-startup-timeout-s", type=float, default=75.,
                        help="maximum time to first accepted MPC cycle, including localization")
    parser.add_argument("--post-target-lap-guard-s", type=float, default=0.5,
                        help="keep collision monitoring active after the target lap")
    parser.add_argument(
        "--minimum-lap-distance-m", type=float, default=0.0,
        help="require this much accumulated odometry travel per counted lap; "
             "prevents stale/erroneous lap counters from ending a run early")
    parser.add_argument(
        "--stop-simulator-container",
        metavar="NAME",
        help="send SIGINT to this simulator Docker container when the watch ends",
    )
    args = parser.parse_args()
    if ((args.target_lap < 1 and not args.collision_only and
         not args.mpc_run and
         not args.csv_command_test) or
            (args.collision_only and args.target_lap != 0) or
            (args.mpc_run and
             (args.collision_only or args.csv_command_test or
              args.csv_path_follow)) or
            (args.csv_command_test and
             (args.target_lap != 0 or args.collision_only)) or
            (args.csv_path_follow and
             (args.target_lap < 1 or args.collision_only or
              args.csv_command_test)) or
            args.timeout_s <= 0.0 or args.stall_timeout_s <= 0.0 or
            args.mpc_startup_timeout_s <= 0.0 or
            args.post_target_lap_guard_s < 0.0 or
            args.minimum_lap_distance_m < 0.0):
        parser.error(
            "set a positive target-lap or collision-only, and a positive timeout-s")

    rclpy.init()
    watcher = SimRunWatcher(
        args.mpc_run, args.csv_command_test, args.csv_path_follow)
    print("WATCHER_READY: collision, lap, odometry"
          + (", and MPC rejection" if args.mpc_run else "")
          + (", and CSV replay completion" if args.csv_command_test else "")
          + (", CSV follower abort, and target-lap completion"
             if args.csv_path_follow else "")
          + " subscriptions created", flush=True)
    started = time.monotonic()
    deadline = started + args.timeout_s
    target_lap_reached_at: float | None = None
    lap_counter_ahead_of_distance = False
    result = 2
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(watcher, timeout_sec=0.02)
            if watcher.collision_count is not None and watcher.collision_count > 0:
                print("SCREEN_ABORT: collision", flush=True)
                result = 3
                break
            if watcher.mpc_fault_reason is not None:
                print(f"SCREEN_ABORT: MPC fault ({watcher.mpc_fault_reason})",
                      flush=True)
                result = 5
                break
            if args.mpc_run and not watcher.mpc_active and (
                    time.monotonic() - started >= args.mpc_startup_timeout_s):
                print("SCREEN_ABORT: MPC startup timeout; no accepted controller "
                      "cycle or localized pose", flush=True)
                result = 5
                break
            if watcher.csv_command_event == "abort":
                print("SCREEN_ABORT: CSV command replay aborted", flush=True)
                result = 5
                break
            if watcher.csv_command_event == "complete":
                if args.csv_path_follow:
                    lap_target_reached = (
                        watcher.initial_lap_count is not None and
                        watcher.lap_count is not None and
                        watcher.lap_count >= watcher.initial_lap_count +
                        args.target_lap)
                    if not lap_target_reached:
                        print("SCREEN_ABORT: CSV route exhausted before target laps",
                              flush=True)
                        result = 5
                        break
                else:
                    print("SCREEN_COMPLETE: CSV command replay lap distance reached",
                          flush=True)
                    result = 0
                    break
            if (not args.collision_only and
                    stall_watch_armed(args.mpc_run, watcher.mpc_active) and
                    watcher.last_progress_time is not None and
                    time.monotonic() - watcher.last_progress_time >=
                    args.stall_timeout_s):
                print("SCREEN_ABORT: no odometry progress", flush=True)
                result = 4
                break
            lap_count_reached = (
                args.target_lap > 0 and watcher.lap_count is not None and
                watcher.initial_lap_count is not None and
                watcher.lap_count >=
                watcher.initial_lap_count + args.target_lap)
            if lap_count_reached:
                distance_required = (
                    args.target_lap * args.minimum_lap_distance_m)
                target_satisfied = lap_target_satisfied(
                    watcher.initial_lap_count, watcher.lap_count,
                    args.target_lap, watcher.total_odometry_distance_m,
                    args.minimum_lap_distance_m)
                if not target_satisfied:
                    if not lap_counter_ahead_of_distance:
                        print(
                            "LAP_COUNT_AHEAD_OF_DISTANCE: "
                            f"counter={watcher.lap_count}/"
                            f"{watcher.initial_lap_count + args.target_lap} "
                            f"travel={watcher.total_odometry_distance_m:.2f}m "
                            f"required={distance_required:.2f}m; continuing",
                            flush=True)
                    lap_counter_ahead_of_distance = True
                    target_lap_reached_at = None
                elif target_lap_reached_at is None:
                    target_lap_reached_at = time.monotonic()
                    lap_counter_ahead_of_distance = False
                    print(
                        f"target laps completed: {args.target_lap}; "
                        f"odometry_travel={watcher.total_odometry_distance_m:.2f}m; "
                        "retaining collision watch", flush=True)
                elif (time.monotonic() - target_lap_reached_at >=
                      args.post_target_lap_guard_s):
                    print("SCREEN_COMPLETE: lap limit", flush=True)
                    result = 0
                    break
            elif not lap_count_reached:
                target_lap_reached_at = None
                lap_counter_ahead_of_distance = False
        else:
            print("SCREEN_ABORT: timeout", flush=True)
    finally:
        watcher.destroy_node()
        rclpy.shutdown()
        if args.stop_simulator_container:
            try:
                stopped = subprocess.run(
                    ["docker", "kill", "--signal=SIGINT",
                     args.stop_simulator_container],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
                if stopped.returncode == 0:
                    print(
                        f"SIMULATOR_STOP_REQUESTED: "
                        f"{args.stop_simulator_container}", flush=True)
                else:
                    detail = (stopped.stderr or stopped.stdout).strip()
                    print(
                        f"SIMULATOR_STOP_FAILED: "
                        f"{args.stop_simulator_container}: {detail}",
                        file=sys.stderr, flush=True)
            except (OSError, subprocess.TimeoutExpired) as error:
                print(
                    f"SIMULATOR_STOP_FAILED: "
                    f"{args.stop_simulator_container}: {error}",
                    file=sys.stderr, flush=True)
    return result


if __name__ == "__main__":
    sys.exit(main())

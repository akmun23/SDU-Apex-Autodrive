"""Development-only GT path follower using a spatial CSV control profile.

Ground truth indexes the saved path continuously. For the native-controls
schema, pure pursuit aligns the car first; then native steering and throttle
are interpolated at the current projected path progress. Leaving the path
returns steering to pure pursuit. The node runs until the end of the available
CSV path or an abort; it does not freeze or stop after a short command horizon.
"""

from __future__ import annotations

import bisect
import csv
from itertools import chain
import json
import math
from pathlib import Path
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sdu_apex_msgs.msg import DriveCommand
from std_msgs.msg import String


GT_TOPIC = "/autodrive/roboracer_1/odom"
COMMAND_TOPIC = "/mpc/drive_command"
DIAGNOSTICS_TOPIC = "/csv_replay/diagnostics"
MAX_STEERING_RAD = 0.5235987756
NATIVE_CONTROL_FULL_WEIGHT_DISTANCE_M = 0.03
NATIVE_CONTROL_ZERO_WEIGHT_DISTANCE_M = 0.10
NATIVE_CONTROL_FULL_WEIGHT_HEADING_RAD = 0.02
NATIVE_CONTROL_ZERO_WEIGHT_HEADING_RAD = 0.10
SPEED_PROFILE_BRAKE_OVERSHOOT_MPS = 0.05


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class CsvCommandReplay(Node):
    """Follow the entire CSV path using GT, PP steering, and CSV throttle."""

    def __init__(self) -> None:
        super().__init__("csv_command_replay")
        self.declare_parameter("csv_file", "")
        self.declare_parameter("path_repeats", 3)
        self.declare_parameter("wheelbase_m", 0.324)
        self.declare_parameter("lookahead_base_m", 0.65)
        self.declare_parameter("lookahead_speed_gain_s", 0.12)
        self.declare_parameter("native_control_delay_s", 0.05)
        self.declare_parameter("max_start_distance_m", 0.50)
        self.declare_parameter("max_start_heading_error_rad", 0.60)
        self.declare_parameter("search_forward_distance_m", 1.50)
        self.declare_parameter("search_back_distance_m", 0.10)
        self.declare_parameter("gt_timeout_s", 0.15)

        csv_file = str(self.get_parameter("csv_file").value).strip()
        if not csv_file:
            raise ValueError("csv_file must be set")
        self.csv_input_format = ""
        reference_rows = self._load_csv(Path(csv_file))
        path_repeats = int(self.get_parameter("path_repeats").value)
        if path_repeats < 1:
            raise ValueError("path_repeats must be at least 1")
        self.reference_row_count = len(reference_rows)
        self.rows = self._repeat_path(reference_rows, path_repeats)
        self.progress = [row["progress_m"] for row in self.rows]
        self.start_progress = self.progress[0]
        self.final_progress = self.progress[-1]
        self.wheelbase = float(self.get_parameter("wheelbase_m").value)
        self.lookahead_base = float(
            self.get_parameter("lookahead_base_m").value)
        self.lookahead_speed_gain = float(
            self.get_parameter("lookahead_speed_gain_s").value)
        self.native_control_delay_s = float(
            self.get_parameter("native_control_delay_s").value)
        self.max_start_distance = float(
            self.get_parameter("max_start_distance_m").value)
        self.max_start_heading_error = float(
            self.get_parameter("max_start_heading_error_rad").value)
        self.search_forward = float(
            self.get_parameter("search_forward_distance_m").value)
        self.search_back = float(
            self.get_parameter("search_back_distance_m").value)
        self.gt_timeout_s = float(self.get_parameter("gt_timeout_s").value)

        positive = (
            self.wheelbase, self.lookahead_base, self.max_start_distance,
            self.max_start_heading_error, self.search_forward,
            self.gt_timeout_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("CSV follower limits must be finite and positive")
        if (not math.isfinite(self.lookahead_speed_gain) or
                self.lookahead_speed_gain < 0.0 or
                not math.isfinite(self.native_control_delay_s) or
                self.native_control_delay_s < 0.0 or
                not math.isfinite(self.search_back) or self.search_back < 0.0):
            raise ValueError("CSV follower projection parameters are invalid")

        self.latest_gt: Odometry | None = None
        self.latest_gt_received_ns: int | None = None
        self.last_progress: float | None = None
        self.last_tick_ns: int | None = None
        self.commands_sent = 0
        self.finished = False

        qos = QoSProfile(depth=10,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.command_pub = self.create_publisher(
            DriveCommand, COMMAND_TOPIC, 10)
        self.diagnostics_pub = self.create_publisher(
            String, DIAGNOSTICS_TOPIC, 10)
        self.gt_sub = self.create_subscription(
            Odometry, GT_TOPIC, self._on_ground_truth, qos)
        # Keep command cadence fixed at 40 Hz despite sensor packet jitter.
        self.timer = self.create_timer(0.025, self._on_tick)
        control_description = (
            "GT-projected native control blended with pure pursuit path "
            f"feedback and {self.native_control_delay_s:.3f}s spatial "
            "delay compensation"
            if self.csv_input_format == "native_controls_raceline_11col" else
            "GT-indexed pure-pursuit steering and CSV throttle")
        self.get_logger().info(
            f"Loaded {self.reference_row_count} CSV points repeated "
            f"{path_repeats} times over "
            f"{self.final_progress - self.start_progress:.3f}m; "
            f"input={self.csv_input_format}; following continuously with "
            f"{control_description} until its final point")

    def _load_csv(self, path: Path) -> list[dict[str, float]]:
        follower_fields = {
            "progress_m", "steering_command_rad", "throttle_command_norm",
            "target_speed_mps", "x_m", "y_m", "yaw_rad", "u_mps", "v_mps",
            "body_yaw_rate_radps",
        }
        with path.open(newline="", encoding="utf-8") as stream:
            first_line = stream.readline()
            # Raceline exports may comment their header with a leading '# '.
            # Feed the normalized header to DictReader without rewriting the
            # supplied file.
            if first_line.startswith("#"):
                first_line = first_line[1:].lstrip()
            reader = csv.DictReader(chain((first_line,), stream))
            fields = set(reader.fieldnames or ())
            native_raceline_fields = {
                "s_m", "x_m", "y_m", "psi_rad", "kappa_radpm",
                "velocity_mps", "acceleration_mps2", "d_left_m",
                "d_right_m", "observed_native_applied_steering_rad",
                "observed_native_applied_throttle_norm",
            }
            if follower_fields.issubset(fields):
                input_format = "csv_command_profile"
                required = follower_fields
            elif native_raceline_fields.issubset(fields):
                input_format = "native_controls_raceline_11col"
                required = native_raceline_fields
            else:
                missing = sorted(follower_fields - fields)
                raise ValueError(
                    "CSV is neither a command-profile CSV nor an 11-column "
                    "native-controls raceline; missing command-profile fields: "
                    + ", ".join(missing))

            rows = []
            for line_number, raw in enumerate(reader, start=2):
                source_row = {name: float(raw[name]) for name in required}
                if not all(math.isfinite(value)
                           for value in source_row.values()):
                    raise ValueError(
                        f"non-finite CSV value at line {line_number}")
                if input_format == "native_controls_raceline_11col":
                    row = {
                        "progress_m": source_row["s_m"],
                        "x_m": source_row["x_m"],
                        "y_m": source_row["y_m"],
                        "yaw_rad": source_row["psi_rad"],
                        "steering_command_rad": source_row[
                            "observed_native_applied_steering_rad"],
                        "throttle_command_norm": source_row[
                            "observed_native_applied_throttle_norm"],
                        "target_speed_mps": source_row["velocity_mps"],
                    }
                else:
                    row = source_row
                if not 0.0 <= row["throttle_command_norm"] <= 1.0:
                    raise ValueError(f"invalid throttle at line {line_number}")
                if abs(row["steering_command_rad"]) > MAX_STEERING_RAD + 1.0e-4:
                    raise ValueError(
                        f"steering exceeds actuator limit at line {line_number}")
                if rows and row["progress_m"] <= rows[-1]["progress_m"]:
                    raise ValueError("CSV progress must increase strictly")
                rows.append(row)
        if len(rows) < 3:
            raise ValueError("CSV has too few command rows")
        # Preserve the selected schema for the startup diagnostic without
        # adding synthetic velocity or yaw-rate labels to the input rows.
        self.csv_input_format = input_format
        return rows

    @staticmethod
    def _repeat_path(rows: list[dict[str, float]],
                     repeats: int) -> list[dict[str, float]]:
        if repeats == 1:
            return rows
        first, last = rows[0], rows[-1]
        closing_distance = math.hypot(
            last["x_m"] - first["x_m"], last["y_m"] - first["y_m"])
        stride = last["progress_m"] - first["progress_m"] + closing_distance
        if not math.isfinite(stride) or stride <= 0.0:
            raise ValueError("CSV path cannot be repeated: invalid closure span")
        repeated = []
        for repeat in range(repeats):
            progress_offset = repeat * stride
            for row in rows:
                copy = row.copy()
                copy["progress_m"] += progress_offset
                repeated.append(copy)
        return repeated

    def _emit(self, values: dict) -> None:
        message = String()
        message.data = json.dumps(values, separators=(",", ":"),
                                  allow_nan=False)
        self.diagnostics_pub.publish(message)

    def _abort(self, reason: str, details: dict) -> None:
        if self.finished:
            return
        self._emit({"event": "abort", "reason": reason, **details})
        self.get_logger().error(f"CSV path follower abort: {reason}: {details}")
        self.finished = True

    def _on_ground_truth(self, message: Odometry) -> None:
        self.latest_gt = message
        self.latest_gt_received_ns = time.monotonic_ns()

    def _project(self, x: float, y: float, yaw: float,
                 initialize: bool) -> tuple[float, int, float, float] | None:
        if initialize:
            first_segment, last_segment = 0, len(self.rows) - 2
            minimum_s, maximum_s = -math.inf, math.inf
        else:
            assert self.last_progress is not None
            first_segment = max(0, bisect.bisect_right(
                self.progress, self.last_progress - self.search_back) - 1)
            last_segment = min(len(self.rows) - 2, bisect.bisect_left(
                self.progress, self.last_progress + self.search_forward))
            minimum_s = self.last_progress - self.search_back
            maximum_s = self.last_progress + self.search_forward

        best: tuple[float, float, int, float, float] | None = None
        for index in range(first_segment, last_segment + 1):
            a, b = self.rows[index], self.rows[index + 1]
            dx, dy = b["x_m"] - a["x_m"], b["y_m"] - a["y_m"]
            length_sq = dx * dx + dy * dy
            if length_sq <= 1.0e-12:
                continue
            fraction = max(0.0, min(1.0,
                ((x - a["x_m"]) * dx + (y - a["y_m"]) * dy) / length_sq))
            s = a["progress_m"] + fraction * (
                b["progress_m"] - a["progress_m"])
            if s < minimum_s or s > maximum_s:
                continue
            px, py = a["x_m"] + fraction * dx, a["y_m"] + fraction * dy
            distance = math.hypot(x - px, y - py)
            tangent_yaw = math.atan2(dy, dx)
            heading_error = abs(wrap_angle(yaw - tangent_yaw))
            score = distance + 0.10 * heading_error
            candidate = (score, s, index, distance, heading_error)
            if best is None or score < best[0]:
                best = candidate
        if best is None:
            return None
        _, s, index, distance, heading_error = best
        if not initialize and self.last_progress is not None:
            s = max(self.last_progress, s)
        return s, index, distance, heading_error

    def _at_progress(
        self, s: float,
    ) -> tuple[int, float, dict[str, float | None]]:
        index = min(len(self.rows) - 2,
                    max(0, bisect.bisect_right(self.progress, s) - 1))
        a, b = self.rows[index], self.rows[index + 1]
        span = b["progress_m"] - a["progress_m"]
        fraction = 0.0 if span <= 0.0 else max(
            0.0, min(1.0, (s - a["progress_m"]) / span))
        fields = (
            "x_m", "y_m", "yaw_rad", "steering_command_rad",
            "throttle_command_norm", "target_speed_mps", "u_mps", "v_mps",
            "body_yaw_rate_radps",
        )
        values = {
            name: (a[name] + fraction * (b[name] - a[name])
                   if name in a and name in b else None)
            for name in fields
        }
        return index, fraction, values

    def _control_for_state(
        self, gt: Odometry, progress: float, path_distance: float,
        path_heading_error: float,
    ) -> tuple[float, float, float | None, float, float, float]:
        pose = gt.pose.pose
        yaw = yaw_from_quaternion(pose.orientation)
        twist = gt.twist.twist
        speed = math.hypot(float(twist.linear.x), float(twist.linear.y))
        lookahead = self.lookahead_base + self.lookahead_speed_gain * speed
        target_progress = min(self.final_progress, progress + lookahead)
        _, _, target = self._at_progress(target_progress)
        def pure_pursuit(x: float, y: float, heading: float) -> float:
            dx = target["x_m"] - x
            dy = target["y_m"] - y
            lateral_body = -math.sin(heading) * dx + math.cos(heading) * dy
            curvature = 2.0 * lateral_body / max(
                lookahead * lookahead, 1.0e-6)
            return math.atan(self.wheelbase * curvature)

        current_pp = pure_pursuit(
            float(pose.position.x), float(pose.position.y), yaw)
        # The recorded native controls describe the command needed at their
        # path station. The command takes two 40 Hz packets to affect the car,
        # so send the profile value for the station expected at application
        # time. GT projection remains the state/index origin; this is only the
        # explicit actuator-delay compensation.
        forward_speed = max(0.0, float(twist.linear.x))
        profile_progress = min(
            self.final_progress,
            progress + forward_speed * self.native_control_delay_s,
        )
        _, _, profile = self._at_progress(profile_progress)
        csv_steering = profile["steering_command_rad"]
        if self.csv_input_format == "native_controls_raceline_11col":
            distance_weight = max(0.0, min(1.0,
                (NATIVE_CONTROL_ZERO_WEIGHT_DISTANCE_M - path_distance) /
                (NATIVE_CONTROL_ZERO_WEIGHT_DISTANCE_M -
                 NATIVE_CONTROL_FULL_WEIGHT_DISTANCE_M)))
            heading_weight = max(0.0, min(1.0,
                (NATIVE_CONTROL_ZERO_WEIGHT_HEADING_RAD - path_heading_error) /
                (NATIVE_CONTROL_ZERO_WEIGHT_HEADING_RAD -
                 NATIVE_CONTROL_FULL_WEIGHT_HEADING_RAD)))
            native_weight = distance_weight * heading_weight
            steering = (native_weight * csv_steering +
                        (1.0 - native_weight) * current_pp)
            native_steering = csv_steering
        else:
            native_weight = 0.0
            steering = current_pp
            native_steering = None

        steering = max(-MAX_STEERING_RAD, min(MAX_STEERING_RAD, steering))
        throttle = profile["throttle_command_norm"]
        speed_error = forward_speed - profile["target_speed_mps"]
        # In this simulator throttle=0 is hard braking, not coasting. The
        # recorded native throttle is a feed-forward profile; apply its drive
        # value only while the measured forward speed is not already above
        # the (delay-aligned) speed target.
        speed_governor_braking = (
            speed_error > SPEED_PROFILE_BRAKE_OVERSHOOT_MPS)
        if speed_governor_braking:
            throttle = 0.0
        return (steering, throttle, native_steering, current_pp, native_weight,
                profile_progress, speed_error, speed_governor_braking)

    def _publish(self, gt: Odometry, steering: float, throttle: float) -> None:
        command = DriveCommand()
        command.header = gt.header
        command.steering_angle_rad = steering
        command.throttle_normalized = throttle
        self.command_pub.publish(command)
        self.commands_sent += 1

    def _status(self, event: str, gt: Odometry, progress: float,
                segment: int, distance: float, heading_error: float,
                tick_interval_ms: float | None, **extra) -> dict:
        pose, twist = gt.pose.pose, gt.twist.twist
        stamp = gt.header.stamp
        return {
            "event": event,
            "gt_stamp_ns": int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
            "gt_age_ms": ((time.monotonic_ns() - self.latest_gt_received_ns) * 1.0e-6
                          if self.latest_gt_received_ns is not None else None),
            "command_timer_interval_ms": tick_interval_ms,
            "csv_index": segment,
            "csv_progress_m": progress,
            "csv_total_progress_m": self.final_progress - self.start_progress,
            "ground_truth_x_m": float(pose.position.x),
            "ground_truth_y_m": float(pose.position.y),
            "ground_truth_yaw_rad": yaw_from_quaternion(pose.orientation),
            "ground_truth_u_mps": float(twist.linear.x),
            "ground_truth_v_mps": float(twist.linear.y),
            "ground_truth_yaw_rate_radps": float(twist.angular.z),
            "commands_sent": self.commands_sent,
            "csv_input_format": self.csv_input_format,
            **extra,
        }

    def _on_tick(self) -> None:
        if self.finished or self.latest_gt is None:
            return
        now_ns = time.monotonic_ns()
        tick_interval_ms = (None if self.last_tick_ns is None else
                            (now_ns - self.last_tick_ns) * 1.0e-6)
        self.last_tick_ns = now_ns
        gt_age_s = ((now_ns - self.latest_gt_received_ns) * 1.0e-9
                    if self.latest_gt_received_ns is not None else math.inf)
        if gt_age_s > self.gt_timeout_s:
            self._abort("ground_truth_stream_stale", {
                "gt_age_ms": gt_age_s * 1000.0,
                "commands_sent": self.commands_sent,
            })
            return

        pose = self.latest_gt.pose.pose
        x, y = float(pose.position.x), float(pose.position.y)
        yaw = yaw_from_quaternion(pose.orientation)
        projected = self._project(x, y, yaw, self.last_progress is None)
        if projected is None:
            self._abort("no_forward_path_projection", {
                "ground_truth_x_m": x,
                "ground_truth_y_m": y,
                "last_progress_m": self.last_progress,
            })
            return
        progress, segment, distance, heading_error = projected

        if self.last_progress is None:
            if (distance > self.max_start_distance or
                    heading_error > self.max_start_heading_error):
                self._abort("csv_start_does_not_match_ground_truth", {
                    "distance_m": distance,
                    "heading_error_rad": heading_error,
                    "ground_truth_x_m": x,
                    "ground_truth_y_m": y,
                    "ground_truth_yaw_rad": yaw,
                })
                return
            self.get_logger().info(
                f"Starting continuous CSV path follow at "
                f"s={progress:.3f}m/{self.final_progress:.3f}m "
                f"(distance={distance:.3f}m, heading error="
                f"{heading_error:.3f}rad)")

        self.last_progress = max(self.last_progress or progress, progress)
        segment, fraction, profile = self._at_progress(self.last_progress)
        if self.last_progress >= self.final_progress - 0.02:
            self._emit(self._status(
                "complete", self.latest_gt, self.last_progress, segment,
                distance, heading_error, tick_interval_ms,
                final_csv_index=len(self.rows) - 1,
                final_command_index=len(self.rows) - 1,
            ))
            self.get_logger().info(
                f"Reached final CSV progress after {self.commands_sent} "
                "40 Hz commands")
            self.finished = True
            return

        (steering, throttle, native_steering, pp_steering, native_weight,
         profile_progress, speed_error, speed_governor_braking) = self._control_for_state(
            self.latest_gt, self.last_progress, distance, heading_error)
        self._publish(self.latest_gt, steering, throttle)
        self._emit(self._status(
            "tracking", self.latest_gt, self.last_progress, segment,
            distance, heading_error, tick_interval_ms,
            csv_fraction=fraction,
            reference_distance_m=distance,
            reference_heading_error_rad=heading_error,
            pp_steering_rad=pp_steering,
            csv_steering_rad=(
                native_steering if native_steering is not None else
                profile["steering_command_rad"]),
            published_steering_rad=steering,
            published_throttle_norm=throttle,
            native_control_blend_weight=native_weight,
            control_profile_progress_m=profile_progress,
            command_delay_compensation_s=self.native_control_delay_s,
            speed_profile_error_mps=speed_error,
            speed_governor_braking=speed_governor_braking,
            csv_target_speed_mps=profile["target_speed_mps"],
            csv_recorded_u_mps=profile.get("u_mps"),
            csv_target_v_mps=profile.get("v_mps"),
            csv_target_yaw_rate_radps=profile.get("body_yaw_rate_radps"),
            csv_native_steering_rad=native_steering,
        ))


def main() -> None:
    rclpy.init()
    node = CsvCommandReplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

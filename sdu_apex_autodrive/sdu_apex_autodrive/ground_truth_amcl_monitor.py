"""Diagnostics-only AMCL, EKF, odometry, and simulator pose comparison.

The simulator ground-truth topics are intentionally diagnostic only.  This
node never publishes a pose, changes AMCL, or feeds a controller.  Messages
are paired by their ROS timestamps so motion between callbacks does not look
like localization error.
"""

import csv
from collections import deque
import math
from pathlib import Path
from typing import Deque, Optional, Tuple

from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node


PoseSample = Tuple[float, float, float, float]


def _yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _rotate(x: float, y: float, angle: float) -> Tuple[float, float]:
    c = math.cos(angle)
    s = math.sin(angle)
    return c * x - s * y, s * x + c * y


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _interpolate(
    samples: Deque[PoseSample], stamp: float, max_gap: float
) -> Optional[Tuple[float, float, float]]:
    """Return x/y/yaw at stamp, using only nearby timestamped samples."""
    if not samples or not math.isfinite(stamp):
        return None

    before = None
    after = None
    for sample in samples:
        if sample[0] <= stamp:
            before = sample
        if sample[0] >= stamp:
            after = sample
            break

    if before is None:
        candidate = samples[0]
        if candidate[0] - stamp > max_gap:
            return None
        return candidate[1], candidate[2], candidate[3]
    if after is None:
        candidate = samples[-1]
        if stamp - candidate[0] > max_gap:
            return None
        return candidate[1], candidate[2], candidate[3]
    if stamp - before[0] > max_gap or after[0] - stamp > max_gap:
        return None
    if after[0] <= before[0] + 1.0e-9:
        return before[1], before[2], before[3]

    ratio = (stamp - before[0]) / (after[0] - before[0])
    x = before[1] + ratio * (after[1] - before[1])
    y = before[2] + ratio * (after[2] - before[2])
    yaw = _wrap(before[3] + ratio * _wrap(after[3] - before[3]))
    return x, y, yaw


class GroundTruthAmclMonitor(Node):
    """Compare AMCL map pose with aligned simulator ground truth."""

    def __init__(self) -> None:
        super().__init__("ground_truth_amcl_monitor")
        self.declare_parameter("ground_truth_topic", "/autodrive/roboracer_1/ips")
        self.declare_parameter("ground_truth_odom_topic", "/autodrive/roboracer_1/odom")
        self.declare_parameter("amcl_topic", "/amcl_pose")
        self.declare_parameter("report_period_sec", 1.0)
        self.declare_parameter("pair_timeout_sec", 0.30)
        self.declare_parameter("output_csv", "")

        gt_topic = str(self.get_parameter("ground_truth_topic").value)
        gt_odom_topic = str(self.get_parameter("ground_truth_odom_topic").value)
        amcl_topic = str(self.get_parameter("amcl_topic").value)
        self.report_period = max(0.1, float(self.get_parameter("report_period_sec").value))
        self.pair_timeout = max(0.05, float(self.get_parameter("pair_timeout_sec").value))
        self.startup_time = self.get_clock().now()

        # Four short histories are enough for interpolation at the native
        # simulator cadence while preventing old poses surviving a reset.
        self.gt_samples: Deque[PoseSample] = deque(maxlen=64)
        self.amcl_samples: Deque[PoseSample] = deque(maxlen=64)
        self.odom_samples: Deque[PoseSample] = deque(maxlen=128)
        self.ekf_samples: Deque[PoseSample] = deque(maxlen=128)
        self.last_gt_position: Optional[Tuple[float, float]] = None
        self.gt_position: Optional[Tuple[float, float]] = None
        self.gt_yaw: Optional[float] = None
        self.gt_speed_mps = math.nan

        self.map_to_world_yaw: Optional[float] = None
        self.map_to_world_translation: Optional[Tuple[float, float]] = None
        # (odom x, odom y, odom yaw, map x, map y, map yaw) at alignment.
        self.odom_map_reference: Optional[Tuple[float, float, float, float, float, float]] = None

        self.sum_sq_xy = 0.0
        self.sum_sq_yaw = 0.0
        self.samples = 0
        self.max_xy = 0.0
        self.max_yaw = 0.0
        self.last_report = None

        output_csv = str(self.get_parameter("output_csv").value)
        self.csv_stream = None
        self.csv_writer = None
        if output_csv:
            path = Path(output_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_stream = path.open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_stream)
            self.csv_writer.writerow((
                "stamp_s", "time_s", "gt_x_m", "gt_y_m", "gt_yaw_rad",
                "amcl_x_m", "amcl_y_m", "amcl_yaw_rad", "amcl_error_m", "amcl_error_rad",
                "ekf_x_m", "ekf_y_m", "ekf_yaw_rad", "ekf_error_m", "ekf_error_rad",
                "odom_x_m", "odom_y_m", "odom_yaw_rad", "odom_error_m", "odom_error_rad",
                "gt_speed_mps",
            ))

        self.create_subscription(Point, gt_topic, self._on_ground_truth, 10)
        self.create_subscription(Odometry, gt_odom_topic, self._on_ground_truth_odom, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, amcl_topic, self._on_amcl, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/ekf_pose", self._on_ekf, 10)
        self.get_logger().warn(
            "Ground-truth monitor is diagnostics-only; simulator IPS/odom are not controller inputs")

    def _reset_alignment(self, reason: str) -> None:
        self.map_to_world_yaw = None
        self.map_to_world_translation = None
        self.odom_map_reference = None
        self.amcl_samples.clear()
        self.odom_samples.clear()
        self.ekf_samples.clear()
        self.sum_sq_xy = 0.0
        self.sum_sq_yaw = 0.0
        self.samples = 0
        self.max_xy = 0.0
        self.max_yaw = 0.0
        self.last_report = None
        self.get_logger().info("%s; waiting for fresh AMCL alignment" % reason)

    def _append(self, history: Deque[PoseSample], sample: PoseSample) -> None:
        if history and sample[0] < history[-1][0]:
            # DDS can deliver a late sample; keep interpolation monotonic.
            return
        history.append(sample)

    def _on_ground_truth(self, msg: Point) -> None:
        # IPS has no Header in the official message.  Keep it as a fallback
        # only; timestamped simulator odometry is the primary source.
        if math.isfinite(msg.x) and math.isfinite(msg.y):
            self.gt_position = (float(msg.x), float(msg.y))

    def _on_ground_truth_odom(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return
        stamp = _stamp_seconds(msg.header.stamp)
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1.0e-9
        if (self.last_gt_position is not None and
                math.hypot(x - self.last_gt_position[0], y - self.last_gt_position[1]) > 2.0):
            self._reset_alignment("Ground-truth jump/reset detected")
            self.gt_samples.clear()
            self.gt_speed_mps = math.nan
        if self.gt_samples:
            previous = self.gt_samples[-1]
            dt = stamp - previous[0]
            distance = math.hypot(x - previous[1], y - previous[2])
            if 1.0e-4 < dt <= 1.0 and math.isfinite(distance):
                self.gt_speed_mps = distance / dt
        self.last_gt_position = (x, y)
        self.gt_position = (x, y)
        self.gt_yaw = yaw
        self._append(self.gt_samples, (stamp, x, y, yaw))

    def _on_amcl(self, msg: PoseWithCovarianceStamped) -> None:
        stamp = _stamp_seconds(msg.header.stamp)
        startup_sec = self.startup_time.nanoseconds * 1.0e-9
        if stamp > 0.0 and stamp + self.pair_timeout < startup_sec:
            # Do not align with an old pose still present in DDS after a
            # workspace restart.
            return
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1.0e-9
        self._append(self.amcl_samples, (stamp, x, y, yaw))
        self._compare_at(stamp)

    def _on_odom(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return
        stamp = _stamp_seconds(msg.header.stamp)
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1.0e-9
        self._append(self.odom_samples, (stamp, x, y, yaw))

    def _on_ekf(self, msg: PoseWithCovarianceStamped) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return
        stamp = _stamp_seconds(msg.header.stamp)
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1.0e-9
        self._append(self.ekf_samples, (stamp, x, y, yaw))

    def _map_pose_from_odom(
        self, odom_pose: Tuple[float, float, float]
    ) -> Optional[Tuple[float, float, float]]:
        if self.odom_map_reference is None:
            return None
        ox0, oy0, oyaw0, mx0, my0, myaw0 = self.odom_map_reference
        ox, oy, oyaw = odom_pose
        dx, dy = _rotate(ox - ox0, oy - oy0, myaw0)
        return mx0 + dx, my0 + dy, _wrap(myaw0 + _wrap(oyaw - oyaw0))

    def _compare_at(self, stamp: float) -> None:
        gt_pose = _interpolate(self.gt_samples, stamp, self.pair_timeout)
        amcl_pose = _interpolate(self.amcl_samples, stamp, self.pair_timeout)
        if gt_pose is None and self.gt_position is not None and self.gt_yaw is not None:
            # Fallback is only used if the simulator odom stream is absent.
            gt_pose = self.gt_position[0], self.gt_position[1], self.gt_yaw
        if gt_pose is None or amcl_pose is None:
            return

        gt_x, gt_y, gt_yaw = gt_pose
        amcl_x, amcl_y, amcl_yaw = amcl_pose
        if self.map_to_world_yaw is None:
            self.map_to_world_yaw = _wrap(gt_yaw - amcl_yaw)
            rx, ry = _rotate(amcl_x, amcl_y, self.map_to_world_yaw)
            self.map_to_world_translation = (gt_x - rx, gt_y - ry)
            odom_pose = _interpolate(self.odom_samples, stamp, self.pair_timeout)
            if odom_pose is not None:
                self.odom_map_reference = (
                    odom_pose[0], odom_pose[1], odom_pose[2],
                    amcl_x, amcl_y, amcl_yaw,
                )
            self.get_logger().info(
                "Ground-truth alignment initialized: map->world yaw=%.3f, translation=(%.3f, %.3f)"
                % (self.map_to_world_yaw, self.map_to_world_translation[0],
                   self.map_to_world_translation[1]))
            return

        tx, ty = self.map_to_world_translation
        dx = gt_x - tx
        dy = gt_y - ty
        expected_x, expected_y = _rotate(dx, dy, -self.map_to_world_yaw)
        expected_yaw = _wrap(gt_yaw - self.map_to_world_yaw)
        error_xy = math.hypot(amcl_x - expected_x, amcl_y - expected_y)
        error_yaw = abs(_wrap(amcl_yaw - expected_yaw))

        ekf_pose = _interpolate(self.ekf_samples, stamp, self.pair_timeout)
        ekf_error_xy = math.nan
        ekf_error_yaw = math.nan
        ekf_report = "ekf=unavailable"
        if ekf_pose is not None:
            ex, ey, eyaw = ekf_pose
            ekf_error_xy = math.hypot(ex - expected_x, ey - expected_y)
            ekf_error_yaw = abs(_wrap(eyaw - expected_yaw))
            ekf_report = "ekf=(%.3f, %.3f, %.3f) error=%.3f m / %.3f rad" % (
                ex, ey, eyaw, ekf_error_xy, ekf_error_yaw)

        odom_pose = _interpolate(self.odom_samples, stamp, self.pair_timeout)
        odom_map_pose = self._map_pose_from_odom(odom_pose) if odom_pose else None
        odom_error_xy = math.nan
        odom_error_yaw = math.nan
        odom_report = "odom=unavailable"
        if odom_map_pose is not None:
            ox, oy, oyaw = odom_map_pose
            odom_error_xy = math.hypot(ox - expected_x, oy - expected_y)
            odom_error_yaw = abs(_wrap(oyaw - expected_yaw))
            odom_report = (
                "odom=(%.3f, %.3f, %.3f) error=%.3f m / %.3f rad"
                % (ox, oy, oyaw, odom_error_xy, odom_error_yaw))

        self.samples += 1
        self.sum_sq_xy += error_xy * error_xy
        self.sum_sq_yaw += error_yaw * error_yaw
        self.max_xy = max(self.max_xy, error_xy)
        self.max_yaw = max(self.max_yaw, error_yaw)

        now = self.get_clock().now()
        should_report = self.last_report is None
        if self.last_report is not None:
            elapsed = (now - self.last_report).nanoseconds / 1.0e9
            should_report = elapsed >= self.report_period
        if should_report:
            self.last_report = now
            rms_xy = math.sqrt(self.sum_sq_xy / self.samples)
            rms_yaw = math.sqrt(self.sum_sq_yaw / self.samples)
            self.get_logger().info(
                "AMCL vs GT: gt=(%.3f, %.3f) amcl=(%.3f, %.3f) error=%.3f m / %.3f rad; %s; %s; "
                "RMS=%.3f m / %.3f rad max=%.3f m / %.3f rad samples=%d"
                % (expected_x, expected_y, amcl_x, amcl_y, error_xy, error_yaw,
                   odom_report, ekf_report, rms_xy, rms_yaw, self.max_xy,
                   self.max_yaw, self.samples))

        if self.csv_writer is not None:
            elapsed = stamp - (self.startup_time.nanoseconds * 1.0e-9)
            ex = ey = etheta = math.nan
            if ekf_pose is not None:
                ex, ey, etheta = ekf_pose
            ox = oy = otheta = math.nan
            if odom_map_pose is not None:
                ox, oy, otheta = odom_map_pose
            self.csv_writer.writerow((
                f"{stamp:.6f}", f"{elapsed:.6f}",
                f"{expected_x:.6f}", f"{expected_y:.6f}", f"{expected_yaw:.6f}",
                f"{amcl_x:.6f}", f"{amcl_y:.6f}", f"{amcl_yaw:.6f}",
                f"{error_xy:.6f}", f"{error_yaw:.6f}",
                f"{ex:.6f}", f"{ey:.6f}", f"{etheta:.6f}",
                f"{ekf_error_xy:.6f}", f"{ekf_error_yaw:.6f}",
                f"{ox:.6f}", f"{oy:.6f}", f"{otheta:.6f}",
                f"{odom_error_xy:.6f}", f"{odom_error_yaw:.6f}",
                f"{self.gt_speed_mps:.6f}",
            ))
            self.csv_stream.flush()

    def destroy_node(self):
        if self.csv_stream is not None and not self.csv_stream.closed:
            self.csv_stream.flush()
            self.csv_stream.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundTruthAmclMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

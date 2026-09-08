"""Mapping-only path-frame pose from simulator ground-truth odometry.

The mapping run may use simulator truth to make traversal deterministic.  This
node estimates one rigid transform between the supplied closed trajectory and
the simulator spawn pose, then publishes the ground-truth pose in the
trajectory's ``map`` frame for the mapping-only Pure Pursuit controller.
It is not included in racing or localization launches.
"""

import csv
import math

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class GroundTruthPathPose(Node):
    """Publish simulator truth expressed in the trajectory coordinate frame."""

    def __init__(self) -> None:
        super().__init__("ground_truth_path_pose")
        self.declare_parameter("ground_truth_topic", "/autodrive/roboracer_1/odom")
        self.declare_parameter("pose_topic", "/current_map_pose")
        self.declare_parameter("trajectory_file", "")
        self.declare_parameter("path_frame", "map")

        trajectory_file = str(self.get_parameter("trajectory_file").value)
        self.path_frame = str(self.get_parameter("path_frame").value)
        self.path_start_x, self.path_start_y, self.path_start_yaw = (
            self._load_start(trajectory_file))
        self._alignment = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            str(self.get_parameter("pose_topic").value),
            qos,
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            str(self.get_parameter("ground_truth_topic").value),
            self._on_odom,
            qos,
        )
        self.get_logger().info(
            f"Mapping-only GT path pose ready: start=({self.path_start_x:.3f}, "
            f"{self.path_start_y:.3f}, {self.path_start_yaw:.3f})")

    @staticmethod
    def _load_start(path: str):
        if not path:
            raise ValueError("trajectory_file is required for GT path pose")
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if not row or row[0].strip().startswith("#"):
                    continue
                if len(row) < 4:
                    continue
                try:
                    return float(row[1]), float(row[2]), float(row[3])
                except ValueError:
                    continue
        raise ValueError(f"trajectory_file contains no valid path point: {path}")

    def _on_odom(self, msg: Odometry) -> None:
        gt_x = float(msg.pose.pose.position.x)
        gt_y = float(msg.pose.pose.position.y)
        gt_yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        if not all(math.isfinite(value) for value in (gt_x, gt_y, gt_yaw)):
            return

        if self._alignment is None:
            delta = _wrap(gt_yaw - self.path_start_yaw)
            cos_delta = math.cos(delta)
            sin_delta = math.sin(delta)
            translated_x = (
                cos_delta * self.path_start_x -
                sin_delta * self.path_start_y)
            translated_y = (
                sin_delta * self.path_start_x +
                cos_delta * self.path_start_y)
            self._alignment = (delta, gt_x - translated_x, gt_y - translated_y)
            self.get_logger().info(
                f"Aligned trajectory to GT spawn: rotation={delta:.3f} rad, "
                f"translation=({self._alignment[1]:.3f}, {self._alignment[2]:.3f})")

        delta, offset_x, offset_y = self._alignment
        cos_delta = math.cos(delta)
        sin_delta = math.sin(delta)
        # Invert the rigid transform path -> ground truth.
        shifted_x = gt_x - offset_x
        shifted_y = gt_y - offset_y
        path_x = cos_delta * shifted_x + sin_delta * shifted_y
        path_y = -sin_delta * shifted_x + cos_delta * shifted_y
        path_yaw = _wrap(gt_yaw - delta)

        pose = PoseWithCovarianceStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.path_frame
        pose.pose.pose.position.x = path_x
        pose.pose.pose.position.y = path_y
        pose.pose.pose.orientation.z = math.sin(path_yaw * 0.5)
        pose.pose.pose.orientation.w = math.cos(path_yaw * 0.5)
        pose.pose.covariance[0] = 1.0e-4
        pose.pose.covariance[7] = 1.0e-4
        pose.pose.covariance[35] = 1.0e-4
        self._pose_pub.publish(pose)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundTruthPathPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

"""Mapping-only ground-truth odometry transform.

The simulator ground-truth odometry is permitted while constructing a map. It
is deliberately isolated in the mapping launch and uses a separate ``gt_odom``
frame, so it cannot overwrite the racing stack's sensor ``odom`` transform or
be consumed by AMCL/controller nodes.
"""

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster


class GroundTruthMappingTf(Node):
    """Publish the simulator pose as a mapping-only odometry TF."""

    def __init__(self) -> None:
        super().__init__("ground_truth_mapping_tf")
        self.declare_parameter("ground_truth_topic", "/autodrive/roboracer_1/odom")
        self.declare_parameter("odom_frame", "gt_odom")
        self.declare_parameter("base_frame", "base_link")

        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tf_broadcaster = TransformBroadcaster(self)
        # The simulator may deliver packets in short bursts. Keep a deeper
        # reliable queue so a burst cannot remove the transform needed by a
        # LiDAR scan before TF consumers can process it.
        qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            Odometry,
            str(self.get_parameter("ground_truth_topic").value),
            self._on_odom,
            qos,
        )
        self.get_logger().info(
            f"Mapping-only ground truth TF: {self.odom_frame} -> {self.base_frame}")

    def _on_odom(self, msg: Odometry) -> None:
        transform = TransformStamped()
        transform.header = msg.header
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundTruthMappingTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

"""Forward the MPC's steering and throttle command to AutoDRIVE."""

import math
import json
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sdu_apex_msgs.msg import DriveCommand
from std_msgs.msg import Float32, String


MIN_DRIVE_THROTTLE = 0.01


class ActuatorInterface(Node):
    def __init__(self, *, parameter_overrides=None) -> None:
        super().__init__("autodrive_actuator_interface",
                         parameter_overrides=parameter_overrides)
        self.declare_parameter("drive_command_topic", "/mpc/drive_command")
        self.declare_parameter(
            "steering_topic", "/autodrive/roboracer_1/steering_command")
        self.declare_parameter(
            "throttle_topic", "/autodrive/roboracer_1/throttle_command")
        self.declare_parameter("development_timing_topic", "")
        self.declare_parameter("max_steering_angle_rad", 0.5235987756)
        self.declare_parameter("minimum_throttle_normalized", MIN_DRIVE_THROTTLE)

        self.max_steering = float(
            self.get_parameter("max_steering_angle_rad").value)
        if not math.isfinite(self.max_steering) or self.max_steering <= 0.0:
            raise ValueError("maximum steering angle must be positive")
        self.minimum_throttle = float(
            self.get_parameter("minimum_throttle_normalized").value)
        if (not math.isfinite(self.minimum_throttle) or
                not 0.0 <= self.minimum_throttle <= 1.0):
            raise ValueError("minimum throttle must be finite and in [0, 1]")
        self.command_sequence = 0
        timing_topic = str(
            self.get_parameter("development_timing_topic").value).strip()

        self.steering_pub = self.create_publisher(
            Float32, self.get_parameter("steering_topic").value, 10)
        self.throttle_pub = self.create_publisher(
            Float32, self.get_parameter("throttle_topic").value, 10)
        self.command_sub = self.create_subscription(
            DriveCommand,
            self.get_parameter("drive_command_topic").value,
            self._on_command,
            1,
        )
        self.timing_pub = self.create_publisher(
            String, timing_topic, 10) if timing_topic else None

    def _on_command(self, message: DriveCommand) -> None:
        received_monotonic_ns = time.monotonic_ns()
        self.command_sequence += 1
        steering = float(message.steering_angle_rad)
        throttle = float(message.throttle_normalized)
        if (not math.isfinite(steering) or not math.isfinite(throttle) or
                not -self.max_steering <= steering <= self.max_steering or
                throttle < self.minimum_throttle - 1.0e-6 or throttle > 1.0):
            self.get_logger().error(
                f"Rejected invalid MPC actuator command: steering={steering:.9g} rad "
                f"(limit={self.max_steering:.9g}), "
                f"throttle={throttle:.9g} "
                f"(range=[{self.minimum_throttle:g},1])")
            self._publish_command_timing(
                message, received_monotonic_ns, "rejected_invalid")
            return
        self._publish(steering / self.max_steering, throttle)
        self._publish_command_timing(
            message, received_monotonic_ns, "forwarded")

    def _publish_command_timing(self, message: DriveCommand,
                                received_monotonic_ns: int,
                                status: str) -> None:
        if self.timing_pub is None:
            return
        published_monotonic_ns = time.monotonic_ns()
        diagnostic = String()
        diagnostic.data = json.dumps({
            "schema_version": 1,
            "command_sequence": self.command_sequence,
            "source_stamp_ns": (
                int(message.header.stamp.sec) * 1_000_000_000 +
                int(message.header.stamp.nanosec)),
            "interface_receive_monotonic_ns": received_monotonic_ns,
            "actuator_publish_monotonic_ns": published_monotonic_ns,
            "status": status,
            "steering_angle_rad": float(message.steering_angle_rad),
            "throttle_normalized": float(message.throttle_normalized),
        }, separators=(",", ":"))
        self.timing_pub.publish(diagnostic)

    def _publish(self, steering: float, throttle: float) -> None:
        self.steering_pub.publish(Float32(data=float(steering)))
        self.throttle_pub.publish(Float32(data=float(throttle)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActuatorInterface()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

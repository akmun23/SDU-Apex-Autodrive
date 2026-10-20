"""Forward the MPC's steering and throttle command to AutoDRIVE."""

import math
import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sdu_apex_msgs.msg import DriveCommand
from std_msgs.msg import Bool, Float32


class ActuatorInterface(Node):
    def __init__(self) -> None:
        super().__init__("autodrive_actuator_interface")
        self.declare_parameter("drive_command_topic", "/mpc/drive_command")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter(
            "steering_topic", "/autodrive/roboracer_1/steering_command")
        self.declare_parameter(
            "throttle_topic", "/autodrive/roboracer_1/throttle_command")
        self.declare_parameter("external_stop_topic", "")
        self.declare_parameter("max_steering_angle_rad", 0.5235987756)
        self.declare_parameter("command_timeout_sec", 0.15)
        self.declare_parameter("odom_timeout_sec", 0.15)
        self.declare_parameter("watchdog_rate_hz", 40.0)

        self.max_steering = float(
            self.get_parameter("max_steering_angle_rad").value)
        self.command_timeout = float(
            self.get_parameter("command_timeout_sec").value)
        self.odom_timeout = float(self.get_parameter("odom_timeout_sec").value)
        watchdog_rate_hz = float(
            self.get_parameter("watchdog_rate_hz").value)
        if (not all(math.isfinite(value) for value in (
                self.max_steering, self.command_timeout,
                self.odom_timeout, watchdog_rate_hz)) or
                self.max_steering <= 0.0 or self.command_timeout <= 0.0 or
                self.odom_timeout <= 0.0 or watchdog_rate_hz <= 0.0):
            raise ValueError("actuator limits, watchdogs, and rate must be positive")

        self.command = None
        self.command_received_at = None
        self.odom_received_at = None
        self.stop_latched = False
        self.last_neutral_reason = None
        self.output_active = False

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
        self.odom_sub = self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value,
            self._on_odom, 1)

        stop_topic = str(self.get_parameter("external_stop_topic").value).strip()
        self.stop_sub = self.create_subscription(
            Bool, stop_topic, self._on_external_stop, 1) if stop_topic else None
        # The timer only enforces command/odometry freshness. Valid MPC output
        # is forwarded from _on_command immediately; a second periodic
        # resampling clock here can make the Bridge send the previous command
        # for one more 40 Hz request.
        self.timer = self.create_timer(1.0 / watchdog_rate_hz, self._tick)

    def _on_command(self, message: DriveCommand) -> None:
        steering = float(message.steering_angle_rad)
        throttle = float(message.throttle_normalized)
        if (not math.isfinite(steering) or not math.isfinite(throttle) or
                not -self.max_steering <= steering <= self.max_steering or
                not 0.0 <= throttle <= 1.0):
            self.get_logger().error(
                f"Rejected invalid MPC actuator command: steering={steering:.9g} rad "
                f"(limit={self.max_steering:.9g}), "
                f"throttle={throttle:.9g} (range=[0,1])")
            return
        self.command = (steering, throttle)
        self.command_received_at = time.monotonic()
        now = self.command_received_at
        if self.stop_latched:
            self._neutral("external stop")
        elif not self._odom_is_fresh(now):
            self._neutral("odometry timeout")
        else:
            self._publish(steering / self.max_steering, throttle)
            self.output_active = True
            self.last_neutral_reason = None

    def _on_odom(self, message: Odometry) -> None:
        twist = message.twist.twist
        values = (twist.linear.x, twist.linear.y, twist.angular.z)
        if not all(math.isfinite(value) for value in values):
            return
        self.odom_received_at = time.monotonic()

    def _on_external_stop(self, message: Bool) -> None:
        if message.data:
            self.stop_latched = True
            self.command = None
            self.command_received_at = None
            self._neutral("external stop")

    def _odom_is_fresh(self, now: float) -> bool:
        return (self.odom_received_at is not None and
                now - self.odom_received_at <= self.odom_timeout)

    def _tick(self) -> None:
        now = time.monotonic()
        if self.stop_latched:
            self._neutral("external stop")
            return
        if self.command is None or self.command_received_at is None:
            self._neutral("no MPC command")
            return
        if now - self.command_received_at > self.command_timeout:
            self._neutral("MPC command timeout")
            return
        if not self._odom_is_fresh(now):
            self._neutral("odometry timeout")
            return

        if not self.output_active:
            steering_angle, throttle = self.command
            self._publish(steering_angle / self.max_steering, throttle)
            self.output_active = True
        self.last_neutral_reason = None

    def _publish(self, steering: float, throttle: float) -> None:
        self.steering_pub.publish(Float32(data=float(steering)))
        self.throttle_pub.publish(Float32(data=float(throttle)))

    def _neutral(self, reason: str) -> None:
        self._publish(0.0, 0.0)
        self.output_active = False
        if reason != self.last_neutral_reason:
            self.get_logger().warning(f"Neutral actuator output: {reason}")
            self.last_neutral_reason = reason


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActuatorInterface()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._publish(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

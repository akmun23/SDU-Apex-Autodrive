"""Safety boundary from Ackermann targets to native AutoDRIVE commands."""

import math
import signal

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Float32

from .mapping import (
    ForwardSpeedController,
    InterfaceConfig,
    SpeedControllerConfig,
    bound_target_speed,
    steering_angle_to_command,
    validate_interface,
    validate_speed_controller,
)


class CommandAdapter(Node):
    """Validate, bound, watchdog, then drive AutoDRIVE's native interface."""

    def __init__(self) -> None:
        super().__init__('autodrive_command_adapter')

        self.declare_parameter('input_topic', '/cmd/selected')
        self.declare_parameter('odom_topic', '/autodrive/roboracer_1/odom')
        self.declare_parameter(
            'steering_topic', '/autodrive/roboracer_1/steering_command')
        self.declare_parameter(
            'throttle_topic', '/autodrive/roboracer_1/throttle_command')
        self.declare_parameter('enable_topic', '/autodrive/adapter/enable')
        self.declare_parameter('armed_topic', '/autodrive/adapter/armed')
        self.declare_parameter('enabled_at_startup', False)
        self.declare_parameter('command_timeout_sec', 0.25)
        self.declare_parameter('odom_timeout_sec', 0.25)
        self.declare_parameter('publish_rate_hz', 20.0)

        # Native 2026 AutoDRIVE RoboRacer interface values.
        self.declare_parameter('max_steering_angle_rad', 0.5236)
        self.declare_parameter('steering_command_min', -1.0)
        self.declare_parameter('steering_command_max', 1.0)
        self.declare_parameter('throttle_command_min', -1.0)
        self.declare_parameter('throttle_command_max', 1.0)
        self.declare_parameter('max_target_speed_mps', 2.5)

        # Longitudinal controller tuning, separate from actuator conversion.
        self.declare_parameter('speed_feedforward_gain', 0.04)
        self.declare_parameter('speed_proportional_gain', 0.02)
        self.declare_parameter('speed_integral_gain', 0.005)
        self.declare_parameter('speed_integral_limit', 1.0)
        self.declare_parameter('maximum_forward_throttle', 0.07)
        self.declare_parameter('throttle_rise_rate_per_sec', 0.10)
        self.declare_parameter('throttle_fall_rate_per_sec', 0.20)
        self.declare_parameter('stop_speed_threshold_mps', 0.01)

        self._interface = InterfaceConfig(
            max_steering_angle_rad=self._float_param('max_steering_angle_rad'),
            steering_command_min=self._float_param('steering_command_min'),
            steering_command_max=self._float_param('steering_command_max'),
            throttle_command_min=self._float_param('throttle_command_min'),
            throttle_command_max=self._float_param('throttle_command_max'),
            max_target_speed_mps=self._float_param('max_target_speed_mps'),
        )
        speed_config = SpeedControllerConfig(
            feedforward_gain=self._float_param('speed_feedforward_gain'),
            proportional_gain=self._float_param('speed_proportional_gain'),
            integral_gain=self._float_param('speed_integral_gain'),
            integral_limit=self._float_param('speed_integral_limit'),
            maximum_forward_throttle=self._float_param(
                'maximum_forward_throttle'),
            throttle_rise_rate_per_sec=self._float_param(
                'throttle_rise_rate_per_sec'),
            throttle_fall_rate_per_sec=self._float_param(
                'throttle_fall_rate_per_sec'),
            stop_speed_threshold_mps=self._float_param(
                'stop_speed_threshold_mps'),
        )
        validate_interface(self._interface)
        validate_speed_controller(speed_config, self._interface)
        self._speed_controller = ForwardSpeedController(
            speed_config, self._interface)

        self._enabled = bool(self.get_parameter('enabled_at_startup').value)
        self._command_timeout_sec = max(
            0.01, self._float_param('command_timeout_sec'))
        self._odom_timeout_sec = max(
            0.01, self._float_param('odom_timeout_sec'))
        publish_rate = max(1.0, self._float_param('publish_rate_hz'))
        self._nominal_dt = 1.0 / publish_rate
        self._last_command: tuple[float, float] | None = None
        self._last_command_time = None
        self._current_speed_mps: float | None = None
        self._last_odom_time = None
        self._last_control_time = None
        self._last_reason = ''

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        status_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._steering_pub = self.create_publisher(
            Float32, self.get_parameter('steering_topic').value, reliable_qos)
        self._throttle_pub = self.create_publisher(
            Float32, self.get_parameter('throttle_topic').value, reliable_qos)
        self._armed_pub = self.create_publisher(
            Bool, self.get_parameter('armed_topic').value, status_qos)
        self._command_sub = self.create_subscription(
            AckermannDriveStamped,
            self.get_parameter('input_topic').value,
            self._on_command,
            reliable_qos,
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._on_odom,
            reliable_qos,
        )
        self._enable_sub = self.create_subscription(
            Bool,
            self.get_parameter('enable_topic').value,
            self._on_enable,
            reliable_qos,
        )
        self._timer = self.create_timer(1.0 / publish_rate, self._on_timer)

        self._publish_armed()
        self._publish_neutral('startup')
        self.get_logger().info(
            'Adapter ready; armed=%s, steering_limit=%.4f rad, input=%s' % (
                self._enabled,
                self._interface.max_steering_angle_rad,
                self.get_parameter('input_topic').value,
            ))

    def _float_param(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _on_command(self, message: AckermannDriveStamped) -> None:
        steering = float(message.drive.steering_angle)
        target_speed = float(message.drive.speed)
        if not math.isfinite(steering) or not math.isfinite(target_speed):
            self.get_logger().error('Rejected non-finite Ackermann command')
            self._last_command = None
            self._last_command_time = None
            return
        self._last_command = (
            steering,
            bound_target_speed(target_speed, self._interface),
        )
        self._last_command_time = self.get_clock().now()

    def _on_odom(self, message: Odometry) -> None:
        vx = float(message.twist.twist.linear.x)
        vy = float(message.twist.twist.linear.y)
        if not math.isfinite(vx) or not math.isfinite(vy):
            self._current_speed_mps = None
            self._last_odom_time = None
            return
        self._current_speed_mps = math.hypot(vx, vy)
        self._last_odom_time = self.get_clock().now()

    def _on_enable(self, message: Bool) -> None:
        self._enabled = bool(message.data)
        self.get_logger().info('Adapter armed=%s' % self._enabled)
        if not self._enabled:
            self._publish_neutral('disarmed')
        self._publish_armed()

    def _on_timer(self) -> None:
        if not self._enabled:
            self._publish_neutral('disarmed')
            return
        if self._last_command is None or self._last_command_time is None:
            self._publish_neutral('no command')
            return
        if self._current_speed_mps is None or self._last_odom_time is None:
            self._publish_neutral('no odometry')
            return

        now = self.get_clock().now()
        command_age = (now - self._last_command_time).nanoseconds / 1e9
        if command_age > self._command_timeout_sec:
            self._publish_neutral('command timeout')
            return
        odom_age = (now - self._last_odom_time).nanoseconds / 1e9
        if odom_age > self._odom_timeout_sec:
            self._publish_neutral('odometry timeout')
            return

        dt = self._nominal_dt
        if self._last_control_time is not None:
            dt = (now - self._last_control_time).nanoseconds / 1e9
            dt = min(max(dt, 1e-3), 0.5)
        self._last_control_time = now

        try:
            steering = steering_angle_to_command(
                self._last_command[0], self._interface)
            throttle = self._speed_controller.update(
                self._last_command[1], self._current_speed_mps, dt)
        except ValueError as error:
            self.get_logger().error('Rejected command: %s' % error)
            self._publish_neutral('invalid command')
            return
        self._publish(steering, throttle)
        self._last_reason = ''

    def _publish(self, steering: float, throttle: float) -> None:
        self._steering_pub.publish(Float32(data=float(steering)))
        self._throttle_pub.publish(Float32(data=float(throttle)))

    def _publish_neutral(self, reason: str) -> None:
        self._speed_controller.reset()
        self._last_control_time = None
        self._publish(0.0, 0.0)
        if (
            reason != self._last_reason
            and reason not in ('startup', 'disarmed')
        ):
            self.get_logger().warn('Neutral output: %s' % reason)
        self._last_reason = reason

    def _publish_armed(self) -> None:
        self._armed_pub.publish(Bool(data=self._enabled))

    def destroy_node(self) -> bool:
        if rclpy.ok(context=self.context):
            self.publish_shutdown_neutral()
        return super().destroy_node()

    def publish_shutdown_neutral(self) -> None:
        """Publish neutral redundantly while the ROS context is still valid."""
        for _ in range(3):
            self._publish_neutral('shutdown')


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = CommandAdapter()
    shutdown_requested = False

    def handle_shutdown(_signum, _frame) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        if rclpy.ok(context=node.context):
            node.publish_shutdown_neutral()
            rclpy.shutdown(context=node.context)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(context=node.context):
            rclpy.shutdown(context=node.context)

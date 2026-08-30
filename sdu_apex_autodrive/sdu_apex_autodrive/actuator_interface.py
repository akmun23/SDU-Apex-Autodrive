"""Always-active watchdog boundary from Ackermann to AutoDRIVE actuators."""

from dataclasses import replace
import math
import signal

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float32

from .actuator_math import (
    ActuatorLimits,
    bound_target_speed,
    steering_angle_to_normalized,
    validate_limits,
)
from .speed_controller import (
    SpeedControllerConfig,
    TargetSpeedController,
    validate_speed_controller,
)
from .watchdog import watchdog_failure


class ActuatorInterface(Node):
    """Drive while inputs are fresh; publish neutral for every invalid state."""

    _DYNAMIC_SPEED_PARAMETERS = {
        'kp',
        'ki',
        'ka',
        'integral_limit',
        'throttle_min_forward',
        'throttle_max_forward',
        'throttle_rise_rate_per_sec',
        'throttle_fall_rate_per_sec',
        'stop_speed_threshold_mps',
    }

    def __init__(self) -> None:
        super().__init__('autodrive_actuator_interface')
        self._declare_parameters()

        self._limits = ActuatorLimits(
            max_steering_angle_rad=self._float_param(
                'max_steering_angle_rad'),
            steering_min=self._float_param('steering_command_min'),
            steering_max=self._float_param('steering_command_max'),
            throttle_min=self._float_param('throttle_command_min'),
            throttle_max=self._float_param('throttle_command_max'),
            max_target_speed_mps=self._float_param('max_target_speed_mps'),
        )
        validate_limits(self._limits)
        speed_config = self._speed_config_from_parameters()
        validate_speed_controller(speed_config)
        if speed_config.throttle_max_forward > self._limits.throttle_max:
            raise ValueError('forward throttle maximum exceeds native limit')
        self._speed_controller = TargetSpeedController(speed_config)

        self._speed_measurement = str(
            self.get_parameter('speed_measurement').value).lower()
        if self._speed_measurement not in ('magnitude', 'forward'):
            raise ValueError('speed_measurement must be magnitude or forward')

        self._command_timeout_sec = max(
            0.01, self._float_param('command_timeout_sec'))
        self._odom_timeout_sec = max(
            0.01, self._float_param('odom_timeout_sec'))
        publish_rate = max(1.0, self._float_param('publish_rate_hz'))
        self._nominal_dt = 1.0 / publish_rate

        self._last_command: tuple[float, float, float] | None = None
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
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._steering_pub = self.create_publisher(
            Float32, self.get_parameter('steering_topic').value, reliable_qos)
        self._throttle_pub = self.create_publisher(
            Float32, self.get_parameter('throttle_topic').value, reliable_qos)
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
            odom_qos,
        )
        self._parameter_callback = self.add_on_set_parameters_callback(
            self._on_parameters)
        self._timer = self.create_timer(1.0 / publish_rate, self._on_timer)

        self._publish_neutral('startup')
        self.get_logger().info(
            'Always-active actuator interface ready; input=%s, watchdogs=%dms' % (
                self.get_parameter('input_topic').value,
                round(1000.0 * min(
                    self._command_timeout_sec, self._odom_timeout_sec)),
            ))

    def _declare_parameters(self) -> None:
        self.declare_parameter('input_topic', '/cmd/controller')
        self.declare_parameter('odom_topic', '/autodrive/roboracer_1/odom')
        self.declare_parameter(
            'steering_topic', '/autodrive/roboracer_1/steering_command')
        self.declare_parameter(
            'throttle_topic', '/autodrive/roboracer_1/throttle_command')
        self.declare_parameter('command_timeout_sec', 0.25)
        self.declare_parameter('odom_timeout_sec', 0.25)
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('max_steering_angle_rad', 0.5236)
        self.declare_parameter('steering_command_min', -1.0)
        self.declare_parameter('steering_command_max', 1.0)
        self.declare_parameter('throttle_command_min', -1.0)
        self.declare_parameter('throttle_command_max', 1.0)
        self.declare_parameter('max_target_speed_mps', 4.0)
        self.declare_parameter('speed_measurement', 'magnitude')
        self.declare_parameter('kp', 0.015)
        self.declare_parameter('ki', 0.003)
        self.declare_parameter('ka', 0.0)
        self.declare_parameter('integral_limit', 1.0)
        self.declare_parameter('throttle_min_forward', 0.0)
        self.declare_parameter('throttle_max_forward', 0.10)
        self.declare_parameter('throttle_rise_rate_per_sec', 0.20)
        self.declare_parameter('throttle_fall_rate_per_sec', 0.50)
        self.declare_parameter('stop_speed_threshold_mps', 0.02)
        # Provisional points derived from bounded track driving. Replace with
        # raw open-world characterization before high-speed tuning.
        self.declare_parameter(
            'feedforward_speed_mps', [0.0, 0.6, 1.0, 1.5, 1.8, 2.5])
        self.declare_parameter(
            'feedforward_throttle', [0.0, 0.025, 0.035, 0.055, 0.065, 0.085])

    def _float_param(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _speed_config_from_parameters(self) -> SpeedControllerConfig:
        return SpeedControllerConfig(
            kp=self._float_param('kp'),
            ki=self._float_param('ki'),
            ka=self._float_param('ka'),
            integral_limit=self._float_param('integral_limit'),
            throttle_min_forward=self._float_param('throttle_min_forward'),
            throttle_max_forward=self._float_param('throttle_max_forward'),
            throttle_rise_rate_per_sec=self._float_param(
                'throttle_rise_rate_per_sec'),
            throttle_fall_rate_per_sec=self._float_param(
                'throttle_fall_rate_per_sec'),
            stop_speed_threshold_mps=self._float_param(
                'stop_speed_threshold_mps'),
            feedforward_speed_mps=tuple(float(value) for value in
                                        self.get_parameter(
                                            'feedforward_speed_mps').value),
            feedforward_throttle=tuple(float(value) for value in
                                       self.get_parameter(
                                           'feedforward_throttle').value),
        )

    def _on_parameters(self, parameters) -> SetParametersResult:
        array_names = {'feedforward_speed_mps', 'feedforward_throttle'}
        if any(parameter.name in array_names for parameter in parameters):
            return SetParametersResult(
                successful=False,
                reason='feedforward table changes require node restart',
            )
        changes = {
            parameter.name: float(parameter.value)
            for parameter in parameters
            if parameter.name in self._DYNAMIC_SPEED_PARAMETERS
        }
        if not changes:
            return SetParametersResult(successful=True)
        try:
            candidate = replace(self._speed_controller.config, **changes)
            validate_speed_controller(candidate)
            if candidate.throttle_max_forward > self._limits.throttle_max:
                raise ValueError('forward throttle maximum exceeds native limit')
        except (TypeError, ValueError) as error:
            return SetParametersResult(successful=False, reason=str(error))
        self._speed_controller.reconfigure(candidate, reset_integral=True)
        self._last_control_time = None
        self.get_logger().info('Speed-controller tuning updated; integral reset')
        return SetParametersResult(successful=True)

    def _on_command(self, message: AckermannDriveStamped) -> None:
        steering = float(message.drive.steering_angle)
        target_speed = float(message.drive.speed)
        target_accel = float(message.drive.acceleration)
        if not all(math.isfinite(value) for value in (
            steering, target_speed, target_accel
        )):
            self.get_logger().error('Rejected non-finite Ackermann command')
            self._last_command = None
            self._last_command_time = None
            return
        self._last_command = (
            steering,
            bound_target_speed(target_speed, self._limits),
            target_accel,
        )
        self._last_command_time = self.get_clock().now()

    def _on_odom(self, message: Odometry) -> None:
        vx = float(message.twist.twist.linear.x)
        vy = float(message.twist.twist.linear.y)
        if not math.isfinite(vx) or not math.isfinite(vy):
            self._current_speed_mps = None
            self._last_odom_time = None
            return
        self._current_speed_mps = (
            math.hypot(vx, vy)
            if self._speed_measurement == 'magnitude'
            else max(0.0, vx)
        )
        self._last_odom_time = self.get_clock().now()

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        command_age = (
            math.inf if self._last_command_time is None
            else (now - self._last_command_time).nanoseconds / 1e9)
        odom_age = (
            math.inf if self._last_odom_time is None
            else (now - self._last_odom_time).nanoseconds / 1e9)
        failure = watchdog_failure(
            has_command=(
                self._last_command is not None
                and self._last_command_time is not None),
            command_age_sec=command_age,
            command_timeout_sec=self._command_timeout_sec,
            has_odometry=(
                self._current_speed_mps is not None
                and self._last_odom_time is not None),
            odom_age_sec=odom_age,
            odom_timeout_sec=self._odom_timeout_sec,
        )
        if failure is not None:
            self._publish_neutral(failure)
            return

        dt = self._nominal_dt
        if self._last_control_time is not None:
            dt = (now - self._last_control_time).nanoseconds / 1e9
            dt = min(max(dt, 1e-3), 0.5)
        self._last_control_time = now

        assert self._last_command is not None
        assert self._current_speed_mps is not None
        try:
            steering = steering_angle_to_normalized(
                self._last_command[0], self._limits)
            throttle = self._speed_controller.update(
                self._last_command[1],
                self._current_speed_mps,
                self._last_command[2],
                dt,
            )
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
        if reason != self._last_reason and reason != 'startup':
            self.get_logger().warn('Neutral output: %s' % reason)
        self._last_reason = reason

    def publish_shutdown_neutral(self) -> None:
        for _ in range(3):
            self._publish_neutral('shutdown')

    def destroy_node(self) -> bool:
        if rclpy.ok(context=self.context):
            self.publish_shutdown_neutral()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ActuatorInterface()
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

"""Team Ackermann commands -> AutoDRIVE normalized actuator channels."""

from dataclasses import replace
import math

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, Int32

from .speed_controller import (
    LongitudinalStateEstimator,
    SpeedControllerConfig,
    TargetSpeedController,
    clamp,
)


class ActuatorInterface(Node):
    TUNABLE = {
        "kp", "ki", "ka", "integral_limit", "throttle_max_forward",
        "throttle_rise_rate_per_sec", "throttle_fall_rate_per_sec",
        "stop_speed_threshold_mps", "overspeed_coast_threshold_mps",
    }

    def __init__(self) -> None:
        super().__init__("autodrive_actuator_interface")
        self._declare_parameters()

        self.max_steering = float(self.get_parameter("max_steering_angle_rad").value)
        self.max_target_speed = float(self.get_parameter("max_target_speed_mps").value)
        self.command_timeout = float(self.get_parameter("command_timeout_sec").value)
        self.odom_timeout = float(self.get_parameter("odom_timeout_sec").value)
        rate = float(self.get_parameter("publish_rate_hz").value)
        if min(self.max_steering, self.max_target_speed, self.command_timeout, self.odom_timeout, rate) <= 0.0:
            raise ValueError("actuator limits, timeouts and rate must be > 0")

        self.nominal_dt = 1.0 / rate
        self.speed_controller = TargetSpeedController(self._speed_config())

        self.command = None
        self.command_time = None
        self.speed = None
        self.odom_time = None
        self.acceleration = 0.0
        self.imu_time = None
        self.speed_estimator = LongitudinalStateEstimator(
            acceleration_filter_alpha=float(
                self.get_parameter("acceleration_filter_alpha").value),
            slip_threshold_mps=float(
                self.get_parameter("slip_threshold_mps").value),
            slip_ratio=float(self.get_parameter("slip_ratio").value),
            odom_correction_gain=float(
                self.get_parameter("odom_correction_gain").value),
        )
        self.control_time = None
        self.last_neutral_reason = None
        self.external_stop_latched = False
        self.collision_count = None
        self.collision_baseline_ready = False
        self.collision_baseline_candidate = None
        self.collision_baseline_candidate_since = None
        self.collision_reset_enabled = bool(
            self.get_parameter("collision_reset_enabled").value)
        self.collision_reset_pulse_sec = float(
            self.get_parameter("collision_reset_pulse_sec").value)
        self.collision_baseline_stable_sec = max(
            0.1, float(self.get_parameter("collision_baseline_stable_sec").value))
        self.reset_release_time = None
        self.reset_release_sent = False

        self.steering_pub = self.create_publisher(
            Float32, self.get_parameter("steering_topic").value, 10)
        self.throttle_pub = self.create_publisher(
            Float32, self.get_parameter("throttle_topic").value, 10)
        self.command_sub = self.create_subscription(
            AckermannDriveStamped, self.get_parameter("input_topic").value,
            self._on_command, 10)
        self.odom_sub = self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value,
            self._on_odom, 10)
        self.imu_sub = self.create_subscription(
            Imu, self.get_parameter("imu_topic").value,
            self._on_imu, rclpy.qos.qos_profile_sensor_data)
        self.reset_pub = None
        self.collision_sub = None
        if self.collision_reset_enabled:
            self.reset_pub = self.create_publisher(
                Bool, self.get_parameter("reset_command_topic").value, 10)
            self.collision_sub = self.create_subscription(
                Int32, self.get_parameter("collision_topic").value,
                self._on_collision_count, 10)
        external_stop_topic = str(self.get_parameter("external_stop_topic").value)
        self.external_stop_sub = None
        if external_stop_topic:
            stop_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.external_stop_sub = self.create_subscription(
                Bool, external_stop_topic, self._on_external_stop, stop_qos)
        self.add_on_set_parameters_callback(self._on_parameters)
        self.timer = self.create_timer(self.nominal_dt, self._tick)

        self._publish(0.0, 0.0)
        self.get_logger().info("Actuator interface ready; no arming state")

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_topic", "/cmd/controller")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("imu_topic", "/autodrive/roboracer_1/imu")
        self.declare_parameter("steering_topic", "/autodrive/roboracer_1/steering_command")
        self.declare_parameter("throttle_topic", "/autodrive/roboracer_1/throttle_command")
        # Mapping completion is not part of the race actuator contract. The
        # mapping launch enables its optional hook explicitly when needed.
        self.declare_parameter("external_stop_topic", "")
        self.declare_parameter("command_timeout_sec", 0.25)
        self.declare_parameter("odom_timeout_sec", 0.25)
        # The official telemetry/control event is approximately 10 Hz. The
        # actuator repeats the last accepted command at that same cadence and
        # retains its timeout watchdog for missing commands.
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("max_steering_angle_rad", 0.5236)
        self.declare_parameter("max_target_speed_mps", 22.88)
        self.declare_parameter("collision_topic", "/autodrive/roboracer_1/collision_count")
        self.declare_parameter("reset_command_topic", "/autodrive/reset_command")
        self.declare_parameter("collision_reset_enabled", False)
        self.declare_parameter("collision_reset_pulse_sec", 0.5)
        # The official counter is cumulative and can be published as zero
        # before the bridge delivers the simulator's existing count. Require a
        # stable observation before treating a later increment as this run's
        # terminal collision.
        self.declare_parameter("collision_baseline_stable_sec", 1.0)

        self.declare_parameter("kp", 0.003)
        self.declare_parameter("ki", 0.0001)
        self.declare_parameter("ka", 0.001)
        self.declare_parameter("integral_limit", 1.0)
        # 1.0 is the AutoDRIVE normalized forward-throttle protocol bound,
        # not a tuning ceiling. The controller determines the actual output.
        self.declare_parameter("throttle_max_forward", 1.0)
        self.declare_parameter("throttle_rise_rate_per_sec", 0.90)
        self.declare_parameter("throttle_fall_rate_per_sec", 4.0)
        self.declare_parameter("stop_speed_threshold_mps", 0.02)
        # Small overshoots must be corrected with the calibrated throttle
        # feedback.  A hard coast at 0.25 m/s causes large oscillations at
        # low speed because passive simulator deceleration is steep.  Reserve
        # forced coasting for a materially large overspeed.
        self.declare_parameter("overspeed_coast_threshold_mps", 1.0)
        # Allowed-input longitudinal observer.  It rejects encoder wheel-spin
        # when the IMU-integrated body speed disagrees materially.
        self.declare_parameter("acceleration_filter_alpha", 0.35)
        self.declare_parameter("slip_threshold_mps", 0.75)
        self.declare_parameter("slip_ratio", 0.20)
        self.declare_parameter("odom_correction_gain", 0.25)
        self.declare_parameter("speed_error_to_accel_gain", 1.25)
        self.declare_parameter("speed_error_integral_to_accel_gain", 0.05)
        self.declare_parameter("max_acceleration_mps2", 6.0)
        self.declare_parameter("max_deceleration_mps2", 8.0)
        # IMU acceleration is useful as a secondary signal, but isolated
        # simulator samples can spike at the native 10 Hz cadence. Let the
        # calibrated speed/acceleration feed-forward and odom feedback remain
        # dominant instead of cutting throttle on one such spike.
        self.declare_parameter("acceleration_feedback_gain", 0.05)
        self.declare_parameter("acceleration_integral_gain", 0.003)
        self.declare_parameter("acceleration_integral_limit", 2.0)
        self.declare_parameter(
            "acceleration_speed_mps",
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0,
             20.0, 23.0])
        self.declare_parameter(
            "acceleration_throttle_per_mps2",
            [0.158521, 0.160881, 0.163312, 0.165818, 0.168402, 0.171068,
             0.173819, 0.176661, 0.179597, 0.182632, 0.185771, 0.190688])
        self.declare_parameter(
            "feedforward_speed_mps",
            [0.0, 0.5016, 1.2497, 1.9930, 2.4859, 2.9768,
             3.7096, 4.4382, 4.9216, 5.6440, 6.3621, 7.3135,
             8.4935, 9.6633, 11.9725, 14.2418, 16.4701, 18.6555,
             22.8836])
        self.declare_parameter(
            "feedforward_throttle",
            [0.0, 0.020, 0.050, 0.080, 0.100, 0.120, 0.150,
             0.180, 0.200, 0.230, 0.260, 0.300, 0.350, 0.400,
             0.500, 0.600, 0.700, 0.800, 1.000])

    def _speed_config(self) -> SpeedControllerConfig:
        return SpeedControllerConfig(
            kp=float(self.get_parameter("kp").value),
            ki=float(self.get_parameter("ki").value),
            ka=float(self.get_parameter("ka").value),
            integral_limit=float(self.get_parameter("integral_limit").value),
            throttle_max_forward=float(self.get_parameter("throttle_max_forward").value),
            throttle_rise_rate_per_sec=float(self.get_parameter("throttle_rise_rate_per_sec").value),
            throttle_fall_rate_per_sec=float(self.get_parameter("throttle_fall_rate_per_sec").value),
            stop_speed_threshold_mps=float(self.get_parameter("stop_speed_threshold_mps").value),
            overspeed_coast_threshold_mps=float(
                self.get_parameter("overspeed_coast_threshold_mps").value),
            feedforward_speed_mps=tuple(float(v) for v in self.get_parameter("feedforward_speed_mps").value),
            feedforward_throttle=tuple(float(v) for v in self.get_parameter("feedforward_throttle").value),
            speed_error_to_accel_gain=float(
                self.get_parameter("speed_error_to_accel_gain").value),
            speed_error_integral_to_accel_gain=float(
                self.get_parameter("speed_error_integral_to_accel_gain").value),
            max_acceleration_mps2=float(
                self.get_parameter("max_acceleration_mps2").value),
            max_deceleration_mps2=float(
                self.get_parameter("max_deceleration_mps2").value),
            acceleration_feedback_gain=float(
                self.get_parameter("acceleration_feedback_gain").value),
            acceleration_integral_gain=float(
                self.get_parameter("acceleration_integral_gain").value),
            acceleration_integral_limit=float(
                self.get_parameter("acceleration_integral_limit").value),
            acceleration_speed_mps=tuple(float(v) for v in self.get_parameter(
                "acceleration_speed_mps").value),
            acceleration_throttle_per_mps2=tuple(float(v) for v in self.get_parameter(
                "acceleration_throttle_per_mps2").value),
        )

    def _on_parameters(self, parameters) -> SetParametersResult:
        if any(p.name in {"feedforward_speed_mps", "feedforward_throttle"} for p in parameters):
            return SetParametersResult(
                successful=False, reason="feedforward table changes require restart")
        changes = {p.name: float(p.value) for p in parameters if p.name in self.TUNABLE}
        if not changes:
            return SetParametersResult(successful=True)
        try:
            candidate = replace(self.speed_controller.config, **changes)
            candidate.validate()
            self.speed_controller.reconfigure(candidate)
        except (TypeError, ValueError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))
        self.control_time = None
        return SetParametersResult(successful=True)

    def _on_command(self, msg: AckermannDriveStamped) -> None:
        steering = float(msg.drive.steering_angle)
        speed = float(msg.drive.speed)
        accel = float(msg.drive.acceleration)
        if not all(math.isfinite(v) for v in (steering, speed, accel)):
            self.command = None
            self.command_time = None
            return
        self.command = (
            clamp(steering, -self.max_steering, self.max_steering),
            clamp(speed, 0.0, self.max_target_speed),
            accel,
        )
        self.command_time = self.get_clock().now()

    def _on_odom(self, msg: Odometry) -> None:
        speed = float(msg.twist.twist.linear.x)
        if not math.isfinite(speed):
            self.speed = None
            self.odom_time = None
            return
        now = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9
        self.speed = self.speed_estimator.update_odometry(speed, now_sec)
        self.acceleration = self.speed_estimator.acceleration_mps2
        self.odom_time = now

    def _on_imu(self, msg: Imu) -> None:
        acceleration = float(msg.linear_acceleration.x)
        if not math.isfinite(acceleration):
            return
        now = self.get_clock().now()
        self.speed_estimator.update_acceleration(
            acceleration, now.nanoseconds / 1e9)
        self.speed = self.speed_estimator.speed_mps
        self.acceleration = self.speed_estimator.acceleration_mps2
        self.imu_time = now

    def _on_collision_count(self, msg: Int32) -> None:
        """Latch a terminal failure and reset the simulator vehicle.

        This subscription is intentionally isolated to the actuator safety
        layer. ``collision_count`` and ``reset_command`` are simulator-only
        diagnostics/control and must not be used by competition controller
        logic.
        """
        count = max(0, int(msg.data))
        now_sec = self.get_clock().now().nanoseconds / 1e9
        previous = self.collision_count
        self.collision_count = count
        # The simulator keeps collision_count cumulative across vehicle
        # resets and bridge sessions. The bridge can briefly publish its
        # default zero before the first actual simulator sample, so a simple
        # previous-value comparison would mistake an old collision for a new
        # one. Wait for one stable count, then only a subsequent increment is
        # considered a collision in this actuator run.
        if not self.collision_baseline_ready:
            if self.collision_baseline_candidate != count:
                self.collision_baseline_candidate = count
                self.collision_baseline_candidate_since = now_sec
                return
            if self.collision_baseline_candidate_since is None:
                self.collision_baseline_candidate_since = now_sec
                return
            if now_sec - self.collision_baseline_candidate_since >= self.collision_baseline_stable_sec:
                self.collision_baseline_ready = True
                self.get_logger().info(
                    f"Collision baseline established at cumulative count {count}")
            return

        # A simulator/bridge reset may make the cumulative counter decrease.
        # Re-baseline that explicit external state instead of treating a later
        # count as an artificial increment.
        if previous is not None and count < previous:
            self.collision_baseline_ready = False
            self.collision_baseline_candidate = count
            self.collision_baseline_candidate_since = now_sec
            return

        collision_detected = previous is not None and count > previous
        if collision_detected and not self.external_stop_latched:
            self._latch_collision(count)

    def _latch_collision(self, count: int) -> None:
        self.external_stop_latched = True
        self.command = None
        self.command_time = None
        self.speed_controller.reset()
        self.speed_estimator.reset()
        self.speed = 0.0
        self.acceleration = 0.0
        self.control_time = None
        if self.reset_pub is not None:
            now = self.get_clock().now()
            self.reset_release_time = now.nanoseconds / 1e9 + self.collision_reset_pulse_sec
            self.reset_release_sent = False
            self.reset_pub.publish(Bool(data=True))
        self.get_logger().error(
            f"Collision count increased to {count}; terminal stop latched and simulator reset requested")

    def _service_reset_pulse(self, now) -> None:
        if self.reset_pub is None or self.reset_release_time is None:
            return
        now_sec = now.nanoseconds / 1e9
        if now_sec < self.reset_release_time:
            self.reset_pub.publish(Bool(data=True))
            return
        if not self.reset_release_sent:
            self.reset_pub.publish(Bool(data=False))
            self.reset_release_sent = True
            self.reset_release_time = None
            self.get_logger().error(
                "Simulator reset pulse completed; actuator remains latched neutral after collision")

    def _on_external_stop(self, msg: Bool) -> None:
        if not msg.data or self.external_stop_latched:
            return
        self.external_stop_latched = True
        self.command = None
        self.command_time = None
        self.get_logger().warn(
            "External stop latched; restarting the actuator node is required to drive again")

    def _tick(self) -> None:
        now = self.get_clock().now()
        self._service_reset_pulse(now)
        if self.external_stop_latched:
            return self._neutral("external stop")
        if self.command is None or self.command_time is None:
            return self._neutral("no command")
        if self.speed is None or self.odom_time is None:
            return self._neutral("no odometry")
        if (now - self.command_time).nanoseconds / 1e9 > self.command_timeout:
            return self._neutral("command timeout")
        if (now - self.odom_time).nanoseconds / 1e9 > self.odom_timeout:
            return self._neutral("odometry timeout")

        dt = self.nominal_dt
        if self.control_time is not None:
            dt = clamp((now - self.control_time).nanoseconds / 1e9, 1e-3, 0.5)
        self.control_time = now

        steering_angle, target_speed, target_accel = self.command
        steering = clamp(steering_angle / self.max_steering, -1.0, 1.0)
        throttle = self.speed_controller.update(
            target_speed, self.speed, target_accel, dt, self.acceleration)
        self._publish(steering, throttle)
        self.last_neutral_reason = None

    def _publish(self, steering: float, throttle: float) -> None:
        self.steering_pub.publish(Float32(data=float(steering)))
        self.throttle_pub.publish(Float32(data=float(throttle)))

    def _neutral(self, reason: str) -> None:
        self.speed_controller.reset()
        self.control_time = None
        self._publish(0.0, 0.0)
        if reason != self.last_neutral_reason:
            self.get_logger().warn(f"Neutral output: {reason}")
            self.last_neutral_reason = reason


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActuatorInterface()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.reset_pub is not None and node.reset_release_time is not None:
            node.reset_pub.publish(Bool(data=False))
        if rclpy.ok():
            node._publish(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

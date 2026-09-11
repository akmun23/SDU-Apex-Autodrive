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
        "speed_hold_error_deadband_mps",
        "speed_hold_recovery_error_mps",
        "speed_hold_acceleration_deadband_mps2",
        "speed_boost_error_mps", "speed_hold_prediction_horizon_sec",
        "speed_hold_entry_margin_mps", "speed_downshift_stable_sec",
        "speed_downshift_band_mps", "speed_overspeed_confirmation_sec",
        "hard_overspeed_cutoff_mps",
        "speed_error_to_accel_gain", "speed_error_integral_to_accel_gain",
        "acceleration_feedback_gain", "acceleration_integral_gain",
        "acceleration_integral_limit",
        "acceleration_throttle_rise_rate_per_sec",
        "acceleration_throttle_fall_rate_per_sec",
    }

    def __init__(self) -> None:
        super().__init__("autodrive_actuator_interface")
        self._declare_parameters()

        self.max_steering = float(self.get_parameter("max_steering_angle_rad").value)
        self.max_target_speed = float(self.get_parameter("max_target_speed_mps").value)
        self.command_timeout = float(self.get_parameter("command_timeout_sec").value)
        self.odom_timeout = float(self.get_parameter("odom_timeout_sec").value)
        self.max_feedback_speed = float(
            self.get_parameter("max_feedback_speed_mps").value)
        rate = float(self.get_parameter("publish_rate_hz").value)
        if min(
            self.max_steering, self.max_target_speed, self.command_timeout,
            self.odom_timeout, self.max_feedback_speed, rate,
        ) <= 0.0:
            raise ValueError("actuator limits, timeouts and rate must be > 0")
        if self.max_feedback_speed < self.max_target_speed:
            raise ValueError("max_feedback_speed_mps must cover max_target_speed_mps")

        self.nominal_dt = 1.0 / rate
        self.speed_controller = TargetSpeedController(self._speed_config())

        self.command = None
        self.command_time = None
        self.raw_throttle_override = None
        self.raw_throttle_override_time = None
        self.speed = None
        # The conditioned speed drives ordinary feedback. Keep the latest
        # accepted raw odometry sample for the hard overspeed interlock so a
        # low-pass filter cannot hide a large target crossing.
        self.raw_speed = None
        # Source time drives derivatives and controller freshness. Arrival
        # time is kept separately for the transport watchdog: callback jitter
        # must not change the physical dt used by the observer, and a
        # duplicate source sample must not be treated as new control state.
        self.odom_arrival_time = None
        self.odom_source_stamp_ns = None
        self.last_controller_odom_source_stamp_ns = None
        # Compatibility alias for older diagnostics; runtime freshness uses
        # odom_arrival_time explicitly.
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
            speed_measurement_filter_alpha=float(
                self.get_parameter("speed_measurement_filter_alpha").value),
        )
        self.control_time = None
        self.last_controller_odom_time = None
        self.last_controller_odom_source_stamp_ns = None
        self.last_neutral_reason = None
        self.external_stop_latched = False
        self.collision_count = None
        self.collision_baseline_ready = False
        self.collision_baseline_candidate = None
        self.collision_baseline_candidate_since = None
        self.collision_reset_enabled = bool(
            self.get_parameter("collision_reset_enabled").value)
        self.collision_terminal_stop = bool(
            self.get_parameter("collision_terminal_stop").value)
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
        self.raw_throttle_override_sub = None
        if bool(self.get_parameter("allow_raw_throttle_override").value):
            self.raw_throttle_override_sub = self.create_subscription(
                Float32,
                self.get_parameter("raw_throttle_override_topic").value,
                self._on_raw_throttle_override,
                10,
            )
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
        # Monitor collisions independently from reset handling. Mapping runs
        # must abort on a crash without publishing a simulator reset command.
        if self.collision_reset_enabled or self.collision_terminal_stop:
            self.collision_sub = self.create_subscription(
                Int32, self.get_parameter("collision_topic").value,
                self._on_collision_count, 10)
        external_stop_topic = str(self.get_parameter("external_stop_topic").value)
        self.external_stop_sub = None
        if external_stop_topic:
            stop_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                # Completion is a current-run event. Do not replay a
                # transient ``true`` from an earlier mapping session into a
                # freshly started actuator; the actuator still latches any
                # true event received during this process lifetime.
                durability=DurabilityPolicy.VOLATILE,
            )
            self.external_stop_sub = self.create_subscription(
                Bool, external_stop_topic, self._on_external_stop, stop_qos)
        self.add_on_set_parameters_callback(self._on_parameters)
        self.timer = self.create_timer(self.nominal_dt, self._tick)

        self._publish(0.0, 0.0)
        self.get_logger().info("Actuator interface ready; no arming state")

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_topic", "/cmd/speed")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("imu_topic", "/autodrive/roboracer_1/imu")
        self.declare_parameter("steering_topic", "/autodrive/roboracer_1/steering_command")
        self.declare_parameter("throttle_topic", "/autodrive/roboracer_1/throttle_command")
        # Diagnostics-only escape hatch for calibration phases that must
        # apply raw zero throttle. Production controllers never enable this:
        # acceleration=0 is a valid hold-acceleration request, not neutral.
        self.declare_parameter("allow_raw_throttle_override", False)
        self.declare_parameter(
            "raw_throttle_override_topic",
            "/autodrive/roboracer_1/raw_throttle_override",
        )
        # Mapping completion is not part of the race actuator contract. The
        # mapping launch enables its optional hook explicitly when needed.
        self.declare_parameter("external_stop_topic", "")
        self.declare_parameter("command_timeout_sec", 0.25)
        # Tolerate a few missed 40 Hz source periods, but do not allow a
        # stale speed estimate to drive for the old 300 ms window.
        self.declare_parameter("odom_timeout_sec", 0.125)
        self.declare_parameter("max_feedback_speed_mps", 30.0)
        # Match the accepted native simulator source cadence. The timeout
        # watchdog still neutralizes the outputs if commands or odometry stop.
        self.declare_parameter("publish_rate_hz", 40.0)
        self.declare_parameter("max_steering_angle_rad", 0.5236)
        self.declare_parameter("max_target_speed_mps", 22.88)
        self.declare_parameter("collision_topic", "/autodrive/roboracer_1/collision_count")
        self.declare_parameter("reset_command_topic", "/autodrive/reset_command")
        self.declare_parameter("collision_reset_enabled", False)
        self.declare_parameter("collision_reset_pulse_sec", 0.5)
        # Collision stopping is terminal. Reset handling is a separate opt-in
        # hook and is not required for collision monitoring.
        self.declare_parameter("collision_terminal_stop", True)
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
        # Once target speed and acceleration are settled, hold the calibrated
        # feed-forward throttle until either deadband is left.
        self.declare_parameter("speed_hold_error_deadband_mps", 0.25)
        self.declare_parameter("speed_hold_recovery_error_mps", 0.25)
        self.declare_parameter("speed_hold_acceleration_deadband_mps2", 0.35)
        # Far below target, use the calibrated acceleration envelope. Handoff
        # is predictive so the vehicle reaches the target without a large
        # overshoot, then the target-speed feed-forward value is held.
        self.declare_parameter("speed_boost_error_mps", 1.5)
        self.declare_parameter("speed_hold_prediction_horizon_sec", 0.25)
        self.declare_parameter("speed_hold_entry_margin_mps", 0.15)
        self.declare_parameter("speed_downshift_stable_sec", 0.20)
        self.declare_parameter("speed_downshift_band_mps", 0.10)
        self.declare_parameter("speed_overspeed_confirmation_sec", 0.30)
        self.declare_parameter("hard_overspeed_cutoff_mps", 0.50)
        # Allowed-input longitudinal observer.  It rejects encoder wheel-spin
        # when the IMU-integrated body speed disagrees materially.
        self.declare_parameter("acceleration_filter_alpha", 0.20)
        self.declare_parameter("slip_threshold_mps", 0.75)
        self.declare_parameter("slip_ratio", 0.20)
        self.declare_parameter("odom_correction_gain", 0.25)
        # Condition only the speed signal consumed by the actuator loop. The
        # estimator's published /odom topic remains the calibrated sensor
        # fusion output and is not low-pass filtered here.
        self.declare_parameter("speed_measurement_filter_alpha", 0.35)
        self.declare_parameter("speed_error_to_accel_gain", 1.25)
        self.declare_parameter("speed_error_integral_to_accel_gain", 0.05)
        # Full-throttle open-ground acceleration is speed dependent. The
        # scalar is the low-speed cap; the envelope below is the measured
        # monotonic capability curve used at runtime.
        self.declare_parameter("max_acceleration_mps2", 5.5)
        self.declare_parameter(
            "max_acceleration_speed_mps",
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
             18.0, 20.0, 22.0, 23.0])
        self.declare_parameter(
            "max_acceleration_envelope_mps2",
            [5.5, 4.4, 4.4, 3.568, 3.175, 2.562, 2.043, 1.565,
             0.956, 0.529, 0.529, 0.529, 0.086])
        self.declare_parameter("max_deceleration_mps2", 8.0)
        # The simulator's acceleration derivative contains alternating sign
        # bursts even while calibrated speed is increasing. Do not close the
        # acceleration loop on that signal: the validated inverse model and
        # actuator slew limits provide a stable command, while acceleration is
        # retained for diagnostics and future validated sensor profiles.
        self.declare_parameter("acceleration_feedback_gain", 0.0)
        self.declare_parameter("acceleration_integral_gain", 0.0)
        self.declare_parameter("acceleration_integral_limit", 2.0)
        self.declare_parameter(
            "acceleration_speed_mps",
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0,
             20.0, 23.0])
        self.declare_parameter(
            "acceleration_throttle_per_mps2",
            [0.055, 0.056, 0.058, 0.060, 0.062, 0.064,
             0.066, 0.068, 0.070, 0.072, 0.074, 0.078])
        self.declare_parameter("acceleration_throttle_rise_rate_per_sec", 2.0)
        self.declare_parameter("acceleration_throttle_fall_rate_per_sec", 4.0)
        self.declare_parameter(
            "feedforward_speed_mps",
            [0.0, 0.7537, 1.5008, 2.4883, 3.7113, 4.9226,
             7.3138, 9.6633, 11.9725, 15.3610, 18.6554, 22.8834])
        self.declare_parameter(
            "feedforward_throttle",
            [0.0, 0.030, 0.060, 0.100, 0.150, 0.200,
             0.300, 0.400, 0.500, 0.650, 0.800, 1.000])

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
            speed_hold_error_deadband_mps=float(
                self.get_parameter("speed_hold_error_deadband_mps").value),
            speed_hold_recovery_error_mps=float(
                self.get_parameter("speed_hold_recovery_error_mps").value),
            speed_hold_acceleration_deadband_mps2=float(
                self.get_parameter("speed_hold_acceleration_deadband_mps2").value),
            speed_boost_error_mps=float(
                self.get_parameter("speed_boost_error_mps").value),
            speed_hold_prediction_horizon_sec=float(
                self.get_parameter("speed_hold_prediction_horizon_sec").value),
            speed_hold_entry_margin_mps=float(
                self.get_parameter("speed_hold_entry_margin_mps").value),
            speed_downshift_stable_sec=float(
                self.get_parameter("speed_downshift_stable_sec").value),
            speed_downshift_band_mps=float(
                self.get_parameter("speed_downshift_band_mps").value),
            speed_overspeed_confirmation_sec=float(
                self.get_parameter("speed_overspeed_confirmation_sec").value),
            hard_overspeed_cutoff_mps=float(
                self.get_parameter("hard_overspeed_cutoff_mps").value),
            feedforward_speed_mps=tuple(float(v) for v in self.get_parameter("feedforward_speed_mps").value),
            feedforward_throttle=tuple(float(v) for v in self.get_parameter("feedforward_throttle").value),
            speed_error_to_accel_gain=float(
                self.get_parameter("speed_error_to_accel_gain").value),
            speed_error_integral_to_accel_gain=float(
                self.get_parameter("speed_error_integral_to_accel_gain").value),
            max_acceleration_mps2=float(
                self.get_parameter("max_acceleration_mps2").value),
            max_acceleration_speed_mps=tuple(float(v) for v in self.get_parameter(
                "max_acceleration_speed_mps").value),
            max_acceleration_envelope_mps2=tuple(float(v) for v in self.get_parameter(
                "max_acceleration_envelope_mps2").value),
            max_deceleration_mps2=float(
                self.get_parameter("max_deceleration_mps2").value),
            acceleration_feedback_gain=float(
                self.get_parameter("acceleration_feedback_gain").value),
            acceleration_integral_gain=float(
                self.get_parameter("acceleration_integral_gain").value),
            acceleration_integral_limit=float(
                self.get_parameter("acceleration_integral_limit").value),
            acceleration_throttle_rise_rate_per_sec=float(
                self.get_parameter("acceleration_throttle_rise_rate_per_sec").value),
            acceleration_throttle_fall_rate_per_sec=float(
                self.get_parameter("acceleration_throttle_fall_rate_per_sec").value),
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
        self.last_controller_odom_time = None
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

    def _on_raw_throttle_override(self, msg: Float32) -> None:
        value = float(msg.data)
        if not math.isfinite(value):
            self.raw_throttle_override = None
            self.raw_throttle_override_time = None
            return
        self.raw_throttle_override = clamp(value, 0.0, self.speed_controller.config.throttle_max_forward)
        self.raw_throttle_override_time = self.get_clock().now()

    def _on_odom(self, msg: Odometry) -> None:
        speed = float(msg.twist.twist.linear.x)
        if (not math.isfinite(speed) or speed < -0.05 or
                speed > self.max_feedback_speed):
            self.speed = None
            self.raw_speed = None
            self.odom_arrival_time = None
            self.odom_time = None
            self.odom_source_stamp_ns = None
            return
        arrival_time = self.get_clock().now()
        source_stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000 +
            int(msg.header.stamp.nanosec)
        )
        # A zero stamp is invalid for the source-time contract. Keep the
        # watchdog alive, but use the arrival clock only as an explicit
        # fallback so malformed telemetry cannot create an enormous dt.
        if source_stamp_ns <= 0:
            source_stamp_ns = arrival_time.nanoseconds
        source_sec = source_stamp_ns / 1e9
        speed = max(0.0, speed)
        self.raw_speed = speed
        self.speed = self.speed_estimator.update_odometry(speed, source_sec)
        self.acceleration = self.speed_estimator.acceleration_mps2
        self.odom_arrival_time = arrival_time
        self.odom_time = arrival_time
        self.odom_source_stamp_ns = source_stamp_ns

    def _on_imu(self, msg: Imu) -> None:
        acceleration = float(msg.linear_acceleration.x)
        if not math.isfinite(acceleration):
            return
        arrival_time = self.get_clock().now()
        source_stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000 +
            int(msg.header.stamp.nanosec)
        )
        # The observer's derivative must follow the sensor source clock. The
        # callback clock is retained only as an explicit fallback for a
        # malformed zero-stamped message; callback jitter must not become
        # physical acceleration.
        source_time_s = (
            source_stamp_ns / 1e9 if source_stamp_ns > 0
            else arrival_time.nanoseconds / 1e9
        )
        self.speed_estimator.update_acceleration(
            acceleration, source_time_s)
        # Once calibrated /odom is live, it is the absolute speed measurement
        # used by the controller. The estimator still filters IMU acceleration
        # for the acceleration-loop feedback, but its open-loop integral must
        # not overwrite the newer odometry speed between two odom callbacks.
        if not self.speed_estimator.has_odom:
            self.speed = self.speed_estimator.speed_mps
        self.acceleration = self.speed_estimator.acceleration_mps2
        self.imu_time = arrival_time

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
        self.external_stop_latched = self.collision_terminal_stop
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
        if self.reset_pub is not None:
            self.get_logger().error(
                f"Collision count increased to {count}; simulator reset requested")
        else:
            self.get_logger().error(
                f"Collision count increased to {count}; terminal stop latched, "
                "no simulator reset published")

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
            if self.collision_terminal_stop:
                self.get_logger().error(
                    "Simulator reset pulse completed; actuator remains latched neutral after collision")
            else:
                self.external_stop_latched = False
                self.get_logger().warn(
                    "Simulator reset pulse completed; mapping actuator resumed after collision")

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
        if (self.raw_throttle_override is not None and
                self.raw_throttle_override_time is not None and
                (now - self.raw_throttle_override_time).nanoseconds / 1e9
                <= self.command_timeout):
            # Calibration owns the raw actuator only while this explicit,
            # diagnostics-only override is fresh. Reset controller state so
            # its previous closed-loop target cannot resume during braking.
            self.speed_controller.reset()
            self.control_time = None
            self._publish(0.0, self.raw_throttle_override)
            self.last_neutral_reason = None
            return
        if self.command is None or self.command_time is None:
            return self._neutral("no command")
        if self.speed is None or self.odom_arrival_time is None:
            return self._neutral("no odometry")
        if self.raw_speed is None:
            return self._neutral("invalid odometry")
        if (now - self.command_time).nanoseconds / 1e9 > self.command_timeout:
            return self._neutral("command timeout")
        if (now - self.odom_arrival_time).nanoseconds / 1e9 > self.odom_timeout:
            return self._neutral("odometry timeout")

        dt = self.nominal_dt
        if self.control_time is not None:
            dt = clamp((now - self.control_time).nanoseconds / 1e9, 1e-3, 0.5)
        self.control_time = now

        steering_angle, target_speed, target_accel = self.command
        steering = clamp(steering_angle / self.max_steering, -1.0, 1.0)
        # The speed interface owns the speed target. Any acceleration field in
        # an incoming speed command is diagnostic metadata and is ignored.
        # Use raw accepted odometry for this safety decision so filtering can
        # never retain forward throttle after a large target crossing.
        if (self.raw_speed - target_speed >=
                self.speed_controller.config.hard_overspeed_cutoff_mps):
            self.speed_controller.reset()
            self.control_time = None
            throttle = 0.0
        else:
            fresh_odom = (
                self.odom_source_stamp_ns is not None and
                (self.last_controller_odom_source_stamp_ns is None or
                 self.odom_source_stamp_ns !=
                 self.last_controller_odom_source_stamp_ns))
            throttle = self.speed_controller.update(
                target_speed, self.speed, 0.0, dt, self.acceleration,
                measurement_fresh=fresh_odom)
            self.last_controller_odom_time = self.odom_time
            if fresh_odom:
                self.last_controller_odom_source_stamp_ns = self.odom_source_stamp_ns
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

"""Single calibration/recording node for AutoDRIVE."""

import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Bool, Float32, Float64, Float64MultiArray, Int32


# The simulator bridge is best-effort and can deliver several native samples
# in one executor burst. The default sensor-data depth is only five, which
# lets the Python diagnostics recorder drop source events while it flushes
# CSV rows. Keep the source timestamps and retain enough backlog to measure
# the bridge's actual cadence without changing runtime topics.
SOURCE_SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
DERIVED_ODOMETRY_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


FIELDS = (
    # stamp_s is the absolute ROS/system timestamp used to align this
    # recorder with timestamped diagnostic files. time_s remains the
    # human-friendly elapsed time within this recorder process.
    "stamp_s", "time_s", "phase", "source_event_name", "source_event_count",
    "source_event_stamp_s", "target_speed_mps", "target_accel_mps2",
    "controller_speed_mps", "controller_accel_mps2", "controller_steering_rad",
    "throttle_command", "throttle_feedback", "steering_command",
    "steering_feedback", "speed_mps", "speed_rate_mps2", "gt_speed_rate_mps2",
    "ax_mps2",
    "ay_mps2", "az_mps2",
    "imu_accel_norm_mps2", "yaw_rate_radps", "odom_yaw_rate_radps",
    "imu_yaw_rate_radps", "imu_yaw_rad", "imu_stamp_s", "odom_stamp_s",
    "gt_odom_stamp_s", "left_encoder_stamp_s", "right_encoder_stamp_s",
    "left_encoder_rad",
    "right_encoder_rad", "left_encoder_speed_radps", "right_encoder_speed_radps",
    "left_encoder_dt_s", "right_encoder_dt_s", "encoder_wheel_speed_mps",
    "gt_slip_speed_mps", "gt_slip_ratio", "x_odom_m", "y_odom_m", "yaw_odom_rad",
    # Team odometry diagnostics. These are allowed-state diagnostics recorded
    # for estimator identification; the recorder does not feed them back into
    # the estimator or controller.
    "odom_raw_wheel_speed_mps", "odom_corrected_wheel_speed_mps",
    "odom_longitudinal_slip_ratio", "odom_wheel_observation_confidence",
    "odom_imu_acceleration_bias_mps2", "odom_encoder_reset_count",
    "odom_imu_speed_mps", "odom_frozen_encoder_model_active",
    "odom_frozen_encoder_model_decel_mps2",
    "odom_sensor_fusion_model_speed_mps", "odom_sensor_fusion_model_spread_mps",
    "odom_sensor_fusion_model_active", "odom_sensor_motion_regime",
    "odom_sensor_fusion_window_raw_speed_mps",
    "odom_sensor_fusion_window_mapped_speed_mps",
    "odom_diagnostics_stamp_s", "odom_imu_observer_acceleration_mps2",
    "odom_imu_pair_speed_mps", "odom_imu_pair_lead_s",
    "odom_diagnostics_speed_mps", "odom_v2_global_speed_mps",
    "odom_v2_base_speed_mps", "odom_v2_braking_speed_mps",
    "odom_v2_braking_blend",
    "odom_v2_global_residual", "odom_v2_braking_residual", "odom_v2_active",
    # Brake/coast completion diagnostics. A reset is permitted only after
    # fresh encoder, IMU, and local-odom evidence has remained stopped.
    "brake_encoder_stopped", "brake_imu_stopped", "brake_odom_stopped",
    "brake_gt_stopped",
    "brake_stop_confirmed", "brake_stop_elapsed_s",
    "speed_settled", "speed_settle_rate_mps2", "speed_settle_elapsed_s",
    # Simulator ground truth. The timestamped simulator odometry pose is the
    # primary truth stream. IPS has no header and is logged separately below;
    # it must never overwrite the timestamped pose used for fitting.
    "gt_x_m", "gt_y_m", "gt_z_m", "gt_yaw_rad",
    "gt_vx_mps", "gt_vy_mps", "gt_vz_mps", "gt_speed_mps", "gt_ax_mps2",
    "gt_ay_mps2", "gt_accel_mps2", "gt_longitudinal_accel_mps2", "gt_dt_s",
    "gt_yaw_rate_radps", "gt_collision_count",
    "gt_odom_x_m", "gt_odom_y_m", "gt_odom_z_m", "gt_odom_yaw_rad",
    "gt_ips_x_m", "gt_ips_y_m", "gt_ips_z_m",
    "x_amcl_m", "y_amcl_m", "yaw_amcl_rad", "amcl_xy_variance", "amcl_yaw_variance",
    "x_ekf_m", "y_ekf_m", "yaw_ekf_rad", "ekf_xy_variance", "ekf_yaw_variance",
    "lidar_rate_hz", "lidar_event_count",
    "imu_rate_hz", "imu_event_count",
    "left_encoder_rate_hz", "left_encoder_event_count",
    "right_encoder_rate_hz", "right_encoder_event_count",
    "odom_rate_hz", "odom_event_count",
    "odom_diagnostics_rate_hz", "odom_diagnostics_event_count",
    "gt_odom_rate_hz", "gt_odom_event_count",
    "gt_ips_rate_hz", "gt_ips_event_count",
    "collision_rate_hz", "collision_event_count",
    "amcl_rate_hz", "amcl_event_count",
    "ekf_rate_hz", "ekf_event_count",
    "speed_command_rate_hz", "speed_command_event_count",
    "acceleration_command_rate_hz", "acceleration_command_event_count",
    "throttle_command_rate_hz", "throttle_command_event_count",
    "steering_command_rate_hz", "steering_command_event_count",
    "throttle_feedback_rate_hz", "throttle_feedback_event_count",
    "steering_feedback_rate_hz", "steering_feedback_event_count",
    "amcl_timing_ms", "amcl_gpu_transfer_ms", "amcl_gpu_pf_ms",
    "amcl_gpu_callback_ms", "amcl_pose_published", "amcl_cluster_weight",
    "amcl_kld_pre_particles", "amcl_kld_bins", "amcl_kld_target",
    "amcl_kld_particles",
    # Production controllers publish separate physical-unit abstractions.
    # Both command streams are logged independently so a fit can distinguish
    # speed-loop and acceleration-loop behavior.
    "speed_command_mps", "speed_command_accel_mps2",
    "speed_command_steering_rad", "speed_command_stamp_s",
    "acceleration_command_mps", "acceleration_command_accel_mps2",
    "acceleration_command_steering_rad", "acceleration_command_stamp_s",
)


def _yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class Calibration(Node):
    def __init__(self) -> None:
        super().__init__("calibration")
        self._declare_parameters()
        self.mode = str(self.get_parameter("mode").value)
        allowed = {
            "sensor_record", "throttle_sweep", "throttle_steps",
            "zero_throttle_decel", "speed_steps", "speed_ramp",
            "acceleration_steps",
            "steering_steps", "steering_response", "throttle_speed_grid",
            "identification_grid",
            "full_suite",
        }
        if self.mode not in allowed:
            raise ValueError(f"unsupported mode: {self.mode}")

        output_dir = Path(str(self.get_parameter("output_dir").value))
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.output_path = output_dir / f"{self.mode}_{stamp}.csv"
        self.stream = self.output_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=FIELDS)
        self.writer.writeheader()

        self.state = {field: math.nan for field in FIELDS}
        self.state["phase"] = "waiting"
        self.start = self.get_clock().now()
        self.phase_start = self.start
        self.last_odom = None
        self.last_odom_speed_sample = None
        self.last_gt_odom = None
        self.last_gt_sample = None
        self.last_encoder_sample = {"left": None, "right": None}
        self.rate_event_names = (
            "lidar", "imu", "left_encoder", "right_encoder", "odom",
            "odom_diagnostics", "gt_odom", "gt_ips", "collision", "amcl", "ekf",
            "speed_command", "acceleration_command",
            "throttle_command", "steering_command",
            "throttle_feedback", "steering_feedback",
        )
        self.event_counts = {name: 0 for name in self.rate_event_names}
        self.event_window_counts = {name: 0 for name in self.rate_event_names}
        self.event_rates = {name: math.nan for name in self.rate_event_names}
        self.rate_window_start = self.start
        self.capture_source_events = bool(
            self.get_parameter("capture_source_events").value)
        self.source_event_rows = []
        self.finished = False
        self.reset_pending = False
        self.reset_odom_confirmed = False
        self.reset_odom_baseline_event = 0
        self.reset_wait_start = None
        self.reset_zero_odom_since = None
        self.reset_gt_confirmed = False
        self.reset_gt_baseline_event = 0
        self.reset_zero_gt_since = None
        self.boundary_reset_count = 0
        self.active_phase = None
        self.brake_stop_since = None
        self.brake_event_baseline = None
        self.speed_settle_since = None
        self.speed_settle_event_baseline = None
        self.reset_signal_sent = False
        self.reset_connection_ready = False
        self.reset_signal_cleared = False

        self.max_speed = float(self.get_parameter("maximum_test_speed_mps").value)
        self.max_throttle = float(self.get_parameter("maximum_throttle").value)
        self.max_steering = float(self.get_parameter("maximum_steering_command").value)
        self.encoder_wheel_radius = float(
            self.get_parameter("encoder_wheel_radius_m").value)
        self.telemetry_timeout = float(self.get_parameter("telemetry_timeout_sec").value)
        self.startup_timeout = max(
            self.telemetry_timeout, float(self.get_parameter("startup_timeout_sec").value))
        self.duration = float(self.get_parameter("duration_sec").value)
        self.get_logger().info(
            "Calibration mode=%s duration=%.3f s output=%s"
            % (self.mode, self.duration, self.output_path))
        self.reset_between_steps = bool(
            self.get_parameter("reset_between_steps").value)
        self.reset_pulse_sec = max(
            0.05, float(self.get_parameter("reset_pulse_sec").value))
        self.reset_signal_hold_sec = min(
            self.reset_pulse_sec,
            max(0.10, float(self.get_parameter("reset_signal_hold_sec").value)))

        self.throttle_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/throttle_command", 10)
        self.steering_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/steering_command", 10)
        self.speed_drive_pub = self.create_publisher(
            AckermannDriveStamped, "/cmd/speed", 10)
        self.acceleration_drive_pub = self.create_publisher(
            AckermannDriveStamped, "/cmd/acceleration", 10)
        # Only diagnostic test launches consume this topic. It prevents a
        # production acceleration=0 hold request from fighting the recorder's
        # raw zero-throttle brake/coast phase.
        self.raw_throttle_override_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/raw_throttle_override", 10)
        self.reset_pub = None
        if self.reset_between_steps:
            # Diagnostics-only simulator reset.  No controller subscribes to
            # this topic and the option is disabled in all production modes.
            self.reset_pub = self.create_publisher(
                Bool, "/autodrive/reset_command", 10)

        self.create_subscription(
            Odometry, "/odom", self._on_odom, DERIVED_ODOMETRY_QOS)
        self.create_subscription(
            Float64MultiArray, "/odom/diagnostics", self._on_odom_diagnostics,
            DERIVED_ODOMETRY_QOS)
        # Restricted simulator ground truth is intentionally subscribed to by
        # this diagnostics recorder only. It provides the reference needed to
        # fit acceleration, delay, encoder scale, and collision metrics.
        self.create_subscription(
            Odometry, "/autodrive/roboracer_1/odom", self._on_gt_odom,
            SOURCE_SENSOR_QOS)
        self.create_subscription(
            Point, "/autodrive/roboracer_1/ips", self._on_gt_ips,
            SOURCE_SENSOR_QOS)
        self.create_subscription(
            Int32, "/autodrive/roboracer_1/collision_count",
            self._on_collision_count, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/ekf_pose", self._on_ekf, 10)
        self.create_subscription(
            Imu, "/autodrive/roboracer_1/imu",
            self._on_imu, SOURCE_SENSOR_QOS)
        self.create_subscription(
            JointState, "/autodrive/roboracer_1/left_encoder",
            lambda m: self._on_encoder(m, "left_encoder_rad"), SOURCE_SENSOR_QOS)
        self.create_subscription(
            JointState, "/autodrive/roboracer_1/right_encoder",
            lambda m: self._on_encoder(m, "right_encoder_rad"), SOURCE_SENSOR_QOS)
        self.create_subscription(
            LaserScan, "/autodrive/roboracer_1/lidar", self._on_lidar,
            SOURCE_SENSOR_QOS)
        self.create_subscription(
            AckermannDriveStamped, "/cmd/speed",
            lambda m: self._on_controller_command(m, "speed"), 10)
        self.create_subscription(
            AckermannDriveStamped, "/cmd/acceleration",
            lambda m: self._on_controller_command(m, "acceleration"), 10)
        self.create_subscription(
            Float32, "/autodrive/roboracer_1/throttle",
            lambda m: self._on_scalar(m, "throttle_feedback", "throttle_feedback"), 10)
        self.create_subscription(
            Float32, "/autodrive/roboracer_1/steering",
            lambda m: self._on_scalar(m, "steering_feedback", "steering_feedback"), 10)
        # These are diagnostics emitted by the team AMCL implementation. They
        # are recorded for offline tuning only and are not controller inputs.
        self.create_subscription(
            Float64, "/amcl_timing", lambda m: self._set("amcl_timing_ms", m.data), 10)
        self.create_subscription(
            Float64MultiArray, "/amcl_gpu_timing", self._on_amcl_gpu_timing, 10)
        self.create_subscription(
            Float64MultiArray, "/amcl_kld_diagnostics", self._on_amcl_kld, 10)
        self.create_subscription(
            Float32, "/autodrive/roboracer_1/throttle_command",
            lambda m: self._on_scalar(m, "throttle_command", "throttle_command"), 10)
        self.create_subscription(
            Float32, "/autodrive/roboracer_1/steering_command",
            lambda m: self._on_scalar(m, "steering_command", "steering_command"), 10)

        self.phases = self._build_phases()
        self.phase_index = 0
        rate = max(1.0, float(self.get_parameter("sample_rate_hz").value))
        self.timer = self.create_timer(1.0 / rate, self._tick)

    def _declare_parameters(self) -> None:
        self.declare_parameter("mode", "sensor_record")
        self.declare_parameter(
            "output_dir", "/workspace/src/sdu_apex_autodrive/artifacts/calibration/raw")
        self.declare_parameter("sample_rate_hz", 50.0)
        # Timer snapshots are useful for command/phase diagnostics, but a
        # burst of native callbacks can make them skip source samples.  When
        # enabled, important sensor callbacks also enqueue exact snapshots;
        # the analyzer selects the GT-aligned rows for model fitting.
        self.declare_parameter("capture_source_events", False)
        self.declare_parameter("duration_sec", 0.0)
        self.declare_parameter(
            "throttle_sequence", [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        self.declare_parameter(
            "speed_sequence_mps", [0.0, 0.5, 1.0, 1.5, 2.0, 1.0, 0.0])
        self.declare_parameter(
            "acceleration_sequence_mps2", [0.0, 1.0, 2.0, 4.0, 6.0, 2.0, -2.0])
        self.declare_parameter(
            "steering_sequence",
            [0.0, -0.25, 0.0, 0.25, 0.0, -0.5, 0.0, 0.5, 0.0])
        self.declare_parameter(
            "grid_speed_sequence_mps", [0.0, 3.0, 6.0, 9.0, 12.0])
        self.declare_parameter(
            "grid_base_throttle_sequence", [0.0, 0.08, 0.18, 0.30, 0.50])
        self.declare_parameter(
            "grid_throttle_sequence",
            [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0])
        self.declare_parameter("grid_steering_enabled", False)
        self.declare_parameter(
            "grid_steering_speed_sequence_mps", [3.0, 9.0, 16.0, 20.0])
        self.declare_parameter(
            "grid_steering_base_throttle_sequence", [0.08, 0.25, 0.50, 0.65])
        self.declare_parameter(
            "grid_steering_sequence", [0.0, -0.25, 0.25, -0.50, 0.50])
        self.declare_parameter("grid_steering_hold_sec", 2.0)
        self.declare_parameter("grid_base_hold_sec", 2.0)
        self.declare_parameter("grid_step_hold_sec", 1.5)
        self.declare_parameter("grid_brake_timeout_sec", 30.0)
        self.declare_parameter("throttle_settle_timeout_sec", 15.0)
        self.declare_parameter("throttle_settle_min_sec", 1.5)
        self.declare_parameter("throttle_settle_stable_sec", 0.75)
        self.declare_parameter("throttle_settle_speed_rate_mps2", 0.15)
        self.declare_parameter("speed_settle_error_mps", 0.25)
        self.declare_parameter("speed_settle_timeout_sec", 15.0)
        self.declare_parameter("brake_stop_encoder_speed_radps", 0.5)
        self.declare_parameter("brake_stop_imu_accel_mps2", 0.25)
        self.declare_parameter("brake_stop_imu_yaw_rate_radps", 0.05)
        self.declare_parameter("brake_stop_odom_speed_mps", 0.20)
        self.declare_parameter("brake_stop_gt_speed_mps", 0.20)
        self.declare_parameter("brake_stop_stable_sec", 0.75)
        self.declare_parameter("steering_test_throttle", 0.23)
        self.declare_parameter("hold_sec", 3.0)
        self.declare_parameter("zero_settle_sec", 2.0)
        # Diagnostic guard defaults to the documented simulator command
        # envelope.  It is not a controller speed or throttle ceiling.
        self.declare_parameter("maximum_test_speed_mps", 22.88)
        self.declare_parameter("maximum_acceleration_mps2", 8.0)
        self.declare_parameter("maximum_throttle", 1.0)
        self.declare_parameter("maximum_steering_command", 0.50)
        self.declare_parameter("encoder_wheel_radius_m", 0.0590)
        self.declare_parameter("telemetry_timeout_sec", 0.50)
        # Simulator startup can take several seconds after the GUI appears;
        # this is only a pre-telemetry grace period, not a runtime watchdog.
        self.declare_parameter("startup_timeout_sec", 30.0)
        self.declare_parameter("reset_between_steps", False)
        # Speed-transition diagnostics can either isolate every target with
        # a brake phase or exercise the production controller continuously
        # from one target to the next. The latter is useful for up/down
        # tracking validation and remains diagnostics-only.
        self.declare_parameter("speed_steps_brake_between_targets", True)
        self.declare_parameter("reset_pulse_sec", 0.25)
        # The paced bridge samples its latched reset level at 40 Hz. The
        # simulator currently requires reset=true to remain asserted across
        # paced commands until the teleport is acknowledged; the harness only
        # enters this phase after the brake/stop gate has completed and clears
        # it immediately after confirmation.
        self.declare_parameter("reset_signal_hold_sec", 0.20)
        # A diagnostics reset must reach both the simulator bridge and the
        # local epoch subscribers.  Counting only the local odometry
        # subscriber can advance the pulse before the bridge has discovered
        # the volatile publisher, leaving the simulator at the old position.
        self.declare_parameter("reset_min_subscribers", 2)
        self.declare_parameter("reset_odom_speed_threshold_mps", 0.20)
        self.declare_parameter("reset_zero_odom_stable_sec", 30.0)
        # Open-world diagnostics must confirm that the simulator is back at
        # its spawn pose, not merely stationary at the previous position.
        self.declare_parameter("reset_ground_truth_position_tolerance_m", 2.0)
        # The official bridge can deliver the post-reset encoder zero several
        # tens of seconds after the simulator teleport. Give the reset enough
        # time rather than silently shortening the requested test.
        self.declare_parameter("reset_confirmation_timeout_sec", 60.0)
        # Diagnostics-only guard for the finite open-ground simulator plane.
        # It prevents a long full-throttle phase from leaving the plane and
        # contaminating the tail with invalid ground-truth samples. This is
        # never loaded by a competition controller.
        self.declare_parameter("ground_truth_boundary_guard_enabled", True)
        self.declare_parameter("ground_truth_boundary_distance_m", 450.0)

    def _build_phases(self):
        if self.mode == "sensor_record":
            return []

        hold = float(self.get_parameter("hold_sec").value)
        settle = float(self.get_parameter("zero_settle_sec").value)

        if self.mode in {"throttle_sweep", "throttle_steps"}:
            sequence = [float(v) for v in self.get_parameter("throttle_sequence").value]
            if any(v < 0.0 or v > self.max_throttle for v in sequence):
                raise ValueError("throttle sequence exceeds configured limit")
            phases = []
            for value in sequence:
                if self.reset_between_steps:
                    phases.append(("reset", "reset", 1.0, self.reset_pulse_sec))
                phases.extend([
                    ("settle", "raw_throttle", 0.0, settle),
                    (f"throttle_{value:.3f}", "raw_throttle", value, hold),
                ])
            return phases + [("final_zero", "raw_throttle", 0.0, settle)]

        if self.mode == "zero_throttle_decel":
            value = min(
                max(float(v) for v in self.get_parameter("throttle_sequence").value),
                self.max_throttle,
            )
            return [
                ("settle", "raw_throttle", 0.0, settle),
                ("accelerate", "raw_throttle", value, hold),
                ("zero", "raw_throttle", 0.0, 2.0 * hold),
            ]

        if self.mode in {"speed_steps", "speed_ramp"}:
            sequence = [float(v) for v in self.get_parameter("speed_sequence_mps").value]
            if any(v < 0.0 or v > self.max_speed for v in sequence):
                raise ValueError("speed sequence exceeds configured limit")
            phases = []
            # A no-reset validation run may be launched immediately after a
            # manual simulator teleport. Give the bridge and all diagnostic
            # subscriptions one neutral interval before the first target so
            # startup latency cannot hide the acceleration transient.
            if not self.reset_between_steps:
                phases.append(("pretest_settle", "raw_throttle", 0.0, settle))
            for value in sequence:
                if self.reset_between_steps:
                    phases.append(("reset", "reset", 1.0, self.reset_pulse_sec))
                    phases.append(("settle", "raw_throttle", 0.0, settle))
                phases.append((f"speed_{value:.2f}", self.mode, value, hold))
                if bool(self.get_parameter(
                        "speed_steps_brake_between_targets").value):
                    phases.append((
                        f"grid_brake_speed_{value:.2f}", "raw_throttle", 0.0,
                        max(1.0, float(self.get_parameter(
                            "grid_brake_timeout_sec").value))))
            if (sequence and not bool(self.get_parameter(
                    "speed_steps_brake_between_targets").value)):
                # The final zero target normally already coasts to rest, but
                # retain the independent stop confirmation so this continuous
                # transition test cannot finish while the car is sliding.
                phases.append((
                    f"grid_brake_speed_{sequence[-1]:.2f}", "raw_throttle", 0.0,
                    max(1.0, float(self.get_parameter(
                        "grid_brake_timeout_sec").value))))
            return phases

        if self.mode == "acceleration_steps":
            sequence = [float(v) for v in self.get_parameter(
                "acceleration_sequence_mps2").value]
            maximum = float(self.get_parameter("maximum_acceleration_mps2").value)
            if any(not math.isfinite(v) or abs(v) > maximum for v in sequence):
                raise ValueError("acceleration sequence exceeds configured limit")
            brake_timeout = max(1.0, float(self.get_parameter(
                "grid_brake_timeout_sec").value))
            phases = []
            for value in sequence:
                if self.reset_between_steps:
                    phases.append(("reset", "reset", 1.0, self.reset_pulse_sec))
                    phases.append(("settle", "raw_throttle", 0.0, settle))
                phases.append((f"acceleration_{value:.2f}",
                               "acceleration_steps", value, hold))
                phases.append((f"grid_brake_acceleration_{value:.2f}",
                               "raw_throttle", 0.0, brake_timeout))
            return phases + [("final_zero", "raw_throttle", 0.0, settle)]

        if self.mode == "throttle_speed_grid":
            speeds = [float(v) for v in self.get_parameter(
                "grid_speed_sequence_mps").value]
            base_throttles = [float(v) for v in self.get_parameter(
                "grid_base_throttle_sequence").value]
            throttles = [float(v) for v in self.get_parameter(
                "grid_throttle_sequence").value]
            if len(speeds) != len(base_throttles):
                raise ValueError(
                    "grid speed and base-throttle sequences must have equal length")
            if any(v < 0.0 or v > self.max_throttle
                   for v in (*base_throttles, *throttles)):
                raise ValueError("grid throttle sequence exceeds configured limit")
            if any(v < 0.0 or v > self.max_speed for v in speeds):
                raise ValueError("grid speed sequence exceeds configured limit")
            base_hold = max(0.1, float(self.get_parameter(
                "grid_base_hold_sec").value))
            step_hold = max(0.1, float(self.get_parameter(
                "grid_step_hold_sec").value))
            phases = []
            # Each row is reset and re-accelerated from open ground. The
            # nominal speed is encoded in the phase name; the actual speed at
            # every throttle sample comes from simulator truth in the CSV.
            # This avoids using truth in the controller while still exposing
            # the full throttle x operating-speed response offline.
            for nominal_speed, base_throttle in zip(speeds, base_throttles):
                if self.reset_between_steps:
                    phases.append(("reset", "reset", 1.0, self.reset_pulse_sec))
                    phases.append(("settle", "raw_throttle", 0.0, settle))
                if base_throttle > 0.0:
                    phases.append((
                        f"grid_base_{nominal_speed:.2f}",
                        "raw_throttle", base_throttle, base_hold))
                for throttle in throttles:
                    phases.append((
                        f"grid_throttle_{throttle:.3f}_at_{nominal_speed:.2f}",
                        "raw_throttle", throttle, step_hold))
            return phases + [("final_zero", "raw_throttle", 0.0, settle)]

        if self.mode == "identification_grid":
            speeds = [float(v) for v in self.get_parameter(
                "grid_speed_sequence_mps").value]
            base_throttles = [float(v) for v in self.get_parameter(
                "grid_base_throttle_sequence").value]
            throttles = [float(v) for v in self.get_parameter(
                "grid_throttle_sequence").value]
            if len(speeds) != len(base_throttles):
                raise ValueError(
                    "grid speed and base-throttle sequences must have equal length")
            if any(v < 0.0 or v > self.max_throttle
                   for v in (*base_throttles, *throttles)):
                raise ValueError("grid throttle sequence exceeds configured limit")
            if any(v < 0.0 or v > self.max_speed for v in speeds):
                raise ValueError("grid speed sequence exceeds configured limit")

            base_hold = max(0.1, float(self.get_parameter(
                "grid_base_hold_sec").value))
            step_hold = max(0.1, float(self.get_parameter(
                "grid_step_hold_sec").value))
            brake_timeout = max(1.0, float(self.get_parameter(
                "grid_brake_timeout_sec").value))
            phases = [("pretest_settle", "raw_throttle", 0.0, settle)]
            for nominal_speed, base_throttle in zip(speeds, base_throttles):
                for throttle in throttles:
                    # The open-ground plane is finite. Every Cartesian grid
                    # point is therefore an independent experiment: reset,
                    # settle, accelerate to its operating regime, apply one
                    # throttle value, then record a full zero-throttle
                    # brake/coast response. The brake is data, never a reset
                    # condition even when the encoder freezes while truth
                    # continues moving.
                    point = f"{nominal_speed:.2f}_throttle_{throttle:.3f}"
                    phases.extend([
                        (f"grid_reset_{point}", "reset", 1.0,
                         self.reset_pulse_sec),
                        (f"grid_settle_{point}", "raw_throttle", 0.0,
                         settle),
                    ])
                    if base_throttle > 0.0:
                        phases.append((
                            f"grid_base_{point}",
                            "raw_throttle", base_throttle, base_hold))
                    phases.append((
                        f"grid_throttle_{throttle:.3f}_at_{nominal_speed:.2f}",
                        "raw_throttle", throttle, step_hold))
                    phases.append((
                        f"grid_brake_{nominal_speed:.2f}_throttle_{throttle:.3f}",
                        "raw_throttle", 0.0, brake_timeout))

            if bool(self.get_parameter("grid_steering_enabled").value):
                steering_speeds = [float(v) for v in self.get_parameter(
                    "grid_steering_speed_sequence_mps").value]
                steering_bases = [float(v) for v in self.get_parameter(
                    "grid_steering_base_throttle_sequence").value]
                steering_values = [float(v) for v in self.get_parameter(
                    "grid_steering_sequence").value]
                if len(steering_speeds) != len(steering_bases):
                    raise ValueError(
                        "steering speed and base-throttle sequences must have equal length")
                if any(v < 0.0 or v > self.max_speed for v in steering_speeds):
                    raise ValueError("steering speed sequence exceeds configured limit")
                if any(v < 0.0 or v > self.max_throttle for v in steering_bases):
                    raise ValueError("steering base-throttle sequence exceeds configured limit")
                if any(abs(v) > self.max_steering for v in steering_values):
                    raise ValueError("grid steering sequence exceeds configured limit")
                steering_hold = max(0.1, float(self.get_parameter(
                    "grid_steering_hold_sec").value))
                for nominal_speed, base_throttle in zip(steering_speeds, steering_bases):
                    for steering in steering_values:
                        label = f"steering_at_{nominal_speed:.2f}_{steering:.3f}"
                        phases.extend([
                            (f"grid_reset_{label}", "reset", 1.0, self.reset_pulse_sec),
                            (f"grid_settle_{label}", "raw_throttle", 0.0, settle),
                            (f"grid_base_{label}", "raw_throttle", base_throttle, base_hold),
                        ])
                        phases.append((
                            f"steering_{steering:.3f}_at_{nominal_speed:.2f}",
                            "raw_steering_with_throttle", steering, steering_hold))
                        phases.append((
                            f"grid_brake_{label}_{steering:.3f}",
                            "raw_throttle", 0.0, brake_timeout))
            return phases + [("final_zero", "raw_throttle", 0.0, 2.0 * settle)]

        if self.mode == "steering_response":
            sequence = [float(v) for v in self.get_parameter(
                "steering_sequence").value]
            if any(abs(v) > self.max_steering for v in sequence):
                raise ValueError("steering sequence exceeds configured limit")
            throttle = float(self.get_parameter("steering_test_throttle").value)
            if not 0.0 <= throttle <= self.max_throttle:
                raise ValueError("steering_test_throttle exceeds configured limit")
            phases = []
            for value in sequence:
                if self.reset_between_steps:
                    phases.append(("reset", "reset", 1.0, self.reset_pulse_sec))
                    phases.append(("settle", "raw_throttle", 0.0, settle))
                phases.append((
                    f"steering_{value:.3f}",
                    "raw_steering_with_throttle", value, hold))
            return phases + [("final_zero", "raw_throttle", 0.0, settle)]

        if self.mode == "full_suite":
            throttle_sequence = [
                float(v) for v in self.get_parameter("throttle_sequence").value
            ]
            speed_sequence = [
                float(v) for v in self.get_parameter("speed_sequence_mps").value
            ]
            if any(v < 0.0 or v > self.max_throttle for v in throttle_sequence):
                raise ValueError("throttle sequence exceeds configured limit")
            if any(v < 0.0 or v > self.max_speed for v in speed_sequence):
                raise ValueError("speed sequence exceeds configured limit")

            phases = [("suite_start_settle", "raw_throttle", 0.0, settle)]
            # Open-ground step response across the complete normalized forward
            # command range. Every point is isolated so its acceleration and
            # steady-state speed can be compared directly to ground truth.
            for value in throttle_sequence:
                if self.reset_between_steps:
                    phases.append(("reset", "reset", 1.0, self.reset_pulse_sec))
                phases.extend([
                    ("settle", "raw_throttle", 0.0, settle),
                    (f"throttle_{value:.3f}", "raw_throttle", value, hold),
                ])

            # Full-throttle release gives the free deceleration and stopping
            # response without relying on a project-specific throttle cap.
            if self.reset_between_steps:
                phases.append(("reset", "reset", 1.0, self.reset_pulse_sec))
                phases.append(("settle", "raw_throttle", 0.0, settle))
            phases.extend([
                ("full_throttle_accelerate", "raw_throttle", self.max_throttle, hold),
                ("zero_throttle_decel", "raw_throttle", 0.0, 3.0 * hold),
            ])

            # Closed-loop speed steps exercise the actual production speed
            # controller at low, medium, high, and stop targets.
            for value in speed_sequence:
                if self.reset_between_steps:
                    phases.append(("reset", "reset", 1.0, self.reset_pulse_sec))
                    phases.append(("settle", "raw_throttle", 0.0, settle))
                phases.append((f"speed_{value:.2f}", "speed_steps", value, hold))
            phases.append(("final_zero", "speed_steps", 0.0, max(settle, 2.0)))
            return phases

        sequence = [float(v) for v in self.get_parameter("steering_sequence").value]
        if any(abs(v) > self.max_steering for v in sequence):
            raise ValueError("steering sequence exceeds configured limit")
        return [(f"steering_{v:.2f}", "raw_steering", v, hold) for v in sequence]

    def _set(self, name, value):
        self.state[name] = value

    def _capture_source_event(self, name: str, stamp_s: float) -> None:
        """Queue an exact callback snapshot when source logging is enabled."""
        if not self.capture_source_events:
            return
        if name not in {
                "imu", "left_encoder", "right_encoder", "odom",
                "odom_diagnostics", "gt_odom"}:
            return
        now = self.get_clock().now()
        row = dict(self.state)
        row["stamp_s"] = now.nanoseconds * 1.0e-9
        row["time_s"] = (now - self.start).nanoseconds * 1.0e-9
        row["phase"] = self.active_phase or str(self.state.get("phase", "waiting"))
        row["source_event_name"] = name
        row["source_event_count"] = self.event_counts[name]
        row["source_event_stamp_s"] = stamp_s
        self.source_event_rows.append(row)

    def _flush_source_event_rows(self) -> None:
        if not self.source_event_rows:
            return
        for row in self.source_event_rows:
            self.writer.writerow(row)
        self.source_event_rows.clear()
        self.stream.flush()

    def _message_stamp_s(self, msg) -> float:
        """Return the source timestamp used for derivative/rate estimates."""
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is not None:
            value = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
            if value > 0.0:
                return value
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _update_encoder_slip(self) -> None:
        left = self.state.get("left_encoder_speed_radps", math.nan)
        right = self.state.get("right_encoder_speed_radps", math.nan)
        if not (math.isfinite(left) and math.isfinite(right)):
            return
        wheel_speed = self.encoder_wheel_radius * 0.5 * (left + right)
        self.state["encoder_wheel_speed_mps"] = wheel_speed
        gt_speed = self.state.get("gt_speed_mps", math.nan)
        if math.isfinite(gt_speed):
            self.state["gt_slip_speed_mps"] = wheel_speed - gt_speed
            self.state["gt_slip_ratio"] = (
                (wheel_speed - gt_speed) / max(abs(gt_speed), 0.5))

    def _record_event(self, name) -> None:
        """Count a callback and maintain a common rolling cadence estimate."""
        if name not in self.event_counts:
            return
        self.event_counts[name] += 1
        self.event_window_counts[name] += 1
        now = self.get_clock().now()
        elapsed = (now - self.rate_window_start).nanoseconds / 1e9
        if elapsed >= 2.0:
            for event_name in self.rate_event_names:
                self.event_rates[event_name] = (
                    self.event_window_counts[event_name] / max(elapsed, 1e-6)
                )
                self.event_window_counts[event_name] = 0
            self.rate_window_start = now
        rate_field = f"{name}_rate_hz"
        count_field = f"{name}_event_count"
        if rate_field in self.state:
            self.state[rate_field] = self.event_rates[name]
        if count_field in self.state:
            self.state[count_field] = self.event_counts[name]

    def _on_scalar(self, msg: Float32, field: str, event_name: str) -> None:
        self._record_event(event_name)
        self._set(field, float(msg.data))

    def _on_odom(self, msg: Odometry) -> None:
        self._record_event("odom")
        speed = max(0.0, float(msg.twist.twist.linear.x))
        self.state["speed_mps"] = speed
        stamp_s = self._message_stamp_s(msg)
        previous_speed = self.last_odom_speed_sample
        if previous_speed is not None:
            previous_stamp, previous_value = previous_speed
            dt = stamp_s - previous_stamp
            if 1.0e-3 <= dt <= 1.0:
                self.state["speed_rate_mps2"] = (speed - previous_value) / dt
            else:
                self.state["speed_rate_mps2"] = math.nan
        else:
            self.state["speed_rate_mps2"] = math.nan
        self.last_odom_speed_sample = (stamp_s, speed)
        self.state["odom_stamp_s"] = stamp_s
        self.state["odom_yaw_rate_radps"] = float(msg.twist.twist.angular.z)
        if not math.isfinite(self.state.get("imu_yaw_rate_radps", math.nan)):
            self.state["yaw_rate_radps"] = float(msg.twist.twist.angular.z)
        self.state["x_odom_m"] = float(msg.pose.pose.position.x)
        self.state["y_odom_m"] = float(msg.pose.pose.position.y)
        self.state["yaw_odom_rad"] = _yaw_from_quaternion(msg.pose.pose.orientation)
        self.last_odom = self.get_clock().now()
        if self.reset_pending and self.event_counts["odom"] > self.reset_odom_baseline_event:
            now = self.get_clock().now()
            if self.state["speed_mps"] <= float(self.get_parameter(
                    "reset_odom_speed_threshold_mps").value):
                if self.reset_zero_odom_since is None:
                    self.reset_zero_odom_since = now
                stable_sec = (now - self.reset_zero_odom_since).nanoseconds / 1e9
                if stable_sec >= max(0.1, float(self.get_parameter(
                        "reset_zero_odom_stable_sec").value)):
                    self.reset_odom_confirmed = True
            else:
                # A delayed bridge/encoder sample reintroduced motion. The
                # zero interval must start over so that stale queued samples
                # cannot contaminate the next phase.
                self.reset_zero_odom_since = None
        self._capture_source_event("odom", stamp_s)

    def _on_odom_diagnostics(self, msg: Float64MultiArray) -> None:
        """Record the fixed-order sensor-odometry diagnostic vector."""
        if len(msg.data) < 20:
            return
        self._record_event("odom_diagnostics")
        fields = (
            "odom_raw_wheel_speed_mps",
            "odom_corrected_wheel_speed_mps",
            "odom_longitudinal_slip_ratio",
            "odom_wheel_observation_confidence",
            "odom_imu_acceleration_bias_mps2",
            "odom_encoder_reset_count",
            "odom_imu_speed_mps",
            "odom_frozen_encoder_model_active",
            "odom_frozen_encoder_model_decel_mps2",
            "odom_sensor_fusion_model_speed_mps",
            "odom_sensor_fusion_model_spread_mps",
            "odom_sensor_fusion_model_active",
            "odom_sensor_motion_regime",
            "odom_sensor_fusion_window_raw_speed_mps",
            "odom_sensor_fusion_window_mapped_speed_mps",
            "odom_diagnostics_stamp_s",
            "odom_imu_observer_acceleration_mps2",
            "odom_imu_pair_speed_mps",
            "odom_imu_pair_lead_s",
            "odom_diagnostics_speed_mps",
            "odom_v2_global_speed_mps",
            "odom_v2_base_speed_mps",
            "odom_v2_braking_speed_mps",
            "odom_v2_braking_blend",
            "odom_v2_global_residual",
            "odom_v2_braking_residual",
            "odom_v2_active",
        )
        for field, value in zip(fields, msg.data[:len(fields)]):
            if math.isfinite(float(value)):
                self.state[field] = float(value)
        diagnostics_stamp = self.state.get("odom_diagnostics_stamp_s")
        if diagnostics_stamp is not None and math.isfinite(diagnostics_stamp):
            self._capture_source_event("odom_diagnostics", diagnostics_stamp)

    def _on_gt_odom(self, msg: Odometry) -> None:
        """Record simulator truth for offline calibration and validation only."""
        self._record_event("gt_odom")
        stamp_s = self._message_stamp_s(msg)
        self.state["gt_odom_stamp_s"] = stamp_s
        pose = msg.pose.pose
        twist = msg.twist.twist
        vx = float(twist.linear.x)
        vy = float(twist.linear.y)
        speed = math.hypot(vx, vy)
        yaw = _yaw_from_quaternion(pose.orientation)
        gt_dt = math.nan
        gt_ax = math.nan
        gt_ay = math.nan
        gt_accel = math.nan
        gt_longitudinal_accel = math.nan
        if self.last_gt_sample is not None:
            previous_stamp, previous_vx, previous_vy, previous_speed = self.last_gt_sample
            gt_dt = stamp_s - previous_stamp
            if 1.0e-3 <= gt_dt <= 1.0:
                gt_ax = (vx - previous_vx) / gt_dt
                gt_ay = (vy - previous_vy) / gt_dt
                gt_accel = math.hypot(gt_ax, gt_ay)
                gt_longitudinal_accel = (speed - previous_speed) / gt_dt
        self.last_gt_sample = (stamp_s, vx, vy, speed)
        self.state["gt_x_m"] = float(pose.position.x)
        self.state["gt_y_m"] = float(pose.position.y)
        self.state["gt_z_m"] = float(pose.position.z)
        self.state["gt_yaw_rad"] = yaw
        self.state["gt_odom_x_m"] = float(pose.position.x)
        self.state["gt_odom_y_m"] = float(pose.position.y)
        self.state["gt_odom_z_m"] = float(pose.position.z)
        self.state["gt_odom_yaw_rad"] = yaw
        self.state["gt_vx_mps"] = vx
        self.state["gt_vy_mps"] = vy
        self.state["gt_vz_mps"] = float(twist.linear.z)
        self.state["gt_speed_mps"] = speed
        self.state["gt_ax_mps2"] = gt_ax
        self.state["gt_ay_mps2"] = gt_ay
        self.state["gt_accel_mps2"] = gt_accel
        self.state["gt_longitudinal_accel_mps2"] = gt_longitudinal_accel
        self.state["gt_speed_rate_mps2"] = gt_longitudinal_accel
        self.state["gt_dt_s"] = gt_dt
        self.state["gt_yaw_rate_radps"] = float(twist.angular.z)
        self._update_encoder_slip()
        self.last_gt_odom = self.get_clock().now()
        if (self.reset_pending and
                self.event_counts["gt_odom"] > self.reset_gt_baseline_event):
            now = self.get_clock().now()
            position_distance = math.hypot(
                float(pose.position.x), float(pose.position.y))
            position_limit = max(0.1, float(self.get_parameter(
                "reset_ground_truth_position_tolerance_m").value))
            if (speed <= float(self.get_parameter(
                    "reset_odom_speed_threshold_mps").value) and
                    position_distance <= position_limit):
                if self.reset_zero_gt_since is None:
                    self.reset_zero_gt_since = now
                stable_sec = (now - self.reset_zero_gt_since).nanoseconds / 1e9
                if stable_sec >= max(0.1, float(self.get_parameter(
                        "reset_zero_odom_stable_sec").value)):
                    self.reset_gt_confirmed = True
            else:
                self.reset_zero_gt_since = None
        self._capture_source_event("gt_odom", stamp_s)

    def _on_gt_ips(self, msg: Point) -> None:
        """Record the simulator IPS position as a second truth stream."""
        self._record_event("gt_ips")
        self.state["gt_ips_x_m"] = float(msg.x)
        self.state["gt_ips_y_m"] = float(msg.y)
        self.state["gt_ips_z_m"] = float(msg.z)

    def _on_collision_count(self, msg: Int32) -> None:
        self._record_event("collision")
        self.state["gt_collision_count"] = int(msg.data)

    def _on_imu(self, msg: Imu) -> None:
        self._record_event("imu")
        self.state["ax_mps2"] = float(msg.linear_acceleration.x)
        self.state["ay_mps2"] = float(msg.linear_acceleration.y)
        self.state["az_mps2"] = float(msg.linear_acceleration.z)
        self.state["imu_accel_norm_mps2"] = math.sqrt(
            float(msg.linear_acceleration.x) ** 2 +
            float(msg.linear_acceleration.y) ** 2 +
            float(msg.linear_acceleration.z) ** 2)
        self.state["imu_yaw_rate_radps"] = float(msg.angular_velocity.z)
        self.state["yaw_rate_radps"] = float(msg.angular_velocity.z)
        self.state["imu_yaw_rad"] = _yaw_from_quaternion(msg.orientation)
        stamp_s = self._message_stamp_s(msg)
        self.state["imu_stamp_s"] = stamp_s
        self._capture_source_event("imu", stamp_s)

    def _on_encoder(self, msg: JointState, field: str) -> None:
        side = "left" if field.startswith("left") else "right"
        self._record_event(f"{side}_encoder")
        if msg.position:
            value = float(msg.position[0])
            if math.isfinite(value):
                self.state[field] = value
                stamp_s = self._message_stamp_s(msg)
                self.state[f"{side}_encoder_stamp_s"] = stamp_s
                previous = self.last_encoder_sample[side]
                dt_field = f"{side}_encoder_dt_s"
                speed_field = f"{side}_encoder_speed_radps"
                if previous is not None:
                    previous_stamp, previous_value = previous
                    dt = stamp_s - previous_stamp
                    self.state[dt_field] = dt
                    if 1.0e-3 <= dt <= 1.0:
                        self.state[speed_field] = (value - previous_value) / dt
                    else:
                        self.state[speed_field] = math.nan
                self.last_encoder_sample[side] = (stamp_s, value)
                self._update_encoder_slip()
                self._capture_source_event(f"{side}_encoder", stamp_s)

    def _on_pose(self, msg: PoseWithCovarianceStamped, prefix: str) -> None:
        pose = msg.pose.pose
        self.state[f"x_{prefix}_m"] = float(pose.position.x)
        self.state[f"y_{prefix}_m"] = float(pose.position.y)
        self.state[f"yaw_{prefix}_rad"] = _yaw_from_quaternion(pose.orientation)
        self.state[f"{prefix}_xy_variance"] = max(
            float(msg.pose.covariance[0]), float(msg.pose.covariance[7]))
        self.state[f"{prefix}_yaw_variance"] = float(msg.pose.covariance[35])

    def _on_amcl(self, msg: PoseWithCovarianceStamped) -> None:
        self._record_event("amcl")
        self._on_pose(msg, "amcl")

    def _on_ekf(self, msg: PoseWithCovarianceStamped) -> None:
        self._record_event("ekf")
        self._on_pose(msg, "ekf")

    def _on_amcl_gpu_timing(self, msg: Float64MultiArray) -> None:
        values = list(msg.data)
        if len(values) >= 20:
            self._set("amcl_gpu_transfer_ms", values[3])
            self._set("amcl_gpu_pf_ms", values[18])
            self._set("amcl_gpu_callback_ms", values[19])
            self._set("amcl_pose_published", values[20] if len(values) > 20 else math.nan)
            self._set("amcl_cluster_weight", values[21] if len(values) > 21 else math.nan)

    def _on_amcl_kld(self, msg: Float64MultiArray) -> None:
        values = list(msg.data)
        if len(values) >= 4:
            self._set("amcl_kld_pre_particles", values[0])
            self._set("amcl_kld_bins", values[1])
            self._set("amcl_kld_target", values[2])
            self._set("amcl_kld_particles", values[3])

    def _on_controller_command(
            self, msg: AckermannDriveStamped, source: str) -> None:
        """Record command abstractions without merging their semantics."""
        speed = float(msg.drive.speed)
        acceleration = float(msg.drive.acceleration)
        steering = float(msg.drive.steering_angle)
        stamp_s = self._message_stamp_s(msg)
        self._record_event(f"{source}_command")
        if source == "speed":
            self.state["target_speed_mps"] = speed
            self.state["target_accel_mps2"] = acceleration
            self.state["controller_speed_mps"] = speed
            self.state["controller_accel_mps2"] = acceleration
            self.state["controller_steering_rad"] = steering
            self.state["speed_command_mps"] = speed
            self.state["speed_command_accel_mps2"] = acceleration
            self.state["speed_command_steering_rad"] = steering
            self.state["speed_command_stamp_s"] = stamp_s
        elif source == "acceleration":
            self.state["target_speed_mps"] = speed
            self.state["target_accel_mps2"] = acceleration
            self.state["controller_speed_mps"] = speed
            self.state["controller_accel_mps2"] = acceleration
            self.state["controller_steering_rad"] = steering
            self.state["acceleration_command_mps"] = speed
            self.state["acceleration_command_accel_mps2"] = acceleration
            self.state["acceleration_command_steering_rad"] = steering
            self.state["acceleration_command_stamp_s"] = stamp_s

    def _on_lidar(self, _msg: LaserScan) -> None:
        self._record_event("lidar")

    def _neutral(self) -> None:
        self.throttle_pub.publish(Float32(data=0.0))
        self.steering_pub.publish(Float32(data=0.0))
        self.raw_throttle_override_pub.publish(Float32(data=0.0))
        if self.reset_pub is not None:
            self.reset_pub.publish(Bool(data=False))
        self._publish_neutral_commands()

    def _publish_neutral_commands(self) -> None:
        neutral = AckermannDriveStamped()
        self.speed_drive_pub.publish(neutral)
        self.acceleration_drive_pub.publish(neutral)
        self.raw_throttle_override_pub.publish(Float32(data=0.0))

    def _finish(self, reason: str = "completed") -> None:
        if self.finished:
            return
        self.finished = True
        self.timer.cancel()
        self._neutral()
        self.stream.flush()
        self.get_logger().info(f"Finished ({reason}); Data: {self.output_path}")
        # This node is a finite diagnostics job.  Humble's executor can catch
        # SystemExit raised from a timer callback and leave the ros2-run child
        # alive after the CSV is complete.  The stream is flushed and the
        # actuator has received neutral before the process-level exit, so no
        # calibration process can remain as a hidden publisher.
        os._exit(0)

    def _command(self, kind: str, value: float, progress: float) -> None:
        if kind == "reset":
            if not self.reset_pending:
                self.reset_pending = True
                self.reset_odom_confirmed = False
                self.reset_odom_baseline_event = self.event_counts["odom"]
                self.reset_wait_start = self.get_clock().now()
                self.reset_zero_odom_since = None
                self.reset_gt_confirmed = False
                self.reset_gt_baseline_event = self.event_counts["gt_odom"]
                self.reset_zero_gt_since = None
                self.reset_signal_sent = False
                self.reset_connection_ready = False
                self.reset_signal_cleared = False
            if self.reset_pub is not None:
                if not self.reset_connection_ready:
                    # The first volatile message can be published before the
                    # bridge has discovered this subscription. Repeat true
                    # until the bridge and local epoch subscriber are both
                    # present; after discovery it is a single rising edge
                    # and cannot queue teleports.
                    self.reset_pub.publish(Bool(data=True))
                    self.reset_connection_ready = (
                        self.reset_pub.get_subscription_count() >= max(
                            1, int(self.get_parameter(
                                "reset_min_subscribers").value)))
                    self.reset_signal_sent = self.reset_connection_ready
                elif (not self.reset_signal_cleared and
                      progress * self.reset_pulse_sec >= self.reset_signal_hold_sec):
                    # Complete the pulse explicitly. Leaving the bridge's
                    # reset flag latched makes the car appear stationary
                    # while wheel/IMU values continue to change.
                    self.reset_pub.publish(Bool(data=False))
                    self.reset_signal_cleared = True
            # Do not write the previous run's ground truth into the new
            # experiment while the simulator processes the reset pulse.
            for field in (
                "gt_x_m", "gt_y_m", "gt_z_m", "gt_yaw_rad",
                "gt_vx_mps", "gt_vy_mps", "gt_vz_mps", "gt_speed_mps",
                "gt_ax_mps2", "gt_ay_mps2", "gt_accel_mps2",
                "gt_longitudinal_accel_mps2", "gt_dt_s",
                "gt_yaw_rate_radps", "gt_collision_count",
                "gt_slip_speed_mps", "gt_slip_ratio",
            ):
                self.state[field] = math.nan
            for side in ("left", "right"):
                self.last_encoder_sample[side] = None
                self.state[f"{side}_encoder_speed_radps"] = math.nan
                self.state[f"{side}_encoder_dt_s"] = math.nan
                self.state[f"{side}_encoder_stamp_s"] = math.nan
            for field in ("gt_odom_stamp_s", "odom_stamp_s"):
                self.state[field] = math.nan
            self.last_gt_sample = None
            # Keep the last callback time for the outer watchdog. The fresh
            # post-reset event baseline above still prevents old GT data from
            # confirming reset or entering the fit, while reset_pending makes
            # the watchdog defer to reset_confirmation_timeout_sec.
            self.last_odom_speed_sample = None
            self.state["speed_rate_mps2"] = math.nan
            self.state["encoder_wheel_speed_mps"] = math.nan
            # Keep the production actuator from replaying the preceding
            # closed-loop target while the simulator handles the diagnostic
            # reset.  Raw calibration outputs and the actuator otherwise
            # share the same normalized topics and would race each other.
            self._publish_neutral_commands()
            self.steering_pub.publish(Float32(data=0.0))
            self.throttle_pub.publish(Float32(data=0.0))
            return
        if kind == "raw_throttle":
            self._publish_neutral_commands()
            if self.reset_pub is not None:
                self.reset_pub.publish(Bool(data=False))
            self.steering_pub.publish(Float32(data=0.0))
            self.throttle_pub.publish(Float32(data=value))
            self.raw_throttle_override_pub.publish(Float32(data=value))
            return
        if kind == "raw_steering":
            self._publish_neutral_commands()
            if self.reset_pub is not None:
                self.reset_pub.publish(Bool(data=False))
            self.throttle_pub.publish(Float32(data=0.0))
            self.steering_pub.publish(Float32(data=value))
            self.raw_throttle_override_pub.publish(Float32(data=0.0))
            return
        if kind == "raw_steering_with_throttle":
            self._publish_neutral_commands()
            if self.reset_pub is not None:
                self.reset_pub.publish(Bool(data=False))
            throttle = float(self.get_parameter("steering_test_throttle").value)
            self.throttle_pub.publish(Float32(data=throttle))
            self.steering_pub.publish(Float32(data=value))
            self.raw_throttle_override_pub.publish(Float32(data=throttle))
            return

        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        if kind == "speed_ramp" and self.phase_index > 0:
            previous = self.phases[self.phase_index - 1][2]
            value = previous + progress * (value - previous)
        if kind == "acceleration_steps":
            # Acceleration mode is a separate abstraction boundary: MPC's
            # physical acceleration target belongs in AckermannDrive's
            # acceleration field, while speed is only diagnostic metadata.
            drive.drive.speed = max(0.0, float(self.state.get("speed_mps", 0.0)))
            drive.drive.acceleration = value
            self.state["target_accel_mps2"] = value
            self.raw_throttle_override_pub.publish(Float32(data=math.nan))
            if self.reset_pub is not None:
                self.reset_pub.publish(Bool(data=False))
            self.acceleration_drive_pub.publish(drive)
        else:
            drive.drive.speed = value
            self.state["target_speed_mps"] = value
            self.raw_throttle_override_pub.publish(Float32(data=math.nan))
            if self.reset_pub is not None:
                self.reset_pub.publish(Bool(data=False))
            self.speed_drive_pub.publish(drive)

    def _ground_truth_boundary_reached(self) -> bool:
        """Return true when a diagnostic run approaches the open-scene edge."""
        if not bool(self.get_parameter("ground_truth_boundary_guard_enabled").value):
            return False
        x = self.state.get("gt_x_m", math.nan)
        y = self.state.get("gt_y_m", math.nan)
        distance = math.hypot(float(x), float(y))
        limit = max(1.0, float(self.get_parameter(
            "ground_truth_boundary_distance_m").value))
        return math.isfinite(distance) and distance >= limit

    def _insert_boundary_reset(self, now) -> bool:
        """Restart the current long phase after a diagnostics-only reset."""
        if self.reset_pub is None:
            self.get_logger().error(
                "Open-ground boundary reached but reset_between_steps is disabled")
            self._finish()
            return True
        phase, kind, value, duration = self.phases[self.phase_index]
        if kind == "reset" or phase == "settle":
            return False
        settle = float(self.get_parameter("zero_settle_sec").value)
        brake_timeout = max(1.0, float(self.get_parameter(
            "grid_brake_timeout_sec").value))
        self.phases[self.phase_index:self.phase_index] = [
            # Never teleport a moving car merely because the finite open
            # plane was reached. Boundary recovery obeys the same stop gate
            # as every normal grid point.
            ("boundary_brake", "raw_throttle", 0.0, brake_timeout),
            ("boundary_reset", "reset", 1.0, self.reset_pulse_sec),
            ("settle", "raw_throttle", 0.0, settle),
        ]
        self.boundary_reset_count += 1
        self.phase_start = now
        self.get_logger().warn(
            f"Ground-truth boundary guard at {math.hypot(float(self.state['gt_x_m']), float(self.state['gt_y_m'])):.1f} m; "
            f"resetting and restarting {phase} (guard reset {self.boundary_reset_count})")
        return True

    def _update_speed_settled(self, now) -> bool:
        """Return true only after local measured speed has settled."""
        baseline = self.speed_settle_event_baseline or {}
        fresh_gt = self.event_counts["gt_odom"] > baseline.get("gt_odom", -1)
        rate = self.state.get("gt_speed_rate_mps2", math.nan)
        fresh_odom = self.event_counts["odom"] > baseline.get("odom", -1)
        if not (fresh_gt and math.isfinite(rate)):
            # Keep the gate usable in non-GT diagnostic profiles, while the
            # identification_grid path always prefers simulator truth.
            fresh_gt = False
            rate = self.state.get("speed_rate_mps2", math.nan)
            fresh_odom = self.event_counts["odom"] > baseline.get("odom", -1)
        fresh_motion = fresh_gt or fresh_odom
        rate_limit = max(0.0, float(self.get_parameter(
            "throttle_settle_speed_rate_mps2").value))
        instant_stable = fresh_motion and math.isfinite(rate) and abs(rate) <= rate_limit
        self.state["speed_settle_rate_mps2"] = rate
        if instant_stable:
            if self.speed_settle_since is None:
                self.speed_settle_since = now
            stable_elapsed = (now - self.speed_settle_since).nanoseconds / 1e9
        else:
            self.speed_settle_since = None
            stable_elapsed = 0.0
        confirmed = instant_stable and stable_elapsed >= max(0.1, float(
            self.get_parameter("throttle_settle_stable_sec").value))
        self.state["speed_settled"] = float(confirmed)
        self.state["speed_settle_elapsed_s"] = stable_elapsed
        return confirmed

    def _update_target_speed_settled(self, now) -> bool:
        """Require target speed and acceleration rate to settle before brake."""
        baseline = self.speed_settle_event_baseline or {}
        fresh_gt = self.event_counts["gt_odom"] > baseline.get("gt_odom", -1)
        fresh_odom = self.event_counts["odom"] > baseline.get("odom", -1)
        target = self.state.get("target_speed_mps", math.nan)
        truth_speed = self.state.get("gt_speed_mps", math.nan)
        truth_rate = self.state.get("gt_speed_rate_mps2", math.nan)
        if fresh_gt and math.isfinite(truth_speed) and math.isfinite(truth_rate):
            measured = truth_speed
            rate = truth_rate
        else:
            measured = self.state.get("speed_mps", math.nan)
            rate = self.state.get("speed_rate_mps2", math.nan)
        error_limit = max(0.0, float(self.get_parameter(
            "speed_settle_error_mps").value))
        rate_limit = max(0.0, float(self.get_parameter(
            "throttle_settle_speed_rate_mps2").value))
        instant_stable = (
            (fresh_gt or fresh_odom) and
            all(math.isfinite(value) for value in (target, measured, rate)) and
            abs(measured - target) <= error_limit and
            abs(rate) <= rate_limit
        )
        self.state["speed_settle_rate_mps2"] = rate
        if instant_stable:
            if self.speed_settle_since is None:
                self.speed_settle_since = now
            stable_elapsed = (now - self.speed_settle_since).nanoseconds / 1e9
        else:
            self.speed_settle_since = None
            stable_elapsed = 0.0
        confirmed = instant_stable and stable_elapsed >= max(0.1, float(
            self.get_parameter("throttle_settle_stable_sec").value))
        self.state["speed_settled"] = float(confirmed)
        self.state["speed_settle_elapsed_s"] = stable_elapsed
        return confirmed

    def _update_brake_stop(self, now) -> bool:
        """Require independent fresh motion evidence before leaving braking.

        Encoder speed can become zero while the chassis is still sliding, so
        encoder zero alone is deliberately insufficient.  The IMU must also
        show no translational or yaw motion, while simulator truth verifies
        that the chassis itself has stopped.  All conditions must remain true
        continuously for the configured stable interval.
        """
        baseline = self.brake_event_baseline or {}
        fresh_left = self.event_counts["left_encoder"] > baseline.get(
            "left_encoder", -1)
        fresh_right = self.event_counts["right_encoder"] > baseline.get(
            "right_encoder", -1)
        fresh_imu = self.event_counts["imu"] > baseline.get("imu", -1)
        fresh_odom = self.event_counts["odom"] > baseline.get("odom", -1)
        fresh_gt = self.event_counts["gt_odom"] > baseline.get("gt_odom", -1)

        encoder_limit = max(0.0, float(self.get_parameter(
            "brake_stop_encoder_speed_radps").value))
        imu_accel_limit = max(0.0, float(self.get_parameter(
            "brake_stop_imu_accel_mps2").value))
        imu_yaw_limit = max(0.0, float(self.get_parameter(
            "brake_stop_imu_yaw_rate_radps").value))
        odom_speed_limit = max(0.0, float(self.get_parameter(
            "brake_stop_odom_speed_mps").value))

        left_speed = self.state.get("left_encoder_speed_radps", math.nan)
        right_speed = self.state.get("right_encoder_speed_radps", math.nan)
        encoder_stopped = (
            fresh_left and fresh_right and
            math.isfinite(left_speed) and math.isfinite(right_speed) and
            abs(left_speed) <= encoder_limit and
            abs(right_speed) <= encoder_limit
        )
        imu_accel = self.state.get("imu_accel_norm_mps2", math.nan)
        imu_yaw_rate = self.state.get("imu_yaw_rate_radps", math.nan)
        imu_stopped = (
            fresh_imu and math.isfinite(imu_accel) and
            math.isfinite(imu_yaw_rate) and
            imu_accel <= imu_accel_limit and
            abs(imu_yaw_rate) <= imu_yaw_limit
        )
        odom_speed = self.state.get("speed_mps", math.nan)
        odom_stopped = (
            fresh_odom and math.isfinite(odom_speed) and
            odom_speed <= odom_speed_limit
        )
        gt_speed = self.state.get("gt_speed_mps", math.nan)
        gt_stopped = (
            fresh_gt and math.isfinite(gt_speed) and
            gt_speed <= float(self.get_parameter(
                "brake_stop_gt_speed_mps").value)
        )

        self.state["brake_encoder_stopped"] = float(encoder_stopped)
        self.state["brake_imu_stopped"] = float(imu_stopped)
        self.state["brake_odom_stopped"] = float(odom_stopped)
        self.state["brake_gt_stopped"] = float(gt_stopped)

        # Ground truth is permitted only as the diagnostic movement oracle for
        # this test. It is recorded and never fed into /odom, EKF, AMCL, or a
        # production controller.
        all_stopped = encoder_stopped and imu_stopped and gt_stopped
        if all_stopped:
            if self.brake_stop_since is None:
                self.brake_stop_since = now
            stable_elapsed = (now - self.brake_stop_since).nanoseconds / 1e9
        else:
            self.brake_stop_since = None
            stable_elapsed = 0.0
        confirmed = all_stopped and stable_elapsed >= max(0.1, float(
            self.get_parameter("brake_stop_stable_sec").value))
        self.state["brake_stop_confirmed"] = float(confirmed)
        self.state["brake_stop_elapsed_s"] = stable_elapsed
        return confirmed

    def _tick(self) -> None:
        now = self.get_clock().now()
        elapsed = (now - self.start).nanoseconds / 1e9
        self.state["stamp_s"] = now.nanoseconds * 1.0e-9
        self.state["time_s"] = elapsed
        # Source callbacks can arrive in bursts between timer invocations.
        # Flush their queued snapshots before the timer row so no native
        # sensor event is lost merely because the recorder cadence is lower.
        self._flush_source_event_rows()

        if self.mode != "sensor_record":
            telemetry_time = (self.last_gt_odom
                              if self.mode == "identification_grid"
                              else self.last_odom)
            if telemetry_time is None:
                self._neutral()
                # A diagnostic reset intentionally invalidates the recorded
                # sensor state while the simulator processes its one-shot
                # reset pulse. The reset confirmation gate owns that wait;
                # do not report it as a process-start telemetry failure.
                if not self.reset_pending and elapsed > self.startup_timeout:
                    self._finish("startup telemetry timeout")
                return
            if (not self.reset_pending and
                    (now - telemetry_time).nanoseconds / 1e9 > self.telemetry_timeout):
                self._finish("telemetry timeout")
                return
            observed_speed = (self.state.get("gt_speed_mps", math.nan)
                              if self.mode == "identification_grid"
                              else self.state.get("speed_mps", math.nan))
            # identification_grid begins with an explicit simulator reset.
            # Let that planned reset run even when a stale session left the
            # vehicle outside the open scene or above the test envelope; the
            # normal safety guard applies from the first post-reset phase.
            planned_reset = (
                self.phase_index < len(self.phases) and
                self.phases[self.phase_index][1] == "reset")
            if (math.isfinite(observed_speed) and observed_speed > self.max_speed
                    and not planned_reset):
                self._finish("ground-truth speed safety limit")
                return

        if self.mode == "sensor_record":
            self.state["phase"] = "record"
            self.writer.writerow(self.state)
            self.stream.flush()
            if self.duration > 0.0 and elapsed >= self.duration:
                self._finish("recording duration reached")
            return

        if self.phase_index >= len(self.phases):
            self._finish("phase list completed")
            return

        phase, kind, value, duration = self.phases[self.phase_index]
        phase_elapsed = (now - self.phase_start).nanoseconds / 1e9

        if phase != self.active_phase:
            self.active_phase = phase
            if phase.startswith(("grid_brake", "boundary_brake")):
                self.brake_stop_since = None
                self.brake_event_baseline = {
                    "left_encoder": self.event_counts["left_encoder"],
                    "right_encoder": self.event_counts["right_encoder"],
                    "imu": self.event_counts["imu"],
                    "odom": self.event_counts["odom"],
                    "gt_odom": self.event_counts["gt_odom"],
                }
                for field in (
                    "brake_encoder_stopped", "brake_imu_stopped",
                    "brake_odom_stopped", "brake_gt_stopped",
                    "brake_stop_confirmed",
                ):
                    self.state[field] = 0.0
                self.state["brake_stop_elapsed_s"] = 0.0
            elif phase.startswith("grid_base_") or phase.startswith("grid_throttle_"):
                self.speed_settle_since = None
                self.speed_settle_event_baseline = {
                    "odom": self.event_counts["odom"],
                    "gt_odom": self.event_counts["gt_odom"],
                }
                self.state["speed_settled"] = 0.0
                self.state["speed_settle_rate_mps2"] = math.nan
                self.state["speed_settle_elapsed_s"] = 0.0
            elif phase.startswith("speed_"):
                self.speed_settle_since = None
                self.speed_settle_event_baseline = {
                    "odom": self.event_counts["odom"],
                    "gt_odom": self.event_counts["gt_odom"],
                }
                self.state["speed_settled"] = 0.0
                self.state["speed_settle_rate_mps2"] = math.nan
                self.state["speed_settle_elapsed_s"] = 0.0

        # Long direct-throttle phases must be allowed to settle fully, but the
        # simulator's open plane is finite. Reset and restart the same phase
        # before reaching its edge so the requested hold duration remains
        # meaningful and no out-of-world tail is used in the fit.
        if (kind != "reset" and not phase.startswith("grid_settle") and
                not phase.startswith(("grid_brake", "boundary_brake")) and
                phase != "settle" and
                self._ground_truth_boundary_reached()):
            self._insert_boundary_reset(now)
            return

        # Do not leave the reset pulse until the bridge has discovered the
        # publisher. With volatile QoS, advancing after the nominal pulse
        # duration would otherwise turn a missed reset into a false local
        # odometry confirmation.
        if kind == "reset":
            self.state["phase"] = phase
            progress = min(max(phase_elapsed / max(duration, 1e-6), 0.0), 1.0)
            self._command(kind, value, progress)
            self.writer.writerow(self.state)
            self.stream.flush()
            reset_timeout = max(duration, float(self.get_parameter(
                "reset_confirmation_timeout_sec").value))
            if (self.reset_connection_ready and phase_elapsed >= duration):
                self.phase_index += 1
                self.phase_start = now
            elif phase_elapsed >= reset_timeout:
                self.get_logger().error(
                    "Reset bridge subscription was not discovered in time")
                self._finish("reset publisher connection timeout")
            return

        # A simulator teleport can complete before the official encoder and
        # IMU streams deliver their first post-reset sample. Do not start a
        # target-speed phase while /odom still contains the previous run's
        # speed. Wait for a fresh near-zero odometry sample and then give the
        # configured settle interval its full duration.
        if (phase == "settle" or phase.startswith("grid_settle")) and self.reset_pending:
            # Ground truth is the only reliable confirmation that the
            # simulator itself teleported back to spawn. Local /odom can
            # reset independently and must not authorize a new test segment.
            reset_confirmed = self.reset_gt_confirmed
            if reset_confirmed:
                self.reset_pending = False
                self.reset_odom_confirmed = False
                self.reset_gt_confirmed = False
                self.reset_wait_start = None
                self.phase_start = now
                phase_elapsed = 0.0
            else:
                wait_start = self.reset_wait_start or now
                wait_elapsed = (now - wait_start).nanoseconds / 1e9
                timeout = float(self.get_parameter(
                    "reset_confirmation_timeout_sec").value)
                if wait_elapsed >= max(0.1, timeout):
                    self.get_logger().error(
                        "No fresh near-zero diagnostic motion sample after reset")
                    self._finish("reset confirmation timeout")
                    return
                self.state["phase"] = phase
                self._command("raw_throttle", 0.0, 0.0)
                self.writer.writerow(self.state)
                self.stream.flush()
                return

        if phase.startswith("grid_base_") or phase.startswith("grid_throttle_"):
            self.state["phase"] = phase
            progress = min(max(phase_elapsed / max(duration, 1e-6), 0.0), 1.0)
            self._command(kind, value, progress)
            settled = self._update_speed_settled(now)
            self.writer.writerow(self.state)
            self.stream.flush()
            minimum_hold = duration
            if phase.startswith("grid_throttle_"):
                minimum_hold = max(duration, float(self.get_parameter(
                    "throttle_settle_min_sec").value))
            if settled and phase_elapsed >= minimum_hold:
                self.phase_index += 1
                self.phase_start = now
            elif phase_elapsed >= max(duration, float(self.get_parameter(
                    "throttle_settle_timeout_sec").value)):
                self.get_logger().error(
                    f"Throttle phase {phase} did not settle within "
                    f"{self.get_parameter('throttle_settle_timeout_sec').value}s; "
                    "refusing to start braking")
                self._finish("throttle settle timeout")
            return

        if phase.startswith(("grid_brake", "boundary_brake")):
            self.state["phase"] = phase
            self._command("raw_throttle", 0.0, 0.0)
            confirmed = self._update_brake_stop(now)
            self.writer.writerow(self.state)
            self.stream.flush()
            if confirmed:
                self.phase_index += 1
                self.phase_start = now
            elif phase_elapsed >= duration:
                self.get_logger().error(
                    f"Brake stop confirmation timed out after {duration:.1f}s; "
                    "refusing to reset while motion remains")
                self._finish("brake stop confirmation timeout")
            return

        if phase.startswith("speed_"):
            self.state["phase"] = phase
            progress = min(max(phase_elapsed / max(duration, 1e-6), 0.0), 1.0)
            self._command(kind, value, progress)
            settled = self._update_target_speed_settled(now)
            self.writer.writerow(self.state)
            self.stream.flush()
            if settled and phase_elapsed >= duration:
                self.phase_index += 1
                self.phase_start = now
            elif phase_elapsed >= duration + max(0.1, float(self.get_parameter(
                    "speed_settle_timeout_sec").value)):
                self.get_logger().error(
                    f"Speed phase {phase} did not settle within "
                    f"{self.get_parameter('speed_settle_timeout_sec').value}s "
                    "after its minimum hold; refusing to start braking")
                self._finish("speed target settle timeout")
            return

        if phase_elapsed >= duration:
            self.phase_index += 1
            self.phase_start = now
            return

        self.state["phase"] = phase
        progress = min(max(phase_elapsed / max(duration, 1e-6), 0.0), 1.0)
        self._command(kind, value, progress)
        self.writer.writerow(self.state)
        self.stream.flush()

    def destroy_node(self):
        if not self.finished and rclpy.ok(context=self.context):
            self._neutral()
        self._flush_source_event_rows()
        if not self.stream.closed:
            self.stream.flush()
            self.stream.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Calibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

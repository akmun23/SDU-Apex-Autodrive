#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <ament_index_cpp/get_package_prefix.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "f1tenth_localization/odometry_observer.hpp"
#include "f1tenth_localization/physx_odom_runtime.hpp"
#include "f1tenth_localization/sensor_packet_assembler.hpp"

namespace
{

// The simulator physics is sampled by the 40 Hz bridge once per 25 ms
// physics interval. ROS message stamps are assigned at bridge receipt, so
// their jitter is not a valid integration timestep.
constexpr int64_t kNominalPhysicsPacketPeriodNs = 25'000'000;

double wrap_angle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

geometry_msgs::msg::Quaternion quaternion_from_yaw(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  const double half = 0.5 * yaw;
  q.z = std::sin(half);
  q.w = std::cos(half);
  return q;
}

int64_t stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL +
    static_cast<int64_t>(stamp.nanosec);
}

}  // namespace

class SensorOdometryNode final : public rclcpp::Node
{
public:
  SensorOdometryNode()
  : Node("sensor_odometry"),
    observer_(default_observer_config())
  {
    const auto observer_defaults = default_observer_config();
    declare_parameter("left_encoder_topic", "/autodrive/roboracer_1/left_encoder");
    declare_parameter("right_encoder_topic", "/autodrive/roboracer_1/right_encoder");
    declare_parameter("imu_topic", "/autodrive/roboracer_1/imu");
    declare_parameter("steering_feedback_topic", "/autodrive/roboracer_1/steering");
    declare_parameter("throttle_feedback_topic", "/autodrive/roboracer_1/throttle");
    declare_parameter(
      "steering_command_topic", "/autodrive/roboracer_1/steering_command");
    declare_parameter(
      "throttle_command_topic", "/autodrive/roboracer_1/throttle_command");
    declare_parameter("use_physx_odom_velocity", false);
    declare_parameter("odom_topic", "/odom");
    declare_parameter("diagnostics_topic", "/odom/diagnostics");

    declare_parameter("odom_frame", "odom");
    declare_parameter("base_frame", "base_link");
    declare_parameter("publish_tf", true);
    declare_parameter("lidar_frame", "lidar");
    declare_parameter("imu_frame", "imu");
    declare_parameter("lidar_x_m", 0.2733);
    declare_parameter("lidar_y_m", 0.0);
    declare_parameter("lidar_z_m", 0.096);
    declare_parameter("imu_x_m", 0.08);
    declare_parameter("imu_y_m", 0.0);
    declare_parameter("imu_z_m", 0.055);
    declare_parameter(
      "imu_acceleration_reference_x_m",
      observer_defaults.imu_acceleration_reference_x_m);
    declare_parameter("physx_velocity_blend", observer_defaults.physx_velocity_blend);
    declare_parameter(
      "physx_forward_velocity_blend", observer_defaults.physx_forward_velocity_blend);
    declare_parameter(
      "physx_low_speed_velocity_blend",
      observer_defaults.physx_low_speed_velocity_blend);
    declare_parameter(
      "physx_low_speed_blend_transition_mps",
      observer_defaults.physx_low_speed_blend_transition_mps);
    declare_parameter("imu_orientation_correction_gain", 1.0);
    declare_parameter("max_imu_orientation_step_rad", 0.30);
    declare_parameter("pose_xy_variance", 0.01);
    // An isolated bridge gap is diagnostic evidence, not a permanent loss of
    // localization. Keep its covariance conservative but below the controller
    // stop gate; the EKF treats source covariance as a floor for the epoch.
    declare_parameter("timing_degraded_pose_xy_variance", 0.01);
    declare_parameter("pose_yaw_variance", 0.0001);
    declare_parameter("twist_linear_variance", 0.04);
    declare_parameter("twist_yaw_variance", 0.04);

    declare_parameter("wheel_radius_m", observer_defaults.wheel_radius_m);
    declare_parameter("wheel_speed_scale", observer_defaults.wheel_speed_scale);
    declare_parameter<std::vector<double>>(
      "wheel_speed_scale_speeds_mps", observer_defaults.wheel_speed_scale_speeds_mps);
    declare_parameter<std::vector<double>>(
      "wheel_speed_scale_values", observer_defaults.wheel_speed_scale_values);
    declare_parameter("reset_encoder_jump_rad", observer_defaults.reset_encoder_jump_rad);
    declare_parameter("wheel_speed_window_s", observer_defaults.wheel_speed_window_s);
    declare_parameter("normal_packet_dt_max_s", observer_defaults.normal_packet_dt_max_s);
    declare_parameter("decel_detect_ax_mps2", observer_defaults.decel_detect_ax_mps2);
    declare_parameter("decel_ax_scale", observer_defaults.decel_ax_scale);
    declare_parameter("decel_ax_offset_mps2", observer_defaults.decel_ax_offset_mps2);
    declare_parameter(
      "wheel_dropout_positive_ax_max_mps2",
      observer_defaults.wheel_dropout_positive_ax_max_mps2);
    declare_parameter(
      "wheel_update_ax_abs_max_mps2", observer_defaults.wheel_update_ax_abs_max_mps2);
    declare_parameter("wheel_freeze_speed_mps", observer_defaults.wheel_freeze_speed_mps);
    declare_parameter(
      "wheel_recovery_launch_speed_mps", observer_defaults.wheel_recovery_launch_speed_mps);
    declare_parameter(
      "wheel_recovery_launch_innovation_mps",
      observer_defaults.wheel_recovery_launch_innovation_mps);
    declare_parameter(
      "wheel_recovery_launch_wheel_speed_mps",
      observer_defaults.wheel_recovery_launch_wheel_speed_mps);
    declare_parameter(
      "stationary_speed_threshold_mps", observer_defaults.stationary_speed_threshold_mps);
    declare_parameter(
      "wheel_burst_disagreement_mps", observer_defaults.wheel_burst_disagreement_mps);
    declare_parameter(
      "wheel_burst_catchup_accel_mps2", observer_defaults.wheel_burst_catchup_accel_mps2);
    declare_parameter(
      "wheel_burst_catchup_max_mps", observer_defaults.wheel_burst_catchup_max_mps);
    declare_parameter(
      "allow_turn_current_packet_recovery", observer_defaults.allow_turn_current_packet_recovery);
    declare_parameter(
      "turn_current_packet_max_increase_mps",
      observer_defaults.turn_current_packet_max_increase_mps);
    declare_parameter(
      "turn_current_packet_max_decrease_mps",
      observer_defaults.turn_current_packet_max_decrease_mps);
    declare_parameter(
      "turn_wheel_update_start_speed_mps",
      observer_defaults.turn_wheel_update_start_speed_mps);
    declare_parameter(
      "turn_wheel_update_min_ax_mps2",
      observer_defaults.turn_wheel_update_min_ax_mps2);
    declare_parameter(
      "turn_wheel_update_innovation_threshold_mps",
      observer_defaults.turn_wheel_update_innovation_threshold_mps);
    declare_parameter(
      "turn_wheel_update_slip_beta",
      observer_defaults.turn_wheel_update_slip_beta);
    declare_parameter(
      "use_turn_speed_bias_model", observer_defaults.use_turn_speed_bias_model);
    declare_parameter(
      "turn_speed_bias_constant_mps", observer_defaults.turn_speed_bias_constant_mps);
    declare_parameter("turn_speed_bias_speed_mps", observer_defaults.turn_speed_bias_speed_mps);
    declare_parameter(
      "turn_speed_bias_speed_squared_mps", observer_defaults.turn_speed_bias_speed_squared_mps);
    declare_parameter(
      "turn_speed_bias_yaw_rate_abs_mps",
      observer_defaults.turn_speed_bias_yaw_rate_abs_mps);
    declare_parameter(
      "turn_speed_bias_yaw_rate_squared_mps",
      observer_defaults.turn_speed_bias_yaw_rate_squared_mps);
    declare_parameter(
      "turn_speed_bias_speed_yaw_rate_abs_mps",
      observer_defaults.turn_speed_bias_speed_yaw_rate_abs_mps);
    declare_parameter("turn_speed_bias_max_mps", observer_defaults.turn_speed_bias_max_mps);
    declare_parameter(
      "turn_speed_bias_start_speed_mps",
      observer_defaults.turn_speed_bias_start_speed_mps);
    declare_parameter(
      "turn_speed_bias_full_speed_mps",
      observer_defaults.turn_speed_bias_full_speed_mps);
    declare_parameter(
      "use_turn_speed_bias_lateral_accel_gate",
      observer_defaults.use_turn_speed_bias_lateral_accel_gate);
    declare_parameter(
      "turn_speed_bias_lateral_accel_gate_start_mps2",
      observer_defaults.turn_speed_bias_lateral_accel_gate_start_mps2);
    declare_parameter(
      "turn_speed_bias_lateral_accel_gate_full_mps2",
      observer_defaults.turn_speed_bias_lateral_accel_gate_full_mps2);
    declare_parameter(
      "turn_slip_speed_bias_lateral_accel_threshold_mps2",
      observer_defaults.turn_slip_speed_bias_lateral_accel_threshold_mps2);
    declare_parameter(
      "turn_slip_speed_bias_gain_mps_per_mps2",
      observer_defaults.turn_slip_speed_bias_gain_mps_per_mps2);
    declare_parameter(
      "turn_slip_speed_bias_start_speed_mps",
      observer_defaults.turn_slip_speed_bias_start_speed_mps);
    declare_parameter(
      "turn_slip_speed_bias_full_speed_mps",
      observer_defaults.turn_slip_speed_bias_full_speed_mps);
    declare_parameter(
      "turn_slip_speed_bias_fade_start_mps",
      observer_defaults.turn_slip_speed_bias_fade_start_mps);
    declare_parameter(
      "turn_slip_speed_bias_fade_end_mps",
      observer_defaults.turn_slip_speed_bias_fade_end_mps);
    declare_parameter(
      "use_coherent_packet_velocity_for_pose",
      observer_defaults.use_coherent_packet_velocity_for_pose);
    declare_parameter(
      "coherent_packet_pose_blend", observer_defaults.coherent_packet_pose_blend);
    declare_parameter(
      "wheel_speed_slew_limit_mps2", observer_defaults.wheel_speed_slew_limit_mps2);
    declare_parameter("wheel_update_beta", observer_defaults.wheel_update_beta);
    declare_parameter("stationary_hold_s", observer_defaults.stationary_hold_s);
    declare_parameter(
      "stationary_ax_abs_max_mps2", observer_defaults.stationary_ax_abs_max_mps2);
    declare_parameter(
      "stationary_ay_abs_max_mps2", observer_defaults.stationary_ay_abs_max_mps2);
    declare_parameter(
      "stationary_yaw_rate_abs_max_radps",
      observer_defaults.stationary_yaw_rate_abs_max_radps);
    declare_parameter(
      "turn_enter_yaw_rate_radps", observer_defaults.turn_enter_yaw_rate_radps);
    declare_parameter("turn_enter_abs_ay_mps2", observer_defaults.turn_enter_abs_ay_mps2);
    declare_parameter("turn_exit_yaw_rate_radps", observer_defaults.turn_exit_yaw_rate_radps);
    declare_parameter("turn_exit_abs_ay_mps2", observer_defaults.turn_exit_abs_ay_mps2);
    declare_parameter("turn_exit_hold_s", observer_defaults.turn_exit_hold_s);
    declare_parameter(
      "turn_wheel_braking_ax_mps2", observer_defaults.turn_wheel_braking_ax_mps2);
    declare_parameter(
      "integrate_lateral_acceleration_in_turn",
      observer_defaults.integrate_lateral_acceleration_in_turn);
    declare_parameter(
      "use_high_yaw_wheel_slip_observer",
      observer_defaults.use_high_yaw_wheel_slip_observer);
    declare_parameter(
      "high_yaw_wheel_slip_enter_radps",
      observer_defaults.high_yaw_wheel_slip_enter_radps);
    declare_parameter(
      "high_yaw_wheel_slip_exit_radps",
      observer_defaults.high_yaw_wheel_slip_exit_radps);
    declare_parameter(
      "high_yaw_wheel_slip_exit_hold_s",
      observer_defaults.high_yaw_wheel_slip_exit_hold_s);
    declare_parameter(
      "high_yaw_wheel_slip_innovation_mps",
      observer_defaults.high_yaw_wheel_slip_innovation_mps);
    declare_parameter(
      "use_learned_u_residual_model",
      observer_defaults.use_learned_u_residual_model);
    declare_parameter(
      "learned_u_residual_min_speed_mps",
      observer_defaults.learned_u_residual_min_speed_mps);
    declare_parameter(
      "learned_u_residual_max_speed_mps",
      observer_defaults.learned_u_residual_max_speed_mps);
    declare_parameter<std::vector<double>>(
      "u_correction_support_speed_edges_mps",
      observer_defaults.u_correction_support_speed_edges_mps);
    declare_parameter<std::vector<double>>(
      "u_correction_support_max_abs_steering_rad",
      observer_defaults.u_correction_support_max_abs_steering_rad);
    declare_parameter(
      "learned_u_residual_intercept_mps",
      observer_defaults.learned_u_residual_intercept_mps);
    declare_parameter<std::vector<double>>(
      "learned_u_residual_coefficients",
      observer_defaults.learned_u_residual_coefficients);
    declare_parameter("use_unity_actuator_u_experts",
      observer_defaults.use_unity_actuator_u_experts);
    declare_parameter("unity_actuator_u_min_speed_mps",
      observer_defaults.unity_actuator_u_min_speed_mps);
    declare_parameter("unity_actuator_u_max_speed_mps",
      observer_defaults.unity_actuator_u_max_speed_mps);
    declare_parameter("unity_zero_throttle_intercept_mps",
      observer_defaults.unity_zero_throttle_intercept_mps);
    declare_parameter<std::vector<double>>(
      "unity_zero_throttle_coefficients",
      observer_defaults.unity_zero_throttle_coefficients);
    declare_parameter("unity_nonzero_throttle_intercept_mps",
      observer_defaults.unity_nonzero_throttle_intercept_mps);
    declare_parameter<std::vector<double>>(
      "unity_nonzero_throttle_coefficients",
      observer_defaults.unity_nonzero_throttle_coefficients);
    declare_parameter("use_high_steer_unity_actuator_u_expert",
      observer_defaults.use_high_steer_unity_actuator_u_expert);
    declare_parameter("high_steer_unity_actuator_u_min_speed_mps",
      observer_defaults.high_steer_unity_actuator_u_min_speed_mps);
    declare_parameter("high_steer_unity_actuator_u_max_speed_mps",
      observer_defaults.high_steer_unity_actuator_u_max_speed_mps);
    declare_parameter("high_steer_unity_actuator_u_max_speed_low_throttle_mps",
      observer_defaults.high_steer_unity_actuator_u_max_speed_low_throttle_mps);
    declare_parameter("high_steer_unity_actuator_u_throttle_split",
      observer_defaults.high_steer_unity_actuator_u_throttle_split);
    declare_parameter("high_steer_unity_actuator_u_min_abs_yaw_rate_radps",
      observer_defaults.high_steer_unity_actuator_u_min_abs_yaw_rate_radps);
    declare_parameter("high_steer_unity_actuator_u_min_abs_steering_rad",
      observer_defaults.high_steer_unity_actuator_u_min_abs_steering_rad);
    declare_parameter("high_steer_unity_actuator_u_max_abs_steering_rad",
      observer_defaults.high_steer_unity_actuator_u_max_abs_steering_rad);
    declare_parameter("high_steer_unity_actuator_u_min_throttle_feedback",
      observer_defaults.high_steer_unity_actuator_u_min_throttle_feedback);
    declare_parameter("high_steer_unity_actuator_u_max_throttle_feedback",
      observer_defaults.high_steer_unity_actuator_u_max_throttle_feedback);
    declare_parameter("high_steer_unity_actuator_u_intercept_mps",
      observer_defaults.high_steer_unity_actuator_u_intercept_mps);
    declare_parameter<std::vector<double>>(
      "high_steer_unity_actuator_u_coefficients",
      observer_defaults.high_steer_unity_actuator_u_coefficients);
    declare_parameter("unity_actuator_feedback_settle_delay_ms", 0.0);
    declare_parameter(
      "use_kinematic_lateral_slip_model", observer_defaults.use_kinematic_lateral_slip_model);
    declare_parameter(
      "lateral_velocity_yaw_rate_gain_m",
      observer_defaults.lateral_velocity_yaw_rate_gain_m);
    declare_parameter(
      "lateral_velocity_speed_yaw_rate_gain_s",
      observer_defaults.lateral_velocity_speed_yaw_rate_gain_s);
    declare_parameter(
      "lateral_velocity_max_mps", observer_defaults.lateral_velocity_max_mps);
    declare_parameter(
      "lateral_velocity_reference_forward_offset_m",
      observer_defaults.lateral_velocity_reference_forward_offset_m);
    declare_parameter("max_imu_ax_abs_mps2", observer_defaults.max_imu_ax_abs_mps2);

    observer_config_ = load_observer_config();
    observer_ = f1tenth_localization::OdometryObserver(observer_config_);
    use_physx_odom_velocity_ = get_parameter("use_physx_odom_velocity").as_bool();
    if (use_physx_odom_velocity_) {
      const std::filesystem::path prefix(
        ament_index_cpp::get_package_prefix("f1tenth_localization"));
      const std::filesystem::path share(
        ament_index_cpp::get_package_share_directory("f1tenth_localization"));
      const auto executable = prefix / "lib/f1tenth_localization/physx_native_vehicle_step";
      const auto initial_state = share / "config/physx_odom_reset_initial.json";
      const auto shape_audit = share / "config/physx_odom_vehicle_shapes.json";
      if (!std::filesystem::is_regular_file(executable) ||
        !std::filesystem::is_regular_file(initial_state) ||
        !std::filesystem::is_regular_file(shape_audit))
      {
        throw std::runtime_error("installed PhysX Odom runtime asset is missing");
      }
      physx_odom_runtime_ = std::make_unique<f1tenth_localization::PhysxOdomRuntime>(
        executable.string(), initial_state.string(), shape_audit.string());
      RCLCPP_INFO(
        get_logger(),
        "PhysX Odom predictor enabled: 25 x 1 ms steps, forward-velocity blend %.3f; lateral blend %.3f (low-speed %.3f fading by %.2f m/s)",
        observer_config_.physx_forward_velocity_blend,
        observer_config_.physx_velocity_blend,
        observer_config_.physx_low_speed_velocity_blend,
        observer_config_.physx_low_speed_blend_transition_mps);
    }
    actuator_feedback_settle_delay_ms_ = get_parameter(
      "unity_actuator_feedback_settle_delay_ms").as_double();
    if (!std::isfinite(actuator_feedback_settle_delay_ms_) ||
      actuator_feedback_settle_delay_ms_ < 0.0 ||
      actuator_feedback_settle_delay_ms_ > 10.0)
    {
      throw std::invalid_argument(
              "unity_actuator_feedback_settle_delay_ms must be in [0, 10]");
    }
    odom_frame_ = get_parameter("odom_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();
    publish_tf_ = get_parameter("publish_tf").as_bool();
    lidar_frame_ = get_parameter("lidar_frame").as_string();
    imu_frame_ = get_parameter("imu_frame").as_string();
    lidar_x_m_ = get_parameter("lidar_x_m").as_double();
    lidar_y_m_ = get_parameter("lidar_y_m").as_double();
    lidar_z_m_ = get_parameter("lidar_z_m").as_double();
    imu_x_m_ = get_parameter("imu_x_m").as_double();
    imu_y_m_ = get_parameter("imu_y_m").as_double();
    imu_z_m_ = get_parameter("imu_z_m").as_double();
    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
      static_tf_broadcaster_ =
        std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    }
    imu_orientation_correction_gain_ = std::clamp(
      get_parameter("imu_orientation_correction_gain").as_double(), 0.0, 1.0);
    max_imu_orientation_step_rad_ = std::max(
      0.05, get_parameter("max_imu_orientation_step_rad").as_double());
    packet_assembler_.set_packet_callback(
      [this](const f1tenth_localization::SensorPacket & packet) {
        if (actuator_feedback_settle_delay_ms_ > 0.0) {
          queue_packet_for_feedback_settle(packet);
        } else {
          process_packet(packet);
        }
      });
    if (actuator_feedback_settle_delay_ms_ > 0.0) {
      actuator_settle_timer_ = create_wall_timer(
        std::chrono::milliseconds(1), [this]() {process_settled_packets();});
    }
    pose_xy_variance_ = std::max(0.0, get_parameter("pose_xy_variance").as_double());
    timing_degraded_pose_xy_variance_ = std::max(
      pose_xy_variance_, get_parameter("timing_degraded_pose_xy_variance").as_double());
    pose_yaw_variance_ = std::max(0.0, get_parameter("pose_yaw_variance").as_double());
    twist_linear_variance_ = std::max(
      0.0, get_parameter("twist_linear_variance").as_double());
    twist_yaw_variance_ = std::max(
      0.0, get_parameter("twist_yaw_variance").as_double());

    const auto derived_qos = rclcpp::QoS(100);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("odom_topic").as_string(), derived_qos);
    diagnostics_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      get_parameter("diagnostics_topic").as_string(), derived_qos);

    const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(100);
    left_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      get_parameter("left_encoder_topic").as_string(), sensor_qos,
      [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
        encoder_callback(*msg, true);
      });
    right_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      get_parameter("right_encoder_topic").as_string(), sensor_qos,
      [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
        encoder_callback(*msg, false);
      });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      get_parameter("imu_topic").as_string(), sensor_qos,
      [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) {
        imu_callback(*msg);
      });
    steering_feedback_sub_ = create_subscription<std_msgs::msg::Float32>(
      get_parameter("steering_feedback_topic").as_string(), sensor_qos,
      [this](std_msgs::msg::Float32::ConstSharedPtr msg) {
        actuator_feedback_callback(*msg, true);
      });
    throttle_feedback_sub_ = create_subscription<std_msgs::msg::Float32>(
      get_parameter("throttle_feedback_topic").as_string(), sensor_qos,
      [this](std_msgs::msg::Float32::ConstSharedPtr msg) {
        actuator_feedback_callback(*msg, false);
      });
    steering_command_sub_ = create_subscription<std_msgs::msg::Float32>(
      get_parameter("steering_command_topic").as_string(), sensor_qos,
      [this](std_msgs::msg::Float32::ConstSharedPtr msg) {
        actuator_command_callback(*msg, true);
      });
    throttle_command_sub_ = create_subscription<std_msgs::msg::Float32>(
      get_parameter("throttle_command_topic").as_string(), sensor_qos,
      [this](std_msgs::msg::Float32::ConstSharedPtr msg) {
        actuator_command_callback(*msg, false);
      });
    if (publish_tf_) {
      publish_static_transforms();
    }
    RCLCPP_INFO(
      get_logger(),
      "Deterministic odometry observer: timestamp-paired encoders + IMU; fixed 25 ms physics step");
  }

private:
  static f1tenth_localization::OdometryObserverConfig default_observer_config()
  {
    return f1tenth_localization::deployment_observer_config();
  }

  f1tenth_localization::OdometryObserverConfig load_observer_config() const
  {
    auto config = default_observer_config();
    config.wheel_radius_m = get_parameter("wheel_radius_m").as_double();
    config.wheel_speed_scale = std::max(
      0.0, get_parameter("wheel_speed_scale").as_double());
    config.wheel_speed_scale_speeds_mps = get_parameter(
      "wheel_speed_scale_speeds_mps").as_double_array();
    config.wheel_speed_scale_values = get_parameter(
      "wheel_speed_scale_values").as_double_array();
    config.reset_encoder_jump_rad = get_parameter("reset_encoder_jump_rad").as_double();
    config.wheel_speed_window_s = std::max(
      0.0, get_parameter("wheel_speed_window_s").as_double());
    config.normal_packet_dt_max_s = get_parameter("normal_packet_dt_max_s").as_double();
    config.decel_detect_ax_mps2 = get_parameter("decel_detect_ax_mps2").as_double();
    config.decel_ax_scale = get_parameter("decel_ax_scale").as_double();
    config.decel_ax_offset_mps2 = get_parameter("decel_ax_offset_mps2").as_double();
    config.wheel_dropout_positive_ax_max_mps2 = std::max(
      0.0, get_parameter("wheel_dropout_positive_ax_max_mps2").as_double());
    config.wheel_update_ax_abs_max_mps2 = get_parameter(
      "wheel_update_ax_abs_max_mps2").as_double();
    config.wheel_freeze_speed_mps = get_parameter("wheel_freeze_speed_mps").as_double();
    config.wheel_recovery_launch_speed_mps = std::max(
      0.0, get_parameter("wheel_recovery_launch_speed_mps").as_double());
    config.wheel_recovery_launch_innovation_mps = std::max(
      0.0, get_parameter("wheel_recovery_launch_innovation_mps").as_double());
    config.wheel_recovery_launch_wheel_speed_mps = std::max(
      0.0, get_parameter("wheel_recovery_launch_wheel_speed_mps").as_double());
    config.wheel_burst_disagreement_mps = get_parameter(
      "wheel_burst_disagreement_mps").as_double();
    config.wheel_burst_catchup_accel_mps2 = std::max(
      0.0, get_parameter("wheel_burst_catchup_accel_mps2").as_double());
    config.wheel_burst_catchup_max_mps = std::max(
      0.0, get_parameter("wheel_burst_catchup_max_mps").as_double());
    config.allow_turn_current_packet_recovery = get_parameter(
      "allow_turn_current_packet_recovery").as_bool();
    config.turn_current_packet_max_increase_mps = std::max(
      0.0, get_parameter("turn_current_packet_max_increase_mps").as_double());
    config.turn_current_packet_max_decrease_mps = std::max(
      0.0, get_parameter("turn_current_packet_max_decrease_mps").as_double());
    config.turn_wheel_update_start_speed_mps = std::max(
      0.0, get_parameter("turn_wheel_update_start_speed_mps").as_double());
    config.turn_wheel_update_min_ax_mps2 = get_parameter(
      "turn_wheel_update_min_ax_mps2").as_double();
    config.turn_wheel_update_innovation_threshold_mps = std::max(
      0.0, get_parameter("turn_wheel_update_innovation_threshold_mps").as_double());
    config.turn_wheel_update_slip_beta = std::clamp(
      get_parameter("turn_wheel_update_slip_beta").as_double(), 0.0, 1.0);
    config.use_turn_speed_bias_model = get_parameter(
      "use_turn_speed_bias_model").as_bool();
    config.turn_speed_bias_constant_mps = get_parameter(
      "turn_speed_bias_constant_mps").as_double();
    config.turn_speed_bias_speed_mps = get_parameter(
      "turn_speed_bias_speed_mps").as_double();
    config.turn_speed_bias_speed_squared_mps = get_parameter(
      "turn_speed_bias_speed_squared_mps").as_double();
    config.turn_speed_bias_yaw_rate_abs_mps = get_parameter(
      "turn_speed_bias_yaw_rate_abs_mps").as_double();
    config.turn_speed_bias_yaw_rate_squared_mps = get_parameter(
      "turn_speed_bias_yaw_rate_squared_mps").as_double();
    config.turn_speed_bias_speed_yaw_rate_abs_mps = get_parameter(
      "turn_speed_bias_speed_yaw_rate_abs_mps").as_double();
    config.turn_speed_bias_max_mps = std::max(
      0.0, get_parameter("turn_speed_bias_max_mps").as_double());
    config.turn_speed_bias_start_speed_mps = get_parameter(
      "turn_speed_bias_start_speed_mps").as_double();
    config.turn_speed_bias_full_speed_mps = get_parameter(
      "turn_speed_bias_full_speed_mps").as_double();
    config.use_turn_speed_bias_lateral_accel_gate = get_parameter(
      "use_turn_speed_bias_lateral_accel_gate").as_bool();
    config.turn_speed_bias_lateral_accel_gate_start_mps2 = get_parameter(
      "turn_speed_bias_lateral_accel_gate_start_mps2").as_double();
    config.turn_speed_bias_lateral_accel_gate_full_mps2 = get_parameter(
      "turn_speed_bias_lateral_accel_gate_full_mps2").as_double();
    config.turn_slip_speed_bias_lateral_accel_threshold_mps2 = get_parameter(
      "turn_slip_speed_bias_lateral_accel_threshold_mps2").as_double();
    config.turn_slip_speed_bias_gain_mps_per_mps2 = get_parameter(
      "turn_slip_speed_bias_gain_mps_per_mps2").as_double();
    config.turn_slip_speed_bias_start_speed_mps = get_parameter(
      "turn_slip_speed_bias_start_speed_mps").as_double();
    config.turn_slip_speed_bias_full_speed_mps = get_parameter(
      "turn_slip_speed_bias_full_speed_mps").as_double();
    config.turn_slip_speed_bias_fade_start_mps = get_parameter(
      "turn_slip_speed_bias_fade_start_mps").as_double();
    config.turn_slip_speed_bias_fade_end_mps = get_parameter(
      "turn_slip_speed_bias_fade_end_mps").as_double();
    config.use_coherent_packet_velocity_for_pose = get_parameter(
      "use_coherent_packet_velocity_for_pose").as_bool();
    config.coherent_packet_pose_blend = std::clamp(
      get_parameter("coherent_packet_pose_blend").as_double(), 0.0, 1.0);
    config.wheel_speed_slew_limit_mps2 = std::max(
      0.0, get_parameter("wheel_speed_slew_limit_mps2").as_double());
    config.stationary_speed_threshold_mps = get_parameter(
      "stationary_speed_threshold_mps").as_double();
    config.wheel_update_beta = get_parameter("wheel_update_beta").as_double();
    config.stationary_hold_s = get_parameter("stationary_hold_s").as_double();
    config.stationary_ax_abs_max_mps2 = get_parameter(
      "stationary_ax_abs_max_mps2").as_double();
    config.stationary_ay_abs_max_mps2 = get_parameter(
      "stationary_ay_abs_max_mps2").as_double();
    config.stationary_yaw_rate_abs_max_radps = get_parameter(
      "stationary_yaw_rate_abs_max_radps").as_double();
    config.turn_enter_yaw_rate_radps = get_parameter(
      "turn_enter_yaw_rate_radps").as_double();
    config.turn_enter_abs_ay_mps2 = get_parameter(
      "turn_enter_abs_ay_mps2").as_double();
    config.turn_exit_yaw_rate_radps = get_parameter(
      "turn_exit_yaw_rate_radps").as_double();
    config.turn_exit_abs_ay_mps2 = get_parameter(
      "turn_exit_abs_ay_mps2").as_double();
    config.turn_exit_hold_s = get_parameter("turn_exit_hold_s").as_double();
    config.turn_wheel_braking_ax_mps2 = get_parameter(
      "turn_wheel_braking_ax_mps2").as_double();
    config.integrate_lateral_acceleration_in_turn = get_parameter(
      "integrate_lateral_acceleration_in_turn").as_bool();
    config.use_high_yaw_wheel_slip_observer = get_parameter(
      "use_high_yaw_wheel_slip_observer").as_bool();
    config.high_yaw_wheel_slip_enter_radps = get_parameter(
      "high_yaw_wheel_slip_enter_radps").as_double();
    config.high_yaw_wheel_slip_exit_radps = get_parameter(
      "high_yaw_wheel_slip_exit_radps").as_double();
    config.high_yaw_wheel_slip_exit_hold_s = get_parameter(
      "high_yaw_wheel_slip_exit_hold_s").as_double();
    config.high_yaw_wheel_slip_innovation_mps = get_parameter(
      "high_yaw_wheel_slip_innovation_mps").as_double();
    config.use_learned_u_residual_model = get_parameter(
      "use_learned_u_residual_model").as_bool();
    config.learned_u_residual_min_speed_mps = get_parameter(
      "learned_u_residual_min_speed_mps").as_double();
    config.learned_u_residual_max_speed_mps = get_parameter(
      "learned_u_residual_max_speed_mps").as_double();
    config.u_correction_support_speed_edges_mps = get_parameter(
      "u_correction_support_speed_edges_mps").as_double_array();
    config.u_correction_support_max_abs_steering_rad = get_parameter(
      "u_correction_support_max_abs_steering_rad").as_double_array();
    config.learned_u_residual_intercept_mps = get_parameter(
      "learned_u_residual_intercept_mps").as_double();
    config.learned_u_residual_coefficients = get_parameter(
      "learned_u_residual_coefficients").as_double_array();
    config.use_unity_actuator_u_experts = get_parameter(
      "use_unity_actuator_u_experts").as_bool();
    config.unity_actuator_u_min_speed_mps = get_parameter(
      "unity_actuator_u_min_speed_mps").as_double();
    config.unity_actuator_u_max_speed_mps = get_parameter(
      "unity_actuator_u_max_speed_mps").as_double();
    config.unity_zero_throttle_intercept_mps = get_parameter(
      "unity_zero_throttle_intercept_mps").as_double();
    config.unity_zero_throttle_coefficients = get_parameter(
      "unity_zero_throttle_coefficients").as_double_array();
    config.unity_nonzero_throttle_intercept_mps = get_parameter(
      "unity_nonzero_throttle_intercept_mps").as_double();
    config.unity_nonzero_throttle_coefficients = get_parameter(
      "unity_nonzero_throttle_coefficients").as_double_array();
    config.use_high_steer_unity_actuator_u_expert = get_parameter(
      "use_high_steer_unity_actuator_u_expert").as_bool();
    config.high_steer_unity_actuator_u_min_speed_mps = get_parameter(
      "high_steer_unity_actuator_u_min_speed_mps").as_double();
    config.high_steer_unity_actuator_u_max_speed_mps = get_parameter(
      "high_steer_unity_actuator_u_max_speed_mps").as_double();
    config.high_steer_unity_actuator_u_max_speed_low_throttle_mps = get_parameter(
      "high_steer_unity_actuator_u_max_speed_low_throttle_mps").as_double();
    config.high_steer_unity_actuator_u_throttle_split = get_parameter(
      "high_steer_unity_actuator_u_throttle_split").as_double();
    config.high_steer_unity_actuator_u_min_abs_yaw_rate_radps = get_parameter(
      "high_steer_unity_actuator_u_min_abs_yaw_rate_radps").as_double();
    config.high_steer_unity_actuator_u_min_abs_steering_rad = get_parameter(
      "high_steer_unity_actuator_u_min_abs_steering_rad").as_double();
    config.high_steer_unity_actuator_u_max_abs_steering_rad = get_parameter(
      "high_steer_unity_actuator_u_max_abs_steering_rad").as_double();
    config.high_steer_unity_actuator_u_min_throttle_feedback = get_parameter(
      "high_steer_unity_actuator_u_min_throttle_feedback").as_double();
    config.high_steer_unity_actuator_u_max_throttle_feedback = get_parameter(
      "high_steer_unity_actuator_u_max_throttle_feedback").as_double();
    config.high_steer_unity_actuator_u_intercept_mps = get_parameter(
      "high_steer_unity_actuator_u_intercept_mps").as_double();
    config.high_steer_unity_actuator_u_coefficients = get_parameter(
      "high_steer_unity_actuator_u_coefficients").as_double_array();
    config.use_kinematic_lateral_slip_model = get_parameter(
      "use_kinematic_lateral_slip_model").as_bool();
    config.lateral_velocity_yaw_rate_gain_m = get_parameter(
      "lateral_velocity_yaw_rate_gain_m").as_double();
    config.lateral_velocity_speed_yaw_rate_gain_s = get_parameter(
      "lateral_velocity_speed_yaw_rate_gain_s").as_double();
    config.lateral_velocity_max_mps = std::max(
      0.0, get_parameter("lateral_velocity_max_mps").as_double());
    config.lateral_velocity_reference_forward_offset_m = get_parameter(
      "lateral_velocity_reference_forward_offset_m").as_double();
    config.imu_acceleration_reference_x_m = get_parameter(
      "imu_acceleration_reference_x_m").as_double();
    config.physx_velocity_blend = get_parameter("physx_velocity_blend").as_double();
    config.physx_forward_velocity_blend = get_parameter(
      "physx_forward_velocity_blend").as_double();
    config.physx_low_speed_velocity_blend = get_parameter(
      "physx_low_speed_velocity_blend").as_double();
    config.physx_low_speed_blend_transition_mps = get_parameter(
      "physx_low_speed_blend_transition_mps").as_double();
    config.max_imu_ax_abs_mps2 = get_parameter("max_imu_ax_abs_mps2").as_double();
    return config;
  }

  void publish_static_transforms()
  {
    geometry_msgs::msg::TransformStamped lidar;
    lidar.header.stamp = now();
    lidar.header.frame_id = base_frame_;
    lidar.child_frame_id = lidar_frame_;
    lidar.transform.translation.x = lidar_x_m_;
    lidar.transform.translation.y = lidar_y_m_;
    lidar.transform.translation.z = lidar_z_m_;
    lidar.transform.rotation.w = 1.0;

    geometry_msgs::msg::TransformStamped imu;
    imu.header.stamp = now();
    imu.header.frame_id = base_frame_;
    imu.child_frame_id = imu_frame_;
    imu.transform.translation.x = imu_x_m_;
    imu.transform.translation.y = imu_y_m_;
    imu.transform.translation.z = imu_z_m_;
    imu.transform.rotation.w = 1.0;

    static_tf_broadcaster_->sendTransform({lidar, imu});
  }

  void imu_callback(const sensor_msgs::msg::Imu & msg)
  {
    const double yaw = yaw_from_quaternion(msg.orientation);
    if (!std::isfinite(yaw) || !std::isfinite(msg.linear_acceleration.x) ||
      !std::isfinite(msg.linear_acceleration.y) ||
      !std::isfinite(msg.angular_velocity.z))
    {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    packet_assembler_.add_imu_sample(
      stamp_ns(msg.header.stamp), msg.linear_acceleration.x,
      msg.linear_acceleration.y, msg.angular_velocity.z, yaw);
  }

  void encoder_callback(const sensor_msgs::msg::JointState & msg, bool left)
  {
    if (msg.position.empty() || !std::isfinite(msg.position.front())) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    packet_assembler_.add_encoder_sample(
      stamp_ns(msg.header.stamp), msg.position.front(), left);
  }

  void actuator_feedback_callback(
    const std_msgs::msg::Float32 & msg, bool steering)
  {
    const double value = static_cast<double>(msg.data);
    if (!std::isfinite(value)) {
      return;
    }
    std::lock_guard<std::mutex> lock(actuator_feedback_mutex_);
    const auto received = std::chrono::steady_clock::now();
    if (steering) {
      latest_steering_feedback_rad_ = value;
      latest_steering_feedback_received_ = received;
      have_steering_feedback_ = true;
    } else {
      latest_throttle_feedback_ = value;
      latest_throttle_feedback_received_ = received;
      have_throttle_feedback_ = true;
    }
  }

  void actuator_command_callback(
    const std_msgs::msg::Float32 & msg, bool steering)
  {
    const double value = static_cast<double>(msg.data);
    if (!std::isfinite(value)) {
      return;
    }
    std::lock_guard<std::mutex> lock(actuator_command_mutex_);
    if (steering) {
      latest_steering_command_norm_ = value;
    } else {
      latest_throttle_command_norm_ = value;
    }
  }

  void queue_packet_for_feedback_settle(
    const f1tenth_localization::SensorPacket & packet)
  {
    PendingPacket pending;
    pending.packet = packet;
    pending.queued_at = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> command_lock(actuator_command_mutex_);
      pending.steering_command_norm = latest_steering_command_norm_;
      pending.throttle_command_norm = latest_throttle_command_norm_;
    }
    std::lock_guard<std::mutex> lock(pending_packets_mutex_);
    pending_packets_.push_back(std::move(pending));
  }

  void process_settled_packets()
  {
    while (true) {
      PendingPacket pending;
      {
        std::lock_guard<std::mutex> lock(pending_packets_mutex_);
        if (pending_packets_.empty()) {
          return;
        }
        const auto age = std::chrono::steady_clock::now() -
          pending_packets_.front().queued_at;
        if (age < std::chrono::duration<double, std::milli>(
            actuator_feedback_settle_delay_ms_))
        {
          return;
        }
        pending = std::move(pending_packets_.front());
        pending_packets_.pop_front();
      }
      process_packet(
        pending.packet, pending.steering_command_norm,
        pending.throttle_command_norm);
    }
  }

  double continuous_yaw(const f1tenth_localization::SensorPacket & packet)
  {
    const double raw_yaw = packet.imu_yaw_rad;
    if (!yaw_initialized_) {
      yaw_initialized_ = true;
      yaw_reference_rad_ = raw_yaw;
      previous_raw_relative_yaw_rad_ = 0.0;
      continuous_yaw_rad_ = 0.0;
      previous_yaw_stamp_s_ = static_cast<double>(packet.stamp_ns) * 1.0e-9;
      previous_yaw_rate_radps_ = packet.yaw_rate_radps;
      return continuous_yaw_rad_;
    }

    const double stamp_s = static_cast<double>(packet.stamp_ns) * 1.0e-9;
    const double dt = stamp_s - previous_yaw_stamp_s_;
    const double raw_relative = wrap_angle(raw_yaw - yaw_reference_rad_);
    const double raw_delta = wrap_angle(raw_relative - previous_raw_relative_yaw_rad_);
    if (dt > 0.0 && dt <= 0.5) {
      continuous_yaw_rad_ = wrap_angle(
        continuous_yaw_rad_ + packet.yaw_rate_radps * dt);
      if (std::abs(raw_delta) <= max_imu_orientation_step_rad_) {
        continuous_yaw_rad_ = wrap_angle(
          continuous_yaw_rad_ + imu_orientation_correction_gain_ *
          wrap_angle(raw_relative - continuous_yaw_rad_));
      } else {
        yaw_reference_rad_ = wrap_angle(raw_yaw - continuous_yaw_rad_);
        previous_raw_relative_yaw_rad_ = continuous_yaw_rad_;
      }
    }
    if (std::abs(raw_delta) <= max_imu_orientation_step_rad_) {
      previous_raw_relative_yaw_rad_ = raw_relative;
    }
    previous_yaw_stamp_s_ = stamp_s;
    previous_yaw_rate_radps_ = packet.yaw_rate_radps;
    return continuous_yaw_rad_;
  }

  void process_packet(const f1tenth_localization::SensorPacket & packet)
  {
    double steering_command = 0.0;
    double throttle_command = 0.0;
    {
      std::lock_guard<std::mutex> lock(actuator_command_mutex_);
      steering_command = latest_steering_command_norm_;
      throttle_command = latest_throttle_command_norm_;
    }
    process_packet(packet, steering_command, throttle_command);
  }

  void process_packet(
    const f1tenth_localization::SensorPacket & packet,
    double steering_command, double throttle_command)
  {
    auto processed_packet = packet;
    const auto source_stamp_ns = processed_packet.stamp_ns;
    const auto previous_source_stamp_ns = last_processed_source_stamp_ns_;
    const bool was_retimestamped = processed_packet.stamp_ns <= last_processed_stamp_ns_;
    if (was_retimestamped) {
      // Source-time reversals are delivered at callback time rather than
      // discarded. Keep the original sensor values and use a strictly newer
      // local ROS stamp only to preserve published-message ordering. Vehicle
      // propagation uses the fixed 25 ms packet clock below.
      int64_t receive_stamp_ns = now().nanoseconds();
      if (receive_stamp_ns <= last_processed_stamp_ns_)
        receive_stamp_ns = last_processed_stamp_ns_ + 1;
      processed_packet.stamp_ns = receive_stamp_ns;
      ++retimestamped_packet_count_;
    }
    last_processed_stamp_ns_ = processed_packet.stamp_ns;
    last_processed_source_stamp_ns_ = source_stamp_ns;

    if (!observer_clock_initialized_) {
      observer_stamp_ns_ = processed_packet.stamp_ns;
      observer_clock_initialized_ = true;
    } else {
      observer_stamp_ns_ += kNominalPhysicsPacketPeriodNs;
    }

    auto physics_packet = processed_packet;
    physics_packet.stamp_ns = observer_stamp_ns_;

    f1tenth_localization::OdometryObservation observation;
    observation.stamp_s = static_cast<double>(observer_stamp_ns_) * 1.0e-9;
    observation.left_angle_rad = processed_packet.left_angle_rad;
    observation.right_angle_rad = processed_packet.right_angle_rad;
    observation.ax_mps2 = processed_packet.ax_mps2;
    observation.ay_mps2 = processed_packet.ay_mps2;
    observation.yaw_rate_radps = processed_packet.yaw_rate_radps;
    observation.yaw_rad = continuous_yaw(physics_packet);
    {
      std::lock_guard<std::mutex> lock(actuator_feedback_mutex_);
      const auto actuator_now = std::chrono::steady_clock::now();
      constexpr auto kActuatorFeedbackMaxAge = std::chrono::milliseconds(100);
      observation.actuator_feedback_valid = have_steering_feedback_ &&
        have_throttle_feedback_ &&
        actuator_now - latest_steering_feedback_received_ <= kActuatorFeedbackMaxAge &&
        actuator_now - latest_throttle_feedback_received_ <= kActuatorFeedbackMaxAge;
      if (observation.actuator_feedback_valid) {
        observation.steering_feedback_rad = latest_steering_feedback_rad_;
        observation.throttle_feedback = latest_throttle_feedback_;
      }
    }

    if (physx_odom_runtime_) {
      const auto velocity = physx_odom_runtime_->advance(
        steering_command, throttle_command,
        observation.actuator_feedback_valid,
        observation.steering_feedback_rad,
        observation.throttle_feedback);
      observation.physx_forward_velocity_mps = velocity.u_mps;
      observation.physx_lateral_velocity_mps = velocity.v_mps;
      observation.physx_velocity_valid = true;
    }

    auto estimate = observer_.update(observation);
    const double receive_dt_s = previous_source_stamp_ns > 0 ?
      static_cast<double>(source_stamp_ns - previous_source_stamp_ns) * 1.0e-9 : 0.0;
    if (previous_source_stamp_ns > 0 &&
      (receive_dt_s <= 0.0 ||
      receive_dt_s > observer_config_.normal_packet_dt_max_s))
    {
      // Keep delivery timing visible to the existing diagnostic/covariance
      // path, but never use this receipt interval for vehicle propagation.
      estimate.timing_degraded = true;
    }
    publish_estimate(
      estimate, processed_packet.stamp_ns, source_stamp_ns, previous_source_stamp_ns,
      was_retimestamped, retimestamped_packet_count_);
  }

  void publish_estimate(
    const f1tenth_localization::OdometryEstimate & estimate,
    int64_t published_stamp_ns,
    int64_t packet_source_stamp_ns,
    int64_t previous_packet_source_stamp_ns,
    bool was_retimestamped,
    std::uint64_t retimestamped_packet_count)
  {
    const rclcpp::Time stamp(published_stamp_ns, RCL_ROS_TIME);
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = estimate.x_m;
    odom.pose.pose.position.y = estimate.y_m;
    odom.pose.pose.orientation = quaternion_from_yaw(estimate.yaw_rad);
    odom.twist.twist.linear.x = estimate.body_u_mps;
    odom.twist.twist.linear.y = estimate.body_v_mps;
    odom.twist.twist.angular.z = estimate.yaw_rate_radps;

    // A rejected IMU outlier does not move the causal pose. Keep its pose
    // covariance at the normal floor so one discarded sample cannot poison
    // the EKF covariance for the remainder of a run. An isolated timing gap
    // gets a bounded inflation, not the old 0.25 m^2 stop-gate value: the EKF
    // uses this covariance as a floor for the rest of the epoch.
    const double pose_variance = estimate.timing_degraded &&
      !estimate.sensor_outlier ?
      timing_degraded_pose_xy_variance_ : pose_xy_variance_;
    odom.pose.covariance[0] = pose_variance;
    odom.pose.covariance[7] = pose_variance;
    odom.pose.covariance[35] = pose_yaw_variance_;
    odom.twist.covariance[0] = twist_linear_variance_;
    odom.twist.covariance[7] = twist_linear_variance_;
    odom.twist.covariance[35] = twist_yaw_variance_;
    odom_pub_->publish(odom);

    std_msgs::msg::Float64MultiArray diagnostics;
    diagnostics.layout.dim.resize(1);
    diagnostics.layout.dim[0].label = "deterministic_odometry_v7";
    diagnostics.layout.dim[0].size = 42;
    diagnostics.layout.dim[0].stride = 42;
    diagnostics.data = {
      7.0,
      estimate.stamp_s,
      estimate.dt_s,
      estimate.wheel_raw_mps,
      estimate.wheel_mapped_mps,
      estimate.speed_pred_mps,
      estimate.speed_mps,
      estimate.body_u_mps,
      estimate.body_v_mps,
      estimate.ax_mps2,
      estimate.ay_mps2,
      estimate.yaw_rate_radps,
      estimate.wheel_update_used ? 1.0 : 0.0,
      estimate.turn_mode ? 1.0 : 0.0,
      estimate.reset_epoch ? 1.0 : 0.0,
      estimate.timing_degraded ? 1.0 : 0.0,
      static_cast<double>(packet_assembler_.packet_drop_count()),
      static_cast<double>(packet_assembler_.packet_coherence_fault_count()),
      estimate.x_m,
      estimate.y_m,
      estimate.yaw_rad,
      estimate.sensor_outlier ? 1.0 : 0.0,
      estimate.left_angle_rad,
      estimate.right_angle_rad,
      estimate.imu_yaw_rad,
      estimate.wheel_packet_mps,
      estimate.wheel_burst_rejected ? 1.0 : 0.0,
      estimate.turn_speed_bias_mps,
      static_cast<double>(packet_source_stamp_ns) * 1.0e-9,
      static_cast<double>(previous_packet_source_stamp_ns) * 1.0e-9,
      was_retimestamped ? 1.0 : 0.0,
      static_cast<double>(retimestamped_packet_count),
      static_cast<double>(packet_assembler_.diagnostics().latest_seen_source_stamp_ns) * 1.0e-9,
      static_cast<double>(packet_assembler_.diagnostics().last_emitted_source_stamp_ns) * 1.0e-9,
      static_cast<double>(packet_assembler_.diagnostics().source_reversal_count),
      static_cast<double>(packet_assembler_.diagnostics().duplicate_source_count),
      static_cast<double>(packet_assembler_.diagnostics().late_completed_packet_count),
      static_cast<double>(packet_assembler_.diagnostics().incomplete_packet_drop_count),
      static_cast<double>(packet_assembler_.diagnostics().maximum_reorder_depth),
      static_cast<double>(packet_assembler_.diagnostics().maximum_reorder_time_ns) * 1.0e-9,
      estimate.pre_wheel_update_speed_mps,
      estimate.pre_wheel_update_speed_valid ? 1.0 : 0.0};
    diagnostics_pub_->publish(diagnostics);

    if (publish_tf_) {
      geometry_msgs::msg::TransformStamped transform;
      transform.header = odom.header;
      transform.child_frame_id = base_frame_;
      transform.transform.translation.x = estimate.x_m;
      transform.transform.translation.y = estimate.y_m;
      transform.transform.rotation = odom.pose.pose.orientation;
      tf_broadcaster_->sendTransform(transform);
    }

  }

  f1tenth_localization::OdometryObserverConfig observer_config_;
  f1tenth_localization::OdometryObserver observer_;
  f1tenth_localization::SensorPacketAssembler packet_assembler_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr diagnostics_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr right_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr steering_feedback_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr throttle_feedback_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr steering_command_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr throttle_command_sub_;
  rclcpp::TimerBase::SharedPtr actuator_settle_timer_;

  std::mutex mutex_;
  std::mutex actuator_feedback_mutex_;
  std::mutex actuator_command_mutex_;
  std::mutex pending_packets_mutex_;
  struct PendingPacket
  {
    f1tenth_localization::SensorPacket packet;
    std::chrono::steady_clock::time_point queued_at;
    double steering_command_norm{0.0};
    double throttle_command_norm{0.0};
  };
  std::deque<PendingPacket> pending_packets_;
  double actuator_feedback_settle_delay_ms_{0.0};
  double latest_steering_feedback_rad_{0.0};
  double latest_throttle_feedback_{0.0};
  std::chrono::steady_clock::time_point latest_steering_feedback_received_{};
  std::chrono::steady_clock::time_point latest_throttle_feedback_received_{};
  bool have_steering_feedback_{false};
  bool have_throttle_feedback_{false};
  double latest_steering_command_norm_{0.0};
  double latest_throttle_command_norm_{0.0};
  bool use_physx_odom_velocity_{false};
  std::unique_ptr<f1tenth_localization::PhysxOdomRuntime> physx_odom_runtime_;
  int64_t last_processed_stamp_ns_{0};
  int64_t last_processed_source_stamp_ns_{0};
  int64_t observer_stamp_ns_{0};
  std::uint64_t retimestamped_packet_count_{0};

  bool observer_clock_initialized_{false};
  bool yaw_initialized_{false};
  double yaw_reference_rad_{0.0};
  double previous_raw_relative_yaw_rad_{0.0};
  double continuous_yaw_rad_{0.0};
  double previous_yaw_stamp_s_{0.0};
  double previous_yaw_rate_radps_{0.0};
  double imu_orientation_correction_gain_{1.0};
  double max_imu_orientation_step_rad_{0.30};
  bool publish_tf_{true};

  std::string odom_frame_;
  std::string base_frame_;
  std::string lidar_frame_;
  std::string imu_frame_;
  double lidar_x_m_{0.2733};
  double lidar_y_m_{0.0};
  double lidar_z_m_{0.096};
  double imu_x_m_{0.08};
  double imu_y_m_{0.0};
  double imu_z_m_{0.055};
  double pose_xy_variance_{0.01};
  double timing_degraded_pose_xy_variance_{0.01};
  double pose_yaw_variance_{0.0001};
  double twist_linear_variance_{0.04};
  double twist_yaw_variance_{0.04};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorOdometryNode>());
  rclcpp::shutdown();
  return 0;
}

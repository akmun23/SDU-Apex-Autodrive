#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
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
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include "f1tenth_localization/sensor_fusion_model.hpp"

namespace {

double wrap_angle(double a)
{
  return std::atan2(std::sin(a), std::cos(a));
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

}  // namespace

using f1tenth_localization::SensorMotionRegime;

class SensorOdometryNode final : public rclcpp::Node
{
public:
  SensorOdometryNode()
  : Node("sensor_odometry"),
    tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)),
    static_tf_broadcaster_(
      std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this))
  {
    declare_parameter("left_encoder_topic", "/autodrive/roboracer_1/left_encoder");
    declare_parameter("right_encoder_topic", "/autodrive/roboracer_1/right_encoder");
    declare_parameter("imu_topic", "/autodrive/roboracer_1/imu");
    declare_parameter("odom_topic", "/odom");
    declare_parameter("diagnostics_topic", "/odom/diagnostics");
    // Simulator epoch resets are diagnostics-only. Production odometry must
    // remain continuous and leaves this disabled.
    declare_parameter("reset_enabled", false);
    declare_parameter("reset_topic", "/autodrive/reset_command");

    declare_parameter("odom_frame", "odom");
    declare_parameter("base_frame", "base_link");
    declare_parameter("lidar_frame", "lidar");
    declare_parameter("imu_frame", "imu");

    // Fixed documented RoboRacer geometry; this is not runtime-calibrated.
    declare_parameter("wheel_radius_m", 0.0590);
    declare_parameter("encoder_angle_scale", 1.0);
    // At the native ~10 Hz telemetry cadence, valid full-speed wheel motion
    // can exceed 2.7 m per sample. Reset jumps are tens to hundreds of metres,
    // so 4 m rejects resets without rejecting valid racing motion.
    declare_parameter("max_encoder_step_m", 4.0);
    declare_parameter("max_encoder_pair_skew_s", 0.05);
    declare_parameter("lidar_x_m", 0.2733);
    declare_parameter("lidar_y_m", 0.0);
    declare_parameter("lidar_z_m", 0.096);
    declare_parameter("imu_x_m", 0.08);
    declare_parameter("imu_y_m", 0.0);
    declare_parameter("imu_z_m", 0.055);
    // The official orientation is accurate in the simulation.  Use it
    // directly when its step is plausible; gyro integration remains the
    // fallback for an isolated discontinuity.
    declare_parameter("imu_orientation_correction_gain", 1.0);
    declare_parameter("max_imu_orientation_step_rad", 0.30);
    declare_parameter("max_imu_dt_s", 0.5);

    // Longitudinal observer: the documented encoder geometry remains the
    // mechanical conversion, followed by the measured simulator slip map.
    // The map is active from standstill because the clean open-ground sweep
    // shows measurable slip before the old high-speed-only threshold.
    declare_parameter("imu_acceleration_filter_alpha", 0.70);
    // The simulator occasionally emits isolated longitudinal IMU spikes at
    // the native telemetry rate. Keep the raw value for the fitted model
    // feature, but use a short causal median for integration and regime
    // selection so one packet cannot create a false speed/acceleration mode.
    declare_parameter("imu_acceleration_median_window", 3);
    declare_parameter("wheel_slip_threshold_mps", 0.75);
    declare_parameter("wheel_slip_ratio", 0.20);
    // Sx is ill-conditioned near standstill.  Do not classify launch
    // quantisation as tire slip until the IMU speed makes the documented
    // slip ratio observable.
    declare_parameter("slip_observation_min_speed_mps", 0.75);
    declare_parameter("wheel_slip_activation_speed_mps", 0.0);
    declare_parameter("encoder_reset_threshold_rad", 0.50);
    // Innovation gate tuned from the clean truth/IMU response: normal body
    // acceleration is accepted, while a delayed wheel burst is not allowed
    // to jump the body-speed estimate to the wheel-spin speed.
    declare_parameter("max_observer_accel_mps2", 8.0);
    declare_parameter("observer_speed_tolerance_mps", 1.0);
    declare_parameter("wheel_observer_correction_gain", 0.10);
    declare_parameter("sensor_fusion_model_enabled", true);
    // In the steady regime the bounded wheel/IMU baseline is already an
    // absolute observation. The learned steady branch is retained for
    // diagnostics but is disabled by default because it can turn a valid
    // low-speed wheel update into a stale launch-speed bias.
    declare_parameter("sensor_fusion_steady_model_enabled", false);
    declare_parameter("regime_acceleration_threshold_mps2", 0.50);
    declare_parameter("regime_model_max_spread_mps", 0.75);
    declare_parameter("regime_model_max_baseline_delta_mps", 0.75);
    // A quiet, consistent wheel/map measurement below the IMU observer is
    // evidence of accumulated positive IMU-speed bias rather than driven
    // wheel overspeed. Allow that sensor-only branch to re-anchor the
    // observer without using throttle, ground truth, or AMCL.
    declare_parameter("steady_encoder_reanchor_enabled", true);
    // Keep the IMU state independent until a live bias-correction fit is
    // accepted; the YAML candidate currently leaves this at zero.
    declare_parameter("imu_speed_correction_gain", 0.0);
    declare_parameter("stationary_speed_threshold_mps", 0.15);
    // AutoDRIVE documents the longitudinal WheelCollider curve using these
    // slip/force knots.  Runtime estimation uses the extremum/asymptote slip
    // boundaries to decide when wheel speed is no longer a valid body-speed
    // observation; the force values are retained for validation and model
    // identification, not treated as a hidden acceleration measurement.
    declare_parameter("forward_extremum_slip", 0.15);
    declare_parameter("forward_extremum_value", 0.72);
    declare_parameter("forward_asymptote_slip", 0.25);
    declare_parameter("forward_asymptote_value", 0.464);
    // When the driven-wheel encoder freezes during simulator coasting, use
    // the allowed IMU integration until a low-speed/low-acceleration stop is
    // confirmed.  This avoids cutting off the coast distance while preventing
    // residual IMU bias from producing indefinite motion at rest.
    declare_parameter("imu_stop_speed_threshold_mps", 2.0);
    declare_parameter("imu_stationary_acceleration_threshold_mps2", 0.30);
    declare_parameter("zero_encoder_stop_confirm_sec", 0.80);
    // Offline identification found a bounded braking deceleration trend when
    // the driven encoder is frozen. This is a sensor-only fallback for the
    // unobservable case where the IMU goes quiet before its integrated speed
    // has decayed; it predicts remaining slip distance and never teleports
    // pose to zero.
    declare_parameter("frozen_encoder_model_enabled", true);
    declare_parameter("frozen_encoder_model_delay_sec", 0.15);
    declare_parameter("frozen_encoder_decel_intercept_mps2", 5.5);
    declare_parameter("frozen_encoder_decel_speed_gain_per_s", 0.27);
    declare_parameter("frozen_encoder_decel_max_mps2", 12.0);
    declare_parameter("slip_pose_xy_variance", 0.10);
    declare_parameter("encoder_reset_covariance_duration_s", 1.0);
    declare_parameter("encoder_reset_pose_xy_variance", 0.25);
    declare_parameter("encoder_reset_twist_linear_variance", 0.50);
    declare_parameter(
      "wheel_speed_map_wheel_mps",
      std::vector<double>{0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
        16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0});
    declare_parameter(
      "wheel_speed_map_body_mps",
      std::vector<double>{0.0, 1.988601, 3.967925, 5.888532, 8.549802,
        9.662430, 11.536114, 13.473328, 15.270020, 16.855770,
        18.570086, 19.279928, 19.987528, 20.287600, 20.287600,
        20.287600});

    declare_parameter("pose_xy_variance", 0.01);
    declare_parameter("pose_yaw_variance", 0.01);
    declare_parameter("velocity_filter_alpha", 1.0);
    declare_parameter("velocity_median_window", 3);
    declare_parameter("velocity_filter_decel_alpha", 1.0);
    // Outlier guard only; do not make this a hidden low-speed ceiling.
    declare_parameter("max_velocity_accel_mps2", 40.0);
    declare_parameter("twist_linear_variance", 0.04);
    declare_parameter("twist_yaw_variance", 0.04);

    wheel_radius_ = get_parameter("wheel_radius_m").as_double();
    encoder_scale_ = get_parameter("encoder_angle_scale").as_double();
    max_encoder_step_m_ = get_parameter("max_encoder_step_m").as_double();
    max_encoder_pair_skew_s_ = std::max(
      0.0, get_parameter("max_encoder_pair_skew_s").as_double());
    pose_xy_var_ = get_parameter("pose_xy_variance").as_double();
    pose_yaw_var_ = get_parameter("pose_yaw_variance").as_double();
    velocity_filter_alpha_ = std::clamp(
      get_parameter("velocity_filter_alpha").as_double(), 0.01, 1.0);
    velocity_filter_decel_alpha_ = std::clamp(
      get_parameter("velocity_filter_decel_alpha").as_double(), 0.01, 1.0);
    const auto median_window = get_parameter("velocity_median_window").as_int();
    velocity_median_window_ = static_cast<size_t>(std::max<int64_t>(
      1, std::min<int64_t>(9, median_window)));
    max_velocity_accel_mps2_ = std::max(
      0.1, get_parameter("max_velocity_accel_mps2").as_double());
    twist_linear_var_ = get_parameter("twist_linear_variance").as_double();
    twist_yaw_var_ = get_parameter("twist_yaw_variance").as_double();
    imu_orientation_correction_gain_ = std::clamp(
      get_parameter("imu_orientation_correction_gain").as_double(), 0.0, 1.0);
    max_imu_orientation_step_rad_ = std::max(
      0.05, get_parameter("max_imu_orientation_step_rad").as_double());
    max_imu_dt_s_ = std::max(0.05, get_parameter("max_imu_dt_s").as_double());
    imu_acceleration_filter_alpha_ = std::clamp(
      get_parameter("imu_acceleration_filter_alpha").as_double(), 0.01, 1.0);
    const auto imu_median_window = get_parameter(
      "imu_acceleration_median_window").as_int();
    imu_acceleration_median_window_ = static_cast<size_t>(std::max<int64_t>(
      1, std::min<int64_t>(9, imu_median_window)));
    wheel_slip_threshold_mps_ = std::max(
      0.0, get_parameter("wheel_slip_threshold_mps").as_double());
    wheel_slip_ratio_ = std::max(
      0.0, get_parameter("wheel_slip_ratio").as_double());
    stationary_speed_threshold_mps_ = std::max(
      0.0, get_parameter("stationary_speed_threshold_mps").as_double());
    slip_observation_min_speed_mps_ = std::max(
      stationary_speed_threshold_mps_, get_parameter(
        "slip_observation_min_speed_mps").as_double());
    wheel_slip_activation_speed_mps_ = std::max(
      0.0, get_parameter("wheel_slip_activation_speed_mps").as_double());
    encoder_reset_threshold_rad_ = std::max(
      0.05, get_parameter("encoder_reset_threshold_rad").as_double());
    max_observer_accel_mps2_ = std::max(
      0.1, get_parameter("max_observer_accel_mps2").as_double());
    observer_speed_tolerance_mps_ = std::max(
      0.0, get_parameter("observer_speed_tolerance_mps").as_double());
    wheel_observer_correction_gain_ = std::clamp(
      get_parameter("wheel_observer_correction_gain").as_double(), 0.0, 1.0);
    sensor_fusion_model_enabled_ = get_parameter(
      "sensor_fusion_model_enabled").as_bool();
    sensor_fusion_steady_model_enabled_ = get_parameter(
      "sensor_fusion_steady_model_enabled").as_bool();
    regime_acceleration_threshold_mps2_ = std::max(
      0.05, get_parameter("regime_acceleration_threshold_mps2").as_double());
    regime_model_max_spread_mps_ = std::max(
      0.0, get_parameter("regime_model_max_spread_mps").as_double());
    regime_model_max_baseline_delta_mps_ = std::max(
      0.0, get_parameter("regime_model_max_baseline_delta_mps").as_double());
    steady_encoder_reanchor_enabled_ = get_parameter(
      "steady_encoder_reanchor_enabled").as_bool();
    imu_speed_correction_gain_ = std::clamp(
      get_parameter("imu_speed_correction_gain").as_double(), 0.0, 1.0);
    forward_extremum_slip_ = get_parameter("forward_extremum_slip").as_double();
    forward_extremum_value_ = get_parameter("forward_extremum_value").as_double();
    forward_asymptote_slip_ = get_parameter("forward_asymptote_slip").as_double();
    forward_asymptote_value_ = get_parameter("forward_asymptote_value").as_double();
    if (!std::isfinite(forward_extremum_slip_) ||
      !std::isfinite(forward_extremum_value_) ||
      !std::isfinite(forward_asymptote_slip_) ||
      !std::isfinite(forward_asymptote_value_) ||
      forward_extremum_slip_ <= 0.0 ||
      forward_asymptote_slip_ <= forward_extremum_slip_ ||
      forward_extremum_value_ <= 0.0 ||
      forward_asymptote_value_ <= 0.0 ||
      forward_asymptote_value_ > forward_extremum_value_) {
      throw std::runtime_error("invalid documented longitudinal friction curve");
    }
    imu_stop_speed_threshold_mps_ = std::max(
      stationary_speed_threshold_mps_, get_parameter(
        "imu_stop_speed_threshold_mps").as_double());
    imu_stationary_acceleration_threshold_mps2_ = std::max(
      0.0, get_parameter("imu_stationary_acceleration_threshold_mps2").as_double());
    zero_encoder_stop_confirm_sec_ = std::max(
      0.0, get_parameter("zero_encoder_stop_confirm_sec").as_double());
    frozen_encoder_model_enabled_ = get_parameter(
      "frozen_encoder_model_enabled").as_bool();
    frozen_encoder_model_delay_sec_ = std::max(
      0.0, get_parameter("frozen_encoder_model_delay_sec").as_double());
    frozen_encoder_decel_intercept_mps2_ = std::max(
      0.0, get_parameter("frozen_encoder_decel_intercept_mps2").as_double());
    frozen_encoder_decel_speed_gain_per_s_ = std::max(
      0.0, get_parameter("frozen_encoder_decel_speed_gain_per_s").as_double());
    frozen_encoder_decel_max_mps2_ = std::max(
      0.1, get_parameter("frozen_encoder_decel_max_mps2").as_double());
    if (frozen_encoder_decel_intercept_mps2_ > frozen_encoder_decel_max_mps2_) {
      throw std::runtime_error("frozen encoder deceleration intercept exceeds maximum");
    }
    slip_pose_xy_var_ = std::max(
      pose_xy_var_, get_parameter("slip_pose_xy_variance").as_double());
    encoder_reset_covariance_duration_s_ = std::max(
      0.0, get_parameter("encoder_reset_covariance_duration_s").as_double());
    encoder_reset_pose_xy_var_ = std::max(
      pose_xy_var_, get_parameter("encoder_reset_pose_xy_variance").as_double());
    encoder_reset_twist_linear_var_ = std::max(
      twist_linear_var_, get_parameter("encoder_reset_twist_linear_variance").as_double());
    wheel_speed_map_wheel_mps_ = get_parameter(
      "wheel_speed_map_wheel_mps").as_double_array();
    wheel_speed_map_body_mps_ = get_parameter(
      "wheel_speed_map_body_mps").as_double_array();
    if (wheel_speed_map_wheel_mps_.size() < 2 ||
        wheel_speed_map_wheel_mps_.size() != wheel_speed_map_body_mps_.size()) {
      throw std::runtime_error("wheel-speed slip maps must have equal length >= 2");
    }
    for (size_t index = 0; index < wheel_speed_map_wheel_mps_.size(); ++index) {
      if (!std::isfinite(wheel_speed_map_wheel_mps_[index]) ||
          !std::isfinite(wheel_speed_map_body_mps_[index]) ||
          wheel_speed_map_wheel_mps_[index] < 0.0 ||
          wheel_speed_map_body_mps_[index] < 0.0 ||
          (index > 0 && wheel_speed_map_wheel_mps_[index] <=
            wheel_speed_map_wheel_mps_[index - 1]) ||
          (index > 0 && wheel_speed_map_body_mps_[index] <
            wheel_speed_map_body_mps_[index - 1])) {
        throw std::runtime_error("wheel-speed slip maps must be finite and monotonic");
      }
    }
    odom_frame_ = get_parameter("odom_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();

    if (wheel_radius_ <= 0.0 || encoder_scale_ <= 0.0 ||
        max_encoder_step_m_ <= 0.0) {
      throw std::runtime_error("invalid sensor odometry parameters");
    }

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("odom_topic").as_string(), rclcpp::QoS(10));
    diagnostics_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      get_parameter("diagnostics_topic").as_string(), rclcpp::QoS(10));

    auto sensor_qos = rclcpp::SensorDataQoS().keep_last(5);
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
      std::bind(&SensorOdometryNode::imu_callback, this, std::placeholders::_1));
    if (get_parameter("reset_enabled").as_bool()) {
      reset_sub_ = create_subscription<std_msgs::msg::Bool>(
        get_parameter("reset_topic").as_string(), rclcpp::QoS(10),
        [this](std_msgs::msg::Bool::ConstSharedPtr msg) {
          if (msg->data) {
            reset_diagnostic_epoch();
          }
        });
    }
    publish_static_transforms();
    RCLCPP_INFO(
      get_logger(),
      "Allowed sensor odometry: rear encoders + IMU -> /odom");
  }

private:

  void reset_diagnostic_epoch()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // The calibration harness verifies that the vehicle has stopped before
    // sending this event. Rebaseline every local state variable so /odom
    // starts at the simulator spawn for the next independent grid point.
    have_left_ = false;
    have_right_ = false;
    left_updated_ = false;
    right_updated_ = false;
    encoder_initialized_ = false;
    imu_initialized_ = false;
    left_angle_ = 0.0;
    right_angle_ = 0.0;
    prev_left_ = 0.0;
    prev_right_ = 0.0;
    left_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    right_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    prev_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    imu_yaw_zero_ = 0.0;
    imu_raw_yaw_ = 0.0;
    last_imu_relative_yaw_ = 0.0;
    imu_integrated_yaw_ = 0.0;
    imu_yaw_rate_ = 0.0;
    odom_yaw_ = 0.0;
    prev_yaw_ = 0.0;
    last_imu_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    have_imu_stamp_ = false;
    speed_mps_ = 0.0;
    recent_raw_speeds_.clear();
    raw_wheel_speed_mps_ = 0.0;
    corrected_wheel_speed_mps_ = 0.0;
    longitudinal_slip_ = 0.0;
    wheel_slip_detected_ = false;
    wheel_observation_confidence_ = 1.0;
    frozen_encoder_model_active_ = false;
    frozen_encoder_model_decel_mps2_ = 0.0;
    sensor_fusion_window_raw_speed_mps_ = 0.0;
    sensor_fusion_window_mapped_speed_mps_ = 0.0;
    sensor_fusion_model_active_ = false;
    sensor_fusion_model_speed_mps_ = 0.0;
    sensor_fusion_model_spread_mps_ = 0.0;
    motion_regime_ = SensorMotionRegime::STEADY;
    sensor_fusion_raw_history_.clear();
    sensor_fusion_mapped_history_.clear();
    sensor_fusion_imu_history_.clear();
    sensor_fusion_gap_history_.clear();
    sensor_fusion_encoder_history_.clear();
    steady_mapped_speed_history_.clear();
    zero_encoder_duration_s_ = 0.0;
    last_motion_sign_ = 1.0;
    imu_raw_acceleration_mps2_ = 0.0;
    imu_lateral_acceleration_mps2_ = 0.0;
    x_ = 0.0;
    y_ = 0.0;
    reset_longitudinal_observer();
    encoder_reset_active_ = false;
    RCLCPP_INFO(get_logger(), "Diagnostic odometry epoch reset to spawn origin");
  }

  void publish_static_transforms()
  {
    geometry_msgs::msg::TransformStamped lidar;
    lidar.header.stamp = now();
    lidar.header.frame_id = base_frame_;
    lidar.child_frame_id = get_parameter("lidar_frame").as_string();
    lidar.transform.translation.x = get_parameter("lidar_x_m").as_double();
    lidar.transform.translation.y = get_parameter("lidar_y_m").as_double();
    lidar.transform.translation.z = get_parameter("lidar_z_m").as_double();
    lidar.transform.rotation.w = 1.0;

    geometry_msgs::msg::TransformStamped imu;
    imu.header.stamp = now();
    imu.header.frame_id = base_frame_;
    imu.child_frame_id = get_parameter("imu_frame").as_string();
    imu.transform.translation.x = get_parameter("imu_x_m").as_double();
    imu.transform.translation.y = get_parameter("imu_y_m").as_double();
    imu.transform.translation.z = get_parameter("imu_z_m").as_double();
    imu.transform.rotation.w = 1.0;

    static_tf_broadcaster_->sendTransform({lidar, imu});
  }

  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
  {
    const double raw_yaw = yaw_from_quaternion(msg->orientation);
    const double gyro_z = msg->angular_velocity.z;
    const double longitudinal_acceleration = msg->linear_acceleration.x;
    if (!std::isfinite(raw_yaw) || !std::isfinite(gyro_z)) {
      return;
    }

    const rclcpp::Time stamp(msg->header.stamp, get_clock()->get_clock_type());
    std::lock_guard<std::mutex> lock(mutex_);
    imu_raw_yaw_ = raw_yaw;
    if (!imu_initialized_) {
      imu_yaw_zero_ = raw_yaw;
      last_imu_relative_yaw_ = 0.0;
      imu_integrated_yaw_ = 0.0;
      last_imu_stamp_ = stamp;
      have_imu_stamp_ = true;
      odom_yaw_ = 0.0;
      imu_initialized_ = true;
    } else {
      const double raw_relative_yaw = wrap_angle(raw_yaw - imu_yaw_zero_);
      const double raw_delta = wrap_angle(raw_relative_yaw - last_imu_relative_yaw_);
      double dt = 0.0;
      if (have_imu_stamp_) {
        dt = (stamp - last_imu_stamp_).seconds();
      }
      if (dt > 1.0e-4 && dt <= max_imu_dt_s_) {
        const double gyro_delta = gyro_z * dt;
        imu_integrated_yaw_ = wrap_angle(imu_integrated_yaw_ + gyro_delta);

        // AutoDRIVE occasionally emits an orientation sample with a large
        // discontinuity although its gyro stream remains smooth. This also
        // occurs when the simulator teleports the vehicle during a diagnostic
        // reset. Never let that sample, or the next correction, teleport
        // odometry. Rebase the raw orientation while preserving the integrated
        // yaw and let the gyro continue from the continuous state.
        if (std::abs(raw_delta) <= max_imu_orientation_step_rad_) {
          const double correction = wrap_angle(raw_relative_yaw - imu_integrated_yaw_);
          imu_integrated_yaw_ = wrap_angle(
            imu_integrated_yaw_ + imu_orientation_correction_gain_ * correction);
        } else {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Ignoring discontinuous IMU yaw sample (step=%.3f rad); using gyro integration",
            raw_delta);
          imu_yaw_zero_ = wrap_angle(raw_yaw - imu_integrated_yaw_);
          last_imu_relative_yaw_ = imu_integrated_yaw_;
        }
      }
      odom_yaw_ = imu_integrated_yaw_;
      if (std::abs(raw_delta) <= max_imu_orientation_step_rad_) {
        last_imu_relative_yaw_ = raw_relative_yaw;
      }
      last_imu_stamp_ = stamp;
      have_imu_stamp_ = true;
    }
    imu_yaw_rate_ = gyro_z;
    imu_raw_acceleration_mps2_ = std::isfinite(longitudinal_acceleration) ?
      longitudinal_acceleration : 0.0;
    imu_lateral_acceleration_mps2_ = std::isfinite(msg->linear_acceleration.y) ?
      msg->linear_acceleration.y : 0.0;

    update_longitudinal_observer(longitudinal_acceleration, stamp);

    // The official bridge publishes both encoder messages before the IMU in
    // one telemetry cycle.  Integrating from an encoder callback therefore
    // uses the previous cycle's yaw.  Complete the pair after the matching
    // IMU callback so the planar increment is time-aligned.
    if (have_left_ && have_right_ && left_updated_ && right_updated_ && imu_initialized_) {
      const double pair_skew = std::abs((left_stamp_ - right_stamp_).seconds());
      if (pair_skew > max_encoder_pair_skew_s_) {
        // Keep the newer sample and wait for the other wheel from the same
        // bridge cycle. Never combine a delayed wheel message with a newer
        // one, which would create false yaw/velocity spikes.
        if (left_stamp_ < right_stamp_) {
          left_updated_ = false;
        } else {
          right_updated_ = false;
        }
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Waiting for synchronized encoder pair (skew=%.3f s)", pair_skew);
      } else {
        integrate_pair();
        left_updated_ = false;
        right_updated_ = false;
      }
    }
  }

  void update_longitudinal_observer(
    double acceleration_mps2, const rclcpp::Time & stamp)
  {
    if (!std::isfinite(acceleration_mps2)) {
      return;
    }
    const double acceleration = std::clamp(acceleration_mps2, -25.0, 25.0);
    const double robust_acceleration = sensor_history_median(
      imu_acceleration_history_, acceleration);
    imu_acceleration_history_.push_back(acceleration);
    while (imu_acceleration_history_.size() > imu_acceleration_median_window_) {
      imu_acceleration_history_.pop_front();
    }
    if (!have_imu_speed_stamp_) {
      imu_acceleration_filtered_mps2_ = acceleration;
      imu_acceleration_robust_mps2_ = robust_acceleration;
      imu_observer_acceleration_mps2_ = acceleration;
      imu_speed_mps_ = 0.0;
      last_imu_speed_stamp_ = stamp;
      have_imu_speed_stamp_ = true;
      imu_speed_ready_ = true;
      return;
    }

    const double dt = (stamp - last_imu_speed_stamp_).seconds();
    if (dt > 1.0e-4 && dt <= max_imu_dt_s_) {
      imu_acceleration_filtered_mps2_ =
        imu_acceleration_filter_alpha_ * acceleration +
        (1.0 - imu_acceleration_filter_alpha_) * imu_acceleration_filtered_mps2_;
      imu_acceleration_robust_mps2_ = robust_acceleration;
      // The causal median remains the outlier/stop guard, but it delays a
      // real braking sign change by two native telemetry samples. Integrate
      // the EMA of the raw longitudinal signal so a sustained brake is
      // reflected promptly in the motion state.
      imu_observer_acceleration_mps2_ =
        imu_acceleration_filter_alpha_ * acceleration +
        (1.0 - imu_acceleration_filter_alpha_) * imu_observer_acceleration_mps2_;
      imu_speed_mps_ = std::clamp(
        imu_speed_mps_ + imu_observer_acceleration_mps2_ * dt, 0.0, 30.0);
    }
    last_imu_speed_stamp_ = stamp;
    have_imu_speed_stamp_ = true;
    imu_speed_ready_ = true;
  }

  void encoder_callback(const sensor_msgs::msg::JointState & msg, bool left)
  {
    if (msg.position.empty()) {
      return;
    }
    const double angle = msg.position.front() * encoder_scale_;
    if (!std::isfinite(angle)) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (left) {
      left_angle_ = angle;
      left_stamp_ = rclcpp::Time(msg.header.stamp, get_clock()->get_clock_type());
      have_left_ = true;
      left_updated_ = true;
    } else {
      right_angle_ = angle;
      right_stamp_ = rclcpp::Time(msg.header.stamp, get_clock()->get_clock_type());
      have_right_ = true;
      right_updated_ = true;
    }

  }

  void integrate_pair()
  {
    const rclcpp::Time stamp = left_stamp_ > right_stamp_ ? left_stamp_ : right_stamp_;
    const double pair_skew = std::abs((left_stamp_ - right_stamp_).seconds());
    frozen_encoder_model_active_ = false;
    frozen_encoder_model_decel_mps2_ = 0.0;

    if (!encoder_initialized_) {
      prev_left_ = left_angle_;
      prev_right_ = right_angle_;
      prev_stamp_ = stamp;
      prev_yaw_ = odom_yaw_;
      speed_mps_ = 0.0;
      reset_longitudinal_observer();
      recent_raw_speeds_.clear();
      sensor_fusion_raw_history_.clear();
      sensor_fusion_mapped_history_.clear();
      sensor_fusion_imu_history_.clear();
      sensor_fusion_gap_history_.clear();
      sensor_fusion_encoder_history_.clear();
      steady_mapped_speed_history_.clear();
      raw_wheel_speed_mps_ = 0.0;
      corrected_wheel_speed_mps_ = 0.0;
      longitudinal_slip_ = 0.0;
      wheel_observation_confidence_ = 1.0;
      encoder_initialized_ = true;
      publish_odom(stamp, 0.0);
      return;
    }

    const double dt = (stamp - prev_stamp_).seconds();
    if (dt <= 1.0e-4 || dt > 0.5) {
      prev_left_ = left_angle_;
      prev_right_ = right_angle_;
      prev_stamp_ = stamp;
      prev_yaw_ = odom_yaw_;
      // A long sensor gap invalidates the cumulative window as well as the
      // single-sample derivative. Do not let pre-gap travel leak into the
      // next causal model feature vector.
      recent_raw_speeds_.clear();
      sensor_fusion_raw_history_.clear();
      sensor_fusion_mapped_history_.clear();
      sensor_fusion_imu_history_.clear();
      sensor_fusion_gap_history_.clear();
      sensor_fusion_encoder_history_.clear();
      steady_mapped_speed_history_.clear();
      sensor_fusion_window_raw_speed_mps_ = 0.0;
      sensor_fusion_window_mapped_speed_mps_ = 0.0;
      return;
    }

    // JointState.position is the encoder angle exposed by the official bridge.
    // The bridge uses the same raw angle for the rear-wheel rotation.
    const double dl = wheel_radius_ * (left_angle_ - prev_left_);
    const double dr = wheel_radius_ * (right_angle_ - prev_right_);

    // Do not turn a simulator reset, encoder rollover, or dropped/replayed
    // encoder sample into metres of fictitious motion.  Re-baselining keeps
    // /odom continuous and lets the next valid pair resume integration.
    const bool paired_encoder_reset =
      dl < -encoder_reset_threshold_rad_ * wheel_radius_ &&
      dr < -encoder_reset_threshold_rad_ * wheel_radius_;
    if (paired_encoder_reset || std::abs(dl) > max_encoder_step_m_ ||
        std::abs(dr) > max_encoder_step_m_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Encoder discontinuity/reset (left=%.3f m right=%.3f m); rebaselining only",
        dl, dr);
      prev_left_ = left_angle_;
      prev_right_ = right_angle_;
      prev_stamp_ = stamp;
      recent_raw_speeds_.clear();
      sensor_fusion_raw_history_.clear();
      sensor_fusion_mapped_history_.clear();
      sensor_fusion_imu_history_.clear();
      sensor_fusion_gap_history_.clear();
      sensor_fusion_encoder_history_.clear();
      steady_mapped_speed_history_.clear();
      sensor_fusion_window_raw_speed_mps_ = 0.0;
      sensor_fusion_window_mapped_speed_mps_ = 0.0;
      prev_yaw_ = odom_yaw_;
      raw_wheel_speed_mps_ = std::numeric_limits<double>::quiet_NaN();
      corrected_wheel_speed_mps_ = std::numeric_limits<double>::quiet_NaN();
      longitudinal_slip_ = std::numeric_limits<double>::quiet_NaN();
      ++encoder_reset_count_;
      encoder_reset_active_ = true;
      last_encoder_reset_stamp_ = stamp;
      publish_odom(stamp, speed_mps_);
      return;
    }

    const double wheel_distance = 0.5 * (dl + dr);
    const double wheel_speed = wheel_distance / dt;
    const double wheel_speed_abs = std::abs(wheel_speed);
    const double encoder_position_m = 0.5 * wheel_radius_ *
      (left_angle_ + right_angle_);
    const double window_raw_speed = sensor_fusion_window_raw_speed(
      stamp.seconds(), encoder_position_m, wheel_speed_abs);
    sensor_fusion_window_raw_speed_mps_ = window_raw_speed;
    sensor_fusion_window_mapped_speed_mps_ =
      body_speed_from_wheel_speed(window_raw_speed);
    if (wheel_distance > 0.0) {
      last_motion_sign_ = 1.0;
    } else if (wheel_distance < 0.0) {
      last_motion_sign_ = -1.0;
    }
    const double mapped_speed = body_speed_from_wheel_speed(wheel_speed_abs);
    const double imu_body_speed = std::clamp(imu_speed_mps_, 0.0, 30.0);
    const double slip_denominator = std::max(imu_body_speed, 0.25);
    const double imu_longitudinal_slip =
      (wheel_speed - imu_body_speed) / slip_denominator;
    const double absolute_longitudinal_slip = std::abs(imu_longitudinal_slip);
    raw_wheel_speed_mps_ = wheel_speed;
    corrected_wheel_speed_mps_ = std::copysign(mapped_speed, wheel_speed);
    const bool slip_observable = imu_speed_ready_ &&
      imu_body_speed > slip_observation_min_speed_mps_;
    const bool high_slip = slip_observable &&
      absolute_longitudinal_slip >= forward_extremum_slip_;
    const bool asymptotic_slip = high_slip &&
      absolute_longitudinal_slip >= forward_asymptote_slip_;
    // The documented force curve is not a velocity inversion.  It does,
    // however, provide a physically meaningful confidence transition: once
    // the tire force falls from the extremum to the asymptote, the driven
    // wheel angle carries progressively less information about body speed.
    // Use that transition as a wheel-confidence feature for the fitted
    // sensor-only fusion model and as the conservative fallback weight when
    // the learned model is disabled.
    const double wheel_observation_confidence = slip_observable ?
      longitudinal_wheel_observation_confidence(absolute_longitudinal_slip) : 1.0;
    const double slip_threshold = std::max(
      wheel_slip_threshold_mps_,
      wheel_slip_ratio_ * std::max(wheel_speed_abs, imu_speed_mps_));
    const bool slip_map_active = wheel_speed_abs >= wheel_slip_activation_speed_mps_;
    double fused_speed = mapped_speed;
    bool encoder_dropout_hold = false;
    sensor_fusion_model_active_ = false;
    sensor_fusion_model_speed_mps_ = 0.0;
    sensor_fusion_model_spread_mps_ = 0.0;
    if (std::abs(wheel_speed) <= stationary_speed_threshold_mps_) {
      zero_encoder_duration_s_ += std::max(0.0, dt);
      // The simulator may freeze the driven-wheel encoder during a brake or
      // passive coast. Continue with the independent IMU prediction while it
      // is moving; only declare a stop after speed and acceleration have
      // stayed near zero for the confirmation interval. No command or
      // actuator feedback is required by this estimator.
      const bool imu_acceleration_quiet =
        std::abs(imu_acceleration_robust_mps2_) <=
        imu_stationary_acceleration_threshold_mps2_;
      const bool imu_stop_confirmed =
        zero_encoder_duration_s_ >= zero_encoder_stop_confirm_sec_ &&
        imu_acceleration_quiet &&
        imu_speed_mps_ <= imu_stop_speed_threshold_mps_;
      if (!imu_stop_confirmed && imu_speed_mps_ > stationary_speed_threshold_mps_) {
        if (frozen_encoder_model_enabled_ && imu_acceleration_quiet &&
            zero_encoder_duration_s_ >= frozen_encoder_model_delay_sec_) {
          // The fit is deliberately a deceleration prior, not a velocity
          // reset. It keeps the car moving for the modelled remaining slip
          // distance and then lets the normal stop confirmation gate close.
          frozen_encoder_model_decel_mps2_ = std::clamp(
            frozen_encoder_decel_intercept_mps2_ +
            frozen_encoder_decel_speed_gain_per_s_ * imu_speed_mps_,
            0.0, frozen_encoder_decel_max_mps2_);
          imu_speed_mps_ = std::max(
            0.0, imu_speed_mps_ - frozen_encoder_model_decel_mps2_ * dt);
          frozen_encoder_model_active_ = true;
        }
        fused_speed = frozen_encoder_model_active_ ?
          std::clamp(imu_speed_mps_, 0.0, 30.0) : imu_body_speed;
        encoder_dropout_hold = true;
      } else {
        fused_speed = 0.0;
        imu_speed_mps_ = 0.0;
      }
      // A frozen wheel at non-zero body speed has Sx approximately -1.0,
      // which is beyond the documented asymptote. Mark it as uncertain so
      // the downstream EKF gives AMCL more authority over the pose.
      wheel_slip_detected_ = asymptotic_slip || encoder_dropout_hold;
      wheel_observation_confidence_ = encoder_dropout_hold ? 0.0 : 1.0;
    } else if (imu_speed_ready_) {
      zero_encoder_duration_s_ = 0.0;
      // The calibrated map converts the documented encoder geometry into a
      // body-speed measurement, including the measured driven-wheel slip.
      // Use it as an absolute correction only when it is physically plausible
      // relative to the IMU prediction. Within the calibrated operating
      // envelope the map and IMU should agree; under a spinning-wheel burst
      // the map can be much higher than body speed and must be rejected.
      const double imu_body_speed = std::clamp(imu_speed_mps_, 0.0, 30.0);
      const double wheel_imu_gap = wheel_speed_abs - imu_body_speed;
      wheel_slip_detected_ = slip_map_active &&
        (high_slip || wheel_imu_gap > slip_threshold);
      const double max_map_prediction_gap = observer_speed_tolerance_mps_ +
        max_observer_accel_mps2_ * dt;
      if (!asymptotic_slip &&
        std::abs(mapped_speed - imu_body_speed) <= max_map_prediction_gap) {
        // Keep the instantaneous correction bounded. Replacing the IMU
        // integration with every accepted wheel sample lets a short wheel-spin
        // burst become the next prediction and can hold the speed controller
        // in false overspeed.
        fused_speed = imu_body_speed + wheel_observer_correction_gain_ * (
          wheel_observation_confidence * (mapped_speed - imu_body_speed));
        // Slowly bring the IMU-speed state toward a trusted wheel/map
        // observation. This prevents long-run acceleration-bias drift while
        // preserving the high-slip innovation rejection above.
        imu_speed_mps_ = std::clamp(
          imu_body_speed + imu_speed_correction_gain_ *
          wheel_observation_confidence * (mapped_speed - imu_body_speed),
          0.0, 30.0);
      } else {
        fused_speed = imu_body_speed;
        // Do not repeatedly compare the same delayed encoder burst with the
        // IMU prediction on the next callback.  The next valid pair is the
        // first opportunity to resynchronize the absolute wheel measurement.
        prev_left_ = left_angle_;
        prev_right_ = right_angle_;
        prev_stamp_ = stamp;
        recent_raw_speeds_.clear();
      }
      wheel_observation_confidence_ = wheel_observation_confidence;
    } else {
      wheel_slip_detected_ = false;
      wheel_observation_confidence_ = 1.0;
    }

    // This model is fitted offline against simulator truth but is causal at
    // runtime: it only sees the independent IMU observer, the current
    // encoder increment, the documented wheel-speed map, timing, signed IMU
    // acceleration, and short sensor histories. The frozen model is trained
    // only on moving zero-encoder samples; stationary samples never select a
    // speed model.
    const double model_imu_speed = std::clamp(imu_speed_mps_, 0.0, 30.0);
    const bool frozen_motion = encoder_dropout_hold &&
      model_imu_speed > stationary_speed_threshold_mps_;
    const bool model_motion =
      wheel_speed_abs > stationary_speed_threshold_mps_ || frozen_motion;
    if (imu_speed_ready_ && model_motion) {
      motion_regime_ = frozen_motion ?
        SensorMotionRegime::FROZEN : classify_motion_regime();
    }
    const bool learned_steady_model_allowed =
      sensor_fusion_steady_model_enabled_ ||
      motion_regime_ != SensorMotionRegime::STEADY;
    if (sensor_fusion_model_enabled_ && imu_speed_ready_ &&
      model_motion &&
      learned_steady_model_allowed)
    {
      const auto features = sensor_fusion_features(
        wheel_speed, mapped_speed, model_imu_speed, dt, pair_skew,
        wheel_observation_confidence, window_raw_speed);
      const auto prediction = sensor_fusion_model_.predict(motion_regime_, features);
      sensor_fusion_model_speed_mps_ = prediction.speed_mps;
      sensor_fusion_model_spread_mps_ = prediction.spread_mps;
      const bool prediction_finite = prediction.valid &&
        std::isfinite(prediction.speed_mps) &&
        std::isfinite(prediction.spread_mps);
      const bool prediction_bounded = prediction.spread_mps <=
        regime_model_max_spread_mps_ &&
        std::abs(prediction.speed_mps - fused_speed) <=
        regime_model_max_baseline_delta_mps_;
      if (prediction_finite && prediction_bounded) {
        sensor_fusion_model_active_ = true;
        fused_speed = prediction.speed_mps;
      }
    }

    // Only collect absolute wheel/map observations while the observer is in
    // the steady regime. A launch or braking sample at the same speed is not
    // interchangeable with a constant-speed sample and must not be allowed
    // to re-anchor the IMU state later.
    // The IMU-integrated observer can retain a positive speed bias after a
    // hard launch. Once longitudinal acceleration is quiet, a stable mapped
    // wheel observation below that observer is the useful absolute speed
    // reference. This is deliberately one-sided: a wheel speed above the
    // IMU remains subject to the documented driven-wheel slip gates. The
    // short history also prevents one delayed/quantized encoder sample from
    // becoming a new observer state.
    // Calculate the reference from prior steady observations. Including the
    // current sample twice would make one low encoder packet look like a
    // stable speed and could re-anchor the observer to that outlier.
    const double steady_mapped_speed = sensor_history_median_existing(
      steady_mapped_speed_history_);
    const bool steady_wheel_consistent = std::isfinite(steady_mapped_speed) &&
      std::abs(mapped_speed - steady_mapped_speed) <=
      observer_speed_tolerance_mps_ + max_observer_accel_mps2_ * dt;
    const bool steady_encoder_reanchor = steady_encoder_reanchor_enabled_ &&
      motion_regime_ == SensorMotionRegime::STEADY &&
      !encoder_dropout_hold &&
      steady_mapped_speed_history_.size() >= 2 &&
      std::abs(imu_acceleration_robust_mps2_) <=
      regime_acceleration_threshold_mps2_ &&
      mapped_speed > stationary_speed_threshold_mps_ &&
      steady_mapped_speed > stationary_speed_threshold_mps_ &&
      // At low and moderate speed the calibrated wheel/map observation is
      // the useful absolute velocity reference.  The old one-sided gate
      // only accepted a wheel value below the IMU observer, so a launch bias
      // could leave /odom permanently low even after the car had settled.
      // Reject large disagreements here; the slip-confidence gate below
      // keeps wheel-spin bursts from becoming a new steady reference.
      std::abs(steady_mapped_speed - model_imu_speed) <=
      observer_speed_tolerance_mps_ &&
      wheel_observation_confidence >= 0.75 &&
      steady_wheel_consistent;
    if (steady_encoder_reanchor) {
      imu_speed_mps_ = std::clamp(steady_mapped_speed, 0.0, 30.0);
      fused_speed = imu_speed_mps_;
      wheel_slip_detected_ = false;
      wheel_observation_confidence_ = 1.0;
    }

    if (motion_regime_ != SensorMotionRegime::STEADY) {
      steady_mapped_speed_history_.clear();
    } else if (mapped_speed > stationary_speed_threshold_mps_ &&
      wheel_observation_confidence >= 0.75) {
      // Keep low-confidence wheel-spin samples out of the steady reference
      // history.  Otherwise a single driven-wheel burst can poison the
      // median for the next several native telemetry cycles.
      steady_mapped_speed_history_.push_back(mapped_speed);
      while (steady_mapped_speed_history_.size() > 5) {
        steady_mapped_speed_history_.pop_front();
      }
    }

    fused_speed = std::clamp(fused_speed, 0.0, 30.0);
    // Report slip against the final allowed body-speed estimate. The IMU
    // innovation still controls wheel trust; this diagnostic is less
    // sensitive to IMU integration lag than the raw IMU ratio alone.
    longitudinal_slip_ = (wheel_speed - fused_speed) /
      std::max(std::abs(fused_speed), 0.25);
    // Integrate the observer's body speed for both pose and twist. Raw wheel
    // distance is not a body-distance increment while driven-wheel slip is
    // present; yaw still comes from the allowed IMU gyro/orientation.
    // Do not integrate a repeated zero encoder position as motion.  Only the
    // explicitly bounded dropout hold may use the IMU speed when the wheel
    // delta is zero.
    const double ds = (wheel_speed_abs > stationary_speed_threshold_mps_ ||
      encoder_dropout_hold) ?
      std::copysign(fused_speed * dt, wheel_distance == 0.0 ?
      (last_motion_sign_ >= 0.0 ? 1.0 : -1.0) : wheel_distance) : 0.0;

    remember_sensor_fusion_sample(
      std::abs(wheel_speed), mapped_speed, imu_body_speed,
      std::abs(mapped_speed - imu_body_speed));
    remember_sensor_fusion_encoder_sample(stamp.seconds(), encoder_position_m);

    const double dyaw = wrap_angle(odom_yaw_ - prev_yaw_);
    const double yaw_mid = prev_yaw_ + 0.5 * dyaw;
    x_ += ds * std::cos(yaw_mid);
    y_ += ds * std::sin(yaw_mid);

    prev_left_ = left_angle_;
    prev_right_ = right_angle_;
    prev_stamp_ = stamp;
    prev_yaw_ = odom_yaw_;
    double raw_speed = fused_speed;
    // Encoder messages can arrive in a short burst after bridge scheduling.
    // Limit only the velocity sample used by downstream speed control; the
    // integrated pose above still uses the complete valid mapped increment.
    if (std::isfinite(speed_mps_)) {
      const double max_speed_step = max_velocity_accel_mps2_ * dt;
      raw_speed = std::clamp(
        raw_speed, speed_mps_ - max_speed_step, speed_mps_ + max_speed_step);
    }
    recent_raw_speeds_.push_back(raw_speed);
    if (wheel_speed_abs <= stationary_speed_threshold_mps_) {
      // Keep a moving IMU observer through a repeated encoder position. Only
      // publish zero when the observer itself has reached standstill.
      const double previous_speed = speed_mps_;
      std::vector<double> sorted_speeds(
        recent_raw_speeds_.begin(), recent_raw_speeds_.end());
      std::sort(sorted_speeds.begin(), sorted_speeds.end());
      const double robust_speed = sorted_speeds.empty() ?
        fused_speed : sorted_speeds[sorted_speeds.size() / 2];
      const double filter_alpha = robust_speed < previous_speed ?
        velocity_filter_decel_alpha_ : velocity_filter_alpha_;
      speed_mps_ = filter_alpha * robust_speed +
        (1.0 - filter_alpha) * previous_speed;
      recent_raw_speeds_.clear();
    } else {
      while (recent_raw_speeds_.size() > velocity_median_window_) {
        recent_raw_speeds_.pop_front();
      }
      std::vector<double> sorted_speeds(
        recent_raw_speeds_.begin(), recent_raw_speeds_.end());
      std::sort(sorted_speeds.begin(), sorted_speeds.end());
      const double robust_speed = sorted_speeds[sorted_speeds.size() / 2];
      const double filter_alpha = robust_speed < speed_mps_ ?
        velocity_filter_decel_alpha_ : velocity_filter_alpha_;
      speed_mps_ = filter_alpha * robust_speed +
        (1.0 - filter_alpha) * speed_mps_;
    }
    publish_odom(stamp, speed_mps_);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Odometry: dl=%.3f dr=%.3f ds=%.3f wheel_speed=%.3f mapped=%.3f body_speed=%.3f "
      "speed=%.3f pose=(%.3f, %.3f, %.3f) enc=(%.3f, %.3f) wheel_slip=%s confidence=%.2f",
      dl, dr, ds, wheel_speed_abs, mapped_speed, fused_speed, speed_mps_, x_, y_, odom_yaw_,
      left_angle_, right_angle_,
      wheel_slip_detected_ ? "true" : "false", wheel_observation_confidence);
  }

  void reset_longitudinal_observer()
  {
    imu_speed_mps_ = 0.0;
    imu_acceleration_filtered_mps2_ = 0.0;
    imu_acceleration_robust_mps2_ = 0.0;
    imu_observer_acceleration_mps2_ = 0.0;
    imu_acceleration_history_.clear();
    imu_speed_ready_ = false;
    have_imu_speed_stamp_ = false;
    zero_encoder_duration_s_ = 0.0;
    wheel_slip_detected_ = false;
    wheel_observation_confidence_ = 1.0;
    frozen_encoder_model_active_ = false;
    frozen_encoder_model_decel_mps2_ = 0.0;
  }

  double body_speed_from_wheel_speed(double wheel_speed) const
  {
    const double speed = std::max(0.0, wheel_speed);
    if (speed <= wheel_speed_map_wheel_mps_.front()) {
      return wheel_speed_map_body_mps_.front();
    }
    if (speed >= wheel_speed_map_wheel_mps_.back()) {
      return wheel_speed_map_body_mps_.back();
    }
    const auto upper = std::upper_bound(
      wheel_speed_map_wheel_mps_.begin(), wheel_speed_map_wheel_mps_.end(), speed);
    const size_t upper_index = static_cast<size_t>(
      std::distance(wheel_speed_map_wheel_mps_.begin(), upper));
    const size_t lower_index = upper_index - 1;
    const double lower_speed = wheel_speed_map_wheel_mps_[lower_index];
    const double upper_speed = wheel_speed_map_wheel_mps_[upper_index];
    const double ratio = (speed - lower_speed) / (upper_speed - lower_speed);
    return wheel_speed_map_body_mps_[lower_index] + ratio * (
      wheel_speed_map_body_mps_[upper_index] -
      wheel_speed_map_body_mps_[lower_index]);
  }

  double longitudinal_wheel_observation_confidence(double absolute_slip) const
  {
    if (!std::isfinite(absolute_slip) ||
      absolute_slip >= forward_asymptote_slip_)
    {
      return 0.0;
    }
    if (absolute_slip <= forward_extremum_slip_) {
      return 1.0;
    }

    // Hermite interpolation of the documented force values with zero slope
    // at both knots.  Normalize the force fall so confidence is one at the
    // extremum and zero at the asymptote.
    const double span = forward_asymptote_slip_ - forward_extremum_slip_;
    const double t = std::clamp(
      (absolute_slip - forward_extremum_slip_) / span, 0.0, 1.0);
    const double smoothstep = t * t * (3.0 - 2.0 * t);
    const double force = forward_extremum_value_ +
      (forward_asymptote_value_ - forward_extremum_value_) * smoothstep;
    return std::clamp(
      (force - forward_asymptote_value_) /
      (forward_extremum_value_ - forward_asymptote_value_), 0.0, 1.0);
  }

  static double sensor_history_median(
    const std::deque<double> & history, double current)
  {
    std::vector<double> values;
    constexpr size_t kHistoryWindow = 9;
    const size_t first = history.size() > kHistoryWindow ?
      history.size() - kHistoryWindow : 0;
    values.reserve(history.size() - first + 1);
    for (size_t index = first; index < history.size(); ++index) {
      values.push_back(history[index]);
    }
    values.push_back(current);
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  }

  static double sensor_history_median_existing(
    const std::deque<double> & history)
  {
    if (history.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> values(history.begin(), history.end());
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  }

  std::array<double, f1tenth_localization::SensorFusionModel::kFeatureCount>
  sensor_fusion_features(
    double wheel_speed, double mapped_speed, double imu_speed, double dt,
    double pair_skew, double confidence, double window_raw_speed) const
  {
    const double raw = std::abs(wheel_speed);
    const double mapped = std::abs(mapped_speed);
    const double imu = std::max(0.0, imu_speed);
    const double signed_gap = mapped - imu;
    const double gap = std::abs(signed_gap);
    const double previous_raw = sensor_fusion_raw_history_.empty() ?
      raw : sensor_fusion_raw_history_.back();
    const double previous_mapped = sensor_fusion_mapped_history_.empty() ?
      mapped : sensor_fusion_mapped_history_.back();
    const double previous_imu = sensor_fusion_imu_history_.empty() ?
      imu : sensor_fusion_imu_history_.back();
    return {
      imu,
      mapped,
      raw,
      signed_gap,
      gap,
      std::max(0.0, confidence),
      imu_raw_acceleration_mps2_,
      imu_acceleration_filtered_mps2_,
      std::abs(imu_lateral_acceleration_mps2_),
      std::abs(imu_yaw_rate_),
      std::max(0.0, dt),
      std::max(0.0, pair_skew),
      raw <= stationary_speed_threshold_mps_ ? 1.0 : 0.0,
      sensor_history_median(sensor_fusion_raw_history_, raw),
      sensor_history_median(sensor_fusion_mapped_history_, mapped),
      sensor_history_median(sensor_fusion_imu_history_, imu),
      sensor_history_median(sensor_fusion_gap_history_, signed_gap),
      raw - previous_raw,
      mapped - previous_mapped,
      imu - previous_imu,
      window_raw_speed,
      body_speed_from_wheel_speed(window_raw_speed)};
  }

  SensorMotionRegime classify_motion_regime() const
  {
    // Use the promptly filtered acceleration for regime transitions. The
    // causal median remains the stop/outlier guard; using it here would delay
    // a real brake sign change by multiple native telemetry samples.
    if (imu_acceleration_filtered_mps2_ > regime_acceleration_threshold_mps2_) {
      return SensorMotionRegime::ACCELERATING;
    }
    if (imu_acceleration_filtered_mps2_ < -regime_acceleration_threshold_mps2_) {
      return SensorMotionRegime::DECELERATING;
    }
    return SensorMotionRegime::STEADY;
  }

  double motion_regime_code() const
  {
    switch (motion_regime_) {
      case SensorMotionRegime::ACCELERATING:
        return 0.0;
      case SensorMotionRegime::STEADY:
        return 1.0;
      case SensorMotionRegime::DECELERATING:
        return 2.0;
      case SensorMotionRegime::FROZEN:
        return 3.0;
    }
    return 1.0;
  }

  void remember_sensor_fusion_sample(
    double raw, double mapped, double imu, double gap)
  {
    sensor_fusion_raw_history_.push_back(raw);
    sensor_fusion_mapped_history_.push_back(mapped);
    sensor_fusion_imu_history_.push_back(imu);
    sensor_fusion_gap_history_.push_back(gap);
    while (sensor_fusion_raw_history_.size() > 9) {
      sensor_fusion_raw_history_.pop_front();
      sensor_fusion_mapped_history_.pop_front();
      sensor_fusion_imu_history_.pop_front();
      sensor_fusion_gap_history_.pop_front();
    }
  }

  double sensor_fusion_window_raw_speed(
    double stamp_s, double position_m, double current_speed) const
  {
    if (sensor_fusion_encoder_history_.empty() ||
      stamp_s <= sensor_fusion_encoder_history_.front().first)
    {
      return current_speed;
    }
    return std::abs(position_m - sensor_fusion_encoder_history_.front().second) /
      (stamp_s - sensor_fusion_encoder_history_.front().first);
  }

  void remember_sensor_fusion_encoder_sample(double stamp_s, double position_m)
  {
    sensor_fusion_encoder_history_.emplace_back(stamp_s, position_m);
    while (sensor_fusion_encoder_history_.size() > 9) {
      sensor_fusion_encoder_history_.pop_front();
    }
  }

  void publish_odom(const rclcpp::Time & stamp, double speed)
  {
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = odom_frame_;
    msg.child_frame_id = base_frame_;
    msg.pose.pose.position.x = x_;
    msg.pose.pose.position.y = y_;
    msg.pose.pose.orientation = quaternion_from_yaw(odom_yaw_);
    msg.twist.twist.linear.x = speed;
    msg.twist.twist.angular.z = imu_yaw_rate_;

    const double confidence = std::clamp(wheel_observation_confidence_, 0.0, 1.0);
    const double pose_variance = pose_xy_var_ +
      (slip_pose_xy_var_ - pose_xy_var_) * (1.0 - confidence);
    const double reset_age_s = encoder_reset_active_ ?
      (stamp - last_encoder_reset_stamp_).seconds() :
      std::numeric_limits<double>::infinity();
    const bool reset_covariance_active = encoder_reset_active_ &&
      reset_age_s >= 0.0 && reset_age_s <= encoder_reset_covariance_duration_s_;
    const double published_pose_variance = reset_covariance_active ?
      std::max(pose_variance, encoder_reset_pose_xy_var_) : pose_variance;
    const double published_twist_variance = reset_covariance_active ?
      std::max(twist_linear_var_, encoder_reset_twist_linear_var_) :
      twist_linear_var_;
    msg.pose.covariance[0] = published_pose_variance;
    msg.pose.covariance[7] = published_pose_variance;
    msg.pose.covariance[35] = pose_yaw_var_;
    msg.twist.covariance[0] = published_twist_variance;
    msg.twist.covariance[35] = twist_yaw_var_;
    odom_pub_->publish(msg);

    std_msgs::msg::Float64MultiArray diagnostics;
    diagnostics.layout.dim.resize(1);
    diagnostics.layout.dim[0].label =
      "raw_wheel_speed_mps,corrected_wheel_speed_mps,longitudinal_slip_ratio,"
      "wheel_observation_confidence,imu_acceleration_bias_mps2,encoder_reset_count,"
      "imu_speed_mps,frozen_encoder_model_active,frozen_encoder_model_decel_mps2,"
      "sensor_fusion_model_speed_mps,sensor_fusion_model_spread_mps,"
      "sensor_fusion_model_active,sensor_motion_regime,"
      "sensor_fusion_window_raw_speed_mps,sensor_fusion_window_mapped_speed_mps";
    diagnostics.layout.dim[0].size = 15;
    diagnostics.layout.dim[0].stride = 15;
    diagnostics.data = {
      raw_wheel_speed_mps_, corrected_wheel_speed_mps_, longitudinal_slip_,
      confidence, 0.0, static_cast<double>(encoder_reset_count_),
      imu_speed_mps_, frozen_encoder_model_active_ ? 1.0 : 0.0,
      frozen_encoder_model_decel_mps2_, sensor_fusion_model_speed_mps_,
      sensor_fusion_model_spread_mps_, sensor_fusion_model_active_ ? 1.0 : 0.0,
      motion_regime_code(), sensor_fusion_window_raw_speed_mps_,
      sensor_fusion_window_mapped_speed_mps_};
    diagnostics_pub_->publish(diagnostics);

    geometry_msgs::msg::TransformStamped tf;
    tf.header = msg.header;
    tf.child_frame_id = base_frame_;
    tf.transform.translation.x = x_;
    tf.transform.translation.y = y_;
    tf.transform.rotation = msg.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf);

  }

  std::mutex mutex_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr diagnostics_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr right_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr reset_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  std::string odom_frame_;
  std::string base_frame_;

  double wheel_radius_{0.0590};
  double encoder_scale_{1.0};
  double max_encoder_step_m_{4.0};
  double max_encoder_pair_skew_s_{0.05};
  double pose_xy_var_{0.01};
  double pose_yaw_var_{0.01};
  double velocity_filter_alpha_{0.70};
  double velocity_filter_decel_alpha_{0.70};
  size_t velocity_median_window_{5};
  double max_velocity_accel_mps2_{40.0};
  double twist_linear_var_{0.04};
  double twist_yaw_var_{0.04};
  double encoder_reset_covariance_duration_s_{1.0};
  double encoder_reset_pose_xy_var_{0.25};
  double encoder_reset_twist_linear_var_{0.50};

  bool have_left_{false};
  bool have_right_{false};
  bool left_updated_{false};
  bool right_updated_{false};
  bool encoder_initialized_{false};
  bool imu_initialized_{false};

  double left_angle_{0.0};
  double right_angle_{0.0};
  double prev_left_{0.0};
  double prev_right_{0.0};
  rclcpp::Time left_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time right_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time prev_stamp_{0, 0, RCL_ROS_TIME};

  double imu_yaw_zero_{0.0};
  double imu_raw_yaw_{0.0};
  double last_imu_relative_yaw_{0.0};
  double imu_integrated_yaw_{0.0};
  double imu_yaw_rate_{0.0};
  double odom_yaw_{0.0};
  double prev_yaw_{0.0};
  double speed_mps_{0.0};
  std::deque<double> recent_raw_speeds_;
  rclcpp::Time last_imu_stamp_{0, 0, RCL_ROS_TIME};
  bool have_imu_stamp_{false};
  double imu_orientation_correction_gain_{0.08};
  double max_imu_orientation_step_rad_{0.30};
  double max_imu_dt_s_{0.5};
  double imu_acceleration_filter_alpha_{0.70};
  size_t imu_acceleration_median_window_{3};
  double wheel_slip_threshold_mps_{0.75};
  double wheel_slip_ratio_{0.20};
  double slip_observation_min_speed_mps_{0.75};
  double wheel_slip_activation_speed_mps_{16.0};
  double encoder_reset_threshold_rad_{0.50};
  double max_observer_accel_mps2_{12.0};
  double observer_speed_tolerance_mps_{2.0};
  double wheel_observer_correction_gain_{0.10};
  double imu_speed_correction_gain_{0.0};
  bool sensor_fusion_model_enabled_{true};
  bool sensor_fusion_steady_model_enabled_{false};
  double regime_acceleration_threshold_mps2_{0.50};
  double regime_model_max_spread_mps_{0.75};
  double regime_model_max_baseline_delta_mps_{0.75};
  bool steady_encoder_reanchor_enabled_{true};
  f1tenth_localization::SensorFusionModel sensor_fusion_model_;
  double stationary_speed_threshold_mps_{0.15};
  double forward_extremum_slip_{0.15};
  double forward_extremum_value_{0.72};
  double forward_asymptote_slip_{0.25};
  double forward_asymptote_value_{0.464};
  double imu_stop_speed_threshold_mps_{2.0};
  double imu_stationary_acceleration_threshold_mps2_{0.30};
  double zero_encoder_stop_confirm_sec_{0.80};
  bool frozen_encoder_model_enabled_{true};
  double frozen_encoder_model_delay_sec_{0.15};
  double frozen_encoder_decel_intercept_mps2_{5.5};
  double frozen_encoder_decel_speed_gain_per_s_{0.27};
  double frozen_encoder_decel_max_mps2_{12.0};
  double slip_pose_xy_var_{0.10};
  std::vector<double> wheel_speed_map_wheel_mps_;
  std::vector<double> wheel_speed_map_body_mps_;
  double imu_speed_mps_{0.0};
  double imu_acceleration_filtered_mps2_{0.0};
  double imu_acceleration_robust_mps2_{0.0};
  double imu_observer_acceleration_mps2_{0.0};
  std::deque<double> imu_acceleration_history_;
  double imu_raw_acceleration_mps2_{0.0};
  double imu_lateral_acceleration_mps2_{0.0};
  rclcpp::Time last_imu_speed_stamp_{0, 0, RCL_ROS_TIME};
  bool have_imu_speed_stamp_{false};
  bool imu_speed_ready_{false};
  bool wheel_slip_detected_{false};
  double wheel_observation_confidence_{1.0};
  bool frozen_encoder_model_active_{false};
  double frozen_encoder_model_decel_mps2_{0.0};
  double sensor_fusion_window_raw_speed_mps_{0.0};
  double sensor_fusion_window_mapped_speed_mps_{0.0};
  double raw_wheel_speed_mps_{0.0};
  double corrected_wheel_speed_mps_{0.0};
  double longitudinal_slip_{0.0};
  uint32_t encoder_reset_count_{0};
  bool encoder_reset_active_{false};
  rclcpp::Time last_encoder_reset_stamp_{0, 0, RCL_ROS_TIME};
  double zero_encoder_duration_s_{0.0};
  double last_motion_sign_{1.0};
  bool sensor_fusion_model_active_{false};
  double sensor_fusion_model_speed_mps_{0.0};
  double sensor_fusion_model_spread_mps_{0.0};
  SensorMotionRegime motion_regime_{SensorMotionRegime::STEADY};
  std::deque<double> sensor_fusion_raw_history_;
  std::deque<double> sensor_fusion_mapped_history_;
  std::deque<double> sensor_fusion_imu_history_;
  std::deque<double> sensor_fusion_gap_history_;
  std::deque<double> steady_mapped_speed_history_;
  std::deque<std::pair<double, double>> sensor_fusion_encoder_history_;
  double x_{0.0};
  double y_{0.0};

};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorOdometryNode>());
  rclcpp::shutdown();
  return 0;
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

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
    // Official actuator feedback is allowed runtime input.  It tells the
    // observer whether a frozen driven-wheel encoder is a passive coast or a
    // temporary sensor dropout while propulsion is still applied.
    declare_parameter("throttle_topic", "/autodrive/roboracer_1/throttle");
    declare_parameter("odom_topic", "/odom");

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
    declare_parameter("imu_acceleration_filter_alpha", 0.35);
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
    declare_parameter("wheel_observer_correction_gain", 0.25);
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
    // Ground-truth coast-down identification supplies the process prior used
    // only while the official encoder is in the documented high-slip region.
    // It is blended with the allowed IMU integration, never substituted for
    // the sensor stream during ordinary wheel motion.
    declare_parameter("coast_model_enabled", true);
    declare_parameter("coast_throttle_threshold", 0.02);
    declare_parameter("coast_model_imu_weight", 0.60);
    declare_parameter("coast_deceleration_intercept_mps2", 5.3);
    declare_parameter("coast_deceleration_speed_gain_s_inv", 0.26);
    declare_parameter("coast_deceleration_max_mps2", 12.0);
    declare_parameter("throttle_feedback_timeout_s", 0.5);
    // When the driven-wheel encoder freezes during simulator coasting, use
    // the allowed IMU integration until a low-speed/low-acceleration stop is
    // confirmed.  This avoids cutting off the coast distance while preventing
    // residual IMU bias from producing indefinite motion at rest.
    declare_parameter("imu_stop_speed_threshold_mps", 2.0);
    declare_parameter("imu_stationary_acceleration_threshold_mps2", 0.30);
    declare_parameter("zero_encoder_stop_confirm_sec", 0.80);
    declare_parameter("slip_pose_xy_variance", 0.10);
    declare_parameter(
      "wheel_speed_map_wheel_mps",
      std::vector<double>{0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
        16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0});
    declare_parameter(
      "wheel_speed_map_body_mps",
      std::vector<double>{0.0, 1.979669, 3.928270, 5.845092, 7.738538,
        9.599276, 11.438255, 13.183116, 14.958249, 16.881441,
        18.882715, 20.485775, 22.088834, 22.883600, 22.883600,
        22.883600});

    declare_parameter("pose_xy_variance", 0.01);
    declare_parameter("pose_yaw_variance", 0.01);
    declare_parameter("velocity_filter_alpha", 0.35);
    declare_parameter("velocity_median_window", 3);
    declare_parameter("velocity_filter_decel_alpha", 0.85);
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
    forward_extremum_slip_ = get_parameter("forward_extremum_slip").as_double();
    forward_extremum_value_ = get_parameter("forward_extremum_value").as_double();
    forward_asymptote_slip_ = get_parameter("forward_asymptote_slip").as_double();
    forward_asymptote_value_ = get_parameter("forward_asymptote_value").as_double();
    coast_model_enabled_ = get_parameter("coast_model_enabled").as_bool();
    coast_throttle_threshold_ = std::clamp(
      get_parameter("coast_throttle_threshold").as_double(), 0.0, 1.0);
    coast_model_imu_weight_ = std::clamp(
      get_parameter("coast_model_imu_weight").as_double(), 0.0, 1.0);
    coast_deceleration_intercept_mps2_ = std::max(
      0.0, get_parameter("coast_deceleration_intercept_mps2").as_double());
    coast_deceleration_speed_gain_s_inv_ = std::max(
      0.0, get_parameter("coast_deceleration_speed_gain_s_inv").as_double());
    coast_deceleration_max_mps2_ = std::max(
      0.1, get_parameter("coast_deceleration_max_mps2").as_double());
    throttle_feedback_timeout_s_ = std::max(
      0.05, get_parameter("throttle_feedback_timeout_s").as_double());
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
    slip_pose_xy_var_ = std::max(
      pose_xy_var_, get_parameter("slip_pose_xy_variance").as_double());
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
    throttle_sub_ = create_subscription<std_msgs::msg::Float32>(
      get_parameter("throttle_topic").as_string(), sensor_qos,
      [this](std_msgs::msg::Float32::ConstSharedPtr msg) {
        throttle_callback(*msg);
      });
    publish_static_transforms();
    RCLCPP_INFO(
      get_logger(),
      "Allowed sensor odometry: rear encoders + IMU -> /odom");
  }

private:

  void throttle_callback(const std_msgs::msg::Float32 & msg)
  {
    if (!std::isfinite(msg.data)) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    throttle_feedback_ = std::clamp(static_cast<double>(msg.data), -1.0, 1.0);
    last_throttle_feedback_time_ = now();
    have_throttle_feedback_ = true;
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
        // discontinuity although its gyro stream remains smooth. Never let
        // that one sample teleport odometry. When the absolute orientation is
        // plausible, use it only as a slow drift correction to the gyro.
        if (std::abs(raw_delta) <= max_imu_orientation_step_rad_) {
          const double correction = wrap_angle(raw_relative_yaw - imu_integrated_yaw_);
          imu_integrated_yaw_ = wrap_angle(
            imu_integrated_yaw_ + imu_orientation_correction_gain_ * correction);
        } else {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Ignoring discontinuous IMU yaw sample (step=%.3f rad); using gyro integration",
            raw_delta);
        }
      }
      odom_yaw_ = imu_integrated_yaw_;
      last_imu_relative_yaw_ = raw_relative_yaw;
      last_imu_stamp_ = stamp;
      have_imu_stamp_ = true;
    }
    imu_yaw_rate_ = gyro_z;

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
    if (!have_imu_speed_stamp_) {
      imu_acceleration_filtered_mps2_ = acceleration_mps2;
      imu_speed_mps_ = 0.0;
      last_imu_speed_stamp_ = stamp;
      have_imu_speed_stamp_ = true;
      imu_speed_ready_ = true;
      return;
    }

    const double dt = (stamp - last_imu_speed_stamp_).seconds();
    if (dt > 1.0e-4 && dt <= max_imu_dt_s_) {
      const double acceleration = std::clamp(acceleration_mps2, -25.0, 25.0);
      imu_acceleration_filtered_mps2_ =
        imu_acceleration_filter_alpha_ * acceleration +
        (1.0 - imu_acceleration_filter_alpha_) * imu_acceleration_filtered_mps2_;
      imu_speed_mps_ = std::clamp(
        imu_speed_mps_ + imu_acceleration_filtered_mps2_ * dt, 0.0, 30.0);
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

    if (!encoder_initialized_) {
      prev_left_ = left_angle_;
      prev_right_ = right_angle_;
      prev_stamp_ = stamp;
      prev_yaw_ = odom_yaw_;
      speed_mps_ = 0.0;
      reset_longitudinal_observer();
      recent_raw_speeds_.clear();
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
        "Encoder discontinuity/reset (left=%.3f m right=%.3f m); resetting odometry origin",
        dl, dr);
      prev_left_ = left_angle_;
      prev_right_ = right_angle_;
      prev_stamp_ = stamp;
      x_ = 0.0;
      y_ = 0.0;
      if (imu_initialized_ && std::isfinite(imu_raw_yaw_)) {
        imu_yaw_zero_ = imu_raw_yaw_;
        last_imu_relative_yaw_ = 0.0;
        imu_integrated_yaw_ = 0.0;
        odom_yaw_ = 0.0;
      }
      speed_mps_ = 0.0;
      reset_longitudinal_observer();
      recent_raw_speeds_.clear();
      prev_yaw_ = odom_yaw_;
      publish_odom(stamp, 0.0);
      return;
    }

    const double wheel_distance = 0.5 * (dl + dr);
    const double wheel_speed = wheel_distance / dt;
    const double wheel_speed_abs = std::abs(wheel_speed);
    if (wheel_distance > 0.0) {
      last_motion_sign_ = 1.0;
    } else if (wheel_distance < 0.0) {
      last_motion_sign_ = -1.0;
    }
    const double mapped_speed = body_speed_from_wheel_speed(wheel_speed_abs);
    const double imu_body_speed = std::clamp(imu_speed_mps_, 0.0, 30.0);
    const double slip_denominator = std::max(imu_body_speed, 0.25);
    const double longitudinal_slip =
      (wheel_speed - imu_body_speed) / slip_denominator;
    const double absolute_longitudinal_slip = std::abs(longitudinal_slip);
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
    // Use that transition only as the wheel/IMU fusion weight; the IMU and
    // coast model remain the speed propagation sources beyond the asymptote.
    const double wheel_observation_confidence = slip_observable ?
      longitudinal_wheel_observation_confidence(absolute_longitudinal_slip) : 1.0;
    const double slip_threshold = std::max(
      wheel_slip_threshold_mps_,
      wheel_slip_ratio_ * std::max(wheel_speed_abs, imu_speed_mps_));
    const bool slip_map_active = wheel_speed_abs >= wheel_slip_activation_speed_mps_;
    double fused_speed = mapped_speed;
    bool encoder_dropout_hold = false;
    if (std::abs(wheel_speed) <= stationary_speed_threshold_mps_) {
      zero_encoder_duration_s_ += std::max(0.0, dt);
      // The simulator may freeze the driven-wheel encoder for the complete
      // passive coast after throttle release.  Continue with the independent
      // IMU prediction while it is moving; only declare a stop after both
      // speed and acceleration have stayed near zero for the confirmation
      // interval.  This also covers a single repeated sample without making
      // the pose drift forever after the car stops.
      const bool imu_stop_confirmed =
        zero_encoder_duration_s_ >= zero_encoder_stop_confirm_sec_ &&
        imu_speed_mps_ <= imu_stop_speed_threshold_mps_ &&
        std::abs(imu_acceleration_filtered_mps2_) <=
        imu_stationary_acceleration_threshold_mps2_;
      if (!imu_stop_confirmed && imu_speed_mps_ > stationary_speed_threshold_mps_) {
        const bool throttle_feedback_fresh = have_throttle_feedback_ &&
          std::abs((now() - last_throttle_feedback_time_).seconds()) <=
          throttle_feedback_timeout_s_;
        const bool passive_coast = coast_model_enabled_ &&
          throttle_feedback_fresh &&
          throttle_feedback_ <= coast_throttle_threshold_ &&
          asymptotic_slip;
        if (passive_coast) {
          if (!coast_model_active_) {
            coast_model_speed_mps_ = std::max(speed_mps_, imu_body_speed);
            coast_model_active_ = true;
          }
          const double model_deceleration = std::min(
            coast_deceleration_max_mps2_,
            coast_deceleration_intercept_mps2_ +
            coast_deceleration_speed_gain_s_inv_ * coast_model_speed_mps_);
          coast_model_speed_mps_ = std::max(
            0.0, coast_model_speed_mps_ - model_deceleration * dt);
          fused_speed = coast_model_imu_weight_ * imu_body_speed +
            (1.0 - coast_model_imu_weight_) * coast_model_speed_mps_;
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Asymptotic passive coast: slip=%.3f model=%.3f imu=%.3f fused=%.3f throttle=%.3f",
            longitudinal_slip, coast_model_speed_mps_, imu_body_speed, fused_speed,
            throttle_feedback_);
        } else {
          coast_model_active_ = false;
          fused_speed = imu_body_speed;
        }
        encoder_dropout_hold = true;
      } else {
        fused_speed = 0.0;
        imu_speed_mps_ = 0.0;
        coast_model_active_ = false;
      }
      // A frozen wheel at non-zero body speed has Sx approximately -1.0,
      // which is beyond the documented asymptote. Mark it as uncertain so
      // the downstream EKF gives AMCL more authority over the pose.
      wheel_slip_detected_ = asymptotic_slip || encoder_dropout_hold;
      wheel_observation_confidence_ = encoder_dropout_hold ? 0.0 : 1.0;
    } else if (imu_speed_ready_) {
      zero_encoder_duration_s_ = 0.0;
      coast_model_active_ = false;
      // The calibrated map converts the documented encoder geometry into a
      // body-speed measurement, including the measured driven-wheel slip.
      // Use it as an absolute correction only when it is physically plausible
      // relative to the IMU prediction. Direct-throttle calibration shows
      // that the map and IMU agree when wheel slip is in the calibrated
      // operating envelope; under a spinning-wheel burst the map can be much
      // higher than body speed and must not make the speed controller coast.
      const double imu_body_speed = std::clamp(imu_speed_mps_, 0.0, 30.0);
      const double wheel_imu_gap = wheel_speed_abs - imu_body_speed;
      wheel_slip_detected_ = slip_map_active &&
        (high_slip || wheel_imu_gap > slip_threshold);
      const double max_map_prediction_gap = observer_speed_tolerance_mps_ +
        max_observer_accel_mps2_ * dt;
      if (!asymptotic_slip &&
        std::abs(mapped_speed - imu_body_speed) <= max_map_prediction_gap) {
        // Keep the IMU integration independent.  Replacing it with every
        // accepted wheel sample lets a short wheel-spin burst become the new
        // prediction and can hold the speed controller in false overspeed.
        fused_speed = imu_body_speed + wheel_observer_correction_gain_ * (
          wheel_observation_confidence * (mapped_speed - imu_body_speed));
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
    fused_speed = std::clamp(fused_speed, 0.0, 30.0);
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
      recent_raw_speeds_.clear();
      speed_mps_ = fused_speed;
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
    imu_speed_ready_ = false;
    have_imu_speed_stamp_ = false;
    zero_encoder_duration_s_ = 0.0;
    wheel_slip_detected_ = false;
    wheel_observation_confidence_ = 1.0;
    coast_model_speed_mps_ = 0.0;
    coast_model_active_ = false;
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
    msg.pose.covariance[0] = pose_variance;
    msg.pose.covariance[7] = pose_variance;
    msg.pose.covariance[35] = pose_yaw_var_;
    msg.twist.covariance[0] = twist_linear_var_;
    msg.twist.covariance[35] = twist_yaw_var_;
    odom_pub_->publish(msg);

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
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr right_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr throttle_sub_;
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
  double velocity_filter_alpha_{0.40};
  double velocity_filter_decel_alpha_{0.85};
  size_t velocity_median_window_{5};
  double max_velocity_accel_mps2_{40.0};
  double twist_linear_var_{0.04};
  double twist_yaw_var_{0.04};

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
  double imu_acceleration_filter_alpha_{0.35};
  double wheel_slip_threshold_mps_{0.75};
  double wheel_slip_ratio_{0.20};
  double slip_observation_min_speed_mps_{0.75};
  double wheel_slip_activation_speed_mps_{16.0};
  double encoder_reset_threshold_rad_{0.50};
  double max_observer_accel_mps2_{12.0};
  double observer_speed_tolerance_mps_{2.0};
  double wheel_observer_correction_gain_{0.25};
  double stationary_speed_threshold_mps_{0.15};
  double forward_extremum_slip_{0.15};
  double forward_extremum_value_{0.72};
  double forward_asymptote_slip_{0.25};
  double forward_asymptote_value_{0.464};
  bool coast_model_enabled_{true};
  double coast_throttle_threshold_{0.02};
  double coast_model_imu_weight_{0.60};
  double coast_deceleration_intercept_mps2_{5.3};
  double coast_deceleration_speed_gain_s_inv_{0.26};
  double coast_deceleration_max_mps2_{12.0};
  double throttle_feedback_timeout_s_{0.5};
  double imu_stop_speed_threshold_mps_{2.0};
  double imu_stationary_acceleration_threshold_mps2_{0.30};
  double zero_encoder_stop_confirm_sec_{0.80};
  double slip_pose_xy_var_{0.10};
  std::vector<double> wheel_speed_map_wheel_mps_;
  std::vector<double> wheel_speed_map_body_mps_;
  double imu_speed_mps_{0.0};
  double imu_acceleration_filtered_mps2_{0.0};
  rclcpp::Time last_imu_speed_stamp_{0, 0, RCL_ROS_TIME};
  bool have_imu_speed_stamp_{false};
  bool imu_speed_ready_{false};
  bool wheel_slip_detected_{false};
  double wheel_observation_confidence_{1.0};
  double zero_encoder_duration_s_{0.0};
  double throttle_feedback_{0.0};
  rclcpp::Time last_throttle_feedback_time_{0, 0, RCL_ROS_TIME};
  bool have_throttle_feedback_{false};
  double coast_model_speed_mps_{0.0};
  bool coast_model_active_{false};
  double last_motion_sign_{1.0};
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

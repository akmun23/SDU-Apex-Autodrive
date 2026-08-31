#include "gpu_amcl_cpp/core/ekf_node.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <Eigen/Cholesky>

namespace gpu_amcl_cpp {

namespace {

constexpr double kMinVariance = 1.0e-6;

double finite_or(double value, double fallback)
{
  return std::isfinite(value) ? value : fallback;
}

}  // namespace

EkfNode::EkfNode(const rclcpp::NodeOptions & options)
: Node("ekf_localization", options)
{
  declare_all_parameters();
  load_parameters();

  pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
    output_topic_, rclcpp::QoS(10).reliable());

  const auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_, sensor_qos,
    std::bind(&EkfNode::odom_callback, this, std::placeholders::_1));
  amcl_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    amcl_topic_, rclcpp::QoS(10).reliable(),
    std::bind(&EkfNode::amcl_callback, this, std::placeholders::_1));

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  RCLCPP_INFO(
    get_logger(),
    "EKF ready: odom=%s + AMCL=%s -> %s; odom-dominant causal fusion",
    odom_topic_.c_str(), amcl_topic_.c_str(), output_topic_.c_str());
}

void EkfNode::declare_all_parameters()
{
  declare_parameter("amcl_topic", amcl_topic_);
  declare_parameter("odom_topic", odom_topic_);
  declare_parameter("output_topic", output_topic_);
  declare_parameter("global_frame", global_frame_);
  declare_parameter("odom_frame", odom_frame_);
  declare_parameter("base_frame", base_frame_);
  declare_parameter("transform_tolerance_s", transform_tolerance_s_);
  declare_parameter("process_noise_scale", process_noise_scale_);
  declare_parameter("amcl_max_latency_s", amcl_max_latency_s_);
  declare_parameter("odom_history_duration_s", odom_history_duration_s_);
  declare_parameter("amcl_position_variance_floor", amcl_position_variance_floor_);
  declare_parameter("amcl_yaw_variance_floor", amcl_yaw_variance_floor_);
  declare_parameter("amcl_innovation_gate_distance_m", amcl_innovation_gate_distance_m_);
  declare_parameter("amcl_innovation_gate_yaw_rad", amcl_innovation_gate_yaw_rad_);
  declare_parameter("amcl_jump_reset_enabled", amcl_jump_reset_enabled_);
  declare_parameter("amcl_jump_reset_distance_m", amcl_jump_reset_distance_m_);
  declare_parameter("amcl_jump_reset_yaw_rad", amcl_jump_reset_yaw_rad_);
}

void EkfNode::load_parameters()
{
  amcl_topic_ = get_parameter("amcl_topic").as_string();
  odom_topic_ = get_parameter("odom_topic").as_string();
  output_topic_ = get_parameter("output_topic").as_string();
  global_frame_ = get_parameter("global_frame").as_string();
  odom_frame_ = get_parameter("odom_frame").as_string();
  base_frame_ = get_parameter("base_frame").as_string();
  transform_tolerance_s_ = std::max(0.0, get_parameter("transform_tolerance_s").as_double());
  process_noise_scale_ = std::max(0.0, get_parameter("process_noise_scale").as_double());
  amcl_max_latency_s_ = std::max(0.05, get_parameter("amcl_max_latency_s").as_double());
  odom_history_duration_s_ = std::max(
    0.25, get_parameter("odom_history_duration_s").as_double());
  amcl_position_variance_floor_ = std::max(
    kMinVariance, get_parameter("amcl_position_variance_floor").as_double());
  amcl_yaw_variance_floor_ = std::max(
    kMinVariance, get_parameter("amcl_yaw_variance_floor").as_double());
  amcl_innovation_gate_distance_m_ = std::max(
    0.0, get_parameter("amcl_innovation_gate_distance_m").as_double());
  amcl_innovation_gate_yaw_rad_ = std::max(
    0.0, get_parameter("amcl_innovation_gate_yaw_rad").as_double());
  amcl_jump_reset_enabled_ = get_parameter("amcl_jump_reset_enabled").as_bool();
  amcl_jump_reset_distance_m_ = std::max(
    0.0, get_parameter("amcl_jump_reset_distance_m").as_double());
  amcl_jump_reset_yaw_rad_ = std::max(
    0.0, get_parameter("amcl_jump_reset_yaw_rad").as_double());
}

void EkfNode::push_odom_sample(
  const rclcpp::Time & stamp, const Eigen::Vector3d & pose)
{
  if (!odom_history_.empty() && stamp <= odom_history_.back().stamp) {
    return;
  }
  odom_history_.push_back({stamp, pose});
  while (odom_history_.size() > 2 &&
    (stamp - odom_history_.front().stamp).seconds() > odom_history_duration_s_)
  {
    odom_history_.pop_front();
  }
}

bool EkfNode::interpolate_odom_pose(
  const rclcpp::Time & stamp, Eigen::Vector3d & pose_out) const
{
  if (odom_history_.empty()) {
    return false;
  }
  if (stamp <= odom_history_.front().stamp) {
    pose_out = odom_history_.front().pose;
    return true;
  }
  if (stamp >= odom_history_.back().stamp) {
    pose_out = odom_history_.back().pose;
    return true;
  }

  for (size_t i = 1; i < odom_history_.size(); ++i) {
    const auto & before = odom_history_[i - 1];
    const auto & after = odom_history_[i];
    if (stamp > after.stamp) {
      continue;
    }
    const double span = (after.stamp - before.stamp).seconds();
    if (span <= 1.0e-9) {
      pose_out = after.pose;
      return true;
    }
    const double ratio = std::clamp(
      (stamp - before.stamp).seconds() / span, 0.0, 1.0);
    pose_out[0] = before.pose[0] + ratio * (after.pose[0] - before.pose[0]);
    pose_out[1] = before.pose[1] + ratio * (after.pose[1] - before.pose[1]);
    pose_out[2] = math_utils::normalize_angle(
      before.pose[2] + ratio * math_utils::angle_diff(after.pose[2], before.pose[2]));
    return true;
  }
  return false;
}

Eigen::Matrix3d EkfNode::odom_process_covariance(
  const nav_msgs::msg::Odometry & msg) const
{
  Eigen::Matrix3d q = Eigen::Matrix3d::Zero();
  const auto & c = msg.pose.covariance;
  q(0, 0) = std::max(kMinVariance, finite_or(c[0], 0.01));
  q(0, 1) = finite_or(c[1], 0.0);
  q(1, 0) = finite_or(c[6], 0.0);
  q(1, 1) = std::max(kMinVariance, finite_or(c[7], 0.01));
  q(2, 2) = std::max(kMinVariance, finite_or(c[35], 0.01));
  return 0.5 * (q + q.transpose());
}

Eigen::Matrix3d EkfNode::amcl_measurement_covariance(
  const geometry_msgs::msg::PoseWithCovarianceStamped & msg) const
{
  Eigen::Matrix3d r = Eigen::Matrix3d::Zero();
  const auto & c = msg.pose.covariance;
  r(0, 0) = std::max(
    amcl_position_variance_floor_, finite_or(c[0], amcl_position_variance_floor_));
  r(1, 1) = std::max(
    amcl_position_variance_floor_, finite_or(c[7], amcl_position_variance_floor_));
  r(2, 2) = std::max(
    amcl_yaw_variance_floor_, finite_or(c[35], amcl_yaw_variance_floor_));
  return r;
}

Eigen::Matrix3d EkfNode::robust_amcl_covariance(
  const Eigen::Vector3d & innovation,
  const Eigen::Matrix3d & measurement_covariance) const
{
  /*
   * AMCL is the map-frame correction, not the motion model.  A scan matcher
   * can produce a plausible but locally biased pose in a symmetric corner.
   * Inflate only the affected measurement covariance as the innovation grows;
   * small corrections remain normal EKF updates and large corrections are
   * still handled by should_reset_from_amcl() as an explicit relocalisation.
   */
  Eigen::Matrix3d robust = measurement_covariance;
  const double distance = std::hypot(innovation[0], innovation[1]);
  if (amcl_innovation_gate_distance_m_ > 0.0 &&
    distance > amcl_innovation_gate_distance_m_)
  {
    const double scale = distance / amcl_innovation_gate_distance_m_;
    robust(0, 0) *= scale * scale;
    robust(1, 1) *= scale * scale;
  }

  const double yaw = std::abs(innovation[2]);
  if (amcl_innovation_gate_yaw_rad_ > 0.0 &&
    yaw > amcl_innovation_gate_yaw_rad_)
  {
    const double scale = yaw / amcl_innovation_gate_yaw_rad_;
    robust(2, 2) *= scale * scale;
  }
  return robust;
}

void EkfNode::predict(const Eigen::Vector3d & delta, const Eigen::Matrix3d & q, double dt)
{
  const double theta = state_[2];
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  Eigen::Matrix3d f = Eigen::Matrix3d::Identity();
  f(0, 2) = -delta[0] * s - delta[1] * c;
  f(1, 2) = delta[0] * c - delta[1] * s;
  state_ = math_utils::se2_compose(state_, delta);
  const double noise_dt = std::clamp(dt, 0.0, 0.25);
  covariance_ = f * covariance_ * f.transpose() +
    process_noise_scale_ * noise_dt * q;
  covariance_ = 0.5 * (covariance_ + covariance_.transpose());
}

void EkfNode::correct(const Eigen::Vector3d & measurement, const Eigen::Matrix3d & r)
{
  Eigen::Vector3d innovation;
  innovation[0] = measurement[0] - state_[0];
  innovation[1] = measurement[1] - state_[1];
  innovation[2] = math_utils::angle_diff(measurement[2], state_[2]);

  const Eigen::Matrix3d s = covariance_ + r;
  const Eigen::Matrix3d gain = s.ldlt().solve(covariance_).transpose();
  state_ += gain * innovation;
  state_[2] = math_utils::normalize_angle(state_[2]);

  const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d residual = identity - gain;
  covariance_ = residual * covariance_ * residual.transpose() +
    gain * r * gain.transpose();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose());
  covariance_(0, 0) = std::max(kMinVariance, covariance_(0, 0));
  covariance_(1, 1) = std::max(kMinVariance, covariance_(1, 1));
  covariance_(2, 2) = std::max(kMinVariance, covariance_(2, 2));
}

bool EkfNode::should_reset_from_amcl(const Eigen::Vector3d & measurement) const
{
  if (!amcl_jump_reset_enabled_ || !initialized_) {
    return false;
  }
  const double distance = std::hypot(
    measurement[0] - state_[0], measurement[1] - state_[1]);
  const double yaw = std::abs(math_utils::angle_diff(measurement[2], state_[2]));
  return distance > amcl_jump_reset_distance_m_ || yaw > amcl_jump_reset_yaw_rad_;
}

void EkfNode::odom_callback(nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  const rclcpp::Time stamp(msg->header.stamp, get_clock()->get_clock_type());
  const Eigen::Vector3d odom_pose = math_utils::pose_to_vec(msg->pose.pose);
  if (!odom_pose.allFinite()) {
    return;
  }

  bool publish = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (odom_received_ && stamp <= previous_odom_stamp_) {
      return;
    }
    push_odom_sample(stamp, odom_pose);

    if (!odom_received_) {
      previous_odom_ = odom_pose;
      previous_odom_stamp_ = stamp;
      odom_received_ = true;
      return;
    }

    const Eigen::Vector3d delta = math_utils::se2_relative(previous_odom_, odom_pose);
    const double dt = (stamp - previous_odom_stamp_).seconds();
    previous_odom_ = odom_pose;
    previous_odom_stamp_ = stamp;

    if (!initialized_) {
      return;
    }

    if (std::hypot(delta[0], delta[1]) > 2.0 || std::abs(delta[2]) > 1.5) {
      initialized_ = false;
      covariance_.setIdentity();
      have_last_amcl_stamp_ = false;
      odom_history_.clear();
      odom_history_.push_back({stamp, odom_pose});
      RCLCPP_WARN(get_logger(), "EKF odometry discontinuity; waiting for fresh AMCL lock");
      return;
    }

    predict(delta, odom_process_covariance(*msg), dt);
    publish = true;
  }

  if (publish) {
    publish_and_broadcast(stamp);
  }
}

void EkfNode::amcl_callback(
  geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg)
{
  if (!msg->header.frame_id.empty() && msg->header.frame_id != global_frame_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Ignoring AMCL pose in frame '%s'; expected '%s'",
      msg->header.frame_id.c_str(), global_frame_.c_str());
    return;
  }

  const rclcpp::Time stamp(msg->header.stamp, get_clock()->get_clock_type());
  const double age = (now() - stamp).seconds();
  if (age > amcl_max_latency_s_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Dropping delayed AMCL update: age=%.3f s limit=%.3f s",
      age, amcl_max_latency_s_);
    return;
  }

  rclcpp::Time publish_stamp = stamp;
  bool publish = false;
  bool reset = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (have_last_amcl_stamp_ && stamp <= last_amcl_stamp_) {
      return;
    }
    last_amcl_stamp_ = stamp;
    have_last_amcl_stamp_ = true;

    Eigen::Vector3d measurement = math_utils::pose_to_vec(msg->pose.pose);
    if (!measurement.allFinite()) {
      return;
    }

    Eigen::Vector3d odom_at_measurement;
    if (odom_received_ && interpolate_odom_pose(stamp, odom_at_measurement)) {
      const Eigen::Vector3d odom_delta =
        math_utils::se2_relative(odom_at_measurement, previous_odom_);
      measurement = math_utils::se2_compose(measurement, odom_delta);
      publish_stamp = previous_odom_stamp_;
    }

    const Eigen::Matrix3d r = amcl_measurement_covariance(*msg);
    if (!initialized_) {
      state_ = measurement;
      covariance_ = r;
      initialized_ = true;
      reset = true;
      publish = true;
    } else if (should_reset_from_amcl(measurement)) {
      state_ = measurement;
      covariance_ = r;
      reset = true;
      publish = true;
    } else {
      Eigen::Vector3d innovation;
      innovation[0] = measurement[0] - state_[0];
      innovation[1] = measurement[1] - state_[1];
      innovation[2] = math_utils::angle_diff(measurement[2], state_[2]);
      correct(measurement, robust_amcl_covariance(innovation, r));
      publish = true;
    }
  }

  if (reset) {
    RCLCPP_INFO(get_logger(), "EKF accepted AMCL map lock/relocalization");
  }
  if (publish) {
    publish_and_broadcast(publish_stamp);
  }
}

void EkfNode::publish_and_broadcast(const rclcpp::Time & stamp)
{
  Eigen::Vector3d state;
  Eigen::Matrix3d covariance;
  Eigen::Vector3d odom_base;
  bool have_odom = false;
  rclcpp::Time publish_stamp = stamp;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return;
    }
    state = state_;
    covariance = covariance_;
    if (odom_received_) {
      publish_stamp = previous_odom_stamp_;
      odom_base = previous_odom_;
      have_odom = true;
    } else {
      odom_base.setZero();
    }
  }

  geometry_msgs::msg::PoseWithCovarianceStamped output;
  output.header.stamp = publish_stamp;
  output.header.frame_id = global_frame_;
  output.pose.pose = math_utils::vec_to_pose(state);
  auto & c = output.pose.covariance;
  std::fill(c.begin(), c.end(), 0.0);
  c[0] = covariance(0, 0);
  c[1] = covariance(0, 1);
  c[5] = covariance(0, 2);
  c[6] = covariance(1, 0);
  c[7] = covariance(1, 1);
  c[11] = covariance(1, 2);
  c[30] = covariance(2, 0);
  c[31] = covariance(2, 1);
  c[35] = covariance(2, 2);
  pose_pub_->publish(output);

  if (have_odom) {
    broadcast_tf(publish_stamp, state, odom_base);
  }
}

void EkfNode::broadcast_tf(
  const rclcpp::Time & stamp,
  const Eigen::Vector3d & map_base,
  const Eigen::Vector3d & odom_base)
{
  const Eigen::Vector3d map_odom = math_utils::se2_compose(
    map_base, math_utils::se2_inverse(odom_base));
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = stamp + rclcpp::Duration::from_seconds(transform_tolerance_s_);
  transform.header.frame_id = global_frame_;
  transform.child_frame_id = odom_frame_;
  transform.transform.translation.x = map_odom[0];
  transform.transform.translation.y = map_odom[1];
  transform.transform.rotation = math_utils::yaw_to_quaternion(map_odom[2]);
  tf_broadcaster_->sendTransform(transform);
}

}  // namespace gpu_amcl_cpp

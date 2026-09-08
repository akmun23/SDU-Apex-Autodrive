#include "gpu_amcl_cpp/core/ekf_node.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

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
  odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
    output_odom_topic_, rclcpp::QoS(100).reliable());
  if (publish_tf_) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }

  const auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_, sensor_qos,
    std::bind(&EkfNode::odom_callback, this, std::placeholders::_1));

  if (reset_enabled_) {
    reset_sub_ = create_subscription<std_msgs::msg::Bool>(
      reset_topic_, rclcpp::QoS(10).reliable(),
      std::bind(&EkfNode::reset_callback, this, std::placeholders::_1));
  }

  RCLCPP_INFO(
    get_logger(),
    "Local odometry trust filter ready: %s -> %s and %s in frame %s; reset events=%s",
    odom_topic_.c_str(), output_odom_topic_.c_str(), output_topic_.c_str(), odom_frame_.c_str(),
    reset_enabled_ ? reset_topic_.c_str() : "disabled");
}

void EkfNode::declare_all_parameters()
{
  declare_parameter("odom_topic", odom_topic_);
  declare_parameter("output_topic", output_topic_);
  declare_parameter("output_odom_topic", output_odom_topic_);
  declare_parameter("odom_frame", odom_frame_);
  declare_parameter("process_noise_scale", process_noise_scale_);
  declare_parameter("process_noise_xy_m2_per_m", process_noise_xy_m2_per_m_);
  declare_parameter("process_noise_xy_m2_per_s", process_noise_xy_m2_per_s_);
  declare_parameter("process_noise_yaw2_per_rad", process_noise_yaw2_per_rad_);
  declare_parameter("process_noise_yaw2_per_m", process_noise_yaw2_per_m_);
  declare_parameter("max_odom_delta_m", max_odom_delta_m_);
  declare_parameter("reset_enabled", reset_enabled_);
  declare_parameter("reset_topic", reset_topic_);
  declare_parameter("publish_tf", publish_tf_);
}

void EkfNode::load_parameters()
{
  odom_topic_ = get_parameter("odom_topic").as_string();
  output_topic_ = get_parameter("output_topic").as_string();
  output_odom_topic_ = get_parameter("output_odom_topic").as_string();
  odom_frame_ = get_parameter("odom_frame").as_string();
  process_noise_scale_ = std::max(
    0.0, get_parameter("process_noise_scale").as_double());
  process_noise_xy_m2_per_m_ = std::max(
    0.0, get_parameter("process_noise_xy_m2_per_m").as_double());
  process_noise_xy_m2_per_s_ = std::max(
    0.0, get_parameter("process_noise_xy_m2_per_s").as_double());
  process_noise_yaw2_per_rad_ = std::max(
    0.0, get_parameter("process_noise_yaw2_per_rad").as_double());
  process_noise_yaw2_per_m_ = std::max(
    0.0, get_parameter("process_noise_yaw2_per_m").as_double());
  max_odom_delta_m_ = std::max(
    1.0, get_parameter("max_odom_delta_m").as_double());
  reset_enabled_ = get_parameter("reset_enabled").as_bool();
  reset_topic_ = get_parameter("reset_topic").as_string();
  publish_tf_ = get_parameter("publish_tf").as_bool();
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

void EkfNode::predict(
  const Eigen::Vector3d & delta, const Eigen::Matrix3d & q, double dt)
{
  const Eigen::Vector3d previous_state = state_;
  const double distance = std::hypot(delta[0], delta[1]);
  const double bounded_dt = std::clamp(dt, 0.0, 0.5);
  // The covariance on the raw /odom message describes the current absolute
  // observer uncertainty. It is not a per-sample process covariance; adding
  // it at every 20 Hz callback would inflate the EKF variance by roughly
  // 20x per second. Use the configured motion-noise densities for propagation
  // and retain the raw covariance only as a conservative lower bound below.
  Eigen::Matrix3d motion_q = Eigen::Matrix3d::Zero();
  motion_q(0, 0) = process_noise_xy_m2_per_m_ * distance +
    process_noise_xy_m2_per_s_ * bounded_dt;
  motion_q(1, 1) = motion_q(0, 0);
  motion_q(2, 2) = process_noise_yaw2_per_rad_ * std::abs(delta[2]) +
    process_noise_yaw2_per_m_ * distance;
  covariance_ = localization_math::propagate_pose_covariance(
    covariance_, previous_state, delta, process_noise_scale_ * motion_q);
  state_ = math_utils::se2_compose(state_, delta);
  // The source covariance remains a floor on the propagated uncertainty. It
  // is never repeatedly injected as process noise.
  for (int i = 0; i < 3; ++i) {
    covariance_(i, i) = std::max(covariance_(i, i), q(i, i));
  }
  covariance_(0, 0) = std::max(kMinVariance, covariance_(0, 0));
  covariance_(1, 1) = std::max(kMinVariance, covariance_(1, 1));
  covariance_(2, 2) = std::max(kMinVariance, covariance_(2, 2));
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

    if (!odom_received_) {
      // The local EKF frame starts at the first valid sensor-odometry sample.
      // This is an origin choice, not a reset based on wheel motion.
      previous_odom_ = odom_pose;
      previous_odom_stamp_ = stamp;
      odom_received_ = true;
      initialized_ = true;
      state_.setZero();
      covariance_ = odom_process_covariance(*msg);
      publish = true;
    } else if (reset_pending_) {
      // The diagnostic suite explicitly announces each simulator reset. The
      // official sensor odometry topic retains its world-relative pose across
      // that teleport, so a pose discontinuity cannot reliably identify the
      // new epoch. Reset only on this scheduled event; frozen wheel motion
      // during braking is never treated as a reset.
      previous_odom_ = odom_pose;
      previous_odom_stamp_ = stamp;
      state_.setZero();
      covariance_ = odom_process_covariance(*msg);
      reset_pending_ = false;
      initialized_ = true;
      publish = true;
    } else {
      const Eigen::Vector3d delta =
        math_utils::se2_relative(previous_odom_, odom_pose);
      const double dt = (stamp - previous_odom_stamp_).seconds();

      // Only reject an impossible pose discontinuity. Encoder standstill
      // during full braking is intentionally not a discontinuity condition;
      // the sensor odometry and IMU-derived motion remain the source data.
      if (std::hypot(delta[0], delta[1]) > max_odom_delta_m_ ||
        std::abs(delta[2]) > 3.141592653589793) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Ignoring impossible sensor-odometry jump: %.3f m, %.3f rad",
          std::hypot(delta[0], delta[1]), std::abs(delta[2]));
        previous_odom_ = odom_pose;
        previous_odom_stamp_ = stamp;
        // In the reset-enabled diagnostic suite a simulator teleport can be
        // delivered before the reset Bool callback. Treat that impossible
        // pose jump as the epoch boundary so the EKF cannot publish one
        // stale pre-reset pose. Production keeps reset_enabled=false and
        // therefore retains the normal impossible-jump rejection behavior.
        if (reset_enabled_) {
          state_.setZero();
          covariance_ = odom_process_covariance(*msg);
          reset_pending_ = false;
          initialized_ = true;
          publish = true;
        } else {
          return;
        }
      } else {
        previous_odom_ = odom_pose;
        previous_odom_stamp_ = stamp;

        predict(delta, odom_process_covariance(*msg), dt);
        publish = true;
      }
    }
  }

  if (publish) {
    publish_pose(stamp);
    publish_odom(stamp, *msg);
  }
}

void EkfNode::reset_callback(std_msgs::msg::Bool::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!msg->data) {
    reset_epoch_active_ = false;
    return;
  }
  if (!reset_epoch_active_) {
    reset_pending_ = true;
    reset_epoch_active_ = true;
    RCLCPP_INFO(get_logger(), "Scheduled local EKF epoch reset armed");
  }
}

void EkfNode::publish_pose(const rclcpp::Time & stamp)
{
  Eigen::Vector3d state;
  Eigen::Matrix3d covariance;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return;
    }
    state = state_;
    covariance = covariance_;
  }

  geometry_msgs::msg::PoseWithCovarianceStamped output;
  output.header.stamp = stamp;
  output.header.frame_id = odom_frame_;
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
}

void EkfNode::publish_odom(
  const rclcpp::Time & stamp, const nav_msgs::msg::Odometry & source)
{
  Eigen::Vector3d state;
  Eigen::Matrix3d covariance;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return;
    }
    state = state_;
    covariance = covariance_;
  }

  nav_msgs::msg::Odometry output;
  output.header.stamp = stamp;
  output.header.frame_id = odom_frame_;
  output.child_frame_id = source.child_frame_id.empty() ? "base_link" : source.child_frame_id;
  output.pose.pose = math_utils::vec_to_pose(state);
  output.twist = source.twist;
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
  output.twist.covariance[0] = std::max(output.twist.covariance[0], 1.0e-6);
  output.twist.covariance[7] = std::max(output.twist.covariance[7], 1.0e-6);
  output.twist.covariance[35] = std::max(output.twist.covariance[35], 1.0e-6);
  odom_pub_->publish(output);

  if (publish_tf_ && tf_broadcaster_) {
    geometry_msgs::msg::TransformStamped transform;
    transform.header = output.header;
    transform.child_frame_id = output.child_frame_id;
    transform.transform.translation.x = state[0];
    transform.transform.translation.y = state[1];
    transform.transform.rotation = output.pose.pose.orientation;
    tf_broadcaster_->sendTransform(transform);
  }
}

}  // namespace gpu_amcl_cpp

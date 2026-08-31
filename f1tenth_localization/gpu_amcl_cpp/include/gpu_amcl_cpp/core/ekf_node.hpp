#pragma once

#include "gpu_amcl_cpp/helpers/math_utils.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Core>

#include <deque>
#include <mutex>
#include <string>

namespace gpu_amcl_cpp {

/**
 * @brief Causal planar EKF that fuses allowed sensor odometry with AMCL.
 *
 * sensor_odometry_node owns the continuous odom-frame motion estimate from
 * the official rear encoder and IMU topics. This node predicts from that
 * estimate and applies AMCL's map-frame pose as a delayed correction. The
 * published /ekf_pose is continuous during motion while remaining anchored
 * to the saved map.
 */
class EkfNode final : public rclcpp::Node
{
public:
  explicit EkfNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  struct OdomSample
  {
    rclcpp::Time stamp;
    Eigen::Vector3d pose;
  };

  void odom_callback(nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void amcl_callback(
    geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg);

  void declare_all_parameters();
  void load_parameters();
  void push_odom_sample(const rclcpp::Time & stamp, const Eigen::Vector3d & pose);
  bool interpolate_odom_pose(
    const rclcpp::Time & stamp, Eigen::Vector3d & pose_out) const;
  Eigen::Matrix3d odom_process_covariance(
    const nav_msgs::msg::Odometry & msg) const;
  Eigen::Matrix3d amcl_measurement_covariance(
    const geometry_msgs::msg::PoseWithCovarianceStamped & msg) const;
  Eigen::Matrix3d robust_amcl_covariance(
    const Eigen::Vector3d & innovation,
    const Eigen::Matrix3d & measurement_covariance) const;

  void predict(const Eigen::Vector3d & delta, const Eigen::Matrix3d & q, double dt);
  void correct(const Eigen::Vector3d & measurement, const Eigen::Matrix3d & r);
  bool should_reset_from_amcl(const Eigen::Vector3d & measurement) const;
  void publish_and_broadcast(const rclcpp::Time & stamp);
  void broadcast_tf(
    const rclcpp::Time & stamp,
    const Eigen::Vector3d & map_base,
    const Eigen::Vector3d & odom_base);

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  Eigen::Vector3d state_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d covariance_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d previous_odom_{Eigen::Vector3d::Zero()};
  rclcpp::Time previous_odom_stamp_{0, 0, RCL_ROS_TIME};
  std::deque<OdomSample> odom_history_;

  bool initialized_{false};
  bool odom_received_{false};
  bool have_last_amcl_stamp_{false};
  rclcpp::Time last_amcl_stamp_{0, 0, RCL_ROS_TIME};

  double odom_history_duration_s_{0.75};
  double transform_tolerance_s_{0.0};
  double process_noise_scale_{1.0};
  double amcl_max_latency_s_{0.35};
  double amcl_position_variance_floor_{0.15};
  double amcl_yaw_variance_floor_{0.06};
  double amcl_innovation_gate_distance_m_{0.08};
  double amcl_innovation_gate_yaw_rad_{0.06};
  bool amcl_jump_reset_enabled_{true};
  double amcl_jump_reset_distance_m_{1.0};
  double amcl_jump_reset_yaw_rad_{1.2};

  std::string amcl_topic_{"/amcl_pose"};
  std::string odom_topic_{"/odom"};
  std::string output_topic_{"/ekf_pose"};
  std::string global_frame_{"map"};
  std::string odom_frame_{"odom"};
  std::string base_frame_{"base_link"};

  mutable std::mutex mutex_;
};

}  // namespace gpu_amcl_cpp

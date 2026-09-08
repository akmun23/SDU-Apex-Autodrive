#pragma once

#include "gpu_amcl_cpp/helpers/math_utils.hpp"
#include "gpu_amcl_cpp/helpers/localization_math.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Core>

#include <mutex>
#include <string>

namespace gpu_amcl_cpp {

/**
 * @brief Causal local planar odometry trust filter.
 *
 * sensor_odometry_node owns the continuous local motion estimate from the
 * official rear encoders and IMU. This node preserves that causal estimate,
 * rejects impossible source jumps, and provides the covariance-bearing
 * nav_msgs/Odometry stream consumed by AMCL. It is intentionally a local
 * odometry filter: AMCL is not a measurement input and simulator ground truth
 * is never used at runtime.
 *
 * The recorder retains the raw encoder and IMU streams so a later identified
 * vehicle model can replace this conservative propagation stage. In
 * particular, a frozen encoder during braking is not an EKF reset condition.
 */
class EkfNode final : public rclcpp::Node
{
public:
  explicit EkfNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void odom_callback(nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void reset_callback(std_msgs::msg::Bool::ConstSharedPtr msg);

  void declare_all_parameters();
  void load_parameters();
  Eigen::Matrix3d odom_process_covariance(
    const nav_msgs::msg::Odometry & msg) const;

  void predict(const Eigen::Vector3d & delta, const Eigen::Matrix3d & q, double dt);
  void publish_pose(const rclcpp::Time & stamp);
  void publish_odom(const rclcpp::Time & stamp, const nav_msgs::msg::Odometry & source);

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr reset_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  Eigen::Vector3d state_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d covariance_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d previous_odom_{Eigen::Vector3d::Zero()};
  rclcpp::Time previous_odom_stamp_{0, 0, RCL_ROS_TIME};

  bool initialized_{false};
  bool odom_received_{false};
  bool reset_pending_{false};
  bool reset_epoch_active_{false};

  double process_noise_scale_{1.0};
  double process_noise_xy_m2_per_m_{0.0004};
  double process_noise_xy_m2_per_s_{0.0001};
  double process_noise_yaw2_per_rad_{0.0004};
  double process_noise_yaw2_per_m_{0.0001};
  double max_odom_delta_m_{20.0};
  bool reset_enabled_{false};
  bool publish_tf_{false};

  std::string odom_topic_{"/odom"};
  std::string output_topic_{"/ekf_pose"};
  std::string output_odom_topic_{"/ekf_odom"};
  std::string odom_frame_{"odom"};
  std::string reset_topic_{"/autodrive/reset_command"};

  mutable std::mutex mutex_;
};

}  // namespace gpu_amcl_cpp

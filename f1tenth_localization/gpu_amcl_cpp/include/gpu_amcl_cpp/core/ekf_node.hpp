#pragma once

#include "gpu_amcl_cpp/helpers/math_utils.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/bool.hpp>
#include <Eigen/Core>

#include <mutex>
#include <string>

namespace gpu_amcl_cpp {

/**
 * @brief Causal local planar filter driven by sensor odometry.
 *
 * sensor_odometry_node owns the continuous local motion estimate from the
 * official rear encoders and IMU. This node propagates that estimate in the
 * odom frame and publishes a local covariance-bearing pose. AMCL is a
 * separate map-frame localization source; it is deliberately not subscribed
 * to here and must never be used as an EKF measurement.
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

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr reset_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;

  Eigen::Vector3d state_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d covariance_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d previous_odom_{Eigen::Vector3d::Zero()};
  rclcpp::Time previous_odom_stamp_{0, 0, RCL_ROS_TIME};

  bool initialized_{false};
  bool odom_received_{false};
  bool reset_pending_{false};
  bool reset_epoch_active_{false};

  double process_noise_scale_{1.0};
  double max_odom_delta_m_{20.0};
  bool reset_enabled_{false};

  std::string odom_topic_{"/odom"};
  std::string output_topic_{"/ekf_pose"};
  std::string odom_frame_{"odom"};
  std::string reset_topic_{"/autodrive/reset_command"};

  mutable std::mutex mutex_;
};

}  // namespace gpu_amcl_cpp

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/float32.hpp"
#include "sdu_apex_msgs/msg/vehicle_control_state.hpp"

namespace f1tenth_mpc {
namespace {

double stamp_seconds(const builtin_interfaces::msg::Time &stamp)
{
  return static_cast<double>(stamp.sec) + 1e-9 * static_cast<double>(stamp.nanosec);
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion &q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

}  // namespace

class ControlStateNode final : public rclcpp::Node {
public:
  ControlStateNode()
  : Node("control_state_node")
  {
    odom_topic_ = declare_parameter("odom_topic", std::string("/odom"));
    pose_topic_ = declare_parameter("map_pose_topic", std::string("/current_map_pose"));
    imu_topic_ = declare_parameter("imu_topic", std::string("/autodrive/roboracer_1/imu"));
    steering_topic_ = declare_parameter(
      "steering_feedback_topic", std::string("/autodrive/roboracer_1/steering"));
    state_topic_ = declare_parameter("state_topic", std::string("/mpc/control_state"));
    max_pose_age_s_ = declare_parameter("max_pose_age_s", 0.125);
    max_steering_age_s_ = declare_parameter("max_steering_age_s", 0.125);
    max_imu_age_s_ = declare_parameter("max_imu_age_s", 0.125);
    body_v_variance_ = declare_parameter("body_v_variance_when_unavailable", 1e6);

    state_pub_ = create_publisher<sdu_apex_msgs::msg::VehicleControlState>(state_topic_, 10);
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::QoS(20).reliable(),
      std::bind(&ControlStateNode::odom_callback, this, std::placeholders::_1));
    pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      pose_topic_, rclcpp::QoS(20).reliable(),
      std::bind(&ControlStateNode::pose_callback, this, std::placeholders::_1));
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ControlStateNode::imu_callback, this, std::placeholders::_1));
    steering_sub_ = create_subscription<std_msgs::msg::Float32>(
      steering_topic_, rclcpp::QoS(20).reliable(),
      std::bind(&ControlStateNode::steering_callback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Publishing source-time control state on %s", state_topic_.c_str());
  }

private:
  void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    pose_ = msg;
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    imu_ = msg;
  }

  void steering_callback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    const double value = static_cast<double>(msg->data);
    if (std::isfinite(value)) {
      steering_ = value;
      steering_arrival_ = now();
      steering_valid_ = true;
    }
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const double odom_time = stamp_seconds(msg->header.stamp);
    sdu_apex_msgs::msg::VehicleControlState output;
    output.header = msg->header;
    output.body_u_mps = std::isfinite(msg->twist.twist.linear.x)
      ? msg->twist.twist.linear.x : 0.0;
    // The production odometry intentionally reports no validated lateral
    // velocity. Keep it zero and make that fact machine-readable instead of
    // quietly presenting it as a measured dynamic state.
    output.body_v_mps = 0.0;
    output.dynamics_state_valid = false;
    output.body_v_variance = body_v_variance_;
    output.yaw_rate_radps = std::isfinite(msg->twist.twist.angular.z)
      ? msg->twist.twist.angular.z : 0.0;
    output.body_u_variance = std::max(0.0, msg->twist.covariance[0]);
    output.yaw_rate_variance = std::max(0.0, msg->twist.covariance[35]);

    if (pose_) {
      const double pose_time = stamp_seconds(pose_->header.stamp);
      const double age = odom_time - pose_time;
      output.x_map_m = pose_->pose.pose.position.x;
      output.y_map_m = pose_->pose.pose.position.y;
      output.yaw_map_rad = yaw_from_quaternion(pose_->pose.pose.orientation);
      output.pose_xy_variance = std::max(
        0.0, std::max(pose_->pose.covariance[0], pose_->pose.covariance[7]));
      output.pose_yaw_variance = std::max(0.0, pose_->pose.covariance[35]);
      output.localization_valid = std::isfinite(age) && age >= -0.020 && age <= max_pose_age_s_;
    } else {
      output.localization_valid = false;
      output.pose_xy_variance = body_v_variance_;
      output.pose_yaw_variance = body_v_variance_;
    }

    if (imu_) {
      const double imu_time = stamp_seconds(imu_->header.stamp);
      const double age = odom_time - imu_time;
      const bool usable = std::isfinite(age) && age >= -0.020 && age <= max_imu_age_s_;
      output.ax_mps2 = usable && std::isfinite(imu_->linear_acceleration.x)
        ? imu_->linear_acceleration.x : 0.0;
      output.ay_mps2 = usable && std::isfinite(imu_->linear_acceleration.y)
        ? imu_->linear_acceleration.y : 0.0;
    }
    const double steering_age = (now() - steering_arrival_).seconds();
    output.steering_angle_rad = steering_;
    output.steering_valid = steering_valid_ && steering_age >= 0.0 &&
      steering_age <= max_steering_age_s_ && std::isfinite(steering_);
    output.reset_epoch = reset_epoch_;
    state_pub_->publish(output);
  }

  std::string odom_topic_;
  std::string pose_topic_;
  std::string imu_topic_;
  std::string steering_topic_;
  std::string state_topic_;
  double max_pose_age_s_{0.125};
  double max_steering_age_s_{0.125};
  double max_imu_age_s_{0.125};
  double body_v_variance_{1e6};
  bool steering_valid_{false};
  double steering_{0.0};
  uint64_t reset_epoch_{0};
  rclcpp::Time steering_arrival_{0, 0, RCL_SYSTEM_TIME};
  geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose_;
  sensor_msgs::msg::Imu::SharedPtr imu_;
  rclcpp::Publisher<sdu_apex_msgs::msg::VehicleControlState>::SharedPtr state_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr steering_sub_;
};

}  // namespace f1tenth_mpc

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<f1tenth_mpc::ControlStateNode>());
  rclcpp::shutdown();
  return 0;
}

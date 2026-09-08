#ifndef F1TENTH_CONTROL_FTG_NODE_HPP_
#define F1TENTH_CONTROL_FTG_NODE_HPP_

#include "algorithms/follow_the_gap.hpp"

#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <memory>
#include <string>

namespace f1tenth_control {

class FTGNode : public rclcpp::Node
{
public:
  explicit FTGNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void declareParameters();
  void loadParameters();
  void scanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg);
  void explorationOdomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void publishDriveCommand(const DriveCommand & cmd);
  void publishDebugScan(
    const sensor_msgs::msg::LaserScan & source,
    const FTGOutput & output);
  void publishDiagnostics(
    const sensor_msgs::msg::LaserScan & source,
    const FTGOutput & output);

  std::unique_ptr<FollowTheGap> ftg_;
  FTGConfig config_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

  std::string scan_topic_{"/autodrive/roboracer_1/lidar"};
  std::string command_topic_{"/cmd/speed"};
  std::string command_frame_{"base_link"};
  std::string processed_scan_topic_{"/ftg/processed_scan"};
  std::string diagnostics_topic_{"/ftg/diagnostics"};
  std::string exploration_odom_topic_{};
  bool publish_debug_topics_{true};

  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr processed_scan_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr diagnostics_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr exploration_odom_sub_;
};

}  // namespace f1tenth_control
#endif

#ifndef F1TENTH_CONTROL_FTG_NODE_HPP_
#define F1TENTH_CONTROL_FTG_NODE_HPP_

#include "algorithms/follow_the_gap.hpp"

#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

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
  void publishDriveCommand(const DriveCommand & cmd);

  std::unique_ptr<FollowTheGap> ftg_;
  FTGConfig config_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

  std::string scan_topic_{"/autodrive/roboracer_1/lidar"};
  std::string command_topic_{"/cmd/controller"};
  std::string command_frame_{"base_link"};
};

}  // namespace f1tenth_control
#endif

#ifndef F1TENTH_CONTROL_FTG_NODE_HPP_
#define F1TENTH_CONTROL_FTG_NODE_HPP_

#include <memory>
#include <string>
#include <type_traits>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sdu_apex_msgs/msg/drive_command.hpp>
#include "algorithms/follow_the_gap.hpp"

namespace f1tenth_control {
class FTGNode : public rclcpp::Node {
public:
  explicit FTGNode(const rclcpp::NodeOptions & options);
private:
  template<class T> T fixedParameter(const std::string & name, const T & value) {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.read_only = true;
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      return static_cast<T>(declare_parameter<int64_t>(
        name, static_cast<int64_t>(value), descriptor));
    } else {
      return declare_parameter<T>(name, value, descriptor);
    }
  }
  void scanCallback(sensor_msgs::msg::LaserScan::ConstSharedPtr scan);
  FTGConfig config_;
  std::unique_ptr<FollowTheGap> ftg_;
  double throttle_cap_;
  std::string command_frame_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<sdu_apex_msgs::msg::DriveCommand>::SharedPtr drive_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr diagnostics_pub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr processed_scan_pub_;
};
}
#endif

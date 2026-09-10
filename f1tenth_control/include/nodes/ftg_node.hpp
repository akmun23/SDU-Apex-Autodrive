#ifndef F1TENTH_CONTROL_FTG_NODE_HPP_
#define F1TENTH_CONTROL_FTG_NODE_HPP_

/**
 * @file ftg_node.hpp
 * @brief ROS 2 node wrapper for the Follow-The-Gap reactive controller.
 *
 * The node has one input (LaserScan) and one control output.  All steering
 * state and rate limiting belong to FollowTheGap; mapping completion and
 * run-level telemetry belong to their respective nodes.
 */
#include "algorithms/follow_the_gap.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>

#include <memory>

namespace f1tenth_control {

class FTGNode : public rclcpp::Node {
public:
    explicit FTGNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
    std::unique_ptr<FollowTheGap> ftg_;
    FTGConfig config_;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;

    void declareParameters();
    void loadParameters();
    void scanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg);
    void publishDriveCommand(const DriveCommand& cmd);
    void publishDiagnostics(const FTGOutput& output);
};

}  // namespace f1tenth_control

#endif  // F1TENTH_CONTROL_FTG_NODE_HPP_

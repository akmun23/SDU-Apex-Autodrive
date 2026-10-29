#include "nodes/ftg_node.hpp"
#include <stdexcept>
#include <rclcpp_components/register_node_macro.hpp>

namespace f1tenth_control {
FTGNode::FTGNode(const rclcpp::NodeOptions & options) : Node("ftg_node", options) {
#define FTG_PARAMETER(name) config_.name = fixedParameter(#name, config_.name)
  FTG_PARAMETER(wheelbase);
  FTG_PARAMETER(car_width);
  FTG_PARAMETER(max_speed);
  FTG_PARAMETER(min_speed);
  FTG_PARAMETER(speed_full_range);
  FTG_PARAMETER(steer_slowdown_gain);
  FTG_PARAMETER(max_steering);
  FTG_PARAMETER(steering_gain);
  FTG_PARAMETER(max_steering_rate);
  FTG_PARAMETER(target_ema_alpha);
  FTG_PARAMETER(heading_weight);
  FTG_PARAMETER(score_power);
  FTG_PARAMETER(clearance_cone_scale);
  FTG_PARAMETER(min_score_range);
  FTG_PARAMETER(gap_switch_margin);
  FTG_PARAMETER(gap_switch_scans);
  FTG_PARAMETER(car_length);
  FTG_PARAMETER(rear_overhang);
  FTG_PARAMETER(lidar_to_rear_axle);
  FTG_PARAMETER(footprint_margin);
  FTG_PARAMETER(rollout_distance);
  FTG_PARAMETER(rollout_step);
  FTG_PARAMETER(trajectory_min_free_distance);
  FTG_PARAMETER(trajectory_candidate_count);
  FTG_PARAMETER(emergency_brake_distance);
  FTG_PARAMETER(disparity_threshold);
  FTG_PARAMETER(wall_margin);
  FTG_PARAMETER(gap_threshold);
  FTG_PARAMETER(min_gap_width);
#undef FTG_PARAMETER
#define LIDAR_PARAMETER(name) config_.lidar_config.name = fixedParameter("lidar." #name, config_.lidar_config.name)
  LIDAR_PARAMETER(range_min);
  LIDAR_PARAMETER(range_max);
  LIDAR_PARAMETER(angle_min);
  LIDAR_PARAMETER(angle_max);
  LIDAR_PARAMETER(apply_median_filter);
  LIDAR_PARAMETER(median_window_size);
#undef LIDAR_PARAMETER
  throttle_cap_ = fixedParameter("mapping_throttle_normalized", 0.02);
  command_frame_ = fixedParameter("command_frame", std::string("base_link"));
  if (!std::isfinite(throttle_cap_) || throttle_cap_ <= 0.0 || throttle_cap_ > 1.0 ||
      !std::isfinite(config_.max_speed) || config_.max_speed <= 0.0 ||
      config_.min_speed < 0.0 || config_.min_speed > config_.max_speed ||
      config_.wheelbase <= 0.0 || config_.rollout_step <= 0.0 ||
      config_.trajectory_candidate_count < 1) {
    throw std::invalid_argument("Invalid FTG throttle, speed or rollout configuration");
  }
  ftg_ = std::make_unique<FollowTheGap>(config_);
  drive_pub_ = create_publisher<sdu_apex_msgs::msg::DriveCommand>(
    fixedParameter("command_topic", std::string("/mpc/drive_command")), 1);
  diagnostics_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/ftg/diagnostics", 1);
  if (fixedParameter("publish_debug_scan", false)) {
    processed_scan_pub_ = create_publisher<sensor_msgs::msg::LaserScan>(
      "/ftg/processed_scan", rclcpp::SensorDataQoS().keep_last(1));
  }
  scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
    fixedParameter("scan_topic", std::string("/autodrive/roboracer_1/lidar")),
    rclcpp::SensorDataQoS().keep_last(1),
    std::bind(&FTGNode::scanCallback, this, std::placeholders::_1));
  RCLCPP_INFO(get_logger(), "FTG mapping: swept footprint, source-timed steering, throttle cap %.4f",
    throttle_cap_);
}

void FTGNode::scanCallback(sensor_msgs::msg::LaserScan::ConstSharedPtr scan) {
  const double stamp = rclcpp::Time(scan->header.stamp).seconds();
  const auto output = ftg_->compute(scan->ranges, scan->angle_min,
    scan->angle_max, scan->angle_increment, stamp);
  sdu_apex_msgs::msg::DriveCommand command;
  command.header.stamp = now();
  command.header.frame_id = command_frame_;
  command.steering_angle_rad = output.command.steering_angle;
  // This is a conservative mapping schedule, not a vehicle speed controller.
  // Every requested slowdown reduces torque; an invalid path requests q=0.
  command.throttle_normalized = throttle_cap_ *
    std::clamp(output.command.speed / config_.max_speed, 0.0, 1.0);
  drive_pub_->publish(command);
  const auto & gap = output.selected_drivable_gap;
  std_msgs::msg::Float64MultiArray diagnostic;
  diagnostic.data = {
    stamp, output.raw_target_angle, output.raw_steering,
    command.steering_angle_rad, output.command.speed, command.throttle_normalized,
    output.closest_point_dist, output.forward_clearance,
    output.trajectory_free_distance, output.trajectory_min_clearance,
    output.no_path ? 1.0 : 0.0, output.emergency_stop ? 1.0 : 0.0,
    output.has_selected_drivable_gap ? 1.0 : 0.0,
    gap.start_angle, gap.end_angle, gap.score,
    output.trajectory_collision_free ? 1.0 : 0.0};
  diagnostics_pub_->publish(diagnostic);
  if (processed_scan_pub_) {
    auto debug = *scan;
    const auto & processed = output.processed_scan;
    for (size_t i = 0; i < processed.filtered_ranges.size(); ++i)
      debug.ranges[i] = processed.valid[i] ?
        static_cast<float>(processed.filtered_ranges[i]) :
        std::numeric_limits<float>::quiet_NaN();
    processed_scan_pub_->publish(debug);
  }
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
    "FTG steer=%.3f q=%.4f free=%.2fm clearance=%.3fm gap=%.2f:%.2f stop=%d",
    command.steering_angle_rad, command.throttle_normalized,
    output.trajectory_free_distance, output.trajectory_min_clearance,
    gap.start_angle, gap.end_angle, output.no_path || output.emergency_stop);
}
}
RCLCPP_COMPONENTS_REGISTER_NODE(f1tenth_control::FTGNode)

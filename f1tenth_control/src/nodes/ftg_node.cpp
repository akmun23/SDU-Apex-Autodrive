#include "nodes/ftg_node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace f1tenth_control {

FTGNode::FTGNode(const rclcpp::NodeOptions & options)
: Node("ftg_node", options)
{
  declareParameters();
  loadParameters();
  ftg_ = std::make_unique<FollowTheGap>(config_);

  scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
    scan_topic_, rclcpp::SensorDataQoS().keep_last(1),
    std::bind(&FTGNode::scanCallback, this, std::placeholders::_1));
  if (!exploration_odom_topic_.empty() && config_.exploration_history_enabled) {
    exploration_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      exploration_odom_topic_, rclcpp::QoS(20),
      std::bind(&FTGNode::explorationOdomCallback, this, std::placeholders::_1));
  }
  drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
    command_topic_, rclcpp::QoS(10));
  if (publish_debug_topics_) {
    processed_scan_pub_ = create_publisher<sensor_msgs::msg::LaserScan>(
      processed_scan_topic_, rclcpp::SensorDataQoS().keep_last(2));
    diagnostics_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      diagnostics_topic_, rclcpp::QoS(10));
  }

  RCLCPP_INFO(get_logger(), "FTG ready: %s -> %s",
              scan_topic_.c_str(), command_topic_.c_str());
}

void FTGNode::declareParameters()
{
  declare_parameter("scan_topic", scan_topic_);
  declare_parameter("command_topic", command_topic_);
  declare_parameter("command_frame", command_frame_);
  declare_parameter("processed_scan_topic", processed_scan_topic_);
  declare_parameter("diagnostics_topic", diagnostics_topic_);
  declare_parameter("exploration_odom_topic", exploration_odom_topic_);
  declare_parameter("publish_debug_topics", publish_debug_topics_);

  declare_parameter("wheelbase", 0.324);
  declare_parameter("car_length", 0.500);
  declare_parameter("car_width", 0.270);
  declare_parameter("rear_overhang", 0.080);
  declare_parameter("lidar_offset_x", 0.2733);
  declare_parameter("virtual_width_inflation", 0.0);
  declare_parameter("virtual_front_inflation", 0.0);
  declare_parameter("virtual_rear_inflation", 0.0);

  declare_parameter("max_speed", 1.8);
  declare_parameter("min_speed", 0.6);
  declare_parameter("speed_full_range", 4.0);
  declare_parameter("steer_slowdown_gain", 1.0);

  declare_parameter("max_steering", 0.5236);
  declare_parameter("steering_gain", 1.1);
  declare_parameter("max_steering_rate", 3.2);
  declare_parameter("target_ema_alpha", 0.55);

  declare_parameter("heading_weight", 0.7);
  declare_parameter("score_power", 2.0);
  declare_parameter("clearance_cone_scale", 1.0);
  declare_parameter("min_score_range", 0.35);
  declare_parameter("select_single_gap", false);
  declare_parameter("avoid_close_side", false);
  declare_parameter("close_side_turn_angle", 0.22);
  declare_parameter("gap_switch_confirm_cycles", 6);

  declare_parameter("emergency_brake_distance", 0.18);
  declare_parameter("emergency_rolling_speed", 0.0);
  declare_parameter("footprint_clearance", 0.08);
  declare_parameter("side_recovery_distance", 0.35);
  declare_parameter("side_recovery_max_angle", 1.20);
  declare_parameter("side_recovery_front_angle", 0.65);
  declare_parameter("side_recovery_min_steering", 0.12);
  declare_parameter("side_recovery_full_steering_distance", 0.30);
  declare_parameter("recovery_switch_confirm_cycles", 4);
  declare_parameter("recovery_clear_confirm_cycles", 5);
  declare_parameter("lock_recovery_side_until_clear", false);
  declare_parameter("recovery_override_selected_gap", true);
  declare_parameter("recovery_gap_blend", 1.0);
  declare_parameter("ambiguous_front_recovery_sign", 0.0);
  declare_parameter("use_rear_context", false);
  declare_parameter("rear_context_min_angle", 1.57079632679);
  declare_parameter("rear_context_max_angle", 2.35619449019);
  declare_parameter("rear_context_min_range", 0.20);
  declare_parameter("rear_context_min_advantage", 0.20);
  declare_parameter("emergency_prefer_last_steering", false);
  declare_parameter("exploration_history_enabled", false);
  declare_parameter("exploration_probe_distance", 2.5);
  declare_parameter("exploration_recent_exclusion_distance", 1.5);
  declare_parameter("exploration_history_radius", 0.75);
  declare_parameter("exploration_branch_penalty", 5.0);
  declare_parameter("disparity_threshold", 0.5);
  declare_parameter("wall_margin", 0.05);
  declare_parameter("side_safety_margin", 0.10);
  declare_parameter("max_disparity_extension_angle", 0.7854);
  declare_parameter("gap_threshold", 0.5);
  declare_parameter("min_gap_width", 0.15);

  declare_parameter("lidar.range_min", 0.06);
  declare_parameter("lidar.range_max", 10.0);
  declare_parameter("lidar.angle_min", -1.57079632679);
  declare_parameter("lidar.angle_max", 1.57079632679);
  declare_parameter("lidar.apply_median_filter", true);
  declare_parameter("lidar.median_window_size", 3);

}

void FTGNode::loadParameters()
{
  scan_topic_ = get_parameter("scan_topic").as_string();
  command_topic_ = get_parameter("command_topic").as_string();
  command_frame_ = get_parameter("command_frame").as_string();
  processed_scan_topic_ = get_parameter("processed_scan_topic").as_string();
  diagnostics_topic_ = get_parameter("diagnostics_topic").as_string();
  exploration_odom_topic_ = get_parameter("exploration_odom_topic").as_string();
  publish_debug_topics_ = get_parameter("publish_debug_topics").as_bool();

  config_.wheelbase = get_parameter("wheelbase").as_double();
  config_.car_length = get_parameter("car_length").as_double();
  config_.car_width = get_parameter("car_width").as_double();
  config_.rear_overhang = get_parameter("rear_overhang").as_double();
  config_.lidar_offset_x = get_parameter("lidar_offset_x").as_double();
  config_.virtual_width_inflation = std::max(
    0.0, get_parameter("virtual_width_inflation").as_double());
  config_.virtual_front_inflation = std::max(
    0.0, get_parameter("virtual_front_inflation").as_double());
  config_.virtual_rear_inflation = std::max(
    0.0, get_parameter("virtual_rear_inflation").as_double());

  config_.max_speed = get_parameter("max_speed").as_double();
  config_.min_speed = get_parameter("min_speed").as_double();
  config_.speed_full_range = get_parameter("speed_full_range").as_double();
  config_.steer_slowdown_gain = get_parameter("steer_slowdown_gain").as_double();

  config_.max_steering = get_parameter("max_steering").as_double();
  config_.steering_gain = get_parameter("steering_gain").as_double();
  config_.max_steering_rate = get_parameter("max_steering_rate").as_double();
  config_.target_ema_alpha = get_parameter("target_ema_alpha").as_double();

  config_.heading_weight = get_parameter("heading_weight").as_double();
  config_.score_power = get_parameter("score_power").as_double();
  config_.clearance_cone_scale = get_parameter("clearance_cone_scale").as_double();
  config_.min_score_range = get_parameter("min_score_range").as_double();
  config_.select_single_gap = get_parameter("select_single_gap").as_bool();
  config_.avoid_close_side = get_parameter("avoid_close_side").as_bool();
  config_.close_side_turn_angle = std::max(
    0.0, get_parameter("close_side_turn_angle").as_double());
  config_.gap_switch_confirm_cycles = std::max(
    1, static_cast<int>(get_parameter("gap_switch_confirm_cycles").as_int()));

  config_.emergency_brake_distance = get_parameter("emergency_brake_distance").as_double();
  config_.emergency_rolling_speed = std::max(
    0.0, get_parameter("emergency_rolling_speed").as_double());
  config_.footprint_clearance = get_parameter("footprint_clearance").as_double();
  config_.side_recovery_distance = std::max(
    config_.emergency_brake_distance,
    get_parameter("side_recovery_distance").as_double());
  config_.side_recovery_max_angle = std::clamp(
    get_parameter("side_recovery_max_angle").as_double(), 0.18, 1.70);
  config_.side_recovery_front_angle = std::clamp(
    get_parameter("side_recovery_front_angle").as_double(),
    0.15, config_.side_recovery_max_angle);
  config_.side_recovery_min_steering = std::clamp(
    get_parameter("side_recovery_min_steering").as_double(),
    0.0, config_.max_steering);
  config_.side_recovery_full_steering_distance = std::clamp(
    get_parameter("side_recovery_full_steering_distance").as_double(),
    config_.emergency_brake_distance + 0.01,
    config_.side_recovery_distance);
  config_.recovery_switch_confirm_cycles = std::max(
    1, static_cast<int>(get_parameter("recovery_switch_confirm_cycles").as_int()));
  config_.recovery_clear_confirm_cycles = std::max(
    1, static_cast<int>(get_parameter("recovery_clear_confirm_cycles").as_int()));
  config_.lock_recovery_side_until_clear =
    get_parameter("lock_recovery_side_until_clear").as_bool();
  config_.recovery_override_selected_gap =
    get_parameter("recovery_override_selected_gap").as_bool();
  config_.recovery_gap_blend = std::clamp(
    get_parameter("recovery_gap_blend").as_double(), 0.0, 1.0);
  config_.ambiguous_front_recovery_sign = std::clamp(
    get_parameter("ambiguous_front_recovery_sign").as_double(), -1.0, 1.0);
  config_.use_rear_context = get_parameter("use_rear_context").as_bool();
  config_.rear_context_min_angle = std::clamp(
    get_parameter("rear_context_min_angle").as_double(), 1.0, 2.30);
  config_.rear_context_max_angle = std::clamp(
    get_parameter("rear_context_max_angle").as_double(),
    config_.rear_context_min_angle + 0.05, 2.35619449019);
  config_.rear_context_min_range = std::max(
    config_.lidar_config.range_min,
    get_parameter("rear_context_min_range").as_double());
  config_.rear_context_min_advantage = std::max(
    0.0, get_parameter("rear_context_min_advantage").as_double());
  config_.emergency_prefer_last_steering =
    get_parameter("emergency_prefer_last_steering").as_bool();
  config_.exploration_history_enabled =
    get_parameter("exploration_history_enabled").as_bool();
  config_.exploration_probe_distance = std::max(
    1.0, get_parameter("exploration_probe_distance").as_double());
  config_.exploration_recent_exclusion_distance = std::max(
    0.0, get_parameter("exploration_recent_exclusion_distance").as_double());
  config_.exploration_history_radius = std::max(
    0.05, get_parameter("exploration_history_radius").as_double());
  config_.exploration_branch_penalty = std::max(
    0.0, get_parameter("exploration_branch_penalty").as_double());
  config_.disparity_threshold = get_parameter("disparity_threshold").as_double();
  config_.wall_margin = get_parameter("wall_margin").as_double();
  config_.side_safety_margin = get_parameter("side_safety_margin").as_double();
  config_.max_disparity_extension_angle =
    get_parameter("max_disparity_extension_angle").as_double();
  config_.gap_threshold = get_parameter("gap_threshold").as_double();
  config_.min_gap_width = get_parameter("min_gap_width").as_double();

  config_.lidar_config.range_min = get_parameter("lidar.range_min").as_double();
  config_.lidar_config.range_max = get_parameter("lidar.range_max").as_double();
  config_.lidar_config.angle_min = get_parameter("lidar.angle_min").as_double();
  config_.lidar_config.angle_max = get_parameter("lidar.angle_max").as_double();
  config_.lidar_config.apply_median_filter =
    get_parameter("lidar.apply_median_filter").as_bool();
  config_.lidar_config.median_window_size =
    get_parameter("lidar.median_window_size").as_int();

}

void FTGNode::explorationOdomCallback(
  const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  const auto & position = msg->pose.pose.position;
  const auto & orientation = msg->pose.pose.orientation;
  const double yaw = std::atan2(
    2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
    1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z));
  ftg_->updateExplorationPose(position.x, position.y, yaw);
}

void FTGNode::scanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg)
{
  const auto output = ftg_->compute(
    msg->ranges, msg->angle_min, msg->angle_max, msg->angle_increment);
  if (output.emergency_stop) {
    const double closest_angle = msg->angle_min +
      static_cast<double>(output.closest_point_idx) * msg->angle_increment;
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "FTG emergency stop: closest %.3f m at %.3f rad (index %zu)",
      output.closest_point_dist, closest_angle, output.closest_point_idx);
  } else if (output.side_recovery) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "FTG side recovery: closest %.3f m at %.3f rad, recovery sign %.0f, sides L/R %.3f/%.3f rear L/R %.3f/%.3f, raw %.3f, commanded steer %.3f, speed %.3f",
      output.closest_point_dist,
      msg->angle_min + static_cast<double>(output.closest_point_idx) * msg->angle_increment,
      output.recovery_steering_sign,
      output.recovery_left_clearance, output.recovery_right_clearance,
      output.recovery_rear_left_clearance, output.recovery_rear_right_clearance,
      output.raw_steering, output.command.steering_angle, output.command.speed);
  } else if (output.footprint_clearance_limited) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "FTG footprint recovery: closest %.3f m, command speed %.3f steer %.3f",
      output.closest_point_dist, output.command.speed,
      output.command.steering_angle);
  } else {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "FTG target %.3f rad, raw steer %.3f, command speed %.3f steer %.3f",
      output.target_angle, output.raw_steering,
      output.command.speed, output.command.steering_angle);
  }
  publishDebugScan(*msg, output);
  publishDiagnostics(*msg, output);
  publishDriveCommand(output.command);
}

void FTGNode::publishDebugScan(
  const sensor_msgs::msg::LaserScan & source,
  const FTGOutput & output)
{
  if (!processed_scan_pub_) {
    return;
  }
  sensor_msgs::msg::LaserScan scan;
  scan.header = source.header;
  scan.angle_min = output.processed_scan.angle_min;
  scan.angle_max = output.processed_scan.angle_max;
  scan.angle_increment = output.processed_scan.angle_increment;
  scan.time_increment = source.time_increment;
  scan.scan_time = source.scan_time;
  scan.range_min = output.processed_scan.range_min;
  scan.range_max = output.processed_scan.range_max;
  scan.ranges.resize(output.processed_scan.filtered_ranges.size());
  scan.intensities.resize(scan.ranges.size());
  for (std::size_t index = 0; index < scan.ranges.size(); ++index) {
    const bool valid = index < output.processed_scan.valid.size() &&
      output.processed_scan.valid[index];
    scan.ranges[index] = valid ?
      static_cast<float>(output.processed_scan.filtered_ranges[index]) :
      std::numeric_limits<float>::quiet_NaN();
    const bool disparity = index < output.processed_scan.disparity_blocked.size() &&
      output.processed_scan.disparity_blocked[index];
    const bool bubble = index < output.processed_scan.bubble_blocked.size() &&
      output.processed_scan.bubble_blocked[index];
    scan.intensities[index] = static_cast<float>((disparity ? 1 : 0) + (bubble ? 2 : 0));
  }
  processed_scan_pub_->publish(scan);
}

void FTGNode::publishDiagnostics(
  const sensor_msgs::msg::LaserScan & source,
  const FTGOutput & output)
{
  if (!diagnostics_pub_) {
    return;
  }
  const double closest_angle = source.angle_min +
    static_cast<double>(output.closest_point_idx) * source.angle_increment;
  const auto & gap = output.selected_gap;
  std_msgs::msg::Float64MultiArray diagnostic;
  // Fixed schema, kept deliberately flat for rosbag-to-CSV conversion:
  // [0] time, [1] closest range, [2] closest angle, [3] target angle,
  // [4] raw steer, [5] command steer, [6] command speed, [7] recovery sign,
  // [8:12] front/rear side clearances L/R, [12:17] selected gap geometry,
  // [17] gap count, [18:21] emergency/footprint/side flags,
  // [21] valid beams, [22] disparity-blocked beams, [23] exploration overlap,
  // [24] exploration-history enabled, [25:26] left/right branch overlap.
  diagnostic.data = {
    now().seconds(), output.closest_point_dist, closest_angle,
    output.target_angle, output.raw_steering, output.command.steering_angle,
    output.command.speed, output.recovery_steering_sign,
    output.recovery_left_clearance, output.recovery_right_clearance,
    output.recovery_rear_left_clearance, output.recovery_rear_right_clearance,
    gap.start_angle, gap.end_angle, gap.min_range, gap.max_range,
    gap.angular_width, static_cast<double>(output.all_gaps.size()),
    output.emergency_stop ? 1.0 : 0.0,
    output.footprint_clearance_limited ? 1.0 : 0.0,
    output.side_recovery ? 1.0 : 0.0,
    static_cast<double>(std::count(
      output.processed_scan.valid.begin(), output.processed_scan.valid.end(), true)),
    static_cast<double>(std::count(
      output.processed_scan.disparity_blocked.begin(),
      output.processed_scan.disparity_blocked.end(), true)),
    output.exploration_overlap,
    config_.exploration_history_enabled ? 1.0 : 0.0,
    output.exploration_left_overlap,
    output.exploration_right_overlap,
  };
  diagnostics_pub_->publish(diagnostic);
}

void FTGNode::publishDriveCommand(const DriveCommand & cmd)
{
  ackermann_msgs::msg::AckermannDriveStamped msg;
  msg.header.stamp = now();
  msg.header.frame_id = command_frame_;
  msg.drive.speed = cmd.speed;
  msg.drive.steering_angle = cmd.steering_angle;
  drive_pub_->publish(msg);
}

}  // namespace f1tenth_control

RCLCPP_COMPONENTS_REGISTER_NODE(f1tenth_control::FTGNode)

#include "nodes/ftg_node.hpp"

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
  drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
    command_topic_, rclcpp::QoS(10));

  RCLCPP_INFO(get_logger(), "FTG ready: %s -> %s",
              scan_topic_.c_str(), command_topic_.c_str());
}

void FTGNode::declareParameters()
{
  declare_parameter("scan_topic", scan_topic_);
  declare_parameter("command_topic", command_topic_);
  declare_parameter("command_frame", command_frame_);

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
  declare_parameter("footprint_clearance", 0.08);
  declare_parameter("side_recovery_distance", 0.35);
  declare_parameter("side_recovery_max_angle", 1.20);
  declare_parameter("side_recovery_min_steering", 0.12);
  declare_parameter("side_recovery_full_steering_distance", 0.30);
  declare_parameter("recovery_switch_confirm_cycles", 4);
  declare_parameter("recovery_clear_confirm_cycles", 5);
  declare_parameter("lock_recovery_side_until_clear", false);
  declare_parameter("ambiguous_front_recovery_sign", 0.0);
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
  config_.footprint_clearance = get_parameter("footprint_clearance").as_double();
  config_.side_recovery_distance = std::max(
    config_.emergency_brake_distance,
    get_parameter("side_recovery_distance").as_double());
  config_.side_recovery_max_angle = std::clamp(
    get_parameter("side_recovery_max_angle").as_double(), 0.18, 1.50);
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
  config_.ambiguous_front_recovery_sign = std::clamp(
    get_parameter("ambiguous_front_recovery_sign").as_double(), -1.0, 1.0);
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
      "FTG side recovery: closest %.3f m at %.3f rad, recovery sign %.0f, sides L/R %.3f/%.3f, raw %.3f, commanded steer %.3f, speed %.3f",
      output.closest_point_dist,
      msg->angle_min + static_cast<double>(output.closest_point_idx) * msg->angle_increment,
      output.recovery_steering_sign,
      output.recovery_left_clearance, output.recovery_right_clearance,
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
  publishDriveCommand(output.command);
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

#include "nodes/ftg_node.hpp"

#include <functional>


namespace f1tenth_control {

FTGNode::FTGNode(const rclcpp::NodeOptions& options)
    : Node("ftg_node", options)
{
    // Declare and load parameters
    declareParameters();
    loadParameters();
    
    // Create algorithm instance
    ftg_ = std::make_unique<FollowTheGap>(config_);
    
    // QoS profiles
    auto sensor_qos = rclcpp::SensorDataQoS().keep_last(1);
    auto reliable_qos = rclcpp::QoS(10);
    
    // Subscribers
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
        "scan", sensor_qos,
        std::bind(&FTGNode::scanCallback, this, std::placeholders::_1)
    );
    
    // Publishers
    drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
        "drive", reliable_qos
    );
    diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "ftg/diagnostics", reliable_qos
    );
    
    RCLCPP_INFO(get_logger(), "FTG Node initialized");
    RCLCPP_INFO(get_logger(), "  Subscribing to: scan");
    RCLCPP_INFO(get_logger(), "  Publishing to: drive");
}

void FTGNode::declareParameters() {
    // Vehicle parameters
    declare_parameter("wheelbase", 0.324);
    declare_parameter("car_width", 0.273);

    // Speed control
    declare_parameter("max_speed", 0.40);
    declare_parameter("min_speed", 0.12);
    declare_parameter("speed_full_range", 2.0);
    declare_parameter("steer_slowdown_gain", 1.0);

    // Steering control
    declare_parameter("max_steering", 0.5236);
    declare_parameter("steering_gain", 1.0);
    declare_parameter("max_steering_rate", 3.2);
    declare_parameter("target_ema_alpha", 1.0);

    // Weighted free-space scoring
    declare_parameter("heading_weight", 0.20);
    declare_parameter("score_power", 1.5);
    declare_parameter("clearance_cone_scale", 1.05);
    declare_parameter("min_score_range", 0.25);
    declare_parameter("gap_switch_margin", 0.15);
    declare_parameter("gap_switch_scans", 2);

    // Ackermann swept-footprint rollout
    declare_parameter("car_length", 0.510);
    declare_parameter("rear_overhang", 0.080);
    declare_parameter("lidar_to_rear_axle", 0.2733);
    declare_parameter("footprint_margin", 0.03);
    declare_parameter("rollout_distance", 1.50);
    declare_parameter("rollout_step", 0.05);
    declare_parameter("trajectory_min_free_distance", 0.25);
    declare_parameter("trajectory_candidate_count", 31);

    // Safety
    declare_parameter("emergency_brake_distance", 0.10);

    // LiDAR processing
    declare_parameter("disparity_threshold", 0.5);
    declare_parameter("wall_margin", 0.03);
    declare_parameter("min_gap_width", 0.10);

    // Generic LiDAR preprocessing
    declare_parameter("lidar.range_min", 0.06);
    declare_parameter("lidar.range_max", 10.0);
    declare_parameter("lidar.angle_min", -2.0944);
    declare_parameter("lidar.angle_max", 2.0944);
    declare_parameter("lidar.apply_median_filter", true);
    declare_parameter("lidar.median_window_size", 3);
}

void FTGNode::loadParameters() {
    // Vehicle parameters
    config_.wheelbase = get_parameter("wheelbase").as_double();
    config_.car_width = get_parameter("car_width").as_double();

    // Speed control
    config_.max_speed = get_parameter("max_speed").as_double();
    config_.min_speed = get_parameter("min_speed").as_double();
    config_.speed_full_range = get_parameter("speed_full_range").as_double();
    config_.steer_slowdown_gain = get_parameter("steer_slowdown_gain").as_double();

    // Steering control
    config_.max_steering = get_parameter("max_steering").as_double();
    config_.steering_gain = get_parameter("steering_gain").as_double();
    config_.max_steering_rate = get_parameter("max_steering_rate").as_double();
    config_.target_ema_alpha = get_parameter("target_ema_alpha").as_double();

    // Weighted free-space scoring
    config_.heading_weight = get_parameter("heading_weight").as_double();
    config_.score_power = get_parameter("score_power").as_double();
    config_.clearance_cone_scale = get_parameter("clearance_cone_scale").as_double();
    config_.min_score_range = get_parameter("min_score_range").as_double();
    config_.gap_switch_margin = get_parameter("gap_switch_margin").as_double();
    config_.gap_switch_scans = get_parameter("gap_switch_scans").as_int();

    config_.car_length = get_parameter("car_length").as_double();
    config_.rear_overhang = get_parameter("rear_overhang").as_double();
    config_.lidar_to_rear_axle = get_parameter("lidar_to_rear_axle").as_double();
    config_.footprint_margin = get_parameter("footprint_margin").as_double();
    config_.rollout_distance = get_parameter("rollout_distance").as_double();
    config_.rollout_step = get_parameter("rollout_step").as_double();
    config_.trajectory_min_free_distance = get_parameter("trajectory_min_free_distance").as_double();
    config_.trajectory_candidate_count = get_parameter("trajectory_candidate_count").as_int();

    // Safety
    config_.emergency_brake_distance = get_parameter("emergency_brake_distance").as_double();

    // LiDAR processing
    config_.disparity_threshold = get_parameter("disparity_threshold").as_double();
    config_.wall_margin = get_parameter("wall_margin").as_double();
    config_.min_gap_width = get_parameter("min_gap_width").as_double();

    // Generic LiDAR preprocessing config
    config_.lidar_config.range_min = get_parameter("lidar.range_min").as_double();
    config_.lidar_config.range_max = get_parameter("lidar.range_max").as_double();
    config_.lidar_config.angle_min = get_parameter("lidar.angle_min").as_double();
    config_.lidar_config.angle_max = get_parameter("lidar.angle_max").as_double();
    config_.lidar_config.apply_median_filter = get_parameter("lidar.apply_median_filter").as_bool();
    config_.lidar_config.median_window_size = get_parameter("lidar.median_window_size").as_int();
}

void FTGNode::scanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg) {
    // Run FTG algorithm
    FTGOutput output = ftg_->compute(
        msg->ranges,
        msg->angle_min,
        msg->angle_max,
        msg->angle_increment,
        static_cast<double>(msg->header.stamp.sec)
            + static_cast<double>(msg->header.stamp.nanosec) * 1.0e-9
    );
    
    // Publish drive command
    publishDriveCommand(output.command);
    publishDiagnostics(output);
    
    // Performance logging (throttled to reduce overhead)
    RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
        "FTG: drivable_gaps=%zu, cmd=(%.2f, %.2f), closest=%.2fm",
        output.drivable_gaps.size(), output.command.speed, output.command.steering_angle,
        output.closest_point_dist);
    
    if (output.emergency_stop) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "Emergency stop! Closest: %.2fm", output.closest_point_dist);
    }
}

void FTGNode::publishDriveCommand(const DriveCommand& cmd) {
    auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
    drive_msg.header.stamp = now();
    drive_msg.header.frame_id = "base_link";
    drive_msg.drive.speed = cmd.speed;
    drive_msg.drive.steering_angle = cmd.steering_angle;
    
    drive_pub_->publish(drive_msg);
}

void FTGNode::publishDiagnostics(const FTGOutput& output) {
    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "ftg/decision";
    status.level = output.emergency_stop
        ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
        : (output.no_path ? diagnostic_msgs::msg::DiagnosticStatus::WARN
                          : diagnostic_msgs::msg::DiagnosticStatus::OK);
    status.message = output.emergency_stop
        ? "emergency_stop"
        : (output.no_path ? "no_path" : "gap_selected");

    auto add = [&](const std::string& key, double value) {
        diagnostic_msgs::msg::KeyValue item;
        item.key = key;
        item.value = std::to_string(value);
        status.values.push_back(item);
    };
    add("source_stamp_s", output.source_stamp_s);
    add("closest_point_dist_m", output.closest_point_dist);
    add("drivable_gap_count", static_cast<double>(output.drivable_gaps.size()));
    for (size_t i = 0; i < output.drivable_gaps.size(); ++i) {
        const auto& gap = output.drivable_gaps[i];
        const std::string prefix = "gap_" + std::to_string(i) + "_";
        add(prefix + "start_angle", gap.start_angle);
        add(prefix + "end_angle", gap.end_angle);
        add(prefix + "width", gap.angular_width);
        add(prefix + "max_clearance", gap.max_clearance);
        add(prefix + "mean_clearance", gap.mean_clearance);
        add(prefix + "mean_clipped_clearance", gap.mean_clipped_clearance);
        add(prefix + "weighted_center", gap.weighted_center_angle);
        add(prefix + "deepest_angle", gap.deepest_angle);
        add(prefix + "score", gap.score);
    }
    if (output.has_selected_drivable_gap) {
        const auto& gap = output.selected_drivable_gap;
        add("selected_gap_start_angle", gap.start_angle);
        add("selected_gap_end_angle", gap.end_angle);
        add("selected_gap_width", gap.angular_width);
        add("selected_gap_max_clearance", gap.max_clearance);
        add("selected_gap_mean_clearance", gap.mean_clearance);
        add("selected_gap_mean_clipped_clearance", gap.mean_clipped_clearance);
        add("selected_gap_score", gap.score);
        add("selected_gap_weighted_center", gap.weighted_center_angle);
        add("selected_gap_deepest_angle", gap.deepest_angle);
    }
    add("raw_target_angle", output.raw_target_angle);
    add("smoothed_target_angle", output.smoothed_target_angle);
    add("raw_steering", output.raw_steering);
    add("rate_limited_steering", output.rate_limited_steering);
    add("forward_clearance_m", output.forward_clearance);
    add("trajectory_free_distance_m", output.trajectory_free_distance);
    add("trajectory_min_clearance_m", output.trajectory_min_clearance);
    add("trajectory_collision_free", output.trajectory_collision_free ? 1.0 : 0.0);
    add("command_speed_mps", output.command.speed);
    add("command_steering_rad", output.command.steering_angle);
    add("no_path", output.no_path ? 1.0 : 0.0);
    add("emergency_stop", output.emergency_stop ? 1.0 : 0.0);
    array.status.push_back(status);
    diagnostics_pub_->publish(array);
}

}  // namespace f1tenth_control

// Register as composable node for zero-copy intra-process communication
RCLCPP_COMPONENTS_REGISTER_NODE(f1tenth_control::FTGNode)

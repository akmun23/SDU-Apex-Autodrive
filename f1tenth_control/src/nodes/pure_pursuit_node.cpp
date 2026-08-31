#include "nodes/pure_pursuit_node.hpp"

#include <chrono>

namespace f1tenth_control {

PurePursuitNode::PurePursuitNode(const rclcpp::NodeOptions& options)
    : Node("pure_pursuit_node", options)
{
    RCLCPP_INFO(get_logger(), "Initializing Pure Pursuit Node");
    
    // Declare and load parameters
    declareParameters();
    loadParameters();

    // Create controller
    controller_ = std::make_unique<PurePursuit>(config_);
    
    // Load trajectory
    if (!loadTrajectory()) {
        RCLCPP_ERROR(get_logger(), "Failed to load trajectory from: %s", trajectory_file_.c_str());
        RCLCPP_WARN(get_logger(), "Controller will be disabled until trajectory is loaded");
    }
    
    // Match the reliable QoS used by the team state publishers.  The official
    // bridge remains the only source of simulator telemetry.
    const auto state_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, state_qos,
        std::bind(&PurePursuitNode::odomCallback, this, std::placeholders::_1)
    );

    pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        pose_topic_, state_qos,
        std::bind(&PurePursuitNode::poseCallback, this, std::placeholders::_1)
    );

    // The official practice API supplies complete LiDAR scans at about 10 Hz.
    // One scan is one controller event; no synthetic wall-timer commands are
    // generated between sensor updates.
    lidar_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
        lidar_topic_, state_qos,
        std::bind(&PurePursuitNode::lidarCallback, this, std::placeholders::_1)
    );

    steering_feedback_sub_ = create_subscription<std_msgs::msg::Float32>(
        steering_feedback_topic_, state_qos,
        std::bind(&PurePursuitNode::steeringFeedbackCallback, this, std::placeholders::_1)
    );
    
    // Setup publishers
    drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
        command_topic_, 10
    );

    // Setup parameter callback
    param_callback_handle_ = add_on_set_parameters_callback(
        std::bind(&PurePursuitNode::parametersCallback, this, std::placeholders::_1)
    );
    
    RCLCPP_INFO(get_logger(), "Pure Pursuit Node initialized");
    RCLCPP_INFO(get_logger(), "  Trajectory: %s (%zu points)", 
                trajectory_file_.c_str(), controller_->getTrajectory().size());
    RCLCPP_INFO(get_logger(), "  Lookahead: %.2f - %.2f m (gain: %.2f)",
                config_.min_lookahead, config_.max_lookahead, config_.lookahead_gain);
    RCLCPP_INFO(get_logger(), "  Max speed cap: %.2f m/s", max_speed_);
    RCLCPP_INFO(get_logger(), "  Lookahead adapt: cte_weight=%.2f cte_gain=%.3f curvature_gain=%.3f",
                config_.cte_lookahead_weight, config_.cte_lookahead_gain, config_.curvature_lookahead_gain);
    RCLCPP_INFO(get_logger(), "  Speed adapt: curvature_factor=%.3f floor_ratio=%.2f",
                config_.curvature_speed_factor, config_.curvature_speed_floor_ratio);
    RCLCPP_INFO(get_logger(), "  Speed adapt (CTE): factor=%.3f floor_ratio=%.2f",
                config_.cte_speed_factor, config_.cte_speed_floor_ratio);
    RCLCPP_INFO(get_logger(), "  Speed limits: max_lat_accel=%.2f min_reg_speed=%.2f",
                config_.max_lateral_accel, config_.min_regulated_speed);
    RCLCPP_INFO(get_logger(), "  Yaw-rate damping: %.3f s", config_.yaw_rate_damping);
    RCLCPP_INFO(get_logger(), "  Command shaping: steer_rate=%.2f accel=%.2f decel=%.2f",
                max_steering_rate_, max_accel_cmd_, max_decel_cmd_);
    RCLCPP_INFO(get_logger(), "  Pose: %s (%s frame)", pose_topic_.c_str(), path_frame_.c_str());
    RCLCPP_INFO(get_logger(), "  Odom: %s", odom_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  Sensor trigger: %s (one command per scan; nominal %.1f Hz)",
                lidar_topic_.c_str(), control_rate_hz_);
    RCLCPP_INFO(get_logger(), "  Steering feedback: %s (lead gain %.2f)",
                steering_feedback_topic_.c_str(), steering_feedback_lead_gain_);
    RCLCPP_INFO(get_logger(), "  Command: %s", command_topic_.c_str());
}

void PurePursuitNode::declareParameters() {
    // Trajectory
    declare_parameter("trajectory_file", "");
    declare_parameter("odom_topic", odom_topic_);
    declare_parameter("pose_topic", pose_topic_);
    declare_parameter("lidar_topic", lidar_topic_);
    declare_parameter("command_topic", command_topic_);
    declare_parameter("steering_feedback_topic", steering_feedback_topic_);
    declare_parameter("path_frame", path_frame_);
    declare_parameter("command_frame", command_frame_);
    
    // Lookahead - sweep-optimized defaults
    declare_parameter("min_lookahead", 0.65);
    declare_parameter("max_lookahead", 1.15);
    declare_parameter("lookahead_gain", 0.14);
    declare_parameter("max_speed", 22.88);
    declare_parameter("cte_lookahead_weight", 1.0);
    declare_parameter("cte_lookahead_gain", 0.041540516);
    declare_parameter("curvature_lookahead_gain", 1.9003721);
    declare_parameter("curvature_speed_factor", 0.1015252);
    declare_parameter("curvature_speed_floor_ratio", 0.52401066);
    declare_parameter("cte_speed_factor", 1.50);
    declare_parameter("cte_speed_floor_ratio", 0.55);
    declare_parameter("max_lateral_accel", 6.50);
    declare_parameter("min_regulated_speed", 0.12);
    declare_parameter("speed_preview_distance", 4.0);
    declare_parameter("speed_profile_braking_decel", 1.50);
    declare_parameter("curvature_preview_factor", 1.6245233);
    declare_parameter("curvature_feedforward_gain", 0.25);
    declare_parameter("yaw_rate_damping", 0.0);
    
    // Corridor-aware width regulation
    declare_parameter("vehicle_half_width", 0.1365);
    declare_parameter("wall_safety_margin", 0.03);
    declare_parameter("corridor_half_width_ref", 0.35);
    declare_parameter("corridor_speed_floor_ratio", 0.25);
    declare_parameter("corridor_lookahead_factor", 2.0);
    declare_parameter("wall_bias_gain", 0.25);
    declare_parameter("wall_bias_max_m", 0.10);
    
    // Steering
    declare_parameter("max_steering", 0.5236);
    
    // Vehicle
    declare_parameter("wheelbase", 0.324);
    
    // Misc
    declare_parameter("pose_timeout_s", 0.1);
    declare_parameter("odom_timeout_s", 0.2);
    declare_parameter("state_extrapolation_max_s", 0.12);
    declare_parameter("control_rate_hz", 10.0);
    declare_parameter("localization_covariance_xy_max", 0.25);
    declare_parameter("localization_covariance_yaw_max", 0.12);
    declare_parameter("localization_required_updates", 5);
    // AutoDRIVE's documented centre-steering rate limit.
    declare_parameter("max_steering_rate", 3.2);
    declare_parameter("max_accel_cmd", 3.0);
    declare_parameter("max_decel_cmd", 8.0);
    declare_parameter("steering_feedback_timeout_s", 0.25);
    declare_parameter("steering_feedback_lead_gain", 0.25);
}

void PurePursuitNode::loadParameters() {
    trajectory_file_ = get_parameter("trajectory_file").as_string();
    odom_topic_ = get_parameter("odom_topic").as_string();
    pose_topic_ = get_parameter("pose_topic").as_string();
    lidar_topic_ = get_parameter("lidar_topic").as_string();
    command_topic_ = get_parameter("command_topic").as_string();
    steering_feedback_topic_ = get_parameter("steering_feedback_topic").as_string();
    path_frame_ = get_parameter("path_frame").as_string();
    command_frame_ = get_parameter("command_frame").as_string();

    config_.min_lookahead = std::max(0.05, get_parameter("min_lookahead").as_double());
    config_.max_lookahead = std::max(config_.min_lookahead, get_parameter("max_lookahead").as_double());
    config_.lookahead_gain = std::max(0.0, get_parameter("lookahead_gain").as_double());
    max_speed_ = std::max(0.0, get_parameter("max_speed").as_double());
    config_.cte_lookahead_weight = std::max(0.0, get_parameter("cte_lookahead_weight").as_double());
    config_.cte_lookahead_gain = std::max(0.0, get_parameter("cte_lookahead_gain").as_double());
    config_.curvature_lookahead_gain = std::max(0.0, get_parameter("curvature_lookahead_gain").as_double());
    config_.curvature_speed_factor = std::max(0.0, get_parameter("curvature_speed_factor").as_double());
    config_.curvature_speed_floor_ratio = std::clamp(
        get_parameter("curvature_speed_floor_ratio").as_double(), 0.0, 1.0);
    config_.cte_speed_factor = std::max(0.0, get_parameter("cte_speed_factor").as_double());
    config_.cte_speed_floor_ratio = std::clamp(
        get_parameter("cte_speed_floor_ratio").as_double(), 0.0, 1.0);
    config_.max_lateral_accel = std::max(0.5, get_parameter("max_lateral_accel").as_double());
    config_.min_regulated_speed = std::max(0.0, get_parameter("min_regulated_speed").as_double());
    config_.speed_preview_distance = std::max(
        0.0, get_parameter("speed_preview_distance").as_double());
    config_.speed_profile_braking_decel = std::max(
        0.0, get_parameter("speed_profile_braking_decel").as_double());
    config_.curvature_preview_factor = std::max(1.0, get_parameter("curvature_preview_factor").as_double());
    config_.curvature_feedforward_gain = std::clamp(
        get_parameter("curvature_feedforward_gain").as_double(), 0.0, 1.0);
    config_.yaw_rate_damping = std::max(0.0, get_parameter("yaw_rate_damping").as_double());
    
    // Corridor-aware width regulation
    config_.vehicle_half_width = std::max(0.01, get_parameter("vehicle_half_width").as_double());
    config_.wall_safety_margin = std::max(0.0, get_parameter("wall_safety_margin").as_double());
    config_.corridor_half_width_ref = std::max(0.01, get_parameter("corridor_half_width_ref").as_double());
    config_.corridor_speed_floor_ratio = std::clamp(
        get_parameter("corridor_speed_floor_ratio").as_double(), 0.0, 1.0);
    config_.corridor_lookahead_factor = std::max(0.0, get_parameter("corridor_lookahead_factor").as_double());
    config_.wall_bias_gain = std::clamp(
        get_parameter("wall_bias_gain").as_double(), 0.0, 1.0);
    config_.wall_bias_max_m = std::max(
        0.0, get_parameter("wall_bias_max_m").as_double());

    config_.max_steering = std::max(1e-3, get_parameter("max_steering").as_double());
    config_.wheelbase = std::max(1e-3, get_parameter("wheelbase").as_double());
    
    pose_topic_ = get_parameter("pose_topic").as_string();
    pose_timeout_s_ = std::max(0.01, get_parameter("pose_timeout_s").as_double());
    odom_timeout_s_ = std::max(0.01, get_parameter("odom_timeout_s").as_double());
    state_extrapolation_max_s_ = std::clamp(
        get_parameter("state_extrapolation_max_s").as_double(), 0.0, 0.5);
    control_rate_hz_ = std::max(1.0, get_parameter("control_rate_hz").as_double());
    localization_covariance_xy_max_ = std::max(
        0.0, get_parameter("localization_covariance_xy_max").as_double());
    localization_covariance_yaw_max_ = std::max(
        0.0, get_parameter("localization_covariance_yaw_max").as_double());
    localization_required_updates_ = std::max(
        1, static_cast<int>(get_parameter("localization_required_updates").as_int()));
    max_steering_rate_ = std::max(0.1, get_parameter("max_steering_rate").as_double());
    max_accel_cmd_ = std::max(0.1, get_parameter("max_accel_cmd").as_double());
    max_decel_cmd_ = std::max(0.1, get_parameter("max_decel_cmd").as_double());
    steering_feedback_timeout_s_ = std::max(
        0.01, get_parameter("steering_feedback_timeout_s").as_double());
    steering_feedback_lead_gain_ = std::clamp(
        get_parameter("steering_feedback_lead_gain").as_double(), 0.0, 1.0);
}

rcl_interfaces::msg::SetParametersResult PurePursuitNode::parametersCallback(
    const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = false;

    PurePursuitConfig candidate = config_;
    double candidate_max_speed = max_speed_;
    double candidate_pose_timeout = pose_timeout_s_;
    double candidate_odom_timeout = odom_timeout_s_;
    double candidate_state_extrapolation = state_extrapolation_max_s_;
    double candidate_max_steering_rate = max_steering_rate_;
    double candidate_max_accel_cmd = max_accel_cmd_;
    double candidate_max_decel_cmd = max_decel_cmd_;
    double candidate_feedback_timeout = steering_feedback_timeout_s_;
    double candidate_feedback_lead_gain = steering_feedback_lead_gain_;

    for (const auto& param : parameters) {
        if (param.get_name() == "odom_topic" ||
            param.get_name() == "pose_topic" ||
            param.get_name() == "lidar_topic" ||
            param.get_name() == "command_topic" ||
            param.get_name() == "steering_feedback_topic") {
            result.reason = "topic parameters require node restart";
            return result;
        }
        if (param.get_name() == "min_lookahead") {
            candidate.min_lookahead = param.as_double();
        } else if (param.get_name() == "max_lookahead") {
            candidate.max_lookahead = param.as_double();
        } else if (param.get_name() == "lookahead_gain") {
            candidate.lookahead_gain = param.as_double();
        } else if (param.get_name() == "max_speed") {
            candidate_max_speed = param.as_double();
        } else if (param.get_name() == "cte_lookahead_weight") {
            candidate.cte_lookahead_weight = param.as_double();
        } else if (param.get_name() == "cte_lookahead_gain") {
            candidate.cte_lookahead_gain = param.as_double();
        } else if (param.get_name() == "curvature_lookahead_gain") {
            candidate.curvature_lookahead_gain = param.as_double();
        } else if (param.get_name() == "curvature_speed_factor") {
            candidate.curvature_speed_factor = param.as_double();
        } else if (param.get_name() == "curvature_speed_floor_ratio") {
            candidate.curvature_speed_floor_ratio = param.as_double();
        } else if (param.get_name() == "cte_speed_factor") {
            candidate.cte_speed_factor = param.as_double();
        } else if (param.get_name() == "cte_speed_floor_ratio") {
            candidate.cte_speed_floor_ratio = param.as_double();
        } else if (param.get_name() == "max_lateral_accel") {
            candidate.max_lateral_accel = param.as_double();
        } else if (param.get_name() == "min_regulated_speed") {
            candidate.min_regulated_speed = param.as_double();
        } else if (param.get_name() == "speed_preview_distance") {
            candidate.speed_preview_distance = param.as_double();
        } else if (param.get_name() == "speed_profile_braking_decel") {
            candidate.speed_profile_braking_decel = param.as_double();
        } else if (param.get_name() == "curvature_preview_factor") {
            candidate.curvature_preview_factor = param.as_double();
        } else if (param.get_name() == "curvature_feedforward_gain") {
            candidate.curvature_feedforward_gain = param.as_double();
        } else if (param.get_name() == "yaw_rate_damping") {
            candidate.yaw_rate_damping = param.as_double();
        } else if (param.get_name() == "vehicle_half_width") {
            candidate.vehicle_half_width = param.as_double();
        } else if (param.get_name() == "wall_safety_margin") {
            candidate.wall_safety_margin = param.as_double();
        } else if (param.get_name() == "corridor_half_width_ref") {
            candidate.corridor_half_width_ref = param.as_double();
        } else if (param.get_name() == "corridor_speed_floor_ratio") {
            candidate.corridor_speed_floor_ratio = param.as_double();
        } else if (param.get_name() == "corridor_lookahead_factor") {
            candidate.corridor_lookahead_factor = param.as_double();
        } else if (param.get_name() == "wall_bias_gain") {
            candidate.wall_bias_gain = param.as_double();
        } else if (param.get_name() == "wall_bias_max_m") {
            candidate.wall_bias_max_m = param.as_double();
        } else if (param.get_name() == "max_steering") {
            candidate.max_steering = param.as_double();
        } else if (param.get_name() == "wheelbase") {
            candidate.wheelbase = param.as_double();
        } else if (param.get_name() == "pose_timeout_s") {
            candidate_pose_timeout = param.as_double();
        } else if (param.get_name() == "odom_timeout_s") {
            candidate_odom_timeout = param.as_double();
        } else if (param.get_name() == "state_extrapolation_max_s") {
            candidate_state_extrapolation = param.as_double();
        } else if (param.get_name() == "max_steering_rate") {
            candidate_max_steering_rate = param.as_double();
        } else if (param.get_name() == "max_accel_cmd") {
            candidate_max_accel_cmd = param.as_double();
        } else if (param.get_name() == "max_decel_cmd") {
            candidate_max_decel_cmd = param.as_double();
        } else if (param.get_name() == "steering_feedback_timeout_s") {
            candidate_feedback_timeout = param.as_double();
        } else if (param.get_name() == "steering_feedback_lead_gain") {
            candidate_feedback_lead_gain = param.as_double();
        }
    }

    auto finite = [](double v) { return std::isfinite(v); };
    auto finite_and_nonnegative = [&](double v) { return finite(v) && v >= 0.0; };

    if (!finite(candidate.min_lookahead) || candidate.min_lookahead < 0.05) {
        result.reason = "min_lookahead must be finite and >= 0.05";
        return result;
    }
    if (!finite(candidate.max_lookahead) || candidate.max_lookahead < candidate.min_lookahead) {
        result.reason = "max_lookahead must be finite and >= min_lookahead";
        return result;
    }
    if (!finite_and_nonnegative(candidate.lookahead_gain)) {
        result.reason = "lookahead_gain must be finite and >= 0";
        return result;
    }
    if (!finite_and_nonnegative(candidate_max_speed)) {
        result.reason = "max_speed must be finite and >= 0";
        return result;
    }
    if (!finite_and_nonnegative(candidate.cte_lookahead_weight) ||
        !finite_and_nonnegative(candidate.cte_lookahead_gain) ||
        !finite_and_nonnegative(candidate.curvature_lookahead_gain) ||
        !finite_and_nonnegative(candidate.curvature_speed_factor) ||
        !finite_and_nonnegative(candidate.cte_speed_factor)) {
        result.reason = "lookahead/speed gains must be finite and >= 0";
        return result;
    }
    if (!finite(candidate.curvature_speed_floor_ratio) ||
        candidate.curvature_speed_floor_ratio < 0.0 ||
        candidate.curvature_speed_floor_ratio > 1.0) {
        result.reason = "curvature_speed_floor_ratio must be in [0,1]";
        return result;
    }
    if (!finite(candidate.cte_speed_floor_ratio) ||
        candidate.cte_speed_floor_ratio < 0.0 ||
        candidate.cte_speed_floor_ratio > 1.0) {
        result.reason = "cte_speed_floor_ratio must be in [0,1]";
        return result;
    }
    if (!finite(candidate.max_lateral_accel) || candidate.max_lateral_accel <= 0.1) {
        result.reason = "max_lateral_accel must be finite and > 0.1";
        return result;
    }
    if (!finite_and_nonnegative(candidate.min_regulated_speed)) {
        result.reason = "min_regulated_speed must be finite and >= 0";
        return result;
    }
    if (!finite_and_nonnegative(candidate.speed_preview_distance) ||
        !finite_and_nonnegative(candidate.speed_profile_braking_decel)) {
        result.reason = "speed preview parameters must be finite and >= 0";
        return result;
    }
    if (!finite(candidate.curvature_preview_factor) || candidate.curvature_preview_factor < 1.0) {
        result.reason = "curvature_preview_factor must be finite and >= 1.0";
        return result;
    }
    if (!finite(candidate.curvature_feedforward_gain) ||
        candidate.curvature_feedforward_gain < 0.0 ||
        candidate.curvature_feedforward_gain > 1.0) {
        result.reason = "curvature_feedforward_gain must be finite and in [0,1]";
        return result;
    }
    if (!finite_and_nonnegative(candidate.yaw_rate_damping)) {
        result.reason = "yaw_rate_damping must be finite and >= 0";
        return result;
    }
    // Corridor-aware width regulation validation
    if (!finite(candidate.vehicle_half_width) || candidate.vehicle_half_width <= 0.01) {
        result.reason = "vehicle_half_width must be finite and > 0.01";
        return result;
    }
    if (!finite_and_nonnegative(candidate.wall_safety_margin)) {
        result.reason = "wall_safety_margin must be finite and >= 0";
        return result;
    }
    if (!finite(candidate.corridor_half_width_ref) || candidate.corridor_half_width_ref <= 0.01) {
        result.reason = "corridor_half_width_ref must be finite and > 0.01";
        return result;
    }
    if (!finite(candidate.corridor_speed_floor_ratio) ||
        candidate.corridor_speed_floor_ratio < 0.0 ||
        candidate.corridor_speed_floor_ratio > 1.0) {
        result.reason = "corridor_speed_floor_ratio must be in [0,1]";
        return result;
    }
    if (!finite_and_nonnegative(candidate.corridor_lookahead_factor)) {
        result.reason = "corridor_lookahead_factor must be finite and >= 0";
        return result;
    }
    if (!finite(candidate.wall_bias_gain) ||
        candidate.wall_bias_gain < 0.0 || candidate.wall_bias_gain > 1.0) {
        result.reason = "wall_bias_gain must be finite and in [0,1]";
        return result;
    }
    if (!finite_and_nonnegative(candidate.wall_bias_max_m)) {
        result.reason = "wall_bias_max_m must be finite and >= 0";
        return result;
    }
    if (!finite(candidate.max_steering) || candidate.max_steering <= 0.0) {
        result.reason = "max_steering must be finite and > 0";
        return result;
    }
    if (!finite(candidate.wheelbase) || candidate.wheelbase <= 0.0) {
        result.reason = "wheelbase must be finite and > 0";
        return result;
    }
    if (!finite(candidate_pose_timeout) || candidate_pose_timeout <= 0.0) {
        result.reason = "pose_timeout_s must be finite and > 0";
        return result;
    }
    if (!finite(candidate_odom_timeout) || candidate_odom_timeout <= 0.0) {
        result.reason = "odom_timeout_s must be finite and > 0";
        return result;
    }
    if (!finite(candidate_state_extrapolation) ||
        candidate_state_extrapolation < 0.0 || candidate_state_extrapolation > 0.5) {
        result.reason = "state_extrapolation_max_s must be finite and in [0, 0.5]";
        return result;
    }
    if (!finite(candidate_max_steering_rate) || candidate_max_steering_rate <= 0.0) {
        result.reason = "max_steering_rate must be finite and > 0";
        return result;
    }
    if (!finite(candidate_max_accel_cmd) || candidate_max_accel_cmd <= 0.0) {
        result.reason = "max_accel_cmd must be finite and > 0";
        return result;
    }
    if (!finite(candidate_max_decel_cmd) || candidate_max_decel_cmd <= 0.0) {
        result.reason = "max_decel_cmd must be finite and > 0";
        return result;
    }
    if (!finite(candidate_feedback_timeout) || candidate_feedback_timeout <= 0.0) {
        result.reason = "steering_feedback_timeout_s must be finite and > 0";
        return result;
    }
    if (!finite(candidate_feedback_lead_gain) ||
        candidate_feedback_lead_gain < 0.0 || candidate_feedback_lead_gain > 1.0) {
        result.reason = "steering_feedback_lead_gain must be finite and in [0,1]";
        return result;
    }

    candidate.curvature_speed_floor_ratio = std::clamp(candidate.curvature_speed_floor_ratio, 0.0, 1.0);
    candidate.cte_speed_floor_ratio = std::clamp(candidate.cte_speed_floor_ratio, 0.0, 1.0);
    candidate.corridor_speed_floor_ratio = std::clamp(candidate.corridor_speed_floor_ratio, 0.0, 1.0);
    candidate.curvature_feedforward_gain = std::clamp(candidate.curvature_feedforward_gain, 0.0, 1.0);

    {
        std::scoped_lock lock(state_mutex_, controller_mutex_);
        config_ = candidate;
        max_speed_ = candidate_max_speed;
        pose_timeout_s_ = candidate_pose_timeout;
        odom_timeout_s_ = candidate_odom_timeout;
        state_extrapolation_max_s_ = candidate_state_extrapolation;
        max_steering_rate_ = candidate_max_steering_rate;
        max_accel_cmd_ = candidate_max_accel_cmd;
        max_decel_cmd_ = candidate_max_decel_cmd;
        steering_feedback_timeout_s_ = candidate_feedback_timeout;
        steering_feedback_lead_gain_ = candidate_feedback_lead_gain;
        if (controller_) {
            controller_->setConfig(config_);
        }
    }

    result.successful = true;
    return result;
}

bool PurePursuitNode::loadTrajectory() {
    if (trajectory_file_.empty()) {
        return false;
    }

    bool ok = false;
    {
        std::lock_guard<std::mutex> lock(controller_mutex_);
        ok = controller_->loadTrajectory(trajectory_file_);
    }

    if (ok) {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            trajectory_loaded_ = true;
        }
        size_t count = 0;
        double len = 0.0;
        {
            std::lock_guard<std::mutex> lock(controller_mutex_);
            count = controller_->getTrajectory().size();
            len = controller_->getTrajectoryLength();
        }
        RCLCPP_INFO(get_logger(), "Loaded trajectory with %zu waypoints (%.1f m)",
                    count, len);
        return true;
    }
    
    return false;
}

void PurePursuitNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_state_.velocity = msg->twist.twist.linear.x;
        current_state_.angular_velocity = msg->twist.twist.angular.z;
        odom_received_ = true;
        last_odom_time_ = now();
        last_odom_stamp_ = rclcpp::Time(msg->header.stamp, get_clock()->get_clock_type());
        if (last_odom_stamp_.nanoseconds() == 0) {
            last_odom_stamp_ = last_odom_time_;
        }
    }
}

void PurePursuitNode::steeringFeedbackCallback(
    const std_msgs::msg::Float32::ConstSharedPtr msg) {
    if (!std::isfinite(msg->data)) {
        return;
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    // AutoDRIVE publishes steering feedback in radians, unlike the normalized
    // Float32 steering_command sent by the actuator adapter.
    steering_feedback_angle_ = static_cast<double>(msg->data);
    steering_feedback_received_ = true;
    last_steering_feedback_time_ = now();
}

void PurePursuitNode::poseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    if (!msg->header.frame_id.empty() && msg->header.frame_id != path_frame_) {
        RCLCPP_ERROR_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Rejected pose in frame '%s'; expected '%s'",
            msg->header.frame_id.c_str(), path_frame_.c_str());
        return;
    }

    const double covariance_xy = std::max(
        msg->pose.covariance[0], msg->pose.covariance[7]);
    const double covariance_yaw = msg->pose.covariance[35];
    const bool covariance_good =
        std::isfinite(covariance_xy) && std::isfinite(covariance_yaw) &&
        covariance_xy >= 0.0 && covariance_yaw >= 0.0 &&
        covariance_xy <= localization_covariance_xy_max_ &&
        covariance_yaw <= localization_covariance_yaw_max_;

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!covariance_good) {
            localization_good_updates_ = 0;
            pose_received_ = false;
        } else {
            localization_good_updates_++;
        }
        current_state_.pose.x = msg->pose.pose.position.x;
        current_state_.pose.y = msg->pose.pose.position.y;
        const double qx = msg->pose.pose.orientation.x;
        const double qy = msg->pose.pose.orientation.y;
        const double qz = msg->pose.pose.orientation.z;
        const double qw = msg->pose.pose.orientation.w;
        current_state_.pose.theta = std::atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz));
        if (localization_good_updates_ >= localization_required_updates_) {
            pose_received_ = true;
            last_pose_time_ = now();
            last_pose_stamp_ = rclcpp::Time(msg->header.stamp, get_clock()->get_clock_type());
            if (last_pose_stamp_.nanoseconds() == 0) {
                last_pose_stamp_ = last_pose_time_;
            }
        }
    }

    if (!covariance_good) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "Waiting for AMCL covariance: xy=%.3f yaw=%.3f",
            covariance_xy, covariance_yaw);
    }

}

void PurePursuitNode::lidarCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg) {
    // Use the official LiDAR event as the control cadence. controlLoop() reads
    // the newest AMCL pose and encoder/IMU odometry and fails safe if either is
    // stale or not yet qualified.
    rclcpp::Time event_stamp(msg->header.stamp, get_clock()->get_clock_type());
    if (event_stamp.nanoseconds() == 0) {
        event_stamp = now();
    }
    controlLoop(event_stamp);
}

void PurePursuitNode::controlLoop(const rclcpp::Time & event_stamp) {
    std::unique_lock<std::mutex> control_lock(control_mutex_, std::try_to_lock);
    if (!control_lock.owns_lock()) {
        return;
    }

    bool trajectory_loaded = false;
    bool pose_received = false;
    bool odom_received = false;
    rclcpp::Time last_pose_time;
    rclcpp::Time last_odom_time;
    rclcpp::Time last_pose_stamp;
    rclcpp::Time last_odom_stamp;
    double pose_timeout_s = 0.1;
    double odom_timeout_s = 0.2;
    double state_extrapolation_max_s = 0.0;
    double max_speed = 0.0;
    bool steering_feedback_received = false;
    double steering_feedback_angle = 0.0;
    rclcpp::Time last_steering_feedback_time;
    double steering_feedback_timeout_s = 0.25;
    double steering_feedback_lead_gain = 0.0;

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        trajectory_loaded = trajectory_loaded_;
        pose_received = pose_received_;
        odom_received = odom_received_;
        last_pose_time = last_pose_time_;
        last_odom_time = last_odom_time_;
        last_pose_stamp = last_pose_stamp_;
        last_odom_stamp = last_odom_stamp_;
        pose_timeout_s = pose_timeout_s_;
        odom_timeout_s = odom_timeout_s_;
        state_extrapolation_max_s = state_extrapolation_max_s_;
        max_speed = max_speed_;
        steering_feedback_received = steering_feedback_received_;
        steering_feedback_angle = steering_feedback_angle_;
        last_steering_feedback_time = last_steering_feedback_time_;
        steering_feedback_timeout_s = steering_feedback_timeout_s_;
        steering_feedback_lead_gain = steering_feedback_lead_gain_;
    }

    if (!trajectory_loaded) {
        publishDriveCommand(0.0, 0.0);
        return;
    }

    if (!pose_received) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "No pose received on %s yet", pose_topic_.c_str());
        publishDriveCommand(0.0, 0.0);
        return;
    }

    if (!odom_received) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "No odometry received yet; issuing stop for fail-safe");
        publishDriveCommand(0.0, 0.0);
        return;
    }

    const double pose_age = (now() - last_pose_time).seconds();
    if (pose_age > pose_timeout_s) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "Pose timeout %.3fs > %.3fs; issuing stop for fail-safe",
                             pose_age, pose_timeout_s);
        publishDriveCommand(0.0, 0.0);
        return;
    }

    const double odom_age = (now() - last_odom_time).seconds();
    if (odom_age > odom_timeout_s) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "Odom timeout %.3fs > %.3fs; issuing stop for fail-safe",
                             odom_age, odom_timeout_s);
        publishDriveCommand(0.0, 0.0);
        return;
    }

    VehicleState state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state = current_state_;
    }

    // AMCL/EKF is scan-driven while odometry is available continuously at the
    // bridge cadence.  Compensate from the accepted pose timestamp to the
    // command publication time, using only allowed odometry velocity and yaw
    // rate.  The scan timestamp describes when the measurement was taken; the
    // command is applied now, so stopping the prediction at the scan event
    // would intentionally leave one delivery interval of turn-in lag.
    const rclcpp::Time command_time = now();
    const rclcpp::Time prediction_stamp =
        event_stamp > command_time ? event_stamp : command_time;
    if (state_extrapolation_max_s > 0.0 &&
        last_pose_stamp.nanoseconds() != 0 && prediction_stamp.nanoseconds() != 0) {
        // The bridge can deliver a correctly timestamped pose immediately
        // after the simulator has already advanced one telemetry cycle.  In
        // that case receipt age is small while stamp age captures the motion
        // that occurred before delivery.  Conversely, a delayed DDS sample
        // has a larger receipt age than its sensor-stamp age.  Use the larger
        // non-negative age so either form of latency is compensated without
        // inventing a fixed-rate control timer.
        const double stamped_gap =
            std::max(0.0, (prediction_stamp - last_pose_stamp).seconds());
        const double received_gap =
            std::max(0.0, (command_time - last_pose_time).seconds());
        const double sensor_gap = std::max(stamped_gap, received_gap);
        const double dt_predict = std::clamp(sensor_gap, 0.0, state_extrapolation_max_s);
        if (dt_predict > 1.0e-4 && std::isfinite(state.velocity) &&
            std::isfinite(state.angular_velocity)) {
            const double yaw_mid = state.pose.theta + 0.5 * state.angular_velocity * dt_predict;
            state.pose.x += state.velocity * dt_predict * std::cos(yaw_mid);
            state.pose.y += state.velocity * dt_predict * std::sin(yaw_mid);
            state.pose.theta = std::atan2(
                std::sin(state.pose.theta + state.angular_velocity * dt_predict),
                std::cos(state.pose.theta + state.angular_velocity * dt_predict));
        }
    }
    
    // Compute control (protected against concurrent trajectory/config updates)
    PurePursuitOutput output;
    {
        std::lock_guard<std::mutex> lock(controller_mutex_);
        if (!controller_ || !controller_->hasTrajectory()) {
            publishDriveCommand(0.0, 0.0);
            return;
        }
        output = controller_->compute(state);
    }
    
    if (output.valid) {
        output.target_speed = std::clamp(output.target_speed, 0.0, max_speed);

        const rclcpp::Time now_t = now();
        double dt_cmd = 0.01;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!cmd_history_initialized_) {
                last_cmd_time_ = now_t;
                // Start from the physical neutral command.  Initialising the
                // history to the requested output bypasses the rate limiter
                // on the first scan, which is unsafe when the spawn pose is
                // a little off the raceline.
                last_cmd_steering_ = 0.0;
                last_cmd_speed_ = 0.0;
                cmd_history_initialized_ = true;
            }
            dt_cmd = std::max(1e-3, (now_t - last_cmd_time_).seconds());
        }

        const double max_delta_steer = max_steering_rate_ * dt_cmd;
        double cmd_steer = output.steering_angle;
        double cmd_speed = output.target_speed;

        // AutoDRIVE publishes steering feedback in radians. Lead the
        // requested angle only when the actuator is lagging in the same
        // direction. Never amplify a sign reversal: at 10 Hz an old feedback
        // sample can legitimately have the opposite sign while the requested
        // command is changing sides.
        const bool feedback_fresh = steering_feedback_received &&
            (now_t - last_steering_feedback_time).seconds() <= steering_feedback_timeout_s;
        if (feedback_fresh && steering_feedback_lead_gain > 0.0) {
            const double feedback_angle = std::clamp(
                steering_feedback_angle, -config_.max_steering, config_.max_steering);
            const double command_epsilon = 1.0e-4;
            const bool same_direction =
                std::abs(cmd_steer) <= command_epsilon ||
                std::abs(feedback_angle) <= command_epsilon ||
                cmd_steer * feedback_angle > 0.0;
            if (same_direction && std::abs(cmd_steer) > std::abs(feedback_angle)) {
                cmd_steer += steering_feedback_lead_gain * (cmd_steer - feedback_angle);
                cmd_steer = std::clamp(cmd_steer, -config_.max_steering, config_.max_steering);
            }
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            const double steer_err = cmd_steer - last_cmd_steering_;
            const double steer_step = std::clamp(steer_err, -max_delta_steer, max_delta_steer);
            cmd_steer = last_cmd_steering_ + steer_step;

            const double max_speed_step_up = max_accel_cmd_ * dt_cmd;
            const double max_speed_step_down = max_decel_cmd_ * dt_cmd;
            const double speed_err = cmd_speed - last_cmd_speed_;
            if (speed_err >= 0.0) {
                cmd_speed = last_cmd_speed_ + std::min(speed_err, max_speed_step_up);
            } else {
                cmd_speed = last_cmd_speed_ + std::max(speed_err, -max_speed_step_down);
            }

            cmd_speed = std::clamp(cmd_speed, 0.0, max_speed);
            last_cmd_steering_ = cmd_steer;
            last_cmd_speed_ = cmd_speed;
            last_cmd_time_ = now_t;
        }

        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "Pure Pursuit: pose=(%.3f, %.3f, %.3f) cte=%.3f closest=%zu target=%zu "
            "steer=%.3f speed=%.3f",
            state.pose.x, state.pose.y, state.pose.theta,
            output.cross_track_error, output.closest_idx, output.target_idx,
            cmd_steer, cmd_speed);
        
        const double requested_accel = std::clamp(
            (cmd_speed - std::max(0.0, state.velocity)) / std::max(dt_cmd, 0.05),
            -max_decel_cmd_, max_accel_cmd_);
        publishDriveCommand(cmd_steer, cmd_speed, requested_accel);
    } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, 
                            "Invalid Pure Pursuit output");
        publishDriveCommand(0.0, 0.0);
    }
}

void PurePursuitNode::publishDriveCommand(double steering, double speed, double acceleration) {
    if (!std::isfinite(steering) || !std::isfinite(speed) || !std::isfinite(acceleration)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "Non-finite command detected; publishing safe stop");
        steering = 0.0;
        speed = 0.0;
        acceleration = 0.0;
    }

    auto msg = ackermann_msgs::msg::AckermannDriveStamped();
    msg.header.stamp = now();
    msg.header.frame_id = command_frame_;
    msg.drive.steering_angle = steering;
    msg.drive.speed = speed;
    msg.drive.acceleration = acceleration;
    drive_pub_->publish(msg);
}

}  // namespace f1tenth_control

// Component registration for composable nodes
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(f1tenth_control::PurePursuitNode)

// Main entry point for standalone executable
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<f1tenth_control::PurePursuitNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

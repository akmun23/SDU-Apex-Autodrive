// One ROS adapter around the BachelorProject MPC core.  It deliberately uses
// the same legal pose/odometry and Ackermann target-speed boundary as the
// established Pure Pursuit controller.  Unity truth/contact data is absent.

#include "mpc_rti.h"
#include "mpc_control_time_predictor.hpp"
#include "mpc_state_synchronizer.hpp"

#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace f1tenth_mpc {
namespace {

double clamp(double value, double lower, double upper)
{
    return std::min(std::max(value, lower), upper);
}

using Waypoint = MpcTrajectorySample_t;

int64_t steady_time_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

class MpcControllerNode final : public rclcpp::Node {
public:
    explicit MpcControllerNode(const rclcpp::NodeOptions & options)
        : Node("mpc_controller_node", options)
    {
        enabled_ = declare_parameter<bool>("enabled", false);
        shadow_mode_ = declare_parameter<bool>("shadow_mode", false);
        if (enabled_ && shadow_mode_)
            throw std::runtime_error(
                "MPC cannot have command authority and shadow mode simultaneously");
        odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom");
        pose_topic_ = declare_parameter<std::string>(
            "pose_topic", "/current_map_pose");
        command_topic_ = declare_parameter<std::string>("command_topic", "/cmd/speed");
        diagnostics_topic_ = declare_parameter<std::string>(
            "diagnostics_topic", "/mpc/diagnostics");
        path_frame_ = declare_parameter<std::string>("path_frame", "map");
        command_frame_ = declare_parameter<std::string>("command_frame", "base_link");
        trajectory_file_ = declare_parameter<std::string>("trajectory_file", "");
        max_speed_mps_ = clamp(declare_parameter<double>("max_speed_mps", 16.0), 0.0, 16.0);
        startup_speed_mps_ = clamp(
            declare_parameter<double>("startup_speed_mps", 1.5), 0.0, max_speed_mps_);
        startup_ramp_laps_ = std::max(
            0.0, declare_parameter<double>("startup_ramp_laps", 1.0));
        state_extrapolation_max_s_ = std::clamp(
            declare_parameter<double>("state_extrapolation_max_s", 0.12),
            0.0, 0.5);
        control_time_mode_name_ = declare_parameter<std::string>(
            "control_time_predictor_mode", "ct2");
        if (control_time_mode_name_ == "ct0") {
            control_time_mode_ = MpcControlTimeMode::kNoExtrapolation;
        } else if (control_time_mode_name_ == "ct1") {
            control_time_mode_ = MpcControlTimeMode::kConstantBodyTwist;
        } else if (control_time_mode_name_ == "ct2") {
            control_time_mode_ =
                MpcControlTimeMode::kAcceptedModelCommandHistory;
        } else {
            throw std::runtime_error(
                "control_time_predictor_mode must be ct0, ct1, or ct2");
        }
        control_time_predictor_config_.maximum_state_age_s =
            state_extrapolation_max_s_;
        control_time_predictor_config_.model_integration_step_s =
            TIME_STEP_SECONDS;
        control_time_predictor_config_.maximum_command_speed_mps =
            max_speed_mps_;
        pose_odom_max_skew_s_ = std::clamp(
            declare_parameter<double>("pose_odom_max_skew_s", 0.12),
            0.0, 0.5);
        source_dt_min_s_ = std::max(
            0.001, declare_parameter<double>("source_dt_min_s", 0.001));
        source_dt_max_s_ = std::max(
            source_dt_min_s_, declare_parameter<double>("source_dt_max_s", 0.250));
        MpcSyncConfig sync_config;
        sync_config.source_dt_min_s = source_dt_min_s_;
        sync_config.source_dt_max_s = source_dt_max_s_;
        sync_config.max_pose_odom_skew_s = pose_odom_max_skew_s_;
        sync_config.max_state_age_s = state_extrapolation_max_s_;
        state_synchronizer_ = MpcStateSynchronizer(sync_config);
        startup_path_max_distance_m_ = std::max(
            0.0, declare_parameter<double>("startup_path_max_distance_m", 0.80));
        startup_path_heading_tolerance_rad_ = std::max(
            0.0, declare_parameter<double>("startup_path_heading_tolerance_rad", 0.75));
        localization_covariance_xy_max_ = std::max(
            0.0, declare_parameter<double>("localization_covariance_xy_max", 0.25));
        localization_covariance_yaw_max_ = std::max(
            0.0, declare_parameter<double>("localization_covariance_yaw_max", 0.12));
        localization_required_updates_ = std::max(
            1, static_cast<int>(
                declare_parameter<int>("localization_required_updates", 5)));
        projection_search_distance_m_ = std::clamp(
            declare_parameter<double>("projection_search_distance_m", 3.0),
            0.25, 6.0);

        if (enabled_) {
            command_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
                command_topic_, rclcpp::QoS(10));
        }
        if (shadow_mode_ || enabled_) {
            diagnostics_pub_ = create_publisher<std_msgs::msg::String>(
                diagnostics_topic_, rclcpp::QoS(rclcpp::KeepLast(2)));
        }
        if (shadow_mode_) {
            observed_command_sub_ =
                create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
                    command_topic_, rclcpp::QoS(rclcpp::KeepLast(1)),
                    std::bind(&MpcControllerNode::command_callback, this,
                              std::placeholders::_1));
        }
        // Control must consume the newest state, not replay a queue of stale
        // 40 Hz samples after an executor/DDS scheduling pause.
        const auto latest_state_qos = rclcpp::QoS(rclcpp::KeepLast(1));
        pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            pose_topic_, latest_state_qos,
            std::bind(&MpcControllerNode::pose_callback, this, std::placeholders::_1));
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, latest_state_qos,
            std::bind(&MpcControllerNode::odom_callback, this, std::placeholders::_1));

        auto declare_weight = [this](const char * name, float fallback) {
            return static_cast<float>(declare_parameter<double>(name, fallback));
        };
        rti_config_.model.weight_e_y = declare_weight("weight_e_y", 1500.0f);
        rti_config_.model.weight_e_psi = declare_weight("weight_e_psi", 50.0f);
        rti_config_.model.weight_u = declare_weight("weight_u", 200.0f);
        rti_config_.model.weight_target_speed_state = declare_weight(
            "weight_target_speed_state", 20.0f);
        rti_config_.model.weight_v = declare_weight("weight_v", 0.0f);
        rti_config_.model.weight_r = declare_weight("weight_r", 1.5f);
        rti_config_.model.weight_steering_command = declare_weight(
            "weight_steering_command", 1.0f);
        rti_config_.model.weight_steering_rate = declare_weight(
            "weight_steering_rate", 2.0f);
        rti_config_.model.weight_target_speed_rate = declare_weight(
            "weight_target_speed_rate", 0.5f);
        rti_config_.model.weight_steering_rate_change = declare_weight(
            "weight_steering_rate_change", 5.0f);
        rti_config_.model.weight_target_speed_rate_change = declare_weight(
            "weight_target_speed_rate_change", 5.0f);
        rti_config_.model.terminal_multiplier = declare_weight(
            "terminal_multiplier", 3.0f);
        rti_config_.model.max_speed_mps = static_cast<float>(max_speed_mps_);
        rti_config_.model.active_speed_ceiling_mps =
            static_cast<float>(max_speed_mps_);
        rti_config_.model.max_steering_rad = static_cast<float>(
            declare_parameter<double>("max_steering_rad", SOURCE_MAX_STEERING_RAD));
        rti_config_.model.max_steering_rate_radps = static_cast<float>(
            declare_parameter<double>("max_steering_rate_radps",
                                      SOURCE_STEERING_RATE_RADPS));
        rti_config_.model.max_target_speed_rate_increase_mps2 = static_cast<float>(
            declare_parameter<double>("max_target_speed_rate_increase_mps2", 3.0));
        rti_config_.model.max_target_speed_rate_reduction_mps2 = static_cast<float>(
            declare_parameter<double>("max_target_speed_rate_reduction_mps2", 8.0));
        rti_config_.model.corridor_margin_m = static_cast<float>(
            declare_parameter<double>("corridor_margin_m", 0.30));
        rti_config_.model.first_prediction_corridor_margin_m =
            static_cast<float>(declare_parameter<double>(
                "first_prediction_corridor_margin_m",
                rti_config_.model.corridor_margin_m));
        /* Match the exact min-time planner's AutoDRIVE footprint contract:
         * 0.30 m minimum planning width, 0.273 m physical width, 0.43 m from
         * rear axle to front bumper, and 0.15 m wall clearance. */
        rti_config_.model.planning_half_width_m = static_cast<float>(
            declare_parameter<double>("planning_half_width_m", 0.15));
        rti_config_.model.vehicle_half_width_m = static_cast<float>(
            declare_parameter<double>("vehicle_half_width_m", 0.1365));
        rti_config_.model.vehicle_longitudinal_extent_m = static_cast<float>(
            declare_parameter<double>("vehicle_longitudinal_extent_m", 0.43));
        rti_config_.model.wall_clearance_m = static_cast<float>(
            declare_parameter<double>("wall_clearance_m", 0.15));
        rti_config_.model.corridor_preview_halfwidth_m = static_cast<float>(
            declare_parameter<double>("corridor_preview_halfwidth_m", 0.10));
        rti_config_.model.nonlinear_corridor_tolerance_m = static_cast<float>(
            declare_parameter<double>("nonlinear_corridor_tolerance_m", 0.001));
        const std::string recovery_seed_policy = declare_parameter<std::string>(
            "recovery_seed_policy", "nominal");
        if (recovery_seed_policy == "nominal") {
            rti_config_.model.recovery_seed_policy =
                MPC_RTI_RECOVERY_SEED_NOMINAL;
        } else if (recovery_seed_policy == "heading_feedback") {
            rti_config_.model.recovery_seed_policy =
                MPC_RTI_RECOVERY_SEED_HEADING_FEEDBACK;
        } else if (recovery_seed_policy == "brake_heading_feedback") {
            rti_config_.model.recovery_seed_policy =
                MPC_RTI_RECOVERY_SEED_BRAKE_HEADING_FEEDBACK;
        } else {
            throw std::runtime_error(
                "recovery_seed_policy must be nominal, heading_feedback, "
                "or brake_heading_feedback");
        }
        rti_config_.model.recovery_steering_k_e_y = static_cast<float>(
            declare_parameter<double>("recovery_steering_k_e_y", 0.0));
        rti_config_.model.recovery_steering_k_e_psi = static_cast<float>(
            declare_parameter<double>("recovery_steering_k_e_psi", 0.0));
        rti_config_.model.recovery_steering_k_r = static_cast<float>(
            declare_parameter<double>("recovery_steering_k_r", 0.0));
        const int configured_iterations = declare_parameter<int>(
            "max_solver_iterations", 100);
        rti_config_.solver.max_iterations = static_cast<uint16_t>(
            std::clamp(configured_iterations, 1, 100));
        rti_config_.solver.rho = static_cast<float>(
            declare_parameter<double>("admm_rho", 7.0));
        rti_config_.solver.rho_u = static_cast<float>(
            declare_parameter<double>("admm_rho_input", 7.0));
        rti_config_.solver.tolerance = static_cast<float>(
            declare_parameter<double>("solver_tolerance", 0.01));
        rti_config_.solver.adaptive_rho = declare_parameter<bool>(
            "adaptive_rho", true) ? 1 : 0;
        rti_config_.solver.shared_rho = 0;
        rti_config_.solver.use_prefactorization = declare_parameter<bool>(
            "use_riccati_prefactorization", true) ? 1 : 0;
        const std::string refinement_mode = declare_parameter<std::string>(
            "rti_refinement_mode", "r1");
        if (refinement_mode == "r1") {
            rti_config_.refinement_mode = MPC_RTI_REFINEMENT_R1;
        } else if (refinement_mode == "r2") {
            rti_config_.refinement_mode = MPC_RTI_REFINEMENT_R2;
        } else if (refinement_mode == "adaptive" || refinement_mode == "ra") {
            rti_config_.refinement_mode = MPC_RTI_REFINEMENT_ADAPTIVE;
        } else {
            throw std::runtime_error(
                "rti_refinement_mode must be r1, r2, or adaptive");
        }
        rti_config_.rti2_progress_error_trigger_m = static_cast<float>(
            declare_parameter<double>("rti2_progress_error_trigger_m", 0.10));
        rti_config_.rti2_curvature_error_trigger_per_m = static_cast<float>(
            declare_parameter<double>("rti2_curvature_error_trigger_per_m", 0.02));
        rti_config_.rti2_bound_error_trigger_m = static_cast<float>(
            declare_parameter<double>("rti2_bound_error_trigger_m", 0.05));
        rti_config_.rti2_min_corridor_slack_trigger_m = static_cast<float>(
            declare_parameter<double>("rti2_min_corridor_slack_trigger_m", 0.25));
        rti_config_.rti2_steering_rate_correction_trigger_radps =
            static_cast<float>(declare_parameter<double>(
                "rti2_steering_rate_correction_trigger_radps", 0.50));
        rti_config_.rti2_target_speed_rate_correction_trigger_mps2 =
            static_cast<float>(declare_parameter<double>(
                "rti2_target_speed_rate_correction_trigger_mps2", 1.0));
        rti_config_.rti2_residual_imbalance_trigger = static_cast<float>(
            declare_parameter<double>("rti2_residual_imbalance_trigger", 8.0));
        rti_config_.rti2_lateral_load_trigger_mps2 = static_cast<float>(
            declare_parameter<double>("rti2_lateral_load_trigger_mps2", 3.0));
        rti_config_.rti2_nonsmooth_columns_trigger = declare_parameter<int>(
            "rti2_nonsmooth_columns_trigger", 50);
        rti_config_.rti2_residual_recovery_limit = static_cast<float>(
            declare_parameter<double>("rti2_residual_recovery_limit", 0.25));
        rti_config_.degraded_residual_limit = static_cast<float>(
            declare_parameter<double>("solver_degraded_tolerance", 0.05));
        rti_config_.maximum_regularization = static_cast<float>(
            declare_parameter<double>("solver_max_regularization", 0.01));
        rti_config_.max_consecutive_degraded_solves =
            declare_parameter<int>("max_consecutive_degraded_solves", 3);
        mpc_rti_memory_reset(&rti_memory_);
        trajectory_loaded_ = load_trajectory(trajectory_file_);
        if (!trajectory_loaded_) {
            RCLCPP_ERROR(get_logger(),
                "MPC has no valid trajectory; controller remains inhibited");
        }
        if (!enabled_ && !shadow_mode_) {
            RCLCPP_WARN(get_logger(),
                "MPC is inactive; no command publisher is created");
        } else if (shadow_mode_) {
            RCLCPP_WARN(get_logger(),
                "MPC shadow active: subscribing to legal inputs; command authority is disabled");
        }
    }

private:
    bool load_trajectory(const std::string & path)
    {
        if (path.empty()) {
            return false;
        }
        std::ifstream input(path);
        if (!input.is_open()) {
            RCLCPP_ERROR(get_logger(), "Cannot open MPC trajectory: %s", path.c_str());
            return false;
        }

        std::vector<Waypoint> parsed;
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line.front() == '#') {
                continue;
            }
            std::stringstream stream(line);
            std::string token;
            std::vector<double> values;
            bool valid = true;
            while (std::getline(stream, token, ',')) {
                try {
                    values.push_back(std::stod(token));
                } catch (...) {
                    valid = false;
                    break;
                }
            }
            if (!valid || values.size() < 6) {
                continue;
            }
            Waypoint point;
            point.s = values[0];
            point.x = values[1];
            point.y = values[2];
            point.heading = values[3];
            point.curvature = values[4];
            point.speed = values[5];
            point.acceleration = values.size() >= 7 ? values[6] : 0.0;
            if (values.size() >= 9) {
                point.left_bound = values[7];
                point.right_bound = values[8];
            }
            if (!std::isfinite(point.s) || !std::isfinite(point.x) ||
                !std::isfinite(point.y) || !std::isfinite(point.heading) ||
                !std::isfinite(point.curvature) || !std::isfinite(point.speed) ||
                !std::isfinite(point.acceleration)) {
                continue;
            }
            if (!std::isfinite(point.left_bound) || point.left_bound <= 0.0) {
                point.left_bound = 1.0;
            }
            if (!std::isfinite(point.right_bound) || point.right_bound <= 0.0) {
                point.right_bound = 1.0;
            }
            parsed.push_back(point);
        }
        trajectory_ = std::move(parsed);
        std::size_t point_count = trajectory_.size();
        if (!mpc_trajectory_prepare(
                trajectory_.data(), &point_count, &track_length_m_)) {
            trajectory_.clear();
            track_length_m_ = 0.0;
            return false;
        }
        trajectory_.resize(point_count);
        RCLCPP_INFO(get_logger(), "Loaded MPC trajectory with %zu points (%.2f m)",
                    trajectory_.size(), track_length_m_);
        return true;
    }

    void update_progress(const MpcPathProjection_t & projection)
    {
        if (!progress_initialized_) {
            progress_initialized_ = true;
            last_closest_index_ = projection.segment;
            last_projected_s_ = projection.s;
            return;
        }
        double unwrapped_s = projection.s;
        while (unwrapped_s - last_projected_s_ < -0.5 * track_length_m_)
            unwrapped_s += track_length_m_;
        while (unwrapped_s - last_projected_s_ > 0.5 * track_length_m_)
            unwrapped_s -= track_length_m_;
        const double delta = unwrapped_s - last_projected_s_;
        if (delta > 0.0 && delta < 0.5 * track_length_m_)
            startup_progress_m_ += delta;
        last_projected_s_ = unwrapped_s;
        last_closest_index_ = projection.segment;
    }

    double active_speed_ceiling() const
    {
        if (startup_ramp_laps_ <= 1.0e-6 || track_length_m_ <= 1.0e-6) {
            return max_speed_mps_;
        }
        const double fraction = clamp(
            startup_progress_m_ / (startup_ramp_laps_ * track_length_m_), 0.0, 1.0);
        return startup_speed_mps_ + (max_speed_mps_ - startup_speed_mps_) * fraction;
    }

    double local_raceline_speed_cap() const
    {
        if (!trajectory_loaded_ || trajectory_.empty() ||
            !progress_initialized_) {
            return std::min(startup_speed_mps_, max_speed_mps_);
        }
        MpcTrajectorySample_t sample{};
        if (!mpc_trajectory_sample(trajectory_.data(), trajectory_.size(),
                track_length_m_, last_projected_s_, &sample)) {
            return std::min(startup_speed_mps_, max_speed_mps_);
        }
        return clamp(sample.speed, 0.0, max_speed_mps_);
    }

    void publish_command(double steering, double speed)
    {
        if (!command_pub_) return;
        auto command = ackermann_msgs::msg::AckermannDriveStamped();
        const rclcpp::Time command_time = now();
        command.header.stamp = command_time;
        command.header.frame_id = command_frame_;
        command.drive.steering_angle = static_cast<float>(steering);
        command.drive.speed = static_cast<float>(speed);
        // The actuator interface owns the safe speed-to-throttle conversion.
        command.drive.acceleration = 0.0f;
        {
            std::lock_guard<std::mutex> lock(command_history_mutex_);
            command_history_.push({command_time.nanoseconds(), steering, speed});
        }
        command_pub_->publish(command);
    }

    void publish_stop(const char * reason, bool emit_shadow_diagnostic = true)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s", reason);
        mpc_rti_memory_reset(&rti_memory_);
        accepted_command_established_ = false;
        if (enabled_) {
            target_speed_mps_ = 0.0;
            target_speed_initialized_ = true;
            last_steering_command_rad_ = 0.0;
            last_steering_rate_radps_ = 0.0;
            last_target_speed_rate_mps2_ = 0.0;
            publish_command(0.0, 0.0);
        } else if (shadow_mode_) {
            target_speed_initialized_ = false;
        }
        if (diagnostics_pub_ && emit_shadow_diagnostic)
            publish_shadow_failure(reason);
    }

    /* A transient estimator/RTI failure should not freeze a stale cornering
     * command. Preserve the last accepted RTI warm start, reduce target speed
     * at the source-valid braking slew, and—when a legal path projection is
     * available—rate-limit steering toward the local raceline feedforward.
     * Before the first legal state, remain stopped; AMCL publishes a
     * provisional pose specifically so the controller can then perform the
     * slow startup travel required for global lock. */
    void publish_driving_fallback(const char *reason,
                                  double observed_speed_mps,
                                  double requested_speed_mps,
                                  bool emit_shadow_diagnostic = true,
                                  const MpcPathProjection_t *path_projection = nullptr)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s; "
            "using bounded decelerating recovery command", reason);
        if (enabled_) {
            (void)observed_speed_mps;
            (void)requested_speed_mps;
            if (!accepted_command_established_) {
                // Before the first accepted MPC solution there is no safe
                // steering command to hold. Match the proven Pure Pursuit
                // startup contract: stay neutral until localization is
                // qualified and RTI has produced one legal command.
                target_speed_mps_ = 0.0;
                target_speed_initialized_ = true;
                last_steering_command_rad_ = 0.0;
                last_steering_rate_radps_ = 0.0;
                last_target_speed_rate_mps2_ = 0.0;
                publish_command(0.0, 0.0);
                if (diagnostics_pub_ && emit_shadow_diagnostic)
                    publish_shadow_failure(reason);
                return;
            }
            const double speed_ceiling = active_speed_ceiling();
            const double previous_target = target_speed_initialized_
                ? clamp(target_speed_mps_, 0.0, speed_ceiling) : 0.0;
            const double reduction =
                rti_config_.model.max_target_speed_rate_reduction_mps2 *
                TIME_STEP_SECONDS;
            const double recovery_speed = target_speed_initialized_
                ? std::max(0.0, previous_target - reduction) : 0.0;

            const double previous_steering = clamp(
                std::isfinite(last_steering_command_rad_)
                    ? last_steering_command_rad_ : 0.0,
                -rti_config_.model.max_steering_rad,
                rti_config_.model.max_steering_rad);
            double recovery_steering = previous_steering;
            if (path_projection && trajectory_loaded_) {
                MpcTrajectorySample_t sample{};
                if (mpc_trajectory_sample(
                        trajectory_.data(), trajectory_.size(), track_length_m_,
                        path_projection->s, &sample)) {
                    double desired = std::atan(
                        sample.curvature / MPC_YAW_RATE_STEERING_GAIN_PER_M);
                    desired -= rti_config_.model.recovery_steering_k_e_y *
                        path_projection->lateral_error;
                    desired -= rti_config_.model.recovery_steering_k_e_psi *
                        path_projection->heading_error;
                    desired = clamp(
                        desired, -rti_config_.model.max_steering_rad,
                        rti_config_.model.max_steering_rad);
                    const double max_step =
                        rti_config_.model.max_steering_rate_radps *
                        TIME_STEP_SECONDS;
                    recovery_steering = clamp(
                        desired, recovery_steering - max_step,
                        recovery_steering + max_step);
                }
            }

            last_steering_rate_radps_ =
                (recovery_steering - previous_steering) / TIME_STEP_SECONDS;
            last_target_speed_rate_mps2_ =
                (recovery_speed - previous_target) / TIME_STEP_SECONDS;
            last_steering_command_rad_ = recovery_steering;
            target_speed_mps_ = recovery_speed;
            target_speed_initialized_ = true;
            publish_command(last_steering_command_rad_, target_speed_mps_);
        } else if (shadow_mode_) {
            target_speed_initialized_ = false;
        }
        if (diagnostics_pub_ && emit_shadow_diagnostic)
            publish_shadow_failure(reason);
    }

    void command_callback(
        const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr message)
    {
        int64_t stamp_ns = rclcpp::Time(
            message->header.stamp, get_clock()->get_clock_type()).nanoseconds();
        if (stamp_ns <= 0) stamp_ns = now().nanoseconds();
        const double speed = message->drive.speed;
        const double steering = message->drive.steering_angle;
        if (!std::isfinite(speed) || speed < 0.0 ||
            speed > max_speed_mps_ || !std::isfinite(steering) ||
            std::abs(steering) > rti_config_.model.max_steering_rad) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC shadow ignored invalid/out-of-envelope command history");
            return;
        }
        std::lock_guard<std::mutex> lock(command_history_mutex_);
        if (observed_command_stamp_ns_ > 0) {
            const int64_t delta_ns = stamp_ns - observed_command_stamp_ns_;
            if (delta_ns > 0) {
                const double dt = static_cast<double>(delta_ns) * 1.0e-9;
                observed_steering_rate_radps_ =
                    (steering - observed_steering_command_rad_) / dt;
                observed_target_speed_rate_mps2_ =
                    (speed - observed_target_speed_mps_) / dt;
            } else {
                // Keep the newest delivered command through timestamp
                // reversals; a derivative is undefined for this pair.
                observed_steering_rate_radps_ = 0.0;
                observed_target_speed_rate_mps2_ = 0.0;
            }
        }
        if (!command_history_.push({stamp_ns, steering, speed})) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC shadow rejected command history outside its causal envelope");
            return;
        }
        observed_command_stamp_ns_ = stamp_ns;
        observed_target_speed_mps_ = speed;
        observed_steering_command_rad_ = steering;
    }

    void publish_shadow_failure(const char * reason)
    {
        if (!diagnostics_pub_) return;
        std_msgs::msg::String message;
        std::ostringstream json;
        json << "{\"status\":\"rejected\",\"reason\":\"" << reason
             << "\",\"source_stamp_ns\":" << last_source_stamp_.nanoseconds()
             << '}';
        message.data = json.str();
        diagnostics_pub_->publish(message);
    }

    static const char * cycle_status_name(MpcRtiCycleStatus_t status)
    {
        switch (status) {
        case MPC_RTI_CYCLE_ACCEPTED_OPTIMAL: return "accepted_optimal";
        case MPC_RTI_CYCLE_ACCEPTED_DEGRADED: return "accepted_degraded";
        case MPC_RTI_CYCLE_REJECTED_INPUT: return "rejected_input";
        case MPC_RTI_CYCLE_REJECTED_SOLVER: return "rejected_solver";
        case MPC_RTI_CYCLE_REJECTED_RESIDUAL: return "rejected_residual";
        case MPC_RTI_CYCLE_REJECTED_REGULARIZATION:
            return "rejected_regularization";
        case MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT:
            return "rejected_nonlinear_rollout";
        }
        return "unknown";
    }

    static const char * cycle_status_name_code(int status)
    {
        if (status < 0) return "not_run";
        return cycle_status_name(static_cast<MpcRtiCycleStatus_t>(status));
    }

    static const char * nonlinear_failure_reason_name(int reason)
    {
        switch (reason) {
        case MPC_RTI_ROLLOUT_OK: return "none";
        case MPC_RTI_ROLLOUT_INVALID_INPUT: return "invalid_input";
        case MPC_RTI_ROLLOUT_INVALID_MODEL: return "invalid_model";
        case MPC_RTI_ROLLOUT_COMMAND_LIMIT: return "command_limit";
        case MPC_RTI_ROLLOUT_STATE_LIMIT: return "state_limit";
        case MPC_RTI_ROLLOUT_CORRIDOR: return "corridor";
        default: return "not_applicable";
        }
    }

    void publish_shadow_result(const MpcRtiState_t &state,
        const MpcSynchronizedState &synchronized,
        const MpcControlTimePrediction &control_prediction,
        double control_prediction_us, double progress,
        double source_dt_s, const MpcRtiCycleResult_t &result,
        double solve_us, int64_t control_ros_stamp_ns,
        int64_t callback_steady_ns, int64_t synchronize_steady_ns)
    {
        if (!diagnostics_pub_) return;
        const int steps[] = {1, 5, 10, 20, 30};
        int64_t observed_command_stamp_ns = 0;
        double observed_target_speed_mps = 0.0;
        double observed_steering_command_rad = 0.0;
        double observed_steering_rate_radps = 0.0;
        double observed_target_speed_rate_mps2 = 0.0;
        {
            std::lock_guard<std::mutex> lock(command_history_mutex_);
            observed_command_stamp_ns = observed_command_stamp_ns_;
            observed_target_speed_mps = observed_target_speed_mps_;
            observed_steering_command_rad = observed_steering_command_rad_;
            observed_steering_rate_radps = observed_steering_rate_radps_;
            observed_target_speed_rate_mps2 = observed_target_speed_rate_mps2_;
        }
        std::ostringstream json;
        const auto json_number = [&json](double value) {
            if (std::isfinite(value)) json << value;
            else json << "null";
        };
        MpcTrajectorySample_t current_path_sample{};
        const bool current_path_sample_valid = mpc_trajectory_sample(
            trajectory_.data(), trajectory_.size(), track_length_m_, progress,
            &current_path_sample);
        json << std::setprecision(9)
            << "{\"status\":\"" << cycle_status_name(result.status)
            << "\",\"nonlinear_failure_stage\":"
            << result.nonlinear_failure_stage
            << ",\"nonlinear_failure_reason\":\""
            << nonlinear_failure_reason_name(result.nonlinear_failure_reason)
            << "\",\"r1_nonlinear_failure_stage\":"
            << result.r1_nonlinear_failure_stage
            << ",\"r1_nonlinear_failure_reason\":\""
            << nonlinear_failure_reason_name(
                result.r1_nonlinear_failure_reason)
            << "\",\"r2_nonlinear_failure_stage\":"
            << result.r2_nonlinear_failure_stage
            << ",\"r2_nonlinear_failure_reason\":\""
            << nonlinear_failure_reason_name(
                result.r2_nonlinear_failure_reason) << '"'
            << ",\"source_stamp_ns\":" << synchronized.source_stamp_ns
            << ",\"odom_source_stamp_ns\":"
            << synchronized.odom_source_stamp_ns
            << ",\"map_pose_source_stamp_ns\":"
            << synchronized.map_pose_source_stamp_ns
            << ",\"control_ros_stamp_ns\":" << control_ros_stamp_ns
            << ",\"source_age_s\":" << synchronized.source_age_s
            << ",\"source_dt_s\":" << source_dt_s
            << ",\"pose_odom_skew_s\":" << synchronized.pose_odom_skew_s
            << ",\"control_time_prediction\":{\"mode\":\""
            << control_time_mode_name_ << "\",\"age_s\":"
            << control_prediction.age_s << ",\"elapsed_us\":"
            << control_prediction_us << ",\"command_changes_used\":"
            << control_prediction.command_changes_used
            << ",\"command_event_stamps_ns\":[";
        for (std::size_t i = 0;
             i < control_prediction.command_event_stamp_count; ++i) {
            if (i > 0) json << ',';
            json << control_prediction.command_event_stamps_ns[i];
        }
        json << "]"
            << ",\"command_fallback\":"
            << (control_prediction.used_command_fallback ? "true" : "false")
            << ",\"time_fallback\":"
            << (control_prediction.used_time_fallback ? "true" : "false")
            << ",\"source_map_pose\":[" << synchronized.map_x << ','
            << synchronized.map_y << ',' << synchronized.map_yaw
            << "],\"predicted_map_pose\":["
            << control_prediction.state.map_x << ','
            << control_prediction.state.map_y << ','
            << control_prediction.state.map_yaw << ']'
            << ",\"predicted_speed_state\":["
            << control_prediction.state.u << ',' << control_prediction.state.v
            << ',' << control_prediction.state.yaw_rate << ']'
            << ",\"prediction_previous_command_rate\":["
            << control_prediction.previous_steering_rate_radps << ','
            << control_prediction.previous_target_speed_rate_mps2 << ']'
            << "}"
            << ",\"host_timing_ns\":{" << "\"callback\":"
            << callback_steady_ns << ",\"synchronize\":"
            << synchronize_steady_ns << '}'
            << ",\"observed_command\":{" << "\"stamp_ns\":"
            << observed_command_stamp_ns << ",\"target_speed_mps\":"
            << observed_target_speed_mps << ",\"steering_rad\":"
            << observed_steering_command_rad << ",\"steering_rate_radps\":"
            << observed_steering_rate_radps << ",\"target_speed_rate_mps2\":"
            << observed_target_speed_rate_mps2 << '}'
            << ",\"progress_m\":" << progress
            << ",\"path_curvature_per_m\":";
        json_number(current_path_sample_valid ? current_path_sample.curvature :
            std::numeric_limits<double>::quiet_NaN());
        json << ",\"reference\":{\"acceleration_mps2\":";
        json_number(current_path_sample_valid ? current_path_sample.acceleration :
            std::numeric_limits<double>::quiet_NaN());
        const double current_reference_speed = current_path_sample_valid
            ? std::min(current_path_sample.speed,
                static_cast<double>(rti_config_.model.active_speed_ceiling_mps))
            : std::numeric_limits<double>::quiet_NaN();
        const double current_reference_rate = current_path_sample_valid
            ? clamp(current_path_sample.acceleration,
                -rti_config_.model.max_target_speed_rate_reduction_mps2,
                rti_config_.model.max_target_speed_rate_increase_mps2)
            : std::numeric_limits<double>::quiet_NaN();
        const double current_target_feedforward = current_path_sample_valid
            ? clamp(current_reference_speed +
                (current_reference_rate - MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 -
                 MPC_LONGITUDINAL_SPEED_COEFF_PER_S * current_reference_speed -
                 MPC_LONGITUDINAL_TARGET_RATE_COEFF * current_reference_rate) /
                MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S -
                0.5 * current_reference_rate * TIME_STEP_SECONDS,
                0.0, rti_config_.model.active_speed_ceiling_mps)
            : std::numeric_limits<double>::quiet_NaN();
        json << ",\"speed_mps\":";
        json_number(current_reference_speed);
        json << ",\"speed_rate_mps2\":";
        json_number(current_reference_rate);
        json << ",\"target_speed_mps\":";
        json_number(current_target_feedforward);
        json << "}";
        const double current_physical_half_extent =
            std::max(
                static_cast<double>(rti_config_.model.planning_half_width_m),
                static_cast<double>(rti_config_.model.vehicle_half_width_m) *
                    std::abs(std::cos(state.plant.e_psi)) +
                static_cast<double>(
                    rti_config_.model.vehicle_longitudinal_extent_m) *
                    std::abs(std::sin(state.plant.e_psi)));
        const double current_required_wall_margin = std::max(
            static_cast<double>(rti_config_.model.corridor_margin_m),
            static_cast<double>(rti_config_.model.wall_clearance_m) +
                current_physical_half_extent);
        const double current_raw_wall_clearance = current_path_sample_valid
            ? std::min(
                current_path_sample.left_bound - state.plant.e_y,
                current_path_sample.right_bound + state.plant.e_y)
            : std::numeric_limits<double>::quiet_NaN();
        json << ",\"footprint_required_wall_margin_m\":";
        json_number(current_required_wall_margin);
        json << ",\"current_raw_wall_clearance_m\":";
        json_number(current_raw_wall_clearance);
        json << ",\"current_physical_wall_slack_m\":";
        json_number(current_path_sample_valid
            ? current_raw_wall_clearance - current_required_wall_margin
            : std::numeric_limits<double>::quiet_NaN());
        json << ",\"state\":[" << state.plant.e_y << ','
            << state.plant.e_psi << ',' << state.plant.u << ','
            << state.plant.v << ',' << state.plant.r << ','
            << state.plant.target_speed << ',' << state.plant.steering_command
            << ',' << state.previous_steering_rate << ','
            << state.previous_target_speed_rate << ']'
            << ",\"solver\":{\"iterations\":" << result.solver_iterations
            << ",\"primal_residual\":" << result.primal_residual
            << ",\"dual_residual\":" << result.dual_residual
            << ",\"max_regularization\":"
            << result.maximum_regularization
            << ",\"regularization_count\":" << result.regularization_count
            << ",\"nonsmooth_columns\":"
            << result.nonsmooth_jacobian_columns
            << ",\"rho_start\":" << result.rho_start
            << ",\"rho_u_start\":" << result.rho_u_start
            << ",\"rho_final\":" << result.rho_final
            << ",\"rho_u_final\":" << result.rho_u_final
            << ",\"rho_change_count\":" << result.rho_change_count
            << ",\"factorization_count\":" << result.factorization_count
            << ",\"factorization_time_ns\":" << result.factorization_time_ns
            << ",\"solve_us\":" << solve_us << '}'
            << ",\"first_action\":["
            << result.published_steering_command << ','
            << result.published_target_speed << ','
            << result.first_control.steering_rate << ','
            << result.first_control.target_speed_rate << ']'
            << ",\"rti_iterations_used\":" << result.rti_iterations_used
            << ",\"rti2_triggered\":"
            << (result.rti2_triggered ? "true" : "false")
            << ",\"rti2_trigger_reason_mask\":"
            << result.rti2_trigger_reason_mask
            << ",\"rti2_budget_skipped\":"
            << (result.rti2_budget_skipped ? "true" : "false")
            << ",\"r1_status\":\""
            << cycle_status_name_code(result.r1_status) << '\"'
            << ",\"r2_status\":\""
            << cycle_status_name_code(result.r2_status) << '\"'
            << ",\"r1_nonlinear_objective\":";
        json_number(result.r1_nonlinear_objective);
        json << ",\"r2_nonlinear_objective\":";
        json_number(result.r2_nonlinear_objective);
        json << ",\"r1_min_corridor_slack\":";
        json_number(result.r1_min_corridor_slack);
        json << ",\"r2_min_corridor_slack\":";
        json_number(result.r2_min_corridor_slack);
        json << ",\"r1_first_action\":["
             << result.r1_first_action.steering_rate << ','
             << result.r1_first_action.target_speed_rate << ']'
             << ",\"r2_first_action\":["
             << result.r2_first_action.steering_rate << ','
             << result.r2_first_action.target_speed_rate << ']'
             << ",\"r1_solver_iterations\":"
             << result.r1_solver_iterations
             << ",\"r2_solver_iterations\":"
             << result.r2_solver_iterations
             << ",\"r1_solve_us\":" << result.r1_solve_us
             << ",\"r2_solve_us\":" << result.r2_solve_us
             << ",\"total_rti_us\":" << result.total_rti_us
             << ",\"selected_candidate\":\""
             << (result.selected_candidate == 2 ? "R2" :
                 result.selected_candidate == 1 ? "R1" : "none") << '\"'
             << ",\"nominal_vs_candidate_progress_error_max_m\":";
        json_number(result.max_candidate_progress_error_m);
        json << ",\"nominal_vs_candidate_curvature_error_max_per_m\":";
        json_number(result.max_candidate_curvature_error_per_m);
        json << ",\"nominal_vs_candidate_left_bound_error_max_m\":";
        json_number(result.max_candidate_left_bound_error_m);
        json << ",\"nominal_vs_candidate_right_bound_error_max_m\":";
        json_number(result.max_candidate_right_bound_error_m);
        json << ",\"minimum_predicted_corridor_slack_m\":";
        json_number(result.minimum_predicted_corridor_slack_m);
        json << ",\"corridor_margin_m\":"
             << rti_config_.model.corridor_margin_m
             << ",\"first_prediction_corridor_margin_m\":"
             << rti_config_.model.first_prediction_corridor_margin_m
             << ",\"rti2_residual_recovery_limit\":"
             << rti_config_.rti2_residual_recovery_limit
             << ",\"recovery_active\":"
             << (result.recovery_active ? "true" : "false")
             << ",\"recovery_not_found\":"
             << (result.recovery_not_found ? "true" : "false")
             << ",\"recovery_reentry_stage\":"
             << result.recovery_reentry_stage
             << ",\"recovery_initial_normal_violation_m\":";
        json_number(result.recovery_initial_normal_violation_m);
        json << ",\"recovery_max_seed_violation_m\":";
        json_number(result.recovery_max_seed_violation_m);
        json << ",\"lateral_accel_proxy_mps2\":";
        json_number(result.lateral_accel_proxy_mps2);
        json << ",\"lateral_accel_proxy_stage\":"
             << result.lateral_accel_proxy_stage
             << ",\"lateral_accel_proxy_by_stage_mps2\":[";
        for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
            if (k > 0) json << ',';
            json_number(result.lateral_accel_proxy_by_stage_mps2[k]);
        }
        json << ']';
        json << ",\"predictions\":[";

        if (result.status != MPC_RTI_CYCLE_ACCEPTED_OPTIMAL &&
            result.status != MPC_RTI_CYCLE_ACCEPTED_DEGRADED) {
            json << "],\"diagnostic_publish_steady_ns\":"
                 << steady_time_ns() << '}';
            std_msgs::msg::String message;
            message.data = json.str();
            diagnostics_pub_->publish(message);
            return;
        }

        bool first = true;
        for (const int step : steps) {
            if (!first) json << ',';
            first = false;
            const MpcRtiState_t &predicted = rti_memory_.nominal.states[step];
            MpcTrajectorySample_t sample{};
            const double predicted_s = rti_memory_.nominal.progress[step];
            if (!mpc_trajectory_sample(trajectory_.data(), trajectory_.size(),
                    track_length_m_, predicted_s, &sample)) {
                publish_shadow_failure("prediction_reference_unavailable");
                return;
            }
            const double reference_speed = std::min(
                sample.speed,
                static_cast<double>(
                    rti_config_.model.active_speed_ceiling_mps));
            const double reference_rate = clamp(
                sample.acceleration,
                -rti_config_.model.max_target_speed_rate_reduction_mps2,
                rti_config_.model.max_target_speed_rate_increase_mps2);
            const double target_speed_reference = clamp(reference_speed +
                (reference_rate - MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 -
                 MPC_LONGITUDINAL_SPEED_COEFF_PER_S * reference_speed -
                 MPC_LONGITUDINAL_TARGET_RATE_COEFF * reference_rate) /
                MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S -
                0.5 * reference_rate * TIME_STEP_SECONDS,
                0.0, rti_config_.model.active_speed_ceiling_mps);
            const double feedforward = clamp(
                std::atan(sample.curvature /
                    MPC_YAW_RATE_STEERING_GAIN_PER_M),
                -rti_config_.model.max_steering_rad,
                rti_config_.model.max_steering_rad);
            const MpcModelControl_t &control =
                rti_memory_.nominal.controls[step - 1];
            json << "{\"n\":" << step << ",\"s_m\":" << predicted_s
                << ",\"state\":[" << predicted.plant.e_y << ','
                << predicted.plant.e_psi << ',' << predicted.plant.u << ','
                << predicted.plant.v << ',' << predicted.plant.r << ','
                << predicted.plant.target_speed << ','
                << predicted.plant.steering_command << ']'
                << ",\"reference\":[0,0," << reference_speed
                << ",0," << sample.curvature * reference_speed << ','
                << feedforward << "," << target_speed_reference << ','
                << reference_rate << ']'
                << ",\"input\":[" << control.steering_rate << ','
                << control.target_speed_rate << ']'
                << ",\"corridor\":[" << sample.left_bound << ','
                << sample.right_bound << "]}";
        }
        json << "],\"diagnostic_publish_steady_ns\":"
             << steady_time_ns() << '}';
        std_msgs::msg::String message;
        message.data = json.str();
        diagnostics_pub_->publish(message);
    }

    void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr message)
    {
        if (!message->header.frame_id.empty() && message->header.frame_id != path_frame_) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
                "Rejected MPC pose frame '%s'; expected '%s'",
                message->header.frame_id.c_str(), path_frame_.c_str());
            return;
        }
        const auto & q = message->pose.pose.orientation;
        const double yaw = std::atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        const double covariance_x = message->pose.covariance[0];
        const double covariance_y = message->pose.covariance[7];
        const double covariance_yaw = message->pose.covariance[35];
        const double covariance_xy = std::max(covariance_x, covariance_y);
        const bool covariance_good =
            std::isfinite(covariance_x) && std::isfinite(covariance_y) &&
            std::isfinite(covariance_yaw) &&
            covariance_x >= 0.0 && covariance_y >= 0.0 &&
            covariance_yaw >= 0.0 &&
            covariance_xy <= localization_covariance_xy_max_ &&
            covariance_yaw <= localization_covariance_yaw_max_;
        if (!std::isfinite(message->pose.pose.position.x) ||
            !std::isfinite(message->pose.pose.position.y) || !std::isfinite(yaw) ||
            !covariance_good) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            localization_good_updates_ = 0;
            localization_ready_ = false;
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "MPC waiting for qualified map pose: covariance xy=%.6g yaw=%.6g",
                covariance_xy, covariance_yaw);
            return;
        }
        rclcpp::Time stamp(message->header.stamp, get_clock()->get_clock_type());
        if (stamp.nanoseconds() <= 0) stamp = now();
        std::lock_guard<std::mutex> lock(state_mutex_);
        const auto status = state_synchronizer_.set_map_pose({
            stamp.nanoseconds(), message->pose.pose.position.x,
            message->pose.pose.position.y, yaw});
        if (status != MpcSyncStatus::kOk) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC map-pose handoff rejected: %s",
                MpcStateSynchronizer::status_name(status));
            return;
        }
        localization_good_updates_ =
            std::min(localization_good_updates_ + 1, localization_required_updates_);
        localization_ready_ =
            localization_good_updates_ >= localization_required_updates_;
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr message)
    {
        const int64_t callback_steady_ns = steady_time_ns();
        const auto & odom_pose = message->pose.pose;
        const auto & q = odom_pose.orientation;
        const double odom_yaw = std::atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        rclcpp::Time odom_stamp(
            message->header.stamp, get_clock()->get_clock_type());
        if (!std::isfinite(odom_pose.position.x) ||
            !std::isfinite(odom_pose.position.y) || !std::isfinite(odom_yaw) ||
            !std::isfinite(message->twist.twist.linear.x) ||
            !std::isfinite(message->twist.twist.linear.y) ||
            !std::isfinite(message->twist.twist.angular.z)) {
            publish_driving_fallback(
                "MPC received uncertain odometry", 0.0, target_speed_mps_);
            return;
        }
        if (odom_stamp.nanoseconds() <= 0) odom_stamp = now();

        if (!enabled_ && !shadow_mode_) return;
        if (!trajectory_loaded_) {
            publish_stop("MPC requires a valid trajectory");
            return;
        }

        MpcSyncStatus odom_status;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            odom_status = state_synchronizer_.push_odometry({
                odom_stamp.nanoseconds(), odom_pose.position.x,
                odom_pose.position.y, odom_yaw,
                message->twist.twist.linear.x,
                message->twist.twist.linear.y,
                message->twist.twist.angular.z});
        }

        if (odom_status != MpcSyncStatus::kOk) {
            publish_driving_fallback(
                "MPC rejected uncertain legal odometry sample", 0.0,
                target_speed_mps_);
            return;
        }

        bool localization_ready = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            localization_ready = localization_ready_;
        }
        if (!localization_ready) {
            publish_driving_fallback(
                "MPC waiting for qualified localization", 0.0,
                target_speed_mps_);
            return;
        }

        MpcSynchronizedState coherent_state;
        MpcSyncStatus sync_status;
        const rclcpp::Time control_ros_time = now();
        const int64_t synchronize_steady_ns = steady_time_ns();
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            sync_status = state_synchronizer_.synchronize(
                control_ros_time.nanoseconds(), &coherent_state);
        }
        if (sync_status != MpcSyncStatus::kOk) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC waiting for legal state input: %s",
                MpcStateSynchronizer::status_name(sync_status));
            // A delayed/temporarily missing AMCL sample is not a command to
            // stop. Refresh the last bounded command so a downstream command
            // watchdog cannot turn estimator uncertainty into a vehicle stop.
            // The next legal map pose is still allowed to correct the state.
            publish_driving_fallback(
                "MPC waiting for legal state input", 0.0, target_speed_mps_);
            return;
        }

        const rclcpp::Time state_stamp(
            coherent_state.source_stamp_ns, get_clock()->get_clock_type());
        const double source_dt = last_source_stamp_.nanoseconds() == 0 ?
            0.0 : (state_stamp - last_source_stamp_).seconds();
        last_source_stamp_ = state_stamp;

        MpcPathProjection_t projection{};
        const std::size_t previous_segment = progress_initialized_ ?
            last_closest_index_ : std::numeric_limits<std::size_t>::max();
        const std::size_t projection_radius =
            previous_segment < trajectory_.size()
            ? mpc_trajectory_search_radius_for_distance(
                  trajectory_.data(), trajectory_.size(), track_length_m_,
                  previous_segment, projection_search_distance_m_)
            : 0;
        if (!mpc_trajectory_project(trajectory_.data(), trajectory_.size(),
                track_length_m_, coherent_state.map_x, coherent_state.map_y,
                coherent_state.map_yaw, previous_segment, projection_radius,
                &projection)) {
            publish_driving_fallback("MPC could not project uncertain pose",
                coherent_state.u, target_speed_mps_);
            return;
        }
        if (!startup_path_validated_) {
            if (projection.distance > startup_path_max_distance_m_ ||
                std::abs(projection.heading_error) > startup_path_heading_tolerance_rad_) {
                publish_driving_fallback("MPC startup pose is not yet aligned",
                    coherent_state.u, target_speed_mps_);
                return;
            }
            startup_path_validated_ = true;
        }
        MpcCommandHistory command_history_snapshot;
        {
            std::lock_guard<std::mutex> lock(command_history_mutex_);
            command_history_snapshot = command_history_;
        }
        MpcControlTimePrediction control_time_prediction{};
        const auto prediction_start = std::chrono::steady_clock::now();
        const MpcControlTimeStatus prediction_status = predict_to_control_time(
            control_time_mode_, coherent_state, control_ros_time.nanoseconds(),
            command_history_snapshot, trajectory_.data(), trajectory_.size(),
            track_length_m_, projection.segment, &projection,
            control_time_predictor_config_, &control_time_prediction);
        const auto prediction_finish = std::chrono::steady_clock::now();
        const double control_prediction_us =
            std::chrono::duration<double, std::micro>(
                prediction_finish - prediction_start).count();
        if (prediction_status != MpcControlTimeStatus::kOk) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC command-time prediction rejected state: %s",
                control_time_status_name(prediction_status));
            publish_driving_fallback(
                "MPC command-time state prediction is uncertain",
                coherent_state.u, target_speed_mps_, true, &projection);
            return;
        }
        const MpcSynchronizedState &command_time_state =
            control_time_prediction.state;
        const MpcPathProjection_t &command_time_projection =
            control_time_prediction.projection;
        update_progress(command_time_projection);
        /* Shadow compares against the running controller's actual profile; it
         * must not apply the MPC-only first-lap ramp to its inputs. */
        const double speed_ceiling = shadow_mode_ ?
            max_speed_mps_ : active_speed_ceiling();
        rti_config_.model.max_speed_mps = static_cast<float>(max_speed_mps_);
        rti_config_.model.active_speed_ceiling_mps =
            static_cast<float>(speed_ceiling);

        double commanded_speed = control_time_prediction.target_speed_mps;
        double commanded_steering =
            control_time_prediction.steering_command_rad;
        double previous_steering_rate =
            control_time_prediction.previous_steering_rate_radps;
        double previous_target_speed_rate =
            control_time_prediction.previous_target_speed_rate_mps2;
        if (!target_speed_initialized_) {
            if (!shadow_mode_ || command_history_snapshot.size() == 0)
                commanded_speed = clamp(
                    std::max(0.0, command_time_state.u), 0.0, speed_ceiling);
            if (enabled_) target_speed_mps_ = commanded_speed;
            target_speed_initialized_ = true;
        }

        MpcRtiState_t state{};
        state.plant.e_y = static_cast<float>(command_time_projection.lateral_error);
        state.plant.e_psi = static_cast<float>(command_time_projection.heading_error);
        state.plant.u = static_cast<float>(std::max(0.0, command_time_state.u));
        state.plant.v = static_cast<float>(command_time_state.v);
        state.plant.r = static_cast<float>(command_time_state.yaw_rate);
        state.plant.target_speed = static_cast<float>(commanded_speed);
        state.plant.steering_command = static_cast<float>(commanded_steering);
        state.previous_steering_rate = static_cast<float>(previous_steering_rate);
        state.previous_target_speed_rate =
            static_cast<float>(previous_target_speed_rate);

        MpcRtiCycleResult_t result{};
        const auto solve_start = std::chrono::steady_clock::now();
        const MpcRtiCycleStatus_t status = mpc_rti_solve_cycle(
            &state, last_projected_s_, trajectory_.data(), trajectory_.size(),
            track_length_m_, TIME_STEP_SECONDS, PREDICTION_HORIZON,
            &rti_config_, &rti_memory_, &result);
        const auto solve_finish = std::chrono::steady_clock::now();
        const double solve_us = std::chrono::duration<double, std::micro>(
            solve_finish - solve_start).count();
        if (status != MPC_RTI_CYCLE_ACCEPTED_OPTIMAL &&
            status != MPC_RTI_CYCLE_ACCEPTED_DEGRADED) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC RTI rejected: %s iterations=%d residual=(%.6g,%.6g) "
                "regularization=(%.6g,%d) nonsmooth_columns=%d "
                "nonlinear_failure_stage=%d",
                cycle_status_name(status), result.solver_iterations,
                result.primal_residual, result.dual_residual,
                result.maximum_regularization, result.regularization_count,
                result.nonsmooth_jacobian_columns,
                result.nonlinear_failure_stage);
            if (diagnostics_pub_)
                publish_shadow_result(state, coherent_state,
                    control_time_prediction, control_prediction_us,
                    last_projected_s_, source_dt, result, solve_us,
                    control_ros_time.nanoseconds(), callback_steady_ns,
                    synchronize_steady_ns);
            publish_driving_fallback(
                "MPC RTI cycle rejected its candidate",
                command_time_state.u, commanded_speed, false,
                &command_time_projection);
            return;
        }

        if (diagnostics_pub_)
            publish_shadow_result(state, coherent_state,
                control_time_prediction, control_prediction_us,
                last_projected_s_, source_dt, result, solve_us,
                control_ros_time.nanoseconds(), callback_steady_ns,
                synchronize_steady_ns);
        if (enabled_) {
            accepted_command_established_ = true;
            target_speed_mps_ = result.published_target_speed;
            last_steering_command_rad_ = result.published_steering_command;
            last_steering_rate_radps_ = result.first_control.steering_rate;
            last_target_speed_rate_mps2_ = result.first_control.target_speed_rate;
            publish_command(last_steering_command_rad_, target_speed_mps_);
        }
    }

    bool enabled_{};
    bool shadow_mode_{};
    bool trajectory_loaded_{};
    bool startup_path_validated_{};
    bool progress_initialized_{};
    bool target_speed_initialized_{};
    bool accepted_command_established_{};
    bool localization_ready_{};
    std::string odom_topic_;
    std::string pose_topic_;
    std::string command_topic_;
    std::string diagnostics_topic_;
    std::string path_frame_;
    std::string command_frame_;
    std::string trajectory_file_;
    std::string control_time_mode_name_;
    MpcControlTimeMode control_time_mode_{
        MpcControlTimeMode::kAcceptedModelCommandHistory};
    MpcControlTimePredictorConfig control_time_predictor_config_{};
    double max_speed_mps_{};
    double startup_speed_mps_{};
    double startup_ramp_laps_{};
    double state_extrapolation_max_s_{};
    double pose_odom_max_skew_s_{};
    double source_dt_min_s_{};
    double source_dt_max_s_{};
    double startup_path_max_distance_m_{};
    double startup_path_heading_tolerance_rad_{};
    double localization_covariance_xy_max_{};
    double localization_covariance_yaw_max_{};
    int localization_required_updates_{5};
    int localization_good_updates_{};
    double projection_search_distance_m_{};
    double track_length_m_{};
    double startup_progress_m_{};
    double last_projected_s_{};
    double target_speed_mps_{};
    double last_steering_command_rad_{};
    double last_steering_rate_radps_{};
    double last_target_speed_rate_mps2_{};
    double observed_target_speed_mps_{};
    double observed_steering_command_rad_{};
    double observed_steering_rate_radps_{};
    double observed_target_speed_rate_mps2_{};
    int64_t observed_command_stamp_ns_{};
    std::size_t last_closest_index_{};
    rclcpp::Time last_source_stamp_{0, 0, RCL_ROS_TIME};
    MpcRtiCycleConfiguration_t rti_config_{};
    MpcRtiMemory_t rti_memory_{};
    std::vector<Waypoint> trajectory_;
    MpcStateSynchronizer state_synchronizer_;
    MpcCommandHistory command_history_;
    std::mutex state_mutex_;
    std::mutex command_history_mutex_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr command_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
        observed_command_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr diagnostics_pub_;
};

}  // namespace f1tenth_mpc

RCLCPP_COMPONENTS_REGISTER_NODE(f1tenth_mpc::MpcControllerNode)

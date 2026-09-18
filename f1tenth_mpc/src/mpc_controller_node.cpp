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
            "diagnostics_topic", "/mpc_shadow/diagnostics");
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

        if (enabled_) {
            command_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
                command_topic_, rclcpp::QoS(10));
        }
        if (shadow_mode_) {
            diagnostics_pub_ = create_publisher<std_msgs::msg::String>(
                diagnostics_topic_, rclcpp::QoS(rclcpp::KeepLast(2)));
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
            declare_parameter<double>("corridor_margin_m", 0.05));
        rti_config_.model.corridor_preview_halfwidth_m = static_cast<float>(
            declare_parameter<double>("corridor_preview_halfwidth_m", 0.10));
        rti_config_.model.nonlinear_corridor_tolerance_m = static_cast<float>(
            declare_parameter<double>("nonlinear_corridor_tolerance_m", 0.001));
        const int configured_iterations = declare_parameter<int>(
            "max_solver_iterations", 100);
        rti_config_.solver.max_iterations = static_cast<uint16_t>(
            std::clamp(configured_iterations, 1, 1000));
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
            if (values.size() >= 9) {
                point.left_bound = values[7];
                point.right_bound = values[8];
            }
            if (!std::isfinite(point.s) || !std::isfinite(point.x) ||
                !std::isfinite(point.y) || !std::isfinite(point.heading) ||
                !std::isfinite(point.curvature) || !std::isfinite(point.speed)) {
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
        if (shadow_mode_ && emit_shadow_diagnostic)
            publish_shadow_failure(reason);
    }

    void command_callback(
        const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr message)
    {
        const int64_t stamp_ns = rclcpp::Time(
            message->header.stamp, get_clock()->get_clock_type()).nanoseconds();
        const double speed = message->drive.speed;
        const double steering = message->drive.steering_angle;
        if (stamp_ns <= 0 || !std::isfinite(speed) || speed < 0.0 ||
            speed > max_speed_mps_ || !std::isfinite(steering) ||
            std::abs(steering) > rti_config_.model.max_steering_rad) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC shadow ignored invalid/out-of-envelope command history");
            return;
        }
        std::lock_guard<std::mutex> lock(command_history_mutex_);
        if (observed_command_stamp_ns_ > 0) {
            const int64_t delta_ns = stamp_ns - observed_command_stamp_ns_;
            if (delta_ns <= 0) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                    "MPC shadow rejected out-of-order /cmd/speed history");
                return;
            }
            const double dt = static_cast<double>(delta_ns) * 1.0e-9;
            observed_steering_rate_radps_ =
                (steering - observed_steering_command_rad_) / dt;
            observed_target_speed_rate_mps2_ =
                (speed - observed_target_speed_mps_) / dt;
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
        json << std::setprecision(9)
            << "{\"status\":\"" << cycle_status_name(result.status)
            << "\",\"source_stamp_ns\":" << synchronized.source_stamp_ns
            << ",\"control_ros_stamp_ns\":" << control_ros_stamp_ns
            << ",\"source_age_s\":" << synchronized.source_age_s
            << ",\"source_dt_s\":" << source_dt_s
            << ",\"pose_odom_skew_s\":" << synchronized.pose_odom_skew_s
            << ",\"control_time_prediction\":{\"mode\":\""
            << control_time_mode_name_ << "\",\"age_s\":"
            << control_prediction.age_s << ",\"elapsed_us\":"
            << control_prediction_us << ",\"command_changes_used\":"
            << control_prediction.command_changes_used
            << ",\"command_fallback\":"
            << (control_prediction.used_command_fallback ? "true" : "false")
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
            << ",\"state\":[" << state.plant.e_y << ','
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
            << ",\"solve_us\":" << solve_us << '}'
            << ",\"first_action\":["
            << result.published_steering_command << ','
            << result.published_target_speed << ','
            << result.first_control.steering_rate << ','
            << result.first_control.target_speed_rate << ']'
            << ",\"predictions\":[";

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
                << feedforward << ']'
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
        if (!std::isfinite(message->pose.pose.position.x) ||
            !std::isfinite(message->pose.pose.position.y) || !std::isfinite(yaw)) {
            return;
        }
        const rclcpp::Time stamp(
            message->header.stamp, get_clock()->get_clock_type());
        std::lock_guard<std::mutex> lock(state_mutex_);
        const auto status = state_synchronizer_.set_map_pose({
            stamp.nanoseconds(), message->pose.pose.position.x,
            message->pose.pose.position.y, yaw});
        if (status != MpcSyncStatus::kOk) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC map-pose handoff rejected: %s",
                MpcStateSynchronizer::status_name(status));
        }
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr message)
    {
        const int64_t callback_steady_ns = steady_time_ns();
        const auto & odom_pose = message->pose.pose;
        const auto & q = odom_pose.orientation;
        const double odom_yaw = std::atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        const rclcpp::Time odom_stamp(
            message->header.stamp, get_clock()->get_clock_type());
        if (odom_stamp.nanoseconds() <= 0 ||
            !std::isfinite(odom_pose.position.x) ||
            !std::isfinite(odom_pose.position.y) || !std::isfinite(odom_yaw) ||
            !std::isfinite(message->twist.twist.linear.x) ||
            !std::isfinite(message->twist.twist.linear.y) ||
            !std::isfinite(message->twist.twist.angular.z)) {
            publish_stop("MPC rejected zero-stamped or invalid odometry");
            return;
        }

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

        if (odom_status == MpcSyncStatus::kTimestampOrderFault ||
            odom_status == MpcSyncStatus::kSourceGapFault) {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                state_synchronizer_.reset();
            }
            mpc_rti_memory_reset(&rti_memory_);
            target_speed_initialized_ = false;
            last_source_stamp_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
            publish_stop(odom_status == MpcSyncStatus::kTimestampOrderFault ?
                "MPC odometry source ordering fault; synchronizer reset" :
                "MPC odometry source gap fault; synchronizer reset");
            return;
        }
        if (odom_status != MpcSyncStatus::kOk) {
            publish_stop("MPC rejected invalid legal odometry sample");
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
                "MPC source-time handoff rejected state: %s",
                MpcStateSynchronizer::status_name(sync_status));
            publish_stop("MPC coherent control-time state unavailable");
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
        if (!mpc_trajectory_project(trajectory_.data(), trajectory_.size(),
                track_length_m_, coherent_state.map_x, coherent_state.map_y,
                coherent_state.map_yaw, previous_segment, 160, &projection)) {
            publish_stop("MPC could not project pose onto raceline");
            return;
        }
        if (!startup_path_validated_) {
            if (projection.distance > startup_path_max_distance_m_ ||
                std::abs(projection.heading_error) > startup_path_heading_tolerance_rad_) {
                publish_stop("MPC startup path gate rejected pose");
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
            publish_stop("MPC command-time state prediction failed");
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
            if (shadow_mode_)
                publish_shadow_result(state, coherent_state,
                    control_time_prediction, control_prediction_us,
                    last_projected_s_, source_dt, result, solve_us,
                    control_ros_time.nanoseconds(), callback_steady_ns,
                    synchronize_steady_ns);
            publish_stop("MPC RTI cycle rejected its candidate",
                         !shadow_mode_);
            return;
        }

        if (enabled_) {
            target_speed_mps_ = result.published_target_speed;
            last_steering_command_rad_ = result.published_steering_command;
            last_steering_rate_radps_ = result.first_control.steering_rate;
            last_target_speed_rate_mps2_ = result.first_control.target_speed_rate;
            publish_command(last_steering_command_rad_, target_speed_mps_);
        } else {
            publish_shadow_result(state, coherent_state, control_time_prediction,
                control_prediction_us, last_projected_s_, source_dt, result,
                solve_us, control_ros_time.nanoseconds(), callback_steady_ns,
                synchronize_steady_ns);
        }
    }

    bool enabled_{};
    bool shadow_mode_{};
    bool trajectory_loaded_{};
    bool startup_path_validated_{};
    bool progress_initialized_{};
    bool target_speed_initialized_{};
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

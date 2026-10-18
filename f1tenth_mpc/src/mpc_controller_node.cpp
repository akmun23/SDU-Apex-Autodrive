// The MPC consumes synchronized localization and permitted sensor inputs,
// then directly publishes steering and normalized throttle.

#include "mpc_rti.h"
#include "mpc_control_time_predictor.hpp"
#include "mpc_state_synchronizer.hpp"
#include "unity_source_plant.hpp"
#include "vehicle_model.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sdu_apex_msgs/msg/drive_command.hpp>
#include <sdu_apex_msgs/msg/unity_plant_state.hpp>
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

constexpr std::uint32_t kUnityPlantStateSchemaVersion = 7;
constexpr double kStickyForwardTimerThresholdS = 1.0;
constexpr double kStickySideTimerThresholdS = 1.0;
constexpr double kStickyTimerStepS = 0.001;

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

void configure_solver_state_scaling(RiccatiAdmmConfig_t & solver, bool enabled)
{
    solver.use_scaling = enabled ? 1 : 0;
    if (!enabled) return;

    // x = D*x_scaled and u = Du*u_scaled. These fixed characteristic scales
    // normalize the mixed SI units without changing physical costs, bounds,
    // plant equations, or actuator commands.
    constexpr float state_scale[RICCATI_MAX_NX] = {
        0.10f, 0.10f, 4.0f, 1.0f, 4.0f,
        0.5f,
        400.0f, 400.0f, 400.0f, 400.0f,
        0.10f, 1.0f, 0.25f, 4.0f, 0.25f, 4.0f,
        0.05f, 0.05f, 0.05f, 0.05f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        0.5f, 1.0f};
    constexpr float input_scale[MPC_MODEL_NU] = {0.5f, 1.0f};
    std::copy_n(state_scale, RICCATI_MAX_NX, solver.state_scale);
    std::copy_n(input_scale, MPC_MODEL_NU, solver.input_scale);
}

}  // namespace

class MpcControllerNode final : public rclcpp::Node {
public:
    explicit MpcControllerNode(const rclcpp::NodeOptions & options)
        : Node("mpc_controller_node", options)
    {
        enabled_ = declare_parameter<bool>("enabled", false);
        odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom");
        pose_topic_ = declare_parameter<std::string>(
            "pose_topic", "/current_map_pose");
        drive_command_topic_ = declare_parameter<std::string>(
            "drive_command_topic", "/mpc/drive_command");
        diagnostics_topic_ = declare_parameter<std::string>(
            "diagnostics_topic", "/mpc/diagnostics");
        const bool publish_diagnostics = declare_parameter<bool>(
            "publish_diagnostics", false);
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
        const double command_actuation_delay_s = std::clamp(
            declare_parameter<double>("command_actuation_delay_s", 0.0),
            0.0, 1.0);
        unity_plant_state_topic_ = declare_parameter<std::string>(
            "unity_plant_state_topic", "/odom/unity_plant_state");
        control_time_predictor_config_.maximum_state_age_s =
            state_extrapolation_max_s_;
        control_time_predictor_config_.model_integration_step_s =
            TIME_STEP_SECONDS;
        control_time_predictor_config_.command_actuation_delay_s =
            command_actuation_delay_s;
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
            drive_command_pub_ = create_publisher<sdu_apex_msgs::msg::DriveCommand>(
                drive_command_topic_, rclcpp::QoS(rclcpp::KeepLast(1)));
        }
        if (enabled_ && publish_diagnostics) {
            diagnostics_pub_ = create_publisher<std_msgs::msg::String>(
                diagnostics_topic_, rclcpp::QoS(rclcpp::KeepLast(2)));
        }
        // State is time-sensitive: if executor/DDS scheduling falls behind,
        // drop old samples instead of replaying them as current control input.
        const auto latest_state_qos = rclcpp::SensorDataQoS().keep_last(1);
        pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            pose_topic_, latest_state_qos,
            std::bind(&MpcControllerNode::pose_callback, this, std::placeholders::_1));
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, latest_state_qos,
            std::bind(&MpcControllerNode::odom_callback, this, std::placeholders::_1));
        unity_plant_state_sub_ =
            create_subscription<sdu_apex_msgs::msg::UnityPlantState>(
                unity_plant_state_topic_, latest_state_qos,
                std::bind(&MpcControllerNode::unity_plant_state_callback, this,
                          std::placeholders::_1));
        auto declare_weight = [this](const char * name, float fallback) {
            return static_cast<float>(declare_parameter<double>(name, fallback));
        };
                /* Keep parameter fallbacks identical to the canonical competition YAML.
         * Otherwise a missing/renamed YAML silently selected the historical
         * BachelorProject objective while the offline replay and documented
         * production profile used different weights. */
        rti_config_.model.weight_e_y = declare_weight("weight_e_y", 125.0f);
        rti_config_.model.weight_e_psi = declare_weight("weight_e_psi", 2.0f);
        rti_config_.model.weight_u = declare_weight("weight_u", 200.0f);
        rti_config_.model.weight_u_overspeed = declare_weight(
            "weight_u_overspeed", 200.0f);
        rti_config_.model.weight_longitudinal_slip = declare_weight(
            "weight_longitudinal_slip", 50.0f);
        rti_config_.model.weight_v = declare_weight("weight_v", 5.0f);
        rti_config_.model.weight_r = declare_weight("weight_r", 1.5f);
        rti_config_.model.weight_steering_command = declare_weight(
            "weight_steering_command", 3.0f);
        rti_config_.model.weight_steering_target = declare_weight(
            "weight_steering_target", 5.0f);
        rti_config_.model.weight_throttle_command = declare_weight(
            "weight_throttle_command", 0.5f);
        rti_config_.model.weight_steering_target_change = declare_weight(
            "weight_steering_target_change", 10.0f);
        rti_config_.model.weight_throttle_command_change = declare_weight(
            "weight_throttle_command_change", 5.0f);
        rti_config_.model.terminal_multiplier = declare_weight(
            "terminal_multiplier", 3.0f);
        rti_config_.model.max_speed_mps = static_cast<float>(max_speed_mps_);
        rti_config_.model.active_speed_ceiling_mps =
            static_cast<float>(max_speed_mps_);
        actuator_steering_limit_rad_ = declare_parameter<double>(
            "max_steering_rad", SOURCE_MAX_STEERING_RAD);
        rti_config_.model.max_steering_rad = static_cast<float>(
            actuator_steering_limit_rad_);
        rti_config_.model.corridor_margin_m = static_cast<float>(
            declare_parameter<double>("corridor_margin_m",
                                      MPC_REQUIRED_WALL_CLEARANCE_M));
        rti_config_.model.first_prediction_corridor_margin_m =
            static_cast<float>(declare_parameter<double>(
                "first_prediction_corridor_margin_m",
                rti_config_.model.corridor_margin_m));
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
        rti_config_.model.jacobian_threads = static_cast<int>(std::clamp<long>(
            declare_parameter<int>("jacobian_threads", 4), 0L, 16L));
        const int configured_iterations = declare_parameter<int>(
            "max_solver_iterations", 160);
        rti_config_.solver.max_iterations = static_cast<uint16_t>(
            std::clamp(configured_iterations, 1, 160));
        const double rti_cycle_budget_ms = declare_parameter<double>(
            "rti_cycle_budget_ms", 20.0);
        if (!std::isfinite(rti_cycle_budget_ms) || rti_cycle_budget_ms < 0.0) {
            throw std::runtime_error(
                "rti_cycle_budget_ms must be finite and nonnegative");
        }
        rti_config_.max_cycle_time_us = rti_cycle_budget_ms * 1000.0;
        rti_config_.solver.rho = static_cast<float>(
            declare_parameter<double>("admm_rho", 7.0));
        rti_config_.solver.rho_u = static_cast<float>(
            declare_parameter<double>("admm_rho_input", 7.0));
        rti_config_.solver.tolerance = static_cast<float>(
            declare_parameter<double>("solver_tolerance", 0.01));
        rti_config_.solver.adaptive_rho = declare_parameter<bool>(
            "adaptive_rho", true) ? 1 : 0;
        const int adaptive_rho_update_period = static_cast<int>(
            declare_parameter<int>("adaptive_rho_update_period", 1));
        rti_config_.solver.adaptive_rho_update_period = std::clamp(
            adaptive_rho_update_period, 1, 100);
        rti_config_.solver.shared_rho = 0;
        rti_config_.solver.over_relaxation = static_cast<float>(
            declare_parameter<double>("admm_over_relaxation", 1.6));
        configure_solver_state_scaling(rti_config_.solver,
            declare_parameter<bool>("solver_state_scaling_enabled", true));
        const std::string refinement_mode = declare_parameter<std::string>(
            "rti_refinement_mode", "adaptive");
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
        rti_config_.rti2_steering_target_correction_trigger_rad =
            static_cast<float>(declare_parameter<double>(
                "rti2_steering_target_correction_trigger_rad", 0.50));
        rti_config_.rti2_throttle_correction_trigger =
            static_cast<float>(declare_parameter<double>(
                "rti2_throttle_correction_trigger", 0.05));
        rti_config_.rti2_residual_imbalance_trigger = static_cast<float>(
            declare_parameter<double>("rti2_residual_imbalance_trigger", 8.0));
        rti_config_.rti2_lateral_load_trigger_mps2 = static_cast<float>(
            declare_parameter<double>("rti2_lateral_load_trigger_mps2", 3.0));
        rti_config_.rti2_nonsmooth_columns_trigger = declare_parameter<int>(
            "rti2_nonsmooth_columns_trigger", 50);
        rti_config_.rti2_residual_recovery_limit = static_cast<float>(
            declare_parameter<double>("rti2_residual_recovery_limit", 0.25));
        rti_config_.degraded_residual_limit = static_cast<float>(
            declare_parameter<double>("solver_degraded_tolerance", 0.01));
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
        if (!enabled_) {
            RCLCPP_WARN(get_logger(),
                "MPC is inactive; no command publisher is created");
        }
    }

    ~MpcControllerNode() override
    {
        report_health("shutdown");
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

    void publish_command(double steering, double throttle_normalized = 0.0)
    {
        if (!drive_command_pub_) return;
        const double finite_steering = std::isfinite(steering) ? steering : 0.0;
        const double finite_throttle = std::isfinite(throttle_normalized)
            ? throttle_normalized : 0.0;
        const double bounded_steering = std::clamp(finite_steering,
            -actuator_steering_limit_rad_, actuator_steering_limit_rad_);
        const double bounded_throttle = clamp(finite_throttle, 0.0, 1.0);
        sdu_apex_msgs::msg::DriveCommand command;
        const rclcpp::Time command_time = now();
        command.header.stamp = command_time;
        command.header.frame_id = command_frame_;
        command.steering_angle_rad = bounded_steering;
        command.throttle_normalized = bounded_throttle;
        {
            std::lock_guard<std::mutex> lock(command_history_mutex_);
            command_history_.push({command_time.nanoseconds(), bounded_steering,
                bounded_throttle, true});
        }
        drive_command_pub_->publish(command);
    }

    void publish_stop(const char * reason, bool emit_diagnostic = true)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s", reason);
        mpc_rti_memory_reset(&rti_memory_);
        if (enabled_) {
            last_steering_command_rad_ = 0.0;
            last_throttle_command_ = 0.0;
            publish_command(0.0, 0.0);
        }
        if (diagnostics_pub_ && emit_diagnostic)
            publish_cycle_failure(reason);
    }

    /* A rejected MPC candidate is never published as if it were valid. A
     * direct-throttle controller has no speed-loop fallback: zero throttle is
     * the simulator's active-brake command. Preserve bounded steering only. */
    void publish_driving_fallback(const char *reason,
                                  bool emit_diagnostic = true,
                                  bool reset_rti_memory = true)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s; "
            "holding bounded driving command while localization recovers", reason);
        if (reset_rti_memory) {
            mpc_rti_memory_reset(&rti_memory_);
        }
        if (enabled_) {
            last_steering_command_rad_ = clamp(
                std::isfinite(last_steering_command_rad_)
                    ? last_steering_command_rad_ : 0.0,
                -rti_config_.model.max_steering_rad,
                rti_config_.model.max_steering_rad);
            last_throttle_command_ = 0.0;
            publish_command(last_steering_command_rad_, 0.0);
        }
        if (diagnostics_pub_ && emit_diagnostic)
            publish_cycle_failure(reason);
    }

    void publish_cycle_failure(const char * reason)
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

    void record_cycle_health(MpcRtiCycleStatus_t status,
                             const MpcRtiCycleResult_t & result,
                             double solve_us)
    {
        ++control_cycles_;
        if (status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL) {
            ++accepted_optimal_;
        } else if (status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED) {
            ++accepted_degraded_;
        }
        if (result.residual_candidate_published) ++residual_candidate_published_;
        if (result.corridor_repair_used) ++corridor_repair_used_;
        max_solver_iterations_seen_ = std::max(
            max_solver_iterations_seen_, result.solver_iterations);
        max_solve_us_ = std::max(max_solve_us_, solve_us);
        if (control_cycles_ % 400 == 0) report_health("periodic");
    }

    void report_health(const char * phase)
    {
        RCLCPP_INFO(get_logger(),
            "MPC health (%s): control_cycles=%llu accepted_optimal=%llu "
            "accepted_degraded=%llu residual_candidate_published=%llu "
            "corridor_repair_used=%llu "
            "hard_fallback_cycles=%llu max_solver_iterations_seen=%d "
            "max_solve_us=%.1f",
            phase,
            static_cast<unsigned long long>(control_cycles_),
            static_cast<unsigned long long>(accepted_optimal_),
            static_cast<unsigned long long>(accepted_degraded_),
            static_cast<unsigned long long>(residual_candidate_published_),
            static_cast<unsigned long long>(corridor_repair_used_),
            static_cast<unsigned long long>(hard_fallback_cycles_),
            max_solver_iterations_seen_, max_solve_us_);
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

    static const char * input_failure_reason_name(int reason)
    {
        switch (reason) {
        case MPC_RTI_INPUT_FAILURE_NONE: return "none";
        case MPC_RTI_INPUT_FAILURE_CYCLE_ARGUMENTS: return "cycle_arguments";
        case MPC_RTI_INPUT_FAILURE_PLANT_STATE: return "plant_state";
        case MPC_RTI_INPUT_FAILURE_NOMINAL_BUILD: return "nominal_build";
        case MPC_RTI_INPUT_FAILURE_CORRIDOR_SCHEDULE:
            return "corridor_schedule";
        }
        return "unknown";
    }

    static const char * nominal_build_failure_stage_name(int stage)
    {
        switch (stage) {
        case MPC_RTI_NOMINAL_BUILD_OK: return "none";
        case MPC_RTI_NOMINAL_BUILD_INVALID_INPUT: return "invalid_input";
        case MPC_RTI_NOMINAL_BUILD_SPEED_CEILING: return "speed_ceiling";
        case MPC_RTI_NOMINAL_BUILD_SPEED_SEED: return "speed_seed";
        case MPC_RTI_NOMINAL_BUILD_SEED_REFERENCE: return "seed_reference";
        case MPC_RTI_NOMINAL_BUILD_SEED_STEP: return "seed_step";
        case MPC_RTI_NOMINAL_BUILD_PASS_A_REFERENCE: return "pass_a_reference";
        case MPC_RTI_NOMINAL_BUILD_PASS_A_ROLLOUT: return "pass_a_rollout";
        case MPC_RTI_NOMINAL_BUILD_PASS_B_REFERENCE: return "pass_b_reference";
        case MPC_RTI_NOMINAL_BUILD_SOURCE_SPEED_ENVELOPE:
            return "source_speed_envelope";
        case MPC_RTI_NOMINAL_BUILD_THROTTLE_MODE: return "throttle_mode";
        case MPC_RTI_NOMINAL_BUILD_FINAL_ROLLOUT: return "final_rollout";
        case MPC_RTI_NOMINAL_BUILD_COMMAND_ENVELOPE:
            return "command_envelope";
        }
        return "unknown";
    }

    static const char * command_envelope_failure_name(int failure)
    {
        switch (failure) {
        case MPC_RTI_COMMAND_ENVELOPE_OK: return "none";
        case MPC_RTI_COMMAND_ENVELOPE_NONFINITE_STATE: return "nonfinite_state";
        case MPC_RTI_COMMAND_ENVELOPE_SPEED_BELOW_ZERO:
            return "speed_below_zero";
        case MPC_RTI_COMMAND_ENVELOPE_SPEED_ABOVE_MAX:
            return "speed_above_max";
        case MPC_RTI_COMMAND_ENVELOPE_STEERING_QUEUE:
            return "steering_queue";
        case MPC_RTI_COMMAND_ENVELOPE_ACTUAL_STEERING:
            return "actual_steering";
        case MPC_RTI_COMMAND_ENVELOPE_THROTTLE_QUEUE:
            return "throttle_queue";
        }
        return "unknown";
    }

    void publish_cycle_diagnostics(const MpcRtiState_t &state,
        const MpcSynchronizedState &synchronized,
        const MpcControlTimePrediction &control_prediction,
        double control_prediction_us, double progress,
        double source_dt_s, const MpcRtiCycleResult_t &result,
        double solve_us, int64_t control_ros_stamp_ns,
        int64_t callback_steady_ns, int64_t synchronize_steady_ns)
    {
        if (!diagnostics_pub_) return;
        const int steps[] = {1, 5, 10, 20, 30};
        std::ostringstream json;
        const auto json_number = [&json](double value) {
            if (std::isfinite(value)) json << value;
            else json << "null";
        };
        const auto append_rollout_failure =
            [&json, &json_number](const char *name,
                const MpcRtiRolloutFailure_t &failure) {
            json << ",\"" << name << "\":{\"valid\":"
                 << (failure.valid ? "true" : "false")
                 << ",\"stage\":" << failure.stage
                 << ",\"progress_m\":";
            json_number(failure.progress);
            json << ",\"state\":[";
            json_number(failure.state.plant.e_y);
            json << ',';
            json_number(failure.state.plant.e_psi);
            json << ',';
            json_number(failure.state.plant.u);
            json << ',';
            json_number(failure.state.plant.v);
            json << ',';
            json_number(failure.state.plant.r);
            json << ',';
            json_number(failure.state.plant.delayed_steering_command_2);
            json << ',';
            json_number(failure.state.plant.actual_steering_angle);
            json << "],\"reference\":[";
            json_number(failure.reference.e_y);
            json << ',';
            json_number(failure.reference.e_psi);
            json << ',';
            json_number(failure.reference.path_curvature);
            json << ',';
            json_number(failure.reference.left_bound);
            json << ',';
            json_number(failure.reference.right_bound);
            json << "],\"margin_m\":";
            json_number(failure.margin_m);
            json << ",\"lower_bound_m\":";
            json_number(failure.lower_bound_m);
            json << ",\"upper_bound_m\":";
            json_number(failure.upper_bound_m);
            json << '}';
        };
        MpcTrajectorySample_t current_path_sample{};
        const bool current_path_sample_valid = mpc_trajectory_sample(
            trajectory_.data(), trajectory_.size(), track_length_m_, progress,
            &current_path_sample);
        json << std::setprecision(9)
            << "{\"status\":\"" << cycle_status_name(result.status)
            << "\",\"input_failure_reason\":\""
            << input_failure_reason_name(result.input_failure_reason)
            << "\",\"nominal_build_failure_stage\":\""
            << nominal_build_failure_stage_name(
                result.nominal_build_failure_stage)
            << "\",\"nominal_build_failure_constraint\":\""
            << command_envelope_failure_name(
                result.nominal_build_failure_constraint)
            << "\",\"nominal_build_failure_index\":"
            << result.nominal_build_failure_index
            << ",\"nonlinear_failure_stage\":"
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
                result.r2_nonlinear_failure_reason) << '"';
        append_rollout_failure("r1_nonlinear_failure",
            result.r1_nonlinear_failure);
        append_rollout_failure("r2_nonlinear_failure",
            result.r2_nonlinear_failure);
        json
            << ",\"source_stamp_ns\":" << synchronized.source_stamp_ns
            << ",\"odom_source_stamp_ns\":"
            << synchronized.odom_source_stamp_ns
            << ",\"map_pose_source_stamp_ns\":"
            << synchronized.map_pose_source_stamp_ns
            << ",\"control_ros_stamp_ns\":" << control_ros_stamp_ns
            << ",\"source_age_s\":" << synchronized.source_age_s
            << ",\"source_dt_s\":" << source_dt_s
            << ",\"pose_odom_skew_s\":" << synchronized.pose_odom_skew_s
            << ",\"control_time_prediction\":{\"mode\":\"unity_plant\","
            << "\"age_s\":"
            << control_prediction.age_s << ",\"elapsed_us\":"
            << control_prediction_us << ",\"command_changes_used\":"
            << control_prediction.command_changes_used
            << ",\"plant_source_stamp_ns\":"
            << control_prediction.plant_source_stamp_ns
            << ",\"plant_handoff_stamp_ns\":"
            << control_prediction.plant_handoff_stamp_ns
            << ",\"plant_handoff_progress_m\":"
            << control_prediction.plant_handoff_progress_m
            << ",\"plant_to_sensor_s\":"
            << control_prediction.plant_to_sensor_s
            << ",\"sensor_state_fused\":"
            << (control_prediction.sensor_state_fused ? "true" : "false")
            << ",\"command_actuation_delay_s\":"
            << control_time_predictor_config_.command_actuation_delay_s
            << ",\"command_event_stamps_ns\":[";
        for (std::size_t i = 0;
             i < control_prediction.command_event_stamp_count; ++i) {
            if (i > 0) json << ',';
            json << control_prediction.command_event_stamps_ns[i];
        }
        json << "]"
            << ",\"plant_state_propagated\":"
            << (control_prediction.plant_state_valid ? "true" : "false")
            << ",\"odom_hidden_state_used\":"
            << (unity_plant_state_used_ ? "true" : "false")
            << ",\"source_map_pose\":[" << synchronized.map_x << ','
            << synchronized.map_y << ',' << synchronized.map_yaw
            << "],\"predicted_map_pose\":["
            << control_prediction.state.map_x << ','
            << control_prediction.state.map_y << ','
            << control_prediction.state.map_yaw << ']'
            << ",\"predicted_speed_state\":["
            << control_prediction.state.u << ',' << control_prediction.state.v
            << ',' << control_prediction.state.yaw_rate << ']'
            << "}"
            << ",\"host_timing_ns\":{" << "\"callback\":"
            << callback_steady_ns << ",\"synchronize\":"
            << synchronize_steady_ns << '}'
            << ",\"progress_m\":" << progress
            << ",\"path_curvature_per_m\":";
        json_number(current_path_sample_valid ? current_path_sample.curvature :
            std::numeric_limits<double>::quiet_NaN());
        json << ",\"reference\":{\"trajectory_acceleration_mps2\":";
        json_number(current_path_sample_valid ? current_path_sample.acceleration :
            std::numeric_limits<double>::quiet_NaN());
        json << ",\"acceleration_reference_mps2\":";
        json_number(current_path_sample_valid
            ? mpc_reference_acceleration_under_speed_ceiling(
                static_cast<float>(current_path_sample.speed),
                static_cast<float>(current_path_sample.acceleration),
                rti_config_.model.active_speed_ceiling_mps)
            : std::numeric_limits<double>::quiet_NaN());
        const double current_reference_speed = current_path_sample_valid
            ? std::min(current_path_sample.speed,
                static_cast<double>(rti_config_.model.active_speed_ceiling_mps))
            : std::numeric_limits<double>::quiet_NaN();
        json << ",\"speed_reference_mps\":";
        json_number(current_reference_speed);
        json << "}";
        json << ",\"state\":[" << state.plant.e_y << ','
            << state.plant.e_psi << ',' << state.plant.u << ','
            << state.plant.v << ',' << state.plant.r << ','
            << state.plant.delayed_steering_command_1
            << ',' << state.plant.delayed_steering_command_2
            << ',' << state.plant.actual_steering_angle
            << ',' << state.plant.delayed_throttle_command_1
            << ',' << state.plant.delayed_throttle_command_2 << ']'
            << ",\"plant_state_source\":\"odometry_v7\"";
        json << ",\"solver\":{\"iterations\":" << result.solver_iterations
            << ",\"primal_residual\":";
        json_number(result.primal_residual);
        json << ",\"dual_residual\":";
        json_number(result.dual_residual);
        json << ",\"max_regularization\":";
        json_number(result.maximum_regularization);
        json
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
            << result.first_control.throttle_command << ']'
            << ",\"rti_iterations_used\":" << result.rti_iterations_used
            << ",\"rti2_triggered\":"
            << (result.rti2_triggered ? "true" : "false")
            << ",\"rti2_trigger_reason_mask\":"
            << result.rti2_trigger_reason_mask
            << ",\"rti2_budget_skipped\":"
            << (result.rti2_budget_skipped ? "true" : "false")
            << ",\"r1_status\":\""
            << cycle_status_name_code(result.r1_status) << '\"'
            << ",\"linearization_failure_stage\":"
            << result.linearization_failure_stage
            << ",\"linearization_failure_reason\":"
            << result.linearization_failure_reason
            << ",\"linearization_failure_index\":"
            << result.linearization_failure_index
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
             << result.r1_first_action.steering_target_rad << ','
             << result.r1_first_action.throttle_command << ']'
             << ",\"r2_first_action\":["
             << result.r2_first_action.steering_target_rad << ','
             << result.r2_first_action.throttle_command << ']'
             << ",\"r1_solver_iterations\":"
             << result.r1_solver_iterations
             << ",\"r2_solver_iterations\":"
             << result.r2_solver_iterations
             << ",\"r1_solve_us\":" << result.r1_solve_us
             << ",\"r2_solve_us\":" << result.r2_solve_us
             << ",\"r1_qp_build_us\":" << result.r1_qp_build_us
             << ",\"r1_admm_us\":" << result.r1_admm_us
             << ",\"r1_exact_rollout_us\":"
             << result.r1_exact_rollout_us
             << ",\"r1_nonlinear_validation_us\":"
             << result.r1_nonlinear_validation_us
             << ",\"r2_qp_build_us\":" << result.r2_qp_build_us
             << ",\"r2_admm_us\":" << result.r2_admm_us
             << ",\"r2_exact_rollout_us\":"
             << result.r2_exact_rollout_us
             << ",\"r2_nonlinear_validation_us\":"
             << result.r2_nonlinear_validation_us
             << ",\"total_rti_us\":" << result.total_rti_us
            << ",\"selected_candidate\":\""
             << (result.selected_candidate == 3 ? "corridor_repair" :
                 result.selected_candidate == 2 ? "R2" :
                 result.selected_candidate == 1 ? "R1" : "none") << '\"'
             << ",\"corridor_repair_used\":"
             << (result.corridor_repair_used ? "true" : "false")
             << ",\"corridor_repair_trials\":"
             << result.corridor_repair_trials
             << ",\"corridor_repair_us\":";
        json_number(result.corridor_repair_us);
        json
             << ",\"residual_candidate_published\":"
             << (result.residual_candidate_published ? "true" : "false")
            << ",\"corridor_repair_alpha\":";
        json_number(result.corridor_repair_alpha);
        json << ",\"nominal_vs_candidate_progress_error_max_m\":";
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
                publish_cycle_failure("prediction_reference_unavailable");
                return;
            }
            const double reference_speed = std::min(
                sample.speed,
                static_cast<double>(
                    rti_config_.model.active_speed_ceiling_mps));
            const double feedforward = clamp(
                vehicle_model_steering_for_curvature(
                    static_cast<float>(sample.curvature)),
                -rti_config_.model.max_steering_rad,
                rti_config_.model.max_steering_rad);
            const MpcModelControl_t &control =
                rti_memory_.nominal.controls[step - 1];
            MpcTrajectorySample_t transition_sample{};
            MpcStageResult_t diagnostic_transition{};
            bool have_transition_output =
                step <= rti_memory_.nominal.horizon &&
                mpc_trajectory_sample(trajectory_.data(), trajectory_.size(),
                    track_length_m_,
                    rti_memory_.nominal.progress[step - 1],
                    &transition_sample);
            if (have_transition_output) {
                diagnostic_transition = mpc_vehicle_model_step(
                    &rti_memory_.nominal.states[step - 1].plant,
                    &control, PREDICTION_DT_SECONDS,
                    static_cast<float>(transition_sample.curvature));
                have_transition_output = diagnostic_transition.valid != 0;
            }
            if (!have_transition_output) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                    "MPC diagnostic plant output could not be reconstructed");
                return;
            }
            json << "{\"n\":" << step << ",\"s_m\":" << predicted_s
                << ",\"state\":[" << predicted.plant.e_y << ','
                << predicted.plant.e_psi << ',' << predicted.plant.u << ','
                << predicted.plant.v << ',' << predicted.plant.r << ','
                << predicted.plant.delayed_steering_command_1 << ','
                << predicted.plant.delayed_steering_command_2 << ','
                << predicted.plant.actual_steering_angle << ']'
                << ",\"source_wheel_omega_radps\":["
                << predicted.plant.source_plant_internal_state[
                    MPC_PLANT_STATE_WHEEL_OMEGA_FL]
                << ',' << predicted.plant.source_plant_internal_state[
                    MPC_PLANT_STATE_WHEEL_OMEGA_FR]
                << ',' << predicted.plant.source_plant_internal_state[
                    MPC_PLANT_STATE_WHEEL_OMEGA_RL]
                << ',' << predicted.plant.source_plant_internal_state[
                    MPC_PLANT_STATE_WHEEL_OMEGA_RR]
                << "]"
                << ",\"source_longitudinal_slip\":["
                << diagnostic_transition.output.longitudinal_slip[0] << ','
                << diagnostic_transition.output.longitudinal_slip[1] << ','
                << diagnostic_transition.output.longitudinal_slip[2] << ','
                << diagnostic_transition.output.longitudinal_slip[3]
                << "]"
                << ",\"source_lateral_slip\":["
                << diagnostic_transition.output.lateral_slip[0] << ','
                << diagnostic_transition.output.lateral_slip[1] << ','
                << diagnostic_transition.output.lateral_slip[2] << ','
                << diagnostic_transition.output.lateral_slip[3]
                << "]"
                << ",\"source_normal_load_n\":["
                << diagnostic_transition.output.normal_load_n[0] << ','
                << diagnostic_transition.output.normal_load_n[1] << ','
                << diagnostic_transition.output.normal_load_n[2] << ','
                << diagnostic_transition.output.normal_load_n[3]
                << "]"
                << ",\"reference\":[0,0," << reference_speed
                << ",0," << sample.curvature * reference_speed << ','
                << feedforward << ','
                << mpc_reference_acceleration_under_speed_ceiling(
                    static_cast<float>(sample.speed),
                    static_cast<float>(sample.acceleration),
                    rti_config_.model.active_speed_ceiling_mps) << ']'
                << ",\"input\":[" << control.steering_target_rad << ','
                << control.throttle_command << ']'
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
        }
    }

    void unity_plant_state_callback(
        const sdu_apex_msgs::msg::UnityPlantState::SharedPtr message)
    {
        if (message->state_schema_version != kUnityPlantStateSchemaVersion ||
            message->state_origin != 1 ||
            message->header.stamp.sec < 0 ||
            (message->header.stamp.sec == 0 && message->header.stamp.nanosec == 0)) {
            return;
        }
        const auto finite_array = [](const auto & values) {
            return std::all_of(values.begin(), values.end(),
                [](double value) { return std::isfinite(value); });
        };
        if (!finite_array(message->wheel_omega_radps) ||
            !finite_array(message->wheel_rotation_angle_rad) ||
            !finite_array(message->suspension_jounce_m) ||
            !finite_array(message->low_forward_speed_time_s) ||
            !finite_array(message->low_side_speed_time_s) ||
            !finite_array(message->longitudinal_slip) ||
            !finite_array(message->lateral_slip) ||
            !finite_array(message->normal_load_n) ||
            !finite_array(message->suspension_force_n) ||
            !finite_array(message->suspension_jounce_rate_mps) ||
            !finite_array(message->tire_longitudinal_force_n) ||
            !finite_array(message->tire_lateral_force_n) ||
            !std::isfinite(message->longitudinal_acceleration_mps2) ||
            !std::isfinite(message->lateral_acceleration_mps2) ||
            !std::isfinite(message->yaw_acceleration_radps2) ||
            !std::isfinite(message->body_height_m) ||
            !std::isfinite(message->body_height_rate_mps) ||
            !std::isfinite(message->roll_rad) ||
            !std::isfinite(message->roll_rate_radps) ||
            !std::isfinite(message->pitch_rad) ||
            !std::isfinite(message->pitch_rate_radps) ||
            !std::isfinite(message->physical_steering_rad) ||
            !std::isfinite(message->longitudinal_velocity_mps) ||
            !std::isfinite(message->lateral_velocity_com_mps) ||
            !std::isfinite(message->lateral_velocity_rear_axle_mps) ||
            !std::isfinite(message->yaw_rate_radps) ||
            !std::isfinite(message->gyro_yaw_rate_radps) ||
            !std::isfinite(message->steering_command_normalized) ||
            !std::isfinite(message->throttle_command_normalized) ||
            !std::isfinite(message->delayed_steering_command_1_normalized) ||
            !std::isfinite(message->delayed_steering_command_2_normalized) ||
            !std::isfinite(message->delayed_throttle_command_1_normalized) ||
            !std::isfinite(message->delayed_throttle_command_2_normalized) ||
            !std::isfinite(message->steering_feedback_rad) ||
            !std::isfinite(message->throttle_feedback_normalized)) {
            return;
        }
        if (message->steering_command_normalized < -1.0 ||
            message->steering_command_normalized > 1.0 ||
            message->delayed_steering_command_1_normalized < -1.0 ||
            message->delayed_steering_command_1_normalized > 1.0 ||
            message->delayed_steering_command_2_normalized < -1.0 ||
            message->delayed_steering_command_2_normalized > 1.0 ||
            message->throttle_command_normalized < 0.0 ||
            message->throttle_command_normalized > 1.0 ||
            message->delayed_throttle_command_1_normalized < 0.0 ||
            message->delayed_throttle_command_1_normalized > 1.0 ||
            message->delayed_throttle_command_2_normalized < 0.0 ||
            message->delayed_throttle_command_2_normalized > 1.0 ||
            (message->actuator_feedback_valid &&
             (std::abs(message->steering_feedback_rad) >
                    rti_config_.model.max_steering_rad + 1.0e-3f ||
              message->throttle_feedback_normalized < 0.0 ||
              message->throttle_feedback_normalized > 1.0))) {
            return;
        }
        for (std::size_t wheel = 0; wheel < 4; ++wheel) {
            if (message->low_forward_speed_time_s[wheel] < 0.0 ||
                message->low_side_speed_time_s[wheel] < 0.0) {
                return;
            }
        }
        const int64_t stamp_ns = rclcpp::Time(
            message->header.stamp, get_clock()->get_clock_type()).nanoseconds();
        if (stamp_ns <= 0) return;
        std::lock_guard<std::mutex> lock(unity_plant_state_mutex_);
        if (unity_plant_state_received_ &&
            stamp_ns < unity_plant_state_stamp_ns_) {
            return;
        }
        latest_unity_plant_state_ = *message;
        unity_plant_state_stamp_ns_ = stamp_ns;
        unity_plant_state_received_ = true;
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
                "MPC received uncertain odometry");
            return;
        }
        if (odom_stamp.nanoseconds() <= 0) odom_stamp = now();

        if (!enabled_) return;
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
                "MPC rejected uncertain legal odometry sample");
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
            // Do not keep racing on stale odometry/localization. The direct
            // throttle fallback below commands active braking immediately.
            const std::string reason = std::string(
                "MPC rejected unsynchronized legal state: ") +
                MpcStateSynchronizer::status_name(sync_status);
            publish_driving_fallback(
                reason.c_str());
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
            publish_driving_fallback("MPC could not project uncertain pose");
            return;
        }
        if (!startup_path_validated_) {
            if (projection.distance > startup_path_max_distance_m_ ||
                std::abs(projection.heading_error) > startup_path_heading_tolerance_rad_) {
                publish_driving_fallback("MPC startup pose is not yet aligned");
                return;
            }
            startup_path_validated_ = true;
        }
        MpcCommandHistory command_history_snapshot;
        {
            std::lock_guard<std::mutex> lock(command_history_mutex_);
            command_history_snapshot = command_history_;
        }
        sdu_apex_msgs::msg::UnityPlantState unity_plant_snapshot;
        int64_t unity_plant_snapshot_stamp_ns = 0;
        bool unity_plant_snapshot_received = false;
        {
            std::lock_guard<std::mutex> lock(unity_plant_state_mutex_);
            unity_plant_snapshot_received = unity_plant_state_received_;
            if (unity_plant_snapshot_received) {
                unity_plant_snapshot = latest_unity_plant_state_;
                unity_plant_snapshot_stamp_ns = unity_plant_state_stamp_ns_;
            }
        }
        MpcControlTimePrediction control_time_prediction{};
        const int64_t source_plant_gap_ns = control_ros_time.nanoseconds() -
            source_plant_handoff_stamp_ns_;
        const bool source_plant_handoff_usable =
            source_plant_handoff_valid_ && source_plant_gap_ns >= 0 &&
            static_cast<double>(source_plant_gap_ns) * 1.0e-9 <=
                state_extrapolation_max_s_;
        const bool unity_plant_snapshot_usable =
            is_causal_plant_state_usable(
                unity_plant_snapshot_received,
                unity_plant_snapshot_stamp_ns,
                coherent_state.odom_source_stamp_ns,
                coherent_state.source_stamp_ns,
                state_extrapolation_max_s_);
        if (!unity_plant_snapshot_usable && !source_plant_handoff_usable) {
            publish_driving_fallback(
                "waiting for a synchronized complete odometry plant state");
            return;
        }
        const MpcSynchronizedState &prediction_source = coherent_state;
        const MpcPathProjection_t &prediction_source_projection = projection;
        MpcModelState_t source_time_plant{};
        int64_t source_plant_stamp_ns = coherent_state.source_stamp_ns;
        double source_plant_progress_m = projection.s;
        if (source_plant_handoff_usable) {
            /* Carry hidden wheel/suspension/actuator state from the previous
             * sensor epoch. The predictor advances it to this sensor epoch,
             * fuses legal observed motion there, snapshots it, then advances
             * a separate copy to the control instant. */
            source_time_plant = source_plant_handoff_state_;
            source_plant_stamp_ns = source_plant_handoff_stamp_ns_;
            source_plant_progress_m = source_plant_handoff_progress_m_;
        }
        if (!source_plant_handoff_usable) {
            source_time_plant = MpcModelState_t{};
            source_plant_stamp_ns = coherent_state.source_stamp_ns;
            source_plant_progress_m = projection.s;
        }
        if (unity_plant_snapshot_usable) {
            /* Odom owns the causal hidden-state estimate. Start the MPC plant
             * from that exact odometry epoch instead of independently
             * recreating suspension, body attitude, and wheel-spin state. The
             * source model remains an estimate; current legal odom/encoders/
             * IMU and actuator feedback are fused below at their timestamps. */
            auto * const plant_state =
                source_time_plant.source_plant_internal_state;
            std::copy(unity_plant_snapshot.wheel_omega_radps.begin(),
                unity_plant_snapshot.wheel_omega_radps.end(),
                plant_state + MPC_PLANT_STATE_WHEEL_OMEGA_FL);
            plant_state[MPC_PLANT_STATE_BODY_HEIGHT] =
                unity_plant_snapshot.body_height_m;
            plant_state[MPC_PLANT_STATE_BODY_HEIGHT_RATE] =
                unity_plant_snapshot.body_height_rate_mps;
            plant_state[MPC_PLANT_STATE_ROLL] = unity_plant_snapshot.roll_rad;
            plant_state[MPC_PLANT_STATE_ROLL_RATE] =
                unity_plant_snapshot.roll_rate_radps;
            plant_state[MPC_PLANT_STATE_PITCH] = unity_plant_snapshot.pitch_rad;
            plant_state[MPC_PLANT_STATE_PITCH_RATE] =
                unity_plant_snapshot.pitch_rate_radps;
            std::copy(unity_plant_snapshot.suspension_jounce_m.begin(),
                unity_plant_snapshot.suspension_jounce_m.end(),
                plant_state + MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL);
            // PxVehicle only tests these timers against zero and one second.
            // Saturating at threshold + one source fixed step is therefore
            // an exact state reduction for future tire-constraint behavior.
            for (std::size_t wheel = 0; wheel < 4; ++wheel) {
                plant_state[MPC_PLANT_STATE_LOW_FORWARD_SPEED_TIME_FL + wheel] =
                    std::min(unity_plant_snapshot.low_forward_speed_time_s[wheel],
                        kStickyForwardTimerThresholdS + kStickyTimerStepS);
                plant_state[MPC_PLANT_STATE_LOW_SIDE_SPEED_TIME_FL + wheel] =
                    std::min(unity_plant_snapshot.low_side_speed_time_s[wheel],
                        kStickySideTimerThresholdS + kStickyTimerStepS);
                plant_state[MPC_PLANT_STATE_WHEEL_ROTATION_ANGLE_FL + wheel] =
                    unity_plant_snapshot.wheel_rotation_angle_rad[wheel];
            }
            source_time_plant.source_plant_initialized = 1;
            source_time_plant.source_suspension_history_initialized =
                unity_plant_snapshot.suspension_history_initialized ? 1 : 0;
            source_time_plant.source_imu_state_initialized = 1;
            source_time_plant.u = static_cast<float>(
                unity_plant_snapshot.longitudinal_velocity_mps);
            source_time_plant.r = static_cast<float>(
                unity_plant_snapshot.yaw_rate_radps);
            source_time_plant.v = static_cast<float>(
                unity_plant_snapshot.lateral_velocity_rear_axle_mps);
            source_time_plant.actual_steering_angle = static_cast<float>(
                unity_plant_snapshot.physical_steering_rad);
            source_time_plant.delayed_steering_command_1 = static_cast<float>(
                unity_steering_command_normalized_to_physical_rad(
                    unity_plant_snapshot.delayed_steering_command_1_normalized,
                    rti_config_.model.max_steering_rad));
            source_time_plant.delayed_steering_command_2 = static_cast<float>(
                unity_steering_command_normalized_to_physical_rad(
                    unity_plant_snapshot.delayed_steering_command_2_normalized,
                    rti_config_.model.max_steering_rad));
            source_time_plant.delayed_throttle_command_1 = static_cast<float>(
                unity_plant_snapshot.delayed_throttle_command_1_normalized);
            source_time_plant.delayed_throttle_command_2 = static_cast<float>(
                unity_plant_snapshot.delayed_throttle_command_2_normalized);
            source_time_plant.longitudinal_acceleration_mps2 = static_cast<float>(
                unity_plant_snapshot.longitudinal_acceleration_mps2);
            source_plant_stamp_ns = unity_plant_snapshot_stamp_ns;
            source_plant_progress_m = projection.s;
            unity_plant_state_used_ = true;
        } else {
            unity_plant_state_used_ = false;
        }
        const auto prediction_start = std::chrono::steady_clock::now();
        const MpcControlTimeStatus prediction_status = predict_to_control_time(
            MpcControlTimeMode::kAcceptedModelCommandHistory,
            prediction_source,
            control_ros_time.nanoseconds(),
            command_history_snapshot, trajectory_.data(), trajectory_.size(),
            track_length_m_, prediction_source_projection.segment,
            &prediction_source_projection,
            &source_time_plant, source_plant_stamp_ns,
            source_plant_progress_m,
            control_time_predictor_config_, &control_time_prediction);
        const auto prediction_finish = std::chrono::steady_clock::now();
        const double control_prediction_us =
            std::chrono::duration<double, std::micro>(
                prediction_finish - prediction_start).count();
        if (prediction_status != MpcControlTimeStatus::kOk) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC command-time prediction rejected state: %s",
                control_time_status_name(prediction_status));
            publish_driving_fallback("MPC command-time state prediction is uncertain");
            return;
        }
        if (control_time_prediction.plant_handoff_state_valid) {
            source_plant_handoff_state_ =
                control_time_prediction.plant_handoff_state;
            source_plant_handoff_stamp_ns_ =
                control_time_prediction.plant_handoff_stamp_ns;
            source_plant_handoff_progress_m_ =
                control_time_prediction.plant_handoff_progress_m;
            source_plant_handoff_valid_ = true;
        }
        const MpcSynchronizedState &command_time_state =
            control_time_prediction.state;
        const MpcPathProjection_t &command_time_projection =
            control_time_prediction.projection;
        update_progress(command_time_projection);
        const double speed_ceiling = active_speed_ceiling();
        rti_config_.model.max_speed_mps = static_cast<float>(max_speed_mps_);
        rti_config_.model.active_speed_ceiling_mps =
            static_cast<float>(speed_ceiling);

        MpcRtiState_t state{};
        if (control_time_prediction.plant_state_valid) {
            state.plant = control_time_prediction.plant_state;
        } else if (source_plant_handoff_usable) {
            state.plant = source_plant_handoff_state_;
        } else {
            state.plant = source_time_plant;
            state.plant.delayed_throttle_command_2 = 0.0f;
        }
        state.plant.e_y = static_cast<float>(command_time_projection.lateral_error);
        state.plant.e_psi = static_cast<float>(command_time_projection.heading_error);
        state.plant.u = static_cast<float>(std::max(0.0, command_time_state.u));
        state.plant.v = static_cast<float>(command_time_state.v);
        state.plant.r = static_cast<float>(command_time_state.yaw_rate);
        MpcRtiCycleResult_t result{};
        const auto solve_start = std::chrono::steady_clock::now();
        const MpcRtiCycleStatus_t status = mpc_rti_solve_cycle(
            &state, last_projected_s_, trajectory_.data(), trajectory_.size(),
            track_length_m_, TIME_STEP_SECONDS, PREDICTION_HORIZON,
            &rti_config_, &rti_memory_, &result);
        const auto solve_finish = std::chrono::steady_clock::now();
        const double solve_us = std::chrono::duration<double, std::micro>(
            solve_finish - solve_start).count();
        record_cycle_health(status, result, solve_us);
        if (result.corridor_repair_used) {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC exact corridor repair accepted: alpha=%.4f "
                "r1_status=%d r2_status=%d",
                result.corridor_repair_alpha, result.r1_status,
                result.r2_status);
        }
        if (status != MPC_RTI_CYCLE_ACCEPTED_OPTIMAL &&
            status != MPC_RTI_CYCLE_ACCEPTED_DEGRADED) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC RTI rejected: %s iterations=%d residual=(%.6g,%.6g) "
                "regularization=(%.6g,%d) nonsmooth_columns=%d "
                "rho=(%.4g->%.4g,%.4g->%.4g) rho_changes=%d "
                "nonlinear_failure=(reason=%d,stage=%d) "
                "r1=(status=%d,reason=%d,stage=%d) "
                "r2=(status=%d,reason=%d,stage=%d) "
                "rti2_triggered=%d state=(ey=%.3f,epsi=%.3f,u=%.3f,r=%.3f)",
                cycle_status_name(status), result.solver_iterations,
                result.primal_residual, result.dual_residual,
                result.maximum_regularization, result.regularization_count,
                result.nonsmooth_jacobian_columns,
                result.rho_start, result.rho_final,
                result.rho_u_start, result.rho_u_final,
                result.rho_change_count,
                result.nonlinear_failure_reason,
                result.nonlinear_failure_stage,
                result.r1_status, result.r1_nonlinear_failure_reason,
                result.r1_nonlinear_failure_stage,
                result.r2_status, result.r2_nonlinear_failure_reason,
                result.r2_nonlinear_failure_stage,
                result.rti2_triggered,
                state.plant.e_y, state.plant.e_psi, state.plant.u,
                state.plant.r);
            if (diagnostics_pub_)
                publish_cycle_diagnostics(state, coherent_state,
                    control_time_prediction, control_prediction_us,
                    last_projected_s_, source_dt, result, solve_us,
                    control_ros_time.nanoseconds(), callback_steady_ns,
                    synchronize_steady_ns);
            /* No exact-feasible candidate exists. Keep command continuity,
             * but apply the configured bounded speed reduction so a numerical
             * failure cannot preserve an unsafe racing speed. The RTI core
             * deliberately retains its ADMM iterate for residual-only
             * failures; preserve that warm start so the next 40 Hz solve can
             * continue convergence. Hard input, nonlinear, and solver errors
             * still reset it. */
            const bool reset_rti_memory =
                status != MPC_RTI_CYCLE_REJECTED_RESIDUAL;
            publish_driving_fallback(
                "MPC RTI cycle had no exact-feasible candidate",
                false, reset_rti_memory);
            ++hard_fallback_cycles_;
            return;
        }

        if (result.residual_candidate_published) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC published an exact-feasible residual-limited candidate "
                "(r1=%d,r2=%d,iterations=%d)", result.r1_status,
                result.r2_status, result.solver_iterations);
        }

        if (diagnostics_pub_)
            publish_cycle_diagnostics(state, coherent_state,
                control_time_prediction, control_prediction_us,
                last_projected_s_, source_dt, result, solve_us,
                control_ros_time.nanoseconds(), callback_steady_ns,
                synchronize_steady_ns);
        if (enabled_) {
            last_steering_command_rad_ = result.published_steering_command;
            last_throttle_command_ = result.published_throttle;
            publish_command(last_steering_command_rad_, result.published_throttle);
        }
    }

    bool enabled_{};
    bool trajectory_loaded_{};
    bool startup_path_validated_{};
    bool progress_initialized_{};
    std::string odom_topic_;
    std::string pose_topic_;
    std::string diagnostics_topic_;
    std::string path_frame_;
    std::string command_frame_;
    std::string trajectory_file_;
    std::string drive_command_topic_;
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
    double actuator_steering_limit_rad_{SOURCE_MAX_STEERING_RAD};
    double track_length_m_{};
    double startup_progress_m_{};
    double last_projected_s_{};
    double last_steering_command_rad_{};
    std::string unity_plant_state_topic_;
    bool source_plant_handoff_valid_{};
    bool unity_plant_state_used_{};
    int64_t source_plant_handoff_stamp_ns_{};
    double source_plant_handoff_progress_m_{};
    MpcModelState_t source_plant_handoff_state_{};
    double last_throttle_command_{};
    std::size_t last_closest_index_{};
    rclcpp::Time last_source_stamp_{0, 0, RCL_ROS_TIME};
    MpcRtiCycleConfiguration_t rti_config_{};
    MpcRtiMemory_t rti_memory_{};
    std::vector<Waypoint> trajectory_;
    MpcStateSynchronizer state_synchronizer_;
    MpcCommandHistory command_history_;
    std::mutex state_mutex_;
    std::mutex command_history_mutex_;
    std::mutex unity_plant_state_mutex_;
    sdu_apex_msgs::msg::UnityPlantState latest_unity_plant_state_{};
    int64_t unity_plant_state_stamp_ns_{};
    bool unity_plant_state_received_{};
    rclcpp::Publisher<sdu_apex_msgs::msg::DriveCommand>::SharedPtr
        drive_command_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sdu_apex_msgs::msg::UnityPlantState>::SharedPtr
        unity_plant_state_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr diagnostics_pub_;
    std::uint64_t control_cycles_{};
    std::uint64_t accepted_optimal_{};
    std::uint64_t accepted_degraded_{};
    std::uint64_t residual_candidate_published_{};
    std::uint64_t corridor_repair_used_{};
    std::uint64_t hard_fallback_cycles_{};
    int max_solver_iterations_seen_{};
    double max_solve_us_{};
};

}  // namespace f1tenth_mpc

RCLCPP_COMPONENTS_REGISTER_NODE(f1tenth_mpc::MpcControllerNode)

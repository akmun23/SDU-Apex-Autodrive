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

constexpr std::uint32_t kUnityPlantStateSchemaVersion = 11;
constexpr int64_t kBridgeControlPeriodNs = 25'000'000;
/* A fresh command takes effect after the two queued bridge packets. Validate
 * those two intervals plus the first interval affected by this command. */
constexpr int kCommandEffectValidationHorizon = 3;
/* Only a truly stationary, plan-less spawn may wait for the next solve tick
 * when cold Jacobian construction consumes the reserved ADMM budget. */
constexpr float kStationaryStartupRetrySpeedMps = 0.05f;
constexpr float kStationaryStartupRetryYawRateRadps = 0.05f;

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

bool next_bridge_request_epoch(int64_t source_request_stamp_ns,
                               int64_t predicted_publish_stamp_ns,
                               int64_t * request_epoch_stamp_ns)
{
    if (!request_epoch_stamp_ns || source_request_stamp_ns <= 0 ||
        predicted_publish_stamp_ns < source_request_stamp_ns) {
        return false;
    }
    const int64_t elapsed_ns =
        predicted_publish_stamp_ns - source_request_stamp_ns;
    const int64_t periods = elapsed_ns / kBridgeControlPeriodNs + 1;
    if (periods > (std::numeric_limits<int64_t>::max() -
            source_request_stamp_ns) / kBridgeControlPeriodNs) {
        return false;
    }
    *request_epoch_stamp_ns = source_request_stamp_ns +
        periods * kBridgeControlPeriodNs;
    return true;
}

void configure_solver_state_scaling(RiccatiAdmmConfig_t & solver, bool enabled)
{
    solver.use_scaling = enabled ? 1 : 0;
    if (!enabled) return;

    // x = D*x_scaled and u = Du*u_scaled. These are characteristic physical
    // state/input scales, not vehicle parameters; the transform preserves the
    // physical cost, bounds, plant equations, and actuator commands.
    constexpr float state_scale[RICCATI_MAX_NX] = {
        0.10f, 0.10f, 0.5f, 1.0f, 4.0f,
        0.5f,
        400.0f, 400.0f, 400.0f, 400.0f,
        0.10f, 1.0f, 0.25f, 4.0f, 0.25f, 4.0f,
        0.05f, 0.05f, 0.05f, 0.05f,
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
        open_loop_horizon_test_enabled_ = declare_parameter<bool>(
            "open_loop_horizon_test_enabled", false);
        open_loop_horizon_test_min_speed_mps_ = clamp(
            declare_parameter<double>(
                "open_loop_horizon_test_min_speed_mps", 6.0), 0.0, 16.0);
        open_loop_horizon_test_topic_ = declare_parameter<std::string>(
            "open_loop_horizon_test_topic", "/mpc/open_loop_test");
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
        control_time_predictor_config_.maximum_prediction_lead_s =
            std::clamp(declare_parameter<double>(
                "maximum_command_publication_lead_s", 0.500), 0.0, 0.750);
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
                diagnostics_topic_, rclcpp::QoS(rclcpp::KeepLast(10)));
        }
        if (enabled_ && open_loop_horizon_test_enabled_) {
            open_loop_horizon_test_pub_ =
                create_publisher<std_msgs::msg::String>(
                    open_loop_horizon_test_topic_, rclcpp::QoS(10));
            RCLCPP_WARN(get_logger(),
                "ONE-SHOT MPC OPEN-LOOP TEST ARMED: latch one accepted %d-step "
                "plan at >= %.2f m/s and execute it without replanning",
                PREDICTION_HORIZON,
                open_loop_horizon_test_min_speed_mps_);
        }
        // State is time-sensitive: if executor/DDS scheduling falls behind,
        // drop old samples instead of replaying them as current control input.
        control_callback_group_ = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
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
        /* Keep parameter defaults identical to the canonical competition
         * YAML. A missing/renamed config must not silently select a different
         * steering/yaw objective than saved-state replay and production. */
        rti_config_.model.weight_e_y = declare_weight("weight_e_y", 125.0f);
        rti_config_.model.weight_e_psi = declare_weight("weight_e_psi", 31.25f);
        rti_config_.model.weight_u = declare_weight("weight_u", 500.0f);
        rti_config_.model.weight_u_overspeed = declare_weight(
            "weight_u_overspeed", 300.0f);
        rti_config_.model.weight_longitudinal_slip = declare_weight(
            "weight_longitudinal_slip", 5.0f);
        rti_config_.model.weight_v = declare_weight("weight_v", 5.0f);
        rti_config_.model.weight_r = declare_weight("weight_r", 7.8125f);
        rti_config_.model.weight_steering_command = declare_weight(
            "weight_steering_command", 3.0f);
        rti_config_.model.weight_steering_target = declare_weight(
            "weight_steering_target", 5.0f);
        rti_config_.model.weight_throttle_command = declare_weight(
            "weight_throttle_command", 0.1f);
        rti_config_.model.weight_steering_slew = declare_weight(
            "weight_steering_slew", 25.0f);
        rti_config_.model.weight_throttle_slew = declare_weight(
            "weight_throttle_slew", 2.0f);
        rti_config_.model.startup_nominal_throttle_seed = static_cast<float>(
            declare_parameter<double>("startup_nominal_throttle_seed", 0.0));
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
        rti_config_.model.jacobian_threads = static_cast<int>(std::clamp<long>(
            declare_parameter<int>("jacobian_threads", 4), 0L,
            MPC_RTI_MAX_JACOBIAN_THREADS));
        const int configured_iterations = declare_parameter<int>(
            "max_solver_iterations", MAX_ITERATIONS);
        // Keep the ROS parameter consistent with the solver's compile-time
        // storage bound. The production node caps the runtime request here.
        rti_config_.solver.max_iterations = static_cast<uint16_t>(
            std::clamp(configured_iterations, 1, MAX_ITERATIONS));
        const double rti_cycle_budget_ms = declare_parameter<double>(
            "rti_cycle_budget_ms", 20.0);
        if (!std::isfinite(rti_cycle_budget_ms) || rti_cycle_budget_ms < 0.0) {
            throw std::runtime_error(
                "rti_cycle_budget_ms must be finite and nonnegative");
        }
        rti_config_.max_cycle_time_us = rti_cycle_budget_ms * 1000.0;
        end_to_end_cycle_budget_ms_ = declare_parameter<double>(
            "end_to_end_cycle_budget_ms", 20.0);
        max_command_start_lateness_ms_ = declare_parameter<double>(
            "max_command_start_lateness_ms", 2.0);
        const double initial_command_publication_lead_ms =
            declare_parameter<double>(
                "initial_command_publication_lead_ms", 15.0);
        if (!std::isfinite(end_to_end_cycle_budget_ms_) ||
            end_to_end_cycle_budget_ms_ <= 0.0 ||
            !std::isfinite(max_command_start_lateness_ms_) ||
            max_command_start_lateness_ms_ < 0.0 ||
            !std::isfinite(initial_command_publication_lead_ms) ||
            initial_command_publication_lead_ms < 0.0) {
            throw std::runtime_error(
                "end-to-end budget must be positive and command-start "
                "lateness and initial command lead must be finite and "
                "nonnegative");
        }
        last_command_publication_lead_ns_ = static_cast<int64_t>(
            std::llround(initial_command_publication_lead_ms * 1.0e6));
        rti_config_.solver.rho = static_cast<float>(
            declare_parameter<double>("admm_rho", 7.0));
        rti_config_.solver.rho_u = static_cast<float>(
            declare_parameter<double>("admm_rho_input", 7.0));
        rti_config_.solver.rho_x_by_state[MPC_OPT_IDX_EY] = static_cast<float>(
            declare_parameter<double>("admm_rho_ey", 0.0));
        rti_config_.solver.rho_x_by_state[MPC_OPT_IDX_U] = static_cast<float>(
            declare_parameter<double>("admm_rho_forward_speed", 0.0));
        rti_config_.solver.rho_x_by_state[
            MPC_OPT_IDX_ACTUAL_STEERING_ANGLE] = static_cast<float>(
                declare_parameter<double>("admm_rho_actual_steering", 0.0));
        for (const float rho_state : rti_config_.solver.rho_x_by_state) {
            if (!std::isfinite(rho_state) ||
                (rho_state != 0.0f &&
                 (rho_state < 1.0f || rho_state > 127.0f))) {
                throw std::runtime_error(
                    "ADMM state-rho overrides must be zero or in [1, 127]");
            }
        }
        rti_config_.solver.rho_u_by_input[0] = static_cast<float>(
            declare_parameter<double>("admm_rho_steering", 0.0));
        rti_config_.solver.rho_u_by_input[1] = static_cast<float>(
            declare_parameter<double>("admm_rho_throttle", 0.0));
        for (const float rho_input : rti_config_.solver.rho_u_by_input) {
            if (!std::isfinite(rho_input) ||
                (rho_input != 0.0f &&
                 (rho_input < 1.0f || rho_input > 127.0f))) {
                throw std::runtime_error(
                    "admm_rho_steering/throttle must be zero or in [1, 127]");
            }
        }
        rti_config_.solver.rho_max = static_cast<float>(
            declare_parameter<double>("admm_rho_max", 32.0));
        if (!std::isfinite(rti_config_.solver.rho_max) ||
            rti_config_.solver.rho_max < 1.0f ||
            rti_config_.solver.rho_max > 127.0f) {
            throw std::runtime_error(
                "admm_rho_max must be finite and in [1, 127]");
        }
        rti_config_.solver.tolerance = static_cast<float>(
            declare_parameter<double>("solver_tolerance", 0.01));
        rti_config_.solver.adaptive_rho = declare_parameter<bool>(
            "adaptive_rho", true) ? 1 : 0;
        rti_config_.solver.cold_start_each_solve = declare_parameter<bool>(
            "solver_cold_start_each_solve", false) ? 1 : 0;
        const int adaptive_rho_update_period = static_cast<int>(
            declare_parameter<int>("adaptive_rho_update_period", 1));
        rti_config_.solver.adaptive_rho_update_period = std::clamp(
            adaptive_rho_update_period, 1, 100);
        rti_config_.solver.shared_rho = 0;
        rti_config_.solver.over_relaxation = static_cast<float>(
            declare_parameter<double>("admm_over_relaxation", 1.6));
        configure_solver_state_scaling(rti_config_.solver,
            declare_parameter<bool>("solver_state_scaling_enabled", true));
        RCLCPP_INFO(get_logger(),
            "MPC solver config: ADMM per-solve max=%u, rho adaptation=%s/%d, "
            "rho-x overrides=[ey=%.2f,u=%.2f,steer=%.2f], "
            "rho-u override=[%.2f,%.2f], cold-start=%s, "
            "Jacobian workers=%d, RTI budget=%.1f ms, end-to-end="
            "%.1f ms, max command-start lateness=%.1f ms, "
            "initial publication lead=%.1f ms, "
            "decision state=%d, plant state=%d, horizon=%d",
            static_cast<unsigned>(rti_config_.solver.max_iterations),
            rti_config_.solver.adaptive_rho ? "on" : "off",
            rti_config_.solver.adaptive_rho_update_period,
            rti_config_.solver.rho_x_by_state[MPC_OPT_IDX_EY],
            rti_config_.solver.rho_x_by_state[MPC_OPT_IDX_U],
            rti_config_.solver.rho_x_by_state[
                MPC_OPT_IDX_ACTUAL_STEERING_ANGLE],
            rti_config_.solver.rho_u_by_input[0],
            rti_config_.solver.rho_u_by_input[1],
            rti_config_.solver.cold_start_each_solve ? "on" : "off",
            rti_config_.model.jacobian_threads,
            rti_config_.max_cycle_time_us / 1000.0,
            end_to_end_cycle_budget_ms_, max_command_start_lateness_ms_,
            initial_command_publication_lead_ms,
            MPC_OPTIMIZER_NX, MPC_OPT_PHYSICAL_NX, PREDICTION_HORIZON);
        mpc_rti_memory_reset(&rti_memory_);
        trajectory_loaded_ = load_trajectory(trajectory_file_);
        if (!trajectory_loaded_) {
            RCLCPP_ERROR(get_logger(),
                "MPC has no valid trajectory; controller remains inhibited");
        }
        if (!enabled_) {
            RCLCPP_WARN(get_logger(),
                "MPC is inactive; no command publisher is created");
        } else {
            /* Keep sensor callbacks responsive while the nonlinear solve runs.
             * The control group is mutually exclusive, so only one solve may
             * touch RTI memory at a time. Input subscriptions remain in the
             * node's separate default group and can refresh the synchronized
             * state concurrently in the four-thread component executor. */
            control_timer_ = create_wall_timer(
                std::chrono::milliseconds(5),
                std::bind(&MpcControllerNode::control_timer_callback, this),
                control_callback_group_);
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

    int64_t publish_command(double steering, double throttle_normalized)
    {
        if (!drive_command_pub_) return 0;
        if (!std::isfinite(steering) || !std::isfinite(throttle_normalized) ||
            std::abs(steering) > actuator_steering_limit_rad_ + 1.0e-6 ||
            throttle_normalized < MPC_THROTTLE_MIN_DRIVE - 1.0e-6 ||
            throttle_normalized > 1.0 + 1.0e-6) return 0;
        const double bounded_steering = clamp(steering,
            -actuator_steering_limit_rad_, actuator_steering_limit_rad_);
        const double bounded_throttle = clamp(throttle_normalized,
            MPC_THROTTLE_MIN_DRIVE, 1.0);
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
        command_published_this_cycle_ = true;
        return command_time.nanoseconds();
    }

    void report_control_fault(const char * reason,
                              bool emit_diagnostic = true,
                              bool reset_rti_memory = true)
    {
        if (control_fault_latched_) return;
        control_fault_latched_ = true;
        ++control_fault_cycles_;
        abort_open_loop_horizon_test(reason);
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "%s; MPC fault latched and no replacement drive command published",
            reason);
        if (reset_rti_memory) {
            mpc_rti_memory_reset(&rti_memory_);
        }
        /* A rejected plan is a latched controller fault, not a recovery
         * command. The run supervisor owns stopping the simulator; the MPC
         * never substitutes Unity's q=0 active-brake mode. */
        if (diagnostics_pub_ && emit_diagnostic)
            publish_cycle_failure(reason);
    }

    bool cycle_time_is_valid(int64_t callback_steady_ns,
                             int64_t candidate_command_stamp_ns,
                             const char ** fault_reason,
                             bool exact_prior_plan = false)
    {
        const double end_to_end_ms = std::max(0.0,
            static_cast<double>(steady_time_ns() - callback_steady_ns) *
                1.0e-6);
        last_publication_gate_elapsed_ms_ = end_to_end_ms;
        const double command_start_lateness_ms = std::max(0.0,
            static_cast<double>(now().nanoseconds() -
                candidate_command_stamp_ns) * 1.0e-6);
        if (command_start_lateness_ms > max_command_start_lateness_ms_) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC command-start prediction was late by %.3f ms "
                "(diagnostic threshold %.3f ms); retaining the validated "
                "candidate because cycle time is %.3f ms",
                command_start_lateness_ms, max_command_start_lateness_ms_,
                end_to_end_ms);
        }
        if (end_to_end_ms < end_to_end_cycle_budget_ms_) return true;

        /* A cycle overrun must not discard a still-safe command sequence.
         * The prior-plan caller reaches this only after exact nonlinear
         * validation from the current state through the delayed command
         * effect. Continue that plan only while its scheduled command epoch
         * is still timely; never turn an overdue action into a fallback. */
        if (exact_prior_plan && command_start_lateness_ms <=
                max_command_start_lateness_ms_) {
            RCLCPP_WARN(get_logger(),
                "MPC cycle took %.3f ms (budget %.3f ms); publishing the "
                "exact-validated prior-plan action because its command "
                "start remains timely (lateness %.3f ms)",
                end_to_end_ms, end_to_end_cycle_budget_ms_,
                command_start_lateness_ms);
            return true;
        }

        if (fault_reason)
            *fault_reason = "MPC cycle reached or exceeded its 20 ms budget";
        RCLCPP_ERROR(get_logger(),
            "MPC candidate not published: end-to-end cycle %.3f ms "
            "must remain below %.3f ms (command-start lateness %.3f ms "
            "is diagnostic only)",
            end_to_end_ms, end_to_end_cycle_budget_ms_,
            command_start_lateness_ms);
        return false;
    }

    bool try_publish_prior_plan(
        const MpcRtiState_t &state,
        double progress,
        int64_t callback_steady_ns,
        int64_t control_stamp_ns,
        int64_t candidate_command_stamp_ns,
        MpcRtiCycleResult_t *result,
        const char **fault_reason)
    {
        if (fault_reason) *fault_reason = nullptr;
        prior_plan_revalidation_attempted_ = true;
        const MpcRtiNominal_t &previous = rti_memory_.nominal;
        if (!result || !previous.valid || previous.horizon < 2) {
            if (fault_reason)
                *fault_reason = "no unconsumed accepted MPC plan remains";
            return false;
        }

        /* A distant failure in the old plan must not discard a still-safe
         * prefix. Validate through the first interval where the command being
         * published affects the plant (two queued packets plus one interval),
         * not just the already-queued motion. The plan cursor is finite. */
        const int next_control_index = prior_plan_control_index_ + 1;
        if (next_control_index + kCommandEffectValidationHorizon >
            previous.horizon) {
            if (fault_reason)
                *fault_reason = "accepted MPC plan has insufficient actions "
                    "to validate through the delayed command effect";
            return false;
        }

        MpcModelControl_t controls[kCommandEffectValidationHorizon]{};
        for (int i = 0; i < kCommandEffectValidationHorizon; ++i) {
            controls[i] = previous.controls[next_control_index + i];
            if (!std::isfinite(controls[i].steering_target_rad) ||
                !std::isfinite(controls[i].throttle_command) ||
                controls[i].throttle_command <
                    MPC_THROTTLE_MIN_DRIVE - 1.0e-6f ||
                controls[i].throttle_command > 1.0f + 1.0e-6f ||
                std::abs(controls[i].steering_target_rad) >
                    actuator_steering_limit_rad_ + 1.0e-6) {
                if (fault_reason)
                    *fault_reason = "accepted-plan prefix violates the "
                        "production drive-command envelope";
                return false;
            }
            /* Match publish_command(): clamp only float-rounding overrun;
             * validate the exact bounded sequence that will be commanded. */
            controls[i].steering_target_rad = static_cast<float>(clamp(
                controls[i].steering_target_rad,
                -actuator_steering_limit_rad_, actuator_steering_limit_rad_));
            controls[i].throttle_command = static_cast<float>(clamp(
                controls[i].throttle_command, MPC_THROTTLE_MIN_DRIVE, 1.0));
        }

        MpcRtiNominal_t revalidated{};
        int failure_stage = -1;
        prior_plan_validated_horizon_ = kCommandEffectValidationHorizon;
        const auto validation_start = std::chrono::steady_clock::now();
        const MpcRtiRolloutStatus_t rollout_status =
            mpc_rti_validate_candidate_plan(&state, progress, controls,
                trajectory_.data(), trajectory_.size(), track_length_m_,
                kCommandEffectValidationHorizon, TIME_STEP_SECONDS,
                &rti_config_, &revalidated,
                &failure_stage);
        prior_plan_validation_status_ = static_cast<int>(rollout_status);
        prior_plan_validation_failure_stage_ = failure_stage;
        prior_plan_validation_us_ = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - validation_start).count();
        if (rollout_status != MPC_RTI_ROLLOUT_OK) {
            if (fault_reason)
                *fault_reason = "accepted-plan prefix failed exact "
                    "delayed-command rollout/corridor validation";
            RCLCPP_WARN(get_logger(),
                "Prior MPC prefix not reused: exact 75 ms rollout "
                "status=%d stage=%d",
                static_cast<int>(rollout_status), failure_stage);
            return false;
        }
        prior_plan_exact_valid_ = true;
        recovered_action_revalidated_nominal_ = revalidated;
        prior_plan_validated_control_index_ = next_control_index;
        prior_plan_actions_remaining_ = previous.horizon -
            next_control_index - 1;

        const char * timing_fault_reason = nullptr;
        if (!cycle_time_is_valid(callback_steady_ns,
                candidate_command_stamp_ns, &timing_fault_reason, true)) {
            prior_plan_timing_rejected_ = true;
            if (fault_reason) *fault_reason = timing_fault_reason;
            return false;
        }
        const MpcModelControl_t &first = revalidated.controls[0];
        const int64_t command_stamp_ns = publish_command(
            first.steering_target_rad, first.throttle_command);
        if (command_stamp_ns == 0) {
            if (fault_reason)
                *fault_reason = "revalidated prior-plan action could not be "
                    "published within the command envelope";
            return false;
        }

        /* Keep the original converged sequence and advance its cursor only
         * after publishing this exactly validated action. This preserves the
         * remaining plan if another isolated solve rejection follows. */
        prior_plan_control_index_ = next_control_index;
        consecutive_candidate_prefix_actions_ = 0;
        ++consecutive_prior_plan_actions_;
        last_command_publication_lead_ns_ = command_stamp_ns >= control_stamp_ns
            ? command_stamp_ns - control_stamp_ns : 0;
        result->status = MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN;
        result->first_control = first;
        result->published_steering_command = first.steering_target_rad;
        result->published_throttle = first.throttle_command;
        result->candidate_control_count = 0;
        return true;
    }

    bool try_publish_candidate_prefix(
        const MpcRtiState_t &state,
        double progress,
        int64_t callback_steady_ns,
        int64_t control_stamp_ns,
        int64_t candidate_command_stamp_ns,
        MpcRtiCycleResult_t *result,
        const char **fault_reason)
    {
        if (fault_reason) *fault_reason = nullptr;
        if (!result || result->status !=
                MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT ||
            result->candidate_control_count < 1) return false;

        candidate_prefix_revalidation_attempted_ = true;
        const bool solver_converged =
            result->admm_status == RICCATI_STATUS_OPTIMAL;
        const bool solver_iteration_limited =
            result->admm_status == RICCATI_STATUS_MAX_ITERATIONS;
        if (!solver_converged && !solver_iteration_limited) {
            if (fault_reason)
                *fault_reason = "full-horizon-invalid candidate did not "
                    "produce a finite ADMM iterate";
            return false;
        }
        candidate_prefix_solver_converged_ = solver_converged;
        candidate_prefix_iteration_limited_ = solver_iteration_limited;
        if (result->candidate_control_count <
                kCommandEffectValidationHorizon) {
            if (fault_reason)
                *fault_reason = "rejected candidate is too short to validate "
                    "through the delayed command effect";
            return false;
        }
        MpcModelControl_t controls[kCommandEffectValidationHorizon]{};
        for (int i = 0; i < kCommandEffectValidationHorizon; ++i) {
            controls[i] = result->candidate_controls[i];
            if (!std::isfinite(controls[i].steering_target_rad) ||
                !std::isfinite(controls[i].throttle_command) ||
                controls[i].throttle_command <
                    MPC_THROTTLE_MIN_DRIVE - 1.0e-6f ||
                controls[i].throttle_command > 1.0f + 1.0e-6f ||
                std::abs(controls[i].steering_target_rad) >
                    actuator_steering_limit_rad_ + 1.0e-6) {
                if (fault_reason)
                    *fault_reason = "rejected candidate prefix violates "
                        "the production drive-command envelope";
                return false;
            }
            controls[i].steering_target_rad = static_cast<float>(clamp(
                controls[i].steering_target_rad,
                -actuator_steering_limit_rad_, actuator_steering_limit_rad_));
            controls[i].throttle_command = static_cast<float>(clamp(
                controls[i].throttle_command, MPC_THROTTLE_MIN_DRIVE, 1.0));
        }

        MpcRtiNominal_t revalidated{};
        int failure_stage = -1;
        candidate_prefix_validation_horizon_ =
            kCommandEffectValidationHorizon;
        const auto validation_start = std::chrono::steady_clock::now();
        const MpcRtiRolloutStatus_t rollout_status =
            mpc_rti_validate_candidate_plan(&state, progress,
                controls, trajectory_.data(), trajectory_.size(),
                track_length_m_, kCommandEffectValidationHorizon,
                TIME_STEP_SECONDS, &rti_config_, &revalidated,
                &failure_stage);
        candidate_prefix_validation_status_ = static_cast<int>(rollout_status);
        candidate_prefix_validation_failure_stage_ = failure_stage;
        candidate_prefix_validation_us_ =
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - validation_start).count();
        if (rollout_status != MPC_RTI_ROLLOUT_OK) {
            if (fault_reason)
                *fault_reason = "rejected candidate prefix failed exact "
                    "delayed-command rollout/corridor validation";
            RCLCPP_WARN(get_logger(),
                "Rejected horizon prefix not reused: exact 75 ms "
                "rollout status=%d stage=%d",
                static_cast<int>(rollout_status), failure_stage);
            return false;
        }
        candidate_prefix_exact_valid_ = true;
        recovered_action_revalidated_nominal_ = revalidated;

        const char * timing_fault_reason = nullptr;
        if (!cycle_time_is_valid(callback_steady_ns,
                candidate_command_stamp_ns, &timing_fault_reason)) {
            candidate_prefix_timing_rejected_ = true;
            if (fault_reason) *fault_reason = timing_fault_reason;
            return false;
        }
        const MpcModelControl_t &first = revalidated.controls[0];
        const int64_t command_stamp_ns = publish_command(
            first.steering_target_rad, first.throttle_command);
        if (command_stamp_ns == 0) {
            if (fault_reason)
                *fault_reason = "exact-validated candidate prefix could not "
                    "be published within the command envelope";
            return false;
        }

        /* The stored accepted plan advanced in time even though this newer
         * candidate supplied the output. Skip the now-stale corresponding
         * action before it can be used as a later prior-plan recovery. */
        const MpcRtiNominal_t &prior = rti_memory_.nominal;
        if (prior.valid && prior_plan_control_index_ < prior.horizon - 1)
            ++prior_plan_control_index_;
        /* This is a newly solved control, not stale-plan replay. Even when
         * ADMM hit its iteration cap, publish only if the exact three-step
         * delayed-action prefix is corridor-safe and meets the deadline.
         * Re-solve at the next tick; never repeat a stale candidate blindly. */
        consecutive_prior_plan_actions_ = 0;
        ++consecutive_candidate_prefix_actions_;
        last_command_publication_lead_ns_ = command_stamp_ns >= control_stamp_ns
            ? command_stamp_ns - control_stamp_ns : 0;
        result->status = MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX;
        result->first_control = first;
        result->published_steering_command = first.steering_target_rad;
        result->published_throttle = first.throttle_command;
        result->candidate_control_count = 0;
        return true;
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

    void publish_waiting_for_localization()
    {
        if (!diagnostics_pub_) return;
        std_msgs::msg::String message;
        message.data = "{\"status\":\"waiting_for_localization\","
            "\"reason\":\"initial_map_pose_not_received\"}";
        diagnostics_pub_->publish(message);
    }

    void publish_waiting_for_plant_state()
    {
        if (!diagnostics_pub_) return;
        std_msgs::msg::String message;
        message.data = "{\"status\":\"waiting_for_plant_state\","
            "\"reason\":\"initial_causal_plant_snapshot_not_received\"}";
        diagnostics_pub_->publish(message);
    }

    bool actuator_subscription_matched() const
    {
        if (!drive_command_pub_) return false;
        const auto endpoints = get_subscriptions_info_by_topic(
            drive_command_topic_);
        return std::any_of(
            endpoints.begin(), endpoints.end(), [](const auto & endpoint) {
                return endpoint.endpoint_type() ==
                        rclcpp::EndpointType::Subscription &&
                    endpoint.node_name() == "autodrive_actuator_interface";
            });
    }

    void publish_waiting_for_actuator()
    {
        if (!diagnostics_pub_) return;
        std_msgs::msg::String message;
        message.data = "{\"status\":\"waiting_for_actuator\","
            "\"reason\":\"drive_command_subscription_not_matched\"}";
        diagnostics_pub_->publish(message);
    }

    void publish_open_loop_horizon_event(const std::string & json)
    {
        if (!open_loop_horizon_test_pub_) return;
        std_msgs::msg::String message;
        message.data = json;
        open_loop_horizon_test_pub_->publish(message);
    }

    void abort_open_loop_horizon_test(const char * reason)
    {
        if (!open_loop_horizon_test_active_) return;
        std::ostringstream json;
        json << "{\"event\":\"abort\",\"plan_id\":"
             << open_loop_horizon_test_plan_id_
             << ",\"next_stage\":" << open_loop_horizon_test_next_stage_
             << ",\"reason\":\"" << reason << "\"}";
        publish_open_loop_horizon_event(json.str());
        open_loop_horizon_test_active_ = false;
        open_loop_horizon_test_finished_ = true;
        if (open_loop_horizon_timer_) open_loop_horizon_timer_->cancel();
    }

    int open_loop_stage_at(int64_t stamp_ns) const
    {
        if (open_loop_horizon_start_stamp_ns_ <= 0 ||
            stamp_ns <= open_loop_horizon_start_stamp_ns_) return 0;
        const double elapsed_s = static_cast<double>(
            stamp_ns - open_loop_horizon_start_stamp_ns_) * 1.0e-9;
        return std::clamp(static_cast<int>(std::llround(
            elapsed_s / TIME_STEP_SECONDS)), 0,
            open_loop_horizon_test_plan_.horizon);
    }

    void publish_open_loop_observation_gap(int stage, const char * reason)
    {
        std::ostringstream gap;
        gap << "{\"event\":\"observation_gap\",\"plan_id\":"
            << open_loop_horizon_test_plan_id_
            << ",\"stage\":" << stage
            << ",\"reason\":\"" << reason << "\"}";
        publish_open_loop_horizon_event(gap.str());
    }

    void record_open_loop_observation_gaps_through(
        int observed_stage, const char * reason)
    {
        observed_stage = std::clamp(observed_stage, 0,
            open_loop_horizon_test_plan_.horizon);
        for (int stage = open_loop_horizon_last_sample_stage_ + 1;
             stage < observed_stage; ++stage) {
            publish_open_loop_observation_gap(stage, reason);
        }
    }

    void finish_open_loop_horizon_test(bool terminal_state_observed)
    {
        if (!open_loop_horizon_test_active_) return;
        open_loop_horizon_test_active_ = false;
        open_loop_horizon_test_finished_ = true;
        std::ostringstream completed;
        completed << "{\"event\":\"complete\",\"plan_id\":"
                  << open_loop_horizon_test_plan_id_
                  << ",\"outputs\":"
                  << open_loop_horizon_test_plan_.horizon
                  << ",\"terminal_state_observed\":"
                  << (terminal_state_observed ? "true" : "false") << '}';
        publish_open_loop_horizon_event(completed.str());
        if (open_loop_horizon_timer_) open_loop_horizon_timer_->cancel();
    }

    void open_loop_horizon_timer_callback()
    {
        if (!open_loop_horizon_test_active_) return;
        if (open_loop_horizon_test_next_stage_ <
            open_loop_horizon_test_plan_.horizon) {
            const int stage = open_loop_horizon_test_next_stage_++;
            if (publish_open_loop_horizon_command(stage) == 0) {
                report_control_fault(
                    "latched MPC plan contains a command outside the drive envelope");
                return;
            }
            if (open_loop_horizon_test_next_stage_ ==
                open_loop_horizon_test_plan_.horizon) {
                open_loop_horizon_all_outputs_sent_ = true;
            }
            return;
        }

        // Observe the terminal state without inventing a new actuator action.
        if (!open_loop_horizon_terminal_waiting_) {
            open_loop_horizon_terminal_waiting_ = true;
            open_loop_horizon_terminal_wait_start_ns_ = steady_time_ns();
            if (open_loop_horizon_last_sample_stage_ >=
                open_loop_horizon_test_plan_.horizon) {
                finish_open_loop_horizon_test(true);
            }
            return;
        }
        if (steady_time_ns() - open_loop_horizon_terminal_wait_start_ns_ >=
            100000000LL) {
            finish_open_loop_horizon_test(false);
        }
    }

    void publish_open_loop_horizon_plan(
        const MpcRtiState_t & initial_state, int64_t stamp_ns)
    {
        std::ostringstream json;
        json << std::setprecision(9)
             << "{\"event\":\"plan\",\"plan_id\":"
             << open_loop_horizon_test_plan_id_
             << ",\"stamp_ns\":" << stamp_ns
             << ",\"horizon\":" << open_loop_horizon_test_plan_.horizon
             << ",\"step_s\":" << TIME_STEP_SECONDS
             << ",\"trigger_speed_mps\":" << initial_state.plant.u
             << ",\"controls\":[";
        for (int step = 0; step < open_loop_horizon_test_plan_.horizon; ++step) {
            if (step > 0) json << ',';
            const MpcModelControl_t & control =
                open_loop_horizon_test_plan_.controls[step];
            json << '[' << control.steering_target_rad << ','
                 << control.throttle_command << ']';
        }
        json << "],\"predictions\":[";
        for (int step = 0; step <= open_loop_horizon_test_plan_.horizon; ++step) {
            if (step > 0) json << ',';
            const MpcRtiState_t & predicted =
                open_loop_horizon_test_plan_.states[step];
            const double predicted_s =
                open_loop_horizon_test_plan_.progress[step];
            MpcTrajectorySample_t reference{};
            if (!mpc_trajectory_sample(trajectory_.data(), trajectory_.size(),
                    track_length_m_, predicted_s, &reference)) {
                json << "null";
                continue;
            }
            const double predicted_x = reference.x -
                std::sin(reference.heading) * predicted.plant.e_y;
            const double predicted_y = reference.y +
                std::cos(reference.heading) * predicted.plant.e_y;
            const double predicted_yaw =
                reference.heading + predicted.plant.e_psi;
            json << "{\"stage\":" << step
                 << ",\"time_s\":" << step * TIME_STEP_SECONDS
                 << ",\"progress_m\":" << predicted_s
                 << ",\"pose\":[" << predicted_x << ',' << predicted_y
                 << ',' << predicted_yaw << "],\"state\":["
                 << predicted.plant.e_y << ',' << predicted.plant.e_psi << ','
                 << predicted.plant.u << ',' << predicted.plant.v << ','
                 << predicted.plant.r << ','
                 << predicted.plant.actual_steering_angle;
            for (int wheel = 0; wheel < 4; ++wheel) {
                json << ',' << predicted.plant.source_plant_internal_state[
                    MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel];
            }
            json << "]}";
        }
        json << "]}";
        publish_open_loop_horizon_event(json.str());
    }

    void publish_open_loop_horizon_sample(
        int stage, const MpcRtiState_t & actual,
        const MpcSynchronizedState & synchronized,
        const MpcSynchronizedState & command_time_state,
        const MpcPathProjection_t & actual_projection,
        int64_t control_stamp_ns)
    {
        if (stage < 0 || stage > open_loop_horizon_test_plan_.horizon) return;
        const MpcRtiState_t & predicted =
            open_loop_horizon_test_plan_.states[stage];
        MpcTrajectorySample_t reference{};
        const double predicted_s =
            open_loop_horizon_test_plan_.progress[stage];
        if (!mpc_trajectory_sample(trajectory_.data(), trajectory_.size(),
                track_length_m_, predicted_s, &reference)) {
            abort_open_loop_horizon_test("predicted-reference-unavailable");
            return;
        }

        const double predicted_x = reference.x -
            std::sin(reference.heading) * predicted.plant.e_y;
        const double predicted_y = reference.y +
            std::cos(reference.heading) * predicted.plant.e_y;
        const double predicted_yaw = reference.heading + predicted.plant.e_psi;
        const auto append_state = [](std::ostringstream & out,
                                     const MpcModelState_t & state) {
            out << '[' << state.e_y << ',' << state.e_psi << ',' << state.u
                << ',' << state.v << ',' << state.r << ','
                << state.actual_steering_angle;
            for (int wheel = 0; wheel < 4; ++wheel) {
                out << ',' << state.source_plant_internal_state[
                    MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel];
            }
            out << ']';
        };

        std::ostringstream json;
        json << std::setprecision(9)
             << "{\"event\":\"sample\",\"plan_id\":"
             << open_loop_horizon_test_plan_id_
             << ",\"stage\":" << stage
             << ",\"control_stamp_ns\":" << control_stamp_ns
             << ",\"source_stamp_ns\":" << synchronized.source_stamp_ns
             << ",\"odom_source_stamp_ns\":"
             << synchronized.odom_source_stamp_ns
             << ",\"actual_progress_m\":" << actual_projection.s
             << ",\"predicted_progress_m\":" << predicted_s
             << ",\"predicted_pose\":[" << predicted_x << ','
             << predicted_y << ',' << predicted_yaw
             << "],\"actual_pose\":[" << command_time_state.map_x << ','
             << command_time_state.map_y << ',' << command_time_state.map_yaw
             << "],\"source_pose\":[" << synchronized.map_x << ','
             << synchronized.map_y << ',' << synchronized.map_yaw
             << "],\"predicted_state\":";
        append_state(json, predicted.plant);
        json << ",\"actual_state\":";
        append_state(json, actual.plant);
        if (stage < open_loop_horizon_test_plan_.horizon) {
            const MpcModelControl_t & control =
                open_loop_horizon_test_plan_.controls[stage];
            json << ",\"planned_control\":["
                 << control.steering_target_rad << ','
                 << control.throttle_command << ']';
        } else {
            json << ",\"planned_control\":null";
        }
        json << '}';
        publish_open_loop_horizon_event(json.str());
    }

    int64_t publish_open_loop_horizon_command(int stage)
    {
        const MpcModelControl_t & control =
            open_loop_horizon_test_plan_.controls[stage];
        if (control.throttle_command < MPC_THROTTLE_MIN_DRIVE ||
            control.throttle_command > 1.0 ||
            std::abs(control.steering_target_rad) >
                actuator_steering_limit_rad_) return 0;
        const int64_t command_stamp_ns = publish_command(
            control.steering_target_rad, control.throttle_command);
        if (command_stamp_ns == 0) return 0;
        std::ostringstream output;
        output << std::setprecision(9)
               << "{\"event\":\"output\",\"plan_id\":"
               << open_loop_horizon_test_plan_id_
               << ",\"stage\":" << stage
               << ",\"steady_stamp_ns\":" << steady_time_ns()
               << ",\"control\":[" << control.steering_target_rad << ','
               << control.throttle_command << "]}";
        publish_open_loop_horizon_event(output.str());
        return command_stamp_ns;
    }

    void record_cycle_health(MpcRtiCycleStatus_t status,
                             const MpcRtiCycleResult_t & result,
                             double solve_us)
    {
        ++control_cycles_;
        if (status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL) ++accepted_optimal_;
        else if (status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE)
            ++accepted_feasible_;
        else if (status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH)
            ++accepted_line_search_;
        else if (status == MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX)
            ++accepted_candidate_prefix_;
        else if (status == MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN)
            ++accepted_prior_plan_;
        max_solver_iterations_seen_ = std::max(
            max_solver_iterations_seen_, result.solver_iterations);
        max_solve_us_ = std::max(max_solve_us_, solve_us);
        if (control_cycles_ % 400 == 0) report_health("periodic");
    }

    void report_health(const char * phase)
    {
        RCLCPP_INFO(get_logger(),
            "MPC health (%s): control_cycles=%llu accepted_optimal=%llu "
            "accepted_feasible=%llu accepted_line_search=%llu "
            "accepted_candidate_prefix=%llu "
            "accepted_prior_plan=%llu "
            "control_fault_cycles=%llu max_solver_iterations_seen=%d "
            "max_solve_us=%.1f",
            phase,
            static_cast<unsigned long long>(control_cycles_),
            static_cast<unsigned long long>(accepted_optimal_),
            static_cast<unsigned long long>(accepted_feasible_),
            static_cast<unsigned long long>(accepted_line_search_),
            static_cast<unsigned long long>(accepted_candidate_prefix_),
            static_cast<unsigned long long>(accepted_prior_plan_),
            static_cast<unsigned long long>(control_fault_cycles_),
            max_solver_iterations_seen_, max_solve_us_);
    }

    static const char * cycle_status_name(MpcRtiCycleStatus_t status)
    {
        switch (status) {
        case MPC_RTI_CYCLE_ACCEPTED_OPTIMAL: return "accepted_optimal";
        case MPC_RTI_CYCLE_ACCEPTED_FEASIBLE: return "accepted_feasible";
        case MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH: return "accepted_line_search";
        case MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX:
            return "accepted_candidate_prefix";
        case MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN: return "accepted_prior_plan";
        case MPC_RTI_CYCLE_REJECTED_INPUT: return "rejected_input";
        case MPC_RTI_CYCLE_REJECTED_SOLVER: return "rejected_solver";
        case MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT:
            return "rejected_nonlinear_rollout";
        case MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET:
            return "waiting_for_solver_budget";
        }
        return "unknown";
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
        int64_t candidate_command_stamp_ns,
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
            << "\",\"action_source\":\""
            << (!command_published_this_cycle_ ? "none_published" :
                (result.status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH
                    ? "current_solver_line_search" :
                (result.status == MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX
                    ? "revalidated_candidate_prefix" :
                    (result.status == MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN
                        ? "revalidated_prior_plan"
                        : "current_solver_candidate"))))
            << "\",\"command_published\":"
            << (command_published_this_cycle_ ? "true" : "false")
            << ",\"solver_skipped_for_deadline\":"
            << (result.solver_skipped_for_deadline ? "true" : "false")
            << ",\"prior_plan_revalidation\":{\"attempted\":"
            << (prior_plan_revalidation_attempted_ ? "true" : "false")
            << ",\"exact_valid\":"
            << (prior_plan_exact_valid_ ? "true" : "false")
            << ",\"timing_rejected\":"
            << (prior_plan_timing_rejected_ ? "true" : "false")
            << ",\"validation_horizon\":"
            << prior_plan_validated_horizon_
            << ",\"control_index\":"
            << prior_plan_validated_control_index_
            << ",\"actions_remaining\":"
            << prior_plan_actions_remaining_
            << ",\"validation_status\":"
            << prior_plan_validation_status_
            << ",\"failure_stage\":"
            << prior_plan_validation_failure_stage_
            << ",\"validation_us\":";
        json_number(prior_plan_validation_us_);
        json << "},\"candidate_prefix_revalidation\":{\"attempted\":"
            << (candidate_prefix_revalidation_attempted_ ? "true" : "false")
            << ",\"exact_valid\":"
            << (candidate_prefix_exact_valid_ ? "true" : "false")
            << ",\"timing_rejected\":"
            << (candidate_prefix_timing_rejected_ ? "true" : "false")
            << ",\"solver_converged\":"
            << (candidate_prefix_solver_converged_ ? "true" : "false")
            << ",\"solver_iteration_limited\":"
            << (candidate_prefix_iteration_limited_ ? "true" : "false")
            << ",\"validation_horizon\":"
            << candidate_prefix_validation_horizon_
            << ",\"validation_status\":"
            << candidate_prefix_validation_status_
            << ",\"failure_stage\":"
            << candidate_prefix_validation_failure_stage_
            << ",\"validation_us\":";
        json_number(candidate_prefix_validation_us_);
        json << "},\"consecutive_prior_plan_actions\":"
            << consecutive_prior_plan_actions_
            << ",\"prior_plan_actions_remaining\":"
            << prior_plan_actions_remaining_
            << ",\"consecutive_candidate_prefix_actions\":"
            << consecutive_candidate_prefix_actions_
            << ",\"input_failure_reason\":\""
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
            << '"';
        append_rollout_failure("nonlinear_failure", result.nonlinear_failure);
        json
            << ",\"source_stamp_ns\":" << synchronized.source_stamp_ns
            << ",\"odom_source_stamp_ns\":"
            << synchronized.odom_source_stamp_ns
            << ",\"map_pose_source_stamp_ns\":"
            << synchronized.map_pose_source_stamp_ns
            << ",\"control_ros_stamp_ns\":" << control_ros_stamp_ns
            << ",\"candidate_command_stamp_ns\":"
            << candidate_command_stamp_ns
            << ",\"candidate_command_lead_s\":"
            << static_cast<double>(candidate_command_stamp_ns -
                control_ros_stamp_ns) * 1.0e-9
            << ",\"candidate_command_lateness_ms\":"
            << std::max(0.0, static_cast<double>(now().nanoseconds() -
                candidate_command_stamp_ns) * 1.0e-6)
            << ",\"max_command_start_lateness_ms\":"
            << max_command_start_lateness_ms_
            << ",\"end_to_end_control_ms\":"
            << std::max(0.0, static_cast<double>(steady_time_ns() -
                callback_steady_ns) * 1.0e-6)
            << ",\"publication_gate_elapsed_ms\":";
        if (last_publication_gate_elapsed_ms_ >= 0.0)
            json_number(last_publication_gate_elapsed_ms_);
        else
            json << "null";
        json
            << ",\"end_to_end_cycle_budget_ms\":"
            << end_to_end_cycle_budget_ms_
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
            << ",\"active_speed_ceiling_mps\":"
            << rti_config_.model.active_speed_ceiling_mps
            << ",\"path_curvature_per_m\":";
        json_number(current_path_sample_valid ? current_path_sample.curvature :
            std::numeric_limits<double>::quiet_NaN());
        json << ",\"reference\":{\"trajectory_acceleration_mps2\":";
        json_number(current_path_sample_valid ? current_path_sample.acceleration :
            std::numeric_limits<double>::quiet_NaN());
        const double current_reference_speed = current_path_sample_valid
            ? std::min(current_path_sample.speed,
                static_cast<double>(rti_config_.model.active_speed_ceiling_mps))
            : std::numeric_limits<double>::quiet_NaN();
        json << ",\"trajectory_speed_mps\":";
        json_number(current_path_sample_valid ? current_path_sample.speed :
            std::numeric_limits<double>::quiet_NaN());
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
            << ",\"plant_internal_state\":[";
        for (int i = 0; i < MPC_PLANT_INTERNAL_STATE_COUNT; ++i) {
            if (i > 0) json << ',';
            json_number(state.plant.source_plant_internal_state[i]);
        }
        json << "],\"plant_state_flags\":["
            << state.plant.source_plant_initialized << ','
            << state.plant.source_suspension_history_initialized << ','
            << state.plant.source_imu_state_initialized << ']'
            << ",\"plant_state_source\":\""
            << (unity_plant_state_used_
                    ? "odom_unity_plant_state"
                    : "source_plant_from_odom_or_handoff")
            << "\"";
        json << ",\"solver\":{\"iterations\":" << result.solver_iterations
            << ",\"max_iterations_configured\":"
            << rti_config_.solver.max_iterations
            << ",\"time_budget_exhausted\":"
            << (result.solver_time_budget_exhausted ? "true" : "false")
            << ",\"sqp_iterations_attempted\":"
            << result.sqp_iterations_attempted
            << ",\"sqp_iterations_completed\":"
            << result.sqp_iterations_completed
            << ",\"sqp_iterations_accepted\":"
            << result.sqp_iterations_accepted
            << ",\"admm_status\":" << result.admm_status
            << ",\"primal_residual\":";
        json_number(result.primal_residual);
        json << ",\"dual_residual\":";
        json_number(result.dual_residual);
        json << ",\"max_regularization\":";
        json_number(result.maximum_regularization);
        json << ",\"regularization_count\":"
             << result.regularization_count
             << ",\"nonsmooth_columns\":"
             << result.nonsmooth_jacobian_columns
             << ",\"rho_start\":" << result.rho_start
             << ",\"rho_u_start\":" << result.rho_u_start
             << ",\"rho_final\":" << result.rho_final
             << ",\"rho_u_final\":" << result.rho_u_final
             << ",\"rho_change_count\":" << result.rho_change_count
             << ",\"factorization_count\":" << result.factorization_count
             << ",\"factorization_time_ns\":"
             << result.factorization_time_ns
             << ",\"cold_start_initialization_time_ns\":"
             << result.solver_cold_start_initialization_time_ns
             << ",\"riccati_rhs_time_ns\":"
             << result.riccati_rhs_time_ns
             << ",\"admm_projection_time_ns\":"
             << result.admm_projection_time_ns
             << ",\"rho_update_time_ns\":" << result.rho_update_time_ns
             << ",\"scaling_transform_time_ns\":"
             << result.solver_scaling_time_ns
             << ",\"output_unscale_time_ns\":"
             << result.solver_output_unscale_time_ns
             << ",\"qp_build_us\":" << result.qp_build_us
             << ",\"jacobian_wall_us\":" << result.jacobian_wall_us
             << ",\"jacobian_thread_cpu_us\":"
             << result.jacobian_thread_cpu_us
             << ",\"jacobian_slowest_stage_us\":"
             << result.jacobian_slowest_stage_us
             << ",\"jacobian_stages_completed\":"
             << result.jacobian_stages_completed
             << ",\"jacobian_threads_used\":"
             << result.jacobian_threads_used
             << ",\"jacobian_source_sensitivity_calls\":"
             << result.jacobian_source_sensitivity_calls
             << ",\"jacobian_finite_difference_columns\":"
             << result.jacobian_finite_difference_columns
             << ",\"nominal_build_us\":" << result.nominal_build_us
             << ",\"reference_seed_us\":" << result.reference_seed_us
             << ",\"cold_start_queue_seed_us\":"
             << result.cold_start_queue_seed_us
             << ",\"nominal_rollout_a_us\":"
             << result.nominal_rollout_a_us
             << ",\"progress_reference_resampling_us\":"
             << result.progress_reference_resampling_us
             << ",\"nominal_rollout_b_us\":"
             << result.nominal_rollout_b_us
             << ",\"admm_us\":" << result.admm_us
             << ",\"exact_rollout_us\":" << result.exact_rollout_us
             << ",\"nonlinear_validation_us\":"
             << result.nonlinear_validation_us
             << ",\"line_search_attempted\":"
             << (result.line_search_attempted ? "true" : "false")
             << ",\"line_search_trials\":" << result.line_search_trials
             << ",\"line_search_accepted\":"
             << (result.line_search_accepted ? "true" : "false")
             << ",\"line_search_skipped_budget\":"
             << (result.line_search_skipped_budget ? "true" : "false")
             << ",\"line_search_skipped_command_epoch\":"
             << (result.line_search_skipped_command_epoch ? "true" : "false")
             << ",\"line_search_full_step_failure_reason\":"
             << result.line_search_full_step_failure_reason
             << ",\"line_search_full_step_failure_stage\":"
             << result.line_search_full_step_failure_stage
             << ",\"line_search_last_failure_reason\":"
             << result.line_search_last_failure_reason
             << ",\"line_search_last_failure_stage\":"
             << result.line_search_last_failure_stage
             << ",\"line_search_selected_alpha\":"
             << result.line_search_selected_alpha
             << ",\"line_search_us\":" << result.line_search_us
             << ",\"solve_us\":" << result.total_solve_us
             << ",\"node_solve_us\":" << solve_us
             << ",\"cycle_us\":" << result.total_cycle_us
             << ",\"objective\":";
        json_number(result.nonlinear_objective);
        json << '}'
             << ",\"first_action\":["
             << result.published_steering_command << ','
             << result.published_throttle << ']'
             << ",\"nominal_first_action\":["
             << result.nominal_first_control.steering_target_rad << ','
             << result.nominal_first_control.throttle_command << ']'
             << ",\"linearization_failure\":["
             << result.linearization_failure_stage << ','
             << result.linearization_failure_reason << ','
             << result.linearization_failure_index << ']'
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
        if (result.candidate_control_count > 0) {
            const bool accepted_controls =
                result.status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
                result.status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
                result.status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH ||
                result.status == MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX ||
                result.status == MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN;
            json << (accepted_controls ? ",\"candidate_controls\":["
                                      : ",\"rejected_candidate_controls\":[");
            for (int k = 0; k < result.candidate_control_count; ++k) {
                if (k > 0) json << ',';
                json << '['
                     << result.candidate_controls[k].steering_target_rad
                     << ',' << result.candidate_controls[k].throttle_command
                     << ']';
            }
            json << ']';
        }
        json << ",\"predictions\":[";

        const bool accepted =
            result.status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
            result.status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
            result.status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH ||
            result.status == MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX ||
            result.status == MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN;
        if (!accepted) {
            json << "],\"diagnostic_publish_steady_ns\":"
                 << steady_time_ns() << '}';
            std_msgs::msg::String message;
            message.data = json.str();
            diagnostics_pub_->publish(message);
            return;
        }

        bool first = true;
        const MpcRtiNominal_t &diagnostic_nominal =
            (result.status == MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX ||
             result.status == MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN)
                ? recovered_action_revalidated_nominal_
                : rti_memory_.nominal;
        for (const int step : steps) {
            if (step > diagnostic_nominal.horizon) continue;
            if (!first) json << ',';
            first = false;
            const MpcRtiState_t &predicted = diagnostic_nominal.states[step];
            MpcTrajectorySample_t sample{};
            const double predicted_s = diagnostic_nominal.progress[step];
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
                diagnostic_nominal.controls[step - 1];
            MpcTrajectorySample_t transition_sample{};
            MpcStageResult_t diagnostic_transition{};
            bool have_transition_output =
                step <= diagnostic_nominal.horizon &&
                mpc_trajectory_sample(trajectory_.data(), trajectory_.size(),
                    track_length_m_,
                    diagnostic_nominal.progress[step - 1],
                    &transition_sample);
            if (have_transition_output) {
                diagnostic_transition = mpc_vehicle_model_step(
                    &diagnostic_nominal.states[step - 1].plant,
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
                << feedforward << ']'
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
            !finite_array(message->suspension_jounce_m) ||
            !finite_array(message->low_forward_speed_time_s) ||
            !finite_array(message->low_side_speed_time_s) ||
            !std::isfinite(message->body_height_m) ||
            !std::isfinite(message->body_height_rate_mps) ||
            !std::isfinite(message->roll_rad) ||
            !std::isfinite(message->roll_rate_radps) ||
            !std::isfinite(message->pitch_rad) ||
            !std::isfinite(message->pitch_rate_radps) ||
            !std::isfinite(message->physical_steering_rad) ||
            !std::isfinite(message->longitudinal_velocity_mps) ||
            !std::isfinite(message->lateral_velocity_rear_axle_mps) ||
            !std::isfinite(message->gyro_yaw_rate_radps) ||
            !std::isfinite(message->delayed_steering_command_1_normalized) ||
            !std::isfinite(message->delayed_steering_command_2_normalized) ||
            !std::isfinite(message->delayed_throttle_command_1_normalized) ||
            !std::isfinite(message->delayed_throttle_command_2_normalized)) {
            return;
        }
        if (message->delayed_steering_command_1_normalized < -1.0 ||
            message->delayed_steering_command_1_normalized > 1.0 ||
            message->delayed_steering_command_2_normalized < -1.0 ||
            message->delayed_steering_command_2_normalized > 1.0 ||
            message->throttle_command_normalized < 0.0 ||
            message->throttle_command_normalized > 1.0 ||
            message->delayed_throttle_command_1_normalized < 0.0 ||
            message->delayed_throttle_command_1_normalized > 1.0 ||
            message->delayed_throttle_command_2_normalized < 0.0 ||
            message->delayed_throttle_command_2_normalized > 1.0) {
            return;
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
            std::lock_guard<std::mutex> lock(state_mutex_);
            latest_odom_ingest_status_ = MpcSyncStatus::kInvalidInput;
            ++odom_generation_;
            return;
        }
        if (odom_stamp.nanoseconds() <= 0) odom_stamp = now();

        if (!enabled_ || control_fault_latched_) return;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            latest_odom_ingest_status_ = state_synchronizer_.push_odometry({
                odom_stamp.nanoseconds(), odom_pose.position.x,
                odom_pose.position.y, odom_yaw,
                message->twist.twist.linear.x,
                message->twist.twist.linear.y,
                message->twist.twist.angular.z});
            ++odom_generation_;
        }
    }

    void control_timer_callback()
    {
        const double callback_start_monotonic_us =
            mpc_rti_monotonic_time_us();
        const int64_t callback_steady_ns = steady_time_ns();
        double cycle_budget_us = end_to_end_cycle_budget_ms_ * 1000.0;
        if (rti_config_.max_cycle_time_us > 0.0)
            cycle_budget_us = std::min(
                cycle_budget_us, rti_config_.max_cycle_time_us);
        const double callback_deadline_us =
            callback_start_monotonic_us + cycle_budget_us;
        last_publication_gate_elapsed_ms_ = -1.0;
        command_published_this_cycle_ = false;
        prior_plan_revalidation_attempted_ = false;
        prior_plan_exact_valid_ = false;
        prior_plan_timing_rejected_ = false;
        prior_plan_validation_us_ = 0.0;
        prior_plan_validated_horizon_ = 0;
        prior_plan_validated_control_index_ = -1;
        prior_plan_actions_remaining_ = 0;
        candidate_prefix_revalidation_attempted_ = false;
        candidate_prefix_exact_valid_ = false;
        candidate_prefix_timing_rejected_ = false;
        candidate_prefix_validation_us_ = 0.0;
        candidate_prefix_validation_horizon_ = 0;
        candidate_prefix_iteration_limited_ = false;
        prior_plan_validation_status_ = -1;
        prior_plan_validation_failure_stage_ = -1;
        candidate_prefix_solver_converged_ = false;
        candidate_prefix_validation_status_ = -1;
        candidate_prefix_validation_failure_stage_ = -1;
        if (!enabled_ || control_fault_latched_) return;

        /* The development recorder also subscribes to drive_command, so a
         * publisher subscriber-count check is not sufficient. Wait for the
         * actuator endpoint itself before spending a solve or publishing a
         * plan that cannot reach Unity. DDS discovery can complete after the
         * component's first timer callbacks. */
        if (!actuator_subscription_matched()) {
            if (!waiting_for_actuator_) {
                RCLCPP_WARN(get_logger(),
                    "MPC waiting for the actuator command subscription to match");
                publish_waiting_for_actuator();
                waiting_for_actuator_ = true;
            }
            return;
        }
        if (waiting_for_actuator_) {
            RCLCPP_INFO(get_logger(),
                "MPC actuator command subscription is matched; enabling solves");
            waiting_for_actuator_ = false;
        }

        std::uint64_t odom_generation = 0;
        MpcSyncStatus odom_status = MpcSyncStatus::kMissingOdom;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            odom_generation = odom_generation_;
            odom_status = latest_odom_ingest_status_;
        }
        if (odom_generation == last_control_odom_generation_) return;
        last_control_odom_generation_ = odom_generation;

        if (!trajectory_loaded_) {
            report_control_fault("MPC requires a valid trajectory");
            return;
        }
        if (odom_status != MpcSyncStatus::kOk) {
            if (open_loop_horizon_test_active_) {
                record_open_loop_observation_gaps_through(
                    open_loop_stage_at(now().nanoseconds()),
                    MpcStateSynchronizer::status_name(odom_status));
                return;
            }
            report_control_fault(
                "MPC rejected uncertain legal odometry sample");
            return;
        }

        MpcSynchronizedState coherent_state;
        MpcSyncStatus sync_status;
        const rclcpp::Time control_ros_time = now();
        const int64_t control_stamp_ns = control_ros_time.nanoseconds();
        const int64_t synchronize_steady_ns = steady_time_ns();
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            sync_status = state_synchronizer_.synchronize(
                control_ros_time.nanoseconds(), &coherent_state);
        }
        if (sync_status != MpcSyncStatus::kOk) {
            if (!state_initialized_ &&
                sync_status == MpcSyncStatus::kMissingMapPose) {
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                    "MPC waiting for its first localized map pose");
                publish_waiting_for_localization();
                return;
            }
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC waiting for legal state input: %s",
                MpcStateSynchronizer::status_name(sync_status));
            // Do not keep racing on stale odometry/localization. The direct
            // controller-fault path below publishes no replacement command.
            const std::string reason = std::string(
                "MPC rejected unsynchronized legal state: ") +
                MpcStateSynchronizer::status_name(sync_status);
            if (open_loop_horizon_test_active_) {
                record_open_loop_observation_gaps_through(
                    open_loop_stage_at(control_ros_time.nanoseconds()),
                    MpcStateSynchronizer::status_name(sync_status));
                return;
            }
            report_control_fault(
                reason.c_str());
            return;
        }
        state_initialized_ = true;

        const int64_t publication_lead_ns =
            std::max<int64_t>(0, last_command_publication_lead_ns_);
        if (publication_lead_ns > 0 && control_stamp_ns >
            std::numeric_limits<int64_t>::max() - publication_lead_ns) {
            report_control_fault("MPC command-time prediction stamp overflow");
            return;
        }
        const int64_t predicted_publish_stamp_ns =
            control_stamp_ns + publication_lead_ns;
        int64_t candidate_command_stamp_ns = 0;
        /* Bridge packets are requested on a fixed 25 ms phase, and every
         * sensor message carries its associated request timestamp. A command
         * published before the next request is consumed at that request; use
         * the next request epoch after the measured publication-time estimate
         * rather than letting host callback jitter shift the plant phase. */
        if (!next_bridge_request_epoch(coherent_state.source_stamp_ns,
                predicted_publish_stamp_ns, &candidate_command_stamp_ns)) {
            report_control_fault(
                "MPC could not align command start to the bridge request epoch");
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
            if (open_loop_horizon_test_active_) {
                record_open_loop_observation_gaps_through(
                    open_loop_stage_at(control_ros_time.nanoseconds()),
                    "trajectory_projection_failed");
                return;
            }
            report_control_fault("MPC could not project uncertain pose");
            return;
        }
        if (!startup_path_validated_) {
            if (projection.distance > startup_path_max_distance_m_ ||
                std::abs(projection.heading_error) > startup_path_heading_tolerance_rad_) {
                report_control_fault("MPC startup pose is not yet aligned");
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
            if (!source_plant_handoff_valid_) {
                /* The first hidden-state sample can arrive after the matching
                 * odometry callback because these are separate DDS topics.
                 * Wait for the next source epoch; do not latch a permanent
                 * controller fault before MPC has ever had a usable plant
                 * state. Once initialized, a missing/stale plant remains a
                 * real fault and is handled below. */
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                    "MPC waiting for its first causal odometry plant snapshot");
                publish_waiting_for_plant_state();
                return;
            }
            report_control_fault(
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
            std::copy(unity_plant_snapshot.low_forward_speed_time_s.begin(),
                unity_plant_snapshot.low_forward_speed_time_s.end(),
                plant_state + MPC_PLANT_STATE_LOW_FORWARD_SPEED_TIME_FL);
            std::copy(unity_plant_snapshot.low_side_speed_time_s.begin(),
                unity_plant_snapshot.low_side_speed_time_s.end(),
                plant_state + MPC_PLANT_STATE_LOW_SIDE_SPEED_TIME_FL);
            source_time_plant.source_plant_initialized = 1;
            source_time_plant.source_suspension_history_initialized =
                unity_plant_snapshot.suspension_history_initialized ? 1 : 0;
            source_time_plant.source_imu_state_initialized = 1;
            source_time_plant.u = static_cast<float>(
                unity_plant_snapshot.longitudinal_velocity_mps);
            // IMU.cs publishes the Rigidbody's local angular velocity
            // directly (no gyro filter/state). Seed the causal rollout from
            // that permitted measurement; all later horizon yaw-rate change
            // comes from the vehicle plant equations.
            source_time_plant.r = static_cast<float>(
                unity_plant_snapshot.gyro_yaw_rate_radps);
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
            candidate_command_stamp_ns,
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
            if (open_loop_horizon_test_active_) {
                record_open_loop_observation_gaps_through(
                    open_loop_stage_at(control_ros_time.nanoseconds()),
                    control_time_status_name(prediction_status));
                return;
            }
            report_control_fault("MPC command-time state prediction is uncertain");
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

        if (open_loop_horizon_test_active_) {
            const int stage = open_loop_stage_at(
                control_ros_time.nanoseconds());
            record_open_loop_observation_gaps_through(stage, "no_sample");
            if (stage > open_loop_horizon_last_sample_stage_ &&
                (stage < open_loop_horizon_test_plan_.horizon ||
                 open_loop_horizon_all_outputs_sent_)) {
                publish_open_loop_horizon_sample(stage, state, coherent_state,
                    command_time_state, command_time_projection,
                    control_ros_time.nanoseconds());
                if (!open_loop_horizon_test_active_) {
                    // A diagnostic failure must not fall through into normal
                    // MPC output or synthesize a braking command.
                    return;
                }
                open_loop_horizon_last_sample_stage_ = stage;
            }
            if (stage >= open_loop_horizon_test_plan_.horizon &&
                open_loop_horizon_all_outputs_sent_) {
                finish_open_loop_horizon_test(true);
            }
            return;
        }
        if (open_loop_horizon_test_finished_) {
            // The run supervisor stops the simulator after observing this
            // terminal event; MPC does not issue an unplanned brake command.
            return;
        }

        MpcRtiCycleResult_t result{};
        const auto solve_start = std::chrono::steady_clock::now();
        MpcRtiCycleConfiguration_t cycle_configuration = rti_config_;
        cycle_configuration.absolute_deadline_monotonic_us =
            callback_deadline_us;
        MpcRtiCycleStatus_t status = mpc_rti_solve_cycle(
            &state, last_projected_s_, trajectory_.data(), trajectory_.size(),
            track_length_m_, TIME_STEP_SECONDS, PREDICTION_HORIZON,
            &cycle_configuration, &rti_memory_, &result);
        const auto solve_finish = std::chrono::steady_clock::now();
        const double solve_us = std::chrono::duration<double, std::micro>(
            solve_finish - solve_start).count();
        if (status == MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET &&
            !rti_memory_.nominal.valid &&
            std::abs(state.plant.u) <= kStationaryStartupRetrySpeedMps &&
            std::abs(state.plant.v) <= kStationaryStartupRetrySpeedMps &&
            std::abs(state.plant.r) <=
                kStationaryStartupRetryYawRateRadps) {
            /* No command has been accepted, so there is no safe prior action
             * to reuse. At spawn the vehicle can remain stationary for one
             * 25 ms tick and retry the cold solve; never turn this into a
             * moving-state or late-command fallback. */
            record_cycle_health(status, result, solve_us);
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "Startup solve deferred: Jacobian build consumed the reserved "
                "ADMM budget; retrying next control tick while stationary");
            if (diagnostics_pub_)
                publish_cycle_diagnostics(state, coherent_state,
                    control_time_prediction, control_prediction_us,
                    last_projected_s_, source_dt, result, solve_us,
                    control_ros_time.nanoseconds(), candidate_command_stamp_ns,
                    callback_steady_ns, synchronize_steady_ns);
            return;
        }
        if (status == MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET) {
            /* A solver-budget miss is retryable only at a motionless spawn.
             * During motion, route it through the normal fault/reuse path. */
            status = MPC_RTI_CYCLE_REJECTED_SOLVER;
            result.status = status;
        }
        if (status != MPC_RTI_CYCLE_ACCEPTED_OPTIMAL &&
            status != MPC_RTI_CYCLE_ACCEPTED_FEASIBLE &&
            status != MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH) {
            const char *candidate_prefix_failure_reason = nullptr;
            if (try_publish_candidate_prefix(state, last_projected_s_,
                    callback_steady_ns, control_stamp_ns,
                    candidate_command_stamp_ns, &result,
                    &candidate_prefix_failure_reason)) {
                record_cycle_health(
                    MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX,
                    result, solve_us);
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                    "Full MPC horizon rejected; published its exact-validated "
                    "next 25 ms action after the prior converged plan was "
                    "unavailable or failed revalidation");
                if (diagnostics_pub_)
                    publish_cycle_diagnostics(state, coherent_state,
                        control_time_prediction, control_prediction_us,
                        last_projected_s_, source_dt, result, solve_us,
                        control_ros_time.nanoseconds(),
                        candidate_command_stamp_ns, callback_steady_ns,
                        synchronize_steady_ns);
                return;
            }
            if (candidate_prefix_timing_rejected_) {
                record_cycle_health(status, result, solve_us);
                if (diagnostics_pub_)
                    publish_cycle_diagnostics(state, coherent_state,
                        control_time_prediction, control_prediction_us,
                        last_projected_s_, source_dt, result, solve_us,
                        control_ros_time.nanoseconds(),
                        candidate_command_stamp_ns, callback_steady_ns,
                        synchronize_steady_ns);
                report_control_fault(candidate_prefix_failure_reason, false);
                publish_cycle_failure(candidate_prefix_failure_reason);
                return;
            }
            const char * reuse_failure_reason = nullptr;
            if (try_publish_prior_plan(state, last_projected_s_,
                    callback_steady_ns, control_stamp_ns,
                    candidate_command_stamp_ns,
                    &result, &reuse_failure_reason)) {
                record_cycle_health(MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN,
                    result, solve_us);
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                    "MPC solve status=%s; published next action from the "
                    "last accepted plan after exact current-state validation",
                    cycle_status_name(status));
                if (diagnostics_pub_)
                    publish_cycle_diagnostics(state, coherent_state,
                        control_time_prediction, control_prediction_us,
                        last_projected_s_, source_dt, result, solve_us,
                        control_ros_time.nanoseconds(),
                        candidate_command_stamp_ns, callback_steady_ns,
                        synchronize_steady_ns);
                return;
            }
            record_cycle_health(status, result, solve_us);
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC cycle rejected: %s ADMM=%d iterations=%d "
                "residual=(%.6g,%.6g) "
                "regularization=(%.6g,%d) nonsmooth_columns=%d "
                "rho=(%.4g->%.4g,%.4g->%.4g) rho_changes=%d "
                "nonlinear_failure=(reason=%d,stage=%d) "
                "state=(ey=%.3f,epsi=%.3f,u=%.3f,r=%.3f)",
                cycle_status_name(status), result.admm_status,
                result.solver_iterations,
                result.primal_residual, result.dual_residual,
                result.maximum_regularization, result.regularization_count,
                result.nonsmooth_jacobian_columns,
                result.rho_start, result.rho_final,
                result.rho_u_start, result.rho_u_final,
                result.rho_change_count,
                result.nonlinear_failure_reason,
                result.nonlinear_failure_stage,
                state.plant.e_y, state.plant.e_psi, state.plant.u,
                state.plant.r);
            if (diagnostics_pub_)
                publish_cycle_diagnostics(state, coherent_state,
                    control_time_prediction, control_prediction_us,
                    last_projected_s_, source_dt, result, solve_us,
                    control_ros_time.nanoseconds(), candidate_command_stamp_ns,
                    callback_steady_ns,
                    synchronize_steady_ns);
            report_control_fault(
                reuse_failure_reason ? reuse_failure_reason :
                    (candidate_prefix_failure_reason
                        ? candidate_prefix_failure_reason
                        : "MPC RTI cycle had no exact-feasible candidate"));
            return;
        }

        /* A numerically accepted plan is not publishable if its command-time
         * state is already stale or if state acquisition through validation
         * crossed the 20 ms compute budget. Command-start prediction age is
         * reported separately and does not invalidate an exact-feasible plan.
         * Full diagnostics are serialized after command publication so a
         * debug recording cannot consume the control deadline. */
        const char * timing_fault_reason = nullptr;
        if (enabled_ && !cycle_time_is_valid(callback_steady_ns,
                candidate_command_stamp_ns, &timing_fault_reason)) {
            if (diagnostics_pub_)
                publish_cycle_diagnostics(state, coherent_state,
                    control_time_prediction, control_prediction_us,
                    last_projected_s_, source_dt, result, solve_us,
                    control_ros_time.nanoseconds(), candidate_command_stamp_ns,
                    callback_steady_ns, synchronize_steady_ns);
            record_cycle_health(status, result, solve_us);
            report_control_fault(timing_fault_reason, false);
            publish_cycle_failure(timing_fault_reason);
            return;
        }
        record_cycle_health(status, result, solve_us);
        if (enabled_ && open_loop_horizon_test_enabled_ &&
            !open_loop_horizon_test_triggered_ &&
            state.plant.u >= open_loop_horizon_test_min_speed_mps_ &&
            rti_memory_.nominal.valid &&
            rti_memory_.nominal.horizon == PREDICTION_HORIZON) {
            open_loop_horizon_test_plan_ = rti_memory_.nominal;
            ++open_loop_horizon_test_plan_id_;
            open_loop_horizon_test_triggered_ = true;
            open_loop_horizon_test_active_ = true;
            open_loop_horizon_test_next_stage_ = 1;
            open_loop_horizon_start_stamp_ns_ =
                candidate_command_stamp_ns;
            open_loop_horizon_last_sample_stage_ = 0;
            open_loop_horizon_all_outputs_sent_ = false;
            open_loop_horizon_terminal_waiting_ = false;
            publish_open_loop_horizon_plan(
                state, candidate_command_stamp_ns);
            publish_open_loop_horizon_sample(0, state, coherent_state,
                command_time_state, command_time_projection,
                candidate_command_stamp_ns);
            if (!open_loop_horizon_test_active_) {
                return;
            }
            if (!cycle_time_is_valid(callback_steady_ns,
                    candidate_command_stamp_ns, &timing_fault_reason)) {
                if (diagnostics_pub_)
                    publish_cycle_diagnostics(state, coherent_state,
                        control_time_prediction, control_prediction_us,
                        last_projected_s_, source_dt, result, solve_us,
                        control_ros_time.nanoseconds(),
                        candidate_command_stamp_ns, callback_steady_ns,
                        synchronize_steady_ns);
                report_control_fault(timing_fault_reason, false);
                publish_cycle_failure(timing_fault_reason);
                return;
            }
            RCLCPP_WARN(get_logger(),
                "Latched MPC plan_id=%llu at %.3f m/s; next %d command "
                "outputs will be emitted on the fixed 25 ms clock",
                static_cast<unsigned long long>(open_loop_horizon_test_plan_id_),
                state.plant.u, PREDICTION_HORIZON);
            const int64_t command_stamp_ns =
                publish_open_loop_horizon_command(0);
            if (command_stamp_ns == 0) {
                report_control_fault(
                    "latched MPC plan starts outside the drive command envelope");
                return;
            }
            last_command_publication_lead_ns_ = command_stamp_ns >=
                control_stamp_ns ? command_stamp_ns - control_stamp_ns : 0;
            open_loop_horizon_timer_ = create_wall_timer(
                std::chrono::milliseconds(25),
                std::bind(&MpcControllerNode::open_loop_horizon_timer_callback,
                    this), control_callback_group_);
            if (diagnostics_pub_)
                publish_cycle_diagnostics(state, coherent_state,
                    control_time_prediction, control_prediction_us,
                    last_projected_s_, source_dt, result, solve_us,
                    control_ros_time.nanoseconds(), candidate_command_stamp_ns,
                    callback_steady_ns, synchronize_steady_ns);
            return;
        }
        if (enabled_) {
            const int64_t command_stamp_ns = publish_command(
                result.published_steering_command, result.published_throttle);
            if (command_stamp_ns == 0) {
                if (diagnostics_pub_)
                    publish_cycle_diagnostics(state, coherent_state,
                        control_time_prediction, control_prediction_us,
                        last_projected_s_, source_dt, result, solve_us,
                        control_ros_time.nanoseconds(),
                        candidate_command_stamp_ns, callback_steady_ns,
                        synchronize_steady_ns);
                report_control_fault(
                    "accepted MPC action violates the positive-throttle command envelope");
                return;
            }
            last_command_publication_lead_ns_ = command_stamp_ns >=
                control_stamp_ns ? command_stamp_ns - control_stamp_ns : 0;
            prior_plan_control_index_ = 0;
            consecutive_prior_plan_actions_ = 0;
            consecutive_candidate_prefix_actions_ = 0;
        }
        if (diagnostics_pub_)
            publish_cycle_diagnostics(state, coherent_state,
                control_time_prediction, control_prediction_us,
                last_projected_s_, source_dt, result, solve_us,
                control_ros_time.nanoseconds(), candidate_command_stamp_ns,
                callback_steady_ns, synchronize_steady_ns);
    }

    bool enabled_{};
    bool trajectory_loaded_{};
    bool startup_path_validated_{};
    bool progress_initialized_{};
    bool state_initialized_{};
    bool control_fault_latched_{};
    bool waiting_for_actuator_{};
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
    double end_to_end_cycle_budget_ms_{20.0};
    double last_publication_gate_elapsed_ms_{-1.0};
    bool command_published_this_cycle_{};
    bool prior_plan_revalidation_attempted_{};
    bool prior_plan_exact_valid_{};
    bool prior_plan_timing_rejected_{};
    double prior_plan_validation_us_{};
    int prior_plan_control_index_{};
    int consecutive_prior_plan_actions_{};
    int consecutive_candidate_prefix_actions_{};
    MpcRtiNominal_t recovered_action_revalidated_nominal_{};
    int prior_plan_validated_horizon_{};
    int prior_plan_validated_control_index_{-1};
    int prior_plan_actions_remaining_{};
    int prior_plan_validation_status_{-1};
    int prior_plan_validation_failure_stage_{-1};
    bool candidate_prefix_revalidation_attempted_{};
    bool candidate_prefix_exact_valid_{};
    bool candidate_prefix_timing_rejected_{};
    bool candidate_prefix_solver_converged_{};
    bool candidate_prefix_iteration_limited_{};
    double candidate_prefix_validation_us_{};
    int candidate_prefix_validation_horizon_{};
    int candidate_prefix_validation_status_{-1};
    int candidate_prefix_validation_failure_stage_{-1};
    double max_command_start_lateness_ms_{2.0};
    double track_length_m_{};
    double startup_progress_m_{};
    double last_projected_s_{};
    std::string unity_plant_state_topic_;
    std::string open_loop_horizon_test_topic_;
    bool open_loop_horizon_test_enabled_{};
    double open_loop_horizon_test_min_speed_mps_{6.0};
    bool open_loop_horizon_test_triggered_{};
    bool open_loop_horizon_test_active_{};
    bool open_loop_horizon_test_finished_{};
    bool open_loop_horizon_all_outputs_sent_{};
    bool open_loop_horizon_terminal_waiting_{};
    int open_loop_horizon_test_next_stage_{};
    int open_loop_horizon_last_sample_stage_{-1};
    int64_t open_loop_horizon_start_stamp_ns_{};
    int64_t open_loop_horizon_terminal_wait_start_ns_{};
    std::uint64_t open_loop_horizon_test_plan_id_{};
    MpcRtiNominal_t open_loop_horizon_test_plan_{};
    bool source_plant_handoff_valid_{};
    bool unity_plant_state_used_{};
    int64_t source_plant_handoff_stamp_ns_{};
    /* Until the first command is published, seed from the measured practice
     * node-cycle median; subsequent values are measured callback-to-publish. */
    int64_t last_command_publication_lead_ns_{};
    double source_plant_handoff_progress_m_{};
    MpcModelState_t source_plant_handoff_state_{};
    std::size_t last_closest_index_{};
    rclcpp::Time last_source_stamp_{0, 0, RCL_ROS_TIME};
    MpcRtiCycleConfiguration_t rti_config_{};
    MpcRtiMemory_t rti_memory_{};
    std::vector<Waypoint> trajectory_;
    MpcStateSynchronizer state_synchronizer_;
    MpcSyncStatus latest_odom_ingest_status_{MpcSyncStatus::kMissingOdom};
    std::uint64_t odom_generation_{};
    std::uint64_t last_control_odom_generation_{};
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
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr
        open_loop_horizon_test_pub_;
    rclcpp::CallbackGroup::SharedPtr control_callback_group_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::TimerBase::SharedPtr open_loop_horizon_timer_;
    std::uint64_t control_cycles_{};
    std::uint64_t accepted_optimal_{};
    std::uint64_t accepted_feasible_{};
    std::uint64_t accepted_line_search_{};
    std::uint64_t accepted_candidate_prefix_{};
    std::uint64_t accepted_prior_plan_{};
    std::uint64_t control_fault_cycles_{};
    int max_solver_iterations_seen_{};
    double max_solve_us_{};
};

}  // namespace f1tenth_mpc

RCLCPP_COMPONENTS_REGISTER_NODE(f1tenth_mpc::MpcControllerNode)

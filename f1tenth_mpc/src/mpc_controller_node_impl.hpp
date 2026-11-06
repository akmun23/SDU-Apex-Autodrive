#pragma once
// Private controller class. Diagnostics and recovery have separate units.
// The MPC consumes synchronized localization and permitted sensor inputs,
// then directly publishes steering and normalized throttle.

#include "mpc_diagnostic_worker.hpp"
#include "mpc_rti.h"
#include "mpc_control_time_predictor.hpp"
#include "mpc_drive_command.hpp"
#include "mpc_publication_deadline.hpp"
#include "mpc_solver_memory_transaction.hpp"
#include "mpc_state_synchronizer.hpp"
#include "mpc_trajectory_io.hpp"
#include "mpc_speed_envelope.hpp"
#include "mpc_robust_contour.hpp"
#include "mpc_world_footprint.hpp"
#include "unity_source_plant.hpp"
#include "vehicle_model.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <sdu_apex_msgs/msg/drive_command.hpp>
#include <sdu_apex_msgs/msg/unity_plant_state.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace f1tenth_mpc {
namespace controller_detail {

constexpr std::uint32_t kUnityPlantStateSchemaVersion = 11;
constexpr int64_t kBridgeControlPeriodNs = 25'000'000;
/* Only a truly stationary, plan-less spawn may wait for the next solve tick
 * when cold Jacobian construction consumes the reserved ADMM budget. */
constexpr float kStationaryStartupRetrySpeedMps = 0.05f;
constexpr float kStationaryStartupRetryYawRateRadps = 0.05f;

inline double clamp(double value, double lower, double upper)
{
    return std::min(std::max(value, lower), upper);
}

using Waypoint = MpcTrajectorySample_t;

inline bool next_bridge_request_epoch(int64_t source_request_stamp_ns,
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

}  // namespace controller_detail
using namespace controller_detail;

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
        // Optional load-time curvature/braking speed envelope. Zero disables;
        // this does not override any full-horizon physical collision limits.
        curvature_speed_envelope_lateral_accel_mps2_ =
            declare_parameter<double>("curvature_speed_envelope_lateral_accel_mps2", 0.0);
        curvature_speed_envelope_braking_mps2_ =
            declare_parameter<double>("curvature_speed_envelope_braking_mps2", 5.0);
        if (!std::isfinite(curvature_speed_envelope_lateral_accel_mps2_) ||
            curvature_speed_envelope_lateral_accel_mps2_ < 0.0 ||
            curvature_speed_envelope_lateral_accel_mps2_ > 40.0 ||
            !std::isfinite(curvature_speed_envelope_braking_mps2_) ||
            curvature_speed_envelope_braking_mps2_ <= 0.0 ||
            curvature_speed_envelope_braking_mps2_ > 30.0)
            throw std::runtime_error("invalid curvature speed envelope limits");
        path_frame_ = declare_parameter<std::string>("path_frame", "map");
        // Experimental physical map authority. Disabled by default. Unknown
        // cells and a missing/invalid occupancy map never authorize driving.
        world_map_authority_enabled_ = declare_parameter<bool>(
            "world_map_authority_enabled", false);
        world_map_rejection_only_ = declare_parameter<bool>(
            "world_map_rejection_only", false);
        world_map_additional_uncertainty_m_ = std::clamp(
            declare_parameter<double>(
                "world_map_additional_uncertainty_m", 0.005), 0.0, 0.10);
        world_map_topic_ = declare_parameter<std::string>(
            "world_map_topic", "/map");
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
        state_stale_fault_grace_s_ = std::clamp(
            declare_parameter<double>("state_stale_fault_grace_s", 0.5),
            0.0, 2.0);
        state_freshness_guard_ = MpcStateFreshnessGuard(
            state_stale_fault_grace_s_);
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
            diagnostics_worker_ =
                std::make_unique<DiagnosticPublisherWorker>(diagnostics_pub_);
        }
        rti_config_.capture_diagnostic_outputs =
            diagnostics_pub_ != nullptr;
        // State is time-sensitive: if executor/DDS scheduling falls behind,
        // drop old samples instead of replaying them as current control input.
        control_callback_group_ = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive, false);
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
        bridge_request_command_sub_ =
            create_subscription<sdu_apex_msgs::msg::DriveCommand>(
                "/odom/bridge_request_command", rclcpp::QoS(100),
                [this](sdu_apex_msgs::msg::DriveCommand::ConstSharedPtr message) {
                    const rclcpp::Time stamp(message->header.stamp,
                        get_clock()->get_clock_type());
                    std::lock_guard<std::mutex> lock(command_history_mutex_);
                    command_history_.push({stamp.nanoseconds(),
                        message->steering_angle_rad,
                        message->throttle_normalized, true});
                });
        if (world_map_authority_enabled_) {
            // Nav2 publishes a transient-local map; only the subscriber
            // callback builds EDT. Never allocate or recompute it in MPC.
            world_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
                world_map_topic_,rclcpp::QoS(1).reliable().transient_local(),
                std::bind(&MpcControllerNode::world_map_callback,this,
                          std::placeholders::_1));
        }
        auto declare_weight = [this](const char * name, float fallback) {
            return static_cast<float>(declare_parameter<double>(name, fallback));
        };
        /* Keep parameter defaults identical to the canonical competition
         * YAML. A missing/renamed config must not silently select a different
         * steering/yaw objective than saved-state replay and production. */
        rti_config_.model.weight_e_y = declare_weight("weight_e_y", 4000.0f);
        rti_config_.model.weight_e_psi = declare_weight("weight_e_psi", 1500.0f);
        rti_config_.model.tracking_lateral_tolerance_m =
            declare_weight("tracking_lateral_tolerance_m", 0.0f);
        rti_config_.model.tracking_heading_tolerance_rad =
            declare_weight("tracking_heading_tolerance_rad", 0.0f);
        rti_config_.model.weight_u = declare_weight("weight_u", 35.0f);
        rti_config_.model.weight_u_overspeed = declare_weight(
            "weight_u_overspeed", 20.0f);
        rti_config_.model.overspeed_curvature_gain_m = declare_weight(
            "overspeed_curvature_gain_m", 0.5f);
        rti_config_.model.underspeed_curvature_gain_m = declare_weight(
            "underspeed_curvature_gain_m", 0.0f);
        rti_config_.model.weight_longitudinal_slip = declare_weight(
            "weight_longitudinal_slip", 5.0f);
        rti_config_.model.longitudinal_slip_curvature_gain_m = declare_weight(
            "longitudinal_slip_curvature_gain_m", 0.0f);
        rti_config_.model.weight_v = declare_weight("weight_v", 5.0f);
        rti_config_.model.weight_r = declare_weight("weight_r", 7.8125f);
        rti_config_.model.weight_steering_command = declare_weight(
            "weight_steering_command", 3.0f);
        rti_config_.model.weight_steering_target = declare_weight(
            "weight_steering_target", 5.0f);
        geometry_only_raceline_reference_ =
            declare_parameter<bool>("geometry_only_raceline_reference",false);
        rti_config_.model.weight_throttle_command = declare_weight(
            "weight_throttle_command", 0.1f);
        rti_config_.model.weight_steering_slew = declare_weight(
            "weight_steering_slew", 2000.0f);
        rti_config_.model.weight_steering_correction_variation =
            declare_weight("weight_steering_correction_variation",0.0f);
        rti_config_.model.steering_correction_variation_epsilon_rad =
            static_cast<float>(declare_parameter<double>(
                "steering_correction_variation_epsilon_rad",0.02));
        if (!(rti_config_.model.steering_correction_variation_epsilon_rad
                >=0.005f &&
              rti_config_.model.steering_correction_variation_epsilon_rad
                <=0.10f))
            throw std::runtime_error(
                "steering_correction_variation_epsilon_rad must be in [0.005,0.10]");
        rti_config_.model.weight_throttle_slew = declare_weight(
            "weight_throttle_slew", 50.0f);
        rti_config_.model.startup_nominal_throttle_seed = static_cast<float>(
            declare_parameter<double>("startup_nominal_throttle_seed", 0.25));
        rti_config_.model.terminal_multiplier = declare_weight(
            "terminal_multiplier", 3.0f);
        // Source-exact nonlinear SQP is the only controller, not a plugin.
        rti_config_.source_exact_nlp_live=1;
        rti_config_.source_exact_nlp_constrained_qp=1;
        rti_config_.source_exact_nlp_compact_qp=1;
        rti_config_.source_exact_nlp_full_step_sqp=1;
        rti_config_.capture_source_benchmark=
            declare_parameter<bool>("capture_source_benchmark",false)?1:0;
        rti_config_.source_exact_nlp_qp_soft_weight=
            static_cast<float>(declare_parameter<double>(
                "source_exact_nlp_qp_soft_weight",1000000.));
        rti_config_.source_exact_nlp_qp_regularization=
            static_cast<float>(declare_parameter<double>(
                "source_exact_nlp_qp_regularization",1000.));
        rti_config_.source_exact_nlp_iterations=std::clamp<int>(
            declare_parameter<int>("source_exact_nlp_iterations",3),1,8);
        rti_config_.source_exact_nlp_extra_clearance_m=static_cast<float>(
            declare_parameter<double>(
                "source_exact_nlp_extra_clearance_m",0.03));
        if(!std::isfinite(
               rti_config_.source_exact_nlp_qp_soft_weight) ||
           rti_config_.source_exact_nlp_qp_soft_weight<=0.f ||
           rti_config_.source_exact_nlp_qp_soft_weight>1.e9f)
            throw std::runtime_error(
                "source_exact_nlp_qp_soft_weight must be in (0,1e9]");
        if(!std::isfinite(
               rti_config_.source_exact_nlp_qp_regularization) ||
           rti_config_.source_exact_nlp_qp_regularization<=0.f ||
           rti_config_.source_exact_nlp_qp_regularization>100000.f)
            throw std::runtime_error(
                "source_exact_nlp_qp_regularization must be in (0,100000]");
        if(!std::isfinite(
               rti_config_.source_exact_nlp_extra_clearance_m) ||
           rti_config_.source_exact_nlp_extra_clearance_m<0.f ||
           rti_config_.source_exact_nlp_extra_clearance_m>.1f)
            throw std::runtime_error(
                "source_exact_nlp_extra_clearance_m must be in [0,0.1]");
        rti_config_.model.max_speed_mps = static_cast<float>(max_speed_mps_);
        rti_config_.model.active_speed_ceiling_mps =
            static_cast<float>(max_speed_mps_);
        actuator_steering_limit_rad_ = declare_parameter<double>(
            "max_steering_rad", SOURCE_MAX_STEERING_RAD);
        rti_config_.model.max_steering_rad = static_cast<float>(
            actuator_steering_limit_rad_);
        // The source NLP evaluates original actuator command-rate
        // inequalities even though the retired incremental Riccati QP
        // is gone. These two limits MUST NOT remain zero-initialized:
        // zero makes every nonzero cold-start control source-infeasible.
        // Full command-domain differences are nonrestrictive, while
        // physical actuator rate/lag remains in the exact Unity plant.
        // These are NOT trust regions around an SQP iterate.
        rti_config_.model.max_steering_command_step_rad =
            2.f * rti_config_.model.max_steering_rad;
        rti_config_.model.max_throttle_command_step =
            MPC_THROTTLE_MAX - MPC_THROTTLE_MIN_DRIVE;
        rti_config_.model.corridor_margin_m = static_cast<float>(
            declare_parameter<double>("corridor_margin_m",
                                      MPC_DEFAULT_WALL_CLEARANCE_M));
        rti_config_.model.first_prediction_corridor_margin_m =
            static_cast<float>(declare_parameter<double>(
                "first_prediction_corridor_margin_m",
                rti_config_.model.corridor_margin_m));
        rti_config_.model.corridor_preview_halfwidth_m = static_cast<float>(
            declare_parameter<double>("corridor_preview_halfwidth_m", 0.10));
        rti_config_.model.reference_clearance_reserve_m = static_cast<float>(
            declare_parameter<double>("reference_clearance_reserve_m", 0.0));
        rti_config_.model.near_wall_tracking_reserve_m = static_cast<float>(
            declare_parameter<double>("near_wall_tracking_reserve_m", 0.0));
        rti_config_.model.weight_source_corridor_reserve =
            static_cast<float>(declare_parameter<double>(
                "weight_source_corridor_reserve",0.0));
        rti_config_.model.weight_source_lateral_slip =
            static_cast<float>(declare_parameter<double>(
                "weight_source_lateral_slip",0.0));
        robust_contour_extra_clearance_m_ = std::clamp(
            declare_parameter<double>(
                "robust_contour_extra_clearance_m",0.0),0.0,0.20);
        robust_contour_max_offset_m_ = std::clamp(
            declare_parameter<double>(
                "robust_contour_max_offset_m",0.10),0.0,0.20);
        robust_contour_smoothing_sigma_m_ = std::clamp(
            declare_parameter<double>(
                "robust_contour_smoothing_sigma_m",0.70),0.2,2.5);
        robust_contour_forward_preview_m_ = std::clamp(
            declare_parameter<double>(
                "robust_contour_forward_preview_m",0.45),0.0,2.0);
        rti_config_.model.nonlinear_corridor_tolerance_m = static_cast<float>(
            declare_parameter<double>("nonlinear_corridor_tolerance_m", 0.005));
        rti_config_.model.jacobian_threads = static_cast<int>(std::clamp<long>(
            declare_parameter<int>("jacobian_threads", 8), 0L,
            MPC_RTI_MAX_JACOBIAN_THREADS));
        const double rti_cycle_budget_ms = declare_parameter<double>(
            "rti_cycle_budget_ms", 22.0);
        if (!std::isfinite(rti_cycle_budget_ms) || rti_cycle_budget_ms < 0.0) {
            throw std::runtime_error(
                "rti_cycle_budget_ms must be finite and nonnegative");
        }
        rti_config_.max_cycle_time_us = rti_cycle_budget_ms * 1000.0;
        end_to_end_cycle_budget_ms_ = declare_parameter<double>(
            "end_to_end_cycle_budget_ms", 22.0);
        max_command_start_lateness_ms_ = declare_parameter<double>(
            "max_command_start_lateness_ms", 2.0);
        const double initial_command_publication_lead_ms =
            declare_parameter<double>(
                "initial_command_publication_lead_ms", 15.0);
        if (!f1tenth_mpc::mpc_publication_budget_valid(
                end_to_end_cycle_budget_ms_) ||
            !std::isfinite(max_command_start_lateness_ms_) ||
            max_command_start_lateness_ms_ < 0.0 ||
            !std::isfinite(initial_command_publication_lead_ms) ||
            initial_command_publication_lead_ms < 0.0) {
            throw std::runtime_error(
                "end-to-end budget must be positive and at most 22 ms; "
                "command-start lateness and initial command lead must be "
                "finite and nonnegative");
        }
        last_command_publication_lead_ns_ = static_cast<int64_t>(
            std::llround(initial_command_publication_lead_ms * 1.0e6));
        RCLCPP_INFO(get_logger(),
            "Source-exact full-step SQP: max_iterations=%d "
            "curvature=%.1f soft_physical_weight=%.1f "
            "clearance=%.3fm budget=%.1f/%.1fms "
            "full_command_range=[%.5f,%.5f]",
            rti_config_.source_exact_nlp_iterations,
            rti_config_.source_exact_nlp_qp_regularization,
            rti_config_.source_exact_nlp_qp_soft_weight,
            rti_config_.source_exact_nlp_extra_clearance_m,
            rti_config_.max_cycle_time_us/1000.,
            end_to_end_cycle_budget_ms_,
            rti_config_.model.max_steering_command_step_rad,
            rti_config_.model.max_throttle_command_step);
        mpc_rti_memory_reset(&rti_memory_);
        if (world_map_authority_enabled_) {
            rti_config_.model.world_sweep_check =
                &MpcControllerNode::world_map_sweep_check;
            rti_config_.model.world_sweep_context=this;
            rti_config_.model.world_map_rejection_only=
                world_map_rejection_only_?1:0;
        }
        trajectory_loaded_ = load_trajectory(trajectory_file_);
        if (!trajectory_loaded_) {
            RCLCPP_ERROR(get_logger(),
                "MPC has no valid trajectory; controller remains inhibited");
        }
        if (!enabled_) {
            RCLCPP_WARN(get_logger(),
                "MPC is inactive; no command publisher is created");
        } else {
            /* One persistent control owner reuses one OpenMP worker team.
             * Letting the component executor migrate this group between its
             * four threads created four separate eight-worker teams in live
             * captures. Sensor callbacks remain on the component executor. */
            control_timer_ = create_wall_timer(
                std::chrono::milliseconds(5),
                std::bind(&MpcControllerNode::control_timer_callback, this),
                control_callback_group_);
            rclcpp::ExecutorOptions options;
            options.context = get_node_base_interface()->get_context();
            control_executor_ =
                std::make_unique<rclcpp::executors::SingleThreadedExecutor>(options);
            control_executor_->add_callback_group(
                control_callback_group_, get_node_base_interface());
            // The source dual plant costs ~7 ms on a P-core but ~14 ms on
            // an E-core on the benchmark i5-13400F. Explicit affinity
            // binds ONLY this persistent control executor thread, never
            // the sensor callbacks or all ROS/Unity worker processes.
            const int cpu_affinity=
                declare_parameter<int>("control_thread_cpu_affinity",-1);
#if defined(__linux__)
            if(cpu_affinity>=0) {
                cpu_set_t allowed{};
                if(cpu_affinity>=CPU_SETSIZE ||
                   sched_getaffinity(0,sizeof(allowed),&allowed)!=0 ||
                   !CPU_ISSET(cpu_affinity,&allowed))
                    throw std::runtime_error(
                        "control_thread_cpu_affinity must be an allowed Linux CPU");
            }
#else
            if(cpu_affinity>=0)
                throw std::runtime_error(
                    "control_thread_cpu_affinity requires Linux");
#endif
            control_thread_ = std::thread([this,cpu_affinity] {
#if defined(__linux__)
                if(cpu_affinity>=0) {
                    cpu_set_t selected{};
                    CPU_ZERO(&selected);
                    CPU_SET(cpu_affinity,&selected);
                    const int rc=pthread_setaffinity_np(
                        pthread_self(),sizeof(selected),&selected);
                    if(rc!=0)
                        RCLCPP_ERROR(get_logger(),
                            "Failed to pin the control executor thread to CPU %d: error %d",
                            cpu_affinity,rc);
                    else
                        RCLCPP_INFO(get_logger(),
                            "Dedicated MPC control thread pinned to CPU %d",
                            cpu_affinity);
                }
#endif
                control_executor_->spin();
            });
        }
    }

    ~MpcControllerNode() override
    {
        if (control_timer_) control_timer_->cancel();
        if (control_executor_) control_executor_->cancel();
        if (control_thread_.joinable()) control_thread_.join();
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

        try {
            trajectory_ = read_mpc_controller_trajectory(input, &track_length_m_);
            if (curvature_speed_envelope_lateral_accel_mps2_ > 0.0) {
                const auto summary = apply_curvature_speed_envelope(
                    trajectory_, track_length_m_,
                    curvature_speed_envelope_lateral_accel_mps2_,
                    curvature_speed_envelope_braking_mps2_);
                RCLCPP_INFO(get_logger(),
                    "Curvature speed envelope ay=%.2f m/s2 brake=%.2f m/s2: "
                    "reduced %zu/%zu knots; range [%.2f, %.2f] m/s; "
                    "source witness invalidated=%s",
                    curvature_speed_envelope_lateral_accel_mps2_,
                    curvature_speed_envelope_braking_mps2_,
                    summary.reduced_points, trajectory_.size(),
                    summary.minimum_speed_mps, summary.maximum_speed_mps,
                    summary.witness_invalidated ? "true" : "false");
            }
            if (robust_contour_extra_clearance_m_>0.0) {
                const auto contour=f1tenth_mpc::precompute_robust_contour(
                    trajectory_,track_length_m_,
                    rti_config_.model.corridor_margin_m,
                    robust_contour_extra_clearance_m_,
                    robust_contour_max_offset_m_,
                    robust_contour_smoothing_sigma_m_,
                    robust_contour_forward_preview_m_);
                RCLCPP_INFO(get_logger(),
                    "Smooth robust contour: adjusted=%zu/%zu max_offset=%.3fm max_slope=%.4f",
                    contour.affected_knots,trajectory_.size(),
                    contour.max_abs_offset_m,contour.max_neighbor_slope);
            }
        } catch (const std::exception &error) {
            RCLCPP_ERROR(get_logger(), "Invalid MPC trajectory '%s': %s",
                         path.c_str(), error.what());
            trajectory_.clear();
            track_length_m_ = 0.0;
            return false;
        }
        if (geometry_only_raceline_reference_) {
            // The offline IPOPT witness is a feasibility certificate for a
            // DIFFERENT open-loop initial state. It is not a control policy.
            // Consume only its path geometry, speed and corridor boundaries;
            // all steering, throttle, lateral-slip, yaw and heading targets
            // are rebuilt from the vehicle model and geometric curvature.
            // Invalidate in-memory witness flags AFTER trajectory parsing
            // and contour preprocessing, without changing the source CSV.
            for (auto &point : trajectory_)
                point.source_witness_valid=0;
            RCLCPP_INFO(get_logger(),
                "Geometry-only raceline: ignored IPOPT state/control witness; "
                "tracking path, curvature, speed and wall boundaries");
        }
        RCLCPP_INFO(get_logger(),
            "Loaded MPC trajectory from '%s' with %zu points (%.2f m); "
            "controller geometry-only: IPOPT commands/state witness excluded",
            trajectory_file_.c_str(), trajectory_.size(), track_length_m_);
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

    int64_t publish_command(
        double steering, double throttle_normalized,
        int64_t callback_steady_ns,
        const char **fault_reason)
    {
        if (!drive_command_pub_) return 0;
        MpcDriveCommand bounded{};
        if (!bound_mpc_drive_command(
                steering, throttle_normalized,
                actuator_steering_limit_rad_, &bounded)) return 0;
        sdu_apex_msgs::msg::DriveCommand command;
        const rclcpp::Time command_time = now();
        command.header.stamp = command_time;
        command.header.frame_id = command_frame_;
        command.steering_angle_rad = bounded.steering_target_rad;
        command.throttle_normalized = bounded.throttle_normalized;

        /* Recheck adjacent to publication. Earlier preflight checks avoid
         * spending time on work that is already over budget; this wrapper
         * makes the actual publisher invocation conditional on the strict
         * callback-to-publication deadline. */
        if (!f1tenth_mpc::mpc_publish_before_deadline(
                callback_steady_ns, end_to_end_cycle_budget_ms_,
                &last_publication_gate_elapsed_ms_,
                [] { return steady_time_ns(); },
                [&] {
                    const int64_t begin = diagnostics_pub_ ? steady_time_ns() : 0;
                    drive_command_pub_->publish(command);
                    if (begin > 0) {
                        last_middleware_publish_end_ns_ = steady_time_ns();
                        last_middleware_publish_us_ = static_cast<double>(
                            last_middleware_publish_end_ns_ - begin) * 1.e-3;
                    }
                })) {
            if (fault_reason)
                *fault_reason =
                    "MPC cycle reached or exceeded its publication budget";
            RCLCPP_ERROR(get_logger(),
                "MPC candidate not published: callback-to-publication "
                "elapsed %.3f ms must remain below %.3f ms",
                last_publication_gate_elapsed_ms_, end_to_end_cycle_budget_ms_);
            return 0;
        }
        /* Lateness is already in the recorded diagnostics. Formatting and
         * logging it on the control owner adds no control/safety decision. */
        command_published_this_cycle_ = true;
        consecutive_no_publication_cycles_=0;
        last_published_command_stamp_ns_ = command_time.nanoseconds();
        return last_published_command_stamp_ns_;
    }

    void report_control_fault(const char *reason,
                              bool emit_diagnostic=true)
    {
        ++control_fault_cycles_;
        RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,"%s",reason);
        if(emit_diagnostic && diagnostics_pub_)
            publish_cycle_failure(reason);
    }

    bool cycle_time_is_valid(int64_t callback_steady_ns,
                             const char ** fault_reason)
    {
        const double end_to_end_ms = static_cast<double>(
            steady_time_ns() - callback_steady_ns) * 1.0e-6;
        last_publication_gate_elapsed_ms_ = end_to_end_ms;
        if (mpc_cycle_within_publication_deadline(
                end_to_end_ms, end_to_end_cycle_budget_ms_)) return true;

        if (fault_reason)
            *fault_reason =
                "MPC cycle reached or exceeded its publication budget";
        RCLCPP_ERROR(get_logger(),
            "MPC candidate not published: end-to-end cycle %.3f ms "
            "must remain below %.3f ms",
            end_to_end_ms, end_to_end_cycle_budget_ms_);
        return false;
    }

    void publish_cycle_failure(const char * reason)
    {
        if (!diagnostics_pub_) return;
        std::ostringstream json;
        json << "{\"status\":\"rejected\",\"reason\":\"" << reason
             << "\",\"source_stamp_ns\":" << last_source_stamp_.nanoseconds()
             << '}';
        enqueue_diagnostic_stream(std::move(json));
    }

    void publish_waiting_for_localization()
    {
        if (!diagnostics_pub_) return;
        enqueue_diagnostic_text("{\"status\":\"waiting_for_localization\","
            "\"reason\":\"initial_map_pose_not_received\"}");
    }

    void publish_waiting_for_plant_state()
    {
        if (!diagnostics_pub_) return;
        enqueue_diagnostic_text("{\"status\":\"waiting_for_plant_state\","
            "\"reason\":\"initial_causal_plant_snapshot_not_received\"}");
    }

    void enqueue_diagnostic_stream(std::ostringstream && json)
    {
        if (diagnostics_worker_)
            diagnostics_worker_->enqueue(std::move(json));
    }

    void enqueue_diagnostic_text(std::string message)
    {
        if (diagnostics_worker_)
            diagnostics_worker_->enqueue(std::move(message));
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
        enqueue_diagnostic_text("{\"status\":\"waiting_for_actuator\","
            "\"reason\":\"drive_command_subscription_not_matched\"}");
    }

    void record_cycle_health(MpcRtiCycleStatus_t status,
                             const MpcRtiCycleResult_t &result,
                             double solve_us)
    {
        ++control_cycles_;
        if(status==MPC_RTI_CYCLE_ACCEPTED_FEASIBLE)
            ++accepted_feasible_;
        max_solver_iterations_seen_=std::max(
            max_solver_iterations_seen_,
            result.source_exact_nlp_source_trial_evaluations);
        max_solve_us_=std::max(max_solve_us_,solve_us);
        if(control_cycles_%400==0)report_health("periodic");
    }

    void record_no_publication()
    {
        ++consecutive_no_publication_cycles_;
        maximum_no_publication_cycles_=std::max(maximum_no_publication_cycles_,
            consecutive_no_publication_cycles_);
    }

    void report_health(const char *phase)
    {
        RCLCPP_INFO(get_logger(),
            "Source SQP health (%s): cycles=%llu feasible=%llu "
            "publication_misses=%llu input_faults=%llu "
            "max_sqp_iterations=%d max_solve_us=%.1f",
            phase,
            static_cast<unsigned long long>(control_cycles_),
            static_cast<unsigned long long>(accepted_feasible_),
            static_cast<unsigned long long>(deadline_miss_cycles_),
            static_cast<unsigned long long>(control_fault_cycles_),
            max_solver_iterations_seen_,max_solve_us_);
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
        case MPC_RTI_CYCLE_ACCEPTED_VIABILITY_RECOVERY:
            return "accepted_viability_recovery";
        case MPC_RTI_CYCLE_DEFERRED_DEADLINE:
            return "deferred_deadline_recovery";
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
        case MPC_RTI_ROLLOUT_TERMINAL_VIABILITY: return "terminal_viability";
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
        int64_t callback_steady_ns, int64_t synchronize_steady_ns);

    static int world_map_sweep_check(
        void *context,double x0,double y0,double yaw0,
        double x1,double y1,double yaw1,double required_margin_m,
        float *minimum_slack_m)
    {
        if (!context || !minimum_slack_m) return 0;
        const auto *node=static_cast<const MpcControllerNode*>(context);
        // The control callback retains ONE immutable geometry snapshot
        // throughout its R1, R2, line searches and independently validated
        // recovery rollouts. A new map cannot silently change the rules
        // between stages. No per-sweep shared_ptr atomic acquisition.
        const auto &grid=node->world_sweep_snapshot_;
        if (!grid) return 0;
        // Physical source rear-axle body + exact occupancy EDT,
        // interpolated segment, and explicitly budgeted uncertainty.
        double slack=-INFINITY;
        const bool safe=grid->interpolated_sweep_passes(
            {x0,y0,yaw0},{x1,y1,yaw1},
            required_margin_m+node->world_map_additional_uncertainty_m_,
            &slack);
        *minimum_slack_m=static_cast<float>(slack);
        return safe?1:0;
    }

    void world_map_callback(
        const nav_msgs::msg::OccupancyGrid::SharedPtr message)
    {
        if (!message || message->header.frame_id!=path_frame_) {
            // Do not continue using an obsolete accepted map after the
            // provider starts publishing geometry for another frame.
            std::atomic_store(&world_grid_,
                std::shared_ptr<const MpcWorldFootprint>{});
            RCLCPP_ERROR_THROTTLE(get_logger(),*get_clock(),5000,
                "MPC map geometry not certified: occupancy frame mismatch");
            return;
        }
        const auto &pose=message->info.origin;
        const auto &q=pose.orientation;
        const double map_yaw=std::atan2(
            2.0*(q.w*q.z+q.x*q.y),1.0-2.0*(q.y*q.y+q.z*q.z));
        const auto grid=MpcWorldFootprint::from_occupancy(
            static_cast<int>(message->info.width),
            static_cast<int>(message->info.height),
            message->info.resolution,pose.position.x,pose.position.y,
            map_yaw,message->data);
        if (!grid) {
            std::atomic_store(&world_grid_,
                std::shared_ptr<const MpcWorldFootprint>{});
            RCLCPP_ERROR(get_logger(),
                "MPC map geometry not certified: invalid occupancy grid");
            return;
        }
        std::atomic_store(&world_grid_,grid);
        RCLCPP_INFO(get_logger(),
            "MPC map world/body authority ready: resolution=%.4fm "
            "extra uncertainty=%.3fm",grid->resolution(),
            world_map_additional_uncertainty_m_);
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

        if (!enabled_) return;
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
        control_callback_interval_ms_ = last_control_callback_start_ns_ > 0
            ? static_cast<double>(callback_steady_ns -
                last_control_callback_start_ns_) * 1.0e-6 : -1.0;
        last_control_callback_start_ns_ = callback_steady_ns;
        // Retain the last ISSUED header across callbacks. Clearing it here
        // disabled SolverMemoryTransaction::begin on every later callback.
        double cycle_budget_us = end_to_end_cycle_budget_ms_ * 1000.0;
        if (rti_config_.max_cycle_time_us > 0.0)
            cycle_budget_us = std::min(
                cycle_budget_us, rti_config_.max_cycle_time_us);
        const double callback_deadline_us =
            callback_start_monotonic_us + cycle_budget_us;
        last_publication_gate_elapsed_ms_ = -1.0;
        last_middleware_publish_us_ = NAN;
        last_command_history_us_ = NAN;
        last_middleware_publish_end_ns_ = 0;
        command_published_this_cycle_ = false;
        if (!enabled_) return;
        // Only this control thread writes the snapshot. The map subscription
        // atomically publishes a new immutable grid, which will become
        // authoritative on the NEXT control cycle. This pins every sweep
        // of one two-pass SQP to the identical map and removes repeated
        // shared_ptr atomics from the nonlinear simulation inner loop.
        if (world_map_authority_enabled_) {
            world_sweep_snapshot_ = std::atomic_load(&world_grid_);
            if (!world_sweep_snapshot_) {
                RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,
                    "MPC awaiting authoritative occupancy map; no commands");
                return;
            }
        }

        const auto handle_stale_state = [this](
            const MpcSynchronizedState &stale_state,
            int64_t control_ros_stamp_ns,
            int64_t control_steady_stamp_ns) {
            const MpcFreshnessDecision decision = state_freshness_guard_.observe(
                MpcSyncStatus::kStateTooOld, control_steady_stamp_ns);
            if (decision == MpcFreshnessDecision::kWaitForFreshState) {
                const double stale_wait_s =
                    state_freshness_guard_.stale_duration_s(
                        control_steady_stamp_ns);
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 250,
                    "MPC waiting for fresh legal state after transport delay "
                    "(sample_age=%.1f ms, wait=%.1f/%.1f ms); retaining the "
                    "last issued command",
                    stale_state.source_age_s * 1000.0,
                    stale_wait_s * 1000.0,
                    state_stale_fault_grace_s_ * 1000.0);
                constexpr int64_t kStaleDiagnosticPeriodNs = 100'000'000;
                if (diagnostics_worker_ &&
                    (last_stale_wait_diagnostic_steady_ns_ <= 0 ||
                     control_steady_stamp_ns -
                         last_stale_wait_diagnostic_steady_ns_ >=
                             kStaleDiagnosticPeriodNs)) {
                    std::ostringstream json;
                    json << "{\"status\":\"state_stale_wait\","
                         << "\"sync_status\":\"state_too_old\","
                         << "\"source_stamp_ns\":"
                         << stale_state.source_stamp_ns << ','
                         << "\"odom_source_stamp_ns\":"
                         << stale_state.odom_source_stamp_ns << ','
                         << "\"map_pose_source_stamp_ns\":"
                         << stale_state.map_pose_source_stamp_ns << ','
                         << "\"control_ros_stamp_ns\":"
                         << control_ros_stamp_ns << ','
                         << "\"source_age_s\":"
                         << stale_state.source_age_s << ','
                         << "\"stale_wait_s\":" << stale_wait_s << ','
                         << "\"stale_fault_grace_s\":"
                         << state_stale_fault_grace_s_ << '}';
                    diagnostics_worker_->enqueue(json.str());
                    last_stale_wait_diagnostic_steady_ns_ =
                        control_steady_stamp_ns;
                }
                return;
            }
            state_freshness_guard_.reset();
            report_control_fault(
                "legal vehicle state remained stale beyond configured grace");
        };

        const auto handle_epoch_wait = [&](MpcSyncStatus status) {
            const auto decision = state_freshness_guard_.observe(
                status, steady_time_ns());
            if (decision == MpcFreshnessDecision::kFault) {
                report_control_fault("odometry/localization epoch bracket unavailable beyond grace");
                return;
            }
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC waiting for matching odometry/localization epochs: %s",
                MpcStateSynchronizer::status_name(status));
        };

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
        if (odom_generation == last_control_odom_generation_) {
            if (state_initialized_) {
                const int64_t freshness_ros_ns = now().nanoseconds();
                const int64_t freshness_steady_ns = steady_time_ns();
                MpcSynchronizedState freshness_state;
                MpcSyncStatus freshness_status;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    freshness_status = state_synchronizer_.synchronize(
                        freshness_ros_ns, &freshness_state);
                }
                if (freshness_status == MpcSyncStatus::kStateTooOld) {
                    handle_stale_state(freshness_state, freshness_ros_ns,
                                       freshness_steady_ns);
                } else if (freshness_status == MpcSyncStatus::kOk) {
                    state_freshness_guard_.observe(
                        MpcSyncStatus::kOk, freshness_steady_ns);
                } else if (freshness_status == MpcSyncStatus::kMapPoseFutureOfOdom ||
                           freshness_status == MpcSyncStatus::kNoOdomBracket) {
                    handle_epoch_wait(freshness_status);
                } else {
                    report_control_fault(
                        "MPC legal state became unsynchronizable while waiting "
                        "for a new odometry sample");
                }
            }
            return;
        }

        if (!trajectory_loaded_) {
            report_control_fault("MPC requires a valid trajectory");
            return;
        }
        if (odom_status != MpcSyncStatus::kOk) {
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
            if (sync_status == MpcSyncStatus::kMapPoseFutureOfOdom ||
                sync_status == MpcSyncStatus::kNoOdomBracket) {
                // Retry this odometry generation when its independent pose
                // callback arrives. No solve may use nearest-epoch data.
                handle_epoch_wait(sync_status);
                return;
            }
            if (!state_initialized_ &&
                sync_status == MpcSyncStatus::kMissingMapPose) {
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                    "MPC waiting for its first localized map pose");
                publish_waiting_for_localization();
                return;
            }
            if (sync_status == MpcSyncStatus::kStateTooOld) {
                handle_stale_state(coherent_state,
                    control_ros_time.nanoseconds(), steady_time_ns());
                return;
            }
            state_freshness_guard_.reset();
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC waiting for legal state input: %s",
                MpcStateSynchronizer::status_name(sync_status));
            // Do not keep racing on stale odometry/localization. The direct
            // controller-fault path below publishes no replacement command.
            const std::string reason = std::string(
                "MPC rejected unsynchronized legal state: ") +
                MpcStateSynchronizer::status_name(sync_status);
            report_control_fault(reason.c_str());
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

        active_command_epoch_ns_ = candidate_command_stamp_ns;

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
        MpcCommandHistoryEntry source_request_command{};
        if (!command_history_snapshot.latest_at_or_before(
                coherent_state.source_stamp_ns, &source_request_command) ||
            source_request_command.stamp_ns != coherent_state.source_stamp_ns) {
            // Reliable DDS topics still have independent callback ordering.
            // Retry the same source epoch; publication is not application.
            handle_epoch_wait(MpcSyncStatus::kNoOdomBracket);
            return;
        }
        bridge_request_unchanged_command_age_ms_=double(source_request_command.stamp_ns-
            source_request_command.unchanged_since_ns)*1.e-6;
        state_freshness_guard_.observe(MpcSyncStatus::kOk, steady_time_ns());
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
        bool unity_plant_snapshot_usable =
            is_causal_plant_state_usable(
                unity_plant_snapshot_received,
                unity_plant_snapshot_stamp_ns,
                coherent_state.odom_source_stamp_ns,
                coherent_state.source_stamp_ns,
                state_extrapolation_max_s_);
        MpcPathProjection_t plant_epoch_projection = projection;
        if (unity_plant_snapshot_usable &&
            unity_plant_snapshot_stamp_ns != coherent_state.source_stamp_ns) {
            MpcSynchronizedState plant_epoch_state{};
            MpcSyncStatus plant_epoch_status;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                plant_epoch_status = state_synchronizer_.state_at_odometry_epoch(
                    unity_plant_snapshot_stamp_ns, &plant_epoch_state);
            }
            unity_plant_snapshot_usable = plant_epoch_status == MpcSyncStatus::kOk &&
                plant_epoch_state.map_pose_source_stamp_ns ==
                    coherent_state.map_pose_source_stamp_ns &&
                mpc_trajectory_project(trajectory_.data(), trajectory_.size(),
                    track_length_m_, plant_epoch_state.map_x,
                    plant_epoch_state.map_y, plant_epoch_state.map_yaw,
                    projection.segment, 160,
                    &plant_epoch_projection);
        }
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
            source_plant_progress_m = plant_epoch_projection.s;
            source_time_plant.e_y = static_cast<float>(
                plant_epoch_projection.lateral_error);
            source_time_plant.e_psi = static_cast<float>(
                plant_epoch_projection.heading_error);
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
            report_control_fault("MPC command-time state prediction is uncertain");
            return;
        }
        last_control_odom_generation_ = odom_generation;
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


        // One source-exact nonlinear SQP, one validated full-horizon
        // solution, one actuator publication. No prior-plan publication,
        // short prefix, alternate QP, recovery trajectory or brake fallback.
        if (rti_memory_.command_epoch_ns>0 &&
            rti_memory_.warm_start_controls_valid &&
            mpc_rti_command_epoch_shift(rti_memory_.command_epoch_ns,
                candidate_command_stamp_ns,TIME_STEP_SECONDS)<0 &&
            candidate_command_stamp_ns<rti_memory_.command_epoch_ns) {
            ++older_command_epoch_rejections_;
            RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,
                "Ignoring older MPC command epoch (%lld < %lld)",
                static_cast<long long>(candidate_command_stamp_ns),
                static_cast<long long>(rti_memory_.command_epoch_ns));
            return;
        }
        if(rti_memory_.command_epoch_ns>candidate_command_stamp_ns)
            ++same_stage_epoch_revisions_;
        const auto solve_begin=std::chrono::steady_clock::now();
        MpcRtiCycleConfiguration_t cycle_configuration=rti_config_;
        cycle_configuration.command_epoch_ns=candidate_command_stamp_ns;
        cycle_configuration.absolute_deadline_monotonic_us=
            callback_deadline_us;
        MpcRtiCycleResult_t result{};
        solver_memory_transaction_.begin(rti_memory_,
            candidate_command_stamp_ns,
            rti_memory_.nominal.valid &&
            last_published_command_stamp_ns_>0);
        const auto status=mpc_rti_solve_cycle(
            &state,last_projected_s_,trajectory_.data(),trajectory_.size(),
            track_length_m_,TIME_STEP_SECONDS,PREDICTION_HORIZON,
            &cycle_configuration,&rti_memory_,&result);
        const double solve_us=std::chrono::duration<double,std::micro>(
            std::chrono::steady_clock::now()-solve_begin).count();
        record_cycle_health(status,result,solve_us);
        if(status!=MPC_RTI_CYCLE_ACCEPTED_FEASIBLE) {
            record_no_publication();
            // An unsuccessful solve never issues an unchecked or stale
            // command. It also never latches a permanent fault, so the
            // SAME NLP can run again at the next synchronized epoch.
            solver_memory_transaction_.rollback(
                &rti_memory_,candidate_command_stamp_ns);
            RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,
                "Source SQP could not publish: status=%s, "
                "feasible=%d, SQP_steps=%d, source_evals=%d, "
                "clearance=%.4f m, wall=%.3f ms",
                cycle_status_name(status),
                result.source_exact_nlp_final_source_feasible,
                result.source_exact_nlp_accepted_iterations,
                result.source_exact_nlp_source_trial_evaluations,
                result.source_exact_nlp_final_original_margin_m,
                solve_us/1000.);
            if(diagnostics_pub_)
                publish_cycle_diagnostics(state,coherent_state,
                    control_time_prediction,control_prediction_us,
                    last_projected_s_,source_dt,result,solve_us,
                    control_ros_time.nanoseconds(),
                    candidate_command_stamp_ns,callback_steady_ns,
                    synchronize_steady_ns);
            return;
        }
        const char *failure_reason=nullptr;
        if(!cycle_time_is_valid(callback_steady_ns,&failure_reason)) {
            // Optimization is not complete until a command is published.
            // Roll back the non-issued warm-start transaction.
            if(!solver_memory_transaction_.rollback(
                   &rti_memory_,candidate_command_stamp_ns))
                mpc_rti_memory_reset(&rti_memory_);
            ++deadline_miss_cycles_;
            record_no_publication();
            result.status=MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET;
            result.source_solver_status=MPC_SOURCE_DEADLINE_EXCEEDED;
            if(diagnostics_pub_)
                publish_cycle_diagnostics(state,coherent_state,
                    control_time_prediction,control_prediction_us,
                    last_projected_s_,source_dt,result,solve_us,
                    control_ros_time.nanoseconds(),
                    candidate_command_stamp_ns,callback_steady_ns,
                    synchronize_steady_ns);
            return;
        }
        if(enabled_) {
            const int64_t issued_stamp=publish_command(
                result.published_steering_command,result.published_throttle,
                callback_steady_ns,&failure_reason);
            if(!issued_stamp) {
                if(!solver_memory_transaction_.rollback(
                       &rti_memory_,candidate_command_stamp_ns))
                    mpc_rti_memory_reset(&rti_memory_);
                const bool publication_deadline_missed=last_publication_gate_elapsed_ms_>=end_to_end_cycle_budget_ms_;
                if(publication_deadline_missed)++deadline_miss_cycles_;
                else ++control_fault_cycles_;
                record_no_publication();
                result.status=publication_deadline_missed?MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET:
                    MPC_RTI_CYCLE_REJECTED_SOLVER;
                result.source_solver_status=publication_deadline_missed?MPC_SOURCE_DEADLINE_EXCEEDED:
                    MPC_SOURCE_INVALID_MODEL;
                RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,
                    "Source SQP publication rejected: %s",
                    failure_reason?failure_reason:"invalid command or deadline");
                if(diagnostics_pub_)
                    publish_cycle_diagnostics(state,coherent_state,
                        control_time_prediction,control_prediction_us,
                        last_projected_s_,source_dt,result,solve_us,
                        control_ros_time.nanoseconds(),
                        candidate_command_stamp_ns,callback_steady_ns,
                        synchronize_steady_ns);
                return;
            }
            last_command_publication_lead_ns_=
                issued_stamp>=control_stamp_ns?
                    issued_stamp-control_stamp_ns:0;
        }
        solver_memory_transaction_.clear();
        if(diagnostics_pub_)
            publish_cycle_diagnostics(state,coherent_state,
                control_time_prediction,control_prediction_us,
                last_projected_s_,source_dt,result,solve_us,
                control_ros_time.nanoseconds(),candidate_command_stamp_ns,
                callback_steady_ns,synchronize_steady_ns);
    }

    bool enabled_{};
    bool trajectory_loaded_{};
    bool startup_path_validated_{};
    bool progress_initialized_{};
    bool state_initialized_{};
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
    double state_stale_fault_grace_s_{0.5};
    double pose_odom_max_skew_s_{};
    double source_dt_min_s_{};
    double source_dt_max_s_{};
    double startup_path_max_distance_m_{};
    double startup_path_heading_tolerance_rad_{};
    double actuator_steering_limit_rad_{SOURCE_MAX_STEERING_RAD};
    double end_to_end_cycle_budget_ms_{20.0};
    double last_publication_gate_elapsed_ms_{-1.0};
    double last_middleware_publish_us_{NAN};
    double last_command_history_us_{NAN};
    int64_t last_middleware_publish_end_ns_{};
    DiagnosticTimingSample last_diagnostic_timing_{};
    int64_t last_control_callback_start_ns_{};
    int64_t last_stale_wait_diagnostic_steady_ns_{};
    int64_t last_published_command_stamp_ns_{};
    uint64_t older_command_epoch_rejections_{};
    uint64_t same_stage_epoch_revisions_{};
    uint64_t consecutive_no_publication_cycles_{};
    uint64_t maximum_no_publication_cycles_{};
    double bridge_request_unchanged_command_age_ms_{NAN};
    double control_callback_interval_ms_{-1.0};
    bool command_published_this_cycle_{};
    double max_command_start_lateness_ms_{2.0};
    double track_length_m_{};
    double startup_progress_m_{};
    double last_projected_s_{};
    std::string unity_plant_state_topic_;
    bool geometry_only_raceline_reference_{};

    double curvature_speed_envelope_lateral_accel_mps2_{0.0};
    double curvature_speed_envelope_braking_mps2_{5.0};
    double robust_contour_extra_clearance_m_{};
    double robust_contour_max_offset_m_{0.10};
    double robust_contour_smoothing_sigma_m_{0.70};
    double robust_contour_forward_preview_m_{0.45};
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
    f1tenth_mpc::SolverMemoryTransaction solver_memory_transaction_{};
    int64_t active_command_epoch_ns_{};
    std::vector<Waypoint> trajectory_;
    MpcStateSynchronizer state_synchronizer_;
    MpcStateFreshnessGuard state_freshness_guard_{};
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
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr world_map_sub_;
    std::shared_ptr<const MpcWorldFootprint> world_grid_;
    // Control-thread-owned immutable map generation. It must never be
    // written from the asynchronous ROS map subscription callback.
    std::shared_ptr<const MpcWorldFootprint> world_sweep_snapshot_;
    bool world_map_authority_enabled_{};
    bool world_map_rejection_only_{};
    std::string world_map_topic_{"/map"};
    double world_map_additional_uncertainty_m_{0.005};
    rclcpp::Subscription<sdu_apex_msgs::msg::UnityPlantState>::SharedPtr
        unity_plant_state_sub_;
    rclcpp::Subscription<sdu_apex_msgs::msg::DriveCommand>::SharedPtr
        bridge_request_command_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr diagnostics_pub_;
    std::unique_ptr<DiagnosticPublisherWorker> diagnostics_worker_;
    rclcpp::CallbackGroup::SharedPtr control_callback_group_;
    std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> control_executor_;
    std::thread control_thread_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    std::uint64_t control_cycles_{};
    std::uint64_t accepted_feasible_{};
    std::uint64_t deadline_miss_cycles_{};
    std::uint64_t control_fault_cycles_{};
    int max_solver_iterations_seen_{};
    double max_solve_us_{};
};

}  // namespace f1tenth_mpc

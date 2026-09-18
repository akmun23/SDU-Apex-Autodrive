#define _POSIX_C_SOURCE 200809L

#include "mpc_rti.h"
#include "riccati_solver.h"
#include "mpc_state_synchronizer.hpp"
#include "replay_metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef MPC_RTI_REPLAY_TRAJECTORY_PATH
#error "MPC_RTI_REPLAY_TRAJECTORY_PATH must name the active raceline"
#endif

using f1tenth_mpc::MpcMapPoseAnchor;
using f1tenth_mpc::MpcOdomSample;
using f1tenth_mpc::MpcStateSynchronizer;
using f1tenth_mpc::MpcSyncConfig;
using f1tenth_mpc::MpcSyncStatus;
using f1tenth_mpc::MpcSynchronizedState;

namespace {

enum { kMaximumTrajectoryPoints = 4000 };

struct ReplayMetrics {
    std::size_t odometry_samples{};
    std::size_t synchronized_samples{};
    std::size_t projected_samples{};
    std::size_t solves{};
    std::size_t status_counts[7]{};
    std::size_t startup_path_rejections{};
    std::size_t projection_rejections{};
    std::size_t ordering_faults{};
    std::size_t source_gap_faults{};
    std::vector<double> solve_ms;
    std::vector<double> iterations;
    std::vector<double> final_rho;
    std::vector<double> final_rho_u;
    std::vector<double> primal_residual;
    std::vector<double> dual_residual;
    std::vector<double> state_age_ms;
    std::vector<double> minimum_clearance_m;
    std::vector<double> candidate_progress_error_m;
    std::vector<double> candidate_curvature_error_per_m;
    std::vector<double> candidate_left_bound_error_m;
    std::vector<double> candidate_right_bound_error_m;
    std::vector<double> first_action_steering_rate_delta;
    std::vector<double> first_action_target_rate_delta;
    double maximum_steering_rate{};
    double maximum_target_speed_rate{};
    double minimum_target_speed{std::numeric_limits<double>::infinity()};
    double maximum_target_speed{-std::numeric_limits<double>::infinity()};
    double maximum_steering_command{};
    double maximum_regularization{};
    std::size_t regularization_count{};
    std::size_t nonlinear_rollout_stages{};
    std::size_t diagnostic_original_residual_rejections{};
    std::size_t diagnostic_nonlinear_feasible{};
    std::size_t diagnostic_nonlinear_rejections{};
    std::size_t diagnostic_not_reached{};
    std::vector<double> source_dt_ms;
    std::size_t residual_limit_rejections{};
    std::size_t degraded_streak_rejections{};
    std::size_t quadratic_factorizations{};
    std::size_t rti2_triggers{};
    std::size_t rti2_selected{};
    std::size_t rti2_fallbacks{};
    std::size_t rti2_reason_counts[11]{};
};

void fail(const std::string &message)
{
    std::cerr << "replay error: " << message << '\n';
    std::exit(2);
}

std::vector<std::string> split_csv(const std::string &line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (c == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    fields.push_back(field);
    return fields;
}

bool json_number(const std::string &json, const char *key, double *value)
{
    const std::string token = std::string("\"") + key + "\"";
    const std::size_t position = json.find(token);
    if (position == std::string::npos) return false;
    const std::size_t colon = json.find(':', position + token.size());
    if (colon == std::string::npos) return false;
    char *end = nullptr;
    const double parsed = std::strtod(json.c_str() + colon + 1, &end);
    if (end == json.c_str() + colon + 1 || !std::isfinite(parsed)) return false;
    *value = parsed;
    return true;
}

bool load_trajectory(std::vector<MpcTrajectorySample_t> *points,
                     double *lap_length)
{
    std::ifstream input(MPC_RTI_REPLAY_TRAJECTORY_PATH);
    if (!input.good()) return false;
    std::string line;
    while (std::getline(input, line)) {
        const auto fields = split_csv(line);
        if (fields.size() < 6) continue;
        double values[9]{};
        bool valid = true;
        const std::size_t count = std::min<std::size_t>(fields.size(), 9);
        for (std::size_t i = 0; i < count; ++i) {
            char *end = nullptr;
            values[i] = std::strtod(fields[i].c_str(), &end);
            if (end == fields[i].c_str() || *end != '\0' ||
                !std::isfinite(values[i])) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;  // Skip the CSV header.
        MpcTrajectorySample_t point{};
        point.s = values[0];
        point.x = values[1];
        point.y = values[2];
        point.heading = values[3];
        point.curvature = values[4];
        point.speed = values[5];
        point.left_bound = count >= 9 ? values[7] : 1.0;
        point.right_bound = count >= 9 ? values[8] : 1.0;
        points->push_back(point);
        if (points->size() > kMaximumTrajectoryPoints) return false;
    }
    std::size_t count = points->size();
    if (!mpc_trajectory_prepare(points->data(), &count, lap_length))
        return false;
    points->resize(count);
    return true;
}

MpcRtiCycleConfiguration_t replay_configuration(int max_iterations,
                                                float tolerance,
                                                bool use_fd_jacobian,
                                                bool use_prefactorization,
                                                bool use_scaling,
                                                bool adaptive_rho,
                                                MpcRtiRefinementMode_t refinement_mode,
                                                float rho,
                                                float rho_u,
                                                float degraded_residual_limit,
                                                int max_degraded_solves,
                                                float corridor_margin_m,
                                                float first_prediction_corridor_margin_m,
                                                float corridor_preview_halfwidth_m)
{
    MpcRtiConfiguration_t model{};
    model.weight_e_y = 1500.0f;
    model.weight_e_psi = 50.0f;
    model.weight_u = 200.0f;
    model.weight_v = 0.0f;
    model.weight_r = 1.5f;
    model.weight_steering_command = 1.0f;
    model.weight_steering_rate = 2.0f;
    model.weight_target_speed_rate = 0.5f;
    model.weight_steering_rate_change = 5.0f;
    model.weight_target_speed_rate_change = 5.0f;
    model.terminal_multiplier = 3.0f;
    model.max_speed_mps = 16.0f;
    model.active_speed_ceiling_mps = 16.0f;
    model.max_steering_rad = SOURCE_MAX_STEERING_RAD;
    model.max_steering_rate_radps = SOURCE_STEERING_RATE_RADPS;
    model.max_target_speed_rate_increase_mps2 = 3.0f;
    model.max_target_speed_rate_reduction_mps2 = 8.0f;
    model.corridor_margin_m = corridor_margin_m;
    model.first_prediction_corridor_margin_m =
        first_prediction_corridor_margin_m;
    model.corridor_preview_halfwidth_m = corridor_preview_halfwidth_m;
    model.nonlinear_corridor_tolerance_m = 0.001f;
    model.use_fd_jacobian_oracle = use_fd_jacobian ? 1 : 0;

    MpcRtiCycleConfiguration_t configuration{};
    configuration.model = model;
    configuration.solver.rho = rho;
    configuration.solver.rho_u = rho_u;
    configuration.solver.tolerance = tolerance;
    configuration.solver.max_iterations = max_iterations;
    configuration.solver.adaptive_rho = adaptive_rho ? 1 : 0;
    configuration.solver.shared_rho = 0;
    configuration.solver.use_prefactorization = use_prefactorization ? 1 : 0;
    configuration.solver.use_scaling = use_scaling ? 1 : 0;
    if (use_scaling) {
        const float state_scale[MPC_RTI_NX] = {
            0.10f, 0.25f, 10.0f, 0.25f, 3.2f, 10.0f, 0.5f, 1.2f, 8.0f};
        const float input_scale[MPC_RTI_NU] = {1.2f, 8.0f};
        std::copy(state_scale, state_scale + MPC_RTI_NX,
                  configuration.solver.state_scale);
        std::copy(input_scale, input_scale + MPC_RTI_NU,
                  configuration.solver.input_scale);
    }
    configuration.refinement_mode = refinement_mode;
    configuration.rti2_progress_error_trigger_m = 0.10f;
    configuration.rti2_curvature_error_trigger_per_m = 0.02f;
    configuration.rti2_bound_error_trigger_m = 0.05f;
    configuration.rti2_min_corridor_slack_trigger_m = 0.25f;
    configuration.rti2_steering_rate_correction_trigger_radps = 0.50f;
    configuration.rti2_target_speed_rate_correction_trigger_mps2 = 1.0f;
    configuration.rti2_residual_imbalance_trigger = 8.0f;
    configuration.rti2_lateral_load_trigger_mps2 = 3.0f;
    configuration.rti2_nonsmooth_columns_trigger = 50;
    configuration.degraded_residual_limit = degraded_residual_limit;
    configuration.maximum_regularization = 1.0e-2f;
    configuration.max_consecutive_degraded_solves = max_degraded_solves;
    return configuration;
}

using f1tenth_mpc::replay_metrics::maximum;
using f1tenth_mpc::replay_metrics::minimum;
using f1tenth_mpc::replay_metrics::percentile;

bool replay_events(const std::string &events_path, int max_iterations,
                   float tolerance, bool use_fd_jacobian,
                   bool use_prefactorization,
                   bool use_scaling, bool adaptive_rho,
                   MpcRtiRefinementMode_t refinement_mode,
                   float rho, float rho_u,
                   float degraded_residual_limit,
                   int max_degraded_solves,
                   float corridor_margin_m,
                   float first_prediction_corridor_margin_m,
                   float corridor_preview_halfwidth_m,
                   bool diagnostic_relaxed_residual_gate,
                   const std::string &actions_path,
                   const std::string &trajectory_path,
                   const std::vector<MpcTrajectorySample_t> &trajectory,
                   double lap_length)
{
    std::ifstream input(events_path);
    if (!input.good()) fail("cannot open event stream: " + events_path);
    std::ofstream action_output;
    if (!actions_path.empty()) {
        action_output.open(actions_path);
        if (!action_output.good())
            fail("cannot open action output: " + actions_path);
        action_output << "event_index,source_stamp_ns,status,steering_rate_radps,"
                         "target_speed_rate_mps2,steering_command_rad,"
                         "target_speed_mps,iterations,primal_residual,dual_residual,"
                         "rti_iterations_used,rti2_triggered,rti2_trigger_reason_mask,"
                         "rti2_budget_skipped,"
                         "r1_status,r2_status,r1_objective,r2_objective,"
                         "r1_slack_m,r2_slack_m,selected_candidate,r1_iterations,"
                         "r2_iterations,r1_steering_rate_radps,"
                         "r1_target_speed_rate_mps2,r1_next_progress_m,"
                         "r1_next_e_y_m,r1_next_raw_bound_clearance_m,"
                         "r1_next_first_step_clearance_m,"
                         "r2_steering_rate_radps,r2_target_speed_rate_mps2,"
                         "r2_next_progress_m,r2_next_e_y_m,"
                         "r2_next_raw_bound_clearance_m,"
                         "r2_next_first_step_clearance_m,"
                         "r1_solve_us,r2_solve_us,total_rti_us,"
                         "rho_start,rho_final,rho_u_start,rho_u_final,"
                         "rho_change_count,factorization_count,"
                         "progress_m,e_y_m,e_psi_rad,u_mps,v_mps,r_radps,"
                         "target_speed_state_mps,steering_state_rad,state_age_s,"
                         "path_curvature_per_m,raw_left_bound_m,"
                         "raw_right_bound_m,current_raw_bound_clearance_m,"
                         "current_inset_corridor_clearance_m,"
                         "first_prediction_corridor_margin_m,"
                         "lateral_accel_proxy_mps2,"
                         "minimum_predicted_corridor_slack_m,"
                         "candidate_progress_error_max_m,"
                         "candidate_curvature_error_max_per_m,"
                         "candidate_left_bound_error_max_m,"
                         "candidate_right_bound_error_max_m,"
                         "nonlinear_failure_stage,nonlinear_failure_reason\n";
        action_output << std::setprecision(10);
    }
    std::ofstream trajectory_output;
    if (!trajectory_path.empty()) {
        trajectory_output.open(trajectory_path);
        if (!trajectory_output.good())
            fail("cannot open trajectory output: " + trajectory_path);
        trajectory_output << "event_index,source_stamp_ns,stage,progress_m,e_y_m,"
                             "e_psi_rad,u_mps,v_mps,r_radps,target_speed_mps,"
                             "steering_command_rad,previous_steering_rate_radps,"
                             "previous_target_speed_rate_mps2,steering_rate_radps,"
                             "target_speed_rate_mps2\n";
        trajectory_output << std::setprecision(10);
    }

    MpcSyncConfig sync_config;
    sync_config.source_dt_min_s = 0.001;
    sync_config.source_dt_max_s = 0.250;
    sync_config.max_pose_odom_skew_s = 0.120;
    sync_config.max_state_age_s = 0.120;
    MpcStateSynchronizer synchronizer(sync_config);
    const MpcRtiCycleConfiguration_t configuration =
        replay_configuration(max_iterations, tolerance, use_fd_jacobian,
                             use_prefactorization, use_scaling, adaptive_rho,
                             refinement_mode,
                             rho, rho_u, degraded_residual_limit,
                             max_degraded_solves, corridor_margin_m,
                             first_prediction_corridor_margin_m,
                             corridor_preview_halfwidth_m);
    MpcRtiMemory_t memory{};
    mpc_rti_memory_reset(&memory);
    ReplayMetrics metrics;

    bool command_seen = false;
    double target_speed = 0.0;
    double steering_command = 0.0;
    double previous_steering_rate = 0.0;
    double previous_target_speed_rate = 0.0;
    int64_t previous_command_stamp_ns = 0;
    double previous_command_steering = 0.0;
    double previous_command_speed = 0.0;
    int64_t previous_odom_stamp_ns = 0;
    bool progress_initialized = false;
    double last_unwrapped_progress = 0.0;
    std::size_t previous_segment = std::numeric_limits<std::size_t>::max();

    std::string line;
    while (std::getline(input, line)) {
        const auto fields = split_csv(line);
        if (fields.size() < 8 || fields[0] == "event_index") continue;
        const std::string &topic = fields[3];
        const int64_t arrival_epoch_ns = std::strtoll(fields[2].c_str(), nullptr, 10);
        const int64_t source_stamp_ns = std::strtoll(fields[5].c_str(), nullptr, 10);
        const std::string &payload = fields[7];

        if (topic == "/cmd/speed") {
            double next_speed = 0.0;
            double next_steering = 0.0;
            if (!json_number(payload, "speed_mps", &next_speed) ||
                !json_number(payload, "steering_angle_rad", &next_steering))
                continue;
            if (command_seen && source_stamp_ns > previous_command_stamp_ns) {
                const double dt = static_cast<double>(
                    source_stamp_ns - previous_command_stamp_ns) * 1.0e-9;
                previous_steering_rate =
                    (next_steering - previous_command_steering) / dt;
                previous_target_speed_rate =
                    (next_speed - previous_command_speed) / dt;
            }
            target_speed = next_speed;
            steering_command = next_steering;
            previous_command_stamp_ns = source_stamp_ns;
            previous_command_steering = next_steering;
            previous_command_speed = next_speed;
            command_seen = true;
            continue;
        }

        if (topic == "/current_map_pose") {
            double x = 0.0, y = 0.0, yaw = 0.0;
            if (json_number(payload, "x_m", &x) &&
                json_number(payload, "y_m", &y) &&
                json_number(payload, "yaw_rad", &yaw)) {
                const MpcSyncStatus status = synchronizer.set_map_pose(
                    MpcMapPoseAnchor{source_stamp_ns, x, y, yaw});
                if (status == MpcSyncStatus::kTimestampOrderFault)
                    ++metrics.ordering_faults;
            }
            continue;
        }
        if (topic != "/odom") continue;

        double x = 0.0, y = 0.0, yaw = 0.0;
        double u = 0.0, v = 0.0, r = 0.0;
        if (!json_number(payload, "x_m", &x) ||
            !json_number(payload, "y_m", &y) ||
            !json_number(payload, "yaw_rad", &yaw) ||
            !json_number(payload, "speed_mps", &u) ||
            !json_number(payload, "lateral_speed_mps", &v) ||
            !json_number(payload, "yaw_rate_radps", &r)) continue;
        ++metrics.odometry_samples;
        const MpcSyncStatus push_status = synchronizer.push_odometry(
            MpcOdomSample{source_stamp_ns, x, y, yaw, u, v, r});
        if (push_status == MpcSyncStatus::kTimestampOrderFault) {
            ++metrics.ordering_faults;
            synchronizer.reset();
            mpc_rti_memory_reset(&memory);
            previous_odom_stamp_ns = 0;
            continue;
        }
        if (push_status == MpcSyncStatus::kSourceGapFault) {
            ++metrics.source_gap_faults;
            synchronizer.reset();
            mpc_rti_memory_reset(&memory);
            previous_odom_stamp_ns = 0;
            continue;
        }
        if (push_status != MpcSyncStatus::kOk) continue;

        MpcSynchronizedState synchronized{};
        const MpcSyncStatus sync_status = synchronizer.synchronize(
            arrival_epoch_ns, &synchronized);
        if (sync_status != MpcSyncStatus::kOk) {
            mpc_rti_memory_reset(&memory);
            previous_odom_stamp_ns = source_stamp_ns;
            continue;
        }
        ++metrics.synchronized_samples;
        metrics.state_age_ms.push_back(synchronized.source_age_s * 1000.0);
        if (!command_seen) {
            target_speed = std::max(0.0, synchronized.u);
            steering_command = 0.0;
        }

        MpcPathProjection_t projection{};
        if (!mpc_trajectory_project(trajectory.data(), trajectory.size(),
                lap_length, synchronized.map_x, synchronized.map_y,
                synchronized.map_yaw, previous_segment, 160, &projection)) {
            ++metrics.projection_rejections;
            mpc_rti_memory_reset(&memory);
            previous_odom_stamp_ns = source_stamp_ns;
            continue;
        }
        ++metrics.projected_samples;
        previous_segment = projection.segment;
        double progress = projection.s;
        if (progress_initialized) {
            while (progress - last_unwrapped_progress < -0.5 * lap_length)
                progress += lap_length;
            while (progress - last_unwrapped_progress > 0.5 * lap_length)
                progress -= lap_length;
            if (progress - last_unwrapped_progress > 0.0 &&
                progress - last_unwrapped_progress < 0.5 * lap_length) {
                last_unwrapped_progress = progress;
            }
        } else {
            progress_initialized = true;
            last_unwrapped_progress = progress;
        }
        progress = last_unwrapped_progress;

        if (projection.distance > 0.80 ||
            std::abs(projection.heading_error) > 0.75) {
            ++metrics.startup_path_rejections;
            mpc_rti_memory_reset(&memory);
            previous_odom_stamp_ns = source_stamp_ns;
            continue;
        }
        if (!command_seen) previous_target_speed_rate = 0.0;

        MpcRtiState_t state{};
        state.plant.e_y = static_cast<float>(projection.lateral_error);
        state.plant.e_psi = static_cast<float>(projection.heading_error);
        state.plant.u = static_cast<float>(synchronized.u);
        state.plant.v = static_cast<float>(synchronized.v);
        state.plant.r = static_cast<float>(synchronized.yaw_rate);
        state.plant.target_speed = static_cast<float>(target_speed);
        state.plant.steering_command = static_cast<float>(steering_command);
        state.previous_steering_rate =
            static_cast<float>(previous_steering_rate);
        state.previous_target_speed_rate =
            static_cast<float>(previous_target_speed_rate);

        if (previous_odom_stamp_ns > 0 &&
            source_stamp_ns > previous_odom_stamp_ns)
            metrics.source_dt_ms.push_back(static_cast<double>(
                source_stamp_ns - previous_odom_stamp_ns) * 1.0e-6);
        previous_odom_stamp_ns = source_stamp_ns;

        MpcRtiCycleResult_t result{};
        const int degraded_streak_before =
            memory.consecutive_degraded_solves;
        const auto start = std::chrono::steady_clock::now();
        const MpcRtiCycleStatus_t status = mpc_rti_solve_cycle(
            &state, progress, trajectory.data(), trajectory.size(), lap_length,
            0.025f, PREDICTION_HORIZON, &configuration, &memory, &result);
        const auto finish = std::chrono::steady_clock::now();
        RiccatiDebugInfo_t solver_debug{};
        riccati_debug_get_last(&solver_debug);
        metrics.quadratic_factorizations += static_cast<std::size_t>(
            std::max(0, solver_debug.quadratic_factorization_count));
        metrics.final_rho.push_back(solver_debug.rho);
        metrics.final_rho_u.push_back(solver_debug.rho_u);
        if (result.rti2_triggered) ++metrics.rti2_triggers;
        if (result.selected_candidate == 2) ++metrics.rti2_selected;
        if (result.rti2_triggered && result.selected_candidate == 1 &&
            result.r2_status >= MPC_RTI_CYCLE_REJECTED_INPUT)
            ++metrics.rti2_fallbacks;
        for (unsigned int bit = 0; bit < 11; ++bit) {
            if ((result.rti2_trigger_reason_mask & (1u << bit)) != 0u)
                ++metrics.rti2_reason_counts[bit];
        }
        const bool original_residual_reject =
            std::max(result.primal_residual, result.dual_residual) > 0.05f ||
            degraded_streak_before + 1 > 3;
        if (diagnostic_relaxed_residual_gate && original_residual_reject) {
            ++metrics.diagnostic_original_residual_rejections;
            if (status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED) {
                ++metrics.diagnostic_nonlinear_feasible;
            } else if (status == MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT) {
                ++metrics.diagnostic_nonlinear_rejections;
            } else {
                ++metrics.diagnostic_not_reached;
            }
        }
        const double solve_ms = std::chrono::duration<double, std::milli>(
            finish - start).count();
        if (action_output.good()) {
            MpcTrajectorySample_t current_sample{};
            const bool current_sample_valid = mpc_trajectory_sample(
                trajectory.data(), trajectory.size(), lap_length, progress,
                &current_sample) != 0;
            const auto pass_has_action = [](int pass_status) {
                return pass_status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
                    pass_status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED ||
                    pass_status == MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
            };
            double r1_next_progress = NAN;
            double r1_next_e_y = NAN;
            double r1_next_raw_clearance = NAN;
            double r2_next_progress = NAN;
            double r2_next_e_y = NAN;
            double r2_next_raw_clearance = NAN;
            if (current_sample_valid && pass_has_action(result.r1_status)) {
                const MpcStageResult_t step = mpc_vehicle_model_step(
                    &state.plant, &result.r1_first_action, 0.025f,
                    static_cast<float>(current_sample.curvature));
                if (step.valid) {
                    r1_next_progress = progress + step.delta_s_m;
                    r1_next_e_y = step.next.e_y;
                    MpcTrajectorySample_t next_sample{};
                    if (mpc_trajectory_sample(trajectory.data(),
                            trajectory.size(), lap_length, r1_next_progress,
                            &next_sample))
                        r1_next_raw_clearance = std::min(
                            next_sample.left_bound - step.next.e_y,
                            next_sample.right_bound + step.next.e_y);
                }
            }
            if (current_sample_valid && pass_has_action(result.r2_status)) {
                const MpcStageResult_t step = mpc_vehicle_model_step(
                    &state.plant, &result.r2_first_action, 0.025f,
                    static_cast<float>(current_sample.curvature));
                if (step.valid) {
                    r2_next_progress = progress + step.delta_s_m;
                    r2_next_e_y = step.next.e_y;
                    MpcTrajectorySample_t next_sample{};
                    if (mpc_trajectory_sample(trajectory.data(),
                            trajectory.size(), lap_length, r2_next_progress,
                            &next_sample))
                        r2_next_raw_clearance = std::min(
                            next_sample.left_bound - step.next.e_y,
                            next_sample.right_bound + step.next.e_y);
                }
            }
            const double current_raw_clearance = current_sample_valid
                ? std::min(current_sample.left_bound - state.plant.e_y,
                           current_sample.right_bound + state.plant.e_y)
                : NAN;
            const double current_inset_clearance = current_sample_valid
                ? current_raw_clearance -
                    configuration.model.corridor_margin_m
                : NAN;
            action_output << fields[0] << ',' << source_stamp_ns << ','
                << static_cast<int>(status) << ','
                << result.first_control.steering_rate << ','
                << result.first_control.target_speed_rate << ','
                << result.published_steering_command << ','
                << result.published_target_speed << ','
                << result.solver_iterations << ',' << result.primal_residual
                << ',' << result.dual_residual << ','
                << result.rti_iterations_used << ','
                << (result.rti2_triggered ? 1 : 0) << ','
                << result.rti2_trigger_reason_mask << ','
                << (result.rti2_budget_skipped ? 1 : 0) << ','
                << result.r1_status << ',' << result.r2_status << ','
                << result.r1_nonlinear_objective << ','
                << result.r2_nonlinear_objective << ','
                << result.r1_min_corridor_slack << ','
                << result.r2_min_corridor_slack << ','
                << (result.selected_candidate == 2 ? "R2" :
                    result.selected_candidate == 1 ? "R1" : "none") << ','
                << result.r1_solver_iterations << ','
                << result.r2_solver_iterations << ','
                << (pass_has_action(result.r1_status)
                        ? result.r1_first_action.steering_rate : NAN) << ','
                << (pass_has_action(result.r1_status)
                        ? result.r1_first_action.target_speed_rate : NAN) << ','
                << r1_next_progress << ',' << r1_next_e_y << ','
                << r1_next_raw_clearance << ','
                << r1_next_raw_clearance -
                    configuration.model.first_prediction_corridor_margin_m << ','
                << (pass_has_action(result.r2_status)
                        ? result.r2_first_action.steering_rate : NAN) << ','
                << (pass_has_action(result.r2_status)
                        ? result.r2_first_action.target_speed_rate : NAN) << ','
                << r2_next_progress << ',' << r2_next_e_y << ','
                << r2_next_raw_clearance << ','
                << r2_next_raw_clearance -
                    configuration.model.first_prediction_corridor_margin_m << ','
                << result.r1_solve_us << ',' << result.r2_solve_us << ','
                << result.total_rti_us << ',' << result.rho_start << ','
                << result.rho_final << ',' << result.rho_u_start << ','
                << result.rho_u_final << ',' << result.rho_change_count << ','
                << result.factorization_count << ',' << progress << ','
                << state.plant.e_y << ',' << state.plant.e_psi << ','
                << state.plant.u << ',' << state.plant.v << ','
                << state.plant.r << ',' << state.plant.target_speed << ','
                << state.plant.steering_command << ','
                << synchronized.source_age_s << ','
                << (current_sample_valid ? current_sample.curvature : NAN) << ','
                << (current_sample_valid ? current_sample.left_bound : NAN) << ','
                << (current_sample_valid ? current_sample.right_bound : NAN) << ','
                << current_raw_clearance << ',' << current_inset_clearance << ','
                << configuration.model.first_prediction_corridor_margin_m << ','
                << result.lateral_accel_proxy_mps2 << ','
                << result.minimum_predicted_corridor_slack_m << ','
                << result.max_candidate_progress_error_m << ','
                << result.max_candidate_curvature_error_per_m << ','
                << result.max_candidate_left_bound_error_m << ','
                << result.max_candidate_right_bound_error_m << ','
                << result.nonlinear_failure_stage << ','
                << result.nonlinear_failure_reason << '\n';
        }
        metrics.solve_ms.push_back(solve_ms);
        ++metrics.solves;
        ++metrics.status_counts[static_cast<int>(status)];
        if (status == MPC_RTI_CYCLE_REJECTED_RESIDUAL) {
            if (std::max(result.primal_residual, result.dual_residual) >
                configuration.degraded_residual_limit)
                ++metrics.residual_limit_rejections;
            else if (degraded_streak_before + 1 >
                     configuration.max_consecutive_degraded_solves)
                ++metrics.degraded_streak_rejections;
        }
        metrics.iterations.push_back(result.solver_iterations);
        metrics.primal_residual.push_back(result.primal_residual);
        metrics.dual_residual.push_back(result.dual_residual);
        metrics.maximum_regularization = std::max<double>(
            metrics.maximum_regularization, result.maximum_regularization);
        metrics.regularization_count += result.regularization_count;

        if (status != MPC_RTI_CYCLE_ACCEPTED_OPTIMAL &&
            status != MPC_RTI_CYCLE_ACCEPTED_DEGRADED) continue;

        if (trajectory_output.good()) {
            for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
                const MpcRtiState_t &predicted = memory.nominal.states[k];
                trajectory_output << fields[0] << ',' << source_stamp_ns << ','
                    << k << ',' << memory.nominal.progress[k] << ','
                    << predicted.plant.e_y << ','
                    << predicted.plant.e_psi << ',' << predicted.plant.u << ','
                    << predicted.plant.v << ',' << predicted.plant.r << ','
                    << predicted.plant.target_speed << ','
                    << predicted.plant.steering_command << ','
                    << predicted.previous_steering_rate << ','
                    << predicted.previous_target_speed_rate << ',';
                if (k < PREDICTION_HORIZON) {
                    trajectory_output
                        << memory.nominal.controls[k].steering_rate << ','
                        << memory.nominal.controls[k].target_speed_rate;
                } else {
                    trajectory_output << ',';
                }
                trajectory_output << '\n';
            }
        }

        if (result.candidate_path_delta.sample_count != PREDICTION_HORIZON + 1)
            fail("accepted candidate lacks full path-schedule mismatch diagnostics");
        for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
            metrics.candidate_progress_error_m.push_back(
                result.candidate_path_delta.progress_error_m[k]);
            metrics.candidate_curvature_error_per_m.push_back(
                result.candidate_path_delta.curvature_error_per_m[k]);
            metrics.candidate_left_bound_error_m.push_back(
                result.candidate_path_delta.left_bound_error_m[k]);
            metrics.candidate_right_bound_error_m.push_back(
                result.candidate_path_delta.right_bound_error_m[k]);
        }
        metrics.first_action_steering_rate_delta.push_back(std::abs(
            result.first_control.steering_rate -
            result.nominal_first_control.steering_rate));
        metrics.first_action_target_rate_delta.push_back(std::abs(
            result.first_control.target_speed_rate -
            result.nominal_first_control.target_speed_rate));

        metrics.nonlinear_rollout_stages += PREDICTION_HORIZON;
        for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
            const MpcRtiState_t &predicted = memory.nominal.states[k];
            if (!std::isfinite(predicted.plant.e_y) ||
                !std::isfinite(predicted.plant.e_psi) ||
                !std::isfinite(predicted.plant.u) ||
                !std::isfinite(predicted.plant.target_speed) ||
                !std::isfinite(predicted.plant.steering_command))
                fail("accepted candidate contains a nonfinite state");
            metrics.minimum_target_speed = std::min<double>(
                metrics.minimum_target_speed, predicted.plant.target_speed);
            metrics.maximum_target_speed = std::max<double>(
                metrics.maximum_target_speed, predicted.plant.target_speed);
            metrics.maximum_steering_command = std::max<double>(
                metrics.maximum_steering_command,
                std::abs(predicted.plant.steering_command));
            MpcTrajectorySample_t sample{};
            if (!mpc_trajectory_sample(trajectory.data(), trajectory.size(),
                    lap_length, memory.nominal.progress[k], &sample))
                fail("cannot sample trajectory at accepted predicted progress");
            /* x0 is measured and not constrained by the QP. Report future
             * predicted clearance only, consistent with candidate acceptance. */
            if (k > 0) {
                const double lower = configuration.model.corridor_margin_m -
                    sample.right_bound;
                const double upper = sample.left_bound -
                    configuration.model.corridor_margin_m;
                metrics.minimum_clearance_m.push_back(std::min(
                    predicted.plant.e_y - lower,
                    upper - predicted.plant.e_y));
            }
        }
        for (int k = 0; k < PREDICTION_HORIZON; ++k) {
            metrics.maximum_steering_rate = std::max<double>(
                metrics.maximum_steering_rate,
                std::abs(memory.nominal.controls[k].steering_rate));
            metrics.maximum_target_speed_rate = std::max<double>(
                metrics.maximum_target_speed_rate,
                std::abs(memory.nominal.controls[k].target_speed_rate));
        }
        /* This override is an offline diagnostic only.  Reset after each
         * candidate which the production 0.05 residual gate would reject so
         * subsequent warm starts follow the production replay sequence. */
        if (diagnostic_relaxed_residual_gate && original_residual_reject)
            mpc_rti_memory_reset(&memory);
    }

    const std::size_t accepted = metrics.status_counts[
        MPC_RTI_CYCLE_ACCEPTED_OPTIMAL] +
        metrics.status_counts[MPC_RTI_CYCLE_ACCEPTED_DEGRADED];
    const auto &s = metrics.status_counts;
    std::cout << std::fixed << std::setprecision(6)
        << "trace=" << events_path << '\n'
        << "odom=" << metrics.odometry_samples
        << " synchronized=" << metrics.synchronized_samples
        << " projected=" << metrics.projected_samples
        << " solves=" << metrics.solves << " accepted=" << accepted
        << " degraded=" << s[MPC_RTI_CYCLE_ACCEPTED_DEGRADED]
        << " rejected=" << metrics.solves - accepted << '\n'
        << "rejection_statuses[input,solver,residual,regularization,nonlinear]="
        << '[' << s[MPC_RTI_CYCLE_REJECTED_INPUT] << ','
        << s[MPC_RTI_CYCLE_REJECTED_SOLVER] << ','
        << s[MPC_RTI_CYCLE_REJECTED_RESIDUAL] << ','
        << s[MPC_RTI_CYCLE_REJECTED_REGULARIZATION] << ','
        << s[MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT] << "]\n"
        << "residual_rejections[over_limit,streak_limit]="
        << metrics.residual_limit_rejections << ','
        << metrics.degraded_streak_rejections << '\n'
        << "solver_profile[max_iterations,tolerance,degraded_limit]="
        << configuration.solver.max_iterations << ','
        << configuration.solver.tolerance << ','
        << configuration.degraded_residual_limit << '\n'
        << "diagnostic_relaxed_residual_gate="
        << (diagnostic_relaxed_residual_gate ? "true" : "false") << '\n'
        << "diagnostic_residual_candidates[original_reject,nonlinear_feasible,"
           "nonlinear_reject,not_reached]="
        << metrics.diagnostic_original_residual_rejections << ','
        << metrics.diagnostic_nonlinear_feasible << ','
        << metrics.diagnostic_nonlinear_rejections << ','
        << metrics.diagnostic_not_reached << '\n'
        << "admm_penalties[rho,rho_u]=" << configuration.solver.rho << ','
        << configuration.solver.rho_u << '\n'
        << "corridor[margin_m,first_prediction_margin_m,preview_halfwidth_m]="
        << configuration.model.corridor_margin_m << ','
        << configuration.model.first_prediction_corridor_margin_m << ','
        << configuration.model.corridor_preview_halfwidth_m << '\n'
        << "adaptive_rho=" << configuration.solver.adaptive_rho << '\n'
        << "rti_refinement_mode="
        << (configuration.refinement_mode == MPC_RTI_REFINEMENT_R2 ? "R2" :
            configuration.refinement_mode == MPC_RTI_REFINEMENT_ADAPTIVE ?
                "adaptive" : "R1")
        << " rti2_triggered=" << metrics.rti2_triggers
        << " selected_r2=" << metrics.rti2_selected
        << " r2_fallback_to_r1=" << metrics.rti2_fallbacks
        << " trigger_rate=" << (metrics.solves > 0
            ? static_cast<double>(metrics.rti2_triggers) / metrics.solves
            : 0.0) << '\n'
        << "rti2_reason_counts[progress,curvature,left_bound,right_bound,"
           "low_slack,nonsmooth,action_correction,degraded,residual_imbalance,"
           "steering_reversal,lateral_load]=";
    for (std::size_t index = 0; index < 11; ++index)
        std::cout << (index == 0 ? "" : ",")
                  << metrics.rti2_reason_counts[index];
    std::cout << '\n'
        << "quadratic_factorizations=" << metrics.quadratic_factorizations
        << " factor_per_solve=" << (metrics.solves > 0
            ? static_cast<double>(metrics.quadratic_factorizations) /
                  static_cast<double>(metrics.solves)
            : 0.0) << '\n'
        << "final_rho[rho,rho_u] p50/p95/max="
        << percentile(metrics.final_rho, 0.50) << '/'
        << percentile(metrics.final_rho, 0.95) << '/'
        << maximum(metrics.final_rho) << ','
        << percentile(metrics.final_rho_u, 0.50) << '/'
        << percentile(metrics.final_rho_u, 0.95) << '/'
        << maximum(metrics.final_rho_u) << '\n'
        << "jacobian=" << (use_fd_jacobian ? "fd_oracle" : "analytic")
        << '\n'
        << "riccati=" << (use_prefactorization ? "prefactorized" : "reference")
        << '\n'
        << "scaling=" << (use_scaling ? "dimensionless" : "physical") << '\n'
        << "prediction[horizon_steps,dt_s]=" << PREDICTION_HORIZON
        << ",0.025000\n"
        << "state_age_ms[p50,p95,max]="
        << percentile(metrics.state_age_ms, 0.50) << ','
        << percentile(metrics.state_age_ms, 0.95) << ','
        << percentile(metrics.state_age_ms, 1.0) << '\n'
        << "source_dt_ms[p50,p95,max]="
        << percentile(metrics.source_dt_ms, 0.50) << ','
        << percentile(metrics.source_dt_ms, 0.95) << ','
        << percentile(metrics.source_dt_ms, 1.0) << '\n'
        << "solve_ms[p50,p95,p99,max]="
        << percentile(metrics.solve_ms, 0.50) << ','
        << percentile(metrics.solve_ms, 0.95) << ','
        << percentile(metrics.solve_ms, 0.99) << ','
        << percentile(metrics.solve_ms, 1.0) << '\n'
        << "iterations[p50,p95,p99]="
        << percentile(metrics.iterations, 0.50) << ','
        << percentile(metrics.iterations, 0.95) << ','
        << percentile(metrics.iterations, 0.99) << '\n'
        << "residual_p95[primal,dual]="
        << percentile(metrics.primal_residual, 0.95) << ','
        << percentile(metrics.dual_residual, 0.95) << '\n'
        << "regularization[max,sum]=" << metrics.maximum_regularization << ','
        << metrics.regularization_count << '\n'
        << "predicted_min_corridor_clearance_m="
        << minimum(metrics.minimum_clearance_m) << '\n'
        << "predicted_max_corridor_clearance_m="
        << maximum(metrics.minimum_clearance_m) << '\n'
        << "candidate_vs_nominal_progress_error_m[p95,max]="
        << percentile(metrics.candidate_progress_error_m, 0.95) << ','
        << maximum(metrics.candidate_progress_error_m) << '\n'
        << "candidate_vs_nominal_curvature_error_per_m[p95,max]="
        << percentile(metrics.candidate_curvature_error_per_m, 0.95) << ','
        << maximum(metrics.candidate_curvature_error_per_m) << '\n'
        << "candidate_vs_nominal_left_bound_delta_m[p95,max]="
        << percentile(metrics.candidate_left_bound_error_m, 0.95) << ','
        << maximum(metrics.candidate_left_bound_error_m) << '\n'
        << "candidate_vs_nominal_right_bound_delta_m[p95,max]="
        << percentile(metrics.candidate_right_bound_error_m, 0.95) << ','
        << maximum(metrics.candidate_right_bound_error_m) << '\n'
        << "first_action_delta[abs_steering_rate_p95,max,"
           "abs_target_rate_p95,max]="
        << percentile(metrics.first_action_steering_rate_delta, 0.95) << ','
        << maximum(metrics.first_action_steering_rate_delta) << ','
        << percentile(metrics.first_action_target_rate_delta, 0.95) << ','
        << maximum(metrics.first_action_target_rate_delta) << '\n'
        << "predicted_command_envelope[max_abs_qdelta,max_abs_qv,"
           "max_abs_delta,target_speed_min,max]="
        << metrics.maximum_steering_rate << ','
        << metrics.maximum_target_speed_rate << ','
        << metrics.maximum_steering_command << ','
        << metrics.minimum_target_speed << ','
        << metrics.maximum_target_speed << '\n'
        << "nonlinear_rollout_stages=" << metrics.nonlinear_rollout_stages
        << " projection_rejections=" << metrics.projection_rejections
        << " startup_path_rejections=" << metrics.startup_path_rejections
        << " source_order_faults=" << metrics.ordering_faults
        << " source_gap_faults=" << metrics.source_gap_faults << '\n';
    const bool no_solver_rejections = accepted == metrics.solves;
    const bool no_path_faults = metrics.projection_rejections == 0 &&
        metrics.startup_path_rejections == 0;
    return metrics.solves > 0 && no_solver_rejections && no_path_faults &&
        metrics.ordering_faults == 0 && metrics.source_gap_faults == 0;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "usage: mpc_rti_offline_replay EVENTS_CSV "
                     "[MAX_ITERATIONS TOLERANCE] [--fd-jacobian] "
                     "[--prefactorized] [--scaled] [--rho VALUE] "
                     "[--rho-u VALUE] [--adaptive-rho] "
                     "[--diagnostic-residual-limit VALUE] "
                     "[--rti-mode r1|r2|adaptive] "
                     "[--corridor-margin METERS] "
                     "[--first-prediction-corridor-margin METERS] "
                     "[--corridor-preview METERS] "
                     "[--actions OUTPUT_CSV] "
                     "[--trajectory OUTPUT_CSV]\n";
        return 2;
    }
    int max_iterations = 50;
    float tolerance = 0.01f;
    bool numeric_settings_seen = false;
    bool use_fd_jacobian = false;
    bool use_prefactorization = false;
    bool use_scaling = false;
    bool adaptive_rho = false;
    MpcRtiRefinementMode_t refinement_mode = MPC_RTI_REFINEMENT_R1;
    bool diagnostic_relaxed_residual_gate = false;
    float rho = 7.0f;
    float rho_u = 7.0f;
    float degraded_residual_limit = 0.05f;
    float corridor_margin_m = 0.05f;
    float first_prediction_corridor_margin_m = 0.05f;
    float corridor_preview_halfwidth_m = 0.10f;
    std::string actions_path;
    std::string trajectory_path;
    for (int index = 2; index < argc; ++index) {
        const std::string option(argv[index]);
        if (option == "--fd-jacobian") {
            use_fd_jacobian = true;
        } else if (option == "--prefactorized") {
            use_prefactorization = true;
        } else if (option == "--scaled") {
            use_scaling = true;
        } else if (option == "--adaptive-rho") {
            adaptive_rho = true;
        } else if (option == "--rti-mode" && index + 1 < argc) {
            const std::string mode(argv[++index]);
            if (mode == "r1") refinement_mode = MPC_RTI_REFINEMENT_R1;
            else if (mode == "r2") refinement_mode = MPC_RTI_REFINEMENT_R2;
            else if (mode == "adaptive" || mode == "ra")
                refinement_mode = MPC_RTI_REFINEMENT_ADAPTIVE;
            else {
                std::cerr << "rti mode must be r1, r2, or adaptive\n";
                return 2;
            }
        } else if (option == "--diagnostic-residual-limit" &&
                   index + 1 < argc) {
            degraded_residual_limit = std::stof(argv[++index]);
            diagnostic_relaxed_residual_gate = true;
        } else if (option == "--corridor-margin" && index + 1 < argc) {
            corridor_margin_m = std::stof(argv[++index]);
        } else if (option == "--first-prediction-corridor-margin" &&
                   index + 1 < argc) {
            first_prediction_corridor_margin_m = std::stof(argv[++index]);
        } else if (option == "--corridor-preview" && index + 1 < argc) {
            corridor_preview_halfwidth_m = std::stof(argv[++index]);
        } else if (option == "--rho" && index + 1 < argc) {
            rho = std::stof(argv[++index]);
        } else if (option == "--rho-u" && index + 1 < argc) {
            rho_u = std::stof(argv[++index]);
        } else if (option == "--actions" && index + 1 < argc) {
            actions_path = argv[++index];
        } else if (option == "--trajectory" && index + 1 < argc) {
            trajectory_path = argv[++index];
        } else if (!numeric_settings_seen && index + 1 < argc) {
            max_iterations = std::stoi(argv[index]);
            tolerance = std::stof(argv[++index]);
            numeric_settings_seen = true;
        } else {
            std::cerr << "invalid replay option: " << option << '\n';
            return 2;
        }
    }
    if (!std::isfinite(rho) || !std::isfinite(rho_u) ||
        rho < 1.0f || rho > 127.0f || rho_u < 1.0f || rho_u > 127.0f) {
        std::cerr << "rho and rho-u must lie in [1, 127]\n";
        return 2;
    }
    if (!std::isfinite(corridor_margin_m) || corridor_margin_m < 0.0f ||
        !std::isfinite(first_prediction_corridor_margin_m) ||
        first_prediction_corridor_margin_m < 0.0f ||
        first_prediction_corridor_margin_m > corridor_margin_m ||
        !std::isfinite(corridor_preview_halfwidth_m) ||
        corridor_preview_halfwidth_m < 0.0f) {
        std::cerr << "corridor margins and preview must be finite, nonnegative, "
                     "and first-step margin <= horizon margin\n";
        return 2;
    }
    if (!std::isfinite(degraded_residual_limit) ||
        degraded_residual_limit < 0.05f ||
        degraded_residual_limit > 1000.0f) {
        std::cerr << "diagnostic residual limit must lie in [0.05, 1000]\n";
        return 2;
    }
    const int max_degraded_solves = diagnostic_relaxed_residual_gate
        ? std::numeric_limits<int>::max() : 3;
    std::vector<MpcTrajectorySample_t> trajectory;
    double lap_length = 0.0;
    if (!load_trajectory(&trajectory, &lap_length))
        fail("cannot load the accepted raceline CSV");
    return replay_events(argv[1], max_iterations, tolerance,
        use_fd_jacobian, use_prefactorization, use_scaling, adaptive_rho,
        refinement_mode,
        rho, rho_u, degraded_residual_limit, max_degraded_solves,
        corridor_margin_m, first_prediction_corridor_margin_m,
        corridor_preview_halfwidth_m,
        diagnostic_relaxed_residual_gate,
        actions_path, trajectory_path,
        trajectory, lap_length) ? 0 : 1;
}

#define _POSIX_C_SOURCE 200809L

#include "mpc_rti.h"
#include "mpc_state_synchronizer.hpp"

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
    std::vector<double> primal_residual;
    std::vector<double> dual_residual;
    std::vector<double> state_age_ms;
    std::vector<double> minimum_clearance_m;
    double maximum_steering_rate{};
    double maximum_target_speed_rate{};
    double minimum_target_speed{std::numeric_limits<double>::infinity()};
    double maximum_target_speed{-std::numeric_limits<double>::infinity()};
    double maximum_steering_command{};
    double maximum_regularization{};
    std::size_t regularization_count{};
    std::size_t nonlinear_rollout_stages{};
    std::vector<double> source_dt_ms;
    std::size_t residual_limit_rejections{};
    std::size_t degraded_streak_rejections{};
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
                                                float tolerance)
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
    model.corridor_margin_m = 0.05f;
    model.corridor_preview_halfwidth_m = 0.10f;
    model.nonlinear_corridor_tolerance_m = 0.001f;

    MpcRtiCycleConfiguration_t configuration{};
    configuration.model = model;
    configuration.solver.rho = 7.0f;
    configuration.solver.rho_u = 7.0f;
    configuration.solver.tolerance = tolerance;
    configuration.solver.max_iterations = max_iterations;
    configuration.solver.adaptive_rho = 0;
    configuration.solver.shared_rho = 0;
    configuration.degraded_residual_limit = 0.05f;
    configuration.maximum_regularization = 1.0e-2f;
    configuration.max_consecutive_degraded_solves = 3;
    return configuration;
}

double percentile(std::vector<double> values, double fraction)
{
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

bool replay_events(const std::string &events_path, int max_iterations,
                   float tolerance,
                   const std::vector<MpcTrajectorySample_t> &trajectory,
                   double lap_length)
{
    std::ifstream input(events_path);
    if (!input.good()) fail("cannot open event stream: " + events_path);

    MpcSyncConfig sync_config;
    sync_config.source_dt_min_s = 0.001;
    sync_config.source_dt_max_s = 0.250;
    sync_config.max_pose_odom_skew_s = 0.120;
    sync_config.max_state_age_s = 0.120;
    MpcStateSynchronizer synchronizer(sync_config);
    const MpcRtiCycleConfiguration_t configuration =
        replay_configuration(max_iterations, tolerance);
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
        const double solve_ms = std::chrono::duration<double, std::milli>(
            finish - start).count();
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
            const double lower = configuration.model.corridor_margin_m -
                sample.right_bound;
            const double upper = sample.left_bound -
                configuration.model.corridor_margin_m;
            metrics.minimum_clearance_m.push_back(std::min(
                predicted.plant.e_y - lower, upper - predicted.plant.e_y));
        }
        for (int k = 0; k < PREDICTION_HORIZON; ++k) {
            metrics.maximum_steering_rate = std::max<double>(
                metrics.maximum_steering_rate,
                std::abs(memory.nominal.controls[k].steering_rate));
            metrics.maximum_target_speed_rate = std::max<double>(
                metrics.maximum_target_speed_rate,
                std::abs(memory.nominal.controls[k].target_speed_rate));
        }
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
        << percentile(metrics.minimum_clearance_m, 0.0) << '\n'
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
    if (argc != 2 && argc != 4) {
        std::cerr << "usage: mpc_rti_offline_replay EVENTS_CSV "
                     "[MAX_ITERATIONS TOLERANCE]\n";
        return 2;
    }
    const int max_iterations = argc == 4 ? std::stoi(argv[2]) : 50;
    const float tolerance = argc == 4 ? std::stof(argv[3]) : 0.01f;
    std::vector<MpcTrajectorySample_t> trajectory;
    double lap_length = 0.0;
    if (!load_trajectory(&trajectory, &lap_length))
        fail("cannot load the accepted raceline CSV");
    return replay_events(argv[1], max_iterations, tolerance,
        trajectory, lap_length) ? 0 : 1;
}

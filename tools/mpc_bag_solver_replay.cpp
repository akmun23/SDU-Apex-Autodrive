#include "mpc_rti.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kMaxSteeringRad = 0.5235987755982988;
constexpr std::size_t kInputColumnCount = 38;

std::vector<double> parse_csv_numbers(const std::string &line)
{
    std::vector<double> values;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        if (!field.empty()) values.push_back(std::stod(field));
    }
    return values;
}

std::vector<MpcTrajectorySample_t> load_trajectory(const char *path,
                                                   double *lap_length)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open trajectory CSV");
    std::vector<MpcTrajectorySample_t> points;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        const auto row = parse_csv_numbers(line);
        if (row.size() < 9) continue;
        points.push_back(MpcTrajectorySample_t{
            row[0], row[1], row[2], row[3], row[4], row[5], row[7], row[8],
            row[6]});
    }
    std::size_t count = points.size();
    if (!mpc_trajectory_prepare(points.data(), &count, lap_length))
        throw std::runtime_error("trajectory validation failed");
    points.resize(count);
    return points;
}

double number_arg(char **argv, int index, double fallback)
{
    return argv[index] ? std::stod(argv[index]) : fallback;
}

void configure(MpcRtiCycleConfiguration_t *config,
               double rho, double rho_steer, double rho_throttle,
               double alpha, double cost_scale, const std::string &mode,
               double rho_ey, double rho_u_state, double rho_steer_state,
               double rho_max, int rho_period, double tolerance,
               bool cold_start_each_solve, int jacobian_threads,
               double startup_throttle_seed,
               double lateral_error_weight_multiplier,
               double heading_error_weight_multiplier,
               double lateral_state_scale,
               double heading_state_scale,
               double corridor_margin,
               double first_corridor_margin,
               int recovery_policy,
               double recovery_k_ey,
               double recovery_k_epsi,
               double recovery_k_r)
{
    *config = MpcRtiCycleConfiguration_t{};
    auto &model = config->model;
    model.weight_e_y = static_cast<float>(
        125.0 * cost_scale * lateral_error_weight_multiplier);
    model.weight_e_psi = static_cast<float>(
        31.25 * cost_scale * heading_error_weight_multiplier);
    model.weight_u = static_cast<float>(500.0 * cost_scale);
    model.weight_u_overspeed = static_cast<float>(300.0 * cost_scale);
    model.weight_longitudinal_slip = static_cast<float>(5.0 * cost_scale);
    model.weight_v = static_cast<float>(5.0 * cost_scale);
    model.weight_r = static_cast<float>(7.8125 * cost_scale);
    model.weight_steering_command = static_cast<float>(3.0 * cost_scale);
    model.weight_steering_target = static_cast<float>(5.0 * cost_scale);
    model.weight_throttle_command = static_cast<float>(0.1 * cost_scale);
    model.startup_nominal_throttle_seed = static_cast<float>(
        startup_throttle_seed);
    model.terminal_multiplier = 3.0f;
    model.max_speed_mps = 16.0f;
    model.active_speed_ceiling_mps = 16.0f;
    model.max_steering_rad = static_cast<float>(kMaxSteeringRad);
    model.corridor_margin_m = static_cast<float>(corridor_margin);
    model.first_prediction_corridor_margin_m =
        static_cast<float>(first_corridor_margin);
    model.corridor_preview_halfwidth_m = 0.10f;
    model.nonlinear_corridor_tolerance_m = 0.005f;
    model.recovery_seed_policy = recovery_policy;
    model.recovery_steering_k_e_y = static_cast<float>(recovery_k_ey);
    model.recovery_steering_k_e_psi = static_cast<float>(recovery_k_epsi);
    model.recovery_steering_k_r = static_cast<float>(recovery_k_r);
    model.jacobian_threads = jacobian_threads;

    auto &solver = config->solver;
    solver.rho = static_cast<float>(rho);
    solver.rho_u = 40.0f;
    solver.rho_x_by_state[MPC_OPT_IDX_EY] = static_cast<float>(rho_ey);
    solver.rho_x_by_state[MPC_OPT_IDX_U] = static_cast<float>(rho_u_state);
    solver.rho_x_by_state[MPC_OPT_IDX_ACTUAL_STEERING_ANGLE] =
        static_cast<float>(rho_steer_state);
    solver.rho_u_by_input[0] = static_cast<float>(rho_steer);
    solver.rho_u_by_input[1] = static_cast<float>(rho_throttle);
    solver.rho_max = static_cast<float>(rho_max);
    solver.tolerance = static_cast<float>(tolerance);
    solver.max_iterations = 49;
    solver.adaptive_rho = 1;
    solver.cold_start_each_solve = cold_start_each_solve ? 1 : 0;
    solver.adaptive_rho_update_period = rho_period;
    solver.shared_rho = 0;
    solver.over_relaxation = static_cast<float>(alpha);
    solver.use_scaling = 1;
    const float state_scale[RICCATI_MAX_NX] = {
        0.10f, 0.10f, 0.5f, 1.0f, 4.0f, 0.5f,
        400.0f, 400.0f, 400.0f, 400.0f,
        0.10f, 1.0f, 0.25f, 4.0f, 0.25f, 4.0f,
        0.05f, 0.05f, 0.05f, 0.05f};
    std::copy_n(state_scale, RICCATI_MAX_NX, solver.state_scale);
    solver.state_scale[MPC_OPT_IDX_EY] = static_cast<float>(lateral_state_scale);
    solver.state_scale[MPC_OPT_IDX_EPSI] = static_cast<float>(heading_state_scale);
    solver.input_scale[0] = 0.5f;
    solver.input_scale[1] = 1.0f;

    config->refinement_mode = mode == "r2"
        ? MPC_RTI_REFINEMENT_R2 : MPC_RTI_REFINEMENT_R1;
    config->rti2_progress_error_trigger_m = 0.10f;
    config->rti2_curvature_error_trigger_per_m = 0.02f;
    config->rti2_bound_error_trigger_m = 0.05f;
    config->rti2_min_corridor_slack_trigger_m = 0.25f;
    config->rti2_steering_target_correction_trigger_rad = 0.50f;
    config->rti2_throttle_correction_trigger = 0.05f;
    config->rti2_residual_imbalance_trigger = 8.0f;
    config->rti2_lateral_load_trigger_mps2 = 3.0f;
    config->rti2_nonsmooth_columns_trigger = 50;
    config->degraded_residual_limit = static_cast<float>(tolerance);
    config->max_consecutive_degraded_solves = 3;
    config->max_cycle_time_us = 0.0;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 8 || argc > 30) {
        std::cerr << "usage: mpc_bag_solver_replay TRAJECTORY INPUT_CSV "
                     "RHO RHO_STEER RHO_THROTTLE ALPHA COST_SCALE "
                     "[r1|r2] [RHO_EY RHO_U_STATE RHO_STEER_STATE "
                     "[RHO_MAX PERIOD TOLERANCE "
                     "[sequence|independent] [warm|cold] [REPEATS "
                     "[JACOBIAN_THREADS [STARTUP_SEED "
                     "[EY_WEIGHT_MULTIPLIER [EPSI_WEIGHT_MULTIPLIER "
                     "[EY_STATE_SCALE [EPSI_STATE_SCALE "
                     "[CORRIDOR_MARGIN [FIRST_CORRIDOR_MARGIN "
                     "[RECOVERY_POLICY [K_EY [K_EPSI [K_R]]]]]]]]]]\n";
        return 2;
    }

    try {
        const double rho = number_arg(argv, 3, 2.0);
        const double rho_steer = number_arg(argv, 4, 1.0);
        const double rho_throttle = number_arg(argv, 5, 7.0);
        const double alpha = number_arg(argv, 6, 1.6);
        const double cost_scale = number_arg(argv, 7, 0.001);
        const std::string mode = argc >= 9 ? argv[8] : "r1";
        const double rho_ey = argc >= 10 ? number_arg(argv, 9, 0.0) : 0.0;
        const double rho_u_state = argc >= 11 ? number_arg(argv, 10, 0.0) : 0.0;
        const double rho_steer_state = argc >= 12
            ? number_arg(argv, 11, 0.0) : 0.0;
        const double rho_max = argc >= 13 ? number_arg(argv, 12, 32.0) : 32.0;
        const int rho_period = argc >= 14
            ? static_cast<int>(number_arg(argv, 13, 10.0)) : 10;
        const double tolerance = argc >= 15
            ? number_arg(argv, 14, 0.02) : 0.02;
        const std::string memory_mode = argc >= 16 ? argv[15] : "sequence";
        const std::string solver_start = argc >= 17 ? argv[16] : "warm";
        const int max_repeats = argc >= 18
            ? static_cast<int>(number_arg(argv, 17, 0.0)) : 0;
        const int jacobian_threads = argc >= 19
            ? static_cast<int>(number_arg(argv, 18, 4.0)) : 4;
        const double startup_seed = argc >= 20
            ? number_arg(argv, 19, 0.25) : 0.25;
        const double ey_weight_multiplier = argc >= 21
            ? number_arg(argv, 20, 1.0) : 1.0;
        const double epsi_weight_multiplier = argc >= 22
            ? number_arg(argv, 21, 1.0) : 1.0;
        const double ey_state_scale = argc >= 23
            ? number_arg(argv, 22, 0.10) : 0.10;
        const double epsi_state_scale = argc >= 24
            ? number_arg(argv, 23, 0.10) : 0.10;
        const double corridor_margin = argc >= 25
            ? number_arg(argv, 24, 0.05) : 0.05;
        const double first_corridor_margin = argc >= 26
            ? number_arg(argv, 25, corridor_margin) : corridor_margin;
        const int recovery_policy = argc >= 27
            ? static_cast<int>(number_arg(argv, 26, 0.0))
            : MPC_RTI_RECOVERY_SEED_NOMINAL;
        const double recovery_k_ey = argc >= 28
            ? number_arg(argv, 27, 0.0) : 0.0;
        const double recovery_k_epsi = argc >= 29
            ? number_arg(argv, 28, 0.0) : 0.0;
        const double recovery_k_r = argc >= 30
            ? number_arg(argv, 29, 0.0) : 0.0;
        if ((mode != "r1" && mode != "r2") || !(cost_scale > 0.0) ||
            !(tolerance > 0.0) ||
            (memory_mode != "sequence" && memory_mode != "independent") ||
            (solver_start != "warm" && solver_start != "cold") ||
            max_repeats < 0 || max_repeats > 10 ||
            jacobian_threads < 0 || jacobian_threads > 4 ||
            startup_seed < 0.0 || startup_seed > 1.0 ||
            !(ey_weight_multiplier > 0.0) || ey_weight_multiplier > 100.0 ||
            !(epsi_weight_multiplier > 0.0) ||
            epsi_weight_multiplier > 100.0 ||
            !(ey_state_scale > 0.0) || ey_state_scale > 10.0 ||
            !(epsi_state_scale > 0.0) || epsi_state_scale > 10.0 ||
            corridor_margin < 0.0 ||
            first_corridor_margin < 0.0 ||
            first_corridor_margin > corridor_margin ||
            recovery_policy < MPC_RTI_RECOVERY_SEED_NOMINAL ||
            recovery_policy > MPC_RTI_RECOVERY_SEED_BRAKE_HEADING_FEEDBACK ||
            recovery_k_ey < 0.0 || recovery_k_epsi < 0.0 ||
            recovery_k_r < 0.0)
            throw std::runtime_error("invalid mode or cost scale");

        double lap_length = 0.0;
        const auto trajectory = load_trajectory(argv[1], &lap_length);
        std::ifstream input(argv[2]);
        if (!input) throw std::runtime_error("cannot open input CSV");

        MpcRtiCycleConfiguration_t config{};
        configure(&config, rho, rho_steer, rho_throttle, alpha, cost_scale,
                  mode, rho_ey, rho_u_state, rho_steer_state,
                  rho_max, rho_period, tolerance, solver_start == "cold",
                  jacobian_threads, startup_seed, ey_weight_multiplier,
                  epsi_weight_multiplier, ey_state_scale, epsi_state_scale,
                  corridor_margin, first_corridor_margin, recovery_policy,
                  recovery_k_ey, recovery_k_epsi, recovery_k_r);
        const char *trace_row_text = std::getenv("MPC_REPLAY_TRACE_ROW");
        const long trace_row = trace_row_text
            ? std::strtol(trace_row_text, nullptr, 10) : -1;
        const char *diagnostic_row_text = std::getenv("MPC_REPLAY_DIAG_ROW");
        const long diagnostic_row = diagnostic_row_text
            ? std::strtol(diagnostic_row_text, nullptr, 10) : -1;
        riccati_debug_set_trace_enabled(trace_row >= 0);
        std::cout << "stamp_ns,status,r1_admm_status,r1_iterations,r1_primal,"
                     "r1_dual,r1_state_primal,r1_state_primal_stage,"
                     "r1_state_primal_index,r1_state_dual,r1_state_dual_stage,"
                     "r1_state_dual_index,r1_input_primal,"
                     "r1_input_primal_stage,r1_input_primal_index,"
                     "r1_input_dual,r1_input_dual_stage,r1_input_dual_index,"
                     "r2_admm_status,r2_iterations,r2_primal,r2_dual,rti_calls,"
                     "convergence_retry_count,convergence_retry_cycle_status,"
                     "convergence_retry_r1_admm_status,"
                     "convergence_retry_r2_admm_status,"
                     "convergence_retry_r1_iterations,"
                     "convergence_retry_r2_iterations,"
                     "convergence_retry_us,"
                     "r1_attempts,r2_attempts,r1_iterations_total,r2_iterations_total,"
                     "r1_qp_padding_m,r2_qp_padding_m,"
                     "r1_solve_us,r2_solve_us,total_rti_us,"
                     "r1_first_steering,r1_first_throttle,"
                     "r1_cycle_status,r1_nonlinear_reason,r1_nonlinear_stage,"
                     "r2_cycle_status,r2_nonlinear_reason,r2_nonlinear_stage,"
                     "corridor_repair_used,corridor_repair_alpha,"
                     "r1_nonlinear_objective,r2_nonlinear_objective,"
                     "selected_candidate,nonlinear_failure_reason,"
                     "nonlinear_failure_stage,"
                     "r1_failure_valid,r1_failure_ey,r1_failure_epsi,"
                     "r1_failure_lower,r1_failure_upper,"
                     "r2_failure_valid,r2_failure_ey,r2_failure_epsi,"
                     "r2_failure_lower,r2_failure_upper,"
                     "selected_steering,selected_throttle\n";

        std::string line;
        std::size_t row_index = 0;
        MpcRtiMemory_t memory{};
        mpc_rti_memory_reset(&memory);
        while (std::getline(input, line)) {
            if (line.empty() || line.front() == '#') continue;
            const auto row = parse_csv_numbers(line);
            if (row.size() != kInputColumnCount)
                throw std::runtime_error("input row has unexpected column count");

            MpcRtiState_t state{};
            state.plant.e_y = static_cast<float>(row[3]);
            state.plant.e_psi = static_cast<float>(row[4]);
            state.plant.u = static_cast<float>(row[5]);
            state.plant.v = static_cast<float>(row[6]);
            state.plant.r = static_cast<float>(row[7]);
            state.plant.actual_steering_angle = static_cast<float>(row[8]);
            state.plant.delayed_steering_command_1 = static_cast<float>(row[9]);
            state.plant.delayed_steering_command_2 = static_cast<float>(row[10]);
            state.plant.delayed_throttle_command_1 = static_cast<float>(row[11]);
            state.plant.delayed_throttle_command_2 = static_cast<float>(row[12]);
            state.plant.source_suspension_history_initialized =
                row[13] != 0.0 ? 1 : 0;
            state.plant.source_plant_initialized = 1;
            state.plant.source_imu_state_initialized = 1;
            /* Columns 14-15 retain the exported bag schema's optional yaw
             * innovation fields. The production state no longer contains or
             * applies that correction, so replay intentionally ignores them. */
            for (int i = 0; i < MPC_PLANT_INTERNAL_STATE_COUNT; ++i)
                state.plant.source_plant_internal_state[i] = row[16 + i];

            config.model.active_speed_ceiling_mps = static_cast<float>(row[2]);
            MpcRtiCycleResult_t result{};
            if (memory_mode == "independent") mpc_rti_memory_reset(&memory);
            MpcRtiCycleStatus_t status = MPC_RTI_CYCLE_REJECTED_INPUT;
            int rti_calls = 0;
            for (int attempt = 0; attempt <= max_repeats; ++attempt) {
                status = mpc_rti_solve_cycle(
                    &state, row[1], trajectory.data(), trajectory.size(),
                    lap_length, PREDICTION_DT_SECONDS, PREDICTION_HORIZON,
                    &config, &memory, &result);
                ++rti_calls;
                const bool r1_converged = result.r1_admm_status == 0 &&
                    result.r1_solver_iterations < 50;
                const bool r2_converged = result.r2_admm_status == 0 &&
                    result.r2_solver_iterations < 50;
                const bool retry_converged =
                    (result.convergence_retry_r1_admm_status == 0 &&
                     result.convergence_retry_r1_iterations < 50) ||
                    (result.convergence_retry_r2_admm_status == 0 &&
                     result.convergence_retry_r2_iterations < 50);
                if (r1_converged || r2_converged || retry_converged) break;
            }
            if (static_cast<long>(row_index) == trace_row) {
                const int trace_count = riccati_debug_get_trace_count();
                for (int i = 0; i < trace_count; ++i) {
                    RiccatiDebugIterSample_t sample{};
                    if (riccati_debug_get_trace_sample(i, &sample) == 0) {
                        std::cerr << "TRACE row=" << row_index
                                  << " iter=" << sample.iter
                                  << " p=" << sample.primal_residual
                                  << " d=" << sample.dual_residual
                                  << " px=" << sample.state_primal_residual
                                  << " dx=" << sample.state_dual_residual
                                  << " pu=" << sample.ctrl_primal_residual
                                  << " du=" << sample.ctrl_dual_residual
                                  << " rho=" << sample.rho
                                  << " rho_u=" << sample.rho_u
                                  << " u0=" << sample.u0_steer << ','
                                  << sample.u0_throttle
                                  << " z0=" << sample.z0_steer << ','
                                  << sample.z0_throttle
                                  << " rho_scale=" << sample.scale_rho << ','
                                  << sample.scale_rho_u << '\n';
                    }
                }
            }
            if (static_cast<long>(row_index) == diagnostic_row) {
                std::cerr << "SCHEDULE row=" << row_index
                          << " status=" << status
                          << " recovery=" << result.recovery_active
                          << " not_found=" << result.recovery_not_found
                          << " initial_violation="
                          << result.recovery_initial_normal_violation_m
                          << " max_seed_violation="
                          << result.recovery_max_seed_violation_m
                          << " reentry=" << result.recovery_reentry_stage
                          << " state={ey:" << state.plant.e_y
                          << ",e_psi:" << state.plant.e_psi
                          << ",u:" << state.plant.u
                          << ",r:" << state.plant.r << "}\n";
                std::cerr << "stage,seed_ey,seed_violation,normal_lower,normal_upper,active_lower,active_upper,allowance,recovery\n";
                for (int k = 0; k <= PREDICTION_HORIZON; ++k)
                    std::cerr << k << ',' << result.recovery_seed_e_y_m[k]
                              << ',' << result.recovery_seed_violation_m[k]
                              << ',' << result.recovery_normal_lower_m[k]
                              << ',' << result.recovery_normal_upper_m[k]
                              << ',' << result.recovery_active_lower_m[k]
                              << ',' << result.recovery_active_upper_m[k]
                              << ',' << result.recovery_seed_violation_m[k]
                              << ',' << result.recovery_stage[k] << '\n';
                auto print_failure = [](const char *label,
                                        const MpcRtiRolloutFailure_t &failure) {
                    std::cerr << label << " valid=" << failure.valid
                              << " stage=" << failure.stage
                              << " ey=" << failure.state.plant.e_y
                              << " epsi=" << failure.state.plant.e_psi
                              << " lower=" << failure.lower_bound_m
                              << " upper=" << failure.upper_bound_m << '\n';
                };
                print_failure("R1_FAILURE", result.r1_nonlinear_failure);
                print_failure("R2_FAILURE", result.r2_nonlinear_failure);
            }
            if (status == MPC_RTI_CYCLE_REJECTED_INPUT &&
                result.input_failure_reason != MPC_RTI_INPUT_FAILURE_NONE)
                std::cerr << "input row " << row_index << " rejected before solve: "
                          << result.input_failure_reason << '\n';
            std::cout.precision(9);
            std::cout << static_cast<std::uint64_t>(row[0]) << ',' << status << ','
                      << result.r1_admm_status << ',' << result.r1_solver_iterations
                      << ',' << result.r1_primal_residual << ','
                      << result.r1_dual_residual << ','
                      << result.r1_state_primal_residual << ','
                      << result.r1_state_primal_stage << ','
                      << result.r1_state_primal_index << ','
                      << result.r1_state_dual_residual << ','
                      << result.r1_state_dual_stage << ','
                      << result.r1_state_dual_index << ','
                      << result.r1_input_primal_residual << ','
                      << result.r1_input_primal_stage << ','
                      << result.r1_input_primal_index << ','
                      << result.r1_input_dual_residual << ','
                      << result.r1_input_dual_stage << ','
                      << result.r1_input_dual_index << ','
                      << result.r2_admm_status << ',' << result.r2_solver_iterations
                      << ',' << result.r2_primal_residual << ','
                      << result.r2_dual_residual << ',' << rti_calls << ','
                      << result.convergence_retry_count << ','
                      << result.convergence_retry_cycle_status << ','
                      << result.convergence_retry_r1_admm_status << ','
                      << result.convergence_retry_r2_admm_status << ','
                      << result.convergence_retry_r1_iterations << ','
                      << result.convergence_retry_r2_iterations << ','
                      << result.convergence_retry_us << ','
                      << result.r1_solver_attempts << ','
                      << result.r2_solver_attempts << ','
                      << result.r1_solver_iterations_total << ','
                      << result.r2_solver_iterations_total << ','
                      << result.r1_qp_corridor_padding_m << ','
                      << result.r2_qp_corridor_padding_m << ','
                      << result.r1_solve_us << ',' << result.r2_solve_us << ','
                      << result.total_rti_us << ','
                      << result.r1_first_action.steering_target_rad << ','
                      << result.r1_first_action.throttle_command << ','
                      << result.r1_status << ','
                      << result.r1_nonlinear_failure_reason << ','
                      << result.r1_nonlinear_failure_stage << ','
                      << result.r2_status << ','
                      << result.r2_nonlinear_failure_reason << ','
                      << result.r2_nonlinear_failure_stage << ','
                      << result.corridor_repair_used << ','
                      << result.corridor_repair_alpha << ','
                      << result.r1_nonlinear_objective << ','
                      << result.r2_nonlinear_objective << ','
                      << result.selected_candidate << ','
                      << result.nonlinear_failure_reason << ','
                      << result.nonlinear_failure_stage << ','
                      << result.r1_nonlinear_failure.valid << ','
                      << result.r1_nonlinear_failure.state.plant.e_y << ','
                      << result.r1_nonlinear_failure.state.plant.e_psi << ','
                      << result.r1_nonlinear_failure.lower_bound_m << ','
                      << result.r1_nonlinear_failure.upper_bound_m << ','
                      << result.r2_nonlinear_failure.valid << ','
                      << result.r2_nonlinear_failure.state.plant.e_y << ','
                      << result.r2_nonlinear_failure.state.plant.e_psi << ','
                      << result.r2_nonlinear_failure.lower_bound_m << ','
                      << result.r2_nonlinear_failure.upper_bound_m << ','
                      << result.first_control.steering_target_rad << ','
                      << result.first_control.throttle_command << '\n';
            ++row_index;
        }
        return row_index == 0 ? 1 : 0;
    } catch (const std::exception &error) {
        std::cerr << "mpc_bag_solver_replay: " << error.what() << '\n';
        return 2;
    }
}

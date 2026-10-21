#include "mpc_rti.h"
#include "mpc_qp_fixture_format.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <set>
#include <string>
#include <vector>

namespace {

MpcRtiQpCapture_t g_qp_capture{};

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
               double alpha, double cost_scale,
               double rho_ey, double rho_u_state, double rho_steer_state,
               double rho_max, int rho_period, double tolerance,
               bool cold_start_each_solve, int jacobian_threads,
               int max_iterations,
               double startup_throttle_seed,
               double lateral_error_weight_multiplier,
               double heading_error_weight_multiplier,
               double lateral_state_scale,
               double heading_state_scale,
               double corridor_margin,
               double first_corridor_margin,
               double rho_u,
               double steering_slew_weight,
               double throttle_slew_weight,
               bool adaptive_rho,
               double speed_tracking_weight_multiplier)
{
    *config = MpcRtiCycleConfiguration_t{};
    auto &model = config->model;
    model.weight_e_y = static_cast<float>(
        125.0 * cost_scale * lateral_error_weight_multiplier);
    model.weight_e_psi = static_cast<float>(
        31.25 * cost_scale * heading_error_weight_multiplier);
    model.weight_u = static_cast<float>(500.0 * cost_scale);
    model.weight_u_overspeed = static_cast<float>(300.0 * cost_scale);
    model.weight_u *= static_cast<float>(speed_tracking_weight_multiplier);
    model.weight_u_overspeed *=
        static_cast<float>(speed_tracking_weight_multiplier);
    model.weight_longitudinal_slip = static_cast<float>(5.0 * cost_scale);
    model.weight_v = static_cast<float>(5.0 * cost_scale);
    model.weight_r = static_cast<float>(7.8125 * cost_scale);
    model.weight_steering_command = static_cast<float>(3.0 * cost_scale);
    model.weight_steering_target = static_cast<float>(5.0 * cost_scale);
    model.weight_throttle_command = static_cast<float>(0.1 * cost_scale);
    model.weight_steering_slew = static_cast<float>(steering_slew_weight);
    model.weight_throttle_slew = static_cast<float>(throttle_slew_weight);
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
    model.jacobian_threads = jacobian_threads;

    auto &solver = config->solver;
    solver.rho = static_cast<float>(rho);
    solver.rho_u = static_cast<float>(rho_u);
    solver.rho_x_by_state[MPC_OPT_IDX_EY] = static_cast<float>(rho_ey);
    solver.rho_x_by_state[MPC_OPT_IDX_U] = static_cast<float>(rho_u_state);
    solver.rho_x_by_state[MPC_OPT_IDX_ACTUAL_STEERING_ANGLE] =
        static_cast<float>(rho_steer_state);
    solver.rho_u_by_input[0] = static_cast<float>(rho_steer);
    solver.rho_u_by_input[1] = static_cast<float>(rho_throttle);
    solver.rho_max = static_cast<float>(rho_max);
    solver.tolerance = static_cast<float>(tolerance);
    solver.max_iterations = max_iterations;
    solver.adaptive_rho = adaptive_rho ? 1 : 0;
    solver.cold_start_each_solve = cold_start_each_solve ? 1 : 0;
    solver.adaptive_rho_update_period = rho_period;
    solver.shared_rho = 0;
    solver.over_relaxation = static_cast<float>(alpha);
    solver.use_scaling = 1;
    const std::array<float, RICCATI_MAX_NX> state_scale = {{
        0.10f, 0.10f, 0.5f, 1.0f, 4.0f, 0.5f,
        400.0f, 400.0f, 400.0f, 400.0f,
        0.10f, 1.0f, 0.25f, 4.0f, 0.25f, 4.0f,
        0.05f, 0.05f, 0.05f, 0.05f,
        0.5f, 1.0f}};
    static_assert(state_scale.size() == RICCATI_MAX_NX,
        "replay scaling must cover every optimizer state");
    std::copy(state_scale.begin(), state_scale.end(), solver.state_scale);
    solver.state_scale[MPC_OPT_IDX_EY] = static_cast<float>(lateral_state_scale);
    solver.state_scale[MPC_OPT_IDX_EPSI] = static_cast<float>(heading_state_scale);
    solver.input_scale[0] = 0.5f;
    solver.input_scale[1] = 1.0f;

    config->max_cycle_time_us = 0.0;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 8 || (argc > 31 && argc != 43)) {
        std::cerr << "usage: mpc_bag_solver_replay TRAJECTORY INPUT_CSV "
                     "RHO RHO_STEER RHO_THROTTLE ALPHA COST_SCALE "
                     "[RHO_EY RHO_U_STATE RHO_STEER_STATE "
                     "[RHO_MAX PERIOD TOLERANCE "
                     "[sequence|independent|nominal-cold|reference-seed] "
                     "[warm|cold|dual-cold] "
                     "[JACOBIAN_THREADS [MAX_ITERATIONS [STARTUP_SEED "
                     "[EY_WEIGHT_MULTIPLIER [EPSI_WEIGHT_MULTIPLIER "
                     "[EY_STATE_SCALE [EPSI_STATE_SCALE "
                     "[CORRIDOR_MARGIN [FIRST_CORRIDOR_MARGIN "
                     "[RHO_U [STEERING_SLEW_WEIGHT "
                     "[THROTTLE_SLEW_WEIGHT [ADAPTIVE_RHO "
                     "[LINE_SEARCH_BACKTRACKS "
                     "[SPEED_TRACKING_WEIGHT_MULTIPLIER "
                     "[DIRECT_WEIGHTS: EY EPSI U U_OVERSPEED SLIP V R "
                     "STEER_COMMAND STEER_TARGET THROTTLE_COMMAND "
                     "STEER_SLEW THROTTLE_SLEW]]]]]]]]]]]]]]]]] "
                     "(MAX_ITERATIONS: 1..200)\n";
        return 2;
    }

    try {
        const double rho = number_arg(argv, 3, 2.0);
        const double rho_steer = number_arg(argv, 4, 1.0);
        const double rho_throttle = number_arg(argv, 5, 7.0);
        const double alpha = number_arg(argv, 6, 1.6);
        const double cost_scale = number_arg(argv, 7, 0.001);
        const double rho_ey = argc >= 9 ? number_arg(argv, 8, 0.0) : 0.0;
        const double rho_u_state = argc >= 10 ? number_arg(argv, 9, 0.0) : 0.0;
        const double rho_steer_state = argc >= 11
            ? number_arg(argv, 10, 0.0) : 0.0;
        const double rho_max = argc >= 12 ? number_arg(argv, 11, 32.0) : 32.0;
        const int rho_period = argc >= 13
            ? static_cast<int>(number_arg(argv, 12, 10.0)) : 10;
        const double tolerance = argc >= 14
            ? number_arg(argv, 13, 0.02) : 0.02;
        const std::string memory_mode = argc >= 15 ? argv[14] : "sequence";
        const std::string solver_start = argc >= 16 ? argv[15] : "warm";
        const int jacobian_threads = argc >= 17
            ? static_cast<int>(number_arg(argv, 16, 4.0)) : 4;
        const int max_iterations = argc >= 18
            ? static_cast<int>(number_arg(argv, 17, 100.0)) : 100;
        const double startup_seed = argc >= 19
            ? number_arg(argv, 18, 0.25) : 0.25;
        const double ey_weight_multiplier = argc >= 20
            ? number_arg(argv, 19, 1.0) : 1.0;
        const double epsi_weight_multiplier = argc >= 21
            ? number_arg(argv, 20, 1.0) : 1.0;
        const double ey_state_scale = argc >= 22
            ? number_arg(argv, 21, 0.10) : 0.10;
        const double epsi_state_scale = argc >= 23
            ? number_arg(argv, 22, 0.10) : 0.10;
        const double corridor_margin = argc >= 24
            ? number_arg(argv, 23, 0.05) : 0.05;
        const double first_corridor_margin = argc >= 25
            ? number_arg(argv, 24, corridor_margin) : corridor_margin;
        const double rho_u = argc >= 26 ? number_arg(argv, 25, 7.0) : 7.0;
        const double steering_slew_weight = argc >= 27
            ? number_arg(argv, 26, 25.0) : 25.0;
        const double throttle_slew_weight = argc >= 28
            ? number_arg(argv, 27, 2.0) : 2.0;
        const double adaptive_rho_value = argc >= 29
            ? number_arg(argv, 28, 1.0) : 1.0;
        const bool adaptive_rho = adaptive_rho_value != 0.0;
        const int line_search_backtracks = argc >= 30
            ? static_cast<int>(number_arg(argv, 29, 0.0)) : 0;
        const double speed_tracking_weight_multiplier = argc >= 31
            ? number_arg(argv, 30, 1.0) : 1.0;
        if (!(cost_scale > 0.0) ||
            !(tolerance > 0.0) ||
            (adaptive_rho_value != 0.0 && adaptive_rho_value != 1.0) ||
            !std::isfinite(rho_u) || !(rho_u > 0.0) || rho_u > 127.0 ||
            (memory_mode != "sequence" && memory_mode != "independent" &&
             memory_mode != "nominal-cold" &&
             memory_mode != "reference-seed") ||
            (solver_start != "warm" && solver_start != "cold" &&
             solver_start != "dual-cold") ||
            jacobian_threads < 0 ||
                jacobian_threads > MPC_RTI_MAX_JACOBIAN_THREADS ||
            max_iterations < 1 || max_iterations > 200 ||
            startup_seed < 0.0 || startup_seed > 1.0 ||
            !(ey_weight_multiplier > 0.0) || ey_weight_multiplier > 100.0 ||
            !(epsi_weight_multiplier > 0.0) ||
            epsi_weight_multiplier > 100.0 ||
            !(ey_state_scale > 0.0) || ey_state_scale > 10.0 ||
            !(epsi_state_scale > 0.0) || epsi_state_scale > 10.0 ||
            corridor_margin < 0.0 ||
            first_corridor_margin < 0.0 ||
            first_corridor_margin > corridor_margin ||
            !std::isfinite(steering_slew_weight) ||
            steering_slew_weight < 0.0 ||
            !std::isfinite(throttle_slew_weight) ||
            throttle_slew_weight < 0.0 ||
            line_search_backtracks < 0 || line_search_backtracks > 2 ||
            !std::isfinite(speed_tracking_weight_multiplier) ||
            !(speed_tracking_weight_multiplier > 0.0) ||
            speed_tracking_weight_multiplier > 100.0)
            throw std::runtime_error("invalid replay configuration");

        double lap_length = 0.0;
        const auto trajectory = load_trajectory(argv[1], &lap_length);
        std::ifstream input(argv[2]);
        if (!input) throw std::runtime_error("cannot open input CSV");

        MpcRtiCycleConfiguration_t config{};
        configure(&config, rho, rho_steer, rho_throttle, alpha, cost_scale,
                  rho_ey, rho_u_state, rho_steer_state,
                  rho_max, rho_period, tolerance, solver_start == "cold",
                  jacobian_threads, max_iterations, startup_seed,
                  ey_weight_multiplier,
                  epsi_weight_multiplier, ey_state_scale, epsi_state_scale,
                  corridor_margin, first_corridor_margin, rho_u,
                  steering_slew_weight, throttle_slew_weight, adaptive_rho,
                  speed_tracking_weight_multiplier);
        /* Production tuning sweeps use the exact twelve production-stage
         * weights, rather than trying to synthesize them from this replay
         * executable's historical global cost multipliers. Keep the older
         * multiplier interface intact for existing benchmark artifacts. */
        if (argc == 43) {
            float * const direct_weights[] = {
                &config.model.weight_e_y,
                &config.model.weight_e_psi,
                &config.model.weight_u,
                &config.model.weight_u_overspeed,
                &config.model.weight_longitudinal_slip,
                &config.model.weight_v,
                &config.model.weight_r,
                &config.model.weight_steering_command,
                &config.model.weight_steering_target,
                &config.model.weight_throttle_command,
                &config.model.weight_steering_slew,
                &config.model.weight_throttle_slew,
            };
            for (int index = 0; index < 12; ++index) {
                const double value = number_arg(argv, 31 + index, 0.0);
                if (!std::isfinite(value) || value < 0.0 || value > 1.0e8)
                    throw std::runtime_error(
                        "direct replay weights must be finite and in [0, 1e8]");
                *direct_weights[index] = static_cast<float>(value);
            }
        }
        config.enable_nonlinear_line_search = line_search_backtracks > 0;
        config.line_search_max_backtracks = line_search_backtracks;
        /* This executable is an offline fixed-state evaluator; it has no
         * bridge command epoch. Live code leaves the corresponding gate off. */
        config.line_search_command_epoch_valid = 1;
        const char *trace_row_text = std::getenv("MPC_REPLAY_TRACE_ROW");
        const long trace_row = trace_row_text
            ? std::strtol(trace_row_text, nullptr, 10) : -1;
        const char *diagnostic_row_text = std::getenv("MPC_REPLAY_DIAG_ROW");
        const long diagnostic_row = diagnostic_row_text
            ? std::strtol(diagnostic_row_text, nullptr, 10) : -1;
        const char *qp_capture_dir_text =
            std::getenv("MPC_REPLAY_QP_CAPTURE_DIR");
        const char *qp_capture_row_text =
            std::getenv("MPC_REPLAY_QP_CAPTURE_ROW");
        const char *qp_capture_rows_text =
            std::getenv("MPC_REPLAY_QP_CAPTURE_ROWS");
        const long qp_capture_row = qp_capture_row_text
            ? std::strtol(qp_capture_row_text, nullptr, 10) : -1;
        std::set<long> qp_capture_rows;
        if (qp_capture_rows_text && qp_capture_rows_text[0] != '\0') {
            std::stringstream rows_stream(qp_capture_rows_text);
            std::string row_token;
            while (std::getline(rows_stream, row_token, ',')) {
                if (row_token.empty())
                    throw std::runtime_error(
                        "empty row in MPC_REPLAY_QP_CAPTURE_ROWS");
                std::size_t parsed = 0;
                const long row = std::stol(row_token, &parsed);
                if (parsed != row_token.size() || row < 0 ||
                    !qp_capture_rows.insert(row).second)
                    throw std::runtime_error(
                        "invalid or duplicate MPC_REPLAY_QP_CAPTURE_ROWS row");
            }
        }
        const bool qp_capture_enabled = qp_capture_dir_text &&
            qp_capture_dir_text[0] != '\0';
        const std::filesystem::path qp_capture_dir = qp_capture_enabled
            ? std::filesystem::path(qp_capture_dir_text)
            : std::filesystem::path{};
        if (qp_capture_enabled)
            std::filesystem::create_directories(qp_capture_dir);
        riccati_debug_set_trace_enabled(trace_row >= 0);
        std::cout << "stamp_ns,status,admm_status,time_budget_exhausted,iterations,"
                     "sqp_iterations_attempted,sqp_iterations_completed,"
                     "sqp_iterations_accepted,primal,dual,"
                     "rho_changes,factorizations,"
                     "input_u_mps,input_v_mps,input_r_radps,"
                     "input_abs_steering_rad,input_delayed_throttle,"
                     "reference_speed_mps,reference_curvature_per_m,"
                     "solve_us,cycle_us,nominal_build_us,reference_seed_us,"
                     "cold_start_queue_seed_us,rollout_a_us,"
                     "progress_reference_resampling_us,rollout_b_us,"
                     "qp_build_us,jacobian_wall_us,jacobian_thread_cpu_us,"
                     "jacobian_slowest_stage_us,jacobian_stages_completed,"
                     "jacobian_threads_used,jacobian_source_sensitivity_calls,"
                     "jacobian_finite_difference_columns,admm_us,"
                     "solver_cold_start_initialization_ns,"
                     "riccati_factorization_ns,riccati_rhs_ns,"
                     "admm_projection_ns,rho_update_ns,scaling_transform_ns,"
                     "output_unscale_ns,rollout_us,"
                     "nonlinear_validation_us,min_corridor_slack_m,"
                     "nonlinear_failure_reason,nonlinear_failure_stage,"
                     "line_search_attempted,line_search_trials,"
                     "line_search_accepted,line_search_skipped_budget,"
                     "line_search_skipped_command_epoch,"
                     "line_search_full_step_failure_reason,"
                     "line_search_full_step_failure_stage,"
                     "line_search_last_failure_reason,"
                     "line_search_last_failure_stage,"
                     "line_search_selected_alpha,line_search_us,"
                     "first_steering,first_throttle,"
                     "max_plan_steering_delta,max_plan_throttle_delta\n";

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

            MpcTrajectorySample_t operating_reference{};
            const bool operating_reference_valid = mpc_trajectory_sample(
                trajectory.data(), trajectory.size(), lap_length, row[1],
                &operating_reference) != 0;

            config.model.active_speed_ceiling_mps = static_cast<float>(row[2]);
            const bool capture_this_qp = qp_capture_enabled &&
                ((qp_capture_row < 0 && qp_capture_rows.empty()) ||
                 qp_capture_row == static_cast<long>(row_index) ||
                 qp_capture_rows.count(static_cast<long>(row_index)) != 0);
            if (capture_this_qp) {
                g_qp_capture = MpcRtiQpCapture_t{};
                config.qp_capture = &g_qp_capture;
            } else {
                config.qp_capture = nullptr;
            }
            MpcRtiCycleResult_t result{};
            if (memory_mode == "independent") {
                mpc_rti_memory_reset(&memory);
            } else if (memory_mode == "nominal-cold") {
                /* Keep the solver's ADMM iterate, but discard only the
                 * previous nonlinear nominal. This isolates whether a
                 * shifted trajectory—not the QP warm state—is the source of
                 * sequence-replay convergence failures. */
                memory.nominal.valid = 0;
            } else if (memory_mode == "reference-seed") {
                /* Offline counterfactual for PR-C: preserve ADMM z/y/rho
                 * history but rebuild the next physical nominal from the
                 * trajectory/source-plant initializer, not the previous
                 * rejected QP suffix. The current sensor state still carries
                 * the immutable in-flight actuator queue. This is a replay
                 * experiment only; it is not a publishable recovery action
                 * and does not reconstruct an unavailable certified suffix. */
                memory.nominal.valid = 0;
                memory.warm_start_controls_valid = 0;
                memory.warm_start_control_count = 0;
            }
            if (solver_start == "dual-cold") {
                /* Experimental hybrid ADMM seed: retain shifted primal
                 * projections (z), adapted rho values, and nominal/command
                 * history, but discard dual multipliers (y). This isolates
                 * whether stale duals hurt difficult corridor states without
                 * discarding the current command-consistent warm start.
                 * Offline replay only; production remains warm. */
                for (int stage = 0; stage <= RICCATI_HORIZON; ++stage) {
                    for (int state_index = 0;
                         state_index < RICCATI_MAX_NX; ++state_index) {
                        memory.solver_state.y_x[stage][state_index] = 0.0f;
                    }
                }
                for (int stage = 0; stage < RICCATI_HORIZON; ++stage) {
                    for (int input_index = 0;
                         input_index < RICCATI_MAX_NU; ++input_index) {
                        memory.solver_state.y_u[stage][input_index] = 0.0f;
                    }
                }
            }
            const MpcRtiCycleStatus_t status = mpc_rti_solve_cycle(
                &state, row[1], trajectory.data(), trajectory.size(),
                lap_length, PREDICTION_DT_SECONDS, PREDICTION_HORIZON,
                &config, &memory, &result);
            if (capture_this_qp && g_qp_capture.valid) {
                char filename[32];
                std::snprintf(filename, sizeof(filename), "qp_%06zu.fqp",
                              row_index);
                const auto path = qp_capture_dir / filename;
                if (!sdu_apex::qp_fixture::write_capture(
                        path.string(), g_qp_capture))
                    throw std::runtime_error(
                        "failed to write frozen QP fixture: " + path.string());
                std::cerr << "QP_CAPTURE row=" << row_index
                          << " passes=" << g_qp_capture.capture_count
                          << " path=" << path.string() << '\n';
            }
            float max_plan_steering_delta = 0.0f;
            float max_plan_throttle_delta = 0.0f;
            if ((status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
                 status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
                 status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH) &&
                memory.nominal.valid && memory.nominal.horizon > 0) {
                float previous_steering = state.plant.delayed_steering_command_2;
                float previous_throttle = state.plant.delayed_throttle_command_2;
                for (int k = 0; k < memory.nominal.horizon; ++k) {
                    const MpcModelControl_t &control = memory.nominal.controls[k];
                    max_plan_steering_delta = std::max(
                        max_plan_steering_delta,
                        std::abs(control.steering_target_rad - previous_steering));
                    max_plan_throttle_delta = std::max(
                        max_plan_throttle_delta,
                        std::abs(control.throttle_command - previous_throttle));
                    previous_steering = control.steering_target_rad;
                    previous_throttle = control.throttle_command;
                }
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
                std::cerr << "CYCLE row=" << row_index
                          << " status=" << status
                          << " admm=" << result.admm_status
                          << " iterations=" << result.solver_iterations
                          << " nonlinear_failure="
                          << result.nonlinear_failure_reason
                          << " stage=" << result.nonlinear_failure_stage
                          << " state={ey:" << state.plant.e_y
                          << ",e_psi:" << state.plant.e_psi
                          << ",u:" << state.plant.u
                          << ",r:" << state.plant.r << "}\n";
                std::cerr << "NOMINAL_FIRST row=" << row_index
                          << " steering="
                          << result.nominal_first_control.steering_target_rad
                          << " throttle="
                          << result.nominal_first_control.throttle_command
                          << " progress=" << row[1]
                          << " curvature=" << row[2] << '\n';
                const auto &failure = result.nonlinear_failure;
                std::cerr << "NONLINEAR_FAILURE valid=" << failure.valid
                          << " stage=" << failure.stage
                          << " ey=" << failure.state.plant.e_y
                          << " epsi=" << failure.state.plant.e_psi
                          << " lower=" << failure.lower_bound_m
                          << " upper=" << failure.upper_bound_m << '\n';
                if (result.candidate_control_count > 0) {
                    std::cerr << "REJECTED_CANDIDATE_CONTROLS row="
                              << row_index;
                    for (int k = 0;
                         k < result.candidate_control_count; ++k) {
                        const MpcModelControl_t &control =
                            result.candidate_controls[k];
                        std::cerr << " [k=" << k << ",delta="
                                  << control.steering_target_rad
                                  << ",q=" << control.throttle_command << ']';
                    }
                    std::cerr << '\n';
                }
                if ((status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
                     status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
                     status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH) &&
                    memory.nominal.horizon > 0) {
                    const int horizon = memory.nominal.horizon;
                    const int sample_stages[] = {
                        0, 1, 2, 3, 4, horizon / 3,
                        (2 * horizon) / 3, horizon - 1};
                    std::cerr << "PLAN_SAMPLES row=" << row_index;
                    int previous_stage = -1;
                    for (const int stage : sample_stages) {
                        if (stage < 0 || stage >= horizon ||
                            stage == previous_stage) continue;
                        previous_stage = stage;
                        const MpcModelControl_t &control =
                            memory.nominal.controls[stage];
                        const float predicted_u =
                            memory.nominal.states[stage + 1].plant.u;
                        std::cerr << " [k=" << stage << ",delta="
                                  << control.steering_target_rad
                                  << ',' << control.throttle_command << ']';
                        std::cerr << " u=" << predicted_u;
                    }
                    std::cerr << " terminal_u="
                              << memory.nominal.states[horizon].plant.u;
                    std::cerr << '\n';
                    std::cerr << "PLAN_CONTROLS row=" << row_index;
                    for (int stage = 0; stage < horizon; ++stage) {
                        const MpcModelControl_t &control =
                            memory.nominal.controls[stage];
                        std::cerr << " [k=" << stage << ':'
                                  << control.steering_target_rad << ','
                                  << control.throttle_command << ']';
                    }
                    std::cerr << '\n';
                }
            }
            if (status == MPC_RTI_CYCLE_REJECTED_INPUT &&
                result.input_failure_reason != MPC_RTI_INPUT_FAILURE_NONE)
                std::cerr << "input row " << row_index << " rejected before solve: "
                          << result.input_failure_reason << '\n';
            std::cout.precision(9);
            std::cout << static_cast<std::uint64_t>(row[0]) << ',' << status << ','
                      << result.admm_status << ','
                      << result.solver_time_budget_exhausted << ','
                      << result.solver_iterations << ','
                      << result.sqp_iterations_attempted << ','
                      << result.sqp_iterations_completed << ','
                      << result.sqp_iterations_accepted << ','
                      << result.primal_residual << ',' << result.dual_residual << ','
                      << result.rho_change_count << ','
                      << result.factorization_count << ','
                      << state.plant.u << ',' << state.plant.v << ','
                      << state.plant.r << ','
                      << std::abs(state.plant.actual_steering_angle) << ','
                      << state.plant.delayed_throttle_command_2 << ','
                      << (operating_reference_valid
                          ? operating_reference.speed
                          : std::numeric_limits<double>::quiet_NaN()) << ','
                      << (operating_reference_valid
                          ? operating_reference.curvature
                          : std::numeric_limits<double>::quiet_NaN()) << ','
                      << result.total_solve_us << ',' << result.total_cycle_us << ','
                      << result.nominal_build_us << ',' << result.reference_seed_us << ','
                      << result.cold_start_queue_seed_us << ','
                      << result.nominal_rollout_a_us << ','
                      << result.progress_reference_resampling_us << ','
                      << result.nominal_rollout_b_us << ','
                      << result.qp_build_us << ',' << result.jacobian_wall_us << ','
                      << result.jacobian_thread_cpu_us << ','
                      << result.jacobian_slowest_stage_us << ','
                      << result.jacobian_stages_completed << ','
                      << result.jacobian_threads_used << ','
                      << result.jacobian_source_sensitivity_calls << ','
                      << result.jacobian_finite_difference_columns << ','
                      << result.admm_us << ','
                      << result.solver_cold_start_initialization_time_ns << ','
                      << result.factorization_time_ns << ','
                      << result.riccati_rhs_time_ns << ','
                      << result.admm_projection_time_ns << ','
                      << result.rho_update_time_ns << ','
                      << result.solver_scaling_time_ns << ','
                      << result.solver_output_unscale_time_ns << ','
                      << result.exact_rollout_us << ','
                      << result.nonlinear_validation_us << ','
                      << result.minimum_predicted_corridor_slack_m << ','
                      << result.nonlinear_failure_reason << ','
                      << result.nonlinear_failure_stage << ','
                      << result.line_search_attempted << ','
                      << result.line_search_trials << ','
                      << result.line_search_accepted << ','
                      << result.line_search_skipped_budget << ','
                      << result.line_search_skipped_command_epoch << ','
                      << result.line_search_full_step_failure_reason << ','
                      << result.line_search_full_step_failure_stage << ','
                      << result.line_search_last_failure_reason << ','
                      << result.line_search_last_failure_stage << ','
                      << result.line_search_selected_alpha << ','
                      << result.line_search_us << ','
                      << result.first_control.steering_target_rad << ','
                      << result.first_control.throttle_command << ','
                      << max_plan_steering_delta << ','
                      << max_plan_throttle_delta << '\n';
            ++row_index;
        }
        return row_index == 0 ? 1 : 0;
    } catch (const std::exception &error) {
        std::cerr << "mpc_bag_solver_replay: " << error.what() << '\n';
        return 2;
    }
}

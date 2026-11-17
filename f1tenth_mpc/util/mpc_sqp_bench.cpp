// Offline benchmark of the production SQP on frozen controller problems.
//
//   mpc_sqp_bench solve CASE [ITERATIONS=4] [RADIUS=0.5]
//       One SQP solve from the case's nominal controls; prints per-iteration
//       model-versus-exact data as JSON.
//   mpc_sqp_bench drive CASE CYCLES [ITERATIONS=4] [DEADLINE_MS=22] [RADIUS=0.5]
//                       [MEASUREMENT_E_Y_SD_M=0] [MEASUREMENT_E_PSI_SD_RAD=0] [SEED=1]
//       Closed loop through mpc_cycle_solve on the source plant itself:
//       the first command of each published plan is applied for 25 ms, a
//       deferred cycle holds the previous command, and the run ends at the
//       first wall contact. Wall-clock timing is real. The controller sees
//       the plant state with errors added to e_y and e_psi each cycle
//       (localization error); the plant itself does not. The errors are
//       independent Gaussians, or, when NOISE_FILE names a file of
//       "e_y_error e_psi_error" lines (blank line between recordings), a
//       randomly chosen recording replayed from a random offset: the real
//       localization error with its tails and time structure.
//
// CASE is a file written by mpc_export_case. This tool never touches ROS or the simulator.
#include "mpc_case.hpp"
#include "mpc_source_sqp.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

using nlohmann::json;
namespace fs = std::filesystem;
using sdu_apex::mpc_util::Case;

namespace {
bool load_case(const fs::path &path, Case *ctx,
               MpcModelControl_t controls[PREDICTION_HORIZON])
{
    if (!sdu_apex::mpc_util::read_case(path.string(), ctx)) return false;
    std::copy_n(ctx->controls, PREDICTION_HORIZON, controls);
    return true;
}

// MPC_OVERRIDE="corridor_margin_m=0.05,weight_grip_loss=0" replaces the
// captured objective weights and margins for this run.
void apply_overrides(MpcSolverConfiguration_t *model)
{
    if (const char *threads = std::getenv("MPC_THREADS")) model->jacobian_threads = std::atoi(threads);
    const char *text = std::getenv("MPC_OVERRIDE");
    if (!text) return;
    const std::pair<const char *, float MpcSolverConfiguration_t::*> fields[] = {
        {"weight_e_y", &MpcSolverConfiguration_t::weight_e_y},
        {"weight_e_psi", &MpcSolverConfiguration_t::weight_e_psi},
        {"weight_u", &MpcSolverConfiguration_t::weight_u},
        {"weight_u_overspeed", &MpcSolverConfiguration_t::weight_u_overspeed},
        {"overspeed_curvature_gain_m", &MpcSolverConfiguration_t::overspeed_curvature_gain_m},
        {"underspeed_curvature_gain_m", &MpcSolverConfiguration_t::underspeed_curvature_gain_m},
        {"weight_grip_loss", &MpcSolverConfiguration_t::weight_grip_loss},
        {"grip_loss_tolerance", &MpcSolverConfiguration_t::grip_loss_tolerance},
        {"weight_v", &MpcSolverConfiguration_t::weight_v},
        {"weight_r", &MpcSolverConfiguration_t::weight_r},
        {"weight_steering_command", &MpcSolverConfiguration_t::weight_steering_command},
        {"weight_steering_target", &MpcSolverConfiguration_t::weight_steering_target},
        {"weight_throttle_command", &MpcSolverConfiguration_t::weight_throttle_command},
        {"weight_steering_slew", &MpcSolverConfiguration_t::weight_steering_slew},
        {"weight_throttle_slew", &MpcSolverConfiguration_t::weight_throttle_slew},
        {"terminal_multiplier", &MpcSolverConfiguration_t::terminal_multiplier},
        {"corridor_margin_m", &MpcSolverConfiguration_t::corridor_margin_m},
        {"first_prediction_corridor_margin_m", &MpcSolverConfiguration_t::first_prediction_corridor_margin_m},
        {"nonlinear_corridor_tolerance_m", &MpcSolverConfiguration_t::nonlinear_corridor_tolerance_m}};
    std::string list = text;
    for (size_t begin = 0; begin < list.size();) {
        size_t end = list.find(',', begin);
        if (end == std::string::npos) end = list.size();
        const std::string item = list.substr(begin, end - begin);
        const size_t eq = item.find('=');
        bool known = false;
        for (const auto &[name, member] : fields)
            if (eq != std::string::npos && item.substr(0, eq) == name) {
                model->*member = std::stof(item.substr(eq + 1));
                known = true;
            }
        if (!known) throw std::runtime_error("unknown MPC_OVERRIDE entry: " + item);
        begin = end + 1;
    }
}

MpcCycleConfiguration_t make_configuration(const Case &ctx,
                                              int iterations)
{
    MpcCycleConfiguration_t cfg{};
    cfg.model = ctx.model;
    apply_overrides(&cfg.model);
    cfg.sqp_violation_penalty = 1.e6f;
    cfg.sqp_accuracy_tolerance = 0.05f;
    if (const char *r = std::getenv("SQP_ACCURACY")) cfg.sqp_accuracy_tolerance = std::stof(r);
    cfg.sqp_iterations = iterations;
    // The production configurations cap the derivative passes at one force substep.
    cfg.derivative_force_substeps = std::getenv("DERIVATIVE_SUBSTEPS") ? std::atoi(std::getenv("DERIVATIVE_SUBSTEPS")) : 1;
    return cfg;
}

json percentile_summary(std::vector<double> v)
{
    if (v.empty()) return nullptr;
    std::sort(v.begin(), v.end());
    const auto at = [&](double q) { return v[static_cast<size_t>(q * (v.size() - 1))]; };
    return {{"p50", at(.5)}, {"p95", at(.95)}, {"p99", at(.99)}, {"max", v.back()}};
}

int solve_case(const fs::path &dir, int iterations, double radius)
{
    auto ctx = std::make_unique<Case>();
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    if (!load_case(dir, ctx.get(), controls)) return 3;
    auto cfg = make_configuration(*ctx, iterations);
    cfg.objective_schedule = &ctx->objective;
    sdu_apex::source_sqp::Options options{};
    options.maximum_iterations = iterations;
    options.radius = radius;
    if (const char *r = std::getenv("SQP_PENALTY")) options.violation_penalty = std::atof(r);
    options.accuracy_tolerance = cfg.sqp_accuracy_tolerance;
    auto result = std::make_unique<sdu_apex::source_sqp::Result>();
    const bool ok = sdu_apex::source_sqp::solve(
        ctx->initial_state, ctx->progress_m, ctx->trajectory.data(),
        ctx->trajectory.size(), ctx->lap_length_m, ctx->prediction_dt, cfg,
        controls, options, result.get());
    json out = {{"case", dir.string()}, {"ok", ok}};
    if (ok) {
        json history = json::array();
        for (const auto &it : result->history)
            history.push_back({
                {"accepted", it.accepted}, {"accurate", it.accurate},
                {"feasible_before", it.feasible_before},
                {"violation_before", it.violation_before}, {"violation_trial", it.violation_trial},
                {"qp_converged", it.qp_converged},
                {"qp_iterations", it.qp_iterations}, {"qp_rounds", it.qp_rounds},
                {"step_inf", it.step_inf}, {"radius", it.radius}, {"active", it.radius_active},
                {"cost_before", it.cost_before}, {"cost_trial", it.cost_trial},
                {"predicted", it.predicted_reduction}, {"actual", it.actual_reduction},
                {"ratio", it.ratio}, {"rejected", it.rejected_trials},
                {"slip_before", it.grip_cost_before}, {"slip_trial", it.grip_cost_trial},
                {"slip_predicted", it.grip_cost_predicted},
                {"new_slip", it.newly_active_grip}, {"lin_ms", it.linearization_ms}, {"model_ms", it.model_ms},
                {"qp_ms", it.qp_ms}, {"roll_ms", it.rollout_ms}, {"ms", it.total_ms},
                {"corrections", it.corrections}, {"attempts", it.attempts}});
        out["initial_feasible"] = result->initial_feasible;
        out["final_feasible"] = result->final_feasible;
        out["final_publishable"] = result->final_publishable;
        out["converged"] = result->converged;
        out["restarted"] = result->restarted;
        out["termination"] = static_cast<int>(result->termination);
        out["initial_cost"] = result->initial_cost;
        out["final_cost"] = result->final_cost;
        out["initial_violation"] = result->initial_violation;
        out["final_violation"] = result->final_violation;
        out["final_margin"] = result->final_minimum_corridor_m;
        out["rollouts"] = result->rollouts;
        out["radius"] = result->radius;
        out["history"] = std::move(history);
        out["controls"] = json::array();
        for (const auto &control : result->optimal.controls)
            out["controls"].push_back({control.steering_target_rad, control.throttle_command});
        out["rollout"] = json::array();
        for (int k = 1; k <= PREDICTION_HORIZON; ++k) {
            const auto &state = result->optimal.states[k].plant;
            MpcCycleSourceCorridorCorners_t g{};
            if (!mpc_cycle_source_exact_corridor_corners(&state, ctx->trajectory.data(),
                    ctx->trajectory.size(), ctx->lap_length_m, result->optimal.progress[k],
                    &ctx->model, k, &g)) return 4;
            out["rollout"].push_back({{"stage", k}, {"s", result->optimal.progress[k]},
                {"u", state.u}, {"e_y", state.e_y}, {"e_psi", state.e_psi},
                {"steering", state.actual_steering_angle},
                {"margin", std::min(*std::min_element(g.left, g.left + 4),
                                    *std::min_element(g.right, g.right + 4))}});
        }
    }
    std::cout << out.dump() << '\n';
    return 0;
}

// Recorded localization errors, one vector per recording.
std::vector<std::vector<std::pair<double, double>>> load_noise_recordings(const char *path)
{
    std::vector<std::vector<std::pair<double, double>>> recordings(1);
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot open NOISE_FILE ") + path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            if (!recordings.back().empty()) recordings.emplace_back();
            continue;
        }
        std::istringstream fields(line);
        double e_y = 0., e_psi = 0.;
        fields >> e_y >> e_psi;
        recordings.back().emplace_back(e_y, e_psi);
    }
    if (recordings.back().empty()) recordings.pop_back();
    if (recordings.empty()) throw std::runtime_error("empty NOISE_FILE");
    return recordings;
}

int drive(const fs::path &dir, int cycles, int iterations, double deadline_ms, double radius,
          double e_y_sd, double e_psi_sd, unsigned noise_seed)
{
    std::mt19937 generator(noise_seed);
    std::normal_distribution<double> unit(0., 1.);
    std::vector<std::vector<std::pair<double, double>>> recordings;
    const std::pair<double, double> *recording = nullptr;
    size_t recording_size = 0, recording_offset = 0;
    double recording_sign = 1.;
    if (const char *path = std::getenv("NOISE_FILE")) {
        recordings = load_noise_recordings(path);
        const auto &chosen = recordings[generator() % recordings.size()];
        recording = chosen.data();
        recording_size = chosen.size();
        recording_offset = generator() % recording_size;
        recording_sign = (generator() & 1u) ? 1. : -1.;
    }
    auto ctx = std::make_unique<Case>();
    MpcModelControl_t seed[PREDICTION_HORIZON]{};
    if (!load_case(dir, ctx.get(), seed)) return 3;
    auto cfg = make_configuration(*ctx, iterations);
    auto memory = std::make_unique<MpcCycleMemory_t>();
    memory->sqp_radius = radius;
    memory->command_epoch_ns = ctx->command_epoch_ns;
    memory->warm_start_controls_valid = 1;
    memory->warm_start_control_count = PREDICTION_HORIZON;
    std::copy_n(seed, PREDICTION_HORIZON, memory->warm_start_controls);
    auto state = ctx->initial_state;
    double progress = ctx->progress_m;
    const auto epoch_step = static_cast<std::int64_t>(std::llround(1.e9 * ctx->prediction_dt));
    MpcModelControl_t held = seed[0];
    int published = 0, deferred = 0, longest_deferral = 0, run = 0, crashed_at = -1;
    int slow = 0, unconverged = 0, rejected_horizons = 0;
    std::vector<double> solve_ms, margins;
    json cycle_log = json::array();
    std::vector<double> racing_errors;     // |predicted - exact| / cost_before, accepted racing steps
    std::vector<double> rejected_errors;   // the same for trials that were not taken
    auto physical_corridor = ctx->model;
    physical_corridor.corridor_margin_m = physical_corridor.first_prediction_corridor_margin_m = 0.f;
    physical_corridor.nonlinear_corridor_tolerance_m = 0.f;
    for (int tick = 0; tick < cycles; ++tick) {
        cfg.command_epoch_ns = ctx->command_epoch_ns + epoch_step * tick;
        cfg.absolute_deadline_monotonic_us = mpc_cycle_monotonic_time_us() + deadline_ms * 1000.;
        auto result = std::make_unique<MpcCycleCycleResult_t>();
        auto measured = state;
        if (recording) {
            const auto &error = recording[(recording_offset + tick) % recording_size];
            measured.plant.e_y += recording_sign * error.first;
            measured.plant.e_psi += recording_sign * error.second;
        } else {
            measured.plant.e_y += e_y_sd * unit(generator);
            measured.plant.e_psi += e_psi_sd * unit(generator);
        }
        const auto status = mpc_cycle_solve(
            &measured, progress, ctx->trajectory.data(), ctx->trajectory.size(),
            ctx->lap_length_m, ctx->prediction_dt, PREDICTION_HORIZON, &cfg,
            memory.get(), result.get());
        for (int i = 0; i < result->source_iteration_metrics_count; ++i) {
            const auto &m = result->source_iteration_metrics[i];
            if (m.feasible_before && m.cost_before > 0. && std::isfinite(m.cost_trial) &&
                m.predicted_reduction > 0.)
                (m.accepted ? racing_errors : rejected_errors).push_back(std::abs(m.predicted_reduction -
                                                 (m.cost_before - m.cost_trial)) / m.cost_before);
        }
        const double ms = result->total_cycle_us / 1000.;
        solve_ms.push_back(ms);
        slow += ms >= deadline_ms;
        const bool ok = status == MPC_CYCLE_CYCLE_ACCEPTED_FEASIBLE;
        rejected_horizons += status == MPC_CYCLE_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
        if (ok) {
            // Exactly what the controller node would publish.
            held.steering_target_rad = result->published_steering_command;
            held.throttle_command = result->published_throttle;
            ++published;
            run = 0;
            unconverged += !result->source_exact_nlp_kkt_converged;
        } else {
            ++deferred;
            longest_deferral = std::max(longest_deferral, ++run);
        }
        MpcCycleState_t next{};
        double next_progress = 0.;
        if (!mpc_cycle_source_exact_stage(&state, progress, &held, ctx->trajectory.data(),
                ctx->trajectory.size(), ctx->lap_length_m, ctx->prediction_dt, &next,
                &next_progress)) {
            crashed_at = tick;
            break;
        }
        MpcCycleSourceCorridorCorners_t g{};
        double margin = INFINITY;
        if (mpc_cycle_source_exact_corridor_corners(&next.plant, ctx->trajectory.data(),
                ctx->trajectory.size(), ctx->lap_length_m, next_progress, &ctx->model, 1, &g))
            for (int c = 0; c < 4; ++c) margin = std::min({margin, g.left[c], g.right[c]});
        double clearance = INFINITY;
        if (!mpc_cycle_source_exact_corridor_corners(&next.plant, ctx->trajectory.data(),
                ctx->trajectory.size(), ctx->lap_length_m, next_progress, &physical_corridor, 1, &g))
            return 4;
        for (int c = 0; c < 4; ++c) clearance = std::min({clearance, g.left[c], g.right[c]});
        margins.push_back(margin);
        cycle_log.push_back({{"t", tick}, {"s", progress}, {"u", state.plant.u},
                             {"status", static_cast<int>(status)}, {"seed_failure", result->sqp_seed_failure}, {"ms", ms},
                             {"margin", margin},
                             {"physical_clearance", clearance},
                             {"steering", state.plant.actual_steering_angle},
                             {"e_y", state.plant.e_y}, {"e_psi", state.plant.e_psi},
                             {"initial_feasible", bool(result->source_exact_nlp_initial_source_feasible)},
                             {"final_feasible", bool(result->source_exact_nlp_final_source_feasible)},
                             {"horizon_margin", result->source_exact_nlp_final_original_margin_m},
                             {"violation", result->source_exact_nlp_final_original_violation},
                             {"initial_cost", result->source_exact_nlp_initial_source_cost},
                             {"final_cost", result->source_exact_nlp_final_source_cost},
                             {"iters", result->sqp_iterations_attempted}});
        state = next;
        progress = next_progress;
        if (clearance < 0.) {
            crashed_at = tick;
            break;
        }
    }
    json out = {{"cycles", cycles}, {"ran", static_cast<int>(solve_ms.size())},
                {"published", published}, {"rejected_horizons", rejected_horizons}, {"deferred", deferred},
                {"longest_deferral", longest_deferral}, {"slow_cycles", slow},
                {"unconverged_published", unconverged}, {"crashed_at", crashed_at},
                {"final_progress", progress}, {"solve_ms", percentile_summary(solve_ms)},
                {"racing_errors", racing_errors}, {"rejected_errors", rejected_errors},
                {"log", std::move(cycle_log)}};
    std::cout << out.dump() << '\n';
    return crashed_at >= 0 ? 1 : 0;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "usage: mpc_sqp_bench {solve|drive} CASE ...\n";
        return 2;
    }
    const std::string mode = argv[1];
    const fs::path dir = argv[2];
    const auto number = [&](int i, double fallback) {
        return argc > i ? std::stod(argv[i]) : fallback;
    };
    if (mode == "solve")
        return solve_case(dir, static_cast<int>(number(3, 4)), number(4, .5));
    if (mode == "drive")
        return drive(dir, static_cast<int>(number(3, 400)), static_cast<int>(number(4, 4)),
                     number(5, 22.), number(6, .5), number(7, 0.), number(8, 0.),
                     static_cast<unsigned>(number(9, 1.)));
    return 2;
}

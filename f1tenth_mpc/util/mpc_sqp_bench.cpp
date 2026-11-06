// Offline benchmark of the production SQP on frozen controller problems.
//
//   mpc_sqp_bench solve CASE_DIR [ITERATIONS=4] [RADIUS=0.5] [FLOOR=1]
//       One SQP solve from CASE_DIR/initial.controls; prints per-iteration
//       model-versus-exact data as JSON.
//   mpc_sqp_bench drive CASE_DIR CYCLES [ITERATIONS=4] [DEADLINE_MS=22]
//       Closed loop through mpc_rti_solve_cycle on the source plant itself:
//       the first command of each published plan is applied for 25 ms, a
//       deferred cycle holds the previous command, and the run ends at the
//       first wall contact. Wall-clock timing is real.
//
// CASE_DIR contains source_context.mctx and initial.controls, as written by
// the live fixture exporter. This tool never touches ROS or the simulator.
#include "mpc_qp_failure_context.hpp"
#include "mpc_source_sqp.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using nlohmann::json;
namespace fs = std::filesystem;
using sdu_apex::qp_fixture::FailureReplayContext;

namespace {
bool load_case(const fs::path &dir, FailureReplayContext *ctx,
               MpcModelControl_t controls[PREDICTION_HORIZON])
{
    if (!sdu_apex::qp_fixture::read_failure_context(
            (dir / "source_context.mctx").string(), ctx))
        return false;
    std::ifstream in(dir / "initial.controls");
    for (int k = 0; k < PREDICTION_HORIZON; ++k)
        if (!(in >> controls[k].steering_target_rad >> controls[k].throttle_command))
            return false;
    return true;
}

MpcRtiCycleConfiguration_t make_configuration(const FailureReplayContext &ctx,
                                              int iterations)
{
    MpcRtiCycleConfiguration_t cfg{};
    cfg.model = ctx.model;
    cfg.model.world_sweep_check = nullptr;
    cfg.model.world_sweep_context = nullptr;
    cfg.model.world_map_rejection_only = 0;
    cfg.source_exact_nlp_live = 1;
    cfg.source_exact_nlp_constrained_qp = 1;
    cfg.source_exact_nlp_compact_qp = 1;
    cfg.source_exact_nlp_full_step_sqp = 1;
    cfg.source_exact_nlp_qp_regularization = 1.f;
    cfg.source_exact_nlp_qp_soft_weight = 1.e6f;
    cfg.source_exact_nlp_iterations = iterations;
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
    auto ctx = std::make_unique<FailureReplayContext>();
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    if (!load_case(dir, ctx.get(), controls)) return 3;
    auto cfg = make_configuration(*ctx, iterations);
    cfg.objective_schedule = &ctx->objective;
    sdu_apex::source_sqp::Options options{};
    options.maximum_iterations = iterations;
    options.radius = radius;
    if (const char *r = std::getenv("SQP_PENALTY")) options.violation_penalty = std::atof(r);
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
                {"accepted", it.accepted}, {"feasible_before", it.feasible_before},
                {"qp_iterations", it.qp_iterations}, {"qp_rounds", it.qp_rounds},
                {"step_inf", it.step_inf}, {"radius", it.radius}, {"active", it.radius_active},
                {"cost_before", it.cost_before}, {"cost_trial", it.cost_trial},
                {"predicted", it.predicted_reduction}, {"actual", it.actual_reduction},
                {"ratio", it.ratio}, {"rejected", it.rejected_trials},
                {"slip_before", it.slip_cost_before}, {"slip_trial", it.slip_cost_trial},
                {"slip_predicted", it.slip_cost_predicted},
                {"new_slip", it.newly_active_slip}, {"lin_ms", it.linearization_ms},
                {"qp_ms", it.qp_ms}, {"roll_ms", it.rollout_ms}, {"ms", it.total_ms}});
        out["initial_feasible"] = result->initial_feasible;
        out["final_feasible"] = result->final_feasible;
        out["converged"] = result->converged;
        out["termination"] = static_cast<int>(result->termination);
        out["initial_cost"] = result->initial_cost;
        out["final_cost"] = result->final_cost;
        out["initial_violation"] = result->initial_violation;
        out["final_violation"] = result->final_violation;
        out["rollouts"] = result->rollouts;
        out["radius"] = result->radius;
        out["history"] = std::move(history);
    }
    std::cout << out.dump() << '\n';
    return 0;
}

int drive(const fs::path &dir, int cycles, int iterations, double deadline_ms)
{
    auto ctx = std::make_unique<FailureReplayContext>();
    MpcModelControl_t seed[PREDICTION_HORIZON]{};
    if (!load_case(dir, ctx.get(), seed)) return 3;
    auto cfg = make_configuration(*ctx, iterations);
    auto memory = std::make_unique<MpcRtiMemory_t>();
    memory->command_epoch_ns = ctx->command_epoch_ns;
    memory->warm_start_controls_valid = 1;
    memory->warm_start_control_count = PREDICTION_HORIZON;
    std::copy_n(seed, PREDICTION_HORIZON, memory->warm_start_controls);
    auto state = ctx->initial_state;
    double progress = ctx->progress_m;
    const auto epoch_step = static_cast<std::int64_t>(std::llround(1.e9 * ctx->prediction_dt));
    MpcModelControl_t held = seed[0];
    int published = 0, deferred = 0, longest_deferral = 0, run = 0, crashed_at = -1;
    int slow = 0, unconverged = 0, degraded = 0;
    std::vector<double> solve_ms, margins;
    json cycle_log = json::array();
    for (int tick = 0; tick < cycles; ++tick) {
        cfg.command_epoch_ns = ctx->command_epoch_ns + epoch_step * tick;
        cfg.absolute_deadline_monotonic_us = mpc_rti_monotonic_time_us() + deadline_ms * 1000.;
        auto result = std::make_unique<MpcRtiCycleResult_t>();
        const auto status = mpc_rti_solve_cycle(
            &state, progress, ctx->trajectory.data(), ctx->trajectory.size(),
            ctx->lap_length_m, ctx->prediction_dt, PREDICTION_HORIZON, &cfg,
            memory.get(), result.get());
        const double ms = result->total_cycle_us / 1000.;
        solve_ms.push_back(ms);
        slow += ms >= deadline_ms;
        const bool ok = status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
                        status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED;
        degraded += status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED;
        if (ok) {
            held = result->candidate_controls[0];
            ++published;
            run = 0;
            unconverged += !result->source_exact_nlp_kkt_converged;
        } else {
            ++deferred;
            longest_deferral = std::max(longest_deferral, ++run);
        }
        MpcRtiState_t next{};
        double next_progress = 0.;
        if (!mpc_rti_source_exact_stage(&state, progress, &held, ctx->trajectory.data(),
                ctx->trajectory.size(), ctx->lap_length_m, ctx->prediction_dt, &next,
                &next_progress)) {
            crashed_at = tick;
            break;
        }
        MpcRtiSourceCorridorCorners_t g{};
        double margin = INFINITY;
        if (mpc_rti_source_exact_corridor_corners(&next.plant, ctx->trajectory.data(),
                ctx->trajectory.size(), ctx->lap_length_m, next_progress, &ctx->model, 1, &g))
            for (int c = 0; c < 4; ++c) margin = std::min({margin, g.left[c], g.right[c]});
        margins.push_back(margin);
        cycle_log.push_back({{"t", tick}, {"s", progress}, {"u", state.plant.u},
                             {"status", static_cast<int>(status)}, {"ms", ms},
                             {"margin", margin},
                             {"iters", result->sqp_iterations_attempted}});
        state = next;
        progress = next_progress;
        if (margin < 0.) {
            crashed_at = tick;
            break;
        }
    }
    json out = {{"cycles", cycles}, {"ran", static_cast<int>(solve_ms.size())},
                {"published", published}, {"degraded", degraded}, {"deferred", deferred},
                {"longest_deferral", longest_deferral}, {"slow_cycles", slow},
                {"unconverged_published", unconverged}, {"crashed_at", crashed_at},
                {"final_progress", progress}, {"solve_ms", percentile_summary(solve_ms)},
                {"log", std::move(cycle_log)}};
    std::cout << out.dump() << '\n';
    return crashed_at >= 0 ? 1 : 0;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "usage: mpc_sqp_bench {solve|drive} CASE_DIR ...\n";
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
                     number(5, 22.));
    return 2;
}

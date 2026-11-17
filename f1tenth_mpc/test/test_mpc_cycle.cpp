// End-to-end controller cycles on the checked-in practice raceline: the SQP
// must return a feasible plan, the published command must be that plan's first
// command, and consecutive receding-horizon cycles must keep succeeding.
#include "mpc_cycle.h"
#include "mpc_trajectory_io.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <memory>

#ifndef MPC_TEST_RACELINE
#error "MPC_TEST_RACELINE must name the practice raceline CSV"
#endif

namespace {
MpcCycleConfiguration_t practice_configuration()
{
    MpcCycleConfiguration_t cfg{};
    auto &m = cfg.model;
    m.weight_e_y = 4500.f; m.weight_e_psi = 1450.f; m.weight_u = 65.f;
    m.weight_u_overspeed = 20.f; m.overspeed_curvature_gain_m = .5f;
    m.underspeed_curvature_gain_m = .6f; m.weight_grip_loss = 200.f; m.grip_loss_tolerance = .2f;
    m.weight_v = 5.f; m.weight_r = 7.8125f; m.weight_steering_command = 4.5f;
    m.weight_steering_target = 5.f; m.weight_throttle_command = .1f;
    m.weight_steering_slew = 1500.f; m.weight_throttle_slew = 50.f;
    m.startup_nominal_throttle_seed = .25f; m.terminal_multiplier = 3.f;
    m.max_speed_mps = 16.f; m.active_speed_ceiling_mps = 16.f;
    m.max_steering_rad = .5235987756f;
    m.corridor_margin_m = .02f; m.first_prediction_corridor_margin_m = .02f;
    m.corridor_preview_halfwidth_m = .1f; m.jacobian_threads = 2;
    cfg.sqp_iterations = 3;
    cfg.sqp_violation_penalty = 1.e6f;
    cfg.sqp_accuracy_tolerance = 0.05f;
    return cfg;
}
} // namespace

TEST(MpcCycle, ReserveRestorationKeepsPhysicalWallsHard)
{
    MpcTrajectorySample_t track[6]{};
    const double x[6] = {0., 100., 100., 0., -100., -100.};
    const double y[6] = {0., 0., 100., 100., 100., 0.};
    for (int i = 0; i < 6; ++i) {
        track[i].s = 100. * i;
        track[i].x = x[i]; track[i].y = y[i];
        track[i].speed = 3.;
        track[i].left_bound = track[i].right_bound = .3;
    }
    size_t count = 6;
    double lap;
    ASSERT_TRUE(mpc_trajectory_prepare(track, &count, &lap));
    auto cfg = practice_configuration();
    cfg.model.corridor_margin_m = cfg.model.first_prediction_corridor_margin_m = .05f;
    for (float ey : {.1435f, .3f}) {
        auto state = std::make_unique<MpcCycleState_t>();
        state->plant.u = 3.f;
        state->plant.e_y = ey;
        ASSERT_TRUE(vehicle_model_initialize_plant_state(&state->plant));
        auto memory = std::make_unique<MpcCycleMemory_t>();
        auto result = std::make_unique<MpcCycleCycleResult_t>();
        const auto status = mpc_cycle_solve(state.get(), 5., track, count, lap,
            PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &cfg, memory.get(), result.get());
        if (ey == .1435f) {
            // Only 20 mm physical clearance: the 50 mm reserve cannot be
            // acquired instantly, but the entire physical plan must be safe.
            ASSERT_EQ(status, MPC_CYCLE_CYCLE_ACCEPTED_FEASIBLE);
            EXPECT_FALSE(result->source_exact_nlp_final_reserve_satisfied);
            EXPECT_GE(result->source_exact_nlp_final_physical_margin_m, 0.);
            auto physical = cfg;
            physical.model.corridor_margin_m = physical.model.first_prediction_corridor_margin_m = 0.f;
            auto checked = std::make_unique<MpcCycleNominal_t>();
            int failed;
            EXPECT_EQ(mpc_cycle_validate_candidate_plan(state.get(), 5., result->candidate_controls,
                track, count, lap, PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &physical,
                checked.get(), nullptr, &failed), MPC_CYCLE_ROLLOUT_OK);
        } else {
            EXPECT_EQ(status, MPC_CYCLE_CYCLE_REJECTED_NONLINEAR_ROLLOUT);
            EXPECT_LT(result->source_exact_nlp_final_physical_margin_m, 0.);
            EXPECT_EQ(result->candidate_control_count, 0);
            EXPECT_EQ(result->nonlinear_rollout_status, MPC_CYCLE_ROLLOUT_CORRIDOR);
            EXPECT_GE(result->nonlinear_failure_stage, 0);
            EXPECT_FALSE(memory->warm_start_controls_valid);
        }
    }
}

TEST(MpcCycle, ReceedingHorizonCyclesPublishTheirFirstCommand)
{
    std::ifstream csv(MPC_TEST_RACELINE);
    ASSERT_TRUE(csv.good());
    double lap_length = 0.;
    const auto trajectory = f1tenth_mpc::read_mpc_controller_trajectory(csv, &lap_length);
    ASSERT_GT(lap_length, 20.);

    auto state = std::make_unique<MpcCycleState_t>();
    state->plant.u = 5.f;
    ASSERT_TRUE(vehicle_model_initialize_plant_state(&state->plant));
    auto memory = std::make_unique<MpcCycleMemory_t>();
    mpc_cycle_memory_reset(memory.get());
    auto cfg = practice_configuration();
    double progress = 5.;
    constexpr std::int64_t kStageNs = 25000000;
    for (int cycle = 0; cycle < 6; ++cycle) {
        cfg.command_epoch_ns = (cycle + 1) * kStageNs;
        auto result = std::make_unique<MpcCycleCycleResult_t>();
        const auto status = mpc_cycle_solve(
            state.get(), progress, trajectory.data(), trajectory.size(), lap_length,
            PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &cfg, memory.get(), result.get());
        ASSERT_EQ(status, MPC_CYCLE_CYCLE_ACCEPTED_FEASIBLE) << "cycle " << cycle;
        ASSERT_EQ(result->candidate_control_count, PREDICTION_HORIZON);
        const auto &first = result->candidate_controls[0];
        EXPECT_EQ(result->published_steering_command, first.steering_target_rad);
        EXPECT_EQ(result->published_throttle, first.throttle_command);
        EXPECT_GE(result->published_throttle, MPC_THROTTLE_MIN_DRIVE);
        EXPECT_LE(result->published_throttle, MPC_THROTTLE_MAX);
        EXPECT_LE(std::abs(result->published_steering_command), cfg.model.max_steering_rad);
        EXPECT_TRUE(result->source_exact_nlp_final_source_feasible);
        // Every accepted racing step not marked as the exact-decrease fallback
        // predicted its exact reduction within the accuracy tolerance (5% of
        // the starting cost).
        for (int i = 0; i < result->source_iteration_metrics_count; ++i) {
            const auto &m = result->source_iteration_metrics[i];
            if (!m.feasible_before || !m.accepted || !m.accurate) continue;
            const double exact = m.cost_before - m.cost_trial;
            EXPECT_LE(std::abs(m.predicted_reduction - exact), .05 * m.cost_before)
                << "cycle " << cycle << " iteration " << i;
        }
        // Advance the simulated plant by the published command.
        MpcCycleState_t next{};
        double next_progress = 0.;
        ASSERT_TRUE(mpc_cycle_source_exact_stage(state.get(), progress, &first, trajectory.data(),
            trajectory.size(), lap_length, PREDICTION_DT_SECONDS, &next, &next_progress));
        *state = next;
        progress = next_progress;
    }
}

// End-to-end controller cycles on the checked-in practice raceline: the SQP
// must return a feasible plan, the published command must be that plan's first
// command, and consecutive receding-horizon cycles must keep succeeding.
#include "mpc_rti.h"
#include "mpc_trajectory_io.hpp"

#include <gtest/gtest.h>
#include <fstream>
#include <memory>

#ifndef MPC_TEST_RACELINE
#error "MPC_TEST_RACELINE must name the practice raceline CSV"
#endif

namespace {
MpcRtiCycleConfiguration_t practice_configuration()
{
    MpcRtiCycleConfiguration_t cfg{};
    auto &m = cfg.model;
    m.weight_e_y = 4500.f; m.weight_e_psi = 1450.f; m.weight_u = 65.f;
    m.weight_u_overspeed = 20.f; m.overspeed_curvature_gain_m = .5f;
    m.underspeed_curvature_gain_m = .6f; m.weight_longitudinal_slip = 5.f;
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
    return cfg;
}
} // namespace

TEST(MpcCycle, ReceedingHorizonCyclesPublishTheirFirstCommand)
{
    std::ifstream csv(MPC_TEST_RACELINE);
    ASSERT_TRUE(csv.good());
    double lap_length = 0.;
    const auto trajectory = f1tenth_mpc::read_mpc_controller_trajectory(csv, &lap_length);
    ASSERT_GT(lap_length, 20.);

    auto state = std::make_unique<MpcRtiState_t>();
    state->plant.u = 5.f;
    ASSERT_TRUE(vehicle_model_initialize_plant_state(&state->plant));
    auto memory = std::make_unique<MpcRtiMemory_t>();
    mpc_rti_memory_reset(memory.get());
    auto cfg = practice_configuration();
    double progress = 5.;
    constexpr std::int64_t kStageNs = 25000000;
    for (int cycle = 0; cycle < 6; ++cycle) {
        cfg.command_epoch_ns = (cycle + 1) * kStageNs;
        auto result = std::make_unique<MpcRtiCycleResult_t>();
        const auto status = mpc_rti_solve_cycle(
            state.get(), progress, trajectory.data(), trajectory.size(), lap_length,
            PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &cfg, memory.get(), result.get());
        ASSERT_EQ(status, MPC_RTI_CYCLE_ACCEPTED_FEASIBLE) << "cycle " << cycle;
        ASSERT_EQ(result->candidate_control_count, PREDICTION_HORIZON);
        const auto &first = result->candidate_controls[0];
        EXPECT_EQ(result->published_steering_command, first.steering_target_rad);
        EXPECT_EQ(result->published_throttle, first.throttle_command);
        EXPECT_GE(result->published_throttle, MPC_THROTTLE_MIN_DRIVE);
        EXPECT_LE(result->published_throttle, MPC_THROTTLE_MAX);
        EXPECT_LE(std::abs(result->published_steering_command), cfg.model.max_steering_rad);
        EXPECT_TRUE(result->source_exact_nlp_final_source_feasible);
        // Advance the simulated plant by the published command.
        MpcRtiState_t next{};
        double next_progress = 0.;
        ASSERT_TRUE(mpc_rti_source_exact_stage(state.get(), progress, &first, trajectory.data(),
            trajectory.size(), lap_length, PREDICTION_DT_SECONDS, &next, &next_progress));
        *state = next;
        progress = next_progress;
    }
}

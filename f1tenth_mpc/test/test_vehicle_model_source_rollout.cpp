#include "mpc_steering_variation.hpp"
#include "mpc_linearization.h"
#include "mpc_rti.h"
#include "../src/mpc_rti_internal.hpp"
#include "riccati_state_polygon.h"
#include "mpc_robust_contour.hpp"
#include "vehicle_model.h"
#include "unity_source_plant.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

TEST(MpcRtiRefinementQuality, KeepsCompletedPassWhenRefinementNeverRunsAdmm)
{
  EXPECT_TRUE(mpc_rti_should_keep_first_pass_after_refinement_deadline_skip(
      RICCATI_STATUS_MAX_ITERATIONS, -1, 1));
  EXPECT_TRUE(mpc_rti_should_keep_first_pass_after_refinement_deadline_skip(
      RICCATI_STATUS_OPTIMAL, -1, 1));
  EXPECT_FALSE(mpc_rti_should_keep_first_pass_after_refinement_deadline_skip(
      -1, -1, 1));
  EXPECT_FALSE(mpc_rti_should_keep_first_pass_after_refinement_deadline_skip(
      RICCATI_STATUS_MAX_ITERATIONS, RICCATI_STATUS_MAX_ITERATIONS, 1));
  EXPECT_FALSE(mpc_rti_should_keep_first_pass_after_refinement_deadline_skip(
      RICCATI_STATUS_MAX_ITERATIONS, -1, 0));
}

TEST(MpcRtiLineSearchClassification, ExactSourceMeritFailureGetsNormalGlobalization)
{
  // Captured run at progress 303.913m: the actual source rollout returned
  // OK, subsequent exact nonlinear merit validation returned INVALID_MODEL,
  // while the nominal source path remained exact-valid. Before this fix,
  // it triggered rejected_nonlinear_rollout with ZERO line-search trials,
  // despite an already-converged QP and time left in the callback.
  EXPECT_TRUE(mpc_rti_line_search_should_backtrack(
      MPC_RTI_ROLLOUT_OK, 0, 1, 0));
  EXPECT_TRUE(mpc_rti_line_search_should_backtrack(
      MPC_RTI_ROLLOUT_INVALID_MODEL, 0, 1, 0));
  EXPECT_TRUE(mpc_rti_line_search_should_backtrack(
      MPC_RTI_ROLLOUT_CORRIDOR, 0, 0, 0));
  EXPECT_TRUE(mpc_rti_line_search_should_backtrack(
      MPC_RTI_ROLLOUT_OK, 1, 1, 1));
  EXPECT_FALSE(mpc_rti_line_search_should_backtrack(
      MPC_RTI_ROLLOUT_OK, 1, 1, 0));
  EXPECT_FALSE(mpc_rti_line_search_should_backtrack(
      MPC_RTI_ROLLOUT_INVALID_MODEL, 0, 0, 0));
  EXPECT_FALSE(mpc_rti_line_search_should_backtrack(
      MPC_RTI_ROLLOUT_STATE_LIMIT, 0, 1, 0));
  EXPECT_FALSE(mpc_rti_line_search_should_backtrack(
      MPC_RTI_ROLLOUT_COMMAND_LIMIT, 0, 1, 0));
  // One invalid intermediate source rollout does not invalidate a smaller
  // step, but ONLY when the exact nominal is already certified.
  EXPECT_TRUE(mpc_rti_line_search_should_continue(
      MPC_RTI_ROLLOUT_INVALID_MODEL, 1));
  EXPECT_FALSE(mpc_rti_line_search_should_continue(
      MPC_RTI_ROLLOUT_INVALID_MODEL, 0));
  EXPECT_TRUE(mpc_rti_line_search_should_continue(
      MPC_RTI_ROLLOUT_CORRIDOR, 0));
  EXPECT_FALSE(mpc_rti_line_search_should_continue(
      MPC_RTI_ROLLOUT_STATE_LIMIT, 1));
}

TEST(MpcRtiLineSearchBudget, AllowsMeasuredRecoveryTrialsButRejectsInvalidTime)
{
  EXPECT_FALSE(mpc_rti_line_search_budget_available(2199.0));
  EXPECT_TRUE(mpc_rti_line_search_budget_available(2200.0));
  EXPECT_TRUE(mpc_rti_line_search_budget_available(3000.0));
  EXPECT_FALSE(mpc_rti_line_search_budget_available(0.0));
  EXPECT_FALSE(mpc_rti_line_search_budget_available(-1.0));
  EXPECT_FALSE(mpc_rti_line_search_budget_available(NAN));
  EXPECT_FALSE(mpc_rti_line_search_budget_available(INFINITY));
}

TEST(MpcRtiRefinementQuality, SelectsOnlyLowerExactCostValidAction)
{
  static_assert(MPC_RECOVERY_PREFIX_HORIZON == 3,
      "core proposal comparison and controller must validate the same 75 ms");
  EXPECT_TRUE(mpc_rti_refinement_improves_action(1, 100, 1, 90));
  EXPECT_FALSE(mpc_rti_refinement_improves_action(1, 100, 1, 101));
  EXPECT_FALSE(mpc_rti_refinement_improves_action(1, 100, 1, 100));
  EXPECT_FALSE(mpc_rti_refinement_improves_action(1, 100, 0, 90));
  EXPECT_TRUE(mpc_rti_refinement_improves_action(0, INFINITY, 1, 150));
  EXPECT_FALSE(mpc_rti_refinement_improves_action(1, 100, 1, NAN));
  EXPECT_FALSE(mpc_rti_refinement_improves_action(1, 100, 1, INFINITY));
  // Rejected horizons are compared on equal-duration exact prefixes, not
  // on their meaningless infinite full-tail objectives. Preserve R1 on ties
  // and when neither prefix validates; R2 must earn replacement.
  EXPECT_FALSE(mpc_rti_refinement_improves_action(0, NAN, 0, NAN));
  EXPECT_FALSE(mpc_rti_refinement_improves_action(1, -10, 1, -9));
  EXPECT_TRUE(mpc_rti_refinement_improves_action(1, -10, 1, -11));
}

TEST(RiccatiWarmStartEpoch, RefinementDoesNotAdvanceTheCommandSequence)
{
  RiccatiStepData_t steps[RICCATI_HORIZON]{};
  float terminal_Q[RICCATI_MAX_NX]{}, terminal_q[RICCATI_MAX_NX]{};
  float lower[RICCATI_MAX_NX], upper[RICCATI_MAX_NX], x0[RICCATI_MAX_NX]{};
  std::fill_n(lower, RICCATI_MAX_NX, -BIG_BOUND);
  std::fill_n(upper, RICCATI_MAX_NX, BIG_BOUND);
  for (auto &step : steps) {
    std::fill_n(step.x_lb, RICCATI_MAX_NX, -BIG_BOUND);
    std::fill_n(step.x_ub, RICCATI_MAX_NX, BIG_BOUND);
    std::fill_n(step.u_lb, RICCATI_MAX_NU, -1.f);
    std::fill_n(step.u_ub, RICCATI_MAX_NU, 1.f);
    for (int i = 0; i < RICCATI_MAX_NX; ++i) step.A[i][i] = 1.f;
  }
  RiccatiAdmmConfig_t config{};
  config.rho = config.rho_u = 1.f;
  config.tolerance = .01f;
  config.max_iterations = 1;
  config.over_relaxation = 1.f;
  for (int same_epoch : {0, 1}) {
    RiccatiAdmmState_t memory{};
    memory.initialized = 1;
    memory.rho = memory.rho_u = 1.f;
    for (int k = 0; k < RICCATI_HORIZON; ++k)
      memory.z_u[k][0] = .01f * (k + 1);
    config.reuse_same_epoch = same_epoch;
    RiccatiSolution_t result{};
    ASSERT_NE(riccati_admm_solve(steps, terminal_Q, terminal_q, lower, upper,
                                x0, &config, &memory, &result), RICCATI_STATUS_ERROR);
    for (int k = 0; k < RICCATI_HORIZON; ++k) {
      const int source = same_epoch ? k : std::min(k + 1, RICCATI_HORIZON - 1);
      EXPECT_FLOAT_EQ(result.u[k][0], .01f * (source + 1));
    }
  }
}

MpcModelState_t turning_state()
{
  MpcModelState_t state{};
  state.u = 6.0f;
  state.delayed_steering_command_1 = 0.35f;
  state.delayed_steering_command_2 = 0.35f;
  state.actual_steering_angle = 0.35f;
  state.delayed_throttle_command_1 = 0.30f;
  state.delayed_throttle_command_2 = 0.30f;
  return state;
}

TEST(VehicleModelSourceRollout, TrackingTransformClosesExactOffsetCircle)
{
  for (const float curvature : {-1.2f, -0.2f, 0.2f, 1.2f}) {
    MpcModelState_t state{};
    state.e_y = 0.12f;
    const float ds = 0.3f;
    const float angle = curvature * ds;
    const float radius = 1.0f / curvature - state.e_y;
    MpcStageResult_t physical{};
    physical.valid = 1;
    physical.next = state;
    physical.body_forward_delta_m = radius * std::sin(angle);
    physical.body_lateral_delta_m = radius * (1.0f - std::cos(angle));
    physical.body_yaw_delta_rad = angle;
    MpcStageResult_t projected{};
    ASSERT_TRUE(vehicle_model_reproject_tracking(
        &state, &physical, curvature, &projected));
    EXPECT_NEAR(projected.delta_s_m, ds, 2.0e-6f);
    EXPECT_NEAR(projected.next.e_y, state.e_y, 2.0e-6f);
    EXPECT_NEAR(projected.next.e_psi, 0.0f, 2.0e-6f);
  }
}

TEST(MpcReference, LoadedFrameClosesWorldMotionAndDifferentialAcrossKnots)
{
  MpcTrajectorySample_t points[4] = {
      {0, 0, 0, 0, .2, 5, 2, 2, 0},
      {1, 1, 0, .2, .4, 5, 2, 2, 0},
      {2, 2, .3, .6, .4, 5, 2, 2, 0},
      {3, 3, 1, 1, .4, 5, 2, 2, 0}};
  size_t count = 4;
  double length;
  ASSERT_TRUE(mpc_trajectory_prepare(points, &count, &length));
  const double progress = .7;
  const float arguments[5] = {.12f, -.08f, .4f, .015f, .12f};
  double next_progress;
  float ey, epsi, jacobian[3][6];
  ASSERT_TRUE(mpc_trajectory_advance_tracking_with_progress(points, count, length, progress,
      arguments[0], arguments[1], arguments[2], arguments[3], arguments[4],
      1.1, &next_progress, &ey, &epsi, jacobian));
  ASSERT_GT(next_progress, 1.0);
  MpcTrajectorySample_t start{}, end{};
  ASSERT_TRUE(mpc_trajectory_sample(points, count, length, progress, &start));
  ASSERT_TRUE(mpc_trajectory_sample(points, count, length, next_progress, &end));
  const double yaw = start.heading + arguments[1];
  const double world_x = start.x - std::sin(start.heading) * arguments[0] +
      std::cos(yaw) * arguments[2] - std::sin(yaw) * arguments[3];
  const double world_y = start.y + std::cos(start.heading) * arguments[0] +
      std::sin(yaw) * arguments[2] + std::cos(yaw) * arguments[3];
  EXPECT_NEAR(end.x - std::sin(end.heading) * ey, world_x, 1.0e-7);
  EXPECT_NEAR(end.y + std::cos(end.heading) * ey, world_y, 1.0e-7);
  EXPECT_NEAR(end.heading + epsi, yaw + arguments[4], 1.0e-7);
  for (int column = 0; column < 6; ++column) {
    float plus[5], minus[5];
    std::copy_n(arguments, 5, plus);
    std::copy_n(arguments, 5, minus);
    constexpr float epsilon = 2.0e-4f;
    if (column < 5) {
      plus[column] += epsilon;
      minus[column] -= epsilon;
    }
    double sp, sm;
    float yp, ym, hp, hm;
    ASSERT_TRUE(mpc_trajectory_advance_tracking(points, count, length,
        progress + (column == 5 ? epsilon : 0),
        plus[0], plus[1], plus[2], plus[3], plus[4], next_progress,
        &sp, &yp, &hp, nullptr));
    ASSERT_TRUE(mpc_trajectory_advance_tracking(points, count, length,
        progress - (column == 5 ? epsilon : 0),
        minus[0], minus[1], minus[2], minus[3], minus[4], next_progress,
        &sm, &ym, &hm, nullptr));
    EXPECT_NEAR(jacobian[0][column], (yp - ym) / (2 * epsilon), 3.0e-4f);
    EXPECT_NEAR(jacobian[1][column], (hp - hm) / (2 * epsilon), 3.0e-4f);
    EXPECT_NEAR(jacobian[2][column], (sp - sm) / (2 * epsilon), 3.0e-4f);
  }
}

TEST(MpcReference, ProjectedTerminalSampleExactlyMatchesResampling)
{
  MpcTrajectorySample_t points[4] = {
      {0, 0, 0, 0, .2, 5, 2, 2, 0},
      {1, 1, 0, .2, .4, 5, 2, 2, 0},
      {2, 2, .3, .6, .4, 5, 2, 2, 0},
      {3, 3, 1, 1, .4, 5, 2, 2, 0}};
  size_t count = 4;
  double length = 0;
  ASSERT_TRUE(mpc_trajectory_prepare(points, &count, &length));
  for (double start : {.6, 1.4, 2.25}) {
    double previous_progress = 0, sampled_progress = 0;
    float prev_ey = 0, prev_epsi = 0, ey = 0, epsi = 0;
    float previous_jac[3][6]{}, sampled_jac[3][6]{};
    MpcTrajectorySample_t reused{}, independently_resampled{};
    ASSERT_TRUE(mpc_trajectory_advance_tracking_with_progress(
        points, count, length, start, .02f, .01f, .09f,
        .001f, .005f, start + .09, &previous_progress,
        &prev_ey, &prev_epsi, previous_jac));
    ASSERT_TRUE(mpc_trajectory_advance_tracking_with_progress_sampled(
        points, count, length, start, .02f, .01f, .09f,
        .001f, .005f, start + .09, &sampled_progress,
        &ey, &epsi, sampled_jac, &reused));
    ASSERT_TRUE(mpc_trajectory_sample(
        points, count, length, sampled_progress, &independently_resampled));
    EXPECT_DOUBLE_EQ(previous_progress, sampled_progress);
    EXPECT_FLOAT_EQ(prev_ey, ey);
    EXPECT_FLOAT_EQ(prev_epsi, epsi);
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 6; ++col)
        EXPECT_FLOAT_EQ(previous_jac[row][col], sampled_jac[row][col]);
    EXPECT_DOUBLE_EQ(reused.s, independently_resampled.s);
    EXPECT_DOUBLE_EQ(reused.x, independently_resampled.x);
    EXPECT_DOUBLE_EQ(reused.y, independently_resampled.y);
    EXPECT_DOUBLE_EQ(reused.heading, independently_resampled.heading);
    EXPECT_DOUBLE_EQ(reused.curvature, independently_resampled.curvature);
    EXPECT_DOUBLE_EQ(reused.speed, independently_resampled.speed);
    EXPECT_DOUBLE_EQ(reused.left_bound, independently_resampled.left_bound);
    EXPECT_DOUBLE_EQ(reused.right_bound, independently_resampled.right_bound);
    EXPECT_DOUBLE_EQ(reused.acceleration, independently_resampled.acceleration);
    EXPECT_EQ(reused.source_witness_valid,
              independently_resampled.source_witness_valid);
  }
}

TEST(MpcReference, SourceWitnessInterpolatesStatesButHoldsCommands)
{
  MpcTrajectorySample_t points[3] = {
      {0., 0., 0., 0., 0., 4., 10., 10., 0.},
      {10., 10., 0., 0., 0., 4., 10., 10., 0.},
      {20., 20., 0., 0., 0., 4., 10., 10., 0.}};
  for (int k=0; k<3; ++k) {
    points[k].source_witness_valid = 1;
    points[k].source_heading_error = .05+k*.02;
    points[k].source_v_rear = -.2+k*.1;
    points[k].source_yaw_rate = .3;
    points[k].source_physical_steering = .02+k*.01;
    points[k].source_steering_target = .04+k*.1;
    points[k].source_throttle = .11+k*.1;
  }
  size_t count=3;
  double length;
  ASSERT_TRUE(mpc_trajectory_prepare(points, &count, &length));
  MpcTrajectorySample_t sample{};
  ASSERT_TRUE(mpc_trajectory_sample(points, count, length, 5., &sample));
  EXPECT_TRUE(sample.source_witness_valid);
  EXPECT_NEAR(sample.source_heading_error, .06, 1.e-12);
  EXPECT_NEAR(sample.source_v_rear, -.15, 1.e-12);
  EXPECT_NEAR(sample.source_physical_steering, .025, 1.e-12);
  EXPECT_DOUBLE_EQ(sample.source_steering_target, .04);
  EXPECT_DOUBLE_EQ(sample.source_throttle, .11);
  points[1].source_witness_valid=0;
  EXPECT_FALSE(mpc_trajectory_prepare(points, &count, &length));
  points[1].source_witness_valid=1;
  points[1].source_throttle=0.;
  EXPECT_FALSE(mpc_trajectory_prepare(points, &count, &length));
}

TEST(MpcReference, SourceWitnessSeedsTorqueAndUsesItsLateralReferences)
{
  MpcTrajectorySample_t points[3] = {
      {0., 0., 0., 0., 0., 4., 10., 10., 0.},
      {10., 10., 0., 0., 0., 4., 10., 10., 0.},
      {20., 20., 0., 0., 0., 4., 10., 10., 0.}};
  for (auto &p : points) {
    p.source_witness_valid=1;
    p.source_heading_error=.07; p.source_v_rear=-.2; p.source_yaw_rate=.3;
    p.source_physical_steering=.02; p.source_steering_target=.04; p.source_throttle=.11;
  }
  size_t count=3; double length;
  ASSERT_TRUE(mpc_trajectory_prepare(points, &count, &length));
  MpcRtiConfiguration_t c{};
  c.weight_e_y=c.weight_e_psi=c.weight_u=c.weight_u_overspeed=1.f;
  c.weight_v=c.weight_r=c.weight_steering_command=c.weight_steering_target=1.f;
  c.weight_throttle_command=c.terminal_multiplier=1.f;
  c.max_speed_mps=c.active_speed_ceiling_mps=16.f;
  c.max_steering_rad=SOURCE_MAX_STEERING_RAD;
  c.corridor_margin_m=c.first_prediction_corridor_margin_m=.05f;
  MpcRtiState_t initial{};
  initial.plant.u=4.f;
  initial.plant.delayed_throttle_command_1=initial.plant.delayed_throttle_command_2=.2f;
  MpcRtiNominal_t nominal{};
  MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0., nullptr, points, count, length,
      PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &c, &nominal, refs));
  for (int k=0; k<PREDICTION_HORIZON; ++k) {
    EXPECT_FLOAT_EQ(nominal.controls[k].steering_target_rad, .04f);
    EXPECT_FLOAT_EQ(nominal.controls[k].throttle_command, .11f);
    EXPECT_FLOAT_EQ(refs[k].e_psi, .07f);
    EXPECT_FLOAT_EQ(refs[k].v, -.2f);
    EXPECT_FLOAT_EQ(refs[k].r, .3f);
    EXPECT_FLOAT_EQ(refs[k].steering_command, .02f);
  }
  c.active_speed_ceiling_mps=3.f;
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0., nullptr, points, count, length,
      PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &c, &nominal, refs));
  for (int k=1; k<PREDICTION_HORIZON; ++k) {
    EXPECT_EQ(refs[k].source_witness_valid, 0);
    EXPECT_FLOAT_EQ(refs[k].u, 3.f);
    EXPECT_FLOAT_EQ(refs[k].v, 0.f);
    EXPECT_FLOAT_EQ(refs[k].e_psi, 0.f);
    EXPECT_FLOAT_EQ(nominal.controls[k].throttle_command, .2f);
  }
  c.active_speed_ceiling_mps=16.f;
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0., nullptr, points, count, length,
      PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &c, &nominal, refs));
  EXPECT_EQ(refs[1].source_witness_valid, 1);
  EXPECT_FLOAT_EQ(refs[1].v, -.2f);
  EXPECT_FLOAT_EQ(nominal.controls[1].throttle_command, .11f);
  c.reference_clearance_reserve_m=.05f;
  for (auto &point : points) point.left_bound=.25;
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0., nullptr, points, count, length,
      PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &c, &nominal, refs));
  EXPECT_LT(refs[1].e_y, 0.f);
  EXPECT_FLOAT_EQ(c.corridor_margin_m,.05f);
  MpcModelState_t target{};
  target.e_y=refs[1].e_y; target.e_psi=refs[1].e_psi;
  float lower_slack, upper_slack;
  ASSERT_TRUE(mpc_rti_compute_corridor_slack(&target, refs[1].left_bound,
      refs[1].right_bound, &c, 2, &lower_slack, &upper_slack));
  EXPECT_GE(lower_slack,.04999f);
  EXPECT_GE(upper_slack,.04999f);
  // Removing the optional soft target shift restores the native path without
  // weakening the body-footprint wall constraints or altering witness states.
  const float shifted_left_bound = refs[1].left_bound;
  const float shifted_right_bound = refs[1].right_bound;
  c.reference_clearance_reserve_m=0.f;
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0., nullptr, points, count, length,
      PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &c, &nominal, refs));
  EXPECT_FLOAT_EQ(refs[1].e_y,0.f);
  EXPECT_FLOAT_EQ(refs[1].e_psi,.07f);
  EXPECT_FLOAT_EQ(refs[1].v,-.2f);
  EXPECT_FLOAT_EQ(refs[1].r,.3f);
  EXPECT_FLOAT_EQ(refs[1].source_steering_target,.04f);
  EXPECT_FLOAT_EQ(refs[1].left_bound,shifted_left_bound);
  EXPECT_FLOAT_EQ(refs[1].right_bound,shifted_right_bound);
  EXPECT_FLOAT_EQ(c.corridor_margin_m,.05f);
}

TEST(MpcReference, ProgressSensitivityPersistsAcrossLaterStages)
{
  MpcTrajectorySample_t points[4] = {
      {0, 0, 0, 0, .2, 5, 2, 2, 0},
      {1, 1, 0, .2, .4, 5, 2, 2, 0},
      {2, 2, .3, .6, .4, 5, 2, 2, 0},
      {3, 3, 1, 1, .4, 5, 2, 2, 0}};
  size_t count = 4;
  double length;
  ASSERT_TRUE(mpc_trajectory_prepare(points, &count, &length));
  auto advance = [&](double impulse, double result[3], bool derivatives) {
    double station = .75;
    float ey = .05f, epsi = .03f;
    double sensitivity[3] = {};
    for (int k = 0; k < 6; ++k) {
      double next;
      float y, p, jac[3][6];
      const float dx = .16f + (k == 0 ? impulse : 0);
      EXPECT_TRUE(mpc_trajectory_advance_tracking_with_progress(
          points, count, length, station, ey, epsi, dx, .002f, .045f,
          station + .16, &next, &y, &p, jac));
      double next_sensitivity[3] = {};
      for (int row = 0; row < 3; ++row)
        next_sensitivity[row] = jac[row][0]*sensitivity[0]
            + jac[row][1]*sensitivity[1] + jac[row][5]*sensitivity[2]
            + (k == 0 ? jac[row][2] : 0);
      std::copy_n(next_sensitivity, 3, sensitivity);
      station = next; ey = y; epsi = p;
    }
    if (derivatives) std::copy_n(sensitivity, 3, result);
    else { result[0] = ey; result[1] = epsi; result[2] = station; }
  };
  double plus[3], minus[3], sensitivity[3];
  constexpr double h = .0002;
  advance(h, plus, false);
  advance(-h, minus, false);
  advance(0, sensitivity, true);
  for (int row = 0; row < 3; ++row)
    EXPECT_NEAR(sensitivity[row], (plus[row]-minus[row])/(2*h), .001);
}

TEST(MpcReference, SourceMotionSensitivityChainsThroughLoadedPath)
{
  MpcTrajectorySample_t points[4] = {
      {0, 0, 0, 0, .2, 5, 2, 2, 0},
      {1, 1, 0, .2, .4, 5, 2, 2, 0},
      {2, 2, .3, .6, .4, 5, 2, 2, 0},
      {3, 3, 1, 1, .4, 5, 2, 2, 0}};
  size_t count = 4;
  double length;
  ASSERT_TRUE(mpc_trajectory_prepare(points, &count, &length));
  MpcModelState_t state = turning_state();
  state.u = 4.0f;
  state.e_y = .05f;
  state.e_psi = .08f;
  state.actual_steering_angle = .10f;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&state));
  const MpcModelControl_t control{.11f, .2f};
  MpcStageResult_t step{};
  MpcStageLinearization_t lin{};
  ASSERT_TRUE(mpc_vehicle_model_step_with_jacobian(
      &state, &control, .025f, 0.0f, &step, &lin));
  double station;
  float ey, epsi, geometry[2][5];
  ASSERT_TRUE(mpc_trajectory_advance_tracking(points, count, length, .96,
      state.e_y, state.e_psi, step.body_forward_delta_m,
      step.body_lateral_delta_m, step.body_yaw_delta_rad, 1.06,
      &station, &ey, &epsi, geometry));
  for (int input = 0; input < 2; ++input) {
    MpcModelControl_t cp = control, cm = control;
    constexpr float epsilon = 1.0e-4f;
    if (input == 0) {
      cp.steering_target_rad += epsilon;
      cm.steering_target_rad -= epsilon;
    } else {
      cp.throttle_command += epsilon;
      cm.throttle_command -= epsilon;
    }
    const auto plus = mpc_vehicle_model_step(&state, &cp, .025f, 0.0f);
    const auto minus = mpc_vehicle_model_step(&state, &cm, .025f, 0.0f);
    ASSERT_TRUE(plus.valid && minus.valid);
    double sp, sm;
    float yp, ym, hp, hm;
    ASSERT_TRUE(mpc_trajectory_advance_tracking(points, count, length, .96,
        state.e_y, state.e_psi, plus.body_forward_delta_m,
        plus.body_lateral_delta_m, plus.body_yaw_delta_rad, station,
        &sp, &yp, &hp, nullptr));
    ASSERT_TRUE(mpc_trajectory_advance_tracking(points, count, length, .96,
        state.e_y, state.e_psi, minus.body_forward_delta_m,
        minus.body_lateral_delta_m, minus.body_yaw_delta_rad, station,
        &sm, &ym, &hm, nullptr));
    for (int row = 0; row < 2; ++row) {
      float analytic = 0;
      for (int motion = 0; motion < 3; ++motion)
        analytic += geometry[row][motion + 2] * lin.body_motion_B[motion][input];
      EXPECT_NEAR(analytic, (row == 0 ? yp - ym : hp - hm) /
          (2 * epsilon), 5.0e-4f);
    }
  }
}

TEST(VehicleModelSourceRollout, CarriesLateralYawWheelAndSuspensionMotion)
{
  MpcModelState_t state = turning_state();
  const MpcModelControl_t control{0.35f, 0.30f};
  const MpcStageResult_t first = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(first.valid);
  ASSERT_TRUE(first.next.source_plant_initialized);
  ASSERT_TRUE(first.next.source_suspension_history_initialized);
  EXPECT_GT(std::abs(first.next.v), 1.0e-5f);
  EXPECT_GT(std::abs(first.next.r), 1.0e-5f);

  const MpcStageResult_t second = mpc_vehicle_model_step(
      &first.next, &control, 0.025f, 0.0f);
  ASSERT_TRUE(second.valid);
  EXPECT_NE(second.next.source_plant_internal_state[
                MPC_PLANT_STATE_WHEEL_OMEGA_FL],
            first.next.source_plant_internal_state[
                MPC_PLANT_STATE_WHEEL_OMEGA_FL]);
  EXPECT_NE(second.next.source_plant_internal_state[MPC_PLANT_STATE_ROLL],
            first.next.source_plant_internal_state[MPC_PLANT_STATE_ROLL]);
}

TEST(MpcRtiMemory, RecoveryResetClearsAdmmButPreservesIssuedSequence)
{
  MpcRtiMemory_t memory{};
  mpc_rti_memory_reset(&memory);

  memory.solver_state.initialized = 1;
  memory.solver_state.rho = 13.0f;
  memory.solver_state.rho_u = 17.0f;
  memory.solver_state.z_u[0][0] = 0.31f;
  memory.solver_state.y_x[2][3] = -0.27f;
  memory.nominal.valid = 1;
  memory.nominal.horizon = PREDICTION_HORIZON;
  memory.nominal.controls[0] = {0.12f, 0.37f};

  MpcModelControl_t issued[PREDICTION_HORIZON]{};
  for (int k = 0; k < PREDICTION_HORIZON; ++k)
    issued[k] = {0.01f * static_cast<float>(k),
        0.2f + 0.01f * static_cast<float>(k)};
  ASSERT_TRUE(mpc_rti_memory_set_warm_start_controls(
      &memory, issued, PREDICTION_HORIZON));

  mpc_rti_memory_reset_solver_state(&memory);

  EXPECT_EQ(memory.solver_state.initialized, 0);
  EXPECT_FLOAT_EQ(memory.solver_state.rho, 0.0f);
  EXPECT_FLOAT_EQ(memory.solver_state.rho_u, 0.0f);
  EXPECT_FLOAT_EQ(memory.solver_state.z_u[0][0], 0.0f);
  EXPECT_FLOAT_EQ(memory.solver_state.y_x[2][3], 0.0f);
  ASSERT_EQ(memory.warm_start_controls_valid, 1);
  ASSERT_EQ(memory.warm_start_control_count, PREDICTION_HORIZON);
  EXPECT_FLOAT_EQ(memory.warm_start_controls[0].steering_target_rad,
      issued[0].steering_target_rad);
  EXPECT_FLOAT_EQ(memory.warm_start_controls[0].throttle_command,
      issued[0].throttle_command);
  ASSERT_EQ(memory.nominal.valid, 1);
  EXPECT_FLOAT_EQ(memory.nominal.controls[0].steering_target_rad, 0.12f);
  EXPECT_FLOAT_EQ(memory.nominal.controls[0].throttle_command, 0.37f);
}

TEST(VehicleModelSourceRollout, YawAndPoseUseOnlySourcePlantTransition)
{
  MpcModelState_t state = turning_state();
  state.u = 3.5f;
  state.delayed_steering_command_1 = 0.15f;
  state.delayed_steering_command_2 = -0.25f;
  state.actual_steering_angle = -0.25f;
  const MpcModelControl_t control{-0.25f, 0.25f};
  MpcModelState_t initialized = state;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&initialized));

  UnitySourcePlantInputC input{};
  input.state[UNITY_SOURCE_U] = initialized.u;
  input.state[UNITY_SOURCE_V] =
      initialized.v + initialized.r * 0.15532;
  input.state[UNITY_SOURCE_YAW_RATE] = initialized.r;
  std::copy_n(initialized.source_plant_internal_state +
                  MPC_PLANT_STATE_WHEEL_OMEGA_FL,
              4, input.state + UNITY_SOURCE_WHEEL_OMEGA_FL);
  input.state[UNITY_SOURCE_BODY_HEIGHT] = initialized.source_plant_internal_state[
      MPC_PLANT_STATE_BODY_HEIGHT];
  input.state[UNITY_SOURCE_BODY_HEIGHT_RATE] = initialized.source_plant_internal_state[
      MPC_PLANT_STATE_BODY_HEIGHT_RATE];
  input.state[UNITY_SOURCE_ROLL] = initialized.source_plant_internal_state[
      MPC_PLANT_STATE_ROLL];
  input.state[UNITY_SOURCE_ROLL_RATE] = initialized.source_plant_internal_state[
      MPC_PLANT_STATE_ROLL_RATE];
  input.state[UNITY_SOURCE_PITCH] = initialized.source_plant_internal_state[
      MPC_PLANT_STATE_PITCH];
  input.state[UNITY_SOURCE_PITCH_RATE] = initialized.source_plant_internal_state[
      MPC_PLANT_STATE_PITCH_RATE];
  input.state[UNITY_SOURCE_PHYSICAL_STEERING] =
      initialized.actual_steering_angle;
  std::copy_n(initialized.source_plant_internal_state +
                  MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL,
              4, input.state + UNITY_SOURCE_SUSPENSION_JOUNCE_FL);
  std::copy_n(initialized.source_plant_internal_state +
                  MPC_PLANT_STATE_LOW_FORWARD_SPEED_TIME_FL,
              4, input.state + UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL);
  std::copy_n(initialized.source_plant_internal_state +
                  MPC_PLANT_STATE_LOW_SIDE_SPEED_TIME_FL,
              4, input.state + UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL);
  input.steering_target_rad = control.steering_target_rad;
  input.throttle_normalized = control.throttle_command;
  input.suspension_history_initialized =
      initialized.source_suspension_history_initialized;

  UnitySourcePlantOutputC expected{};
  ASSERT_TRUE(unity_source_plant_step_c(&input, 0.025, &expected));
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);
  EXPECT_NEAR(stage.next.r, expected.state[UNITY_SOURCE_YAW_RATE], 1.0e-6f);
  EXPECT_NEAR(stage.next.v,
      expected.state[UNITY_SOURCE_V] -
          expected.state[UNITY_SOURCE_YAW_RATE] * 0.15532,
      1.0e-6f);
  EXPECT_NEAR(stage.body_yaw_delta_rad,
      expected.integrated_yaw_delta_rad, 1.0e-6f);
}

TEST(VehicleModelSourceRollout,
     MaximumFloatSteeringCommandClipsOnlyRepresentationOverrun)
{
  MpcModelState_t state{};
  state.u = 1.0f;
  state.delayed_steering_command_1 = SOURCE_MAX_STEERING_RAD;
  state.delayed_steering_command_2 = SOURCE_MAX_STEERING_RAD;
  state.actual_steering_angle = SOURCE_MAX_STEERING_RAD;
  state.delayed_throttle_command_1 = 0.05f;
  state.delayed_throttle_command_2 = 0.05f;

  const MpcStageResult_t stage = mpc_vehicle_model_propagate(
      &state, 0.005f, 0.0f);

  ASSERT_TRUE(stage.valid);
  EXPECT_TRUE(std::isfinite(stage.next.u));
  EXPECT_NEAR(stage.next.actual_steering_angle,
      static_cast<double>(SOURCE_MAX_STEERING_RAD), 1.0e-6);
}

TEST(VehicleModelSourceRollout, ZeroThrottleUsesTheSourceBrakeBranch)
{
  MpcModelState_t state{};
  state.u = 3.0f;
  state.delayed_throttle_command_1 = 0.5f;
  state.delayed_throttle_command_2 = 0.0f;
  const MpcModelControl_t control{};
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);
  EXPECT_FLOAT_EQ(stage.next.delayed_throttle_command_2, 0.0f);
  EXPECT_NE(stage.branch_flags & MPC_STAGE_PLANT_BRAKE, 0U);
  EXPECT_LT(stage.next.u, state.u);
}

TEST(VehicleModelSourceRollout,
     BrakeLockedWheelReportsZeroForwardSlipAtLowSpeed)
{
  MpcModelState_t state{};
  state.u = 0.12f;
  state.delayed_throttle_command_1 = 0.0f;
  state.delayed_throttle_command_2 = 0.0f;
  state.source_plant_initialized = 1;
  state.source_suspension_history_initialized = 1;
  state.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT] =
      0.00464804098;
  state.source_plant_internal_state[MPC_PLANT_STATE_WHEEL_OMEGA_FL] = 0.0;
  state.source_plant_internal_state[MPC_PLANT_STATE_WHEEL_OMEGA_FR] = 0.0;
  state.source_plant_internal_state[MPC_PLANT_STATE_WHEEL_OMEGA_RL] = 0.0;
  state.source_plant_internal_state[MPC_PLANT_STATE_WHEEL_OMEGA_RR] = 0.0;

  const MpcStageResult_t stage = mpc_vehicle_model_propagate(
      &state, 0.001f, 0.0f);

  ASSERT_TRUE(stage.valid);
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(stage.next.source_plant_internal_state[
        MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel], 0.0);
    // The capture records zero WheelHit.forwardSlip for locked wheels in this
    // low-speed hard-braking regime; do not preserve the old inferred slip.
    EXPECT_DOUBLE_EQ(stage.output.longitudinal_slip[wheel], 0.0);
  }
  EXPECT_EQ(MPC_PLANT_INTERNAL_STATE_COUNT, 22);
  EXPECT_EQ(MPC_OPT_PHYSICAL_NX, MPC_MODEL_NX - 4);
}

TEST(VehicleModelSourceRollout,
     DirectThrottleCommandDrivesThePlantAndAdvancesHistory)
{
  MpcModelState_t state = turning_state();
  state.delayed_throttle_command_1 = 0.20f;
  state.delayed_throttle_command_2 = 0.30f;
  const MpcModelControl_t control{0.0f, 0.70f};
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);
  /* The command drives this command-time stage immediately and is retained
   * in slot 2 for the next causal sensor-time prediction. */
  EXPECT_FLOAT_EQ(stage.next.delayed_throttle_command_1, 0.30f);
  EXPECT_FLOAT_EQ(stage.next.delayed_throttle_command_2, 0.70f);
  EXPECT_TRUE(std::isfinite(stage.next.u));
  const MpcStageResult_t queued = mpc_vehicle_model_propagate(
      &state, 0.025f, 0.0f);
  ASSERT_TRUE(queued.valid);
  EXPECT_NE(stage.next.source_plant_internal_state[
                MPC_PLANT_STATE_WHEEL_OMEGA_RL],
            queued.next.source_plant_internal_state[
                MPC_PLANT_STATE_WHEEL_OMEGA_RL]);
}

TEST(VehicleModelSourceRollout,
     ContinuousPropagationDoesNotAdvancePacketCommandQueues)
{
  MpcModelState_t state = turning_state();
  state.delayed_steering_command_1 = 0.12f;
  state.delayed_steering_command_2 = -0.18f;
  state.delayed_throttle_command_1 = 0.24f;
  state.delayed_throttle_command_2 = 0.32f;
  const MpcStageResult_t stage = mpc_vehicle_model_propagate(
      &state, 0.005f, 0.0f);
  ASSERT_TRUE(stage.valid);
  EXPECT_FLOAT_EQ(stage.next.delayed_steering_command_1, 0.12f);
  EXPECT_FLOAT_EQ(stage.next.delayed_steering_command_2, -0.18f);
  EXPECT_FLOAT_EQ(stage.next.delayed_throttle_command_1, 0.24f);
  EXPECT_FLOAT_EQ(stage.next.delayed_throttle_command_2, 0.32f);
}

TEST(VehicleModelSourceRollout,
     SteeringCommandShiftsPastHistoryAndQueuesNextCommand)
{
  MpcModelState_t state = turning_state();
  state.delayed_steering_command_1 = 0.18f;
  state.delayed_steering_command_2 = -0.12f;
  const MpcModelControl_t control{0.30f, 0.30f};
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);
  EXPECT_FLOAT_EQ(stage.next.delayed_steering_command_1, -0.12f);
  EXPECT_FLOAT_EQ(stage.next.delayed_steering_command_2, 0.30f);
}

TEST(VehicleModelSourceRollout,
     ConvertsRearAxleLateralVelocityAtTheUnityPlantBoundary)
{
  constexpr double com_forward_from_rear_axle_m = 0.15532;
  MpcModelState_t state = turning_state();
  state.r = 0.9f;
  state.v = -0.08f;
  state.source_plant_initialized = 1;
  state.source_suspension_history_initialized = 1;
  state.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT] =
      0.00464804098;
  state.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT_RATE] = 0.001;
  state.source_plant_internal_state[MPC_PLANT_STATE_ROLL] = 0.01;
  state.source_plant_internal_state[MPC_PLANT_STATE_ROLL_RATE] = 0.02;
  state.source_plant_internal_state[MPC_PLANT_STATE_PITCH] = -0.01;
  state.source_plant_internal_state[MPC_PLANT_STATE_PITCH_RATE] = 0.01;
  for (int wheel = 0; wheel < 4; ++wheel)
    state.source_plant_internal_state[
        MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] =
        state.u / 0.059;
  state.source_plant_internal_state[MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL] =
      0.0001;
  state.source_plant_internal_state[MPC_PLANT_STATE_SUSPENSION_JOUNCE_FR] =
      -0.0001;
  state.source_plant_internal_state[MPC_PLANT_STATE_SUSPENSION_JOUNCE_RL] =
      0.0002;
  state.source_plant_internal_state[MPC_PLANT_STATE_SUSPENSION_JOUNCE_RR] =
      -0.0002;

  const MpcModelControl_t control{};
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);

  /* Independently step the source plant from the same physical state. Its v
   * is COM-referenced; the public MPC state is rear-axle referenced. */
  UnitySourcePlantInputC reference_input{};
  reference_input.state[UNITY_SOURCE_U] = state.u;
  reference_input.state[UNITY_SOURCE_V] =
      state.v + state.r * com_forward_from_rear_axle_m;
  reference_input.state[UNITY_SOURCE_YAW_RATE] = state.r;
  for (int wheel = 0; wheel < 4; ++wheel)
    reference_input.state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] =
        state.source_plant_internal_state[
            MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel];
  reference_input.state[UNITY_SOURCE_BODY_HEIGHT] =
      state.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT];
  reference_input.state[UNITY_SOURCE_BODY_HEIGHT_RATE] =
      state.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT_RATE];
  reference_input.state[UNITY_SOURCE_ROLL] =
      state.source_plant_internal_state[MPC_PLANT_STATE_ROLL];
  reference_input.state[UNITY_SOURCE_ROLL_RATE] =
      state.source_plant_internal_state[MPC_PLANT_STATE_ROLL_RATE];
  reference_input.state[UNITY_SOURCE_PITCH] =
      state.source_plant_internal_state[MPC_PLANT_STATE_PITCH];
  reference_input.state[UNITY_SOURCE_PITCH_RATE] =
      state.source_plant_internal_state[MPC_PLANT_STATE_PITCH_RATE];
  for (int wheel = 0; wheel < 4; ++wheel)
    reference_input.state[UNITY_SOURCE_SUSPENSION_JOUNCE_FL + wheel] =
        state.source_plant_internal_state[
            MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL + wheel];
  reference_input.state[UNITY_SOURCE_PHYSICAL_STEERING] =
      state.actual_steering_angle;
  reference_input.steering_target_rad = control.steering_target_rad;
  reference_input.throttle_normalized = control.throttle_command;
  reference_input.suspension_history_initialized = 1;
  UnitySourcePlantOutputC reference_output{};
  ASSERT_TRUE(unity_source_plant_step_c(
      &reference_input, 0.025, &reference_output));

  EXPECT_NEAR(stage.next.u,
              reference_output.state[UNITY_SOURCE_U], 1.0e-6);
  EXPECT_NEAR(stage.next.r,
              reference_output.state[UNITY_SOURCE_YAW_RATE], 1.0e-6);
  EXPECT_NEAR(stage.next.v,
              reference_output.state[UNITY_SOURCE_V] -
                  reference_output.state[UNITY_SOURCE_YAW_RATE] *
                      com_forward_from_rear_axle_m,
              1.0e-6);
  constexpr int source_internal_index[MPC_OPT_PHYSICAL_NX - 6] = {
      UNITY_SOURCE_WHEEL_OMEGA_FL, UNITY_SOURCE_WHEEL_OMEGA_FR,
      UNITY_SOURCE_WHEEL_OMEGA_RL, UNITY_SOURCE_WHEEL_OMEGA_RR,
      UNITY_SOURCE_BODY_HEIGHT, UNITY_SOURCE_BODY_HEIGHT_RATE,
      UNITY_SOURCE_ROLL, UNITY_SOURCE_ROLL_RATE,
      UNITY_SOURCE_PITCH, UNITY_SOURCE_PITCH_RATE,
      UNITY_SOURCE_SUSPENSION_JOUNCE_FL,
      UNITY_SOURCE_SUSPENSION_JOUNCE_FR,
      UNITY_SOURCE_SUSPENSION_JOUNCE_RL,
      UNITY_SOURCE_SUSPENSION_JOUNCE_RR};
  for (int index = 0; index < MPC_OPT_PHYSICAL_NX - 6; ++index) {
    EXPECT_DOUBLE_EQ(stage.next.source_plant_internal_state[index],
        reference_output.state[source_internal_index[index]]);
  }
  // Wheel rotation is an odometry encoder accumulator, not an MPC dynamic
  // state. The MPC retains only values that feed subsequent plant dynamics.
  for (int wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(stage.output.longitudinal_slip[wheel],
                     reference_output.longitudinal_slip[wheel]);
    EXPECT_DOUBLE_EQ(stage.output.lateral_slip[wheel],
                     reference_output.lateral_slip[wheel]);
    EXPECT_DOUBLE_EQ(stage.output.normal_load_n[wheel],
                     reference_output.normal_load_n[wheel]);
  }
}

TEST(VehicleModelSourceRollout,
     UsesDirectHorizonControlAndQueuedPreCommandPropagation)
{
  MpcModelState_t state = turning_state();
  state.delayed_throttle_command_1 = 0.30f;
  state.delayed_throttle_command_2 = 0.0f;
  const MpcModelControl_t control{0.0f, 0.40f};
  const MpcStageResult_t horizon_step = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(horizon_step.valid);
  EXPECT_EQ(horizon_step.branch_flags & MPC_STAGE_PLANT_BRAKE, 0U);
  EXPECT_FLOAT_EQ(horizon_step.next.delayed_throttle_command_1, 0.0f);
  EXPECT_FLOAT_EQ(horizon_step.next.delayed_throttle_command_2, 0.40f);

  const MpcStageResult_t pre_command_step = mpc_vehicle_model_propagate(
      &state, 0.025f, 0.0f);
  ASSERT_TRUE(pre_command_step.valid);
  EXPECT_NE(pre_command_step.branch_flags & MPC_STAGE_PLANT_BRAKE, 0U);
  EXPECT_LT(pre_command_step.next.u, horizon_step.next.u);

  MpcModelState_t queued_state = state;
  queued_state.delayed_throttle_command_2 = 0.30f;
  const MpcStageResult_t queued_step = mpc_vehicle_model_propagate(
      &queued_state, 0.025f, 0.0f);
  ASSERT_TRUE(queued_step.valid);
  EXPECT_GT(queued_step.next.u, pre_command_step.next.u);
}

TEST(RiccatiSolverDeadline, StopsOnlyAfterACompleteAdmmIteration)
{
  RiccatiStepData_t steps[RICCATI_HORIZON] = {};
  float terminal_q[RICCATI_MAX_NX] = {};
  float terminal_x_lb[RICCATI_MAX_NX] = {};
  float terminal_x_ub[RICCATI_MAX_NX] = {};
  float x0[RICCATI_MAX_NX] = {};
  for (int i = 0; i < RICCATI_MAX_NX; ++i) {
    terminal_q[i] = 1.0f;
    terminal_x_lb[i] = i == 0 ? -0.05f : -BIG_BOUND;
    terminal_x_ub[i] = i == 0 ? 0.05f : BIG_BOUND;
    x0[i] = 0.0f;
  }
  for (int k = 0; k < RICCATI_HORIZON; ++k) {
    for (int i = 0; i < RICCATI_MAX_NX; ++i) {
      steps[k].A[i][i] = 1.0f;
      steps[k].Q_diag[i] = 1.0f;
      steps[k].x_lb[i] = i == 0 ? -0.05f : -BIG_BOUND;
      steps[k].x_ub[i] = i == 0 ? 0.05f : BIG_BOUND;
    }
    steps[k].B[0][0] = 1.0f;
    steps[k].R_diag[0] = 1.0f;
    steps[k].u_lb[0] = -1.0f;
    steps[k].u_ub[0] = 1.0f;
    steps[k].R_diag[1] = 1.0f;
    steps[k].u_lb[1] = -1.0f;
    steps[k].u_ub[1] = 1.0f;
  }
  x0[0] = 2.0f;

  RiccatiAdmmConfig_t config{};
  config.rho = 7.0f;
  config.rho_u = 7.0f;
  config.tolerance = 1.0e-9f;
  config.max_iterations = 100;
  config.over_relaxation = 1.0f;
  config.max_solve_time_us = 0.001;

  RiccatiAdmmState_t memory{};
  RiccatiSolution_t solution{};
  const RiccatiStatus_t status = riccati_admm_solve(
      steps, terminal_q, terminal_q, terminal_x_lb, terminal_x_ub, x0,
      &config, &memory, &solution);
  EXPECT_EQ(status, RICCATI_STATUS_MAX_ITERATIONS);
  EXPECT_GE(solution.iterations, 1);
  EXPECT_LT(solution.iterations, config.max_iterations);
  EXPECT_EQ(solution.time_budget_exhausted, 1);
  EXPECT_TRUE(memory.initialized);
}

TEST(RiccatiSolverColdStart, AuxiliaryIndefiniteSchurDoesNotRejectAugmentedQp)
{
  // Reproduce the failure mechanism reported by the 4th long race: an
  // auxiliary zero-rho Riccati initialization can fail BEFORE iteration 0,
  // even when the actual positively regularized ADMM linear solves exist.
  // This synthetic example isolates that numerical property; it is NOT a
  // certificate of convexity for an arbitrary indefinite original objective.
  RiccatiStepData_t steps[RICCATI_HORIZON]{};
  float terminal_Q[RICCATI_MAX_NX]{};
  float terminal_q[RICCATI_MAX_NX]{};
  float low[RICCATI_MAX_NX]{},high[RICCATI_MAX_NX]{},x0[RICCATI_MAX_NX]{};
  for (int s = 0; s < RICCATI_MAX_NX; ++s) {
    low[s] = -BIG_BOUND;
    high[s] = BIG_BOUND;
    terminal_Q[s] = 1.f;
  }
  for (int k = 0; k < RICCATI_HORIZON; ++k) {
    for (int s = 0; s < RICCATI_MAX_NX; ++s) {
      steps[k].A[s][s] = 1.f;
      steps[k].Q_diag[s] = 1.f;
      steps[k].x_lb[s] = -BIG_BOUND;
      steps[k].x_ub[s] = BIG_BOUND;
    }
    steps[k].R_diag[0] = .5f;
    steps[k].R_diag[1] = .5f;
    steps[k].R_cross = .8f;
    steps[k].u_lb[0] = steps[k].u_lb[1] = -1.f;
    steps[k].u_ub[0] = steps[k].u_ub[1] = 1.f;
  }
  RiccatiAdmmState_t aux{};
  RiccatiSolution_t test_solution{};
  EXPECT_EQ(0, riccati_solver_pass(
      steps,terminal_Q,terminal_q,low,high,x0,0.f,0.f,
      aux.z_x,aux.y_x,aux.z_u,aux.y_u,
      test_solution.x,test_solution.u));
  RiccatiAdmmConfig_t config{};
  config.rho = 7.f;
  config.rho_u = 7.f;
  config.max_iterations = 20;
  config.adaptive_rho = 0;
  config.tolerance = .01f;
  RiccatiAdmmState_t memory{};
  RiccatiSolution_t solution{};
  const auto result=riccati_admm_solve(steps,terminal_Q,terminal_q,
      low,high,x0,&config,&memory,&solution);
  EXPECT_NE(result,RICCATI_STATUS_ERROR);
  EXPECT_GT(solution.iterations,0);
  EXPECT_TRUE(std::isfinite(solution.u[0][0]));
  EXPECT_TRUE(std::isfinite(solution.u[0][1]));
}

TEST(RiccatiSolverScaling, PreservesThePhysicalSolution)
{
  RiccatiStepData_t steps[RICCATI_HORIZON] = {};
  float terminal_Q[RICCATI_MAX_NX] = {};
  float terminal_q[RICCATI_MAX_NX] = {};
  float terminal_x_lb[RICCATI_MAX_NX] = {};
  float terminal_x_ub[RICCATI_MAX_NX] = {};
  float x0[RICCATI_MAX_NX] = {};
  for (int i = 0; i < RICCATI_MAX_NX; ++i) {
    terminal_Q[i] = 0.5f + 0.1f * static_cast<float>(i);
    terminal_x_lb[i] = -BIG_BOUND;
    terminal_x_ub[i] = BIG_BOUND;
  }
  terminal_x_lb[0] = -0.15f;
  terminal_x_ub[0] = 0.15f;
  x0[0] = 0.30f;
  // An unbounded wheel-rate coordinate can legitimately exceed BIG_BOUND's
  // marker value after its solver scaling; scaling must not turn the marker
  // into a real ±50 rad/s constraint.
  x0[6] = 80.0f;
  for (int k = 0; k < RICCATI_HORIZON; ++k) {
    for (int i = 0; i < RICCATI_MAX_NX; ++i) {
      steps[k].A[i][i] = 1.0f;
      steps[k].Q_diag[i] = 0.5f + 0.1f * static_cast<float>(i);
      steps[k].x_lb[i] = -BIG_BOUND;
      steps[k].x_ub[i] = BIG_BOUND;
    }
    steps[k].x_lb[0] = -0.15f;
    steps[k].x_ub[0] = 0.15f;
    steps[k].B[0][0] = 0.25f;
    steps[k].N[0][0] = -0.03f;
    steps[k].N[1][1] = -0.02f;
    steps[k].R_diag[0] = 0.5f;
    steps[k].R_diag[1] = 0.5f;
    steps[k].u_lb[0] = -1.0f;
    steps[k].u_ub[0] = 1.0f;
    steps[k].u_lb[1] = -1.0f;
    steps[k].u_ub[1] = 1.0f;
  }

  RiccatiAdmmConfig_t plain{};
  plain.rho = 7.0f;
  plain.rho_u = 7.0f;
  plain.tolerance = 1.0e-5f;
  plain.max_iterations = 500;
  plain.adaptive_rho = 1;
  plain.adaptive_rho_update_period = 5;
  plain.over_relaxation = 1.4f;
  RiccatiAdmmConfig_t scaled = plain;
  scaled.use_scaling = 1;
  const float scales[RICCATI_MAX_NX] = {
      0.10f, 0.10f, 0.5f, 1.0f, 4.0f, 0.5f,
      400.0f, 400.0f, 400.0f, 400.0f,
      0.10f, 1.0f, 0.25f, 4.0f, 0.25f, 4.0f,
      0.05f, 0.05f, 0.05f, 0.05f,
      1.0f, 0.5f, 1.0f};
  std::copy_n(scales, RICCATI_MAX_NX, scaled.state_scale);
  scaled.input_scale[0] = 0.5f;
  scaled.input_scale[1] = 1.0f;

  RiccatiAdmmState_t plain_memory{};
  RiccatiAdmmState_t scaled_memory{};
  RiccatiSolution_t plain_solution{};
  RiccatiSolution_t scaled_solution{};
  const RiccatiStatus_t plain_status = riccati_admm_solve(
      steps, terminal_Q, terminal_q, terminal_x_lb, terminal_x_ub, x0,
      &plain, &plain_memory, &plain_solution);
  const RiccatiStatus_t scaled_status = riccati_admm_solve(
      steps, terminal_Q, terminal_q, terminal_x_lb, terminal_x_ub, x0,
      &scaled, &scaled_memory, &scaled_solution);
  ASSERT_NE(plain_status, RICCATI_STATUS_ERROR);
  ASSERT_NE(scaled_status, RICCATI_STATUS_ERROR);
  EXPECT_EQ(plain_solution.status, plain_status);
  EXPECT_EQ(scaled_solution.status, scaled_status);
  EXPECT_FLOAT_EQ(plain_solution.primal_residual,
      std::max(plain_solution.state_primal_residual,
               plain_solution.input_primal_residual));
  EXPECT_FLOAT_EQ(plain_solution.dual_residual,
      std::max(plain_solution.state_dual_residual,
               plain_solution.input_dual_residual));
  for (int k = 0; k < RICCATI_HORIZON; ++k) {
    for (int a = 0; a < MPC_MODEL_NU; ++a)
      EXPECT_NEAR(plain_solution.u[k][a], scaled_solution.u[k][a], 2.0e-3f);
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      EXPECT_NEAR(plain_solution.x[k + 1][i], scaled_solution.x[k + 1][i],
          2.0e-3f);
  }

  RiccatiAdmmConfig_t capped = scaled;
  capped.rho_max = 32.0f;
  capped.tolerance = 1.0e-8f;
  capped.max_iterations = 50;
  capped.adaptive_rho = 1;
  capped.adaptive_rho_update_period = 1;
  RiccatiAdmmState_t capped_memory{};
  capped_memory.initialized = 1;
  capped_memory.scaling_enabled = 1;
  capped_memory.rho = 100.0f;
  capped_memory.rho_u = 100.0f;
  std::copy_n(capped.state_scale, RICCATI_MAX_NX,
      capped_memory.state_scale);
  std::copy_n(capped.input_scale, MPC_MODEL_NU,
      capped_memory.input_scale);
  RiccatiSolution_t capped_solution{};
  ASSERT_NE(RICCATI_STATUS_ERROR, riccati_admm_solve(
      steps, terminal_Q, terminal_q, terminal_x_lb, terminal_x_ub, x0,
      &capped, &capped_memory, &capped_solution));
  RiccatiDebugInfo_t capped_debug{};
  riccati_debug_get_last(&capped_debug);
  EXPECT_FLOAT_EQ(capped_debug.rho_start, 32.0f);
  EXPECT_FLOAT_EQ(capped_debug.rho_u_start, 32.0f);
  EXPECT_LE(capped_debug.rho, 32.0f);
  EXPECT_LE(capped_debug.rho_u, 32.0f);
}

TEST(RiccatiFeedback, ExportsPhysicalGainWithoutChangingTheQpSolution)
{
  RiccatiStepData_t steps[RICCATI_HORIZON]{};
  float terminal_Q[RICCATI_MAX_NX]{}, terminal_q[RICCATI_MAX_NX]{};
  float lo[RICCATI_MAX_NX], hi[RICCATI_MAX_NX], x0[RICCATI_MAX_NX]{};
  for (int i=0; i<RICCATI_MAX_NX; ++i) {lo[i]=-BIG_BOUND;hi[i]=BIG_BOUND;}
  lo[0]=-1.f; hi[0]=1.f; terminal_Q[0]=3.f;
  for (auto &stage : steps) {
    for (int i=0; i<RICCATI_MAX_NX; ++i) {
      stage.A[i][i]=1.f; stage.x_lb[i]=lo[i]; stage.x_ub[i]=hi[i];
    }
    stage.B[0][0]=.1f; stage.Q_diag[0]=3.f;
    for (int a=0; a<2; ++a) {stage.R_diag[a]=4.f;stage.u_lb[a]=-1.f;stage.u_ub[a]=1.f;}
  }
  RiccatiAdmmConfig_t config{};
  config.rho=config.rho_u=7.f; config.rho_max=32.f;
  config.max_iterations=1;config.tolerance=1.e-6f;
  config.use_scaling=1;
  std::fill_n(config.state_scale,RICCATI_MAX_NX,1.f);
  config.state_scale[0]=2.f;
  config.input_scale[0]=.5f;config.input_scale[1]=1.f;
  RiccatiAdmmState_t old_memory{}, feedback_memory{};
  RiccatiSolution_t old_solution{}, feedback_solution{};
  float gain[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX];
  const auto a=riccati_admm_solve(steps,terminal_Q,terminal_q,lo,hi,x0,&config,&old_memory,&old_solution);
  const auto b=riccati_admm_solve_with_feedback(steps,terminal_Q,terminal_q,lo,hi,x0,&config,
      &feedback_memory,&feedback_solution,nullptr,gain);
  EXPECT_EQ(a,b);
  EXPECT_EQ(std::memcmp(&old_solution,&feedback_solution,sizeof(old_solution)),0);
  const float b_scaled=.1f*.5f/2.f;
  const float terminal_augmented=3.f*2.f*2.f+7.f;
  const float expected_scaled=-b_scaled*terminal_augmented/(4.f*.5f*.5f+7.f+b_scaled*b_scaled*terminal_augmented);
  EXPECT_NEAR(gain[RICCATI_HORIZON-1][0][0], .5f/2.f*expected_scaled,1.e-7f);
  // Invalid core configuration fails after the scaling wrapper has started.
  // Optional outputs must not be read/unscaled if the core produced none.
  config.over_relaxation = 3.f;
  float primal[RICCATI_HORIZON][RICCATI_MAX_NU];
  for (int k=0; k<RICCATI_HORIZON; ++k)
    for (int a=0; a<RICCATI_MAX_NU; ++a) {
      primal[k][a]=7.f;
      std::fill_n(gain[k][a],RICCATI_MAX_NX,7.f);
    }
  EXPECT_EQ(riccati_admm_solve_with_feedback(steps,terminal_Q,terminal_q,lo,hi,x0,
      &config,&feedback_memory,&feedback_solution,primal,gain),RICCATI_STATUS_ERROR);
  EXPECT_FLOAT_EQ(gain[0][0][0],7.f);
  EXPECT_FLOAT_EQ(primal[0][0],7.f);
}

TEST(RiccatiSolver, SymmetricValueRecursionMatchesUnfactoredPass)
{
  constexpr int nx = RICCATI_MAX_NX;
  constexpr int nu = MPC_MODEL_NU;
  constexpr int horizon = RICCATI_HORIZON;
  RiccatiStepData_t steps[horizon] = {};
  float terminal_q[RICCATI_MAX_NX] = {};
  float terminal_Q[RICCATI_MAX_NX] = {};
  float terminal_lb[RICCATI_MAX_NX] = {};
  float terminal_ub[RICCATI_MAX_NX] = {};
  float x0[RICCATI_MAX_NX] = {};
  float z_x[horizon + 1][RICCATI_MAX_NX] = {};
  float y_x[horizon + 1][RICCATI_MAX_NX] = {};
  float z_u[horizon][RICCATI_MAX_NU] = {};
  float y_u[horizon][RICCATI_MAX_NU] = {};
  float x_unfactored[horizon + 1][RICCATI_MAX_NX] = {};
  float u_unfactored[horizon][RICCATI_MAX_NU] = {};
  float x_factored[horizon + 1][RICCATI_MAX_NX] = {};
  float u_factored[horizon][RICCATI_MAX_NU] = {};

  for (int i = 0; i < nx; ++i) {
    terminal_Q[i] = 1.0f + 0.01f * static_cast<float>(i);
    terminal_lb[i] = -BIG_BOUND;
    terminal_ub[i] = BIG_BOUND;
    x0[i] = 0.001f * static_cast<float>(i + 1);
  }
  for (int k = 0; k < horizon; ++k) {
    RiccatiStepData_t &step = steps[k];
    for (int i = 0; i < nx; ++i) {
      step.A[i][i] = 0.8f;
      step.d[i] = 0.0001f * static_cast<float>(i + k);
      step.Q_diag[i] = 0.5f + 0.005f * static_cast<float>(i);
      step.x_lb[i] = -BIG_BOUND;
      step.x_ub[i] = BIG_BOUND;
      for (int j = 0; j < nx; ++j) {
        if (i != j) {
          step.A[i][j] = 0.0001f * static_cast<float>(1 + (i + j) % 5);
        }
      }
      for (int a = 0; a < nu; ++a) {
        step.B[i][a] = 0.001f * static_cast<float>(1 + (i + a) % 3);
        step.N[i][a] = 0.0001f * static_cast<float>(1 + (i + 2 * a) % 5);
      }
    }
    for (int a = 0; a < nu; ++a) {
      step.R_diag[a] = 1.0f + 0.1f * static_cast<float>(a);
      step.r[a] = 0.001f * static_cast<float>(k + a);
      step.u_lb[a] = -1.0f;
      step.u_ub[a] = 1.0f;
    }
  }

  RiccatiFactorization_t factorization{};
  ASSERT_TRUE(riccati_solver_factorize(steps, terminal_Q, terminal_lb,
      terminal_ub, 7.0f, 7.0f, &factorization));
  for (int k = 0; k <= horizon; ++k) {
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < nx; ++j) {
        EXPECT_TRUE(std::isfinite(factorization.P[k][i][j]));
        EXPECT_FLOAT_EQ(factorization.P[k][i][j],
                        factorization.P[k][j][i]);
      }
    }
  }

  ASSERT_TRUE(riccati_solver_pass(steps, terminal_Q, terminal_q,
      terminal_lb, terminal_ub, x0, 7.0f, 7.0f,
      z_x, y_x, z_u, y_u, x_unfactored, u_unfactored));
  ASSERT_TRUE(riccati_solver_pass_factored(steps, terminal_q, terminal_lb,
      terminal_ub, x0, &factorization, z_x, y_x, z_u, y_u,
      x_factored, u_factored));
  for (int k = 0; k <= horizon; ++k) {
    for (int i = 0; i < nx; ++i) {
      EXPECT_NEAR(x_factored[k][i], x_unfactored[k][i], 1.0e-5f);
    }
  }
  for (int k = 0; k < horizon; ++k) {
    for (int a = 0; a < nu; ++a) {
      EXPECT_NEAR(u_factored[k][a], u_unfactored[k][a], 1.0e-5f);
    }
  }
  // Quadratic factor reuse must still allow NEW affine dynamics. The private
  // ADMM P*d cache must never leak into this public RHS-only reuse contract.
  for (int k = 0; k < horizon; ++k)
    for (int i = 0; i < nx; ++i) steps[k].d[i] *= -3.0f;
  ASSERT_TRUE(riccati_solver_pass(steps, terminal_Q, terminal_q,
      terminal_lb, terminal_ub, x0, 7.0f, 7.0f,
      z_x, y_x, z_u, y_u, x_unfactored, u_unfactored));
  ASSERT_TRUE(riccati_solver_pass_factored(steps, terminal_q, terminal_lb,
      terminal_ub, x0, &factorization, z_x, y_x, z_u, y_u,
      x_factored, u_factored));
  for (int k = 0; k < horizon; ++k)
    for (int a = 0; a < nu; ++a)
      EXPECT_NEAR(u_factored[k][a], u_unfactored[k][a], 1.e-5f);
}

TEST(MpcRtiSlewCost, CrossTermEqualsSquaredCommandIncrement)
{
  RiccatiStepData_t stage{};
  float state[MPC_RTI_NX] = {};
  float control[MPC_RTI_NU] = {};
  constexpr float steering_weight = 17.0f;
  constexpr float throttle_weight = 3.5f;
  state[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] = 0.2f;
  state[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] = 0.4f;
  control[0] = -0.1f;
  control[1] = 0.7f;
  stage.Q_diag[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] =
      2.0f * steering_weight;
  stage.R_diag[0] = 2.0f * steering_weight;
  stage.N[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND][0] =
      -2.0f * steering_weight;
  stage.Q_diag[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] =
      2.0f * throttle_weight;
  stage.R_diag[1] = 2.0f * throttle_weight;
  stage.N[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND][1] =
      -2.0f * throttle_weight;

  const float expected = steering_weight * 0.3f * 0.3f +
      throttle_weight * 0.3f * 0.3f;
  EXPECT_NEAR(mpc_rti_matrix_stage_cost(&stage, state, control),
      expected, 1.0e-6f);
}

TEST(MpcRtiDirectInputCondensing, PreservesPlantJacobianAndBasePoint)
{
  constexpr int horizon = PREDICTION_HORIZON;
  MpcRtiState_t nominal_states[PREDICTION_HORIZON + 1] = {};
  MpcModelControl_t nominal_controls[PREDICTION_HORIZON] = {};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
  MpcRtiConfiguration_t config{};
  config.weight_e_y = 1.0f;
  config.weight_e_psi = 1.0f;
  config.weight_u = 1.0f;
  config.weight_u_overspeed = 1.0f;
  config.weight_v = 1.0f;
  config.weight_r = 1.0f;
  config.weight_steering_command = 1.0f;
  config.weight_steering_target = 1.0f;
  config.weight_throttle_command = 1.0f;
  config.weight_steering_slew = 20.0f;
  config.weight_throttle_slew = 2.0f;
  config.terminal_multiplier = 1.0f;
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 10.0f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m = 0.05f;
  config.first_prediction_corridor_margin_m = 0.05f;
  config.corridor_preview_halfwidth_m = 0.10f;
  config.nonlinear_corridor_tolerance_m = 0.01f;
  config.jacobian_threads = 4;

  MpcModelState_t plant = turning_state();
  plant.u = 5.0f;
  plant.actual_steering_angle = 0.25f;
  for (int k = 0; k <= horizon; ++k) {
    nominal_states[k].plant = plant;
    references[k].u = plant.u;
    references[k].steering_command =
        plant.delayed_steering_command_2 + 0.01f * k;
    references[k].left_bound = 1.0f;
    references[k].right_bound = 1.0f;
  }
  for (int k = 0; k < horizon; ++k) {
    nominal_controls[k].steering_target_rad = (k % 2 == 0) ? 0.1f : -0.1f;
    nominal_controls[k].throttle_command = k % 2 == 0 ? 0.2f : 0.3f;
  }

  ASSERT_TRUE(vehicle_model_set_active_target_speed_ceiling(
      config.active_speed_ceiling_mps));
  for (int k = 0; k < horizon; ++k) {
    const MpcStageResult_t step = mpc_vehicle_model_step(
        &nominal_states[k].plant, &nominal_controls[k],
        PREDICTION_DT_SECONDS, references[k].path_curvature);
    ASSERT_TRUE(step.valid);
    nominal_states[k + 1].plant = step.next;
    nominal_states[k + 1].incoming_branch_flags = step.branch_flags;
    nominal_states[k + 1].incoming_body_forward_delta_m =
        step.body_forward_delta_m;
    nominal_states[k + 1].incoming_body_lateral_delta_m =
        step.body_lateral_delta_m;
    nominal_states[k + 1].incoming_body_yaw_delta_rad =
        step.body_yaw_delta_rad;
    nominal_states[k + 1].incoming_transition_valid = 1;
  }
  MpcStageLinearization_t serial[PREDICTION_HORIZON] = {};
  for (int k = 0; k < horizon; ++k) {
    ASSERT_TRUE(mpc_model_linearize(&nominal_states[k].plant,
        &nominal_controls[k], PREDICTION_DT_SECONDS,
        references[k].path_curvature, &serial[k]));
  }

  MpcRtiProblem_t problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, nominal_controls,
      references, horizon, PREDICTION_DT_SECONDS, &config, &problem));
  // The QP search domain must allow the full physical command range at
  // every stage, irrespective of the current SQP nominal actuator sequence.
  for (int k = 0; k < RICCATI_HORIZON; ++k) {
    EXPECT_FLOAT_EQ(problem.steps[k].u_lb[0], -config.max_steering_rad);
    EXPECT_FLOAT_EQ(problem.steps[k].u_ub[0], config.max_steering_rad);
    EXPECT_FLOAT_EQ(problem.steps[k].u_lb[1], MPC_THROTTLE_MIN_DRIVE);
    EXPECT_FLOAT_EQ(problem.steps[k].u_ub[1], MPC_THROTTLE_MAX);
  }
  MpcRtiConfiguration_t no_steering_slew_config = config;
  no_steering_slew_config.weight_steering_slew = 0.0f;
  MpcRtiProblem_t no_steering_slew_problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, nominal_controls,
      references, horizon, PREDICTION_DT_SECONDS,
      &no_steering_slew_config, &no_steering_slew_problem));
  const float expected_reference_slew_linear_term =
      2.f * config.weight_steering_slew *
      (references[1].steering_command - references[0].steering_command);
  EXPECT_NEAR(problem.steps[0].q[
          MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] -
          no_steering_slew_problem.steps[0].q[
              MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND],
      expected_reference_slew_linear_term, 1.0e-5f);
  EXPECT_NEAR(problem.steps[0].r[0] -
          no_steering_slew_problem.steps[0].r[0],
      -expected_reference_slew_linear_term, 1.0e-5f);
  // Native commanded-target jumps must not replace the physical rack/path
  // increments used by the successful v1 objective.
  MpcRtiReference_t source_references[PREDICTION_HORIZON + 1];
  std::copy_n(references, PREDICTION_HORIZON + 1, source_references);
  for (int k = 0; k <= horizon; ++k) {
    source_references[k].source_witness_valid = 1;
    source_references[k].source_steering_target = k == 0 ? 0.30f : -0.20f;
    source_references[k].source_throttle = 0.2f;
  }
  MpcRtiProblem_t source_problem{}, source_no_slew{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, nominal_controls,
      source_references, horizon, PREDICTION_DT_SECONDS, &config, &source_problem));
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, nominal_controls,
      source_references, horizon, PREDICTION_DT_SECONDS, &no_steering_slew_config, &source_no_slew));
  for (int k : {0, 1, 2}) {
    const float expected = 2.f * config.weight_steering_slew *
        (source_references[k + 1].steering_command - source_references[k].steering_command);
    EXPECT_NEAR(source_problem.steps[k].r[0] - source_no_slew.steps[k].r[0], -expected, 1.e-5f);
    EXPECT_NEAR(source_problem.steps[k].q[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] -
        source_no_slew.steps[k].q[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND], expected, 1.e-5f);
  }
  auto no_target_config = config;
  no_target_config.weight_steering_target = 0.f;
  MpcRtiProblem_t no_target_problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, nominal_controls,
      source_references, horizon, PREDICTION_DT_SECONDS, &no_target_config, &no_target_problem));
  for (int k : {0, 1}) {
    // The controller tracks curvature-derived steering, regardless of what
    // IPOPT stored as its open-loop command at these source knots.
    EXPECT_NEAR(source_problem.steps[k].r[0]-no_target_problem.steps[k].r[0],
        -2.f*config.weight_steering_target*source_references[k].steering_command,1.e-5f);
    EXPECT_NEAR(source_problem.steps[k].R_diag[0]-no_target_problem.steps[k].R_diag[0],
        2.f*config.weight_steering_target,1.e-5f);
  }
  MpcModelState_t prepared_optimizer_plant =
      nominal_states[MPC_COMMAND_DELAY_STEPS].plant;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&prepared_optimizer_plant));
  EXPECT_EQ(MPC_RTI_NX, 23);
  EXPECT_EQ(RICCATI_HORIZON,
      PREDICTION_HORIZON - MPC_COMMAND_DELAY_STEPS);
  EXPECT_FLOAT_EQ(problem.x0[MPC_OPT_IDX_EY],
      nominal_states[MPC_COMMAND_DELAY_STEPS].plant.e_y);
  EXPECT_FLOAT_EQ(problem.x0[MPC_OPT_IDX_U],
      nominal_states[MPC_COMMAND_DELAY_STEPS].plant.u);
  EXPECT_FLOAT_EQ(problem.x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND],
      nominal_states[0].plant
          .delayed_steering_command_2);
  EXPECT_FLOAT_EQ(problem.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND],
      nominal_states[0].plant
          .delayed_throttle_command_2);
  EXPECT_FLOAT_EQ(problem.x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND],
      nominal_states[MPC_COMMAND_DELAY_STEPS].plant
          .delayed_steering_command_2);
  EXPECT_FLOAT_EQ(problem.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND],
      nominal_states[MPC_COMMAND_DELAY_STEPS].plant
          .delayed_throttle_command_2);
  EXPECT_FLOAT_EQ(problem.steps[0].u_lb[1],
      MPC_THROTTLE_MIN_DRIVE);
  EXPECT_FLOAT_EQ(problem.steps[0].u_ub[1],
      MPC_THROTTLE_MAX);
  EXPECT_FLOAT_EQ(problem.steps[1].u_lb[1],
      MPC_THROTTLE_MIN_DRIVE);
  EXPECT_FLOAT_EQ(problem.steps[1].u_ub[1],
      MPC_THROTTLE_MAX);
  EXPECT_FLOAT_EQ(problem.steps[0].B[
      MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND][0], 1.0f);
  EXPECT_FLOAT_EQ(problem.steps[0].B[
      MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND][1], 1.0f);
  EXPECT_FLOAT_EQ(problem.steps[0].Q_diag[
      MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND], 40.0f);
  EXPECT_FLOAT_EQ(problem.steps[0].N[
      MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND][0], -40.0f);
  EXPECT_FLOAT_EQ(problem.steps[0].Q_diag[
      MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND], 4.0f);
  EXPECT_FLOAT_EQ(problem.steps[0].N[
      MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND][1], -4.0f);
  EXPECT_FLOAT_EQ(problem.x0[MPC_OPT_IDX_ACTUAL_STEERING_ANGLE],
      prepared_optimizer_plant.actual_steering_angle);
  for (int wheel = 0; wheel < 4; ++wheel) {
    EXPECT_FLOAT_EQ(problem.x0[MPC_OPT_IDX_WHEEL_OMEGA_FL + wheel],
        static_cast<float>(prepared_optimizer_plant.source_plant_internal_state[
            MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel]));
    EXPECT_FLOAT_EQ(problem.x0[MPC_OPT_IDX_SUSPENSION_JOUNCE_FL + wheel],
        static_cast<float>(prepared_optimizer_plant.source_plant_internal_state[
            MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL + wheel]));
  }
  const auto model_index = [](int optimizer_index) {
    if (optimizer_index < 5) return optimizer_index;
    if (optimizer_index == MPC_OPT_IDX_ACTUAL_STEERING_ANGLE)
      return static_cast<int>(MPC_STATE_IDX_ACTUAL_STEERING_ANGLE);
    return static_cast<int>(MPC_STATE_IDX_WHEEL_OMEGA_FL +
        optimizer_index - MPC_OPT_IDX_WHEEL_OMEGA_FL);
  };
  for (int k = 0; k < RICCATI_HORIZON; ++k) {
    const int model_stage = k + MPC_COMMAND_DELAY_STEPS;
    for (int row = 0; row < MPC_OPT_PHYSICAL_NX; ++row) {
      const int model_row = model_index(row);
      EXPECT_FLOAT_EQ(problem.steps[k].B[row][0],
          serial[model_stage].B[model_row][0]);
      EXPECT_FLOAT_EQ(problem.steps[k].B[row][1],
          serial[model_stage].B[model_row][1]);
      for (int col = 0; col < MPC_OPT_PHYSICAL_NX; ++col) {
        SCOPED_TRACE(::testing::Message() << "stage=" << k
            << " model_stage=" << model_stage
            << " row=" << row << " col=" << col
            << " model_row=" << model_row
            << " model_col=" << model_index(col));
        EXPECT_FLOAT_EQ(problem.steps[k].A[row][col],
            serial[model_stage].A[model_row][model_index(col)]);
      }
    }
  }

  // Replay every nominal decision through the reduced affine model and
  // compare against the corresponding direct-input plant states.
  float condensed_state[RICCATI_MAX_NX] = {};
  for (int index = 0; index < RICCATI_MAX_NX; ++index) {
    condensed_state[index] = problem.x0[index];
  }
  for (int j = 0; j < RICCATI_HORIZON; ++j) {
    const int model_stage = j + MPC_COMMAND_DELAY_STEPS;
    const MpcModelControl_t applied = nominal_controls[j];

    float nominal_optimizer_state[RICCATI_MAX_NX] = {};
    MpcModelState_t nominal_plant = nominal_states[model_stage].plant;
    ASSERT_TRUE(vehicle_model_initialize_plant_state(&nominal_plant));
    nominal_optimizer_state[0] = nominal_plant.e_y;
    nominal_optimizer_state[1] = nominal_plant.e_psi;
    nominal_optimizer_state[2] = nominal_plant.u;
    nominal_optimizer_state[3] = nominal_plant.v;
    nominal_optimizer_state[4] = nominal_plant.r;
    nominal_optimizer_state[5] = nominal_plant.actual_steering_angle;
    // The 4 accumulated wheel-rotation angles are plant outputs, not RTI
    // coordinates. The physical optimizer vector carries the 14 recursively
    // relevant wheel/body/suspension states after its six path/body values.
    for (int internal = 0; internal < MPC_OPT_PHYSICAL_NX - 6; ++internal) {
      nominal_optimizer_state[6 + internal] = static_cast<float>(
          nominal_plant.source_plant_internal_state[internal]);
    }
    // Accumulate this cancellation-heavy affine row in double precision. The
    // model's new attitude kinematics make the individual float A*x terms
    // larger while their sum remains the small yaw-rate state; float-only
    // accumulation exceeded the tolerance by 1.8e-6 without indicating a
    // plant/Jacobian mismatch.
    double nominal_yaw_rate_reconstruction = problem.steps[j].d[4];
    for (int column = 0; column < MPC_RTI_NX; ++column) {
      nominal_yaw_rate_reconstruction +=
          problem.steps[j].A[4][column] *
          nominal_optimizer_state[column];
    }
    for (int input = 0; input < MPC_MODEL_NU; ++input) {
      nominal_yaw_rate_reconstruction +=
          problem.steps[j].B[4][input] *
          (input == 0 ? applied.steering_target_rad :
                        applied.throttle_command);
    }
    EXPECT_NEAR(nominal_yaw_rate_reconstruction,
        nominal_states[model_stage + 1].plant.r, 1.0e-4)
        << "stage " << model_stage
        << " affine yaw-rate base-point reconstruction";

    float next[MPC_RTI_NX] = {};
    for (int row = 0; row < MPC_RTI_NX; ++row) {
      next[row] = problem.steps[j].d[row];
      for (int column = 0; column < MPC_RTI_NX; ++column) {
        next[row] += problem.steps[j].A[row][column] *
            condensed_state[column];
      }
      for (int input = 0; input < MPC_MODEL_NU; ++input) {
        next[row] += problem.steps[j].B[row][input] *
            (input == 0 ? applied.steering_target_rad :
                          applied.throttle_command);
      }
    }
    std::copy_n(next, MPC_RTI_NX, condensed_state);

    const MpcModelState_t & expected =
        nominal_states[model_stage + 1].plant;
    float expected_physical[MPC_OPT_PHYSICAL_NX] = {
        expected.e_y, expected.e_psi, expected.u, expected.v, expected.r,
        expected.actual_steering_angle};
    for (int index = 0; index < MPC_OPT_PHYSICAL_NX; ++index) {
      if (index >= 6) {
        expected_physical[index] = static_cast<float>(
            expected.source_plant_internal_state[index - 6]);
      }
      EXPECT_NEAR(condensed_state[index], expected_physical[index], 4.0e-4f)
          << "stage " << model_stage << " optimizer state " << index;
    }
  }
  EXPECT_GT(problem.terminal_Q[MPC_OPT_IDX_U], 0.0f);
  EXPECT_EQ(problem.horizon, RICCATI_HORIZON);

  // The last issued command is the delayed-command channel in the terminal
  // state. Its feedforward cost has the SAME terminal multiplier as the exact
  // nonlinear objective, even though that channel was eliminated from QP x.
  MpcRtiConfiguration_t terminal_config = config;
  terminal_config.terminal_multiplier = 3.0f;
  MpcRtiProblem_t terminal_problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, nominal_controls,
      references, horizon, PREDICTION_DT_SECONDS, &terminal_config,
      &terminal_problem));
  EXPECT_FLOAT_EQ(terminal_problem.steps[0].R_diag[0],
      problem.steps[0].R_diag[0]);
  EXPECT_NEAR(terminal_problem.steps[RICCATI_HORIZON-1].R_diag[0] -
      problem.steps[RICCATI_HORIZON-1].R_diag[0],
      4.0f*config.weight_steering_command, 1.e-5f);
  EXPECT_NEAR(terminal_problem.steps[RICCATI_HORIZON-1].r[0] -
      problem.steps[RICCATI_HORIZON-1].r[0],
      -4.0f*config.weight_steering_command*references[horizon].steering_command,
      1.e-5f);

}

TEST(MpcReference, NominalHorizonKeepsTrackSpeedTarget)
{
  MpcTrajectorySample_t trajectory[3] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 10.0, 10.0, 3.0},
      {10.0, 10.0, 0.0, 0.0, 0.0, 4.0, 10.0, 10.0, 3.0},
      {20.0, 10.0, 10.0, 1.5707963267948966, 0.0, 4.0, 10.0, 10.0, 3.0}};
  size_t trajectory_count = 3;
  double lap_length = 0.0;
  ASSERT_TRUE(mpc_trajectory_prepare(
      trajectory, &trajectory_count, &lap_length));

  MpcRtiConfiguration_t config{};
  config.weight_e_y = 1.0f;
  config.weight_e_psi = 1.0f;
  config.weight_u = 1.0f;
  config.weight_u_overspeed = 1.0f;
  config.weight_v = 1.0f;
  config.weight_r = 1.0f;
  config.weight_steering_command = 1.0f;
  config.weight_steering_target = 1.0f;
  config.weight_throttle_command = 1.0f;
  config.terminal_multiplier = 1.0f;
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 1.5f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m = 0.05f;
  config.first_prediction_corridor_margin_m = 0.05f;
  config.corridor_preview_halfwidth_m = 0.10f;
  config.nonlinear_corridor_tolerance_m = 0.01f;

  MpcRtiState_t initial{};
  initial.plant.u = 0.5f;
  initial.plant.delayed_throttle_command_1 = 0.2f;
  initial.plant.delayed_throttle_command_2 = 0.2f;
  MpcRtiNominal_t nominal{};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0.0, nullptr,
      trajectory, trajectory_count, lap_length, PREDICTION_DT_SECONDS,
      PREDICTION_HORIZON, &config, &nominal, references))
      << "nominal build stage=" << nominal.build_failure_stage
      << " constraint=" << nominal.build_failure_constraint
      << " index=" << nominal.build_failure_index
      << " u=" << nominal.states[nominal.build_failure_index].plant.u
      << " u17=" << nominal.states[17].plant.u
      << " u18=" << nominal.states[18].plant.u
      << " u19=" << nominal.states[19].plant.u
      << " q16=" << nominal.controls[16].throttle_command
      << " q17=" << nominal.controls[17].throttle_command
      << " q18=" << nominal.controls[18].throttle_command;

  EXPECT_TRUE(nominal.speed_reference_valid);
  EXPECT_FLOAT_EQ(references[0].u, initial.plant.u);
  EXPECT_FLOAT_EQ(references[1].u, 1.5f);
  EXPECT_FLOAT_EQ(references[PREDICTION_HORIZON].u, 1.5f);
  for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
    EXPECT_LE(references[k].u, 1.5f);
    EXPECT_FLOAT_EQ(nominal.speed_reference[k], references[k].u);
    EXPECT_NEAR(references[k].r,
        references[k].path_curvature * references[k].u, 1.0e-6f);
  }
  config.active_speed_ceiling_mps = 5.0f;
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0.0, nullptr,
      trajectory, trajectory_count, lap_length, PREDICTION_DT_SECONDS,
      PREDICTION_HORIZON, &config, &nominal, references));
  EXPECT_FLOAT_EQ(references[0].u, initial.plant.u);
  for (int k = 0; k <= PREDICTION_HORIZON; ++k)
    EXPECT_LE(references[k].u, 5.0f);
  EXPECT_NEAR(references[5].u, 4.0f, 1.0e-4f)
      << "The target must remain the trajectory request; the plant rollout, "
         "not a pre-clipped target, determines whether it is reachable.";
}

TEST(MpcReference, InfeasibleWarmStartCanReachQpButCannotBePublished)
{
  // A finite nominal is a linearization point, not an accepted trajectory.
  // Start from the final measured state of the Oct 28 slip-cost trial. A
  // sustained tight-turn seed eventually predicts reverse forward velocity;
  // the actual vehicle is still moving forwards when optimization begins.
  MpcTrajectorySample_t trajectory[3] = {
      {0., 0., 0., 0., 0., 12., 100., 100., 0.},
      {100., 100., 0., 0., 0., 12., 100., 100., 0.},
      {200., 200., 0., 0., 0., 12., 100., 100., 0.}};
  size_t count = 3;
  double length = 0.;
  ASSERT_TRUE(mpc_trajectory_prepare(trajectory, &count, &length));
  MpcRtiConfiguration_t config{};
  config.weight_e_y = config.weight_e_psi = config.weight_u = 1.f;
  config.weight_u_overspeed = config.weight_v = config.weight_r = 1.f;
  config.weight_steering_command = config.weight_steering_target = 1.f;
  config.weight_throttle_command = config.terminal_multiplier = 1.f;
  config.max_speed_mps = 16.f;
  config.active_speed_ceiling_mps = 16.f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m = config.first_prediction_corridor_margin_m = .05f;
  MpcRtiState_t initial{};
  initial.plant.u = 5.24309015f;
  initial.plant.v = .509816468f;
  initial.plant.r = -2.16397238f;
  initial.plant.actual_steering_angle = -.253153682f;
  initial.plant.delayed_steering_command_1 = -.338617772f;
  initial.plant.delayed_steering_command_2 = -.338617772f;
  initial.plant.delayed_throttle_command_1 = .267959505f;
  initial.plant.delayed_throttle_command_2 = .268189043f;
  initial.plant.source_plant_initialized = 1;
  initial.plant.source_suspension_history_initialized = 1;
  const double hidden[] = {113.274164, 114.357796, 112.407531, 113.8122,
      .00409496345, -.000176888811, -.079873332, .0974897421,
      .0165529907, .11159703, .00709799916, -.0114578217,
      .0116765382, -.00722836564, 0., 0., 0., 0., 0., 0., 0., 0.};
  std::copy_n(hidden, 22, initial.plant.source_plant_internal_state);
  MpcModelControl_t controls[PREDICTION_HORIZON]{};
  for (auto &control : controls) {
    control.throttle_command = .268189043f;
    control.steering_target_rad = -.5f;
  }
  MpcRtiNominal_t nominal{};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1]{};
  ASSERT_TRUE(mpc_rti_build_nominal_with_warm_start(&initial, 0., nullptr,
      controls, PREDICTION_HORIZON, trajectory, count, length,
      PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &config, &nominal, references))
      << "stage=" << nominal.build_failure_stage << " index="
      << nominal.build_failure_index;
  float minimum_u = initial.plant.u;
  for (const auto &state : nominal.states)
    minimum_u = std::min(minimum_u, state.plant.u);
  ASSERT_LT(minimum_u, -MPC_PHYSX_BRAKE_REVERSE_TRANSIENT_MPS);
  MpcRtiProblem_t problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal.states, nominal.controls,
      references, PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &config, &problem));
  for (int k = 0; k < problem.horizon; ++k) {
    EXPECT_GE(problem.steps[k].x_lb[MPC_OPT_IDX_U],
              -MPC_PHYSX_BRAKE_REVERSE_TRANSIENT_MPS);
    EXPECT_LE(problem.steps[k].x_ub[MPC_OPT_IDX_U], config.max_speed_mps);
  }
  MpcRtiState_t states[PREDICTION_HORIZON + 1]{};
  EXPECT_EQ(mpc_rti_rollout_candidate(&initial, 0., controls, trajectory,
      count, length, nullptr, nullptr, nullptr, PREDICTION_HORIZON,
      PREDICTION_DT_SECONDS, &config, states, nullptr, nullptr),
      MPC_RTI_ROLLOUT_STATE_LIMIT);

  // Measured violations remain fatal; they must not become repairable seeds.
  for (float invalid_speed : {-1.f, 17.f}) {
    initial.plant.u = invalid_speed;
    EXPECT_FALSE(mpc_rti_build_nominal(&initial, 0., nullptr, trajectory,
        count, length, PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &config,
        &nominal, references));
    EXPECT_EQ(nominal.build_failure_index, 0);
    EXPECT_EQ(nominal.build_failure_stage, MPC_RTI_NOMINAL_BUILD_COMMAND_ENVELOPE);
  }
}

TEST(MpcReference, ColdStartThrottleSeedMatchesExactQueueShift)
{
  MpcTrajectorySample_t trajectory[3] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 10.0, 10.0, 3.0},
      {10.0, 10.0, 0.0, 0.0, 0.0, 4.0, 10.0, 10.0, 3.0},
      {20.0, 20.0, 0.0, 0.0, 0.0, 4.0, 10.0, 10.0, 3.0}};
  size_t trajectory_count = 3;
  double lap_length = 0.0;
  ASSERT_TRUE(mpc_trajectory_prepare(
      trajectory, &trajectory_count, &lap_length));

  MpcRtiConfiguration_t config{};
  config.weight_e_y = config.weight_e_psi = config.weight_u = 1.0f;
  config.weight_u_overspeed = config.weight_v = config.weight_r = 1.0f;
  config.weight_steering_command = config.weight_steering_target = 1.0f;
  config.weight_throttle_command = config.terminal_multiplier = 1.0f;
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 1.5f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m = 0.05f;
  config.first_prediction_corridor_margin_m = 0.05f;
  config.corridor_preview_halfwidth_m = 0.10f;
  config.nonlinear_corridor_tolerance_m = 0.01f;

  const float queued_throttles[] = {0.0f, 0.2f, 0.8f};
  for (const float queued_throttle : queued_throttles) {
    MpcRtiState_t initial{};
    initial.plant.u = 0.7f;  // Do not activate the separate launch seed.
    initial.plant.delayed_throttle_command_1 = 0.65f;
    initial.plant.delayed_throttle_command_2 = queued_throttle;

    MpcRtiNominal_t nominal{};
    MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
    ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0.0, nullptr,
        trajectory, trajectory_count, lap_length, PREDICTION_DT_SECONDS,
        PREDICTION_HORIZON, &config, &nominal, references));

    // Recreate the prior seed algorithm with the source plant. The selected
    // throttle depends only on the exact queued-command shift, not body state.
    MpcModelState_t plant = initial.plant;
    for (int k = 0; k < PREDICTION_HORIZON; ++k) {
      const float expected_throttle = std::clamp(
          std::max(plant.delayed_throttle_command_2,
                   MPC_THROTTLE_MIN_DRIVE),
          MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
      EXPECT_FLOAT_EQ(nominal.controls[k].throttle_command,
                      expected_throttle) << "stage=" << k;
      const MpcModelControl_t control{
          references[k + 1].steering_command, expected_throttle};
      const MpcStageResult_t step = mpc_vehicle_model_step(
          &plant, &control, PREDICTION_DT_SECONDS,
          references[k].path_curvature);
      ASSERT_TRUE(step.valid) << "reference queue stage=" << k;
      plant = step.next;
    }
  }
}

TEST(MpcReference, LaunchesFromRestWhenSpeedReferenceIsPositive)
{
  MpcTrajectorySample_t trajectory[3] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.0, 1.0, 0.0},
      {10.0, 10.0, 0.0, 0.0, 0.0, 1.5, 1.0, 1.0, 0.0},
      {20.0, 10.0, 10.0, 0.0, 0.0, 1.5, 1.0, 1.0, 0.0}};
  size_t trajectory_count = 3;
  double lap_length = 0.0;
  ASSERT_TRUE(mpc_trajectory_prepare(
      trajectory, &trajectory_count, &lap_length));

  MpcRtiConfiguration_t config{};
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 1.5f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m = 0.05f;
  config.first_prediction_corridor_margin_m = 0.05f;
  config.corridor_preview_halfwidth_m = 0.10f;
  config.nonlinear_corridor_tolerance_m = 0.01f;
  config.startup_nominal_throttle_seed = 0.25f;

  MpcRtiState_t initial{};
  // Live odometry reports a tiny positive residual at standstill (about
  // 4e-11 m/s). It must not make the zero-speed launch envelope look like an
  // overspeed/braking state.
  initial.plant.u = 4.0e-11f;
  MpcRtiNominal_t nominal{};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0.0, nullptr,
      trajectory, trajectory_count, lap_length, PREDICTION_DT_SECONDS,
      PREDICTION_HORIZON, &config, &nominal, references));

  ASSERT_TRUE(nominal.speed_reference_valid);
  EXPECT_GT(nominal.speed_reference[PREDICTION_HORIZON], 0.0f);
  EXPECT_GE(nominal.controls[0].throttle_command, MPC_THROTTLE_MIN_DRIVE)
      << "The first command must select drive so the delayed plant can leave rest.";

  MpcRtiNominal_t stale_nominal{};
  stale_nominal.valid = 1;
  stale_nominal.horizon = PREDICTION_HORIZON;
  for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
    stale_nominal.states[k] = initial;
    stale_nominal.progress[k] = 0.0;
  }
  for (int k = 0; k < PREDICTION_HORIZON; ++k)
    stale_nominal.controls[k].throttle_command = MPC_THROTTLE_MIN_DRIVE;
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0.0, &stale_nominal,
      trajectory, trajectory_count, lap_length, PREDICTION_DT_SECONDS,
      PREDICTION_HORIZON, &config, &nominal, references));
  EXPECT_NEAR(nominal.controls[0].throttle_command,
      config.startup_nominal_throttle_seed, 1.0e-6f)
      << "A warm shifted nominal must not preserve minimum drive throttle "
         "when a stopped car is still well below its requested speed.";

  // A previously accepted nominal can be passed directly as the destination.
  // This must preserve the same warm seed without copying every nominal on
  // the ordinary disjoint-memory controller path.
  const MpcRtiNominal_t old_nominal = nominal;
  MpcRtiNominal_t disjoint{}, aliased = old_nominal;
  MpcRtiReference_t disjoint_references[PREDICTION_HORIZON + 1]{};
  MpcRtiReference_t aliased_references[PREDICTION_HORIZON + 1]{};
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0., &old_nominal,
      trajectory, trajectory_count, lap_length, PREDICTION_DT_SECONDS,
      PREDICTION_HORIZON, &config, &disjoint, disjoint_references));
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0., &aliased,
      trajectory, trajectory_count, lap_length, PREDICTION_DT_SECONDS,
      PREDICTION_HORIZON, &config, &aliased, aliased_references));
  for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
    EXPECT_DOUBLE_EQ(disjoint.progress[k], aliased.progress[k]);
    EXPECT_FLOAT_EQ(disjoint.states[k].plant.u, aliased.states[k].plant.u);
    if (k < PREDICTION_HORIZON) {
      EXPECT_FLOAT_EQ(disjoint.controls[k].steering_target_rad,
                      aliased.controls[k].steering_target_rad);
      EXPECT_FLOAT_EQ(disjoint.controls[k].throttle_command,
                      aliased.controls[k].throttle_command);
    }
  }
}

TEST(MpcQpSourceSchedule, FrozenCostsDoNotFreezePhysicalCorridorGeometry)
{
  MpcTrajectorySample_t trajectory[]={
      {0.,0.,0.,0.,0.,2.5,4.,4.,0.},
      {10.,10.,0.,0.,0.,2.5,4.,4.,0.},
      {20.,20.,0.,0.,0.,2.5,4.,4.,0.}};
  size_t count=3;double length=0.;
  ASSERT_TRUE(mpc_trajectory_prepare(trajectory,&count,&length));
  MpcRtiConfiguration_t model{};
  model.weight_e_y=4500.f;
  model.weight_e_psi=1450.f;
  model.weight_u=45.f;
  model.weight_u_overspeed=20.f;
  model.weight_v=5.f;
  model.weight_r=7.8125f;
  model.weight_steering_command=4.5f;
  model.weight_steering_target=5.f;
  model.weight_steering_slew=1500.f;
  model.weight_throttle_slew=50.f;
  model.terminal_multiplier=3.f;
  model.max_speed_mps=model.active_speed_ceiling_mps=16.f;
  model.max_steering_rad=SOURCE_MAX_STEERING_RAD;
  model.command_increment_qp_enabled=1;
  model.max_steering_command_step_rad=2.f*SOURCE_MAX_STEERING_RAD;
  model.max_throttle_command_step=1.f;
  model.corridor_margin_m=model.first_prediction_corridor_margin_m=.05f;
  model.corridor_preview_halfwidth_m=.1f;
  MpcRtiState_t initial{};
  initial.plant.u=2.f;
  initial.plant.delayed_steering_command_2=.02f;
  initial.plant.delayed_throttle_command_2=.3f;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&initial.plant));
  MpcModelControl_t controls[PREDICTION_HORIZON]{};
  for (auto &u:controls) {
    u.steering_target_rad=.02f;
    u.throttle_command=.3f;
  }
  auto nominal=std::make_unique<MpcRtiNominal_t>();
  MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
  ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
      &initial,0.,nullptr,controls,PREDICTION_HORIZON,
      trajectory,count,length,PREDICTION_DT_SECONDS,
      PREDICTION_HORIZON,&model,nominal.get(),refs));
  auto original=std::make_unique<MpcRtiProblem_t>();
  auto modified=std::make_unique<MpcRtiProblem_t>();
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal->states,nominal->controls,
      refs,PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&model,original.get()));

  MpcRtiObjectiveSchedule_t frozen{};
  ASSERT_TRUE(mpc_rti_build_objective_schedule(nominal.get(),refs,&model,&frozen));
  ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
      nominal->states,nominal->controls,refs,PREDICTION_HORIZON,
      PREDICTION_DT_SECONDS,&model,&frozen,modified.get()));
  // First RTI pass with its own exact-source nominal MUST be numerically
  // identical to the original QP, including all 30 stage cross terms.
  for(int k=0;k<RICCATI_HORIZON;++k) {
    const auto &a=original->steps[k], &b=modified->steps[k];
    for(int i=0;i<RICCATI_MAX_NX;++i) {
      EXPECT_FLOAT_EQ(a.Q_diag[i],b.Q_diag[i])<<k<<" "<<i;
      EXPECT_FLOAT_EQ(a.q[i],b.q[i])<<k<<" "<<i;
      for(int j=0;j<RICCATI_MAX_NX;++j) {
        EXPECT_FLOAT_EQ(a.A[i][j],b.A[i][j])<<k<<" "<<i<<" "<<j;
        EXPECT_FLOAT_EQ(a.Q_offdiag[i][j],b.Q_offdiag[i][j])
            <<k<<" "<<i<<" "<<j;
      }
      for(int j=0;j<RICCATI_MAX_NU;++j)
        EXPECT_FLOAT_EQ(a.N[i][j],b.N[i][j])<<k<<" "<<i<<" "<<j;
    }
    for(int j=0;j<RICCATI_MAX_NU;++j) {
      EXPECT_FLOAT_EQ(a.r[j],b.r[j])<<k<<" "<<j;
      EXPECT_FLOAT_EQ(a.R_diag[j],b.R_diag[j])<<k<<" "<<j;
    }
  }
  // An R2 iterate must change the *physical* nominal, not the frozen merit.
  // Exercise the true two-pass contract: independent exact source rollout,
  // independent source A/B and footprint geometry, identical cost targets.
  MpcModelControl_t changed_controls[PREDICTION_HORIZON]{};
  for(int k=0;k<PREDICTION_HORIZON;++k) {
    changed_controls[k]=controls[k];
    changed_controls[k].steering_target_rad+=.012f*std::sin(.18f*k);
    changed_controls[k].throttle_command+=.025f*std::cos(.2f*k);
  }
  auto changed_nominal=std::make_unique<MpcRtiNominal_t>();
  MpcRtiReference_t changed_refs[PREDICTION_HORIZON+1]{};
  ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
      &initial,0.,nullptr,changed_controls,PREDICTION_HORIZON,
      trajectory,count,length,PREDICTION_DT_SECONDS,
      PREDICTION_HORIZON,&model,changed_nominal.get(),changed_refs));
  auto changed_problem=std::make_unique<MpcRtiProblem_t>();
  ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
      changed_nominal->states,changed_nominal->controls,changed_refs,
      PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&model,&frozen,
      changed_problem.get()));
  // With the objective schedule held fixed, target/weight QP derivatives
  // use the ORIGINAL trajectory even though the source dynamics are fresh.
  EXPECT_NEAR(changed_problem->steps[1].Q_diag[MPC_OPT_IDX_U],
      modified->steps[1].Q_diag[MPC_OPT_IDX_U],1.e-2);
  EXPECT_NEAR(changed_problem->steps[1].q[MPC_OPT_IDX_U],
      modified->steps[1].q[MPC_OPT_IDX_U],1.e-2);
  EXPECT_GT(std::abs(changed_nominal->states[20].plant.actual_steering_angle-
                     nominal->states[20].plant.actual_steering_angle),.001f);
  EXPECT_GT(std::abs(changed_problem->steps[20].A[MPC_OPT_IDX_R][MPC_OPT_IDX_U]-
                     modified->steps[20].A[MPC_OPT_IDX_R][MPC_OPT_IDX_U]),1.e-4);
  // The exact scorer would use these frozen quantities if a later SQP
  // nominal crossed a speed branch or shifted to another path location.
  for(int k=0;k<=PREDICTION_HORIZON;++k) {
    frozen.speed_weights[k]=90.f;
    frozen.references[k].left_bound=.2f;
    frozen.references[k].right_bound=.3f;
  }
  ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
      nominal->states,nominal->controls,refs,PREDICTION_HORIZON,
      PREDICTION_DT_SECONDS,&model,&frozen,modified.get()));
  // The accelerated source QP and the exact merit now share the same
  // frozen speed-weight branch, including the terminal multiplier.
  const double original_speed_weight=45.;
  const double delta=90.-original_speed_weight;
  EXPECT_NEAR(
      modified->steps[1].Q_diag[MPC_OPT_IDX_U] -
      original->steps[1].Q_diag[MPC_OPT_IDX_U],2.*delta,2.e-3);
  EXPECT_NEAR(
      modified->steps[1].q[MPC_OPT_IDX_U] -
      original->steps[1].q[MPC_OPT_IDX_U],-2.*delta*refs[1].u,2.e-3);
  const double old_terminal_weight =
      nominal->states[PREDICTION_HORIZON].plant.u > refs[PREDICTION_HORIZON].u
          ? model.weight_u_overspeed : model.weight_u;
  EXPECT_NEAR(
      modified->terminal_Q[MPC_OPT_IDX_U] -
      original->terminal_Q[MPC_OPT_IDX_U],
      2.*(90.-old_terminal_weight)*model.terminal_multiplier,2.e-3);
  // Yet corridor bounds and geometric sensitivities MUST remain those of
  // the current exact plant nominal, not stale objective reference metadata.
  EXPECT_FLOAT_EQ(modified->steps[10].x_lb[MPC_OPT_IDX_EY],
                  original->steps[10].x_lb[MPC_OPT_IDX_EY]);
  EXPECT_FLOAT_EQ(modified->steps[10].x_ub[MPC_OPT_IDX_EY],
                  original->steps[10].x_ub[MPC_OPT_IDX_EY]);
  EXPECT_FLOAT_EQ(modified->steps[10].A[MPC_OPT_IDX_EY][MPC_OPT_IDX_U],
                  original->steps[10].A[MPC_OPT_IDX_EY][MPC_OPT_IDX_U]);
  frozen.speed_weights[17]=-1.f;
  EXPECT_FALSE(mpc_rti_build_ltv_qp_with_objective_schedule(
      nominal->states,nominal->controls,refs,PREDICTION_HORIZON,
      PREDICTION_DT_SECONDS,&model,&frozen,modified.get()));
}

TEST(MpcSourceAdjoint, PhysicalSteeringAndThrottleGradientAgreesWithExactPlant)
{
  MpcTrajectorySample_t trajectory[]={
      {0.,0.,0.,0.,0.,2.5,4.,4.,0.},
      {10.,10.,0.,0.,0.,2.5,4.,4.,0.},
      {20.,20.,0.,0.,0.,2.5,4.,4.,0.}};
  size_t count=3;double length=0.;
  ASSERT_TRUE(mpc_trajectory_prepare(trajectory,&count,&length));
  MpcRtiConfiguration_t model{};
  model.weight_e_y=4500.f;
  model.weight_e_psi=1450.f;
  model.weight_u=45.f;
  model.weight_u_overspeed=20.f;
  model.weight_v=5.f;
  model.weight_r=7.8125f;
  model.weight_longitudinal_slip=5.f;
  model.weight_steering_command=4.5f;
  model.weight_steering_target=5.f;
  model.weight_throttle_command=.1f;
  model.weight_steering_slew=1500.f;
  model.weight_throttle_slew=50.f;
  model.terminal_multiplier=3.f;
  model.max_speed_mps=model.active_speed_ceiling_mps=16.f;
  model.max_steering_rad=SOURCE_MAX_STEERING_RAD;
  model.command_increment_qp_enabled=1;
  model.max_steering_command_step_rad=2.f*SOURCE_MAX_STEERING_RAD;
  model.max_throttle_command_step=1.f;
  model.corridor_margin_m=model.first_prediction_corridor_margin_m=.05f;
  model.corridor_preview_halfwidth_m=.10f;
  model.nonlinear_corridor_tolerance_m=.005f;

  MpcRtiState_t initial{};
  initial.plant.u=2.f;
  initial.plant.e_y=.01f;
  initial.plant.e_psi=.005f;
  initial.plant.actual_steering_angle=.02f;
  initial.plant.delayed_steering_command_1=.02f;
  initial.plant.delayed_steering_command_2=.02f;
  initial.plant.delayed_throttle_command_1=.3f;
  initial.plant.delayed_throttle_command_2=.3f;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&initial.plant));
  MpcModelControl_t controls[PREDICTION_HORIZON]{};
  for(int k=0;k<PREDICTION_HORIZON;++k) {
    controls[k].steering_target_rad=.022f+.004f*std::sin(.12*k);
    controls[k].throttle_command=.31f+.012f*std::cos(.15*k);
  }
  MpcRtiNominal_t nominal{};
  MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
  ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
      &initial,0.,nullptr,controls,PREDICTION_HORIZON,
      trajectory,count,length,PREDICTION_DT_SECONDS,
      PREDICTION_HORIZON,&model,&nominal,refs));
  MpcRtiObjectiveSchedule_t frozen{};
  ASSERT_TRUE(mpc_rti_build_objective_schedule(
      &nominal,refs,&model,&frozen));
  double adjoint[PREDICTION_HORIZON][MPC_RTI_NU]{};
  ASSERT_TRUE(mpc_rti_source_adjoint_gradient(
      &initial,0.,trajectory,count,length,PREDICTION_DT_SECONDS,
      &model,controls,&frozen,adjoint));

  MpcRtiCycleConfiguration_t cycle{};
  cycle.model=model;
  cycle.objective_schedule=&frozen;
  MpcRtiNominal_t exact{};
  for(const int k:{0,3,11,26,29}) {
    for(int a=0;a<2;++a) {
      MpcModelControl_t perturbed[PREDICTION_HORIZON]{};
      std::copy_n(controls,PREDICTION_HORIZON,perturbed);
      float &u=a==0?perturbed[k].steering_target_rad:
          perturbed[k].throttle_command;
      const float original=u;
      const float h=a==0?.0001f:.0003f;
      u=original+h;
      int failure=-1;
      ASSERT_EQ(mpc_rti_validate_candidate_plan(
          &initial,0.,perturbed,trajectory,count,length,
          PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&cycle,&exact,
          &failure),MPC_RTI_ROLLOUT_OK)<<k<<" "<<a<<" "<<failure;
      const double high=exact.validation_objective;
      u=original-h;
      ASSERT_EQ(mpc_rti_validate_candidate_plan(
          &initial,0.,perturbed,trajectory,count,length,
          PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&cycle,&exact,
          &failure),MPC_RTI_ROLLOUT_OK)<<k<<" "<<a<<" "<<failure;
      const double derivative=(high-exact.validation_objective)/(2.*h);
      EXPECT_NEAR(adjoint[k][a],derivative,
          std::max(1.0,.08*std::abs(derivative)))
          <<"stage="<<k<<" input="<<a
          <<" adjoint="<<adjoint[k][a]
          <<" exact_fd="<<derivative;
    }
  }
  // Stronger regression: emulate a later SQP iterate whose live speed and
  // progress differ from the objective originally frozen at cycle start.
  // Both reference targets AND overspeed/underspeed branches must remain
  // those used by the independent exact source-merit evaluator.
  for(int k=0;k<=PREDICTION_HORIZON;++k) {
    frozen.speed_weights[k] = (k % 2) ? 95.f : 72.f;
    frozen.references[k].u += .31f;
    frozen.references[k].steering_command += .012f;
  }
  ASSERT_TRUE(mpc_rti_source_adjoint_gradient(
      &initial,0.,trajectory,count,length,PREDICTION_DT_SECONDS,
      &model,controls,&frozen,adjoint));
  for(const int k:{0,9,21,29}) {
    for(int a=0;a<2;++a) {
      MpcModelControl_t perturbed[PREDICTION_HORIZON]{};
      std::copy_n(controls,PREDICTION_HORIZON,perturbed);
      float &input=a==0?perturbed[k].steering_target_rad:
          perturbed[k].throttle_command;
      const float original=input;
      const float h=.0001f;
      int failure=-1;
      input=original+h;
      ASSERT_EQ(mpc_rti_validate_candidate_plan(
          &initial,0.,perturbed,trajectory,count,length,
          PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&cycle,&exact,
          &failure),MPC_RTI_ROLLOUT_OK);
      const double high=exact.validation_objective;
      input=original-h;
      ASSERT_EQ(mpc_rti_validate_candidate_plan(
          &initial,0.,perturbed,trajectory,count,length,
          PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&cycle,&exact,
          &failure),MPC_RTI_ROLLOUT_OK);
      const double fd=(high-exact.validation_objective)/(2.*h);
      // Exact-source tire branches can change inside +/-3e-4 throttle.
      // Test the infinitesimal derivative, not a secant across branches:
      // at stage 21 a +/-1e-4 FD matches the adjoint within 0.02%, while
      // +/-3e-4 changes the finite-difference slope by about 19%.
      EXPECT_NEAR(adjoint[k][a],fd,std::max(1.0,.08*std::abs(fd)))
          << "frozen SQP branch stage=" << k << " input=" << a
          << " adjoint=" << adjoint[k][a] << " source_fd=" << fd;
    }
  }
}

TEST(MpcSourceNonlinear, ExactSourceCandidatesPreserveOriginalHardCorridor)
{
  MpcTrajectorySample_t trajectory[]={
      {0.,0.,0.,0.,0.,2.5,4.,4.,0.},
      {10.,10.,0.,0.,0.,2.5,4.,4.,0.},
      {20.,20.,0.,0.,0.,2.5,4.,4.,0.}};
  size_t count=3;double length=0.;
  ASSERT_TRUE(mpc_trajectory_prepare(trajectory,&count,&length));
  MpcRtiConfiguration_t model{};
  model.weight_e_y=4500.f;
  model.weight_e_psi=1450.f;
  model.weight_u=45.f;
  model.weight_u_overspeed=20.f;
  model.weight_v=5.f;
  model.weight_r=7.8125f;
  model.weight_steering_command=4.5f;
  model.weight_steering_target=5.f;
  model.weight_steering_slew=1500.f;
  model.weight_throttle_slew=50.f;
  model.terminal_multiplier=3.f;
  model.max_speed_mps=model.active_speed_ceiling_mps=16.f;
  model.max_steering_rad=SOURCE_MAX_STEERING_RAD;
  model.command_increment_qp_enabled=1;
  model.max_steering_command_step_rad=2.f*SOURCE_MAX_STEERING_RAD;
  model.max_throttle_command_step=1.f;
  model.corridor_margin_m=model.first_prediction_corridor_margin_m=.05f;
  model.corridor_preview_halfwidth_m=.1f;
  MpcRtiState_t initial{};
  initial.plant.u=2.f;
  initial.plant.e_y=.02f;
  initial.plant.e_psi=.012f;
  initial.plant.delayed_steering_command_1=.02f;
  initial.plant.delayed_steering_command_2=.02f;
  initial.plant.delayed_throttle_command_1=.3f;
  initial.plant.delayed_throttle_command_2=.3f;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&initial.plant));
  MpcModelControl_t controls[PREDICTION_HORIZON]{};
  for(int k=0;k<PREDICTION_HORIZON;++k) {
    controls[k].steering_target_rad=.02f+.002f*std::sin(.12*k);
    controls[k].throttle_command=.3f+.01f*std::cos(.15*k);
  }
  auto nominal=std::make_unique<MpcRtiNominal_t>();
  MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
  ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
      &initial,0.,nullptr,controls,PREDICTION_HORIZON,
      trajectory,count,length,PREDICTION_DT_SECONDS,
      PREDICTION_HORIZON,&model,nominal.get(),refs));
  MpcRtiObjectiveSchedule_t frozen{};
  ASSERT_TRUE(mpc_rti_build_objective_schedule(
      nominal.get(),refs,&model,&frozen));
  MpcRtiCycleConfiguration_t cycle{};
  cycle.model=model;
  cycle.objective_schedule=&frozen;

  MpcRtiSourceNonlinearOptions_t options{};
  options.max_iterations=2;
  options.max_exact_trials_per_iteration=5;
  options.initial_step_scale=.1f;
  options.minimum_step_scale=.005f;
  options.armijo_fraction=.0001f;
  options.steering_direction_norm_rad=.15f;
  options.throttle_direction_norm=.25f;
  options.directional_check_step_scale=.0003f;
  auto result=std::make_unique<MpcRtiSourceNonlinearResult_t>();
  auto cached=std::make_unique<MpcRtiNominal_t>();
  ASSERT_TRUE(mpc_rti_source_nonlinear_optimize(
      &initial,0.,trajectory,count,length,PREDICTION_DT_SECONDS,
      &cycle,&frozen,controls,&options,result.get(),cached.get()));
  EXPECT_TRUE(cached->valid);
  EXPECT_EQ(cached->horizon,PREDICTION_HORIZON);
  ASSERT_TRUE(result->initial_valid);
  ASSERT_TRUE(result->final_valid);
  EXPECT_GE(result->initial_objective,result->final_objective);
  EXPECT_GE(result->exact_rollout_evaluations,2);
  EXPECT_EQ(result->source_adjoint_evaluations,result->iterations_attempted);
  EXPECT_EQ(result->source_derivative_discrepancies,0);
  EXPECT_GE(result->minimum_corridor_clearance_m,0.);
  EXPECT_EQ(result->budget_exhausted,0);
  for(int i=0;i<result->iterations_accepted;++i)
    EXPECT_LT(result->accepted_exact_objectives[i+1],
              result->accepted_exact_objectives[i]);
  auto independent=std::make_unique<MpcRtiNominal_t>();
  int fail=-1;
  EXPECT_EQ(mpc_rti_validate_candidate_plan(
      &initial,0.,result->controls,trajectory,count,length,
      PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&cycle,
      independent.get(),&fail),MPC_RTI_ROLLOUT_OK)<<fail;
  EXPECT_NEAR(independent->validation_objective,
              result->final_objective,1.e-5);
  for(int k=0;k<=PREDICTION_HORIZON;++k) {
      EXPECT_NEAR(cached->progress[k],independent->progress[k],1.e-7)<<k;
      EXPECT_NEAR(cached->states[k].plant.e_y,
                  independent->states[k].plant.e_y,1.e-7)<<k;
      EXPECT_NEAR(cached->states[k].plant.r,
                  independent->states[k].plant.r,1.e-7)<<k;
  }
  // An exact-validated plan must give the SAME source adjoint as the old
  // independent same-epoch re-integration; otherwise caching physical
  // states would silently optimize a different plant/Jacobian.
  double old_adjoint[PREDICTION_HORIZON][MPC_RTI_NU]{};
  double reused_adjoint[PREDICTION_HORIZON][MPC_RTI_NU]{};
  ASSERT_TRUE(mpc_rti_source_adjoint_gradient(
      &initial,0.,trajectory,count,length,PREDICTION_DT_SECONDS,
      &model,result->controls,&frozen,old_adjoint));
  ASSERT_TRUE(mpc_rti_source_adjoint_gradient_from_exact_nominal(
      independent.get(),trajectory,count,length,PREDICTION_DT_SECONDS,
      &model,&frozen,reused_adjoint));
  double max_abs_adjoint_diff=0.;
  for(int k=0;k<PREDICTION_HORIZON;++k)
    for(int a=0;a<MPC_RTI_NU;++a) {
      max_abs_adjoint_diff=std::max(max_abs_adjoint_diff,
          std::abs(old_adjoint[k][a]-reused_adjoint[k][a]));
      EXPECT_NEAR(reused_adjoint[k][a],old_adjoint[k][a],
          std::max(1.e-3,1.e-5*std::abs(old_adjoint[k][a])))
          << "stage=" <<k<<" input="<<a;
    }
  EXPECT_LT(max_abs_adjoint_diff,.01);
  // Neither an invalid reference objective nor an untrusted initial seed
  // may be silently made "source-valid" by the optimizer.
  frozen.valid=0;
  EXPECT_FALSE(mpc_rti_source_nonlinear_optimize(
      &initial,0.,trajectory,count,length,PREDICTION_DT_SECONDS,
      &cycle,&frozen,controls,&options,result.get(),nullptr));
  frozen.valid=1;
  options.maximum_wall_time_us=.000001;
  ASSERT_TRUE(mpc_rti_source_nonlinear_optimize(
      &initial,0.,trajectory,count,length,PREDICTION_DT_SECONDS,
      &cycle,&frozen,controls,&options,result.get(),nullptr));
  EXPECT_EQ(result->budget_exhausted,1);
  EXPECT_EQ(result->iterations_accepted,0);
  EXPECT_EQ(result->final_objective,result->initial_objective);
}

TEST(MpcRtiNominal, SameEpochRebuildMatchesExactControlsAndSourceTrajectory)
{
  MpcTrajectorySample_t trajectory[] = {
      {0.,0.,0.,0.,0.,2.,4.,4.,0.},
      {10.,10.,0.,0.,0.,2.,4.,4.,0.},
      {20.,20.,0.,0.,0.,2.,4.,4.,0.}};
  size_t count=3;double length=0.;
  ASSERT_TRUE(mpc_trajectory_prepare(trajectory,&count,&length));
  MpcRtiConfiguration_t config{};
  config.max_speed_mps=config.active_speed_ceiling_mps=16.f;
  config.max_steering_rad=SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m=config.first_prediction_corridor_margin_m=.05f;
  config.command_increment_qp_enabled=1;
  config.max_steering_command_step_rad=2.f*SOURCE_MAX_STEERING_RAD;
  config.max_throttle_command_step=1.f;
  MpcRtiState_t initial{};
  initial.plant.u=2.f;
  initial.plant.delayed_throttle_command_1=.3f;
  initial.plant.delayed_throttle_command_2=.3f;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&initial.plant));
  MpcModelControl_t commands[PREDICTION_HORIZON]{};
  for(int k=0;k<PREDICTION_HORIZON;++k) {
    commands[k].steering_target_rad=.03f+.002f*std::sin(.3*k);
    commands[k].throttle_command=.25f+.04f*std::cos(.2*k);
  }
  MpcRtiNominal_t same{},shifted{},exact{};
  MpcRtiReference_t refs[PREDICTION_HORIZON+1]{},shifted_refs[PREDICTION_HORIZON+1]{};
  ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
      &initial,0.,nullptr,commands,PREDICTION_HORIZON,
      trajectory,count,length,PREDICTION_DT_SECONDS,PREDICTION_HORIZON,
      &config,&same,refs));
  for(int k=0;k<PREDICTION_HORIZON;++k) {
    EXPECT_FLOAT_EQ(same.controls[k].steering_target_rad,
        commands[k].steering_target_rad);
    EXPECT_FLOAT_EQ(same.controls[k].throttle_command,
        commands[k].throttle_command);
  }
  MpcRtiCycleConfiguration_t cycle{};
  cycle.model=config;
  int failed=-1;
  ASSERT_EQ(mpc_rti_validate_candidate_plan(&initial,0.,commands,
      trajectory,count,length,PREDICTION_HORIZON,
      PREDICTION_DT_SECONDS,&cycle,&exact,&failed),
      MPC_RTI_ROLLOUT_OK)<<failed;
  for(int k=0;k<=PREDICTION_HORIZON;++k) {
    EXPECT_NEAR(same.states[k].plant.e_y,exact.states[k].plant.e_y,1e-7f);
    EXPECT_NEAR(same.states[k].plant.r,exact.states[k].plant.r,1e-6f);
    EXPECT_NEAR(same.progress[k],exact.progress[k],1e-8);
  }
  ASSERT_TRUE(mpc_rti_build_nominal_with_warm_start(
      &initial,0.,nullptr,commands,PREDICTION_HORIZON,
      trajectory,count,length,PREDICTION_DT_SECONDS,PREDICTION_HORIZON,
      &config,&shifted,shifted_refs));
  EXPECT_FLOAT_EQ(shifted.controls[0].steering_target_rad,
      commands[1].steering_target_rad);
  EXPECT_FLOAT_EQ(shifted.controls[0].throttle_command,
      commands[1].throttle_command);
}

TEST(MpcReference, IssuedCandidateSequenceSeedsNextNominalAfterShift)
{
  MpcTrajectorySample_t trajectory[3] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 2.0, 2.0, 0.0},
      {10.0, 10.0, 0.0, 0.0, 0.0, 1.5, 2.0, 2.0, 0.0},
      {20.0, 10.0, 10.0, 0.0, 0.0, 1.5, 2.0, 2.0, 0.0}};
  size_t trajectory_count = 3;
  double lap_length = 0.0;
  ASSERT_TRUE(mpc_trajectory_prepare(
      trajectory, &trajectory_count, &lap_length));

  MpcRtiConfiguration_t config{};
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 1.5f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m = 0.05f;
  config.first_prediction_corridor_margin_m = 0.05f;
  config.corridor_preview_halfwidth_m = 0.10f;
  config.nonlinear_corridor_tolerance_m = 0.01f;
  config.startup_nominal_throttle_seed = 0.9f;

  MpcRtiState_t initial{};
  MpcModelControl_t just_issued[PREDICTION_HORIZON]{};
  for (int k = 0; k < PREDICTION_HORIZON; ++k) {
    just_issued[k].steering_target_rad = 0.01f * (k + 1);
    just_issued[k].throttle_command = 0.20f + 0.01f * k;
  }
  MpcRtiMemory_t memory{};
  mpc_rti_memory_reset(&memory);
  ASSERT_TRUE(mpc_rti_memory_set_warm_start_controls(
      &memory, just_issued, PREDICTION_HORIZON));
  ASSERT_EQ(memory.warm_start_control_count, PREDICTION_HORIZON);
  ASSERT_EQ(memory.warm_start_controls_valid, 1);
  MpcRtiNominal_t nominal{};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
  ASSERT_TRUE(mpc_rti_build_nominal_with_warm_start(&initial, 0.0, nullptr,
      memory.warm_start_controls, memory.warm_start_control_count,
      trajectory, trajectory_count,
      lap_length, PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &config,
      &nominal, references));

  EXPECT_FLOAT_EQ(nominal.controls[0].steering_target_rad,
      just_issued[1].steering_target_rad);
  EXPECT_FLOAT_EQ(nominal.controls[0].throttle_command,
      just_issued[1].throttle_command);
  EXPECT_FLOAT_EQ(nominal.controls[1].steering_target_rad,
      just_issued[2].steering_target_rad);
  EXPECT_FLOAT_EQ(nominal.controls[1].throttle_command,
      just_issued[2].throttle_command);
  EXPECT_NE(nominal.controls[0].throttle_command,
      config.startup_nominal_throttle_seed)
      << "An explicit published-sequence seed must not be overwritten by "
         "the generic launch initialization.";
  // A short certified suffix contains no information about its missing tail.
  // Initialize that tail from the path, not a repeated expired steering input.
  ASSERT_TRUE(mpc_rti_build_nominal_with_warm_start(&initial, 0.0, nullptr,
      just_issued, 2, trajectory, trajectory_count, lap_length,
      PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &config,
      &nominal, references));
  EXPECT_FLOAT_EQ(nominal.controls[0].steering_target_rad,
      just_issued[1].steering_target_rad);
  EXPECT_FLOAT_EQ(nominal.controls[1].steering_target_rad, 0.0f);
  EXPECT_FLOAT_EQ(nominal.controls[PREDICTION_HORIZON - 1].steering_target_rad,
      0.0f);
  EXPECT_FLOAT_EQ(nominal.controls[1].throttle_command,
      just_issued[1].throttle_command);
  mpc_rti_memory_reset(&memory);
  EXPECT_EQ(memory.warm_start_controls_valid, 0);
  EXPECT_EQ(memory.warm_start_control_count, 0);
}

TEST(MpcReference,
     WheelSlipBelowSpeedTargetDoesNotForceUnityFullBrakeMode)
{
  MpcTrajectorySample_t trajectory[3] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 2.0, 2.0, 1.0},
      {10.0, 10.0, 0.0, 0.0, 0.0, 8.0, 2.0, 2.0, 1.0},
      {20.0, 20.0, 0.0, 0.0, 0.0, 8.0, 2.0, 2.0, 1.0}};
  size_t trajectory_count = 3;
  double lap_length = 0.0;
  ASSERT_TRUE(mpc_trajectory_prepare(
      trajectory, &trajectory_count, &lap_length));

  MpcRtiConfiguration_t config{};
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 8.0f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m = 0.05f;
  config.first_prediction_corridor_margin_m = 0.05f;
  config.corridor_preview_halfwidth_m = 0.10f;
  config.nonlinear_corridor_tolerance_m = 0.01f;
  ASSERT_TRUE(vehicle_model_set_active_target_speed_ceiling(8.0f));

  MpcRtiState_t initial{};
  initial.plant.u = 0.4f;
  initial.plant.delayed_throttle_command_1 = 0.1f;
  initial.plant.delayed_throttle_command_2 = 0.1f;
  initial.plant.source_plant_initialized = 1;
  initial.plant.source_suspension_history_initialized = 1;
  initial.plant.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT] =
      0.00464804098;
  for (int wheel = 0; wheel < 4; ++wheel)
    initial.plant.source_plant_internal_state[
        MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] = 40.0;

  MpcRtiNominal_t nominal{};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
  ASSERT_TRUE(mpc_rti_build_nominal(&initial, 0.0, nullptr,
      trajectory, trajectory_count, lap_length, PREDICTION_DT_SECONDS,
      PREDICTION_HORIZON, &config, &nominal, references));

  ASSERT_GT(nominal.states[3].incoming_max_positive_wheel_slip,
            UNITY_SOURCE_FORWARD_ASYMPTOTE_SLIP);
  EXPECT_GE(references[3].u + 1.0e-5f, nominal.states[3].plant.u);
  EXPECT_FLOAT_EQ(references[3].u, 8.0f)
      << "Keep the raceline's requested speed; the nonlinear plant rollout "
         "determines the acceleration actually reachable from wheel spin.";
  EXPECT_GE(nominal.controls[0].throttle_command, MPC_THROTTLE_MIN_DRIVE)
      << "Wheel-spin penalty must shape positive drive torque; only a braking "
         "or overspeed target selects Unity's q=0 full-brake mode.";
}

TEST(MpcRtiCommandEnvelope, FullBrakeAtRestDoesNotCreateReverseSpeed)
{
  MpcModelState_t state{};
  state.delayed_steering_command_1 = -0.2f;
  state.delayed_steering_command_2 = -0.2f;
  state.actual_steering_angle = -0.2f;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&state));

  const MpcModelControl_t full_brake{-0.2f, MPC_THROTTLE_MIN};
  const MpcStageResult_t step = mpc_vehicle_model_step(
      &state, &full_brake, PREDICTION_DT_SECONDS, 0.0f);
  ASSERT_TRUE(step.valid);
  EXPECT_GE(step.next.u, -1.0e-6f)
      << "Unity's zero-throttle brake command must not create meaningful "
         "reverse body motion from rest.";
}

TEST(MpcCandidate, PlantRolloutPreservesDriveAndBrakeBranches)
{
  MpcTrajectorySample_t trajectory[3] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 100.0, 0.0},
      {10.0, 10.0, 0.0, 0.0, 0.0, 1.0, 100.0, 100.0, 0.0},
      {20.0, 10.0, 10.0, 1.5707963267948966, 0.0, 1.0, 100.0, 100.0, 0.0}};
  size_t trajectory_count = 3;
  double lap_length = 0.0;
  ASSERT_TRUE(mpc_trajectory_prepare(
      trajectory, &trajectory_count, &lap_length));

  MpcRtiConfiguration_t config{};
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 10.0f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m = 0.05f;
  config.first_prediction_corridor_margin_m = 0.05f;
  config.corridor_preview_halfwidth_m = 0.0f;
  config.nonlinear_corridor_tolerance_m = 0.0f;

  MpcRtiState_t initial{};
  initial.plant.u = 10.0f;
  initial.plant.delayed_throttle_command_1 = 0.001f;
  initial.plant.delayed_throttle_command_2 = 0.001f;
  initial.plant.source_plant_initialized = 1;
  for (int wheel = 0; wheel < 4; ++wheel)
    initial.plant.source_plant_internal_state[
        MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] = 250.0;

  MpcModelControl_t controls[PREDICTION_HORIZON] = {};
  MpcRtiState_t states[PREDICTION_HORIZON + 1] = {};
  double progress[PREDICTION_HORIZON + 1] = {};
  for (int k = 0; k < PREDICTION_HORIZON; ++k)
    controls[k].throttle_command = MPC_THROTTLE_MIN_DRIVE;
  EXPECT_EQ(mpc_rti_rollout_candidate(&initial, 0.0,
      controls, trajectory, trajectory_count, lap_length, nullptr, nullptr,
      nullptr, 4, PREDICTION_DT_SECONDS, &config, states, progress,
      nullptr), MPC_RTI_ROLLOUT_OK);
  EXPECT_GT(states[1].incoming_max_positive_wheel_slip,
            UNITY_SOURCE_FORWARD_ASYMPTOTE_SLIP);
  EXPECT_GT(states[1].incoming_wheel_slip_excess_mps[2], 0.0);

  for (int k = 0; k < PREDICTION_HORIZON; ++k)
    controls[k].throttle_command = MPC_THROTTLE_MIN;
  EXPECT_EQ(mpc_rti_rollout_candidate(&initial, 0.0,
      controls, trajectory, trajectory_count, lap_length, nullptr, nullptr,
      nullptr, 4, PREDICTION_DT_SECONDS, &config, states, progress,
      nullptr), MPC_RTI_ROLLOUT_OK);
}

TEST(MpcCandidate, PriorPlanSuffixRequiresExactCurrentStateValidation)
{
  MpcTrajectorySample_t trajectory[3] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 5.0, 0.0},
      {10.0, 10.0, 0.0, 0.0, 0.0, 1.0, 5.0, 5.0, 0.0},
      {20.0, 20.0, 0.0, 0.0, 0.0, 1.0, 5.0, 5.0, 0.0}};
  size_t trajectory_count = 3;
  double lap_length = 0.0;
  ASSERT_TRUE(mpc_trajectory_prepare(
      trajectory, &trajectory_count, &lap_length));

  MpcRtiCycleConfiguration_t config{};
  config.model.max_speed_mps = 16.0f;
  config.model.active_speed_ceiling_mps = 4.0f;
  config.model.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.model.corridor_margin_m = 0.05f;
  config.model.first_prediction_corridor_margin_m = 0.05f;
  config.model.corridor_preview_halfwidth_m = 0.10f;
  config.model.nonlinear_corridor_tolerance_m = 0.001f;
  config.capture_diagnostic_outputs = 1;

  MpcRtiState_t initial{};
  initial.plant.u = 1.0f;
  initial.plant.delayed_throttle_command_1 = 0.1f;
  initial.plant.delayed_throttle_command_2 = 0.1f;
  initial.plant.source_plant_initialized = 1;
  initial.plant.source_suspension_history_initialized = 1;
  initial.plant.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT] =
      0.00464804098;
  for (int wheel = 0; wheel < 4; ++wheel)
    initial.plant.source_plant_internal_state[
        MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] = 1.0 / 0.059;

  MpcModelControl_t controls[PREDICTION_HORIZON]{};
  controls[0].throttle_command = 0.1f;
  controls[1].throttle_command = 0.1f;
  MpcRtiNominal_t validated{};
  MpcRtiDiagnosticStageOutput_t diagnostic_outputs[
      MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT]{};
  int diagnostic_output_count = -1;
  int failure_stage = -1;
  EXPECT_EQ(mpc_rti_validate_candidate_plan_with_diagnostics(
      &initial, 0.0, controls,
      trajectory, trajectory_count, lap_length, 2, PREDICTION_DT_SECONDS,
      &config, &validated, diagnostic_outputs, &diagnostic_output_count,
      &failure_stage), MPC_RTI_ROLLOUT_OK);
  ASSERT_TRUE(validated.valid);
  EXPECT_EQ(validated.horizon, 2);
  EXPECT_FLOAT_EQ(validated.controls[0].throttle_command, 0.1f);
  ASSERT_EQ(diagnostic_output_count, 2);
  EXPECT_EQ(diagnostic_outputs[0].stage, 1);
  EXPECT_EQ(diagnostic_outputs[1].stage, 2);
  ASSERT_TRUE(diagnostic_outputs[0].valid);
  ASSERT_TRUE(diagnostic_outputs[1].valid);
  ASSERT_TRUE(vehicle_model_set_active_target_speed_ceiling(
      config.model.active_speed_ceiling_mps));
  const MpcStageResult_t expected_step_1 = mpc_vehicle_model_step(
      &initial.plant, &controls[0], PREDICTION_DT_SECONDS, 0.0f);
  ASSERT_TRUE(expected_step_1.valid);
  const MpcStageResult_t expected_step_2 = mpc_vehicle_model_step(
      &expected_step_1.next, &controls[1], PREDICTION_DT_SECONDS, 0.0f);
  ASSERT_TRUE(expected_step_2.valid);
  for (int wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(diagnostic_outputs[0].output.longitudinal_slip[wheel],
        expected_step_1.output.longitudinal_slip[wheel]);
    EXPECT_DOUBLE_EQ(diagnostic_outputs[1].output.lateral_slip[wheel],
        expected_step_2.output.lateral_slip[wheel]);
    EXPECT_DOUBLE_EQ(diagnostic_outputs[0].output.normal_load_n[wheel],
        expected_step_1.output.normal_load_n[wheel]);
  }

  config.capture_diagnostic_outputs = 0;
  diagnostic_output_count = -1;
  EXPECT_EQ(mpc_rti_validate_candidate_plan_with_diagnostics(
      &initial, 0.0, controls, trajectory, trajectory_count, lap_length, 2,
      PREDICTION_DT_SECONDS, &config, &validated, diagnostic_outputs,
      &diagnostic_output_count, &failure_stage), MPC_RTI_ROLLOUT_OK);
  EXPECT_EQ(diagnostic_output_count, 0);

  for (int k = 0; k < PREDICTION_HORIZON - 1; ++k) {
    controls[k].steering_target_rad = 0.0f;
    controls[k].throttle_command = 0.1f;
  }
  config.capture_diagnostic_outputs = 1;
  diagnostic_output_count = -1;
  EXPECT_EQ(mpc_rti_validate_candidate_plan_with_diagnostics(
      &initial, 0.0, controls, trajectory, trajectory_count, lap_length,
      PREDICTION_HORIZON - 1, PREDICTION_DT_SECONDS, &config, &validated,
      diagnostic_outputs, &diagnostic_output_count, &failure_stage),
      MPC_RTI_ROLLOUT_OK);
  ASSERT_TRUE(validated.valid);
  EXPECT_EQ(validated.horizon, PREDICTION_HORIZON - 1);
  EXPECT_GT(diagnostic_output_count, 0);
  EXPECT_LE(diagnostic_output_count, MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT);

  controls[1].steering_target_rad = SOURCE_MAX_STEERING_RAD + 0.01f;
  EXPECT_EQ(mpc_rti_validate_candidate_plan(&initial, 0.0, controls,
      trajectory, trajectory_count, lap_length, 2, PREDICTION_DT_SECONDS,
      &config, &validated, &failure_stage),
      MPC_RTI_ROLLOUT_COMMAND_LIMIT);
  EXPECT_FALSE(validated.valid);
}

TEST(MpcRti, IterationLimitedCandidateIsAcceptedOnlyAfterExactValidation)
{
  MpcTrajectorySample_t trajectory[3] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0},
      {10.0, 10.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0},
      {20.0, 20.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0}};
  size_t trajectory_count = 3;
  double lap_length = 0.0;
  ASSERT_TRUE(mpc_trajectory_prepare(
      trajectory, &trajectory_count, &lap_length));

  MpcRtiCycleConfiguration_t config{};
  config.model.weight_e_y = 20.0f;
  config.model.weight_e_psi = 5.0f;
  config.model.weight_u = 5.0f;
  config.model.weight_u_overspeed = 5.0f;
  config.model.weight_steering_command = 1.0f;
  config.model.weight_steering_target = 1.0f;
  config.model.weight_throttle_command = 0.1f;
  config.model.weight_steering_slew = 2.0f;
  config.model.weight_throttle_slew = 0.2f;
  config.model.terminal_multiplier = 1.0f;
  config.model.max_speed_mps = 16.0f;
  config.model.active_speed_ceiling_mps = 2.0f;
  config.model.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.model.corridor_margin_m = 0.05f;
  config.model.first_prediction_corridor_margin_m = 0.05f;
  config.model.corridor_preview_halfwidth_m = 0.10f;
  config.model.nonlinear_corridor_tolerance_m = 0.001f;
  config.model.jacobian_threads = 0;
  config.solver.rho = 7.0f;
  config.solver.rho_u = 7.0f;
  config.solver.rho_max = 32.0f;
  config.solver.tolerance = 1.0e-9f;
  config.solver.max_iterations = 1;
  config.solver.over_relaxation = 1.0f;
  MpcRtiState_t initial{};
  initial.plant.e_y = 0.15f;
  initial.plant.u = 2.0f;
  initial.plant.delayed_throttle_command_1 = 0.1f;
  initial.plant.delayed_throttle_command_2 = 0.1f;
  initial.plant.source_plant_initialized = 1;
  initial.plant.source_suspension_history_initialized = 1;
  initial.plant.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT] =
      0.00464804098;
  for (int wheel = 0; wheel < 4; ++wheel)
    initial.plant.source_plant_internal_state[
        MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] = 2.0 / 0.059;

  MpcRtiMemory_t memory{};
  mpc_rti_memory_reset(&memory);
  MpcRtiCycleResult_t result{};
  const MpcRtiCycleStatus_t status = mpc_rti_solve_cycle(
      &initial, 0.0, trajectory, trajectory_count, lap_length,
      PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &config, &memory, &result);
  EXPECT_EQ(result.admm_status, RICCATI_STATUS_MAX_ITERATIONS);
  EXPECT_EQ(result.solver_time_budget_exhausted, 0);
  EXPECT_EQ(status, MPC_RTI_CYCLE_ACCEPTED_FEASIBLE)
      << "An iteration-limited action is publishable only after the exact "
         "nonlinear state/corridor validation succeeds.";
  EXPECT_EQ(result.line_search_attempted, 0);
  EXPECT_EQ(result.line_search_accepted, 0);
  EXPECT_TRUE(memory.nominal.valid);
  EXPECT_TRUE(std::isfinite(result.first_control.throttle_command));

  config.feedback_rollout = 1;
  mpc_rti_memory_reset(&memory);
  const auto feedback_status = mpc_rti_solve_cycle(&initial, 0., trajectory,
      trajectory_count, lap_length, PREDICTION_DT_SECONDS, PREDICTION_HORIZON,
      &config, &memory, &result);
  ASSERT_TRUE(feedback_status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
      feedback_status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
      feedback_status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH);
  MpcRtiNominal_t explicit_replay{};
  int explicit_failure;
  ASSERT_EQ(mpc_rti_validate_candidate_plan(&initial, 0., memory.nominal.controls,
      trajectory, trajectory_count, lap_length, PREDICTION_HORIZON,
      PREDICTION_DT_SECONDS, &config, &explicit_replay, &explicit_failure), MPC_RTI_ROLLOUT_OK);
  for (int k=0; k<=PREDICTION_HORIZON; ++k) {
    EXPECT_NEAR(explicit_replay.states[k].plant.e_y, memory.nominal.states[k].plant.e_y,1.e-6f);
    EXPECT_NEAR(explicit_replay.states[k].plant.r, memory.nominal.states[k].plant.r,1.e-6f);
    EXPECT_NEAR(explicit_replay.progress[k], memory.nominal.progress[k],1.e-6);
  }

  mpc_rti_memory_reset(&memory);
  config.absolute_deadline_monotonic_us = mpc_rti_monotonic_time_us() - 1.0;
  const MpcRtiCycleStatus_t expired_status = mpc_rti_solve_cycle(
      &initial, 0.0, trajectory, trajectory_count, lap_length,
      PREDICTION_DT_SECONDS, PREDICTION_HORIZON, &config, &memory, &result);
  EXPECT_EQ(expired_status, MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET);
  EXPECT_EQ(result.solver_skipped_for_deadline, 1);
  EXPECT_EQ(result.solver_time_budget_exhausted, 0);
  EXPECT_EQ(result.solver_iterations, 0);
}

TEST(MpcRtiObjective, NearWallBiasIsBoundedFrozenAndInactiveAwayFromWall)
{
  MpcRtiConfiguration_t config{};
  config.corridor_margin_m = 0.02f;
  config.first_prediction_corridor_margin_m = 0.02f;
  config.nonlinear_corridor_tolerance_m = 0.f;
  config.near_wall_tracking_reserve_m = 0.025f;
  config.weight_u = 1.f;
  MpcRtiNominal_t nominal{};
  nominal.valid = 1;
  nominal.horizon = 3;
  MpcRtiReference_t reference[PREDICTION_HORIZON+1]{};
  for (int k=0; k<=nominal.horizon; ++k) {
    reference[k].u = 3.f;
    reference[k].left_bound = 1.f;
    reference[k].right_bound = 0.4f;
  }
  float lower=0.f, upper=0.f;
  ASSERT_TRUE(mpc_rti_compute_corridor_slack(&nominal.states[1].plant,
      reference[1].left_bound, reference[1].right_bound, &config, 1,
      &lower, &upper));
  nominal.states[1].plant.e_y = 0.004f-lower;
  ASSERT_TRUE(mpc_rti_compute_corridor_slack(&nominal.states[1].plant,
      reference[1].left_bound, reference[1].right_bound, &config, 1,
      &lower, &upper));
  ASSERT_NEAR(lower, 0.004f, 1.e-5f);
  ASSERT_GT(upper, config.near_wall_tracking_reserve_m);
  /* Mirror the test on the other corridor wall. */
  reference[2].left_bound = 0.4f;
  reference[2].right_bound = 1.f;
  ASSERT_TRUE(mpc_rti_compute_corridor_slack(&nominal.states[2].plant,
      reference[2].left_bound, reference[2].right_bound, &config, 2,
      &lower, &upper));
  nominal.states[2].plant.e_y = upper-0.004f;
  MpcRtiObjectiveSchedule_t schedule{};
  ASSERT_TRUE(mpc_rti_build_objective_schedule(&nominal, reference,
      &config, &schedule));
  EXPECT_TRUE(schedule.valid);
  EXPECT_FLOAT_EQ(schedule.references[0].e_y, 0.f);
  EXPECT_NEAR(schedule.references[1].e_y, 0.021f, 1.e-5f);
  EXPECT_NEAR(schedule.references[2].e_y, -0.021f, 1.e-5f);
  EXPECT_FLOAT_EQ(schedule.references[3].e_y, 0.f);
  EXPECT_FLOAT_EQ(reference[1].e_y, 0.f);
  /* This remains a pure opt-in objective change. No hard wall constraints
   * or live source-plant validation are weakened. */
  config.near_wall_tracking_reserve_m = 0.f;
  ASSERT_TRUE(mpc_rti_build_objective_schedule(&nominal, reference,
      &config, &schedule));
  for (int k=0; k<=nominal.horizon; ++k)
    EXPECT_FLOAT_EQ(schedule.references[k].e_y, reference[k].e_y);
}

TEST(MpcRtiObjective, FrozenTargetsKeepCandidateProgressCorridorExact)
{
  MpcTrajectorySample_t trajectory[] = {
      {0.,0.,0.,0.,0.,2.,2.,2.,0.},
      {10.,10.,0.,0.,0.,2.,.30,2.,0.},
      {20.,20.,0.,0.,0.,2.,2.,2.,0.}};
  size_t count=3;
  double length=0.;
  ASSERT_TRUE(mpc_trajectory_prepare(trajectory,&count,&length));
  MpcRtiCycleConfiguration_t config{};
  config.model.weight_e_y=10.f;
  config.model.weight_u=2.f;
  config.model.weight_u_overspeed=10.f;
  config.model.weight_steering_target=3.f;
  config.model.terminal_multiplier=2.f;
  config.model.corridor_margin_m=.05f;
  config.model.first_prediction_corridor_margin_m=.05f;
  MpcRtiNominal_t plan{};
  plan.valid=1;
  plan.horizon=1;
  plan.controls[0]={.1f,.2f};
  for (int k=0;k<2;++k) {
    plan.states[k].plant.u=2.f;
    plan.states[k].plant.e_y=.2f;
  }
  MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
  for (int k=0;k<2;++k) {
    references[k].u=2.f;
    references[k].source_witness_valid=1;
    references[k].source_steering_target=.2f;
    references[k].left_bound=references[k].right_bound=100.f;
  }
  MpcRtiObjectiveSchedule_t schedule{};
  ASSERT_TRUE(mpc_rti_build_objective_schedule(&plan,references,&config.model,&schedule));
  config.objective_schedule=&schedule;
  plan.progress[1]=.1;
  ASSERT_TRUE(mpc_rti_score_candidate_plan(&plan,trajectory,count,length,&config));
  const float objective=plan.validation_objective;
  plan.progress[1]=5.;
  ASSERT_TRUE(mpc_rti_score_candidate_plan(&plan,trajectory,count,length,&config));
  EXPECT_FLOAT_EQ(plan.validation_objective,objective);
  // The frozen objective's deliberately wide bounds cannot widen the real
  // wall corridor sampled at the candidate's new progress.
  plan.progress[1]=10.;
  EXPECT_FALSE(mpc_rti_score_candidate_plan(&plan,trajectory,count,length,&config));
}

TEST(MpcRtiCostGradient, CondensedDirectionMatchesExactRollout)
{
  for (int increment_mode : {0, 1}) {
  for (int scenario : {0, -1, 1, 3, -2, 2}) {
    const bool source_witness=std::abs(scenario)>1;
    const int turn=scenario==3 ? 0 : scenario<0 ? -1 : scenario>0 ? 1 : 0;
    SCOPED_TRACE(turn);
    const float direction = turn < 0 ? -1.f : 1.f;
    std::vector<MpcTrajectorySample_t> samples = {
        {0., 0., 0., 0., 0., 5., 10., 10., 0.},
        {30., 30., 0., 0., 0., 5., 10., 10., 0.},
        {60., 60., 0., 0., 0., 5., 10., 10., 0.}};
    if (turn) {
      samples.clear();
      double station = 0.;
      for (int i = 0; i < 240; ++i) {
        const double angle = i * 2. * M_PI / 240.;
        const double x = 3.*std::cos(angle), y = turn*5.*std::sin(angle);
        if (i) station += std::hypot(x-samples.back().x, y-samples.back().y);
        samples.push_back({station, x, y,
            std::atan2(turn*5.*std::cos(angle), -3.*std::sin(angle)),
            turn*15./std::pow(9.*std::sin(angle)*std::sin(angle) +
                25.*std::cos(angle)*std::cos(angle), 1.5),
            5., 10., 10., 0.});
      }
    }
    auto *trajectory = samples.data();
    if (source_witness) {
      for (auto &p : samples) {
        p.source_witness_valid=1;
        p.source_heading_error=.03*direction+.01*std::sin(.3*p.s);
        p.source_v_rear=-.1*direction+.04*std::sin(.6*p.s);
        p.source_yaw_rate=.2*direction+.2*std::sin(.4*p.s);
        p.source_physical_steering=.03*direction+.02*std::sin(.3*p.s);
        p.source_steering_target=.04*direction+.06*std::sin(.8*p.s);
        p.source_throttle=.2;
      }
    }
    size_t count = samples.size();
    double length = 0.;
    ASSERT_TRUE(mpc_trajectory_prepare(trajectory, &count, &length));
    MpcRtiCycleConfiguration_t config{};
    auto &c = config.model;
    c.weight_e_y = 4000.f;
    c.weight_e_psi = 1500.f;
    c.weight_u = 35.f;
    c.weight_u_overspeed = 20.f;
    c.overspeed_curvature_gain_m = .5f;
    c.weight_v = 5.f;
    c.weight_r = 7.8125f;
    c.weight_longitudinal_slip = 5.f;
    c.weight_steering_command = 3.f;
    c.weight_steering_target = 5.f;
    c.weight_throttle_command = .1f;
    c.weight_steering_slew = 2000.f;
    c.weight_throttle_slew = 50.f;
    c.terminal_multiplier = 3.f;
    c.max_speed_mps = c.active_speed_ceiling_mps = 16.f;
    c.max_steering_rad = SOURCE_MAX_STEERING_RAD;
    // The same QP/exact objective-gradient contract must hold after
    // reparameterizing controls as successive command increments.
    c.command_increment_qp_enabled = increment_mode;
    if (increment_mode) {
      c.max_steering_command_step_rad = 2.f*SOURCE_MAX_STEERING_RAD;
      c.max_throttle_command_step = 1.f;
    }
    c.corridor_margin_m = c.first_prediction_corridor_margin_m = .05f;
    MpcRtiState_t initial{};
    initial.plant.u = 5.f;
    initial.plant.e_y = direction * .08f;
    initial.plant.e_psi = direction * .02f;
    initial.plant.delayed_throttle_command_1 = .2f;
    initial.plant.delayed_throttle_command_2 = .2f;
    ASSERT_TRUE(vehicle_model_initialize_plant_state(&initial.plant));
    MpcRtiNominal_t previous{}, nominal{};
    previous.valid = 1;
    previous.horizon = PREDICTION_HORIZON;
    for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
      previous.progress[k] = 5. + k * .125;
      if (k < PREDICTION_HORIZON)
        previous.controls[k] = {direction * (.02f + .01f * std::sin(.3f*k)),
            .2f + .01f * std::cos(.2f*k)};
    }
    MpcRtiReference_t references[PREDICTION_HORIZON + 1];
    ASSERT_TRUE(mpc_rti_build_nominal(&initial, 5., &previous,
        trajectory, count, length, PREDICTION_DT_SECONDS, PREDICTION_HORIZON,
        &c, &nominal, references));
    MpcRtiProblem_t problem{};
    ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal.states, nominal.controls,
        references, PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &c, &problem));
    MpcRtiObjectiveSchedule_t schedule{};
    ASSERT_TRUE(mpc_rti_build_objective_schedule(&nominal,references,&c,&schedule));
    config.objective_schedule=&schedule;
    for (int input : {0, 1}) {
      for (int changed_stage : {0, 8, 20}) {
        double x[RICCATI_MAX_NX], dx[RICCATI_MAX_NX]{};
        std::copy_n(problem.x0, RICCATI_MAX_NX, x);
        double derivative = 0.;
        for (int k = 0; k < RICCATI_HORIZON; ++k) {
          const auto &stage = problem.steps[k];
          double u[2] = {nominal.controls[k].steering_target_rad,
              nominal.controls[k].throttle_command};
          double du[2]{};
          if (increment_mode) {
            u[0] -= x[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND];
            u[1] -= x[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
          }
          if (k == changed_stage) du[input] = 1.;
          // An absolute single-stage command perturbation also alters the
          // NEXT increment by -delta, because v[k+1]=u[k+1]-u[k].
          if (increment_mode && k == changed_stage + 1) du[input] = -1.;
          for (int i = 0; i < RICCATI_MAX_NX; ++i) {
            double gx = stage.Q_diag[i] * x[i] + stage.q[i];
            for(int j=0;j<RICCATI_MAX_NX;++j)
                gx+=stage.Q_offdiag[i][j]*x[j];
            for (int a = 0; a < 2; ++a) gx += stage.N[i][a] * u[a];
            derivative += gx * dx[i];
          }
          for (int a = 0; a < 2; ++a) {
            double gu = stage.R_diag[a] * u[a] + stage.r[a];
            gu+=stage.R_cross*u[1-a];
            for (int i = 0; i < RICCATI_MAX_NX; ++i) gu += stage.N[i][a] * x[i];
            derivative += gu * du[a];
          }
          double next[RICCATI_MAX_NX]{}, dnext[RICCATI_MAX_NX]{};
          for (int i = 0; i < RICCATI_MAX_NX; ++i) {
            next[i] = stage.d[i];
            for (int j = 0; j < RICCATI_MAX_NX; ++j) {
              next[i] += stage.A[i][j] * x[j];
              dnext[i] += stage.A[i][j] * dx[j];
            }
            for (int a = 0; a < 2; ++a) {
              next[i] += stage.B[i][a] * u[a];
              dnext[i] += stage.B[i][a] * du[a];
            }
          }
          std::copy_n(next, RICCATI_MAX_NX, x);
          std::copy_n(dnext, RICCATI_MAX_NX, dx);
        }
        for (int i = 0; i < RICCATI_MAX_NX; ++i)
          derivative += (problem.terminal_Q[i] * x[i] + problem.terminal_q[i]) * dx[i];
        double costs[2];
        constexpr float epsilon = .0001f;
        // A 5e-4 throttle perturbation produced a 1--2% recursive secant/local
        // gradient difference. Use 1e-4 for the local check; do not mistake the
        // larger finite step for the derivative or attribute it to an unproven
        // specific physics branch. Account for float objective resolution.
        for (int sign = 0; sign < 2; ++sign) {
          MpcModelControl_t controls[PREDICTION_HORIZON];
          std::copy_n(nominal.controls, PREDICTION_HORIZON, controls);
          float &value = input == 0 ? controls[changed_stage].steering_target_rad
              : controls[changed_stage].throttle_command;
          value += (sign == 0 ? -epsilon : epsilon);
          MpcRtiNominal_t exact{};
          int failed_stage;
          ASSERT_EQ(mpc_rti_validate_candidate_plan(&initial, 5., controls,
              trajectory, count, length, PREDICTION_HORIZON,
              PREDICTION_DT_SECONDS, &config, &exact, &failed_stage), MPC_RTI_ROLLOUT_OK);
          costs[sign] = exact.validation_objective;
        }
        const double finite_derivative = (costs[1] - costs[0]) / (2. * epsilon);
        EXPECT_NEAR(derivative, finite_derivative,
            .005 * std::max(1., std::abs(finite_derivative)))
            << "input=" << input << " stage=" << changed_stage;
      }
    }
  }  }
}

TEST(MpcSlipCost, SourceSlipLossEntersQpAndExactStageObjective)
{
  MpcRtiConfiguration_t config{};
  config.weight_longitudinal_slip = 50.0f;
  config.terminal_multiplier = 1.0f;
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 16.0f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.first_prediction_corridor_margin_m = 0.0f;
  config.corridor_margin_m = 0.0f;

  MpcRtiState_t nominal_states[PREDICTION_HORIZON + 1] = {};
  nominal_states[0].plant.u = 8.0f;
  nominal_states[0].plant.delayed_throttle_command_1 = 0.001f;
  nominal_states[0].plant.delayed_throttle_command_2 = 0.001f;
  nominal_states[0].plant.source_plant_initialized = 1;
  nominal_states[0].plant.source_suspension_history_initialized = 1;
  nominal_states[0].plant.source_plant_internal_state[
      MPC_PLANT_STATE_BODY_HEIGHT] = 0.00464804098;
  for (int wheel = 0; wheel < 4; ++wheel)
    nominal_states[0].plant.source_plant_internal_state[
        MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] = 8.0 / 0.059;
  nominal_states[0].plant.source_plant_internal_state[
      MPC_PLANT_STATE_WHEEL_OMEGA_RL] = 250.0;

  MpcModelControl_t controls[PREDICTION_HORIZON] = {};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
  for (int stage = 0; stage <= PREDICTION_HORIZON; ++stage) {
    references[stage].u = 8.0f;
    references[stage].path_curvature = 0.5f;
    references[stage].left_bound = 10.0f;
    references[stage].right_bound = 10.0f;
  }
  for (int k = 0; k < PREDICTION_HORIZON; ++k)
    controls[k].throttle_command = 0.001f;
  ASSERT_TRUE(vehicle_model_set_active_target_speed_ceiling(16.0f));
  for (int k = 0; k < PREDICTION_HORIZON; ++k) {
    const MpcStageResult_t predicted = mpc_vehicle_model_step(
        &nominal_states[k].plant, &controls[k],
        PREDICTION_DT_SECONDS, references[k].path_curvature);
    ASSERT_TRUE(predicted.valid);
    nominal_states[k + 1].plant = predicted.next;
    nominal_states[k + 1].incoming_branch_flags = predicted.branch_flags;
    nominal_states[k + 1].incoming_body_forward_delta_m =
        predicted.body_forward_delta_m;
    nominal_states[k + 1].incoming_body_lateral_delta_m =
        predicted.body_lateral_delta_m;
    nominal_states[k + 1].incoming_body_yaw_delta_rad =
        predicted.body_yaw_delta_rad;
    nominal_states[k + 1].incoming_transition_valid = 1;
  }
  MpcRtiProblem_t problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, controls, references,
      PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &config, &problem));
  bool slip_cost_retained_after_queued_prefix = false;
  int slip_cost_solver_stage = -1;
  for (int solver_stage = 1; solver_stage < RICCATI_HORIZON;
       ++solver_stage) {
    const int model_stage = solver_stage + MPC_COMMAND_DELAY_STEPS;
    double excess[4] = {}, derivative[4] = {};
    ASSERT_TRUE(vehicle_model_longitudinal_slip_excess(
        &nominal_states[model_stage].plant, excess, derivative));
    if (excess[2] > 0.0 && std::abs(derivative[2]) > 1.0e-9) {
      EXPECT_GT(problem.steps[solver_stage].Q_diag[6 + 2], 0.0f);
      slip_cost_retained_after_queued_prefix = true;
      slip_cost_solver_stage = solver_stage;
      break;
    }
  }
  EXPECT_TRUE(slip_cost_retained_after_queued_prefix);

  // Compare the QP's gradient with an independent held-interval peak cost.
  // The former wheel-only instantaneous residual had zero body-speed
  // gradient here, despite the exact peak clearly depending on body speed.
  MpcRtiConfiguration_t no_slip_config = config;
  no_slip_config.weight_longitudinal_slip = 0.0f;
  MpcRtiProblem_t no_slip_problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, controls, references,
      PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &no_slip_config,
      &no_slip_problem));
  auto plus_plant = nominal_states[0].plant;
  auto minus_plant = nominal_states[0].plant;
  constexpr double epsilon = 1.e-3;
  plus_plant.u += epsilon;
  minus_plant.u -= epsilon;
  const auto plus_peak = mpc_vehicle_model_step(&plus_plant, &controls[0],
      PREDICTION_DT_SECONDS, references[0].path_curvature);
  const auto minus_peak = mpc_vehicle_model_step(&minus_plant, &controls[0],
      PREDICTION_DT_SECONDS, references[0].path_curvature);
  ASSERT_TRUE(plus_peak.valid && minus_peak.valid);
  double finite_gradient = 0.;
  for (int wheel = 0; wheel < 4; ++wheel) {
    const double plus = plus_peak.output.maximum_longitudinal_slip_excess_mps[wheel];
    const double minus = minus_peak.output.maximum_longitudinal_slip_excess_mps[wheel];
    finite_gradient += config.weight_longitudinal_slip *
        (plus*plus-minus*minus)/(plus_plant.u-minus_plant.u);
  }
  // Full GN includes state/state, state/control and input/input couplings.
  // Evaluate its ENTIRE gradient at the nominal physical state, not only Qii.
  const auto &with=problem.steps[0],&without=no_slip_problem.steps[0];
  float xnom[MPC_RTI_NX]{};
  xnom[MPC_OPT_IDX_EY]=nominal_states[0].plant.e_y;
  xnom[MPC_OPT_IDX_EPSI]=nominal_states[0].plant.e_psi;
  xnom[MPC_OPT_IDX_U]=nominal_states[0].plant.u;
  xnom[MPC_OPT_IDX_V]=nominal_states[0].plant.v;
  xnom[MPC_OPT_IDX_R]=nominal_states[0].plant.r;
  xnom[MPC_OPT_IDX_ACTUAL_STEERING_ANGLE]=
      nominal_states[0].plant.actual_steering_angle;
  for(int i=0;i<14;++i)
    xnom[6+i]=static_cast<float>(
        nominal_states[0].plant.source_plant_internal_state[i]);
  double qp_gradient=(with.Q_diag[MPC_OPT_IDX_U]-
      without.Q_diag[MPC_OPT_IDX_U])*xnom[MPC_OPT_IDX_U] +
      with.q[MPC_OPT_IDX_U]-without.q[MPC_OPT_IDX_U];
  for(int j=0;j<MPC_RTI_NX;++j)
    qp_gradient+=(with.Q_offdiag[MPC_OPT_IDX_U][j]-
        without.Q_offdiag[MPC_OPT_IDX_U][j])*xnom[j];
  for(int a=0;a<MPC_RTI_NU;++a)
    qp_gradient+=(with.N[MPC_OPT_IDX_U][a]-
        without.N[MPC_OPT_IDX_U][a])*
        (a==0?controls[0].steering_target_rad:controls[0].throttle_command);
  EXPECT_GT(std::abs(finite_gradient), 1.);
  EXPECT_NEAR(qp_gradient, finite_gradient, 0.15);

  MpcRtiState_t measured{};
  measured.plant = nominal_states[0].plant;
  const MpcModelControl_t control{};
  config.weight_longitudinal_slip = 0.0f;
  const float cost_without = mpc_rti_scalar_stage_cost(
      &config, &references[0], &measured, &control);
  config.weight_longitudinal_slip = 50.0f;
  const float cost_with = mpc_rti_scalar_stage_cost(
      &config, &references[0], &measured, &control);
  double excess[4] = {0}, derivative[4] = {0};
  ASSERT_TRUE(vehicle_model_longitudinal_slip_excess(
      &measured.plant, excess, derivative));
  double expected_penalty = 0.0;
  for (double value : excess) expected_penalty += 50.0 * value * value;
  EXPECT_NEAR(cost_with - cost_without, expected_penalty, 1.0e-4);

  config.longitudinal_slip_curvature_gain_m = 1.0f;
  MpcRtiProblem_t curvature_scheduled_problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, controls, references,
      PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &config,
      &curvature_scheduled_problem));
  ASSERT_GE(slip_cost_solver_stage, 0);
  const float uniform_wheel_hessian =
      problem.steps[slip_cost_solver_stage].Q_diag[6 + 2];
  const float scheduled_wheel_hessian = curvature_scheduled_problem
      .steps[slip_cost_solver_stage].Q_diag[6 + 2];
  EXPECT_NEAR(scheduled_wheel_hessian / uniform_wheel_hessian, 1.5f,
      1.0e-4f);
  const float scheduled_cost = mpc_rti_scalar_stage_cost(
      &config, &references[0], &measured, &control);
  double expected_scheduled_penalty = 0.0;
  for (double value : excess)
    expected_scheduled_penalty += 75.0 * value * value;
  EXPECT_NEAR(scheduled_cost - cost_without,
      expected_scheduled_penalty, 1.0e-4);

  config.weight_u = 7.5f;
  config.weight_u_overspeed = 60.0f;
  config.longitudinal_slip_curvature_gain_m = 0.0f;
  config.overspeed_curvature_gain_m = 0.0f;
  measured.plant.u = 9.0f;
  references[0].u = 8.0f;
  const float overspeed_cost_uniform = mpc_rti_scalar_stage_cost(
      &config, &references[0], &measured, &control);
  config.overspeed_curvature_gain_m = 1.0f;
  const float overspeed_cost_curvature = mpc_rti_scalar_stage_cost(
      &config, &references[0], &measured, &control);
  EXPECT_NEAR(overspeed_cost_curvature - overspeed_cost_uniform,
      60.0f * 0.5f, 1.0e-4f);
  measured.plant.u = 7.0f;
  const float underspeed_cost_uniform = mpc_rti_scalar_stage_cost(
      &config, &references[0], &measured, &control);
  config.overspeed_curvature_gain_m = 0.0f;
  const float underspeed_cost_without_curvature = mpc_rti_scalar_stage_cost(
      &config, &references[0], &measured, &control);
  EXPECT_FLOAT_EQ(underspeed_cost_uniform, underspeed_cost_without_curvature);

  config.overspeed_curvature_gain_m = 0.0f;
  for (int stage = 0; stage <= PREDICTION_HORIZON; ++stage)
    references[stage].u = 7.0f;
  MpcRtiProblem_t uniform_speed_problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, controls, references,
      PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &config,
      &uniform_speed_problem));
  config.overspeed_curvature_gain_m = 1.0f;
  MpcRtiProblem_t curvature_speed_problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, controls, references,
      PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &config,
      &curvature_speed_problem));
  EXPECT_NEAR(
      curvature_speed_problem.steps[1].Q_diag[MPC_OPT_IDX_U] -
          uniform_speed_problem.steps[1].Q_diag[MPC_OPT_IDX_U],
      2.0f * config.weight_u_overspeed * 0.5f, 1.0e-4f);

  /* Speed catch-up stays strong on straights but is moderated in a bend.
   * The nominal rollout remains below this 10 m/s reference, so both the QP
   * and exact nonlinear objective must select the underspeed schedule. */
  config.weight_u = 40.0f;
  config.underspeed_curvature_gain_m = 0.0f;
  measured.plant.u = 7.0f;
  references[0].u = 8.0f;
  const float underspeed_cost_flat = mpc_rti_scalar_stage_cost(
      &config, &references[0], &measured, &control);
  config.underspeed_curvature_gain_m = 4.0f;
  const float underspeed_cost_scheduled = mpc_rti_scalar_stage_cost(
      &config, &references[0], &measured, &control);
  EXPECT_NEAR(underspeed_cost_scheduled - underspeed_cost_flat,
      (40.0f / 3.0f - 40.0f), 1.0e-4f);

  for (int stage = 0; stage <= PREDICTION_HORIZON; ++stage)
    references[stage].u = 10.0f;
  config.underspeed_curvature_gain_m = 0.0f;
  MpcRtiProblem_t uniform_underspeed_problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, controls, references,
      PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &config,
      &uniform_underspeed_problem));
  config.underspeed_curvature_gain_m = 4.0f;
  MpcRtiProblem_t scheduled_underspeed_problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, controls, references,
      PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &config,
      &scheduled_underspeed_problem));
  ASSERT_LT(nominal_states[3].plant.u, references[3].u);
  EXPECT_NEAR(
      scheduled_underspeed_problem.steps[1].Q_diag[MPC_OPT_IDX_U] -
          uniform_underspeed_problem.steps[1].Q_diag[MPC_OPT_IDX_U],
      2.0f * (40.0f / 3.0f - 40.0f), 1.0e-3f);
}

TEST(VehicleModelSourceRollout, SensitivityOnlyRtiMatchesFullAffineJacobian)
{
  /* The RTI-only no-affine variant must retain identical source derivatives
   * and branch diagnostics for the same known PhysX stage. */
  for (float steering : {-0.2f, 0.0f, 0.22f}) {
    MpcModelState_t state = turning_state();
    state.u = 4.0f;
    state.actual_steering_angle = steering;
    state.e_y = 0.02f;
    state.e_psi = -0.04f;
    ASSERT_TRUE(vehicle_model_initialize_plant_state(&state));
    const MpcModelControl_t control{steering, 0.2f};
    const auto step = mpc_vehicle_model_step(&state, &control, 0.025f, 0.0f);
    ASSERT_TRUE(step.valid);
    MpcStageLinearization_t full{};
    MpcStageLinearization_t fast{};
    ASSERT_TRUE(mpc_model_linearize_from_known_step(
        &state, &control, 0.025f, 0.0f, &step, &full));
    ASSERT_TRUE(mpc_model_linearize_sensitivities_from_known_step(
        &state, &control, 0.025f, 0.0f, &step, &fast));
    EXPECT_EQ(fast.valid, full.valid);
    EXPECT_EQ(fast.nominal_branch_flags, full.nominal_branch_flags);
    EXPECT_EQ(fast.nonsmooth_column_mask, full.nonsmooth_column_mask);
    EXPECT_EQ(fast.nonsmooth_column_count, full.nonsmooth_column_count);
    EXPECT_EQ(fast.finite_difference_column_count, full.finite_difference_column_count);
    EXPECT_EQ(fast.source_sensitivity_call_count, full.source_sensitivity_call_count);
    EXPECT_EQ(0, std::memcmp(fast.A, full.A, sizeof(full.A)));
    EXPECT_EQ(0, std::memcmp(fast.B, full.B, sizeof(full.B)));
    EXPECT_EQ(0, std::memcmp(fast.body_motion_A, full.body_motion_A,
                             sizeof(full.body_motion_A)));
    EXPECT_EQ(0, std::memcmp(fast.body_motion_B, full.body_motion_B,
                             sizeof(full.body_motion_B)));
    EXPECT_EQ(0, std::memcmp(fast.maximum_slip_excess_A,
                             full.maximum_slip_excess_A,
                             sizeof(full.maximum_slip_excess_A)));
    EXPECT_EQ(0, std::memcmp(fast.maximum_slip_excess_B,
                             full.maximum_slip_excess_B,
                             sizeof(full.maximum_slip_excess_B)));
    for (float d : fast.d) EXPECT_FLOAT_EQ(d, 0.0f);
  }
}

TEST(VehicleModelSourceRollout, JacobianDifferentiatesTheSameSourceStep)
{
  MpcModelState_t state = turning_state();
  state.source_plant_initialized = 1;
  for (int wheel = 0; wheel < 4; ++wheel)
    state.source_plant_internal_state[
        MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] =
        state.u / 0.059;

  const MpcModelControl_t control{};
  MpcStageResult_t nominal{};
  MpcStageLinearization_t linearization{};
  ASSERT_TRUE(mpc_vehicle_model_step_with_jacobian(
      &state, &control, 0.025f, 0.0f, &nominal, &linearization));
  ASSERT_TRUE(linearization.valid);
  EXPECT_NEAR(linearization.A[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1]
                                [MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2],
              1.0f, 2.0e-4f);
  EXPECT_FLOAT_EQ(linearization.A[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2]
                                  [MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1],
                  0.0f);

  constexpr float epsilon = 1.0e-3f;
  MpcModelState_t plus_state = state;
  MpcModelState_t minus_state = state;
  plus_state.u += epsilon;
  minus_state.u -= epsilon;
  const MpcStageResult_t plus = mpc_vehicle_model_step(
      &plus_state, &control, 0.025f, 0.0f);
  const MpcStageResult_t minus = mpc_vehicle_model_step(
      &minus_state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(plus.valid);
  ASSERT_TRUE(minus.valid);
  const float finite_difference = (plus.next.r - minus.next.r) /
      (2.0f * epsilon);
  EXPECT_NEAR(linearization.A[4][2], finite_difference,
              0.02f * std::max(1.0f, std::abs(finite_difference)));
}

TEST(VehicleModelSourceRollout,
     JacobianUsesNewestCommandAndExactHistoryShift)
{
  MpcModelState_t state = turning_state();
  state.delayed_steering_command_1 = 0.18f;
  state.delayed_steering_command_2 = -0.12f;
  state.delayed_throttle_command_1 = 0.24f;
  state.delayed_throttle_command_2 = 0.31f;
  state.source_plant_initialized = 1;
  state.source_suspension_history_initialized = 1;
  for (int wheel = 0; wheel < 4; ++wheel)
    state.source_plant_internal_state[
        MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] =
        state.u / 0.059;

  const MpcModelControl_t control{0.30f, 0.45f};
  constexpr float dt = 0.025f;
  constexpr float curvature = 0.17f;
  MpcStageResult_t nominal{};
  MpcStageLinearization_t jacobian{};
  ASSERT_TRUE(mpc_vehicle_model_step_with_jacobian(
      &state, &control, dt, curvature, &nominal, &jacobian));

  EXPECT_NEAR(jacobian.A[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1]
                            [MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2],
              1.0f, 1.0e-6f);
  EXPECT_FLOAT_EQ(jacobian.A[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2]
                              [MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1],
                  0.0f);
  EXPECT_FLOAT_EQ(jacobian.A[MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2]
                              [MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2],
                  0.0f);
  EXPECT_NEAR(jacobian.A[MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1]
                            [MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2],
              1.0f, 1.0e-6f);
  EXPECT_NEAR(jacobian.B[MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2][0],
              1.0f, 1.0e-6f);
  EXPECT_NEAR(jacobian.B[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2][1],
              1.0f, 1.0e-6f);
  for (int row = 0; row < MPC_MODEL_NX; ++row) {
    EXPECT_FLOAT_EQ(
        jacobian.A[row][MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1], 0.0f);
    EXPECT_FLOAT_EQ(
        jacobian.A[row][MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1], 0.0f);
  }

  /* Slot-1 values are historical only: the plant consumes slot 2 and each
   * next slot 1 is overwritten from slot 2. Perturbing slot 1 must therefore
   * leave every physical/path prediction unchanged. */
  MpcModelState_t plus_old_commands = state;
  MpcModelState_t minus_old_commands = state;
  plus_old_commands.delayed_steering_command_1 += 0.1f;
  minus_old_commands.delayed_steering_command_1 -= 0.1f;
  plus_old_commands.delayed_throttle_command_1 += 0.1f;
  minus_old_commands.delayed_throttle_command_1 -= 0.1f;
  const auto plus_old_command_step = mpc_vehicle_model_step(
      &plus_old_commands, &control, dt, curvature);
  const auto minus_old_command_step = mpc_vehicle_model_step(
      &minus_old_commands, &control, dt, curvature);
  ASSERT_TRUE(plus_old_command_step.valid);
  ASSERT_TRUE(minus_old_command_step.valid);
  EXPECT_FLOAT_EQ(plus_old_command_step.next.e_y,
                  minus_old_command_step.next.e_y);
  EXPECT_FLOAT_EQ(plus_old_command_step.next.e_psi,
                  minus_old_command_step.next.e_psi);
  EXPECT_FLOAT_EQ(plus_old_command_step.next.u,
                  minus_old_command_step.next.u);
  EXPECT_FLOAT_EQ(plus_old_command_step.next.v,
                  minus_old_command_step.next.v);
  EXPECT_FLOAT_EQ(plus_old_command_step.next.r,
                  minus_old_command_step.next.r);
  EXPECT_FLOAT_EQ(plus_old_command_step.next.actual_steering_angle,
                  minus_old_command_step.next.actual_steering_angle);
  for (int i = 0; i < MPC_PLANT_INTERNAL_STATE_COUNT; ++i)
    EXPECT_DOUBLE_EQ(
        plus_old_command_step.next.source_plant_internal_state[i],
        minus_old_command_step.next.source_plant_internal_state[i]);
  for (int wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(plus_old_command_step.output.longitudinal_slip[wheel],
                     minus_old_command_step.output.longitudinal_slip[wheel]);
    EXPECT_DOUBLE_EQ(plus_old_command_step.output.lateral_slip[wheel],
                     minus_old_command_step.output.lateral_slip[wheel]);
  }

  constexpr float path_epsilons[] = {1.0e-4f, 1.0e-3f};
  for (int column = 0; column < 2; ++column) {
    const float epsilon = path_epsilons[column];
    MpcModelState_t plus_state = state;
    MpcModelState_t minus_state = state;
    if (column == 0) {
      plus_state.e_y += epsilon;
      minus_state.e_y -= epsilon;
    } else {
      plus_state.e_psi += epsilon;
      minus_state.e_psi -= epsilon;
    }
    const MpcStageResult_t plus = mpc_vehicle_model_step(
        &plus_state, &control, dt, curvature);
    const MpcStageResult_t minus = mpc_vehicle_model_step(
        &minus_state, &control, dt, curvature);
    ASSERT_TRUE(plus.valid);
    ASSERT_TRUE(minus.valid);
    EXPECT_NEAR(jacobian.A[0][column],
        (plus.next.e_y - minus.next.e_y) / (2.0f * epsilon), 2.0e-3f);
    EXPECT_NEAR(jacobian.A[1][column],
        std::remainder(plus.next.e_psi - minus.next.e_psi,
                       6.2831853071795864769f) / (2.0f * epsilon),
        2.0e-3f);
  }

  MpcModelState_t plus_command = state;
  MpcModelState_t minus_command = state;
  plus_command.delayed_steering_command_2 += 1.0e-4f;
  minus_command.delayed_steering_command_2 -= 1.0e-4f;
  const auto plus_steering = mpc_vehicle_model_step(
      &plus_command, &control, dt, curvature);
  const auto minus_steering = mpc_vehicle_model_step(
      &minus_command, &control, dt, curvature);
  ASSERT_TRUE(plus_steering.valid);
  ASSERT_TRUE(minus_steering.valid);
  EXPECT_NEAR(jacobian.A[MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1]
                          [MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2],
      (plus_steering.next.delayed_steering_command_1 -
       minus_steering.next.delayed_steering_command_1) / 2.0e-4f,
      1.0e-3f);

  EXPECT_FLOAT_EQ(jacobian.A[MPC_STATE_IDX_ACTUAL_STEERING_ANGLE]
                              [MPC_STATE_IDX_ACTUAL_STEERING_ANGLE],
                  0.0f);

  EXPECT_FLOAT_EQ(jacobian.A[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1]
                              [MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2],
                  1.0f);
  EXPECT_FLOAT_EQ(jacobian.A[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2]
                              [MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1],
                  0.0f);
  EXPECT_FLOAT_EQ(jacobian.A[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2]
                              [MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2],
                  0.0f);
  MpcModelState_t plus_throttle = state;
  MpcModelState_t minus_throttle = state;
  plus_throttle.delayed_throttle_command_2 += 1.0e-4f;
  minus_throttle.delayed_throttle_command_2 -= 1.0e-4f;
  const auto plus_throttle_step = mpc_vehicle_model_step(
      &plus_throttle, &control, dt, curvature);
  const auto minus_throttle_step = mpc_vehicle_model_step(
      &minus_throttle, &control, dt, curvature);
  ASSERT_TRUE(plus_throttle_step.valid);
  ASSERT_TRUE(minus_throttle_step.valid);
  const float du_dq = jacobian.A[MPC_STATE_IDX_U]
      [MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2];
  const float dr_dq = jacobian.A[MPC_STATE_IDX_R]
      [MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2];
  EXPECT_NEAR(du_dq,
      (plus_throttle_step.next.u - minus_throttle_step.next.u) / 2.0e-4f,
      0.02f * std::max(1.0f, std::abs(du_dq)));
  EXPECT_NEAR(dr_dq,
      (plus_throttle_step.next.r - minus_throttle_step.next.r) / 2.0e-4f,
      0.02f * std::max(1.0f, std::abs(dr_dq)));
}

TEST(VehicleModelSourceRollout,
     BoundedActuatorQueueJacobiansPreserveFeasibleOneSidedCommands)
{
  for (const float command : {0.0f, 1.0f}) {
    MpcModelState_t state = turning_state();
    state.source_plant_initialized = 1;
    state.source_suspension_history_initialized = 1;
    for (int wheel = 0; wheel < 4; ++wheel)
      state.source_plant_internal_state[
          MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] = state.u / 0.059f;

    const MpcModelControl_t control{0.0f, command};
    MpcStageResult_t nominal{};
    MpcStageLinearization_t jacobian{};
    ASSERT_TRUE(mpc_vehicle_model_step_with_jacobian(
        &state, &control, 0.025f, 0.0f, &nominal, &jacobian));

    /* The active [0, 1] command bounds restrict the QP to one feasible
     * direction at either endpoint; they do not erase the queue derivative. */
    EXPECT_FLOAT_EQ(
        jacobian.B[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2][1], 1.0f);
  }
}

TEST(VehicleModelSourceRollout,
     ReprojectedPhysicsCacheMatchesFullPlantForWholeHorizon)
{
  MpcModelState_t pass_a = turning_state();
  MpcModelState_t pass_b = pass_a;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&pass_a));
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&pass_b));

  for (int k = 0; k < PREDICTION_HORIZON; ++k) {
    const float phase = 0.17f * static_cast<float>(k);
    const MpcModelControl_t control{
        0.22f + 0.06f * std::sin(phase),
        0.28f + 0.08f * std::cos(phase)};
    const float curvature_a = 0.12f + 0.07f * std::sin(0.11f * k);
    const float curvature_b = -0.04f + 0.09f * std::cos(0.13f * k);
    const MpcStageResult_t pass_a_step = mpc_vehicle_model_step(
        &pass_a, &control, PREDICTION_DT_SECONDS, curvature_a);
    const MpcStageResult_t full_pass_b_step = mpc_vehicle_model_step(
        &pass_b, &control, PREDICTION_DT_SECONDS, curvature_b);
    MpcStageResult_t cached_pass_b_step{};
    ASSERT_TRUE(pass_a_step.valid);
    ASSERT_TRUE(full_pass_b_step.valid);
    ASSERT_TRUE(vehicle_model_reproject_tracking(
        &pass_b, &pass_a_step, curvature_b, &cached_pass_b_step));

    const auto &full = full_pass_b_step.next;
    const auto &cached = cached_pass_b_step.next;
    EXPECT_NEAR(cached.e_y, full.e_y, 1.0e-7f);
    EXPECT_NEAR(cached.e_psi, full.e_psi, 1.0e-7f);
    EXPECT_FLOAT_EQ(cached.u, full.u);
    EXPECT_FLOAT_EQ(cached.v, full.v);
    EXPECT_FLOAT_EQ(cached.r, full.r);
    EXPECT_FLOAT_EQ(cached.delayed_steering_command_1,
                    full.delayed_steering_command_1);
    EXPECT_FLOAT_EQ(cached.delayed_steering_command_2,
                    full.delayed_steering_command_2);
    EXPECT_FLOAT_EQ(cached.actual_steering_angle,
                    full.actual_steering_angle);
    EXPECT_FLOAT_EQ(cached.delayed_throttle_command_1,
                    full.delayed_throttle_command_1);
    EXPECT_FLOAT_EQ(cached.delayed_throttle_command_2,
                    full.delayed_throttle_command_2);
    EXPECT_EQ(cached.source_plant_initialized,
              full.source_plant_initialized);
    EXPECT_EQ(cached.source_suspension_history_initialized,
              full.source_suspension_history_initialized);
    EXPECT_EQ(cached.source_imu_state_initialized,
              full.source_imu_state_initialized);
    for (int i = 0; i < MPC_PLANT_INTERNAL_STATE_COUNT; ++i)
      EXPECT_DOUBLE_EQ(cached.source_plant_internal_state[i],
                       full.source_plant_internal_state[i]);
    EXPECT_NEAR(cached_pass_b_step.delta_s_m,
                full_pass_b_step.delta_s_m, 1.0e-7f);
    EXPECT_EQ(cached_pass_b_step.branch_flags,
              full_pass_b_step.branch_flags);
    EXPECT_FLOAT_EQ(cached_pass_b_step.body_forward_delta_m,
                    full_pass_b_step.body_forward_delta_m);
    EXPECT_FLOAT_EQ(cached_pass_b_step.body_lateral_delta_m,
                    full_pass_b_step.body_lateral_delta_m);
    EXPECT_FLOAT_EQ(cached_pass_b_step.body_yaw_delta_rad,
                    full_pass_b_step.body_yaw_delta_rad);
    for (int wheel = 0; wheel < 4; ++wheel) {
      EXPECT_DOUBLE_EQ(cached_pass_b_step.output.longitudinal_slip[wheel],
                       full_pass_b_step.output.longitudinal_slip[wheel]);
      EXPECT_DOUBLE_EQ(cached_pass_b_step.output.lateral_slip[wheel],
                       full_pass_b_step.output.lateral_slip[wheel]);
      EXPECT_DOUBLE_EQ(cached_pass_b_step.output.normal_load_n[wheel],
                       full_pass_b_step.output.normal_load_n[wheel]);
    }

    pass_a = pass_a_step.next;
    pass_b = cached_pass_b_step.next;
  }
}

TEST(VehicleModelSourceRollout, RTIJacobianCarriesAllWheelSpinStates)
{
  MpcModelState_t state = turning_state();
  state.u = 3.5f;
  state.v = 0.0f;
  state.r = 0.8f;
  state.actual_steering_angle = 0.32f;
  state.delayed_throttle_command_1 = 0.55f;
  state.delayed_throttle_command_2 = 0.55f;
  state.source_plant_initialized = 1;
  state.source_suspension_history_initialized = 1;
  state.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT] =
      0.00464804098;
  for (int wheel = 0; wheel < 4; ++wheel) {
    state.source_plant_internal_state[
        MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] =
        state.u / 0.059;
  }
  state.source_plant_internal_state[MPC_PLANT_STATE_WHEEL_OMEGA_RL] += 24.0;
  state.source_plant_internal_state[MPC_PLANT_STATE_WHEEL_OMEGA_RR] += 18.0;

  const MpcModelControl_t control{0.0f, 0.0f};
  constexpr float dt = 0.025f;
  MpcStageResult_t nominal{};
  MpcStageLinearization_t jacobian{};
  ASSERT_TRUE(mpc_vehicle_model_step_with_jacobian(
      &state, &control, dt, 0.12f, &nominal, &jacobian));

  for (int wheel = 0; wheel < 4; ++wheel) {
    MpcModelState_t perturbed = state;
    constexpr float epsilon = 1.0e-2f;
    perturbed.source_plant_internal_state[
        MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] +=
        epsilon;
    const MpcStageResult_t changed = mpc_vehicle_model_step(
        &perturbed, &control, dt, 0.12f);
    ASSERT_TRUE(changed.valid);
    const float expected = static_cast<float>(
        (changed.next.source_plant_internal_state[
             MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] -
         nominal.next.source_plant_internal_state[
             MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel]) /
        epsilon);
    const int wheel_state = MPC_STATE_IDX_WHEEL_OMEGA_FL + wheel;
    EXPECT_NEAR(jacobian.A[wheel_state][wheel_state], expected, 2.0e-3f)
        << "wheel index " << wheel;
    EXPECT_TRUE(std::isfinite(jacobian.A[wheel_state][wheel_state]));
  }
}

TEST(VehicleModelSourceRollout, KeepsTheRealTimeRTIStateCompact)
{
  EXPECT_EQ(MPC_MODEL_NX, 24);
  EXPECT_EQ(MPC_RTI_NX, 23);
  EXPECT_EQ(RICCATI_HORIZON,
      PREDICTION_HORIZON - MPC_COMMAND_DELAY_STEPS);
  EXPECT_EQ(MPC_PLANT_INTERNAL_STATE_COUNT, 22);
  EXPECT_EQ(MPC_OPT_PHYSICAL_NX, 20);
  EXPECT_EQ(UNITY_SOURCE_PLANT_STATE_SIZE, 26);

  MpcModelState_t state = turning_state();
  const MpcModelControl_t control{0.0f, 0.0f};
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);
  ASSERT_TRUE(stage.next.source_plant_initialized);
  EXPECT_NE(stage.next.source_plant_internal_state[MPC_PLANT_STATE_ROLL], 0.0);
  EXPECT_NE(stage.next.source_plant_internal_state[MPC_PLANT_STATE_PITCH], 0.0);
  EXPECT_NE(stage.next.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT],
            state.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT]);
  EXPECT_TRUE(stage.next.source_suspension_history_initialized);
}

TEST(MpcRtiCorridorSlack, MatchesTheFootprintAndStageSpecificMargins)
{
  MpcRtiConfiguration_t config{};
  config.corridor_margin_m = 0.05f;
  config.first_prediction_corridor_margin_m = 0.03f;
  config.nonlinear_corridor_tolerance_m = 0.005f;
  MpcModelState_t state{};
  state.e_y = 0.10f;
  state.e_psi = 0.25f;
  constexpr float left_bound = 0.65f;
  constexpr float right_bound = 0.72f;

  const float left_extent = 0.5f * MPC_CAR_WIDTH_M *
          std::fabs(std::cos(state.e_psi)) +
      MPC_REAR_AXLE_TO_FRONT_BUMPER_M *
          std::fabs(std::sin(state.e_psi));
  const float right_extent = 0.5f * MPC_CAR_WIDTH_M *
          std::fabs(std::cos(state.e_psi)) +
      MPC_REAR_OVERHANG_M * std::fabs(std::sin(state.e_psi));
  float lower_stage1 = 0.0f;
  float upper_stage1 = 0.0f;
  ASSERT_TRUE(mpc_rti_compute_corridor_slack(&state, left_bound,
      right_bound, &config, 1, &lower_stage1, &upper_stage1));
  EXPECT_NEAR(lower_stage1,
      state.e_y - (0.03f + right_extent - right_bound - 0.005f), 1.0e-6f);
  EXPECT_NEAR(upper_stage1,
      left_bound - (0.03f + left_extent) + 0.005f - state.e_y, 1.0e-6f);

  float lower_stage2 = 0.0f;
  float upper_stage2 = 0.0f;
  ASSERT_TRUE(mpc_rti_compute_corridor_slack(&state, left_bound,
      right_bound, &config, 2, &lower_stage2, &upper_stage2));
  EXPECT_NEAR(lower_stage1 - lower_stage2, 0.02f, 1.0e-6f);
  EXPECT_NEAR(upper_stage1 - upper_stage2, 0.02f, 1.0e-6f);
}

TEST(MpcRtiCorridorSlack, MatchesIndependentTransformedBodyCornersOnBothSides)
{
  MpcRtiConfiguration_t config{};
  config.corridor_margin_m = 0.05f;
  config.first_prediction_corridor_margin_m = 0.05f;
  config.nonlinear_corridor_tolerance_m = 0.0f;
  for (const float heading : {-2.8f, -1.2f, -0.25f, 0.0f, 0.25f, 1.2f, 2.8f}) {
    MpcModelState_t state{};
    state.e_y = 0.1f;
    state.e_psi = heading;
    double minimum_y = INFINITY, maximum_y = -INFINITY;
    for (const double x : {-double(MPC_REAR_OVERHANG_M),
                          double(MPC_REAR_AXLE_TO_FRONT_BUMPER_M)}) {
      for (const double y : {-0.5 * MPC_CAR_WIDTH_M, 0.5 * MPC_CAR_WIDTH_M}) {
        const double corner_y = x * std::sin(heading) + y * std::cos(heading);
        minimum_y = std::min(minimum_y, corner_y);
        maximum_y = std::max(maximum_y, corner_y);
      }
    }
    float lower, upper;
    ASSERT_TRUE(mpc_rti_compute_corridor_slack(&state, 1.0f, 1.0f,
      &config, 1, &lower, &upper));
    EXPECT_NEAR(lower, 1.0 + state.e_y + minimum_y - 0.05, 1.0e-6) << heading;
    EXPECT_NEAR(upper, 1.0 - state.e_y - maximum_y - 0.05, 1.0e-6) << heading;
  }
}

}  // namespace

// World-map evaluation must see precisely the same interpolated path poses as
// the exact source-model tracking update.  This also protects the performance
// optimization which reuses reference interpolations rather than sampling
// the whole trajectory twice more per swept source step.
TEST(MpcCandidate, WorldSweptFootprintReusesIdenticalInterpolatedPoses)
{
  MpcTrajectorySample_t trajectory[3] = {
      {0.0,0.0,0.0,0.0,0.0,2.0,100.0,100.0,0.0},
      {10.0,10.0,0.0,0.0,0.0,2.0,100.0,100.0,0.0},
      {20.0,20.0,0.0,0.0,0.0,2.0,100.0,100.0,0.0}};
  size_t count=3;
  double lap_length=0.0;
  ASSERT_TRUE(mpc_trajectory_prepare(trajectory,&count,&lap_length));

  struct SweepRecord {
    struct Pose {double x,y,yaw;};
    Pose start[4]{}, end[4]{};
    float required[4]{};
    int calls=0;
    int reject_stage=0;
  } log;
  auto check=[](void* ptr,double x0,double y0,double h0,
                double x1,double y1,double h1,
                double margin,float* slack) -> int {
    auto &data=*static_cast<SweepRecord*>(ptr);
    const int stage=++data.calls;
    if(stage>4)return 0;
    data.start[stage-1]={x0,y0,h0};
    data.end[stage-1]={x1,y1,h1};
    data.required[stage-1]=static_cast<float>(margin);
    *slack=(data.reject_stage==stage) ? -0.01f : 0.4f;
    return data.reject_stage==stage ? 0 : 1;
  };

  MpcRtiConfiguration_t config{};
  config.max_speed_mps=16.f;
  config.active_speed_ceiling_mps=16.f;
  config.max_steering_rad=SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m=.05f;
  config.first_prediction_corridor_margin_m=.03f;
  config.world_sweep_check=check;
  config.world_sweep_context=&log;
  config.world_map_rejection_only=0;

  MpcRtiState_t initial{};
  initial.plant.u=2.f;
  initial.plant.source_plant_initialized=1;
  initial.plant.delayed_throttle_command_1=.01f;
  initial.plant.delayed_throttle_command_2=.01f;
  MpcModelControl_t controls[PREDICTION_HORIZON]{};
  for(auto &control:controls)control.throttle_command=.01f;

  MpcRtiState_t states[PREDICTION_HORIZON+1]{};
  double progress[PREDICTION_HORIZON+1]{};
  int failed=-1;
  ASSERT_EQ(mpc_rti_rollout_candidate(&initial,0.,controls,trajectory,count,
       lap_length,nullptr,nullptr,nullptr,4,PREDICTION_DT_SECONDS,&config,
       states,progress,&failed),MPC_RTI_ROLLOUT_OK);
  EXPECT_EQ(log.calls,4);
  EXPECT_EQ(failed,-1);
  for(int k=0;k<4;++k) {
    for(int side=0;side<2;++side){
      const int at=k+side;
      MpcTrajectorySample_t reference{};
      ASSERT_TRUE(mpc_trajectory_sample(trajectory,count,lap_length,
          progress[at],&reference));
      const auto &state=states[at].plant;
      const double expected_x=reference.x-
          std::sin(reference.heading)*state.e_y;
      const double expected_y=reference.y+
          std::cos(reference.heading)*state.e_y;
      const double expected_heading=reference.heading+state.e_psi;
      const auto &actual=side ? log.end[k] : log.start[k];
      EXPECT_NEAR(actual.x,expected_x,1.e-11);
      EXPECT_NEAR(actual.y,expected_y,1.e-11);
      EXPECT_NEAR(actual.yaw,expected_heading,1.e-11);
    }
    EXPECT_FLOAT_EQ(log.required[k],k==0 ? .03f : .05f);
  }
  // An obstacle at stage 3 remains an unconditional hard rejection.  The
  // reference-cache optimization may not change the safety decision.
  log.calls=0;
  log.reject_stage=3;
  failed=-1;
  EXPECT_EQ(mpc_rti_rollout_candidate(&initial,0.,controls,trajectory,count,
      lap_length,nullptr,nullptr,nullptr,4,PREDICTION_DT_SECONDS,&config,
      states,progress,&failed),MPC_RTI_ROLLOUT_CORRIDOR);
  EXPECT_EQ(failed,2);
  EXPECT_EQ(log.calls,3);
}

// R1/R2 and the existing controller recovery now use this same causal
// terminal-extension probe. The three extra source-plant stages MUST obey
// the exact same physical footprint validator (including map authority).
TEST(MpcRtiTerminalViability, ThreeStepExtensionRetainsPlantAndCollisionGate)
{
  MpcTrajectorySample_t track[3] = {
      {0.,0.,0.,0.,0.,2.,4.,4.,0.},
      {10.,10.,0.,0.,0.,2.,4.,4.,0.},
      {20.,20.,0.,0.,0.,2.,4.,4.,0.}};
  size_t size=3;
  double lap=0.;
  ASSERT_TRUE(mpc_trajectory_prepare(track,&size,&lap));
  MpcRtiCycleConfiguration_t config{};
  config.model.max_speed_mps=16.f;
  config.model.active_speed_ceiling_mps=16.f;
  config.model.max_steering_rad=SOURCE_MAX_STEERING_RAD;
  config.model.corridor_margin_m=.05f;
  config.model.first_prediction_corridor_margin_m=.05f;
  MpcRtiState_t initial{};
  initial.plant.u=1.f;
  initial.plant.source_plant_initialized=1;
  initial.plant.delayed_throttle_command_1=.03f;
  initial.plant.delayed_throttle_command_2=.03f;
  MpcModelControl_t commands[PREDICTION_HORIZON]{};
  for(auto &u:commands)u.throttle_command=.03f;
  MpcRtiNominal_t plan{};
  int failed=-1;
  ASSERT_EQ(mpc_rti_validate_candidate_plan(
      &initial,0.,commands,track,size,lap,PREDICTION_HORIZON,
      PREDICTION_DT_SECONDS,&config,&plan,&failed),
      MPC_RTI_ROLLOUT_OK);
  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(mpc_rti_terminal_extension_probe(&plan,
      track,size,lap,PREDICTION_DT_SECONDS,&config),1);
  // Both nominal terminal commands must fail if authoritative world
  // geometry rejects the continuation; no skipped physical safety.
  config.model.world_sweep_check=[](void*,double,double,double,
      double,double,double,double,float* slack)->int {
    *slack=-.01f; return 0;
  };
  EXPECT_EQ(mpc_rti_terminal_extension_probe(&plan,
      track,size,lap,PREDICTION_DT_SECONDS,&config),0);
  plan.valid=0;
  EXPECT_EQ(mpc_rti_terminal_extension_probe(&plan,
      track,size,lap,PREDICTION_DT_SECONDS,&config),0);
}

TEST(MpcRobustContour, SmoothOffsetAnticipatesNarrowWallWithoutChangingSourcePhysics)
{
    std::vector<MpcTrajectorySample_t> track(100);
    for (std::size_t i=0;i<track.size();++i) {
        auto &p=track[i];
        p.s=.1*i;
        const double a=2.*3.14159265358979323846*i/track.size();
        p.x=5.*std::cos(a);p.y=5.*std::sin(a);
        p.heading=a+1.5707963267948966;
        p.speed=6.;p.left_bound=.8;p.right_bound=.8;
        p.source_witness_valid=1;
        p.source_throttle=.2;
    }
    for(int i=40;i<=50;++i)
        track[i].left_bound=.20;
    const double lap_length=10.;
    const auto untouched=track;
    const auto result=f1tenth_mpc::precompute_robust_contour(
        track,lap_length,.05,.08,.10);
    EXPECT_GT(result.affected_knots,0u);
    EXPECT_LE(result.max_abs_offset_m,.1000001);
    EXPECT_LT(track[38].robust_contour_lateral_offset_m,0.)
        << "Preview must move inward before the wall narrows.";
    EXPECT_LT(track[45].robust_contour_lateral_offset_m,0.);
    EXPECT_NEAR(track[20].robust_contour_lateral_offset_m,0.,.005);
    EXPECT_LT(result.max_neighbor_slope,.5)
        << "Reference offsets must have smooth spatial transitions.";
    for(std::size_t i=0;i<track.size();++i) {
        EXPECT_DOUBLE_EQ(track[i].s,untouched[i].s);
        EXPECT_DOUBLE_EQ(track[i].x,untouched[i].x);
        EXPECT_DOUBLE_EQ(track[i].heading,untouched[i].heading);
        EXPECT_DOUBLE_EQ(track[i].speed,untouched[i].speed);
        EXPECT_DOUBLE_EQ(track[i].source_throttle,untouched[i].source_throttle);
        EXPECT_DOUBLE_EQ(track[i].left_bound,untouched[i].left_bound);
        EXPECT_DOUBLE_EQ(track[i].right_bound,untouched[i].right_bound);
    }
    MpcTrajectorySample_t sample{};
    ASSERT_TRUE(mpc_trajectory_sample(track.data(),track.size(),lap_length,
                                      4.25,&sample));
    EXPECT_NEAR(sample.robust_contour_lateral_offset_m,
        .5*(track[42].robust_contour_lateral_offset_m+
            track[43].robust_contour_lateral_offset_m),1.e-6);
    // Normal production profile must retain a bitwise zero offset.
    const auto inactive=f1tenth_mpc::precompute_robust_contour(
        track,lap_length,.05,0.,.10);
    EXPECT_EQ(inactive.affected_knots,0u);
}

TEST(MpcRtiSteeringVariation, PseudoHuberQPMatchesExactObjectiveGradient)
{
  constexpr double lambda=80.,eps=.02;
  for (double z0 : {-.4,-.08,-.02,0.,.02,.08,.4}) {
    const double h=1.e-5;
    const double derivative=(mpc_steering_variation_exact_cost(
        lambda,eps,z0+h)-mpc_steering_variation_exact_cost(
        lambda,eps,z0-h))/(2*h);
    const double tangent=2*mpc_steering_variation_quadratic_weight(
        lambda,eps,(float)z0)*z0;
    EXPECT_NEAR(derivative,tangent,2.e-3);
  }
  EXPECT_GT(mpc_steering_variation_quadratic_weight(
      lambda,eps,0.),mpc_steering_variation_quadratic_weight(
      lambda,eps,.2f));
  EXPECT_DOUBLE_EQ(mpc_steering_variation_exact_cost(
      0.,eps,.2),0.);
}

// The physical-control coordinate change must preserve BOTH the exact
// one-step plant affine model and the full quadratic objective. This is not
// constrained around the previous physical command, not the SQP seed.
// constrained; a 180-degree steering reversal must take multiple stages.
TEST(MpcRtiCommandIncrements, ExactQuadraticAndDynamicsCoordinateChange)
{
  MpcRtiProblem_t before{},after{};
  MpcRtiConfiguration_t config{};
  config.command_increment_qp_enabled=1;
  config.max_steering_rad=SOURCE_MAX_STEERING_RAD;
  config.max_steering_command_step_rad=.08f;
  config.max_throttle_command_step=.20f;
  before.horizon=RICCATI_HORIZON;
  const int prev[2]={MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND,
                     MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND};
  for(int k=0;k<RICCATI_HORIZON;++k){
    auto &s=before.steps[k];
    for (int i=0;i<MPC_RTI_NX;++i){s.x_lb[i]=-BIG_BOUND;s.x_ub[i]=BIG_BOUND;}
    s.R_diag[0]=2001.f;s.R_diag[1]=50.1f;
    s.N[prev[0]][0]=-2000.f;
    s.N[prev[1]][1]=-50.f;
    // Explicit mixed steering/throttle and physical-state/input terms.
    s.R_cross=0.71f;
    s.N[MPC_OPT_IDX_EY][0]=0.33f;
    s.N[MPC_OPT_IDX_U][1]=-0.09f;
    s.Q_offdiag[MPC_OPT_IDX_EY][MPC_OPT_IDX_U]=0.13f;
    s.Q_offdiag[MPC_OPT_IDX_U][MPC_OPT_IDX_EY]=0.13f;
    s.Q_diag[prev[0]]=2000.f;
    s.Q_diag[prev[1]]=50.f;
    s.q[prev[0]]=8.1f;
    s.q[prev[1]]=3.5f;
    s.r[0]=-9.7f;
    s.r[1]=-3.8f;
    s.A[MPC_OPT_IDX_EY][MPC_OPT_IDX_EY]=1.f;
    s.A[MPC_OPT_IDX_EY][prev[0]]=0.f;
    s.B[MPC_OPT_IDX_EY][0]=.45f;
    s.B[MPC_OPT_IDX_EY][1]=-.17f;
    s.B[prev[0]][0]=1.f;
    s.B[prev[1]][1]=1.f;
  }
  after=before;
  ASSERT_TRUE(mpc_rti_transform_qp_to_command_increments(&after,&config,PREDICTION_DT_SECONDS));
  constexpr float x_steer=.17f,x_throttle=.4f;
  const float cmd[2]={-.28f,.7f};
  const float dcmd[2]={cmd[0]-x_steer,cmd[1]-x_throttle};
  // The original stage cost depends on previous-command state and the
  // absolute command; transformed cost must be identical for every point.
  float x[RICCATI_MAX_NX]{},u[RICCATI_MAX_NU]={cmd[0],cmd[1]};
  float v[RICCATI_MAX_NU]={dcmd[0],dcmd[1]};
  x[prev[0]]=x_steer;x[prev[1]]=x_throttle;x[MPC_OPT_IDX_EY]=.12f;
  const auto &s=before.steps[4],&t=after.steps[4];
  EXPECT_NEAR(mpc_rti_matrix_stage_cost(&s,x,u),
              mpc_rti_matrix_stage_cost(&t,x,v),.001f);
  for (int i : {int(MPC_OPT_IDX_EY),prev[0],prev[1]}) {
    float old_next=s.d[i],new_next=t.d[i];
    for(int j=0;j<MPC_RTI_NX;++j){
      old_next+=s.A[i][j]*x[j];
      new_next+=t.A[i][j]*x[j];
    }
    for(int j=0;j<MPC_RTI_NU;++j){
      old_next+=s.B[i][j]*u[j];
      new_next+=t.B[i][j]*v[j];
    }
    EXPECT_NEAR(old_next,new_next,1.e-6f)<<i;
  }
  EXPECT_FLOAT_EQ(t.u_lb[0],-.08f);
  EXPECT_FLOAT_EQ(t.u_ub[0],.08f);
  EXPECT_FLOAT_EQ(t.u_lb[1],-.20f);
  EXPECT_FLOAT_EQ(t.u_ub[1],.20f);
  EXPECT_NEAR(t.x_lb[prev[0]],-SOURCE_MAX_STEERING_RAD,1.e-6f);
  EXPECT_NEAR(t.x_ub[prev[0]], SOURCE_MAX_STEERING_RAD,1.e-6f);
  EXPECT_NEAR(after.terminal_x_lb[prev[1]],MPC_THROTTLE_MIN_DRIVE,1.e-6f);
  EXPECT_NEAR(after.terminal_x_ub[prev[1]],MPC_THROTTLE_MAX,1.e-6f);
  // The current x0 has a fixed command that may begin at 0.0 full brake.
  // Do not reject it by imposing future-actuator bounds on solver stage 0.
  EXPECT_LT(after.steps[0].x_lb[prev[1]],0.f);
}

TEST(MpcRtiCommandIncrements, MultistageFeedbackPredictionUsesIncrementCoordinates)
{
  // The feedback prediction must propagate physical absolute commands
  // through the SAME affine LTV model as a delta-input QP. Accidentally
  // inserting absolute commands into B after A[p] += B double-counts them.
  MpcRtiProblem_t before{}, after{};
  MpcRtiConfiguration_t config{};
  config.command_increment_qp_enabled = 1;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.max_steering_command_step_rad = .10f;
  config.max_throttle_command_step = .25f;
  before.horizon = RICCATI_HORIZON;
  for (auto &stage : before.steps) {
    stage.A[MPC_OPT_IDX_EY][MPC_OPT_IDX_EY] = 1.f;
    stage.B[MPC_OPT_IDX_EY][0] = .45f;
    stage.B[MPC_OPT_IDX_EY][1] = -.17f;
    stage.B[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND][0] = 1.f;
    stage.B[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND][1] = 1.f;
  }
  after = before;
  ASSERT_TRUE(mpc_rti_transform_qp_to_command_increments(&after, &config,PREDICTION_DT_SECONDS));
  float direct[MPC_RTI_NX]{};
  float increments[MPC_RTI_NX]{};
  direct[MPC_OPT_IDX_EY] = increments[MPC_OPT_IDX_EY] = .12f;
  direct[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] =
      increments[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] = .17f;
  direct[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] =
      increments[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] = .4f;
  for (int k = 0; k < 8; ++k) {
    const float steering = .15f - .025f*k;
    const float throttle = .42f + .015f*k;
    const float previous_s =
        increments[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND];
    const float previous_t =
        increments[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
    const float decisions[2] = {
        mpc_rti_qp_decision_from_absolute_command(steering, previous_s, 1),
        mpc_rti_qp_decision_from_absolute_command(throttle, previous_t, 1)};
    EXPECT_NEAR(mpc_rti_absolute_command_from_qp_decision(
        decisions[0], previous_s, 1), steering, 1.e-6f);
    EXPECT_NEAR(mpc_rti_absolute_command_from_qp_decision(
        decisions[1], previous_t, 1), throttle, 1.e-6f);
    float direct_next[MPC_RTI_NX]{}, increment_next[MPC_RTI_NX]{};
    for (int i = 0; i < MPC_RTI_NX; ++i) {
      double lhs = before.steps[k].d[i] +
          before.steps[k].B[i][0]*steering +
          before.steps[k].B[i][1]*throttle;
      double rhs = after.steps[k].d[i] +
          after.steps[k].B[i][0]*decisions[0] +
          after.steps[k].B[i][1]*decisions[1];
      for (int j = 0; j < MPC_RTI_NX; ++j) {
        lhs += before.steps[k].A[i][j]*direct[j];
        rhs += after.steps[k].A[i][j]*increments[j];
      }
      direct_next[i] = (float)lhs;
      increment_next[i] = (float)rhs;
      EXPECT_NEAR(lhs, rhs, 1.e-6) << "stage=" << k << " state=" << i;
    }
    std::copy_n(direct_next, MPC_RTI_NX, direct);
    std::copy_n(increment_next, MPC_RTI_NX, increments);
  }
}


// Source-Lagrangian Hessian-vector primitive: directionally differentiate
// the ORIGINAL native 1 ms source transition (all wheel/suspension states).
// A meaningful second derivative is branch-local; the function must reject
// physical-bound crossings instead of silently fitting a clipped control.
TEST(MpcSourceSecondOrder, ExactSourceDirectionalSymmetryAndLocalAdAgreement)
{
  MpcModelState_t state{};
  state.u=2.4f;
  state.v=.04f;
  state.r=.08f;
  state.actual_steering_angle=.04f;
  state.e_y=.01f;
  state.e_psi=.02f;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&state));
  const MpcModelControl_t control{.065f,.18f};
  double direction[UNITY_SOURCE_PLANT_STATE_SIZE]{};
  direction[UNITY_SOURCE_U]=.21;
  direction[UNITY_SOURCE_V]=-.03;
  direction[UNITY_SOURCE_YAW_RATE]=.07;
  direction[UNITY_SOURCE_PHYSICAL_STEERING]=-.04;
  direction[UNITY_SOURCE_WHEEL_OMEGA_FL]=.06;
  direction[UNITY_SOURCE_BODY_HEIGHT]=.001;
  const double direction_input[2]={.04,.055};
  MpcSourceDirectionalSecondOrder_t forward{},reversed{};
  ASSERT_TRUE(vehicle_model_full_source_interval_directional_second_order(
      &state,&control,.025f,direction,direction_input,.01,&forward));
  double negative_direction[UNITY_SOURCE_PLANT_STATE_SIZE]{};
  for(int i=0;i<UNITY_SOURCE_PLANT_STATE_SIZE;++i)
      negative_direction[i]=-direction[i];
  const double negative_input[2]={
      -direction_input[0],-direction_input[1]};
  ASSERT_TRUE(vehicle_model_full_source_interval_directional_second_order(
      &state,&control,.025f,negative_direction,negative_input,.01,&reversed));
  for(int i=0;i<UNITY_SOURCE_PLANT_STATE_SIZE;++i) {
      EXPECT_NEAR(forward.first_source[i],-reversed.first_source[i],1e-8);
      EXPECT_NEAR(forward.second_source[i],reversed.second_source[i],1e-8);
  }
  for(int i=0;i<3;++i) {
      EXPECT_NEAR(forward.first_body[i],-reversed.first_body[i],1e-8);
      EXPECT_NEAR(forward.second_body[i],reversed.second_body[i],1e-8);
  }
  // Compare the exact finite-difference directional first derivative
  // against the existing 20-lane FULL source automatic differentiation.
  double seeds[26][20]{},control_seeds[2][20]{};
  double source_jacobian_direction[26][20]{};
  double body_jacobian_direction[3][20]{};
  double slip_jacobian_direction[4][20]{};
  for(int i=0;i<26;++i)seeds[i][0]=direction[i];
  control_seeds[0][0]=direction_input[0];
  control_seeds[1][0]=direction_input[1];
  ASSERT_TRUE(vehicle_model_full_source_interval_sensitivities(
      &state,&control,.025f,seeds,control_seeds,
      source_jacobian_direction,body_jacobian_direction,
      slip_jacobian_direction));
  for(int i:{UNITY_SOURCE_U,UNITY_SOURCE_V,UNITY_SOURCE_YAW_RATE,
             UNITY_SOURCE_PHYSICAL_STEERING}) {
      const double exact_fd=forward.first_source[i];
      const double local_ad=source_jacobian_direction[i][0];
      EXPECT_NEAR(exact_fd,local_ad,
          .05*std::max(1.,std::abs(exact_fd)))
          <<"source coordinate "<<i;
  }
}

TEST(MpcSourceSecondOrder, RejectsPerturbationsOutsidePhysicalDriveBounds)
{
  MpcModelState_t state{};
  state.u=2.f;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&state));
  double direction[UNITY_SOURCE_PLANT_STATE_SIZE]{};
  const double steering_delta[2]={.02,0.};
  const double throttle_delta[2]={0.,.1};
  MpcSourceDirectionalSecondOrder_t curvature{};
  MpcModelControl_t at_steer_limit{.5235987756f,.2f};
  EXPECT_FALSE(vehicle_model_full_source_interval_directional_second_order(
      &state,&at_steer_limit,.025f,direction,steering_delta,.01,
      &curvature));
  MpcModelControl_t at_positive_drive_min{.02f,MPC_THROTTLE_MIN_DRIVE};
  EXPECT_FALSE(vehicle_model_full_source_interval_directional_second_order(
      &state,&at_positive_drive_min,.025f,direction,throttle_delta,.01,
      &curvature));
  const double zero_delta[2]={0.,0.};
  MpcModelControl_t inside{.02f,.2f};
  ASSERT_TRUE(vehicle_model_full_source_interval_directional_second_order(
      &state,&inside,.025f,direction,zero_delta,.01,&curvature));
  for(int i=0;i<UNITY_SOURCE_PLANT_STATE_SIZE;++i) {
      EXPECT_DOUBLE_EQ(curvature.first_source[i],0.);
      EXPECT_DOUBLE_EQ(curvature.second_source[i],0.);
  }
}


TEST(MpcSourceSecondOrder, ExactBranchLocalCurvatureMatchesIndependentSmoothSource)
{
  MpcModelState_t state{};
  state.u=2.4f;state.v=.04f;state.r=.08f;
  state.actual_steering_angle=.04f;
  state.e_y=.01f;state.e_psi=.02f;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&state));
  const MpcModelControl_t control{.065f,.18f};
  double direction[UNITY_SOURCE_PLANT_STATE_SIZE]{};
  direction[UNITY_SOURCE_U]=.21;
  direction[UNITY_SOURCE_V]=-.03;
  direction[UNITY_SOURCE_YAW_RATE]=.07;
  direction[UNITY_SOURCE_PHYSICAL_STEERING]=-.04;
  direction[UNITY_SOURCE_WHEEL_OMEGA_FL]=.06;
  direction[UNITY_SOURCE_BODY_HEIGHT]=.001;
  const double du[2]={.04,.055};
  MpcSourceDirectionalSecondOrder_t analytic{},independent{};
  ASSERT_TRUE(vehicle_model_full_source_interval_analytic_directional_second_order(
      &state,&control,.025f,direction,du,&analytic));
  // Analytic selected-branch second derivatives must agree with centered
  // exact 1ms source evaluations as h->0 where the source is differentiable.
  ASSERT_TRUE(vehicle_model_full_source_interval_directional_second_order(
      &state,&control,.025f,direction,du,1.e-4,&independent));
  double seeds[26][20]{},dcontrol[2][20]{};
  double ad[26][20]{},body[3][20]{},slip[4][20]{};
  for(int i=0;i<26;++i)seeds[i][0]=direction[i];
  dcontrol[0][0]=du[0];dcontrol[1][0]=du[1];
  ASSERT_TRUE(vehicle_model_full_source_interval_sensitivities(
      &state,&control,.025f,seeds,dcontrol,ad,body,slip));
  for(int i : {UNITY_SOURCE_U,UNITY_SOURCE_V,UNITY_SOURCE_YAW_RATE,
               UNITY_SOURCE_PHYSICAL_STEERING}) {
    const double a=analytic.first_source[i];
    const double d=independent.first_source[i];
    EXPECT_NEAR(a,ad[i][0],1.e-3*std::max(1.,std::abs(a)));
    EXPECT_NEAR(a,d,.03*std::max(1.,std::abs(a)))
        <<"first source state "<<i;
    EXPECT_NEAR(analytic.second_source[i],
                independent.second_source[i],
                .15*std::max(1.,std::abs(analytic.second_source[i])))
        <<"second source state "<<i;
  }
}

TEST(MpcSourceSecondOrder, OneMillisecondSourceSplitReproducesFullInterval)
{
  MpcModelState_t state{};
  state.u=3.22f;state.v=.06f;state.r=-1.47f;
  state.actual_steering_angle=-.22f;
  state.e_y=.02f;state.e_psi=.11f;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&state));
  UnitySourcePlantInputC input{};
  input.state[UNITY_SOURCE_U]=state.u;
  input.state[UNITY_SOURCE_V]=state.v+.15532*state.r;
  input.state[UNITY_SOURCE_YAW_RATE]=state.r;
  input.state[UNITY_SOURCE_PHYSICAL_STEERING]=
      state.actual_steering_angle;
  for(int i=0;i<MPC_PLANT_INTERNAL_STATE_COUNT;++i) {
    const int index=i<10?3+i:i+4;
    input.state[index]=state.source_plant_internal_state[i];
  }
  input.steering_target_rad=-.29;
  input.throttle_normalized=.23;
  input.suspension_history_initialized=
      state.source_suspension_history_initialized;
  UnitySourcePlantOutputC whole{};
  ASSERT_TRUE(unity_source_plant_step_c(&input,.025,&whole));
  UnitySourcePlantOutputC last{};
  for(int microstep=0;microstep<25;++microstep) {
    ASSERT_TRUE(unity_source_plant_step_c(&input,.001,&last));
    std::copy_n(last.state,UNITY_SOURCE_PLANT_STATE_SIZE,input.state);
    std::copy_n(last.wheel_rotation_angle_rad,4,
                input.wheel_rotation_angle_rad);
    input.suspension_history_initialized=
        last.suspension_history_initialized;
  }
  for(int i=0;i<UNITY_SOURCE_PLANT_STATE_SIZE;++i)
      EXPECT_DOUBLE_EQ(last.state[i],whole.state[i])
          <<"physical source coordinate "<<i;
  for(int i=0;i<4;++i)
      EXPECT_DOUBLE_EQ(last.wheel_rotation_angle_rad[i],
                       whole.wheel_rotation_angle_rad[i]);
}


TEST(MpcSourceSecondOrder, InnerSubstepBranchDiagnosticsNeverChangePlant)
{
  f1tenth_mpc::UnitySourcePlantState initial{};
  initial.u_mps=3.2;
  initial.v_mps=.11;
  initial.yaw_rate_radps=-1.4;
  initial.physical_steering_rad=-.17;
  initial.body_height_m=.1;
  initial.suspension_history_initialized=false;
  initial.wheel_omega_radps={53.,54.,52.,53.};
  const f1tenth_mpc::UnitySourcePlantInput command{-.29,.23};
  f1tenth_mpc::UnitySourcePlantOutput ordinary{},instrumented{};
  std::vector<f1tenth_mpc::UnitySourceBranchObservation> branches;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      initial,command,.025,&ordinary));
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_trace_branches(
      initial,command,.025,&branches,&instrumented));
  ASSERT_EQ(branches.size(),500u);
  EXPECT_DOUBLE_EQ(ordinary.state.u_mps,instrumented.state.u_mps);
  EXPECT_DOUBLE_EQ(ordinary.state.v_mps,instrumented.state.v_mps);
  EXPECT_DOUBLE_EQ(ordinary.state.yaw_rate_radps,
                   instrumented.state.yaw_rate_radps);
  EXPECT_DOUBLE_EQ(ordinary.state.physical_steering_rad,
                   instrumented.state.physical_steering_rad);
  EXPECT_DOUBLE_EQ(ordinary.integrated_yaw_delta_rad,
                   instrumented.integrated_yaw_delta_rad);
  EXPECT_DOUBLE_EQ(ordinary.integrated_body_forward_displacement_m,
                   instrumented.integrated_body_forward_displacement_m);
  EXPECT_DOUBLE_EQ(ordinary.integrated_body_lateral_displacement_m,
                   instrumented.integrated_body_lateral_displacement_m);
  for(std::size_t w=0;w<4;++w) {
    EXPECT_DOUBLE_EQ(ordinary.state.wheel_omega_radps[w],
                     instrumented.state.wheel_omega_radps[w]);
    EXPECT_DOUBLE_EQ(ordinary.state.suspension_jounce_m[w],
                     instrumented.state.suspension_jounce_m[w]);
  }
  EXPECT_EQ(branches.front().fixed_tick,0);
  EXPECT_EQ(branches.front().vehicle_substep,0);
  EXPECT_EQ(branches.front().vehicle_substep_count,5);
  EXPECT_EQ(branches.back().fixed_tick,24);
  EXPECT_EQ(branches.back().vehicle_substep,4);
  EXPECT_EQ(branches.back().wheel,3);
}

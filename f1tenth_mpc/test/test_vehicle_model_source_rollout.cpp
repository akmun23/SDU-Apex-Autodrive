#include "mpc_linearization.h"
#include "mpc_rti.h"
#include "vehicle_model.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>

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
  EXPECT_TRUE(mpc_rti_refinement_improves_action(1, 100, 1, 90));
  EXPECT_FALSE(mpc_rti_refinement_improves_action(1, 100, 1, 101));
  EXPECT_FALSE(mpc_rti_refinement_improves_action(1, 100, 1, 100));
  EXPECT_FALSE(mpc_rti_refinement_improves_action(1, 100, 0, 90));
  EXPECT_TRUE(mpc_rti_refinement_improves_action(0, INFINITY, 1, 150));
  EXPECT_FALSE(mpc_rti_refinement_improves_action(1, 100, 1, NAN));
  EXPECT_FALSE(mpc_rti_refinement_improves_action(1, 100, 1, INFINITY));
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
  MpcRtiConfiguration_t local_config = config;
  local_config.sqp_steering_trust_region_rad = 0.08f;
  local_config.sqp_throttle_trust_region = 0.10f;
  MpcRtiProblem_t local_problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, nominal_controls,
      references, horizon, PREDICTION_DT_SECONDS, &local_config,
      &local_problem));
  for (int k = 0; k < RICCATI_HORIZON; ++k) {
    EXPECT_NEAR(local_problem.steps[k].u_lb[0],
        std::max(-config.max_steering_rad,
            nominal_controls[k].steering_target_rad - 0.08f), 1.0e-7f);
    EXPECT_NEAR(local_problem.steps[k].u_ub[0],
        std::min(config.max_steering_rad,
            nominal_controls[k].steering_target_rad + 0.08f), 1.0e-7f);
    EXPECT_GE(local_problem.steps[k].u_lb[1], MPC_THROTTLE_MIN_DRIVE);
    EXPECT_LE(local_problem.steps[k].u_ub[1], MPC_THROTTLE_MAX);
    EXPECT_LE(local_problem.steps[k].u_ub[1] -
        local_problem.steps[k].u_lb[1], 0.200001f);
  }
  MpcRtiConfiguration_t no_steering_slew_config = config;
  no_steering_slew_config.weight_steering_slew = 0.0f;
  MpcRtiProblem_t no_steering_slew_problem{};
  ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal_states, nominal_controls,
      references, horizon, PREDICTION_DT_SECONDS,
      &no_steering_slew_config, &no_steering_slew_problem));
  const float expected_reference_slew_linear_term =
      2.0f * config.weight_steering_slew * 0.01f;
  EXPECT_NEAR(problem.steps[0].q[
          MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] -
          no_steering_slew_problem.steps[0].q[
              MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND],
      expected_reference_slew_linear_term, 1.0e-5f);
  EXPECT_NEAR(problem.steps[0].r[0] -
          no_steering_slew_problem.steps[0].r[0],
      -expected_reference_slew_linear_term, 1.0e-5f);
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
  const double qp_gradient =
      (problem.steps[0].Q_diag[MPC_OPT_IDX_U] -
       no_slip_problem.steps[0].Q_diag[MPC_OPT_IDX_U]) * nominal_states[0].plant.u +
      problem.steps[0].q[MPC_OPT_IDX_U] - no_slip_problem.steps[0].q[MPC_OPT_IDX_U];
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

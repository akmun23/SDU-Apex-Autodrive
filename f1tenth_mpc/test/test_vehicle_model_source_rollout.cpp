#include "mpc_linearization.h"
#include "mpc_rti.h"
#include "vehicle_model.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

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

TEST(VehicleModelSourceRollout, CarriesLateralYawWheelAndSuspensionMotion)
{
  MpcModelState_t state = turning_state();
  const MpcModelControl_t control{};
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

TEST(VehicleModelSourceRollout, AppliesCausalYawResidualOnlyInSupportedRegime)
{
  MpcModelState_t state = turning_state();
  state.u = 3.5f;
  state.delayed_steering_command_1 = -0.25f;
  state.delayed_steering_command_2 = -0.25f;
  state.actual_steering_angle = -0.25f;
  const MpcModelControl_t control{-0.25f, 0.25f};
  const MpcStageResult_t raw = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(raw.valid);
  ASSERT_GT(std::abs(raw.next.r), 1.0e-3f);

  state.yaw_rate_prediction_residual_radps = raw.next.r > 0.0f ? -0.08f : 0.08f;
  state.yaw_rate_residual_correction_enabled = 1;
  const MpcStageResult_t corrected = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(corrected.valid);
  EXPECT_NEAR(corrected.next.r, raw.next.r -
      0.5f * state.yaw_rate_prediction_residual_radps, 1.0e-5f);
  EXPECT_EQ(corrected.next.yaw_rate_residual_correction_enabled, 1);
  EXPECT_FLOAT_EQ(corrected.next.yaw_rate_prediction_residual_radps,
      state.yaw_rate_prediction_residual_radps);

  state.u = 6.0f;
  const MpcStageResult_t outside_speed_support = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(outside_speed_support.valid);
  MpcModelState_t state_without_correction = state;
  state_without_correction.yaw_rate_residual_correction_enabled = 0;
  const MpcStageResult_t outside_speed_raw = mpc_vehicle_model_step(
      &state_without_correction, &control, 0.025f, 0.0f);
  ASSERT_TRUE(outside_speed_raw.valid);
  EXPECT_FLOAT_EQ(outside_speed_support.next.r, outside_speed_raw.next.r);
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
     DirectThrottleCommandEntersTheTwoPacketPlantQueue)
{
  MpcModelState_t state = turning_state();
  state.delayed_throttle_command_1 = 0.20f;
  state.delayed_throttle_command_2 = 0.30f;
  const MpcModelControl_t control{0.0f, 0.70f};
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);
  /* The plant sees the already-delayed command; the optimizer's command is
   * queued for the appropriate future packet instead of being converted to a
   * speed target by an internal actuator controller. */
  EXPECT_FLOAT_EQ(stage.next.delayed_throttle_command_1, 0.30f);
  EXPECT_FLOAT_EQ(stage.next.delayed_throttle_command_2, 0.70f);
  EXPECT_TRUE(std::isfinite(stage.next.u));
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
     SteeringCommandAdvancesThroughBothPendingPackets)
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
  reference_input.steering_target_rad =
      state.delayed_steering_command_1;
  reference_input.throttle_normalized =
      state.delayed_throttle_command_1;
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

TEST(VehicleModelSourceRollout, AppliesAndShiftsTheTwoPacketThrottleQueue)
{
  MpcModelState_t state = turning_state();
  state.delayed_throttle_command_1 = 0.0f;
  state.delayed_throttle_command_2 = 0.30f;
  const MpcModelControl_t control{};
  const MpcStageResult_t delayed = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(delayed.valid);
  EXPECT_NE(delayed.branch_flags & MPC_STAGE_PLANT_BRAKE, 0U);

  MpcModelState_t immediate_state = state;
  immediate_state.delayed_throttle_command_1 = 0.30f;
  const MpcStageResult_t immediate = mpc_vehicle_model_step(
      &immediate_state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(immediate.valid);
  EXPECT_LT(delayed.next.u, immediate.next.u);
  EXPECT_FLOAT_EQ(delayed.next.delayed_throttle_command_1, 0.30f);
  EXPECT_FLOAT_EQ(delayed.next.delayed_throttle_command_2, 0.0f);
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
      0.10f, 0.10f, 4.0f, 1.0f, 4.0f, 0.5f,
      400.0f, 400.0f, 400.0f, 400.0f,
      0.10f, 1.0f, 0.25f, 4.0f, 0.25f, 4.0f,
      0.05f, 0.05f, 0.05f, 0.05f};
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
  for (int k = 0; k < RICCATI_HORIZON; ++k) {
    for (int a = 0; a < MPC_MODEL_NU; ++a)
      EXPECT_NEAR(plain_solution.u[k][a], scaled_solution.u[k][a], 2.0e-3f);
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      EXPECT_NEAR(plain_solution.x[k + 1][i], scaled_solution.x[k + 1][i],
          2.0e-3f);
  }
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
}

TEST(MpcRtiDelayCondensing, StartsAfterQueuedPrefixAndPreservesPlantJacobian)
{
  constexpr int horizon = PREDICTION_HORIZON;
  MpcRtiState_t nominal_states[PREDICTION_HORIZON + 1] = {};
  MpcModelControl_t nominal_controls[PREDICTION_HORIZON] = {};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
  MpcRtiConfiguration_t config{};
  config.weight_e_y = 1.0f;
  config.weight_e_psi = 1.0f;
  config.weight_u = 1.0f;
  config.weight_v = 1.0f;
  config.weight_r = 1.0f;
  config.weight_steering_command = 1.0f;
  config.weight_steering_target = 1.0f;
  config.weight_throttle_command = 1.0f;
  config.terminal_multiplier = 1.0f;
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 10.0f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m = 0.05f;
  config.first_prediction_corridor_margin_m = 0.05f;
  config.corridor_preview_halfwidth_m = 0.10f;
  config.nonlinear_corridor_tolerance_m = 0.01f;
  config.recovery_seed_policy = MPC_RTI_RECOVERY_SEED_NOMINAL;
  config.jacobian_threads = 8;

  MpcModelState_t plant = turning_state();
  plant.u = 5.0f;
  plant.actual_steering_angle = 0.25f;
  for (int k = 0; k <= horizon; ++k) {
    nominal_states[k].plant = plant;
    references[k].u = plant.u;
    references[k].steering_command = plant.delayed_steering_command_2;
    references[k].left_bound = 1.0f;
    references[k].right_bound = 1.0f;
  }
  for (int k = 0; k < horizon; ++k) {
    nominal_controls[k].steering_target_rad = (k % 2 == 0) ? 0.1f : -0.1f;
    nominal_controls[k].throttle_command = k % 2 == 0 ? 0.2f : 0.0f;
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
  MpcModelState_t prepared_optimizer_plant =
      nominal_states[MPC_COMMAND_DELAY_STEPS].plant;
  ASSERT_TRUE(vehicle_model_initialize_plant_state(&prepared_optimizer_plant));
  EXPECT_EQ(MPC_RTI_NX, 20);
  EXPECT_EQ(RICCATI_HORIZON,
      PREDICTION_HORIZON - MPC_COMMAND_DELAY_STEPS);
  EXPECT_FLOAT_EQ(problem.x0[MPC_OPT_IDX_EY],
      nominal_states[MPC_COMMAND_DELAY_STEPS].plant.e_y);
  EXPECT_FLOAT_EQ(problem.x0[MPC_OPT_IDX_U],
      nominal_states[MPC_COMMAND_DELAY_STEPS].plant.u);
  EXPECT_FLOAT_EQ(problem.steps[0].u_lb[1],
      MPC_THROTTLE_MIN_DRIVE);
  EXPECT_FLOAT_EQ(problem.steps[0].u_ub[1],
      MPC_THROTTLE_MAX);
  EXPECT_FLOAT_EQ(problem.steps[1].u_lb[1],
      MPC_THROTTLE_MIN);
  EXPECT_FLOAT_EQ(problem.steps[1].u_ub[1],
      MPC_THROTTLE_MIN);
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
          serial[model_stage].A[model_row][MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1]);
      EXPECT_FLOAT_EQ(problem.steps[k].B[row][1],
          serial[model_stage].A[model_row][MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1]);
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

  // The fixed two-command prefix has already been propagated into x0. Replay
  // all remaining nominal decisions through the reduced affine model and
  // compare against the corresponding full delayed-plant states.
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
  EXPECT_EQ(problem.horizon, horizon);
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
}

TEST(MpcRtiCorridorSchedule,
     TapersAnExistingClearanceViolationAcrossThePredictionHorizon)
{
  MpcTrajectorySample_t trajectory[3] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.65, 0.65, 0.0},
      {10.0, 10.0, 0.0, 0.0, 0.0, 1.5, 0.65, 0.65, 0.0},
      {20.0, 20.0, 0.0, 0.0, 0.0, 1.5, 0.65, 0.65, 0.0}};
  size_t trajectory_count = 3;
  double lap_length = 0.0;
  ASSERT_TRUE(mpc_trajectory_prepare(
      trajectory, &trajectory_count, &lap_length));

  MpcRtiConfiguration_t config{};
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 8.0f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m = 0.15f;
  config.first_prediction_corridor_margin_m = 0.15f;
  config.corridor_preview_halfwidth_m = 0.10f;
  config.nonlinear_corridor_tolerance_m = 0.005f;
  config.recovery_seed_policy = MPC_RTI_RECOVERY_SEED_NOMINAL;

  MpcRtiState_t initial{};
  initial.plant.e_y = 0.65f - config.corridor_margin_m -
      0.5f * MPC_CAR_WIDTH_M + 0.01f;  // 10 mm outside physical-clearance bound.
  MpcRtiNominal_t nominal{};
  nominal.valid = 1;
  nominal.horizon = PREDICTION_HORIZON;
  for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
    nominal.states[k] = initial;
    nominal.progress[k] = 0.0;
  }

  MpcRtiCorridorSchedule_t schedule{};
  ASSERT_TRUE(mpc_rti_build_corridor_schedule(&initial, &nominal,
      trajectory, trajectory_count, lap_length, PREDICTION_DT_SECONDS,
      &config, &schedule));
  ASSERT_TRUE(schedule.recovery_active);
  EXPECT_TRUE(schedule.recovery_not_found)
      << "The nominal seed is stationary, but the active envelope must still "
         "allow a bounded recovery rollout.";
  EXPECT_NEAR(schedule.initial_normal_violation, 0.01f, 1.0e-5f);
  EXPECT_GT(schedule.active_upper[1], schedule.normal_upper[1]);
  EXPECT_GT(schedule.active_upper[1] - schedule.normal_upper[1],
      schedule.active_upper[PREDICTION_HORIZON] -
          schedule.normal_upper[PREDICTION_HORIZON]);
  EXPECT_FLOAT_EQ(schedule.active_upper[PREDICTION_HORIZON],
      schedule.normal_upper[PREDICTION_HORIZON]);
  EXPECT_TRUE(schedule.recovery_stage[1]);
  EXPECT_FALSE(schedule.recovery_stage[PREDICTION_HORIZON]);
}

TEST(MpcRtiCorridorRecovery,
     AllowsTransientUseOfFiveCentimeterReserveThenRequiresReentry)
{
  MpcTrajectorySample_t trajectory[3] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.2, 1.2, 0.0},
      {10.0, 10.0, 0.0, 0.0, 0.0, 1.5, 1.2, 1.2, 0.0},
      {20.0, 20.0, 0.0, 0.0, 0.0, 1.5, 1.2, 1.2, 0.0}};
  size_t trajectory_count = 3;
  double lap_length = 0.0;
  ASSERT_TRUE(mpc_trajectory_prepare(
      trajectory, &trajectory_count, &lap_length));

  MpcRtiConfiguration_t config{};
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 8.0f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.corridor_margin_m = 0.15f;
  config.first_prediction_corridor_margin_m = 0.15f;
  config.nonlinear_corridor_tolerance_m = 0.0f;

  MpcRtiState_t initial{};
  initial.plant.e_y = 0.90f;
  initial.plant.e_psi = 0.05f;
  initial.plant.u = 1.5f;
  initial.plant.r = -0.4f;

  MpcModelControl_t controls[PREDICTION_HORIZON] = {};
  for (int k = 0; k < PREDICTION_HORIZON; ++k) {
    controls[k].steering_target_rad = 0.0f;
    controls[k].throttle_command = 0.10f;
  }
  MpcRtiCorridorSchedule_t schedule{};
  schedule.horizon = PREDICTION_HORIZON;
  schedule.initial_normal_violation = 0.01f;
  schedule.recovery_active = 1;
  for (int k = 1; k <= PREDICTION_HORIZON; ++k)
    schedule.recovery_stage[k] = 1;

  MpcRtiState_t states[PREDICTION_HORIZON + 1] = {};
  double progress[PREDICTION_HORIZON + 1] = {};
  int failure_stage = -1;
  MpcRtiRolloutFailure_t failure{};
  const MpcRtiRolloutStatus_t status =
      mpc_rti_rollout_candidate_with_schedule(&initial, 0.0, controls,
          trajectory, trajectory_count, lap_length, nullptr, nullptr, nullptr,
          PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &config, &schedule,
          states, progress, &failure_stage, &failure);

  ASSERT_EQ(status, MPC_RTI_ROLLOUT_OK)
      << "failure stage=" << failure_stage
      << " ey=" << failure.state.plant.e_y
      << " epsi=" << failure.state.plant.e_psi
      << " bounds=[" << failure.lower_bound_m << ','
      << failure.upper_bound_m << ']';
  const auto normal_upper = [&](const MpcRtiState_t & state) {
    const float extent = 0.5f * MPC_CAR_WIDTH_M *
            std::abs(std::cos(state.plant.e_psi)) +
        std::max(MPC_REAR_AXLE_TO_FRONT_BUMPER_M, MPC_REAR_OVERHANG_M) *
            std::abs(std::sin(state.plant.e_psi));
    return trajectory[1].left_bound - config.corridor_margin_m - extent;
  };
  EXPECT_GT(states[1].plant.e_y, normal_upper(states[1]));
  EXPECT_GT(states[PREDICTION_HORIZON].plant.e_y,
      normal_upper(states[PREDICTION_HORIZON]));

  /* The production recovery schedule disables the recovery allowance at its
   * endpoint. The same rollout must therefore fail if it has not re-entered
   * the normal corridor by the horizon. */
  schedule.recovery_stage[PREDICTION_HORIZON] = 0;
  failure_stage = -1;
  EXPECT_EQ(mpc_rti_rollout_candidate_with_schedule(&initial, 0.0, controls,
      trajectory, trajectory_count, lap_length, nullptr, nullptr, nullptr,
      PREDICTION_HORIZON, PREDICTION_DT_SECONDS, &config, &schedule,
      states, progress, &failure_stage, &failure), MPC_RTI_ROLLOUT_CORRIDOR);
  EXPECT_EQ(failure_stage, PREDICTION_HORIZON - 1);
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
  config.recovery_seed_policy = MPC_RTI_RECOVERY_SEED_NOMINAL;
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

TEST(MpcReference,
     FallingReferenceStillSeedsDriveWhenDelayedSpeedIsBelowTarget)
{
  MpcModelControl_t controls[PREDICTION_HORIZON] = {};
  MpcRtiState_t predicted[PREDICTION_HORIZON + 1] = {};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
  controls[0].throttle_command = MPC_THROTTLE_MIN;
  predicted[3].plant.u = 3.0f;
  references[3].u = 4.0f;
  ASSERT_TRUE(mpc_rti_select_throttle_modes(
      controls, predicted, references, PREDICTION_HORIZON));
  EXPECT_FLOAT_EQ(controls[0].throttle_command, MPC_THROTTLE_MIN_DRIVE)
      << "A negative reference-speed derivative must not select Unity's "
         "full-brake q=0 branch while delayed predicted speed is still "
         "below the decreasing speed target; drive-mode magnitude remains "
         "a continuous optimizer decision.";
}

TEST(MpcReference, UnderspeedPreservesContinuousDriveDecision)
{
  MpcModelControl_t controls[PREDICTION_HORIZON] = {};
  MpcRtiState_t predicted[PREDICTION_HORIZON + 1] = {};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
  controls[0].throttle_command = 0.37f;
  predicted[3].plant.u = 1.5f;
  references[3].u = 6.0f;
  ASSERT_TRUE(mpc_rti_select_throttle_modes(
      controls, predicted, references, PREDICTION_HORIZON));
  EXPECT_FLOAT_EQ(controls[0].throttle_command, 0.37f);
}

TEST(MpcReference, PredictedOverspeedSelectsTheUnityFullBrakeBranch)
{
  MpcModelControl_t controls[PREDICTION_HORIZON] = {};
  MpcRtiState_t predicted[PREDICTION_HORIZON + 1] = {};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
  predicted[3].plant.u = 4.5f;
  references[3].u = 4.0f;
  ASSERT_TRUE(mpc_rti_select_throttle_modes(
      controls, predicted, references, PREDICTION_HORIZON));
  EXPECT_FLOAT_EQ(controls[0].throttle_command, MPC_THROTTLE_MIN);
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

TEST(MpcCandidate, RejectsWireDeadbandButAcceptsSourceSaturatedSlip)
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
  controls[0].throttle_command = 0.0005f;
  EXPECT_EQ(mpc_rti_rollout_candidate_with_schedule(&initial, 0.0,
      controls, trajectory, trajectory_count, lap_length, nullptr, nullptr,
      nullptr, 1, PREDICTION_DT_SECONDS, &config, nullptr, states, progress,
      nullptr, nullptr), MPC_RTI_ROLLOUT_COMMAND_LIMIT);

  for (int k = 0; k < PREDICTION_HORIZON; ++k)
    controls[k].throttle_command = 0.001f;
  EXPECT_EQ(mpc_rti_rollout_candidate_with_schedule(&initial, 0.0,
      controls, trajectory, trajectory_count, lap_length, nullptr, nullptr,
      nullptr, 4, PREDICTION_DT_SECONDS, &config, nullptr, states, progress,
      nullptr, nullptr), MPC_RTI_ROLLOUT_OK);
  EXPECT_GT(states[1].incoming_max_positive_wheel_slip,
            UNITY_SOURCE_FORWARD_ASYMPTOTE_SLIP);
  EXPECT_GT(states[1].incoming_wheel_slip_excess_mps[2], 0.0);

  for (int k = 0; k < PREDICTION_HORIZON; ++k)
    controls[k].throttle_command = 0.0f;
  EXPECT_EQ(mpc_rti_rollout_candidate_with_schedule(&initial, 0.0,
      controls, trajectory, trajectory_count, lap_length, nullptr, nullptr,
      nullptr, 4, PREDICTION_DT_SECONDS, &config, nullptr, states, progress,
      nullptr, nullptr), MPC_RTI_ROLLOUT_OK);
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
    references[stage].left_bound = 10.0f;
    references[stage].right_bound = 10.0f;
  }
  for (int k = 0; k < PREDICTION_HORIZON; ++k)
    controls[k].throttle_command = 0.001f;
  ASSERT_TRUE(vehicle_model_set_active_target_speed_ceiling(16.0f));
  for (int k = 0; k < PREDICTION_HORIZON; ++k) {
    const MpcStageResult_t predicted = mpc_vehicle_model_step(
        &nominal_states[k].plant, &controls[k],
        PREDICTION_DT_SECONDS, 0.0f);
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
  for (int solver_stage = 1; solver_stage < RICCATI_HORIZON;
       ++solver_stage) {
    const int model_stage = solver_stage + MPC_COMMAND_DELAY_STEPS;
    double excess[4] = {}, derivative[4] = {};
    ASSERT_TRUE(vehicle_model_longitudinal_slip_excess(
        &nominal_states[model_stage].plant, excess, derivative));
    if (excess[2] > 0.0 && std::abs(derivative[2]) > 1.0e-9) {
      EXPECT_GT(problem.steps[solver_stage].Q_diag[6 + 2], 0.0f);
      slip_cost_retained_after_queued_prefix = true;
      break;
    }
  }
  EXPECT_TRUE(slip_cost_retained_after_queued_prefix);

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
     JacobianUsesExactTwoPacketQueuesAndCheapFrenetColumns)
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
                              [MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2],
                  0.0f);
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
  for (int row = 0; row < MPC_MODEL_NX; ++row) {
    if (row != MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1) {
      EXPECT_FLOAT_EQ(
          jacobian.A[row][MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2], 0.0f);
    }
  }
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
  EXPECT_EQ(MPC_RTI_NX, 20);
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

}  // namespace

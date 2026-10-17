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
  state.longitudinal_acceleration_mps2 = 6.0f;
  state.steering_command = 0.35f;
  state.delayed_steering_command_1 = 0.35f;
  state.delayed_steering_command_2 = 0.35f;
  state.actual_steering_angle = 0.35f;
  state.delayed_throttle_command_1 = 0.30f;
  state.delayed_throttle_command_2 = 0.30f;
  return state;
}

MpcYawRateResponseSurface_t y1_test_surface()
{
  MpcYawRateResponseSurface_t surface{};
  constexpr float steering_knots[MPC_YAW_SURFACE_Q_KNOTS] = {
    0.15f, 0.20f, 0.21f, 0.22f, 0.23f,
    0.25f, 0.30f, 0.35f, 0.42f, 0.50f};
  surface.enabled = 1;
  surface.speed_count = 3;
  surface.response_scale = 1.0f;
  surface.blend_q_start = 0.30f;
  surface.blend_q_end = 0.70f;
  surface.speed_blend_margin_mps = 0.10f;
  surface.low_speed_blend_margin_mps = 0.10f;
  surface.speed_mps[0] = 2.5f;
  surface.speed_mps[1] = 3.0f;
  surface.speed_mps[2] = 3.5f;
  surface.steering_blend_start_rad = 0.23f;
  surface.steering_blend_full_start_rad = 0.25f;
  surface.steering_blend_full_end_rad = 0.42f;
  surface.steering_blend_end_rad = 0.45f;
  surface.hold_response_time_constant_s = 0.1293144252f;
  surface.hold_rate_full_radps = 0.05f;
  surface.hold_rate_zero_radps = 0.15f;
  for (int speed_index = 0; speed_index < surface.speed_count;
       ++speed_index) {
    for (int direction = 0; direction < MPC_YAW_SURFACE_TURN_DIRECTIONS;
         ++direction) {
      for (int knot = 0; knot < MPC_YAW_SURFACE_Q_KNOTS; ++knot) {
        surface.q[speed_index][direction][knot] =
            surface.speed_mps[speed_index] * std::tan(steering_knots[knot]);
        surface.yaw_rate_abs_rps[speed_index][direction][knot] = 2.0f;
      }
    }
  }
  return surface;
}

MpcModelState_t steady_y1_state(float steering, float throttle)
{
  MpcModelState_t state{};
  state.u = 3.0f;
  state.r = steering < 0.0f ? -2.0f : 2.0f;
  state.longitudinal_acceleration_mps2 = 3.0f;
  state.steering_command = steering;
  state.delayed_steering_command_1 = steering;
  state.delayed_steering_command_2 = steering;
  state.actual_steering_angle = steering;
  state.delayed_throttle_command_1 = throttle;
  state.delayed_throttle_command_2 = throttle;
  state.source_plant_initialized = 1;
  state.source_suspension_history_initialized = 1;
  state.source_plant_state[UNITY_SOURCE_U] = state.u;
  state.source_plant_state[UNITY_SOURCE_YAW_RATE] = state.r;
  state.source_plant_state[UNITY_SOURCE_BODY_HEIGHT] = 0.00464804098;
  state.source_plant_state[UNITY_SOURCE_PHYSICAL_STEERING] = steering;
  for (int wheel = 0; wheel < 4; ++wheel) {
    state.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] =
        state.u / 0.059;
  }
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
  EXPECT_NE(second.next.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL],
            first.next.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL]);
  EXPECT_NE(second.next.source_plant_state[UNITY_SOURCE_ROLL],
            first.next.source_plant_state[UNITY_SOURCE_ROLL]);
}

TEST(VehicleModelSourceRollout, ZeroThrottleUsesTheSourceBrakeBranch)
{
  MpcModelState_t state{};
  state.u = 3.0f;
  state.longitudinal_acceleration_mps2 = 0.0f;
  const MpcModelControl_t control{};
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);
  EXPECT_FLOAT_EQ(stage.next.delayed_throttle_command_2, 0.0f);
  EXPECT_NE(stage.branch_flags & MPC_STAGE_PLANT_BRAKE, 0U);
  EXPECT_LT(stage.next.u, state.u);
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
  EXPECT_FLOAT_EQ(stage.next.source_plant_state[UNITY_SOURCE_U], stage.next.u);
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
  state.source_plant_state[UNITY_SOURCE_U] = state.u;
  state.source_plant_state[UNITY_SOURCE_V] = 0.0;
  state.source_plant_state[UNITY_SOURCE_YAW_RATE] = state.r;
  state.source_plant_state[UNITY_SOURCE_BODY_HEIGHT] = 0.00464804098;
  state.source_plant_state[UNITY_SOURCE_PHYSICAL_STEERING] =
      state.actual_steering_angle;
  for (int wheel = 0; wheel < 4; ++wheel)
    state.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] =
        state.u / 0.059;

  const MpcModelControl_t control{};
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);

  /* Independently step the source plant from the same physical state. Its v
   * is COM-referenced; the public MPC state is rear-axle referenced. */
  UnitySourcePlantInputC reference_input{};
  std::memcpy(reference_input.state, state.source_plant_state,
              sizeof(reference_input.state));
  reference_input.state[UNITY_SOURCE_U] = state.u;
  reference_input.state[UNITY_SOURCE_V] =
      state.v + state.r * com_forward_from_rear_axle_m;
  reference_input.state[UNITY_SOURCE_YAW_RATE] = state.r;
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

  EXPECT_NEAR(stage.next.source_plant_state[UNITY_SOURCE_V],
              reference_output.state[UNITY_SOURCE_V], 1.0e-10);
  EXPECT_NEAR(stage.next.source_plant_state[UNITY_SOURCE_YAW_RATE],
              reference_output.state[UNITY_SOURCE_YAW_RATE], 1.0e-10);
  EXPECT_NEAR(stage.next.v,
              reference_output.state[UNITY_SOURCE_V] -
                  reference_output.state[UNITY_SOURCE_YAW_RATE] *
                      com_forward_from_rear_axle_m,
              1.0e-6);
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

TEST(VehicleModelSourceRollout,
     UsesRearEncoderRatesForOneStageThenPropagatesWheelSpin)
{
  MpcModelState_t state = turning_state();
  state.source_plant_initialized = 1;
  state.source_suspension_history_initialized = 1;
  state.source_plant_state[UNITY_SOURCE_U] = state.u;
  state.source_plant_state[UNITY_SOURCE_V] = 0.0;
  state.source_plant_state[UNITY_SOURCE_YAW_RATE] = 0.0;
  state.source_plant_state[UNITY_SOURCE_PHYSICAL_STEERING] =
      state.actual_steering_angle;
  for (int wheel = 0; wheel < 4; ++wheel)
    state.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] =
        state.u / 0.059;
  state.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RL] = 50.0;
  state.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RR] = 52.0;
  state.source_rear_wheel_omega_feedback_radps[0] = 43.0f;
  state.source_rear_wheel_omega_feedback_radps[1] = 47.0f;
  state.source_rear_wheel_omega_feedback_valid = 1;

  MpcModelState_t expected = state;
  expected.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RL] = 43.0;
  expected.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RR] = 47.0;
  expected.source_rear_wheel_omega_feedback_valid = 0;
  const MpcModelControl_t control{};
  const MpcStageResult_t corrected = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  const MpcStageResult_t reference = mpc_vehicle_model_step(
      &expected, &control, 0.025f, 0.0f);
  ASSERT_TRUE(corrected.valid);
  ASSERT_TRUE(reference.valid);
  EXPECT_FALSE(corrected.next.source_rear_wheel_omega_feedback_valid);
  EXPECT_NEAR(corrected.next.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RL],
              reference.next.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RL],
              1.0e-10);
  EXPECT_NEAR(corrected.next.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RR],
              reference.next.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RR],
              1.0e-10);

  const MpcStageResult_t continued = mpc_vehicle_model_step(
      &corrected.next, &control, 0.025f, 0.0f);
  ASSERT_TRUE(continued.valid);
  EXPECT_NE(continued.next.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RL],
            corrected.next.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RL]);
}

TEST(RiccatiSolverDeadline, StopsOnlyAfterACompleteAdmmIteration)
{
  RiccatiStepData_t steps[PREDICTION_HORIZON] = {};
  float terminal_q[RICCATI_MAX_NX] = {};
  float terminal_x_lb[RICCATI_MAX_NX] = {};
  float terminal_x_ub[RICCATI_MAX_NX] = {};
  float x0[RICCATI_MAX_NX] = {};
  for (int k = 0; k < PREDICTION_HORIZON; ++k) {
    steps[k].A[0][0] = 1.0f;
    steps[k].B[0][0] = 1.0f;
    steps[k].Q_diag[0] = 1.0f;
    steps[k].R_diag[0] = 1.0f;
    steps[k].x_lb[0] = -0.05f;
    steps[k].x_ub[0] = 0.05f;
    steps[k].u_lb[0] = -1.0f;
    steps[k].u_ub[0] = 1.0f;
  }
  terminal_q[0] = 1.0f;
  terminal_x_lb[0] = -0.05f;
  terminal_x_ub[0] = 0.05f;
  x0[0] = 2.0f;

  RiccatiAdmmConfig_t config{};
  config.rho = 7.0f;
  config.rho_u = 7.0f;
  config.tolerance = 1.0e-9f;
  config.max_iterations = 100;
  config.over_relaxation = 1.0f;
  config.use_prefactorization = 1;
  config.max_solve_time_us = 0.001;

  RiccatiAdmmState_t memory{};
  RiccatiSolution_t solution{};
  const RiccatiStatus_t status = riccati_admm_solve(
      steps, terminal_q, terminal_q, terminal_x_lb, terminal_x_ub, x0,
      1, 1, PREDICTION_HORIZON, &config, &memory, &solution);
  EXPECT_EQ(status, RICCATI_STATUS_MAX_ITERATIONS);
  EXPECT_GE(solution.iterations, 1);
  EXPECT_LT(solution.iterations, config.max_iterations);
  EXPECT_TRUE(memory.initialized);
}

TEST(MpcRtiStageParallelism, PreservesSerialStageJacobians)
{
  constexpr int horizon = 12;
  MpcRtiState_t nominal_states[PREDICTION_HORIZON + 1] = {};
  MpcModelControl_t nominal_controls[PREDICTION_HORIZON] = {};
  MpcRtiReference_t references[PREDICTION_HORIZON + 1] = {};
  MpcRtiConfiguration_t config{};
  config.weight_e_y = 1.0f;
  config.weight_e_psi = 1.0f;
  config.weight_u = 1.0f;
  config.weight_longitudinal_acceleration = 1.0f;
  config.weight_v = 1.0f;
  config.weight_r = 1.0f;
  config.weight_steering_command = 1.0f;
  config.weight_steering_rate = 1.0f;
  config.weight_throttle_command = 1.0f;
  config.weight_steering_rate_change = 1.0f;
  config.weight_throttle_command_change = 1.0f;
  config.terminal_multiplier = 1.0f;
  config.max_speed_mps = 16.0f;
  config.active_speed_ceiling_mps = 10.0f;
  config.max_steering_rad = SOURCE_MAX_STEERING_RAD;
  config.max_steering_rate_radps = SOURCE_STEERING_RATE_RADPS;
  config.corridor_margin_m = 0.05f;
  config.first_prediction_corridor_margin_m = 0.05f;
  config.corridor_preview_halfwidth_m = 0.10f;
  config.nonlinear_corridor_tolerance_m = 0.01f;
  config.recovery_seed_policy = MPC_RTI_RECOVERY_SEED_NOMINAL;
  config.jacobian_threads = 8;

  MpcModelState_t plant = turning_state();
  plant.u = 5.0f;
  plant.longitudinal_acceleration_mps2 = 5.0f;
  plant.actual_steering_angle = 0.25f;
  for (int k = 0; k <= horizon; ++k) {
    nominal_states[k].plant = plant;
    references[k].u = plant.u;
    references[k].steering_command = plant.steering_command;
    references[k].left_bound = 1.0f;
    references[k].right_bound = 1.0f;
  }
  for (int k = 0; k < horizon; ++k) {
    nominal_controls[k].steering_rate = (k % 2 == 0) ? 0.1f : -0.1f;
    nominal_controls[k].throttle_command = 0.2f;
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
  for (int k = 0; k < horizon; ++k) {
    for (int row = 0; row < MPC_MODEL_NX; ++row) {
      EXPECT_EQ(std::memcmp(problem.steps[k].A[row], serial[k].A[row],
                            sizeof(serial[k].A[row])), 0);
      EXPECT_EQ(std::memcmp(problem.steps[k].B[row], serial[k].B[row],
                            sizeof(serial[k].B[row])), 0);
      EXPECT_FLOAT_EQ(problem.steps[k].d[row], serial[k].d[row]);
    }
  }
}

TEST(VehicleModelSourceRollout,
     Y1EquilibriumCorrectsOnlySupportedSteadySourceYaw)
{
  const MpcYawRateResponseSurface_t disabled{};
  const MpcYawRateResponseSurface_t surface = y1_test_surface();
  ASSERT_TRUE(vehicle_model_set_yaw_rate_response_surface(&disabled));
  const MpcModelState_t state = steady_y1_state(-0.25f, 0.15f);
  const MpcModelControl_t control{};
  const MpcStageResult_t physics_only = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(physics_only.valid);

  ASSERT_TRUE(vehicle_model_set_yaw_rate_response_surface(&surface));
  const MpcStageResult_t corrected = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(corrected.valid);
  EXPECT_GT(corrected.next.source_yaw_surface_blend, 0.99f);
  EXPECT_NEAR(corrected.next.source_yaw_surface_target_radps, -2.0f, 1.0e-4f);
  EXPECT_LT(std::abs(corrected.next.r - state.r),
            std::abs(physics_only.next.r - state.r));
  EXPECT_NEAR(corrected.next.source_plant_state[UNITY_SOURCE_YAW_RATE],
              corrected.next.r, 1.0e-6);
  EXPECT_TRUE(vehicle_model_set_yaw_rate_response_surface(&disabled));
}

TEST(VehicleModelSourceRollout,
     Y1EquilibriumDoesNotCorrectUnsupportedSteeringOrBraking)
{
  const MpcYawRateResponseSurface_t disabled{};
  const MpcYawRateResponseSurface_t surface = y1_test_surface();
  const MpcModelControl_t control{};
  for (const auto & state : {
           steady_y1_state(-0.20f, 0.15f),
           steady_y1_state(-0.25f, 0.0f)}) {
    ASSERT_TRUE(vehicle_model_set_yaw_rate_response_surface(&disabled));
    const MpcStageResult_t physics_only = mpc_vehicle_model_step(
        &state, &control, 0.025f, 0.0f);
    ASSERT_TRUE(physics_only.valid);
    ASSERT_TRUE(vehicle_model_set_yaw_rate_response_surface(&surface));
    const MpcStageResult_t gated = mpc_vehicle_model_step(
        &state, &control, 0.025f, 0.0f);
    ASSERT_TRUE(gated.valid);
    EXPECT_FLOAT_EQ(gated.next.source_yaw_surface_blend, 0.0f);
    EXPECT_NEAR(gated.next.r, physics_only.next.r, 1.0e-7f);
  }
  EXPECT_TRUE(vehicle_model_set_yaw_rate_response_surface(&disabled));
}

TEST(VehicleModelSourceRollout, JacobianDifferentiatesTheSameSourceStep)
{
  MpcModelState_t state = turning_state();
  state.source_plant_initialized = 1;
  state.source_plant_state[UNITY_SOURCE_U] = state.u;
  state.source_plant_state[UNITY_SOURCE_V] = 0.0;
  state.source_plant_state[UNITY_SOURCE_YAW_RATE] = 0.0;
  state.source_plant_state[UNITY_SOURCE_PHYSICAL_STEERING] =
      state.actual_steering_angle;
  for (int wheel = 0; wheel < 4; ++wheel)
    state.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] =
        state.u / 0.059;

  const MpcModelControl_t control{};
  MpcStageResult_t nominal{};
  MpcStageLinearization_t linearization{};
  ASSERT_TRUE(mpc_vehicle_model_step_with_jacobian(
      &state, &control, 0.025f, 0.0f, &nominal, &linearization));
  ASSERT_TRUE(linearization.valid);
  EXPECT_NEAR(linearization.A[10][11], 1.0f, 2.0e-4f);
  EXPECT_FLOAT_EQ(linearization.A[11][10], 0.0f);

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
     JacobianUsesExactCommandQueueAndCheapFrenetColumns)
{
  MpcModelState_t state = turning_state();
  state.steering_command = 0.20f;
  state.delayed_steering_command_1 = 0.18f;
  state.delayed_steering_command_2 = -0.12f;
  state.delayed_throttle_command_1 = 0.24f;
  state.delayed_throttle_command_2 = 0.31f;
  state.source_plant_initialized = 1;
  state.source_suspension_history_initialized = 1;
  state.source_plant_state[UNITY_SOURCE_U] = state.u;
  state.source_plant_state[UNITY_SOURCE_V] = state.v;
  state.source_plant_state[UNITY_SOURCE_YAW_RATE] = state.r;
  state.source_plant_state[UNITY_SOURCE_PHYSICAL_STEERING] =
      state.actual_steering_angle;
  for (int wheel = 0; wheel < 4; ++wheel)
    state.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] =
        state.u / 0.059;

  const MpcModelControl_t control{0.30f, 0.0f};
  constexpr float dt = 0.025f;
  constexpr float curvature = 0.17f;
  MpcStageResult_t nominal{};
  MpcStageLinearization_t jacobian{};
  ASSERT_TRUE(mpc_vehicle_model_step_with_jacobian(
      &state, &control, dt, curvature, &nominal, &jacobian));

  EXPECT_NEAR(jacobian.A[10][11], 1.0f, 1.0e-6f);
  EXPECT_FLOAT_EQ(jacobian.A[11][10], 0.0f);
  EXPECT_NEAR(jacobian.A[7][6], 1.0f, 1.0e-6f);
  EXPECT_FLOAT_EQ(jacobian.A[6][6], 1.0f);
  EXPECT_NEAR(jacobian.B[6][0], dt, 1.0e-6f);

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
  plus_command.steering_command += 1.0e-4f;
  minus_command.steering_command -= 1.0e-4f;
  const auto plus_steering = mpc_vehicle_model_step(
      &plus_command, &control, dt, curvature);
  const auto minus_steering = mpc_vehicle_model_step(
      &minus_command, &control, dt, curvature);
  ASSERT_TRUE(plus_steering.valid);
  ASSERT_TRUE(minus_steering.valid);
  EXPECT_NEAR(jacobian.A[6][6],
      (plus_steering.next.steering_command -
       minus_steering.next.steering_command) / 2.0e-4f, 1.0e-3f);
  EXPECT_NEAR(jacobian.A[7][6],
      (plus_steering.next.delayed_steering_command_1 -
       minus_steering.next.delayed_steering_command_1) / 2.0e-4f,
      1.0e-3f);

  MpcModelState_t unused_steer_queue = state;
  unused_steer_queue.delayed_steering_command_2 += 0.01f;
  const auto unused_queue_result = mpc_vehicle_model_step(
      &unused_steer_queue, &control, dt, curvature);
  ASSERT_TRUE(unused_queue_result.valid);
  for (int row = 0; row < MPC_MODEL_NX; ++row)
    EXPECT_FLOAT_EQ(jacobian.A[row][8], 0.0f);

  EXPECT_FLOAT_EQ(jacobian.A[10][11], 1.0f);
  for (int row = 0; row < MPC_MODEL_NX; ++row) {
    if (row != 10) {
      EXPECT_FLOAT_EQ(jacobian.A[row][11], 0.0f);
    }
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
  state.source_plant_state[UNITY_SOURCE_U] = state.u;
  state.source_plant_state[UNITY_SOURCE_V] = state.v;
  state.source_plant_state[UNITY_SOURCE_YAW_RATE] = state.r;
  state.source_plant_state[UNITY_SOURCE_BODY_HEIGHT] = 0.00464804098;
  state.source_plant_state[UNITY_SOURCE_PHYSICAL_STEERING] =
      state.actual_steering_angle;
  for (int wheel = 0; wheel < 4; ++wheel) {
    state.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] =
        state.u / 0.059;
  }
  state.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RL] += 24.0;
  state.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RR] += 18.0;

  const MpcModelControl_t control{0.0f, 0.0f};
  constexpr float dt = 0.025f;
  MpcStageResult_t nominal{};
  MpcStageLinearization_t jacobian{};
  ASSERT_TRUE(mpc_vehicle_model_step_with_jacobian(
      &state, &control, dt, 0.12f, &nominal, &jacobian));

  for (int wheel = 0; wheel < 4; ++wheel) {
    MpcModelState_t perturbed = state;
    constexpr float epsilon = 1.0e-2f;
    perturbed.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] +=
        epsilon;
    const MpcStageResult_t changed = mpc_vehicle_model_step(
        &perturbed, &control, dt, 0.12f);
    ASSERT_TRUE(changed.valid);
    const float expected = static_cast<float>(
        (changed.next.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] -
         nominal.next.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel]) /
        epsilon);
    EXPECT_NEAR(jacobian.A[12 + wheel][12 + wheel], expected, 2.0e-3f)
        << "wheel index " << wheel;
    EXPECT_TRUE(std::isfinite(jacobian.A[12 + wheel][12 + wheel]));
  }
}

TEST(VehicleModelSourceRollout, KeepsTheRealTimeRTIStateCompact)
{
  EXPECT_EQ(MPC_MODEL_NX, 16);
  EXPECT_EQ(MPC_RTI_PLANT_NX, MPC_MODEL_NX);
  EXPECT_EQ(MPC_RTI_NX, MPC_RTI_PLANT_NX + MPC_RTI_NU);

  MpcModelState_t state = turning_state();
  const MpcModelControl_t control{0.0f, 0.0f};
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);
  ASSERT_TRUE(stage.next.source_plant_initialized);
  EXPECT_NE(stage.next.source_plant_state[UNITY_SOURCE_ROLL], 0.0);
  EXPECT_NE(stage.next.source_plant_state[UNITY_SOURCE_PITCH], 0.0);
  EXPECT_NE(stage.next.source_plant_state[UNITY_SOURCE_BODY_HEIGHT],
            state.source_plant_state[UNITY_SOURCE_BODY_HEIGHT]);
  EXPECT_TRUE(stage.next.source_suspension_history_initialized);
}

}  // namespace

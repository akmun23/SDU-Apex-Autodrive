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
  state.target_speed = 6.0f;
  state.steering_command = 0.35f;
  state.delayed_steering_command_1 = 0.35f;
  state.delayed_steering_command_2 = 0.35f;
  state.actual_steering_angle = 0.35f;
  state.delayed_throttle_command_1 = 0.30f;
  state.delayed_throttle_command_2 = 0.30f;
  state.source_throttle_command = 0.30f;
  state.source_last_target_speed = 6.0f;
  state.source_target_initialized = 1;
  state.source_actuator_speed_mps = 6.0f;
  state.source_actuator_speed_initialized = 1;
  state.source_hold_approach = 1;
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
  state.target_speed = 3.0f;
  state.steering_command = steering;
  state.delayed_steering_command_1 = steering;
  state.delayed_steering_command_2 = steering;
  state.actual_steering_angle = steering;
  state.delayed_throttle_command_1 = throttle;
  state.delayed_throttle_command_2 = throttle;
  state.source_throttle_command = throttle;
  state.source_last_target_speed = state.target_speed;
  state.source_target_initialized = 1;
  state.source_actuator_speed_mps = state.u;
  state.source_actuator_speed_initialized = 1;
  state.source_hold_approach = 1;
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
  state.target_speed = 0.0f;
  state.source_throttle_command = 0.2f;
  const MpcModelControl_t control{};
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);
  EXPECT_EQ(stage.next.source_throttle_command, 0.0f);
  EXPECT_NE(stage.branch_flags & MPC_STAGE_ACTIVE_BRAKE, 0U);
  EXPECT_LT(stage.next.u, state.u);
}

TEST(VehicleModelSourceRollout,
     PredictsThrottleLeadWhenEnteringTheConfiguredSpeedHoldBand)
{
  MpcModelState_t state{};
  state.u = 4.5f;
  state.target_speed = 5.0f;
  state.delayed_throttle_command_1 = 0.40f;
  state.delayed_throttle_command_2 = 0.40f;
  state.source_throttle_command = 0.40f;
  state.source_last_target_speed = 5.0f;
  state.source_target_initialized = 1;
  state.source_actuator_speed_mps = 4.5f;
  state.source_actuator_speed_initialized = 1;

  const MpcModelControl_t control{};
  const MpcStageResult_t stage = mpc_vehicle_model_step(
      &state, &control, 0.025f, 0.0f);
  ASSERT_TRUE(stage.valid);

  /* Production predicts the near-term speed crossing from throttle already
   * above its measured-speed feed-forward, even before the filtered odometry
   * derivative has caught up. */
  EXPECT_EQ(stage.next.source_hold_approach, 1);
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
  EXPECT_FLOAT_EQ(delayed.next.delayed_throttle_command_2,
                  delayed.next.source_throttle_command);
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

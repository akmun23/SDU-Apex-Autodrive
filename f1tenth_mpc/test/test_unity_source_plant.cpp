#include "unity_source_plant.hpp"
#include "unity_source_plant_c.h"
#include "mpc_types.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

namespace {

f1tenth_mpc::UnitySourcePlantState state_at_speed(double speed_mps)
{
  f1tenth_mpc::UnitySourcePlantState state;
  state.u_mps = speed_mps;
  // Keep the production state default: it is the active Unity actor height.
  // WheelCollider's native target-position shift is applied by the plant.
  for (double &omega : state.wheel_omega_radps)
    omega = speed_mps / 0.059;
  return state;
}

double raw_curve(double slip, const std::array<double, 5> &curve)
{
  const double magnitude = std::abs(slip);
  double response;
  if (magnitude < curve[0]) {
    const double t = magnitude / curve[0];
    response = curve[1] * (t + t * t - t * t * t);
  } else if (magnitude < curve[2]) {
    const double t = (magnitude - curve[0]) / (curve[2] - curve[0]);
    const double smooth = 3.0 * t * t - 2.0 * t * t * t;
    response = curve[1] + (curve[3] - curve[1]) * smooth;
  } else {
    response = curve[3];
  }
  return std::copysign(response, slip);
}

double expected_yaw_acceleration_from_wheel_forces(
    const f1tenth_mpc::UnitySourcePlantOutput &output)
{
  constexpr double wheel_x[4] = {
    0.17467999458312988, 0.17467999458312988,
    -0.15532000362873077, -0.15532000362873077};
  constexpr double wheel_y[4] = {
    0.11574111133813858, -0.11590110510587692,
    0.11792000383138657, -0.11807999759912491};
  double torque_z = 0.0;
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    const double fx = output.tire_longitudinal_force_n[wheel];
    const double fy = output.tire_lateral_force_n[wheel];
    torque_z += wheel_x[wheel] * fy - wheel_y[wheel] * fx;
  }

  // PxVehicleUpdate::updateNoDrive makes the temporary COM pose rotation
  // identity before integrateBody. Vehicle substeps therefore use the
  // principal mass-space diagonal, not the rotated scene-solver tensor.
  // The practice player's measured Rigidbody inertiaTensor.y is 0.100282848.
  constexpr double inverse_yaw_inertia = 9.971794977342487;
  return inverse_yaw_inertia * torque_z;
}

}  // namespace

TEST(UnitySourcePlant, UsesExactPracticePlayerWheelSubstepSchedule)
{
  // The exact-build native initialization is
  // setSubStepCount(128.0f, 5, 1); PhysX uses the high-speed branch at equality.
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(0.0), 5U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(4.999), 5U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(-4.999), 5U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(5.0), 5U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(12.0), 5U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(127.999), 5U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(-127.999), 5U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(128.0), 1U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(-128.0), 1U);
}

TEST(UnitySourcePlant, ReconcilesJounceAfterBodyStateCorrection)
{
  auto state = state_at_speed(3.2);
  state.body_height_m = 0.0042;
  state.roll_rad = 0.012;
  state.pitch_rad = -0.018;
  state.physical_steering_rad = 0.21;
  state.suspension_history_initialized = true;
  state.suspension_jounce_m = {0.018, -0.018, 0.012, -0.012};
  const auto unchanged_state = state;

  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_reconcile_suspension_jounce(
      &state));

  EXPECT_DOUBLE_EQ(state.u_mps, unchanged_state.u_mps);
  EXPECT_DOUBLE_EQ(state.v_mps, unchanged_state.v_mps);
  EXPECT_DOUBLE_EQ(state.yaw_rate_radps, unchanged_state.yaw_rate_radps);
  EXPECT_DOUBLE_EQ(state.roll_rad, unchanged_state.roll_rad);
  EXPECT_DOUBLE_EQ(state.pitch_rad, unchanged_state.pitch_rad);
  EXPECT_DOUBLE_EQ(state.physical_steering_rad,
                   unchanged_state.physical_steering_rad);
  EXPECT_TRUE(state.suspension_history_initialized);
  EXPECT_NE(state.suspension_jounce_m,
            unchanged_state.suspension_jounce_m);
  for (double jounce : state.suspension_jounce_m) {
    EXPECT_GE(jounce, -0.025);
    EXPECT_LE(jounce, 0.025);
  }
}

TEST(UnitySourcePlant, WheelPoseAngleMatchesNoDriveUpdateWrap)
{
  constexpr double pi = 3.1415926535897932384626433832795;
  constexpr double wrap = 10.0 * pi;
  auto state = state_at_speed(128.0);  // exactly one native substep
  state.wheel_rotation_angle_rad.fill(wrap - 0.001);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &output));

  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    const double unwrapped = state.wheel_rotation_angle_rad[wheel] +
        output.state.wheel_omega_radps[wheel] * 0.001;
    ASSERT_GE(unwrapped, wrap);
    EXPECT_DOUBLE_EQ(output.state.wheel_rotation_angle_rad[wheel],
                     unwrapped - wrap);
    EXPECT_LT(output.state.wheel_rotation_angle_rad[wheel], wrap);
    EXPECT_GE(output.state.wheel_rotation_angle_rad[wheel], -wrap);
  }
}

TEST(UnitySourcePlant, ZeroSlipAtMatchedWheelSurfaceSpeed)
{
  // At 128 m/s the exact native 128/5/1 schedule selects one substep. This
  // keeps the reported slip at the initially matched wheel/body speed before
  // the nonzero torque changes wheel spin.
  const auto state = state_at_speed(128.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &output));
  for (double slip : output.longitudinal_slip)
    EXPECT_NEAR(slip, 0.0, 1.0e-12);
  for (double slip : output.lateral_slip)
    EXPECT_NEAR(slip, 0.0, 1.0e-12);
  EXPECT_TRUE(std::isfinite(output.longitudinal_acceleration_mps2));
  EXPECT_TRUE(std::isfinite(output.lateral_acceleration_mps2));
  EXPECT_TRUE(std::isfinite(output.yaw_acceleration_radps2));
}

TEST(UnitySourcePlant, MinimalStepsMatchFullStateAndMultiStepDiagnostics)
{
  const auto initial = state_at_speed(4.2);
  const f1tenth_mpc::UnitySourcePlantInput input{0.18, 0.35};
  f1tenth_mpc::UnitySourcePlantOutput full_step{}, minimal_step{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      initial, input, &full_step));
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms_minimal(
      initial, input, &minimal_step));

  const auto expect_same_state = [](const auto &a, const auto &b) {
    EXPECT_DOUBLE_EQ(a.u_mps, b.u_mps);
    EXPECT_DOUBLE_EQ(a.v_mps, b.v_mps);
    EXPECT_DOUBLE_EQ(a.yaw_rate_radps, b.yaw_rate_radps);
    EXPECT_DOUBLE_EQ(a.body_height_m, b.body_height_m);
    EXPECT_DOUBLE_EQ(a.body_height_rate_mps, b.body_height_rate_mps);
    EXPECT_DOUBLE_EQ(a.roll_rad, b.roll_rad);
    EXPECT_DOUBLE_EQ(a.roll_rate_radps, b.roll_rate_radps);
    EXPECT_DOUBLE_EQ(a.pitch_rad, b.pitch_rad);
    EXPECT_DOUBLE_EQ(a.pitch_rate_radps, b.pitch_rate_radps);
    EXPECT_DOUBLE_EQ(a.physical_steering_rad, b.physical_steering_rad);
    EXPECT_EQ(a.suspension_history_initialized,
              b.suspension_history_initialized);
    for (std::size_t wheel = 0; wheel < 4; ++wheel) {
      EXPECT_DOUBLE_EQ(a.wheel_omega_radps[wheel],
                       b.wheel_omega_radps[wheel]);
      EXPECT_DOUBLE_EQ(a.wheel_rotation_angle_rad[wheel],
                       b.wheel_rotation_angle_rad[wheel]);
      EXPECT_DOUBLE_EQ(a.suspension_jounce_m[wheel],
                       b.suspension_jounce_m[wheel]);
      EXPECT_DOUBLE_EQ(a.low_forward_speed_time_s[wheel],
                       b.low_forward_speed_time_s[wheel]);
      EXPECT_DOUBLE_EQ(a.low_side_speed_time_s[wheel],
                       b.low_side_speed_time_s[wheel]);
    }
  };
  expect_same_state(full_step.state, minimal_step.state);
  EXPECT_EQ(full_step.maximum_positive_longitudinal_slip,
            minimal_step.maximum_positive_longitudinal_slip);
  EXPECT_EQ(full_step.maximum_longitudinal_slip_excess_mps,
            minimal_step.maximum_longitudinal_slip_excess_mps);

  f1tenth_mpc::UnitySourcePlantOutput batched{}, explicit_output{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      initial, input, 0.025, &batched));
  auto explicit_state = initial;
  std::array<double, 4> maximum_slip{}, maximum_excess{};
  for (int tick = 0; tick < 25; ++tick) {
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
        explicit_state, input, &explicit_output));
    for (std::size_t wheel = 0; wheel < 4; ++wheel) {
      maximum_slip[wheel] = std::max(maximum_slip[wheel],
          explicit_output.maximum_positive_longitudinal_slip[wheel]);
      maximum_excess[wheel] = std::max(maximum_excess[wheel],
          explicit_output.maximum_longitudinal_slip_excess_mps[wheel]);
    }
    explicit_state = explicit_output.state;
  }
  expect_same_state(explicit_state, batched.state);
  EXPECT_EQ(maximum_slip, batched.maximum_positive_longitudinal_slip);
  EXPECT_EQ(maximum_excess, batched.maximum_longitudinal_slip_excess_mps);
  EXPECT_EQ(explicit_output.longitudinal_slip, batched.longitudinal_slip);
  EXPECT_EQ(explicit_output.lateral_slip, batched.lateral_slip);
  EXPECT_DOUBLE_EQ(explicit_output.longitudinal_acceleration_mps2,
                   batched.longitudinal_acceleration_mps2);
  EXPECT_DOUBLE_EQ(explicit_output.lateral_acceleration_mps2,
                   batched.lateral_acceleration_mps2);
  EXPECT_DOUBLE_EQ(explicit_output.yaw_acceleration_radps2,
                   batched.yaw_acceleration_radps2);
}

TEST(UnitySourcePlant, ForwardSensitivitiesMatchSmoothBranchDifferences)
{
  auto state = state_at_speed(6.2);
  state.v_mps = 0.14;
  state.yaw_rate_radps = 0.72;
  state.physical_steering_rad = 0.24;
  state.suspension_history_initialized = true;
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    state.wheel_omega_radps[wheel] =
        (state.u_mps + (wheel >= 2 ? 0.12 : 0.0)) / 0.059;
    state.suspension_jounce_m[wheel] = 0.0;
  }
  // Keep the target far enough away that the per-1ms steering slew does not
  // land exactly on its clamp in the last integration sample.
  const f1tenth_mpc::UnitySourcePlantInput input{0.50, 0.42};
  constexpr double dt = 0.025;
  constexpr double epsilon[] = {1.0e-3, 1.0e-3, 1.0e-2, 1.0e-4};
  f1tenth_mpc::UnitySourcePlantSensitivitySeeds seeds{};
  seeds.state[UNITY_SOURCE_U][0] = 1.0;
  seeds.state[UNITY_SOURCE_YAW_RATE][1] = 1.0;
  seeds.state[UNITY_SOURCE_WHEEL_OMEGA_RL][2] = 1.0;
  seeds.state[UNITY_SOURCE_PHYSICAL_STEERING][3] = 1.0;
  f1tenth_mpc::UnitySourcePlantSensitivity sensitivity{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_with_sensitivities(
      state, input, dt, seeds, &sensitivity));

  for (std::size_t direction = 0; direction < 4; ++direction) {
    auto plus_state = state;
    auto minus_state = state;
    switch (direction) {
      case 0:
        plus_state.u_mps += epsilon[direction];
        minus_state.u_mps -= epsilon[direction];
        break;
      case 1:
        plus_state.yaw_rate_radps += epsilon[direction];
        minus_state.yaw_rate_radps -= epsilon[direction];
        break;
      case 2:
        plus_state.wheel_omega_radps[2] += epsilon[direction];
        minus_state.wheel_omega_radps[2] -= epsilon[direction];
        break;
      case 3:
        plus_state.physical_steering_rad += epsilon[direction];
        minus_state.physical_steering_rad -= epsilon[direction];
        break;
    }
    f1tenth_mpc::UnitySourcePlantOutput plus{}, minus{};
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        plus_state, input, dt, &plus));
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        minus_state, input, dt, &minus));
    const double denominator = 2.0 * epsilon[direction];
    const auto check = [&](std::size_t output_index,
                           double plus_value, double minus_value) {
      const double finite_difference = (plus_value - minus_value) / denominator;
      EXPECT_NEAR(sensitivity.state[output_index][direction],
                  finite_difference,
                  2.0e-4 + 2.0e-3 * std::abs(finite_difference))
          << "direction=" << direction << " output=" << output_index;
    };
    check(UNITY_SOURCE_U, plus.state.u_mps, minus.state.u_mps);
    check(UNITY_SOURCE_V, plus.state.v_mps, minus.state.v_mps);
    check(UNITY_SOURCE_YAW_RATE,
          plus.state.yaw_rate_radps, minus.state.yaw_rate_radps);
    check(UNITY_SOURCE_WHEEL_OMEGA_RL,
          plus.state.wheel_omega_radps[2], minus.state.wheel_omega_radps[2]);
    const double d_forward =
        (plus.integrated_body_forward_displacement_m -
         minus.integrated_body_forward_displacement_m) / denominator;
    const double d_lateral =
        (plus.integrated_body_lateral_displacement_m -
         minus.integrated_body_lateral_displacement_m) / denominator;
    const double d_yaw = (plus.integrated_yaw_delta_rad -
                          minus.integrated_yaw_delta_rad) / denominator;
    EXPECT_NEAR(sensitivity.body_forward_displacement[direction], d_forward,
                2.0e-5 + 2.0e-3 * std::abs(d_forward));
    EXPECT_NEAR(sensitivity.body_lateral_displacement[direction], d_lateral,
                2.0e-5 + 2.0e-3 * std::abs(d_lateral));
    EXPECT_NEAR(sensitivity.yaw_delta[direction], d_yaw,
                2.0e-5 + 2.0e-3 * std::abs(d_yaw));
  }
}

TEST(UnitySourcePlant, ForwardSensitivitiesSupportUpperEightLanes)
{
  auto state = state_at_speed(6.2);
  state.v_mps = 0.08;
  state.yaw_rate_radps = 0.24;
  state.physical_steering_rad = 0.18;
  state.roll_rad = 0.01;
  state.body_height_m = 0.00464804098;
  state.suspension_history_initialized = true;
  for (std::size_t wheel = 0; wheel < 4; ++wheel)
    state.wheel_omega_radps[wheel] = state.u_mps / 0.059;

  constexpr double epsilon[4] = {2.0e-6, 2.0e-6, 1.0e-4, 1.0e-3};
  const f1tenth_mpc::UnitySourcePlantInput input{0.23, 0.42};
  f1tenth_mpc::UnitySourcePlantSensitivitySeeds seeds{};
  seeds.state[UNITY_SOURCE_BODY_HEIGHT][4] = 1.0;
  seeds.state[UNITY_SOURCE_ROLL][5] = 1.0;
  seeds.steering_target[6] = 1.0;
  seeds.throttle[7] = 1.0;
  f1tenth_mpc::UnitySourcePlantSensitivity sensitivity{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_with_sensitivities(
      state, input, 0.025, seeds, &sensitivity));

  for (std::size_t lane = 0; lane < 4; ++lane) {
    const std::size_t direction = lane + 4;
    auto plus_state = state;
    auto minus_state = state;
    auto plus_input = input;
    auto minus_input = input;
    if (lane == 0) {
      plus_state.body_height_m += epsilon[lane];
      minus_state.body_height_m -= epsilon[lane];
    } else if (lane == 1) {
      plus_state.roll_rad += epsilon[lane];
      minus_state.roll_rad -= epsilon[lane];
    } else if (lane == 2) {
      plus_input.steering_target_rad += epsilon[lane];
      minus_input.steering_target_rad -= epsilon[lane];
    } else {
      plus_input.throttle_normalized += epsilon[lane];
      minus_input.throttle_normalized -= epsilon[lane];
    }
    f1tenth_mpc::UnitySourcePlantOutput plus{}, minus{};
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        plus_state, plus_input, 0.025, &plus));
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        minus_state, minus_input, 0.025, &minus));
    const double denominator = 2.0 * epsilon[lane];
    const auto check = [&](std::size_t output_index,
                           double plus_value, double minus_value) {
      const double finite_difference =
          (plus_value - minus_value) / denominator;
      EXPECT_NEAR(sensitivity.state[output_index][direction], finite_difference,
          2.0e-5 + 2.0e-3 * std::abs(finite_difference))
          << "upper lane=" << lane << " output=" << output_index;
    };
    check(UNITY_SOURCE_U, plus.state.u_mps, minus.state.u_mps);
    check(UNITY_SOURCE_V, plus.state.v_mps, minus.state.v_mps);
    check(UNITY_SOURCE_YAW_RATE,
          plus.state.yaw_rate_radps, minus.state.yaw_rate_radps);
    const double forward_difference =
        (plus.integrated_body_forward_displacement_m -
         minus.integrated_body_forward_displacement_m) / denominator;
    EXPECT_NEAR(sensitivity.body_forward_displacement[direction],
        forward_difference, 2.0e-5 + 2.0e-3 * std::abs(forward_difference));
  }
}

TEST(UnitySourcePlant, ForwardSensitivitiesCoverAttitudeAndJounceStates)
{
  auto state = state_at_speed(6.2);
  state.v_mps = 0.08;
  state.yaw_rate_radps = 0.24;
  state.physical_steering_rad = 0.18;
  state.roll_rad = 0.01;
  state.roll_rate_radps = 0.025;
  state.pitch_rad = -0.006;
  state.pitch_rate_radps = 0.018;
  state.body_height_m = 0.00464804098;
  state.suspension_history_initialized = true;
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    state.wheel_omega_radps[wheel] = state.u_mps / 0.059;
    state.suspension_jounce_m[wheel] = 0.0;
  }
  const f1tenth_mpc::UnitySourcePlantInput input{0.23, 0.42};
  constexpr double epsilon = 2.0e-6;
  f1tenth_mpc::UnitySourcePlantSensitivitySeeds seeds{};
  seeds.state[UNITY_SOURCE_BODY_HEIGHT][0] = 1.0;
  seeds.state[UNITY_SOURCE_ROLL][1] = 1.0;
  seeds.state[UNITY_SOURCE_PITCH][2] = 1.0;
  seeds.state[UNITY_SOURCE_SUSPENSION_JOUNCE_RL][3] = 1.0;
  f1tenth_mpc::UnitySourcePlantSensitivity sensitivity{};
  ASSERT_TRUE(unity_source_plant_step_with_sensitivities(
      state, input, 0.025, seeds, &sensitivity));

  const std::size_t input_state[] = {
      UNITY_SOURCE_BODY_HEIGHT, UNITY_SOURCE_ROLL, UNITY_SOURCE_PITCH,
      UNITY_SOURCE_SUSPENSION_JOUNCE_RL};
  for (std::size_t lane = 0; lane < 4; ++lane) {
    auto plus_state = state;
    auto minus_state = state;
    const auto perturb = [&](f1tenth_mpc::UnitySourcePlantState *target,
                             double amount) {
      switch (input_state[lane]) {
        case UNITY_SOURCE_BODY_HEIGHT: target->body_height_m += amount; break;
        case UNITY_SOURCE_ROLL: target->roll_rad += amount; break;
        case UNITY_SOURCE_PITCH: target->pitch_rad += amount; break;
        case UNITY_SOURCE_SUSPENSION_JOUNCE_RL:
          target->suspension_jounce_m[2] += amount;
          break;
      }
    };
    perturb(&plus_state, epsilon);
    perturb(&minus_state, -epsilon);
    f1tenth_mpc::UnitySourcePlantOutput plus{}, minus{};
    ASSERT_TRUE(unity_source_plant_step(plus_state, input, 0.025, &plus));
    ASSERT_TRUE(unity_source_plant_step(minus_state, input, 0.025, &minus));
    const auto check = [&](std::size_t output_state,
                           double plus_value, double minus_value) {
      const double finite_difference =
          (plus_value - minus_value) / (2.0 * epsilon);
      EXPECT_NEAR(sensitivity.state[output_state][lane], finite_difference,
          2.0e-5 + 2.0e-4 * std::abs(finite_difference))
          << "lane=" << lane << " output=" << output_state;
    };
    check(UNITY_SOURCE_U, plus.state.u_mps, minus.state.u_mps);
    check(UNITY_SOURCE_V, plus.state.v_mps, minus.state.v_mps);
    check(UNITY_SOURCE_YAW_RATE,
          plus.state.yaw_rate_radps, minus.state.yaw_rate_radps);
    check(UNITY_SOURCE_WHEEL_OMEGA_FL,
          plus.state.wheel_omega_radps[0], minus.state.wheel_omega_radps[0]);
    check(UNITY_SOURCE_ROLL_RATE,
          plus.state.roll_rate_radps, minus.state.roll_rate_radps);
  }
}

TEST(UnitySourcePlant, ForwardSensitivitiesMatchAllFourJounceDirections)
{
  auto state = state_at_speed(6.0);
  state.v_mps = 0.08 + 0.24 * 0.15532;
  state.yaw_rate_radps = 0.24;
  state.physical_steering_rad = 0.18;
  state.body_height_m = 0.00464804098;
  state.roll_rad = 0.01;
  state.roll_rate_radps = 0.025;
  state.pitch_rad = -0.006;
  state.pitch_rate_radps = 0.018;
  state.suspension_history_initialized = true;
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    state.wheel_omega_radps[wheel] = state.u_mps / 0.059;
    state.suspension_jounce_m[wheel] = 0.0;
  }
  const f1tenth_mpc::UnitySourcePlantInput input{0.23, 0.42};
  constexpr double epsilon = 2.0e-6;
  f1tenth_mpc::UnitySourcePlantSensitivitySeeds seeds{};
  for (std::size_t wheel = 0; wheel < 4; ++wheel)
    seeds.state[UNITY_SOURCE_SUSPENSION_JOUNCE_FL + wheel][wheel] = 1.0;
  f1tenth_mpc::UnitySourcePlantSensitivity sensitivity{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_with_sensitivities(
      state, input, 0.025, seeds, &sensitivity));

  for (std::size_t lane = 0; lane < 4; ++lane) {
    auto plus_state = state;
    auto minus_state = state;
    plus_state.suspension_jounce_m[lane] += epsilon;
    minus_state.suspension_jounce_m[lane] -= epsilon;
    f1tenth_mpc::UnitySourcePlantOutput plus{}, minus{};
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        plus_state, input, 0.025, &plus));
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        minus_state, input, 0.025, &minus));
    const auto check = [&](std::size_t output_index,
                           double plus_value, double minus_value) {
      const double finite_difference =
          (plus_value - minus_value) / (2.0 * epsilon);
      EXPECT_NEAR(sensitivity.state[output_index][lane], finite_difference,
          2.0e-5 + 2.0e-4 * std::abs(finite_difference))
          << "wheel=" << lane << " output=" << output_index;
    };
    check(UNITY_SOURCE_U, plus.state.u_mps, minus.state.u_mps);
    check(UNITY_SOURCE_V, plus.state.v_mps, minus.state.v_mps);
    check(UNITY_SOURCE_YAW_RATE,
          plus.state.yaw_rate_radps, minus.state.yaw_rate_radps);
    check(UNITY_SOURCE_WHEEL_OMEGA_FL,
          plus.state.wheel_omega_radps[0], minus.state.wheel_omega_radps[0]);
    check(UNITY_SOURCE_WHEEL_OMEGA_RR,
          plus.state.wheel_omega_radps[3], minus.state.wheel_omega_radps[3]);
    check(UNITY_SOURCE_ROLL_RATE,
          plus.state.roll_rate_radps, minus.state.roll_rate_radps);
  }
}

TEST(UnitySourcePlant, SlipExcessMatchesSourceCurveAndHasCorrectWheelDerivative)
{
  auto state = state_at_speed(8.0);
  state.wheel_omega_radps[2] = 250.0;
  std::array<double, 4> slip{}, excess{}, derivative{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_slip_excess(
      state, &slip, &excess, &derivative));
  EXPECT_GT(slip[2], UNITY_SOURCE_FORWARD_EXTREMUM_SLIP);
  EXPECT_GT(excess[2], 0.0);
  EXPECT_NEAR(excess[2], (slip[2] - UNITY_SOURCE_FORWARD_EXTREMUM_SLIP) *
      (std::max(std::abs(state.u_mps),
                std::abs(state.wheel_omega_radps[2] * 0.059)) + 0.1),
      1.0e-12);

  constexpr double epsilon = 1.0e-4;
  auto perturbed = state;
  perturbed.wheel_omega_radps[2] += epsilon;
  std::array<double, 4> perturbed_slip{}, perturbed_excess{},
      perturbed_derivative{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_slip_excess(
      perturbed, &perturbed_slip, &perturbed_excess,
      &perturbed_derivative));
  EXPECT_NEAR(derivative[2], (perturbed_excess[2] - excess[2]) / epsilon,
      1.0e-8);

  state.wheel_omega_radps[2] = 5.0 / 0.059;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_slip_excess(
      state, &slip, &excess, &derivative));
  EXPECT_LT(slip[2], -UNITY_SOURCE_FORWARD_EXTREMUM_SLIP);
  EXPECT_GT(excess[2], 0.0);
}

TEST(UnitySourcePlant, SuspensionJounceHistoryInitializesAndAdvances)
{
  auto state = state_at_speed(5.0);
  // Move the collider ray farther into suspension travel after accounting
  // for WheelCollider.center.
  state.body_height_m = -0.025;
  f1tenth_mpc::UnitySourcePlantOutput first;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &first));
  EXPECT_TRUE(first.state.suspension_history_initialized);
  for (double jounce : first.state.suspension_jounce_m)
    EXPECT_TRUE(std::isfinite(jounce));

  auto displaced = first.state;
  displaced.body_height_m += 0.001;
  f1tenth_mpc::UnitySourcePlantOutput second;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      displaced, {0.0, 0.25}, &second));
  EXPECT_TRUE(second.state.suspension_history_initialized);
  EXPECT_NE(second.state.suspension_jounce_m[0],
            first.state.suspension_jounce_m[0]);
  for (double load : second.normal_load_n)
    EXPECT_TRUE(std::isfinite(load));
}

TEST(UnitySourcePlant, PhysXAppliesNoWheelForceBeyondMaximumDroop)
{
  // This fixture places all four ray intersections beyond max droop. PhysX
  // reports jounce=-maxDroop and leaves the initialized force/slip outputs at
  // zero; a positive spring or filtered tire load here is a model error.
  const auto state = state_at_speed(4.0);
  auto airborne = state;
  airborne.body_height_m = 0.045;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      airborne, {0.0, 0.25}, &output));
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(output.state.suspension_jounce_m[wheel], -0.025);
    EXPECT_DOUBLE_EQ(output.suspension_force_n[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.normal_load_n[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.tire_longitudinal_force_n[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.tire_lateral_force_n[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.longitudinal_slip[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.lateral_slip[wheel], 0.0);
  }
  // No wheel force is generated, but PhysX scene-level Rigidbody drag still
  // applies once per fixed step to an airborne body.
  constexpr double rigidbody_drag_per_s = 0.273;
  EXPECT_NEAR(output.longitudinal_acceleration_mps2,
              -rigidbody_drag_per_s * state.u_mps, 1.0e-9);
  EXPECT_NEAR(output.lateral_acceleration_mps2, 0.0, 1.0e-12);
  EXPECT_NEAR(output.yaw_acceleration_radps2, 0.0, 1.0e-12);
}

TEST(UnitySourcePlant, BrakeLockedWheelReportsZeroForwardSlipAtLowSpeed)
{
  auto state = state_at_speed(0.15);
  state.wheel_omega_radps.fill(0.0);
  state.suspension_history_initialized = true;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(output.state.wheel_omega_radps[wheel], 0.0);
    // At this source-defined low speed, locked wheels and sticky-tire
    // handling produce zero longitudinal slip under hard braking.
    EXPECT_DOUBLE_EQ(output.longitudinal_slip[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.tire_longitudinal_force_n[wheel], 0.0);
  }
}

TEST(UnitySourcePlant, StickyForwardConstraintUsesSourceSpeedBoundary)
{
  auto above = state_at_speed(0.2005);
  above.wheel_omega_radps.fill(0.0);
  above.suspension_history_initialized = true;
  f1tenth_mpc::UnitySourcePlantOutput above_output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      above, {0.0, 0.0}, &above_output));
  for (bool active : above_output.sticky_forward_active)
    EXPECT_FALSE(active);

  auto below = state_at_speed(0.1995);
  below.wheel_omega_radps.fill(0.0);
  below.suspension_history_initialized = true;
  f1tenth_mpc::UnitySourcePlantOutput below_output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      below, {0.0, 0.0}, &below_output));
  for (bool active : below_output.sticky_forward_active)
    EXPECT_TRUE(active);
}

TEST(UnitySourcePlant, StraightLockedWheelBrakeDoesNotCreateYawImpulse)
{
  auto state = state_at_speed(0.15);
  state.wheel_omega_radps.fill(0.0);
  state.suspension_history_initialized = true;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  EXPECT_LT(std::abs(output.state.yaw_rate_radps), 3.0e-3);
}

TEST(UnitySourcePlant, LowSideTimerResetsWhenDriverIntendsToAccelerate)
{
  auto state = state_at_speed(0.0);
  state.wheel_omega_radps.fill(0.0);
  state.low_side_speed_time_s.fill(0.5);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.2}, &output));
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(output.state.low_forward_speed_time_s[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.state.low_side_speed_time_s[wheel], 0.0);
  }
}

TEST(UnitySourcePlant, StickyTireTimersSurviveMpcInternalStateRoundTrip)
{
  std::array<double, UNITY_SOURCE_PLANT_STATE_SIZE> state{};
  state[UNITY_SOURCE_U] = 0.1;
  state[UNITY_SOURCE_BODY_HEIGHT] = -0.025;
  state[UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_RL] = 0.8;
  state[UNITY_SOURCE_LOW_SIDE_SPEED_TIME_RR] = 0.9;
  UnitySourcePlantInputC input{};
  std::copy(state.begin(), state.end(), input.state);
  input.suspension_history_initialized = 1;
  input.throttle_normalized = 0.0;
  UnitySourcePlantOutputC output{};
  ASSERT_EQ(unity_source_plant_step_c(&input, 0.001, &output), 1);
  EXPECT_GT(output.state[UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_RL], 0.8);
  EXPECT_GT(output.state[UNITY_SOURCE_LOW_SIDE_SPEED_TIME_RR], 0.9);
}

TEST(UnitySourcePlant, LowSpeedTireForceYawMirrorsWithSteeringDirection)
{
  // One 1 ms update only changes steering by 0.0032 rad. Use one captured
  // 25 ms control interval so the source steering actuator has an observable
  // input before comparing the mirrored yaw response.
  auto left = state_at_speed(0.12);
  left.wheel_omega_radps.fill(0.0);
  left.physical_steering_rad = 0.15;
  left.suspension_history_initialized = true;
  auto right = left;
  right.physical_steering_rad = -left.physical_steering_rad;

  f1tenth_mpc::UnitySourcePlantOutput left_output, right_output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      left, {left.physical_steering_rad, 0.0}, 0.025, &left_output));
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      right, {right.physical_steering_rad, 0.0}, 0.025, &right_output));
  EXPECT_GT(std::abs(left_output.state.yaw_rate_radps), 1.0e-4);
  // The practice player's measured inertiaTensorRotation is non-identity;
  // mirrored forces are not expected to have identical magnitudes after the
  // scene solver applies the rotated mass-frame tensor. Steering reversal
  // must still reverse the yaw response.
  EXPECT_LT(left_output.state.yaw_rate_radps *
                right_output.state.yaw_rate_radps,
            0.0);
}

TEST(UnitySourcePlant, LeftWheelSupportProducesPositiveSourceFrameRoll)
{
  auto state = state_at_speed(5.0);
  state.suspension_history_initialized = true;
  const double front_jounce = 0.025 -
      (state.body_height_m + 0.08490486745229364 - 0.059);
  const double rear_jounce = 0.025 -
      (state.body_height_m + 0.075 - 0.059);
  state.suspension_jounce_m = {
      front_jounce - 0.001, front_jounce,
      rear_jounce - 0.001, rear_jounce};
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.2}, &output));
  EXPECT_GT(output.state.roll_rate_radps, state.roll_rate_radps);
}

TEST(UnitySourcePlant, PxVehicleSubstepUsesPrincipalInertiaWithoutGyroTerm)
{
  f1tenth_mpc::UnitySourcePlantState state;
  state.u_mps = 128.0;  // native setSubStepCount equality selects one step
  state.roll_rate_radps = 1.0;
  state.v_mps = state.roll_rate_radps *
      -(state.body_height_m + 0.06434);
  for (double &omega : state.wheel_omega_radps) omega = state.u_mps / 0.059;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  const double force_derived_yaw_acceleration =
      expected_yaw_acceleration_from_wheel_forces(output);
  const double expected_rate =
      (state.yaw_rate_radps + 0.001 * force_derived_yaw_acceleration) *
      (1.0 - 0.1 * 0.001);
  EXPECT_NEAR(output.state.yaw_rate_radps, expected_rate, 1.0e-10);
  EXPECT_NEAR(output.yaw_acceleration_radps2,
      (output.state.yaw_rate_radps - state.yaw_rate_radps) / 0.001,
      1.0e-10);
  EXPECT_NEAR(force_derived_yaw_acceleration, 0.0, 1.0e-12);
}

TEST(UnitySourcePlant, RigidbodyAngularUpdateUsesTorqueThenSceneDamping)
{
  auto state = state_at_speed(128.0);  // single vehicle substep at equality
  state.yaw_rate_radps = 0.75;
  state.suspension_history_initialized = true;

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.2}, &output));  // drive intent disables sticky rows

  // PxVehicle integrates angular velocity from the wheel-force torque first;
  // PhysX scene pre-integration then applies Rigidbody angular damping before
  // constraints. At 128 m/s the model uses one vehicle substep, and drive intent
  // disables sticky rows, so this checks the exact ordered update.
  EXPECT_NEAR(output.yaw_acceleration_radps2,
      (output.state.yaw_rate_radps - state.yaw_rate_radps) / 0.001,
      1.0e-12);
}

TEST(UnitySourcePlant, GyroBodyRatesAdvanceTiltUsingEulerKinematics)
{
  auto state = state_at_speed(128.0);  // one native vehicle substep
  state.roll_rad = 0.20;
  state.pitch_rad = -0.15;
  state.roll_rate_radps = 0.30;
  state.pitch_rate_radps = -0.40;
  state.yaw_rate_radps = 2.20;
  state.physical_steering_rad = 0.10;

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {state.physical_steering_rad, 0.0}, &output));

  // The production plant applies Rigidbody angular damping after the
  // substep's pose integration. Undo that final 1 ms damping to recover the
  // body rates used by the semi-implicit pose update.
  constexpr double dt = 0.001;
  const double undamped_p = output.state.roll_rate_radps / (1.0 - 0.1 * dt);
  const double undamped_q = output.state.pitch_rate_radps / (1.0 - 0.1 * dt);
  const double undamped_r = output.state.yaw_rate_radps / (1.0 - 0.1 * dt);
  const double expected_roll_rate = undamped_p +
      std::tan(state.pitch_rad) *
          (std::sin(state.roll_rad) * undamped_q +
           std::cos(state.roll_rad) * undamped_r);
  const double expected_pitch_rate =
      std::cos(state.roll_rad) * undamped_q -
      std::sin(state.roll_rad) * undamped_r;

  EXPECT_NEAR((output.state.roll_rad - state.roll_rad) / dt,
              expected_roll_rate, 2.0e-6);
  EXPECT_NEAR((output.state.pitch_rad - state.pitch_rad) / dt,
              expected_pitch_rate, 2.0e-6);
  EXPECT_GT(std::abs(expected_roll_rate - undamped_p), 0.05);
  EXPECT_GT(std::abs(expected_pitch_rate - undamped_q), 0.05);
}

TEST(UnitySourcePlant, ThrottleChangesWheelSpinAndLongitudinalSlip)
{
  const auto state = state_at_speed(5.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      state, {0.0, 0.35}, 0.025, &output));
  EXPECT_GT(output.state.wheel_omega_radps[2], state.wheel_omega_radps[2]);
  EXPECT_GT(output.longitudinal_slip[2], 0.0);
  EXPECT_GT(output.longitudinal_acceleration_mps2, -2.0);
}

TEST(UnitySourcePlant, WheelSpinUsesPhysXNoDriveImplicitUpdate)
{
  // At 128 m/s PxVehicleNoDrive selects one vehicle substep per 1 ms update.
  // Check the production wheel state against integrateNoDriveWheelSpeeds:
  // (omega + dt/I * (tireTorque + driveTorque)) /
  // (1 + dampingRate * dt/I).
  constexpr double wheel_radius_m = 0.059;
  constexpr double wheel_mass_kg = 0.109;
  constexpr double wheel_moi_kgm2 =
      0.5 * wheel_mass_kg * wheel_radius_m * wheel_radius_m;
  constexpr double wheel_damping_rate = 0.25;
  constexpr double motor_torque_nm = 428.0;
  constexpr double throttle = 0.2;
  constexpr double dt = 0.001;
  const auto state = state_at_speed(128.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, throttle}, &output));

  const double tire_torque_nm =
      -wheel_radius_m * output.tire_longitudinal_force_n[2];
  const double per_wheel_drive_torque_nm =
      motor_torque_nm * throttle / 4.0;
  const double dt_over_moi = dt / wheel_moi_kgm2;
  const double expected_omega =
      (state.wheel_omega_radps[2] + dt_over_moi *
          (tire_torque_nm + per_wheel_drive_torque_nm)) /
      (1.0 + wheel_damping_rate * dt_over_moi);
  EXPECT_NEAR(output.state.wheel_omega_radps[2], expected_omega, 1.0e-10);
}

TEST(UnitySourcePlant, ZeroThrottleUsesFullBrakeTorque)
{
  auto state = state_at_speed(5.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  for (double omega : output.state.wheel_omega_radps)
    EXPECT_NEAR(omega, 0.0, 1.0e-12);
}

TEST(UnitySourcePlant, HardBrakeReverseTransientStaysInsidePlantSweepBound)
{
  constexpr double speeds_mps[] = {0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
  double worst_reverse_mps = 0.0;
  double worst_start_speed_mps = 0.0;
  int worst_tick = -1;
  for (double speed : speeds_mps) {
    auto state = state_at_speed(speed);
    for (int tick = 0; tick < 3000; ++tick) {
      f1tenth_mpc::UnitySourcePlantOutput output;
      ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
          state, {0.0, 0.0}, &output));
      state = output.state;
      if (state.u_mps < worst_reverse_mps) {
        worst_reverse_mps = state.u_mps;
        worst_start_speed_mps = speed;
        worst_tick = tick;
      }
      if (tick > 250 && state.u_mps == 0.0) break;
    }
  }
  EXPECT_GE(worst_reverse_mps, -MPC_PHYSX_BRAKE_REVERSE_TRANSIENT_MPS)
      << "native 1 ms PhysX sweep reached -0.00779 m/s over 0.5--12 m/s; "
      << "plant minimum occurred from " << worst_start_speed_mps
      << " m/s at 1 ms tick " << worst_tick;
}

TEST(UnitySourcePlant, SteeringGeneratesLateralForceAndYaw)
{
  const auto state = state_at_speed(5.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      state, {0.25, 0.2}, 0.025, &output));
  EXPECT_GT(output.lateral_acceleration_mps2, 0.0);
  EXPECT_GT(output.yaw_acceleration_radps2, 0.0);
  // Positive-left steering gives negative lateral contact velocity in the
  // wheel's local right/forward convention. WheelHit reports the opposite
  // sign, while the force still opposes the actual contact velocity.
  EXPECT_GT(output.lateral_slip[0], 0.0);
  EXPECT_GT(output.tire_lateral_force_n[0], 0.0);
  EXPECT_NEAR(output.state.physical_steering_rad,
              0.025 * 3.2, 2.0e-4);

  f1tenth_mpc::UnitySourcePlantOutput reverse_turn;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      state, {-0.25, 0.2}, 0.025, &reverse_turn));
  EXPECT_LT(reverse_turn.lateral_acceleration_mps2, 0.0);
  EXPECT_LT(reverse_turn.yaw_acceleration_radps2, 0.0);
  EXPECT_LT(reverse_turn.lateral_slip[0], 0.0);
  EXPECT_LT(reverse_turn.tire_lateral_force_n[0], 0.0);
  EXPECT_NEAR(reverse_turn.state.physical_steering_rad,
              -0.025 * 3.2, 2.0e-4);
}

TEST(UnitySourcePlant, InertialAccelerationIncludesBodyFrameYawTransport)
{
  auto state = state_at_speed(3.5);
  state.v_mps = -0.35;
  state.yaw_rate_radps = -2.4;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.15}, &output));

  const double expected_lateral_acceleration =
      (output.state.v_mps - state.v_mps) / 0.001 +
      0.5 * (state.yaw_rate_radps + output.state.yaw_rate_radps) *
          0.5 * (state.u_mps + output.state.u_mps);
  EXPECT_NEAR(output.lateral_acceleration_mps2,
              expected_lateral_acceleration, 1.0e-5);
  const double body_coordinate_derivative =
      (output.state.v_mps - state.v_mps) / 0.001;
  EXPECT_GT(std::abs(output.lateral_acceleration_mps2 -
                     body_coordinate_derivative), 5.0);
}

TEST(UnitySourcePlant, NormalizedCommandUsesPhysicalSteeringSign)
{
  constexpr double max_steering_rad = 0.52359877559829887308;
  const double positive_target =
      f1tenth_mpc::unity_steering_command_normalized_to_physical_rad(
          0.5, max_steering_rad);
  const double negative_target =
      f1tenth_mpc::unity_steering_command_normalized_to_physical_rad(
          -0.5, max_steering_rad);
  EXPECT_GT(positive_target, 0.0);
  EXPECT_LT(negative_target, 0.0);
  EXPECT_DOUBLE_EQ(positive_target, -negative_target);

  const auto initial = state_at_speed(5.0);
  f1tenth_mpc::UnitySourcePlantOutput left, right;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      initial, {positive_target, 0.2}, 0.025, &left));
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      initial, {negative_target, 0.2}, 0.025, &right));
  EXPECT_GT(left.state.physical_steering_rad, 0.0);
  EXPECT_LT(right.state.physical_steering_rad, 0.0);
  EXPECT_GT(left.yaw_acceleration_radps2, 0.0);
  EXPECT_LT(right.yaw_acceleration_radps2, 0.0);
}

TEST(UnitySourcePlant, LateralSlipUsesPhysXToleranceScaleFloor)
{
  auto state = state_at_speed(128.0);  // isolate one substep's input slip
  state.v_mps = 0.24;
  state.yaw_rate_radps = 0.55;
  state.physical_steering_rad = 0.18;

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.18, 0.25}, &output));

  const double wheel_forward = output.contact_forward_speed_mps[0];
  const double wheel_lateral = output.contact_lateral_speed_mps[0];
  const double source_slip = std::atan(
      wheel_lateral / (std::abs(wheel_forward) + 1.0));
  const double former_candidate_slip = std::atan(
      wheel_lateral / (std::abs(wheel_forward) + 0.1));

  EXPECT_NEAR(output.lateral_slip[0], -source_slip, 1.0e-12);
  EXPECT_GT(std::abs(output.lateral_slip[0] + former_candidate_slip), 1.0e-3);
}

TEST(UnitySourcePlant, WheelSlipUsesFullAngularVelocityAtContactPoint)
{
  constexpr double wheel_radius = 0.059;
  constexpr double com_height_from_actor = 0.06434;
  constexpr double forward_speed = 128.0;  // one native substep at threshold
  auto state = state_at_speed(forward_speed);
  const double contact_z = -(state.body_height_m + com_height_from_actor);
  state.roll_rate_radps = 0.4;
  f1tenth_mpc::UnitySourcePlantOutput roll_output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &roll_output));
  const double roll_contact_lateral_speed =
      -state.roll_rate_radps * contact_z;
  const double roll_contact_forward_speed =
      std::abs(roll_output.contact_forward_speed_mps[0]);
  EXPECT_NEAR(roll_output.lateral_slip[0],
      -std::atan(roll_contact_lateral_speed /
                 (roll_contact_forward_speed + 1.0)), 1.0e-10);

  state = state_at_speed(forward_speed);
  const double pitch_contact_z =
      -(state.body_height_m + com_height_from_actor);
  state.pitch_rate_radps = 0.4;
  f1tenth_mpc::UnitySourcePlantOutput pitch_output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &pitch_output));
  const double pitch_contact_forward_speed =
      forward_speed + state.pitch_rate_radps * pitch_contact_z;
  const double pitch_contact_forward_abs =
      std::abs(pitch_output.contact_forward_speed_mps[0]);
  const double expected_longitudinal_slip =
      (forward_speed - pitch_contact_forward_speed) /
      (std::max(forward_speed, pitch_contact_forward_abs) + 0.1);
  EXPECT_NEAR(pitch_output.longitudinal_slip[0],
              expected_longitudinal_slip, 1.0e-10);
  EXPECT_NEAR(state.wheel_omega_radps[0] * wheel_radius,
              forward_speed, 1.0e-12);
}

TEST(UnitySourcePlant, ContactJounceUsesPhysXRayPlaneIntersection)
{
  auto state = state_at_speed(128.0);  // one native substep; ray is initial
  state.body_height_m = -0.025;
  state.roll_rad = 0.035;
  state.pitch_rad = -0.045;
  const double sr = std::sin(state.roll_rad), cr = std::cos(state.roll_rad);
  const double sp = std::sin(state.pitch_rad), cp = std::cos(state.pitch_rad);
  const std::array<double, 3> normal_body{-sp, cp * sr, cp * cr};
  // UnityPlayer::AddWheelToExistingVehicle sets chassis-local (0,-1,0)
  // suspension travel for every wheel; front camber only transforms center.
  const std::array<double, 3> suspension_direction{0.0, 0.0, -1.0};
  const std::array<double, 3> collider_transform{
      0.17467999458312988, 0.11792,
      -0.004435138776898384};
  constexpr double center_y = -0.025 * 0.08715574274765817;
  constexpr double center_z = 0.025 * 0.9961946980917455;
  std::array<double, 3> wheel_center{};
  std::array<double, 3> ray_start{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double center_offset = axis == 1 ? center_y :
        (axis == 2 ? center_z : 0.0);
    // UnityPlayer::ComputeWheelCentreOfMassOffset adds
    // suspensionDir * distance * (1-targetPosition). With target=0.5 this is
    // one max-compression interval along native travel.
    wheel_center[axis] = collider_transform[axis] + center_offset +
        suspension_direction[axis] * 0.025;
    ray_start[axis] = wheel_center[axis] -
        suspension_direction[axis] * (0.059 + 0.025);
  }
  const std::array<double, 3> com_offset{-0.00468, 0.00008, 0.06434};
  const auto dot3 = [](const std::array<double, 3> &a,
                       const std::array<double, 3> &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  const double ray_travel = -(state.body_height_m +
      dot3(normal_body, com_offset) + dot3(normal_body, ray_start)) /
      dot3(normal_body, suspension_direction);
  const double expected_jounce = std::clamp(
      2.0 * 0.059 + 0.025 - ray_travel, -0.025, 0.025);

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &output));
  EXPECT_NEAR(output.state.suspension_jounce_m[0], expected_jounce, 1.0e-10);
}

TEST(UnitySourcePlant, TireAndStickyPhysicsUseSameWheelBottomKinematics)
{
  auto state = state_at_speed(3.5);
  state.body_height_m = -0.025;
  state.v_mps = 0.12;
  state.yaw_rate_radps = -0.2;
  state.physical_steering_rad = -0.25;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(output.contact_forward_speed_mps[wheel],
                     output.sticky_contact_forward_speed_mps[wheel]);
    EXPECT_DOUBLE_EQ(output.contact_lateral_speed_mps[wheel],
                     output.sticky_contact_lateral_speed_mps[wheel]);
  }
}

TEST(UnitySourcePlant, UnityPlayerTireForceCallbackAppliesCombinedSlipCircle)
{
  auto state = state_at_speed(4.0);
  state.v_mps = 0.7;
  state.yaw_rate_radps = 0.45;
  state.physical_steering_rad = 0.21;
  for (double &omega : state.wheel_omega_radps)
    omega += 2.2;

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.4, 0.25}, &output));
  const std::array<double, 5> forward{0.15, 0.9, 0.25, 0.58, 0.8};
  const std::array<double, 5> lateral{0.01, 1.0, 0.1, 0.5, 1.0};
  for (std::size_t i = 0; i < 4; ++i) {
    const double raw_x = raw_curve(output.longitudinal_slip[i], forward);
    const double force_model_slip_y = -output.lateral_slip[i];
    const double raw_y = raw_curve(force_model_slip_y, lateral);
    const double utilization = std::sqrt(std::max(
        1.0, std::pow(raw_x / forward[1], 2.0) +
                 std::pow(raw_y / lateral[1], 2.0)));
    EXPECT_NEAR(output.tire_longitudinal_force_n[i],
        output.normal_load_n[i] * raw_x * forward[4] /
            utilization, 1.0e-10);
    EXPECT_NEAR(output.tire_lateral_force_n[i],
        -output.normal_load_n[i] * raw_y * lateral[4] /
            utilization, 1.0e-10);
  }
}


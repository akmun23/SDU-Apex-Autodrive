#include "unity_source_plant.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

f1tenth_mpc::UnitySourcePlantState state_at_speed(double speed_mps)
{
  f1tenth_mpc::UnitySourcePlantState state;
  state.u_mps = speed_mps;
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

}  // namespace

TEST(UnitySourcePlant, ZeroSlipAtMatchedWheelSurfaceSpeed)
{
  const auto state = state_at_speed(5.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  // Slip is computed from the interval's initial wheel/body speeds. One 1 ms
  // step keeps this a direct check of the slip coordinates, before the
  // nonzero torque has time to create wheel-spin mismatch.
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

TEST(UnitySourcePlant, SuspensionJounceHistoryInitializesAndAdvances)
{
  auto state = state_at_speed(5.0);
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

TEST(UnitySourcePlant, PhysXVehicleIntegratorDoesNotAddGyroscopicYawTorque)
{
  f1tenth_mpc::UnitySourcePlantState state;
  state.roll_rate_radps = 1.0;
  for (double &omega : state.wheel_omega_radps) omega = 0.0;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  EXPECT_NEAR(output.yaw_acceleration_radps2, 0.0, 1.0e-12);
  EXPECT_NEAR(output.state.yaw_rate_radps, 0.0, 1.0e-12);
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

TEST(UnitySourcePlant, ZeroThrottleUsesFullBrakeTorque)
{
  auto state = state_at_speed(5.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  for (double omega : output.state.wheel_omega_radps)
    EXPECT_NEAR(omega, 0.0, 1.0e-12);
}

TEST(UnitySourcePlant, SteeringGeneratesLateralForceAndYaw)
{
  const auto state = state_at_speed(5.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      state, {0.25, 0.2}, 0.025, &output));
  EXPECT_GT(output.lateral_acceleration_mps2, 0.0);
  EXPECT_GT(output.yaw_acceleration_radps2, 0.0);
  // Positive-left steering gives negative lateral velocity in the wheel's
  // local right/forward convention; the source tire callback negates the
  // lateral curve to produce a positive-left force on the chassis.
  EXPECT_LT(output.lateral_slip[0], 0.0);
  EXPECT_NEAR(output.state.physical_steering_rad,
              0.025 * 3.2, 2.0e-4);

  f1tenth_mpc::UnitySourcePlantOutput reverse_turn;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      state, {-0.25, 0.2}, 0.025, &reverse_turn));
  EXPECT_LT(reverse_turn.lateral_acceleration_mps2, 0.0);
  EXPECT_LT(reverse_turn.yaw_acceleration_radps2, 0.0);
  EXPECT_NEAR(reverse_turn.state.physical_steering_rad,
              -0.025 * 3.2, 2.0e-4);
}

TEST(UnitySourcePlant, LateralSlipUsesPhysXToleranceScaleFloor)
{
  auto state = state_at_speed(6.0);
  state.v_mps = 0.24;
  state.yaw_rate_radps = 0.55;
  state.physical_steering_rad = 0.18;

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.18, 0.25}, &output));

  const double tangent = std::tan(output.state.physical_steering_rad);
  const double wheel_steering = std::atan2(
      2.0 * 0.324 * tangent, 2.0 * 0.324 - 0.236 * tangent);
  const double wheel_x = 0.17467999458312988;
  const double wheel_y = 0.11574111133813858;
  const double center_forward = state.u_mps - state.yaw_rate_radps * wheel_y;
  const double center_lateral = state.v_mps + state.yaw_rate_radps * wheel_x;
  const double wheel_forward = std::cos(wheel_steering) * center_forward +
      std::sin(wheel_steering) * center_lateral;
  const double wheel_lateral = -std::sin(wheel_steering) * center_forward +
      std::cos(wheel_steering) * center_lateral;
  const double source_slip = std::atan(
      wheel_lateral / (std::abs(wheel_forward) + 1.0));
  const double former_candidate_slip = std::atan(
      wheel_lateral / (std::abs(wheel_forward) + 0.1));

  EXPECT_NEAR(output.lateral_slip[0], source_slip, 1.0e-12);
  EXPECT_GT(std::abs(output.lateral_slip[0] - former_candidate_slip), 1.0e-3);
}

TEST(UnitySourcePlant, CombinedSlipUsesRawCurveResponsesBeforeStiffness)
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
    const double raw_y = raw_curve(output.lateral_slip[i], lateral);
    const double utilization = std::sqrt(std::max(
        1.0, std::pow(raw_x / forward[1], 2.0) +
                 std::pow(raw_y / lateral[1], 2.0)));
    EXPECT_NEAR(output.tire_longitudinal_force_n[i],
        output.normal_load_n[i] * raw_x * forward[4] / utilization, 1.0e-10);
    EXPECT_NEAR(output.tire_lateral_force_n[i],
        -output.normal_load_n[i] * raw_y * lateral[4] / utilization, 1.0e-10);
  }
}


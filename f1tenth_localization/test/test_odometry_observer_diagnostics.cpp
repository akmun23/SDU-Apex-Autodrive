#include <cmath>

#include <gtest/gtest.h>

#include "f1tenth_localization/odometry_observer.hpp"

namespace f1tenth_localization
{
namespace
{

TEST(OdometryObserverDiagnostics, SeparatesPreWheelPredictionFromFusedSpeed)
{
  OdometryObserver observer;
  OdometryObservation sample;
  sample.stamp_s = 1.0;
  auto estimate = observer.update(sample);
  EXPECT_FALSE(estimate.pre_wheel_update_speed_valid);

  sample.stamp_s += 0.025;
  sample.left_angle_rad = 0.30;
  sample.right_angle_rad = 0.30;
  sample.ax_mps2 = 1.0;
  estimate = observer.update(sample);

  ASSERT_TRUE(estimate.pre_wheel_update_speed_valid);
  EXPECT_NEAR(estimate.pre_wheel_update_speed_mps, 0.025, 1.0e-9);
  EXPECT_GT(estimate.speed_mps, estimate.pre_wheel_update_speed_mps + 0.5);
}

TEST(OdometryObserverDiagnostics, InvalidatesReferenceOnTurnEntry)
{
  OdometryObserver observer;
  OdometryObservation sample;
  sample.stamp_s = 1.0;
  observer.update(sample);

  sample.stamp_s += 0.025;
  sample.left_angle_rad = 0.30;
  sample.right_angle_rad = 0.30;
  sample.yaw_rate_radps = 1.0;
  const auto entry = observer.update(sample);
  EXPECT_TRUE(entry.turn_mode);
  EXPECT_FALSE(entry.pre_wheel_update_speed_valid);

  sample.stamp_s += 0.025;
  sample.left_angle_rad += 0.30;
  sample.right_angle_rad += 0.30;
  const auto established_turn = observer.update(sample);
  EXPECT_TRUE(established_turn.turn_mode);
  EXPECT_TRUE(established_turn.pre_wheel_update_speed_valid);
}

TEST(OdometryObserverDiagnostics, HighYawWheelDecouplingCorrectsOnlyPublishedU)
{
  OdometryObserverConfig baseline_config;
  baseline_config.use_kinematic_lateral_slip_model = false;
  OdometryObserverConfig candidate_config = baseline_config;
  candidate_config.use_high_yaw_wheel_slip_observer = true;
  OdometryObserver baseline(baseline_config);
  OdometryObserver candidate(candidate_config);

  OdometryObservation sample;
  sample.stamp_s = 1.0;
  baseline.update(sample);
  candidate.update(sample);
  const double wheel_angle_step = 5.0 * 0.025 / baseline_config.wheel_radius_m;
  for (int index = 0; index < 12; ++index) {
    sample.stamp_s += 0.025;
    sample.left_angle_rad += wheel_angle_step;
    sample.right_angle_rad += wheel_angle_step;
    baseline.update(sample);
    candidate.update(sample);
  }

  bool replaced_u = false;
  bool active_at_end = false;
  for (int index = 0; index < 32; ++index) {
    sample.stamp_s += 0.025;
    sample.left_angle_rad += wheel_angle_step;
    sample.right_angle_rad += wheel_angle_step;
    sample.yaw_rate_radps = 3.0;
    sample.yaw_rad += sample.yaw_rate_radps * 0.025;
    const auto expected = baseline.update(sample);
    const auto actual = candidate.update(sample);
    EXPECT_NEAR(actual.body_v_mps, expected.body_v_mps, 1.0e-12);
    EXPECT_NEAR(actual.yaw_rate_radps, expected.yaw_rate_radps, 1.0e-12);
    active_at_end = actual.high_yaw_wheel_slip_active;
    if (actual.body_u_mps < expected.body_u_mps - 2.0) {
      replaced_u = true;
      EXPECT_FALSE(actual.wheel_update_used);
    }
  }
  EXPECT_TRUE(replaced_u);
  EXPECT_TRUE(active_at_end);

  for (int index = 0; index < 5; ++index) {
    sample.stamp_s += 0.025;
    sample.left_angle_rad += wheel_angle_step;
    sample.right_angle_rad += wheel_angle_step;
    sample.yaw_rate_radps = 0.0;
    sample.yaw_rad += sample.yaw_rate_radps * 0.025;
    baseline.update(sample);
    const auto recovered = candidate.update(sample);
    if (index == 4) {
      EXPECT_FALSE(recovered.high_yaw_wheel_slip_active);
    }
  }
}

TEST(OdometryObserverDiagnostics, HighSpeedBiasGateUsesLateralDemand)
{
  OdometryObserverConfig baseline_config;
  baseline_config.use_kinematic_lateral_slip_model = false;
  baseline_config.use_turn_speed_bias_model = true;
  baseline_config.turn_speed_bias_constant_mps = -0.03;
  baseline_config.turn_speed_bias_yaw_rate_abs_mps = 0.185;
  baseline_config.turn_speed_bias_max_mps = 0.15;
  baseline_config.turn_speed_bias_start_speed_mps = 6.5;
  baseline_config.turn_speed_bias_full_speed_mps = 7.5;
  baseline_config.turn_slip_speed_bias_gain_mps_per_mps2 = 0.0;
  OdometryObserverConfig gated_config = baseline_config;
  gated_config.use_turn_speed_bias_lateral_accel_gate = true;
  gated_config.turn_speed_bias_lateral_accel_gate_start_mps2 = 4.0;
  gated_config.turn_speed_bias_lateral_accel_gate_full_mps2 = 8.0;
  OdometryObserver baseline(baseline_config);
  OdometryObserver gated(gated_config);

  OdometryObservation sample;
  sample.stamp_s = 1.0;
  baseline.update(sample);
  gated.update(sample);
  const double wheel_angle_step = 8.0 * 0.025 / baseline_config.wheel_radius_m;
  for (int index = 0; index < 12; ++index) {
    sample.stamp_s += 0.025;
    sample.left_angle_rad += wheel_angle_step;
    sample.right_angle_rad += wheel_angle_step;
    baseline.update(sample);
    gated.update(sample);
  }

  sample.stamp_s += 0.025;
  sample.left_angle_rad += wheel_angle_step;
  sample.right_angle_rad += wheel_angle_step;
  sample.yaw_rate_radps = 0.8;
  const auto ungated = baseline.update(sample);
  const auto attenuated = gated.update(sample);
  ASSERT_GT(ungated.turn_speed_bias_mps, 0.0);
  EXPECT_GT(attenuated.turn_speed_bias_mps, 0.0);
  EXPECT_LT(attenuated.turn_speed_bias_mps, ungated.turn_speed_bias_mps);

  for (int index = 0; index < 4; ++index) {
    sample.stamp_s += 0.025;
    sample.left_angle_rad += wheel_angle_step;
    sample.right_angle_rad += wheel_angle_step;
    sample.yaw_rate_radps = 1.1;
    const auto ungated_full = baseline.update(sample);
    const auto full_demand = gated.update(sample);
    EXPECT_NEAR(full_demand.turn_speed_bias_mps,
      ungated_full.turn_speed_bias_mps, 1.0e-12);
  }
}

}  // namespace
}  // namespace f1tenth_localization

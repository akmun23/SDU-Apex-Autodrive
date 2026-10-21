#include "f1tenth_localization/physx_odom_runtime.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace f1tenth_localization
{
namespace
{

TEST(SourceDerivedOdomRuntime, UsesNewestQueuedCommandForElapsedPacket)
{
  SourceDerivedOdomRuntime runtime(0.15532);

  const auto first = runtime.advance(0.5, 0.2, false, 0.0);
  EXPECT_DOUBLE_EQ(first.delayed_steering_commands_normalized[0], 0.0);
  EXPECT_DOUBLE_EQ(first.delayed_steering_commands_normalized[1], 0.5);
  EXPECT_DOUBLE_EQ(first.delayed_throttle_commands_normalized[0], 0.0);
  EXPECT_DOUBLE_EQ(first.delayed_throttle_commands_normalized[1], 0.2);

  const auto second = runtime.advance(-0.25, 0.0, false, 0.0);
  EXPECT_GT(second.state[13], 0.0);
  EXPECT_NEAR(second.state[13], 0.025 * 3.2, 0.002);
  EXPECT_DOUBLE_EQ(second.delayed_steering_commands_normalized[0], 0.5);
  EXPECT_DOUBLE_EQ(second.delayed_steering_commands_normalized[1], -0.25);
  EXPECT_DOUBLE_EQ(second.delayed_throttle_commands_normalized[0], 0.2);
  EXPECT_DOUBLE_EQ(second.delayed_throttle_commands_normalized[1], 0.0);

  const auto third = runtime.advance(0.0, 0.1, false, 0.0);
  EXPECT_NEAR(third.state[13], 0.0, 0.002);
  EXPECT_DOUBLE_EQ(third.delayed_steering_commands_normalized[0], -0.25);
  EXPECT_DOUBLE_EQ(third.delayed_throttle_commands_normalized[0], 0.0);
  for (const double value : third.state) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

TEST(SourceDerivedOdomRuntime, DoesNotApplyEndpointThrottleFeedbackRetrospectively)
{
  SourceDerivedOdomRuntime queued_runtime(0.15532);
  SourceDerivedOdomRuntime feedback_runtime(0.15532);
  for (SourceDerivedOdomRuntime * runtime :
    {&queued_runtime, &feedback_runtime})
  {
    runtime->advance(0.0, 0.2, false, 0.0);
    runtime->advance(0.0, 0.0, false, 0.0);
  }

  const auto queue_only = queued_runtime.advance(0.0, 0.1, false, 0.0);
  const auto measured = feedback_runtime.advance(
    0.0, 0.1, false, 0.0, true, 0.5);

  // The endpoint feedback can already describe the next request. The
  // completed interval is propagated with the queued request in both cases;
  // feedback must not retrospectively change any predicted plant state.
  for (std::size_t state = 0; state < measured.state.size(); ++state) {
    EXPECT_DOUBLE_EQ(measured.state[state], queue_only.state[state]);
  }
  EXPECT_DOUBLE_EQ(measured.delayed_throttle_commands_normalized[0],
    queue_only.delayed_throttle_commands_normalized[0]);
  EXPECT_DOUBLE_EQ(measured.delayed_throttle_commands_normalized[1],
    queue_only.delayed_throttle_commands_normalized[1]);
}

TEST(SourceDerivedOdomRuntime, PropagatesPhysXTireTimersAndResetsOnDriveCommand)
{
  SourceDerivedOdomRuntime runtime(0.15532);

  // The reset/settle interval is one second at rest, so the shared plant has
  // entered the persistent low-speed sticky-tire regime before commands are
  // applied. The runtime must preserve this state through its command queue.
  const auto first = runtime.advance(0.0, 0.2, false, 0.0);
  EXPECT_GT(first.state[UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL], 1.0);
  EXPECT_GT(first.state[UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL], 1.0);

  runtime.advance(0.0, 0.2, false, 0.0);
  const auto drive_applied = runtime.advance(0.0, 0.2, false, 0.0);
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(
      drive_applied.state[UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL + wheel], 0.0);
    EXPECT_DOUBLE_EQ(
      drive_applied.state[UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL + wheel], 0.0);
  }
}

TEST(SourceDerivedOdomRuntime, UsesMeasuredSteeringAndCorrectedBodyState)
{
  SourceDerivedOdomRuntime runtime(0.15532);
  runtime.correct(2.0, 0.1, 0.25, 0.7, 0.02, -0.01, 0.0,
    0.0, 0.0, {30.0, 30.0, 32.0, 32.0});

  const auto output = runtime.advance(0.0, 0.0, true, -0.12);
  // VehicleController.Steer() has a sign-asymmetric release branch: for this
  // measured negative physical angle and a queued zero command, it clamps the
  // internal angle to zero on the first fixed step.
  EXPECT_DOUBLE_EQ(output.state[13], 0.0);
  EXPECT_TRUE(std::isfinite(output.rear_axle_x_m));
  EXPECT_TRUE(std::isfinite(output.rear_axle_y_m));
  EXPECT_TRUE(std::isfinite(output.wheel_encoder_angle_prediction_rad[2]));
}

TEST(SourceDerivedOdomRuntime, AdvancesPreSteerFeedbackToRackStateAtPacketBoundary)
{
  SourceDerivedOdomRuntime runtime(0.15532);
  runtime.advance(0.5, 0.0, false, 0.0);
  runtime.advance(0.5, 0.0, false, 0.0);

  // The source getter reports the angle before Steer() updates it. The
  // feedback correction reconstructs that missing fixed-step rack motion,
  // then the 25 future plant ticks advance steering normally.
  const auto output = runtime.advance(0.5, 0.0, true, 0.1);
  constexpr double kSteeringRateRadps =
    183.345993 * 3.14159265358979323846 / 180.0;
  constexpr double kStep = kSteeringRateRadps * 0.001;
  EXPECT_NEAR(output.state[UNITY_SOURCE_PHYSICAL_STEERING],
    0.1 + 26.0 * kStep, 1.0e-7);
}

TEST(SourceDerivedOdomRuntime, AppliesOnlyThePreviousGyroInnovationToYawForecast)
{
  SourceDerivedOdomRuntime runtime(0.15532);
  const auto first = runtime.advance(0.0, 0.0, false, 0.0);
  EXPECT_DOUBLE_EQ(first.yaw_rate_prediction_correction_radps, 0.0);
  EXPECT_DOUBLE_EQ(first.state[2], first.raw_yaw_rate_prediction_radps);
  EXPECT_DOUBLE_EQ(first.corrected_yaw_rate_prediction_radps,
    first.raw_yaw_rate_prediction_radps);

  constexpr double prior_prediction_error_radps = 0.12;
  runtime.correct(3.0, 0.08,
    first.raw_yaw_rate_prediction_radps - prior_prediction_error_radps,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {70.0, 70.0, 70.0, 70.0});

  const auto second = runtime.advance(0.2, 0.15, false, 0.0);
  EXPECT_EQ(second.yaw_rate_prediction_mode, 1);
  EXPECT_NEAR(second.yaw_rate_prediction_correction_radps,
    prior_prediction_error_radps, 1.0e-12);
  EXPECT_NEAR(second.corrected_yaw_rate_prediction_radps,
    second.raw_yaw_rate_prediction_radps - prior_prediction_error_radps,
    1.0e-12);
  EXPECT_DOUBLE_EQ(second.state[2], second.raw_yaw_rate_prediction_radps);

  // The correction is a forecast error-state only: the internal plant is
  // sensor-corrected and the next raw prediction remains separately exposed.
  runtime.correct(3.0, 0.08, -0.04, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, {70.0, 70.0, 70.0, 70.0});
  const auto third = runtime.advance(0.2, 0.15, false, 0.0);
  EXPECT_EQ(third.yaw_rate_prediction_mode, 1);
  EXPECT_NEAR(third.yaw_rate_prediction_correction_radps,
    second.raw_yaw_rate_prediction_radps + 0.04, 1.0e-12);
}

TEST(SourceDerivedOdomRuntime, ExposesCurrentCausalYawPredictionInnovation)
{
  SourceDerivedOdomRuntime runtime(0.15532);
  const auto prediction = runtime.advance(0.0, 0.0, false, 0.0);
  EXPECT_FALSE(runtime.yaw_rate_prediction_residual_valid());

  constexpr double measured_rate_offset = 0.037;
  runtime.correct(0.0, 0.0,
    prediction.raw_yaw_rate_prediction_radps - measured_rate_offset,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {0.0, 0.0, 0.0, 0.0});
  EXPECT_TRUE(runtime.yaw_rate_prediction_residual_valid());
  EXPECT_NEAR(runtime.yaw_rate_prediction_residual_radps(),
    measured_rate_offset, 1.0e-12);
}

TEST(SourceDerivedOdomRuntime, SeparatesSteeringRampReversalAndSettledYawForecasts)
{
  SourceDerivedOdomRuntime runtime(0.15532);
  auto first = runtime.advance(0.0, 0.0, true, 0.0);
  runtime.correct(1.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, {30.0, 30.0, 30.0, 30.0});

  const auto ramp_start = runtime.advance(0.0, 0.0, true, 0.06);
  EXPECT_EQ(ramp_start.yaw_rate_prediction_mode, 0);
  runtime.correct(1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, {30.0, 30.0, 30.0, 30.0});

  const auto continuing_ramp = runtime.advance(0.0, 0.0, true, 0.12);
  EXPECT_EQ(continuing_ramp.yaw_rate_prediction_mode, 2);
  EXPECT_NEAR(continuing_ramp.corrected_yaw_rate_prediction_radps,
    0.3, 1.0e-12);
  runtime.correct(1.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, {30.0, 30.0, 30.0, 30.0});

  const auto reversal = runtime.advance(0.0, 0.0, true, 0.0);
  EXPECT_EQ(reversal.yaw_rate_prediction_mode, 0);
  EXPECT_DOUBLE_EQ(reversal.corrected_yaw_rate_prediction_radps,
    reversal.raw_yaw_rate_prediction_radps);
  runtime.correct(1.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, {30.0, 30.0, 30.0, 30.0});

  const auto first_settling_packet = runtime.advance(0.0, 0.0, true, 0.0);
  EXPECT_EQ(first_settling_packet.yaw_rate_prediction_mode, 0);
  runtime.correct(1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, {30.0, 30.0, 30.0, 30.0});

  const auto settled = runtime.advance(0.0, 0.0, true, 0.0);
  EXPECT_EQ(settled.yaw_rate_prediction_mode, 1);
  EXPECT_NEAR(settled.corrected_yaw_rate_prediction_radps,
    settled.raw_yaw_rate_prediction_radps -
    (first_settling_packet.raw_yaw_rate_prediction_radps - 0.5), 1.0e-12);
  EXPECT_DOUBLE_EQ(settled.state[2], settled.raw_yaw_rate_prediction_radps);
  (void)first;
}

}  // namespace
}  // namespace f1tenth_localization

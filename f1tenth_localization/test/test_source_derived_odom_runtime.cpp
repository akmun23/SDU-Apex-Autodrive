#include "f1tenth_localization/physx_odom_runtime.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace f1tenth_localization
{
namespace
{

TEST(SourceDerivedOdomRuntime, AppliesCommandsAfterTheTwoPacketDelay)
{
  SourceDerivedOdomRuntime runtime(0.15532);

  const auto first = runtime.advance(0.5, 0.2, false, 0.0);
  EXPECT_DOUBLE_EQ(first.delayed_steering_commands_normalized[0], 0.0);
  EXPECT_DOUBLE_EQ(first.delayed_steering_commands_normalized[1], 0.5);
  EXPECT_DOUBLE_EQ(first.delayed_throttle_commands_normalized[0], 0.0);
  EXPECT_DOUBLE_EQ(first.delayed_throttle_commands_normalized[1], 0.2);

  const auto second = runtime.advance(-0.25, 0.0, false, 0.0);
  EXPECT_DOUBLE_EQ(second.delayed_steering_commands_normalized[0], 0.5);
  EXPECT_DOUBLE_EQ(second.delayed_steering_commands_normalized[1], -0.25);
  EXPECT_DOUBLE_EQ(second.delayed_throttle_commands_normalized[0], 0.2);
  EXPECT_DOUBLE_EQ(second.delayed_throttle_commands_normalized[1], 0.0);

  const auto third = runtime.advance(0.0, 0.1, false, 0.0);
  EXPECT_GT(third.state[13], 0.0);
  EXPECT_NEAR(third.state[13], 0.025 * 3.2, 0.002);
  EXPECT_DOUBLE_EQ(third.delayed_steering_commands_normalized[0], -0.25);
  EXPECT_DOUBLE_EQ(third.delayed_throttle_commands_normalized[0], 0.0);
  for (const double value : third.state) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

TEST(SourceDerivedOdomRuntime, UsesMeasuredSteeringAndCorrectedBodyState)
{
  SourceDerivedOdomRuntime runtime(0.15532);
  runtime.correct(2.0, 0.1, 0.25, 0.7, 0.02, -0.01, 0.0,
    0.0, 0.0, {30.0, 30.0, 32.0, 32.0});

  const auto output = runtime.advance(0.0, 0.0, true, -0.12);
  EXPECT_NEAR(output.state[13], -0.12 + 0.025 * 3.2, 0.003);
  EXPECT_TRUE(std::isfinite(output.rear_axle_x_m));
  EXPECT_TRUE(std::isfinite(output.rear_axle_y_m));
  EXPECT_TRUE(std::isfinite(output.wheel_encoder_angle_prediction_rad[2]));
}

}  // namespace
}  // namespace f1tenth_localization

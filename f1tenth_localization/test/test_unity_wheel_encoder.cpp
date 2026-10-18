#include "f1tenth_localization/unity_wheel_encoder.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace f1tenth_localization::detail
{
namespace
{

TEST(UnityWheelEncoderPrediction, ConvertsPreSolveRpmIntoQuantizedRadians)
{
  UnityWheelEncoderPrediction encoder;
  const float rpm = 84.0f * (30.0f / UnityWheelEncoderPrediction::kPi);

  encoder.sample_pre_solve_rpm(rpm, 0.001f);

  // WheelEncoder.cs integrates RPS, truncates at 16*120 ticks/revolution,
  // then converts the integer tick count back to radians.
  EXPECT_EQ(encoder.angle_rad,
    static_cast<float>(25) * (2.0f * UnityWheelEncoderPrediction::kPi) /
    static_cast<float>(UnityWheelEncoderPrediction::kTicksPerRevolution));
  EXPECT_NEAR(encoder.angle_rad, 0.084, 2.0f * UnityWheelEncoderPrediction::kPi /
    static_cast<float>(UnityWheelEncoderPrediction::kTicksPerRevolution));
}

TEST(UnityWheelEncoderPrediction, PreservesReverseDirectionAndReset)
{
  UnityWheelEncoderPrediction encoder;
  const float rpm = -84.0f * (30.0f / UnityWheelEncoderPrediction::kPi);
  encoder.sample_pre_solve_rpm(rpm, 0.001f);
  EXPECT_LT(encoder.angle_rad, 0.0f);

  encoder.reset();
  EXPECT_FLOAT_EQ(encoder.total_revolutions, 0.0f);
  EXPECT_FLOAT_EQ(encoder.angle_rad, 0.0f);
}

}  // namespace
}  // namespace f1tenth_localization::detail

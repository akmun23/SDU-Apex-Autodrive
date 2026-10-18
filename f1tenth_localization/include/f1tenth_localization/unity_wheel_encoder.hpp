#pragma once

#include <cstdint>

namespace f1tenth_localization::detail
{

// Mirrors WheelEncoder.FixedUpdate() in the active F1TENTH prefab. This is a
// quantized sensor angle, not the WheelCollider's physical rotation angle.
struct UnityWheelEncoderPrediction
{
  static constexpr int32_t kTicksPerRevolution = 16 * 120;
  static constexpr float kPi = 3.14159265358979323846f;

  float total_revolutions{0.0f};
  float angle_rad{0.0f};

  void reset() noexcept
  {
    total_revolutions = 0.0f;
    angle_rad = 0.0f;
  }

  void sample_pre_solve_rpm(float wheel_rpm, float fixed_dt_s) noexcept
  {
    const float rps = wheel_rpm / 60.0f;
    total_revolutions += rps * fixed_dt_s;
    const auto total_ticks = static_cast<int32_t>(
      total_revolutions * static_cast<float>(kTicksPerRevolution));
    angle_rad = static_cast<float>(total_ticks) * (2.0f * kPi) /
      static_cast<float>(kTicksPerRevolution);
  }
};

}  // namespace f1tenth_localization::detail

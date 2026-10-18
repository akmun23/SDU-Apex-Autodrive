#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace f1tenth_localization::detail
{

inline constexpr std::size_t kPhysicsStepsPerPacket = 25;
inline constexpr float kSteeringRateDegPerS = 183.346f;
inline constexpr float kSteeringLimitDeg = 30.0f;
inline constexpr float kPi = 3.14159265358979323846f;

struct PhysxPacketActuation
{
  float throttle_normalized{};
  std::array<float, kPhysicsStepsPerPacket> steering_angles_deg{};
};

inline float unity_steering_step(float angle_deg, float command_norm)
{
  const float command = std::clamp(command_norm, -1.0f, 1.0f);
  const float threshold = angle_deg / kSteeringLimitDeg;
  const float increment = kSteeringRateDegPerS * 0.001f;
  const float target = -command * kSteeringLimitDeg;
  if (command > threshold) {
    angle_deg -= increment;
    if (angle_deg <= target) angle_deg = target;
  } else {
    angle_deg += increment;
    if (angle_deg >= target) angle_deg = target;
  }
  return angle_deg;
}

// Feedback is a measurement at the packet boundary. The pending queue head is
// the command that will act over the following 25 Unity fixed steps.
inline PhysxPacketActuation make_physx_packet_actuation(
  float & internal_steering_angle_deg,
  float pending_steering_command,
  float pending_throttle_command,
  bool steering_feedback_valid,
  double steering_feedback_rad)
{
  if (steering_feedback_valid && std::isfinite(steering_feedback_rad)) {
    internal_steering_angle_deg = std::clamp(
      -static_cast<float>(steering_feedback_rad) * (180.0f / kPi),
      -kSteeringLimitDeg, kSteeringLimitDeg);
  }

  PhysxPacketActuation actuation;
  actuation.throttle_normalized = pending_throttle_command;
  for (float & angle : actuation.steering_angles_deg) {
    internal_steering_angle_deg = unity_steering_step(
      internal_steering_angle_deg, pending_steering_command);
    angle = internal_steering_angle_deg;
  }
  return actuation;
}

}  // namespace f1tenth_localization::detail

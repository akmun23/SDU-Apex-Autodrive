#pragma once

#include <cstdio>
#include <cstdint>
#include <string>
#include <sys/types.h>

namespace f1tenth_localization
{

struct PredictedBodyVelocity
{
  double u_mps{0.0};
  double v_mps{0.0};
};

// Persistent 1 kHz public-PhysX source-parity process. Each 40 Hz sensor
// packet advances exactly 25 fixed physics steps. When available, reported
// actuator feedback anchors the measured steering angle and applied torque;
// otherwise the normalized command is used.
class PhysxOdomRuntime
{
public:
  PhysxOdomRuntime(
    const std::string & executable,
    const std::string & initial_state,
    const std::string & shape_audit);
  ~PhysxOdomRuntime();

  PhysxOdomRuntime(const PhysxOdomRuntime &) = delete;
  PhysxOdomRuntime & operator=(const PhysxOdomRuntime &) = delete;

  PredictedBodyVelocity advance(
    double steering_command_norm, double throttle_command_norm,
    bool actuator_feedback_valid, double steering_feedback_rad,
    double throttle_feedback);

private:
  void send_physics_step(float throttle_input);
  PredictedBodyVelocity read_velocity();

  pid_t child_pid_{-1};
  std::FILE * child_input_{nullptr};
  std::FILE * child_output_{nullptr};
  float center_steering_angle_deg_{0.0f};
  std::uint64_t physics_step_index_{0};
};

}  // namespace f1tenth_localization

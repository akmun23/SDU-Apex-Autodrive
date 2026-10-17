#pragma once

#include <cstdio>
#include <cstdint>
#include <array>
#include <string>
#include <sys/types.h>

namespace f1tenth_localization
{

struct PredictedPlantState
{
  std::array<double, 18> state{};
  std::array<double, 4> longitudinal_slip{};
  std::array<double, 4> lateral_slip{};
  std::array<double, 4> normal_load_n{};
  std::array<double, 4> suspension_force_n{};
  std::array<double, 4> suspension_jounce_rate_mps{};
  std::array<double, 4> tire_longitudinal_force_n{};
  std::array<double, 4> tire_lateral_force_n{};
  double longitudinal_acceleration_mps2{0.0};
  double lateral_acceleration_mps2{0.0};
  double yaw_acceleration_radps2{0.0};
  bool suspension_history_initialized{true};
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

  PredictedPlantState advance(
    double steering_command_norm, double throttle_command_norm,
    bool actuator_feedback_valid, double steering_feedback_rad,
    double throttle_feedback);

private:
  void send_physics_step(float throttle_input);
  PredictedPlantState read_state();

  pid_t child_pid_{-1};
  std::FILE * child_input_{nullptr};
  std::FILE * child_output_{nullptr};
  float center_steering_angle_deg_{0.0f};
  std::uint64_t physics_step_index_{0};
};

}  // namespace f1tenth_localization

#ifndef F1TENTH_MPC_UNITY_SOURCE_PLANT_HPP
#define F1TENTH_MPC_UNITY_SOURCE_PLANT_HPP

#include <array>

namespace f1tenth_mpc {

// Planar MPC plant using the checked-in Unity/PhysX source translations.
// Coordinates are x-forward/y-left; wheel order is FL, FR, RL, RR.
// This is intentionally separate from the current production model until
// recursive, held-out practice validation passes.
struct UnitySourcePlantState {
  double u_mps{};
  double v_mps{};
  double yaw_rate_radps{};
  std::array<double, 4> wheel_omega_radps{};
  double body_height_m{0.00464804098};
  double body_height_rate_mps{};
  double roll_rad{};
  double roll_rate_radps{};
  double pitch_rad{};
  double pitch_rate_radps{};
  double physical_steering_rad{};
  // PhysX damper velocity is the finite difference of each wheel's jounce
  // across internal vehicle substeps, not a chassis-rate approximation.
  std::array<double, 4> suspension_jounce_m{};
  bool suspension_history_initialized{};
};

struct UnitySourcePlantInput {
  double steering_target_rad{};
  // Exact VehicleController contract: exactly zero selects CAWB braking;
  // every nonzero value requests signed quarter-share motor torque.
  double throttle_normalized{};
};

struct UnitySourcePlantOutput {
  UnitySourcePlantState state{};
  std::array<double, 4> longitudinal_slip{};
  std::array<double, 4> lateral_slip{};
  std::array<double, 4> normal_load_n{};
  std::array<double, 4> suspension_force_n{};
  std::array<double, 4> suspension_jounce_rate_mps{};
  std::array<double, 4> tire_longitudinal_force_n{};
  std::array<double, 4> tire_lateral_force_n{};
  double longitudinal_acceleration_mps2{};
  double lateral_acceleration_mps2{};
  double yaw_acceleration_radps2{};
};

// Advance a held 40 Hz steering/throttle command using Unity's 1 ms fixed
// step, source Ackermann geometry, four independent wheel speeds, PhysX slip
// coordinates, Unity's tire callback curve/combined-slip response, and the
// source-derived spring/damper load model. This is not the native patched
// WheelCollider contact solver; callers must retain that boundary in reports.
bool unity_source_plant_step_1ms(
    const UnitySourcePlantState &state,
    const UnitySourcePlantInput &input,
    UnitySourcePlantOutput *output);

// Advance a controller interval by repeating the 1 ms source-model step.
bool unity_source_plant_step(
    const UnitySourcePlantState &state,
    const UnitySourcePlantInput &input,
    double duration_s,
    UnitySourcePlantOutput *output);

}  // namespace f1tenth_mpc

#endif

#ifndef F1TENTH_MPC_UNITY_SOURCE_PLANT_HPP
#define F1TENTH_MPC_UNITY_SOURCE_PLANT_HPP

#include <array>

namespace f1tenth_mpc {

// Reduced Unity/PhysX source plant used by the production MPC rollout.
// Coordinates are x-forward/y-left; wheel order is FL, FR, RL, RR. It carries
// body, wheel-spin, sprung-body, steering, and suspension state. It remains a
// source translation built on public PhysX APIs, not a bit-exact copy of the
// simulator's private WheelCollider/contact solver.
struct UnitySourcePlantState {
  double u_mps{};
  double v_mps{};
  double yaw_rate_radps{};
  std::array<double, 4> wheel_omega_radps{};
  std::array<double, 4> wheel_rotation_angle_rad{};
  double body_height_m{0.00464804098};
  double body_height_rate_mps{};
  double roll_rad{};
  // These are body-frame angular-velocity components (mapped to the plant's
  // roll/pitch/yaw axes), as supplied by the gyro. They are not Euler-angle
  // derivatives; the plant converts them through the ZYX attitude kinematics.
  double roll_rate_radps{};
  double pitch_rad{};
  double pitch_rate_radps{};
  double physical_steering_rad{};
  // PhysX damper velocity is the finite difference of each wheel's jounce
  // across internal vehicle substeps, not a chassis-rate approximation.
  std::array<double, 4> suspension_jounce_m{};
  bool suspension_history_initialized{};
  // Public PxVehicle's sticky-tire activation history. These are plant
  // memory, propagated deterministically; they are not optimizer decisions.
  std::array<double, 4> low_forward_speed_time_s{};
  std::array<double, 4> low_side_speed_time_s{};
};

struct UnitySourcePlantInput {
  double steering_target_rad{};
  // Exact VehicleController contract: exactly zero selects CAWB braking;
  // nonzero normalized throttle requests quarter-share motor torque. The
  // production MPC constrains throttle to [0, 1].
  double throttle_normalized{};
  // Defaults to the source-derived production treatment. Offline capture
  // ablations may disable only these constraints to attribute their effect.
  bool apply_sticky_tire_constraints{true};
  bool apply_immediate_sticky_constraints{true};
};

struct UnitySourcePlantOutput {
  UnitySourcePlantState state{};
  // Contact kinematics and active low-speed constraints are exposed for
  // diagnostics; production odometry/MPC continue to consume the propagated
  // state and force channels below.
  std::array<double, 4> contact_forward_speed_mps{};
  std::array<double, 4> contact_lateral_speed_mps{};
  std::array<double, 4> sticky_contact_forward_speed_mps{};
  std::array<double, 4> sticky_contact_lateral_speed_mps{};
  std::array<std::array<double, 3>, 4> sticky_contact_position_body_m{};
  std::array<std::array<double, 3>, 4> sticky_tire_longitudinal_body{};
  std::array<bool, 4> sticky_forward_active{};
  std::array<bool, 4> sticky_side_active{};
  std::array<double, 4> longitudinal_slip{};
  std::array<double, 4> maximum_positive_longitudinal_slip{};
  std::array<double, 4> maximum_longitudinal_slip_excess_mps{};
  // Uses Unity WheelHit.sidewaysSlip sign convention. The force calculation
  // uses the opposite body-left-positive contact-velocity coordinate.
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

// VehicleController stores AutonomousSteering with the opposite sign in its
// internal wheel angle, then publishes CurrentSteeringAngle as the negated
// internal angle. The externally normalized command therefore maps to the
// same-sign physical steering target consumed by this plant.
double unity_steering_command_normalized_to_physical_rad(
    double normalized_command, double max_steering_rad);


// Advance a held 40 Hz steering/throttle command using Unity's 1 ms fixed
// step, source Ackermann geometry, four independent wheel speeds, PhysX slip
// coordinates, Unity's tire callback curve/combined-slip response, and the
// source-derived spring/damper load model. This is not the native patched
// WheelCollider contact solver; callers must retain that boundary in reports.
bool unity_source_plant_step_1ms(
    const UnitySourcePlantState &state,
    const UnitySourcePlantInput &input,
    UnitySourcePlantOutput *output);

// State-only 1 ms step for callers that discard per-step diagnostics and need
// only the propagated plant state and within-step slip maxima. Dynamics and
// branch behavior are identical to unity_source_plant_step_1ms().
bool unity_source_plant_step_1ms_minimal(
    const UnitySourcePlantState &state,
    const UnitySourcePlantInput &input,
    UnitySourcePlantOutput *output);

// Advance a controller interval by repeating the 1 ms source-model step.
bool unity_source_plant_step(
    const UnitySourcePlantState &state,
    const UnitySourcePlantInput &input,
    double duration_s,
    UnitySourcePlantOutput *output);

bool unity_source_plant_slip_excess(
    const UnitySourcePlantState &state,
    std::array<double, 4> *longitudinal_slip,
    std::array<double, 4> *excess_speed_mps,
    std::array<double, 4> *excess_derivative_wheel_omega);

}  // namespace f1tenth_mpc

#endif

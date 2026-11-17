#ifndef F1TENTH_MPC_UNITY_SOURCE_PLANT_HPP
#define F1TENTH_MPC_UNITY_SOURCE_PLANT_HPP

#include <array>

namespace f1tenth_mpc {

// Reduced Unity/PhysX source plant used by the production MPC rollout.
// Coordinates are x-forward/y-left; wheel order is FL, FR, RL, RR. It carries
// body, wheel-spin, sprung-body, steering, and suspension state. It remains a
// source translation built on public PhysX APIs, not a bit-exact copy of the
// simulator's private WheelCollider/contact solver.

template <typename Scalar>
struct UnitySourcePlantStateT {
  Scalar u_mps{};
  Scalar v_mps{};
  Scalar yaw_rate_radps{};
  std::array<Scalar, 4> wheel_omega_radps{};
  std::array<Scalar, 4> wheel_rotation_angle_rad{};
  Scalar body_height_m{0.00464804098};
  Scalar body_height_rate_mps{};
  Scalar roll_rad{};
  // These are body-frame angular-velocity components (mapped to the plant's
  // roll/pitch/yaw axes), as supplied by the gyro. They are not Euler-angle
  // derivatives; the plant converts them through the ZYX attitude kinematics.
  Scalar roll_rate_radps{};
  Scalar pitch_rad{};
  Scalar pitch_rate_radps{};
  Scalar physical_steering_rad{};
  // PhysX damper velocity is the finite difference of each wheel's jounce
  // across internal vehicle substeps, not a chassis-rate approximation.
  std::array<Scalar, 4> suspension_jounce_m{};
  bool suspension_history_initialized{};
  // Public PxVehicle's sticky-tire activation history. These are plant
  // memory, propagated deterministically; they are not optimizer decisions.
  std::array<Scalar, 4> low_forward_speed_time_s{};
  std::array<Scalar, 4> low_side_speed_time_s{};
};

using UnitySourcePlantState = UnitySourcePlantStateT<double>;

template <typename Scalar>
struct UnitySourcePlantInputT {
  Scalar steering_target_rad{};
  // Exact VehicleController contract: exactly zero selects CAWB braking;
  // nonzero normalized throttle requests quarter-share motor torque. The
  // production MPC constrains throttle to [0.001, 1].
  Scalar throttle_normalized{};
};

using UnitySourcePlantInput = UnitySourcePlantInputT<double>;

template <typename Scalar>
struct UnitySourcePlantOutputT {
  UnitySourcePlantStateT<Scalar> state{};
  // Rear-axle planar displacement in the interval's initial body frame and
  // world-yaw change, accumulated at the same 1 ms rate as the source plant.
  // These let the MPC propagate pose without approximating a 25 ms interval
  // from endpoint twist samples.
  Scalar integrated_body_forward_displacement_m{};
  Scalar integrated_body_lateral_displacement_m{};
  Scalar integrated_yaw_delta_rad{};
  // Contact kinematics and active low-speed constraints are exposed for
  // diagnostics; production odometry/MPC continue to consume the propagated
  // state and force channels below.
  std::array<Scalar, 4> contact_forward_speed_mps{};
  std::array<Scalar, 4> contact_lateral_speed_mps{};
  std::array<Scalar, 4> sticky_contact_forward_speed_mps{};
  std::array<Scalar, 4> sticky_contact_lateral_speed_mps{};
  std::array<std::array<Scalar, 3>, 4> sticky_contact_position_body_m{};
  std::array<std::array<Scalar, 3>, 4> sticky_tire_longitudinal_body{};
  std::array<bool, 4> sticky_forward_active{};
  std::array<bool, 4> sticky_side_active{};
  std::array<Scalar, 4> longitudinal_slip{};
  std::array<Scalar, 4> maximum_positive_longitudinal_slip{};
  // Per wheel, maximum over all substeps of loss_x^2 + loss_y^2. loss_x and
  // the front wheels' loss_y are the fraction of the tire curve's peak force
  // lost to sliding past its extremum (0 inside the grip region,
  // 1 - |response| / peak beyond). The rear wheels' loss_y is their lateral
  // slip beyond the force-peak slip as a fraction of that slip.
  std::array<Scalar, 4> maximum_grip_loss{};
  // Uses Unity WheelHit.sidewaysSlip sign convention. The force calculation
  // uses the opposite body-left-positive contact-velocity coordinate.
  std::array<Scalar, 4> lateral_slip{};
  std::array<Scalar, 4> normal_load_n{};
  std::array<Scalar, 4> suspension_force_n{};
  std::array<Scalar, 4> suspension_jounce_rate_mps{};
  std::array<Scalar, 4> tire_longitudinal_force_n{};
  std::array<Scalar, 4> tire_lateral_force_n{};
  Scalar longitudinal_acceleration_mps2{};
  Scalar lateral_acceleration_mps2{};
  Scalar yaw_acceleration_radps2{};
};

using UnitySourcePlantOutput = UnitySourcePlantOutputT<double>;

inline constexpr std::size_t kUnityPlantSensitivityWidth = 28;
inline constexpr std::size_t kUnityPlantSensitivityStateSize = 26;
struct UnitySourcePlantSensitivitySeeds {
  std::array<std::array<double, kUnityPlantSensitivityWidth>,
             kUnityPlantSensitivityStateSize> state{};
  std::array<double, kUnityPlantSensitivityWidth> steering_target{};
  std::array<double, kUnityPlantSensitivityWidth> throttle{};
};

struct UnitySourcePlantSensitivity {
  std::array<std::array<double, kUnityPlantSensitivityWidth>,
             kUnityPlantSensitivityStateSize> state{};
  std::array<double, kUnityPlantSensitivityWidth> body_forward_displacement{};
  std::array<double, kUnityPlantSensitivityWidth> body_lateral_displacement{};
  std::array<double, kUnityPlantSensitivityWidth> yaw_delta{};
  std::array<double, 4> maximum_grip_loss{};
  std::array<std::array<double, kUnityPlantSensitivityWidth>, 4>
      maximum_grip_loss_derivative{};
};

// Optional offline command constraint: RL, RR WheelHit slips at the last
// vehicle substep of each of the 25 source ticks.
// Jacobian columns: state[0..17], steer, throttle;
// positive throttle resets the eight sticky histories.
struct UnitySourcePlantRearSlipTrace {
  std::array<double, 50> value{};
  std::array<std::array<double, 20>, 50> derivative{};
};
bool unity_source_plant_rear_slip_trace(
    const UnitySourcePlantState &state, const UnitySourcePlantInput &input,
    UnitySourcePlantRearSlipTrace *trace, bool with_derivatives);

// Coordinate order shared by differentiation and the C ABI. Wheel pose angles
// do not enter this port's dynamics; they are not independent optimizer states.
template <typename Scalar>
std::array<Scalar *, kUnityPlantSensitivityStateSize>
unity_source_plant_state_coordinates(UnitySourcePlantStateT<Scalar> &s)
{
  return {&s.u_mps, &s.v_mps, &s.yaw_rate_radps,
      &s.wheel_omega_radps[0], &s.wheel_omega_radps[1], &s.wheel_omega_radps[2], &s.wheel_omega_radps[3],
      &s.body_height_m, &s.body_height_rate_mps, &s.roll_rad, &s.roll_rate_radps,
      &s.pitch_rad, &s.pitch_rate_radps, &s.physical_steering_rad,
      &s.suspension_jounce_m[0], &s.suspension_jounce_m[1], &s.suspension_jounce_m[2], &s.suspension_jounce_m[3],
      &s.low_forward_speed_time_s[0], &s.low_forward_speed_time_s[1], &s.low_forward_speed_time_s[2], &s.low_forward_speed_time_s[3],
      &s.low_side_speed_time_s[0], &s.low_side_speed_time_s[1], &s.low_side_speed_time_s[2], &s.low_side_speed_time_s[3]};
}

// Exact selected-branch value, gradient and Hessian of one controller interval.
// Inputs, in order: forward speed, lateral speed, yaw rate, steering target,
// normalized throttle. Outputs, in order: the next u, v, yaw rate and steering
// rack, rear-axle body forward and lateral displacement, yaw delta, and the
// four signed interval slip maxima. All other coordinates enter as constants.
// Branches are selected on the primal path.
inline constexpr std::size_t kUnityPlantHessianInputs = 5;
inline constexpr std::size_t kUnityPlantHessianOutputs = 11;
struct UnitySourcePlantHessian {
  std::array<double, kUnityPlantHessianOutputs> value{};
  std::array<std::array<double, kUnityPlantHessianInputs>,
             kUnityPlantHessianOutputs> gradient{};
  std::array<std::array<std::array<double, kUnityPlantHessianInputs>,
                        kUnityPlantHessianInputs>,
             kUnityPlantHessianOutputs> hessian{};
};
bool unity_source_plant_step_with_hessian(
    const UnitySourcePlantState &state, const UnitySourcePlantInput &input,
    double duration_s, UnitySourcePlantHessian *output,
    unsigned int force_substep_cap = 0);

// VehicleController stores AutonomousSteering with the opposite sign in its
// internal wheel angle, then publishes CurrentSteeringAngle as the negated
// internal angle. The externally normalized command therefore maps to the
// same-sign physical steering target consumed by this plant.
double unity_steering_command_normalized_to_physical_rad(
    double normalized_command, double max_steering_rad);

// Exact pinned practice-player schedule: five substeps below 5 m/s and one at
// or above 5 m/s. The matching UnityPlayer build disassembly confirms
// setSubStepCount(5.0f, 5, 1); see the source constant and boundary test.
unsigned int unity_source_wheel_substep_count(double forward_speed_mps);

// Evaluate the exact native source wheel-contact lateral-slip expression
// at a physical state, without advancing a motor/steering command or
// repeating the expensive 25ms source-plant dynamics. Used by the normal
// SQP's coherent lateral tire-saturation objective and its finite-difference
// state Jacobian. WheelHit.sidewaysSlip sign is retained.
bool unity_source_plant_lateral_slip_at_state(
    const UnitySourcePlantState &state, std::array<double,4> *slip);

// Differentiate the instantaneous WheelHit lateral-slip contact
// equation in all 26 source-state coordinates with the same Dual28 AD
// kernel used by the physical Unity-source plant. One 28-lane pass;
// no finite-difference step or altered tire/contact formula.
bool unity_source_plant_lateral_slip_with_sensitivities(
    const UnitySourcePlantState &state, std::array<double,4> *slip,
    std::array<std::array<double,kUnityPlantSensitivityStateSize>,4> *jacobian);

// Reconcile the suspension history after an observer corrects body pose or
// attitude. PhysX damper velocity is a finite difference from the preceding
// substep; carrying jounce computed from the pre-correction body state would
// create a spurious first-substep spring/damper impulse. This updates only
// the four geometric jounces and marks their history initialized.
bool unity_source_plant_reconcile_suspension_jounce(
    UnitySourcePlantState *state);

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

// Differentiate the same source-derived interval step through its active
// branch for up to 28 independent directions. Branch selection is based on
// primal doubles and is therefore a local derivative, not a smoothing of the
// WheelCollider/PhysX branch transitions. force_substep_cap limits the force
// substeps per millisecond tick (the plant uses five below 5 m/s and one above);
// 0 keeps the plant's schedule, a smaller cap gives a cheaper approximate
// derivative of the same exact plant.
bool unity_source_plant_step_with_sensitivities(
    const UnitySourcePlantState &state,
    const UnitySourcePlantInput &input,
    double duration_s,
    const UnitySourcePlantSensitivitySeeds &seeds,
    UnitySourcePlantSensitivity *sensitivity,
    unsigned int force_substep_cap = 0);

}  // namespace f1tenth_mpc

#endif

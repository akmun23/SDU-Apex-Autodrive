#ifndef F1TENTH_MPC_UNITY_SOURCE_PLANT_HPP
#define F1TENTH_MPC_UNITY_SOURCE_PLANT_HPP

#include <array>
#include <vector>

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
  // production MPC constrains throttle to [0, 1].
  Scalar throttle_normalized{};
  // Defaults to the source-derived production treatment. Offline capture
  // ablations may disable only these constraints to attribute their effect.
  bool apply_sticky_tire_constraints{true};
  bool apply_immediate_sticky_constraints{true};
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
  std::array<Scalar, 4> maximum_longitudinal_slip_excess_mps{};
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

// Derivative of EVERY physical source tire-slip substep in up to 28
// source-state/actuator seed directions, not merely the largest slip.
struct UnitySourceSlipSubstepSensitivity {
  int fixed_tick{};
  int vehicle_substep{};
  int wheel{};
  double positive_excess_mps{};
  std::array<double,kUnityPlantSensitivityWidth> derivative{};
};
struct UnitySourcePlantSensitivity {
  std::array<std::array<double, kUnityPlantSensitivityWidth>,
             kUnityPlantSensitivityStateSize> state{};
  std::array<double, kUnityPlantSensitivityWidth> body_forward_displacement{};
  std::array<double, kUnityPlantSensitivityWidth> body_lateral_displacement{};
  std::array<double, kUnityPlantSensitivityWidth> yaw_delta{};
  std::array<double, 4> maximum_longitudinal_slip_excess_mps{};
  std::array<std::array<double, kUnityPlantSensitivityWidth>, 4>
      maximum_longitudinal_slip_excess_derivative{};
};

// Exact branch-local directional second derivative of the full 1 ms source
// plant integrated over one MPC stage. All 26 physical coordinates and
// both command targets participate in the same fixed source branch. The
// result is one selected piecewise Hessian direction; at force/contact
// switches the global second derivative need not exist. Offline-first API.
struct UnitySourcePlantDirectionalCurvature {
  std::array<double,kUnityPlantSensitivityStateSize> first_state{};
  std::array<double,kUnityPlantSensitivityStateSize> second_state{};
  std::array<double,3> first_body{};
  std::array<double,3> second_body{};
  std::array<double,4> first_slip_excess{};
  std::array<double,4> second_slip_excess{};
};
// One-sided derivatives of the SAME templated physical source equations.
// This is the selected directional derivative of the complete 25 ms
// integrator, including all 1/5 PxVehicle substeps, tire/contact and slip.
// At an actual discrete source switch (zero throttle / wheel-substep count
// or contact continuity failure), the directional derivative is REJECTED
// rather than invented by a smoothed physical model.
struct UnitySourceDirectionalBranchReport {
  int exact_physical_switch_ties{};
  int derivative_selected_switch_ties{};
  int contact_switch_ties{};
  int zero_throttle_brake_switches{};
  int discrete_substep_schedule_switches{};
  int valid_one_sided_directional{};
};
bool unity_source_plant_step_one_sided_directional(
    const UnitySourcePlantState &state,
    const UnitySourcePlantInput &input,
    double duration_s,
    const std::array<double,kUnityPlantSensitivityStateSize> &direction_state,
    const std::array<double,2> &direction_control,
    UnitySourcePlantDirectionalCurvature *output,
    UnitySourceDirectionalBranchReport *branch_report);

bool unity_source_plant_step_with_directional_curvature(
    const UnitySourcePlantState &state,
    const UnitySourcePlantInput &input,
    double duration_s,
    const std::array<double,kUnityPlantSensitivityStateSize> &direction_state,
    const std::array<double,2> &direction_control,
    UnitySourcePlantDirectionalCurvature *output);

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

// Differentiate the ORIGINAL instantaneous WheelHit lateral-slip contact
// equation in all 26 source-state coordinates with the same Dual28 AD
// kernel used by the physical Unity-source plant. One 28-lane pass;
// no finite-difference step or altered tire/contact formula.
bool unity_source_plant_lateral_slip_with_sensitivities(
    const UnitySourcePlantState &state, std::array<double,4> *slip,
    std::array<std::array<double,kUnityPlantSensitivityStateSize>,4> *jacobian);

// Second directional derivative of original instantaneous front-wheel
// lateral slip, including first AND second source-state perturbations. Uses
// SourceJet2 through the SAME contact/attitude/atan formula as the plant.
bool unity_source_plant_lateral_slip_directional_second(
    const UnitySourcePlantState &state,
    const std::array<double,kUnityPlantSensitivityStateSize> &first_state,
    const std::array<double,kUnityPlantSensitivityStateSize> &second_state,
    std::array<double,4> *slip,
    std::array<double,4> *first_slip,
    std::array<double,4> *second_slip);

// Reconcile the suspension history after an observer corrects body pose or
// attitude. PhysX damper velocity is a finite difference from the preceding
// substep; carrying jounce computed from the pre-correction body state would
// create a spurious first-substep spring/damper impulse. This updates only
// the four geometric jounces and marks their history initialized.
bool unity_source_plant_reconcile_suspension_jounce(
    UnitySourcePlantState *state);


// Development-only event trace of every original PxVehicle internal
// wheel substep. At <5 m/s, each source 1 ms tick contains FIVE 0.2 ms
// friction/contact evaluations; stage-end or 1 ms summaries conceal
// regime changes. No alternative plant or hidden smoothing is permitted.
struct UnitySourceBranchObservation {
  int fixed_tick{};
  int vehicle_substep{};
  int vehicle_substep_count{};
  int wheel{};
  bool grounded{};
  bool sticky_forward{};
  bool sticky_side{};
  int longitudinal_curve_region{}; // 0: rising, 1: falling, 2: plateau
  int lateral_curve_region{};
  int tire_load_filter_region{};
  bool combined_slip_saturated{};
  // Further original PhysX/Unity physical switching surfaces. These are
  // recorded only in source-trace mode, NEVER used to approximate forces.
  // -1 / 0 / +1: below minimum / interior / above maximum jounce.
  int suspension_jounce_region{};
  // -1 wheel-ground > wheel-surface, +1 surface > ground (the max
  // denominator derivative changes at equality); 0 exact tie.
  int longitudinal_slip_denominator_region{};
  // -1,0,+1 for distinct absolute-value derivatives at zero speed.
  int forward_contact_speed_sign{};
  int lateral_contact_speed_sign{};
  int wheel_surface_speed_sign{};
  bool steering_target_reached{};
  bool zero_throttle_brake_mode{};
  double raw_suspension_jounce_m{};
  double wheel_ground_forward_speed_mps{};
  double wheel_surface_speed_mps{};
  double unfiltered_tire_load_ratio{};
  double normalized_combined_slip_demand_squared{};
  double longitudinal_slip{};
  // ORIGINAL instantaneous source penalty feature, already computed by
  // the tire model. This is before the exact max over substeps/ticks.
  double positive_longitudinal_slip_excess_mps{};
  double lateral_slip{};
  double combined_slip_utilization{};
  double normal_load_n{};
};
bool unity_source_plant_trace_branches(
    const UnitySourcePlantState &state,
    const UnitySourcePlantInput &input,double duration_s,
    std::vector<UnitySourceBranchObservation> *trace,
    UnitySourcePlantOutput *traced_output=nullptr);

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
// branch for up to twenty independent directions. Branch selection is based on
// primal doubles and is therefore a local derivative, not a smoothing of the
// WheelCollider/PhysX branch transitions.
bool unity_source_plant_step_with_sensitivities(
    const UnitySourcePlantState &state,
    const UnitySourcePlantInput &input,
    double duration_s,
    const UnitySourcePlantSensitivitySeeds &seeds,
    UnitySourcePlantSensitivity *sensitivity,
    std::vector<UnitySourceSlipSubstepSensitivity>
        *all_original_slip_substeps=nullptr);

bool unity_source_plant_slip_excess(
    const UnitySourcePlantState &state,
    std::array<double, 4> *longitudinal_slip,
    std::array<double, 4> *excess_speed_mps,
    std::array<double, 4> *excess_derivative_wheel_omega);

}  // namespace f1tenth_mpc

#endif

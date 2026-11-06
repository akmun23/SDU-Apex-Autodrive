#ifndef F1TENTH_MPC_UNITY_SOURCE_PLANT_C_H
#define F1TENTH_MPC_UNITY_SOURCE_PLANT_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* C ABI for the source-derived plant, shared by the MPC core and offline
 * capture evaluator. state[] ordering is documented in this header so C and
 * Python callers do not depend on the C++ std::array ABI. */
enum { UNITY_SOURCE_PLANT_STATE_SIZE = 26 };
enum { UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH = 28 };
/* Exact physical command limit. Float-valued ROS/MPC interfaces round 30 deg
 * slightly upward, so plant inputs are clipped to this double boundary. */
#define UNITY_SOURCE_MAX_STEERING_RAD 0.52359877559829887308
/* Unity's forward tire curve reaches its force extremum at this slip. */
#define UNITY_SOURCE_FORWARD_EXTREMUM_SLIP 0.15
/* Unity VehicleController's forward friction curve enters its asymptote here.
 * This is a source-curve traction plateau, not an empirical fit. */
#define UNITY_SOURCE_FORWARD_ASYMPTOTE_SLIP 0.25
enum {
  UNITY_SOURCE_U = 0,
  UNITY_SOURCE_V = 1,
  UNITY_SOURCE_YAW_RATE = 2,
  UNITY_SOURCE_WHEEL_OMEGA_FL = 3,
  UNITY_SOURCE_WHEEL_OMEGA_FR = 4,
  UNITY_SOURCE_WHEEL_OMEGA_RL = 5,
  UNITY_SOURCE_WHEEL_OMEGA_RR = 6,
  UNITY_SOURCE_BODY_HEIGHT = 7,
  UNITY_SOURCE_BODY_HEIGHT_RATE = 8,
  UNITY_SOURCE_ROLL = 9,
  UNITY_SOURCE_ROLL_RATE = 10,
  UNITY_SOURCE_PITCH = 11,
  UNITY_SOURCE_PITCH_RATE = 12,
  UNITY_SOURCE_PHYSICAL_STEERING = 13,
  UNITY_SOURCE_SUSPENSION_JOUNCE_FL = 14,
  UNITY_SOURCE_SUSPENSION_JOUNCE_FR = 15,
  UNITY_SOURCE_SUSPENSION_JOUNCE_RL = 16,
  UNITY_SOURCE_SUSPENSION_JOUNCE_RR = 17,
  UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL = 18,
  UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FR = 19,
  UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_RL = 20,
  UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_RR = 21,
  UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL = 22,
  UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FR = 23,
  UNITY_SOURCE_LOW_SIDE_SPEED_TIME_RL = 24,
  UNITY_SOURCE_LOW_SIDE_SPEED_TIME_RR = 25
};

typedef struct {
  double state[UNITY_SOURCE_PLANT_STATE_SIZE];
  double wheel_rotation_angle_rad[4];
  double steering_target_rad;
  double throttle_normalized;
  int suspension_history_initialized;
} UnitySourcePlantInputC;

typedef struct {
  double state[UNITY_SOURCE_PLANT_STATE_SIZE];
  double integrated_body_forward_displacement_m;
  double integrated_body_lateral_displacement_m;
  double integrated_yaw_delta_rad;
  /* PxVehicle wheel pose rotation is persistent and affects the source
   * wheel-mesh collision shapes in the native scene solve. */
  double wheel_rotation_angle_rad[4];
  int suspension_history_initialized;
  double longitudinal_slip[4];
  /* Per-wheel maximum positive drive slip over every internal plant substep
   * in this requested interval. */
  double maximum_positive_longitudinal_slip[4];
  double maximum_longitudinal_slip_excess_mps[4];
  double lateral_slip[4];
  double normal_load_n[4];
  double suspension_force_n[4];
  double suspension_jounce_rate_mps[4];
  double tire_longitudinal_force_n[4];
  double tire_lateral_force_n[4];
  double longitudinal_acceleration_mps2;
  double lateral_acceleration_mps2;
  double yaw_acceleration_radps2;
} UnitySourcePlantOutputC;

typedef struct {
  double state[UNITY_SOURCE_PLANT_STATE_SIZE]
      [UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH];
  double steering_target[UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH];
  double throttle[UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH];
} UnitySourcePlantSensitivitySeedsC;

typedef struct {
  double state[UNITY_SOURCE_PLANT_STATE_SIZE]
      [UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH];
  double body_forward_displacement
      [UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH];
  double body_lateral_displacement
      [UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH];
  double yaw_delta[UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH];
  double maximum_longitudinal_slip_residual_mps[4];
  double maximum_longitudinal_slip_residual_derivative[4]
      [UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH];
} UnitySourcePlantSensitivityC;

/* Recompute source-geometry suspension jounce after an observer corrects the
 * body state. Mutates only state[14..17] and suspension_history_initialized. */
int unity_source_plant_reconcile_suspension_jounce_c(
    double state[UNITY_SOURCE_PLANT_STATE_SIZE],
    int *suspension_history_initialized);

/* Returns 1 on success; rejects non-finite/out-of-contract inputs. duration
 * must be an integer number of 1 ms Unity fixed steps, no longer than 250 ms. */
int unity_source_plant_step_c(const UnitySourcePlantInputC *input,
                              double duration_s,
                              UnitySourcePlantOutputC *output);


/* Branch-local forward-mode derivatives of the exact same source plant. Each
 * direction seeds one or more state/command entries; the returned columns
 * correspond to those directions. */
int unity_source_plant_step_sensitivity_c(
    const UnitySourcePlantInputC *input, double duration_s,
    const UnitySourcePlantSensitivitySeedsC *seeds,
    UnitySourcePlantSensitivityC *sensitivity);

/* Full native-source branch-local second derivative in one 26-state + 2
 * physical control direction. Unlike central finite differences this
 * evaluates the exact selected source branch only; it is NOT a globally
 * valid smooth Hessian across friction/contact switching surfaces. */
typedef struct {
  double first_state[UNITY_SOURCE_PLANT_STATE_SIZE];
  double second_state[UNITY_SOURCE_PLANT_STATE_SIZE];
  double first_body[3];
  double second_body[3];
  double first_slip_excess[4];
  double second_slip_excess[4];
} UnitySourcePlantDirectionalCurvatureC;
int unity_source_plant_step_directional_curvature_c(
    const UnitySourcePlantInputC *input,
    double duration_s,
    const double direction_state[UNITY_SOURCE_PLANT_STATE_SIZE],
    const double direction_control[2],
    UnitySourcePlantDirectionalCurvatureC *out);

/* Separate one-sided directional source-plant interface. Does NOT
 * change existing C source step/Jacobian/curvature ABI. When report.valid=0
 * the sampled physical source direction crosses a discrete/noncontinuous
 * branch and MUST NOT be used as a finite source gradient by the NLP. */
typedef struct {
  int exact_physical_switch_ties;
  int derivative_selected_switch_ties;
  int contact_switch_ties;
  int zero_throttle_brake_switches;
  int discrete_substep_schedule_switches;
  int valid_one_sided_directional;
} UnitySourceDirectionalBranchReportC;

/* Includes A*x'' in the same branch-selected Jet2 pass. A null report
 * retains nominal branch selection; otherwise uses one-sided semantics. */
int unity_source_plant_step_path_curvature_c(
    const UnitySourcePlantInputC *input,double duration_s,
    const double first_state[UNITY_SOURCE_PLANT_STATE_SIZE],
    const double second_state[UNITY_SOURCE_PLANT_STATE_SIZE],
    const double first_control[2],UnitySourcePlantDirectionalCurvatureC *out,
    UnitySourceDirectionalBranchReportC *report);

/* Exact original 1ms/0.2ms physical source branch comparison.
 * Both sides are independently evaluated through the ORIGINAL plant,
 * not through finite-difference forces or surrogate tire curves.
 * Counts compare matching wheel/tick/substep IDs and expose discrete
 * schedule changes separately. Audit-only and excluded from production. */
typedef struct {
  int original_internal_events;
  int perturbed_internal_events;
  int mismatched_substep_schedule_events;
  int contact_grounded_changes;
  int sticky_forward_changes;
  int sticky_side_changes;
  int longitudinal_spline_region_changes;
  int lateral_spline_region_changes;
  int tire_load_filter_changes;
  int combined_slip_saturation_changes;
  int suspension_jounce_region_changes;
  int slip_denominator_region_changes;
  int steering_target_reached_changes;
  int throttle_brake_mode_changes;
  int first_changed_fixed_tick;
  int first_changed_vehicle_substep;
  int first_changed_wheel;
  int valid;
} UnitySourcePhysicalBranchComparisonC;

/* Exact source wheel-contact lateral slip at this state, without
 * advancing the plant. No ABI changes to existing step/sensitivity structs. */
int unity_source_plant_instantaneous_lateral_slip_c(
    const double state[UNITY_SOURCE_PLANT_STATE_SIZE], double slip[4]);

/* Original WheelHit lateral-slip expression and its analytic 26-state
 * source AD Jacobian. New explicit API, existing C structs/ABI unchanged. */
int unity_source_plant_instantaneous_lateral_slip_ad_c(
    const double state[UNITY_SOURCE_PLANT_STATE_SIZE],
    double slip[4],
    double jacobian[4][UNITY_SOURCE_PLANT_STATE_SIZE]);

/* Exact branch-local source directional second derivative for the
 * instantaneous WheelHit lateral-slip objective. The caller supplies both
 * first and second 26-state horizon sensitivities. */
int unity_source_plant_instantaneous_lateral_slip_second_c(
    const double state[UNITY_SOURCE_PLANT_STATE_SIZE],
    const double first_state[UNITY_SOURCE_PLANT_STATE_SIZE],
    const double second_state[UNITY_SOURCE_PLANT_STATE_SIZE],
    double slip[4],double first_slip[4],double second_slip[4]);

/* Offline exact source trace of the internal peak-longitudinal-slip
 * branch. Reuses the original substep tire expression; no extrapolation.
 * Tied maxima are explicitly exposed rather than smoothed. */
typedef struct {
  double actual_source_max_excess_mps;
  double traced_max_excess_mps;
  double runner_up_excess_mps;
  int active_fixed_tick;
  int active_vehicle_substep;
  int equal_max_substeps;
  int traced_substeps;
} UnitySourcePeakSlipTraceC;
/* Every ORIGINAL 1ms/vehicle-substep positive slip excess. The maximum
 * over these values and zero equals the original source slip-cost feature.
 * This is an optional offline nonlinear-epigraph evaluator; no force
 * recomputation or fabricated slip curve. One trace event is produced for
 * every tire and internal substep; airborne wheels have zero excess.
 */
typedef struct {
    int fixed_tick;
    int vehicle_substep;
    int wheel;
    double positive_excess_mps;
} UnitySourceSlipSubstepC;

/* Source AD of every ORIGINAL internal tire-slip substep.
 * The caller selects up to 20 independent source-state/command directions.
 * Results have the same time/wheel indexing as the primal substep trace.
 * At switching surfaces the returned derivative is the selected source
 * one-sided branch, not an assumed smooth derivative.
 */
typedef struct {
    UnitySourceSlipSubstepC original;
    double derivative[UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH];
} UnitySourceSlipSubstepSensitivityC;


int unity_source_plant_peak_slip_trace_c(
    const UnitySourcePlantInputC *input,double duration_s,
    UnitySourcePeakSlipTraceC out[4]);

/* Evaluate Unity's source slip equation at the supplied physical plant state.
 * The excess is wheel-surface-speed mismatch beyond the source curve's force
 * extremum, in m/s; its derivative is with respect to wheel omega. */
int unity_source_plant_slip_excess_c(
    const double state[UNITY_SOURCE_PLANT_STATE_SIZE],
    double longitudinal_slip[4],
    double excess_speed_mps[4],
    double excess_derivative_wheel_omega[4]);

#ifdef __cplusplus
}
#endif

#endif

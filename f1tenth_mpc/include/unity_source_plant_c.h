#ifndef F1TENTH_MPC_UNITY_SOURCE_PLANT_C_H
#define F1TENTH_MPC_UNITY_SOURCE_PLANT_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* C ABI for the source-derived plant, shared by the MPC core and offline
 * capture evaluator. state[] ordering is documented in this header so C and
 * Python callers do not depend on the C++ std::array ABI. */
enum { UNITY_SOURCE_PLANT_STATE_SIZE = 26 };
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
  int sticky_tire_longitudinal_active[4];
  int sticky_tire_lateral_active[4];
  double tire_longitudinal_force_n[4];
  double tire_lateral_force_n[4];
  double longitudinal_acceleration_mps2;
  double lateral_acceleration_mps2;
  double yaw_acceleration_radps2;
} UnitySourcePlantOutputC;

/* Returns 1 on success; rejects non-finite/out-of-contract inputs. duration
 * must be an integer number of 1 ms Unity fixed steps, no longer than 250 ms. */
int unity_source_plant_step_c(const UnitySourcePlantInputC *input,
                              double duration_s,
                              UnitySourcePlantOutputC *output);

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

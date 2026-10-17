#ifndef F1TENTH_MPC_UNITY_SOURCE_PLANT_C_H
#define F1TENTH_MPC_UNITY_SOURCE_PLANT_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* C ABI for the source-derived plant, shared by the MPC core and offline
 * capture evaluator. state[] ordering is documented in this header so C and
 * Python callers do not depend on the C++ std::array ABI. */
enum { UNITY_SOURCE_PLANT_STATE_SIZE = 18 };
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
  UNITY_SOURCE_SUSPENSION_JOUNCE_RR = 17
};

typedef struct {
  double state[UNITY_SOURCE_PLANT_STATE_SIZE];
  double steering_target_rad;
  double throttle_normalized;
  int suspension_history_initialized;
} UnitySourcePlantInputC;

typedef struct {
  double state[UNITY_SOURCE_PLANT_STATE_SIZE];
  int suspension_history_initialized;
  double longitudinal_slip[4];
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

/* Returns 1 on success; rejects non-finite/out-of-contract inputs. duration
 * must be an integer number of 1 ms Unity fixed steps, no longer than 250 ms. */
int unity_source_plant_step_c(const UnitySourcePlantInputC *input,
                              double duration_s,
                              UnitySourcePlantOutputC *output);

#ifdef __cplusplus
}
#endif

#endif

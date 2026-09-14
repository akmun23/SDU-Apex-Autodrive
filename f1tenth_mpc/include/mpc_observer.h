/**
 * @file mpc_observer.h
 * @brief Source-time MPC dynamic-state observer prototype.
 *
 * This is a dev-side observer API, not a runtime ROS node and not an MPC
 * authority path. It consumes only legal runtime measurements. Simulator
 * truth is used by offline tools to score the observer, never by this API.
 */

#ifndef MPC_OBSERVER_H
#define MPC_OBSERVER_H

#include <stdint.h>

typedef struct
{
    float stamp_s;                  /* Source-epoch measurement time [s]. */
    float u_mps;                    /* Longitudinal speed from odometry [m/s]. */
    float yaw_rate_radps;           /* IMU gyro yaw rate [rad/s]. */
    float longitudinal_accel_mps2;  /* IMU longitudinal acceleration [m/s^2]. */
    float lateral_accel_mps2;       /* IMU lateral acceleration [m/s^2]. */
    float steering_rad;             /* Steering feedback/command [rad]. */
    float applied_throttle_norm;    /* Applied drive command [0, 1]. */
} MpcObserverMeasurement_t;

typedef struct
{
    float min_dt_s;
    float max_dt_s;                 /* Strict source cadence upper bound [s]. */
    float u_accel_gain;
    float u_measurement_gain;
    float max_u_mps;
    float steering_time_constant_s;
    float drive_time_constant_s;
    float v_time_constant_s;
    float v_u_coefficient;
    float v_r_coefficient;
    float v_steering_coefficient;
    float v_lateral_accel_coefficient;
    float v_bias_mps;
    float max_abs_v_mps;
} MpcObserverConfiguration_t;

typedef struct
{
    float stamp_s;
    float dt_s;
    float u_mps;
    float v_mps;
    float r_radps;
    float steering_rad;
    float q_drive;
    float state_age_s;
    uint8_t valid;
    uint8_t timing_degraded;
} MpcObserverState_t;

MpcObserverConfiguration_t mpc_observer_default_configuration(void);

void mpc_observer_initialize(
    const MpcObserverConfiguration_t *configuration);

void mpc_observer_reset(void);

int mpc_observer_update(
    const MpcObserverMeasurement_t *measurement,
    MpcObserverState_t *state);

/* Copy the latest source-time state to a later solve/application epoch.
 * This is zero-order-hold propagation until the accepted reduced plant is
 * available; it exposes age explicitly and never invents simulator truth. */
int mpc_observer_propagate_to(
    float target_stamp_s,
    MpcObserverState_t *state);

#endif /* MPC_OBSERVER_H */

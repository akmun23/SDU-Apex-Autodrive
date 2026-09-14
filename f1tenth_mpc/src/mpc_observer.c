/**
 * @file mpc_observer.c
 * @brief Source-time MPC dynamic-state observer prototype.
 */

#include "mpc_observer.h"

#include <math.h>
#include <stddef.h>

#define OBS_DEFAULT_MIN_DT_S 0.015f
#define OBS_DEFAULT_MAX_DT_S 0.035f
#define OBS_DEFAULT_U_ACCEL_GAIN 1.0f
#define OBS_DEFAULT_U_MEASUREMENT_GAIN 0.20f
#define OBS_DEFAULT_MAX_U_MPS 20.0f
#define OBS_DEFAULT_STEERING_TAU_S 0.025f
#define OBS_DEFAULT_DRIVE_TAU_S 0.100f
#define OBS_DEFAULT_V_TAU_S 0.010f

/* Fitted offline from accepted source-time trace 81; validation is run 82.
 * These are observer coefficients, not simulator truth or production tire
 * parameters. Keep them replaceable until cross-track/open-scene validation. */
#define OBS_DEFAULT_V_U_COEFF (-0.000841488814f)
#define OBS_DEFAULT_V_R_COEFF (0.201426313148f)
#define OBS_DEFAULT_V_STEERING_COEFF (-0.242524180360f)
#define OBS_DEFAULT_V_AY_COEFF (-0.012222072104f)
#define OBS_DEFAULT_V_BIAS (0.001891000530f)
#define OBS_DEFAULT_MAX_ABS_V_MPS 1.5f

static MpcObserverConfiguration_t configuration;
static MpcObserverState_t latest;
static int initialized = 0;

static int finite_float(float value)
{
    return isfinite(value) != 0;
}

static float clamp_float(float value, float lower, float upper)
{
    if (value < lower)
        return lower;
    if (value > upper)
        return upper;
    return value;
}

static float one_minus_exp_decay(float dt_s, float tau_s)
{
    if (!(tau_s > 0.0f) || !finite_float(tau_s))
        return 1.0f;
    return clamp_float(1.0f - expf(-dt_s / tau_s), 0.0f, 1.0f);
}

MpcObserverConfiguration_t mpc_observer_default_configuration(void)
{
    MpcObserverConfiguration_t defaults = {
        .min_dt_s = OBS_DEFAULT_MIN_DT_S,
        .max_dt_s = OBS_DEFAULT_MAX_DT_S,
        .u_accel_gain = OBS_DEFAULT_U_ACCEL_GAIN,
        .u_measurement_gain = OBS_DEFAULT_U_MEASUREMENT_GAIN,
        .max_u_mps = OBS_DEFAULT_MAX_U_MPS,
        .steering_time_constant_s = OBS_DEFAULT_STEERING_TAU_S,
        .drive_time_constant_s = OBS_DEFAULT_DRIVE_TAU_S,
        .v_time_constant_s = OBS_DEFAULT_V_TAU_S,
        .v_u_coefficient = OBS_DEFAULT_V_U_COEFF,
        .v_r_coefficient = OBS_DEFAULT_V_R_COEFF,
        .v_steering_coefficient = OBS_DEFAULT_V_STEERING_COEFF,
        .v_lateral_accel_coefficient = OBS_DEFAULT_V_AY_COEFF,
        .v_bias_mps = OBS_DEFAULT_V_BIAS,
        .max_abs_v_mps = OBS_DEFAULT_MAX_ABS_V_MPS,
    };
    return defaults;
}

void mpc_observer_initialize(
    const MpcObserverConfiguration_t *requested_configuration)
{
    configuration = requested_configuration != NULL ?
        *requested_configuration : mpc_observer_default_configuration();
    if (!(configuration.min_dt_s > 0.0f) ||
        !(configuration.max_dt_s >= configuration.min_dt_s) ||
        !(configuration.max_u_mps > 0.0f) ||
        configuration.u_measurement_gain < 0.0f ||
        configuration.u_measurement_gain > 1.0f ||
        !(configuration.max_abs_v_mps > 0.0f) ||
        !finite_float(configuration.min_dt_s) ||
        !finite_float(configuration.max_dt_s) ||
        !finite_float(configuration.u_accel_gain) ||
        !finite_float(configuration.u_measurement_gain) ||
        !finite_float(configuration.max_u_mps) ||
        !finite_float(configuration.max_abs_v_mps)) {
        configuration = mpc_observer_default_configuration();
    }
    mpc_observer_reset();
}

void mpc_observer_reset(void)
{
    latest = (MpcObserverState_t){0};
    initialized = 0;
}

static float lateral_target(const MpcObserverMeasurement_t *measurement)
{
    return configuration.v_u_coefficient * measurement->u_mps +
           configuration.v_r_coefficient * measurement->yaw_rate_radps +
           configuration.v_steering_coefficient * measurement->steering_rad +
           configuration.v_lateral_accel_coefficient *
               measurement->lateral_accel_mps2 +
           configuration.v_bias_mps;
}

int mpc_observer_update(
    const MpcObserverMeasurement_t *measurement,
    MpcObserverState_t *state)
{
    if (measurement == NULL || state == NULL ||
        !finite_float(measurement->stamp_s) ||
        !finite_float(measurement->u_mps) ||
        !finite_float(measurement->yaw_rate_radps) ||
        !finite_float(measurement->longitudinal_accel_mps2) ||
        !finite_float(measurement->lateral_accel_mps2) ||
        !finite_float(measurement->steering_rad) ||
        !finite_float(measurement->applied_throttle_norm)) {
        return 0;
    }

    const float target_v = clamp_float(
        lateral_target(measurement),
        -configuration.max_abs_v_mps, configuration.max_abs_v_mps);
    const float throttle = clamp_float(
        measurement->applied_throttle_norm, 0.0f, 1.0f);

    if (!initialized) {
        latest.stamp_s = measurement->stamp_s;
        latest.dt_s = 0.0f;
        latest.u_mps = measurement->u_mps;
        latest.v_mps = target_v;
        latest.r_radps = measurement->yaw_rate_radps;
        latest.steering_rad = measurement->steering_rad;
        latest.q_drive = throttle;
        latest.state_age_s = 0.0f;
        latest.valid = 1;
        latest.timing_degraded = 0;
        initialized = 1;
        *state = latest;
        return 1;
    }

    const float dt_s = measurement->stamp_s - latest.stamp_s;
    const int timing_ok = dt_s >= configuration.min_dt_s &&
        dt_s <= configuration.max_dt_s;
    if (!(dt_s > 0.0f) || !finite_float(dt_s)) {
        latest.timing_degraded = 1;
        latest.state_age_s = 0.0f;
        *state = latest;
        return 0;
    }

    latest.stamp_s = measurement->stamp_s;
    latest.dt_s = dt_s;
    const float u_prediction = latest.u_mps + dt_s *
        configuration.u_accel_gain * measurement->longitudinal_accel_mps2;
    latest.u_mps = clamp_float(
        u_prediction + configuration.u_measurement_gain *
            (measurement->u_mps - u_prediction),
        0.0f, configuration.max_u_mps);
    latest.r_radps = measurement->yaw_rate_radps;
    latest.steering_rad += one_minus_exp_decay(
        dt_s, configuration.steering_time_constant_s) *
        (measurement->steering_rad - latest.steering_rad);
    latest.q_drive += one_minus_exp_decay(
        dt_s, configuration.drive_time_constant_s) *
        (throttle - latest.q_drive);
    latest.v_mps += one_minus_exp_decay(
        dt_s, configuration.v_time_constant_s) *
        (target_v - latest.v_mps);
    latest.v_mps = clamp_float(
        latest.v_mps, -configuration.max_abs_v_mps,
        configuration.max_abs_v_mps);
    latest.state_age_s = 0.0f;
    latest.valid = 1;
    latest.timing_degraded = timing_ok ? 0 : 1;
    *state = latest;
    return timing_ok;
}

int mpc_observer_propagate_to(
    float target_stamp_s,
    MpcObserverState_t *state)
{
    if (state == NULL || !initialized || !finite_float(target_stamp_s) ||
        target_stamp_s < latest.stamp_s) {
        return 0;
    }
    *state = latest;
    state->state_age_s = target_stamp_s - latest.stamp_s;
    return finite_float(state->state_age_s) ? 1 : 0;
}

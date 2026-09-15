/**
 * @file vehicle_plant.h
 * @brief Native replay plant for the model-identification candidate.
 *
 * This is intentionally separate from vehicle_model.h.  It is an offline
 * replay target until the structured plant passes open-loop and blind-run
 * acceptance; it is not wired into the production MPC solver by default.
 */

#ifndef VEHICLE_PLANT_H
#define VEHICLE_PLANT_H

#include <stdint.h>

typedef struct
{
    float x_m;
    float y_m;
    float yaw_rad;
    float u_mps;
    float v_mps;
    float r_radps;
    float steering_rad;
    float wheel_speed_mps;
} VehiclePlantState_t;

typedef struct
{
    float steering_target_norm;
    float throttle_norm;
} VehiclePlantInput_t;

typedef struct
{
    float mass_kg;
    float lf_m;
    float lr_m;
    float iz_kgm2;
    /* Recorded pose point relative to the body-velocity point [m]. */
    float position_offset_from_velocity_point_x_m;
    float max_steering_rad;
    float steering_rate_radps;
    float max_speed_mps;
    /* Optional measured rigid-body damping terms [1/s].  These are kept
     * explicit so an offline A/B test cannot hide damping in tire forces. */
    float linear_damping_per_s;
    float angular_damping_per_s;
    float force_max_n;
    float hard_brake_force_n;
    float slip_gain_per_mps;
    float coast_speed_drag_n_per_mps;
    float cf_n_per_rad;
    float cr_n_per_rad;
    float df_n;
    float dr_n;
    uint8_t tire_model;
    uint8_t wheel_dynamics_model;
    float wheel_coefficients[6];
} VehiclePlantParameters_t;

enum {
    VEHICLE_PLANT_TIRE_LINEAR_SATURATED = 0,
    VEHICLE_PLANT_TIRE_TANH = 1
};

enum {
    VEHICLE_PLANT_WHEEL_DYNAMICS_DISCRETE = 0,
    VEHICLE_PLANT_WHEEL_DYNAMICS_CONTINUOUS = 1
};

VehiclePlantParameters_t vehicle_plant_default_parameters(void);

void vehicle_plant_step(
    const VehiclePlantState_t *state,
    const VehiclePlantInput_t *input,
    float dt_s,
    const VehiclePlantParameters_t *parameters,
    VehiclePlantState_t *next);

#endif /* VEHICLE_PLANT_H */

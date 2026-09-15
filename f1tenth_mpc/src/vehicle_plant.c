/**
 * @file vehicle_plant.c
 * @brief Native implementation of the offline structured replay plant.
 */

#include "vehicle_plant.h"

#include <math.h>
#include <stddef.h>

#define PLANT_DEFAULT_MASS_KG 3.470f
#define PLANT_DEFAULT_LF_M 0.174679914f
#define PLANT_DEFAULT_LR_M 0.155320086f
#define PLANT_DEFAULT_IZ_KGM2 0.0961908f /* Corrected Unity body-Y inertia projection. */
#define PLANT_DEFAULT_POSITION_OFFSET_X_M -0.155320086f /* rear pose vs COM velocity point */
#define PLANT_DEFAULT_MAX_STEERING_RAD 0.5236f
#define PLANT_DEFAULT_STEERING_RATE_RADPS 3.2f
#define PLANT_DEFAULT_MAX_SPEED_MPS 16.0f /* Project offline identification ceiling. */
#define PLANT_DEFAULT_LINEAR_DAMPING_PER_S 0.273f /* Unity Rigidbody.drag [1/s]. */
#define PLANT_DEFAULT_ANGULAR_DAMPING_PER_S 0.1f /* Unity Rigidbody.angularDrag [1/s]. */
#define PLANT_DEFAULT_FORCE_MAX_N 19.89360065f /* Clean measured-damping holdout fit; offline only. */
#define PLANT_DEFAULT_HARD_BRAKE_FORCE_N 18.57580700f /* Clean measured-damping holdout fit; offline only. */
#define PLANT_DEFAULT_SLIP_GAIN_PER_MPS 0.97150588f /* Clean measured-damping holdout fit; offline only. */
#define PLANT_DEFAULT_DRAG_N_PER_MPS 0.0f /* Explicit Rigidbody drag prevents double counting. */
#define PLANT_DEFAULT_CF_N_PER_RAD 2674.084021452388f /* Manifest candidate Y2. */
#define PLANT_DEFAULT_CR_N_PER_RAD 4853.771529131131f /* Manifest candidate Y2. */
#define PLANT_DEFAULT_DF_N 12.63696205350674f /* Manifest candidate Y2. */
#define PLANT_DEFAULT_DR_N 14.123309383979418f /* Manifest candidate Y2. */
#define PLANT_MIN_SLIP_SPEED_MPS 0.5f
#define PLANT_INTEGRATION_SUBSTEP_S 0.002f

static float clampf_local(float value, float lower, float upper)
{
    if (value < lower)
        return lower;
    if (value > upper)
        return upper;
    return value;
}

static float move_towards(float current, float target, float maximum_delta)
{
    return current + clampf_local(target - current,
                                  -maximum_delta, maximum_delta);
}

VehiclePlantParameters_t vehicle_plant_default_parameters(void)
{
    VehiclePlantParameters_t parameters = {
        .mass_kg = PLANT_DEFAULT_MASS_KG,
        .lf_m = PLANT_DEFAULT_LF_M,
        .lr_m = PLANT_DEFAULT_LR_M,
        .iz_kgm2 = PLANT_DEFAULT_IZ_KGM2,
        .position_offset_from_velocity_point_x_m = PLANT_DEFAULT_POSITION_OFFSET_X_M,
        .max_steering_rad = PLANT_DEFAULT_MAX_STEERING_RAD,
        .steering_rate_radps = PLANT_DEFAULT_STEERING_RATE_RADPS,
        .max_speed_mps = PLANT_DEFAULT_MAX_SPEED_MPS,
        .linear_damping_per_s = PLANT_DEFAULT_LINEAR_DAMPING_PER_S,
        .angular_damping_per_s = PLANT_DEFAULT_ANGULAR_DAMPING_PER_S,
        .force_max_n = PLANT_DEFAULT_FORCE_MAX_N,
        .hard_brake_force_n = PLANT_DEFAULT_HARD_BRAKE_FORCE_N,
        .slip_gain_per_mps = PLANT_DEFAULT_SLIP_GAIN_PER_MPS,
        .coast_speed_drag_n_per_mps = PLANT_DEFAULT_DRAG_N_PER_MPS,
        .cf_n_per_rad = PLANT_DEFAULT_CF_N_PER_RAD,
        .cr_n_per_rad = PLANT_DEFAULT_CR_N_PER_RAD,
        .df_n = PLANT_DEFAULT_DF_N,
        .dr_n = PLANT_DEFAULT_DR_N,
        .tire_model = VEHICLE_PLANT_TIRE_TANH,
        .wheel_dynamics_model = VEHICLE_PLANT_WHEEL_DYNAMICS_CONTINUOUS,
        .wheel_coefficients = {
            -0.05503404f, -38.442266f, 967.47575f,
            1.4255339f, -37.163453f, 0.02705089f,
        },
    };
    return parameters;
}

static float tire_force(float alpha, float stiffness, float peak,
                        uint8_t tire_model)
{
    if (tire_model == VEHICLE_PLANT_TIRE_TANH)
        return peak * tanhf(stiffness * alpha / fmaxf(peak, 1.0e-9f));
    return clampf_local(stiffness * alpha, -peak, peak);
}

static void lateral_forces(float u, float v, float r, float delta,
                           const VehiclePlantParameters_t *p,
                           float *front, float *rear)
{
    const float safe_u = (u < 0.0f) ? -fmaxf(-u, PLANT_MIN_SLIP_SPEED_MPS)
                                   : fmaxf(u, PLANT_MIN_SLIP_SPEED_MPS);
    const float alpha_front = delta - atanf(
        (v + p->lf_m * r) / safe_u);
    const float alpha_rear = -atanf(
        (v - p->lr_m * r) / safe_u);
    *front = tire_force(alpha_front, p->cf_n_per_rad, p->df_n,
                        p->tire_model);
    *rear = tire_force(alpha_rear, p->cr_n_per_rad, p->dr_n,
                       p->tire_model);
}

static float wheel_next(float body_u, float wheel, float throttle, float dt_s,
                        const VehiclePlantParameters_t *p)
{
    const float features[6] = {
        1.0f, wheel, throttle, wheel * throttle,
        throttle * throttle, body_u,
    };
    float result = 0.0f;
    for (int index = 0; index < 6; ++index)
        result += features[index] * p->wheel_coefficients[index];
    if (p->wheel_dynamics_model == VEHICLE_PLANT_WHEEL_DYNAMICS_CONTINUOUS)
        result = wheel + dt_s * result;
    return fmaxf(0.0f, result);
}

void vehicle_plant_step(
    const VehiclePlantState_t *state,
    const VehiclePlantInput_t *input,
    float dt_s,
    const VehiclePlantParameters_t *parameters,
    VehiclePlantState_t *next)
{
    if (state == NULL || input == NULL || parameters == NULL || next == NULL ||
        !(dt_s > 0.0f) || !isfinite(dt_s))
        return;

    *next = *state;
    const float target = clampf_local(input->steering_target_norm, -1.0f, 1.0f) *
                         parameters->max_steering_rad;
    const float throttle = clampf_local(input->throttle_norm, 0.0f, 1.0f);
    const float steering_end = move_towards(
        state->steering_rad, target, parameters->steering_rate_radps * dt_s);
    const float wheel_end = wheel_next(
        state->u_mps, state->wheel_speed_mps, throttle, dt_s, parameters);
    const int count = (int)fmaxf(1.0f, ceilf(dt_s / PLANT_INTEGRATION_SUBSTEP_S));
    const float sub_dt = dt_s / (float)count;

    float x = state->x_m;
    float y = state->y_m;
    float yaw = state->yaw_rad;
    float u = state->u_mps;
    float v = state->v_mps;
    float r = state->r_radps;
    for (int index = 0; index < count; ++index) {
        const float fraction = ((float)index + 0.5f) / (float)count;
        const float delta = state->steering_rad +
            fraction * (steering_end - state->steering_rad);
        const float wheel = state->wheel_speed_mps +
            fraction * (wheel_end - state->wheel_speed_mps);
        float front_force = 0.0f;
        float rear_force = 0.0f;
        lateral_forces(u, v, r, delta, parameters,
                       &front_force, &rear_force);
        // AutoDRIVE's zero-throttle command is not passive coasting: the
        // vehicle controller applies brake torque to all driven wheels. Keep
        // this branch explicit so offline recursive prediction does not
        // systematically under-predict braking distance.
        const float longitudinal_force = throttle > 1.0e-5f ?
            parameters->force_max_n * tanhf(
                parameters->slip_gain_per_mps * (wheel - u)) -
            parameters->coast_speed_drag_n_per_mps * u :
            -fmaxf(0.0f, parameters->hard_brake_force_n) -
            parameters->coast_speed_drag_n_per_mps * u;
        const float u_dot = (longitudinal_force -
                             front_force * sinf(delta)) / parameters->mass_kg + r * v;
        const float v_dot = (front_force * cosf(delta) + rear_force) /
                            parameters->mass_kg - r * u;
        const float r_dot = (parameters->lf_m * front_force * cosf(delta) -
                             parameters->lr_m * rear_force) /
                            parameters->iz_kgm2 -
                            parameters->angular_damping_per_s * r;
        const float pose_v = v +
            parameters->position_offset_from_velocity_point_x_m * r;
        x += sub_dt * (u * cosf(yaw) - pose_v * sinf(yaw));
        y += sub_dt * (u * sinf(yaw) + pose_v * cosf(yaw));
        yaw += sub_dt * r;
        const float damped_u_dot = u_dot -
            parameters->linear_damping_per_s * u;
        const float damped_v_dot = v_dot -
            parameters->linear_damping_per_s * v;
        u = clampf_local(u + sub_dt * damped_u_dot, 0.0f,
                         parameters->max_speed_mps);
        v += sub_dt * damped_v_dot;
        r += sub_dt * r_dot;
    }
    next->x_m = x;
    next->y_m = y;
    next->yaw_rad = yaw;
    next->u_mps = u;
    next->v_mps = v;
    next->r_radps = r;
    next->steering_rad = steering_end;
    next->wheel_speed_mps = wheel_end;
}

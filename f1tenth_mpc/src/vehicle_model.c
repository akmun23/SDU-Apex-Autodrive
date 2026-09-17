/**
 * @file vehicle_model.c
 * @brief Source-command baseline for the current AutoDRIVE Unity object.
 *
 * Unity owns the actual Rigidbody/WheelCollider contact solve.  This module
 * intentionally does not invent a real-car tire law for it.  It carries only
 * the verified steering geometry and the MPC target-speed command policy.
 * The resulting stage map is conservative plumbing for the controller and is
 * explicitly replaced by a measured observable response map before racing
 * promotion (see config/autodrive_simulator_contract.yaml).
 */

#include "vehicle_model.h"

#include <math.h>
#include <stddef.h>

static VehicleParameters_t active_parameters = {
    .steering_wheelbase_m = SOURCE_STEERING_WHEELBASE_M,
    .max_steering_angle = SOURCE_MAX_STEERING_RAD,
    .steering_rate_radps = SOURCE_STEERING_RATE_RADPS,
    .maximum_command_speed_mps = MPC_MAX_COMMAND_SPEED_MPS,
    .maximum_target_speed_rate_increase_mps2 = MPC_TARGET_SPEED_RATE_INCREASE_MAX_MPS2,
    .maximum_target_speed_rate_reduction_mps2 = MPC_TARGET_SPEED_RATE_REDUCTION_MAX_MPS2,
};

static float clampf_local(float value, float lower, float upper)
{
    if (value < lower)
        return lower;
    if (value > upper)
        return upper;
    return value;
}

static int finite_positive(float value)
{
    return isfinite(value) && value > 0.0f;
}

VehicleParameters_t vehicle_model_default_parameters(void)
{
    return (VehicleParameters_t){
        .steering_wheelbase_m = SOURCE_STEERING_WHEELBASE_M,
        .max_steering_angle = SOURCE_MAX_STEERING_RAD,
        .steering_rate_radps = SOURCE_STEERING_RATE_RADPS,
        .maximum_command_speed_mps = MPC_MAX_COMMAND_SPEED_MPS,
        .maximum_target_speed_rate_increase_mps2 = MPC_TARGET_SPEED_RATE_INCREASE_MAX_MPS2,
        .maximum_target_speed_rate_reduction_mps2 = MPC_TARGET_SPEED_RATE_REDUCTION_MAX_MPS2,
    };
}

VehicleParameters_t vehicle_model_get_parameters(void)
{
    return active_parameters;
}

int vehicle_model_set_parameters(const VehicleParameters_t *parameters)
{
    if (parameters == NULL ||
        !finite_positive(parameters->steering_wheelbase_m) ||
        !finite_positive(parameters->max_steering_angle) ||
        !finite_positive(parameters->steering_rate_radps) ||
        !finite_positive(parameters->maximum_command_speed_mps) ||
        !finite_positive(parameters->maximum_target_speed_rate_increase_mps2) ||
        !finite_positive(parameters->maximum_target_speed_rate_reduction_mps2)) {
        return 0;
    }
    active_parameters = *parameters;
    return 1;
}

ControlInput_t vehicle_model_saturate_control(const ControlInput_t *raw_control)
{
    ControlInput_t result = {0.0f, 0.0f};
    if (raw_control == NULL)
        return result;

    if (isfinite(raw_control->steer_ang)) {
        result.steer_ang = clampf_local(
            raw_control->steer_ang,
            -active_parameters.max_steering_angle,
            active_parameters.max_steering_angle);
    }
    if (isfinite(raw_control->target_speed_rate)) {
        result.target_speed_rate = clampf_local(
            raw_control->target_speed_rate,
            -active_parameters.maximum_target_speed_rate_reduction_mps2,
            active_parameters.maximum_target_speed_rate_increase_mps2);
    }
    return result;
}

static float geometric_yaw_rate(float speed_mps, float steering_rad)
{
    /* This is the only nominal lateral relation: Unity's serialized Ackermann
     * geometry. It is deliberately not a tire-force or inertia surrogate. */
    return speed_mps * tanf(steering_rad) /
        active_parameters.steering_wheelbase_m;
}

VehicleState_t vehicle_model_predict_next_state(
    const VehicleState_t *current_state,
    const ControlInput_t *control_input,
    float time_step)
{
    if (current_state == NULL || control_input == NULL ||
        !(time_step > 0.0f) || !isfinite(time_step)) {
        return (VehicleState_t){0};
    }

    const ControlInput_t control = vehicle_model_saturate_control(control_input);
    const float u0 = clampf_local(
        fmaxf(0.0f, current_state->long_vel),
        0.0f, active_parameters.maximum_command_speed_mps);
    const float u1 = clampf_local(
        u0 + time_step * control.target_speed_rate,
        0.0f, active_parameters.maximum_command_speed_mps);
    const float r0 = current_state->yaw_rate;
    const float r1 = geometric_yaw_rate(u1, control.steer_ang);
    const float r_mid = 0.5f * (r0 + r1);
    const float psi_mid = current_state->heading + 0.5f * time_step * r_mid;
    const float u_mid = 0.5f * (u0 + u1);
    const float v = current_state->lat_vel;

    VehicleState_t next = *current_state;
    next.pos_x += time_step * (u_mid * cosf(psi_mid) - v * sinf(psi_mid));
    next.pos_y += time_step * (u_mid * sinf(psi_mid) + v * cosf(psi_mid));
    next.heading = atan2f(
        sinf(current_state->heading + time_step * r_mid),
        cosf(current_state->heading + time_step * r_mid));
    next.long_vel = u1;
    /* A legal-state response map, not this baseline, owns lateral velocity. */
    next.lat_vel = v;
    next.yaw_rate = r1;
    return next;
}

FrenetState_t vehicle_model_predict_next_frenet_state(
    const FrenetState_t *state,
    const ControlInput_t *control_input,
    float time_step,
    float path_curvature)
{
    if (state == NULL || control_input == NULL ||
        !(time_step > 0.0f) || !isfinite(time_step)) {
        return (FrenetState_t){0};
    }

    const ControlInput_t control = vehicle_model_saturate_control(control_input);
    const float u0 = clampf_local(
        fmaxf(0.0f, state->flong_vel),
        0.0f, active_parameters.maximum_command_speed_mps);
    const float u1 = clampf_local(
        u0 + time_step * control.target_speed_rate,
        0.0f, active_parameters.maximum_command_speed_mps);
    const float r1 = geometric_yaw_rate(u1, control.steer_ang);
    const float u_mid = 0.5f * (u0 + u1);
    const float r_mid = 0.5f * (state->fyaw_rate + r1);
    const float heading_mid = state->fhead_error + 0.5f * time_step * r_mid;
    float denominator = 1.0f - path_curvature * state->flat_error;
    if (fabsf(denominator) < 0.05f)
        denominator = denominator < 0.0f ? -0.05f : 0.05f;

    const float path_progress =
        (u_mid * cosf(heading_mid) - state->flat_vel * sinf(heading_mid)) /
        denominator;
    FrenetState_t next = *state;
    next.flat_error += time_step * (
        u_mid * sinf(heading_mid) + state->flat_vel * cosf(heading_mid));
    next.fhead_error = atan2f(
        sinf(state->fhead_error + time_step * (r_mid - path_curvature * path_progress)),
        cosf(state->fhead_error + time_step * (r_mid - path_curvature * path_progress)));
    next.flong_vel = u1;
    next.fyaw_rate = r1;
    return next;
}

void vehicle_model_predict_trajectory(
    const VehicleState_t *initial_state,
    const ControlInput_t *control_sequence,
    float time_step,
    uint16_t step_count,
    VehicleState_t *predicted_trajectory)
{
    if (initial_state == NULL || control_sequence == NULL ||
        predicted_trajectory == NULL) {
        return;
    }
    predicted_trajectory[0] = *initial_state;
    for (uint16_t index = 0; index < step_count; ++index) {
        predicted_trajectory[index + 1] = vehicle_model_predict_next_state(
            &predicted_trajectory[index], &control_sequence[index], time_step);
    }
}

static void state_to_array(const FrenetState_t *state, float values[NX_FRENET])
{
    values[0] = state->flat_error;
    values[1] = state->fhead_error;
    values[2] = state->flong_vel;
    values[3] = state->flat_vel;
    values[4] = state->fyaw_rate;
}

static FrenetState_t array_to_state(const float values[NX_FRENET])
{
    return (FrenetState_t){
        .flat_error = values[0],
        .fhead_error = values[1],
        .flong_vel = values[2],
        .flat_vel = values[3],
        .fyaw_rate = values[4],
    };
}

void vehicle_model_compute_frenet_linearization(
    const FrenetState_t *frenet_state,
    const ControlInput_t *operating_control,
    float time_step,
    float path_curvature,
    float reference_velocity,
    float state_matrix_A[NX_FRENET][NX_FRENET],
    float input_matrix_B[NX_FRENET][NU])
{
    (void)reference_velocity;
    if (frenet_state == NULL || operating_control == NULL ||
        state_matrix_A == NULL || input_matrix_B == NULL) {
        return;
    }

    const float state_epsilon[NX_FRENET] = {
        1.0e-4f, 1.0e-4f, 1.0e-3f, 1.0e-3f, 1.0e-3f};
    const float input_epsilon[NU] = {1.0e-4f, 1.0e-3f};
    float base_values[NX_FRENET];
    state_to_array(frenet_state, base_values);

    for (int column = 0; column < NX_FRENET; ++column) {
        float plus_values[NX_FRENET];
        float minus_values[NX_FRENET];
        for (int index = 0; index < NX_FRENET; ++index) {
            plus_values[index] = base_values[index];
            minus_values[index] = base_values[index];
        }
        plus_values[column] += state_epsilon[column];
        minus_values[column] -= state_epsilon[column];
        const FrenetState_t plus_state = array_to_state(plus_values);
        const FrenetState_t minus_state = array_to_state(minus_values);
        const FrenetState_t plus = vehicle_model_predict_next_frenet_state(
            &plus_state, operating_control, time_step, path_curvature);
        const FrenetState_t minus = vehicle_model_predict_next_frenet_state(
            &minus_state, operating_control, time_step, path_curvature);
        float plus_output[NX_FRENET];
        float minus_output[NX_FRENET];
        state_to_array(&plus, plus_output);
        state_to_array(&minus, minus_output);
        for (int row = 0; row < NX_FRENET; ++row) {
            state_matrix_A[row][column] =
                (plus_output[row] - minus_output[row]) /
                (2.0f * state_epsilon[column]);
        }
    }

    for (int column = 0; column < NU; ++column) {
        ControlInput_t plus = *operating_control;
        ControlInput_t minus = *operating_control;
        if (column == 0) {
            plus.steer_ang += input_epsilon[column];
            minus.steer_ang -= input_epsilon[column];
        } else {
            plus.target_speed_rate += input_epsilon[column];
            minus.target_speed_rate -= input_epsilon[column];
        }
        const FrenetState_t plus_output = vehicle_model_predict_next_frenet_state(
            frenet_state, &plus, time_step, path_curvature);
        const FrenetState_t minus_output = vehicle_model_predict_next_frenet_state(
            frenet_state, &minus, time_step, path_curvature);
        float plus_values[NX_FRENET];
        float minus_values[NX_FRENET];
        state_to_array(&plus_output, plus_values);
        state_to_array(&minus_output, minus_values);
        for (int row = 0; row < NX_FRENET; ++row) {
            input_matrix_B[row][column] =
                (plus_values[row] - minus_values[row]) /
                (2.0f * input_epsilon[column]);
        }
    }
}

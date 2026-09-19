/**
 * @file vehicle_model.c
 * @brief Source-command baseline for the current AutoDRIVE Unity object.
 *
 * Unity owns the actual Rigidbody/WheelCollider contact solve.  This module
 * intentionally does not invent a real-car tire law for it. It combines
 * verified command geometry with the held-out AutoDRIVE yaw-rate response
 * identified from clean 40 Hz raceline captures. Speed/braking response
 * remains a separate model gap before racing promotion (see
 * config/autodrive_simulator_contract.yaml).
 */

#include "vehicle_model.h"
#include "mpc_linearization.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static VehicleParameters_t active_parameters = {
    .steering_wheelbase_m = SOURCE_STEERING_WHEELBASE_M,
    .max_steering_angle = SOURCE_MAX_STEERING_RAD,
    .steering_rate_radps = SOURCE_STEERING_RATE_RADPS,
    .maximum_command_speed_mps = MPC_MAX_COMMAND_SPEED_MPS,
    .maximum_target_speed_rate_increase_mps2 = MPC_TARGET_SPEED_RATE_INCREASE_MAX_MPS2,
    .maximum_target_speed_rate_reduction_mps2 = MPC_TARGET_SPEED_RATE_REDUCTION_MAX_MPS2,
};
static float active_target_speed_ceiling_mps = MPC_MAX_COMMAND_SPEED_MPS;

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
    if (active_target_speed_ceiling_mps > active_parameters.maximum_command_speed_mps)
        active_target_speed_ceiling_mps = active_parameters.maximum_command_speed_mps;
    return 1;
}

float vehicle_model_get_active_target_speed_ceiling(void)
{
    return active_target_speed_ceiling_mps;
}

int vehicle_model_set_active_target_speed_ceiling(float ceiling_mps)
{
    if (!isfinite(ceiling_mps) || ceiling_mps <= 0.0f ||
        ceiling_mps > active_parameters.maximum_command_speed_mps) {
        return 0;
    }
    active_target_speed_ceiling_mps = ceiling_mps;
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

static float yaw_rate_response(
    float speed_mps, float steering_rad, float yaw_rate_radps, float time_step)
{
    /* Exact zero-order-hold update of the identified first-order response:
     * r_dot = (-r + gain * u * tan(delta)) / tau. The gain is fitted from
     * AutoDRIVE data and is intentionally separate from Unity's 0.324 m
     * steering-geometry parameter. */
    const float steady_yaw_rate = speed_mps * tanf(steering_rad) *
        MPC_YAW_RATE_STEERING_GAIN_PER_M;
    const float retention = expf(
        -time_step / MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS);
    return retention * yaw_rate_radps +
        (1.0f - retention) * steady_yaw_rate;
}

static float longitudinal_speed_response(
    float speed_mps,
    float target_speed_mps,
    float target_speed_rate_mps2,
    float time_step)
{
    const float target_next = clampf_local(
        target_speed_mps + target_speed_rate_mps2 * time_step,
        0.0f, active_target_speed_ceiling_mps);
    const float target_mid = 0.5f * (target_speed_mps + target_next);
    float acceleration =
        MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 +
        MPC_LONGITUDINAL_SPEED_COEFF_PER_S * speed_mps +
        MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S * (target_mid - speed_mps) +
        MPC_LONGITUDINAL_TARGET_RATE_COEFF * target_speed_rate_mps2;
    const float braking_limit =
        MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2 +
        MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV * speed_mps;
    acceleration = clampf_local(
        acceleration, -braking_limit, MPC_LONGITUDINAL_ACCEL_LIMIT_MPS2);
    return clampf_local(
        speed_mps + acceleration * time_step,
        0.0f, active_parameters.maximum_command_speed_mps);
}

static int finite_mpc_state(const MpcModelState_t *state)
{
    return state && isfinite(state->e_y) && isfinite(state->e_psi) &&
        isfinite(state->u) && isfinite(state->v) && isfinite(state->r) &&
        isfinite(state->target_speed) && isfinite(state->steering_command);
}

enum { MPC_JET_DERIVATIVES = 9 };

typedef struct
{
    float value;
    float derivative[MPC_JET_DERIVATIVES];
    int differentiated;
} MpcJet_t;

static MpcJet_t jet_constant(float value)
{
    MpcJet_t result = {0};
    result.value = value;
    return result;
}

static MpcJet_t jet_variable(float value, int index, int differentiate)
{
    MpcJet_t result = jet_constant(value);
    if (differentiate) {
        result.derivative[index] = 1.0f;
        result.differentiated = 1;
    }
    return result;
}

static MpcJet_t jet_add(MpcJet_t lhs, MpcJet_t rhs)
{
    MpcJet_t result = jet_constant(lhs.value + rhs.value);
    result.differentiated = lhs.differentiated || rhs.differentiated;
    if (result.differentiated) {
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            result.derivative[i] = lhs.derivative[i] + rhs.derivative[i];
    }
    return result;
}

static MpcJet_t jet_subtract(MpcJet_t lhs, MpcJet_t rhs)
{
    MpcJet_t result = jet_constant(lhs.value - rhs.value);
    result.differentiated = lhs.differentiated || rhs.differentiated;
    if (result.differentiated) {
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            result.derivative[i] = lhs.derivative[i] - rhs.derivative[i];
    }
    return result;
}

static MpcJet_t jet_multiply(MpcJet_t lhs, MpcJet_t rhs)
{
    MpcJet_t result = jet_constant(lhs.value * rhs.value);
    result.differentiated = lhs.differentiated || rhs.differentiated;
    if (result.differentiated) {
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i) {
            result.derivative[i] = lhs.derivative[i] * rhs.value +
                lhs.value * rhs.derivative[i];
        }
    }
    return result;
}

static MpcJet_t jet_divide(MpcJet_t numerator, MpcJet_t denominator)
{
    MpcJet_t result = jet_constant(numerator.value / denominator.value);
    result.differentiated = numerator.differentiated || denominator.differentiated;
    if (result.differentiated) {
        const float denominator_squared = denominator.value * denominator.value;
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i) {
            result.derivative[i] =
                (numerator.derivative[i] * denominator.value -
                 numerator.value * denominator.derivative[i]) /
                denominator_squared;
        }
    }
    return result;
}

static MpcJet_t jet_sine(MpcJet_t input)
{
    MpcJet_t result = jet_constant(sinf(input.value));
    result.differentiated = input.differentiated;
    if (input.differentiated) {
        const float scale = cosf(input.value);
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            result.derivative[i] = scale * input.derivative[i];
    }
    return result;
}

static MpcJet_t jet_cosine(MpcJet_t input)
{
    MpcJet_t result = jet_constant(cosf(input.value));
    result.differentiated = input.differentiated;
    if (input.differentiated) {
        const float scale = -sinf(input.value);
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            result.derivative[i] = scale * input.derivative[i];
    }
    return result;
}

static MpcJet_t jet_tangent(MpcJet_t input)
{
    MpcJet_t result = jet_constant(tanf(input.value));
    result.differentiated = input.differentiated;
    if (input.differentiated) {
        const float scale = 1.0f + result.value * result.value;
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            result.derivative[i] = scale * input.derivative[i];
    }
    return result;
}

static void mark_nonsmooth_crossing(
    MpcJet_t boundary_difference,
    float boundary_scale,
    MpcStageLinearization_t *linearization)
{
    if (linearization == NULL || !boundary_difference.differentiated) return;

    /* Match the perturbation sizes used by the retained FD oracle. A bit is
     * set only when that column's oracle probe could cross this branch. */
    static const float probe_epsilon[MPC_JET_DERIVATIVES] = {
        1.0e-4f, 1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-3f,
        1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-2f};
    const float roundoff = 2.0e-7f * fmaxf(1.0f, fabsf(boundary_scale));
    for (int i = 0; i < MPC_JET_DERIVATIVES; ++i) {
        const float reach = probe_epsilon[i] *
            fabsf(boundary_difference.derivative[i]);
        if (fabsf(boundary_difference.value) <= reach + roundoff)
            linearization->nonsmooth_column_mask |= (uint16_t)(1u << i);
    }
}

static MpcJet_t jet_clip(
    MpcJet_t input,
    MpcJet_t lower,
    MpcJet_t upper,
    unsigned int clipped_flag,
    MpcStageResult_t *stage,
    MpcStageLinearization_t *linearization)
{
    mark_nonsmooth_crossing(
        jet_subtract(input, lower), lower.value, linearization);
    mark_nonsmooth_crossing(
        jet_subtract(input, upper), upper.value, linearization);
    if (input.value < lower.value) {
        stage->branch_flags |= clipped_flag;
        return lower;
    }
    if (input.value > upper.value) {
        stage->branch_flags |= clipped_flag;
        return upper;
    }
    return input;
}

static unsigned int mask_popcount(uint16_t mask)
{
    unsigned int count = 0;
    while (mask != 0u) {
        count += mask & 1u;
        mask >>= 1u;
    }
    return count;
}

static int vehicle_model_step_impl(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    MpcStageResult_t *stage,
    MpcStageLinearization_t *linearization,
    int differentiate)
{
    if (stage == NULL || (differentiate && linearization == NULL)) return 0;
    *stage = (MpcStageResult_t){0};
    if (linearization != NULL)
        memset(linearization, 0, sizeof(*linearization));
    if (!finite_mpc_state(state) || !control ||
        !isfinite(control->steering_rate) ||
        !isfinite(control->target_speed_rate) ||
        !(dt > 0.0f) || !isfinite(dt) || !isfinite(path_curvature)) {
        return 0;
    }

    const VehicleParameters_t parameters = vehicle_model_get_parameters();
    MpcJet_t x[7] = {
        jet_variable(state->e_y, 0, differentiate),
        jet_variable(state->e_psi, 1, differentiate),
        jet_variable(state->u, 2, differentiate),
        jet_variable(state->v, 3, differentiate),
        jet_variable(state->r, 4, differentiate),
        jet_variable(state->target_speed, 5, differentiate),
        jet_variable(state->steering_command, 6, differentiate)};
    MpcJet_t w[2] = {
        jet_variable(control->steering_rate, 7, differentiate),
        jet_variable(control->target_speed_rate, 8, differentiate)};

    MpcJet_t q_delta = jet_clip(w[0],
        jet_constant(-SOURCE_STEERING_RATE_RADPS),
        jet_constant(SOURCE_STEERING_RATE_RADPS),
        MPC_STAGE_CLIPPED_STEERING_RATE, stage, linearization);
    MpcJet_t q_speed = jet_clip(w[1],
        jet_constant(-parameters.maximum_target_speed_rate_reduction_mps2),
        jet_constant(parameters.maximum_target_speed_rate_increase_mps2),
        MPC_STAGE_CLIPPED_SPEED_RATE, stage, linearization);

    MpcJet_t delta_raw = jet_add(x[6], jet_multiply(jet_constant(dt), q_delta));
    MpcJet_t delta_next = jet_clip(delta_raw,
        jet_constant(-parameters.max_steering_angle),
        jet_constant(parameters.max_steering_angle),
        MPC_STAGE_CLIPPED_STEERING_COMMAND, stage, linearization);

    MpcJet_t target_raw = jet_add(x[5],
        jet_multiply(jet_constant(dt), q_speed));
    MpcJet_t target_next = jet_clip(target_raw, jet_constant(0.0f),
        jet_constant(active_target_speed_ceiling_mps),
        MPC_STAGE_CLIPPED_TARGET_SPEED, stage, linearization);
    MpcJet_t target_mid = jet_multiply(jet_constant(0.5f),
        jet_add(x[5], target_next));

    MpcJet_t u0 = jet_clip(x[2], jet_constant(0.0f),
        jet_constant(parameters.maximum_command_speed_mps),
        0u, stage, linearization);
    MpcJet_t acceleration = jet_add(
        jet_constant(MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2),
        jet_multiply(jet_constant(MPC_LONGITUDINAL_SPEED_COEFF_PER_S), u0));
    acceleration = jet_add(acceleration, jet_multiply(
        jet_constant(MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S),
        jet_subtract(target_mid, u0)));
    acceleration = jet_add(acceleration, jet_multiply(
        jet_constant(MPC_LONGITUDINAL_TARGET_RATE_COEFF), q_speed));
    MpcJet_t brake_limit = jet_add(
        jet_constant(MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2),
        jet_multiply(
            jet_constant(MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV), u0));
    acceleration = jet_clip(acceleration,
        jet_multiply(jet_constant(-1.0f), brake_limit),
        jet_constant(MPC_LONGITUDINAL_ACCEL_LIMIT_MPS2),
        MPC_STAGE_CLIPPED_ACCELERATION, stage, linearization);

    MpcJet_t u_raw = jet_add(u0, jet_multiply(jet_constant(dt), acceleration));
    MpcJet_t u_next = jet_clip(u_raw, jet_constant(0.0f),
        jet_constant(parameters.maximum_command_speed_mps),
        MPC_STAGE_CLIPPED_BODY_SPEED, stage, linearization);
    MpcJet_t u_mid = jet_multiply(jet_constant(0.5f), jet_add(u0, u_next));

    const float yaw_retention = expf(
        -dt / MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS);
    MpcJet_t yaw_steady = jet_multiply(
        jet_multiply(
            jet_constant(MPC_YAW_RATE_STEERING_GAIN_PER_M), u_mid),
        jet_tangent(delta_next));
    MpcJet_t r_next = jet_add(
        jet_multiply(jet_constant(yaw_retention), x[4]),
        jet_multiply(jet_constant(1.0f - yaw_retention), yaw_steady));
    MpcJet_t r_mid = jet_multiply(jet_constant(0.5f), jet_add(x[4], r_next));

    MpcJet_t denominator0 = jet_subtract(jet_constant(1.0f),
        jet_multiply(jet_constant(path_curvature), x[0]));
    mark_nonsmooth_crossing(
        jet_subtract(denominator0, jet_constant(0.05f)), 0.05f,
        linearization);
    mark_nonsmooth_crossing(
        jet_add(denominator0, jet_constant(0.05f)), 0.05f,
        linearization);
    if (fabsf(denominator0.value) < 0.05f) return 0;
    MpcJet_t s_dot0 = jet_divide(
        jet_subtract(
            jet_multiply(u0, jet_cosine(x[1])),
            jet_multiply(x[3], jet_sine(x[1]))),
        denominator0);
    MpcJet_t ey_dot0 = jet_add(
        jet_multiply(u0, jet_sine(x[1])),
        jet_multiply(x[3], jet_cosine(x[1])));
    MpcJet_t epsi_dot0 = jet_subtract(x[4],
        jet_multiply(jet_constant(path_curvature), s_dot0));
    MpcJet_t ey_mid = jet_add(x[0],
        jet_multiply(jet_constant(0.5f * dt), ey_dot0));
    MpcJet_t epsi_mid = jet_add(x[1],
        jet_multiply(jet_constant(0.5f * dt), epsi_dot0));
    MpcJet_t denominator_mid = jet_subtract(jet_constant(1.0f),
        jet_multiply(jet_constant(path_curvature), ey_mid));
    mark_nonsmooth_crossing(
        jet_subtract(denominator_mid, jet_constant(0.05f)), 0.05f,
        linearization);
    mark_nonsmooth_crossing(
        jet_add(denominator_mid, jet_constant(0.05f)), 0.05f,
        linearization);
    if (fabsf(denominator_mid.value) < 0.05f) return 0;

    MpcJet_t s_dot_mid = jet_divide(
        jet_subtract(
            jet_multiply(u_mid, jet_cosine(epsi_mid)),
            jet_multiply(x[3], jet_sine(epsi_mid))),
        denominator_mid);
    MpcJet_t ey_dot_mid = jet_add(
        jet_multiply(u_mid, jet_sine(epsi_mid)),
        jet_multiply(x[3], jet_cosine(epsi_mid)));
    MpcJet_t epsi_dot_mid = jet_subtract(r_mid,
        jet_multiply(jet_constant(path_curvature), s_dot_mid));

    MpcJet_t e_y_next = jet_add(x[0],
        jet_multiply(jet_constant(dt), ey_dot_mid));
    MpcJet_t e_psi_unwrapped = jet_add(x[1],
        jet_multiply(jet_constant(dt), epsi_dot_mid));
    MpcJet_t next[7] = {
        e_y_next,
        jet_constant(atan2f(sinf(e_psi_unwrapped.value),
                            cosf(e_psi_unwrapped.value))),
        u_next,
        x[3],
        r_next,
        target_next,
        delta_next};
    /* Wrapping changes the coordinate value, not its local derivative. */
    next[1].differentiated = e_psi_unwrapped.differentiated;
    if (next[1].differentiated) {
        memcpy(next[1].derivative, e_psi_unwrapped.derivative,
               sizeof(next[1].derivative));
    }

    stage->next = (MpcModelState_t){
        .e_y = next[0].value,
        .e_psi = next[1].value,
        .u = next[2].value,
        .v = next[3].value,
        .r = next[4].value,
        .target_speed = next[5].value,
        .steering_command = next[6].value};
    stage->delta_s_m = dt * s_dot_mid.value;
    stage->body_accel_mps2 = acceleration.value;
    stage->valid = finite_mpc_state(&stage->next) &&
        isfinite(stage->delta_s_m) && isfinite(stage->body_accel_mps2);
    if (!stage->valid) return 0;

    if (linearization != NULL) {
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 7; ++column)
                linearization->A[row][column] = next[row].derivative[column];
            for (int input = 0; input < 2; ++input)
                linearization->B[row][input] = next[row].derivative[7 + input];
        }
        linearization->nominal_branch_flags = stage->branch_flags;
        linearization->nonsmooth_column_count =
            (int)mask_popcount(linearization->nonsmooth_column_mask);
        linearization->valid = 1;
    }
    return 1;
}

int mpc_vehicle_model_step_with_jacobian(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    MpcStageResult_t *stage,
    MpcStageLinearization_t *linearization)
{
    return vehicle_model_step_impl(
        state, control, dt, path_curvature, stage, linearization, 1);
}

MpcStageResult_t mpc_vehicle_model_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature)
{
    MpcStageResult_t stage = {0};
    /* The fused map is authoritative. Nonlinear rollouts use the same
     * intermediate expressions but skip derivative propagation. */
    if (!vehicle_model_step_impl(
            state, control, dt, path_curvature, &stage, NULL, 0)) {
        stage.valid = 0;
    }
    return stage;
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
    const float target_speed0 = clampf_local(
        current_state->target_speed_mps,
        0.0f, active_target_speed_ceiling_mps);
    const float target_speed1 = clampf_local(
        target_speed0 + time_step * control.target_speed_rate,
        0.0f, active_target_speed_ceiling_mps);
    const float u1 = longitudinal_speed_response(
        u0, target_speed0, control.target_speed_rate, time_step);
    const float r0 = current_state->yaw_rate;
    const float u_mid = 0.5f * (u0 + u1);
    const float r1 = yaw_rate_response(
        u_mid, control.steer_ang, r0, time_step);
    const float r_mid = 0.5f * (r0 + r1);
    const float psi_mid = current_state->heading + 0.5f * time_step * r_mid;
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
    next.target_speed_mps = target_speed1;
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
    const float target_speed0 = clampf_local(
        state->ftarget_speed_mps,
        0.0f, active_target_speed_ceiling_mps);
    const float target_speed1 = clampf_local(
        target_speed0 + time_step * control.target_speed_rate,
        0.0f, active_target_speed_ceiling_mps);
    const float u1 = longitudinal_speed_response(
        u0, target_speed0, control.target_speed_rate, time_step);
    const float u_mid = 0.5f * (u0 + u1);
    const float r1 = yaw_rate_response(
        u_mid, control.steer_ang, state->fyaw_rate, time_step);
    const float r_mid = 0.5f * (state->fyaw_rate + r1);
    const float v = state->flat_vel;
    const float denominator0 = 1.0f - path_curvature * state->flat_error;
    float safe_denominator0 = denominator0;
    if (fabsf(safe_denominator0) < 0.05f)
        safe_denominator0 = safe_denominator0 < 0.0f ? -0.05f : 0.05f;

    /* Midpoint/RK2 Frenet kinematics.  The initial heading-error derivative
     * must include path-frame rotation (r - kappa*s_dot); omitting it creates
     * artificial lateral motion even for ideal constant-curvature following. */
    const float s_dot0 =
        (u0 * cosf(state->fhead_error) - v * sinf(state->fhead_error)) /
        safe_denominator0;
    const float ey_dot0 =
        u0 * sinf(state->fhead_error) + v * cosf(state->fhead_error);
    const float epsi_dot0 = state->fyaw_rate - path_curvature * s_dot0;
    const float ey_mid = state->flat_error + 0.5f * time_step * ey_dot0;
    const float epsi_mid = state->fhead_error + 0.5f * time_step * epsi_dot0;
    const float denominator_mid = 1.0f - path_curvature * ey_mid;
    float safe_denominator_mid = denominator_mid;
    if (fabsf(safe_denominator_mid) < 0.05f)
        safe_denominator_mid = safe_denominator_mid < 0.0f ? -0.05f : 0.05f;
    const float s_dot_mid =
        (u_mid * cosf(epsi_mid) - v * sinf(epsi_mid)) / safe_denominator_mid;
    const float ey_dot_mid =
        u_mid * sinf(epsi_mid) + v * cosf(epsi_mid);
    const float epsi_dot_mid = r_mid - path_curvature * s_dot_mid;
    FrenetState_t next = *state;
    next.flat_error += time_step * ey_dot_mid;
    next.fhead_error = atan2f(
        sinf(state->fhead_error + time_step * epsi_dot_mid),
        cosf(state->fhead_error + time_step * epsi_dot_mid));
    next.flong_vel = u1;
    next.fyaw_rate = r1;
    next.ftarget_speed_mps = target_speed1;
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
    values[5] = state->ftarget_speed_mps;
}

static FrenetState_t array_to_state(const float values[NX_FRENET])
{
    return (FrenetState_t){
        .flat_error = values[0],
        .fhead_error = values[1],
        .flong_vel = values[2],
        .flat_vel = values[3],
        .fyaw_rate = values[4],
        .ftarget_speed_mps = values[5],
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
        1.0e-4f, 1.0e-4f, 1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-3f};
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

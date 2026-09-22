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
static MpcYawRateModelParameters_t active_yaw_rate_parameters = {
    .response_time_constant_s = MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS,
    .steering_gain_per_m = MPC_YAW_RATE_STEERING_GAIN_PER_M,
    .steering_gain_reduction_per_rad = 0.0f,
    .steering_gain_start_rad = 0.41f,
    .steering_gain_end_rad = 0.46f,
    .curvature_gain_reduction_per_m = 0.0f,
    .curvature_gain_start_per_m = 0.20f,
    .curvature_gain_end_per_m = 0.40f,
    .low_speed_response_time_constant_s = 0.0f,
    .low_speed_transition_speed_mps = 2.0f,
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

float vehicle_model_yaw_rate_gain(float steering_rad)
{
    if (!isfinite(steering_rad)) return NAN;
    const float magnitude = fabsf(steering_rad);
    const float active_interval = clampf_local(
        magnitude - active_yaw_rate_parameters.steering_gain_start_rad,
        0.0f,
        active_yaw_rate_parameters.steering_gain_end_rad -
            active_yaw_rate_parameters.steering_gain_start_rad);
    return active_yaw_rate_parameters.steering_gain_per_m -
        active_yaw_rate_parameters.steering_gain_reduction_per_rad *
            active_interval;
}

MpcYawRateModelParameters_t vehicle_model_default_yaw_rate_parameters(void)
{
    return (MpcYawRateModelParameters_t){
        .response_time_constant_s = MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS,
        .steering_gain_per_m = MPC_YAW_RATE_STEERING_GAIN_PER_M,
        .steering_gain_reduction_per_rad = 0.0f,
        .steering_gain_start_rad = 0.41f,
        .steering_gain_end_rad = 0.46f,
        .curvature_gain_reduction_per_m = 0.0f,
        .curvature_gain_start_per_m = 0.20f,
        .curvature_gain_end_per_m = 0.40f,
        .low_speed_response_time_constant_s = 0.0f,
        .low_speed_transition_speed_mps = 2.0f,
    };
}

MpcYawRateModelParameters_t vehicle_model_get_yaw_rate_parameters(void)
{
    return active_yaw_rate_parameters;
}

int vehicle_model_set_yaw_rate_parameters(
    const MpcYawRateModelParameters_t *parameters)
{
    if (parameters == NULL ||
        !finite_positive(parameters->response_time_constant_s) ||
        !finite_positive(parameters->steering_gain_per_m) ||
        !isfinite(parameters->steering_gain_reduction_per_rad) ||
        parameters->steering_gain_reduction_per_rad < 0.0f ||
        !isfinite(parameters->steering_gain_start_rad) ||
        parameters->steering_gain_start_rad < 0.0f ||
        !isfinite(parameters->steering_gain_end_rad) ||
        parameters->steering_gain_end_rad <= parameters->steering_gain_start_rad ||
        !isfinite(parameters->curvature_gain_reduction_per_m) ||
        parameters->curvature_gain_reduction_per_m < 0.0f ||
        !isfinite(parameters->curvature_gain_start_per_m) ||
        parameters->curvature_gain_start_per_m < 0.0f ||
        !isfinite(parameters->curvature_gain_end_per_m) ||
        parameters->curvature_gain_end_per_m <=
            parameters->curvature_gain_start_per_m ||
        !isfinite(parameters->low_speed_response_time_constant_s) ||
        parameters->low_speed_response_time_constant_s < 0.0f ||
        (parameters->low_speed_response_time_constant_s > 0.0f &&
            parameters->low_speed_response_time_constant_s <
                parameters->response_time_constant_s) ||
        !isfinite(parameters->low_speed_transition_speed_mps) ||
        (parameters->low_speed_response_time_constant_s > 0.0f &&
            parameters->low_speed_transition_speed_mps <= 0.0f) ||
        parameters->steering_gain_per_m -
                parameters->curvature_gain_reduction_per_m *
                    (parameters->curvature_gain_end_per_m -
                    parameters->curvature_gain_start_per_m) -
                parameters->steering_gain_reduction_per_rad *
                    (parameters->steering_gain_end_rad -
                    parameters->steering_gain_start_rad) < 0.01f) {
        return 0;
    }
    active_yaw_rate_parameters = *parameters;
    return 1;
}

float vehicle_model_yaw_rate_gain_for_curvature(float curvature_radpm)
{
    if (!isfinite(curvature_radpm)) return NAN;
    const float magnitude = fabsf(curvature_radpm);
    const float start = active_yaw_rate_parameters.curvature_gain_start_per_m;
    const float end = active_yaw_rate_parameters.curvature_gain_end_per_m;
    const float active_interval = clampf_local(
        magnitude - start, 0.0f, end - start);
    return active_yaw_rate_parameters.steering_gain_per_m -
        active_yaw_rate_parameters.curvature_gain_reduction_per_m *
            active_interval;
}

float vehicle_model_steering_for_curvature(float curvature_radpm)
{
    if (!isfinite(curvature_radpm)) return NAN;
    /* Feed-forward and prediction use the same identified response relation. */
    const float steering = atanf(
        curvature_radpm / vehicle_model_yaw_rate_gain_for_curvature(
            curvature_radpm));
    return clampf_local(steering, -SOURCE_MAX_STEERING_RAD,
        SOURCE_MAX_STEERING_RAD);
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

static float yaw_rate_response(
    float speed_mps, float steering_rad, float yaw_rate_radps, float time_step,
    float path_curvature)
{
    /* Use the identified first-order source-response map directly.  The
     * newer tanh/u^2 saturation looked plausible offline but failed the live
     * authority A/B at the first high-curvature transition: it caused the
     * N30 steering solution to reverse near s=34.6 m.  This is a Unity
     * response fit, not a real-car tire or friction model. */
    const float path_gain = vehicle_model_yaw_rate_gain_for_curvature(path_curvature);
    const float steering_gain = vehicle_model_yaw_rate_gain(steering_rad);
    const float nominal_gain = active_yaw_rate_parameters.steering_gain_per_m;
    const float steady_yaw_rate = speed_mps * tanf(steering_rad) *
        (steering_gain - nominal_gain + path_gain);
    const float response_time_constant =
        active_yaw_rate_parameters.low_speed_response_time_constant_s > 0.0f
        ? active_yaw_rate_parameters.response_time_constant_s +
            (active_yaw_rate_parameters.low_speed_response_time_constant_s -
                active_yaw_rate_parameters.response_time_constant_s) *
            expf(-fmaxf(speed_mps, 0.0f) /
                active_yaw_rate_parameters.low_speed_transition_speed_mps)
        : active_yaw_rate_parameters.response_time_constant_s;
    const float retention = expf(-time_step / response_time_constant);
    return retention * yaw_rate_radps +
        (1.0f - retention) * steady_yaw_rate;
}

static int finite_mpc_state(const MpcModelState_t *state)
{
    return state && isfinite(state->e_y) && isfinite(state->e_psi) &&
        isfinite(state->u) && isfinite(state->v) && isfinite(state->r) &&
        isfinite(state->target_speed) && isfinite(state->steering_command) &&
        isfinite(state->actual_steering_angle);
}

enum { MPC_JET_DERIVATIVES = 12 };

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
    if (!rhs.differentiated && rhs.value == 0.0f) return lhs;
    if (!lhs.differentiated && lhs.value == 0.0f) return rhs;
    MpcJet_t result = jet_constant(lhs.value + rhs.value);
    result.differentiated = lhs.differentiated || rhs.differentiated;
    if (!result.differentiated) return result;
    if (!lhs.differentiated) {
        memcpy(result.derivative, rhs.derivative, sizeof(result.derivative));
        return result;
    }
    if (!rhs.differentiated) {
        memcpy(result.derivative, lhs.derivative, sizeof(result.derivative));
        return result;
    }
    for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
        result.derivative[i] = lhs.derivative[i] + rhs.derivative[i];
    return result;
}

static MpcJet_t jet_subtract(MpcJet_t lhs, MpcJet_t rhs)
{
    if (!rhs.differentiated && rhs.value == 0.0f) return lhs;
    MpcJet_t result = jet_constant(lhs.value - rhs.value);
    result.differentiated = lhs.differentiated || rhs.differentiated;
    if (!result.differentiated) return result;
    if (!rhs.differentiated) {
        memcpy(result.derivative, lhs.derivative, sizeof(result.derivative));
        return result;
    }
    if (!lhs.differentiated) {
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            result.derivative[i] = -rhs.derivative[i];
        return result;
    }
    for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
        result.derivative[i] = lhs.derivative[i] - rhs.derivative[i];
    return result;
}

static MpcJet_t jet_multiply(MpcJet_t lhs, MpcJet_t rhs)
{
    MpcJet_t result = jet_constant(lhs.value * rhs.value);
    if ((!lhs.differentiated && lhs.value == 0.0f) ||
        (!rhs.differentiated && rhs.value == 0.0f)) return result;
    result.differentiated = lhs.differentiated || rhs.differentiated;
    if (!result.differentiated) return result;
    if (!rhs.differentiated) {
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            result.derivative[i] = lhs.derivative[i] * rhs.value;
        return result;
    }
    if (!lhs.differentiated) {
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            result.derivative[i] = lhs.value * rhs.derivative[i];
        return result;
    }
    for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
        result.derivative[i] = lhs.derivative[i] * rhs.value +
            lhs.value * rhs.derivative[i];
    return result;
}

static MpcJet_t jet_divide(MpcJet_t numerator, MpcJet_t denominator)
{
    MpcJet_t result = jet_constant(numerator.value / denominator.value);
    if (!numerator.differentiated && numerator.value == 0.0f) return result;
    result.differentiated = numerator.differentiated || denominator.differentiated;
    if (!result.differentiated) return result;
    const float inverse_denominator = 1.0f / denominator.value;
    if (!denominator.differentiated) {
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            result.derivative[i] = numerator.derivative[i] * inverse_denominator;
        return result;
    }
    const float numerator_scale = -numerator.value * inverse_denominator *
        inverse_denominator;
    for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
        result.derivative[i] = numerator.derivative[i] * inverse_denominator +
            numerator_scale * denominator.derivative[i];
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

static MpcJet_t jet_exponential(MpcJet_t input)
{
    MpcJet_t result = jet_constant(expf(input.value));
    result.differentiated = input.differentiated;
    if (input.differentiated) {
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            result.derivative[i] = result.value * input.derivative[i];
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
        1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-2f};
    const float roundoff = 2.0e-7f * fmaxf(1.0f, fabsf(boundary_scale));
    for (int i = 0; i < MPC_JET_DERIVATIVES; ++i) {
        const float reach = probe_epsilon[i] *
            fabsf(boundary_difference.derivative[i]);
        if (fabsf(boundary_difference.value) <= reach + roundoff)
            linearization->nonsmooth_column_mask |= (uint16_t)(1u << i);
    }
}

static void mark_nonsmooth_difference(
    MpcJet_t lhs,
    MpcJet_t rhs,
    float boundary_scale,
    MpcStageLinearization_t *linearization)
{
    if (linearization == NULL ||
        !(lhs.differentiated || rhs.differentiated)) return;
    static const float probe_epsilon[MPC_JET_DERIVATIVES] = {
        1.0e-4f, 1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-3f,
        1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-2f};
    const float difference = lhs.value - rhs.value;
    const float roundoff = 2.0e-7f * fmaxf(1.0f, fabsf(boundary_scale));
    for (int i = 0; i < MPC_JET_DERIVATIVES; ++i) {
        const float reach = probe_epsilon[i] * fabsf(
            lhs.derivative[i] - rhs.derivative[i]);
        if (fabsf(difference) <= reach + roundoff)
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
    mark_nonsmooth_difference(input, lower, lower.value, linearization);
    mark_nonsmooth_difference(input, upper, upper.value, linearization);
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

/* Nonlinear rollouts dominate each RTI cycle. They do not need derivatives,
 * so keep them on a scalar path instead of constructing nine-component jets
 * and multiplying their zero derivative slots. The equations intentionally
 * mirror vehicle_model_step_impl() below; the differentiated path remains the
 * single source for the analytic Jacobian. */
static int vehicle_model_step_scalar(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    MpcStageResult_t *stage)
{
    if (stage == NULL) return 0;
    *stage = (MpcStageResult_t){0};
    if (!finite_mpc_state(state) || control == NULL ||
        !isfinite(control->steering_rate) ||
        !isfinite(control->target_speed_rate) ||
        !(dt > 0.0f) || !isfinite(dt) || !isfinite(path_curvature)) {
        return 0;
    }

    const VehicleParameters_t parameters = vehicle_model_get_parameters();
    const float q_delta = clampf_local(control->steering_rate,
        -parameters.steering_rate_radps, parameters.steering_rate_radps);
    const float q_speed = clampf_local(control->target_speed_rate,
        -parameters.maximum_target_speed_rate_reduction_mps2,
        parameters.maximum_target_speed_rate_increase_mps2);
    if (q_delta != control->steering_rate)
        stage->branch_flags |= MPC_STAGE_CLIPPED_STEERING_RATE;
    if (q_speed != control->target_speed_rate)
        stage->branch_flags |= MPC_STAGE_CLIPPED_SPEED_RATE;

    const float delta_raw = state->steering_command + dt * q_delta;
    const float delta_next = clampf_local(delta_raw,
        -parameters.max_steering_angle, parameters.max_steering_angle);
    if (delta_next != delta_raw)
        stage->branch_flags |= MPC_STAGE_CLIPPED_STEERING_COMMAND;

    /* VehicleController keeps a physical steering angle separate from the
     * autonomous target. The bridge/source path exposes a two-sample command
     * queue, so the physical state follows the oldest queued target during
     * this interval. Using delta_next directly for yaw makes a reversal look
     * instantaneous and causes the N10--N30 horizon to turn too early. */
    const float actual_rate_raw =
        (state->delayed_steering_command_2 -
            state->actual_steering_angle) / dt;
    const float actual_rate = clampf_local(actual_rate_raw,
        -parameters.steering_rate_radps, parameters.steering_rate_radps);
    const float actual_raw = state->actual_steering_angle + dt * actual_rate;
    const float actual_next = clampf_local(actual_raw,
        -parameters.max_steering_angle, parameters.max_steering_angle);
    if (actual_rate != actual_rate_raw || actual_next != actual_raw)
        stage->branch_flags |= MPC_STAGE_CLIPPED_ACTUAL_STEERING;

    const float target_raw = state->target_speed + dt * q_speed;
    const float target_next = clampf_local(target_raw,
        0.0f, active_target_speed_ceiling_mps);
    if (target_next != target_raw)
        stage->branch_flags |= MPC_STAGE_CLIPPED_TARGET_SPEED;
    const float target_mid = 0.5f * (state->target_speed + target_next);

    const float u0 = clampf_local(state->u, 0.0f,
        parameters.maximum_command_speed_mps);
    float acceleration = MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 +
        MPC_LONGITUDINAL_SPEED_COEFF_PER_S * u0 +
        MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S * (target_mid - u0) +
        MPC_LONGITUDINAL_TARGET_RATE_COEFF * q_speed;
    const float braking_limit =
        MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2 +
        MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV * u0;
    const float unclipped_acceleration = acceleration;
    acceleration = clampf_local(acceleration, -braking_limit,
        MPC_LONGITUDINAL_ACCEL_LIMIT_MPS2);
    if (acceleration != unclipped_acceleration)
        stage->branch_flags |= MPC_STAGE_CLIPPED_ACCELERATION;

    const float u_raw = u0 + dt * acceleration;
    const float u_next = clampf_local(u_raw, 0.0f,
        parameters.maximum_command_speed_mps);
    if (u_next != u_raw)
        stage->branch_flags |= MPC_STAGE_CLIPPED_BODY_SPEED;
    const float u_mid = 0.5f * (u0 + u_next);

    const float r_next = yaw_rate_response(
        u_mid, actual_next, state->r, dt, path_curvature);
    const float r_mid = 0.5f * (state->r + r_next);
    const float denominator0 = 1.0f - path_curvature * state->e_y;
    if (fabsf(denominator0) < 0.05f) return 0;
    const float s_dot0 =
        (u0 * cosf(state->e_psi) - state->v * sinf(state->e_psi)) /
        denominator0;
    const float e_y_dot0 =
        u0 * sinf(state->e_psi) + state->v * cosf(state->e_psi);
    const float e_psi_dot0 = state->r - path_curvature * s_dot0;
    const float e_y_mid = state->e_y + 0.5f * dt * e_y_dot0;
    const float e_psi_mid = state->e_psi + 0.5f * dt * e_psi_dot0;
    const float denominator_mid = 1.0f - path_curvature * e_y_mid;
    if (fabsf(denominator_mid) < 0.05f) return 0;
    const float s_dot_mid =
        (u_mid * cosf(e_psi_mid) - state->v * sinf(e_psi_mid)) /
        denominator_mid;
    const float e_y_dot_mid =
        u_mid * sinf(e_psi_mid) + state->v * cosf(e_psi_mid);
    const float e_psi_dot_mid = r_mid - path_curvature * s_dot_mid;

    stage->next = (MpcModelState_t){
        .e_y = state->e_y + dt * e_y_dot_mid,
        .e_psi = atan2f(sinf(state->e_psi + dt * e_psi_dot_mid),
                        cosf(state->e_psi + dt * e_psi_dot_mid)),
        .u = u_next,
        /* The controller has no causal lateral-velocity input model beyond
         * the current legal /odom state. Holding it through one prediction
         * step is the least-assumptive map; an unvalidated decay fit caused
         * no improvement in live authority and is intentionally not used. */
        .v = state->v,
        .r = r_next,
        .target_speed = target_next,
        .steering_command = delta_next,
        .delayed_steering_command_1 = state->steering_command,
        .delayed_steering_command_2 = state->delayed_steering_command_1,
        .actual_steering_angle = actual_next};
    stage->delta_s_m = dt * s_dot_mid;
    stage->body_accel_mps2 = acceleration;
    stage->valid = finite_mpc_state(&stage->next) &&
        isfinite(stage->delta_s_m) && isfinite(stage->body_accel_mps2);
    return stage->valid;
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
    MpcJet_t x[MPC_MODEL_NX] = {
        jet_variable(state->e_y, 0, differentiate),
        jet_variable(state->e_psi, 1, differentiate),
        jet_variable(state->u, 2, differentiate),
        jet_variable(state->v, 3, differentiate),
        jet_variable(state->r, 4, differentiate),
        jet_variable(state->target_speed, 5, differentiate),
        jet_variable(state->steering_command, 6, differentiate),
        jet_variable(state->delayed_steering_command_1, 7, differentiate),
        jet_variable(state->delayed_steering_command_2, 8, differentiate),
        jet_variable(state->actual_steering_angle, 9, differentiate)};
    MpcJet_t w[2] = {
        jet_variable(control->steering_rate, 10, differentiate),
        jet_variable(control->target_speed_rate, 11, differentiate)};

    MpcJet_t q_delta = jet_clip(w[0],
        jet_constant(-parameters.steering_rate_radps),
        jet_constant(parameters.steering_rate_radps),
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

    MpcJet_t actual_rate_raw = jet_divide(
        jet_subtract(x[8], x[9]), jet_constant(dt));
    MpcJet_t actual_rate = jet_clip(actual_rate_raw,
        jet_constant(-parameters.steering_rate_radps),
        jet_constant(parameters.steering_rate_radps),
        MPC_STAGE_CLIPPED_ACTUAL_STEERING, stage, linearization);
    MpcJet_t actual_raw = jet_add(x[9],
        jet_multiply(jet_constant(dt), actual_rate));
    MpcJet_t actual_next = jet_clip(actual_raw,
        jet_constant(-parameters.max_steering_angle),
        jet_constant(parameters.max_steering_angle),
        MPC_STAGE_CLIPPED_ACTUAL_STEERING, stage, linearization);

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

    MpcJet_t yaw_retention;
    if (active_yaw_rate_parameters.low_speed_response_time_constant_s > 0.0f) {
        const MpcJet_t speed_decay = jet_exponential(jet_multiply(
            jet_constant(-1.0f /
                active_yaw_rate_parameters.low_speed_transition_speed_mps),
            u_mid));
        const MpcJet_t response_time_constant = jet_add(
            jet_constant(active_yaw_rate_parameters.response_time_constant_s),
            jet_multiply(jet_constant(
                active_yaw_rate_parameters.low_speed_response_time_constant_s -
                active_yaw_rate_parameters.response_time_constant_s),
                speed_decay));
        yaw_retention = jet_exponential(jet_divide(
            jet_constant(-dt), response_time_constant));
    } else {
        yaw_retention = jet_constant(expf(
            -dt / active_yaw_rate_parameters.response_time_constant_s));
    }
    const float steering_start = active_yaw_rate_parameters.steering_gain_start_rad;
    const float steering_end = active_yaw_rate_parameters.steering_gain_end_rad;
    mark_nonsmooth_difference(
        actual_next, jet_constant(steering_start), steering_start, linearization);
    mark_nonsmooth_difference(
        actual_next, jet_constant(-steering_start), steering_start, linearization);
    mark_nonsmooth_difference(
        actual_next, jet_constant(steering_end), steering_end, linearization);
    mark_nonsmooth_difference(
        actual_next, jet_constant(-steering_end), steering_end, linearization);
    const float path_gain = vehicle_model_yaw_rate_gain_for_curvature(path_curvature);
    const float steering_magnitude = fabsf(actual_next.value);
    MpcJet_t yaw_gain = jet_constant(
        path_gain - active_yaw_rate_parameters.steering_gain_per_m +
        vehicle_model_yaw_rate_gain(actual_next.value));
    if (actual_next.differentiated &&
        steering_magnitude > steering_start && steering_magnitude < steering_end) {
        const float sign = actual_next.value < 0.0f ? -1.0f : 1.0f;
        const float derivative =
            -active_yaw_rate_parameters.steering_gain_reduction_per_rad * sign;
        yaw_gain.differentiated = 1;
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            yaw_gain.derivative[i] += derivative * actual_next.derivative[i];
    }
    MpcJet_t yaw_steady = jet_multiply(
        yaw_gain,
        jet_multiply(u_mid, jet_tangent(actual_next)));
    MpcJet_t r_next = jet_add(
        jet_multiply(yaw_retention, x[4]),
        jet_multiply(jet_subtract(jet_constant(1.0f), yaw_retention),
            yaw_steady));
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
    MpcJet_t next[MPC_MODEL_NX] = {
        e_y_next,
        jet_constant(atan2f(sinf(e_psi_unwrapped.value),
                            cosf(e_psi_unwrapped.value))),
        u_next,
        x[3],
        r_next,
        target_next,
        delta_next,
        x[6],
        x[7],
        actual_next};
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
        .steering_command = next[6].value,
        .delayed_steering_command_1 = next[7].value,
        .delayed_steering_command_2 = next[8].value,
        .actual_steering_angle = next[9].value};
    stage->delta_s_m = dt * s_dot_mid.value;
    stage->body_accel_mps2 = acceleration.value;
    stage->valid = finite_mpc_state(&stage->next) &&
        isfinite(stage->delta_s_m) && isfinite(stage->body_accel_mps2);
    if (!stage->valid) return 0;

    if (linearization != NULL) {
        for (int row = 0; row < MPC_MODEL_NX; ++row) {
            for (int column = 0; column < MPC_MODEL_NX; ++column)
                linearization->A[row][column] = next[row].derivative[column];
            for (int input = 0; input < 2; ++input)
                linearization->B[row][input] = next[row].derivative[10 + input];
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
    vehicle_model_step_scalar(state, control, dt, path_curvature, &stage);
    return stage;
}

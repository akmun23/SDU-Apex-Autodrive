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
static MpcYawRateResponseSurface_t active_yaw_rate_surface = {0};
static MpcYawRateResidualModel_t active_yaw_rate_residual = {0};

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

int vehicle_model_set_yaw_rate_response_surface(
    const MpcYawRateResponseSurface_t *surface)
{
    if (surface == NULL) return 0;
    if (!surface->enabled) {
        active_yaw_rate_surface = (MpcYawRateResponseSurface_t){0};
        return 1;
    }
    if (!isfinite(surface->blend_q_start) ||
        !isfinite(surface->blend_q_end) ||
        surface->blend_q_start < 0.0f ||
        surface->blend_q_end <= surface->blend_q_start ||
        !isfinite(surface->speed_blend_margin_mps) ||
        surface->speed_blend_margin_mps < 0.0f ||
        !isfinite(surface->low_speed_blend_margin_mps) ||
        surface->low_speed_blend_margin_mps < 0.0f ||
        !isfinite(surface->low_speed_support_fadeout_mps) ||
        surface->low_speed_support_fadeout_mps < 0.0f ||
        !isfinite(surface->high_speed_support_fadein_mps) ||
        surface->high_speed_support_fadein_mps < 0.0f ||
        ((surface->low_speed_support_fadeout_mps > 0.0f) !=
         (surface->high_speed_support_fadein_mps > 0.0f)) ||
        !isfinite(surface->steering_blend_start_rad) ||
        !isfinite(surface->steering_blend_full_start_rad) ||
        !isfinite(surface->steering_blend_full_end_rad) ||
        !isfinite(surface->steering_blend_end_rad) ||
        !isfinite(surface->hold_response_time_constant_s) ||
        !isfinite(surface->hold_rate_full_radps) ||
        !isfinite(surface->hold_rate_zero_radps)) {
        return 0;
    }
    const int steering_gate_enabled =
        surface->steering_blend_start_rad != 0.0f ||
        surface->steering_blend_full_start_rad != 0.0f ||
        surface->steering_blend_full_end_rad != 0.0f ||
        surface->steering_blend_end_rad != 0.0f;
    if (steering_gate_enabled &&
        (surface->steering_blend_start_rad < 0.0f ||
         surface->steering_blend_full_start_rad <=
             surface->steering_blend_start_rad ||
         surface->steering_blend_full_end_rad <
             surface->steering_blend_full_start_rad ||
         surface->steering_blend_end_rad <=
             surface->steering_blend_full_end_rad ||
         surface->steering_blend_end_rad > active_parameters.max_steering_angle)) {
        return 0;
    }
    if (surface->hold_response_time_constant_s > 0.0f) {
        if (!finite_positive(surface->hold_rate_zero_radps) ||
            surface->hold_rate_full_radps < 0.0f ||
            surface->hold_rate_zero_radps <= surface->hold_rate_full_radps) {
            return 0;
        }
    } else if (surface->hold_response_time_constant_s < 0.0f ||
               surface->hold_rate_full_radps != 0.0f ||
               surface->hold_rate_zero_radps != 0.0f) {
        return 0;
    }
    if (surface->speed_count < 3 ||
        surface->speed_count > MPC_YAW_SURFACE_SPEED_KNOTS) return 0;
    for (int speed_index = 0; speed_index < surface->speed_count;
         ++speed_index) {
        if (!finite_positive(surface->speed_mps[speed_index]) ||
            (speed_index > 0 && surface->speed_mps[speed_index] <=
                surface->speed_mps[speed_index - 1])) {
            return 0;
        }
        for (int direction = 0; direction < MPC_YAW_SURFACE_TURN_DIRECTIONS;
             ++direction) {
            for (int knot = 0; knot < MPC_YAW_SURFACE_Q_KNOTS; ++knot) {
                if (!finite_positive(surface->q[speed_index][direction][knot]) ||
                    !finite_positive(surface->yaw_rate_abs_rps
                        [speed_index][direction][knot]) ||
                    (knot > 0 && surface->q[speed_index][direction][knot] <=
                        surface->q[speed_index][direction][knot - 1])) {
                    return 0;
                }
            }
        }
    }
    if (surface->low_speed_support_fadeout_mps > 0.0f &&
        surface->low_speed_support_fadeout_mps +
            surface->high_speed_support_fadein_mps >=
            surface->speed_mps[2] - surface->speed_mps[0]) {
        return 0;
    }
    active_yaw_rate_surface = *surface;
    return 1;
}

MpcYawRateResidualModel_t vehicle_model_default_yaw_rate_residual_model(void)
{
    return (MpcYawRateResidualModel_t){0};
}

int vehicle_model_set_yaw_rate_residual_model(
    const MpcYawRateResidualModel_t *model)
{
    if (model == NULL) return 0;
    if (!model->enabled) {
        active_yaw_rate_residual = (MpcYawRateResidualModel_t){0};
        return 1;
    }
    if (model->enabled != 1 || !isfinite(model->gain) ||
        model->gain < 0.0f || model->gain > 1.0f ||
        !finite_positive(model->correction_clip_radps2) ||
        !isfinite(model->target_speed_zero_mps) ||
        !isfinite(model->target_speed_full_mps) ||
        model->target_speed_full_mps <= model->target_speed_zero_mps ||
        !isfinite(model->speed_deficit_full_mps) ||
        !isfinite(model->speed_deficit_zero_mps) ||
        model->speed_deficit_zero_mps <= model->speed_deficit_full_mps ||
        !isfinite(model->abs_steering_zero_rad) ||
        !isfinite(model->abs_steering_full_rad) ||
        !isfinite(model->actual_speed_zero_mps) ||
        !isfinite(model->actual_speed_full_mps) ||
        !isfinite(model->actual_speed_upper_full_mps) ||
        !isfinite(model->actual_speed_upper_zero_mps) ||
        !isfinite(model->abs_steering_upper_full_rad) ||
        !isfinite(model->abs_steering_upper_zero_rad) ||
        model->abs_steering_zero_rad < 0.0f ||
        model->abs_steering_full_rad <= model->abs_steering_zero_rad) {
        return 0;
    }
    const int actual_speed_window_enabled =
        model->actual_speed_zero_mps != 0.0f ||
        model->actual_speed_full_mps != 0.0f ||
        model->actual_speed_upper_full_mps != 0.0f ||
        model->actual_speed_upper_zero_mps != 0.0f;
    if (actual_speed_window_enabled &&
        (model->actual_speed_zero_mps < 0.0f ||
         model->actual_speed_full_mps <= model->actual_speed_zero_mps ||
         model->actual_speed_upper_full_mps < model->actual_speed_full_mps ||
         model->actual_speed_upper_zero_mps <=
             model->actual_speed_upper_full_mps)) {
        return 0;
    }
    const int steering_upper_gate_enabled =
        model->abs_steering_upper_full_rad != 0.0f ||
        model->abs_steering_upper_zero_rad != 0.0f;
    if (steering_upper_gate_enabled &&
        (model->abs_steering_upper_full_rad <= model->abs_steering_full_rad ||
         model->abs_steering_upper_zero_rad <=
             model->abs_steering_upper_full_rad ||
         model->abs_steering_upper_zero_rad >
             active_parameters.max_steering_angle)) {
        return 0;
    }
    for (int i = 0; i < MPC_YAW_RESIDUAL_FEATURES; ++i) {
        if (!isfinite(model->feature_mean[i]) ||
            !finite_positive(model->feature_scale[i])) return 0;
    }
    for (int i = 0; i < MPC_YAW_RESIDUAL_COEFFICIENTS; ++i) {
        if (!isfinite(model->coefficients[i])) return 0;
    }
    active_yaw_rate_residual = *model;
    return 1;
}

static float smoothstep_scalar(float value, float start, float end)
{
    const float t = clampf_local((value - start) / (end - start), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static float yaw_residual_rate_scalar(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float curvature,
    float acceleration)
{
    if (!active_yaw_rate_residual.enabled) return 0.0f;
    const float delta = state->actual_steering_angle;
    const float q_delta = clampf_local(control->steering_rate,
        -active_parameters.steering_rate_radps,
        active_parameters.steering_rate_radps);
    const float q_speed = clampf_local(control->target_speed_rate,
        -active_parameters.maximum_target_speed_rate_reduction_mps2,
        active_parameters.maximum_target_speed_rate_increase_mps2);
    const float abs_delta = fabsf(delta);
    const float features[MPC_YAW_RESIDUAL_FEATURES] = {
        state->u, state->r, delta, q_delta, state->v, curvature,
        acceleration, q_speed, state->u * delta * abs_delta,
        delta * fabsf(q_delta), state->v * abs_delta,
        abs_delta * acceleration, delta * fabsf(q_speed)};
    float correction = active_yaw_rate_residual.coefficients[0];
    for (int i = 0; i < MPC_YAW_RESIDUAL_FEATURES; ++i) {
        correction += active_yaw_rate_residual.coefficients[i + 1] *
            ((features[i] - active_yaw_rate_residual.feature_mean[i]) /
             active_yaw_rate_residual.feature_scale[i]);
    }
    correction = clampf_local(correction,
        -active_yaw_rate_residual.correction_clip_radps2,
        active_yaw_rate_residual.correction_clip_radps2);
    const float target_gate = smoothstep_scalar(state->target_speed,
        active_yaw_rate_residual.target_speed_zero_mps,
        active_yaw_rate_residual.target_speed_full_mps);
    const float deficit = state->target_speed - state->u;
    const float tracking_gate = 1.0f - smoothstep_scalar(deficit,
        active_yaw_rate_residual.speed_deficit_full_mps,
        active_yaw_rate_residual.speed_deficit_zero_mps);
    float steering_gate = smoothstep_scalar(abs_delta,
        active_yaw_rate_residual.abs_steering_zero_rad,
        active_yaw_rate_residual.abs_steering_full_rad);
    if (active_yaw_rate_residual.abs_steering_upper_zero_rad > 0.0f) {
        steering_gate *= 1.0f - smoothstep_scalar(abs_delta,
            active_yaw_rate_residual.abs_steering_upper_full_rad,
            active_yaw_rate_residual.abs_steering_upper_zero_rad);
    }
    float actual_speed_gate = 1.0f;
    if (active_yaw_rate_residual.actual_speed_upper_zero_mps > 0.0f) {
        actual_speed_gate = smoothstep_scalar(state->u,
            active_yaw_rate_residual.actual_speed_zero_mps,
            active_yaw_rate_residual.actual_speed_full_mps) *
            (1.0f - smoothstep_scalar(state->u,
                active_yaw_rate_residual.actual_speed_upper_full_mps,
                active_yaw_rate_residual.actual_speed_upper_zero_mps));
    }
    return active_yaw_rate_residual.gain * correction * target_gate *
        tracking_gate * steering_gate * actual_speed_gate;
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

typedef struct
{
    float value;
    float slope;
} MpcPchipValue_t;

typedef struct
{
    float yaw_rate_rps;
    float d_yaw_rate_d_speed;
    float d_yaw_rate_d_steering;
} MpcYawSurfaceValue_t;

static float pchip_endpoint_slope(float h0, float h1, float d0, float d1)
{
    float slope = ((2.0f * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if (slope * d0 <= 0.0f) return 0.0f;
    if (d0 * d1 < 0.0f && fabsf(slope) > 3.0f * fabsf(d0))
        return 3.0f * d0;
    return slope;
}

static void pchip_slopes(const float *x, const float *y, int count,
                         float *slopes)
{
    float h[MPC_YAW_SURFACE_Q_KNOTS - 1];
    float secant[MPC_YAW_SURFACE_Q_KNOTS - 1];
    for (int i = 0; i < count - 1; ++i) {
        h[i] = x[i + 1] - x[i];
        secant[i] = (y[i + 1] - y[i]) / h[i];
    }
    slopes[0] = pchip_endpoint_slope(h[0], h[1], secant[0], secant[1]);
    slopes[count - 1] = pchip_endpoint_slope(
        h[count - 2], h[count - 3], secant[count - 2], secant[count - 3]);
    for (int i = 1; i < count - 1; ++i) {
        const float left = secant[i - 1];
        const float right = secant[i];
        if (left * right <= 0.0f) {
            slopes[i] = 0.0f;
            continue;
        }
        const float w_left = 2.0f * h[i] + h[i - 1];
        const float w_right = h[i] + 2.0f * h[i - 1];
        slopes[i] = (w_left + w_right) /
            (w_left / left + w_right / right);
    }
}

static MpcPchipValue_t pchip_value_and_slope(
    const float *x, const float *y, int count, float query)
{
    if (query < x[0]) return (MpcPchipValue_t){y[0], 0.0f};
    if (query > x[count - 1])
        return (MpcPchipValue_t){y[count - 1], 0.0f};

    float slopes[MPC_YAW_SURFACE_Q_KNOTS];
    pchip_slopes(x, y, count, slopes);
    int interval = 0;
    while (interval < count - 2 && query > x[interval + 1]) ++interval;
    const float width = x[interval + 1] - x[interval];
    const float t = (query - x[interval]) / width;
    const float y0 = y[interval];
    const float y1 = y[interval + 1];
    const float m0 = slopes[interval];
    const float m1 = slopes[interval + 1];
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    const float h10 = t3 - 2.0f * t2 + t;
    const float h01 = -2.0f * t3 + 3.0f * t2;
    const float h11 = t3 - t2;
    const float value = h00 * y0 + h10 * width * m0 +
        h01 * y1 + h11 * width * m1;
    const float slope = ((6.0f * t2 - 6.0f * t) * y0 / width) +
        (3.0f * t2 - 4.0f * t + 1.0f) * m0 +
        ((-6.0f * t2 + 6.0f * t) * y1 / width) +
        (3.0f * t2 - 2.0f * t) * m1;
    return (MpcPchipValue_t){value, slope};
}

static MpcYawSurfaceValue_t yaw_surface_value(
    float speed_mps, float steering_rad)
{
    const int direction = steering_rad < 0.0f ? 0 : 1;
    const float turn_sign = steering_rad < 0.0f ? -1.0f : 1.0f;
    const float demand_q = fabsf(speed_mps * tanf(steering_rad));
    const int speed_count = active_yaw_rate_surface.speed_count;
    float row_rate[MPC_YAW_SURFACE_SPEED_KNOTS];
    float row_q_slope[MPC_YAW_SURFACE_SPEED_KNOTS];
    for (int i = 0; i < speed_count; ++i) {
        const MpcPchipValue_t point = pchip_value_and_slope(
            active_yaw_rate_surface.q[i][direction],
            active_yaw_rate_surface.yaw_rate_abs_rps[i][direction],
            MPC_YAW_SURFACE_Q_KNOTS, demand_q);
        row_rate[i] = point.value;
        row_q_slope[i] = point.slope;
    }

    const float speed = clampf_local(speed_mps,
        active_yaw_rate_surface.speed_mps[0],
        active_yaw_rate_surface.speed_mps[speed_count - 1]);
    int interval = 0;
    while (interval < speed_count - 2 &&
           speed > active_yaw_rate_surface.speed_mps[interval + 1]) {
        ++interval;
    }
    const float low_speed = active_yaw_rate_surface.speed_mps[interval];
    const float high_speed = active_yaw_rate_surface.speed_mps[interval + 1];
    const float fraction = (speed - low_speed) / (high_speed - low_speed);
    const float magnitude = row_rate[interval] + fraction *
        (row_rate[interval + 1] - row_rate[interval]);
    const float q_slope = row_q_slope[interval] + fraction *
        (row_q_slope[interval + 1] - row_q_slope[interval]);
    const float speed_slope = (speed_mps <
            active_yaw_rate_surface.speed_mps[0] ||
        speed_mps > active_yaw_rate_surface.speed_mps[
            speed_count - 1]) ? 0.0f :
        (row_rate[interval + 1] - row_rate[interval]) /
            (high_speed - low_speed);
    const float tangent = tanf(steering_rad);
    const float secant_squared = 1.0f + tangent * tangent;
    return (MpcYawSurfaceValue_t){
        turn_sign * magnitude,
        turn_sign * (speed_slope + q_slope * fabsf(tangent)),
        q_slope * fmaxf(speed_mps, 0.0f) * secant_squared};
}

static float smoothstep_scalar(float value, float start, float end);

static float yaw_surface_steering_blend_weight(float steering_rad)
{
    const float start = active_yaw_rate_surface.steering_blend_start_rad;
    const float full_start =
        active_yaw_rate_surface.steering_blend_full_start_rad;
    const float full_end = active_yaw_rate_surface.steering_blend_full_end_rad;
    const float end = active_yaw_rate_surface.steering_blend_end_rad;
    if (start == 0.0f && full_start == 0.0f &&
        full_end == 0.0f && end == 0.0f) return 1.0f;
    const float magnitude = fabsf(steering_rad);
    const float enter = smoothstep_scalar(magnitude, start, full_start);
    const float leave = 1.0f - smoothstep_scalar(magnitude, full_end, end);
    return enter * leave;
}

static float yaw_surface_steering_blend_slope(float steering_rad)
{
    const float start = active_yaw_rate_surface.steering_blend_start_rad;
    const float full_start =
        active_yaw_rate_surface.steering_blend_full_start_rad;
    const float full_end = active_yaw_rate_surface.steering_blend_full_end_rad;
    const float end = active_yaw_rate_surface.steering_blend_end_rad;
    if (start == 0.0f && full_start == 0.0f &&
        full_end == 0.0f && end == 0.0f) return 0.0f;
    const float magnitude = fabsf(steering_rad);
    const float enter = smoothstep_scalar(magnitude, start, full_start);
    const float leave = 1.0f - smoothstep_scalar(magnitude, full_end, end);
    const float enter_t = clampf_local(
        (magnitude - start) / (full_start - start), 0.0f, 1.0f);
    const float leave_t = clampf_local(
        (magnitude - full_end) / (end - full_end), 0.0f, 1.0f);
    const float enter_slope = (enter_t > 0.0f && enter_t < 1.0f)
        ? 6.0f * enter_t * (1.0f - enter_t) / (full_start - start) : 0.0f;
    const float leave_slope = (leave_t > 0.0f && leave_t < 1.0f)
        ? -6.0f * leave_t * (1.0f - leave_t) / (end - full_end) : 0.0f;
    return (steering_rad < 0.0f ? -1.0f : 1.0f) *
        (enter_slope * leave + enter * leave_slope);
}

static float yaw_surface_blend_weight(float demand_q, float steering_rad)
{
    const float width = active_yaw_rate_surface.blend_q_end -
        active_yaw_rate_surface.blend_q_start;
    const float t = clampf_local(
        (demand_q - active_yaw_rate_surface.blend_q_start) / width,
        0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t) *
        yaw_surface_steering_blend_weight(steering_rad);
}

static float yaw_surface_blend_slope(float demand_q)
{
    if (demand_q <= active_yaw_rate_surface.blend_q_start ||
        demand_q >= active_yaw_rate_surface.blend_q_end) return 0.0f;
    const float width = active_yaw_rate_surface.blend_q_end -
        active_yaw_rate_surface.blend_q_start;
    const float t = (demand_q - active_yaw_rate_surface.blend_q_start) / width;
    return 6.0f * t * (1.0f - t) / width;
}

static float smoothstep01(float value)
{
    const float t = clampf_local(value, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static float yaw_surface_speed_support_weight(float speed_mps)
{
    const float low_margin =
        active_yaw_rate_surface.low_speed_support_fadeout_mps;
    const float high_margin =
        active_yaw_rate_surface.high_speed_support_fadein_mps;
    if (low_margin <= 0.0f && high_margin <= 0.0f) return 1.0f;

    const float low_knot = active_yaw_rate_surface.speed_mps[0];
    const float high_knot = active_yaw_rate_surface.speed_mps[2];
    const float low_weight = low_margin > 0.0f
        ? 1.0f - smoothstep01((speed_mps - low_knot) / low_margin)
        : 0.0f;
    const float high_weight = high_margin > 0.0f
        ? smoothstep01((speed_mps - (high_knot - high_margin)) / high_margin)
        : 0.0f;
    return low_weight + high_weight;
}

static float yaw_surface_speed_support_slope(float speed_mps)
{
    const float low_margin =
        active_yaw_rate_surface.low_speed_support_fadeout_mps;
    const float high_margin =
        active_yaw_rate_surface.high_speed_support_fadein_mps;
    float slope = 0.0f;
    if (low_margin > 0.0f) {
        const float t = (speed_mps -
            active_yaw_rate_surface.speed_mps[0]) / low_margin;
        if (t > 0.0f && t < 1.0f) slope -= 6.0f * t * (1.0f - t) / low_margin;
    }
    if (high_margin > 0.0f) {
        const float t = (speed_mps - (
            active_yaw_rate_surface.speed_mps[2] - high_margin)) / high_margin;
        if (t > 0.0f && t < 1.0f) slope += 6.0f * t * (1.0f - t) / high_margin;
    }
    return slope;
}

static float yaw_surface_speed_blend_weight(float speed_mps)
{
    const float low_margin =
        active_yaw_rate_surface.low_speed_blend_margin_mps;
    const float high_margin = active_yaw_rate_surface.speed_blend_margin_mps;
    const float low = active_yaw_rate_surface.speed_mps[0];
    const float high = active_yaw_rate_surface.speed_mps[
        active_yaw_rate_surface.speed_count - 1];
    const float enter = low_margin > 0.0f
        ? smoothstep01((speed_mps - (low - low_margin)) / low_margin)
        : 1.0f;
    const float leave = high_margin > 0.0f
        ? smoothstep01(((high + high_margin) - speed_mps) / high_margin)
        : 1.0f;
    return enter * leave * yaw_surface_speed_support_weight(speed_mps);
}

static float yaw_surface_speed_blend_slope(float speed_mps)
{
    const float low_margin =
        active_yaw_rate_surface.low_speed_blend_margin_mps;
    const float high_margin = active_yaw_rate_surface.speed_blend_margin_mps;
    const float low = active_yaw_rate_surface.speed_mps[0];
    const float high = active_yaw_rate_surface.speed_mps[
        active_yaw_rate_surface.speed_count - 1];
    const float enter = low_margin > 0.0f
        ? (speed_mps - (low - low_margin)) / low_margin : 1.0f;
    const float leave = high_margin > 0.0f
        ? ((high + high_margin) - speed_mps) / high_margin : 1.0f;
    const float enter_weight = low_margin > 0.0f
        ? smoothstep01(enter) : 1.0f;
    const float leave_weight = high_margin > 0.0f
        ? smoothstep01(leave) : 1.0f;
    const float enter_slope = (low_margin > 0.0f && enter > 0.0f && enter < 1.0f)
        ? 6.0f * enter * (1.0f - enter) / low_margin : 0.0f;
    const float leave_slope = (high_margin > 0.0f && leave > 0.0f && leave < 1.0f)
        ? -6.0f * leave * (1.0f - leave) / high_margin : 0.0f;
    const float support = yaw_surface_speed_support_weight(speed_mps);
    const float support_slope = yaw_surface_speed_support_slope(speed_mps);
    return (enter_slope * leave_weight + enter_weight * leave_slope) * support +
        enter_weight * leave_weight * support_slope;
}

static float legacy_yaw_steady_response(
    float speed_mps, float steering_rad, float path_curvature)
{
    const float path_gain = vehicle_model_yaw_rate_gain_for_curvature(path_curvature);
    const float steering_gain = vehicle_model_yaw_rate_gain(steering_rad);
    const float nominal_gain = active_yaw_rate_parameters.steering_gain_per_m;
    return speed_mps * tanf(steering_rad) *
        (steering_gain - nominal_gain + path_gain);
}

static float yaw_steady_response(
    float speed_mps, float steering_rad, float path_curvature)
{
    const float legacy = legacy_yaw_steady_response(
        speed_mps, steering_rad, path_curvature);
    if (!active_yaw_rate_surface.enabled) return legacy;

    const float demand_q = fmaxf(speed_mps, 0.0f) *
        fabsf(tanf(steering_rad));
    const float blend = yaw_surface_blend_weight(demand_q, steering_rad) *
        yaw_surface_speed_blend_weight(speed_mps);
    const float empirical = yaw_surface_value(speed_mps, steering_rad).yaw_rate_rps;
    return legacy + blend * (empirical - legacy);
}

float vehicle_model_steering_for_curvature_at_speed(
    float curvature_radpm, float speed_mps)
{
    if (!isfinite(curvature_radpm) || !isfinite(speed_mps)) return NAN;
    if (!active_yaw_rate_surface.enabled)
        return vehicle_model_steering_for_curvature(curvature_radpm);
    if (speed_mps <= 0.0f || fabsf(curvature_radpm) < 1.0e-6f) return 0.0f;

    const float desired_yaw_rate = curvature_radpm * speed_mps;
    const float direction = desired_yaw_rate < 0.0f ? -1.0f : 1.0f;
    float best_steering = 0.0f;
    float best_error = fabsf(desired_yaw_rate);
    /* The measured response has a real non-monotone steering knee, so invert
     * it by bounded global search instead of Newton iteration on a possibly
     * negative local derivative. This is a feed-forward seed; MPC still
     * optimizes the full command sequence against the same response map. */
    const int intervals = 96;
    for (int i = 1; i <= intervals; ++i) {
        const float steering = direction * SOURCE_MAX_STEERING_RAD *
            ((float)i / (float)intervals);
        const float response = yaw_steady_response(
            speed_mps, steering, curvature_radpm);
        const float error = fabsf(response - desired_yaw_rate);
        if (error < best_error) {
            best_error = error;
            best_steering = steering;
        }
    }
    return best_steering;
}

static float yaw_rate_response(
    float speed_mps, float steering_rad, float yaw_rate_radps, float time_step,
    float path_curvature, float physical_steering_rate_radps)
{
    /* Use the identified first-order source-response map directly.  The
     * newer tanh/u^2 saturation looked plausible offline but failed the live
     * authority A/B at the first high-curvature transition: it caused the
     * N30 steering solution to reverse near s=34.6 m.  This is a Unity
     * response fit, not a real-car tire or friction model. */
    float steady_yaw_rate = yaw_steady_response(
        speed_mps, steering_rad, path_curvature);
    float response_time_constant =
        active_yaw_rate_parameters.low_speed_response_time_constant_s > 0.0f
        ? active_yaw_rate_parameters.response_time_constant_s +
            (active_yaw_rate_parameters.low_speed_response_time_constant_s -
                active_yaw_rate_parameters.response_time_constant_s) *
            expf(-fmaxf(speed_mps, 0.0f) /
                active_yaw_rate_parameters.low_speed_transition_speed_mps)
        : active_yaw_rate_parameters.response_time_constant_s;
    if (active_yaw_rate_surface.enabled &&
        active_yaw_rate_surface.hold_response_time_constant_s > 0.0f) {
        const float demand_q = fmaxf(speed_mps, 0.0f) *
            fabsf(tanf(steering_rad));
        const float phase_t = clampf_local(
            (fabsf(physical_steering_rate_radps) -
             active_yaw_rate_surface.hold_rate_full_radps) /
            (active_yaw_rate_surface.hold_rate_zero_radps -
             active_yaw_rate_surface.hold_rate_full_radps), 0.0f, 1.0f);
        const float moving_weight = phase_t * phase_t * (3.0f - 2.0f * phase_t);
        const float hold_weight = 1.0f - moving_weight;
        const float support = yaw_surface_blend_weight(demand_q, steering_rad) *
            yaw_surface_speed_blend_weight(speed_mps);
        /* The equilibrium map was validated on held steering, not active
         * turn-in/unwind. Fade its contribution with the same causal phase
         * gate as the fitted hold time constant. */
        const float legacy_steady_yaw_rate = legacy_yaw_steady_response(
            speed_mps, steering_rad, path_curvature);
        steady_yaw_rate = legacy_steady_yaw_rate + hold_weight *
            (steady_yaw_rate - legacy_steady_yaw_rate);
        response_time_constant += support * hold_weight *
            (active_yaw_rate_surface.hold_response_time_constant_s -
             response_time_constant);
    }
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

static MpcJet_t jet_absolute(MpcJet_t input,
                             MpcStageLinearization_t *linearization)
{
    mark_nonsmooth_difference(input, jet_constant(0.0f), 0.0f, linearization);
    MpcJet_t result = jet_constant(fabsf(input.value));
    result.differentiated = input.differentiated;
    if (input.differentiated) {
        const float sign = input.value < 0.0f ? -1.0f :
            input.value > 0.0f ? 1.0f : 0.0f;
        for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
            result.derivative[i] = sign * input.derivative[i];
    }
    return result;
}

static MpcJet_t jet_smoothstep(
    MpcJet_t input, float start, float end,
    MpcStageResult_t *stage,
    MpcStageLinearization_t *linearization)
{
    MpcJet_t t = jet_divide(
        jet_subtract(input, jet_constant(start)), jet_constant(end - start));
    t = jet_clip(t, jet_constant(0.0f), jet_constant(1.0f), 0u,
        stage, linearization);
    MpcJet_t t2 = jet_multiply(t, t);
    return jet_multiply(t2, jet_subtract(jet_constant(3.0f),
        jet_multiply(jet_constant(2.0f), t)));
}

static MpcJet_t yaw_residual_rate_jet(
    MpcJet_t x[MPC_MODEL_NX],
    MpcJet_t q_delta,
    MpcJet_t q_speed,
    MpcJet_t curvature,
    MpcJet_t acceleration,
    MpcStageResult_t *stage,
    MpcStageLinearization_t *linearization)
{
    if (!active_yaw_rate_residual.enabled) return jet_constant(0.0f);
    const MpcYawRateResidualModel_t *model = &active_yaw_rate_residual;
    const MpcJet_t delta = x[9];
    const MpcJet_t abs_delta = jet_absolute(delta, linearization);
    const MpcJet_t abs_q_delta = jet_absolute(q_delta, linearization);
    const MpcJet_t abs_q_speed = jet_absolute(q_speed, linearization);
    MpcJet_t features[MPC_YAW_RESIDUAL_FEATURES] = {
        x[2], x[4], delta, q_delta, x[3], curvature, acceleration, q_speed,
        jet_multiply(jet_multiply(x[2], delta), abs_delta),
        jet_multiply(delta, abs_q_delta),
        jet_multiply(x[3], abs_delta),
        jet_multiply(abs_delta, acceleration),
        jet_multiply(delta, abs_q_speed)};
    MpcJet_t correction = jet_constant(model->coefficients[0]);
    for (int i = 0; i < MPC_YAW_RESIDUAL_FEATURES; ++i) {
        const MpcJet_t normalized = jet_divide(
            jet_subtract(features[i], jet_constant(model->feature_mean[i])),
            jet_constant(model->feature_scale[i]));
        correction = jet_add(correction, jet_multiply(
            jet_constant(model->coefficients[i + 1]), normalized));
    }
    correction = jet_clip(correction,
        jet_constant(-model->correction_clip_radps2),
        jet_constant(model->correction_clip_radps2),
        MPC_STAGE_CLIPPED_YAW_RESIDUAL, stage, linearization);
    const MpcJet_t target_gate = jet_smoothstep(x[5],
        model->target_speed_zero_mps, model->target_speed_full_mps,
        stage, linearization);
    const MpcJet_t speed_deficit = jet_subtract(x[5], x[2]);
    const MpcJet_t tracking_gate = jet_subtract(jet_constant(1.0f),
        jet_smoothstep(speed_deficit, model->speed_deficit_full_mps,
            model->speed_deficit_zero_mps, stage, linearization));
    MpcJet_t steering_gate = jet_smoothstep(abs_delta,
        model->abs_steering_zero_rad, model->abs_steering_full_rad,
        stage, linearization);
    if (model->abs_steering_upper_zero_rad > 0.0f) {
        const MpcJet_t steering_upper_gate = jet_subtract(jet_constant(1.0f),
            jet_smoothstep(abs_delta, model->abs_steering_upper_full_rad,
                model->abs_steering_upper_zero_rad, stage, linearization));
        steering_gate = jet_multiply(steering_gate, steering_upper_gate);
    }
    MpcJet_t actual_speed_gate = jet_constant(1.0f);
    if (model->actual_speed_upper_zero_mps > 0.0f) {
        actual_speed_gate = jet_multiply(
            jet_smoothstep(x[2], model->actual_speed_zero_mps,
                model->actual_speed_full_mps, stage, linearization),
            jet_subtract(jet_constant(1.0f),
                jet_smoothstep(x[2], model->actual_speed_upper_full_mps,
                    model->actual_speed_upper_zero_mps, stage, linearization)));
    }
    return jet_multiply(jet_constant(model->gain),
        jet_multiply(correction,
            jet_multiply(target_gate,
                jet_multiply(tracking_gate,
                    jet_multiply(steering_gate, actual_speed_gate)))));
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
     * autonomous target. Held-out command/feedback captures identify one
     * 40 Hz queue interval before the physical angle follows the target.
     * Using delta_next directly would make a reversal instantaneous. */
    const float actual_rate_raw =
        (state->delayed_steering_command_1 -
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
        u_mid, actual_next, state->r, dt, path_curvature, actual_rate) + dt *
        yaw_residual_rate_scalar(state, control, path_curvature, acceleration);
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
        jet_subtract(x[7], x[9]), jet_constant(dt));
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

    MpcJet_t yaw_response_time_constant;
    MpcJet_t yaw_retention;
    if (active_yaw_rate_parameters.low_speed_response_time_constant_s > 0.0f) {
        const MpcJet_t speed_decay = jet_exponential(jet_multiply(
            jet_constant(-1.0f /
                active_yaw_rate_parameters.low_speed_transition_speed_mps),
            u_mid));
        yaw_response_time_constant = jet_add(
            jet_constant(active_yaw_rate_parameters.response_time_constant_s),
            jet_multiply(jet_constant(
                active_yaw_rate_parameters.low_speed_response_time_constant_s -
                active_yaw_rate_parameters.response_time_constant_s),
                speed_decay));
    } else {
        yaw_response_time_constant = jet_constant(
            active_yaw_rate_parameters.response_time_constant_s);
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
    if (active_yaw_rate_surface.enabled &&
        active_yaw_rate_surface.steering_blend_end_rad > 0.0f) {
        const float boundaries[] = {
            active_yaw_rate_surface.steering_blend_start_rad,
            active_yaw_rate_surface.steering_blend_full_start_rad,
            active_yaw_rate_surface.steering_blend_full_end_rad,
            active_yaw_rate_surface.steering_blend_end_rad};
        for (size_t i = 0; i < sizeof(boundaries) / sizeof(boundaries[0]); ++i) {
            mark_nonsmooth_difference(actual_next,
                jet_constant(boundaries[i]), boundaries[i], linearization);
            mark_nonsmooth_difference(actual_next,
                jet_constant(-boundaries[i]), boundaries[i], linearization);
        }
    }
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
    MpcJet_t legacy_yaw_steady = jet_multiply(
        yaw_gain,
        jet_multiply(u_mid, jet_tangent(actual_next)));
    MpcJet_t yaw_steady = legacy_yaw_steady;
    if (active_yaw_rate_surface.enabled) {
        const float tangent = tanf(actual_next.value);
        const float speed_nonnegative = fmaxf(u_mid.value, 0.0f);
        const float demand_q = speed_nonnegative * fabsf(tangent);
        MpcJet_t demand_q_jet = jet_constant(demand_q);
        demand_q_jet.differentiated =
            u_mid.differentiated || actual_next.differentiated;
        if (demand_q_jet.differentiated) {
            const float q_speed_derivative = u_mid.value > 0.0f
                ? fabsf(tangent) : 0.0f;
            const float tangent_sign = tangent < 0.0f ? -1.0f : 1.0f;
            const float q_steering_derivative = speed_nonnegative *
                (1.0f + tangent * tangent) * tangent_sign;
            for (int i = 0; i < MPC_JET_DERIVATIVES; ++i) {
                demand_q_jet.derivative[i] = q_speed_derivative *
                    u_mid.derivative[i] + q_steering_derivative *
                    actual_next.derivative[i];
            }
        }

        const MpcYawSurfaceValue_t empirical_value = yaw_surface_value(
            u_mid.value, actual_next.value);
        MpcJet_t empirical = jet_constant(empirical_value.yaw_rate_rps);
        empirical.differentiated =
            u_mid.differentiated || actual_next.differentiated;
        if (empirical.differentiated) {
            for (int i = 0; i < MPC_JET_DERIVATIVES; ++i) {
                empirical.derivative[i] =
                    empirical_value.d_yaw_rate_d_speed * u_mid.derivative[i] +
                    empirical_value.d_yaw_rate_d_steering *
                        actual_next.derivative[i];
            }
        }

        const float q_width = active_yaw_rate_surface.blend_q_end -
            active_yaw_rate_surface.blend_q_start;
        const float q_t = clampf_local(
            (demand_q - active_yaw_rate_surface.blend_q_start) / q_width,
            0.0f, 1.0f);
        const float q_blend = q_t * q_t * (3.0f - 2.0f * q_t);
        const float steering_blend =
            yaw_surface_steering_blend_weight(actual_next.value);
        const float speed_blend = yaw_surface_speed_blend_weight(u_mid.value);
        MpcJet_t blend = jet_constant(
            q_blend * steering_blend * speed_blend);
        blend.differentiated = demand_q_jet.differentiated ||
            u_mid.differentiated || actual_next.differentiated;
        if (blend.differentiated) {
            const float q_blend_slope = yaw_surface_blend_slope(demand_q);
            const float steering_blend_slope =
                yaw_surface_steering_blend_slope(actual_next.value);
            const float speed_blend_slope =
                yaw_surface_speed_blend_slope(u_mid.value);
            for (int i = 0; i < MPC_JET_DERIVATIVES; ++i) {
                blend.derivative[i] = speed_blend * steering_blend *
                    q_blend_slope * demand_q_jet.derivative[i] +
                    speed_blend * q_blend * steering_blend_slope *
                        actual_next.derivative[i] +
                    q_blend * steering_blend * speed_blend_slope *
                        u_mid.derivative[i];
            }
        }
        MpcJet_t response_blend = blend;
        if (active_yaw_rate_surface.hold_response_time_constant_s > 0.0f) {
            const float rate_t = clampf_local(
                (fabsf(actual_rate.value) -
                 active_yaw_rate_surface.hold_rate_full_radps) /
                (active_yaw_rate_surface.hold_rate_zero_radps -
                 active_yaw_rate_surface.hold_rate_full_radps),
                0.0f, 1.0f);
            const float moving_weight = rate_t * rate_t * (3.0f - 2.0f * rate_t);
            MpcJet_t hold_weight = jet_constant(1.0f - moving_weight);
            hold_weight.differentiated = actual_rate.differentiated;
            if (hold_weight.differentiated && rate_t > 0.0f && rate_t < 1.0f) {
                const float rate_sign = actual_rate.value < 0.0f ? -1.0f : 1.0f;
                const float derivative = -6.0f * rate_t * (1.0f - rate_t) /
                    (active_yaw_rate_surface.hold_rate_zero_radps -
                     active_yaw_rate_surface.hold_rate_full_radps) * rate_sign;
                for (int i = 0; i < MPC_JET_DERIVATIVES; ++i)
                    hold_weight.derivative[i] = derivative * actual_rate.derivative[i];
            }
            response_blend = jet_multiply(blend, hold_weight);
            const MpcJet_t hold_tau_delta = jet_subtract(jet_constant(
                active_yaw_rate_surface.hold_response_time_constant_s),
                yaw_response_time_constant);
            yaw_response_time_constant = jet_add(yaw_response_time_constant,
                jet_multiply(response_blend, hold_tau_delta));
        }
        yaw_retention = jet_exponential(jet_divide(
            jet_constant(-dt), yaw_response_time_constant));
        yaw_steady = jet_add(legacy_yaw_steady, jet_multiply(
            response_blend, jet_subtract(empirical, legacy_yaw_steady)));

        for (int i = 1; i < active_yaw_rate_surface.speed_count; ++i) {
            mark_nonsmooth_difference(u_mid,
                jet_constant(active_yaw_rate_surface.speed_mps[i]),
                active_yaw_rate_surface.speed_mps[i], linearization);
        }
    }
    if (!active_yaw_rate_surface.enabled) {
        yaw_retention = jet_exponential(jet_divide(
            jet_constant(-dt), yaw_response_time_constant));
    }
    MpcJet_t r_next = jet_add(
        jet_multiply(yaw_retention, x[4]),
        jet_multiply(jet_subtract(jet_constant(1.0f), yaw_retention),
            yaw_steady));
    r_next = jet_add(r_next, jet_multiply(jet_constant(dt),
        yaw_residual_rate_jet(x, q_delta, q_speed,
            jet_constant(path_curvature), acceleration,
            stage, linearization)));
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

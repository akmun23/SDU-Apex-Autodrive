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

/* Unity's Rigidbody velocity is measured at its COM. The MPC state and
 * /odom twist are referenced to base_link at the rear axle. The source
 * vehicle geometry puts COM 0.15532 m forward of that axle (see
 * tools/racing/analyze_race_run.py). */
static const double kComForwardFromRearAxleM = 0.15532;

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
        !isfinite(surface->response_scale) ||
        surface->response_scale < 0.0f || surface->response_scale > 1.0f ||
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
    const float blend = active_yaw_rate_surface.response_scale *
        yaw_surface_blend_weight(demand_q, steering_rad) *
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
        /* The surface describes the supported speed/steering equilibrium.
         * Steering phase gates only the separately fitted held-response time
         * constant; suppressing the equilibrium correction during turn-in
         * would restore the legacy model exactly where the surface reports
         * reduced high-steering yaw authority. */
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
    if (!(state && isfinite(state->e_y) && isfinite(state->e_psi) &&
        isfinite(state->u) && isfinite(state->v) && isfinite(state->r) &&
        isfinite(state->target_speed) && isfinite(state->steering_command) &&
        isfinite(state->actual_steering_angle) &&
        isfinite(state->source_rear_wheel_omega_feedback_radps[0]) &&
        isfinite(state->source_rear_wheel_omega_feedback_radps[1]) &&
        isfinite(state->delayed_throttle_command_1) &&
        isfinite(state->delayed_throttle_command_2) &&
        isfinite(state->source_throttle_command) &&
        isfinite(state->source_last_target_speed) &&
        isfinite(state->source_actuator_speed_mps) &&
        isfinite(state->source_actuator_acceleration_mps2) &&
        isfinite(state->source_actuator_integral) &&
        isfinite(state->source_overspeed_elapsed) &&
        isfinite(state->source_downshift_stable_elapsed) &&
        isfinite(state->source_yaw_surface_blend) &&
        isfinite(state->source_yaw_surface_target_radps))) return 0;
    if (state->delayed_throttle_command_1 < 0.0f ||
        state->delayed_throttle_command_1 > 1.0f ||
        state->delayed_throttle_command_2 < 0.0f ||
        state->delayed_throttle_command_2 > 1.0f) return 0;
    if (state->source_actuator_acceleration_history_count > 5U ||
        state->source_actuator_acceleration_history_next >= 5U) return 0;
    for (int index = 0; index < 5; ++index)
        if (!isfinite(state->source_actuator_acceleration_history[index]))
            return 0;
    if (state->source_plant_initialized || state->source_imu_state_initialized) {
        for (int index = 0; index < UNITY_SOURCE_PLANT_STATE_SIZE; ++index)
            if (!isfinite(state->source_plant_state[index])) return 0;
    }
    return 1;
}

enum { MPC_JET_DERIVATIVES = MPC_MODEL_NX + 2 };

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
        1.0e-3f, 1.0e-4f, 1.0e-4f, 1.0e-4f, 1.0e-4f,
        1.0e-3f, 1.0e-3f, 1.0e-2f, 1.0e-2f, 1.0e-2f,
        1.0e-2f, 1.0e-3f, 1.0e-2f};
    const float roundoff = 2.0e-7f * fmaxf(1.0f, fabsf(boundary_scale));
    for (int i = 0; i < MPC_JET_DERIVATIVES; ++i) {
        const float reach = probe_epsilon[i] *
            fabsf(boundary_difference.derivative[i]);
        if (fabsf(boundary_difference.value) <= reach + roundoff)
            linearization->nonsmooth_column_mask |= (uint32_t)(1u << i);
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
        1.0e-3f, 1.0e-4f, 1.0e-4f, 1.0e-4f, 1.0e-4f,
        1.0e-3f, 1.0e-3f, 1.0e-2f, 1.0e-2f, 1.0e-2f,
        1.0e-2f, 1.0e-3f, 1.0e-2f};
    const float difference = lhs.value - rhs.value;
    const float roundoff = 2.0e-7f * fmaxf(1.0f, fabsf(boundary_scale));
    for (int i = 0; i < MPC_JET_DERIVATIVES; ++i) {
        const float reach = probe_epsilon[i] * fabsf(
            lhs.derivative[i] - rhs.derivative[i]);
        if (fabsf(difference) <= reach + roundoff)
            linearization->nonsmooth_column_mask |= (uint32_t)(1u << i);
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

static unsigned int mask_popcount(uint32_t mask)
{
    unsigned int count = 0;
    while (mask != 0u) {
        count += mask & 1u;
        mask >>= 1u;
    }
    return count;
}

static double source_feedforward_throttle(double target_speed_mps)
{
    /* The production speed controller's measured feed-forward map. */
    static const double speed_mps[] = {
        0.0, 0.7537, 1.5008, 2.4883, 3.7113,
        4.9226, 7.3138, 9.6633, 11.9725, 15.3610};
    static const double throttle[] = {
        0.0, 0.030, 0.060, 0.100, 0.150,
        0.200, 0.300, 0.400, 0.500, 0.650};
    const int count = (int)(sizeof(speed_mps) / sizeof(speed_mps[0]));
    if (target_speed_mps <= speed_mps[0]) return throttle[0];
    if (target_speed_mps >= speed_mps[count - 1]) return throttle[count - 1];
    for (int upper = 1; upper < count; ++upper) {
        if (target_speed_mps <= speed_mps[upper]) {
            const double ratio = (target_speed_mps - speed_mps[upper - 1]) /
                (speed_mps[upper] - speed_mps[upper - 1]);
            return throttle[upper - 1] + ratio *
                (throttle[upper] - throttle[upper - 1]);
        }
    }
    return throttle[count - 1];
}

static double source_move_toward(double value, double target, double limit)
{
    if (fabs(target - value) <= limit) return target;
    return value + copysign(limit, target - value);
}

static double source_linear_table(
    double value, const double *x, const double *y, int count)
{
    if (value <= x[0]) return y[0];
    if (value >= x[count - 1]) return y[count - 1];
    for (int upper = 1; upper < count; ++upper) {
        if (value <= x[upper]) {
            const double ratio = (value - x[upper - 1]) /
                (x[upper] - x[upper - 1]);
            return y[upper - 1] + ratio * (y[upper] - y[upper - 1]);
        }
    }
    return y[count - 1];
}

static double source_throttle_per_acceleration(double speed_mps)
{
    static const double speed[] = {
        0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};
    static const double throttle_gain[] = {
        0.055, 0.056, 0.058, 0.060, 0.062, 0.064, 0.066, 0.068, 0.070};
    return source_linear_table(speed_mps, speed, throttle_gain, 9);
}

static double source_maximum_acceleration(double speed_mps)
{
    static const double speed[] = {
        0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};
    static const double envelope[] = {
        5.5, 4.4, 4.4, 3.568, 3.175, 2.562, 2.043, 1.565, 0.956};
    const double base_throttle = source_feedforward_throttle(speed_mps);
    const double gain = source_throttle_per_acceleration(speed_mps);
    const double headroom = fmax(0.0, (1.0 - base_throttle) / gain);
    return fmin(5.5, fmin(
        source_linear_table(speed_mps, speed, envelope, 9), headroom));
}

static void source_update_actuator_measurement(
    const MpcModelState_t *state,
    double raw_speed_mps,
    double dt,
    MpcModelState_t *next_state)
{
    const double measured = fmax(0.0, raw_speed_mps);
    if (!state->source_actuator_speed_initialized) {
        next_state->source_actuator_speed_mps = (float)measured;
        next_state->source_actuator_acceleration_mps2 = 0.0f;
        next_state->source_actuator_acceleration_history_count = 0U;
        next_state->source_actuator_acceleration_history_next = 0U;
        memset(next_state->source_actuator_acceleration_history, 0,
               sizeof(next_state->source_actuator_acceleration_history));
        next_state->source_actuator_speed_initialized = 1;
        return;
    }

    const double previous_speed = state->source_actuator_speed_mps;
    const double filtered_speed = 0.75 * measured + 0.25 * previous_speed;
    double acceleration_sample = clampf_local((float)(
        (filtered_speed - previous_speed) / dt), -25.0f, 25.0f);
    const unsigned int slot =
        state->source_actuator_acceleration_history_next % 5U;
    next_state->source_actuator_acceleration_history[slot] =
        (float)acceleration_sample;
    next_state->source_actuator_acceleration_history_next = (slot + 1U) % 5U;
    if (next_state->source_actuator_acceleration_history_count < 5U)
        ++next_state->source_actuator_acceleration_history_count;

    double ordered[5];
    const unsigned int count =
        next_state->source_actuator_acceleration_history_count;
    for (unsigned int index = 0; index < count; ++index)
        ordered[index] = next_state->source_actuator_acceleration_history[index];
    for (unsigned int index = 1; index < count; ++index) {
        const double value = ordered[index];
        unsigned int position = index;
        while (position > 0U && ordered[position - 1U] > value) {
            ordered[position] = ordered[position - 1U];
            --position;
        }
        ordered[position] = value;
    }
    acceleration_sample = ordered[count / 2U];
    next_state->source_actuator_speed_mps = (float)filtered_speed;
    next_state->source_actuator_acceleration_mps2 = (float)(
        0.20 * acceleration_sample +
        0.80 * state->source_actuator_acceleration_mps2);
    next_state->source_actuator_speed_initialized = 1;
}

static void seed_source_physics(
    const MpcModelState_t *state,
    UnitySourcePlantInputC *input)
{
    memset(input, 0, sizeof(*input));
    input->state[UNITY_SOURCE_U] = state->u;
    input->state[UNITY_SOURCE_V] =
        state->v + state->r * kComForwardFromRearAxleM;
    input->state[UNITY_SOURCE_YAW_RATE] = state->r;
    input->state[UNITY_SOURCE_BODY_HEIGHT] = 0.00464804098;
    input->state[UNITY_SOURCE_PHYSICAL_STEERING] =
        state->actual_steering_angle;

    /* Seed wheel rates from rolling contact kinematics. Later control cycles
     * retain the recursively predicted wheel-spin and suspension state. */
    const double radius = 0.059;
    const double wheel_x[4] = {
        0.174679994338, 0.174679994338,
        -0.155320003629, -0.155320003629};
    const double wheel_y[4] = {
        0.115741111338, -0.115901105106,
        0.117920003832, -0.118079997599};
    const double tangent = tan(state->actual_steering_angle);
    const double numerator = 2.0 * 0.324 * tangent;
    const double steer[4] = {
        atan2(numerator, 2.0 * 0.324 - 0.236 * tangent),
        atan2(numerator, 2.0 * 0.324 + 0.236 * tangent), 0.0, 0.0};
    for (int wheel = 0; wheel < 4; ++wheel) {
        const double vx = state->u - state->r * wheel_y[wheel];
        const double vy = input->state[UNITY_SOURCE_V] +
            state->r * wheel_x[wheel];
        input->state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] =
            (cos(steer[wheel]) * vx + sin(steer[wheel]) * vy) / radius;
    }
}

int vehicle_model_initialize_source_plant_state(MpcModelState_t *state)
{
    if (!state) return 0;
    if (!state->source_plant_initialized) {
        UnitySourcePlantInputC input;
        seed_source_physics(state, &input);
        if (state->source_imu_state_initialized) {
            input.state[UNITY_SOURCE_ROLL] =
                state->source_plant_state[UNITY_SOURCE_ROLL];
            input.state[UNITY_SOURCE_ROLL_RATE] =
                state->source_plant_state[UNITY_SOURCE_ROLL_RATE];
            input.state[UNITY_SOURCE_PITCH] =
                state->source_plant_state[UNITY_SOURCE_PITCH];
            input.state[UNITY_SOURCE_PITCH_RATE] =
                state->source_plant_state[UNITY_SOURCE_PITCH_RATE];
        }
        memcpy(state->source_plant_state, input.state,
               sizeof(state->source_plant_state));
        state->source_plant_initialized = 1;
        state->source_suspension_history_initialized =
            input.suspension_history_initialized;
    }

    if (state->source_rear_wheel_omega_feedback_valid) {
        state->source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RL] =
            state->source_rear_wheel_omega_feedback_radps[0];
        state->source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RR] =
            state->source_rear_wheel_omega_feedback_radps[1];
    }
    state->source_rear_wheel_omega_feedback_valid = 0;
    for (int index = 0; index < UNITY_SOURCE_PLANT_STATE_SIZE; ++index) {
        if (!isfinite(state->source_plant_state[index])) return 0;
    }
    return 1;
}

static double source_actuator_throttle(
    const MpcModelState_t *state,
    double target_speed_mps,
    double dt,
    MpcModelState_t *next_state,
    unsigned int *branch_flags)
{
    const double raw_speed = fmax(0.0, state->u);
    const double measured_speed =
        next_state->source_actuator_speed_mps;
    const double measured_acceleration =
        next_state->source_actuator_acceleration_mps2;
    const double target_change_threshold = 0.15;
    const double downshift_band = 0.10;

    if (target_speed_mps <= 0.02) {
        next_state->source_throttle_command = 0.0f;
        next_state->source_last_target_speed = 0.0f;
        next_state->source_actuator_integral = 0.0f;
        next_state->source_overspeed_elapsed = 0.0f;
        next_state->source_downshift_stable_elapsed = 0.0f;
        next_state->source_target_initialized = 0;
        next_state->source_downshift_guard = 0;
        next_state->source_hold_approach = 0;
        next_state->source_hold_reentry_requires_speed = 0;
        next_state->source_downshift_catch = 0;
        if (branch_flags) *branch_flags |= MPC_STAGE_ACTIVE_BRAKE;
        return 0.0;
    }

    const int target_changed = !state->source_target_initialized ||
        fabs(target_speed_mps - state->source_last_target_speed) >
            target_change_threshold;
    if (target_changed) {
        const int downshift = state->source_target_initialized &&
            target_speed_mps < state->source_last_target_speed -
                target_change_threshold;
        next_state->source_actuator_integral = 0.0f;
        next_state->source_overspeed_elapsed = 0.0f;
        next_state->source_downshift_stable_elapsed = 0.0f;
        next_state->source_downshift_guard = downshift &&
            measured_speed > target_speed_mps +
                fmax(downshift_band, target_change_threshold);
        next_state->source_hold_approach = 0;
        next_state->source_hold_reentry_requires_speed = 0;
        next_state->source_downshift_catch = 0;
    }
    next_state->source_last_target_speed = (float)target_speed_mps;
    next_state->source_target_initialized = 1;

    if (next_state->source_downshift_guard) {
        if (measured_speed > target_speed_mps + downshift_band) {
            const double predicted_speed = measured_speed +
                fmin(0.0, measured_acceleration) * 0.35;
            if (measured_acceleration < 0.0 &&
                predicted_speed <= target_speed_mps + downshift_band) {
                next_state->source_downshift_guard = 0;
                next_state->source_downshift_catch = 1;
                next_state->source_hold_approach = 1;
                next_state->source_downshift_stable_elapsed = 0.0f;
                next_state->source_actuator_integral = 0.0f;
                const double desired = source_feedforward_throttle(
                    target_speed_mps);
                next_state->source_throttle_command = (float)source_move_toward(
                    state->source_throttle_command, desired, 10.0 * dt);
                return next_state->source_throttle_command;
            }
            next_state->source_downshift_stable_elapsed = 0.0f;
            next_state->source_throttle_command = 0.0f;
            if (branch_flags) *branch_flags |= MPC_STAGE_ACTIVE_BRAKE;
            return 0.0;
        } else if (measured_speed <
                   fmax(0.0, target_speed_mps - downshift_band)) {
            next_state->source_downshift_guard = 0;
            next_state->source_downshift_stable_elapsed = 0.0f;
        } else {
            next_state->source_downshift_stable_elapsed += (float)dt;
            if (next_state->source_downshift_stable_elapsed < 0.20f) {
                next_state->source_actuator_integral = 0.0f;
                next_state->source_throttle_command = 0.0f;
                if (branch_flags)
                    *branch_flags |= MPC_STAGE_ACTIVE_BRAKE;
                return 0.0;
            } else {
                next_state->source_downshift_guard = 0;
                next_state->source_hold_approach = 1;
            }
        }
    }

    if (next_state->source_downshift_catch) {
        next_state->source_actuator_integral = 0.0f;
        if (measured_speed <= target_speed_mps + downshift_band) {
            next_state->source_downshift_catch = 0;
        } else {
            const double desired = source_feedforward_throttle(target_speed_mps);
            next_state->source_throttle_command = (float)source_move_toward(
                state->source_throttle_command, desired, 10.0 * dt);
            return next_state->source_throttle_command;
        }
    }

    if (raw_speed - target_speed_mps >= 0.50) {
        next_state->source_actuator_integral = 0.0f;
        next_state->source_hold_approach = 0;
        next_state->source_hold_reentry_requires_speed = 0;
        next_state->source_overspeed_elapsed = 0.0f;
        next_state->source_throttle_command = 0.0f;
        if (branch_flags) *branch_flags |= MPC_STAGE_ACTIVE_BRAKE;
        return 0.0;
    }

    const double overspeed_limit = fmin(
        1.0, fmax(0.15, 0.10 * target_speed_mps));
    if (measured_speed - target_speed_mps > overspeed_limit) {
        next_state->source_overspeed_elapsed += (float)dt;
    } else {
        next_state->source_overspeed_elapsed = 0.0f;
    }
    if (next_state->source_overspeed_elapsed >= 0.05f ||
        measured_speed - target_speed_mps > 1.0) {
        next_state->source_actuator_integral = 0.0f;
        next_state->source_hold_approach = 0;
        next_state->source_hold_reentry_requires_speed = 0;
        next_state->source_throttle_command = 0.0f;
        if (branch_flags) *branch_flags |= MPC_STAGE_ACTIVE_BRAKE;
        return 0.0;
    }

    const double error = target_speed_mps - measured_speed;
    if (next_state->source_hold_approach && error > 0.75) {
        next_state->source_hold_approach = 0;
        next_state->source_hold_reentry_requires_speed = 1;
    }
    if (!next_state->source_hold_approach) {
        const double entry_speed = fmax(0.0, target_speed_mps - 0.15);
        if (next_state->source_hold_reentry_requires_speed)
            next_state->source_hold_reentry_requires_speed =
                measured_speed < entry_speed;
        if (!next_state->source_hold_reentry_requires_speed) {
            /* Match SpeedController._predicted_positive_acceleration(): the
             * allowed measured-acceleration signal is combined with the
             * acceleration implied by the current throttle above the
             * measured-speed feed-forward operating point. Using only the
             * measured derivative delays the production hold handoff when
             * throttle is already above steady demand. */
            double predicted_acceleration = fmax(0.0, measured_acceleration);
            const double measured_feedforward = source_feedforward_throttle(
                measured_speed);
            if (state->source_throttle_command > measured_feedforward) {
                const double throttle_acceleration =
                    (state->source_throttle_command - measured_feedforward) /
                    source_throttle_per_acceleration(measured_speed);
                predicted_acceleration = fmax(
                    predicted_acceleration, throttle_acceleration);
            }
            predicted_acceleration = fmin(
                predicted_acceleration,
                source_maximum_acceleration(measured_speed));
            const double predicted_speed = measured_speed +
                predicted_acceleration * 0.35;
            if (measured_speed >= entry_speed || predicted_speed >= entry_speed)
                next_state->source_hold_approach = 1;
        }
    }

    double desired;
    if (next_state->source_hold_approach) {
        next_state->source_actuator_integral = 0.0f;
        desired = source_feedforward_throttle(target_speed_mps);
    } else {
        const double candidate_integral = fmax(-1.0, fmin(1.0,
            state->source_actuator_integral + error * dt));
        const double unsaturated = source_feedforward_throttle(
                target_speed_mps) + 0.003 * error + 0.0001 * candidate_integral;
        desired = fmax(0.0, fmin(1.0, unsaturated));
        if ((unsaturated > 0.0 && unsaturated < 1.0) ||
            (unsaturated >= 1.0 && error < 0.0) ||
            (unsaturated <= 0.0 && error > 0.0))
            next_state->source_actuator_integral = (float)candidate_integral;
    }
    next_state->source_throttle_command = (float)source_move_toward(
        state->source_throttle_command, desired,
        10.0 * dt);
    return next_state->source_throttle_command;
}

/* The path-coordinate integration depends on the vehicle transition only
 * through its next body velocities. Keeping it separate makes derivatives
 * of e_y/e_psi inexpensive without replaying the 1 ms wheel-contact model. */
static int vehicle_model_tracking_step(
    const MpcModelState_t *state,
    float u_next,
    float v_next,
    float r_next,
    float dt,
    float path_curvature,
    float *e_y_next,
    float *e_psi_next,
    float *delta_s)
{
    if (!state || !e_y_next || !e_psi_next || !delta_s) return 0;
    const float u_mid = 0.5f * (state->u + u_next);
    const float v_mid = 0.5f * (state->v + v_next);
    const float r_mid = 0.5f * (state->r + r_next);
    const float denominator0 = 1.0f - path_curvature * state->e_y;
    if (fabsf(denominator0) < 0.05f) return 0;
    const float s_dot0 =
        (state->u * cosf(state->e_psi) - state->v * sinf(state->e_psi)) /
        denominator0;
    const float e_y_dot0 =
        state->u * sinf(state->e_psi) + state->v * cosf(state->e_psi);
    const float e_psi_dot0 = state->r - path_curvature * s_dot0;
    const float e_y_mid = state->e_y + 0.5f * dt * e_y_dot0;
    const float e_psi_mid = state->e_psi + 0.5f * dt * e_psi_dot0;
    const float denominator_mid = 1.0f - path_curvature * e_y_mid;
    if (fabsf(denominator_mid) < 0.05f) return 0;
    const float s_dot_mid =
        (u_mid * cosf(e_psi_mid) - v_mid * sinf(e_psi_mid)) /
        denominator_mid;
    const float e_y_dot_mid =
        u_mid * sinf(e_psi_mid) + v_mid * cosf(e_psi_mid);
    const float e_psi_dot_mid = r_mid - path_curvature * s_dot_mid;
    const float e_psi_raw = state->e_psi + dt * e_psi_dot_mid;
    *e_y_next = state->e_y + dt * e_y_dot_mid;
    *e_psi_next = atan2f(sinf(e_psi_raw), cosf(e_psi_raw));
    *delta_s = dt * s_dot_mid;
    return isfinite(*e_y_next) && isfinite(*e_psi_next) &&
        isfinite(*delta_s);
}

/* Nonlinear rollouts use the 1 ms source-derived four-wheel plant. The compact
 * RTI state tracks path/body/controller variables, while this carried state
 * preserves wheel spin, sprung-body motion, and suspension history across
 * every stage of the horizon. */
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

    const float target_raw = state->target_speed + dt * q_speed;
    const float target_next = clampf_local(target_raw,
        0.0f, active_target_speed_ceiling_mps);
    if (target_next != target_raw)
        stage->branch_flags |= MPC_STAGE_CLIPPED_TARGET_SPEED;

    MpcModelState_t next = *state;
    UnitySourcePlantInputC source_input;
    if (state->source_plant_initialized) {
        memset(&source_input, 0, sizeof(source_input));
        memcpy(source_input.state, state->source_plant_state,
               sizeof(source_input.state));
    } else {
        seed_source_physics(state, &source_input);
        if (state->source_imu_state_initialized) {
            source_input.state[UNITY_SOURCE_ROLL] =
                state->source_plant_state[UNITY_SOURCE_ROLL];
            source_input.state[UNITY_SOURCE_ROLL_RATE] =
                state->source_plant_state[UNITY_SOURCE_ROLL_RATE];
            source_input.state[UNITY_SOURCE_PITCH] =
                state->source_plant_state[UNITY_SOURCE_PITCH];
            source_input.state[UNITY_SOURCE_PITCH_RATE] =
                state->source_plant_state[UNITY_SOURCE_PITCH_RATE];
        }
    }
    if (state->source_rear_wheel_omega_feedback_valid) {
        source_input.state[UNITY_SOURCE_WHEEL_OMEGA_RL] =
            state->source_rear_wheel_omega_feedback_radps[0];
        source_input.state[UNITY_SOURCE_WHEEL_OMEGA_RR] =
            state->source_rear_wheel_omega_feedback_radps[1];
    }
    /* This is a measured initial condition, not a constant horizon input. */
    next.source_rear_wheel_omega_feedback_valid = 0;
    source_update_actuator_measurement(state, state->u, dt, &next);
    source_input.state[UNITY_SOURCE_U] = state->u;
    source_input.state[UNITY_SOURCE_V] =
        state->v + state->r * kComForwardFromRearAxleM;
    source_input.state[UNITY_SOURCE_YAW_RATE] = state->r;
    source_input.state[UNITY_SOURCE_PHYSICAL_STEERING] =
        state->actual_steering_angle;
    source_input.steering_target_rad = state->delayed_steering_command_1;
    const float generated_throttle = (float)source_actuator_throttle(
        state, target_next, dt, &next, &stage->branch_flags);
    source_input.throttle_normalized = state->delayed_throttle_command_1;
    next.delayed_throttle_command_1 = state->delayed_throttle_command_2;
    next.delayed_throttle_command_2 = generated_throttle;
    if (source_input.throttle_normalized == 0.0f)
        stage->branch_flags |= MPC_STAGE_PLANT_BRAKE;
    source_input.suspension_history_initialized =
        state->source_suspension_history_initialized;

    UnitySourcePlantOutputC source_output;
    if (!unity_source_plant_step_c(
            &source_input, dt, &source_output)) return 0;
    memcpy(next.source_longitudinal_slip, source_output.longitudinal_slip,
           sizeof(next.source_longitudinal_slip));
    memcpy(next.source_lateral_slip, source_output.lateral_slip,
           sizeof(next.source_lateral_slip));
    memcpy(next.source_normal_load_n, source_output.normal_load_n,
           sizeof(next.source_normal_load_n));
    next.source_yaw_surface_blend = 0.0f;
    next.source_yaw_surface_target_radps = 0.0f;
    if (active_yaw_rate_surface.enabled &&
        active_yaw_rate_surface.response_scale > 0.0f &&
        active_yaw_rate_surface.hold_response_time_constant_s > 0.0f) {
        const float speed_mid = 0.5f * (
            state->u + (float)source_output.state[UNITY_SOURCE_U]);
        const float steering_next = (float)
            source_output.state[UNITY_SOURCE_PHYSICAL_STEERING];
        const float steering_rate =
            (steering_next - state->actual_steering_angle) / dt;
        const float demand_q = fmaxf(speed_mid, 0.0f) *
            fabsf(tanf(steering_next));
        const MpcYawSurfaceValue_t empirical = yaw_surface_value(
            speed_mid, steering_next);
        const float measured_response_support =
            yaw_surface_blend_weight(demand_q, steering_next) *
            yaw_surface_speed_blend_weight(speed_mid);
        const float equilibrium_gate = 1.0f - smoothstep_scalar(
            fabsf(state->r - empirical.yaw_rate_rps), 0.15f, 0.25f);
        const float steering_rate_gate = 1.0f - smoothstep_scalar(
            fabsf(steering_rate),
            active_yaw_rate_surface.hold_rate_full_radps,
            active_yaw_rate_surface.hold_rate_zero_radps);
        const float applied_throttle = state->delayed_throttle_command_1;
        const float throttle_magnitude_gate = smoothstep_scalar(
            applied_throttle, 0.05f, 0.10f);
        const float throttle_slew_gate = 1.0f - smoothstep_scalar(
            fabsf(state->delayed_throttle_command_2 - applied_throttle),
            0.01f, 0.03f);
        const float correction_weight = active_yaw_rate_surface.response_scale *
            measured_response_support * equilibrium_gate *
            steering_rate_gate * throttle_magnitude_gate * throttle_slew_gate;

        if (correction_weight > 0.0f) {
            float response_tau =
                active_yaw_rate_parameters.response_time_constant_s;
            if (active_yaw_rate_parameters.low_speed_response_time_constant_s >
                0.0f) {
                response_tau += (
                    active_yaw_rate_parameters.low_speed_response_time_constant_s -
                    response_tau) * expf(-fmaxf(speed_mid, 0.0f) /
                    active_yaw_rate_parameters.low_speed_transition_speed_mps);
            }
            const float hold_phase = steering_rate_gate;
            response_tau += measured_response_support * hold_phase *
                (active_yaw_rate_surface.hold_response_time_constant_s -
                 response_tau);
            const float retention = expf(-dt / response_tau);
            const float empirical_next = retention * state->r +
                (1.0f - retention) * empirical.yaw_rate_rps;
            const float physical_yaw_rate = (float)
                source_output.state[UNITY_SOURCE_YAW_RATE];
            const float corrected_yaw_rate = physical_yaw_rate +
                correction_weight * (empirical_next - physical_yaw_rate);
            source_output.state[UNITY_SOURCE_YAW_RATE] = corrected_yaw_rate;
            source_output.yaw_acceleration_radps2 =
                (corrected_yaw_rate - state->r) / dt;
            next.source_yaw_surface_blend = correction_weight;
            next.source_yaw_surface_target_radps = empirical.yaw_rate_rps;
        }
    }
    memcpy(next.source_plant_state, source_output.state,
           sizeof(next.source_plant_state));
    next.source_plant_initialized = 1;
    next.source_suspension_history_initialized =
        source_output.suspension_history_initialized;

    const float u_next = (float)source_output.state[UNITY_SOURCE_U];
    const float r_next = (float)source_output.state[UNITY_SOURCE_YAW_RATE];
    const float v_next = (float)(
        source_output.state[UNITY_SOURCE_V] -
        r_next * kComForwardFromRearAxleM);
    const float actual_next = (float)
        source_output.state[UNITY_SOURCE_PHYSICAL_STEERING];
    const float acceleration = (float)
        source_output.longitudinal_acceleration_mps2;
    if (!isfinite(u_next) || !isfinite(v_next) || !isfinite(r_next) ||
        !isfinite(actual_next) ||
        fabsf(u_next) > parameters.maximum_command_speed_mps) return 0;

    if (!vehicle_model_tracking_step(
            state, u_next, v_next, r_next, dt, path_curvature,
            &next.e_y, &next.e_psi, &stage->delta_s_m)) return 0;
    next.u = u_next;
    next.v = v_next;
    next.r = r_next;
    next.target_speed = target_next;
    next.steering_command = delta_next;
    next.delayed_steering_command_1 = state->steering_command;
    next.delayed_steering_command_2 = state->delayed_steering_command_1;
    next.actual_steering_angle = actual_next;
    stage->next = next;
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
        jet_variable(state->actual_steering_angle, 9, differentiate),
        jet_variable(state->delayed_throttle_command_1, 10, differentiate),
        jet_variable(state->delayed_throttle_command_2, 11, differentiate)};
    MpcJet_t w[2] = {
        jet_variable(control->steering_rate, MPC_MODEL_NX, differentiate),
        jet_variable(control->target_speed_rate,
                     MPC_MODEL_NX + 1, differentiate)};

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
        MpcJet_t hold_response_blend = blend;
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
            hold_response_blend = jet_multiply(blend, hold_weight);
            const MpcJet_t hold_tau_delta = jet_subtract(jet_constant(
                active_yaw_rate_surface.hold_response_time_constant_s),
                yaw_response_time_constant);
            yaw_response_time_constant = jet_add(yaw_response_time_constant,
                jet_multiply(hold_response_blend, hold_tau_delta));
        }
        yaw_retention = jet_exponential(jet_divide(
            jet_constant(-dt), yaw_response_time_constant));
        const MpcJet_t yaw_surface_response_blend = jet_multiply(
            blend,
            jet_constant(active_yaw_rate_surface.response_scale));
        yaw_steady = jet_add(legacy_yaw_steady, jet_multiply(
            yaw_surface_response_blend,
            jet_subtract(empirical, legacy_yaw_steady)));

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
                linearization->B[row][input] =
                    next[row].derivative[MPC_MODEL_NX + input];
        }
        linearization->nominal_branch_flags = stage->branch_flags;
        linearization->nonsmooth_column_count =
            (int)mask_popcount(linearization->nonsmooth_column_mask);
        linearization->valid = 1;
    }
    return 1;
}

static float mpc_state_component(const MpcModelState_t *state, int index)
{
    switch (index) {
    case 0: return state->e_y;
    case 1: return state->e_psi;
    case 2: return state->u;
    case 3: return state->v;
    case 4: return state->r;
    case 5: return state->target_speed;
    case 6: return state->steering_command;
    case 7: return state->delayed_steering_command_1;
    case 8: return state->delayed_steering_command_2;
    case 9: return state->actual_steering_angle;
    case 10: return state->delayed_throttle_command_1;
    case 11: return state->delayed_throttle_command_2;
    case 12: return (float)state->source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL];
    case 13: return (float)state->source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FR];
    case 14: return (float)state->source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RL];
    case 15: return (float)state->source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RR];
    default: return NAN;
    }
}

static void mpc_state_perturb(MpcModelState_t *state, int index, float amount)
{
    switch (index) {
    case 0: state->e_y += amount; break;
    case 1: state->e_psi += amount; break;
    case 2: state->u += amount; break;
    case 3: state->v += amount; break;
    case 4: state->r += amount; break;
    case 5: state->target_speed += amount; break;
    case 6: state->steering_command += amount; break;
    case 7: state->delayed_steering_command_1 += amount; break;
    case 8: state->delayed_steering_command_2 += amount; break;
    case 9: state->actual_steering_angle += amount; break;
    case 10: state->delayed_throttle_command_1 += amount; break;
    case 11: state->delayed_throttle_command_2 += amount; break;
    case 12: state->source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL] += amount; break;
    case 13: state->source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FR] += amount; break;
    case 14: state->source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RL] += amount; break;
    case 15: state->source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RR] += amount; break;
    default: break;
    }
}

static float mpc_stage_component(const MpcStageResult_t *stage, int index)
{
    return mpc_state_component(&stage->next, index);
}

static float mpc_component_delta(int row, float lhs, float rhs)
{
    return row == 1 ? remainderf(lhs - rhs, 6.2831853071795864769f)
                    : lhs - rhs;
}

static int vehicle_model_linearize_with_known_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    const MpcStageResult_t *known_step,
    MpcStageResult_t *stage_out,
    MpcStageLinearization_t *linearization)
{
    if (!state || !control || !stage_out || !linearization) return 0;
    memset(linearization, 0, sizeof(*linearization));
    linearization->failure_index = -1;
    MpcModelState_t prepared_state = *state;
    if (!vehicle_model_initialize_source_plant_state(&prepared_state)) {
        linearization->failure_reason = 1;
        return 0;
    }
    state = &prepared_state;
    MpcStageResult_t known_storage = {0};
    if (known_step) {
        if (!known_step->valid ||
            !finite_mpc_state(&known_step->next) ||
            !isfinite(known_step->delta_s_m)) {
            linearization->failure_reason = 2;
            return 0;
        }
        known_storage = *known_step;
    } else if (!vehicle_model_step_scalar(
                   state, control, dt, path_curvature, &known_storage)) {
        linearization->failure_reason = 2;
        return 0;
    }
    MpcStageResult_t *stage = &known_storage;
    *stage_out = known_storage;
    linearization->nominal_branch_flags = stage->branch_flags;

    static const float state_epsilon[MPC_MODEL_NX] = {
        1.0e-4f, 1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-3f,
        1.0e-3f, 1.0e-4f, 1.0e-4f, 1.0e-4f, 1.0e-4f,
        1.0e-3f, 1.0e-3f, 1.0e-2f, 1.0e-2f, 1.0e-2f, 1.0e-2f};
    static const float input_epsilon[2] = {1.0e-3f, 1.0e-2f};

    /* These columns do not affect the Unity-source plant transition:
     * - e_y/e_psi enter only the cheap Frenet integration below;
     * - steering_command/steering_rate update the delayed command queue;
     * - delayed_steering_command_2 is not consumed by this stage;
     * - delayed_throttle_command_2 only shifts into queue slot 1.
     * Fill these exact dependencies directly rather than finite-differencing
     * the costly 1 ms tire/contact equations for every bookkeeping state. */
    const VehicleParameters_t parameters = vehicle_model_get_parameters();
    const float steering_rate = clampf_local(
        control->steering_rate, -parameters.steering_rate_radps,
        parameters.steering_rate_radps);
    const float steering_raw = state->steering_command + dt * steering_rate;
    const int steering_unsaturated =
        steering_raw > -parameters.max_steering_angle &&
        steering_raw < parameters.max_steering_angle;
    linearization->A[6][6] = steering_unsaturated ? 1.0f : 0.0f;
    linearization->A[7][6] = 1.0f;
    linearization->B[6][0] =
        steering_unsaturated && steering_rate == control->steering_rate
        ? dt : 0.0f;
    linearization->A[10][11] = 1.0f;
    if (!steering_unsaturated) {
        linearization->nonsmooth_column_mask |=
            (uint32_t)(1u << 6);
        ++linearization->nonsmooth_column_count;
    }
    if (steering_rate != control->steering_rate) {
        linearization->nonsmooth_column_mask |=
            (uint32_t)(1u << (MPC_MODEL_NX + 0));
        ++linearization->nonsmooth_column_count;
    }

    /* Path-coordinate columns need only the already-computed body transition,
     * so use symmetric differences of the exact same midpoint equations. */
    for (int column = 0; column < 2; ++column) {
        const float epsilon = state_epsilon[column];
        MpcModelState_t state_plus = *state;
        MpcModelState_t state_minus = *state;
        mpc_state_perturb(&state_plus, column, epsilon);
        mpc_state_perturb(&state_minus, column, -epsilon);
        float plus_ey = 0.0f, plus_epsi = 0.0f, plus_ds = 0.0f;
        float minus_ey = 0.0f, minus_epsi = 0.0f, minus_ds = 0.0f;
        const int plus_valid = vehicle_model_tracking_step(
            &state_plus, stage->next.u, stage->next.v, stage->next.r,
            dt, path_curvature, &plus_ey, &plus_epsi, &plus_ds);
        const int minus_valid = vehicle_model_tracking_step(
            &state_minus, stage->next.u, stage->next.v, stage->next.r,
            dt, path_curvature, &minus_ey, &minus_epsi, &minus_ds);
        if (plus_valid && minus_valid) {
            linearization->A[0][column] =
                (plus_ey - minus_ey) / (2.0f * epsilon);
            linearization->A[1][column] = mpc_component_delta(
                1, plus_epsi, minus_epsi) / (2.0f * epsilon);
        } else if (plus_valid) {
            linearization->A[0][column] =
                (plus_ey - stage->next.e_y) / epsilon;
            linearization->A[1][column] = mpc_component_delta(
                1, plus_epsi, stage->next.e_psi) / epsilon;
            linearization->nonsmooth_column_mask |=
                (uint32_t)(1u << column);
            ++linearization->nonsmooth_column_count;
        } else if (minus_valid) {
            linearization->A[0][column] =
                (stage->next.e_y - minus_ey) / epsilon;
            linearization->A[1][column] = mpc_component_delta(
                1, stage->next.e_psi, minus_epsi) / epsilon;
            linearization->nonsmooth_column_mask |=
                (uint32_t)(1u << column);
            ++linearization->nonsmooth_column_count;
        } else {
            linearization->failure_reason = 3;
            linearization->failure_index = column;
            return 0;
        }
    }

    for (int column = 0; column < MPC_MODEL_NX + 2; ++column) {
        const int is_state = column < MPC_MODEL_NX;
        const int index = is_state ? column : column - MPC_MODEL_NX;
        const int analytic_column = is_state
            ? (index == 0 || index == 1 || index == 6 || index == 8 || index == 11)
            : index == 0;
        if (analytic_column) continue;
        const float epsilon = is_state
            ? state_epsilon[index] : input_epsilon[index];
        MpcModelState_t state_plus = *state;
        MpcModelControl_t control_plus = *control;
        if (is_state) {
            mpc_state_perturb(&state_plus, index, epsilon);
        } else if (index == 0) {
            control_plus.steering_rate += epsilon;
        } else {
            control_plus.target_speed_rate += epsilon;
        }
        MpcStageResult_t plus = {0};
        MpcStageResult_t minus = {0};
        (void)vehicle_model_step_scalar(
            &state_plus, &control_plus, dt, path_curvature, &plus);
        const int plus_same_branch = plus.valid &&
            plus.branch_flags == stage->branch_flags;

        /* Use a forward difference in smooth regions; pay for the opposite
         * side only when perturbation crosses a plant/model branch. */
        int nonsmooth = !plus_same_branch;
        if (!plus_same_branch) {
            MpcModelState_t state_minus = *state;
            MpcModelControl_t control_minus = *control;
            if (is_state) {
                mpc_state_perturb(&state_minus, index, -epsilon);
            } else if (index == 0) {
                control_minus.steering_rate -= epsilon;
            } else {
                control_minus.target_speed_rate -= epsilon;
            }
            (void)vehicle_model_step_scalar(
                &state_minus, &control_minus, dt, path_curvature, &minus);
        }
        const int minus_same_branch = minus.valid &&
            minus.branch_flags == stage->branch_flags;
        for (int row = 0; row < MPC_MODEL_NX; ++row) {
            const float base_value = mpc_stage_component(stage, row);
            const float plus_value = mpc_stage_component(&plus, row);
            const float minus_value = mpc_stage_component(&minus, row);
            float derivative;
            if (plus_same_branch) {
                derivative = mpc_component_delta(
                    row, plus_value, base_value) / epsilon;
            } else if (minus_same_branch) {
                derivative = mpc_component_delta(
                    row, base_value, minus_value) / epsilon;
            } else if (plus.valid && minus.valid) {
                const unsigned int plus_distance = mask_popcount(
                    (uint16_t)(plus.branch_flags ^ stage->branch_flags));
                const unsigned int minus_distance = mask_popcount(
                    (uint16_t)(minus.branch_flags ^ stage->branch_flags));
                derivative = plus_distance <= minus_distance
                    ? mpc_component_delta(row, plus_value, base_value) / epsilon
                    : mpc_component_delta(row, base_value, minus_value) / epsilon;
            } else if (plus.valid) {
                derivative = mpc_component_delta(
                    row, plus_value, base_value) / epsilon;
            } else if (minus.valid) {
                derivative = mpc_component_delta(
                    row, base_value, minus_value) / epsilon;
            } else {
                linearization->failure_reason = 4;
                linearization->failure_index = column;
                return 0;
            }
            if (!isfinite(derivative)) {
                linearization->failure_reason = 5;
                linearization->failure_index = column;
                return 0;
            }
            if (is_state) linearization->A[row][index] = derivative;
            else linearization->B[row][index] = derivative;
        }
        if (nonsmooth) {
            linearization->nonsmooth_column_mask |=
                (uint32_t)(1u << column);
            ++linearization->nonsmooth_column_count;
        }
    }
    linearization->valid = 1;
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
    return vehicle_model_linearize_with_known_step(
        state, control, dt, path_curvature, NULL, stage, linearization);
}

int mpc_vehicle_model_linearize_from_known_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    const MpcStageResult_t *known_step,
    MpcStageResult_t *stage,
    MpcStageLinearization_t *linearization)
{
    return vehicle_model_linearize_with_known_step(
        state, control, dt, path_curvature, known_step, stage,
        linearization);
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

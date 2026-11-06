// Original racing objective helper weights; not a QP builder.
#include "mpc_rti_internal.hpp"
#include <cmath>

float overspeed_weight_for_reference(
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiReference_t *reference)
{
    const float curvature_scale = 1.0f +
        configuration->overspeed_curvature_gain_m *
        fabsf(reference->path_curvature);
    return configuration->weight_u_overspeed * curvature_scale;
}

float underspeed_weight_for_reference(
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiReference_t *reference)
{
    const float curvature_scale = 1.0f +
        configuration->underspeed_curvature_gain_m *
        fabsf(reference->path_curvature);
    return configuration->weight_u / curvature_scale;
}

float longitudinal_slip_weight_for_reference(
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiReference_t *reference)
{
    const float curvature_scale = 1.0f +
        configuration->longitudinal_slip_curvature_gain_m *
        fabsf(reference->path_curvature);
    return configuration->weight_longitudinal_slip * curvature_scale;
}

/* Preserve the objective used by the successful v1 witness and geometry-only
 * controller: damp correction increments, not the steering change required by
 * the loaded path. This reference is physical rack/feedforward steering, not
 * an independently resampled future command history. Keep QP and exact merit
 * on the same reference schedule. No track-specific behavior belongs here. */
float steering_reference_increment(
    const MpcRtiReference_t *current, const MpcRtiReference_t *next)
{
    return next->steering_command-current->steering_command;
}


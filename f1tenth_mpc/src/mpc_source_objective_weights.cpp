// Racing objective helper weights; not a QP builder.
#include "mpc_cycle_internal.hpp"
#include <cmath>

float overspeed_weight_for_reference(
    const MpcSolverConfiguration_t *configuration,
    const MpcCycleReference_t *reference)
{
    const float curvature_scale = 1.0f +
        configuration->overspeed_curvature_gain_m *
        fabsf(reference->path_curvature);
    return configuration->weight_u_overspeed * curvature_scale;
}

float lateral_weight_for_reference(
    const MpcSolverConfiguration_t *configuration,
    const MpcCycleReference_t *reference)
{
    return configuration->weight_e_y * (1.0f +
        configuration->lateral_curvature_gain_m * fabsf(reference->path_curvature));
}

float heading_weight_for_reference(
    const MpcSolverConfiguration_t *configuration,
    const MpcCycleReference_t *reference)
{
    return configuration->weight_e_psi * (1.0f +
        configuration->heading_curvature_gain_m * fabsf(reference->path_curvature));
}

float underspeed_weight_for_reference(
    const MpcSolverConfiguration_t *configuration,
    const MpcCycleReference_t *reference)
{
    const float curvature_scale = 1.0f +
        configuration->underspeed_curvature_gain_m *
        fabsf(reference->path_curvature);
    return configuration->weight_u / curvature_scale;
}

/* Penalize correction increments relative to the loaded path's rack/feedforward
 * steering, rather than the steering change the path requires. QP and exact
 * merit use the same reference schedule. */
float steering_reference_increment(
    const MpcCycleReference_t *current, const MpcCycleReference_t *next)
{
    return next->steering_command-current->steering_command;
}


#ifndef MPC_TRACKING_TOLERANCE_H
#define MPC_TRACKING_TOLERANCE_H
/*
 * A C1 convex dead-zone squared error. The MPC is permitted to carry small
 * harmless lateral/heading deviations without paying a strong correction cost.
 * The source-plant and physical wall constraints are UNCHANGED.
 *
 * For e = state - reference, the exact nonlinear merit is
 * w * max(0, |e|-tolerance)^2. The local QP active-set target is
 * reference + copysign(tolerance,e) when |e| exceeds the tolerance;
 * otherwise its quadratic weight is exactly zero. This yields the exact
 * local derivative/Hessian of the merit away from its switching surface.
 */
#include <math.h>

static inline float mpc_tracking_excess(float error, float tolerance)
{
    const float magnitude = fmaxf(0.0f, fabsf(error)-tolerance);
    return copysignf(magnitude, error);
}

static inline int mpc_tracking_local_target(float nominal,
    float reference, float tolerance, float *target)
{
    if (tolerance <= 0.0f) {
        *target = reference;
        return 1;
    }
    const float error = nominal-reference;
    if (fabsf(error) <= tolerance) {
        *target = reference;
        return 0;
    }
    *target = reference+copysignf(tolerance,error);
    return 1;
}
#endif

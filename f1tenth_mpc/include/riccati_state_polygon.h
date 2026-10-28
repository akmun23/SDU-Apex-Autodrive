#ifndef RICCATI_STATE_POLYGON_H
#define RICCATI_STATE_POLYGON_H

#include "mpc_types.h"
#include <math.h>
#include <float.h>

/* Exact diagonal-metric projection onto the intersection of a 2D box and
 * affine half-planes. Most iterates are interior or project onto one face;
 * only a multi-face corner needs the bounded vertex enumeration. No heap,
 * iteration cap, approximate alternating projections or hidden fallback. */
static inline int riccati_project_state_polygon(
    const RiccatiStatePolygon_t *polygon, const float input[2],
    const float lower[2], const float upper[2], const float metric[2],
    float output[2])
{
    if (!polygon || polygon->count < 0 ||
        polygon->count > RICCATI_STATE_POLYGON_MAX_PLANES) return 0;
    float a[RICCATI_STATE_POLYGON_MAX_PLANES + 4][2];
    float b[RICCATI_STATE_POLYGON_MAX_PLANES + 4];
    int count = polygon->count;
    for (int axis = 0; axis < 2; ++axis) {
        if (!isfinite(input[axis]) || !isfinite(lower[axis]) ||
            !isfinite(upper[axis]) || lower[axis] > upper[axis] ||
            !isfinite(metric[axis]) || metric[axis] <= 0.f) return 0;
    }
    for (int i = 0; i < count; ++i) {
        a[i][0] = polygon->normal[i][0];
        a[i][1] = polygon->normal[i][1];
        b[i] = polygon->upper[i];
        if (!isfinite(a[i][0]) || !isfinite(a[i][1]) || !isfinite(b[i]) ||
            !(a[i][0]*a[i][0] + a[i][1]*a[i][1] > 0.f)) return 0;
    }
    for (int axis = 0; axis < 2; ++axis) {
        if (lower[axis] > -BIG_BOUND) {
            a[count][0] = a[count][1] = 0.f;
            a[count][axis] = -1.f;
            b[count++] = -lower[axis];
        }
        if (upper[axis] < BIG_BOUND) {
            a[count][0] = a[count][1] = 0.f;
            a[count][axis] = 1.f;
            b[count++] = upper[axis];
        }
    }
    int inside = 1;
    for (int i = 0; i < count; ++i)
        if (a[i][0]*input[0] + a[i][1]*input[1] > b[i]) inside = 0;
    if (inside) {
        output[0] = input[0]; output[1] = input[1];
        return 1;
    }
    const double inverse0 = 1. / metric[0], inverse1 = 1. / metric[1];
    for (int i = 0; i < count; ++i) {
        const double violation = a[i][0]*(double)input[0] +
            a[i][1]*(double)input[1] - b[i];
        if (violation <= 0.) continue;
        const double denominator = a[i][0]*(double)a[i][0]*inverse0 +
            a[i][1]*(double)a[i][1]*inverse1;
        const double lambda = violation / denominator;
        const double x = input[0] - lambda*inverse0*a[i][0];
        const double y = input[1] - lambda*inverse1*a[i][1];
        int feasible = 1;
        for (int j = 0; j < count; ++j)
            if (a[j][0]*x + a[j][1]*y > b[j] + 1.e-7) { feasible = 0; break; }
        /* A projection onto one violated half-space is a lower bound on
         * distance to the intersection. If feasible, it is the exact answer. */
        if (feasible) { output[0] = (float)x; output[1] = (float)y; return 1; }
    }
    double best = DBL_MAX;
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            const double determinant = a[i][0]*(double)a[j][1] -
                a[i][1]*(double)a[j][0];
            if (fabs(determinant) <= 1.e-12) continue;
            const double x = (b[i]*(double)a[j][1] - a[i][1]*(double)b[j]) / determinant;
            const double y = (a[i][0]*(double)b[j] - b[i]*(double)a[j][0]) / determinant;
            int feasible = 1;
            for (int k = 0; k < count; ++k)
                if (a[k][0]*x + a[k][1]*y > b[k] + 1.e-7) { feasible = 0; break; }
            if (!feasible) continue;
            const double dx = x-input[0], dy = y-input[1];
            const double cost = metric[0]*dx*dx + metric[1]*dy*dy;
            if (cost < best) {
                best = cost;
                output[0] = (float)x; output[1] = (float)y;
            }
        }
    }
    return best < DBL_MAX;
}
#endif

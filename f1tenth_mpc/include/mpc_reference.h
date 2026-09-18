#ifndef MPC_REFERENCE_H
#define MPC_REFERENCE_H

#include "mpc_types.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    double s;
    double x;
    double y;
    double heading; /* Unwrapped in-place by mpc_trajectory_prepare(). */
    double curvature;
    double speed;
    double left_bound;
    double right_bound;
} MpcTrajectorySample_t;

typedef struct
{
    double s;
    double x;
    double y;
    double heading;
    double lateral_error;
    double heading_error;
    double distance;
    size_t segment;
} MpcPathProjection_t;

/* Removes a repeated closing row, validates the path, unwraps heading, and
 * computes the periodic length from the final point back to the first. */
int mpc_trajectory_prepare(
    MpcTrajectorySample_t *points,
    size_t *point_count,
    double *lap_length);

/* Periodic piecewise-linear interpolation in arc length. */
int mpc_trajectory_sample(
    const MpcTrajectorySample_t *points,
    size_t point_count,
    double lap_length,
    double s,
    MpcTrajectorySample_t *sample);

/* Continuous projection onto track segments. Pass SIZE_MAX for full-track
 * initialization; otherwise search locally first, then direction-gated full
 * track fallback. */
int mpc_trajectory_project(
    const MpcTrajectorySample_t *points,
    size_t point_count,
    double lap_length,
    double x,
    double y,
    double heading,
    size_t previous_segment,
    size_t local_search_radius,
    MpcPathProjection_t *projection);

/* Cold-start reference seed. The output has horizon+1 state references and
 * progress values; each next sample advances by v_ref(s_k)*dt, never a fixed
 * waypoint count or minimum distance. */
int mpc_reference_build_speed_seed(
    const MpcTrajectorySample_t *points,
    size_t point_count,
    double lap_length,
    double initial_s,
    double speed_ceiling,
    double dt,
    int horizon,
    TrajectoryReferencePoint_t *references,
    double *progress);

#ifdef __cplusplus
}
#endif

#endif

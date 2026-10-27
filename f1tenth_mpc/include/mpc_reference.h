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
    double acceleration;
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

/* Differential of the loaded reference interpolation, in its own s units. */
typedef struct
{
    MpcTrajectorySample_t sample;
    double dx_ds;
    double dy_ds;
    double dheading_ds;
} MpcPathFrame_t;

/* Advance the physical body displacement through the actual loaded path
 * frame. Optional Jacobian columns are e_y, e_psi, body dx, body dy, body yaw.
 * No tire/actuator calculation is repeated here. */
int mpc_trajectory_advance_tracking(
    const MpcTrajectorySample_t *points, size_t count, double lap_length,
    double progress, float e_y, float e_psi,
    float body_dx, float body_dy, float body_yaw, double progress_seed,
    double *next_progress, float *next_e_y, float *next_e_psi,
    float jacobian[2][5]);

/* Full differential: rows next e_y/e_psi/progress, columns the five above
 * followed by initial progress. Progress remains in local nominal units. */
int mpc_trajectory_advance_tracking_with_progress(
    const MpcTrajectorySample_t *points, size_t count, double lap_length,
    double progress, float e_y, float e_psi,
    float body_dx, float body_dy, float body_yaw, double progress_seed,
    double *next_progress, float *next_e_y, float *next_e_psi,
    float jacobian[3][6]);

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

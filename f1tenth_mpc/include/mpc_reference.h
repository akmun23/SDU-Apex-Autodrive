#ifndef MPC_REFERENCE_H
#define MPC_REFERENCE_H

#include "mpc_types.h"

#include <stddef.h>

#ifdef __cplusplus
#define MPC_REFERENCE_ZERO {}
#else
#define MPC_REFERENCE_ZERO
#endif

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
    /* Optional, precomputed source witness. These are reference data, never
     * extra optimizer states or privileged runtime sensor inputs. */
    int source_witness_valid MPC_REFERENCE_ZERO;
    double source_heading_error MPC_REFERENCE_ZERO;
    double source_v_rear MPC_REFERENCE_ZERO;
    double source_yaw_rate MPC_REFERENCE_ZERO;
    double source_physical_steering MPC_REFERENCE_ZERO;
    double source_steering_target MPC_REFERENCE_ZERO;
    double source_throttle MPC_REFERENCE_ZERO;
    /* Precomputed spatially smoothed inboard objective target. Zero by
     * default. Reference geometry and source control remain unmodified. */
    double robust_contour_lateral_offset_m MPC_REFERENCE_ZERO;
    /* Tangent and curvature corrections derived from the same smooth
     * lateral contour, so strong heading/yaw objectives do not oppose
     * the shifted lateral target. */
    double robust_contour_heading_offset_rad MPC_REFERENCE_ZERO;
    double robust_contour_curvature_delta_radpm MPC_REFERENCE_ZERO;
} MpcTrajectorySample_t;

#undef MPC_REFERENCE_ZERO

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

/* Exact one-sided slopes of the ORIGINAL periodic, piecewise-linear
 * trajectory interpolation. Source-optimizer spline control points are
 * already sampled before they reach the MPC; do not invent a different
 * cubic interpolation or smooth over a waypoint discontinuity.
 *
 * Each slope is per metre of unwrapped progress. At an exact trajectory
 * knot the two adjacent segments are returned independently. Held source
 * command targets have zero within-segment derivative, but can jump at
 * knots (at_knot=1): never treat this output as a smooth derivative of a
 * command packet across that discontinuity. */
typedef struct {
    MpcTrajectorySample_t forward;
    MpcTrajectorySample_t backward;
    size_t forward_segment;
    size_t backward_segment;
    int at_knot;
} MpcTrajectoryProgressDifferential_t;

int mpc_trajectory_sample_progress_differential(
    const MpcTrajectorySample_t *points,size_t point_count,
    double lap_length,double s,MpcTrajectoryProgressDifferential_t *out);

/* Chain path tangents through the original implicit tracking projection.
 * Input order matches its Jacobian: ey,epsi,dx,dy,dyaw,progress.
 * Output order: ey,epsi,progress. Reference knots use the segment reached
 * for positive path parameter, selected by first then second tangent. */
int mpc_trajectory_tracking_path_curvature(
    const MpcTrajectorySample_t *points,size_t count,double length,
    double progress,float ey,float epsi,const float body[3],
    double end_progress,float end_ey,
    const double first[6],const double second[6],
    double output_first[3],double output_second[3],int *reference_knots);

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

/* Same loaded-frame propagation and differential; optionally returns the
 * exact path sample already computed in the successful final projection.
 * The returned sample is identical to mpc_trajectory_sample(next_progress)
 * and can be used immediately for source-plant corridor evaluation. */
int mpc_trajectory_advance_tracking_with_progress_sampled(
    const MpcTrajectorySample_t *points, size_t count, double lap_length,
    double progress, float e_y, float e_psi,
    float body_dx, float body_dy, float body_yaw, double progress_seed,
    double *next_progress, float *next_e_y, float *next_e_psi,
    float jacobian[3][6], MpcTrajectorySample_t *end_sample);

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

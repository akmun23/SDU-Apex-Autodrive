#include "mpc_reference.h"

#include <math.h>
#include <stdint.h>

static const double kPi = 3.14159265358979323846;
static double wrap_angle(double angle)
{
    return fabs(angle) <= kPi ? angle : remainder(angle, 2.0 * kPi);
}

static double unwrap_near(double angle, double reference)
{
    return reference + wrap_angle(angle - reference);
}

static int finite_sample(const MpcTrajectorySample_t *point)
{
    return point && isfinite(point->s) && isfinite(point->x) &&
        isfinite(point->y) && isfinite(point->heading) &&
        isfinite(point->curvature) && isfinite(point->speed) &&
        isfinite(point->left_bound) && isfinite(point->right_bound) &&
        isfinite(point->acceleration) &&
        point->speed >= 0.0 && point->left_bound > 0.0 &&
        point->right_bound > 0.0;
}

int mpc_trajectory_prepare(
    MpcTrajectorySample_t *points,
    size_t *point_count,
    double *lap_length)
{
    if (!points || !point_count || !lap_length || *point_count < 3) return 0;

    size_t count = *point_count;
    const double closing_dx = points[0].x - points[count - 1].x;
    const double closing_dy = points[0].y - points[count - 1].y;
    if (hypot(closing_dx, closing_dy) < 1.0e-4 && count > 3) --count;
    if (count < 3) return 0;

    for (size_t i = 0; i < count; ++i) {
        if (!finite_sample(&points[i])) return 0;
        if (i > 0 && !(points[i].s > points[i - 1].s)) return 0;
        if (i > 0) {
            points[i].heading = points[i - 1].heading +
                wrap_angle(points[i].heading - points[i - 1].heading);
        }
    }

    const double ds = points[count - 1].s - points[0].s;
    const double closing_distance = hypot(
        points[0].x - points[count - 1].x,
        points[0].y - points[count - 1].y);
    const double length = ds + closing_distance;
    if (!(length > ds) || !isfinite(length)) return 0;

    *point_count = count;
    *lap_length = length;
    return 1;
}

static int valid_path(
    const MpcTrajectorySample_t *points,
    size_t point_count,
    double lap_length)
{
    return points && point_count >= 3 && isfinite(lap_length) &&
        lap_length > points[point_count - 1].s - points[0].s;
}

static int sample_with_frame(
    const MpcTrajectorySample_t *points,
    size_t point_count,
    double lap_length,
    double s,
    MpcTrajectorySample_t *sample,
    MpcPathFrame_t *frame)
{
    if (!valid_path(points, point_count, lap_length) || !sample || !isfinite(s))
        return 0;

    const double first_s = points[0].s;
    double wrapped = fmod(s - first_s, lap_length);
    if (wrapped < 0.0) wrapped += lap_length;
    wrapped += first_s;

    size_t lower = 0;
    size_t upper = 0;
    double s0 = 0.0;
    double s1 = 0.0;
    if (wrapped >= points[point_count - 1].s) {
        lower = point_count - 1;
        upper = 0;
        s0 = points[lower].s;
        s1 = points[upper].s + lap_length;
    } else {
        size_t lo = 0;
        size_t hi = point_count - 1;
        while (hi - lo > 1) {
            const size_t mid = lo + (hi - lo) / 2;
            if (points[mid].s <= wrapped) lo = mid;
            else hi = mid;
        }
        lower = lo;
        upper = hi;
        s0 = points[lower].s;
        s1 = points[upper].s;
    }

    const double query_s = wrapped < s0 ? wrapped + lap_length : wrapped;
    const double span = s1 - s0;
    if (!(span > 0.0)) return 0;
    const double t = fmax(0.0, fmin(1.0, (query_s - s0) / span));
    const MpcTrajectorySample_t *a = &points[lower];
    const MpcTrajectorySample_t *b = &points[upper];
    /* prepare() already unwraps adjacent stored headings; only the closing
     * segment needs a wrap. Avoid trig in every binary-search interpolation. */
    const double b_heading = upper == 0
        ? unwrap_near(b->heading, a->heading) : b->heading;

    *sample = (MpcTrajectorySample_t){
        .s = wrapped,
        .x = a->x + t * (b->x - a->x),
        .y = a->y + t * (b->y - a->y),
        .heading = a->heading + t * (b_heading - a->heading),
        .curvature = a->curvature + t * (b->curvature - a->curvature),
        .speed = a->speed + t * (b->speed - a->speed),
        .left_bound = a->left_bound + t * (b->left_bound - a->left_bound),
        .right_bound = a->right_bound + t * (b->right_bound - a->right_bound),
        .acceleration = a->acceleration +
            t * (b->acceleration - a->acceleration),
    };
    if (frame) {
        frame->sample = *sample;
        frame->dx_ds = (b->x - a->x) / span;
        frame->dy_ds = (b->y - a->y) / span;
        frame->dheading_ds = (b_heading - a->heading) / span;
    }
    return 1;
}

int mpc_trajectory_sample(
    const MpcTrajectorySample_t *points, size_t count, double length,
    double s, MpcTrajectorySample_t *sample)
{
    return sample_with_frame(points, count, length, s, sample, NULL);
}

/* Invert P=C(s)+e_y*N(s) for the same interpolated heading that the MPC
 * uses to reconstruct its world pose. A chord-only projection and a
 * constant-curvature propagation do not define that same coordinate map. */
static int project_reference_frame(
    const MpcTrajectorySample_t *points, size_t count, double length,
    double x, double y, double heading, double seed,
    MpcPathFrame_t *frame, double *progress, double *ey, double *epsi)
{
    double station = seed;
    for (int i = 0; i < 8; ++i) {
        MpcTrajectorySample_t sample;
        if (!sample_with_frame(points, count, length, station, &sample, frame))
            return 0;
        const double c = cos(sample.heading), s = sin(sample.heading);
        const double ex = x - sample.x, ey_world = y - sample.y;
        const double tangent_error = c * ex + s * ey_world;
        *ey = -s * ex + c * ey_world;
        const double tangent_scale = c * frame->dx_ds + s * frame->dy_ds -
            frame->dheading_ds * (*ey);
        if (!(tangent_scale > 0.05) || !isfinite(tangent_error)) return 0;
        if (fabs(tangent_error) <= 1.0e-8) {
            *progress = station;
            *epsi = wrap_angle(heading - sample.heading);
            return 1;
        }
        const double correction = tangent_error / tangent_scale;
        station += fmax(-0.5, fmin(0.5, correction));
    }
    return 0;
}

int mpc_trajectory_advance_tracking_with_progress(
    const MpcTrajectorySample_t *points, size_t count, double length,
    double progress, float ey, float epsi,
    float dx, float dy, float dyaw, double progress_seed,
    double *next_progress, float *next_ey, float *next_epsi,
    float jacobian[3][6])
{
    if (!next_progress || !next_ey || !next_epsi ||
        !isfinite(ey) || !isfinite(epsi) || !isfinite(dx) ||
        !isfinite(dy) || !isfinite(dyaw)) return 0;
    MpcTrajectorySample_t start;
    MpcPathFrame_t start_frame;
    if (!sample_with_frame(points, count, length, progress,
            &start, &start_frame)) return 0;
    const double c0 = cos(start.heading), s0 = sin(start.heading);
    const double yaw = start.heading + epsi;
    const double cb = cos(yaw), sb = sin(yaw);
    const double x = start.x - s0 * ey + cb * dx - sb * dy;
    const double y = start.y + c0 * ey + sb * dx + cb * dy;
    MpcPathFrame_t end;
    double result_ey, result_epsi;
    if (!project_reference_frame(points, count, length, x, y, yaw + dyaw,
            progress_seed, &end, next_progress, &result_ey, &result_epsi))
        return 0;
    *next_ey = (float)result_ey;
    *next_epsi = (float)result_epsi;
    if (jacobian) {
        const double c = cos(end.sample.heading), s = sin(end.sample.heading);
        const double a = c * end.dx_ds + s * end.dy_ds -
            end.dheading_ds * result_ey;
        const double b = -s * end.dx_ds + c * end.dy_ds;
        const double h0 = start_frame.dheading_ds;
        const double px[6] = {-s0, -sb * dx - cb * dy, cb, -sb, 0.0,
            start_frame.dx_ds - c0 * ey * h0 + (-sb * dx - cb * dy) * h0};
        const double py[6] = { c0,  cb * dx - sb * dy, sb,  cb, 0.0,
            start_frame.dy_ds - s0 * ey * h0 + ( cb * dx - sb * dy) * h0};
        for (int j = 0; j < 6; ++j) {
            const double ds = (c * px[j] + s * py[j]) / a;
            jacobian[0][j] = (float)(-s * px[j] + c * py[j] - b * ds);
            jacobian[1][j] = (float)((j == 5 ? h0 :
                (j == 1 || j == 4 ? 1.0 : 0.0)) -
                end.dheading_ds * ds);
            jacobian[2][j] = (float)ds;
        }
    }
    return 1;
}

int mpc_trajectory_advance_tracking(
    const MpcTrajectorySample_t *points, size_t count, double length,
    double progress, float ey, float epsi,
    float dx, float dy, float dyaw, double progress_seed,
    double *next_progress, float *next_ey, float *next_epsi,
    float jacobian[2][5])
{
    float full[3][6];
    const int ok = mpc_trajectory_advance_tracking_with_progress(points, count,
        length, progress, ey, epsi, dx, dy, dyaw, progress_seed,
        next_progress, next_ey, next_epsi, jacobian ? full : NULL);
    if (ok && jacobian)
        for (int row = 0; row < 2; ++row)
            for (int col = 0; col < 5; ++col)
                jacobian[row][col] = full[row][col];
    return ok;
}

static size_t wrapped_index(long long index, size_t count)
{
    const long long n = (long long)count;
    long long value = index % n;
    if (value < 0) value += n;
    return (size_t)value;
}

static void consider_segment(
    const MpcTrajectorySample_t *points,
    size_t count,
    double lap_length,
    double x,
    double y,
    double pose_heading,
    size_t segment,
    int *found,
    double *best_distance_squared,
    MpcPathProjection_t *best)
{
    const size_t next = (segment + 1) % count;
    const MpcTrajectorySample_t *a = &points[segment];
    const MpcTrajectorySample_t *b = &points[next];
    const double dx = b->x - a->x;
    const double dy = b->y - a->y;
    const double length_squared = dx * dx + dy * dy;
    if (!(length_squared > 1.0e-12)) return;

    const double t = fmax(0.0, fmin(1.0,
        ((x - a->x) * dx + (y - a->y) * dy) / length_squared));
    const double px = a->x + t * dx;
    const double py = a->y + t * dy;
    const double tangent = atan2(dy, dx);
    const double heading_error = wrap_angle(pose_heading - tangent);
    if (fabs(heading_error) > 0.5 * kPi) return;

    const double ex = x - px;
    const double ey = y - py;
    const double distance_squared = ex * ex + ey * ey;
    if (*found && distance_squared >= *best_distance_squared) return;

    const double s0 = a->s;
    const double s1 = next == 0 ? points[0].s + lap_length : b->s;
    *found = 1;
    *best_distance_squared = distance_squared;
    *best = (MpcPathProjection_t){
        .s = s0 + t * (s1 - s0),
        .x = px,
        .y = py,
        .heading = tangent,
        .lateral_error = -sin(tangent) * ex + cos(tangent) * ey,
        .heading_error = heading_error,
        .distance = sqrt(distance_squared),
        .segment = segment,
    };
}

int mpc_trajectory_project(
    const MpcTrajectorySample_t *points,
    size_t point_count,
    double lap_length,
    double x,
    double y,
    double heading,
    size_t previous_segment,
    size_t local_search_radius,
    MpcPathProjection_t *projection)
{
    if (!valid_path(points, point_count, lap_length) || !projection ||
        !isfinite(x) || !isfinite(y) || !isfinite(heading)) return 0;

    int found = 0;
    double best_distance_squared = INFINITY;
    MpcPathProjection_t best = {0};
    if (previous_segment < point_count) {
        const long long radius = (long long)local_search_radius;
        for (long long offset = -radius; offset <= radius; ++offset) {
            consider_segment(points, point_count, lap_length, x, y, heading,
                wrapped_index((long long)previous_segment + offset, point_count),
                &found, &best_distance_squared, &best);
        }
    } else {
        for (size_t i = 0; i < point_count; ++i) {
            consider_segment(points, point_count, lap_length, x, y, heading, i,
                &found, &best_distance_squared, &best);
        }
    }

    if (!found && previous_segment < point_count) {
        best_distance_squared = INFINITY;
        for (size_t i = 0; i < point_count; ++i) {
            consider_segment(points, point_count, lap_length, x, y, heading, i,
                &found, &best_distance_squared, &best);
        }
    }
    if (!found) return 0;
    MpcPathFrame_t frame;
    double station, ey, epsi;
    if (!project_reference_frame(points, point_count, lap_length,
            x, y, heading, best.s, &frame, &station, &ey, &epsi)) return 0;
    *projection = best;
    projection->s = frame.sample.s;
    projection->x = frame.sample.x;
    projection->y = frame.sample.y;
    projection->heading = frame.sample.heading;
    projection->lateral_error = ey;
    projection->heading_error = epsi;
    projection->distance = hypot(x - frame.sample.x, y - frame.sample.y);
    return 1;
}

int mpc_reference_build_speed_seed(
    const MpcTrajectorySample_t *points,
    size_t point_count,
    double lap_length,
    double initial_s,
    double speed_ceiling,
    double dt,
    int horizon,
    TrajectoryReferencePoint_t *references,
    double *progress)
{
    if (!valid_path(points, point_count, lap_length) || !references ||
        !progress || !isfinite(initial_s) || !isfinite(speed_ceiling) ||
        speed_ceiling < 0.0 || !isfinite(dt) || !(dt > 0.0) ||
        horizon <= 0 || horizon > PREDICTION_HORIZON) return 0;

    progress[0] = initial_s;
    for (int k = 0; k <= horizon; ++k) {
        MpcTrajectorySample_t sample;
        if (!mpc_trajectory_sample(
                points, point_count, lap_length, progress[k], &sample)) return 0;
        const double speed = fmin(sample.speed, speed_ceiling);
        references[k] = (TrajectoryReferencePoint_t){
            .reference_lateral_error = 0.0f,
            .reference_heading_error = 0.0f,
            .reference_velocity = (float)speed,
            .reference_lateral_velocity = 0.0f,
            .reference_yaw_rate = (float)(sample.curvature * speed),
            .path_curvature = (float)sample.curvature,
            .left_wall_bound = (float)sample.left_bound,
            .right_wall_bound = (float)sample.right_bound,
        };
        if (k < horizon) progress[k + 1] = progress[k] + dt * speed;
    }
    return 1;
}

#include "mpc_reference.h"

#include <math.h>
#include <stdint.h>

static const double kPi = 3.14159265358979323846;
static double wrap_angle(double angle)
{
    return atan2(sin(angle), cos(angle));
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

int mpc_trajectory_sample(
    const MpcTrajectorySample_t *points,
    size_t point_count,
    double lap_length,
    double s,
    MpcTrajectorySample_t *sample)
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
    const double b_heading = unwrap_near(b->heading, a->heading);

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
    return 1;
}

static size_t wrapped_index(long long index, size_t count)
{
    const long long n = (long long)count;
    long long value = index % n;
    if (value < 0) value += n;
    return (size_t)value;
}

static double segment_arc_length(
    const MpcTrajectorySample_t *points,
    size_t count,
    double lap_length,
    size_t segment)
{
    if (!points || count < 2 || segment >= count) return 0.0;
    const size_t next = (segment + 1) % count;
    const double s0 = points[segment].s;
    const double s1 = next == 0 ? points[0].s + lap_length : points[next].s;
    const double ds = s1 - s0;
    return isfinite(ds) && ds > 0.0 ? ds : 0.0;
}

size_t mpc_trajectory_search_radius_for_distance(
    const MpcTrajectorySample_t *points,
    size_t point_count,
    double lap_length,
    size_t previous_segment,
    double search_distance_m)
{
    if (!valid_path(points, point_count, lap_length) ||
        previous_segment >= point_count || !isfinite(search_distance_m) ||
        search_distance_m <= 0.0) return 0;

    double forward_m = 0.0;
    double backward_m = 0.0;
    size_t radius = 0;
    while (radius + 1 < point_count &&
           (forward_m < search_distance_m ||
            backward_m < search_distance_m)) {
        const size_t next_radius = radius + 1;
        if (forward_m < search_distance_m) {
            const size_t segment = wrapped_index(
                (long long)previous_segment + (long long)radius,
                point_count);
            const double ds = segment_arc_length(
                points, point_count, lap_length, segment);
            if (!(ds > 0.0)) return 0;
            forward_m += ds;
        }
        if (backward_m < search_distance_m) {
            const size_t segment = wrapped_index(
                (long long)previous_segment - (long long)next_radius,
                point_count);
            const double ds = segment_arc_length(
                points, point_count, lap_length, segment);
            if (!(ds > 0.0)) return 0;
            backward_m += ds;
        }
        radius = next_radius;
    }
    return radius;
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

    /* Once progress is initialized, never reacquire from the full closed
     * track. A global nearest-segment fallback can jump to a parallel hairpin
     * branch after a transient localization/control error. Returning failure
     * lets the controller enter its bounded recovery path instead. */
    if (!found) return 0;
    *projection = best;
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

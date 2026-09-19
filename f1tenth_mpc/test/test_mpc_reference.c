#include "mpc_reference.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef MPC_TEST_TRAJECTORY_PATH
#error "MPC_TEST_TRAJECTORY_PATH must point at the current raceline CSV"
#endif

#define TEST_MAX_POINTS 3000

static int failures = 0;
static MpcTrajectorySample_t track_points[TEST_MAX_POINTS];

static uint32_t random_state = 0x5eeda11u;

static double deterministic_unit(void)
{
    random_state = random_state * 1664525u + 1013904223u;
    return (double)random_state / 4294967296.0;
}

static void check_true(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

static void check_near(double actual, double expected, double tolerance,
                       const char *message)
{
    check_true(isfinite(actual) && fabs(actual - expected) <= tolerance,
               message);
}

static size_t make_rectangle(MpcTrajectorySample_t points[4])
{
    points[0] = (MpcTrajectorySample_t){0.0, 0.0, 0.0, 0.0, 0.0, 2.231232, 1.0, 1.0, 0.0};
    points[1] = (MpcTrajectorySample_t){2.0, 2.0, 0.0, 0.0, 0.0, 2.231232, 1.0, 1.0, 0.0};
    points[2] = (MpcTrajectorySample_t){2.1, 2.0, 0.1, 1.57079632679, 0.0, 2.231232, 1.0, 1.0, 0.0};
    points[3] = (MpcTrajectorySample_t){4.1, 0.0, 0.1, 3.14159265359, 0.0, 2.231232, 1.0, 1.0, 0.0};
    return 4;
}

static void test_current_raceline_loads_and_wraps(void)
{
    FILE *input = fopen(MPC_TEST_TRAJECTORY_PATH, "r");
    check_true(input != NULL, "open the project raceline fixture");
    if (!input) return;

    char line[512];
    size_t count = 0;
    while (fgets(line, sizeof(line), input)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        double value[9];
        if (sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                   &value[0], &value[1], &value[2], &value[3], &value[4],
                   &value[5], &value[6], &value[7], &value[8]) != 9) {
            continue;
        }
        if (count >= TEST_MAX_POINTS) {
            check_true(0, "trajectory fits the configured MPC point capacity");
            break;
        }
        track_points[count++] = (MpcTrajectorySample_t){
            .s = value[0], .x = value[1], .y = value[2], .heading = value[3],
            .curvature = value[4], .speed = value[5],
            .left_bound = value[7], .right_bound = value[8],
            .acceleration = value[6],
        };
    }
    fclose(input);

    check_true(count == 313, "current exact trajectory contains 313 rows");
    double lap_length = 0.0;
    check_true(mpc_trajectory_prepare(track_points, &count, &lap_length),
               "current trajectory validates and prepares");
    check_true(count == 313, "current exact trajectory contains 313 unique points");
    check_near(lap_length, 53.21, 0.03,
               "closed-loop length includes the final-to-first segment");

    MpcTrajectorySample_t at_start;
    MpcTrajectorySample_t at_next_lap;
    check_true(mpc_trajectory_sample(
        track_points, count, lap_length, track_points[0].s, &at_start),
        "sample exact starting arc length");
    check_true(mpc_trajectory_sample(
        track_points, count, lap_length, track_points[0].s + lap_length,
        &at_next_lap), "sample one full lap later");
    check_near(at_start.x, track_points[0].x, 1.0e-9,
               "exact waypoint interpolation preserves x");
    check_near(at_next_lap.y, track_points[0].y, 1.0e-9,
               "arc length wraps periodically at lap boundary");

    for (int sample_index = 0; sample_index < 1000; ++sample_index) {
        const size_t segment = (size_t)(deterministic_unit() * (double)count);
        const size_t next = (segment + 1) % count;
        const double fraction = deterministic_unit();
        const double signed_offset = 0.30 * deterministic_unit() - 0.15;
        const double dx = track_points[next].x - track_points[segment].x;
        const double dy = track_points[next].y - track_points[segment].y;
        const double tangent = atan2(dy, dx);
        const double px = track_points[segment].x + fraction * dx;
        const double py = track_points[segment].y + fraction * dy;
        const double query_x = px - signed_offset * sin(tangent);
        const double query_y = py + signed_offset * cos(tangent);
        MpcPathProjection_t projected;
        const int projected_ok = mpc_trajectory_project(
            track_points, count, lap_length, query_x, query_y, tangent,
            segment, 4, &projected);
        check_true(projected_ok,
                   "randomized current-track point projects locally");
        if (projected_ok) {
            check_true(fabs(projected.lateral_error - signed_offset) <= 0.002,
                "continuous projection matches dense segment oracle within 2 mm");
        }
    }
}

static void test_continuous_interpolation_and_heading_wrap(void)
{
    MpcTrajectorySample_t points[4] = {
        {0.0, 0.0, 0.0, 3.10, 0.1, 2.0, 0.8, 0.7, 0.0},
        {1.0, 1.0, 0.0, -3.10, 0.3, 4.0, 0.6, 0.5, 0.0},
        {2.0, 1.0, 1.0, -1.57, 0.5, 3.0, 0.7, 0.6, 0.0},
        {3.0, 0.0, 1.0, 0.0, 0.2, 2.0, 0.8, 0.7, 0.0},
    };
    size_t count = 4;
    double lap_length = 0.0;
    check_true(mpc_trajectory_prepare(points, &count, &lap_length),
               "prepare heading-wrap interpolation fixture");
    MpcTrajectorySample_t midpoint;
    check_true(mpc_trajectory_sample(points, count, lap_length, 0.5, &midpoint),
               "sample between ordinary waypoints");
    check_near(midpoint.x, 0.5, 1.0e-12,
               "position interpolation is continuous within a segment");
    check_true(fabs(midpoint.heading) > 3.13,
               "heading interpolates through pi instead of through zero");
    check_near(midpoint.curvature, 0.2, 1.0e-12,
               "curvature is interpolated rather than nearest-neighbor sampled");
    check_near(midpoint.speed, 3.0, 1.0e-12,
               "speed is interpolated rather than nearest-neighbor sampled");
    check_near(midpoint.acceleration, 0.0, 1.0e-12,
               "acceleration is interpolated rather than nearest-neighbor sampled");
    check_near(midpoint.left_bound, 0.7, 1.0e-12,
               "left corridor bound is interpolated");
}

static void test_continuous_projection_and_direction_gate(void)
{
    MpcTrajectorySample_t points[4];
    size_t count = make_rectangle(points);
    double lap_length = 0.0;
    check_true(mpc_trajectory_prepare(points, &count, &lap_length),
               "prepare projection fixture");

    MpcPathProjection_t projection;
    check_true(mpc_trajectory_project(points, count, lap_length,
        0.5, 0.2, 0.0, SIZE_MAX, 0, &projection),
        "project pose onto continuous path segment");
    check_true(projection.segment == 0,
               "projection identifies the containing segment");
    check_near(projection.s, 0.5, 1.0e-9,
               "projection returns continuous arc length within segment");
    check_near(projection.lateral_error, 0.2, 1.0e-9,
               "projection returns signed left-positive cross-track error");

    check_true(mpc_trajectory_project(points, count, lap_length,
        0.999, 0.1, 0.0, 0, 1, &projection),
        "projection remains continuous at a waypoint boundary");
    check_near(projection.s, 0.999, 0.003,
               "waypoint-boundary projection does not quantize to a sample");

    check_true(mpc_trajectory_project(points, count, lap_length,
        1.0, 0.09, 0.0, SIZE_MAX, 0, &projection),
        "direction-gated projection finds a compatible track segment");
    check_true(projection.segment == 0,
               "direction gate rejects the closer opposite-direction alias");
    check_near(projection.distance, 0.09, 1.0e-9,
               "opposite-direction segment is not selected by proximity alone");
}

static void test_metric_local_projection_window_and_no_global_reacquire(void)
{
    MpcTrajectorySample_t points[4];
    size_t count = make_rectangle(points);
    double lap_length = 0.0;
    check_true(mpc_trajectory_prepare(points, &count, &lap_length),
               "prepare metric projection-window fixture");

    const size_t short_radius = mpc_trajectory_search_radius_for_distance(
        points, count, lap_length, 0, 0.15);
    const size_t long_radius = mpc_trajectory_search_radius_for_distance(
        points, count, lap_length, 0, 2.05);
    check_true(short_radius == 2,
               "metric search covers the requested distance in both directions");
    check_true(long_radius >= short_radius && long_radius < count,
               "metric search radius grows with arc distance, not point count");

    MpcPathProjection_t projection;
    check_true(!mpc_trajectory_project(points, count, lap_length,
        1.0, 0.1, 3.14159265359, 0, 0, &projection),
        "initialized local projection does not snap to remote opposite branch");
    check_true(mpc_trajectory_project(points, count, lap_length,
        1.0, 0.1, 3.14159265359, SIZE_MAX, 0, &projection),
        "full-track projection remains available for first acquisition");
    check_true(projection.segment == 2,
        "first acquisition may choose the heading-compatible remote branch");
}

static void test_speed_seed_uses_n_plus_one_physical_progress(void)
{
    MpcTrajectorySample_t points[4];
    size_t count = make_rectangle(points);
    double lap_length = 0.0;
    check_true(mpc_trajectory_prepare(points, &count, &lap_length),
               "prepare reference progress fixture");

    TrajectoryReferencePoint_t references[PREDICTION_HORIZON + 1];
    double progress[PREDICTION_HORIZON + 1];
    const double speed = 2.231232;
    check_true(mpc_reference_build_speed_seed(
        points, count, lap_length, 0.0, 16.0, 0.025,
        PREDICTION_HORIZON, references, progress),
        "build N+1 reference samples for N dynamics stages");
    check_near(progress[1], speed * 0.025, 1.0e-8,
               "low-speed reference advances by v_ref times dt");
    check_near(progress[PREDICTION_HORIZON],
        PREDICTION_HORIZON * speed * 0.025, 1.0e-7,
        "N30 low-speed reference spans physical distance, not a 0.25 m floor");
    check_true(references[PREDICTION_HORIZON].left_wall_bound > 0.0f &&
               references[PREDICTION_HORIZON].reference_velocity > 0.0f,
        "terminal reference at index N is populated");
}

int main(void)
{
    test_current_raceline_loads_and_wraps();
    test_continuous_interpolation_and_heading_wrap();
    test_continuous_projection_and_direction_gate();
    test_metric_local_projection_window_and_no_global_reacquire();
    test_speed_seed_uses_n_plus_one_physical_progress();
    if (failures != 0) {
        fprintf(stderr, "%d MPC reference test(s) failed\n", failures);
        return 1;
    }
    puts("MPC continuous-reference tests passed");
    return 0;
}

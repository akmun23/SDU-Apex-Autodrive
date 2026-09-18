#define _POSIX_C_SOURCE 200809L

#include "mpc_rti.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef MPC_RTI_BENCH_TRAJECTORY_PATH
#error "MPC_RTI_BENCH_TRAJECTORY_PATH must name the accepted raceline CSV"
#endif

enum { kMaximumPoints = 4000, kBenchmarkCycles = 500 };

static int compare_double(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double percentile(const double *sorted, int count, double fraction)
{
    int index = (int)ceil(fraction * (double)count) - 1;
    if (index < 0) index = 0;
    if (index >= count) index = count - 1;
    return sorted[index];
}

static int load_trajectory(MpcTrajectorySample_t *points, size_t *count,
                           double *lap_length)
{
    FILE *input = fopen(MPC_RTI_BENCH_TRAJECTORY_PATH, "r");
    if (!input) return 0;
    char line[1024];
    *count = 0;
    while (fgets(line, sizeof(line), input)) {
        if (line[0] == '#' || line[0] == '\n' || *count >= kMaximumPoints)
            continue;
        double values[16];
        size_t value_count = 0;
        char *cursor = line;
        while (*cursor && value_count < 16) {
            char *end = NULL;
            const double value = strtod(cursor, &end);
            if (end == cursor) {
                value_count = 0;
                break;
            }
            values[value_count++] = value;
            if (*end == ',') cursor = end + 1;
            else if (*end == '\n' || *end == '\0' || *end == '\r') break;
            else {
                value_count = 0;
                break;
            }
        }
        if (value_count < 6) continue;
        MpcTrajectorySample_t point = {
            .s = values[0], .x = values[1], .y = values[2],
            .heading = values[3], .curvature = values[4],
            .speed = values[5],
            .left_bound = value_count >= 9 ? values[7] : 1.0,
            .right_bound = value_count >= 9 ? values[8] : 1.0};
        points[(*count)++] = point;
    }
    fclose(input);
    return mpc_trajectory_prepare(points, count, lap_length);
}

static double seconds_since(const struct timespec *start,
                            const struct timespec *finish)
{
    return (double)(finish->tv_sec - start->tv_sec) +
        1.0e-9 * (double)(finish->tv_nsec - start->tv_nsec);
}

int main(void)
{
    MpcTrajectorySample_t trajectory[kMaximumPoints];
    size_t trajectory_count = 0;
    double lap_length = 0.0;
    if (!load_trajectory(trajectory, &trajectory_count, &lap_length)) {
        fprintf(stderr, "failed to load benchmark raceline: %s\n",
                MPC_RTI_BENCH_TRAJECTORY_PATH);
        return 2;
    }

    const float speed = (float)trajectory[0].speed;
    const float target_speed = speed -
        (MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 +
         MPC_LONGITUDINAL_SPEED_COEFF_PER_S * speed) /
            MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S;
    const float curvature = (float)trajectory[0].curvature;
    const float steering_command = fmaxf(-SOURCE_MAX_STEERING_RAD,
        fminf(SOURCE_MAX_STEERING_RAD, atanf(curvature /
            MPC_YAW_RATE_STEERING_GAIN_PER_M)));
    const MpcRtiState_t state = {
        .plant = {.e_y = 0.0f, .e_psi = 0.0f, .u = speed, .v = 0.0f,
                  .r = curvature * speed, .target_speed = target_speed,
                  .steering_command = steering_command},
        .previous_steering_rate = 0.0f,
        .previous_target_speed_rate = 0.0f};
    const MpcRtiConfiguration_t model = {
        .weight_e_y = 1500.0f, .weight_e_psi = 50.0f,
        .weight_u = 200.0f, .weight_v = 0.0f, .weight_r = 1.5f,
        .weight_steering_command = 1.0f, .weight_steering_rate = 2.0f,
        .weight_target_speed_rate = 0.5f,
        .weight_steering_rate_change = 5.0f,
        .weight_target_speed_rate_change = 5.0f,
        .terminal_multiplier = 3.0f, .max_speed_mps = 16.0f,
        .active_speed_ceiling_mps = 16.0f,
        .max_steering_rad = SOURCE_MAX_STEERING_RAD,
        .max_steering_rate_radps = SOURCE_STEERING_RATE_RADPS,
        .max_target_speed_rate_increase_mps2 = 3.0f,
        .max_target_speed_rate_reduction_mps2 = 8.0f,
        .corridor_margin_m = 0.05f,
        .corridor_preview_halfwidth_m = 0.10f,
        .nonlinear_corridor_tolerance_m = 0.001f};
    const MpcRtiCycleConfiguration_t configuration = {
        .model = model,
        .solver = {.rho = 7.0f, .rho_u = 7.0f, .tolerance = 0.01f,
                   .max_iterations = 50, .adaptive_rho = 0,
                   .shared_rho = 0},
        .degraded_residual_limit = 0.05f,
        .maximum_regularization = 1.0e-2f,
        .max_consecutive_degraded_solves = 3};

    MpcRtiMemory_t memory = {0};
    MpcRtiNominal_t diagnostic_nominal;
    MpcRtiReference_t diagnostic_references[PREDICTION_HORIZON + 1];
    const int nominal_ok = mpc_rti_build_nominal(
        &state, 0.0, NULL, trajectory, trajectory_count, lap_length,
        0.025f, PREDICTION_HORIZON, &model, &diagnostic_nominal,
        diagnostic_references);
    fprintf(stderr, "benchmark diagnostic nominal_ok=%d first_kappa=%.7g "
            "left=%.7g right=%.7g\n", nominal_ok, curvature,
            (double)trajectory[0].left_bound,
            (double)trajectory[0].right_bound);
    double durations_ms[kBenchmarkCycles];
    int optimal = 0;
    int degraded = 0;
    int rejected = 0;
    int max_iterations = 0;
    double max_regularization = 0.0;
    int statuses[7] = {0};
    for (int i = 0; i < kBenchmarkCycles; ++i) {
        MpcRtiCycleResult_t result;
        struct timespec start;
        struct timespec finish;
        clock_gettime(CLOCK_MONOTONIC, &start);
        const MpcRtiCycleStatus_t status = mpc_rti_solve_cycle(
            &state, 0.0, trajectory, trajectory_count, lap_length,
            0.025f, PREDICTION_HORIZON, &configuration, &memory, &result);
        clock_gettime(CLOCK_MONOTONIC, &finish);
        durations_ms[i] = 1000.0 * seconds_since(&start, &finish);
        if (status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL) ++optimal;
        else if (status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED) ++degraded;
        else ++rejected;
        if ((int)status >= 0 && (int)status < 7) ++statuses[(int)status];
        if (result.solver_iterations > max_iterations)
            max_iterations = result.solver_iterations;
        if (result.maximum_regularization > max_regularization)
            max_regularization = result.maximum_regularization;
    }
    qsort(durations_ms, kBenchmarkCycles, sizeof(durations_ms[0]),
          compare_double);
    printf("RTI N30 40Hz core benchmark on %zu-point raceline, %.3f m lap, "
           "raceline speed %.3f m/s; cycles=%d\n", trajectory_count, lap_length,
           (double)speed,
           kBenchmarkCycles);
    printf("solve ms p50/p95/p99/max=%.3f/%.3f/%.3f/%.3f; "
           "optimal=%d degraded=%d rejected=%d max_iter=%d max_lambda=%.7g\n",
           percentile(durations_ms, kBenchmarkCycles, 0.50),
           percentile(durations_ms, kBenchmarkCycles, 0.95),
           percentile(durations_ms, kBenchmarkCycles, 0.99),
           durations_ms[kBenchmarkCycles - 1], optimal, degraded, rejected,
           max_iterations, max_regularization);
    fprintf(stderr, "cycle statuses [optimal,degraded,input,solver,residual,regularization,nonlinear]=[%d,%d,%d,%d,%d,%d,%d]\n",
            statuses[0], statuses[1], statuses[2], statuses[3], statuses[4],
            statuses[5], statuses[6]);
    return rejected == 0 ? 0 : 1;
}

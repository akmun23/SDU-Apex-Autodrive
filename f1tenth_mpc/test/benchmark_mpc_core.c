#define _POSIX_C_SOURCE 200809L

#include "mpc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { warmup_count = 50, sample_count = 500 };

static double elapsed_ms(const struct timespec *start, const struct timespec *end)
{
    return (double)(end->tv_sec - start->tv_sec) * 1000.0 +
        (double)(end->tv_nsec - start->tv_nsec) / 1.0e6;
}

static int compare_double(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

int main(void)
{
    const MpcConfiguration_t configuration = get_default_configuration();
    TrajectoryReferencePoint_t reference[PREDICTION_HORIZON];
    for (int i = 0; i < PREDICTION_HORIZON; ++i) {
        reference[i] = (TrajectoryReferencePoint_t){
            .reference_velocity = 4.0f,
            .path_curvature = 0.05f,
            .left_wall_bound = 1.0f,
            .right_wall_bound = 1.0f,
        };
    }
    FrenetState_t state = {
        .flat_error = 0.05f,
        .fhead_error = 0.02f,
        .flong_vel = 3.5f,
        .flat_vel = 0.0f,
        .fyaw_rate = 0.0f,
        .ftarget_speed_mps = 3.5f,
    };
    ControlInput_t previous_command = {0.0f, 0.0f};
    double timings[sample_count];
    uint16_t maximum_iterations_used = 0;
    int maximum_iteration_statuses = 0;
    float final_primal_residual = NAN;
    float final_dual_residual = NAN;
    mpc_initialize_with_configuration(&configuration);

    for (int sample = -warmup_count; sample < sample_count; ++sample) {
        MpcSolverResult_t result = {0};
        struct timespec start;
        struct timespec end;
        mpc_set_previous_command_with_dt(
            &previous_command, TIME_STEP_SECONDS, 0.0f, 0);
        clock_gettime(CLOCK_MONOTONIC, &start);
        const MpcSolverStatus_t status = mpc_compute_optimal_control(
            &state, reference, &result);
        clock_gettime(CLOCK_MONOTONIC, &end);
        if (status != MPC_STATUS_SUCCESS &&
            status != MPC_STATUS_MAXIMUM_ITERATIONS_REACHED) {
            fprintf(stderr, "MPC benchmark solve failed with status %d\n", status);
            return 1;
        }
        if (status == MPC_STATUS_MAXIMUM_ITERATIONS_REACHED) {
            ++maximum_iteration_statuses;
        }
        if (result.iterations_used > maximum_iterations_used) {
            maximum_iterations_used = result.iterations_used;
        }
        final_primal_residual = result.primal_residual;
        final_dual_residual = result.dual_residual;
        previous_command = result.optimal_control;
        state.ftarget_speed_mps = fminf(4.0f, fmaxf(0.0f,
            state.ftarget_speed_mps +
            result.optimal_control.target_speed_rate * TIME_STEP_SECONDS));
        if (sample >= 0) {
            timings[sample] = elapsed_ms(&start, &end);
        }
    }

    qsort(timings, sample_count, sizeof(timings[0]), compare_double);
    const double p50 = timings[(sample_count * 50) / 100];
    const double p95 = timings[(sample_count * 95) / 100];
    const double p99 = timings[(sample_count * 99) / 100];
    printf("10-state N30 MPC core solve time (no ROS): "
           "p50=%.3f ms p95=%.3f ms p99=%.3f ms max=%.3f ms\n",
           p50, p95, p99, timings[sample_count - 1]);
    printf("ADMM status: max-iteration samples=%d/%d max-iterations-used=%u "
           "last-primal=%.6g last-dual=%.6g\n",
           maximum_iteration_statuses, sample_count,
           (unsigned)maximum_iterations_used,
           (double)final_primal_residual, (double)final_dual_residual);
    return 0;
}

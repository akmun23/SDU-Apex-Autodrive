#include "mpc.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

static void check_true(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

int main(void)
{
    TrajectoryReferencePoint_t reference[PREDICTION_HORIZON];
    for (int index = 0; index < PREDICTION_HORIZON; ++index) {
        reference[index] = (TrajectoryReferencePoint_t){
            .reference_velocity = 4.0f,
            .path_curvature = 0.05f,
            .left_wall_bound = 1.0f,
            .right_wall_bound = 1.0f,
        };
    }

    mpc_initialize();
    const FrenetState_t state = {
        .flat_error = 0.05f,
        .fhead_error = 0.02f,
        .flong_vel = 3.5f,
        .flat_vel = 0.0f,
        .fyaw_rate = 0.0f,
        .ftarget_speed_mps = 3.5f,
    };
    MpcSolverResult_t result = {0};
    const MpcSolverStatus_t status = mpc_compute_optimal_control(
        &state, reference, &result);

    check_true(status == MPC_STATUS_SUCCESS ||
               status == MPC_STATUS_MAXIMUM_ITERATIONS_REACHED,
               "source-command MPC returns a usable bounded solution");
    check_true(isfinite(result.optimal_control.steer_ang) &&
               isfinite(result.optimal_control.target_speed_rate),
               "MPC output is finite");
    check_true(fabsf(result.optimal_control.steer_ang) <= 0.523599f,
               "MPC steering respects the Unity source limit");
    check_true(result.optimal_control.target_speed_rate <= 3.0001f &&
               result.optimal_control.target_speed_rate >= -8.0001f,
               "MPC longitudinal output respects target-speed policy bounds");

    if (failures != 0) {
        fprintf(stderr, "%d MPC core test(s) failed\n", failures);
        return 1;
    }
    puts("MPC core tests passed");
    return 0;
}

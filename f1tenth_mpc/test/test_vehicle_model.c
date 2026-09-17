#include "vehicle_model.h"

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

static void check_close(float actual, float expected, float tolerance,
                        const char *message)
{
    check_true(isfinite(actual) && fabsf(actual - expected) <= tolerance,
               message);
}

static void test_zero_state_is_finite(void)
{
    const VehicleState_t state = {0};
    const ControlInput_t command = {0};
    const VehicleState_t next = vehicle_model_predict_next_state(
        &state, &command, TIME_STEP_SECONDS);
    check_close(next.pos_x, 0.0f, 1.0e-7f, "zero state retains x");
    check_close(next.pos_y, 0.0f, 1.0e-7f, "zero state retains y");
    check_close(next.long_vel, 0.0f, 1.0e-7f, "zero state retains speed");
    check_true(isfinite(next.heading) && isfinite(next.yaw_rate),
               "zero state remains finite");
}

static void test_source_geometry_and_command_policy(void)
{
    const VehicleParameters_t parameters = vehicle_model_default_parameters();
    check_close(parameters.steering_wheelbase_m, 0.324f, 1.0e-6f,
                "uses Unity steering wheelbase");
    check_close(parameters.max_steering_angle, 0.5235987756f, 1.0e-6f,
                "uses Unity steering limit");
    check_close(parameters.steering_rate_radps, 3.2f, 1.0e-6f,
                "uses Unity steering rate");
    check_close(parameters.maximum_command_speed_mps, 16.0f, 1.0e-6f,
                "uses 16 m/s project command ceiling");
}

static void test_geometric_turn_sign_and_target_speed_bound(void)
{
    const VehicleState_t state = {0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f};
    const ControlInput_t left = {0.10f, 0.0f};
    const ControlInput_t right = {-0.10f, 0.0f};
    const VehicleState_t left_next = vehicle_model_predict_next_state(
        &state, &left, TIME_STEP_SECONDS);
    const VehicleState_t right_next = vehicle_model_predict_next_state(
        &state, &right, TIME_STEP_SECONDS);
    check_true(left_next.yaw_rate > 0.0f && right_next.yaw_rate < 0.0f,
               "source steering signs produce opposite turns");

    const ControlInput_t excessive = {4.0f, 100.0f};
    const ControlInput_t bounded = vehicle_model_saturate_control(&excessive);
    check_true(bounded.steer_ang <= 0.5235988f &&
               bounded.target_speed_rate <= 3.0f,
               "MPC output is bounded by source and command policy limits");
}

static void test_shared_stage_and_numerical_linearization(void)
{
    const FrenetState_t state = {0.1f, -0.05f, 5.0f, 0.02f, 0.1f};
    const ControlInput_t control = {0.08f, 1.0f};
    const FrenetState_t next = vehicle_model_predict_next_frenet_state(
        &state, &control, TIME_STEP_SECONDS, 0.04f);
    check_true(next.flong_vel > state.flong_vel,
               "target-speed command advances the command-stage speed state");
    check_true(isfinite(next.flat_error) && isfinite(next.fhead_error),
               "source-command Frenet stage remains finite");

    float matrix_a[NX_FRENET][NX_FRENET];
    float matrix_b[NX_FRENET][NU];
    vehicle_model_compute_frenet_linearization(
        &state, &control, TIME_STEP_SECONDS, 0.04f, 5.0f,
        matrix_a, matrix_b);
    for (int row = 0; row < NX_FRENET; ++row) {
        for (int column = 0; column < NX_FRENET; ++column)
            check_true(isfinite(matrix_a[row][column]), "finite stage A");
        for (int column = 0; column < NU; ++column)
            check_true(isfinite(matrix_b[row][column]), "finite stage B");
    }
    check_true(fabsf(matrix_b[4][0]) > 1.0e-4f,
               "steering influences the same stage map used by linearization");
}

int main(void)
{
    test_zero_state_is_finite();
    test_source_geometry_and_command_policy();
    test_geometric_turn_sign_and_target_speed_bound();
    test_shared_stage_and_numerical_linearization();
    if (failures != 0) {
        fprintf(stderr, "%d vehicle-model test(s) failed\n", failures);
        return 1;
    }
    puts("Vehicle-model tests passed");
    return 0;
}

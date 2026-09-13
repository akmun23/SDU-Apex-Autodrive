#include "vehicle_plant.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

static void check_true(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static void check_close(float actual, float expected, float tolerance,
                        const char *message)
{
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "FAIL: %s (actual=%g expected=%g)\n", message,
                (double)actual, (double)expected);
        failures++;
    }
}

static void test_zero_input_is_identity(void)
{
    const VehiclePlantParameters_t parameters = vehicle_plant_default_parameters();
    const VehiclePlantState_t state = {1.0f, -2.0f, 0.3f, 0.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f};
    const VehiclePlantInput_t input = {0.0f, 0.0f};
    VehiclePlantState_t next = {0};
    vehicle_plant_step(&state, &input, 0.025f, &parameters, &next);
    check_close(next.x_m, state.x_m, 1.0e-7f, "stationary x remains fixed");
    check_close(next.y_m, state.y_m, 1.0e-7f, "stationary y remains fixed");
    check_close(next.u_mps, 0.0f, 1.0e-7f, "stationary speed remains zero");
    check_true(isfinite(next.v_mps) && isfinite(next.r_radps),
               "stationary lateral state remains finite");
}

static void test_actuator_limits_and_forward_motion(void)
{
    const VehiclePlantParameters_t parameters = vehicle_plant_default_parameters();
    const VehiclePlantState_t state = {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f,
                                       0.0f, 2.0f};
    const VehiclePlantInput_t input = {2.0f, 1.0f};
    VehiclePlantState_t next = {0};
    vehicle_plant_step(&state, &input, 0.025f, &parameters, &next);
    check_true(next.x_m > state.x_m, "positive speed advances world position");
    check_close(next.steering_rad,
                parameters.steering_rate_radps * 0.025f,
                1.0e-6f, "steering rate limit is applied");
    check_true(next.u_mps > state.u_mps, "positive throttle increases speed");
    check_true(next.wheel_speed_mps >= 0.0f,
               "wheel state remains physically nonnegative");
}

int main(void)
{
    test_zero_input_is_identity();
    test_actuator_limits_and_forward_motion();
    if (failures != 0) {
        fprintf(stderr, "%d vehicle-plant test(s) failed\n", failures);
        return 1;
    }
    puts("Vehicle-plant tests passed");
    return 0;
}

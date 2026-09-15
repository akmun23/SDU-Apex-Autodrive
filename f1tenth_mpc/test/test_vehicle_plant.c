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

static void test_default_parameters_use_unity_structural_anchors(void)
{
    const VehiclePlantParameters_t parameters = vehicle_plant_default_parameters();
    check_close(parameters.mass_kg, 3.470f, 1.0e-6f,
                "plant uses Unity Rigidbody mass");
    check_close(parameters.lf_m, 0.174679914f, 1.0e-6f,
                "plant uses measured front contact geometry");
    check_close(parameters.lr_m, 0.155320086f, 1.0e-6f,
                "plant uses measured rear contact geometry");
    check_close(parameters.iz_kgm2, 0.0961908f, 1.0e-7f,
                "plant uses corrected Unity body-Y inertia");
    check_close(parameters.position_offset_from_velocity_point_x_m,
                -0.155320086f, 1.0e-7f,
                "plant uses measured pose/velocity lever arm");
    check_close(parameters.linear_damping_per_s, 0.273f, 1.0e-7f,
                "plant uses measured Unity linear damping");
    check_close(parameters.angular_damping_per_s, 0.1f, 1.0e-7f,
                "plant uses measured Unity angular damping");
    check_close(parameters.hard_brake_force_n, 18.5758070f, 1.0e-5f,
                "plant uses latest clean measured-damping hard-brake force");
    check_true(parameters.tire_model == VEHICLE_PLANT_TIRE_TANH,
               "plant default uses the current tanh tire candidate");
    check_true(parameters.wheel_dynamics_model ==
                 VEHICLE_PLANT_WHEEL_DYNAMICS_CONTINUOUS,
               "plant default uses the current continuous wheel candidate");
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

static void test_zero_throttle_is_active_braking(void)
{
    const VehiclePlantParameters_t parameters = vehicle_plant_default_parameters();
    const VehiclePlantState_t state = {0.0f, 0.0f, 0.0f, 8.0f, 0.0f, 0.0f,
                                       0.0f, 8.0f};
    const VehiclePlantInput_t input = {0.0f, 0.0f};
    VehiclePlantState_t next = {0};
    vehicle_plant_step(&state, &input, 0.025f, &parameters, &next);
    check_true(next.u_mps < state.u_mps,
               "zero throttle applies active braking to moving state");
}

static void test_pose_reference_lever_arm_only_affects_pose(void)
{
    VehiclePlantParameters_t parameters = vehicle_plant_default_parameters();
    const VehiclePlantState_t state = {0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 1.0f,
                                       0.0f, 4.0f};
    const VehiclePlantInput_t input = {0.0f, 1.0f};
    VehiclePlantState_t corrected = {0};
    VehiclePlantState_t same_point = {0};
    vehicle_plant_step(&state, &input, 0.025f, &parameters, &corrected);
    parameters.position_offset_from_velocity_point_x_m = 0.0f;
    vehicle_plant_step(&state, &input, 0.025f, &parameters, &same_point);
    check_true(corrected.y_m < same_point.y_m,
               "rear-position lever arm changes only propagated pose");
    check_close(corrected.u_mps, same_point.u_mps, 1.0e-6f,
                "pose lever arm does not change body longitudinal speed");
    check_close(corrected.v_mps, same_point.v_mps, 1.0e-6f,
                "pose lever arm does not change body lateral speed");
    check_close(corrected.r_radps, same_point.r_radps, 1.0e-6f,
                "pose lever arm does not change body yaw rate");
}

static void test_continuous_wheel_dynamics_uses_dt(void)
{
    VehiclePlantParameters_t parameters = vehicle_plant_default_parameters();
    parameters.wheel_dynamics_model = VEHICLE_PLANT_WHEEL_DYNAMICS_CONTINUOUS;
    parameters.wheel_coefficients[0] = 0.0f;
    parameters.wheel_coefficients[1] = 0.0f;
    parameters.wheel_coefficients[2] = 4.0f;
    parameters.wheel_coefficients[3] = 0.0f;
    parameters.wheel_coefficients[4] = 0.0f;
    parameters.wheel_coefficients[5] = 0.0f;
    const VehiclePlantState_t state = {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f,
                                       0.0f, 2.0f};
    const VehiclePlantInput_t input = {0.0f, 0.5f};
    VehiclePlantState_t short_next = {0};
    VehiclePlantState_t long_next = {0};
    vehicle_plant_step(&state, &input, 0.025f, &parameters, &short_next);
    vehicle_plant_step(&state, &input, 0.050f, &parameters, &long_next);
    check_true(long_next.wheel_speed_mps > short_next.wheel_speed_mps,
               "continuous wheel state responds to measured dt");
    check_close(long_next.wheel_speed_mps,
                2.0f * short_next.wheel_speed_mps - state.wheel_speed_mps,
                1.0e-6f, "continuous wheel state scales with dt");
}

int main(void)
{
    test_default_parameters_use_unity_structural_anchors();
    test_zero_input_is_identity();
    test_actuator_limits_and_forward_motion();
    test_zero_throttle_is_active_braking();
    test_pose_reference_lever_arm_only_affects_pose();
    test_continuous_wheel_dynamics_uses_dt();
    if (failures != 0) {
        fprintf(stderr, "%d vehicle-plant test(s) failed\n", failures);
        return 1;
    }
    puts("Vehicle-plant tests passed");
    return 0;
}

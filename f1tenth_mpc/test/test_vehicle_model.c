#include "vehicle_model.h"

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
        fprintf(stderr, "FAIL: %s (actual=%g expected=%g)\n",
                message, (double)actual, (double)expected);
        failures++;
    }
}

static void test_zero_speed_is_physical(void)
{
    const VehicleState_t state = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const ControlInput_t control = {0.0f, 0.0f};
    const VehicleState_t next = vehicle_model_predict_next_state(
        &state, &control, 0.025f);

    check_close(next.pos_x, 0.0f, 1e-7f, "zero-speed x remains fixed");
    check_close(next.pos_y, 0.0f, 1e-7f, "zero-speed y remains fixed");
    check_close(next.long_vel, 0.0f, 1e-7f,
                "zero-speed model does not invent forward motion");
    check_true(isfinite(next.lat_vel) && isfinite(next.yaw_rate),
               "zero-speed dynamic state remains finite");
}

static void test_forward_acceleration_and_steering_sign(void)
{
    const VehicleState_t state = {0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f};
    const ControlInput_t throttle = {0.0f, 1.0f};
    const VehicleState_t accelerated = vehicle_model_predict_next_state(
        &state, &throttle, 0.025f);
    check_true(accelerated.long_vel > state.long_vel,
               "positive longitudinal input increases speed");

    const ControlInput_t left = {0.10f, 0.0f};
    const ControlInput_t right = {-0.10f, 0.0f};
    const VehicleState_t left_next = vehicle_model_predict_next_state(
        &state, &left, 0.025f);
    const VehicleState_t right_next = vehicle_model_predict_next_state(
        &state, &right, 0.025f);
    check_true(left_next.yaw_rate > 0.0f && right_next.yaw_rate < 0.0f,
               "steering produces opposite yaw-rate signs");
}

static void test_low_speed_linearization_is_finite(void)
{
    const FrenetState_t state = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const ControlInput_t control = {0.0f, 0.0f};
    float matrix_a[NX_FRENET][NX_FRENET];
    float matrix_b[NX_FRENET][NU];
    vehicle_model_compute_frenet_linearization(
        &state, &control, 0.025f, 0.0f, 0.0f, matrix_a, matrix_b);

    for (int row = 0; row < NX_FRENET; ++row) {
        for (int column = 0; column < NX_FRENET; ++column) {
            check_true(isfinite(matrix_a[row][column]),
                       "low-speed A matrix remains finite");
        }
        for (int column = 0; column < NU; ++column) {
            check_true(isfinite(matrix_b[row][column]),
                       "low-speed B matrix remains finite");
        }
    }
}

static void test_rollouts_share_body_force_path(void)
{
    const VehicleState_t state = {1.0f, -2.0f, 0.3f, 3.0f, 0.15f, 0.2f};
    const ControlInput_t control = {0.12f, 0.8f};
    const float dt = 0.01f;
    VehicleBodyDynamics_t dynamics = {0};
    vehicle_model_compute_body_dynamics(&state, &control, &dynamics);

    const VehicleState_t next = vehicle_model_predict_next_state(
        &state, &control, dt);
    check_close(next.long_vel,
                state.long_vel + dt * dynamics.long_acceleration,
                1e-5f,
                "global rollout uses shared longitudinal dynamics");
    check_close(next.lat_vel,
                state.lat_vel + dt * dynamics.lateral_acceleration,
                1e-5f,
                "global rollout uses shared lateral dynamics");
    check_close(next.yaw_rate,
                state.yaw_rate + dt * dynamics.yaw_acceleration,
                1e-5f,
                "global rollout uses shared yaw dynamics");

    const FrenetState_t frenet = {0.1f, -0.05f, state.long_vel,
                                  state.lat_vel, state.yaw_rate};
    const FrenetState_t next_frenet =
        vehicle_model_predict_next_frenet_state(&frenet, &control, dt, 0.02f);
    check_close(next_frenet.flong_vel, next.long_vel, 1e-5f,
                "Frenet rollout shares longitudinal dynamics");
    check_close(next_frenet.flat_vel, next.lat_vel, 1e-5f,
                "Frenet rollout shares lateral dynamics");
    check_close(next_frenet.fyaw_rate, next.yaw_rate, 1e-5f,
                "Frenet rollout shares yaw dynamics");
}

static void test_runtime_parameters_are_active(void)
{
    const VehicleParameters_t defaults = vehicle_model_default_parameters();
    VehicleParameters_t tuned = defaults;
    tuned.max_velocity = 3.05f;
    check_true(vehicle_model_set_parameters(&tuned) == 1,
               "valid runtime model parameters are accepted");

    const VehicleState_t state = {0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f};
    const ControlInput_t control = {0.0f, 4.0f};
    const VehicleState_t next = vehicle_model_predict_next_state(
        &state, &control, 0.5f);
    check_true(next.long_vel <= tuned.max_velocity + 1e-5f,
               "runtime maximum speed is applied by rollout");

    VehicleParameters_t invalid = tuned;
    invalid.vehicle_mass = 0.0f;
    check_true(vehicle_model_set_parameters(&invalid) == 0,
               "invalid runtime model parameters are rejected");
    check_true(vehicle_model_set_parameters(&defaults) == 1,
               "default model parameters can be restored");
}

static void test_unity_structural_anchors(void)
{
    const VehicleParameters_t parameters = vehicle_model_default_parameters();
    check_close(parameters.vehicle_mass, 3.470f, 1.0e-6f,
                "Unity Rigidbody mass is used");
    check_close(parameters.yaw_moment_of_inertia, 0.0961908f, 1.0e-7f,
                "MPC uses corrected Unity body-Y inertia");
    check_close(parameters.wheelbase_meters, 0.330000f, 1.0e-5f,
                "contact wheelbase is used");
    check_close(parameters.distance_cg_to_front_axle, 0.174679914f, 1.0e-6f,
                "front COM contact distance is used");
    check_close(parameters.distance_cg_to_rear_axle, 0.155320086f, 1.0e-6f,
                "rear COM contact distance is used");
    check_close(parameters.max_steering_angle, 0.5235987756f, 1.0e-6f,
                "Unity steering limit is used");
    check_close(parameters.height_cg_to_ground, 0.0f, 1.0e-7f,
                "unvalidated Unity-local COM coordinate is not used as load height");

    float front_load = 0.0f;
    float rear_load = 0.0f;
    vehicle_model_compute_normal_loads(10.0f, &front_load, &rear_load);
    const float static_total_load = parameters.vehicle_mass * 9.82f;
    check_close(front_load + rear_load, static_total_load, 1.0e-4f,
                "unknown CG height preserves total static normal load");
}

int main(void)
{
    test_zero_speed_is_physical();
    test_forward_acceleration_and_steering_sign();
    test_low_speed_linearization_is_finite();
    test_rollouts_share_body_force_path();
    test_runtime_parameters_are_active();
    test_unity_structural_anchors();
    if (failures != 0) {
        fprintf(stderr, "%d vehicle-model test(s) failed\n", failures);
        return 1;
    }
    puts("Vehicle-model tests passed");
    return 0;
}

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

static float steady_target_speed(float speed)
{
    return speed - (MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 +
                    MPC_LONGITUDINAL_SPEED_COEFF_PER_S * speed) /
                       MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S;
}

static float expected_yaw_steady(float speed, float steering)
{
    return speed * tanf(steering) * MPC_YAW_RATE_STEERING_GAIN_PER_M;
}

static void test_source_parameters(void)
{
    const VehicleParameters_t parameters = vehicle_model_default_parameters();
    check_close(parameters.steering_wheelbase_m, 0.324f, 1.0e-6f,
                "uses Unity steering wheelbase");
    check_close(parameters.max_steering_angle, SOURCE_MAX_STEERING_RAD,
                1.0e-6f, "uses Unity steering limit");
    check_close(parameters.steering_rate_radps, SOURCE_STEERING_RATE_RADPS,
                1.0e-6f, "uses Unity steering rate");
    check_close(parameters.maximum_command_speed_mps, 16.0f, 1.0e-6f,
                "uses project command ceiling");
}

static void test_curvature_response_parameter(void)
{
    const MpcYawRateModelParameters_t baseline =
        vehicle_model_default_yaw_rate_parameters();
    const MpcYawRateModelParameters_t active = {
        .response_time_constant_s = baseline.response_time_constant_s,
        .steering_gain_per_m = baseline.steering_gain_per_m,
        .curvature_gain_reduction_per_m = 1.3f,
        .curvature_gain_start_per_m = 0.20f,
        .curvature_gain_end_per_m = 0.40f,
    };
    check_true(vehicle_model_set_yaw_rate_parameters(&active),
               "accepts finite curvature response correction");
    check_close(vehicle_model_yaw_rate_gain_for_curvature(0.10f),
                baseline.steering_gain_per_m, 1.0e-6f,
                "does not alter low-curvature response");
    check_close(vehicle_model_yaw_rate_gain_for_curvature(0.30f),
                baseline.steering_gain_per_m - 0.13f, 1.0e-6f,
                "interpolates curvature response correction");
    check_close(vehicle_model_yaw_rate_gain_for_curvature(0.70f),
                baseline.steering_gain_per_m - 0.26f, 1.0e-6f,
                "saturates curvature response correction");
    check_true(vehicle_model_set_yaw_rate_parameters(&baseline),
               "restores nominal curvature response");
}

static void test_authoritative_stage(void)
{
    const float dt = TIME_STEP_SECONDS;
    const float speed = 4.0f;
    const MpcModelState_t state = {
        .e_y = 0.0f, .e_psi = 0.0f, .u = speed, .v = 0.0f, .r = 0.2f,
        .target_speed = steady_target_speed(speed),
        .steering_command = 0.0f, .actual_steering_angle = 0.0f};
    const MpcModelControl_t control = {
        .steering_rate = 1.0f, .target_speed_rate = 0.0f};
    const MpcStageResult_t stage = mpc_vehicle_model_step(
        &state, &control, dt, 0.0f);

    check_true(stage.valid, "authoritative ten-state stage is valid");
    check_close(stage.next.steering_command, dt, 1.0e-7f,
                "steering command integrates over the prediction step");
    check_close(stage.next.v, state.v, 1.0e-7f,
                "lateral velocity is held over one prediction step");

    const float alpha = expf(-dt /
        MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS);
    const float u_mid = 0.5f * (state.u + stage.next.u);
    const float yaw_steady = expected_yaw_steady(
        u_mid, stage.next.actual_steering_angle);
    check_close(stage.next.r,
        alpha * state.r + (1.0f - alpha) * yaw_steady, 1.0e-6f,
        "same-step yaw response uses the delayed physical steering angle");
}

static void test_measured_yaw_response_branch(void)
{
    const float dt = TIME_STEP_SECONDS;
    const MpcModelState_t state = {
        .e_y = 0.0f, .e_psi = 0.0f, .u = 4.0f, .v = 0.0f, .r = 0.0f,
        .target_speed = 4.0f,
        .steering_command = SOURCE_MAX_STEERING_RAD,
        .delayed_steering_command_1 = SOURCE_MAX_STEERING_RAD,
        .delayed_steering_command_2 = SOURCE_MAX_STEERING_RAD,
        .actual_steering_angle = SOURCE_MAX_STEERING_RAD};
    const MpcStageResult_t stage = mpc_vehicle_model_step(
        &state, &(MpcModelControl_t){0.0f, 0.0f}, dt, 0.0f);
    const float alpha = expf(-dt /
        MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS);
    const float u_mid = 0.5f * (state.u + stage.next.u);
    const float yaw_steady = expected_yaw_steady(
        u_mid, stage.next.actual_steering_angle);
    const float expected = (1.0f - alpha) * yaw_steady;
    check_close(stage.next.r, expected, 1.0e-6f,
                "measured yaw response keeps the identified command branch");
}

static void test_speed_and_branch_limits(void)
{
    MpcModelState_t state = {
        .e_y = 0.0f, .e_psi = 0.0f, .u = 4.0f, .v = 0.0f, .r = 0.0f,
        .target_speed = 4.0f, .steering_command = 0.52f,
        .actual_steering_angle = 0.52f};
    const MpcStageResult_t clipped = mpc_vehicle_model_step(
        &state, &(MpcModelControl_t){3.2f, 3.0f}, TIME_STEP_SECONDS, 0.0f);
    check_close(clipped.next.steering_command, SOURCE_MAX_STEERING_RAD,
                1.0e-6f, "steering command respects source limit");
    check_close(clipped.next.target_speed, 4.075f, 1.0e-6f,
                "target speed integrates the bounded rate command");
    check_true((clipped.branch_flags & MPC_STAGE_CLIPPED_STEERING_COMMAND) != 0,
               "steering saturation reports its active branch");

    state.target_speed = 16.0f;
    state.steering_command = 0.52f;
    const MpcStageResult_t ceiling = mpc_vehicle_model_step(
        &state, &(MpcModelControl_t){3.2f, 3.0f}, TIME_STEP_SECONDS, 0.0f);
    check_close(ceiling.next.target_speed, 16.0f, 1.0e-6f,
                "target speed respects the project ceiling");
    check_true((ceiling.branch_flags & MPC_STAGE_CLIPPED_TARGET_SPEED) != 0,
               "target-speed ceiling reports its active branch");
}

static void test_braking_and_turn_invariance(void)
{
    MpcModelState_t braking_state = {
        .e_y = 0.0f, .e_psi = 0.0f, .u = 6.0f, .v = 0.0f, .r = 0.0f,
        .target_speed = 0.0f, .steering_command = 0.0f,
        .actual_steering_angle = 0.0f};
    const MpcStageResult_t braking = mpc_vehicle_model_step(
        &braking_state, &(MpcModelControl_t){0.0f, 0.0f},
        TIME_STEP_SECONDS, 0.0f);
    const float brake_limit = MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2 +
        MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV * braking_state.u;
    check_close(braking.body_accel_mps2, -brake_limit, 1.0e-5f,
                "braking uses the measured speed-dependent envelope");

    const float curvature = 0.2f;
    const float speed = 2.0f;
    const float steering = atanf(curvature /
        MPC_YAW_RATE_STEERING_GAIN_PER_M);
    const MpcModelState_t turn = {
        .e_y = 0.0f, .e_psi = 0.0f, .u = speed, .v = 0.0f,
        .r = curvature * speed, .target_speed = steady_target_speed(speed),
        .steering_command = steering,
        .delayed_steering_command_1 = steering,
        .delayed_steering_command_2 = steering,
        .actual_steering_angle = steering};
    const MpcStageResult_t turn_stage = mpc_vehicle_model_step(
        &turn, &(MpcModelControl_t){0.0f, 0.0f}, TIME_STEP_SECONDS, curvature);
    check_true(turn_stage.valid, "ideal turn remains inside Frenet domain");
    check_close(turn_stage.next.e_y, 0.0f, 1.0e-5f,
                "ideal turn preserves zero cross-track error");
    check_close(turn_stage.next.e_psi, 0.0f, 1.0e-5f,
                "ideal turn preserves zero heading error");
    check_close(turn_stage.delta_s_m, TIME_STEP_SECONDS * speed, 1.0e-5f,
                "ideal turn progress matches speed times dt");

    MpcModelState_t invalid = turn;
    invalid.e_y = 0.096f;
    const MpcStageResult_t invalid_stage = mpc_vehicle_model_step(
        &invalid, &(MpcModelControl_t){0.0f, 0.0f}, TIME_STEP_SECONDS, 10.0f);
    check_true(!invalid_stage.valid,
               "near-singular Frenet denominator is rejected");
}

int main(void)
{
    test_source_parameters();
    test_curvature_response_parameter();
    test_authoritative_stage();
    test_measured_yaw_response_branch();
    test_speed_and_branch_limits();
    test_braking_and_turn_invariance();
    if (failures != 0) {
        fprintf(stderr, "%d vehicle-model test(s) failed\n", failures);
        return 1;
    }
    puts("Vehicle-model tests passed");
    return 0;
}

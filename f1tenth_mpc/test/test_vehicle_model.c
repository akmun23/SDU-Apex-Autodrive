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
    check_close(MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS, 0.087735f,
                1.0e-6f, "uses held-out AutoDRIVE yaw response time constant");
    check_close(MPC_YAW_RATE_STEERING_GAIN_PER_M, 3.011897f,
                1.0e-6f, "uses held-out AutoDRIVE yaw steering gain");
}

static void test_identified_yaw_response_and_target_speed_bound(void)
{
    const VehicleState_t state = {
        .pos_x = 0.0f, .pos_y = 0.0f, .heading = 0.0f,
        .long_vel = 4.0f, .lat_vel = 0.0f, .yaw_rate = 0.0f,
        .target_speed_mps = 4.0f -
            (MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 +
             MPC_LONGITUDINAL_SPEED_COEFF_PER_S * 4.0f) /
                MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S};
    const ControlInput_t left = {0.10f, 0.0f};
    const ControlInput_t right = {-0.10f, 0.0f};
    const VehicleState_t left_next = vehicle_model_predict_next_state(
        &state, &left, TIME_STEP_SECONDS);
    const VehicleState_t right_next = vehicle_model_predict_next_state(
        &state, &right, TIME_STEP_SECONDS);
    check_true(left_next.yaw_rate > 0.0f && right_next.yaw_rate < 0.0f,
               "source steering signs produce opposite turns");
    const float yaw_target = state.long_vel * tanf(left.steer_ang) *
        MPC_YAW_RATE_STEERING_GAIN_PER_M;
    const float yaw_retention = expf(
        -TIME_STEP_SECONDS / MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS);
    check_close(left_next.yaw_rate,
                (1.0f - yaw_retention) * yaw_target,
                1.0e-6f,
                "yaw rate follows the identified first-order response");
    check_true(left_next.yaw_rate <
                   state.long_vel * tanf(left.steer_ang) /
                       SOURCE_STEERING_WHEELBASE_M,
               "yaw response does not jump instantly to Ackermann steady state");

    VehicleState_t repeated = state;
    for (int step = 0; step < 20; ++step) {
        repeated = vehicle_model_predict_next_state(
            &repeated, &left, TIME_STEP_SECONDS);
    }
    const float expected_after_20 = yaw_target *
        (1.0f - expf(-20.0f * TIME_STEP_SECONDS /
                     MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS));
    check_close(repeated.yaw_rate, expected_after_20, 1.0e-5f,
                "yaw rate converges to the measured steady response");

    const ControlInput_t excessive = {4.0f, 100.0f};
    const ControlInput_t bounded = vehicle_model_saturate_control(&excessive);
    check_true(bounded.steer_ang <= 0.5235988f &&
               bounded.target_speed_rate <= 3.0f,
               "MPC output is bounded by source and command policy limits");
}

static void test_shared_stage_and_numerical_linearization(void)
{
    const FrenetState_t state = {
        .flat_error = 0.1f, .fhead_error = -0.05f,
        .flong_vel = 5.0f, .flat_vel = 0.02f, .fyaw_rate = 0.1f,
        .ftarget_speed_mps = 6.0f};
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
    check_true(matrix_b[IDX_TARGET_SPEED_STATE][1] > 0.0f,
               "speed slew advances the carried actuator target state");
}

static void test_perfect_turn_frenet_invariance(void)
{
    const float curvature = 0.96754f;
    const float speed = 5.749041f;
    const float steady_target = speed -
        (MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 +
         MPC_LONGITUDINAL_SPEED_COEFF_PER_S * speed) /
            MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S;
    const FrenetState_t state = {
        .flat_error = 0.0f,
        .fhead_error = 0.0f,
        .flong_vel = speed,
        .flat_vel = 0.0f,
        .fyaw_rate = curvature * speed,
        .ftarget_speed_mps = steady_target,
    };
    const ControlInput_t command = {
        .steer_ang = atanf(curvature / MPC_YAW_RATE_STEERING_GAIN_PER_M),
        .target_speed_rate = 0.0f,
    };
    const FrenetState_t next = vehicle_model_predict_next_frenet_state(
        &state, &command, TIME_STEP_SECONDS, curvature);

    check_close(next.flat_error, 0.0f, 1.0e-5f,
                "ideal constant-curvature tracking creates no lateral drift");
    check_close(next.fhead_error, 0.0f, 1.0e-5f,
                "ideal constant-curvature tracking preserves zero heading error");
}

static void test_authoritative_seven_state_stage(void)
{
    const float dt = TIME_STEP_SECONDS;
    const float speed = 4.0f;
    const float target = speed -
        (MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 +
         MPC_LONGITUDINAL_SPEED_COEFF_PER_S * speed) /
            MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S;
    const MpcModelState_t state = {
        .e_y = 0.0f,
        .e_psi = 0.0f,
        .u = speed,
        .v = 0.0f,
        .r = 0.2f,
        .target_speed = target,
        .steering_command = 0.0f,
    };
    const MpcModelControl_t control = {
        .steering_rate = 1.0f,
        .target_speed_rate = 0.0f,
    };
    const MpcStageResult_t stage = mpc_vehicle_model_step(
        &state, &control, dt, 0.0f);
    check_true(stage.valid, "authoritative seven-state stage is valid");
    check_close(stage.next.steering_command, dt, 1.0e-7f,
                "steering command integrates q_delta over the prediction dt");
    check_close(stage.next.v, state.v, 1.0e-7f,
                "unidentified lateral velocity remains held constant");

    const float alpha = expf(-dt / MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS);
    const float u_mid = 0.5f * (state.u + stage.next.u);
    const float yaw_steady = MPC_YAW_RATE_STEERING_GAIN_PER_M * u_mid *
        tanf(stage.next.steering_command);
    check_close(stage.next.r,
        alpha * state.r + (1.0f - alpha) * yaw_steady, 1.0e-6f,
        "same-interval yaw response uses the newly published steering command");

    MpcModelState_t speed_state = state;
    speed_state.target_speed = 4.0f;
    const MpcModelControl_t speed_control = {0.0f, 1.0f};
    const MpcStageResult_t speed_stage = mpc_vehicle_model_step(
        &speed_state, &speed_control, dt, 0.0f);
    check_close(speed_stage.next.target_speed, 4.025f, 1.0e-7f,
                "target-speed state integrates q_v over the prediction dt");

    MpcModelState_t clipped_state = state;
    clipped_state.steering_command = 0.52f;
    clipped_state.target_speed = 15.99f;
    const MpcModelControl_t clipped_control = {3.2f, 3.0f};
    const MpcStageResult_t clipped = mpc_vehicle_model_step(
        &clipped_state, &clipped_control, dt, 0.0f);
    check_close(clipped.next.steering_command, SOURCE_MAX_STEERING_RAD,
                1.0e-6f, "steering command respects the source steering limit");
    check_close(clipped.next.target_speed, 16.0f, 1.0e-6f,
                "target speed respects the project command ceiling");
    check_true((clipped.branch_flags & MPC_STAGE_CLIPPED_STEERING_COMMAND) &&
               (clipped.branch_flags & MPC_STAGE_CLIPPED_TARGET_SPEED),
        "stage reports nonsmooth steering and target-speed saturation branches");

    MpcModelState_t brake_state = state;
    brake_state.u = 6.0f;
    brake_state.target_speed = 0.0f;
    const MpcModelControl_t hold_brake = {0.0f, 0.0f};
    const MpcStageResult_t braking = mpc_vehicle_model_step(
        &brake_state, &hold_brake, dt, 0.0f);
    const float brake_limit = MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2 +
        MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV * brake_state.u;
    check_close(braking.body_accel_mps2, -brake_limit, 1.0e-5f,
                "full-brake stage uses the measured speed-dependent envelope");
    check_close(braking.next.u, brake_state.u - dt * brake_limit, 1.0e-5f,
                "braking response integrates the accepted longitudinal model");

    const float curvature = 0.8f;
    const float turn_delta = atanf(
        curvature / MPC_YAW_RATE_STEERING_GAIN_PER_M);
    MpcModelState_t turn = {
        .e_y = 0.0f,
        .e_psi = 0.0f,
        .u = speed,
        .v = 0.0f,
        .r = curvature * speed,
        .target_speed = target,
        .steering_command = turn_delta,
    };
    const MpcModelControl_t hold = {0.0f, 0.0f};
    const MpcStageResult_t turn_stage = mpc_vehicle_model_step(
        &turn, &hold, dt, curvature);
    check_true(turn_stage.valid,
               "ideal constant-curvature stage remains inside Frenet domain");
    check_close(turn_stage.next.e_y, 0.0f, 1.0e-5f,
                "perfect turn preserves zero cross-track error");
    check_close(turn_stage.next.e_psi, 0.0f, 1.0e-5f,
                "perfect turn preserves zero heading error");
    check_close(turn_stage.delta_s_m, dt * speed, 1.0e-5f,
                "stage progress agrees with speed times dt at equilibrium");

    turn.e_y = 0.096f;
    const MpcStageResult_t invalid_frenet = mpc_vehicle_model_step(
        &turn, &hold, dt, 10.0f);
    check_true(!invalid_frenet.valid,
               "Frenet denominator near zero is reported as invalid");
}

static void test_identified_longitudinal_response_and_braking_envelope(void)
{
    FrenetState_t state = {
        .flong_vel = 6.0f,
        .ftarget_speed_mps = 7.0f,
    };
    const ControlInput_t hold_target = {0.0f, 0.0f};
    const FrenetState_t accelerated = vehicle_model_predict_next_frenet_state(
        &state, &hold_target, TIME_STEP_SECONDS, 0.0f);
    check_true(accelerated.flong_vel > state.flong_vel,
               "target-speed error produces fitted forward response");

    state.ftarget_speed_mps = 0.0f;
    const FrenetState_t braked = vehicle_model_predict_next_frenet_state(
        &state, &hold_target, TIME_STEP_SECONDS, 0.0f);
    const float expected_brake = MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2 +
        MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV * state.flong_vel;
    check_close(state.flong_vel - braked.flong_vel,
                expected_brake * TIME_STEP_SECONDS,
                1.0e-5f,
                "full-brake response follows the speed-dependent measured envelope");

    state.flong_vel = 4.0f;
    state.ftarget_speed_mps = 4.0f;
    const ControlInput_t ramp = {0.0f, 1.0f};
    const FrenetState_t ramped = vehicle_model_predict_next_frenet_state(
        &state, &ramp, TIME_STEP_SECONDS, 0.0f);
    check_close(ramped.ftarget_speed_mps, 4.025f, 1.0e-6f,
                "speed-rate input integrates the command state");
}

int main(void)
{
    test_zero_state_is_finite();
    test_source_geometry_and_command_policy();
    test_identified_yaw_response_and_target_speed_bound();
    test_shared_stage_and_numerical_linearization();
    test_perfect_turn_frenet_invariance();
    test_authoritative_seven_state_stage();
    test_identified_longitudinal_response_and_braking_envelope();
    if (failures != 0) {
        fprintf(stderr, "%d vehicle-model test(s) failed\n", failures);
        return 1;
    }
    puts("Vehicle-model tests passed");
    return 0;
}

#include "mpc_linearization.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>

enum { NX = 7, CONTROL_COUNT = 2 };

static int failures = 0;
static float max_directional_relative_error = 0.0f;
static float heading_wrap_jacobian_error = 0.0f;
static float max_analytic_fd_directional_relative_error = 0.0f;
static uint32_t random_state = 0x61c88647u;

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

static void state_to_array(const MpcModelState_t *state, float x[NX])
{
    x[0] = state->e_y;
    x[1] = state->e_psi;
    x[2] = state->u;
    x[3] = state->v;
    x[4] = state->r;
    x[5] = state->target_speed;
    x[6] = state->steering_command;
}

static void output_to_array(const MpcStageResult_t *stage, float y[NX])
{
    state_to_array(&stage->next, y);
}

static float wrapped_difference(float a, float b)
{
    return remainderf(a - b, 6.2831853071795864769f);
}

static float uniform_random(float lower, float upper)
{
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    const float unit = (float)(random_state & 0x00ffffffu) /
        (float)0x01000000u;
    return lower + (upper - lower) * unit;
}

static void check_linearization_base_reconstruction(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float curvature,
    const char *name)
{
    MpcStageLinearization_t linearization;
    const int ok = mpc_model_linearize(
        state, control, 0.025f, curvature, &linearization);
    check_true(ok && linearization.valid, name);
    if (!ok) return;

    float x[NX];
    float u[CONTROL_COUNT] = {
        control->steering_rate, control->target_speed_rate};
    float actual[NX];
    float predicted[NX];
    state_to_array(state, x);
    const MpcStageResult_t stage = mpc_vehicle_model_step(
        state, control, 0.025f, curvature);
    output_to_array(&stage, actual);
    for (int row = 0; row < NX; ++row) {
        predicted[row] = linearization.d[row];
        for (int col = 0; col < NX; ++col)
            predicted[row] += linearization.A[row][col] * x[col];
        for (int col = 0; col < CONTROL_COUNT; ++col)
            predicted[row] += linearization.B[row][col] * u[col];
        const float error = row == 1
            ? wrapped_difference(predicted[row], actual[row])
            : predicted[row] - actual[row];
        check_close(error, 0.0f, 2.0e-5f,
                    "affine stage reproduces nonlinear nominal point");
    }
}

static void check_directional_derivative(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float curvature,
    const char *label)
{
    MpcStageLinearization_t linearization;
    check_true(mpc_model_linearize(
                   state, control, 0.025f, curvature, &linearization),
               label);
    if (!linearization.valid) return;

    const float dx[NX] = {0.21f, 0.07f, 0.53f, -0.13f,
                          0.62f, 0.31f, 0.018f};
    const float du[NU] = {0.34f, -0.42f};
    float x_plus[NX];
    float x_minus[NX];
    float x0[NX];
    state_to_array(state, x0);
    const float h = 1.0e-3f;
    for (int i = 0; i < NX; ++i) {
        x_plus[i] = x0[i] + h * dx[i];
        x_minus[i] = x0[i] - h * dx[i];
    }
    const MpcModelState_t state_plus = {
        .e_y = x_plus[0], .e_psi = x_plus[1], .u = x_plus[2],
        .v = x_plus[3], .r = x_plus[4], .target_speed = x_plus[5],
        .steering_command = x_plus[6]};
    const MpcModelState_t state_minus = {
        .e_y = x_minus[0], .e_psi = x_minus[1], .u = x_minus[2],
        .v = x_minus[3], .r = x_minus[4], .target_speed = x_minus[5],
        .steering_command = x_minus[6]};
    const MpcModelControl_t control_plus = {
        .steering_rate = control->steering_rate + h * du[0],
        .target_speed_rate = control->target_speed_rate + h * du[1]};
    const MpcModelControl_t control_minus = {
        .steering_rate = control->steering_rate - h * du[0],
        .target_speed_rate = control->target_speed_rate - h * du[1]};
    const MpcStageResult_t plus = mpc_vehicle_model_step(
        &state_plus, &control_plus, 0.025f, curvature);
    const MpcStageResult_t minus = mpc_vehicle_model_step(
        &state_minus, &control_minus, 0.025f, curvature);
    check_true(plus.valid && minus.valid, "direction probes remain valid");
    if (!plus.valid || !minus.valid) return;

    float y_plus[NX];
    float y_minus[NX];
    output_to_array(&plus, y_plus);
    output_to_array(&minus, y_minus);
    float error_sq = 0.0f;
    float reference_sq = 0.0f;
    for (int row = 0; row < NX; ++row) {
        const float observed = (row == 1
            ? wrapped_difference(y_plus[row], y_minus[row])
            : y_plus[row] - y_minus[row]) / (2.0f * h);
        float predicted = 0.0f;
        for (int col = 0; col < NX; ++col)
            predicted += linearization.A[row][col] * dx[col];
        for (int col = 0; col < CONTROL_COUNT; ++col)
            predicted += linearization.B[row][col] * du[col];
        const float error = predicted - observed;
        error_sq += error * error;
        reference_sq += observed * observed;
    }
    const float relative_error = sqrtf(error_sq) /
        fmaxf(sqrtf(reference_sq), 1.0e-5f);
    if (relative_error > max_directional_relative_error)
        max_directional_relative_error = relative_error;
    if (!(relative_error < 0.01f)) {
        fprintf(stderr, "FAIL: %s relative error %.6g\n", label,
                relative_error);
        ++failures;
    }
}

static void test_linearizations_away_from_switches(void)
{
    const MpcModelState_t straight = {
        .e_y = 0.08f, .e_psi = 0.03f, .u = 5.0f, .v = 0.05f,
        .r = 0.25f, .target_speed = 5.2f, .steering_command = 0.03f};
    const MpcModelControl_t accelerate = {
        .steering_rate = 0.10f, .target_speed_rate = 0.50f};
    check_linearization_base_reconstruction(
        &straight, &accelerate, 0.0f, "straight stage linearizes");
    check_directional_derivative(
        &straight, &accelerate, 0.0f,
        "straight-stage directional derivative below one percent");

    const MpcModelState_t corner = {
        .e_y = 0.10f, .e_psi = 0.025f, .u = 7.0f, .v = 0.04f,
        .r = 1.20f, .target_speed = 7.1f, .steering_command = 0.06f};
    const MpcModelControl_t turn = {
        .steering_rate = 0.05f, .target_speed_rate = 0.0f};
    check_linearization_base_reconstruction(
        &corner, &turn, 0.15f, "corner stage linearizes");
    check_directional_derivative(
        &corner, &turn, 0.15f,
        "corner-stage directional derivative below one percent");

    const MpcModelState_t braking = {
        .e_y = -0.04f, .e_psi = -0.02f, .u = 8.0f, .v = -0.03f,
        .r = -0.45f, .target_speed = 2.0f, .steering_command = -0.025f};
    const MpcModelControl_t brake = {
        .steering_rate = -0.08f, .target_speed_rate = -0.5f};
    check_linearization_base_reconstruction(
        &braking, &brake, -0.08f, "braking stage linearizes");
    check_directional_derivative(
        &braking, &brake, -0.08f,
        "braking-stage directional derivative below one percent");
}

static void test_saturation_boundary_is_reported(void)
{
    const float dt = 0.025f;
    const float max_delta = SOURCE_MAX_STEERING_RAD;
    MpcModelState_t state = {
        .e_y = 0.0f, .e_psi = 0.0f, .u = 4.0f, .v = 0.0f, .r = 0.0f,
        .target_speed = 4.0f, .steering_command = max_delta - 5.0e-6f};
    const MpcModelControl_t control = {0.0f, 0.0f};
    MpcStageLinearization_t lin;
    check_true(mpc_model_linearize(&state, &control, dt, 0.0f, &lin),
               "linearization remains valid at steering saturation boundary");
    check_true((lin.nonsmooth_column_mask & (1u << 6)) != 0,
               "state perturbation identifies steering saturation boundary");
    check_true(isfinite(lin.A[6][6]),
               "one-sided steering-boundary derivative is finite");

    state.steering_command = 0.0f;
    state.target_speed = 16.0f - 5.0e-4f;
    check_true(mpc_model_linearize(&state, &control, dt, 0.0f, &lin),
               "linearization remains valid at target-speed boundary");
    check_true((lin.nonsmooth_column_mask & (1u << 5)) != 0,
               "state perturbation identifies target-speed saturation boundary");
    check_true(isfinite(lin.A[5][5]),
               "one-sided target-speed-boundary derivative is finite");
}

static void test_invalid_frenet_perturbation_uses_valid_side(void)
{
    const MpcModelState_t state = {
        .e_y = 0.95f, .e_psi = 0.0f, .u = 0.0f, .v = 0.0f, .r = 0.0f,
        .target_speed = 0.0f, .steering_command = 0.0f};
    const MpcModelControl_t hold = {0.0f, 0.0f};
    MpcStageLinearization_t lin;
    check_true(mpc_model_linearize(&state, &hold, 0.025f, 1.0f, &lin),
               "valid Frenet edge linearizes with one invalid perturbation");
    check_true((lin.nonsmooth_column_mask & 1u) != 0,
               "Frenet-domain edge marks the affected lateral-state column");
    check_true(isfinite(lin.A[0][0]),
               "Frenet-domain edge uses the valid one-sided derivative");
}

static void test_heading_wrap_and_mirror_symmetry(void)
{
    MpcModelState_t wrap = {
        .e_y = 0.0f, .e_psi = 3.14159265f - 2.0e-6f,
        .u = 1.0f, .v = 0.0f, .r = 0.0f, .target_speed = 1.0f,
        .steering_command = 0.0f};
    const MpcModelControl_t hold = {0.0f, 0.0f};
    MpcStageLinearization_t lin;
    check_true(mpc_model_linearize(&wrap, &hold, 0.025f, 0.0f, &lin),
               "heading-wrap stage linearizes");
    heading_wrap_jacobian_error = fabsf(lin.A[1][1] - 1.0f);
    check_close(lin.A[1][1], 1.0f, 2.0e-4f,
                "heading Jacobian uses wrapped difference at plus/minus pi");

    const MpcModelState_t left = {
        .e_y = 0.12f, .e_psi = 0.04f, .u = 5.0f, .v = 0.08f,
        .r = 0.5f, .target_speed = 5.1f, .steering_command = 0.035f};
    const MpcModelState_t right = {
        .e_y = -left.e_y, .e_psi = -left.e_psi, .u = left.u,
        .v = -left.v, .r = -left.r, .target_speed = left.target_speed,
        .steering_command = -left.steering_command};
    const MpcModelControl_t left_control = {0.1f, 0.0f};
    const MpcModelControl_t right_control = {-0.1f, 0.0f};
    const MpcStageResult_t left_next = mpc_vehicle_model_step(
        &left, &left_control, 0.025f, 0.12f);
    const MpcStageResult_t right_next = mpc_vehicle_model_step(
        &right, &right_control, 0.025f, -0.12f);
    check_true(left_next.valid && right_next.valid,
               "mirrored turn stages remain valid");
    check_close(left_next.next.e_y, -right_next.next.e_y, 2.0e-6f,
                "mirrored turn preserves lateral sign symmetry");
    check_close(left_next.next.e_psi, -right_next.next.e_psi, 2.0e-6f,
                "mirrored turn preserves heading sign symmetry");
    check_close(left_next.next.r, -right_next.next.r, 2.0e-6f,
                "mirrored turn preserves yaw-rate sign symmetry");
    check_close(left_next.next.u, right_next.next.u, 2.0e-6f,
                "mirrored turn preserves longitudinal response");
    check_close(left_next.delta_s_m, right_next.delta_s_m, 2.0e-6f,
                "mirrored turn preserves along-track progress");
}

static void test_analytic_jacobian_against_fd_oracle(void)
{
    const int sample_count = 10000;
    for (int sample = 0; sample < sample_count; ++sample) {
        MpcModelState_t state = {
            .e_y = uniform_random(-0.30f, 0.30f),
            .e_psi = uniform_random(-0.30f, 0.30f),
            .u = uniform_random(2.0f, 10.0f),
            .v = uniform_random(-0.15f, 0.15f),
            .r = uniform_random(-1.8f, 1.8f),
            .target_speed = 0.0f,
            .steering_command = uniform_random(-0.20f, 0.20f)};
        state.target_speed = state.u + uniform_random(-0.30f, 0.30f);
        const MpcModelControl_t control = {
            .steering_rate = uniform_random(-1.5f, 1.5f),
            .target_speed_rate = uniform_random(-1.0f, 1.0f)};
        const float curvature = uniform_random(-0.18f, 0.18f);

        MpcStageLinearization_t analytic;
        MpcStageLinearization_t oracle;
        if (!mpc_model_linearize(
                &state, &control, 0.025f, curvature, &analytic) ||
            !mpc_model_linearize_fd_oracle(
                &state, &control, 0.025f, curvature, &oracle)) {
            check_true(0, "random analytic and FD Jacobians are valid");
            return;
        }
        check_true(analytic.nominal_branch_flags == 0u,
                   "random parity sample remains on smooth model branches");

        float direction_x[NX];
        float direction_u[CONTROL_COUNT];
        float error_sq = 0.0f;
        float reference_sq = 0.0f;
        for (int i = 0; i < NX; ++i)
            direction_x[i] = uniform_random(-1.0f, 1.0f);
        for (int i = 0; i < CONTROL_COUNT; ++i)
            direction_u[i] = uniform_random(-1.0f, 1.0f);
        for (int row = 0; row < NX; ++row) {
            float predicted_analytic = 0.0f;
            float predicted_oracle = 0.0f;
            for (int column = 0; column < NX; ++column) {
                predicted_analytic +=
                    analytic.A[row][column] * direction_x[column];
                predicted_oracle +=
                    oracle.A[row][column] * direction_x[column];
            }
            for (int input = 0; input < CONTROL_COUNT; ++input) {
                predicted_analytic +=
                    analytic.B[row][input] * direction_u[input];
                predicted_oracle +=
                    oracle.B[row][input] * direction_u[input];
            }
            const float difference = predicted_analytic - predicted_oracle;
            error_sq += difference * difference;
            reference_sq += predicted_oracle * predicted_oracle;
        }
        const float relative_error = sqrtf(error_sq) /
            fmaxf(sqrtf(reference_sq), 1.0e-5f);
        if (relative_error > max_analytic_fd_directional_relative_error)
            max_analytic_fd_directional_relative_error = relative_error;
        if (!(relative_error <= 1.0e-3f)) {
            fprintf(stderr,
                "FAIL: random analytic/FD directional parity sample %d: %.7g "
                "state=[%.6g %.6g %.6g %.6g %.6g %.6g %.6g] "
                "control=[%.6g %.6g] curvature=%.6g\n",
                sample, relative_error, state.e_y, state.e_psi, state.u,
                state.v, state.r, state.target_speed,
                state.steering_command, control.steering_rate,
                control.target_speed_rate, curvature);
            ++failures;
            return;
        }

        float x[NX];
        float u[CONTROL_COUNT] = {
            control.steering_rate, control.target_speed_rate};
        float next[NX];
        state_to_array(&state, x);
        const MpcStageResult_t stage = mpc_vehicle_model_step(
            &state, &control, 0.025f, curvature);
        output_to_array(&stage, next);
        for (int row = 0; row < NX; ++row) {
            float reconstructed = analytic.d[row];
            for (int column = 0; column < NX; ++column)
                reconstructed += analytic.A[row][column] * x[column];
            for (int input = 0; input < CONTROL_COUNT; ++input)
                reconstructed += analytic.B[row][input] * u[input];
            const float error = row == 1
                ? wrapped_difference(reconstructed, next[row])
                : reconstructed - next[row];
            if (!(fabsf(error) <= 2.0e-5f)) {
                fprintf(stderr,
                    "FAIL: random affine parity sample %d row %d: %.7g\n",
                    sample, row, error);
                ++failures;
                return;
            }
        }
    }
}

static void check_branch_jacobian_case(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float curvature,
    unsigned int expected_branch,
    const char *label)
{
    MpcStageLinearization_t analytic;
    MpcStageLinearization_t oracle;
    const int analytic_ok = mpc_model_linearize(
        state, control, 0.025f, curvature, &analytic);
    const int oracle_ok = mpc_model_linearize_fd_oracle(
        state, control, 0.025f, curvature, &oracle);
    check_true(analytic_ok && oracle_ok, label);
    if (!analytic_ok || !oracle_ok) return;
    if ((analytic.nominal_branch_flags & expected_branch) == 0u) {
        fprintf(stderr,
            "FAIL: %s expected branch mask 0x%x, got 0x%x\n", label,
            expected_branch, analytic.nominal_branch_flags);
        ++failures;
    }
    check_true(analytic.nominal_branch_flags == oracle.nominal_branch_flags,
               "analytic and FD oracle select identical nominal branches");

    const uint16_t nonsmooth = analytic.nonsmooth_column_mask |
        oracle.nonsmooth_column_mask;
    for (int row = 0; row < NX; ++row) {
        for (int column = 0; column < NX; ++column) {
            if (nonsmooth & (uint16_t)(1u << column)) continue;
            const float actual = analytic.A[row][column];
            const float expected = oracle.A[row][column];
            if (fabsf(actual - expected) >
                2.0e-3f * fmaxf(1.0f, fabsf(expected))) {
                fprintf(stderr,
                    "FAIL: %s A[%d,%d] analytic %.8g FD %.8g\n",
                    label, row, column, actual, expected);
                ++failures;
            }
        }
        for (int input = 0; input < CONTROL_COUNT; ++input) {
            const int column = NX + input;
            if (nonsmooth & (uint16_t)(1u << column)) continue;
            const float actual = analytic.B[row][input];
            const float expected = oracle.B[row][input];
            if (fabsf(actual - expected) >
                2.0e-3f * fmaxf(1.0f, fabsf(expected))) {
                fprintf(stderr,
                    "FAIL: %s B[%d,%d] analytic %.8g FD %.8g\n",
                    label, row, input, actual, expected);
                ++failures;
            }
        }
    }
}

static void test_analytic_jacobian_clipped_branches(void)
{
    const MpcModelState_t nominal = {
        .e_y = 0.02f, .e_psi = -0.03f, .u = 5.0f, .v = 0.04f,
        .r = 0.2f, .target_speed = 5.0f, .steering_command = 0.02f};
    MpcModelState_t state = nominal;
    MpcModelControl_t control = {0.0f, 0.0f};

    control.steering_rate = 3.5f;
    check_branch_jacobian_case(&state, &control, 0.0f,
        MPC_STAGE_CLIPPED_STEERING_RATE, "steering-rate upper clip parity");

    control = (MpcModelControl_t){0.0f, -9.0f};
    check_branch_jacobian_case(&state, &control, 0.0f,
        MPC_STAGE_CLIPPED_SPEED_RATE, "target-rate lower clip parity");

    state = nominal;
    state.steering_command = SOURCE_MAX_STEERING_RAD - 0.01f;
    control = (MpcModelControl_t){SOURCE_STEERING_RATE_RADPS, 0.0f};
    check_branch_jacobian_case(&state, &control, 0.0f,
        MPC_STAGE_CLIPPED_STEERING_COMMAND,
        "steering-command upper clip parity");

    state = nominal;
    state.target_speed = 15.99f;
    control = (MpcModelControl_t){0.0f, 3.0f};
    check_branch_jacobian_case(&state, &control, 0.0f,
        MPC_STAGE_CLIPPED_TARGET_SPEED, "target-speed upper clip parity");

    state = nominal;
    state.target_speed = 0.0f;
    control = (MpcModelControl_t){0.0f, -8.0f};
    check_branch_jacobian_case(&state, &control, 0.0f,
        MPC_STAGE_CLIPPED_TARGET_SPEED, "target-speed zero clip parity");

    state = nominal;
    state.u = 3.0f;
    state.target_speed = 15.0f;
    control = (MpcModelControl_t){0.0f, 3.0f};
    check_branch_jacobian_case(&state, &control, 0.0f,
        MPC_STAGE_CLIPPED_ACCELERATION, "positive acceleration clip parity");

    state = nominal;
    state.u = 10.0f;
    state.target_speed = 2.0f;
    control = (MpcModelControl_t){0.0f, -8.0f};
    check_branch_jacobian_case(&state, &control, 0.0f,
        MPC_STAGE_CLIPPED_ACCELERATION, "braking acceleration clip parity");

    state = nominal;
    state.u = 0.005f;
    state.target_speed = 0.0f;
    control = (MpcModelControl_t){0.0f, 0.0f};
    check_branch_jacobian_case(&state, &control, 0.0f,
        MPC_STAGE_CLIPPED_BODY_SPEED, "body-speed lower clip parity");
}

int main(void)
{
    test_linearizations_away_from_switches();
    test_saturation_boundary_is_reported();
    test_invalid_frenet_perturbation_uses_valid_side();
    test_heading_wrap_and_mirror_symmetry();
    test_analytic_jacobian_against_fd_oracle();
    test_analytic_jacobian_clipped_branches();
    if (failures) {
        fprintf(stderr, "%d MPC linearization checks failed\n", failures);
        return 1;
    }
    printf("MPC linearization tests passed: max directional relative error %.6g, "
           "heading-wrap Jacobian absolute error %.6g, random analytic/FD max directional error %.6g\n",
           max_directional_relative_error, heading_wrap_jacobian_error,
           max_analytic_fd_directional_relative_error);
    return 0;
}

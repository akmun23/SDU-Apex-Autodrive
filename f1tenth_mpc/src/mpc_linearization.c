#include "mpc_linearization.h"

#include <math.h>
#include <string.h>

#define MPC_MODEL_NX 7
#define MPC_MODEL_NU 2

static const float kStateEpsilon[MPC_MODEL_NX] = {
    1.0e-4f, 1.0e-3f, 1.0e-3f, 1.0e-3f,
    1.0e-3f, 1.0e-3f, 1.0e-5f};
static const float kInputEpsilon[MPC_MODEL_NU] = {1.0e-3f, 1.0e-3f};

static void state_to_array(const MpcModelState_t *state, float x[MPC_MODEL_NX])
{
    x[0] = state->e_y;
    x[1] = state->e_psi;
    x[2] = state->u;
    x[3] = state->v;
    x[4] = state->r;
    x[5] = state->target_speed;
    x[6] = state->steering_command;
}

static MpcModelState_t array_to_state(const float x[MPC_MODEL_NX])
{
    MpcModelState_t state;
    state.e_y = x[0];
    state.e_psi = x[1];
    state.u = x[2];
    state.v = x[3];
    state.r = x[4];
    state.target_speed = x[5];
    state.steering_command = x[6];
    return state;
}

static void control_to_array(const MpcModelControl_t *control,
                             float u[MPC_MODEL_NU])
{
    u[0] = control->steering_rate;
    u[1] = control->target_speed_rate;
}

static MpcModelControl_t array_to_control(const float u[MPC_MODEL_NU])
{
    MpcModelControl_t control;
    control.steering_rate = u[0];
    control.target_speed_rate = u[1];
    return control;
}

static void stage_to_array(const MpcStageResult_t *stage,
                           float next[MPC_MODEL_NX])
{
    state_to_array(&stage->next, next);
}

static float output_difference(int row, float plus, float minus)
{
    if (row == 1) return remainderf(plus - minus, 6.2831853071795864769f);
    return plus - minus;
}

static unsigned int bit_count(unsigned int value)
{
    unsigned int count = 0;
    while (value) {
        count += value & 1u;
        value >>= 1u;
    }
    return count;
}

static float choose_derivative(
    const MpcStageResult_t *base,
    const MpcStageResult_t *plus,
    const MpcStageResult_t *minus,
    const float base_value[MPC_MODEL_NX],
    const float plus_value[MPC_MODEL_NX],
    const float minus_value[MPC_MODEL_NX],
    float epsilon,
    int row,
    int *nonsmooth)
{
    const int plus_same = plus->valid &&
        plus->branch_flags == base->branch_flags;
    const int minus_same = minus->valid &&
        minus->branch_flags == base->branch_flags;
    if (plus_same && minus_same) {
        return output_difference(row, plus_value[row], minus_value[row]) /
            (2.0f * epsilon);
    }

    *nonsmooth = 1;
    if (plus_same && plus->valid) {
        return output_difference(row, plus_value[row], base_value[row]) /
            epsilon;
    }
    if (minus_same && minus->valid) {
        return output_difference(row, base_value[row], minus_value[row]) /
            epsilon;
    }

    const unsigned int plus_distance = bit_count(
        plus->branch_flags ^ base->branch_flags);
    const unsigned int minus_distance = bit_count(
        minus->branch_flags ^ base->branch_flags);
    if (plus->valid && (!minus->valid || plus_distance <= minus_distance)) {
        return output_difference(row, plus_value[row], base_value[row]) /
            epsilon;
    }
    if (minus->valid) {
        return output_difference(row, base_value[row], minus_value[row]) /
            epsilon;
    }
    return NAN;
}

int mpc_model_linearize(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    MpcStageLinearization_t *linearization)
{
    if (!state || !control || !linearization) return 0;
    memset(linearization, 0, sizeof(*linearization));

    const MpcStageResult_t base = mpc_vehicle_model_step(
        state, control, dt, path_curvature);
    if (!base.valid) return 0;
    linearization->nominal_branch_flags = base.branch_flags;

    float x0[MPC_MODEL_NX];
    float u0[MPC_MODEL_NU];
    float f0[MPC_MODEL_NX];
    state_to_array(state, x0);
    control_to_array(control, u0);
    stage_to_array(&base, f0);

    for (int column = 0; column < MPC_MODEL_NX + MPC_MODEL_NU; ++column) {
        const int is_state = column < MPC_MODEL_NX;
        const int index = is_state ? column : column - MPC_MODEL_NX;
        const float epsilon = is_state ? kStateEpsilon[index] : kInputEpsilon[index];
        float x_plus[MPC_MODEL_NX];
        float x_minus[MPC_MODEL_NX];
        float u_plus[MPC_MODEL_NU];
        float u_minus[MPC_MODEL_NU];
        memcpy(x_plus, x0, sizeof(x0));
        memcpy(x_minus, x0, sizeof(x0));
        memcpy(u_plus, u0, sizeof(u0));
        memcpy(u_minus, u0, sizeof(u0));
        if (is_state) {
            x_plus[index] += epsilon;
            x_minus[index] -= epsilon;
        } else {
            u_plus[index] += epsilon;
            u_minus[index] -= epsilon;
        }

        const MpcModelState_t state_plus = array_to_state(x_plus);
        const MpcModelState_t state_minus = array_to_state(x_minus);
        const MpcModelControl_t control_plus = array_to_control(u_plus);
        const MpcModelControl_t control_minus = array_to_control(u_minus);
        const MpcStageResult_t plus = mpc_vehicle_model_step(
            &state_plus, &control_plus, dt, path_curvature);
        const MpcStageResult_t minus = mpc_vehicle_model_step(
            &state_minus, &control_minus, dt, path_curvature);
        float f_plus[MPC_MODEL_NX] = {0};
        float f_minus[MPC_MODEL_NX] = {0};
        if (plus.valid) stage_to_array(&plus, f_plus);
        if (minus.valid) stage_to_array(&minus, f_minus);

        int column_nonsmooth = 0;
        for (int row = 0; row < MPC_MODEL_NX; ++row) {
            const float derivative = choose_derivative(
                &base, &plus, &minus, f0, f_plus, f_minus, epsilon, row,
                &column_nonsmooth);
            if (!isfinite(derivative)) return 0;
            if (is_state) linearization->A[row][index] = derivative;
            else linearization->B[row][index] = derivative;
        }
        if (column_nonsmooth) {
            linearization->nonsmooth_column_mask |=
                (uint16_t)(1u << column);
            ++linearization->nonsmooth_column_count;
        }
    }

    for (int row = 0; row < MPC_MODEL_NX; ++row) {
        float affine = f0[row];
        for (int column = 0; column < MPC_MODEL_NX; ++column)
            affine -= linearization->A[row][column] * x0[column];
        for (int column = 0; column < MPC_MODEL_NU; ++column)
            affine -= linearization->B[row][column] * u0[column];
        linearization->d[row] = affine;
        if (!isfinite(affine)) return 0;
    }
    linearization->valid = 1;
    return 1;
}

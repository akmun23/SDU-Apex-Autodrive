#include "mpc_linearization.h"

#include <math.h>
#include <string.h>

#ifdef MPC_ENABLE_FD_ORACLE
/* Steering-state perturbations are smaller because the measured yaw surface
 * has narrow, smooth support gates; 1e-3 rad crosses enough of a gate to make
 * the central-difference oracle a poor local-derivative check. */
static const float kStateEpsilon[MPC_MODEL_NX] = {
    1.0e-4f, 1.0e-3f, 1.0e-3f, 1.0e-3f,
    1.0e-3f, 1.0e-4f, 1.0e-4f, 1.0e-4f,
    1.0e-3f, 1.0e-3f,
    1.0e-2f, 1.0e-2f, 1.0e-2f, 1.0e-2f,
    1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-3f,
    1.0e-3f, 1.0e-3f,
    1.0e-4f, 1.0e-4f, 1.0e-4f, 1.0e-4f};
static const float kInputEpsilon[MPC_MODEL_NU] = {1.0e-3f, 1.0e-2f};
#endif

static void state_to_array(const MpcModelState_t *state, float x[MPC_MODEL_NX])
{
    MpcModelState_t prepared = *state;
    (void)vehicle_model_initialize_plant_state(&prepared);
    x[MPC_STATE_IDX_EY] = state->e_y;
    x[MPC_STATE_IDX_EPSI] = state->e_psi;
    x[MPC_STATE_IDX_U] = state->u;
    x[MPC_STATE_IDX_V] = state->v;
    x[MPC_STATE_IDX_R] = state->r;
    x[MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1] =
        state->delayed_steering_command_1;
    x[MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2] =
        state->delayed_steering_command_2;
    x[MPC_STATE_IDX_ACTUAL_STEERING_ANGLE] = state->actual_steering_angle;
    x[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1] =
        state->delayed_throttle_command_1;
    x[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2] =
        state->delayed_throttle_command_2;
    for (int wheel = 0; wheel < 4; ++wheel) {
        x[MPC_STATE_IDX_WHEEL_OMEGA_FL + wheel] =
            (float)prepared.source_plant_internal_state[
                MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel];
    }
    x[MPC_STATE_IDX_BODY_HEIGHT] = (float)prepared.source_plant_internal_state[
        MPC_PLANT_STATE_BODY_HEIGHT];
    x[MPC_STATE_IDX_BODY_HEIGHT_RATE] =
        (float)prepared.source_plant_internal_state[
            MPC_PLANT_STATE_BODY_HEIGHT_RATE];
    x[MPC_STATE_IDX_ROLL] = (float)prepared.source_plant_internal_state[
        MPC_PLANT_STATE_ROLL];
    x[MPC_STATE_IDX_ROLL_RATE] = (float)prepared.source_plant_internal_state[
        MPC_PLANT_STATE_ROLL_RATE];
    x[MPC_STATE_IDX_PITCH] = (float)prepared.source_plant_internal_state[
        MPC_PLANT_STATE_PITCH];
    x[MPC_STATE_IDX_PITCH_RATE] =
        (float)prepared.source_plant_internal_state[
            MPC_PLANT_STATE_PITCH_RATE];
    for (int wheel = 0; wheel < 4; ++wheel) {
        x[MPC_STATE_IDX_SUSPENSION_JOUNCE_FL + wheel] = (float)
            prepared.source_plant_internal_state[
                MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL + wheel];
    }
}

#ifdef MPC_ENABLE_FD_ORACLE
static MpcModelState_t array_to_state(
    const float x[MPC_MODEL_NX], const MpcModelState_t *base)
{
    MpcModelState_t state = *base;
    state.e_y = x[MPC_STATE_IDX_EY];
    state.e_psi = x[MPC_STATE_IDX_EPSI];
    state.u = x[MPC_STATE_IDX_U];
    state.v = x[MPC_STATE_IDX_V];
    state.r = x[MPC_STATE_IDX_R];
    state.delayed_steering_command_1 =
        x[MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1];
    state.delayed_steering_command_2 =
        x[MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2];
    state.actual_steering_angle = x[MPC_STATE_IDX_ACTUAL_STEERING_ANGLE];
    state.delayed_throttle_command_1 =
        x[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1];
    state.delayed_throttle_command_2 =
        x[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2];
    if (!vehicle_model_initialize_plant_state(&state)) return state;
    for (int wheel = 0; wheel < 4; ++wheel) {
        state.source_plant_internal_state[
            MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel] =
            x[MPC_STATE_IDX_WHEEL_OMEGA_FL + wheel];
    }
    state.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT] =
        x[MPC_STATE_IDX_BODY_HEIGHT];
    state.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT_RATE] =
        x[MPC_STATE_IDX_BODY_HEIGHT_RATE];
    state.source_plant_internal_state[MPC_PLANT_STATE_ROLL] =
        x[MPC_STATE_IDX_ROLL];
    state.source_plant_internal_state[MPC_PLANT_STATE_ROLL_RATE] =
        x[MPC_STATE_IDX_ROLL_RATE];
    state.source_plant_internal_state[MPC_PLANT_STATE_PITCH] =
        x[MPC_STATE_IDX_PITCH];
    state.source_plant_internal_state[MPC_PLANT_STATE_PITCH_RATE] =
        x[MPC_STATE_IDX_PITCH_RATE];
    for (int wheel = 0; wheel < 4; ++wheel) {
        state.source_plant_internal_state[
            MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL + wheel] =
            x[MPC_STATE_IDX_SUSPENSION_JOUNCE_FL + wheel];
    }
    return state;
}
#endif

static void control_to_array(const MpcModelControl_t *control,
                             float u[MPC_MODEL_NU])
{
    u[0] = control->steering_target_rad;
    u[1] = control->throttle_command;
}

#ifdef MPC_ENABLE_FD_ORACLE
static MpcModelControl_t array_to_control(const float u[MPC_MODEL_NU])
{
    MpcModelControl_t control;
    control.steering_target_rad = u[0];
    control.throttle_command = u[1];
    return control;
}
#endif

static void stage_to_array(const MpcStageResult_t *stage,
                           float next[MPC_MODEL_NX])
{
    state_to_array(&stage->next, next);
}

#ifdef MPC_ENABLE_FD_ORACLE
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
#endif

#ifdef MPC_ENABLE_FD_ORACLE
int mpc_model_linearize_fd_oracle(
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
    for (int wheel = 0; wheel < 4; ++wheel)
        linearization->maximum_slip_excess_mps[wheel] = (float)
            base.output.maximum_longitudinal_slip_excess_mps[wheel];

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

        const MpcModelState_t state_plus = array_to_state(x_plus, state);
        const MpcModelState_t state_minus = array_to_state(x_minus, state);
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
        const float base_motion[3] = {base.body_forward_delta_m,
            base.body_lateral_delta_m, base.body_yaw_delta_rad};
        const float plus_motion[3] = {plus.body_forward_delta_m,
            plus.body_lateral_delta_m, plus.body_yaw_delta_rad};
        const float minus_motion[3] = {minus.body_forward_delta_m,
            minus.body_lateral_delta_m, minus.body_yaw_delta_rad};
        for (int motion = 0; motion < 3; ++motion) {
            /* Motion increments are unwrapped, not path heading states.
             * Use the same branch selection as the physical state rows. */
            const float derivative = choose_derivative(&base, &plus, &minus,
                &base_motion[motion], &plus_motion[motion],
                &minus_motion[motion], epsilon, 0, &column_nonsmooth);
            if (!isfinite(derivative)) return 0;
            if (is_state)
                linearization->body_motion_A[motion][index] = derivative;
            else
                linearization->body_motion_B[motion][index] = derivative;
        }
        for (int wheel = 0; wheel < 4; ++wheel) {
            const float base_excess = (float)
                base.output.maximum_longitudinal_slip_excess_mps[wheel];
            const float plus_excess = (float)
                plus.output.maximum_longitudinal_slip_excess_mps[wheel];
            const float minus_excess = (float)
                minus.output.maximum_longitudinal_slip_excess_mps[wheel];
            const float derivative = choose_derivative(&base, &plus, &minus,
                &base_excess, &plus_excess, &minus_excess,
                epsilon, 0, &column_nonsmooth);
            if (!isfinite(derivative)) return 0;
            if (is_state)
                linearization->maximum_slip_excess_A[wheel][index] = derivative;
            else
                linearization->maximum_slip_excess_B[wheel][index] = derivative;
        }
        if (column_nonsmooth) {
            linearization->nonsmooth_column_mask |= UINT64_C(1) << column;
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
        if (!isfinite(affine)) {
            linearization->failure_reason = 6;
            linearization->failure_index = row;
            return 0;
        }
    }
    linearization->valid = 1;
    return 1;
}
#endif

int mpc_model_linearize(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    MpcStageLinearization_t *linearization)
{
    if (state == NULL || control == NULL || linearization == NULL) return 0;
    MpcStageResult_t stage;
    if (!mpc_vehicle_model_step_with_jacobian(
            state, control, dt, path_curvature, &stage, linearization)) {
        return 0;
    }

    float x[MPC_MODEL_NX];
    float u[MPC_MODEL_NU];
    float f[MPC_MODEL_NX];
    state_to_array(state, x);
    control_to_array(control, u);
    stage_to_array(&stage, f);
    for (int row = 0; row < MPC_MODEL_NX; ++row) {
        float affine = f[row];
        for (int column = 0; column < MPC_MODEL_NX; ++column)
            affine -= linearization->A[row][column] * x[column];
        for (int input = 0; input < MPC_MODEL_NU; ++input)
            affine -= linearization->B[row][input] * u[input];
        if (!isfinite(affine)) return 0;
        linearization->d[row] = affine;
    }
    return 1;
}

int mpc_model_linearize_from_known_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    const MpcStageResult_t *known_step,
    MpcStageLinearization_t *linearization)
{
    if (!state || !control || !known_step || !linearization) return 0;
    MpcStageResult_t stage;
    if (!mpc_vehicle_model_linearize_from_known_step(
            state, control, dt, path_curvature, known_step, &stage,
            linearization)) return 0;

    float x[MPC_MODEL_NX];
    float u[MPC_MODEL_NU];
    float f[MPC_MODEL_NX];
    state_to_array(state, x);
    control_to_array(control, u);
    stage_to_array(&stage, f);
    for (int row = 0; row < MPC_MODEL_NX; ++row) {
        float affine = f[row];
        for (int column = 0; column < MPC_MODEL_NX; ++column)
            affine -= linearization->A[row][column] * x[column];
        for (int input = 0; input < MPC_MODEL_NU; ++input)
            affine -= linearization->B[row][input] * u[input];
        if (!isfinite(affine)) return 0;
        linearization->d[row] = affine;
    }
    return 1;
}

#include "mpc_linearization.h"

#include <math.h>

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

static void control_to_array(const MpcModelControl_t *control,
                             float u[MPC_MODEL_NU])
{
    u[0] = control->steering_target_rad;
    u[1] = control->throttle_command;
}

static void stage_to_array(const MpcStageResult_t *stage,
                           float next[MPC_MODEL_NX])
{
    state_to_array(&stage->next, next);
}


// Full-affine callers need d = f(x,u) - A*x - B*u; the RTI's reduced-state
// QP intentionally uses the separate sensitivity-only path instead.
static int finish_affine_linearization(
    const MpcModelState_t *state, const MpcModelControl_t *control,
    const MpcStageResult_t &stage, MpcStageLinearization_t *linearization)
{
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

    return finish_affine_linearization(state, control, stage, linearization);
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

    return finish_affine_linearization(state, control, stage, linearization);
}

/* Source derivatives are expensive, but their full-state affine offset is
 * discarded by the reduced-state Riccati QP. Avoid preparing x, u, f and
 * their full 24-dimensional A*x+B*u contraction; the QP computes the
 * required affine offsets once in its own coordinates. */
int mpc_model_linearize_sensitivities_from_known_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    const MpcStageResult_t *known_step,
    MpcStageLinearization_t *linearization)
{
    if (!state || !control || !known_step || !linearization) return 0;
    MpcStageResult_t stage;
    return mpc_vehicle_model_linearize_from_known_step(
        state, control, dt, path_curvature, known_step, &stage,
        linearization);
}

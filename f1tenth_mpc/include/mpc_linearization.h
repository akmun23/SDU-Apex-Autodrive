#ifndef MPC_LINEARIZATION_H
#define MPC_LINEARIZATION_H

#include "vehicle_model.h"

#include <stdint.h>

typedef struct
{
    float A[MPC_MODEL_NX][MPC_MODEL_NX];
    float B[MPC_MODEL_NX][2];
    float d[MPC_MODEL_NX];
    unsigned int nominal_branch_flags;
    uint16_t nonsmooth_column_mask;
    int nonsmooth_column_count;
    int valid;
} MpcStageLinearization_t;

/* Production branch-aware analytic Jacobian fused with the authoritative
 * discrete stage map. */
int mpc_model_linearize(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    MpcStageLinearization_t *linearization);

/* Finite-difference oracle retained for tests and offline parity checks. */
int mpc_model_linearize_fd_oracle(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    MpcStageLinearization_t *linearization);

/* Shared-model implementation that returns the nonlinear stage and its
 * branch-aware sensitivities from the same intermediate calculations. */
int mpc_vehicle_model_step_with_jacobian(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    MpcStageResult_t *stage,
    MpcStageLinearization_t *linearization);

#endif

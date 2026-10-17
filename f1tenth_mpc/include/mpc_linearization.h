#ifndef MPC_LINEARIZATION_H
#define MPC_LINEARIZATION_H

#include "vehicle_model.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    float A[MPC_MODEL_NX][MPC_MODEL_NX];
    float B[MPC_MODEL_NX][2];
    float d[MPC_MODEL_NX];
    unsigned int nominal_branch_flags;
    uint32_t nonsmooth_column_mask;
    int nonsmooth_column_count;
    int failure_reason;
    int failure_index;
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

/* Reuse a previously computed exact nonlinear transition when linearizing
 * its nominal state. This avoids simulating the same source-plant step twice.
 */
int mpc_model_linearize_from_known_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    const MpcStageResult_t *known_step,
    MpcStageLinearization_t *linearization);

/* Lower-level transition/Jacobian path used by the affine-map wrapper. */
int mpc_vehicle_model_linearize_from_known_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    const MpcStageResult_t *known_step,
    MpcStageResult_t *stage,
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

#ifdef __cplusplus
}
#endif

#endif

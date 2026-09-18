#ifndef MPC_LINEARIZATION_H
#define MPC_LINEARIZATION_H

#include "vehicle_model.h"

#include <stdint.h>

typedef struct
{
    float A[7][7];
    float B[7][2];
    float d[7];
    unsigned int nominal_branch_flags;
    uint16_t nonsmooth_column_mask;
    int nonsmooth_column_count;
    int valid;
} MpcStageLinearization_t;

/* Branch-aware numerical Jacobian of the authoritative discrete stage map. */
int mpc_model_linearize(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    MpcStageLinearization_t *linearization);

#endif

#pragma once

// Private interface between the quadratic Riccati, ADMM and scaling units.
// Applications and replay tools continue to use only riccati_solver.h.
#include "riccati_solver.h"

// Keep the solver debug state owned by the ADMM unit. Factorization gets
// private access without introducing an externally writable global object.
RiccatiDebugInfo_t &riccati_debug_last_mutable();

#ifdef MPC_ENABLE_RICCATI_PROFILE
uint64_t riccati_profile_now_ns(void);
#endif

int riccati_solver_primal_pass(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    const float rho_state[][RICCATI_MAX_NX],
    const float rho_input[][RICCATI_MAX_NU],
    const float z_x[][RICCATI_MAX_NX],
    const float y_x[][RICCATI_MAX_NX],
    const float z_u[][RICCATI_MAX_NU],
    const float y_u[][RICCATI_MAX_NU],
    float x_out[][RICCATI_MAX_NX],
    float u_out[][RICCATI_MAX_NU],
    RiccatiFactorization_t *factorization,
    const float terminal_augmentation[RICCATI_MAX_NX][RICCATI_MAX_NX]);

RiccatiStatus_t riccati_admm_solve_core(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    const RiccatiAdmmConfig_t *config,
    RiccatiAdmmState_t *admm_state,
    RiccatiSolution_t *solution,
    float primal_u_out[RICCATI_HORIZON][RICCATI_MAX_NU],
    float feedback_out[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX]);


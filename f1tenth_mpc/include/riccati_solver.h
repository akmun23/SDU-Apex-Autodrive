/**
 * @file riccati_solver.h
 * @brief Riccati-ADMM solver for non-condensed MPC.
 * @details Declares API contracts for solving the constrained linear-quadratic
 *          MPC subproblem with an ADMM outer loop and a Riccati recursion inner
 *          solve. Solves the constrained optimal control problem:
 *
 *   min  Σ_{k=0}^{N-1} [l_k(x_k, u_k)] + l_N(x_N)
 *   s.t. x_{k+1} = A_k x_k + B_k u_k
 *        x_lb_k <= x_k <= x_ub_k    (state box constraints)
 *        u_lb_k <= u_k <= u_ub_k    (control box constraints)
 *
 * Using ADMM to handle constraints, with Riccati recursion for the
 * unconstrained LQR sub-problem (O(N) per ADMM iteration).
 *
 * All arithmetic uses native float operations.
 * @dependencies mpc_types.h
 */

#ifndef RICCATI_SOLVER_H
#define RICCATI_SOLVER_H

#include "mpc_types.h"
#include <stdint.h>

#ifdef __cplusplus
#define RICCATI_RESTRICT __restrict
#else
#define RICCATI_RESTRICT restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    float rho_start;
    float rho_u_start;
    float rho;
    float rho_u;
    int rho_change_count;
    float primal_residual;
    float dual_residual;
    float last_invert_det;
    float last_fallback_s00;
    float last_fallback_s11;
    int invert_fallback_count;
    float max_control_hessian_regularization;
    int control_hessian_regularization_count;
    int quadratic_factorization_count;
    uint64_t quadratic_factorization_time_ns;
    uint64_t quadratic_rhs_time_ns;
    uint64_t projection_residual_time_ns;
    uint64_t cold_start_initialization_time_ns;
    uint64_t rho_update_time_ns;
    uint64_t scaling_transform_time_ns;
    uint64_t output_unscale_time_ns;
} RiccatiDebugInfo_t;

#define RICCATI_DEBUG_TRACE_MAX 256

typedef struct
{
    int iter;
    float primal_residual;
    float dual_residual;
    float state_primal_residual;
    float state_dual_residual;
    float ctrl_primal_residual;
    float ctrl_dual_residual;
    float rho;
    float rho_u;
    float u0_steer;
    float u0_throttle;
    float z0_steer;
    float z0_throttle;
    float y0_steer;
    float y0_throttle;
    int scale_rho;
    int scale_rho_u;
} RiccatiDebugIterSample_t;

/**
 * @brief Execute one Riccati backward-forward pass for ADMM primal update.
 * @details Inputs: per-stage model/cost data, terminal costs, initial state,
 *          dimensions, current ADMM penalties, and ADMM z/y iterates.
 *          Purpose: compute unconstrained primal trajectories under the current
 *          augmented-Lagrangian objective.
 *          Outputs: writes x_out/u_out trajectories for all stages.
 * @param step_data Per-step dynamics and cost data.
 * @param terminal_Q Terminal diagonal state cost.
 * @param terminal_q Terminal linear state cost.
 * @param x0 Initial state vector.
 * @param nx State dimension.
 * @param nu Control dimension.
 * @param N Prediction horizon length.
 * @param rho ADMM penalty for constrained state channels.
 * @param rho_u ADMM penalty for control channels.
 * @param z_x ADMM projected state variables.
 * @param y_x ADMM dual state variables.
 * @param z_u ADMM projected control variables.
 * @param y_u ADMM dual control variables.
 * @param x_out Output state trajectory.
 * @param u_out Output control trajectory.
 * @return 1 when the complete pass is finite and valid; otherwise 0.
 */
int riccati_solver_pass(
    const RiccatiStepData_t * RICCATI_RESTRICT step_data,
    const float * RICCATI_RESTRICT terminal_Q,
    const float * RICCATI_RESTRICT terminal_q,
    const float * RICCATI_RESTRICT terminal_x_lb,
    const float * RICCATI_RESTRICT terminal_x_ub,
    const float * RICCATI_RESTRICT x0,
    float rho,
    float rho_u,
    const float z_x[][RICCATI_MAX_NX],
    const float y_x[][RICCATI_MAX_NX],
    const float z_u[][RICCATI_MAX_NU],
    const float y_u[][RICCATI_MAX_NU],
    float x_out[][RICCATI_MAX_NX],
    float u_out[][RICCATI_MAX_NU]);

/** Cached quadratic Riccati terms for repeated ADMM linear-RHS updates. */
typedef struct
{
    float P[RICCATI_HORIZON + 1][RICCATI_MAX_NX][RICCATI_MAX_NX];
    float K[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX];
    float S_inv[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NU];
    float G[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX];
    /* Constant affine dynamics contribution: P[k+1] d[k]. Refreshed
     * together with P and K whenever rho-dependent factors are rebuilt. */
    float affine_state_shift[RICCATI_HORIZON][RICCATI_MAX_NX];
    /* Only private fixed-QP ADMM may assume d[k] never changes while
     * reusing this factorization. Public factored RHS calls explicitly
     * support changing d[k] between calls. */
    uint8_t private_fixed_qp_affine_cache_enabled;
    uint8_t x_is_constrained[RICCATI_HORIZON + 1][RICCATI_MAX_NX];
    /* Diagonal augmented-Lagrangian penalties; scalar fields are summaries
     * retained for the public scalar-factorization API and diagnostics. */
    float rho_state[RICCATI_HORIZON + 1][RICCATI_MAX_NX];
    float rho_input[RICCATI_HORIZON][RICCATI_MAX_NU];
    float rho;
    float rho_u;
    int valid;
} RiccatiFactorization_t;

int riccati_solver_factorize(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    float rho,
    float rho_u,
    RiccatiFactorization_t *factorization);

int riccati_solver_pass_factored(
    const RiccatiStepData_t *step_data,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    const RiccatiFactorization_t *factorization,
    const float z_x[][RICCATI_MAX_NX],
    const float y_x[][RICCATI_MAX_NX],
    const float z_u[][RICCATI_MAX_NU],
    const float y_u[][RICCATI_MAX_NU],
    float x_out[][RICCATI_MAX_NX],
    float u_out[][RICCATI_MAX_NU]);

#undef RICCATI_RESTRICT

/**
 * @brief Zero all ADMM warm-start buffers and mark the state as uninitialized.
 * @details Inputs: state points to writable warm-start buffers.
 *          Purpose: reset ADMM history so the next solve uses a cold start.
 *          Outputs: clears z/y buffers, stored rho values, and sets
 *          initialized = 0.
 *          Call before the first solve, and whenever the problem changes enough
 *          to make the previous warm-start invalid (for example, a large
 *          operating-point change). If state is NULL, the function returns
 *          without side effects.
 * @param state Pointer to ADMM state to clear.
 * @return None.
 */
void riccati_admm_state_init(RiccatiAdmmState_t *state);

/** Shift a valid ADMM warm-start one stage toward the present. */
void riccati_admm_shift_warm_start(
    RiccatiAdmmState_t *state,
    const float *new_x0);

/**
 * @brief Solve constrained LQR over a finite horizon using Riccati-ADMM.
 * @details Inputs: fixed-size per-step dynamics/cost data, terminal cost
 *          vectors, initial state, and solver configuration.
 *          Purpose: compute a control/state trajectory that minimizes stage and
 *          terminal costs while satisfying box constraints through ADMM
 *          projections.
 *          Outputs: writes state and control trajectories, residuals,
 *          iteration count, and solver status to *solution, and updates
 *          *admm_state for warm-start reuse.
 *
 * Each ADMM iteration consists of:
 * 1. Riccati backward pass: compute gains K_k, k_k using
 *    augmented costs (Q + rho*I, R + rho*I) and ADMM offsets
 * 2. Riccati forward pass: roll out x_k, u_k from x0
 * 3. z-update: project (x + y, u + y) onto box constraints
 * 4. y-update: dual variable update
 * 5. Convergence check against the configured absolute maximum residual
 *
 * This solver is specialized for N=RICCATI_HORIZON=30,
 * nx=RICCATI_MAX_NX=23, and nu=RICCATI_MAX_NU=2. These compile-time
 * dimensions are the only supported problem shape.
 *
 * @param step_data   Per-step dynamics and cost data (array of 30 elements)
 * @param terminal_Q  Terminal state cost (diagonal weights, length 23)
 * @param terminal_q  Terminal state linear cost (length 23)
 * @param x0          Initial state (length 23)
 * @param config      ADMM configuration (if NULL, compile-time defaults are used).
 * @param admm_state  ADMM warm-start state (modified in-place)
 * @param solution    Output: optimal trajectories and status
 * @return Status code. Returns RICCATI_STATUS_ERROR if any mandatory pointer
 *         input is NULL.
 */
RiccatiStatus_t riccati_admm_solve(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    const RiccatiAdmmConfig_t *config,
    RiccatiAdmmState_t *admm_state,
    RiccatiSolution_t *solution);

/* Offline diagnostic variant. In addition to the normal projected-control
 * result, writes the final unconstrained Riccati primal controls before the
 * ADMM box projection. Passing this output is optional and never used by the
 * production controller. */
RiccatiStatus_t riccati_admm_solve_with_primal_controls(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    const RiccatiAdmmConfig_t *config,
    RiccatiAdmmState_t *admm_state,
    RiccatiSolution_t *solution,
    float primal_u_out[RICCATI_HORIZON][RICCATI_MAX_NU]);

/* Same solve, additionally return the already-factorized local feedback gain
 * in PHYSICAL coordinates: du = K dx. No new factorization or solver states.
 * This is the augmented Riccati policy, not a constrained-optimum sensitivity;
 * a nonlinear trial using it must still enforce all bounds and be validated. */
RiccatiStatus_t riccati_admm_solve_with_feedback(
    const RiccatiStepData_t *step_data, const float *terminal_Q,
    const float *terminal_q, const float *terminal_x_lb,
    const float *terminal_x_ub, const float *x0,
    const RiccatiAdmmConfig_t *config, RiccatiAdmmState_t *admm_state,
    RiccatiSolution_t *solution,
    float primal_u_out[RICCATI_HORIZON][RICCATI_MAX_NU],
    float feedback_out[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX]);

void riccati_debug_get_last(RiccatiDebugInfo_t *out);
int riccati_debug_get_trace_count(void);
int riccati_debug_get_trace_sample(int index, RiccatiDebugIterSample_t *out);
/** Enable or disable detailed per-iteration trace capture (disabled by default). */
void riccati_debug_set_trace_enabled(int enabled);

/** Debug flag: set to 1 to print ADMM iteration details */
extern int riccati_admm_debug;

#ifdef __cplusplus
}
#endif

#endif /* RICCATI_SOLVER_H */

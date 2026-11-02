/**
 * @file riccati_solver.cpp
 * @brief ADMM constrained-QP solver and warm-start state management.
 * @details Calls the unique factorized Riccati implementation in
 *          riccati_factorization.cpp. QP scaling/unscaling resides in
 *          riccati_scaling.cpp, keeping this file focused on ADMM.
 *
 * Riccati backward pass (per step):
 *   M    = B^T P_{k+1}           (nu×nx)
 *   S    = R + M B               (nu×nu, invert via 2×2 formula)
 *   G    = M A                   (nu×nx)
 *   K_k  = -S^{-1} G             (nu×nx, feedback gain)
 *   kk_k = -S^{-1} (r + B^T p)   (nu×1, feedforward)
 *   P_k  = Q + A^T P A + G^T K   (nx×nx)
 *   p_k  = q + A^T p + G^T kk    (nx×1)
 *
 * Riccati forward pass:
 *   x_0 = given
 *   u_k = K_k x_k + kk_k
 *   x_{k+1} = A x_k + B u_k
 *
 * ADMM loop:
 *   1. Riccati pass with augmented costs (Q+rhoI, R+rhoI)
 *   2. z = clip(x+y, lb, ub) and z_u = clip(u+y_u, u_lb, u_ub)
 *   3. y += x - z, y_u += u - z_u
 *   4. Check convergence
 *
 * All operations use native float32 arithmetic.
 * @dependencies riccati_solver.h, <string.h>, <stdio.h>, <math.h>
 */

#include "riccati_solver_internal.hpp"
#include "riccati_state_polygon.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static uint64_t riccati_monotonic_now_ns(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

#ifdef MPC_ENABLE_RICCATI_PROFILE
uint64_t riccati_profile_now_ns(void)
{
    return riccati_monotonic_now_ns();
}
#endif

/* Debug flag: set to 1 from tests to print ADMM iteration details */
int riccati_admm_debug = 0;
static RiccatiDebugInfo_t g_riccati_debug_last;

RiccatiDebugInfo_t &riccati_debug_last_mutable()
{
    return g_riccati_debug_last;
}
static RiccatiDebugIterSample_t g_riccati_debug_trace[RICCATI_DEBUG_TRACE_MAX];
static int g_riccati_debug_trace_count = 0;
static int g_riccati_debug_trace_enabled = 0;

void riccati_debug_get_last(RiccatiDebugInfo_t *out)
{
    if (!out) return;
    *out = g_riccati_debug_last;
}

int riccati_debug_get_trace_count(void)
{
    return g_riccati_debug_trace_count;
}

int riccati_debug_get_trace_sample(int index, RiccatiDebugIterSample_t *out)
{
    if (!out || index < 0 || index >= g_riccati_debug_trace_count) return -1;
    *out = g_riccati_debug_trace[index];
    return 0;
}

void riccati_debug_set_trace_enabled(int enabled)
{
    g_riccati_debug_trace_enabled = enabled != 0;
    if (!g_riccati_debug_trace_enabled) {
        g_riccati_debug_trace_count = 0;
    }
}

/*===========================================================================
 * ADMM State Initialization
 *===========================================================================*/

void riccati_admm_state_init(RiccatiAdmmState_t *state)
{
    // Guard against NULL pointer input
    if (!state) {
        return;
    }

    // Clear all buffers and reset initialized flag
    memset(state, 0, sizeof(*state));
    state->initialized = 0;
}

void riccati_admm_shift_warm_start(
    RiccatiAdmmState_t *state,
    const float *new_x0)
{
    if (!state || !state->initialized || !new_x0) {
        return;
    }
    enum {
        nx = RICCATI_MAX_NX,
        nu = RICCATI_MAX_NU,
        horizon = RICCATI_HORIZON
    };

    for (int k = 0; k < horizon; ++k) {
        const int source = k + 1;  // k is always in [0, horizon).
        for (int i = 0; i < nx; ++i) {
            state->z_x[k][i] = state->z_x[source][i];
            state->y_x[k][i] = state->y_x[source][i];
            state->rho_state[k][i] = state->rho_state[source][i];
        }
    }
    /* Hold the prior terminal prediction at the new horizon endpoint. */
    for (int i = 0; i < nx; ++i) {
        state->z_x[horizon][i] = state->z_x[horizon - 1][i];
        state->y_x[horizon][i] = state->y_x[horizon - 1][i];
        state->rho_state[horizon][i] =
            state->rho_state[horizon - 1][i];
    }
    for (int k = 0; k < horizon - 1; ++k) {
        for (int a = 0; a < nu; ++a) {
            state->z_u[k][a] = state->z_u[k + 1][a];
            state->y_u[k][a] = state->y_u[k + 1][a];
            state->rho_input[k][a] = state->rho_input[k + 1][a];
        }
    }
    if (horizon > 1) {
        for (int a = 0; a < nu; ++a) {
            state->z_u[horizon - 1][a] = state->z_u[horizon - 2][a];
            state->y_u[horizon - 1][a] = state->y_u[horizon - 2][a];
            state->rho_input[horizon - 1][a] =
                state->rho_input[horizon - 2][a];
        }
    }
    for (int i = 0; i < nx; ++i) {
        state->z_x[0][i] = new_x0[i];
        state->y_x[0][i] = 0.0f;
    }
}

/*===========================================================================
 * Main Solver: Riccati-ADMM
 *===========================================================================*/

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
    float feedback_out[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX])
{
    if (!step_data || !terminal_Q || !terminal_q || !terminal_x_lb || !terminal_x_ub || !x0 || !admm_state || !solution) {
        if (solution) {
            solution->status = RICCATI_STATUS_ERROR;
        }
        return RICCATI_STATUS_ERROR;
    }

    solution->time_budget_exhausted = 0;

    if (config && config->cold_start_each_solve)
        riccati_admm_state_init(admm_state);

    enum { nx = RICCATI_MAX_NX, nu = RICCATI_MAX_NU,
           N = RICCATI_HORIZON };
    const float over_relaxation = config &&
        config->over_relaxation > 0.0f ? config->over_relaxation : 1.0f;
    if (!isfinite(over_relaxation) || over_relaxation < 1.0f ||
        over_relaxation > 2.0f) {
        solution->status = RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }

    const RiccatiTerminalConstraints_t empty_terminal{};
    const auto &terminal = config ? config->terminal_constraints : empty_terminal;
    if (terminal.count<0 || terminal.count>RICCATI_TERMINAL_MAX_PLANES ||
        (terminal.count && (!isfinite(terminal.rho) || terminal.rho<=0.f))) {
        solution->status=RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }
    float terminal_augmentation[nx][nx]{};
    float terminal_z[RICCATI_TERMINAL_MAX_PLANES]{};
    float terminal_y[RICCATI_TERMINAL_MAX_PLANES]{};
    for (int f=0;f<terminal.count;++f) {
        if (!isfinite(terminal.upper[f])) {
            solution->status=RICCATI_STATUS_ERROR; return RICCATI_STATUS_ERROR;
        }
        double norm=0.;
        for (int i=0;i<nx;++i) {
            if (!isfinite(terminal.normal[f][i])) {
                solution->status=RICCATI_STATUS_ERROR; return RICCATI_STATUS_ERROR;
            }
            norm += (double)terminal.normal[f][i]*terminal.normal[f][i];
            for (int j=0;j<nx;++j)
                terminal_augmentation[i][j] += terminal.rho *
                    terminal.normal[f][i]*terminal.normal[f][j];
        }
        if (!(norm>0.)) {
            solution->status=RICCATI_STATUS_ERROR; return RICCATI_STATUS_ERROR;
        }
    }

    const float cfg_rho = (config && config->rho > 0.0f) ? config->rho : ADMM_RHO;
    const float cfg_rho_u = (config && config->rho_u > 0.0f) ? config->rho_u : ADMM_RHO_U;
    float cfg_rho_x_by_state[RICCATI_MAX_NX];
    int per_state_rho_enabled = 0;
    for (int s = 0; s < RICCATI_MAX_NX; ++s) {
        const float configured = config ? config->rho_x_by_state[s] : 0.0f;
        if (!isfinite(configured) || configured < 0.0f ||
            configured > 127.0f || (configured > 0.0f && configured < 1.0f)) {
            solution->status = RICCATI_STATUS_ERROR;
            return RICCATI_STATUS_ERROR;
        }
        cfg_rho_x_by_state[s] = configured > 0.0f ? configured : cfg_rho;
        per_state_rho_enabled |= configured > 0.0f;
    }
    float cfg_rho_u_by_input[RICCATI_MAX_NU];
    for (int a = 0; a < RICCATI_MAX_NU; ++a) {
        const float configured = config ? config->rho_u_by_input[a] : 0.0f;
        if (!isfinite(configured) || configured < 0.0f || configured > 127.0f) {
            solution->status = RICCATI_STATUS_ERROR;
            return RICCATI_STATUS_ERROR;
        }
        cfg_rho_u_by_input[a] = configured > 0.0f ? configured : cfg_rho_u;
    }
    const float cfg_rho_max = config && config->rho_max > 0.0f
        ? fminf(config->rho_max, 127.0f) : 127.0f;
    const int cfg_max_iter = (config && config->max_iterations > 0)
                                 ? config->max_iterations
                                 : MAX_ITERATIONS;
    const float tolerance = (config && config->tolerance > 0.0f)
                                ? config->tolerance
                                : CONVERGENCE_TOLERANCE;
    /* State and input channels use different physical units (m, rad, m/s,
     * rad/s, m/s^2). A single relative scale based on the largest raw state
     * lets a large speed or dual value silently loosen the corridor/control
     * residual gate. Treat the configured tolerance as an absolute maximum
     * residual instead; the RTI layer has a separate, explicit degraded gate. */
    const float residual_tolerance = (tolerance > 1e-6f) ? tolerance : 1e-6f;
    const int adaptive_rho = config ? config->adaptive_rho : 1;
    const int adaptive_rho_update_period = config &&
        config->adaptive_rho_update_period > 0
        ? config->adaptive_rho_update_period : 1;
    const int shared_rho = config ? config->shared_rho : 0;
    const double max_solve_time_us = config ? config->max_solve_time_us : 0.0;
    if (!isfinite(max_solve_time_us) || max_solve_time_us < 0.0) {
        solution->status = RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }
    const uint64_t solve_start_ns = riccati_monotonic_now_ns();

    memset(&g_riccati_debug_last, 0, sizeof(g_riccati_debug_last));
    g_riccati_debug_trace_count = 0;
    RiccatiFactorization_t factorization = {};

    /* Shift split variables AND penalties before choosing the new epoch's
     * channel penalties. Otherwise a shifted dual can be paired with the
     * previous stage's rho during normalization. */
    if (admm_state->initialized && !(config && config->reuse_same_epoch)) {
        riccati_admm_shift_warm_start(admm_state, x0);
    }

    float rho = (admm_state->initialized && admm_state->rho > 0.0f)
                    ? admm_state->rho : cfg_rho;
    float rho_u = (admm_state->initialized && admm_state->rho_u > 0.0f)
                    ? admm_state->rho_u : (cfg_rho_u > 0.0f ? cfg_rho_u : rho);
    float rho_state[RICCATI_HORIZON + 1][RICCATI_MAX_NX];
    float rho_input[RICCATI_HORIZON][RICCATI_MAX_NU];
    /* Penalties track the same horizon coordinates as z/y. Older
     * scalar-only warm states migrate by broadcasting their scalar values. */
    for (int k = 0; k <= N; ++k) {
        for (int s = 0; s < nx; ++s) {
            const float old_rho = admm_state->initialized &&
                    admm_state->rho_state[k][s] > 0.0f
                ? admm_state->rho_state[k][s]
                : (admm_state->initialized && admm_state->rho > 0.0f
                    ? admm_state->rho : cfg_rho_x_by_state[s]);
            rho_state[k][s] = fminf(fmaxf(old_rho, 1.0f), cfg_rho_max);
            if (admm_state->initialized && rho_state[k][s] != old_rho)
                admm_state->y_x[k][s] *= old_rho / rho_state[k][s];
        }
    }
    for (int k = 0; k < N; ++k) {
        for (int a = 0; a < nu; ++a) {
            const float old_rho_u = admm_state->initialized &&
                    admm_state->rho_input[k][a] > 0.0f
                ? admm_state->rho_input[k][a]
                : (admm_state->initialized && admm_state->rho_u > 0.0f
                    ? admm_state->rho_u : cfg_rho_u_by_input[a]);
            rho_input[k][a] =
                fminf(fmaxf(old_rho_u, 1.0f), cfg_rho_max);
            if (admm_state->initialized && rho_input[k][a] != old_rho_u)
                admm_state->y_u[k][a] *=
                    old_rho_u / rho_input[k][a];
        }
    }
    rho_u = 0.0f;
    for (int a = 0; a < nu; ++a)
        rho_u = fmaxf(rho_u, rho_input[0][a]);
    if (shared_rho) {
        /* Preserve the unscaled dual when a caller requests one common
         * penalty after a previous per-coordinate solve. */
        const float shared = fminf(fmaxf(rho, 1.0f), cfg_rho_max);
        for (int k = 0; k <= N; ++k) {
            for (int s = 0; s < nx; ++s) {
                if (admm_state->initialized && rho_state[k][s] != shared)
                    admm_state->y_x[k][s] *= rho_state[k][s] / shared;
                rho_state[k][s] = shared;
            }
        }
        for (int k = 0; k < N; ++k) {
            for (int a = 0; a < nu; ++a) {
                if (admm_state->initialized && rho_input[k][a] != shared)
                    admm_state->y_u[k][a] *= rho_input[k][a] / shared;
                rho_input[k][a] = shared;
            }
        }
        rho = shared;
        rho_u = shared;
    } else {
        /* Keep each state/input channel uniform over stages. Per-state
         * penalties are opt-in; otherwise preserve the scalar state-rho
         * behavior that was already validated on the complete replay bag. */
        if (per_state_rho_enabled) {
            rho = 0.0f;
            for (int s = 0; s < nx; ++s) {
                const float channel_rho = fminf(
                    fmaxf(rho_state[0][s], 1.0f), cfg_rho_max);
                for (int k = 0; k <= N; ++k) {
                    if (admm_state->initialized &&
                        rho_state[k][s] != channel_rho)
                        admm_state->y_x[k][s] *=
                            rho_state[k][s] / channel_rho;
                    rho_state[k][s] = channel_rho;
                }
                rho = fmaxf(rho, channel_rho);
            }
        } else {
            rho = fminf(fmaxf(rho, 1.0f), cfg_rho_max);
            for (int k = 0; k <= N; ++k) {
                for (int s = 0; s < nx; ++s) {
                    if (admm_state->initialized && rho_state[k][s] != rho)
                        admm_state->y_x[k][s] *= rho_state[k][s] / rho;
                    rho_state[k][s] = rho;
                }
            }
        }
        rho_u = 0.0f;
        for (int a = 0; a < nu; ++a) {
            const float channel_rho = fminf(
                fmaxf(rho_input[0][a], 1.0f), cfg_rho_max);
            for (int k = 0; k < N; ++k) {
                if (admm_state->initialized &&
                    rho_input[k][a] != channel_rho)
                    admm_state->y_u[k][a] *=
                        rho_input[k][a] / channel_rho;
                rho_input[k][a] = channel_rho;
            }
            rho_u = fmaxf(rho_u, channel_rho);
        }
    }
    g_riccati_debug_last.rho_start = rho;
    g_riccati_debug_last.rho_u_start = rho_u;
    int max_iter = cfg_max_iter;

    /* ADMM variables (persistent buffers for warm-start reuse). */
    float (*z_x)[RICCATI_MAX_NX] = admm_state->z_x;
    float (*z_u)[RICCATI_MAX_NU] = admm_state->z_u;
    float (*y_x)[RICCATI_MAX_NX] = admm_state->y_x;
    float (*y_u)[RICCATI_MAX_NU] = admm_state->y_u;


    /* Precompute constrained flags */
    uint8_t x_is_constrained[RICCATI_HORIZON + 1][RICCATI_MAX_NX];
    memset(x_is_constrained, 0, sizeof(x_is_constrained));
    for (int k = 0; k <= N; k++) {
        for (int s = 0; s < nx; s++) {
            const float xub = (k < N) ? step_data[k].x_ub[s] : terminal_x_ub[s];
            const float xlb = (k < N) ? step_data[k].x_lb[s] : terminal_x_lb[s];
            x_is_constrained[k][s] = k > 0 &&
                                     (xub < BIG_BOUND || xlb > -BIG_BOUND);
        }
    }

    if (admm_state->initialized) {
        /* A warm start may contain a projected (or simply stale) stage-zero
         * value. Keep the fixed initial state out of the split variables. */
        for (int s = 0; s < nx; s++) {
            z_x[0][s] = x0[s];
            y_x[0][s] = 0.0f;
        }

    }

    /* Remaining initialization and iterations follow. */

    if (!admm_state->initialized) {
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t cold_init_start_ns = riccati_profile_now_ns();
#endif
        /* Cold start */
        memset(z_x, 0, sizeof(admm_state->z_x));
        memset(z_u, 0, sizeof(admm_state->z_u));
        memset(y_x, 0, sizeof(admm_state->y_x));
        memset(y_u, 0, sizeof(admm_state->y_u));

        /* The unconstrained zero-rho solution is a high-quality initial
         * iterate in the usual case. But it is an AUXILIARY Riccati matrix
         * whose 2x2 Schur complement can become indefinite in sharp turns
         * even when the positively augmented ADMM problem is numerically
         * solvable. Its failure must not reject a valid controller cycle
         * before ADMM has executed a single iteration.
         *
         * Preserve the established normal initializer exactly. ONLY when
         * it fails, initialize the SAME QP from its positive ADMM rho;
         * the subsequent real ADMM constraints, convergence checks, and
         * source-validated control sequence remain unchanged. */
        const auto cold_riccati_seed = [&](float initial_rho_x,
                                           float initial_rho_u) {
            return riccati_solver_pass(
                step_data, terminal_Q, terminal_q,
                terminal_x_lb, terminal_x_ub,
                x0, initial_rho_x, initial_rho_u,
                (const float (*)[RICCATI_MAX_NX])z_x,
                (const float (*)[RICCATI_MAX_NX])y_x,
                (const float (*)[RICCATI_MAX_NU])z_u,
                (const float (*)[RICCATI_MAX_NU])y_u,
                solution->x, solution->u);
        };
        if (!cold_riccati_seed(0.0f, 0.0f) &&
            !cold_riccati_seed(rho, rho_u)) {
            solution->status = RICCATI_STATUS_ERROR;
            solution->iterations = 0;
            solution->primal_residual = INFINITY;
            solution->dual_residual = INFINITY;
#ifdef MPC_ENABLE_RICCATI_PROFILE
            const uint64_t cold_init_finish_ns = riccati_profile_now_ns();
            if (cold_init_finish_ns >= cold_init_start_ns)
                g_riccati_debug_last.cold_start_initialization_time_ns +=
                    cold_init_finish_ns - cold_init_start_ns;
#endif
            return RICCATI_STATUS_ERROR;
        }

        /* Initialize z from projection of unconstrained solution */
        for (int k = 0; k <= N; k++) {
            float polygon_projection[2] = {0.f, 0.f};
            const int use_polygon = k > 0 && config &&
                config->state_polygons[k].count > 0;
            if (use_polygon) {
                const float lower[2] = {
                    k < N ? step_data[k].x_lb[0] : terminal_x_lb[0],
                    k < N ? step_data[k].x_lb[1] : terminal_x_lb[1]};
                const float upper[2] = {
                    k < N ? step_data[k].x_ub[0] : terminal_x_ub[0],
                    k < N ? step_data[k].x_ub[1] : terminal_x_ub[1]};
                const float metric[2] = {rho_state[k][0], rho_state[k][1]};
                if (!riccati_project_state_polygon(&config->state_polygons[k],
                        solution->x[k], lower, upper, metric, polygon_projection)) {
                    solution->status = RICCATI_STATUS_ERROR;
                    return RICCATI_STATUS_ERROR;
                }
            }
            for (int s = 0; s < nx; s++) {
                float val = solution->x[k][s];
                const float xlb = (k < N) ? step_data[k].x_lb[s] : terminal_x_lb[s];
                const float xub = (k < N) ? step_data[k].x_ub[s] : terminal_x_ub[s];
                if (x_is_constrained[k][s]) {
                    if (use_polygon && s < 2) val = polygon_projection[s];
                    else {
                        if (val < xlb) val = xlb;
                        if (val > xub) val = xub;
                    }
                }
                z_x[k][s] = val;
            }
        }
        for (int k = 0; k < N; k++) {
            for (int a = 0; a < nu; a++) {
                float val = solution->u[k][a];
                if (val < step_data[k].u_lb[a]) val = step_data[k].u_lb[a];
                if (val > step_data[k].u_ub[a]) val = step_data[k].u_ub[a];
                z_u[k][a] = val;
            }
        }

        /* Start scaled duals at zero.  The projected primal warm seed already
         * supplies a useful feasible initialization; copying x-z into y would
         * inject an arbitrary nonzero multiplier on every cold start. */
        memset(y_x, 0, sizeof(admm_state->y_x));
        memset(y_u, 0, sizeof(admm_state->y_u));

        if (g_riccati_debug_trace_enabled &&
            g_riccati_debug_trace_count < RICCATI_DEBUG_TRACE_MAX) {
            RiccatiDebugIterSample_t *sample =
                &g_riccati_debug_trace[g_riccati_debug_trace_count++];
            memset(sample, 0, sizeof(*sample));
            sample->iter = -1;
            sample->rho = 0.0f;
            sample->rho_u = 0.0f;
            sample->u0_steer = solution->u[0][0];
            sample->u0_throttle = solution->u[0][1];
            sample->z0_steer = z_u[0][0];
            sample->z0_throttle = z_u[0][1];
            sample->y0_steer = y_u[0][0];
            sample->y0_throttle = y_u[0][1];
        }
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t cold_init_finish_ns = riccati_profile_now_ns();
        if (cold_init_finish_ns >= cold_init_start_ns)
            g_riccati_debug_last.cold_start_initialization_time_ns +=
                cold_init_finish_ns - cold_init_start_ns;
#endif
    }

    /* Terminal planes are relinearized each QP. Initialize their split from
     * the current primal seed, with zero multipliers; never reuse multipliers
     * from different planes across SQP passes. */
    for (int f=0;f<terminal.count;++f) {
        double value=0.;
        for(int i=0;i<nx;++i)value+=terminal.normal[f][i]*z_x[N][i];
        terminal_z[f]=fminf((float)value,terminal.upper[f]);
    }

    RiccatiStatus_t status = RICCATI_STATUS_MAX_ITERATIONS;

    for (int iter = 0; iter < max_iter; iter++) {

        /*--- Primal update: Riccati pass with augmented costs ---*/
        float augmented_terminal_q[nx];
        memcpy(augmented_terminal_q,terminal_q,sizeof(augmented_terminal_q));
        for (int f=0;f<terminal.count;++f)
            for (int i=0;i<nx;++i)
                augmented_terminal_q[i] -= terminal.rho*terminal.normal[f][i]*
                    (terminal_z[f]-terminal_y[f]);
        if (!riccati_solver_primal_pass(
            step_data, terminal_Q, augmented_terminal_q, terminal_x_lb, terminal_x_ub, x0,
            (const float (*)[RICCATI_MAX_NX])rho_state,
            (const float (*)[RICCATI_MAX_NU])rho_input,
            (const float (*)[RICCATI_MAX_NX])z_x,
            (const float (*)[RICCATI_MAX_NX])y_x,
            (const float (*)[RICCATI_MAX_NU])z_u,
            (const float (*)[RICCATI_MAX_NU])y_u,
            solution->x, solution->u, &factorization,
            terminal.count ? terminal_augmentation : nullptr)) {
            riccati_admm_state_init(admm_state);
            solution->status = RICCATI_STATUS_ERROR;
            solution->iterations = (uint16_t)iter;
            solution->primal_residual = INFINITY;
            solution->dual_residual = INFINITY;
            return RICCATI_STATUS_ERROR;
        }

        /*--- Fused z-update, y-update, and residual computation ---*/
        float state_primal = 0.0f, state_dual = 0.0f;
        float ctrl_primal = 0.0f, ctrl_dual = 0.0f;
        /* Adaptive-rho balancing is dimensionless, but the production
         * convergence gate below deliberately remains the raw physical
         * residual.  A metre, radian, m/s, and rad/s residual must not be
         * compared as if they were the same channel. */
        float state_primal_balance = 0.0f, state_dual_balance = 0.0f;
        float ctrl_primal_balance = 0.0f, ctrl_dual_balance = 0.0f;
        float state_primal_balance_by_state[RICCATI_MAX_NX] = {};
        float state_dual_balance_by_state[RICCATI_MAX_NX] = {};
        float ctrl_primal_balance_by_input[RICCATI_MAX_NU] = {};
        float ctrl_dual_balance_by_input[RICCATI_MAX_NU] = {};
        int state_primal_stage = -1, state_primal_index = -1;
        int state_dual_stage = -1, state_dual_index = -1;
        int ctrl_primal_stage = -1, ctrl_primal_index = -1;
        int ctrl_dual_stage = -1, ctrl_dual_index = -1;
        /* Channel-normalized residuals influence rho adaptation only.
         * Ordinary primal/dual convergence is always calculated, but the
         * expensive per-channel divisions/maxima are unnecessary on other
         * iterations (and entirely unnecessary with fixed rho). */
        const int rho_balance_due = adaptive_rho && iter > 0 &&
            iter % adaptive_rho_update_period == 0;
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t projection_start_ns = riccati_profile_now_ns();
#endif

        /* State loop */
        for (int k = 0; k <= N; k++) {
            float polygon_projection[2] = {0.f, 0.f};
            const int use_polygon = k > 0 && config &&
                config->state_polygons[k].count > 0;
            if (use_polygon) {
                const float input[2] = {
                    over_relaxation*solution->x[k][0] +
                        (1.f-over_relaxation)*z_x[k][0] + y_x[k][0],
                    over_relaxation*solution->x[k][1] +
                        (1.f-over_relaxation)*z_x[k][1] + y_x[k][1]};
                const float lower[2] = {
                    k < N ? step_data[k].x_lb[0] : terminal_x_lb[0],
                    k < N ? step_data[k].x_lb[1] : terminal_x_lb[1]};
                const float upper[2] = {
                    k < N ? step_data[k].x_ub[0] : terminal_x_ub[0],
                    k < N ? step_data[k].x_ub[1] : terminal_x_ub[1]};
                const float metric[2] = {rho_state[k][0], rho_state[k][1]};
                if (!riccati_project_state_polygon(&config->state_polygons[k],
                        input, lower, upper, metric, polygon_projection)) {
                    solution->status = RICCATI_STATUS_ERROR;
                    solution->iterations = (uint16_t)iter;
                    return RICCATI_STATUS_ERROR;
                }
            }
            for (int s = 0; s < nx; s++) {
                if (x_is_constrained[k][s]) {
                    float x_val = solution->x[k][s];
                    float x_hat = over_relaxation * x_val +
                        (1.0f - over_relaxation) * z_x[k][s];
                    float val = x_hat + y_x[k][s];
                    const float xlb = (k < N) ? step_data[k].x_lb[s] : terminal_x_lb[s];
                    const float xub = (k < N) ? step_data[k].x_ub[s] : terminal_x_ub[s];

                    /* z-update: hard constraint projection (box clipping) */
                    if (use_polygon && s < 2) val = polygon_projection[s];
                    else {
                        if (val < xlb) val = xlb;
                        if (val > xub) val = xub;
                    }

                    float z_new = val;
                    /* Dual residual */
                    float z_prev = z_x[k][s];
                    float dd = fabsf(rho_state[k][s] *
                        (z_new - z_prev));
                    if (dd > state_dual) {
                        state_dual = dd;
                        state_dual_stage = k;
                        state_dual_index = s;
                    }
                    if (rho_balance_due) {
                        const float span = fmaxf(xub-xlb, 1.0e-3f);
                        const float dual_balance = dd / span;
                        state_dual_balance = fmaxf(
                            state_dual_balance, dual_balance);
                        state_dual_balance_by_state[s] = fmaxf(
                            state_dual_balance_by_state[s], dual_balance);
                    }
                    /* y-update: y += x_hat - z.  x_hat is the standard ADMM
                     * over-relaxed primal point; with alpha=1 this is the
                     * original unrelaxed update. */
                    y_x[k][s] = x_hat - z_new + y_x[k][s];
                    /* Primal residual */
                    float pd = fabsf(solution->x[k][s] - z_new);
                    if (pd > state_primal) {
                        state_primal = pd;
                        state_primal_stage = k;
                        state_primal_index = s;
                    }
                    if (rho_balance_due) {
                        const float span = fmaxf(xub-xlb, 1.0e-3f);
                        const float primal_balance = pd / span;
                        state_primal_balance = fmaxf(
                            state_primal_balance, primal_balance);
                        state_primal_balance_by_state[s] = fmaxf(
                            state_primal_balance_by_state[s], primal_balance);
                    }
                    z_x[k][s] = z_new;
                } else {
                    z_x[k][s] = solution->x[k][s];
                }
            }
        }

        /* Control loop */
        for (int k = 0; k < N; k++) {
            const RiccatiStepData_t *sd = &step_data[k];
            for (int a = 0; a < nu; a++) {
                float u_val = solution->u[k][a];
                float u_hat = over_relaxation * u_val +
                    (1.0f - over_relaxation) * z_u[k][a];
                /* z-update: z = clip(u_hat + y, lb, ub) */
                float val = u_hat + y_u[k][a];
                if (val < sd->u_lb[a]) val = sd->u_lb[a];
                if (val > sd->u_ub[a]) val = sd->u_ub[a];
                float z_new = val;
                /* Dual residual */
                float z_prev = z_u[k][a];
                float dd = fabsf(rho_input[k][a] * (z_new - z_prev));
                if (dd > ctrl_dual) {
                    ctrl_dual = dd;
                    ctrl_dual_stage = k;
                    ctrl_dual_index = a;
                }
                if (rho_balance_due) {
                    const float span = fmaxf(
                        sd->u_ub[a]-sd->u_lb[a], 1.0e-3f);
                    const float dual_balance = dd / span;
                    ctrl_dual_balance = fmaxf(
                        ctrl_dual_balance, dual_balance);
                    ctrl_dual_balance_by_input[a] = fmaxf(
                        ctrl_dual_balance_by_input[a], dual_balance);
                }
                /* y-update uses the same over-relaxed primal point as the
                 * state split. */
                y_u[k][a] = u_hat - z_new + y_u[k][a];
                /* Primal residual */
                float pd = fabsf(solution->u[k][a] - z_new);
                if (pd > ctrl_primal) {
                    ctrl_primal = pd;
                    ctrl_primal_stage = k;
                    ctrl_primal_index = a;
                }
                if (rho_balance_due) {
                    const float span = fmaxf(
                        sd->u_ub[a]-sd->u_lb[a], 1.0e-3f);
                    const float primal_balance = pd / span;
                    ctrl_primal_balance = fmaxf(
                        ctrl_primal_balance, primal_balance);
                    ctrl_primal_balance_by_input[a] = fmaxf(
                        ctrl_primal_balance_by_input[a], primal_balance);
                }
                z_u[k][a] = z_new;
            }
        }

        float terminal_dual[nx]{};
        for (int f=0;f<terminal.count;++f) {
            double value=0.;
            for(int i=0;i<nx;++i)value+=terminal.normal[f][i]*solution->x[N][i];
            const float previous=terminal_z[f];
            const float relaxed=over_relaxation*(float)value+
                (1.f-over_relaxation)*previous;
            const float projected=fminf(relaxed+terminal_y[f],terminal.upper[f]);
            terminal_y[f]+=relaxed-projected;
            terminal_z[f]=projected;
            const float residual=fabsf((float)value-projected);
            if(residual>state_primal) {
                state_primal=residual;state_primal_stage=N;state_primal_index=-1;
            }
            for(int i=0;i<nx;++i)
                terminal_dual[i]+=terminal.rho*terminal.normal[f][i]*(projected-previous);
        }
        for(int i=0;i<nx;++i)
            if(fabsf(terminal_dual[i])>state_dual) {
                state_dual=fabsf(terminal_dual[i]);state_dual_stage=N;state_dual_index=i;
            }

        float primal_res = state_primal > ctrl_primal ? state_primal : ctrl_primal;
        float dual_res = state_dual > ctrl_dual ? state_dual : ctrl_dual;
        const float eps_primal = residual_tolerance;
        const float eps_dual = residual_tolerance;

        solution->iterations = iter + 1;
        solution->primal_residual = primal_res;
        solution->dual_residual = dual_res;
        solution->state_primal_residual = state_primal;
        solution->state_dual_residual = state_dual;
        solution->input_primal_residual = ctrl_primal;
        solution->input_dual_residual = ctrl_dual;
        solution->state_primal_stage = state_primal_stage;
        solution->state_primal_index = state_primal_index;
        solution->state_dual_stage = state_dual_stage;
        solution->state_dual_index = state_dual_index;
        solution->input_primal_stage = ctrl_primal_stage;
        solution->input_primal_index = ctrl_primal_index;
        solution->input_dual_stage = ctrl_dual_stage;
        solution->input_dual_index = ctrl_dual_index;
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t projection_finish_ns = riccati_profile_now_ns();
        if (projection_finish_ns >= projection_start_ns)
            g_riccati_debug_last.projection_residual_time_ns +=
                projection_finish_ns - projection_start_ns;
#endif

        /* Debug output */
        if (riccati_admm_debug && (iter < 5 || iter % 50 == 0 || iter == max_iter - 1)) {
            printf("    ADMM[%3d] p=%.4f<=%.4f(s=%.4f,c=%.4f) d=%.4f<=%.4f(s=%.4f,c=%.4f) rho=%.2f rho_u=%.2f u0=[%.4f,%.3f] z0=[%.4f,%.3f] y0=[%.4f,%.3f]\n",
                   iter,
                   (double)primal_res, (double)eps_primal, (double)state_primal, (double)ctrl_primal,
                   (double)dual_res, (double)eps_dual, (double)state_dual, (double)ctrl_dual,
                   (double)rho, (double)rho_u,
                   (double)solution->u[0][0], (double)solution->u[0][1],
                   (double)z_u[0][0], (double)z_u[0][1],
                   (double)y_u[0][0], (double)y_u[0][1]);
        }

        if (g_riccati_debug_trace_enabled &&
            g_riccati_debug_trace_count < RICCATI_DEBUG_TRACE_MAX) {
            RiccatiDebugIterSample_t *sample =
                &g_riccati_debug_trace[g_riccati_debug_trace_count++];
            sample->iter = iter;
            sample->primal_residual = primal_res;
            sample->dual_residual = dual_res;
            sample->state_primal_residual = state_primal;
            sample->state_dual_residual = state_dual;
            sample->ctrl_primal_residual = ctrl_primal;
            sample->ctrl_dual_residual = ctrl_dual;
            sample->rho = rho;
            sample->rho_u = rho_u;
            sample->u0_steer = solution->u[0][0];
            sample->u0_throttle = solution->u[0][1];
            sample->z0_steer = z_u[0][0];
            sample->z0_throttle = z_u[0][1];
            sample->y0_steer = y_u[0][0];
            sample->y0_throttle = y_u[0][1];
            sample->scale_rho = 0;
            sample->scale_rho_u = 0;
        }

        if (primal_res <= eps_primal && dual_res <= eps_dual) {
            status = RICCATI_STATUS_OPTIMAL;
            break;
        }

        /*--- Adaptive rho (aligned with FPGA HLS solver) ---*/
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t rho_update_start_ns = riccati_profile_now_ns();
#endif
        if (rho_balance_due) {
            const float rho_before_adaptation = rho;
            const float rho_u_before_adaptation = rho_u;
            const float adapt_ratio_state = 2.0f;
            const float adapt_ratio_ctrl = 2.0f;
            const float adapt_ratio_shared = 5.0f;
            const float rho_min = 1.0f;
            const float rho_max = cfg_rho_max;
            int scale_rho = 0;
            int scale_rho_u = 0;
            int changed = 0;

            if (shared_rho) {
                const float primal_balance = state_primal_balance >
                    ctrl_primal_balance ? state_primal_balance :
                    ctrl_primal_balance;
                const float dual_balance = state_dual_balance >
                    ctrl_dual_balance ? state_dual_balance : ctrl_dual_balance;
                if (primal_balance > adapt_ratio_shared * dual_balance &&
                    rho < rho_max) {
                    scale_rho = 1;
                } else if (dual_balance > adapt_ratio_shared * primal_balance &&
                           rho > rho_min) {
                    scale_rho = -1;
                }
                scale_rho_u = scale_rho;
            }

            if (shared_rho && scale_rho != 0) {
                const float old = rho;
                const float updated = scale_rho > 0
                    ? fminf(rho * 1.25f, rho_max)
                    : fmaxf(rho * 0.75f, rho_min);
                if (updated != old) {
                    const float y_scale = old / updated;
                    for (int k = 0; k <= N; ++k) {
                        for (int s = 0; s < nx; ++s) {
                            if (x_is_constrained[k][s])
                                y_x[k][s] *= y_scale;
                            rho_state[k][s] = updated;
                        }
                    }
                    for (int k = 0; k < N; ++k) {
                        for (int a = 0; a < nu; ++a) {
                            y_u[k][a] *= y_scale;
                            rho_input[k][a] = updated;
                        }
                    }
                    rho = updated;
                    rho_u = updated;
                    changed = 1;
                }
            } else if (!shared_rho) {
                int direction = 0;
                int state_direction_summary = 0;
                if (per_state_rho_enabled) {
                    for (int s = 0; s < nx; ++s) {
                        direction = 0;
                        const float channel_rho = rho_state[0][s];
                        if (state_primal_balance_by_state[s] >
                                adapt_ratio_state *
                                    state_dual_balance_by_state[s] &&
                            channel_rho < rho_max) {
                            direction = 1;
                        } else if (state_dual_balance_by_state[s] >
                                       adapt_ratio_state *
                                           state_primal_balance_by_state[s] &&
                                   channel_rho > rho_min) {
                            direction = -1;
                        }
                        if (direction == 0) continue;

                        const float updated = direction > 0
                            ? fminf(channel_rho * 1.25f, rho_max)
                            : fmaxf(channel_rho * 0.75f, rho_min);
                        if (updated == channel_rho) continue;

                        const float y_scale = channel_rho / updated;
                        for (int k = 0; k <= N; ++k) {
                            if (x_is_constrained[k][s])
                                y_x[k][s] *= y_scale;
                            rho_state[k][s] = updated;
                        }
                        changed = 1;
                        if (state_direction_summary == 0)
                            state_direction_summary = direction;
                        else if (state_direction_summary != direction)
                            state_direction_summary = 2;
                    }
                    scale_rho = state_direction_summary;
                    rho = 0.0f;
                    for (int s = 0; s < nx; ++s)
                        rho = fmaxf(rho, rho_state[0][s]);
                } else {
                    if (state_primal_balance >
                            adapt_ratio_state * state_dual_balance &&
                        rho < rho_max) {
                        direction = 1;
                    } else if (state_dual_balance >
                                   adapt_ratio_state * state_primal_balance &&
                               rho > rho_min) {
                        direction = -1;
                    }
                    if (direction != 0) {
                        const float old = rho;
                        rho = direction > 0
                            ? fminf(old * 1.25f, rho_max)
                            : fmaxf(old * 0.75f, rho_min);
                        if (rho != old) {
                            const float y_scale = old / rho;
                            for (int k = 0; k <= N; ++k) {
                                for (int s = 0; s < nx; ++s) {
                                    if (x_is_constrained[k][s])
                                        y_x[k][s] *= y_scale;
                                    rho_state[k][s] = rho;
                                }
                            }
                            changed = 1;
                            scale_rho = direction;
                        }
                    }
                }

                for (int a = 0; a < nu; ++a) {
                    direction = 0;
                    const float channel_rho = rho_input[0][a];
                    if (ctrl_primal_balance_by_input[a] >
                            adapt_ratio_ctrl *
                                ctrl_dual_balance_by_input[a] &&
                        channel_rho < rho_max) {
                        direction = 1;
                    } else if (ctrl_dual_balance_by_input[a] >
                                   adapt_ratio_ctrl *
                                       ctrl_primal_balance_by_input[a] &&
                               channel_rho > rho_min) {
                        direction = -1;
                    }
                    if (direction == 0) continue;

                    const float updated = direction > 0
                        ? fminf(channel_rho * 1.25f, rho_max)
                        : fmaxf(channel_rho * 0.8f, rho_min);
                    if (updated == channel_rho) continue;

                    const float y_scale = channel_rho / updated;
                    for (int k = 0; k < N; ++k) {
                        y_u[k][a] *= y_scale;
                        rho_input[k][a] = updated;
                    }
                    changed = 1;
                    if (scale_rho_u == 0) scale_rho_u = direction;
                    else if (scale_rho_u != direction) scale_rho_u = 2;
                }
                rho_u = 0.0f;
                for (int a = 0; a < nu; ++a)
                    rho_u = fmaxf(rho_u, rho_input[0][a]);
            }

            if (shared_rho) {
                rho = rho_state[0][0];
                rho_u = rho_input[0][0];
            }

            if (g_riccati_debug_trace_enabled &&
                g_riccati_debug_trace_count > 0) {
                RiccatiDebugIterSample_t *sample =
                    &g_riccati_debug_trace[g_riccati_debug_trace_count - 1];
                sample->scale_rho = scale_rho;
                sample->scale_rho_u = scale_rho_u;
            }
            // The factorization is private to this solve. Its only mutable
            // quadratic inputs are the per-state/input rho arrays modified
            // here; an actual penalty change must trigger new factors.
            if (changed) factorization.valid = 0;
            if (changed || rho != rho_before_adaptation ||
                rho_u != rho_u_before_adaptation)
                ++g_riccati_debug_last.rho_change_count;
        }
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t rho_update_finish_ns = riccati_profile_now_ns();
        if (rho_update_finish_ns >= rho_update_start_ns)
            g_riccati_debug_last.rho_update_time_ns +=
                rho_update_finish_ns - rho_update_start_ns;
#endif

        /* Stop only between complete ADMM iterations. The caller may use the
         * projected candidate only after its normal residual and nonlinear
         * feasibility gates; a time limit never weakens those checks. */
        if (max_solve_time_us > 0.0) {
            const uint64_t now_ns = riccati_monotonic_now_ns();
            const double elapsed_us = now_ns >= solve_start_ns
                ? (double)(now_ns - solve_start_ns) * 1.0e-3 : 0.0;
            if (elapsed_us >= max_solve_time_us) {
                status = RICCATI_STATUS_MAX_ITERATIONS;
                solution->time_budget_exhausted = 1;
                break;
            }
        }
    }

    /* Save scalar warm-start metadata. Buffers are already updated in-place. */
    admm_state->rho = rho;
    admm_state->rho_u = shared_rho ? rho : rho_u;
    memcpy(admm_state->rho_state, rho_state, sizeof(rho_state));
    memcpy(admm_state->rho_input, rho_input, sizeof(rho_input));
    admm_state->initialized = 1;

    g_riccati_debug_last.rho = rho;
    g_riccati_debug_last.rho_u = rho_u;
    g_riccati_debug_last.primal_residual = solution->primal_residual;
    g_riccati_debug_last.dual_residual = solution->dual_residual;

    /* Preserve the unconstrained Riccati primal only for the development
     * fixture path, then return the normal ADMM-projected controls. */
    if (primal_u_out)
        memcpy(primal_u_out, solution->u, sizeof(admm_state->z_u));
    if (feedback_out)
        memcpy(feedback_out, factorization.K, sizeof(factorization.K));

    /* Return the ADMM projection z_u (not the primal u) as the feasible control,
     * because z_u is guaranteed to satisfy box constraints whereas u may not be. */
    memcpy(solution->u, z_u, sizeof(admm_state->z_u));

    solution->status = status;
    return status;
}


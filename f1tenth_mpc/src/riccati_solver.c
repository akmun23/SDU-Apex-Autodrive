/**
 * @file riccati_solver.c
 * @brief Riccati-ADMM Solver Implementation
 * @details Solves constrained LQR using ADMM with Riccati recursion for the
 *          unconstrained sub-problem. Each ADMM iteration is O(N × nx^3).
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

#include "riccati_solver.h"

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
static uint64_t riccati_profile_now_ns(void)
{
    return riccati_monotonic_now_ns();
}
#endif

/* Debug flag: set to 1 from tests to print ADMM iteration details */
int riccati_admm_debug = 0;
static RiccatiDebugInfo_t g_riccati_debug_last;
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
        const int source = (k + 1 <= horizon) ? k + 1 : horizon;
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
 * 2x2 Matrix Inverse (for S = R + B^T P B)
 *===========================================================================*/

/* The production problem always has two controls. Symmetrize the 2x2 Schur
 * complement and, only when necessary, add bounded diagonal regularization. */
static int invert_regularized_control_hessian(float S[2][2], float Si[2][2])
{
    const float off_diagonal = 0.5f * (S[0][1] + S[1][0]);
    S[0][1] = off_diagonal;
    S[1][0] = off_diagonal;

    const float scale = fmaxf(1.0f, fmaxf(fabsf(S[0][0]),
        fmaxf(fabsf(S[1][1]), fabsf(S[0][1]))));
    const float max_lambda = 1.0e-2f * scale;
    float lambda = 0.0f;
    for (int attempt = 0; attempt < 8; ++attempt) {
        const float s00 = S[0][0] + lambda;
        const float s11 = S[1][1] + lambda;
        const float determinant = s00 * s11 - S[0][1] * S[0][1];
        const float threshold = 1.0e-10f * scale * scale;
        const int positive_definite = isfinite(s00) && isfinite(determinant) &&
            isfinite(s11) && s00 > 1.0e-10f * scale &&
            s11 > 1.0e-10f * scale && determinant > threshold;
        if (positive_definite) {
            const float inverse_determinant = 1.0f / determinant;
            Si[0][0] = s11 * inverse_determinant;
            Si[0][1] = -S[0][1] * inverse_determinant;
            Si[1][0] = Si[0][1];
            Si[1][1] = s00 * inverse_determinant;
            if (lambda > 0.0f) {
                ++g_riccati_debug_last.control_hessian_regularization_count;
                if (lambda >
                    g_riccati_debug_last.max_control_hessian_regularization) {
                    g_riccati_debug_last.max_control_hessian_regularization =
                        lambda;
                }
            }
            return isfinite(Si[0][0]) && isfinite(Si[0][1]) &&
                isfinite(Si[1][0]) && isfinite(Si[1][1]);
        }
        if (lambda >= max_lambda) break;
        lambda = lambda == 0.0f ? 1.0e-7f * scale : 10.0f * lambda;
        if (lambda > max_lambda) lambda = max_lambda;
    }

    g_riccati_debug_last.last_fallback_s00 = S[0][0] + lambda;
    g_riccati_debug_last.last_fallback_s11 = S[1][1] + lambda;
    g_riccati_debug_last.last_invert_det =
        g_riccati_debug_last.last_fallback_s00 *
        g_riccati_debug_last.last_fallback_s11 - S[0][1] * S[0][1];
    ++g_riccati_debug_last.invert_fallback_count;
    return 0;
}

/*===========================================================================
 * Riccati Backward + Forward Pass
 *===========================================================================*/

int riccati_solver_pass(
    const RiccatiStepData_t * restrict step_data,
    const float * restrict terminal_Q,
    const float * restrict terminal_q,
    const float * restrict terminal_x_lb,
    const float * restrict terminal_x_ub,
    const float * restrict x0,
    float rho,
    float rho_u,
    const float z_x[][RICCATI_MAX_NX],
    const float y_x[][RICCATI_MAX_NX],
    const float z_u[][RICCATI_MAX_NU],
    const float y_u[][RICCATI_MAX_NU],
    float x_out[][RICCATI_MAX_NX],
    float u_out[][RICCATI_MAX_NU])
{
    /* Every active state/input entry participates in the Riccati recursion;
     * no sparse state-index layout is assumed. */
    float K[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX] = {{{0}}};
    float kk[RICCATI_HORIZON][RICCATI_MAX_NU] = {{0}};
    float P[RICCATI_MAX_NX][RICCATI_MAX_NX] = {{0}};
    float p[RICCATI_MAX_NX] = {0};

    if (!step_data || !terminal_Q || !terminal_q || !terminal_x_lb ||
        !terminal_x_ub || !x0 || !z_x || !y_x || !z_u || !y_u ||
        !x_out || !u_out) {
        return 0;
    }
    enum { nx = RICCATI_MAX_NX, nu = RICCATI_MAX_NU,
           N = RICCATI_HORIZON };

    for (int i = 0; i < nx; ++i) {
        const int constrained = terminal_x_ub[i] < BIG_BOUND ||
                                terminal_x_lb[i] > -BIG_BOUND;
        P[i][i] = terminal_Q[i] + (constrained ? rho : 0.0f);
        p[i] = terminal_q[i] - (constrained ?
            rho * (z_x[N][i] - y_x[N][i]) : 0.0f);
    }

    for (int k = N - 1; k >= 0; --k) {
        const RiccatiStepData_t *sd = &step_data[k];
        float q_aug_diag[RICCATI_MAX_NX] = {0};
        float q_aug_linear[RICCATI_MAX_NX] = {0};
        float r_aug_diag[RICCATI_MAX_NU] = {0};
        float r_aug_linear[RICCATI_MAX_NU] = {0};
        float M[RICCATI_MAX_NU][RICCATI_MAX_NX] = {{0}};
        float S[2][2] = {{0}};
        float Si[2][2] = {{0}};
        float G[RICCATI_MAX_NU][RICCATI_MAX_NX] = {{0}};
        float p_shift[RICCATI_MAX_NX] = {0};
        float P_A[RICCATI_MAX_NX][RICCATI_MAX_NX] = {{0}};
        float P_next[RICCATI_MAX_NX][RICCATI_MAX_NX] = {{0}};
        float p_next[RICCATI_MAX_NX] = {0};
        float btp[RICCATI_MAX_NU] = {0};

        /* Stage cost plus ADMM's state-box quadratic. x_0 is fixed and is
         * excluded from the split penalty; its ordinary stage cost remains. */
        for (int i = 0; i < nx; ++i) {
            const int constrained = k > 0 &&
                (sd->x_ub[i] < BIG_BOUND || sd->x_lb[i] > -BIG_BOUND);
            q_aug_diag[i] = sd->Q_diag[i] + (constrained ? rho : 0.0f);
            q_aug_linear[i] = sd->q[i] - (constrained ?
                rho * (z_x[k][i] - y_x[k][i]) : 0.0f);
            if (!isfinite(q_aug_diag[i]) || !isfinite(q_aug_linear[i])) {
                return 0;
            }
        }
        for (int a = 0; a < nu; ++a) {
            r_aug_diag[a] = sd->R_diag[a] + rho_u;
            r_aug_linear[a] = sd->r[a] - rho_u * (z_u[k][a] - y_u[k][a]);
            if (!isfinite(r_aug_diag[a]) || !isfinite(r_aug_linear[a])) {
                return 0;
            }
        }

        /* M = B'P; S = R + B'PB; G = B'PA + N'. */
        for (int a = 0; a < nu; ++a) {
            for (int j = 0; j < nx; ++j) {
                for (int s = 0; s < nx; ++s) {
                    M[a][j] += sd->B[s][a] * P[s][j];
                }
            }
        }
        for (int a = 0; a < nu; ++a) {
            for (int b = 0; b < nu; ++b) {
                S[a][b] = (a == b) ? r_aug_diag[a] : 0.0f;
                for (int s = 0; s < nx; ++s) {
                    S[a][b] += M[a][s] * sd->B[s][b];
                }
            }
            for (int j = 0; j < nx; ++j) {
                float value = 0.0f;
                for (int s = 0; s < nx; ++s) {
                    value += M[a][s] * sd->A[s][j];
                }
                G[a][j] = value + sd->N[j][a];
            }
        }

        /* Preserve coupling while allowing a bounded SPD repair for tiny
         * round-off/conditioning defects.  Large indefiniteness still fails. */
        if (!invert_regularized_control_hessian(S, Si)) return 0;

        /* Affine dynamics shift the next value gradient: p_bar = p + P d. */
        for (int i = 0; i < nx; ++i) {
            p_shift[i] = p[i];
            for (int s = 0; s < nx; ++s) {
                p_shift[i] += P[i][s] * sd->d[s];
            }
        }

        /* K = -S^-1 G; kk = -S^-1(r + B'p_bar). */
        for (int a = 0; a < nu; ++a) {
            for (int s = 0; s < nx; ++s) {
                btp[a] += sd->B[s][a] * p_shift[s];
            }
        }
        for (int a = 0; a < nu; ++a) {
            for (int j = 0; j < nx; ++j) {
                for (int b = 0; b < nu; ++b) {
                    K[k][a][j] -= Si[a][b] * G[b][j];
                }
            }
            for (int b = 0; b < nu; ++b) {
                kk[k][a] -= Si[a][b] * (r_aug_linear[b] + btp[b]);
            }
        }

        /* P_A = P A.  Since P and S^-1 are symmetric, the mathematical
         * update Q + A' P A + G'K is symmetric.  Compute its upper triangle
         * once, then mirror it.  This avoids almost half of the dense
         * quadratic-form accumulation and makes symmetry exact in storage. */
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                for (int s = 0; s < nx; ++s) {
                    P_A[i][j] += P[i][s] * sd->A[s][j];
                }
            }
        }
        for (int i = 0; i < nx; ++i) {
            for (int j = i; j < nx; ++j) {
                float value = (i == j) ? q_aug_diag[i] : 0.0f;
                for (int s = 0; s < nx; ++s) {
                    value += sd->A[s][i] * P_A[s][j];
                }
                for (int a = 0; a < nu; ++a) {
                    value += G[a][i] * K[k][a][j];
                }
                P_next[i][j] = value;
                P_next[j][i] = value;
            }
        }

        /* p_k = q + A'(p + P d) + G'kk. */
        for (int i = 0; i < nx; ++i) {
            p_next[i] = q_aug_linear[i];
            for (int s = 0; s < nx; ++s) {
                p_next[i] += sd->A[s][i] * p_shift[s];
            }
            for (int a = 0; a < nu; ++a) {
                p_next[i] += G[a][i] * kk[k][a];
            }
            if (!isfinite(p_next[i])) {
                return 0;
            }
        }
        memcpy(P, P_next, sizeof(P));
        memcpy(p, p_next, sizeof(p));
    }

    /* Forward rollout uses every state row/column, input channel and affine
     * offset.  This is the same model supplied to the backward recursion. */
    for (int i = 0; i < nx; ++i) {
        x_out[0][i] = x0[i];
    }
    for (int k = 0; k < N; ++k) {
        const RiccatiStepData_t *sd = &step_data[k];
        for (int a = 0; a < nu; ++a) {
            u_out[k][a] = kk[k][a];
            for (int s = 0; s < nx; ++s) {
                u_out[k][a] += K[k][a][s] * x_out[k][s];
            }
            if (!isfinite(u_out[k][a])) {
                return 0;
            }
        }
        for (int i = 0; i < nx; ++i) {
            x_out[k + 1][i] = sd->d[i];
            for (int s = 0; s < nx; ++s) {
                x_out[k + 1][i] += sd->A[i][s] * x_out[k][s];
            }
            for (int a = 0; a < nu; ++a) {
                x_out[k + 1][i] += sd->B[i][a] * u_out[k][a];
            }
            if (!isfinite(x_out[k + 1][i])) {
                return 0;
            }
        }
    }
    return 1;
}

static int riccati_solver_factorize_diagonal(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float rho_state[][RICCATI_MAX_NX],
    const float rho_input[][RICCATI_MAX_NU],
    RiccatiFactorization_t *factorization)
{
    enum { nx = RICCATI_MAX_NX, nu = RICCATI_MAX_NU,
           N = RICCATI_HORIZON };
    RiccatiFactorization_t *const factor = factorization;
    if (!step_data || !terminal_Q || !terminal_x_lb || !terminal_x_ub ||
        !rho_state || !rho_input || !factor) {
        return 0;
    }

    memset(factor, 0, sizeof(*factor));
    factor->rho = 0.0f;
    factor->rho_u = 0.0f;
    for (int k = 0; k <= N; ++k) {
        for (int i = 0; i < nx; ++i) {
            if (!isfinite(rho_state[k][i]) || rho_state[k][i] < 0.0f)
                return 0;
            factor->rho_state[k][i] = rho_state[k][i];
            if (rho_state[k][i] > factor->rho)
                factor->rho = rho_state[k][i];
        }
    }
    for (int k = 0; k < N; ++k) {
        for (int a = 0; a < nu; ++a) {
            if (!isfinite(rho_input[k][a]) || rho_input[k][a] < 0.0f)
                return 0;
            factor->rho_input[k][a] = rho_input[k][a];
            if (rho_input[k][a] > factor->rho_u)
                factor->rho_u = rho_input[k][a];
        }
    }

    for (int i = 0; i < nx; ++i) {
        factor->x_is_constrained[N][i] =
            terminal_x_ub[i] < BIG_BOUND || terminal_x_lb[i] > -BIG_BOUND;
        factor->P[N][i][i] = terminal_Q[i] +
            (factor->x_is_constrained[N][i] ? rho_state[N][i] : 0.0f);
        if (!isfinite(factor->P[N][i][i])) return 0;
    }

    for (int k = N - 1; k >= 0; --k) {
        const RiccatiStepData_t *sd = &step_data[k];
        float q_aug_diag[RICCATI_MAX_NX] = {0};
        float r_aug_diag[RICCATI_MAX_NU] = {0};
        float M[RICCATI_MAX_NU][RICCATI_MAX_NX] = {{0}};
        float S[2][2] = {{0}};
        float P_A[RICCATI_MAX_NX][RICCATI_MAX_NX] = {{0}};

        for (int i = 0; i < nx; ++i) {
            const int constrained = k > 0 &&
                (sd->x_ub[i] < BIG_BOUND || sd->x_lb[i] > -BIG_BOUND);
            factor->x_is_constrained[k][i] = (uint8_t)constrained;
            q_aug_diag[i] = sd->Q_diag[i] +
                (constrained ? rho_state[k][i] : 0.0f);
            if (!isfinite(q_aug_diag[i])) return 0;
        }
        for (int a = 0; a < nu; ++a) {
            r_aug_diag[a] = sd->R_diag[a] + rho_input[k][a];
            if (!isfinite(r_aug_diag[a])) return 0;
        }

        for (int a = 0; a < nu; ++a) {
            for (int j = 0; j < nx; ++j) {
                for (int s = 0; s < nx; ++s)
                    M[a][j] += sd->B[s][a] * factor->P[k + 1][s][j];
            }
        }
        for (int a = 0; a < nu; ++a) {
            for (int b = 0; b < nu; ++b) {
                S[a][b] = a == b ? r_aug_diag[a] : 0.0f;
                for (int s = 0; s < nx; ++s)
                    S[a][b] += M[a][s] * sd->B[s][b];
            }
            for (int j = 0; j < nx; ++j) {
                float value = 0.0f;
                for (int s = 0; s < nx; ++s)
                    value += M[a][s] * sd->A[s][j];
                factor->G[k][a][j] = value + sd->N[j][a];
            }
        }
        if (!invert_regularized_control_hessian(
                S, factor->S_inv[k])) return 0;

        for (int a = 0; a < nu; ++a) {
            for (int j = 0; j < nx; ++j) {
                for (int b = 0; b < nu; ++b) {
                    factor->K[k][a][j] -=
                        factor->S_inv[k][a][b] * factor->G[k][b][j];
                }
            }
        }

        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                for (int s = 0; s < nx; ++s)
                    P_A[i][j] += factor->P[k + 1][i][s] * sd->A[s][j];
            }
        }
        for (int i = 0; i < nx; ++i) {
            for (int j = i; j < nx; ++j) {
                float value = i == j ? q_aug_diag[i] : 0.0f;
                for (int s = 0; s < nx; ++s)
                    value += sd->A[s][i] * P_A[s][j];
                for (int a = 0; a < nu; ++a)
                    value += factor->G[k][a][i] * factor->K[k][a][j];
                factor->P[k][i][j] = value;
                factor->P[k][j][i] = value;
            }
        }
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < nx; ++j)
                if (!isfinite(factor->P[k][i][j])) return 0;
    }

    factor->valid = 1;
    ++g_riccati_debug_last.quadratic_factorization_count;
    return 1;
}

int riccati_solver_factorize(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    float rho,
    float rho_u,
    RiccatiFactorization_t *factorization)
{
    float rho_state[RICCATI_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float rho_input[RICCATI_HORIZON][RICCATI_MAX_NU] = {{0}};
    for (int k = 0; k <= RICCATI_HORIZON; ++k)
        for (int i = 0; i < RICCATI_MAX_NX; ++i)
            rho_state[k][i] = rho;
    for (int k = 0; k < RICCATI_HORIZON; ++k)
        for (int a = 0; a < RICCATI_MAX_NU; ++a)
            rho_input[k][a] = rho_u;
    return riccati_solver_factorize_diagonal(
        step_data, terminal_Q, terminal_x_lb, terminal_x_ub,
        (const float (*)[RICCATI_MAX_NX])rho_state,
        (const float (*)[RICCATI_MAX_NU])rho_input, factorization);
}

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
    float u_out[][RICCATI_MAX_NU])
{
    enum { nx = RICCATI_MAX_NX, nu = RICCATI_MAX_NU,
           N = RICCATI_HORIZON };
    const RiccatiFactorization_t *const factor = factorization;
    if (!step_data || !terminal_q || !terminal_x_lb || !terminal_x_ub ||
        !x0 || !factor || !factor->valid || !z_x || !y_x || !z_u || !y_u ||
        !x_out || !u_out) {
        return 0;
    }

    float p[RICCATI_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float kk[RICCATI_HORIZON][RICCATI_MAX_NU] = {{0}};
    for (int i = 0; i < nx; ++i) {
        p[N][i] = terminal_q[i] -
            (factor->x_is_constrained[N][i]
                ? factor->rho_state[N][i] *
                    (z_x[N][i] - y_x[N][i]) : 0.0f);
        if (!isfinite(p[N][i])) return 0;
    }

    for (int k = N - 1; k >= 0; --k) {
        const RiccatiStepData_t *sd = &step_data[k];
        float p_shift[RICCATI_MAX_NX] = {0};
        float q_aug_linear[RICCATI_MAX_NX] = {0};
        float r_aug_linear[RICCATI_MAX_NU] = {0};
        float btp[RICCATI_MAX_NU] = {0};

        for (int i = 0; i < nx; ++i) {
            p_shift[i] = p[k + 1][i];
            for (int s = 0; s < nx; ++s)
                p_shift[i] += factor->P[k + 1][i][s] * sd->d[s];
            q_aug_linear[i] = sd->q[i] -
                (factor->x_is_constrained[k][i]
                    ? factor->rho_state[k][i] *
                        (z_x[k][i] - y_x[k][i]) : 0.0f);
            if (!isfinite(p_shift[i]) || !isfinite(q_aug_linear[i])) return 0;
        }
        for (int a = 0; a < nu; ++a) {
            r_aug_linear[a] = sd->r[a] -
                factor->rho_input[k][a] * (z_u[k][a] - y_u[k][a]);
            for (int s = 0; s < nx; ++s)
                btp[a] += sd->B[s][a] * p_shift[s];
            if (!isfinite(r_aug_linear[a]) || !isfinite(btp[a])) return 0;
        }
        for (int a = 0; a < nu; ++a) {
            for (int b = 0; b < nu; ++b)
                kk[k][a] -= factor->S_inv[k][a][b] *
                    (r_aug_linear[b] + btp[b]);
            if (!isfinite(kk[k][a])) return 0;
        }
        for (int i = 0; i < nx; ++i) {
            p[k][i] = q_aug_linear[i];
            for (int s = 0; s < nx; ++s)
                p[k][i] += sd->A[s][i] * p_shift[s];
            for (int a = 0; a < nu; ++a)
                p[k][i] += factor->G[k][a][i] * kk[k][a];
            if (!isfinite(p[k][i])) return 0;
        }
    }

    for (int i = 0; i < nx; ++i) x_out[0][i] = x0[i];
    for (int k = 0; k < N; ++k) {
        const RiccatiStepData_t *sd = &step_data[k];
        for (int a = 0; a < nu; ++a) {
            u_out[k][a] = kk[k][a];
            for (int s = 0; s < nx; ++s)
                u_out[k][a] += factor->K[k][a][s] * x_out[k][s];
            if (!isfinite(u_out[k][a])) return 0;
        }
        for (int i = 0; i < nx; ++i) {
            x_out[k + 1][i] = sd->d[i];
            for (int s = 0; s < nx; ++s)
                x_out[k + 1][i] += sd->A[i][s] * x_out[k][s];
            for (int a = 0; a < nu; ++a)
                x_out[k + 1][i] += sd->B[i][a] * u_out[k][a];
            if (!isfinite(x_out[k + 1][i])) return 0;
        }
    }
    return 1;
}

static int riccati_solver_primal_pass(
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
    RiccatiFactorization_t *factorization)
{
    if (!factorization) return 0;
    int same_penalties = factorization->valid;
    for (int k = 0; same_penalties && k <= RICCATI_HORIZON; ++k)
        for (int i = 0; same_penalties && i < RICCATI_MAX_NX; ++i)
            same_penalties =
                factorization->rho_state[k][i] == rho_state[k][i];
    for (int k = 0; same_penalties && k < RICCATI_HORIZON; ++k)
        for (int a = 0; same_penalties && a < RICCATI_MAX_NU; ++a)
            same_penalties =
                factorization->rho_input[k][a] == rho_input[k][a];
    if (!same_penalties) {
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t start_ns = riccati_profile_now_ns();
#endif
        if (!riccati_solver_factorize_diagonal(step_data, terminal_Q,
                terminal_x_lb, terminal_x_ub, rho_state, rho_input,
                factorization)) return 0;
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t finish_ns = riccati_profile_now_ns();
        if (finish_ns >= start_ns)
            g_riccati_debug_last.quadratic_factorization_time_ns +=
                finish_ns - start_ns;
#endif
    }
 #ifdef MPC_ENABLE_RICCATI_PROFILE
    const uint64_t rhs_start_ns = riccati_profile_now_ns();
#endif
    const int solved = riccati_solver_pass_factored(
        step_data, terminal_q, terminal_x_lb, terminal_x_ub, x0,
        factorization, z_x, y_x, z_u, y_u, x_out, u_out);
#ifdef MPC_ENABLE_RICCATI_PROFILE
    const uint64_t rhs_finish_ns = riccati_profile_now_ns();
    if (rhs_finish_ns >= rhs_start_ns)
        g_riccati_debug_last.quadratic_rhs_time_ns +=
            rhs_finish_ns - rhs_start_ns;
#endif
    return solved;
}

/*===========================================================================
 * Main Solver: Riccati-ADMM
 *===========================================================================*/

static RiccatiStatus_t riccati_admm_solve_core(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    const RiccatiAdmmConfig_t *config,
    RiccatiAdmmState_t *admm_state,
    RiccatiSolution_t *solution)
{
    if (!step_data || !terminal_Q || !terminal_q || !terminal_x_lb || !terminal_x_ub || !x0 || !admm_state || !solution) {
        if (solution) {
            solution->status = RICCATI_STATUS_ERROR;
        }
        return RICCATI_STATUS_ERROR;
    }

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
    RiccatiFactorization_t factorization = {0};

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

    if (admm_state->initialized) {
        riccati_admm_shift_warm_start(admm_state, x0);
    }

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

        if (!riccati_solver_pass(
            step_data, terminal_Q, terminal_q, terminal_x_lb, terminal_x_ub,
            x0, 0.0f, 0.0f,
            (const float (*)[RICCATI_MAX_NX])z_x,
            (const float (*)[RICCATI_MAX_NX])y_x,
            (const float (*)[RICCATI_MAX_NU])z_u,
            (const float (*)[RICCATI_MAX_NU])y_u,
            solution->x, solution->u)) {
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
            for (int s = 0; s < nx; s++) {
                float val = solution->x[k][s];
                const float xlb = (k < N) ? step_data[k].x_lb[s] : terminal_x_lb[s];
                const float xub = (k < N) ? step_data[k].x_ub[s] : terminal_x_ub[s];
                if (x_is_constrained[k][s]) {
                    if (val < xlb) val = xlb;
                    if (val > xub) val = xub;
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

    RiccatiStatus_t status = RICCATI_STATUS_MAX_ITERATIONS;

    for (int iter = 0; iter < max_iter; iter++) {

        /*--- Primal update: Riccati pass with augmented costs ---*/
        if (!riccati_solver_primal_pass(
            step_data, terminal_Q, terminal_q, terminal_x_lb, terminal_x_ub, x0,
            (const float (*)[RICCATI_MAX_NX])rho_state,
            (const float (*)[RICCATI_MAX_NU])rho_input,
            (const float (*)[RICCATI_MAX_NX])z_x,
            (const float (*)[RICCATI_MAX_NX])y_x,
            (const float (*)[RICCATI_MAX_NU])z_u,
            (const float (*)[RICCATI_MAX_NU])y_u,
            solution->x, solution->u, &factorization)) {
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
        float state_primal_balance_by_state[RICCATI_MAX_NX] = {0};
        float state_dual_balance_by_state[RICCATI_MAX_NX] = {0};
        float ctrl_primal_balance_by_input[RICCATI_MAX_NU] = {0};
        float ctrl_dual_balance_by_input[RICCATI_MAX_NU] = {0};
        int state_primal_stage = -1, state_primal_index = -1;
        int state_dual_stage = -1, state_dual_index = -1;
        int ctrl_primal_stage = -1, ctrl_primal_index = -1;
        int ctrl_dual_stage = -1, ctrl_dual_index = -1;
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t projection_start_ns = riccati_profile_now_ns();
#endif

        /* State loop */
        for (int k = 0; k <= N; k++) {
            for (int s = 0; s < nx; s++) {
                if (x_is_constrained[k][s]) {
                    float x_val = solution->x[k][s];
                    float x_hat = over_relaxation * x_val +
                        (1.0f - over_relaxation) * z_x[k][s];
                    float val = x_hat + y_x[k][s];
                    const float xlb = (k < N) ? step_data[k].x_lb[s] : terminal_x_lb[s];
                    const float xub = (k < N) ? step_data[k].x_ub[s] : terminal_x_ub[s];

                    /* z-update: hard constraint projection (box clipping) */
                    if (val < xlb) val = xlb;
                    if (val > xub) val = xub;

                    float z_new = val;
                    const float span = fmaxf(xub - xlb, 1.0e-3f);
                    /* Dual residual */
                    float z_prev = z_x[k][s];
                    float dd = fabsf(rho_state[k][s] *
                        (z_new - z_prev));
                    if (dd > state_dual) {
                        state_dual = dd;
                        state_dual_stage = k;
                        state_dual_index = s;
                    }
                    const float dual_balance = dd / span;
                    state_dual_balance = fmaxf(
                        state_dual_balance, dual_balance);
                    state_dual_balance_by_state[s] = fmaxf(
                        state_dual_balance_by_state[s], dual_balance);
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
                    const float primal_balance = pd / span;
                    state_primal_balance = fmaxf(
                        state_primal_balance, primal_balance);
                    state_primal_balance_by_state[s] = fmaxf(
                        state_primal_balance_by_state[s], primal_balance);
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
                const float span = fmaxf(sd->u_ub[a] - sd->u_lb[a], 1.0e-3f);
                /* Dual residual */
                float z_prev = z_u[k][a];
                float dd = fabsf(rho_input[k][a] * (z_new - z_prev));
                if (dd > ctrl_dual) {
                    ctrl_dual = dd;
                    ctrl_dual_stage = k;
                    ctrl_dual_index = a;
                }
                const float dual_balance = dd / span;
                ctrl_dual_balance = fmaxf(
                    ctrl_dual_balance, dual_balance);
                ctrl_dual_balance_by_input[a] = fmaxf(
                    ctrl_dual_balance_by_input[a], dual_balance);
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
                const float primal_balance = pd / span;
                ctrl_primal_balance = fmaxf(
                    ctrl_primal_balance, primal_balance);
                ctrl_primal_balance_by_input[a] = fmaxf(
                    ctrl_primal_balance_by_input[a], primal_balance);
                z_u[k][a] = z_new;
            }
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
        if (adaptive_rho && iter > 0 &&
            iter % adaptive_rho_update_period == 0) {
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

    /* Output feasible controls: z_u is the ADMM projection */
    /* Return the ADMM projection z_u (not the primal u) as the feasible control,
     * because z_u is guaranteed to satisfy box constraints whereas u may not be. */
    memcpy(solution->u, z_u, sizeof(admm_state->z_u));

    solution->status = status;
    return status;
}

static int riccati_scaling_signature_matches(
    const RiccatiAdmmState_t *state,
    const RiccatiAdmmConfig_t *config)
{
    enum { nx = RICCATI_MAX_NX, nu = RICCATI_MAX_NU };
    if (!state || !config || !state->initialized ||
        !state->scaling_enabled)
        return 0;
    for (int i = 0; i < nx; ++i)
        if (state->state_scale[i] != config->state_scale[i]) return 0;
    for (int a = 0; a < nu; ++a)
        if (state->input_scale[a] != config->input_scale[a]) return 0;
    return 1;
}

static float scale_state_lower_bound(float bound, float scale)
{
    /* BIG_BOUND is a semantic unconstrained marker, not a physical limit.
     * Dividing it by D can move it inside the core solver's BIG_BOUND test
     * and accidentally constrain scaled wheel/body states. */
    return bound <= -BIG_BOUND ? -BIG_BOUND : bound / scale;
}

static float scale_state_upper_bound(float bound, float scale)
{
    return bound >= BIG_BOUND ? BIG_BOUND : bound / scale;
}

RiccatiStatus_t riccati_admm_solve(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    const RiccatiAdmmConfig_t *config,
    RiccatiAdmmState_t *admm_state,
    RiccatiSolution_t *solution)
{
    enum { nx = RICCATI_MAX_NX, nu = RICCATI_MAX_NU,
           N = RICCATI_HORIZON };
    if (!admm_state || !solution || !step_data) {
        if (solution) solution->status = RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }
    for (int k = 0; k < RICCATI_HORIZON; ++k)
        for (int i = 0; i < RICCATI_MAX_NX; ++i)
            for (int a = 0; a < RICCATI_MAX_NU; ++a)
                if (!isfinite(step_data[k].N[i][a])) {
                    solution->status = RICCATI_STATUS_ERROR;
                    return RICCATI_STATUS_ERROR;
                }

    const int use_scaling = config ? config->use_scaling : 0;
    if (use_scaling != 0 && use_scaling != 1) {
        solution->status = RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }
    if (!use_scaling) {
        if (admm_state->initialized && admm_state->scaling_enabled)
            riccati_admm_state_init(admm_state);
        return riccati_admm_solve_core(
            step_data, terminal_Q, terminal_q, terminal_x_lb, terminal_x_ub,
            x0, config, admm_state, solution);
    }

    if (!step_data || !terminal_Q || !terminal_q || !terminal_x_lb ||
        !terminal_x_ub || !x0) {
        solution->status = RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }
    for (int i = 0; i < nx; ++i) {
        if (!isfinite(config->state_scale[i]) ||
            config->state_scale[i] <= 0.0f) {
            solution->status = RICCATI_STATUS_ERROR;
            return RICCATI_STATUS_ERROR;
        }
    }
    for (int a = 0; a < nu; ++a) {
        if (!isfinite(config->input_scale[a]) ||
            config->input_scale[a] <= 0.0f) {
            solution->status = RICCATI_STATUS_ERROR;
            return RICCATI_STATUS_ERROR;
        }
    }

    if (admm_state->initialized &&
        !riccati_scaling_signature_matches(admm_state, config))
        riccati_admm_state_init(admm_state);

    /* x = Dx*xhat and u = Du*uhat.  Scale dynamics, all diagonal and
     * affine costs, and hard bounds consistently. */
#ifdef MPC_ENABLE_RICCATI_PROFILE
    const uint64_t scaling_start_ns = riccati_profile_now_ns();
#endif
    RiccatiStepData_t scaled_steps[RICCATI_HORIZON] = {0};
    float scaled_terminal_Q[RICCATI_MAX_NX] = {0};
    float scaled_terminal_q[RICCATI_MAX_NX] = {0};
    float scaled_terminal_lb[RICCATI_MAX_NX] = {0};
    float scaled_terminal_ub[RICCATI_MAX_NX] = {0};
    float scaled_x0[RICCATI_MAX_NX] = {0};
    for (int i = 0; i < nx; ++i) {
        const float scale = config->state_scale[i];
        scaled_terminal_Q[i] = terminal_Q[i] * scale * scale;
        scaled_terminal_q[i] = terminal_q[i] * scale;
        scaled_terminal_lb[i] = scale_state_lower_bound(
            terminal_x_lb[i], scale);
        scaled_terminal_ub[i] = scale_state_upper_bound(
            terminal_x_ub[i], scale);
        scaled_x0[i] = x0[i] / scale;
    }
    memcpy(scaled_steps, step_data, (size_t)N * sizeof(scaled_steps[0]));
    for (int k = 0; k < N; ++k) {
        RiccatiStepData_t *scaled = &scaled_steps[k];
        const RiccatiStepData_t *physical = &step_data[k];
        for (int i = 0; i < nx; ++i) {
            const float state_scale = config->state_scale[i];
            scaled->d[i] = physical->d[i] / state_scale;
            scaled->Q_diag[i] = physical->Q_diag[i] *
                state_scale * state_scale;
            scaled->q[i] = physical->q[i] * state_scale;
            for (int a = 0; a < nu; ++a) {
                scaled->N[i][a] = physical->N[i][a] * state_scale *
                    config->input_scale[a];
            }
            scaled->x_lb[i] = scale_state_lower_bound(
                physical->x_lb[i], state_scale);
            scaled->x_ub[i] = scale_state_upper_bound(
                physical->x_ub[i], state_scale);
            for (int j = 0; j < nx; ++j) {
                scaled->A[i][j] = physical->A[i][j] *
                    config->state_scale[j] / state_scale;
            }
            for (int a = 0; a < nu; ++a) {
                scaled->B[i][a] = physical->B[i][a] *
                    config->input_scale[a] / state_scale;
            }
        }
        for (int a = 0; a < nu; ++a) {
            const float input_scale = config->input_scale[a];
            scaled->R_diag[a] = physical->R_diag[a] *
                input_scale * input_scale;
            scaled->r[a] = physical->r[a] * input_scale;
            scaled->u_lb[a] = physical->u_lb[a] / input_scale;
            scaled->u_ub[a] = physical->u_ub[a] / input_scale;
        }
    }

    RiccatiAdmmConfig_t scaled_config = *config;
    scaled_config.use_scaling = 0;
    RiccatiSolution_t scaled_solution = {0};
#ifdef MPC_ENABLE_RICCATI_PROFILE
    const uint64_t scaling_finish_ns = riccati_profile_now_ns();
#endif
    const RiccatiStatus_t status = riccati_admm_solve_core(
        scaled_steps, scaled_terminal_Q, scaled_terminal_q,
        scaled_terminal_lb, scaled_terminal_ub, scaled_x0,
        &scaled_config, admm_state, &scaled_solution);
    *solution = scaled_solution;
#ifdef MPC_ENABLE_RICCATI_PROFILE
    const uint64_t output_unscale_start_ns = riccati_profile_now_ns();
#endif
    for (int k = 0; k <= N; ++k)
        for (int i = 0; i < nx; ++i)
            solution->x[k][i] *= config->state_scale[i];
    for (int k = 0; k < N; ++k)
        for (int a = 0; a < nu; ++a)
            solution->u[k][a] *= config->input_scale[a];

    if (admm_state->initialized) {
        admm_state->scaling_enabled = 1;
        memcpy(admm_state->state_scale, config->state_scale,
               (size_t)nx * sizeof(config->state_scale[0]));
        memcpy(admm_state->input_scale, config->input_scale,
               (size_t)nu * sizeof(config->input_scale[0]));
    }
#ifdef MPC_ENABLE_RICCATI_PROFILE
    const uint64_t output_unscale_finish_ns = riccati_profile_now_ns();
    if (scaling_finish_ns >= scaling_start_ns)
        g_riccati_debug_last.scaling_transform_time_ns +=
            scaling_finish_ns - scaling_start_ns;
    if (output_unscale_finish_ns >= output_unscale_start_ns)
        g_riccati_debug_last.output_unscale_time_ns +=
            output_unscale_finish_ns - output_unscale_start_ns;
#endif
    return status;
}

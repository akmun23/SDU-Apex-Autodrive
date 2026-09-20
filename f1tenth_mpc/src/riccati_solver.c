/**
 * @file riccati_solver.c
 * @brief Riccati-ADMM Solver Implementation
 * @details Solves constrained LQR using ADMM with Riccati recursion for the
 *          unconstrained sub-problem. Each ADMM iteration is O(N × nx^3).
 *
 * Riccati backward pass (per step):
 *   M    = B^T P_{k+1}           (nu×nx)
 *   S    = R + M B               (nu×nu, invert via 2×2 formula)
 *   G    = M A + N^T             (nu×nx, includes cross-cost)
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

#ifdef MPC_ENABLE_RICCATI_PROFILE
#include <time.h>
#endif

/* Debug flag: set to 1 from tests to print ADMM iteration details */
int riccati_admm_debug = 0;
static RiccatiDebugInfo_t g_riccati_debug_last;
static RiccatiDebugIterSample_t g_riccati_debug_trace[RICCATI_DEBUG_TRACE_MAX];
static int g_riccati_debug_trace_count = 0;
static int g_riccati_debug_trace_enabled = 0;

#ifdef MPC_ENABLE_RICCATI_PROFILE
static uint64_t riccati_profile_now_ns(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}
#endif

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
    int nx,
    int nu,
    int horizon,
    const float *new_x0)
{
    if (!state || !state->initialized || !new_x0 ||
        nx <= 0 || nx > RICCATI_MAX_NX ||
        nu <= 0 || nu > RICCATI_MAX_NU ||
        horizon <= 0 || horizon > PREDICTION_HORIZON) {
        return;
    }

    for (int k = 0; k < horizon; ++k) {
        const int source = (k + 1 <= horizon) ? k + 1 : horizon;
        for (int i = 0; i < nx; ++i) {
            state->z_x[k][i] = state->z_x[source][i];
            state->y_x[k][i] = state->y_x[source][i];
        }
    }
    /* Hold the prior terminal prediction at the new horizon endpoint. */
    for (int i = 0; i < nx; ++i) {
        state->z_x[horizon][i] = state->z_x[horizon - 1][i];
        state->y_x[horizon][i] = state->y_x[horizon - 1][i];
    }
    for (int k = 0; k < horizon - 1; ++k) {
        for (int a = 0; a < nu; ++a) {
            state->z_u[k][a] = state->z_u[k + 1][a];
            state->y_u[k][a] = state->y_u[k + 1][a];
        }
    }
    if (horizon > 1) {
        for (int a = 0; a < nu; ++a) {
            state->z_u[horizon - 1][a] = state->z_u[horizon - 2][a];
            state->y_u[horizon - 1][a] = state->y_u[horizon - 2][a];
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

int riccati_invert_2x2(
    float S[2][2],
    float Si[2][2])
{
    float det = S[0][0] * S[1][1] - S[0][1] * S[1][0];

    if (fabsf(det) < 1e-10f) {
        /* Singular or near-singular */
        g_riccati_debug_last.invert_fallback_count++;
        g_riccati_debug_last.last_invert_det = det;
        return -1;
    }

    float inv_det = 1.0f / det;
    Si[0][0] =  S[1][1] * inv_det;
    Si[0][1] = -S[0][1] * inv_det;
    Si[1][0] = -S[1][0] * inv_det;
    Si[1][1] =  S[0][0] * inv_det;

    return 0;
}

/* Symmetrize and, only when necessary, add bounded diagonal regularization
 * before inverting the active 1x1 or 2x2 control Hessian. */
static int invert_regularized_control_hessian(
    float S[2][2], int nu, float Si[2][2])
{
    if (nu < 1 || nu > 2) return 0;
    if (nu == 2) {
        const float off_diagonal = 0.5f * (S[0][1] + S[1][0]);
        S[0][1] = off_diagonal;
        S[1][0] = off_diagonal;
    }

    const float scale = fmaxf(1.0f, fmaxf(fabsf(S[0][0]),
        nu == 2 ? fmaxf(fabsf(S[1][1]), fabsf(S[0][1])) : 0.0f));
    const float max_lambda = 1.0e-2f * scale;
    float lambda = 0.0f;
    for (int attempt = 0; attempt < 8; ++attempt) {
        const float s00 = S[0][0] + lambda;
        const float s11 = nu == 2 ? S[1][1] + lambda : 1.0f;
        const float determinant = nu == 2
            ? s00 * s11 - S[0][1] * S[0][1]
            : s00;
        const float threshold = 1.0e-10f * scale *
            (nu == 2 ? scale : 1.0f);
        const int positive_definite = isfinite(s00) && isfinite(determinant) &&
            s00 > 1.0e-10f * scale &&
            (nu == 1 || (isfinite(s11) && s11 > 1.0e-10f * scale &&
                         determinant > threshold));
        if (positive_definite) {
            if (nu == 1) {
                Si[0][0] = 1.0f / s00;
            } else {
                const float inverse_determinant = 1.0f / determinant;
                Si[0][0] = s11 * inverse_determinant;
                Si[0][1] = -S[0][1] * inverse_determinant;
                Si[1][0] = Si[0][1];
                Si[1][1] = s00 * inverse_determinant;
            }
            if (lambda > 0.0f) {
                ++g_riccati_debug_last.control_hessian_regularization_count;
                if (lambda >
                    g_riccati_debug_last.max_control_hessian_regularization) {
                    g_riccati_debug_last.max_control_hessian_regularization =
                        lambda;
                }
            }
            return isfinite(Si[0][0]) && isfinite(Si[0][1]) &&
                (nu == 1 || (isfinite(Si[1][0]) && isfinite(Si[1][1])));
        }
        if (lambda >= max_lambda) break;
        lambda = lambda == 0.0f ? 1.0e-7f * scale : 10.0f * lambda;
        if (lambda > max_lambda) lambda = max_lambda;
    }

    g_riccati_debug_last.last_fallback_s00 = S[0][0] + lambda;
    g_riccati_debug_last.last_fallback_s11 = nu == 2
        ? S[1][1] + lambda : 0.0f;
    g_riccati_debug_last.last_invert_det = nu == 2
        ? g_riccati_debug_last.last_fallback_s00 *
              g_riccati_debug_last.last_fallback_s11 - S[0][1] * S[0][1]
        : g_riccati_debug_last.last_fallback_s00;
    ++g_riccati_debug_last.invert_fallback_count;
    return 0;
}

enum
{
    ACTIVE_RICCATI_NX = 9,
    ACTIVE_RICCATI_PLANT_NX = 7,
    ACTIVE_RICCATI_NU = 2
};

/* The production RTI wrapper augments a 7-state plant with two exact
 * previous-input memory states. Recognize that layout conservatively so the
 * generic public solver remains available for arbitrary dense test problems. */
static int active_augmented_structure(
    const RiccatiStepData_t *steps,
    int horizon)
{
    if (!steps || horizon <= 0 || horizon > PREDICTION_HORIZON) return 0;
    for (int k = 0; k < horizon; ++k) {
        const RiccatiStepData_t *sd = &steps[k];
        for (int row = 0; row < ACTIVE_RICCATI_PLANT_NX; ++row) {
            if (sd->A[row][ACTIVE_RICCATI_PLANT_NX] != 0.0f ||
                sd->A[row][ACTIVE_RICCATI_PLANT_NX + 1] != 0.0f)
                return 0;
        }
        for (int row = ACTIVE_RICCATI_PLANT_NX; row < ACTIVE_RICCATI_NX; ++row)
            for (int column = 0; column < ACTIVE_RICCATI_NX; ++column)
                if (sd->A[row][column] != 0.0f) return 0;

        if (sd->B[ACTIVE_RICCATI_PLANT_NX][0] != 1.0f ||
            sd->B[ACTIVE_RICCATI_PLANT_NX][1] != 0.0f ||
            sd->B[ACTIVE_RICCATI_PLANT_NX + 1][0] != 0.0f ||
            sd->B[ACTIVE_RICCATI_PLANT_NX + 1][1] != 1.0f)
            return 0;
        if (sd->d[ACTIVE_RICCATI_PLANT_NX] != 0.0f ||
            sd->d[ACTIVE_RICCATI_PLANT_NX + 1] != 0.0f)
            return 0;
        for (int row = 0; row < ACTIVE_RICCATI_PLANT_NX; ++row) {
            if (sd->N[row][0] != 0.0f || sd->N[row][1] != 0.0f)
                return 0;
        }
        if (sd->N[ACTIVE_RICCATI_PLANT_NX][0] == 0.0f ||
            sd->N[ACTIVE_RICCATI_PLANT_NX][1] != 0.0f ||
            sd->N[ACTIVE_RICCATI_PLANT_NX + 1][0] != 0.0f ||
            sd->N[ACTIVE_RICCATI_PLANT_NX + 1][1] == 0.0f)
            return 0;
    }
    return 1;
}

static void active_form_control_terms(
    const RiccatiStepData_t *sd,
    float P[RICCATI_MAX_NX][RICCATI_MAX_NX],
    const float r_aug_diag[ACTIVE_RICCATI_NU],
    float M[ACTIVE_RICCATI_NU][RICCATI_MAX_NX],
    float S[ACTIVE_RICCATI_NU][ACTIVE_RICCATI_NU],
    float G[ACTIVE_RICCATI_NU][RICCATI_MAX_NX])
{
    for (int a = 0; a < ACTIVE_RICCATI_NU; ++a) {
        for (int j = 0; j < ACTIVE_RICCATI_NX; ++j) {
            for (int s = 0; s < ACTIVE_RICCATI_PLANT_NX; ++s)
                M[a][j] += sd->B[s][a] * P[s][j];
            /* The two memory rows of B are an identity matrix. */
            M[a][j] += P[ACTIVE_RICCATI_PLANT_NX + a][j];
        }
    }
    for (int a = 0; a < ACTIVE_RICCATI_NU; ++a) {
        for (int b = 0; b < ACTIVE_RICCATI_NU; ++b) {
            S[a][b] = a == b ? r_aug_diag[a] : 0.0f;
            for (int s = 0; s < ACTIVE_RICCATI_PLANT_NX; ++s)
                S[a][b] += M[a][s] * sd->B[s][b];
            S[a][b] += M[a][ACTIVE_RICCATI_PLANT_NX + b];
        }
        for (int j = 0; j < ACTIVE_RICCATI_NX; ++j) {
            G[a][j] = sd->N[j][a];
            if (j < ACTIVE_RICCATI_PLANT_NX) {
                for (int s = 0; s < ACTIVE_RICCATI_PLANT_NX; ++s)
                    G[a][j] += sd->A[s][j] * M[a][s];
            }
        }
    }
}

static void active_form_p_a(
    const RiccatiStepData_t *sd,
    float P[RICCATI_MAX_NX][RICCATI_MAX_NX],
    float P_A[RICCATI_MAX_NX][RICCATI_MAX_NX])
{
    for (int i = 0; i < ACTIVE_RICCATI_NX; ++i) {
        for (int j = 0; j < ACTIVE_RICCATI_PLANT_NX; ++j) {
            for (int s = 0; s < ACTIVE_RICCATI_PLANT_NX; ++s)
                P_A[i][j] += P[i][s] * sd->A[s][j];
        }
    }
}

static void active_update_p(
    const RiccatiStepData_t *sd,
    const float q_aug_diag[RICCATI_MAX_NX],
    float P_A[RICCATI_MAX_NX][RICCATI_MAX_NX],
    float G[ACTIVE_RICCATI_NU][RICCATI_MAX_NX],
    float K[ACTIVE_RICCATI_NU][RICCATI_MAX_NX],
    float P_next[RICCATI_MAX_NX][RICCATI_MAX_NX])
{
    for (int i = 0; i < ACTIVE_RICCATI_NX; ++i) {
        for (int j = 0; j < ACTIVE_RICCATI_NX; ++j) {
            P_next[i][j] = i == j ? q_aug_diag[i] : 0.0f;
            if (i < ACTIVE_RICCATI_PLANT_NX &&
                j < ACTIVE_RICCATI_PLANT_NX) {
                for (int s = 0; s < ACTIVE_RICCATI_PLANT_NX; ++s)
                    P_next[i][j] += sd->A[s][i] * P_A[s][j];
            }
            for (int a = 0; a < ACTIVE_RICCATI_NU; ++a)
                P_next[i][j] += G[a][i] * K[a][j];
        }
    }
}

static void active_state_affine_shift(
    const float *P,
    const float d[RICCATI_MAX_NX],
    const float p_next[RICCATI_MAX_NX],
    float p_shift[RICCATI_MAX_NX])
{
    for (int i = 0; i < ACTIVE_RICCATI_NX; ++i) {
        p_shift[i] = p_next[i];
        for (int s = 0; s < ACTIVE_RICCATI_PLANT_NX; ++s)
            p_shift[i] += P[i * RICCATI_MAX_NX + s] * d[s];
    }
}

static void active_b_transpose_p(
    const RiccatiStepData_t *sd,
    const float p_shift[RICCATI_MAX_NX],
    float btp[ACTIVE_RICCATI_NU])
{
    for (int a = 0; a < ACTIVE_RICCATI_NU; ++a) {
        for (int s = 0; s < ACTIVE_RICCATI_PLANT_NX; ++s)
            btp[a] += sd->B[s][a] * p_shift[s];
        btp[a] += p_shift[ACTIVE_RICCATI_PLANT_NX + a];
    }
}

static int riccati_solver_pass_active(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    int N,
    float rho,
    float rho_u,
    const float z_x[][RICCATI_MAX_NX],
    const float y_x[][RICCATI_MAX_NX],
    const float z_u[][RICCATI_MAX_NU],
    const float y_u[][RICCATI_MAX_NU],
    float x_out[][RICCATI_MAX_NX],
    float u_out[][RICCATI_MAX_NU])
{
    float K[PREDICTION_HORIZON][ACTIVE_RICCATI_NU][RICCATI_MAX_NX] = {{{0}}};
    float kk[PREDICTION_HORIZON][ACTIVE_RICCATI_NU] = {{0}};
    float P[RICCATI_MAX_NX][RICCATI_MAX_NX] = {{0}};
    float p[RICCATI_MAX_NX] = {0};

    for (int i = 0; i < ACTIVE_RICCATI_NX; ++i) {
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
        float r_aug_linear[ACTIVE_RICCATI_NU] = {0};
        float r_aug_diag[ACTIVE_RICCATI_NU] = {0};
        float M[ACTIVE_RICCATI_NU][RICCATI_MAX_NX] = {{0}};
        float S[ACTIVE_RICCATI_NU][ACTIVE_RICCATI_NU] = {{0}};
        float Si[ACTIVE_RICCATI_NU][ACTIVE_RICCATI_NU] = {{0}};
        float G[ACTIVE_RICCATI_NU][RICCATI_MAX_NX] = {{0}};
        float p_shift[RICCATI_MAX_NX] = {0};
        float P_A[RICCATI_MAX_NX][RICCATI_MAX_NX] = {{0}};
        float P_next[RICCATI_MAX_NX][RICCATI_MAX_NX] = {{0}};
        float p_next[RICCATI_MAX_NX] = {0};
        float btp[ACTIVE_RICCATI_NU] = {0};

        for (int i = 0; i < ACTIVE_RICCATI_NX; ++i) {
            const int constrained = k > 0 &&
                (sd->x_ub[i] < BIG_BOUND || sd->x_lb[i] > -BIG_BOUND);
            q_aug_diag[i] = sd->Q_diag[i] + (constrained ? rho : 0.0f);
            q_aug_linear[i] = sd->q[i] - (constrained ?
                rho * (z_x[k][i] - y_x[k][i]) : 0.0f);
            if (!isfinite(q_aug_diag[i]) || !isfinite(q_aug_linear[i]))
                return 0;
        }
        for (int a = 0; a < ACTIVE_RICCATI_NU; ++a) {
            r_aug_diag[a] = sd->R_diag[a] + rho_u;
            r_aug_linear[a] = sd->r[a] -
                rho_u * (z_u[k][a] - y_u[k][a]);
            if (!isfinite(r_aug_diag[a]) || !isfinite(r_aug_linear[a]))
                return 0;
        }

        active_form_control_terms(sd, P, r_aug_diag, M, S, G);
        if (!invert_regularized_control_hessian(S, ACTIVE_RICCATI_NU, Si))
            return 0;
        active_state_affine_shift(&P[0][0], sd->d, p, p_shift);
        active_b_transpose_p(sd, p_shift, btp);
        for (int a = 0; a < ACTIVE_RICCATI_NU; ++a) {
            for (int j = 0; j < ACTIVE_RICCATI_NX; ++j) {
                for (int b = 0; b < ACTIVE_RICCATI_NU; ++b)
                    K[k][a][j] -= Si[a][b] * G[b][j];
            }
            for (int b = 0; b < ACTIVE_RICCATI_NU; ++b)
                kk[k][a] -= Si[a][b] * (r_aug_linear[b] + btp[b]);
        }
        active_form_p_a(sd, P, P_A);
        active_update_p(sd, q_aug_diag, P_A, G, K[k], P_next);
        for (int i = 0; i < ACTIVE_RICCATI_NX; ++i) {
            p_next[i] = q_aug_linear[i];
            if (i < ACTIVE_RICCATI_PLANT_NX) {
                for (int s = 0; s < ACTIVE_RICCATI_PLANT_NX; ++s)
                    p_next[i] += sd->A[s][i] * p_shift[s];
            }
            for (int a = 0; a < ACTIVE_RICCATI_NU; ++a)
                p_next[i] += G[a][i] * kk[k][a];
            if (!isfinite(p_next[i])) return 0;
        }
        memcpy(P, P_next, sizeof(P));
        memcpy(p, p_next, sizeof(p));
    }

    for (int i = 0; i < ACTIVE_RICCATI_NX; ++i) x_out[0][i] = x0[i];
    for (int k = 0; k < N; ++k) {
        const RiccatiStepData_t *sd = &step_data[k];
        for (int a = 0; a < ACTIVE_RICCATI_NU; ++a) {
            u_out[k][a] = kk[k][a];
            for (int s = 0; s < ACTIVE_RICCATI_NX; ++s)
                u_out[k][a] += K[k][a][s] * x_out[k][s];
            if (!isfinite(u_out[k][a])) return 0;
        }
        for (int i = 0; i < ACTIVE_RICCATI_PLANT_NX; ++i) {
            x_out[k + 1][i] = sd->d[i];
            for (int s = 0; s < ACTIVE_RICCATI_PLANT_NX; ++s)
                x_out[k + 1][i] += sd->A[i][s] * x_out[k][s];
            for (int a = 0; a < ACTIVE_RICCATI_NU; ++a)
                x_out[k + 1][i] += sd->B[i][a] * u_out[k][a];
        }
        x_out[k + 1][ACTIVE_RICCATI_PLANT_NX] = u_out[k][0];
        x_out[k + 1][ACTIVE_RICCATI_PLANT_NX + 1] = u_out[k][1];
        for (int i = 0; i < ACTIVE_RICCATI_NX; ++i)
            if (!isfinite(x_out[k + 1][i])) return 0;
    }
    return 1;
}

static int riccati_solver_pass_factored_active(
    const RiccatiStepData_t *step_data,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    int N,
    const RiccatiFactorization_t *factor,
    const float z_x[][RICCATI_MAX_NX],
    const float y_x[][RICCATI_MAX_NX],
    const float z_u[][RICCATI_MAX_NU],
    const float y_u[][RICCATI_MAX_NU],
    float x_out[][RICCATI_MAX_NX],
    float u_out[][RICCATI_MAX_NU])
{
    (void)terminal_x_lb;
    (void)terminal_x_ub;
    float p[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float kk[PREDICTION_HORIZON][ACTIVE_RICCATI_NU] = {{0}};
    for (int i = 0; i < ACTIVE_RICCATI_NX; ++i) {
        p[N][i] = terminal_q[i] -
            (factor->x_is_constrained[N][i]
                ? factor->rho * (z_x[N][i] - y_x[N][i]) : 0.0f);
        if (!isfinite(p[N][i])) return 0;
    }

    for (int k = N - 1; k >= 0; --k) {
        const RiccatiStepData_t *sd = &step_data[k];
        float p_shift[RICCATI_MAX_NX] = {0};
        float q_aug_linear[RICCATI_MAX_NX] = {0};
        float r_aug_linear[ACTIVE_RICCATI_NU] = {0};
        float btp[ACTIVE_RICCATI_NU] = {0};
        active_state_affine_shift(&factor->P[k + 1][0][0], sd->d,
                                  p[k + 1], p_shift);
        for (int i = 0; i < ACTIVE_RICCATI_NX; ++i) {
            q_aug_linear[i] = sd->q[i] -
                (factor->x_is_constrained[k][i]
                    ? factor->rho * (z_x[k][i] - y_x[k][i]) : 0.0f);
            if (!isfinite(p_shift[i]) || !isfinite(q_aug_linear[i])) return 0;
        }
        for (int a = 0; a < ACTIVE_RICCATI_NU; ++a) {
            r_aug_linear[a] = sd->r[a] -
                factor->rho_u * (z_u[k][a] - y_u[k][a]);
        }
        active_b_transpose_p(sd, p_shift, btp);
        for (int a = 0; a < ACTIVE_RICCATI_NU; ++a) {
            for (int b = 0; b < ACTIVE_RICCATI_NU; ++b)
                kk[k][a] -= factor->S_inv[k][a][b] *
                    (r_aug_linear[b] + btp[b]);
            if (!isfinite(kk[k][a])) return 0;
        }
        for (int i = 0; i < ACTIVE_RICCATI_NX; ++i) {
            p[k][i] = q_aug_linear[i];
            if (i < ACTIVE_RICCATI_PLANT_NX) {
                for (int s = 0; s < ACTIVE_RICCATI_PLANT_NX; ++s)
                    p[k][i] += sd->A[s][i] * p_shift[s];
            }
            for (int a = 0; a < ACTIVE_RICCATI_NU; ++a)
                p[k][i] += factor->G[k][a][i] * kk[k][a];
            if (!isfinite(p[k][i])) return 0;
        }
    }

    for (int i = 0; i < ACTIVE_RICCATI_NX; ++i) x_out[0][i] = x0[i];
    for (int k = 0; k < N; ++k) {
        const RiccatiStepData_t *sd = &step_data[k];
        for (int a = 0; a < ACTIVE_RICCATI_NU; ++a) {
            u_out[k][a] = kk[k][a];
            for (int s = 0; s < ACTIVE_RICCATI_NX; ++s)
                u_out[k][a] += factor->K[k][a][s] * x_out[k][s];
            if (!isfinite(u_out[k][a])) return 0;
        }
        for (int i = 0; i < ACTIVE_RICCATI_PLANT_NX; ++i) {
            x_out[k + 1][i] = sd->d[i];
            for (int s = 0; s < ACTIVE_RICCATI_PLANT_NX; ++s)
                x_out[k + 1][i] += sd->A[i][s] * x_out[k][s];
            for (int a = 0; a < ACTIVE_RICCATI_NU; ++a)
                x_out[k + 1][i] += sd->B[i][a] * u_out[k][a];
        }
        x_out[k + 1][ACTIVE_RICCATI_PLANT_NX] = u_out[k][0];
        x_out[k + 1][ACTIVE_RICCATI_PLANT_NX + 1] = u_out[k][1];
        for (int i = 0; i < ACTIVE_RICCATI_NX; ++i)
            if (!isfinite(x_out[k + 1][i])) return 0;
    }
    return 1;
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
    int nx, int nu, int N,
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
    float K[PREDICTION_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX] = {{{0}}};
    float kk[PREDICTION_HORIZON][RICCATI_MAX_NU] = {{0}};
    float P[RICCATI_MAX_NX][RICCATI_MAX_NX] = {{0}};
    float p[RICCATI_MAX_NX] = {0};

    if (!step_data || !terminal_Q || !terminal_q || !terminal_x_lb ||
        !terminal_x_ub || !x0 || !z_x || !y_x || !z_u || !y_u ||
        !x_out || !u_out || nx <= 0 || nx > RICCATI_MAX_NX ||
        nu <= 0 || nu > RICCATI_MAX_NU || N <= 0 ||
        N > PREDICTION_HORIZON) {
        return 0;
    }

    if (nx == ACTIVE_RICCATI_NX && nu == ACTIVE_RICCATI_NU &&
        active_augmented_structure(step_data, N)) {
        return riccati_solver_pass_active(
            step_data, terminal_Q, terminal_q, terminal_x_lb, terminal_x_ub,
            x0, N, rho, rho_u, z_x, y_x, z_u, y_u, x_out, u_out);
    }

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
                G[a][j] = sd->N[j][a];
                for (int s = 0; s < nx; ++s) {
                    G[a][j] += M[a][s] * sd->A[s][j];
                }
            }
        }

        /* Preserve coupling while allowing a bounded SPD repair for tiny
         * round-off/conditioning defects.  Large indefiniteness still fails. */
        if (!invert_regularized_control_hessian(S, nu, Si)) return 0;

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

        /* P_A = P A; P_k = Q + A' P A + G'K.  Symmetrize round-off so the
         * next control Hessian remains numerically symmetric. */
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                for (int s = 0; s < nx; ++s) {
                    P_A[i][j] += P[i][s] * sd->A[s][j];
                }
            }
        }
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                P_next[i][j] = (i == j) ? q_aug_diag[i] : 0.0f;
                for (int s = 0; s < nx; ++s) {
                    P_next[i][j] += sd->A[s][i] * P_A[s][j];
                }
                for (int a = 0; a < nu; ++a) {
                    P_next[i][j] += G[a][i] * K[k][a][j];
                }
            }
        }
        for (int i = 0; i < nx; ++i) {
            for (int j = i + 1; j < nx; ++j) {
                const float symmetric = 0.5f * (P_next[i][j] + P_next[j][i]);
                P_next[i][j] = symmetric;
                P_next[j][i] = symmetric;
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

int riccati_solver_factorize(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    int nx,
    int nu,
    int horizon,
    float rho,
    float rho_u,
    RiccatiFactorization_t *factorization)
{
    const int N = horizon;
    RiccatiFactorization_t *const factor = factorization;
    if (!step_data || !terminal_Q || !terminal_x_lb || !terminal_x_ub ||
        !factor || nx <= 0 || nx > RICCATI_MAX_NX || nu <= 0 ||
        nu > RICCATI_MAX_NU || N <= 0 || N > PREDICTION_HORIZON ||
        !isfinite(rho) || rho < 0.0f || !isfinite(rho_u) || rho_u < 0.0f) {
        return 0;
    }

    memset(factor, 0, sizeof(*factor));
    factor->nx = nx;
    factor->nu = nu;
    factor->horizon = N;
    factor->rho = rho;
    factor->rho_u = rho_u;

    for (int i = 0; i < nx; ++i) {
        factor->x_is_constrained[N][i] =
            terminal_x_ub[i] < BIG_BOUND || terminal_x_lb[i] > -BIG_BOUND;
        factor->P[N][i][i] = terminal_Q[i] +
            (factor->x_is_constrained[N][i] ? rho : 0.0f);
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
            q_aug_diag[i] = sd->Q_diag[i] + (constrained ? rho : 0.0f);
            if (!isfinite(q_aug_diag[i])) return 0;
        }
        for (int a = 0; a < nu; ++a) {
            r_aug_diag[a] = sd->R_diag[a] + rho_u;
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
                factor->G[k][a][j] = sd->N[j][a];
                for (int s = 0; s < nx; ++s)
                    factor->G[k][a][j] += M[a][s] * sd->A[s][j];
            }
        }
        if (!invert_regularized_control_hessian(
                S, nu, factor->S_inv[k])) return 0;

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
            for (int j = 0; j < nx; ++j) {
                factor->P[k][i][j] = i == j ? q_aug_diag[i] : 0.0f;
                for (int s = 0; s < nx; ++s)
                    factor->P[k][i][j] += sd->A[s][i] * P_A[s][j];
                for (int a = 0; a < nu; ++a)
                    factor->P[k][i][j] +=
                        factor->G[k][a][i] * factor->K[k][a][j];
            }
        }
        for (int i = 0; i < nx; ++i) {
            for (int j = i + 1; j < nx; ++j) {
                const float symmetric =
                    0.5f * (factor->P[k][i][j] + factor->P[k][j][i]);
                factor->P[k][i][j] = symmetric;
                factor->P[k][j][i] = symmetric;
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

int riccati_solver_pass_factored(
    const RiccatiStepData_t *step_data,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    int nx,
    int nu,
    int horizon,
    const RiccatiFactorization_t *factorization,
    const float z_x[][RICCATI_MAX_NX],
    const float y_x[][RICCATI_MAX_NX],
    const float z_u[][RICCATI_MAX_NU],
    const float y_u[][RICCATI_MAX_NU],
    float x_out[][RICCATI_MAX_NX],
    float u_out[][RICCATI_MAX_NU])
{
    const int N = horizon;
    const RiccatiFactorization_t *const factor = factorization;
    if (!step_data || !terminal_q || !terminal_x_lb || !terminal_x_ub ||
        !x0 || !factor || !factor->valid || !z_x || !y_x || !z_u || !y_u ||
        !x_out || !u_out || nx != factor->nx || nu != factor->nu ||
        N != factor->horizon || nx <= 0 || nx > RICCATI_MAX_NX ||
        nu <= 0 || nu > RICCATI_MAX_NU || N <= 0 ||
        N > PREDICTION_HORIZON) {
        return 0;
    }

    if (nx == ACTIVE_RICCATI_NX && nu == ACTIVE_RICCATI_NU &&
        active_augmented_structure(step_data, N)) {
        return riccati_solver_pass_factored_active(
            step_data, terminal_q, terminal_x_lb, terminal_x_ub, x0, N,
            factor, z_x, y_x, z_u, y_u, x_out, u_out);
    }

    float p[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float kk[PREDICTION_HORIZON][RICCATI_MAX_NU] = {{0}};
    for (int i = 0; i < nx; ++i) {
        p[N][i] = terminal_q[i] -
            (factor->x_is_constrained[N][i]
                ? factor->rho * (z_x[N][i] - y_x[N][i]) : 0.0f);
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
                    ? factor->rho * (z_x[k][i] - y_x[k][i]) : 0.0f);
            if (!isfinite(p_shift[i]) || !isfinite(q_aug_linear[i])) return 0;
        }
        for (int a = 0; a < nu; ++a) {
            r_aug_linear[a] = sd->r[a] -
                factor->rho_u * (z_u[k][a] - y_u[k][a]);
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
    int nx,
    int nu,
    int N,
    float rho,
    float rho_u,
    const float z_x[][RICCATI_MAX_NX],
    const float y_x[][RICCATI_MAX_NX],
    const float z_u[][RICCATI_MAX_NU],
    const float y_u[][RICCATI_MAX_NU],
    float x_out[][RICCATI_MAX_NX],
    float u_out[][RICCATI_MAX_NU],
    int use_prefactorization,
    RiccatiFactorization_t *factorization)
{
    if (!use_prefactorization) {
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t start_ns = riccati_profile_now_ns();
#endif
        const int solved = riccati_solver_pass(
            step_data, terminal_Q, terminal_q, terminal_x_lb, terminal_x_ub,
            x0, nx, nu, N, rho, rho_u, z_x, y_x, z_u, y_u, x_out, u_out);
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t finish_ns = riccati_profile_now_ns();
        if (finish_ns >= start_ns)
            g_riccati_debug_last.refactor_pass_time_ns += finish_ns - start_ns;
#endif
        return solved;
    }
    if (!factorization) return 0;
    if (!factorization->valid || factorization->nx != nx ||
        factorization->nu != nu || factorization->horizon != N ||
        factorization->rho != rho || factorization->rho_u != rho_u) {
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t start_ns = riccati_profile_now_ns();
#endif
        if (!riccati_solver_factorize(step_data, terminal_Q,
                terminal_x_lb, terminal_x_ub, nx, nu, N, rho, rho_u,
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
        nx, nu, N, factorization, z_x, y_x, z_u, y_u, x_out, u_out);
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
    int nx, int nu, int N,
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

    if (nx <= 0 || nx > RICCATI_MAX_NX || nu <= 0 || nu > RICCATI_MAX_NU ||
        N <= 0 || N > PREDICTION_HORIZON) {
        solution->status = RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }
    if (config && config->use_prefactorization != 0 &&
        config->use_prefactorization != 1) {
        solution->status = RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }
    const float over_relaxation = config &&
        config->over_relaxation > 0.0f ? config->over_relaxation : 1.0f;
    if (!isfinite(over_relaxation) || over_relaxation < 1.0f ||
        over_relaxation > 2.0f) {
        solution->status = RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }

    if (admm_state->initialized &&
        (admm_state->nx != nx || admm_state->nu != nu ||
         admm_state->horizon != N)) {
        riccati_admm_state_init(admm_state);
    }

    const float cfg_rho = (config && config->rho > 0.0f) ? config->rho : ADMM_RHO;
    const float cfg_rho_u = (config && config->rho_u > 0.0f) ? config->rho_u : ADMM_RHO_U;
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
    const int shared_rho = config ? config->shared_rho : 0;
    const int use_prefactorization =
        config ? config->use_prefactorization : 0;

    memset(&g_riccati_debug_last, 0, sizeof(g_riccati_debug_last));
    g_riccati_debug_trace_count = 0;
    RiccatiFactorization_t factorization = {0};

    float rho = (admm_state->initialized && admm_state->rho > 0.0f)
                    ? admm_state->rho : cfg_rho;
    float rho_u = (admm_state->initialized && admm_state->rho_u > 0.0f)
                    ? admm_state->rho_u : (cfg_rho_u > 0.0f ? cfg_rho_u : rho);
    /* Clamp initial penalties to the configured CPU/replay adaptive range. */
    if (rho < 1.0f) rho = 1.0f;
    if (rho_u < 1.0f) rho_u = 1.0f;
    if (rho > 127.0f) rho = 127.0f;
    if (rho_u > 127.0f) rho_u = 127.0f;
    if (shared_rho) {
        /* A public-API caller may switch an existing warm start from
         * independent penalties to one shared penalty.  Preserve the
         * unscaled control dual before adopting the state rho. */
        if (admm_state->initialized && rho_u != rho) {
            const float y_u_scale = rho_u / rho;
            for (int k = 0; k < N; k++) {
                for (int a = 0; a < nu; a++) {
                    admm_state->y_u[k][a] *= y_u_scale;
                }
            }
        }
        rho_u = rho;
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
        riccati_admm_shift_warm_start(admm_state, nx, nu, N, x0);
    }

    /* Precompute constrained flags */
    uint8_t x_is_constrained[PREDICTION_HORIZON + 1][RICCATI_MAX_NX];
    memset(x_is_constrained, 0, sizeof(x_is_constrained));
    for (int k = 0; k <= N; k++) {
        for (int s = 0; s < nx; s++) {
            const float xub = (k < N) ? step_data[k].x_ub[s] : terminal_x_ub[s];
            const float xlb = (k < N) ? step_data[k].x_lb[s] : terminal_x_lb[s];
            /* x_0 is fixed data.  Only predicted states x_1..x_N can be
             * projected by ADMM. */
            x_is_constrained[k][s] = k > 0 &&
                                     (xub < BIG_BOUND || xlb > -BIG_BOUND);
        }
    }

    if (admm_state->initialized) {
        /* A warm start may contain a projected (or simply stale) stage-zero
         * value.  Keep the fixed initial state out of the split variables. */
        for (int s = 0; s < nx; s++) {
            z_x[0][s] = x0[s];
            y_x[0][s] = 0.0f;
        }
    }

    if (!admm_state->initialized) {
        /* Cold start */
        memset(z_x, 0, sizeof(admm_state->z_x));
        memset(z_u, 0, sizeof(admm_state->z_u));
        memset(y_x, 0, sizeof(admm_state->y_x));
        memset(y_u, 0, sizeof(admm_state->y_u));

        if (!riccati_solver_primal_pass(
            step_data, terminal_Q, terminal_q, terminal_x_lb, terminal_x_ub, x0,
            nx, nu, N, 0.0f, 0.0f,
            (const float (*)[RICCATI_MAX_NX])z_x,
            (const float (*)[RICCATI_MAX_NX])y_x,
            (const float (*)[RICCATI_MAX_NU])z_u,
            (const float (*)[RICCATI_MAX_NU])y_u,
            solution->x, solution->u, 0, NULL)) {
            solution->status = RICCATI_STATUS_ERROR;
            solution->iterations = 0;
            solution->primal_residual = INFINITY;
            solution->dual_residual = INFINITY;
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
            sample->u0_target_speed_rate = solution->u[0][1];
            sample->z0_steer = z_u[0][0];
            sample->z0_target_speed_rate = z_u[0][1];
            sample->y0_steer = y_u[0][0];
            sample->y0_target_speed_rate = y_u[0][1];
        }
    }

    RiccatiStatus_t status = RICCATI_STATUS_MAX_ITERATIONS;

    for (int iter = 0; iter < max_iter; iter++) {

        /*--- Primal update: Riccati pass with augmented costs ---*/
        if (!riccati_solver_primal_pass(
            step_data, terminal_Q, terminal_q, terminal_x_lb, terminal_x_ub, x0,
            nx, nu, N, rho, rho_u,
            (const float (*)[RICCATI_MAX_NX])z_x,
            (const float (*)[RICCATI_MAX_NX])y_x,
            (const float (*)[RICCATI_MAX_NU])z_u,
            (const float (*)[RICCATI_MAX_NU])y_u,
            solution->x, solution->u, use_prefactorization,
            &factorization)) {
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
                    float dd = fabsf(rho * (z_new - z_prev));
                    state_dual = fmaxf(state_dual, dd);
                    state_dual_balance = fmaxf(
                        state_dual_balance, dd / span);
                    /* y-update: y += x_hat - z.  x_hat is the standard ADMM
                     * over-relaxed primal point; with alpha=1 this is the
                     * original unrelaxed update. */
                    y_x[k][s] = x_hat - z_new + y_x[k][s];
                    /* Primal residual */
                    float pd = fabsf(solution->x[k][s] - z_new);
                    state_primal = fmaxf(state_primal, pd);
                    state_primal_balance = fmaxf(
                        state_primal_balance, pd / span);
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
                float dd = fabsf(rho_u * (z_new - z_prev));
                ctrl_dual = fmaxf(ctrl_dual, dd);
                ctrl_dual_balance = fmaxf(ctrl_dual_balance, dd / span);
                /* y-update uses the same over-relaxed primal point as the
                 * state split. */
                y_u[k][a] = u_hat - z_new + y_u[k][a];
                /* Primal residual */
                float pd = fabsf(solution->u[k][a] - z_new);
                ctrl_primal = fmaxf(ctrl_primal, pd);
                ctrl_primal_balance = fmaxf(ctrl_primal_balance, pd / span);
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
            sample->u0_target_speed_rate = solution->u[0][1];
            sample->z0_steer = z_u[0][0];
            sample->z0_target_speed_rate = z_u[0][1];
            sample->y0_steer = y_u[0][0];
            sample->y0_target_speed_rate = y_u[0][1];
            sample->scale_rho = 0;
            sample->scale_rho_u = 0;
        }

        if (primal_res <= eps_primal && dual_res <= eps_dual) {
            status = RICCATI_STATUS_OPTIMAL;
            break;
        }

        /*--- Adaptive rho (aligned with FPGA HLS solver) ---*/
        if (adaptive_rho && iter > 0) {
            const float rho_before_adaptation = rho;
            const float rho_u_before_adaptation = rho_u;
            const float adapt_ratio_state = 2.0f;
            const float adapt_ratio_ctrl = 2.0f;
            const float adapt_ratio_shared = 5.0f;
            const float rho_min = 1.0f;
            const float rho_max = 127.0f;

            int scale_rho = 0;
            int scale_rho_u = 0;

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
            } else {
                if (state_primal_balance > adapt_ratio_state *
                        state_dual_balance && rho < rho_max) {
                    scale_rho = 1;
                } else if (state_dual_balance > adapt_ratio_state *
                               state_primal_balance && rho > rho_min) {
                    scale_rho = -1;
                }

                if (ctrl_primal_balance > adapt_ratio_ctrl *
                        ctrl_dual_balance && rho_u < rho_max) {
                    scale_rho_u = 1;
                } else if (ctrl_dual_balance > adapt_ratio_ctrl *
                               ctrl_primal_balance && rho_u > rho_min) {
                    scale_rho_u = -1;
                }
            }

            if (g_riccati_debug_trace_enabled &&
                g_riccati_debug_trace_count > 0) {
                RiccatiDebugIterSample_t *sample =
                    &g_riccati_debug_trace[g_riccati_debug_trace_count - 1];
                sample->scale_rho = scale_rho;
                sample->scale_rho_u = scale_rho_u;
            }

            /* Update penalties and rescale the scaled dual variables to keep
             * lambda = rho*y invariant.  Deriving the multiplier from the
             * clamped old/new values is important: rho changes by 1.25/0.75
             * for states, by 2/0.5 for controls, and may hit either bound. */
            if (scale_rho != 0) {
                const float old_rho = rho;
                if (scale_rho > 0) {
                    rho *= 1.25f;
                    if (rho > rho_max) rho = rho_max;
                } else {
                    rho *= 0.75f;
                    if (rho < rho_min) rho = rho_min;
                }
                const float y_x_scale = old_rho / rho;
                for (int k = 1; k <= N; k++) {
                    for (int s = 0; s < nx; s++) {
                        if (x_is_constrained[k][s]) {
                            y_x[k][s] *= y_x_scale;
                        }
                    }
                }
            }

            if (shared_rho) {
                const float old_rho_u = rho_u;
                rho_u = rho;
                if (rho_u != old_rho_u) {
                    const float y_u_scale = old_rho_u / rho_u;
                    for (int k = 0; k < N; k++) {
                        for (int a = 0; a < nu; a++) {
                            y_u[k][a] *= y_u_scale;
                        }
                    }
                }
            } else if (scale_rho_u != 0) {
                const float old_rho_u = rho_u;
                if (scale_rho_u > 0) {
                    rho_u *= 2.0f;
                    if (rho_u > rho_max) rho_u = rho_max;
                } else {
                    rho_u *= 0.5f;
                    if (rho_u < rho_min) rho_u = rho_min;
                }
                const float y_u_scale = old_rho_u / rho_u;
                for (int k = 0; k < N; k++) {
                    for (int a = 0; a < nu; a++) {
                        y_u[k][a] *= y_u_scale;
                    }
                }
            }
            if (rho != rho_before_adaptation ||
                rho_u != rho_u_before_adaptation) {
                ++g_riccati_debug_last.rho_change_count;
            }
        }
    }

    /* Save scalar warm-start metadata. Buffers are already updated in-place. */
    admm_state->rho = rho;
    admm_state->rho_u = shared_rho ? rho : rho_u;
    admm_state->nx = nx;
    admm_state->nu = nu;
    admm_state->horizon = N;
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
    const RiccatiAdmmConfig_t *config,
    int nx,
    int nu)
{
    if (!state || !config || !state->initialized ||
        !state->scaling_enabled || state->nx != nx || state->nu != nu)
        return 0;
    for (int i = 0; i < nx; ++i)
        if (state->state_scale[i] != config->state_scale[i]) return 0;
    for (int a = 0; a < nu; ++a)
        if (state->input_scale[a] != config->input_scale[a]) return 0;
    return 1;
}

RiccatiStatus_t riccati_admm_solve(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    int nx, int nu, int horizon,
    const RiccatiAdmmConfig_t *config,
    RiccatiAdmmState_t *admm_state,
    RiccatiSolution_t *solution)
{
    const int N = horizon;
    if (!admm_state || !solution) {
        if (solution) solution->status = RICCATI_STATUS_ERROR;
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
            x0, nx, nu, N, config, admm_state, solution);
    }

    if (!step_data || !terminal_Q || !terminal_q || !terminal_x_lb ||
        !terminal_x_ub || !x0 || nx <= 0 || nx > RICCATI_MAX_NX ||
        nu <= 0 || nu > RICCATI_MAX_NU || N <= 0 ||
        N > PREDICTION_HORIZON) {
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
        !riccati_scaling_signature_matches(admm_state, config, nx, nu))
        riccati_admm_state_init(admm_state);

    /* x = Dx*xhat and u = Du*uhat.  Scale dynamics, all diagonal and
     * cross-cost terms, affine costs, and hard bounds consistently. */
    RiccatiStepData_t scaled_steps[PREDICTION_HORIZON] = {0};
    float scaled_terminal_Q[RICCATI_MAX_NX] = {0};
    float scaled_terminal_q[RICCATI_MAX_NX] = {0};
    float scaled_terminal_lb[RICCATI_MAX_NX] = {0};
    float scaled_terminal_ub[RICCATI_MAX_NX] = {0};
    float scaled_x0[RICCATI_MAX_NX] = {0};
    for (int i = 0; i < nx; ++i) {
        const float scale = config->state_scale[i];
        scaled_terminal_Q[i] = terminal_Q[i] * scale * scale;
        scaled_terminal_q[i] = terminal_q[i] * scale;
        scaled_terminal_lb[i] = terminal_x_lb[i] / scale;
        scaled_terminal_ub[i] = terminal_x_ub[i] / scale;
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
            scaled->x_lb[i] = physical->x_lb[i] / state_scale;
            scaled->x_ub[i] = physical->x_ub[i] / state_scale;
            for (int j = 0; j < nx; ++j) {
                scaled->A[i][j] = physical->A[i][j] *
                    config->state_scale[j] / state_scale;
            }
            for (int a = 0; a < nu; ++a) {
                scaled->B[i][a] = physical->B[i][a] *
                    config->input_scale[a] / state_scale;
                scaled->N[i][a] = physical->N[i][a] *
                    state_scale * config->input_scale[a];
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
    const RiccatiStatus_t status = riccati_admm_solve_core(
        scaled_steps, scaled_terminal_Q, scaled_terminal_q,
        scaled_terminal_lb, scaled_terminal_ub, scaled_x0,
        nx, nu, N, &scaled_config, admm_state, &scaled_solution);
    *solution = scaled_solution;
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
    return status;
}

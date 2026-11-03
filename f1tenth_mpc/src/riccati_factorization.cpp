// Unconstrained Riccati factorization and factored right-hand-side recursion.
// This is the sole numerical pathway for cold, public and warm ADMM solves.
#include "riccati_solver_internal.hpp"
#include <math.h>
#include <string.h>

extern "C" int mpc_riccati_eigen_factored_rhs(
    const RiccatiStepData_t*, const float*, const float*,
    const RiccatiFactorization_t*, const float*, const float*,
    const float*, const float*, float*, float*);
extern "C" void mpc_riccati_eigen_quadratic_update(
    const float*, const float*, const float*, const float*,
    const float[RICCATI_MAX_NX], const float*, float*);

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
                ++riccati_debug_last_mutable().control_hessian_regularization_count;
                if (lambda >
                    riccati_debug_last_mutable().max_control_hessian_regularization) {
                    riccati_debug_last_mutable().max_control_hessian_regularization =
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

    riccati_debug_last_mutable().last_fallback_s00 = S[0][0] + lambda;
    riccati_debug_last_mutable().last_fallback_s11 = S[1][1] + lambda;
    riccati_debug_last_mutable().last_invert_det =
        riccati_debug_last_mutable().last_fallback_s00 *
        riccati_debug_last_mutable().last_fallback_s11 - S[0][1] * S[0][1];
    ++riccati_debug_last_mutable().invert_fallback_count;
    return 0;
}

/*===========================================================================
 * Riccati Backward + Forward Pass
 *===========================================================================*/

/* Public one-shot Riccati pass.  Use the same numeric factorization/RHS
 * as the ADMM code rather than maintaining a second, slowly diverging C
 * implementation.  Public factorization does not enable the fixed-QP
 * affine cache: callers can legally change d between factored RHS calls. */
int riccati_solver_pass(
    const RiccatiStepData_t * __restrict step_data,
    const float * __restrict terminal_Q,
    const float * __restrict terminal_q,
    const float * __restrict terminal_x_lb,
    const float * __restrict terminal_x_ub,
    const float * __restrict x0,
    float rho,
    float rho_u,
    const float z_x[][RICCATI_MAX_NX],
    const float y_x[][RICCATI_MAX_NX],
    const float z_u[][RICCATI_MAX_NU],
    const float y_u[][RICCATI_MAX_NU],
    float x_out[][RICCATI_MAX_NX],
    float u_out[][RICCATI_MAX_NU])
{
    if (!step_data || !terminal_Q || !terminal_q ||
        !terminal_x_lb || !terminal_x_ub || !x0 ||
        !z_x || !y_x || !z_u || !y_u || !x_out || !u_out)
        return 0;
    RiccatiFactorization_t factor = {};
    if (!riccati_solver_factorize(
            step_data, terminal_Q, terminal_x_lb,
            terminal_x_ub, rho, rho_u, &factor))
        return 0;
    return riccati_solver_pass_factored(
        step_data, terminal_q, terminal_x_lb, terminal_x_ub,
        x0, &factor, z_x, y_x, z_u, y_u, x_out, u_out);
}

static int riccati_solver_factorize_diagonal(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float rho_state[][RICCATI_MAX_NX],
    const float rho_input[][RICCATI_MAX_NU],
    RiccatiFactorization_t *factorization,
    const float terminal_augmentation[RICCATI_MAX_NX][RICCATI_MAX_NX],
    const RiccatiStatePolygon_t *polygons)
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
            terminal_x_ub[i] < BIG_BOUND || terminal_x_lb[i] > -BIG_BOUND ||
            riccati_polygon_constrains_state(polygons,N,i);
        factor->P[N][i][i] = terminal_Q[i] +
            (factor->x_is_constrained[N][i] ? rho_state[N][i] : 0.0f);
        if (!isfinite(factor->P[N][i][i])) return 0;
    }

    if (terminal_augmentation)
        for (int i=0;i<nx;++i)
            for (int j=0;j<nx;++j)
                factor->P[N][i][j] += terminal_augmentation[i][j];

    for (int k = N - 1; k >= 0; --k) {
        const RiccatiStepData_t *sd = &step_data[k];
        float q_aug_diag[RICCATI_MAX_NX] = {};
        float r_aug_diag[RICCATI_MAX_NU] = {};
        float M[RICCATI_MAX_NU][RICCATI_MAX_NX] = {{0}};
        float S[2][2] = {{0}};

        for (int i = 0; i < nx; ++i) {
            const int constrained = k > 0 &&
                (sd->x_ub[i] < BIG_BOUND || sd->x_lb[i] > -BIG_BOUND ||
                 riccati_polygon_constrains_state(polygons,k,i));
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
            // Interleave independent output coordinates, NOT the reduction
            // order. Each result still accumulates s=0..nx-1 in float, but
            // the inner loop traverses a contiguous row and permits SIMD.
            for (int s = 0; s < nx; ++s) {
                for (int j = 0; j < nx; ++j)
                    M[a][j] += sd->B[s][a] * factor->P[k + 1][s][j];
            }
        }
        for (int a = 0; a < nu; ++a) {
            for (int b = 0; b < nu; ++b) {
                S[a][b] = a == b ? r_aug_diag[a] : sd->R_cross;
                for (int s = 0; s < nx; ++s)
                    S[a][b] += M[a][s] * sd->B[s][b];
            }
            for (int s = 0; s < nx; ++s)
                for (int j = 0; j < nx; ++j)
                    factor->G[k][a][j] += M[a][s] * sd->A[s][j];
            for (int j = 0; j < nx; ++j)
                factor->G[k][a][j] += sd->N[j][a];
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

        /* P[k+1]*d[k] is fixed within a private QP while rho is fixed.
         * Compute it once per factorization instead of once per ADMM RHS
         * iteration. Public factoring still accepts changed affine d[k]:
         * public callers never activate the private cache below. */
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < nx; ++j)
                factor->affine_state_shift[k][i] +=
                    factor->P[k + 1][i][j] * sd->d[j];
        mpc_riccati_eigen_quadratic_update(
            &factor->P[k+1][0][0],&sd->A[0][0],
            &factor->G[k][0][0],&factor->K[k][0][0],
            q_aug_diag,&sd->Q_offdiag[0][0],&factor->P[k][0][0]);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < nx; ++j)
                if (!isfinite(factor->P[k][i][j])) return 0;
    }

    factor->valid = 1;
    ++riccati_debug_last_mutable().quadratic_factorization_count;
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
        (const float (*)[RICCATI_MAX_NU])rho_input, factorization,
        nullptr,nullptr);
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
    const RiccatiFactorization_t *const factor = factorization;
    if (!step_data || !terminal_q || !terminal_x_lb || !terminal_x_ub ||
        !x0 || !factor || !factor->valid || !z_x || !y_x || !z_u || !y_u ||
        !x_out || !u_out) {
        return 0;
    }

    return mpc_riccati_eigen_factored_rhs(
        step_data,terminal_q,x0,factor,
        &z_x[0][0],&y_x[0][0],&z_u[0][0],&y_u[0][0],
        &x_out[0][0],&u_out[0][0]);
}

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
    const float terminal_augmentation[RICCATI_MAX_NX][RICCATI_MAX_NX],
    const RiccatiStatePolygon_t *polygons)
{
    if (!factorization) return 0;
    // Only the ADMM core calls this private fixed-QP pathway. Its penalty
    // arrays cannot change between iterations except during its explicit
    // rho-adaptation step, which invalidates this factorization immediately.
    // Avoid comparing every one of the 773 penalty coordinates each iterate.
    if (!factorization->valid) {
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t start_ns = riccati_profile_now_ns();
#endif
        if (!riccati_solver_factorize_diagonal(step_data, terminal_Q,
                terminal_x_lb, terminal_x_ub, rho_state, rho_input,
                factorization, terminal_augmentation,polygons)) return 0;
        /* This factorization belongs to one fixed MPC QP and cannot be
         * used through the public RHS reuse contract. Subsequent ADMM
         * iterations only update duals and penalty matrices; a penalty
         * change forces a new factorization (and new P*d cache). */
        factorization->private_fixed_qp_affine_cache_enabled = 1;
#ifdef MPC_ENABLE_RICCATI_PROFILE
        const uint64_t finish_ns = riccati_profile_now_ns();
        if (finish_ns >= start_ns)
            riccati_debug_last_mutable().quadratic_factorization_time_ns +=
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
        riccati_debug_last_mutable().quadratic_rhs_time_ns +=
            rhs_finish_ns - rhs_start_ns;
#endif
    return solved;
}


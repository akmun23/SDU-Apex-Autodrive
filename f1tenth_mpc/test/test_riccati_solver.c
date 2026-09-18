#include "riccati_solver.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define ORACLE_MAX_CONTROLS (PREDICTION_HORIZON * RICCATI_MAX_NU)

static int failures = 0;

static void check_true(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

static void test_warm_start_shifts_every_stage(void)
{
    RiccatiAdmmState_t state = {0};
    const int nx = 9;
    const int nu = 2;
    const int horizon = 3;
    float new_x0[RICCATI_MAX_NX] = {0};
    for (int k = 0; k <= horizon; ++k) {
        for (int i = 0; i < nx; ++i) {
            state.z_x[k][i] = 100.0f * k + i;
            state.y_x[k][i] = -100.0f * k - i;
        }
    }
    for (int k = 0; k < horizon; ++k) {
        for (int j = 0; j < nu; ++j) {
            state.z_u[k][j] = 10.0f * k + j;
            state.y_u[k][j] = -10.0f * k - j;
        }
    }
    state.initialized = 1;
    for (int i = 0; i < nx; ++i) new_x0[i] = -1.0f - i;

    riccati_admm_shift_warm_start(&state, nx, nu, horizon, new_x0);

    for (int i = 0; i < nx; ++i) {
        check_true(state.z_x[0][i] == new_x0[i] && state.y_x[0][i] == 0.0f,
            "warm-start shift pins x0 and clears its dual");
    }
    for (int k = 1; k < horizon; ++k) {
        for (int i = 0; i < nx; ++i) {
            check_true(state.z_x[k][i] == 100.0f * (k + 1) + i,
                "state primal warm start shifts toward stage zero");
            check_true(state.y_x[k][i] == -100.0f * (k + 1) - i,
                "state dual warm start shifts toward stage zero");
        }
    }
    for (int i = 0; i < nx; ++i) {
        check_true(state.z_x[horizon][i] == 100.0f * horizon + i &&
                   state.y_x[horizon][i] == -100.0f * horizon - i,
            "terminal warm-start state repeats the old terminal value");
    }
    for (int k = 0; k < horizon - 1; ++k) {
        for (int j = 0; j < nu; ++j) {
            check_true(state.z_u[k][j] == 10.0f * (k + 1) + j,
                "control primal warm start shifts toward stage zero");
            check_true(state.y_u[k][j] == -10.0f * (k + 1) - j,
                "control dual warm start shifts toward stage zero");
        }
    }
    for (int j = 0; j < nu; ++j) {
        check_true(state.z_u[horizon - 1][j] == 10.0f * (horizon - 1) + j &&
                   state.y_u[horizon - 1][j] == -10.0f * (horizon - 1) - j,
            "terminal warm-start control repeats the old final value");
    }
}

static int solve_dense_system(
    float matrix[ORACLE_MAX_CONTROLS][ORACLE_MAX_CONTROLS],
    float rhs[ORACLE_MAX_CONTROLS],
    int dimension)
{
    for (int pivot = 0; pivot < dimension; ++pivot) {
        int best = pivot;
        for (int row = pivot + 1; row < dimension; ++row) {
            if (fabsf(matrix[row][pivot]) > fabsf(matrix[best][pivot])) {
                best = row;
            }
        }
        if (fabsf(matrix[best][pivot]) < 1.0e-8f) {
            return 0;
        }
        if (best != pivot) {
            for (int column = pivot; column < dimension; ++column) {
                const float temporary = matrix[pivot][column];
                matrix[pivot][column] = matrix[best][column];
                matrix[best][column] = temporary;
            }
            const float temporary = rhs[pivot];
            rhs[pivot] = rhs[best];
            rhs[best] = temporary;
        }
        const float diagonal = matrix[pivot][pivot];
        for (int column = pivot; column < dimension; ++column) {
            matrix[pivot][column] /= diagonal;
        }
        rhs[pivot] /= diagonal;
        for (int row = 0; row < dimension; ++row) {
            if (row == pivot) continue;
            const float factor = matrix[row][pivot];
            for (int column = pivot; column < dimension; ++column) {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    return 1;
}

/* Independent condensed-QP oracle.  It propagates x = a + F U, assembles the
 * full Hessian/gradient from the declared quadratic cost, then solves it with
 * pivoted Gaussian elimination.  Deliberately dense A/B/N entries exercise
 * every state and invalidate assumptions about the old sparse 8+2 layout. */
static int solve_condensed_oracle(
    const RiccatiStepData_t *steps,
    const float *terminal_Q,
    const float *terminal_q,
    const float *x0,
    int nx,
    int nu,
    int horizon,
    float controls[ORACLE_MAX_CONTROLS],
    float states[PREDICTION_HORIZON + 1][RICCATI_MAX_NX])
{
    const int variables = horizon * nu;
    float affine[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float sensitivity[PREDICTION_HORIZON + 1][RICCATI_MAX_NX]
                    [ORACLE_MAX_CONTROLS] = {{{0}}};
    float hessian[ORACLE_MAX_CONTROLS][ORACLE_MAX_CONTROLS] = {{0}};
    float gradient[ORACLE_MAX_CONTROLS] = {0};
    memcpy(affine[0], x0, (size_t)nx * sizeof(float));

    for (int k = 0; k < horizon; ++k) {
        const RiccatiStepData_t *step = &steps[k];
        for (int i = 0; i < nx; ++i) {
            affine[k + 1][i] = step->d[i];
            for (int s = 0; s < nx; ++s) {
                affine[k + 1][i] += step->A[i][s] * affine[k][s];
                for (int column = 0; column < variables; ++column) {
                    sensitivity[k + 1][i][column] +=
                        step->A[i][s] * sensitivity[k][s][column];
                }
            }
            for (int a = 0; a < nu; ++a) {
                sensitivity[k + 1][i][k * nu + a] += step->B[i][a];
            }
        }

        for (int p = 0; p < variables; ++p) {
            for (int i = 0; i < nx; ++i) {
                gradient[p] += sensitivity[k][i][p] *
                    (step->Q_diag[i] * affine[k][i] + step->q[i]);
            }
            for (int q = 0; q < variables; ++q) {
                for (int i = 0; i < nx; ++i) {
                    hessian[p][q] += step->Q_diag[i] *
                        sensitivity[k][i][p] * sensitivity[k][i][q];
                }
            }
        }

        for (int a = 0; a < nu; ++a) {
            const int control_column = k * nu + a;
            hessian[control_column][control_column] += step->R_diag[a];
            gradient[control_column] += step->r[a];
            for (int i = 0; i < nx; ++i) {
                gradient[control_column] += affine[k][i] * step->N[i][a];
            }
            for (int p = 0; p < variables; ++p) {
                float cross = 0.0f;
                for (int i = 0; i < nx; ++i) {
                    cross += sensitivity[k][i][p] * step->N[i][a];
                }
                hessian[p][control_column] += cross;
                hessian[control_column][p] += cross;
            }
        }
    }

    for (int p = 0; p < variables; ++p) {
        for (int i = 0; i < nx; ++i) {
            gradient[p] += sensitivity[horizon][i][p] *
                (terminal_Q[i] * affine[horizon][i] + terminal_q[i]);
        }
        for (int q = 0; q < variables; ++q) {
            for (int i = 0; i < nx; ++i) {
                hessian[p][q] += terminal_Q[i] *
                    sensitivity[horizon][i][p] * sensitivity[horizon][i][q];
            }
        }
        gradient[p] = -gradient[p];
    }

    if (!solve_dense_system(hessian, gradient, variables)) {
        return 0;
    }
    memcpy(controls, gradient, (size_t)variables * sizeof(float));
    for (int k = 0; k <= horizon; ++k) {
        for (int i = 0; i < nx; ++i) {
            states[k][i] = affine[k][i];
            for (int column = 0; column < variables; ++column) {
                states[k][i] += sensitivity[k][i][column] * gradient[column];
            }
        }
    }
    return 1;
}

static void test_dense_n_state_riccati_against_condensed_qp(int nx)
{
    const int nu = 2;
    const int horizon = 4;
    RiccatiStepData_t steps[PREDICTION_HORIZON] = {0};
    float terminal_Q[RICCATI_MAX_NX] = {0};
    float terminal_q[RICCATI_MAX_NX] = {0};
    float terminal_lb[RICCATI_MAX_NX];
    float terminal_ub[RICCATI_MAX_NX];
    float x0[RICCATI_MAX_NX] = {0};
    float z_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float y_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float z_u[PREDICTION_HORIZON][RICCATI_MAX_NU] = {{0}};
    float y_u[PREDICTION_HORIZON][RICCATI_MAX_NU] = {{0}};
    float riccati_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float riccati_u[PREDICTION_HORIZON][RICCATI_MAX_NU] = {{0}};
    float oracle_u[ORACLE_MAX_CONTROLS] = {0};
    float oracle_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};

    for (int i = 0; i < nx; ++i) {
        x0[i] = 0.03f * (float)(i - 4);
        terminal_Q[i] = 1.0f + 0.1f * (float)i;
        terminal_q[i] = 0.015f * (float)(i - 3);
        terminal_lb[i] = -BIG_BOUND;
        terminal_ub[i] = BIG_BOUND;
    }
    for (int k = 0; k < horizon; ++k) {
        RiccatiStepData_t *step = &steps[k];
        for (int i = 0; i < nx; ++i) {
            step->d[i] = 0.002f * (float)(i - 4 + k);
            step->Q_diag[i] = 0.7f + 0.04f * (float)i;
            step->q[i] = 0.008f * (float)(i - 5 + k);
            step->x_lb[i] = -BIG_BOUND;
            step->x_ub[i] = BIG_BOUND;
            for (int j = 0; j < nx; ++j) {
                if (i == j) {
                    step->A[i][j] = 0.45f + 0.01f * (float)i;
                } else {
                    const int residue = (i * 13 + j * 7 + k * 3) % 9;
                    const float sign = ((i + j + k) % 2) ? -1.0f : 1.0f;
                    step->A[i][j] = sign * (0.001f + 0.0002f * (float)residue);
                }
            }
            for (int a = 0; a < nu; ++a) {
                const float sign = ((i + a + k) % 2) ? -1.0f : 1.0f;
                step->B[i][a] = sign * (0.012f + 0.002f * (float)((i + 2 * a) % 5));
                step->N[i][a] = sign * (0.002f + 0.0003f * (float)((i + a) % 4));
            }
        }
        for (int a = 0; a < nu; ++a) {
            step->R_diag[a] = 1.4f + 0.2f * (float)a;
            step->r[a] = 0.01f * (float)(a - k);
            step->u_lb[a] = -BIG_BOUND;
            step->u_ub[a] = BIG_BOUND;
        }
    }

    check_true(solve_condensed_oracle(steps, terminal_Q, terminal_q, x0,
        nx, nu, horizon, oracle_u, oracle_x),
        "independent dense condensed QP oracle is nonsingular");
    const int pass_ok = riccati_solver_pass(
        steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
        nx, nu, horizon, 0.0f, 0.0f,
        (const float (*)[RICCATI_MAX_NX])z_x,
        (const float (*)[RICCATI_MAX_NX])y_x,
        (const float (*)[RICCATI_MAX_NU])z_u,
        (const float (*)[RICCATI_MAX_NU])y_u,
        riccati_x, riccati_u);
    check_true(pass_ok, "dense dimensioned Riccati pass succeeds");
    if (!pass_ok) return;

    float max_control_error = 0.0f;
    float max_state_error = 0.0f;
    for (int k = 0; k < horizon; ++k) {
        for (int a = 0; a < nu; ++a) {
            const float error = fabsf(riccati_u[k][a] - oracle_u[k * nu + a]);
            if (error > max_control_error) max_control_error = error;
        }
    }
    for (int k = 0; k <= horizon; ++k) {
        for (int i = 0; i < nx; ++i) {
            const float error = fabsf(riccati_x[k][i] - oracle_x[k][i]);
            if (error > max_state_error) max_state_error = error;
        }
    }
    check_true(max_control_error < 2.0e-4f,
        "Riccati controls match an independent full-matrix QP solution");
    check_true(max_state_error < 2.0e-4f,
        "Riccati state rollout matches independent full-matrix propagation");
    printf("dense %d-state QP parity: max_control_error=%.7g "
           "max_state_error=%.7g\n",
           nx, max_control_error, max_state_error);
}

static void test_n_state_admm_projects_all_constraint_channels(int nx)
{
    const int nu = 2;
    const int horizon = 8;
    RiccatiStepData_t steps[PREDICTION_HORIZON] = {0};
    float terminal_Q[RICCATI_MAX_NX] = {0};
    float terminal_q[RICCATI_MAX_NX] = {0};
    float terminal_lb[RICCATI_MAX_NX];
    float terminal_ub[RICCATI_MAX_NX];
    float x0[RICCATI_MAX_NX] = {0};
    RiccatiAdmmState_t admm_state;
    RiccatiSolution_t solution;
    const RiccatiAdmmConfig_t config = {
        .rho = 5.0f,
        .rho_u = 5.0f,
        .tolerance = 1.0e-4f,
        .max_iterations = 500,
        .adaptive_rho = 0,
        .shared_rho = 0,
    };
    memset(&solution, 0, sizeof(solution));
    riccati_admm_state_init(&admm_state);

    for (int i = 0; i < nx; ++i) {
        terminal_Q[i] = 0.5f;
        terminal_lb[i] = -BIG_BOUND;
        terminal_ub[i] = BIG_BOUND;
        x0[i] = 0.0f;
        for (int k = 0; k < horizon; ++k) {
            steps[k].A[i][i] = 0.9f;
            steps[k].Q_diag[i] = 0.5f;
            steps[k].x_lb[i] = -BIG_BOUND;
            steps[k].x_ub[i] = BIG_BOUND;
        }
    }

    /* Two distinct constrained states are influenced by separate controls. */
    x0[0] = 0.1f;
    x0[5] = 0.2f;
    terminal_lb[0] = -0.2f;
    terminal_ub[0] = 0.2f;
    terminal_lb[5] = 0.0f;
    terminal_ub[5] = 0.4f;
    terminal_q[0] = -1.0f;
    terminal_q[5] = -0.5f;
    for (int k = 0; k < horizon; ++k) {
        steps[k].B[0][0] = 0.4f;
        steps[k].B[5][1] = 0.3f;
        steps[k].Q_diag[0] = 1.0f;
        steps[k].Q_diag[5] = 1.0f;
        steps[k].q[0] = -1.0f;
        steps[k].q[5] = -0.5f;
        steps[k].x_lb[0] = -0.2f;
        steps[k].x_ub[0] = 0.2f;
        steps[k].x_lb[5] = 0.0f;
        steps[k].x_ub[5] = 0.4f;
        steps[k].R_diag[0] = 0.2f;
        steps[k].R_diag[1] = 0.2f;
        steps[k].u_lb[0] = -0.5f;
        steps[k].u_ub[0] = 0.5f;
        steps[k].u_lb[1] = -0.5f;
        steps[k].u_ub[1] = 0.5f;
    }

    const RiccatiStatus_t status = riccati_admm_solve(
        steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
        nx, nu, horizon, &config, &admm_state, &solution);
    check_true(status == RICCATI_STATUS_OPTIMAL,
        "dimensioned constrained ADMM reaches its declared stopping tolerance");
    for (int k = 1; k <= horizon; ++k) {
        check_true(admm_state.z_x[k][0] >= -0.2001f &&
                   admm_state.z_x[k][0] <= 0.2001f,
            "ADMM projects the cross-track state at every predicted stage");
        check_true(admm_state.z_x[k][5] >= -0.0001f &&
                   admm_state.z_x[k][5] <= 0.4001f,
            "ADMM projects the target-speed state at every predicted stage");
    }
    for (int k = 0; k < horizon; ++k) {
        for (int a = 0; a < nu; ++a) {
            check_true(solution.u[k][a] >= -0.5001f &&
                       solution.u[k][a] <= 0.5001f,
                "ADMM returns projected, control-bound-feasible inputs");
        }
    }
    check_true(solution.primal_residual < 0.05f && solution.dual_residual < 0.05f,
        "ADMM reports finite small primal and dual residuals");
}

static void test_large_unrelated_state_does_not_loosen_residual_gate(void)
{
    const int nx = 9;
    const int nu = 2;
    const int horizon = 1;
    RiccatiStepData_t steps[PREDICTION_HORIZON] = {0};
    float terminal_Q[RICCATI_MAX_NX] = {0};
    float terminal_q[RICCATI_MAX_NX] = {0};
    float terminal_lb[RICCATI_MAX_NX];
    float terminal_ub[RICCATI_MAX_NX];
    float x0[RICCATI_MAX_NX] = {0};
    RiccatiAdmmState_t admm_state;
    RiccatiSolution_t solution = {0};
    const RiccatiAdmmConfig_t config = {
        .rho = 7.0f,
        .rho_u = 7.0f,
        .tolerance = 0.01f,
        .max_iterations = 1,
        .adaptive_rho = 0,
        .shared_rho = 0,
    };

    for (int i = 0; i < nx; ++i) {
        terminal_lb[i] = -BIG_BOUND;
        terminal_ub[i] = BIG_BOUND;
        steps[0].x_lb[i] = -BIG_BOUND;
        steps[0].x_ub[i] = BIG_BOUND;
    }
    for (int a = 0; a < nu; ++a) {
        steps[0].R_diag[a] = 1.0f;
        steps[0].u_lb[a] = -BIG_BOUND;
        steps[0].u_ub[a] = BIG_BOUND;
    }

    /* x0[0] is deliberately large but unconstrained. It creates x1[1]=1.1,
     * only 0.1 beyond the terminal upper bound. A raw relative threshold
     * scaled by max(|x|)=10 would incorrectly accept this against 0.01 as
     * 0.01 + 0.02*10 = 0.21. */
    x0[0] = 10.0f;
    steps[0].A[1][0] = 0.11f;
    terminal_lb[1] = 0.0f;
    terminal_ub[1] = 1.0f;

    riccati_admm_state_init(&admm_state);
    const RiccatiStatus_t status = riccati_admm_solve(
        steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
        nx, nu, horizon, &config, &admm_state, &solution);
    check_true(status == RICCATI_STATUS_MAX_ITERATIONS,
        "large unrelated state cannot make a 0.1 constraint residual optimal");
    check_true(fabsf(solution.primal_residual - 0.1f) < 1.0e-4f,
        "reported primal residual remains in the constrained state units");
}

static void test_bounded_control_hessian_regularization(void)
{
    const int nx = 9;
    const int nu = 2;
    const int horizon = 1;
    RiccatiStepData_t steps[PREDICTION_HORIZON] = {0};
    float terminal_Q[RICCATI_MAX_NX] = {0};
    float terminal_q[RICCATI_MAX_NX] = {0};
    float terminal_lb[RICCATI_MAX_NX];
    float terminal_ub[RICCATI_MAX_NX];
    float x0[RICCATI_MAX_NX] = {0};
    float z_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float y_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float z_u[PREDICTION_HORIZON][RICCATI_MAX_NU] = {{0}};
    float y_u[PREDICTION_HORIZON][RICCATI_MAX_NU] = {{0}};
    float output_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float output_u[PREDICTION_HORIZON][RICCATI_MAX_NU] = {{0}};
    for (int i = 0; i < nx; ++i) {
        terminal_lb[i] = -BIG_BOUND;
        terminal_ub[i] = BIG_BOUND;
        steps[0].x_lb[i] = -BIG_BOUND;
        steps[0].x_ub[i] = BIG_BOUND;
    }
    check_true(riccati_solver_pass(
        steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
        nx, nu, horizon, 0.0f, 0.0f,
        (const float (*)[RICCATI_MAX_NX])z_x,
        (const float (*)[RICCATI_MAX_NX])y_x,
        (const float (*)[RICCATI_MAX_NU])z_u,
        (const float (*)[RICCATI_MAX_NU])y_u,
        output_x, output_u),
        "bounded regularization repairs a zero 9-state control Hessian");

    RiccatiDebugInfo_t debug = {0};
    riccati_debug_get_last(&debug);
    check_true(debug.control_hessian_regularization_count > 0 &&
               debug.max_control_hessian_regularization > 0.0f &&
               debug.max_control_hessian_regularization <= 1.0e-2f,
        "regularization diagnostics expose count and bounded maximum lambda");
    printf("9-state Riccati regularization: count=%d max_lambda=%.7g\n",
        debug.control_hessian_regularization_count,
        debug.max_control_hessian_regularization);

    steps[0].R_diag[0] = -0.1f;
    check_true(!riccati_solver_pass(
        steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
        nx, nu, horizon, 0.0f, 0.0f,
        (const float (*)[RICCATI_MAX_NX])z_x,
        (const float (*)[RICCATI_MAX_NX])y_x,
        (const float (*)[RICCATI_MAX_NU])z_u,
        (const float (*)[RICCATI_MAX_NU])y_u,
        output_x, output_u),
        "materially indefinite control Hessian fails after bounded repair");
}

int main(void)
{
    test_warm_start_shifts_every_stage();
    test_dense_n_state_riccati_against_condensed_qp(9);
    test_dense_n_state_riccati_against_condensed_qp(10);
    test_n_state_admm_projects_all_constraint_channels(9);
    test_n_state_admm_projects_all_constraint_channels(10);
    test_large_unrelated_state_does_not_loosen_residual_gate();
    test_bounded_control_hessian_regularization();
    if (failures != 0) {
        fprintf(stderr, "%d Riccati-ADMM test(s) failed\n", failures);
        return 1;
    }
    puts("Dimensioned Riccati-ADMM tests passed");
    return 0;
}

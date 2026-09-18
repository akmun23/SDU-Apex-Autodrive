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

typedef struct
{
    float c[2];
    float upper;
} OracleHalfspace_t;

static float two_input_objective(
    float H[2][2], const float g[2], const float u[2])
{
    return 0.5f * (u[0] * (H[0][0] * u[0] + H[0][1] * u[1]) +
                   u[1] * (H[1][0] * u[0] + H[1][1] * u[1])) +
        g[0] * u[0] + g[1] * u[1];
}

static int halfspaces_feasible(
    const OracleHalfspace_t *constraints, int count, const float u[2])
{
    for (int i = 0; i < count; ++i) {
        if (constraints[i].c[0] * u[0] + constraints[i].c[1] * u[1] >
            constraints[i].upper + 2.0e-5f) return 0;
    }
    return 1;
}

static void consider_oracle_candidate(
    float H[2][2], const float g[2],
    const OracleHalfspace_t *constraints, int constraint_count,
    const float candidate[2], float *best_cost, float best_u[2])
{
    if (!isfinite(candidate[0]) || !isfinite(candidate[1]) ||
        !halfspaces_feasible(constraints, constraint_count, candidate)) return;
    const float cost = two_input_objective(H, g, candidate);
    if (cost < *best_cost) {
        *best_cost = cost;
        best_u[0] = candidate[0];
        best_u[1] = candidate[1];
    }
}

/* Independent exact active-set oracle for a one-stage, two-input LTV QP.
 * The optimizer has only two decisions, so enumerate unconstrained, one-face,
 * and two-face KKT candidates. The Riccati-ADMM implementation is not used by
 * this oracle. */
static int solve_horizon_one_box_qp_oracle(
    const RiccatiStepData_t *step,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_lb,
    const float *terminal_ub,
    const float *x0,
    int nx,
    float best_u[2],
    float *best_cost)
{
    OracleHalfspace_t constraints[2 * RICCATI_MAX_NX + 4];
    int constraint_count = 0;
    float affine_next[RICCATI_MAX_NX] = {0};
    float H[2][2] = {{0}};
    float g[2] = {0};

    for (int i = 0; i < nx; ++i) {
        affine_next[i] = step->d[i];
        for (int j = 0; j < nx; ++j)
            affine_next[i] += step->A[i][j] * x0[j];
    }
    for (int a = 0; a < 2; ++a) {
        H[a][a] = step->R_diag[a];
        g[a] = step->r[a];
        for (int i = 0; i < nx; ++i) {
            g[a] += x0[i] * step->N[i][a] +
                step->B[i][a] *
                (terminal_Q[i] * affine_next[i] + terminal_q[i]);
            for (int b = 0; b < 2; ++b)
                H[a][b] += step->B[i][a] * terminal_Q[i] * step->B[i][b];
        }

        constraints[constraint_count++] = (OracleHalfspace_t){
            .c = {a == 0 ? 1.0f : 0.0f, a == 1 ? 1.0f : 0.0f},
            .upper = step->u_ub[a]};
        constraints[constraint_count++] = (OracleHalfspace_t){
            .c = {a == 0 ? -1.0f : 0.0f, a == 1 ? -1.0f : 0.0f},
            .upper = -step->u_lb[a]};
    }
    for (int i = 0; i < nx; ++i) {
        if (terminal_ub[i] < BIG_BOUND) {
            constraints[constraint_count++] = (OracleHalfspace_t){
                .c = {step->B[i][0], step->B[i][1]},
                .upper = terminal_ub[i] - affine_next[i]};
        }
        if (terminal_lb[i] > -BIG_BOUND) {
            constraints[constraint_count++] = (OracleHalfspace_t){
                .c = {-step->B[i][0], -step->B[i][1]},
                .upper = affine_next[i] - terminal_lb[i]};
        }
    }

    *best_cost = INFINITY;
    float matrix[ORACLE_MAX_CONTROLS][ORACLE_MAX_CONTROLS] = {{0}};
    float rhs[ORACLE_MAX_CONTROLS] = {0};
    float candidate[2] = {0};

    /* No active constraints. */
    matrix[0][0] = H[0][0];
    matrix[0][1] = H[0][1];
    matrix[1][0] = H[1][0];
    matrix[1][1] = H[1][1];
    rhs[0] = -g[0];
    rhs[1] = -g[1];
    if (solve_dense_system(matrix, rhs, 2)) {
        candidate[0] = rhs[0];
        candidate[1] = rhs[1];
        consider_oracle_candidate(H, g, constraints, constraint_count,
                                  candidate, best_cost, best_u);
    }

    /* One active face: solve [H C'; C 0] [u;lambda] = [-g;b]. */
    for (int i = 0; i < constraint_count; ++i) {
        memset(matrix, 0, sizeof(matrix));
        memset(rhs, 0, sizeof(rhs));
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 2; ++col)
                matrix[row][col] = H[row][col];
            matrix[row][2] = constraints[i].c[row];
            matrix[2][row] = constraints[i].c[row];
        }
        rhs[0] = -g[0];
        rhs[1] = -g[1];
        rhs[2] = constraints[i].upper;
        if (solve_dense_system(matrix, rhs, 3)) {
            candidate[0] = rhs[0];
            candidate[1] = rhs[1];
            consider_oracle_candidate(H, g, constraints, constraint_count,
                                      candidate, best_cost, best_u);
        }
    }

    /* Two active faces meet at a vertex in the two-dimensional decision set. */
    for (int i = 0; i < constraint_count; ++i) {
        for (int j = i + 1; j < constraint_count; ++j) {
            memset(matrix, 0, sizeof(matrix));
            memset(rhs, 0, sizeof(rhs));
            for (int row = 0; row < 2; ++row) {
                matrix[0][row] = constraints[i].c[row];
                matrix[1][row] = constraints[j].c[row];
            }
            rhs[0] = constraints[i].upper;
            rhs[1] = constraints[j].upper;
            if (solve_dense_system(matrix, rhs, 2)) {
                candidate[0] = rhs[0];
                candidate[1] = rhs[1];
                consider_oracle_candidate(H, g, constraints,
                    constraint_count, candidate, best_cost, best_u);
            }
        }
    }
    return isfinite(*best_cost);
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

static void test_constrained_admm_matches_independent_horizon_one_oracle(int nx)
{
    const int nu = 2;
    const int horizon = 1;
    RiccatiStepData_t steps[PREDICTION_HORIZON] = {0};
    float terminal_Q[RICCATI_MAX_NX] = {0};
    float terminal_q[RICCATI_MAX_NX] = {0};
    float terminal_lb[RICCATI_MAX_NX];
    float terminal_ub[RICCATI_MAX_NX];
    float x0[RICCATI_MAX_NX] = {0};
    float oracle_u[2] = {0};
    float oracle_cost = INFINITY;
    RiccatiAdmmState_t admm_state;
    RiccatiSolution_t solution = {0};
    const RiccatiAdmmConfig_t config = {
        .rho = 3.0f,
        .rho_u = 3.0f,
        .tolerance = 1.0e-5f,
        .max_iterations = 5000,
        .adaptive_rho = 0,
        .shared_rho = 0,
    };

    for (int i = 0; i < nx; ++i) {
        x0[i] = 0.045f * (float)(i - 4);
        steps[0].d[i] = 0.012f * (float)(i - 4);
        terminal_Q[i] = 1.0f + 0.1f * (float)i;
        terminal_lb[i] = -BIG_BOUND;
        terminal_ub[i] = BIG_BOUND;
        steps[0].Q_diag[i] = 0.6f + 0.05f * (float)i;
        steps[0].q[i] = 0.01f * (float)(i - 3);
        steps[0].x_lb[i] = -BIG_BOUND;
        steps[0].x_ub[i] = BIG_BOUND;
        for (int j = 0; j < nx; ++j) {
            if (i == j) {
                steps[0].A[i][j] = 0.55f + 0.01f * (float)i;
            } else {
                const int residue = (i * 5 + j * 3) % 5 - 2;
                steps[0].A[i][j] = 0.002f * (float)residue;
            }
        }
        for (int a = 0; a < nu; ++a) {
            const float sign = ((i + a) % 2) ? -1.0f : 1.0f;
            steps[0].B[i][a] = sign *
                (0.012f + 0.003f * (float)((i + 2 * a) % 4));
            steps[0].N[i][a] = sign *
                (0.006f + 0.001f * (float)((i + a) % 3));
        }
    }
    steps[0].B[0][0] = 0.45f;
    steps[0].B[0][1] = 0.10f;
    steps[0].B[5][0] = -0.12f;
    steps[0].B[5][1] = 0.42f;
    terminal_q[0] = -5.0f;
    terminal_q[5] = 3.0f;
    terminal_lb[0] = -0.40f;
    terminal_ub[0] = -0.08f;
    terminal_lb[5] = -0.04f;
    terminal_ub[5] = 0.30f;
    steps[0].R_diag[0] = 0.4f;
    steps[0].R_diag[1] = 0.3f;
    steps[0].r[0] = 0.02f;
    steps[0].r[1] = -0.03f;
    steps[0].u_lb[0] = -0.35f;
    steps[0].u_ub[0] = 0.25f;
    steps[0].u_lb[1] = -0.40f;
    steps[0].u_ub[1] = 0.30f;

    check_true(solve_horizon_one_box_qp_oracle(
            &steps[0], terminal_Q, terminal_q, terminal_lb, terminal_ub,
            x0, nx, oracle_u, &oracle_cost),
        "independent active-set oracle finds a feasible constrained optimum");
    riccati_admm_state_init(&admm_state);
    const RiccatiStatus_t status = riccati_admm_solve(
        steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
        nx, nu, horizon, &config, &admm_state, &solution);
    check_true(status == RICCATI_STATUS_OPTIMAL,
        "affine/cross-cost constrained ADMM reaches absolute tolerance");

    const float control[2] = {solution.u[0][0], solution.u[0][1]};
    float affine_next[RICCATI_MAX_NX] = {0};
    float oracle_H[2][2] = {{0}};
    float oracle_g[2] = {0};
    for (int i = 0; i < nx; ++i) {
        affine_next[i] = steps[0].d[i];
        for (int j = 0; j < nx; ++j)
            affine_next[i] += steps[0].A[i][j] * x0[j];
    }
    for (int a = 0; a < nu; ++a) {
        oracle_H[a][a] = steps[0].R_diag[a];
        oracle_g[a] = steps[0].r[a];
        for (int i = 0; i < nx; ++i) {
            oracle_g[a] += x0[i] * steps[0].N[i][a] +
                steps[0].B[i][a] *
                    (terminal_Q[i] * affine_next[i] + terminal_q[i]);
            for (int b = 0; b < nu; ++b)
                oracle_H[a][b] += steps[0].B[i][a] * terminal_Q[i] *
                    steps[0].B[i][b];
        }
    }
    const float full_returned_cost = two_input_objective(
        oracle_H, oracle_g, control);
    check_true(fabsf(control[0] - oracle_u[0]) < 2.0e-3f &&
               fabsf(control[1] - oracle_u[1]) < 2.0e-3f,
        "constrained ADMM inputs match independent active-set optimum");
    check_true(fabsf(full_returned_cost - oracle_cost) < 2.0e-4f,
        "constrained ADMM objective matches independent active-set optimum");
    check_true(solution.primal_residual <= config.tolerance &&
               solution.dual_residual <= config.tolerance,
        "optimal ADMM status satisfies absolute primal and dual tolerances");

    RiccatiAdmmConfig_t scaled_config = config;
    scaled_config.use_prefactorization = 1;
    scaled_config.use_scaling = 1;
    const float state_scales[RICCATI_MAX_NX] = {
        0.5f, 0.5f, 10.0f, 0.5f, 3.2f, 10.0f, 0.5235988f, 3.2f, 8.0f,
        1.0f};
    const float input_scales[RICCATI_MAX_NU] = {3.2f, 8.0f};
    memcpy(scaled_config.state_scale, state_scales, sizeof(state_scales));
    memcpy(scaled_config.input_scale, input_scales, sizeof(input_scales));
    RiccatiAdmmState_t scaled_state = {0};
    RiccatiSolution_t scaled_solution = {0};
    const RiccatiStatus_t scaled_status = riccati_admm_solve(
        steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
        nx, nu, horizon, &scaled_config, &scaled_state, &scaled_solution);
    check_true(scaled_status == RICCATI_STATUS_OPTIMAL,
        "dimensionless constrained ADMM reaches its declared tolerance");
    check_true(scaled_state.scaling_enabled == 1,
        "scaled warm-start state records its coordinate transform");
    check_true(fabsf(scaled_solution.u[0][0] - control[0]) < 5.0e-4f &&
               fabsf(scaled_solution.u[0][1] - control[1]) < 5.0e-4f,
        "scaled and physical ADMM controls match on affine cross-cost QP");
    check_true(fabsf(scaled_solution.u[0][0] - oracle_u[0]) < 2.0e-3f &&
               fabsf(scaled_solution.u[0][1] - oracle_u[1]) < 2.0e-3f,
        "scaled ADMM preserves the independent active-set optimum");

    /* Changing the scale invalidates normalized z/y warm-start coordinates. */
    scaled_config.state_scale[0] *= 1.1f;
    RiccatiSolution_t invalidated_solution = {0};
    RiccatiAdmmState_t cold_scaled_state = {0};
    RiccatiSolution_t cold_scaled_solution = {0};
    const RiccatiStatus_t invalidated_status = riccati_admm_solve(
        steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
        nx, nu, horizon, &scaled_config, &scaled_state,
        &invalidated_solution);
    const RiccatiStatus_t cold_scaled_status = riccati_admm_solve(
        steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
        nx, nu, horizon, &scaled_config, &cold_scaled_state,
        &cold_scaled_solution);
    check_true(invalidated_status == cold_scaled_status &&
               invalidated_solution.iterations == cold_scaled_solution.iterations,
        "scale changes discard incompatible warm starts");
    check_true(fabsf(invalidated_solution.u[0][0] -
                     cold_scaled_solution.u[0][0]) < 1.0e-7f &&
               fabsf(invalidated_solution.u[0][1] -
                     cold_scaled_solution.u[0][1]) < 1.0e-7f,
        "changed-scale solve matches a cold scaled solve");
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

static void test_factored_riccati_pass_matches_reference(void)
{
    const int nx = 9;
    const int nu = 2;
    const int horizon = PREDICTION_HORIZON;
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
    float reference_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float reference_u[PREDICTION_HORIZON][RICCATI_MAX_NU] = {{0}};
    float factored_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX] = {{0}};
    float factored_u[PREDICTION_HORIZON][RICCATI_MAX_NU] = {{0}};

    for (int i = 0; i < nx; ++i) {
        terminal_Q[i] = 1.0f + 0.1f * (float)i;
        terminal_q[i] = 0.015f * (float)(i - 3);
        terminal_lb[i] = -BIG_BOUND;
        terminal_ub[i] = BIG_BOUND;
        x0[i] = 0.02f * (float)(i - 4);
        if (i == 0 || i == 2 || i == 5 || i == 6) {
            terminal_lb[i] = -0.8f;
            terminal_ub[i] = 0.8f;
        }
        for (int k = 0; k < horizon; ++k) {
            steps[k].A[i][i] = 0.82f + 0.005f * (float)(i % 3);
            for (int j = 0; j < nx; ++j) {
                if (i != j)
                    steps[k].A[i][j] = 0.0007f *
                        (float)(((i + 3 * j + k) % 5) - 2);
            }
            steps[k].d[i] = 0.0003f * (float)((i + 2 * k) % 7 - 3);
            steps[k].Q_diag[i] = 0.7f + 0.08f * (float)(i % 4);
            steps[k].q[i] = 0.012f * (float)((2 * i + k) % 5 - 2);
            steps[k].x_lb[i] = (i == 0 || i == 2 || i == 5 || i == 6)
                ? -0.8f : -BIG_BOUND;
            steps[k].x_ub[i] = (i == 0 || i == 2 || i == 5 || i == 6)
                ? 0.8f : BIG_BOUND;
            z_x[k][i] = 0.004f * (float)((i + k) % 7 - 3);
            y_x[k][i] = 0.002f * (float)((2 * i + k) % 9 - 4);
        }
        z_x[horizon][i] = 0.003f * (float)(i - 4);
        y_x[horizon][i] = -0.001f * (float)(i - 4);
    }
    for (int k = 0; k < horizon; ++k) {
        for (int a = 0; a < nu; ++a) {
            steps[k].R_diag[a] = 0.4f + 0.2f * (float)a;
            steps[k].r[a] = 0.01f * (float)(k - 12 * a);
            steps[k].u_lb[a] = -1.0f;
            steps[k].u_ub[a] = 1.0f;
            z_u[k][a] = 0.01f * (float)((k + a) % 9 - 4);
            y_u[k][a] = -0.005f * (float)((2 * k + a) % 7 - 3);
            for (int i = 0; i < nx; ++i) {
                steps[k].B[i][a] = 0.01f *
                    (float)(((i + 2 * a + k) % 7) - 3);
                steps[k].N[i][a] = 0.002f *
                    (float)(((3 * i + a + k) % 5) - 2);
            }
        }
    }

    const float rho = 7.0f;
    const float rho_u = 7.0f;
    const int reference_ok = riccati_solver_pass(
        steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
        nx, nu, horizon, rho, rho_u,
        (const float (*)[RICCATI_MAX_NX])z_x,
        (const float (*)[RICCATI_MAX_NX])y_x,
        (const float (*)[RICCATI_MAX_NU])z_u,
        (const float (*)[RICCATI_MAX_NU])y_u,
        reference_x, reference_u);
    RiccatiFactorization_t factorization = {0};
    const int factorized_ok = riccati_solver_factorize(
        steps, terminal_Q, terminal_lb, terminal_ub, nx, nu, horizon,
        rho, rho_u, &factorization) &&
        riccati_solver_pass_factored(
            steps, terminal_q, terminal_lb, terminal_ub, x0,
            nx, nu, horizon, &factorization,
            (const float (*)[RICCATI_MAX_NX])z_x,
            (const float (*)[RICCATI_MAX_NX])y_x,
            (const float (*)[RICCATI_MAX_NU])z_u,
            (const float (*)[RICCATI_MAX_NU])y_u,
            factored_x, factored_u);
    check_true(reference_ok && factorized_ok,
        "reference and factorized Riccati passes both complete");
    if (!reference_ok || !factorized_ok) return;

    float max_state_error = 0.0f;
    float max_control_error = 0.0f;
    for (int k = 0; k <= horizon; ++k)
        for (int i = 0; i < nx; ++i)
            max_state_error = fmaxf(max_state_error,
                fabsf(reference_x[k][i] - factored_x[k][i]));
    for (int k = 0; k < horizon; ++k)
        for (int a = 0; a < nu; ++a)
            max_control_error = fmaxf(max_control_error,
                fabsf(reference_u[k][a] - factored_u[k][a]));
    check_true(max_state_error <= 1.0e-5f,
        "factored Riccati states match the refactor-every-pass oracle");
    check_true(max_control_error <= 1.0e-5f,
        "factored Riccati controls match the refactor-every-pass oracle");
    printf("Riccati factorization parity: max_control_error=%.7g "
           "max_state_error=%.7g\n", max_control_error, max_state_error);
}

static void test_prefactorized_admm_matches_reference_over_warm_cycles(void)
{
    const int nx = 9;
    const int nu = 2;
    const int horizon = PREDICTION_HORIZON;
    RiccatiStepData_t steps[PREDICTION_HORIZON] = {0};
    float terminal_Q[RICCATI_MAX_NX] = {0};
    float terminal_q[RICCATI_MAX_NX] = {0};
    float terminal_lb[RICCATI_MAX_NX];
    float terminal_ub[RICCATI_MAX_NX];
    float x0[RICCATI_MAX_NX] = {0};
    RiccatiAdmmState_t reference_state = {0};
    RiccatiAdmmState_t factored_state = {0};
    RiccatiAdmmConfig_t reference_config = {
        .rho = 7.0f, .rho_u = 7.0f, .tolerance = 0.01f,
        .max_iterations = 100, .adaptive_rho = 0, .shared_rho = 0,
        .use_prefactorization = 0};
    RiccatiAdmmConfig_t factored_config = reference_config;
    factored_config.use_prefactorization = 1;

    for (int i = 0; i < nx; ++i) {
        terminal_Q[i] = 1.0f + 0.1f * (float)i;
        terminal_lb[i] = -BIG_BOUND;
        terminal_ub[i] = BIG_BOUND;
        for (int k = 0; k < horizon; ++k) {
            steps[k].A[i][i] = 0.82f + 0.005f * (float)(i % 3);
            steps[k].Q_diag[i] = 0.7f + 0.08f * (float)(i % 4);
            steps[k].x_lb[i] = -BIG_BOUND;
            steps[k].x_ub[i] = BIG_BOUND;
            if (i == 0 || i == 2 || i == 6) {
                steps[k].x_lb[i] = -0.75f;
                steps[k].x_ub[i] = 0.75f;
            }
            for (int j = 0; j < nx; ++j) {
                if (i != j)
                    steps[k].A[i][j] = 0.0004f *
                        (float)(((i + 2 * j + k) % 5) - 2);
            }
            for (int a = 0; a < nu; ++a) {
                steps[k].B[i][a] = 0.006f *
                    (float)(((i + 3 * a + k) % 7) - 3);
                steps[k].N[i][a] = 0.001f *
                    (float)(((2 * i + a + k) % 5) - 2);
            }
        }
    }
    steps[0].B[0][0] = 0.35f;
    steps[0].B[2][1] = 0.28f;
    for (int k = 0; k < horizon; ++k) {
        for (int a = 0; a < nu; ++a) {
            steps[k].R_diag[a] = 0.45f + 0.15f * (float)a;
            steps[k].u_lb[a] = -0.5f;
            steps[k].u_ub[a] = 0.5f;
        }
    }
    terminal_lb[0] = -0.75f;
    terminal_ub[0] = 0.75f;

    for (int cycle = 0; cycle < 4; ++cycle) {
        for (int i = 0; i < nx; ++i) {
            x0[i] = 0.006f * (float)((i + cycle) % 7 - 3);
            terminal_q[i] = 0.01f * (float)((2 * i + cycle) % 7 - 3);
            for (int k = 0; k < horizon; ++k)
                steps[k].q[i] = 0.002f *
                    (float)((i + 2 * k + cycle) % 9 - 4);
        }
        RiccatiSolution_t reference = {0};
        RiccatiSolution_t factored = {0};
        const RiccatiStatus_t reference_status = riccati_admm_solve(
            steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
            nx, nu, horizon, &reference_config, &reference_state, &reference);
        const RiccatiStatus_t factored_status = riccati_admm_solve(
            steps, terminal_Q, terminal_q, terminal_lb, terminal_ub, x0,
            nx, nu, horizon, &factored_config, &factored_state, &factored);
        RiccatiDebugInfo_t debug = {0};
        riccati_debug_get_last(&debug);
        check_true(reference_status == factored_status,
            "prefactorized ADMM preserves status over warm-started QPs");
        check_true(reference.iterations == factored.iterations,
            "prefactorized ADMM preserves iteration count");
        check_true(fabsf(reference.primal_residual - factored.primal_residual)
                       <= 1.0e-6f &&
                   fabsf(reference.dual_residual - factored.dual_residual)
                       <= 1.0e-6f,
            "prefactorized ADMM preserves final residuals");
        check_true(debug.quadratic_factorization_count == 1,
            "fixed-rho prefactorized solve factors once per QP");
        float max_state_error = 0.0f;
        float max_control_error = 0.0f;
        for (int k = 0; k <= horizon; ++k)
            for (int i = 0; i < nx; ++i)
                max_state_error = fmaxf(max_state_error,
                    fabsf(reference.x[k][i] - factored.x[k][i]));
        for (int k = 0; k < horizon; ++k)
            for (int a = 0; a < nu; ++a)
                max_control_error = fmaxf(max_control_error,
                    fabsf(reference.u[k][a] - factored.u[k][a]));
        check_true(max_state_error <= 1.0e-5f &&
                   max_control_error <= 1.0e-5f,
            "prefactorized ADMM matches reference states and controls");
    }
}

int main(void)
{
    test_warm_start_shifts_every_stage();
    test_dense_n_state_riccati_against_condensed_qp(9);
    test_dense_n_state_riccati_against_condensed_qp(10);
    test_n_state_admm_projects_all_constraint_channels(9);
    test_n_state_admm_projects_all_constraint_channels(10);
    test_constrained_admm_matches_independent_horizon_one_oracle(9);
    test_constrained_admm_matches_independent_horizon_one_oracle(10);
    test_large_unrelated_state_does_not_loosen_residual_gate();
    test_bounded_control_hessian_regularization();
    test_factored_riccati_pass_matches_reference();
    test_prefactorized_admm_matches_reference_over_warm_cycles();
    if (failures != 0) {
        fprintf(stderr, "%d Riccati-ADMM test(s) failed\n", failures);
        return 1;
    }
    puts("Dimensioned Riccati-ADMM tests passed");
    return 0;
}

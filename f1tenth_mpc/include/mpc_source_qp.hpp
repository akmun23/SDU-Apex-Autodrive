#pragma once
/* Convex QP of one SQP step, solved by a primal-dual interior-point method.
 *
 *   minimize   1/2 x'Hx + g'x + sum_e ( 1/2 w_e t_e^2 + c_e t_e )
 *   subject to lower <= x <= upper
 *              A x + b >= 0                      (hard rows)
 *              t_e >= 0,  C_e x + d_e + t_e >= 0 (elastic rows)
 *
 * An elastic row with c_e = 0 and w_e > 0 is the exact epigraph of the convex
 * hinge  1/2 w_e max(0, -(C_e x + d_e))^2;  with c_e > 0 it is an exact L1
 * penalty on the row's violation. All inequality blocks are condensed into one
 * dense n x n Cholesky factorization per iteration.
 */
#include <Eigen/Core>

namespace sdu_apex::source_sqp {

struct ConvexQp {
    Eigen::MatrixXd hessian;
    Eigen::VectorXd gradient;
    Eigen::VectorXd lower, upper;  // finite, lower <= upper
    Eigen::MatrixXd hard_jacobian;
    Eigen::VectorXd hard_margin;
    Eigen::MatrixXd elastic_jacobian;
    Eigen::VectorXd elastic_margin;
    Eigen::VectorXd elastic_weight;       // w_e >= 0
    Eigen::VectorXd elastic_linear_cost;  // c_e >= 0
};

struct ConvexQpResult {
    Eigen::VectorXd step;
    int iterations{};
    bool converged{};
    bool deadline_exceeded{};
};

// `tolerance` bounds the stationarity and complementarity residuals in the
// problem's own (unscaled) units. Returns false only for malformed input.
bool solve_convex_qp(const ConvexQp &problem, int maximum_iterations,
                     double tolerance, ConvexQpResult *result,
                     double absolute_deadline_monotonic_us = 0.);

} // namespace sdu_apex::source_sqp

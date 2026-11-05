#pragma once

#include <Eigen/Core>

namespace sdu_apex::source_nlp {

// Convex SQP subproblem only. Physical hard rows are A*x+b >= 0.
// Preferred rows use explicit elastic variables t >= 0, C*x+d+t >= 0
// and cost .5*sum(w*t^2). No physical row receives an elastic variable.
struct SourceConvexQp {
    Eigen::MatrixXd hessian;
    Eigen::VectorXd gradient;
    Eigen::MatrixXd hard_jacobian;
    Eigen::VectorXd hard_margin;
    Eigen::MatrixXd elastic_jacobian;
    Eigen::VectorXd elastic_margin;
    Eigen::VectorXd elastic_weight;
    // Optional exact-L1 elastic SQP globalization. This regularizes an
    // infeasible local linearization; callers still certify the original NLP.
    Eigen::VectorXd elastic_linear_cost;
    // Rows may share one epigraph: t[group] >= -C[row]*x-d[row].
    // Empty means one independent epigraph per row.
    Eigen::VectorXi elastic_groups;
    Eigen::VectorXd initial_hard_multipliers;
};

struct SourceConvexQpResult {
    Eigen::VectorXd step;
    Eigen::VectorXd hard_multipliers;
    Eigen::VectorXd elastic_multipliers;
    Eigen::VectorXd elastic_values;
    double primal_inf{};
    double dual_inf{};
    double complementarity_inf{};
    int iterations{};
    bool converged{};
    bool deadline_exceeded{};
};

// Infeasible-start primal-dual predictor/corrector. Elastic variables and
// slacks are eliminated analytically; each Newton solve has control dimension.
bool solve_source_convex_qp(const SourceConvexQp &problem,
    int maximum_iterations,double tolerance,SourceConvexQpResult *result,
    double absolute_deadline_monotonic_us=0.);

} // namespace sdu_apex::source_nlp

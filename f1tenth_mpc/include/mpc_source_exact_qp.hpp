#pragma once
#include "mpc_source_exact_nlp.hpp"
#include <vector>
#include <array>

namespace sdu_apex::source_nlp {
// One local convex subproblem of the original constrained nonlinear MPC.
// Source dynamics are eliminated by EXACT stage-A/B sensitivities; source
// body-corner and actuator-rate inequalities retain their row identities.
// Positive multipliers are carried across SQP linearizations, not guessed by
// a separate feasibility-first optimizer.
// PSD secant of the ORIGINAL source nonlinear objective and constraint
// merit along a rejected full SQP step. These are Hessian corrections of
// the SAME NLP, not input trust regions or alternate candidate searches.
struct SourceNonlinearCurvatureSecant {
    std::array<double,2*kStages> unit_direction{};
    double curvature{};
};
struct SourceConstrainedQpOptions {
    double requested_corner_clearance_m{0.03};
    double curvature_regularization{1000.};
    double inequality_penalty_weight{1.e6};
    int maximum_qp_iterations{30};
    double kkt_tolerance{1.e-4};
    const std::vector<double> *inequality_multipliers{nullptr};
    const MpcRtiCycleConfiguration_t *source_configuration{nullptr};
    const MpcRtiNominal_t *source_nominal{nullptr};
    const std::vector<SourceNonlinearCurvatureSecant> *
        source_curvature_secants{nullptr};
};
struct SourceConstrainedQpResult {
    double direction[kStages][2]{};
    std::vector<double> inequality_multipliers;
    double source_cost_linear_slope{};
    double source_cost_psd_quadratic{};
    double predicted_original_objective_change{};
    double predicted_linearized_violation_before{};
    double predicted_linearized_violation_after{};
    double primal_residual{};
    double dual_residual{};
    // First-order KKT diagnostics of the ORIGINAL constrained source NLP,
    // never the elastic QP's nominal objective.
    double original_nlp_stationarity_inf{};
    double original_nlp_inequality_residual_inf{};
    double original_nlp_complementarity_inf{};
    double wall_ms{};
    int original_footprint_rows{};
    int original_rate_rows{};
    int original_box_rows{};
    int iterations{};
    bool converged{};
    bool valid{};
};
bool source_exact_constrained_qp_direction(
    const SparseNonlinearProgram &program,
    const double original_objective_control_gradient[kStages][2],
    const MpcModelControl_t nominal_controls[kStages],
    const SourceConstrainedQpOptions &options,
    SourceConstrainedQpResult *output);
} // namespace sdu_apex::source_nlp

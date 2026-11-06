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
struct SourceSlipCurvatureMultiplier {
    int stage{},wheel{},fixed_tick{},vehicle_substep{};
    size_t source_row_index{};
    double multiplier{};
};
struct SourceLagrangianMemory {
    bool valid{};
    bool anchor_valid{};
    bool linearized_controls_valid{};
    std::array<double,2*kStages*2*kStages> hessian{};
    std::array<double,2*kStages*2*kStages> racing_gauss_newton{};
    std::array<double,2*kStages> anchor_controls{};
    std::array<double,2*kStages> anchor_gradient{};
    std::array<double,2*kStages> linearized_controls{};
    std::vector<double> frozen_curvature_multipliers;
    std::vector<SourceSlipCurvatureMultiplier> frozen_slip_multipliers;
};
struct SourceConstrainedQpOptions {
    double requested_corner_clearance_m{0.03};
    double curvature_regularization{1000.};
    double inequality_penalty_weight{1.e6};
    double terminal_speed_viability_weight{};
    int maximum_qp_iterations{30};
    double kkt_tolerance{1.e-4};
    const std::vector<double> *inequality_multipliers{nullptr};
    const MpcRtiCycleConfiguration_t *source_configuration{nullptr};
    const MpcRtiNominal_t *source_nominal{nullptr};
    bool capture_curvature_model{false};
    bool linearization_audit_only{false};
    // Maratos second-order constraint defect, in original row units.
    const std::vector<double> *constraint_margin_corrections{nullptr};
    SourceLagrangianMemory *lagrangian_memory{nullptr};
    double absolute_deadline_monotonic_us{};
    const std::array<double,4*kStages*kStages> *lagrangian_hessian_scaled{};
    bool refresh_known_gauss_newton{true};
    // A failed hard QP need not run feasibility restoration when the
    // original source primal has already passed all hard constraints.
    bool original_source_primal_feasible{false};

};
struct SourceConstrainedQpResult {
    const char *failure_reason{"invalid_or_unsolved_source_qp"};
    double direction[kStages][2]{};
    double normal_direction[kStages][2]{};
    bool normal_step_computed{};
    bool normal_only_step{};
    double normal_primal_residual{};
    double normal_dual_residual{};
    bool original_duals_trustworthy{};
    double normal_linearized_violation{};
    double linearized_physical_violation_inf{};
    std::vector<double> inequality_multipliers;
    std::vector<double> predicted_constraint_margins;
    double source_cost_linear_slope{};
    double source_cost_psd_quadratic{};
    double predicted_original_objective_change{};
    double predicted_linearized_violation_before{};
    double predicted_linearized_violation_after{};
    double predicted_terminal_speed_violation_before{};
    double predicted_terminal_speed_violation_after{};
    double primal_residual{};
    double dual_residual{};
    double complementarity_residual{};
    double hessian_min_eigenvalue{};
    double hessian_max_eigenvalue{};
    double tolerance_used{};
    bool deadline_exceeded{};
    bool lagrangian_bfgs_updated{};
    double lagrangian_bfgs_damping{};
    std::array<double,2*kStages> lagrangian_gradient_at_qp_duals{};
    std::vector<double> curvature_multipliers;
    std::vector<SourceSlipCurvatureMultiplier> slip_curvature_multipliers;
    // First-order KKT diagnostics of the ORIGINAL constrained source NLP,
    // never the elastic QP's nominal objective.
    // Infinity norm in U = D_u z coordinates (0.15 rad / 0.25 throttle).
    double original_nlp_stationarity_inf{};
    double original_nlp_inequality_residual_inf{};
    double original_nlp_complementarity_inf{};
    double original_nlp_dual_feasibility_inf{};
    double wall_ms{};
    int original_footprint_rows{};
    int original_rate_rows{};
    int original_box_rows{};
    int iterations{};
    bool converged{};
    bool valid{};
    // Offline curvature audit only; empty during normal controller cycles.
    std::vector<double> racing_hessian_scaled;
    std::vector<double> stabilized_hessian_scaled;
    std::vector<double> constraint_jacobian_scaled;
    std::vector<int> constraint_row_indices;
    std::vector<double> constraint_margins;
};
bool source_exact_constrained_qp_direction(
    const SparseNonlinearProgram &program,
    const double original_objective_control_gradient[kStages][2],
    const MpcModelControl_t nominal_controls[kStages],
    const SourceConstrainedQpOptions &options,
    SourceConstrainedQpResult *output);
} // namespace sdu_apex::source_nlp

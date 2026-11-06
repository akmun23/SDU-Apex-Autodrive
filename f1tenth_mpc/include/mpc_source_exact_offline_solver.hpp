#pragma once
#include "mpc_source_exact_nlp.hpp"
#include <vector>

namespace sdu_apex::source_nlp {

// One original-source nonlinear program per 40 Hz control epoch.
// The same exact original physics, cost and inequalities determine every
// relinearization, SQP update and publication. No line-search alpha grid,
// alternate optimizer, trust box or recovery controller exists here.
struct OfflineOptions {
    int maximum_iterations{3};
    const MpcRtiNominal_t *initial_same_epoch_source_nominal{nullptr};
    double absolute_deadline_monotonic_us{};
    double minimum_remaining_us_for_new_iteration{};
    // Preferred clearance DIAGNOSTIC. It does not augment the racing objective.
    // The original-source collision boundary cannot be softened.
    double minimum_source_clearance_m{0.03};
    double allowable_corridor_violation_m{5.e-7};
    double allowable_command_rate_violation{1.e-7};
    double minimum_source_cost_reduction{};
    double minimum_violation_reduction{};
    double source_qp_curvature_regularization{1000.};
    double source_qp_soft_physical_weight{1.e6};
    // Audit-only full curvature-regime invalidation. Disabled until the
    // physical switch taxonomy is proven useful: many smooth tire region
    // changes otherwise reset every accepted BFGS update.
    bool strict_accepted_source_branch_curvature_guard{false};
    // Research-only: skip a quasi-Newton secant across a literal source
    // mode discontinuity, but DO NOT discard the existing Hessian. A tire
    // spline C1 boundary alone does not trigger this.
    bool source_discontinuous_secant_guard{false};
    // Research: exact-source second-order physical-constraint correction
    // at infeasible starts even when the first QP duals are not certified.
    // Only affects the same original SQP; never publishes an elastic output.
    bool source_infeasible_second_order_correction{false};
    // Offline experiment: form a physical correction by relinearizing
    // at the exact rejected source trial, not at its obsolete source seed.
    // This is an internal extra SQP step, never a second published control.
    bool source_trial_relinearized_correction{false};
    // Additional elastic terminal speed-operating inequality from the
    // same source rollout and loaded raceline. No alternative controller.
    double terminal_speed_viability_weight{};
    double terminal_speed_allowance_mps{.25};
    int source_qp_max_iterations{30};
    bool use_source_compact_qp_linearization{true};
    // Offline benchmark instrumentation only; no numerical decisions depend
    // on this flag. Live cycles leave it disabled to avoid trajectory copies.
    bool record_primal_trace{false};
    // Offline mathematical comparisons; not enabled by ROS configuration.
    bool source_slip_epigraph{false};
    bool exact_lagrangian_hessian{false};

};

struct OfflineIteration {
    int iteration{};
    bool accepted{};
    bool normal_step_computed{};
    bool normal_only_step{};
    double normal_primal_residual{};
    double normal_dual_residual{};
    bool normal_step_accepted{};
    bool original_duals_trustworthy{};
    double normal_linearized_violation{};
    double linearized_physical_violation_inf{};
    bool linearization_failed{};
    bool qp_failed{};
    bool qp_attempted{};
    bool full_step_rejected{};
    bool full_step_evaluated{};
    bool full_step_accepted{};
    bool nonlinear_evaluation_failed{};
    bool second_order_correction_attempted{};
    bool second_order_correction_accepted{};
    double safeguarded_step_length{1.};
    int safeguarded_evaluations{};
    bool lagrangian_hessian_valid{};
    int hessian_source_directional_evaluations{};
    double hessian_quadratic_consistency_relative{};
    double lagrangian_hessian_wall_ms{};
    double rejected_gradient_bfgs_wall_ms{};
    bool rejected_gradient_bfgs_updated{};
    int rejected_gradient_source_branch_changes{};
    double rejected_gradient_bfgs_damping{};
    double lagrangian_hessian_min_eigenvalue{};
    double second_order_correction_cost{};
    double second_order_correction_clearance{};
    int qp_iterations{};
    bool qp_converged{};
    bool invalid_qp_duals_discarded{};
    // All physical quantities refer to this iteration's exact source primal.
    double physical_violation_before{};
    double physical_violation_trial{};
    double physical_violation_after{};
    double maximum_hard_violation_before{};
    double maximum_hard_violation_after{};
    double accepted_step_inf_scaled{};
    int exact_source_evaluations{};
    double qp_complementarity_residual{};
    double qp_hessian_min_eigenvalue{};
    double qp_hessian_max_eigenvalue{};
    double qp_dual_residual{};
    double qp_primal_residual{};
    double qp_tolerance_used{};
    bool lagrangian_bfgs_updated{};
    double lagrangian_bfgs_damping{};
    double source_nlp_stationarity_inf{};
    double source_nlp_inequality_residual_inf{};
    double source_nlp_complementarity_inf{};
    double source_nlp_dual_feasibility_inf{};
    double qp_wall_ms{};
    double qp_regularization_used{};
    double model_predicted_hard_violation{};
    double model_actual_over_predicted_reduction{};
    double source_peak_slip_cost_before{};
    double source_peak_slip_cost_full_step{};
    int newly_activated_peak_slip_wheels{};
    double model_maximum_hard_constraint_error{};
    bool source_constraint_path_model_valid{};
    int source_constraint_path_first_invalid_stage{-1};
    double physical_violation_directional_first{};
    double physical_violation_directional_second{};
    double one_sided_constraint_direction_disagreement_inf{};
    double model_maximum_quadratic_footprint_error_m{};
    int calibrated_source_secants{};
    double secant_curvature_added{};
    bool reused_exact_source_linearization{};
    bool source_constraint_layout_reset{};
    // A source/contact/actuator/reference/peak branch changed across an
    // accepted nonlinear step. BFGS secants cannot cross these regimes.
    bool accepted_source_branch_curvature_reset{};
    bool accepted_source_discontinuous_secant_skipped{};
    int accepted_source_physical_branch_changes{};
    int accepted_source_peak_branch_changes{};
    int accepted_source_reference_branch_changes{};
    double linearization_wall_ms{};
    double source_rollout_wall_ms{};
    double iteration_wall_ms{};
    double cost_before{};
    double cost_trial{};
    double cost_after{};
    double violation_before{};
    double violation_trial{};
    double violation_after{};
    double predicted_cost_change{};
    double actual_cost_change{};
    double corridor_slack_after{};
};

struct OfflinePrimalTrace {
    int iteration{};
    bool trial_evaluated{};
    bool accepted{};
    Decision warm_start{};
    // QP dual estimate at warm_start, in original physical row order/units.
    // Excludes dynamics placeholders and preferred-reserve elastic rows.
    std::vector<double> physical_multipliers;
    Decision trial{};
    bool normal_evaluated{};
    bool normal_accepted{};
    Decision normal{};
    bool correction_evaluated{};
    bool correction_accepted{};
    Decision correction{};
    std::vector<Decision> safeguarded_trials;
    std::vector<double> safeguarded_step_lengths;
    std::vector<bool> safeguarded_valid;
    bool safeguarded_accepted{};
};

struct OfflineResult {
    MpcSourceSolverStatus_t solver_status{MPC_SOURCE_INVALID_MODEL};
    bool linearization_failed{};
    bool qp_failed{};
    bool numerical_failure{};
    const char *termination_reason{"iteration_limit"};
    bool iteration_admission_stopped{};
    bool initial_source_feasible{};
    bool final_source_feasible{};
    bool deadline_exhausted{};
    bool full_step_sqp_stalled{};
    bool source_nlp_kkt_converged{};
    int accepted_iterations{};
    int source_constraint_layout_resets{};
    int accepted_source_branch_curvature_resets{};
    int accepted_source_discontinuous_secant_skips{};
    int source_trial_evaluations{};
    int exact_source_evaluations{};
    double initial_source_cost{};
    double final_source_cost{};
    double initial_exact_constraint_violation{};
    double final_exact_constraint_violation{};
    double initial_minimum_corridor_slack{};
    double final_minimum_corridor_slack{};
    double final_minimum_rate_margin{};
    double initial_terminal_speed_margin_mps{};
    double final_terminal_speed_margin_mps{};
    double initial_terminal_speed_violation{};
    double final_terminal_speed_violation{};
    Decision optimized{};
    std::vector<double> final_physical_multipliers;
    std::vector<OfflineIteration> history;
    std::vector<OfflinePrimalTrace> primal_trace;
};

bool optimize_offline_exact_source(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const MpcModelControl_t seed[kStages],
    const OfflineOptions &options,OfflineResult *result);
} // namespace sdu_apex::source_nlp

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
    // Desired ELASTIC clearance reserve. The exact physical wall is 0.
    // The original-source collision boundary cannot be softened.
    double minimum_source_clearance_m{0.03};
    double allowable_corridor_violation_m{5.e-7};
    double allowable_command_rate_violation{1.e-7};
    double minimum_source_cost_reduction{};
    double minimum_violation_reduction{};
    double source_qp_curvature_regularization{1000.};
    double source_qp_soft_physical_weight{1.e6};
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
    bool elastic_sqp_globalization{false};
    bool safeguarded_merit_globalization{false};
    bool finite_difference_lagrangian_hessian{false};
    bool absolute_lagrangian_curvature{false};
};

struct OfflineIteration {
    int iteration{};
    bool accepted{};
    bool full_step_rejected{};
    bool nonlinear_evaluation_failed{};
    bool second_order_correction_attempted{};
    bool second_order_correction_accepted{};
    double safeguarded_step_length{1.};
    int safeguarded_evaluations{};
    double lagrangian_hessian_wall_ms{};
    double lagrangian_hessian_min_eigenvalue{};
    double second_order_correction_cost{};
    double second_order_correction_clearance{};
    int qp_iterations{};
    bool qp_converged{};
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
    int calibrated_source_secants{};
    double secant_curvature_added{};
    bool reused_exact_source_linearization{};
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
    Decision trial{};
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
    bool iteration_admission_stopped{};
    bool initial_source_feasible{};
    bool final_source_feasible{};
    bool deadline_exhausted{};
    bool full_step_sqp_stalled{};
    bool source_nlp_kkt_converged{};
    int accepted_iterations{};
    int source_trial_evaluations{};
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

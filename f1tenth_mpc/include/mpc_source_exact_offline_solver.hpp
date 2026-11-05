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
    double minimum_source_clearance_m{0.03};
    double allowable_corridor_violation_m{5.e-7};
    double allowable_command_rate_violation{1.e-7};
    double minimum_source_cost_reduction{1.e-5};
    double minimum_violation_reduction{1.e-10};
    double source_qp_curvature_regularization{1000.};
    double source_qp_soft_physical_weight{1.e6};
    int source_qp_max_iterations{30};
    bool use_source_compact_qp_linearization{true};
};

struct OfflineIteration {
    int iteration{};
    bool accepted{};
    bool full_step_rejected{};
    bool nonlinear_evaluation_failed{};
    int qp_iterations{};
    bool qp_converged{};
    double qp_dual_residual{};
    double qp_primal_residual{};
    double source_nlp_stationarity_inf{};
    double source_nlp_inequality_residual_inf{};
    double source_nlp_complementarity_inf{};
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

struct OfflineResult {
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
    Decision optimized{};
    std::vector<OfflineIteration> history;
};

bool optimize_offline_exact_source(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const MpcModelControl_t seed[kStages],
    const OfflineOptions &options,OfflineResult *result);
} // namespace sdu_apex::source_nlp

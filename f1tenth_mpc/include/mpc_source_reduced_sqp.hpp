#pragma once
/* Native 60-control exact-source SQP, accuracy-first implementation.
 * Full source dynamics (including hidden wheel/suspension/actuator state)
 * are eliminated by exact nonlinear single shooting at a frozen command
 * epoch. Only strictly source-validated full horizons are publishable. */
#include "mpc_rti.h"

namespace sdu_apex::source_sqp {
struct DerivativeSnapshot {
    double objective{};
    double wall[60]{};
    double rate[120]{};
    double gradient[60]{};
    double wall_jacobian[60][60]{};
    double rate_jacobian[120][60]{};
    // Offline-only snapshot of the ACTUAL first nonlinear SQP QP.
    // Unlike a guessed direction, this records the QP direction with
    // its original frozen objective/affine physical constraints.
    double qp_direction[60]{};
    double qp_hessian[60][60]{};
    double qp_wall_duals[60]{};
    bool qp_wall_duals_valid{};
    bool qp_wall_duals_from_elastic{};
    bool qp_direction_valid{};
    bool qp_was_phase1{};
};
struct Options {
    int max_iterations{8};
    // Qualified against 60 normal exact-source frames: 40 and 90 yield
    // identical accepted SQP solutions; 25 loses one QP certificate.
    int qp_max_iterations{40};

    double derivative_step{.002};
    double absolute_deadline_us{0.};
    bool second_order_correction{true};
    // Experimental, must pass exact-source parity before live promotion.
    // Differentiate command k by simulating only stages k..N from the
    // *same exact baseline physical state*, including hidden source states.
    bool reuse_unperturbed_fd_prefix{false};
    // Experimental full 26-source-channel, 20-lane forward derivative.
    // Never use this as production until cost AND wall/rate gradient parity.
    bool use_full_source_forward_sensitivities{false};
    // Experimental full-source residual Gauss-Newton Hessian. The exact
    // objective gradient and full nonlinear acceptance are unchanged.
    bool use_gauss_newton_hessian{false};
    // Experimental parallel stage-local full-source Jacobians (26+2),
    // causally chained to all 60 controls. Never approximate the plant.
    bool use_stage_local_jacobians{false};
    // Experiment: in strictly positive-drive source stages the eight
    // low-speed sticky-tire time memories do not influence any tire force
    // or geometric objective. All 26 primal states are STILL propagated.
    // Omit only their identically inactive command-sensitivity columns,
    // leaving 18 active-state + 2 input seeds = one 20-lane source call.
    bool compress_positive_drive_passive_timer_sensitivities{false};
    // The infeasible Phase-I QP only uses J_corridor and J_rate, not
    // source racing-cost gradients or source racing-cost curvature.
    // Score the same physical footprint directly when optimizing its
    // feasibility. Source model, map, terminal and strict acceptance
    // remain unchanged. Experimental until parity qualification.
    bool use_phase1_corridor_only_scoring{false};
    // Offline opt-in inexact hard Phase-I QP termination. The physical
    // affine primal constraints and complementarity remain rigorously
    // checked, and EVERY returned candidate is independently validated
    // against the complete nonlinear source. Does not loosen source
    // certificate or actuation authority.
    double phase1_inexact_relative_dual{0.};
    // Offline-only second-source-branch SQP experiment. Retain TWO
    // independent affine corridor constraints: one from the exact source
    // incumbent and one from the exact source trajectory at the first
    // QP's proposed step. No source physics or vehicle bounds change.
    // Disabled until source-feasibility parity and 22ms timing qualify.
    bool use_two_source_branch_qp{false};
    // Experimental algebraic presolve. A corridor row is removed ONLY
    // if maximizing its affine LHS over all physically legal command
    // boxes still cannot violate that row (no input-radius restrictions).
    bool use_two_branch_box_presolve{false};
    // Reduce *control decision* stages only (20 or 30); always propagate
    // and strictly validate the original 30-stage physical horizon.
    // Stages [decision_stages..29] retain the physical warm-start commands.
    int control_decision_stages{30};
    bool source_exact_early_stop{false};
    double min_accepted_relative_cost_improvement{.02};
    double min_early_stop_clearance_m{.002};


    // Optional experimental 3-stage EXACT-SOURCE tail under last-command
    // continuation. A sufficient check for this particular tail policy,
    // not a controlled-invariant terminal feasibility proof. Default 0.
    int terminal_lookahead_steps{0}; // supported: 0 or 3
    DerivativeSnapshot *first_derivative_snapshot{};
};
struct Statistics {
    int iterations{};
    int accepted_iterations{};
    int first_feasible_iteration{};
    bool source_exact_early_stop_selected{};
    int source_evaluations{};
    int source_finite_difference_columns{};
    int two_branch_source_probes{};
    int two_branch_qp_solves{};
    int two_branch_box_redundant_wall_rows{};
    int two_branch_relinearizations{};
    int two_branch_selected_directions{};
    double two_branch_initial_max_wall_defect_m{};
    double two_branch_selected_wall_v{};

    int source_forward_sensitivity_intervals{};
    int source_tangent_objective_evaluations{};
    double full_forward_dual_plant_us{};
    double full_forward_tangent_scoring_us{};

    int control_decision_stages{30};

    int qp_solves{};
    int qp_iterations{};

    int line_search_trials{};
    int accepted_full_source_horizons{};
    int near_feasible_source_projection_trials{};
    int near_feasible_source_projection_accepts{};
    int initial_source_failure_stage{-1};
    int last_qp_status{};
    bool feasible{};
    bool deadline_exhausted{};
    double initial_cost{};
    double final_cost{};
    double initial_wall_violation_m2{};
    double final_wall_violation_m2{};
    double initial_rate_violation_m2{};
    double final_rate_violation_m2{};
    bool initial_strict_source_valid{};
    bool final_strict_source_valid{};
    bool final_source_certificate_reused{};
    int final_source_failure_stage{-1};
    double initial_min_clearance_m{};
    double final_min_clearance_m{};
    int terminal_lookahead_steps{};
    double initial_tail_violation_m2{};
    double final_tail_violation_m2{};
    double initial_tail_minimum_clearance_m{};
    double final_tail_minimum_clearance_m{};
    double max_affine_violation{};
    double exact_source_evaluation_us{};
    double derivative_us{};
    double qp_us{};
    double total_us{};
    double first_cost_prediction_error{};
    bool full_step_cost_model_valid[12]{};
    double full_step_cost_prediction_error[12]{};
    double full_step_cost_predicted_change[12]{};
    double full_step_cost_actual_change[12]{};
    bool full_step_evaluated[12]{};
    double full_step_max_wall_defect_m[12]{};
    double full_step_max_rate_defect[12]{};
    double full_step_local_wall_violation_m2[12]{};
    double full_step_source_wall_violation_m2[12]{};
    bool accepted_cost_model_valid[12]{};
    double accepted_cost_predicted_change[12]{};
    double accepted_cost_actual_change[12]{};
    double accepted_cost_prediction_error[12]{};
    double accepted_cost[12]{};
    double accepted_wall_violation_m2[12]{};
    double accepted_alpha[12]{};
    double phase1_slack_sum[12]{};
    double first_qp_direction_norm[12]{};
    double initial_stage_min_clearance_m[PREDICTION_HORIZON+1]{};
    double final_stage_min_clearance_m[PREDICTION_HORIZON+1]{};
};
// Returns true for a valid numerical solve, even when Phase I has no fully
// source-feasible candidate. Then certified->valid is false and publication
// is forbidden. Selected controls expose the diagnostic intermediate only.
bool optimize(const MpcRtiState_t *initial, double progress,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, float dt,
    const MpcRtiCycleConfiguration_t *configuration,
    const MpcModelControl_t initial_controls[PREDICTION_HORIZON],
    const Options &options, Statistics *statistics,
    MpcModelControl_t selected_controls[PREDICTION_HORIZON],
    MpcRtiNominal_t *certified);
} // namespace sdu_apex::source_sqp

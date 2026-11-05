#ifndef MPC_RTI_H
#define MPC_RTI_H

#include "mpc_linearization.h"
#include "mpc_reference.h"

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    MPC_RTI_NX = MPC_OPTIMIZER_NX,
    MPC_RTI_NU = MPC_MODEL_NU,
    MPC_RTI_MAX_JACOBIAN_THREADS = 8,
    MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT = 7
};

typedef struct
{
    int stage;
    int valid;
    MpcModelOutput_t output;
} MpcRtiDiagnosticStageOutput_t;

typedef struct
{
    MpcModelState_t plant;
    unsigned int incoming_branch_flags;
    float incoming_max_positive_wheel_slip;
    double incoming_wheel_slip_excess_mps[4];
    float incoming_body_forward_delta_m;
    float incoming_body_lateral_delta_m;
    float incoming_body_yaw_delta_rad;
    int incoming_transition_valid;
    /* Exact loaded-path frame differential for this incoming physical step. */
    float incoming_tracking_jacobian[3][6];
    int incoming_tracking_jacobian_valid;
} MpcRtiState_t;

typedef struct
{
    float e_y;
    float e_psi;
    float u;
    float v;
    float r;
    float steering_command;
    float path_curvature;
    float left_bound;
    float right_bound;
    int source_witness_valid;
    float source_steering_target;
    float source_throttle;
} MpcRtiReference_t;

/* Tracking targets and nominal speed-weight branches held fixed for one SQP
 * objective. Physical rollout and wall bounds still use candidate progress. */
typedef struct
{
    MpcRtiReference_t references[PREDICTION_HORIZON + 1];
    float speed_weights[PREDICTION_HORIZON + 1];
    MpcRtiReference_t terminal_targets[3];
    float terminal_speed_weights[3];
    float terminal_prediction_dt;
    int terminal_targets_valid;
    int horizon;
    int valid;
    /* Frozen controlled terminal policy shared by R1, R2 and every exact
     * nonlinear acceptance. Expressed as delta from the horizon's previous
     * absolute command, so source sensitivities include actuator coupling.
     * Appended for backwards-compatible versioned .mctx migration. */
    int successor_recourse_valid;
    int successor_recourse_horizon;
    float successor_recourse_dt;
    float successor_recourse_weight;
    float successor_recourse_reserve_m;
    MpcModelControl_t successor_recourse_deltas[8];
} MpcRtiObjectiveSchedule_t;

typedef struct
{
    float weight_e_y;
    float weight_e_psi;
    /* Optional loss dead-zones; zero exactly preserves the original objective.
     * Hard wall/footprint constraints are NOT relaxed by these tolerances. */
    float tracking_lateral_tolerance_m;
    float tracking_heading_tolerance_rad;
    float weight_u;
    /* Used only while the nominal state is above the raceline speed. */
    float weight_u_overspeed;
    /* Scales overspeed cost up with |reference curvature|. A value of zero
     * preserves the uniform overspeed weight; units are metres. */
    float overspeed_curvature_gain_m;
    /* Scales underspeed cost down with |reference curvature|. A value of zero
     * preserves the uniform underspeed weight; units are metres. */
    float underspeed_curvature_gain_m;
    /* Soft cost on per-wheel surface-speed mismatch beyond Unity's forward
     * tire-curve extremum; not a hard slip feasibility limit. */
    float weight_longitudinal_slip;
    /* Scales the source wheel-slip cost up with |reference curvature|.
     * Zero preserves the uniform cost; units are metres. */
    float longitudinal_slip_curvature_gain_m;
    float weight_v;
    float weight_r;
    float weight_steering_command;
    /* Geometric steering target from path curvature, not IPOPT steering
     * actuator output; applies equally to source-derived geometry. */
    float weight_steering_target;
    float weight_throttle_command;
    /* Steering changes relative to the loaded rack/feedforward increment;
     * throttle changes remain zero-centered. Anchor the previous input to
     * the command actually issued, not an unexecuted optimizer proposal. */
    float weight_steering_slew;
    /* Convex pseudo-Huber total variation of steering CORRECTION changes.
     * QP uses a locally tangent quadratic majorizer, exact merit uses the
     * same smooth cost. Allows large curvature-demanded changes without
     * clipping SQP inputs to a nominal trust box. Zero restores legacy cost. */
    float weight_steering_correction_variation;
    float steering_correction_variation_epsilon_rad;
    float weight_throttle_slew;
    /* The normal Riccati QP may optimize physical command INCREMENTS
     * instead of unrestricted absolute steering/throttle commands.
     * These limits apply to successive ACTUAL
     * command targets across the horizon, independently of nominal/iterate.
     * Zero keeps the original absolute-input QP implementation. */
    int command_increment_qp_enabled;
    float max_steering_command_step_rad;
    float max_throttle_command_step;
    /* Optional cold-nominal drive seed when launching below 0.5 m/s toward
     * a materially faster reference. This initializes the linearization only;
     * the optimizer remains free to reduce or reject the command. */
    float startup_nominal_throttle_seed;
    float terminal_multiplier;
    float max_speed_mps;
    float active_speed_ceiling_mps;
    float max_steering_rad;
    float corridor_margin_m;
    /* Optional first predicted sample margin; default equals corridor_margin_m. */
    float first_prediction_corridor_margin_m;
    float corridor_preview_halfwidth_m;
    float nonlinear_corridor_tolerance_m;
    /* Independent horizon-stage Jacobians; zero selects the serial fallback. */
    int jacobian_threads;
    /* Soft geometric reference reserve beyond the HARD wall-clearance margin.
     * Moves the lateral target away from a tight wall when space permits;
     * does not tighten/relax constraints or change the loaded path/speeds. */
    float reference_clearance_reserve_m;
    /* Opt-in physical-footprint-aware frozen lateral target bias. At a
     * nominal stage whose actual body clearance is below this reserve above
     * the existing hard corridor, move that stage's lateral cost target away
     * from the nearby wall. This is an objective target, not relaxed geometry. */
    float near_wall_tracking_reserve_m;
    /* Experimental authoritative swept-world-body check using loaded map,
     * not an extra polygon QP constraint. Fail closed when map unavailable.
     * Callback inputs are two successive source-plant derived rear-axle poses
     * in map frame. Return 1 on physical sweep pass, 0 on rejection.
     * NOTE: interpolation between source step endpoints is checked, not
     * continuous Unity/PhysX substep motion. Supply uncertainty margin. */
    int (*world_sweep_check)(void *ctx,
        double x0,double y0,double yaw0,double x1,double y1,double yaw1,
        double margin_m,float *minimum_slack_m);
    void *world_sweep_context;
    /* When true, preserve original conservative Frenet validation on all
     * admitted stages and only ask map authority to rescue a false reject.
     * Never skip the map on a stage that Frenet considers invalid. */
    int world_map_rejection_only;
    /* Experimental full-footprint reserve cost, consistent in QP and exact
     * source merit. Append at the former ABI padding at the END of this
     * struct so all earlier replay model offsets remain unchanged.
     * Older v1/v2 .mctx reader initializes it to zero explicitly. */
    float weight_source_corridor_reserve;
    /* Opt-in Unity-source lateral tire saturation merit. Applies to exact
     * source WheelHit-equivalent instantaneous front contact slip (radians)
     * and the identical GN approximation in normal QP, not bicycle slip. */
    float weight_source_lateral_slip;
    /* Source-exact steering-actuator decision coordinates: optimize the
     * reachable physical rack increment instead of an arbitrary target that
     * the Unity rate limiter may not attain in this 25ms plant stage. QP
     * target = physical_steer + rack_increment; the command sent to Unity
     * remains an ordinary absolute steering target. Requires the existing
     * command-increment QP for causal input/box constraints. Experimental
     * until strict full races pass, defaults off. */
    int source_reachable_steering_qp;
} MpcRtiConfiguration_t;

typedef enum
{
    MPC_RTI_NOMINAL_BUILD_OK = 0,
    MPC_RTI_NOMINAL_BUILD_INVALID_INPUT,
    MPC_RTI_NOMINAL_BUILD_SPEED_CEILING,
    MPC_RTI_NOMINAL_BUILD_SPEED_SEED,
    MPC_RTI_NOMINAL_BUILD_SEED_REFERENCE,
    MPC_RTI_NOMINAL_BUILD_SEED_STEP,
    MPC_RTI_NOMINAL_BUILD_PASS_A_REFERENCE,
    MPC_RTI_NOMINAL_BUILD_PASS_A_ROLLOUT,
    MPC_RTI_NOMINAL_BUILD_PASS_B_REFERENCE,
    MPC_RTI_NOMINAL_BUILD_FINAL_ROLLOUT,
    MPC_RTI_NOMINAL_BUILD_COMMAND_ENVELOPE
} MpcRtiNominalBuildFailureStage_t;

typedef enum
{
    MPC_RTI_COMMAND_ENVELOPE_OK = 0,
    MPC_RTI_COMMAND_ENVELOPE_NONFINITE_STATE,
    MPC_RTI_COMMAND_ENVELOPE_SPEED_BELOW_ZERO,
    MPC_RTI_COMMAND_ENVELOPE_SPEED_ABOVE_MAX,
    MPC_RTI_COMMAND_ENVELOPE_STEERING_QUEUE,
    MPC_RTI_COMMAND_ENVELOPE_ACTUAL_STEERING,
    MPC_RTI_COMMAND_ENVELOPE_THROTTLE_QUEUE
} MpcRtiCommandEnvelopeFailure_t;

typedef struct
{
    MpcRtiState_t states[PREDICTION_HORIZON + 1];
    MpcModelControl_t controls[PREDICTION_HORIZON];
    double progress[PREDICTION_HORIZON + 1];
    float speed_reference[PREDICTION_HORIZON + 1];
    int horizon;
    int speed_reference_valid;
    int valid;
    /* Filled by exact command-sequence validation, not by nominal building. */
    double validation_objective;
    float validation_minimum_corridor_slack_m;
    /* Per-cycle benchmark instrumentation; microseconds from CLOCK_MONOTONIC. */
    double reference_seed_us;
    double cold_start_queue_seed_us;
    double nominal_rollout_a_us;
    double progress_reference_resampling_us;
    MpcRtiNominalBuildFailureStage_t build_failure_stage;
    MpcRtiCommandEnvelopeFailure_t build_failure_constraint;
    int build_failure_index;
} MpcRtiNominal_t;

typedef enum
{
    MPC_RTI_ROLLOUT_OK = 0,
    MPC_RTI_ROLLOUT_INVALID_INPUT,
    MPC_RTI_ROLLOUT_INVALID_MODEL,
    MPC_RTI_ROLLOUT_COMMAND_LIMIT,
    MPC_RTI_ROLLOUT_STATE_LIMIT,
    MPC_RTI_ROLLOUT_CORRIDOR,
    MPC_RTI_ROLLOUT_TERMINAL_VIABILITY
} MpcRtiRolloutStatus_t;

/* Details of the first state that made an exact nonlinear candidate
 * unacceptable. This is diagnostic data only: it does not widen the
 * corridor or change action selection. */
typedef struct
{
    int valid;
    int stage;
    MpcRtiState_t state;
    MpcRtiReference_t reference;
    double progress;
    float margin_m;
    float lower_bound_m;
    float upper_bound_m;
} MpcRtiRolloutFailure_t;

typedef struct
{
    int sample_count;
    double progress_error_m[PREDICTION_HORIZON + 1];
    float curvature_error_per_m[PREDICTION_HORIZON + 1];
    float left_bound_error_m[PREDICTION_HORIZON + 1];
    float right_bound_error_m[PREDICTION_HORIZON + 1];
    /* Exact nonlinear footprint-clearance margins from the same rollout
     * check. Stage zero is measured input and is not corridor-constrained. */
    float lower_corridor_slack_m[PREDICTION_HORIZON + 1];
    float upper_corridor_slack_m[PREDICTION_HORIZON + 1];
    /* When obtained from one complete exact nonlinear rollout, the world
     * reference is already sampled at each physical stage. Reuse it only
     * for an UNCHANGED trajectory, where all progress and speed references
     * match exactly; otherwise resample during objective evaluation. */
    MpcRtiReference_t sampled_reference[PREDICTION_HORIZON + 1];
    double sampled_progress[PREDICTION_HORIZON + 1];
    int sampled_references_valid;
} MpcRtiCandidatePathDelta_t;

/* Diagnostic form of the nonlinear corridor test, using the same vehicle
 * footprint and first-prediction margin as production validation. */
/* Original-source physical footprint corridor differential, with the
 * exact same heading/corner selection as mpc_rti_compute_corridor_slack.
 * Local heading derivatives are returned only away from a max/abs tie.
 * Progress-dependent left/right bounds may have a one-sided kink at an
 * original raceline interpolation knot or preview minimum; therefore
 * their forward and backward differentials remain SEPARATE.
 *
 * Not a smooth approximation across branches, not a safety certificate.
 * Returns 0 when the heading active corner is ambiguous: the caller
 * must use source-matched finite differences or multiple active cuts.
 */
typedef struct {
    double margin_lower_m;
    double margin_upper_m;
    double lower_d_ey;
    double upper_d_ey;
    double lower_d_epsi;
    double upper_d_epsi;
    double lower_d_progress_forward;
    double upper_d_progress_forward;
    double lower_d_progress_backward;
    double upper_d_progress_backward;
    int source_heading_active_branch;
    int progress_reference_kink; /* directional left/right branch differs */
    double progress_slope_convergence_error; /* diagnostic; source units */
    double progress_derivative_step_m;
} MpcRtiCorridorDifferential_t;
int mpc_rti_corridor_differential_at_progress(
    const MpcModelState_t *state,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,double progress_m,
    const MpcRtiConfiguration_t *configuration,int prediction_index,
    MpcRtiCorridorDifferential_t *out);

/* Exact ORIGINAL source rear-axle rectangle as eight nonlinear corner
 * inequalities. Original source min-slacks equal min(left[0..3]) and
 * min(right[0..3]), including the same first-stage clearance and tolerance.
 * This removes the physical MAX over bumper corners from the formulation:
 * each corner is continuously differentiable in heading, INCLUDING e_psi=0.
 *
 * Source reference-bound slopes remain directional at original loaded-knot
 * and +/- preview-min switches. reference_derivatives_valid=0 means DO NOT
 * treat a selected reference as smooth; primal corner values are still exact.
 *
 * Array order for each side: (rear,-halfwidth),(rear,+halfwidth),
 * (front,-halfwidth),(front,+halfwidth).
 */
typedef struct {
    double left[4];  /* original upper/left wall constraints */
    double right[4]; /* original lower/right wall constraints */
    double left_d_psi[4];
    double right_d_psi[4];
    double left_d_progress_forward;
    double left_d_progress_backward;
    double right_d_progress_forward;
    double right_d_progress_backward;
    int reference_derivatives_valid;
    int reference_progress_kink;
} MpcRtiSourceCorridorCorners_t;
int mpc_rti_source_exact_corridor_corners(
    const MpcModelState_t *state,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,double progress_m,
    const MpcRtiConfiguration_t *configuration,int prediction_index,
    MpcRtiSourceCorridorCorners_t *out);

int mpc_rti_compute_corridor_slack(
    const MpcModelState_t *state,
    float left_bound_m,
    float right_bound_m,
    const MpcRtiConfiguration_t *configuration,
    int prediction_index,
    float *lower_slack_m,
    float *upper_slack_m);

// The sole source-exact nonlinear SQP cycle. These are all the
// runtime inputs; obsolete solver modes are no longer representable.
typedef struct
{
    MpcRtiConfiguration_t model;
    int64_t command_epoch_ns;
    double max_cycle_time_us;
    double absolute_deadline_monotonic_us;
    int source_exact_nlp_live;
    int source_exact_nlp_constrained_qp;
    int source_exact_nlp_compact_qp;
    int source_exact_nlp_full_step_sqp;
    float source_exact_nlp_qp_soft_weight;
    float source_exact_nlp_qp_regularization;
    int source_exact_nlp_iterations;
    float source_exact_nlp_extra_clearance_m;
    int capture_diagnostic_outputs;
    const MpcRtiObjectiveSchedule_t *objective_schedule;
} MpcRtiCycleConfiguration_t;

typedef struct
{
    int valid;
    int accepted;
    int source_exact_rollout_complete;
    int source_exact_rollout_status;
    int source_exact_failure_stage;
    int comparison_includes_feedback;
    int line_search_attempted;
    int line_search_trials;
    int line_search_accepted;
    int line_search_merit_triggered;
    float selected_alpha;
    int state_sample_count;
    int footprint_sample_count;
    int maximum_state_error_stage;
    int maximum_state_error_index;
    int maximum_footprint_error_stage;
    int maximum_source_footprint_violation_stage;
    double qp_predicted_cost_reduction;
    double unscaled_qp_predicted_cost_reduction;
    double source_exact_cost_reduction;
    double realized_source_exact_cost_reduction;
    double selected_qp_predicted_cost_reduction;
    double selected_source_exact_cost_reduction;
    double model_agreement_ratio;
    double realized_model_agreement_ratio;
    double selected_model_agreement_ratio;
    /* Retained only for binary compatibility with captured .mctx fixtures.
     * The abandoned post-QP adaptive trust radius is never applied. */
    float trust_region_scale_before;
    float trust_region_scale_after;
    double maximum_state_error;
    double maximum_footprint_constraint_error_m;
    double maximum_source_footprint_violation_m;
    float state_error_by_stage[PREDICTION_HORIZON + 1];
    int state_error_index_by_stage[PREDICTION_HORIZON + 1];
    float footprint_constraint_error_by_stage_m[PREDICTION_HORIZON + 1];
    float source_footprint_violation_by_stage_m[PREDICTION_HORIZON + 1];
} MpcRtiSqpStepMetrics_t;

typedef struct
{
    /* Last full-horizon nominal retained for progress seeding. This is not
     * the certified plan that may be replayed after a rejected solve. */
    MpcRtiNominal_t nominal;
    /* Sequence rooted at command_epoch_ns. Runtime advances by elapsed
     * command stages; same-epoch callbacks must not consume its first action. */
    MpcModelControl_t warm_start_controls[PREDICTION_HORIZON];
    int warm_start_control_count;
    int warm_start_controls_valid;
    int64_t command_epoch_ns;
    /* Reserved legacy field for preexisting same-build replay snapshots.
     * Not read or written by the production SQP. */
    float sqp_trust_region_scale;
    double solver_nominal_progress[PREDICTION_HORIZON + 1];
    int solver_nominal_progress_valid;
} MpcRtiMemory_t;

typedef enum
{
    MPC_RTI_CYCLE_ACCEPTED_OPTIMAL = 0,
    /* Solver hit its iteration cap, and this candidate passed the full
     * nonlinear plant/corridor validation. */
    MPC_RTI_CYCLE_ACCEPTED_FEASIBLE,
    /* The horizon failed validation, but its next 25 ms action was exact-
     * validated and published; the next cycle must solve/revalidate again. */
    MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX,
    /* The RTI solve failed, but the next action of the last accepted plan was
     * revalidated from the current state and published once. */
    MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN,
    MPC_RTI_CYCLE_REJECTED_INPUT,
    MPC_RTI_CYCLE_REJECTED_SOLVER,
    MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT,
    /* Startup-only retry: nominal/Jacobian work consumed the reserved solver
     * budget before ADMM began. No command is publishable yet. */
    MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET,
    /* A damped QP direction passed the same complete nonlinear checks. */
    MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH,
    /* The command missed its end-to-end deadline. No late output was sent;
     * the next cycle will revalidate the next action of the prior plan. */
    MPC_RTI_CYCLE_DEFERRED_DEADLINE,
    /* Experimental: full-horizon source-certified deterministic rescue. */
    MPC_RTI_CYCLE_ACCEPTED_VIABILITY_RECOVERY
} MpcRtiCycleStatus_t;

typedef enum
{
    MPC_RTI_INPUT_FAILURE_NONE = 0,
    MPC_RTI_INPUT_FAILURE_CYCLE_ARGUMENTS,
    MPC_RTI_INPUT_FAILURE_PLANT_STATE,
    MPC_RTI_INPUT_FAILURE_NOMINAL_BUILD
} MpcRtiInputFailureReason_t;

typedef struct
{
    MpcRtiCycleStatus_t status;
    MpcRtiInputFailureReason_t input_failure_reason;
    MpcRtiNominalBuildFailureStage_t nominal_build_failure_stage;
    MpcRtiCommandEnvelopeFailure_t nominal_build_failure_constraint;
    int nominal_build_failure_index;
    MpcModelControl_t first_control;
    MpcModelControl_t nominal_first_control;
    MpcRtiCandidatePathDelta_t candidate_path_delta;
    float published_steering_command;
    float published_throttle;
    int solver_iterations;
    int solver_skipped_for_deadline;
    int solver_time_budget_exhausted;
    /* Controller discarded stale ADMM iterates after publishing an exactly
     * revalidated recovery action, preserving its nominal/control seed. */
    int admm_state_reset_for_recovery;
    /* Late unpublished solve was rolled back to the last published source
     * nominal and ADMM state, not left as an unissued command. */
    int solver_memory_restored_after_deadline;
    int feasibility_restoration_attempted;
    int feasibility_restoration_accepted;
    int feasibility_restoration_iterations;
    float feasibility_restoration_primal_residual;
    double feasibility_restoration_us;
    int sqp_iterations_attempted;
    int sqp_iterations_completed;
    int sqp_iterations_accepted;
    int native_source_sqp_active;
    int native_source_sqp_source_evaluations;
    int native_source_sqp_fd_columns;
    int native_source_sqp_qp_solves;
    int native_source_sqp_qp_iterations;
    int native_source_sqp_first_feasible_iteration;
    int native_source_sqp_initial_failure_stage;
    double native_source_sqp_initial_wall_violation_m2;
    double native_source_sqp_final_wall_violation_m2;
    double native_source_sqp_qp_us;
    double native_source_sqp_source_evaluation_us;
    double native_source_sqp_initial_rate_violation_m2;
    double native_source_sqp_final_rate_violation_m2;
    int native_source_sqp_initial_strict_source_valid;
    int native_source_sqp_final_strict_source_valid;
    int native_source_sqp_last_qp_status;
    int native_source_sqp_full_forward_intervals;
    int native_source_sqp_tangent_score_evaluations;
    double native_source_sqp_derivative_us;
    double native_source_sqp_dual_plant_worker_us;
    double native_source_sqp_tangent_scoring_worker_us;
    /* Diagnostic only: the first accepted cost-optimizing SQP step, not a
     * Phase-I physical-feasibility QP. Compare the *applied* alpha step's
     * quadratic racing-cost prediction with its exact source-model cost. */
    int native_source_sqp_cost_prediction_valid;
    double native_source_sqp_predicted_cost_change;
    double native_source_sqp_actual_cost_change;
    double native_source_sqp_cost_prediction_error;
    /* QP affine footprint versus EXACT nonlinear source rollout at the
     * candidate full step, for each of the first two source SQP passes.
     * Diagnostic-only; no extra source integration is performed. This
     * measures whether local convexification can be trusted in corners. */
    int native_source_sqp_full_step_evaluated[2];
    int native_source_sqp_accepted_cost_model_valid[2];
    double native_source_sqp_accepted_cost_predicted_change[2];
    double native_source_sqp_accepted_cost_actual_change[2];
    double native_source_sqp_accepted_cost_prediction_error[2];
    int native_source_sqp_full_step_cost_model_valid[2];
    double native_source_sqp_full_step_cost_predicted_change[2];
    double native_source_sqp_full_step_cost_actual_change[2];
    double native_source_sqp_full_step_cost_prediction_error[2];
    double native_source_sqp_qp_predicted_wall_violation_m2[2];
    double native_source_sqp_source_actual_wall_violation_m2[2];
    double native_source_sqp_max_wall_model_defect_m[2];
    double native_source_sqp_selected_alpha[2];
    double native_source_sqp_phase1_slack_sum[2];
    int native_source_sqp_near_feasible_projection_trials;
    int native_source_sqp_near_feasible_projection_accepts;
    int source_exact_nlp_active;
    int source_exact_nlp_accepted_iterations;
    int source_exact_nlp_source_trial_evaluations;
    int source_exact_nlp_deadline_exhausted;
    int source_exact_nlp_initial_source_feasible;
    int source_exact_nlp_final_source_feasible;
    int source_exact_nlp_kkt_converged;
    int source_exact_nlp_full_step_stalled;
    double source_exact_nlp_initial_source_cost;
    double source_exact_nlp_final_source_cost;
    double source_exact_nlp_final_original_margin_m;
    double source_exact_nlp_final_original_violation;
    int source_nominal_repair_attempted;
    int source_nominal_repair_accepted;
    int source_nominal_repair_phase_one_iterations;
    double source_nominal_repair_us;
    int successor_recourse_active;
    double successor_recourse_initialize_us;
    double successor_recourse_qp_us;
    int experimental_source_nonlinear_selected;
    int source_nonlinear_exact_evaluations;
    int source_nonlinear_gradient_discrepancies;
    int source_nonlinear_timed_out;
    double source_nonlinear_wall_us;
    double source_nonlinear_objective_improvement;
    int source_nonlinear_terminal_probe_count;
    int source_nonlinear_terminal_rejected;
    /* Only compared when neither SQP pass has a valid complete horizon.
     * This selects a proposal for the controller's independent prefix gate;
     * it never certifies the rejected tail. 0=not compared, 1=first, 2=second. */
    int refinement_full_horizon_selected_pass; /* 0=none, 1=R1, 2=R2 */
    int refinement_terminal_probe_count;
    int refinement_terminal_selected_pass;
    double refinement_terminal_probe_us;
    double refinement_nominal_build_us;
    int refinement_prefix_selected_pass;
    double refinement_first_prefix_objective;
    double refinement_second_prefix_objective;
    double refinement_prefix_validation_us;
    double first_pass_objective;
    double second_pass_objective;
    double exact_objective_improvement;
    /* Infeasible-source-iterate R1->R2 filter: an accepted intermediate is
     * NEVER publishable; only the strict final horizon can be published. */
    int source_violation_filter_attempted;
    int source_violation_filter_selected;
    float source_violation_filter_alpha;
    double source_violation_initial_m2;
    double source_violation_selected_m2;
    MpcRtiSqpStepMetrics_t r1_step_metrics;
    MpcRtiSqpStepMetrics_t r2_step_metrics;
    int admm_status;
    float primal_residual;
    float dual_residual;
    float state_primal_residual;
    float state_dual_residual;
    float input_primal_residual;
    float input_dual_residual;
    int state_primal_stage;
    int state_primal_index;
    int state_dual_stage;
    int state_dual_index;
    int input_primal_stage;
    int input_primal_index;
    int input_dual_stage;
    int input_dual_index;
    float maximum_regularization;
    int regularization_count;
    int nonsmooth_jacobian_columns;
    int nonlinear_failure_stage;
    int nonlinear_failure_reason;
    MpcRtiRolloutFailure_t nonlinear_failure;
    int line_search_attempted;
    int line_search_trials;
    int line_search_accepted;
    int line_search_skipped_budget;
    int line_search_skipped_command_epoch;
    int line_search_full_step_failure_reason;
    int line_search_full_step_failure_stage;
    int line_search_last_failure_reason;
    int line_search_last_failure_stage;
    int line_search_merit_triggered;
    int line_search_nominal_exact_valid;
    double line_search_nominal_objective;
    float line_search_selected_alpha;
    double line_search_us;
    int diagnostic_output_count;
    MpcRtiDiagnosticStageOutput_t
        diagnostic_outputs[MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT];
    /* The solver's exact action sequence when it converged but nonlinear
     * rollout validation rejected that sequence. Diagnostic-only; never a
     * publishable command on a rejected cycle. */
    int candidate_control_count;
    MpcModelControl_t candidate_controls[PREDICTION_HORIZON];
    MpcModelControl_t nominal_controls[PREDICTION_HORIZON];
    int nominal_control_count;
    /* 0=cold seed, 1=shifted prior nominal, 2=issued command sequence. */
    int nominal_control_seed_source;
    /* -1=cold/unusable epoch; otherwise elapsed 25ms control stages. */
    int command_epoch_shift_stages;
    float rho_start;
    float rho_u_start;
    float rho_final;
    float rho_u_final;
    int rho_change_count;
    int factorization_count;
    uint64_t factorization_time_ns;
    uint64_t solver_cold_start_initialization_time_ns;
    uint64_t riccati_rhs_time_ns;
    uint64_t admm_projection_time_ns;
    uint64_t rho_update_time_ns;
    uint64_t solver_scaling_time_ns;
    uint64_t solver_output_unscale_time_ns;
    float max_candidate_progress_error_m;
    float max_candidate_curvature_error_per_m;
    float max_candidate_left_bound_error_m;
    float max_candidate_right_bound_error_m;
    float minimum_predicted_corridor_slack_m;
    float lateral_accel_proxy_mps2;
    int lateral_accel_proxy_stage;
    float lateral_accel_proxy_by_stage_mps2[PREDICTION_HORIZON + 1];
    int linearization_failure_stage;
    int linearization_failure_reason;
    int linearization_failure_index;
    int jacobian_stages_completed;
    int jacobian_threads_used;
    int jacobian_source_sensitivity_calls;
    int jacobian_finite_difference_columns;
    double jacobian_wall_us;
    double jacobian_thread_cpu_us;
    double jacobian_slowest_stage_us;
    double jacobian_worker_cpu_us[MPC_RTI_MAX_JACOBIAN_THREADS];
    double jacobian_worker_stage_wall_us[MPC_RTI_MAX_JACOBIAN_THREADS];
    int jacobian_worker_stage_count[MPC_RTI_MAX_JACOBIAN_THREADS];
    uint64_t jacobian_worker_os_tid[MPC_RTI_MAX_JACOBIAN_THREADS];
    double qp_build_us;
    double nominal_build_us;
    double reference_seed_us;
    double cold_start_queue_seed_us;
    double nominal_rollout_a_us;
    double progress_reference_resampling_us;
    double admm_us;
    /* Actual thread CPU spent in ADMM, excluding OS descheduling, per solve. */
    double admm_thread_cpu_us;
    double exact_rollout_us;
    double nonlinear_validation_us;
    double total_solve_us;
    double total_cycle_us;
    double nonlinear_objective;
    /* Common cycle objective for cross-pass and candidate/prior comparisons. */
    MpcRtiObjectiveSchedule_t comparison_objective;
} MpcRtiCycleResult_t;

int mpc_rti_build_objective_schedule(
    const MpcRtiNominal_t *nominal,
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    const MpcRtiConfiguration_t *configuration,
    MpcRtiObjectiveSchedule_t *schedule);

/* Runtime reference/bounds interpolation at one physical path-progress point.
 * Exposed for exact replay tooling so corridor audits use the same preview,
 * contour and clearance policy as the controller. */
int mpc_rti_reference_at_progress(
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, double progress,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiReference_t *reference);

/* Score an already propagated plan without rerunning the physical plant. */
int mpc_rti_score_candidate_plan(
    MpcRtiNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcRtiCycleConfiguration_t *configuration);

/* Replay/diagnostic scoring for a complete exact-source rollout whose states
 * may violate the hard footprint constraint. This reports the original
 * frozen objective and minimum clearance but never validates or authorizes
 * the candidate for publication. */
int mpc_rti_score_candidate_plan_for_diagnostics(
    const MpcRtiNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcRtiCycleConfiguration_t *configuration,
    double *objective, float *minimum_corridor_slack_m);

/* Shift (or cold-seed) controls and construct a deterministic two-pass
 * nonlinear nominal/reference horizon using exact model delta_s. */
int mpc_rti_build_nominal(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcRtiNominal_t *previous_nominal,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiNominal_t *nominal,
    MpcRtiReference_t references[PREDICTION_HORIZON + 1]);

/* As above, but shift the supplied command sequence instead of deriving
 * controls from the prior nominal. The sequence must include the most
 * recently issued command at index zero. */
int mpc_rti_build_nominal_with_warm_start(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcRtiNominal_t *previous_nominal,
    const MpcModelControl_t *warm_start_controls,
    int warm_start_control_count,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiNominal_t *nominal,
    MpcRtiReference_t references[PREDICTION_HORIZON + 1]);

/* Rebuild a nonlinear SQP nominal at the SAME physical command epoch.
 * Unlike next-cycle warm_start(), this DOES NOT shift the supplied actuator
 * commands forward one optimization stage. Exposing the distinction avoids
 * accidental one-stage phase errors in RTI relinearization and offline
 * derivative/curvature audits. */
/* The SINGLE authoritative source stage operator used by the original
 * nominal rollout and by the offline nonlinear multiple-shooting dynamics.
 * It advances the full source wheel/suspension/timer states, original 1ms
 * substeps, path projection, branch flags and within-interval peak tire slip.
 * No linear/QP/spline approximation, no candidate-state clamping.
 *
 * Current state and progress belong to the SAME actuator epoch. The
 * original command is applied directly for exactly one prediction interval.
 */
/* Source physical steering-target and throttle command-rate inequalities
 * from the SAME definition used by the native reduced source-SQP.
 * Nonnegative entries mean legal. These are physical/source-epoch
 * constraints, NOT optimization-iterate trust regions.
 * Output: [rack_limit - (steer_target-rack),
 *          rack_limit + (steer_target-rack),
 *          throttle_step - (throttle - preceding_command),
 *          throttle_step + (throttle - preceding_command)].
 */
int mpc_rti_source_command_rate_margins(
    const MpcModelState_t *start,const MpcModelControl_t *command,
    const MpcRtiConfiguration_t *configuration,double out[4]);

int mpc_rti_source_exact_stage(
    const MpcRtiState_t *current,double current_progress,
    const MpcModelControl_t *command,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,MpcRtiState_t *next,
    double *next_progress);

int mpc_rti_build_nominal_same_epoch(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcRtiNominal_t *previous_nominal,
    const MpcModelControl_t *same_epoch_controls,
    int same_epoch_control_count,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiNominal_t *nominal,
    MpcRtiReference_t references[PREDICTION_HORIZON + 1]);

/* Experimental exact-source nonlinear single-shooting adjoint. Builds the
 * same-epoch physical source trajectory and its local A/B derivatives,
 * then returns d(exact-nominal objective)/d(absolute steering,throttle)
 * using a full-horizon costate sweep. The local cost gradient is approximate
 * in changing tire/progress regimes: certify against source finite differences
 * before enabling a runtime policy. Does not solve or publish any command. */
/* Recover the exact same stage references as a freshly integrated
 * same-epoch nominal, without repeating the full PhysX/source rollout.
 * Call only for an independently source-integrated, validated nominal. */
int mpc_rti_references_from_exact_nominal(
    const MpcRtiNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count, double lap_length,
    const MpcRtiConfiguration_t *model,
    MpcRtiReference_t references[PREDICTION_HORIZON + 1]);

/* The adjoint of an ALREADY source-validated full horizon. The passed
 * physical stage states must match the same immutable control sequence.
 * Unlike the legacy wrapper below, it never reintegrates source dynamics. */
int mpc_rti_source_adjoint_gradient_from_exact_nominal(
    const MpcRtiNominal_t *validated_nominal,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count, double lap_length,
    float prediction_dt, const MpcRtiConfiguration_t *model,
    const MpcRtiObjectiveSchedule_t *frozen_objective,
    double gradient_out[PREDICTION_HORIZON][MPC_RTI_NU]);

int mpc_rti_source_adjoint_gradient(
    const MpcRtiState_t *initial,
    double progress,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    const MpcRtiConfiguration_t *model,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcRtiObjectiveSchedule_t *frozen_objective,
    double gradient_out[PREDICTION_HORIZON][MPC_RTI_NU]);

/* Experimental full-source iterative nonlinear MPC. This entrypoint is NOT
 * connected to the normal ROS controller mode: it only evaluates externally
 * supplied, same-command-epoch controls and returns a strictly source-valid
 * improved full horizon. Every trial integrates the source plant and checks
 * the ORIGINAL physical corridor. No QP-predicted path or unrelated merit
 * schedule may substitute for these physical trajectories.
 *
 * A frozen schedule MUST come from the current cycle. First derivatives use
 * the production source adjoint; a separately integrated directional
 * perturbation audits whether it predicts the true nonlinear merit.
 *
 * The iteration cap is deterministic; the caller must independently budget
 * its worst-case duration before ever enabling real-time control. */
typedef struct
{
    int max_iterations; /* 1..8 */
    int max_exact_trials_per_iteration; /* 1..16 */
    float initial_step_scale; /* positive dimensionless search parameter */
    float minimum_step_scale;
    float armijo_fraction; /* 0..0.5, requires true source merit reduction */
    float steering_direction_norm_rad; /* L2 norm of 30 steering changes */
    float throttle_direction_norm; /* L2 norm of 30 throttle changes */
    float directional_check_step_scale; /* relative to selected physical dir */
    double maximum_wall_time_us; /* 0 = offline unbounded; caller reserves publication overhead */
} MpcRtiSourceNonlinearOptions_t;

typedef struct
{
    int initial_valid;
    int final_valid;
    MpcRtiRolloutStatus_t initial_failure_reason;
    int initial_failure_stage;
    int iterations_attempted;
    int iterations_accepted;
    int exact_rollout_evaluations;
    int source_adjoint_evaluations;
    int source_derivative_consistency_checks;
    int source_derivative_discrepancies;
    int trial_corridor_rejections;
    int trial_model_rejections;
    int budget_exhausted;
    int feasibility_iterations_attempted;
    int feasibility_iterations_accepted;
    int feasible_seed_recovered;
    double initial_corridor_violation_m2;
    double final_corridor_violation_m2;
    double initial_objective;
    double final_objective;
    double minimum_corridor_clearance_m;
    double largest_control_step_rad;
    double largest_throttle_step;
    double max_directional_derivative_relative_error;
    double wall_time_ms;
    /* Exact-source objective after each ACCEPTED iteration; the original
     * seed is element zero. Not a QP-predicted merit or relaxed-track score. */
    double accepted_exact_objectives[9];
    double accepted_step_scales[8];
    double directional_relative_error_by_iteration[8];
    MpcModelControl_t controls[PREDICTION_HORIZON];
} MpcRtiSourceNonlinearResult_t;

/* Shared exact-source Phase-I seed repair. Returns 1 only after a full
 * strict source-plant rollout of the repaired controls. Used as an SQP
 * initializer, not a publishable fallback or separate racing optimizer. */
int mpc_rti_source_repair_infeasible_nominal(
    const MpcRtiState_t *initial, double progress,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, float prediction_dt,
    const MpcRtiCycleConfiguration_t *configuration,
    const MpcRtiObjectiveSchedule_t *frozen_objective,
    const MpcModelControl_t seed[PREDICTION_HORIZON],
    const MpcRtiSourceNonlinearOptions_t *options,
    MpcRtiSourceNonlinearResult_t *result,
    MpcRtiNominal_t *certified_nominal);

int mpc_rti_source_nonlinear_optimize(
    const MpcRtiState_t *initial,
    double progress,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    const MpcRtiCycleConfiguration_t *configuration,
    const MpcRtiObjectiveSchedule_t *frozen_objective,
    const MpcModelControl_t seed[PREDICTION_HORIZON],
    const MpcRtiSourceNonlinearOptions_t *options,
    MpcRtiSourceNonlinearResult_t *result,
    MpcRtiNominal_t *certified_nominal);

/* Recursively score a candidate at candidate progress, sampling its exact
 * nonlinear curvature and corridor instead of reusing the nominal schedule. */
MpcRtiRolloutStatus_t mpc_rti_rollout_candidate(
    const MpcRtiState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    /* Optional QP schedule used only to report path-schedule mismatch. */
    const MpcRtiReference_t nominal_references[PREDICTION_HORIZON + 1],
    const double nominal_progress[PREDICTION_HORIZON + 1],
    MpcRtiCandidatePathDelta_t *path_delta,
    int horizon,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiState_t states[PREDICTION_HORIZON + 1],
    double progress[PREDICTION_HORIZON + 1],
    int *failure_stage);

/* Validate a control sequence against the exact nonlinear plant, state and
 * full-footprint corridor checks. Used only to continue a remaining suffix
 * of a previously accepted plan after a later solve fails. */
/* Exact-source scorer with the same physical footprint planes exported
 * for source-SQP derivatives. Avoids re-evaluating reference interpolation
 * and the footprint against the same 30 already scored source states.
 * Margin rows 1..horizon populated; 0 is the caller's measured state. */
int mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
    const MpcRtiNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,const MpcRtiCycleConfiguration_t *configuration,
    double *objective,float *minimum_corridor_slack_m,
    float corridor_by_stage[PREDICTION_HORIZON+1][2]);

MpcRtiRolloutStatus_t mpc_rti_validate_candidate_plan(
    const MpcRtiState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    int horizon,
    float prediction_dt,
    const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiNominal_t *validated_plan,
    int *failure_stage);

/* Causal 75ms post-horizon viability PROBE, not an infinite-horizon safety
 * proof: two legal successors (source witness at terminal station or held
 * final command) are independently simulated from the exact terminal
 * physical state and must pass the same body/map/command validator.
 * The whole car source state and actuator history are preserved. */
int mpc_rti_terminal_extension_probe(
    const MpcRtiNominal_t *validated_plan,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count, double lap_length,
    float prediction_dt,
    const MpcRtiCycleConfiguration_t *configuration);

/* As above, while retaining selected source outputs that were already
 * computed by the exact rollout. This avoids stepping the plant again in the
 * ROS diagnostics publisher. */
MpcRtiRolloutStatus_t mpc_rti_validate_candidate_plan_with_diagnostics(
    const MpcRtiState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    int horizon,
    float prediction_dt,
    const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiNominal_t *validated_plan,
    MpcRtiDiagnosticStageOutput_t diagnostic_outputs[
        MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT],
    int *diagnostic_output_count,
    int *failure_stage);

void mpc_rti_memory_reset(MpcRtiMemory_t *memory);
int mpc_rti_memory_set_warm_start_controls(
    MpcRtiMemory_t *memory,
    const MpcModelControl_t *controls,
    int control_count);
int mpc_rti_memory_set_warm_start_controls_at_epoch(
    MpcRtiMemory_t *memory, const MpcModelControl_t *controls,
    int control_count, int64_t command_epoch_ns);
/* Nearest discrete source stage; -1 means missing/reversed/expired epoch.
 * Host request jitter does not change the source's fixed 25ms interval. */
int mpc_rti_command_epoch_shift(int64_t previous_ns, int64_t current_ns,
                              float prediction_dt);

/* One RTI cycle: build shifted nominal/QP, solve once, validate the exact
 * nonlinear rollout, and commit the candidate only when it is accepted. */
MpcRtiCycleStatus_t mpc_rti_solve_cycle(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiMemory_t *memory,
    MpcRtiCycleResult_t *result);

/* CLOCK_MONOTONIC timestamp in microseconds, shared by the ROS wrapper and
 * RTI core when constructing an absolute per-callback deadline. */
double mpc_rti_monotonic_time_us(void);

#ifdef __cplusplus
}
#endif

#endif

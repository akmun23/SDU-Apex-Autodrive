#ifndef MPC_RTI_H
#define MPC_RTI_H

#include "vehicle_model.h"
#include <stdint.h>
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
    float weight_throttle_slew;
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
    int capture_source_benchmark;
} MpcRtiCycleConfiguration_t;

enum { MPC_RTI_SOURCE_TRACE_CAPACITY = 8 };
typedef struct {
    int iteration;
    int trial_evaluated;
    int accepted;
    MpcModelControl_t warm_start[PREDICTION_HORIZON];
    MpcModelControl_t trial[PREDICTION_HORIZON];
    int correction_evaluated;
    int correction_accepted;
    MpcModelControl_t correction[PREDICTION_HORIZON];
    int normal_accepted;
    int filter_accepted;
    MpcModelControl_t accepted_controls[PREDICTION_HORIZON];
} MpcRtiSourcePrimalTrace_t;

/* Scalar diagnostics are always retained; full primal captures remain opt-in.
 * QP multipliers and NLP residuals refer to the iteration START controls. */
typedef struct {
    int accepted;
    int trial_evaluated;
    int rejected_trials;
    int radius_active;
    int qp_iterations;
    int qp_rounds;
    int newly_activated_slip_wheels;
    double radius;
    double linearization_ms;
    double qp_ms;
    double rollout_ms;
    double total_ms;
    double cost_before;
    double cost_trial;
    double cost_after;
    double predicted_reduction;
    double actual_over_predicted;
    double slip_cost_before;
    double slip_cost_trial;
} MpcRtiSourceIterationMetrics_t;

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
    /* Step bound (scaled control units) the SQP ended the last cycle with;
     * zero selects the default for a cold start. */
    double sqp_radius;
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
    /* The best verified plan still violates a wall after a full restoration
     * effort; it is published because it is never worse than the seed. */
    MPC_RTI_CYCLE_ACCEPTED_DEGRADED,
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

typedef enum {
    MPC_SOURCE_DEADLINE_EXCEEDED = 0,
    MPC_SOURCE_PRIMAL_INFEASIBLE,
    MPC_SOURCE_NONLINEAR_STEP_REJECTED,
    MPC_SOURCE_KKT_NOT_CONVERGED,
    MPC_SOURCE_INVALID_MODEL,
    MPC_SOURCE_FEASIBLE_SOLUTION_AVAILABLE
} MpcSourceSolverStatus_t;

typedef struct
{
    MpcRtiCycleStatus_t status;
    MpcSourceSolverStatus_t source_solver_status;
    int source_linearization_failed;
    int source_qp_failed;
    int source_iteration_admission_stopped;
    MpcModelControl_t nominal_first_control;
    MpcModelControl_t first_control;
    float published_steering_command;
    float published_throttle;
    /* Controller discarded stale ADMM iterates after publishing an exactly
     * revalidated recovery action, preserving its nominal/control seed. */
    /* Late unpublished solve was rolled back to the last published source
     * nominal and ADMM state, not left as an unissued command. */
    int sqp_iterations_attempted;
    /* Diagnostic only: the first accepted cost-optimizing SQP step, not a
     * Phase-I physical-feasibility QP. Compare the *applied* alpha step's
     * quadratic racing-cost prediction with its exact source-model cost. */
    /* QP affine footprint versus EXACT nonlinear source rollout at the
     * candidate full step, for each of the first two source SQP passes.
     * Diagnostic-only; no extra source integration is performed. This
     * measures whether local convexification can be trusted in corners. */
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
    int source_iteration_metrics_count;
    MpcRtiSourceIterationMetrics_t source_iteration_metrics[MPC_RTI_SOURCE_TRACE_CAPACITY];
    /* Only compared when neither SQP pass has a valid complete horizon.
     * This selects a proposal for the controller's independent prefix gate;
     * it never certifies the rejected tail. 0=not compared, 1=first, 2=second. */
    /* Infeasible-source-iterate R1->R2 filter: an accepted intermediate is
     * NEVER publishable; only the strict final horizon can be published. */
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
    /* -1=cold/unusable epoch; otherwise elapsed 25ms control stages. */
    double nominal_build_us;
    /* Actual thread CPU spent in ADMM, excluding OS descheduling, per solve. */
    double total_solve_us;
    double validation_us;
    double total_cycle_us;
    /* Common cycle objective for cross-pass and candidate/prior comparisons. */
    MpcRtiObjectiveSchedule_t comparison_objective;
    /* Benchmark capture, independent of publication. Rejected cycles must
     * retain their final primal too. Trace copies are explicitly opt-in. */
    int source_optimized_control_count;
    MpcModelControl_t source_optimized_controls[PREDICTION_HORIZON];
    int source_primal_trace_count;
    int source_primal_trace_truncated;
    MpcRtiSourcePrimalTrace_t source_primal_trace[MPC_RTI_SOURCE_TRACE_CAPACITY];
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

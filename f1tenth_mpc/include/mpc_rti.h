#ifndef MPC_RTI_H
#define MPC_RTI_H

#include "mpc_linearization.h"
#include "mpc_reference.h"
#include "riccati_solver.h"

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
    float weight_steering_target;
    float weight_throttle_command;
    /* Steering penalizes command-increment error relative to the change in
     * path steering feedforward; throttle penalizes raw command increments. */
    float weight_steering_slew;
    float weight_throttle_slew;
    /* Local SQP decision radius about each nominal input. Zero disables the
     * corresponding bound. These bound a linearization step, not actuator
     * slew across physical epochs; successive solves can traverse the full
     * command envelope. */
    float sqp_steering_trust_region_rad;
    float sqp_throttle_trust_region;
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
    /* FD oracle is available only in BUILD_TESTING builds for A/B replay. */
    int use_fd_jacobian_oracle;
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
    float validation_objective;
    float validation_minimum_corridor_slack_m;
    /* Per-cycle benchmark instrumentation; microseconds from CLOCK_MONOTONIC. */
    double reference_seed_us;
    double cold_start_queue_seed_us;
    double nominal_rollout_a_us;
    double progress_reference_resampling_us;
    double nominal_rollout_b_us;
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
    MPC_RTI_ROLLOUT_CORRIDOR
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
    RiccatiStepData_t steps[RICCATI_HORIZON];
    float terminal_Q[RICCATI_MAX_NX];
    float terminal_q[RICCATI_MAX_NX];
    float terminal_x_lb[RICCATI_MAX_NX];
    float terminal_x_ub[RICCATI_MAX_NX];
    float x0[RICCATI_MAX_NX];
    RiccatiStatePolygon_t state_polygons[RICCATI_HORIZON + 1];
    int horizon;
    int nonsmooth_jacobian_columns;
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
    int linearization_failure_stage;
    int linearization_failure_reason;
    int linearization_failure_index;
} MpcRtiProblem_t;

/* Offline replay only. When non-NULL, solve_rti_pass copies the exact QP,
 * effective solver configuration, pre-solve ADMM memory, and raw solver result
 * at the solve boundary. Production leaves this pointer NULL; serialization
 * and all file I/O belong to a development replay executable. */
typedef struct
{
    MpcRtiProblem_t problem;
    RiccatiAdmmConfig_t solver_config;
    RiccatiAdmmState_t initial_solver_state;
    RiccatiAdmmState_t final_solver_state;
    float raw_primal_u[RICCATI_HORIZON][RICCATI_MAX_NU];
    RiccatiSolution_t raw_solution;
    RiccatiDebugInfo_t debug_info;
    RiccatiStatus_t raw_status;
    int capture_count;
    int valid;
} MpcRtiQpCapture_t;

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
} MpcRtiCandidatePathDelta_t;

/* Diagnostic form of the nonlinear corridor test, using the same vehicle
 * footprint and first-prediction margin as production validation. */
int mpc_rti_compute_corridor_slack(
    const MpcModelState_t *state,
    float left_bound_m,
    float right_bound_m,
    const MpcRtiConfiguration_t *configuration,
    int prediction_index,
    float *lower_slack_m,
    float *upper_slack_m);

/* First-order constraints for ALL four body corners, in (e_y,e_psi).
 * Unlike freezing a support extent, this also handles the heading-zero
 * corner switch without selecting an arbitrary derivative of max(). */
int mpc_rti_build_corridor_polygon(
    const MpcModelState_t *state, float left_bound_m, float right_bound_m,
    const MpcRtiConfiguration_t *configuration, int prediction_index,
    RiccatiStatePolygon_t *polygon);

typedef struct
{
    MpcRtiConfiguration_t model;
    RiccatiAdmmConfig_t solver;
    /* Relative end-to-end budget for offline callers; zero means unbounded. */
    double max_cycle_time_us;
    /* Optional absolute CLOCK_MONOTONIC deadline in microseconds. When set,
     * this deadline takes precedence over max_cycle_time_us. */
    double absolute_deadline_monotonic_us;
    /* Exact nonlinear backtracking/merit guard. Every trial remains subject
     * to source-plant rollout, corridor, command-epoch, and deadline checks;
     * among tested exact-feasible backtracks, the production objective picks
     * the winner, with corridor slack as a numerical tie-break. */
    int enable_nonlinear_line_search;
    int line_search_max_backtracks; /* 0..4: alpha halves through 0.0625 */
    int line_search_command_epoch_valid;
    /* Zero/one preserves the single RTI pass. Two enables one deadline-gated
     * relinearization after a rejected corridor step or alpha=0 no-op. */
    int max_sqp_iterations;
    /* Capture already-computed source-output samples for runtime diagnostics.
     * When false, exact rollouts do not retain or copy this debug data. */
    int capture_diagnostic_outputs;
    /* Optional offline-only POD sink. Must remain NULL in production. */
    MpcRtiQpCapture_t *qp_capture;
} MpcRtiCycleConfiguration_t;

typedef struct
{
    RiccatiAdmmState_t solver_state;
    /* Last full-horizon nominal retained for progress seeding. This is not
     * the certified plan that may be replayed after a rejected solve. */
    MpcRtiNominal_t nominal;
    /* Current sequence, including the command just issued at element zero.
     * The next nominal shifts it once, so stage zero matches the next action. */
    MpcModelControl_t warm_start_controls[PREDICTION_HORIZON];
    int warm_start_control_count;
    int warm_start_controls_valid;
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
    MPC_RTI_CYCLE_DEFERRED_DEADLINE
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
    int sqp_iterations_attempted;
    int sqp_iterations_completed;
    int sqp_iterations_accepted;
    /* Only compared when neither SQP pass has a valid complete horizon.
     * This selects a proposal for the controller's independent prefix gate;
     * it never certifies the rejected tail. 0=not compared, 1=first, 2=second. */
    int refinement_prefix_selected_pass;
    float refinement_first_prefix_objective;
    float refinement_second_prefix_objective;
    double refinement_prefix_validation_us;
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
    float line_search_nominal_objective;
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
    double nominal_rollout_b_us;
    double admm_us;
    double exact_rollout_us;
    double nonlinear_validation_us;
    double total_solve_us;
    double total_cycle_us;
    float nonlinear_objective;
} MpcRtiCycleResult_t;

/* Build an absolute-state/input affine LTV QP from a nonlinear nominal. */
int mpc_rti_build_ltv_qp(
    const MpcRtiState_t nominal_states[PREDICTION_HORIZON + 1],
    const MpcModelControl_t nominal_controls[PREDICTION_HORIZON],
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    int horizon,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiProblem_t *problem);

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
/* Clear ADMM primal/dual history while preserving nominal and issued controls. */
void mpc_rti_memory_reset_solver_state(MpcRtiMemory_t *memory);
int mpc_rti_should_keep_first_pass_after_refinement_deadline_skip(
    int first_pass_admm_status,
    int refinement_admm_status,
    int refinement_skipped_for_deadline);
/* Compare exact-valid actions evaluated with the same speed targets. */
int mpc_rti_refinement_improves_action(
    int first_valid, float first_objective,
    int refined_valid, float refined_objective);

/* True when the cycle has enough time remaining to attempt one nonlinear
 * backtracking rollout without bypassing the final publication deadline. */
int mpc_rti_line_search_budget_available(double remaining_us);
int mpc_rti_memory_set_warm_start_controls(
    MpcRtiMemory_t *memory,
    const MpcModelControl_t *controls,
    int control_count);

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

/* Diagnostic evaluators used to test the exact scalar/matrix cost algebra. */
float mpc_rti_matrix_stage_cost(
    const RiccatiStepData_t *stage,
    const float state[MPC_RTI_NX],
    const float control[MPC_RTI_NU]);
float mpc_rti_scalar_stage_cost(
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiReference_t *reference,
    const MpcRtiState_t *state,
    const MpcModelControl_t *control);

#ifdef __cplusplus
}
#endif

#endif

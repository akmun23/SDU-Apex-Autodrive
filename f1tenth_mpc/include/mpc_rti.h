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
    MPC_RTI_NU = MPC_MODEL_NU
};

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
} MpcRtiReference_t;

typedef struct
{
    float weight_e_y;
    float weight_e_psi;
    float weight_u;
    /* Used only while the nominal state is above the raceline speed. */
    float weight_u_overspeed;
    /* Soft cost on per-wheel surface-speed mismatch beyond Unity's forward
     * tire-curve extremum; not a hard slip feasibility limit. */
    float weight_longitudinal_slip;
    float weight_v;
    float weight_r;
    float weight_steering_command;
    float weight_steering_target;
    float weight_throttle_command;
    /* Per-stage squared command increments, (u[k]-u[k-1])^2. */
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
    int horizon;
    int nonsmooth_jacobian_columns;
    int linearization_failure_stage;
    int linearization_failure_reason;
    int linearization_failure_index;
} MpcRtiProblem_t;

typedef struct
{
    int sample_count;
    double progress_error_m[PREDICTION_HORIZON + 1];
    float curvature_error_per_m[PREDICTION_HORIZON + 1];
    float left_bound_error_m[PREDICTION_HORIZON + 1];
    float right_bound_error_m[PREDICTION_HORIZON + 1];
} MpcRtiCandidatePathDelta_t;

typedef struct
{
    MpcRtiConfiguration_t model;
    RiccatiAdmmConfig_t solver;
    /* End-to-end cycle budget; zero means no deadline. */
    double max_cycle_time_us;
} MpcRtiCycleConfiguration_t;

typedef struct
{
    RiccatiAdmmState_t solver_state;
    MpcRtiNominal_t nominal;
} MpcRtiMemory_t;

typedef enum
{
    MPC_RTI_CYCLE_ACCEPTED_OPTIMAL = 0,
    MPC_RTI_CYCLE_REJECTED_INPUT,
    MPC_RTI_CYCLE_REJECTED_SOLVER,
    MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT
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
    int admm_status;
    float primal_residual;
    float dual_residual;
    float maximum_regularization;
    int regularization_count;
    int nonsmooth_jacobian_columns;
    int nonlinear_failure_stage;
    int nonlinear_failure_reason;
    MpcRtiRolloutFailure_t nonlinear_failure;
    /* The solver's exact action sequence when it converged but nonlinear
     * rollout validation rejected that sequence. Diagnostic-only; never a
     * publishable command on a rejected cycle. */
    int candidate_control_count;
    MpcModelControl_t candidate_controls[PREDICTION_HORIZON];
    float rho_start;
    float rho_u_start;
    float rho_final;
    float rho_u_final;
    int rho_change_count;
    int factorization_count;
    uint64_t factorization_time_ns;
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
    double qp_build_us;
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

void mpc_rti_memory_reset(MpcRtiMemory_t *memory);

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

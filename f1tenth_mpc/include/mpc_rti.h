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
    MPC_RTI_PLANT_NX = 7,
    MPC_RTI_NX = 9,
    MPC_RTI_NU = 2,
    MPC_RTI_IDX_EY = 0,
    MPC_RTI_IDX_EPSI = 1,
    MPC_RTI_IDX_U = 2,
    MPC_RTI_IDX_V = 3,
    MPC_RTI_IDX_R = 4,
    MPC_RTI_IDX_TARGET_SPEED = 5,
    MPC_RTI_IDX_STEERING_COMMAND = 6,
    MPC_RTI_IDX_PREVIOUS_STEERING_RATE = 7,
    MPC_RTI_IDX_PREVIOUS_TARGET_SPEED_RATE = 8
};

typedef struct
{
    MpcModelState_t plant;
    float previous_steering_rate;
    float previous_target_speed_rate;
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
    float weight_v;
    float weight_r;
    float weight_steering_command;
    float weight_steering_rate;
    float weight_target_speed_rate;
    float weight_steering_rate_change;
    float weight_target_speed_rate_change;
    float terminal_multiplier;
    float max_speed_mps;
    float active_speed_ceiling_mps;
    float max_steering_rad;
    float max_steering_rate_radps;
    float max_target_speed_rate_increase_mps2;
    float max_target_speed_rate_reduction_mps2;
    float corridor_margin_m;
    /* Optional first predicted sample margin; default equals corridor_margin_m. */
    float first_prediction_corridor_margin_m;
    float corridor_preview_halfwidth_m;
    float nonlinear_corridor_tolerance_m;
    /* FD oracle is available only in BUILD_TESTING builds for A/B replay. */
    int use_fd_jacobian_oracle;
    /* Recovery-seed policy is used only when the nominal seed violates the
     * normal corridor.  Policy zero preserves exact normal-operation parity. */
    int recovery_seed_policy;
    float recovery_steering_k_e_y;
    float recovery_steering_k_e_psi;
    float recovery_steering_k_r;
} MpcRtiConfiguration_t;

enum
{
    MPC_RTI_RECOVERY_SEED_NOMINAL = 0,
    MPC_RTI_RECOVERY_SEED_HEADING_FEEDBACK = 1,
    MPC_RTI_RECOVERY_SEED_BRAKE_HEADING_FEEDBACK = 2
};

typedef struct
{
    MpcRtiState_t states[PREDICTION_HORIZON + 1];
    MpcModelControl_t controls[PREDICTION_HORIZON];
    double progress[PREDICTION_HORIZON + 1];
    int horizon;
    int valid;
} MpcRtiNominal_t;

/* A single corridor schedule shared by the QP and the exact nonlinear
 * rollout.  The normal bounds are never modified.  Active bounds may be
 * directionally extended only before the deterministic bounded seed has
 * re-entered the normal corridor. */
typedef struct
{
    float normal_lower[PREDICTION_HORIZON + 1];
    float normal_upper[PREDICTION_HORIZON + 1];
    float active_lower[PREDICTION_HORIZON + 1];
    float active_upper[PREDICTION_HORIZON + 1];
    float seed_e_y[PREDICTION_HORIZON + 1];
    float seed_violation[PREDICTION_HORIZON + 1];
    int recovery_stage[PREDICTION_HORIZON + 1];
    int horizon;
    int recovery_active;
    int recovery_not_found;
    int first_normal_feasible_stage;
    float initial_normal_violation;
    float max_seed_violation;
} MpcRtiCorridorSchedule_t;

typedef enum
{
    MPC_RTI_ROLLOUT_OK = 0,
    MPC_RTI_ROLLOUT_INVALID_INPUT,
    MPC_RTI_ROLLOUT_INVALID_MODEL,
    MPC_RTI_ROLLOUT_COMMAND_LIMIT,
    MPC_RTI_ROLLOUT_STATE_LIMIT,
    MPC_RTI_ROLLOUT_CORRIDOR
} MpcRtiRolloutStatus_t;

typedef struct
{
    RiccatiStepData_t steps[PREDICTION_HORIZON];
    float terminal_Q[RICCATI_MAX_NX];
    float terminal_q[RICCATI_MAX_NX];
    float terminal_x_lb[RICCATI_MAX_NX];
    float terminal_x_ub[RICCATI_MAX_NX];
    float x0[RICCATI_MAX_NX];
    int horizon;
    int nonsmooth_jacobian_columns;
} MpcRtiProblem_t;

typedef struct
{
    int sample_count;
    double progress_error_m[PREDICTION_HORIZON + 1];
    float curvature_error_per_m[PREDICTION_HORIZON + 1];
    float left_bound_error_m[PREDICTION_HORIZON + 1];
    float right_bound_error_m[PREDICTION_HORIZON + 1];
} MpcRtiCandidatePathDelta_t;

typedef enum
{
    MPC_RTI_REFINEMENT_R1 = 0,
    MPC_RTI_REFINEMENT_R2 = 1,
    MPC_RTI_REFINEMENT_ADAPTIVE = 2
} MpcRtiRefinementMode_t;

enum
{
    MPC_RTI2_TRIGGER_PROGRESS = 1u << 0,
    MPC_RTI2_TRIGGER_CURVATURE = 1u << 1,
    MPC_RTI2_TRIGGER_LEFT_BOUND = 1u << 2,
    MPC_RTI2_TRIGGER_RIGHT_BOUND = 1u << 3,
    MPC_RTI2_TRIGGER_LOW_SLACK = 1u << 4,
    MPC_RTI2_TRIGGER_NONSMOOTH = 1u << 5,
    MPC_RTI2_TRIGGER_ACTION_CORRECTION = 1u << 6,
    MPC_RTI2_TRIGGER_DEGRADED_SOLVE = 1u << 7,
    MPC_RTI2_TRIGGER_RESIDUAL_IMBALANCE = 1u << 8,
    MPC_RTI2_TRIGGER_STEERING_REVERSAL = 1u << 9,
    MPC_RTI2_TRIGGER_LATERAL_LOAD = 1u << 10,
    MPC_RTI2_TRIGGER_R1_CORRIDOR_REPAIR = 1u << 11,
    MPC_RTI2_TRIGGER_R1_RESIDUAL_RECOVERY = 1u << 12
};

typedef struct
{
    MpcRtiConfiguration_t model;
    RiccatiAdmmConfig_t solver;
    MpcRtiRefinementMode_t refinement_mode;
    float rti2_progress_error_trigger_m;
    float rti2_curvature_error_trigger_per_m;
    float rti2_bound_error_trigger_m;
    float rti2_min_corridor_slack_trigger_m;
    float rti2_steering_rate_correction_trigger_radps;
    float rti2_target_speed_rate_correction_trigger_mps2;
    float rti2_residual_imbalance_trigger;
    float rti2_lateral_load_trigger_mps2;
    int rti2_nonsmooth_columns_trigger;
    /* A finite nonlinear-feasible R1 may be used only as an R2 seed when
     * its residual is above the normal degraded gate but below this bound.
     * Zero disables this recovery path. */
    float rti2_residual_recovery_limit;
    float degraded_residual_limit;
    float maximum_regularization;
    int max_consecutive_degraded_solves;
} MpcRtiCycleConfiguration_t;

typedef struct
{
    RiccatiAdmmState_t solver_state;
    MpcRtiNominal_t nominal;
    int consecutive_degraded_solves;
} MpcRtiMemory_t;

typedef enum
{
    MPC_RTI_CYCLE_ACCEPTED_OPTIMAL = 0,
    MPC_RTI_CYCLE_ACCEPTED_DEGRADED,
    MPC_RTI_CYCLE_REJECTED_INPUT,
    MPC_RTI_CYCLE_REJECTED_SOLVER,
    MPC_RTI_CYCLE_REJECTED_RESIDUAL,
    MPC_RTI_CYCLE_REJECTED_REGULARIZATION,
    MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT
} MpcRtiCycleStatus_t;

typedef struct
{
    MpcRtiCycleStatus_t status;
    MpcModelControl_t first_control;
    MpcModelControl_t nominal_first_control;
    MpcRtiCandidatePathDelta_t candidate_path_delta;
    float published_steering_command;
    float published_target_speed;
    int solver_iterations;
    float primal_residual;
    float dual_residual;
    float maximum_regularization;
    int regularization_count;
    int nonsmooth_jacobian_columns;
    int nonlinear_failure_stage;
    int rti_iterations_used;
    int rti2_triggered;
    unsigned int rti2_trigger_reason_mask;
    int rti2_budget_skipped;
    int r1_status;
    int r2_status;
    int nonlinear_failure_reason;
    int r1_nonlinear_failure_reason;
    int r2_nonlinear_failure_reason;
    int r1_nonlinear_failure_stage;
    int r2_nonlinear_failure_stage;
    float r1_nonlinear_objective;
    float r2_nonlinear_objective;
    float r1_min_corridor_slack;
    float r2_min_corridor_slack;
    MpcModelControl_t r1_first_action;
    MpcModelControl_t r2_first_action;
    int r1_solver_iterations;
    int r2_solver_iterations;
    double r1_solve_us;
    double r2_solve_us;
    double total_rti_us;
    int selected_candidate; /* 1=R1, 2=R2. */
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
    int recovery_active;
    int recovery_not_found;
    int recovery_reentry_stage;
    float recovery_initial_normal_violation_m;
    float recovery_max_seed_violation_m;
    float recovery_normal_lower_m[PREDICTION_HORIZON + 1];
    float recovery_normal_upper_m[PREDICTION_HORIZON + 1];
    float recovery_active_lower_m[PREDICTION_HORIZON + 1];
    float recovery_active_upper_m[PREDICTION_HORIZON + 1];
    float recovery_seed_e_y_m[PREDICTION_HORIZON + 1];
    float recovery_seed_violation_m[PREDICTION_HORIZON + 1];
    int recovery_stage[PREDICTION_HORIZON + 1];
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

/* Scheduled variant used by the RTI cycle and its deterministic regression
 * tests.  Passing NULL is equivalent to the normal hard corridor. */
int mpc_rti_build_ltv_qp_with_schedule(
    const MpcRtiState_t nominal_states[PREDICTION_HORIZON + 1],
    const MpcModelControl_t nominal_controls[PREDICTION_HORIZON],
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    int horizon,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiCorridorSchedule_t *schedule,
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

int mpc_rti_build_corridor_schedule(
    const MpcRtiState_t *current_state,
    const MpcRtiNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiCorridorSchedule_t *schedule);

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

MpcRtiRolloutStatus_t mpc_rti_rollout_candidate_with_schedule(
    const MpcRtiState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    const MpcRtiReference_t nominal_references[PREDICTION_HORIZON + 1],
    const double nominal_progress[PREDICTION_HORIZON + 1],
    MpcRtiCandidatePathDelta_t *path_delta,
    int horizon,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiCorridorSchedule_t *schedule,
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

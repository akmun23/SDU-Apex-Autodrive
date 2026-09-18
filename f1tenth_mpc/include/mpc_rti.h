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
    float corridor_preview_halfwidth_m;
    float nonlinear_corridor_tolerance_m;
    /* FD oracle is available only in BUILD_TESTING builds for A/B replay. */
    int use_fd_jacobian_oracle;
} MpcRtiConfiguration_t;

typedef struct
{
    MpcRtiState_t states[PREDICTION_HORIZON + 1];
    MpcModelControl_t controls[PREDICTION_HORIZON];
    double progress[PREDICTION_HORIZON + 1];
    int horizon;
    int valid;
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

typedef struct
{
    MpcRtiConfiguration_t model;
    RiccatiAdmmConfig_t solver;
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

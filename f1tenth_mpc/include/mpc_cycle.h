#ifndef MPC_CYCLE_H
#define MPC_CYCLE_H

#include "vehicle_model.h"
#include <stdint.h>
#include "mpc_reference.h"

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    MPC_CYCLE_MAX_JACOBIAN_THREADS = 8
};

typedef struct
{
    MpcModelState_t plant;
    /* Worst loss_x^2 + loss_y^2 over the incoming interval; see the plant. */
    double incoming_wheel_grip_loss[4];
    float incoming_body_forward_delta_m;
    float incoming_body_lateral_delta_m;
    float incoming_body_yaw_delta_rad;
    int incoming_transition_valid;
    /* Exact loaded-path frame differential for this incoming physical step. */
    float incoming_tracking_jacobian[3][6];
    int incoming_tracking_jacobian_valid;
} MpcCycleState_t;

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
} MpcCycleReference_t;

/* Tracking targets and nominal speed-weight branches held fixed for one SQP
 * objective. Physical rollout and wall bounds still use candidate progress. */
typedef struct
{
    MpcCycleReference_t references[PREDICTION_HORIZON + 1];
    float speed_weights[PREDICTION_HORIZON + 1];
    float e_y_weights[PREDICTION_HORIZON + 1];
    float e_psi_weights[PREDICTION_HORIZON + 1];
    int horizon;
    int valid;
} MpcCycleObjectiveSchedule_t;

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
    /* Scales the lateral-position weight up with |reference curvature|: where
     * the path turns tightly the walls are close and position is measured
     * well. Zero preserves the uniform weight; units are metres. */
    float lateral_curvature_gain_m;
    /* Scales the heading weight up with |reference curvature|. The car's true
     * heading error in tight corners is real (ground truth: median 25 mrad,
     * 99th percentile 115 mrad against 9 mrad planned), so it is weighted more
     * where the path turns and the walls are close. Zero preserves the uniform
     * weight; units are metres. */
    float heading_curvature_gain_m;
    /* Soft cost w*max(0, L - tolerance^2)^2 on each wheel's grip loss
     * L = loss_x^2 + loss_y^2, where a loss is the fraction of the tire
     * curve's peak force given up by sliding past its extremum. Not a hard
     * feasibility limit. */
    float weight_grip_loss;
    /* Fraction of lost peak force that costs nothing (the grip reserve). */
    float grip_loss_tolerance;
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
    /* Optional first predicted sample clearance reserve; default equals
     * corridor_margin_m. Physical walls are validated without this reserve. */
    float first_prediction_corridor_margin_m;
    float corridor_preview_halfwidth_m;
    float nonlinear_corridor_tolerance_m;
    /* Extra corridor tolerance that grows linearly from zero at
     * far_stage_tolerance_start_stage to far_stage_corridor_tolerance_m at the
     * last stage. Set only on the configuration of the independent publication
     * check; zero everywhere else. */
    float far_stage_corridor_tolerance_m;
    int far_stage_tolerance_start_stage;
    /* Independent horizon-stage Jacobians; zero selects the serial fallback. */
    int jacobian_threads;
} MpcSolverConfiguration_t;

typedef enum
{
    MPC_CYCLE_COMMAND_ENVELOPE_OK = 0,
    MPC_CYCLE_COMMAND_ENVELOPE_NONFINITE_STATE,
    MPC_CYCLE_COMMAND_ENVELOPE_SPEED_BELOW_ZERO,
    MPC_CYCLE_COMMAND_ENVELOPE_SPEED_ABOVE_MAX,
    MPC_CYCLE_COMMAND_ENVELOPE_STEERING_QUEUE,
    MPC_CYCLE_COMMAND_ENVELOPE_ACTUAL_STEERING,
    MPC_CYCLE_COMMAND_ENVELOPE_THROTTLE_QUEUE
} MpcCycleCommandEnvelopeFailure_t;

typedef struct
{
    MpcCycleState_t states[PREDICTION_HORIZON + 1];
    MpcModelControl_t controls[PREDICTION_HORIZON];
    double progress[PREDICTION_HORIZON + 1];
    float speed_reference[PREDICTION_HORIZON + 1];
    int horizon;
    int speed_reference_valid;
    int valid;
} MpcCycleNominal_t;

typedef struct
{
    double cost;
    double excess[PREDICTION_HORIZON][4];
} MpcCycleGripMetrics_t;

typedef struct
{
    double objective;
    float minimum_corridor_slack_m;
} MpcCyclePlanMetrics_t;

typedef enum
{
    MPC_CYCLE_ROLLOUT_OK = 0,
    MPC_CYCLE_ROLLOUT_INVALID_INPUT,
    MPC_CYCLE_ROLLOUT_INVALID_MODEL,
    MPC_CYCLE_ROLLOUT_COMMAND_LIMIT,
    MPC_CYCLE_ROLLOUT_STATE_LIMIT,
    MPC_CYCLE_ROLLOUT_CORRIDOR
} MpcCycleRolloutStatus_t;

/* Rear-axle rectangle as eight nonlinear corner inequalities. Source
 * min-slacks equal min(left[0..3]) and min(right[0..3]), including the same
 * first-stage clearance and tolerance. Each corner is differentiable in
 * heading, including e_psi=0; the maximum over corners need not be.
 *
 * Reference-bound slopes are directional at loaded knots and +/- preview-min
 * switches. reference_derivatives_valid=0 marks a nonsmooth reference;
 * primal corner values remain exact.
 *
 * Array order for each side: (rear,-halfwidth),(rear,+halfwidth),
 * (front,-halfwidth),(front,+halfwidth).
 */
typedef struct {
    double left[4];  /* original upper/left wall constraints */
    double right[4]; /* original lower/right wall constraints */
    double physical_left[4];  /* zero-reserve, zero-tolerance physical walls */
    double physical_right[4];
    double left_d_psi[4];
    double right_d_psi[4];
    double left_d_progress_forward;
    double left_d_progress_backward;
    double right_d_progress_forward;
    double right_d_progress_backward;
    int reference_derivatives_valid;
    int reference_progress_kink;
} MpcCycleSourceCorridorCorners_t;
int mpc_cycle_source_exact_corridor_corners(
    const MpcModelState_t *state,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,double progress_m,
    const MpcSolverConfiguration_t *configuration,int prediction_index,
    MpcCycleSourceCorridorCorners_t *out);

/* Signed slack of the vehicle footprint to the lower and upper lateral limits
 * at one stage. A heading or reserve for which no lateral position fits gives
 * a finite negative slack (an infeasible state the solver can restore), never
 * a failed query. */
int mpc_cycle_compute_corridor_slack(
    const MpcModelState_t *state,
    float left_bound_m,
    float right_bound_m,
    const MpcSolverConfiguration_t *configuration,
    int prediction_index,
    float *lower_slack_m,
    float *upper_slack_m);

// Runtime inputs for the source-exact nonlinear SQP cycle.
typedef struct
{
    MpcSolverConfiguration_t model;
    int64_t command_epoch_ns;
    double max_cycle_time_us;
    double absolute_deadline_monotonic_us;
    int sqp_iterations;
    float sqp_violation_penalty;
    // Accepted prediction error of a racing step as a fraction of its starting
    // cost; it sets how far the trust region may grow (see source_sqp::Options).
    float sqp_accuracy_tolerance;
    /* Force substeps per millisecond tick in the derivative passes: the plant
     * integrates five below 5 m/s, which dominates the cost of a linearization.
     * 0 differentiates the plant's own schedule; 1 differentiates one substep
     * per tick, which is about three times cheaper below 5 m/s (derivative
     * blocks within a few percent). The exact rollouts always use the plant's
     * own schedule. */
    int derivative_force_substeps;
    /* The independent publication check tolerates a wall violation that grows
     * linearly over the late stages, from zero at publication_tolerance_start_stage
     * to publication_far_tolerance_m at the last stage. The next cycles re-plan
     * long before those stages are reached, so a small far violation does not
     * make the plan unpublishable; the solver itself still enforces its rows. */
    float publication_far_tolerance_m;
    int publication_tolerance_start_stage;
    const MpcCycleObjectiveSchedule_t *objective_schedule;
    int capture_source_benchmark;
} MpcCycleConfiguration_t;

enum { MPC_CYCLE_SOURCE_TRACE_CAPACITY = 16 };

/* Scalar diagnostics describe the iteration-start controls and exact trial. */
typedef struct {
    /* The merit an iteration reduces: the racing cost when the iterate is
     * feasible, racing cost plus the configured L1 wall penalty otherwise. */
    int feasible_before;
    int accepted;
    /* The accepted step predicted its exact reduction within the tolerance;
     * 0 marks the exact-decrease fallback of a state with no accurate step. */
    int accurate;
    int trial_evaluated;
    int rejected_trials;
    int radius_active;
    int qp_iterations;
    int qp_rounds;
    int newly_activated_grip_wheels;
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
    double grip_cost_before;
    double grip_cost_trial;
} MpcCycleSourceIterationMetrics_t;

/* Second derivatives of the stage maps, recycled from cycle to cycle. Entry
 * [k][m] is a 9x9 (core state, steering target, throttle) Hessian: m = 0..6
 * the core next-state coordinates, m = 7..10 the four interval slip maxima.
 * Stage k of the next cycle is stage k+1 of this one, shifted by the elapsed
 * command stages; only stages without data and the head are recomputed. */
#define MPC_CYCLE_CURVATURE_OUTPUTS 11
#define MPC_CYCLE_CURVATURE_ENTRIES 81
typedef struct
{
    int64_t epoch_ns;
    unsigned char valid[PREDICTION_HORIZON];
    double hessian[PREDICTION_HORIZON][MPC_CYCLE_CURVATURE_OUTPUTS][MPC_CYCLE_CURVATURE_ENTRIES];
} MpcCycleCurvatureCache_t;

typedef struct
{
    /* Last issued plan, retained for progress seeding and gap continuation. */
    MpcCycleNominal_t nominal;
    /* Sequence rooted at command_epoch_ns. Runtime advances by elapsed
     * command stages; same-epoch callbacks must not consume its first action. */
    MpcModelControl_t warm_start_controls[PREDICTION_HORIZON];
    int warm_start_control_count;
    int warm_start_controls_valid;
    int64_t command_epoch_ns;
    /* Step bound (scaled control units) the SQP ended the last cycle with;
     * zero selects the default for a cold start. */
    double sqp_radius;
    /* Cycles left in which the SQP differentiates with the plant's own substep schedule. */
    int sqp_exact_derivative_cycles;
    MpcCycleCurvatureCache_t curvature_cache;
} MpcCycleMemory_t;

typedef enum
{
    /* Independently validated against every physical wall over all 30 stages. */
    MPC_CYCLE_CYCLE_ACCEPTED_FEASIBLE,
    MPC_CYCLE_CYCLE_REJECTED_INPUT,
    MPC_CYCLE_CYCLE_REJECTED_SOLVER,
    MPC_CYCLE_CYCLE_REJECTED_NONLINEAR_ROLLOUT,
    /* No command was published before the soft deadline. */
    MPC_CYCLE_CYCLE_WAITING_FOR_SOLVER_BUDGET
} MpcCycleCycleStatus_t;

typedef enum {
    MPC_SOURCE_DEADLINE_EXCEEDED = 0,
    MPC_SOURCE_PRIMAL_INFEASIBLE,
    MPC_SOURCE_KKT_NOT_CONVERGED,
    MPC_SOURCE_INVALID_MODEL,
    MPC_SOURCE_FEASIBLE_SOLUTION_AVAILABLE
} MpcSourceSolverStatus_t;

typedef struct
{
    MpcCycleCycleStatus_t status;
    MpcSourceSolverStatus_t source_solver_status;
    MpcCycleRolloutStatus_t nonlinear_rollout_status;
    int nonlinear_failure_stage;
    MpcModelControl_t nominal_first_control;
    MpcModelControl_t first_control;
    float published_steering_command;
    float published_throttle;
    int sqp_iterations_attempted;
    /* The first search ended outside the wall and a coasting restart also ran. */
    int sqp_restarted;
    /* SeedFailure of the solve that could not score its seed plan (0 = none). */
    int sqp_seed_failure;
    int source_exact_nlp_accepted_iterations;
    int source_exact_nlp_source_trial_evaluations;
    int source_exact_nlp_deadline_exhausted;
    int source_exact_nlp_initial_source_feasible;
    int source_exact_nlp_final_source_feasible;
    int source_exact_nlp_final_reserve_satisfied;
    int source_exact_nlp_kkt_converged;
    double source_exact_nlp_initial_source_cost;
    double source_exact_nlp_final_source_cost;
    double source_exact_nlp_final_original_margin_m;
    double source_exact_nlp_final_original_violation;
    double source_exact_nlp_final_physical_margin_m;
    int source_iteration_metrics_count;
    MpcCycleSourceIterationMetrics_t source_iteration_metrics[MPC_CYCLE_SOURCE_TRACE_CAPACITY];
    /* The solver's exact action sequence when it converged but nonlinear
     * rollout validation rejected that sequence. Diagnostic-only; never a
     * publishable command on a rejected cycle. */
    int candidate_control_count;
    MpcModelControl_t candidate_controls[PREDICTION_HORIZON];
    MpcModelControl_t nominal_controls[PREDICTION_HORIZON];
    int nominal_control_count;
    double nominal_build_us;
    /* Elapsed source SQP wall time, excluding independent validation. */
    double total_solve_us;
    double validation_us;
    double total_cycle_us;
    /* Common frozen objective for every candidate in this cycle. */
    MpcCycleObjectiveSchedule_t comparison_objective;
    /* Final solver controls, including rejected cycles. */
    int source_optimized_control_count;
    MpcModelControl_t source_optimized_controls[PREDICTION_HORIZON];
} MpcCycleCycleResult_t;

int mpc_cycle_build_objective_schedule(
    const MpcCycleNominal_t *nominal,
    const MpcCycleReference_t references[PREDICTION_HORIZON + 1],
    const MpcSolverConfiguration_t *configuration,
    MpcCycleObjectiveSchedule_t *schedule);

/* Runtime reference/bounds interpolation at one physical path-progress point.
 * Exposed for exact replay tooling so corridor checks use the same preview,
 * contour and clearance policy as the controller. */
int mpc_cycle_reference_at_progress(
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, double progress,
    const MpcSolverConfiguration_t *configuration,
    MpcCycleReference_t *reference);

/* The authoritative source stage operator used by the
 * nominal rollout and by the offline nonlinear multiple-shooting dynamics.
 * It advances the full source wheel/suspension/timer states, 1 ms
 * substeps, path projection, branch flags and within-interval peak tire slip.
 * No linear/QP/spline approximation, no candidate-state clamping.
 *
 * Current state and progress belong to the same actuator epoch. The
 * command is applied directly for exactly one prediction interval.
 */
int mpc_cycle_source_exact_stage(
    const MpcCycleState_t *current,double current_progress,
    const MpcModelControl_t *command,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,MpcCycleState_t *next,
    double *next_progress);

int mpc_cycle_build_nominal_same_epoch(
    const MpcCycleState_t *current_state,
    double current_progress,
    const MpcCycleNominal_t *previous_nominal,
    const MpcModelControl_t *same_epoch_controls,
    int same_epoch_control_count,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcSolverConfiguration_t *configuration,
    MpcCycleNominal_t *nominal,
    MpcCycleReference_t references[PREDICTION_HORIZON + 1]);

/* Recover the exact same stage references as a freshly integrated
 * same-epoch nominal, without repeating the full PhysX/source rollout.
 * Call only for an independently source-integrated, validated nominal. */
int mpc_cycle_references_from_exact_nominal(
    const MpcCycleNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count, double lap_length,
    const MpcSolverConfiguration_t *model,
    MpcCycleReference_t references[PREDICTION_HORIZON + 1]);

/* Exact-source scorer with the same physical footprint planes exported
 * for source-SQP derivatives. Avoids re-evaluating reference interpolation
 * and the footprint against the same 30 already scored source states.
 * Margin rows 1..horizon populated; 0 is the caller's measured state.
 * Optional grip metrics retain the scoring pass's per-wheel excess/cost. */
int mpc_cycle_score_candidate_plan_and_corridor_for_diagnostics(
    const MpcCycleNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,const MpcCycleConfiguration_t *configuration,
    double *objective,float *minimum_corridor_slack_m,
    float corridor_by_stage[PREDICTION_HORIZON+1][2],
    MpcCycleGripMetrics_t *grip_metrics);

/* Independently integrate and validate controls. Optional metrics are written
 * only on success and describe that freshly integrated plan. */
MpcCycleRolloutStatus_t mpc_cycle_validate_candidate_plan(
    const MpcCycleState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    int horizon,
    float prediction_dt,
    const MpcCycleConfiguration_t *configuration,
    MpcCycleNominal_t *validated_plan,
    MpcCyclePlanMetrics_t *metrics,
    int *failure_stage);

void mpc_cycle_memory_reset(MpcCycleMemory_t *memory);
int mpc_cycle_memory_set_warm_start_controls(
    MpcCycleMemory_t *memory,
    const MpcModelControl_t *controls,
    int control_count);
int mpc_cycle_memory_set_warm_start_controls_at_epoch(
    MpcCycleMemory_t *memory, const MpcModelControl_t *controls,
    int control_count, int64_t command_epoch_ns);
/* Nearest discrete source stage; -1 means missing/reversed/expired epoch.
 * Host request jitter does not change the source's fixed 25ms interval. */
int mpc_cycle_command_epoch_shift(int64_t previous_ns, int64_t current_ns,
                              float prediction_dt);

/* One control cycle: build shifted nominal/QP, solve once, validate the exact
 * nonlinear rollout, and commit the candidate only when it is accepted. */
MpcCycleCycleStatus_t mpc_cycle_solve(
    const MpcCycleState_t *current_state,
    double current_progress,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcCycleConfiguration_t *configuration,
    MpcCycleMemory_t *memory,
    MpcCycleCycleResult_t *result);

/* CLOCK_MONOTONIC timestamp in microseconds, shared by the ROS wrapper and
 * cycle core when constructing an absolute per-callback deadline. */
double mpc_cycle_monotonic_time_us(void);

#ifdef __cplusplus
}
#endif

#endif

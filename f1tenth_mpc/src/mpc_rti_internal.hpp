#pragma once

// Private RTI functions shared by the QP builder and exact rollout/scoring.
// Not a second solver or a public controller API.
#include "mpc_rti.h"
#include <cmath>

double monotonic_microseconds(void);
double thread_cpu_microseconds(void);
int finite_rti_state(const MpcRtiState_t *state);
int finite_reference(const MpcRtiReference_t *reference);
int valid_configuration(const MpcRtiConfiguration_t *configuration);
void serialize_state(const MpcRtiState_t *state, float x[MPC_RTI_NX]);

/* One coordinate convention for Riccati solutions, source rollout and
 * feasibility restoration. The plant always consumes absolute commands;
 * the optional increment QP instead optimizes command minus previous command. */
/* Normal mode is previous-absolute-command increment. The source-reachable
 * mode uses the actual physical rack angle as the steering baseline, while
 * retaining the previous issued throttle. This removes the saturated
 * steering-target Jacobian plateau from feasible SQP directions. */
inline int mpc_rti_steering_qp_state_index(const MpcRtiConfiguration_t *model)
{
    return model && model->source_reachable_steering_qp
        ? MPC_OPT_IDX_ACTUAL_STEERING_ANGLE
        : MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND;
}
inline int mpc_rti_steering_qp_mode(const MpcRtiConfiguration_t *model)
{
    return !model || !model->command_increment_qp_enabled ? 0 :
        model->source_reachable_steering_qp ? 2 : 1;
}
inline float mpc_rti_steering_qp_step_limit(
    const MpcRtiConfiguration_t *model, float dt)
{
    if(!model)return 0.f;
    return model->source_reachable_steering_qp
        ? fminf(model->max_steering_command_step_rad,
            .995f * SOURCE_STEERING_RATE_RADPS * dt)
        : model->max_steering_command_step_rad;
}
inline float mpc_rti_qp_decision_from_absolute_command(
    float command, float previous_command, int increment_qp)
{
    return increment_qp ? command - previous_command : command;
}
inline float mpc_rti_absolute_command_from_qp_decision(
    float decision, float previous_command, int increment_qp)
{
    return increment_qp ? previous_command + decision : decision;
}
float corridor_margin_at_prediction(
    const MpcRtiConfiguration_t *configuration, int prediction_index);
float overspeed_weight_for_reference(
    const MpcRtiConfiguration_t *configuration, const MpcRtiReference_t *reference);
float underspeed_weight_for_reference(
    const MpcRtiConfiguration_t *configuration, const MpcRtiReference_t *reference);
float longitudinal_slip_weight_for_reference(
    const MpcRtiConfiguration_t *configuration, const MpcRtiReference_t *reference);
float steering_reference_increment(
    const MpcRtiReference_t *current, const MpcRtiReference_t *next);
double source_slip_cost(
    const MpcRtiConfiguration_t *configuration, const MpcModelState_t *plant,
    const MpcRtiReference_t *reference);

// The actual Unity sideways tire-force curve peaks at 0.01rad and falls
// towards its asymptote at 0.10rad; penalize entry into the falling part
// before post-peak saturation makes SQP locally misleading.
inline constexpr double kMpcSourceFrontLateralSlipReserveRad=0.05;

// Opt-in normal-QP first-command physical admissibility box from the same
// source and original hard map validator; no independent command output.
int mpc_rti_source_first_step_feasible_box(
    const MpcRtiState_t *measured, double progress,
    const MpcTrajectorySample_t *trajectory, size_t count, double lap,
    float prediction_dt, const MpcRtiCycleConfiguration_t *cycle,
    const MpcModelControl_t *nominal_first, double deadline_us,
    float decision_lb[2], float decision_ub[2], int *source_checks);

// Local strict-corridor feasibility derivative with respect to nominal path
// progress (not a physical state-space pose derivative). Phase-I combines it
// with e_y/e_psi and source dynamics through the same Riccati adjoint.
double mpc_rti_source_corridor_progress_derivative(
    const MpcModelState_t &plant, double station, int stage,
    const MpcTrajectorySample_t *trajectory, size_t count, double lap,
    const MpcRtiConfiguration_t *model);

/* Source adjoint from the QP already built at the exact physical nominal.
 * No duplicate rollout, Jacobian build, objective or controller. */
/* Source-informed PSD curvature preserving the exact nominal gradient. */
int mpc_rti_prepare_elastic_corridor_qp(
    MpcRtiProblem_t *problem,const MpcRtiNominal_t *nominal,
    const MpcRtiReference_t references[PREDICTION_HORIZON+1],
    const MpcRtiConfiguration_t *model,float weight,int *activated);

int mpc_rti_exact_source_corridor_violation(
    const MpcRtiState_t *initial_state,double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float prediction_dt,const MpcRtiConfiguration_t *model,
    double *violation_m2,double *minimum_slack_m);

int mpc_rti_select_source_violation_decreasing_step(
    const MpcRtiState_t *initial,double progress,
    const MpcModelControl_t nominal[PREDICTION_HORIZON],
    const MpcModelControl_t proposal[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,size_t count,
    double lap,float dt,const MpcRtiConfiguration_t *model,
    MpcModelControl_t selected[PREDICTION_HORIZON],
    double *initial_violation,double *best_violation,float *alpha_selected);

int mpc_rti_qp_source_gradient(const MpcRtiProblem_t *problem,
    const MpcModelControl_t controls[PREDICTION_HORIZON], int increment_qp,
    double gradient[PREDICTION_HORIZON][MPC_RTI_NU]);

int mpc_rti_add_progress_aware_reserve_qp(
    MpcRtiProblem_t *problem,const MpcRtiNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    const MpcRtiConfiguration_t *model);

/* Opt-in source-progress-aware local corridor reserve curvature. Corrects
 * the 2D wall cost's missing derivative with respect to where on the
 * physically evolving raceline a candidate enters the corner. This adds
 * PSD progress/cross terms to the same existing normal racing SQP, without
 * any new fallback or changes to hard corridor constraints. */
/* Local active-footprint-plane gradient at the source-exact QP nominal.
 * Reuses the QP source LTV dynamics and corridor planes, including the
 * command-increment chain rule. No plant rollout, no numerical derivatives,
 * and no replacement of strict nonlinear corridor validation. The active
 * plane is piecewise smooth and neglects progress-dependent wall motion;
 * exact-source validation is mandatory for a proposed correction. */
int mpc_rti_qp_clearance_gradient(const MpcRtiProblem_t *problem,
    const MpcRtiNominal_t *source_nominal,int increment_qp,
    double gradient[PREDICTION_HORIZON][MPC_RTI_NU],
    float *nominal_min_slack_m,int *limiting_stage);


/* Eight-step (200ms at 40Hz) source-exact, two-control-channel terminal
 * successor feasibility search. Output is a VALIDATED RESEARCH PROBE only;
 * a single 200ms rollout is not a controlled-invariant certificate.
 * The normal QP and its output selection are not modified by the probe. */
enum { MPC_SUCCESSOR_PROBE_HORIZON = 16 };
struct MpcRtiSuccessorProbe_t {
    int feasible{};
    int candidates_evaluated{};
    int horizon{};
    int feasible_candidates{};
    float minimum_clearance_m{};
    double progress_m{};
    float terminal_heading_error_rad{};
    float terminal_lateral_speed_mps{};
    double best_value{};
    MpcModelControl_t controls[MPC_SUCCESSOR_PROBE_HORIZON]{};
};
int mpc_rti_source_successor_merit_gradient(
    const MpcRtiState_t *terminal,double progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcModelControl_t anchor[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    float dt,int horizon,const MpcRtiConfiguration_t *model,
    double *merit,double gradient[MPC_SUCCESSOR_PROBE_HORIZON][2]);

int mpc_rti_source_successor_probe(
    const MpcRtiState_t *terminal, double progress,
    MpcModelControl_t previous_command,
    const MpcTrajectorySample_t *trajectory, size_t count, double lap,
    float dt, const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiSuccessorProbe_t *result, int steps=8, int candidate_limit=9);

/* Soft, frozen eight-step controlled recourse, differentiated using native
 * source plant A/B and the loaded-progress-frame geometry Jacobian. The
 * score and local QP gradient share exactly one frozen policy per RTI cycle. */
int mpc_rti_initialize_successor_recourse(
    const MpcRtiNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,float dt,
    const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiObjectiveSchedule_t *schedule);
int mpc_rti_successor_recourse_cost(
    const MpcRtiNominal_t *plan,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    const MpcRtiConfiguration_t *model,
    const MpcRtiObjectiveSchedule_t *schedule,double *cost);
int mpc_rti_add_successor_recourse_qp(
    MpcRtiProblem_t *problem,const MpcRtiNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    const MpcRtiConfiguration_t *model,
    const MpcRtiObjectiveSchedule_t *schedule);

int mpc_rti_initialize_terminal_targets(const MpcRtiNominal_t *,
    const MpcTrajectorySample_t *,size_t,double,float,
    const MpcRtiConfiguration_t *,MpcRtiObjectiveSchedule_t *);
int mpc_rti_terminal_value(const MpcRtiNominal_t *,
    const MpcTrajectorySample_t *,size_t,double,const MpcRtiConfiguration_t *,
    const MpcRtiObjectiveSchedule_t *,double *,int verify);
int mpc_rti_add_terminal_viability(MpcRtiProblem_t *,const MpcRtiNominal_t *,
    const MpcTrajectorySample_t *,size_t,double,const MpcRtiConfiguration_t *,
    const MpcRtiObjectiveSchedule_t *);

int mpc_rti_reference_at_progress(const MpcTrajectorySample_t *,size_t,
    double,double,const MpcRtiConfiguration_t *,MpcRtiReference_t *);

struct MpcRtiQpPrimalCertificate_t {
    double max_physical_split_error;
    double max_dynamics_equality_error;
    double max_hard_constraint_error;
    double max_elastic_wall_violation_m;
    int worst_constraint_stage;
};
int mpc_rti_qp_primal_certificate(
    const MpcRtiProblem_t *problem,
    const RiccatiSolution_t *solution,
    const RiccatiAdmmState_t *state,
    const float raw_primal_u[RICCATI_HORIZON][RICCATI_MAX_NU],
    MpcRtiQpPrimalCertificate_t *result);

int mpc_rti_recondition_admm_penalties(
    RiccatiAdmmState_t *state,const RiccatiAdmmConfig_t *config);

void mpc_rti_rebase_solver_warm_start(MpcRtiMemory_t *,const MpcRtiNominal_t *,
    const float *,RiccatiAdmmConfig_t *,int stage_shift=1);

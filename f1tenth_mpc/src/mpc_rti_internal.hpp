#pragma once

// Private RTI functions shared by the QP builder and exact rollout/scoring.
// Not a second solver or a public controller API.
#include "mpc_rti.h"

double monotonic_microseconds(void);
double thread_cpu_microseconds(void);
int finite_rti_state(const MpcRtiState_t *state);
int finite_reference(const MpcRtiReference_t *reference);
int valid_configuration(const MpcRtiConfiguration_t *configuration);
void serialize_state(const MpcRtiState_t *state, float x[MPC_RTI_NX]);

/* One coordinate convention for Riccati solutions, source rollout and
 * feasibility restoration. The plant always consumes absolute commands;
 * the optional increment QP instead optimizes command minus previous command. */
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

/* Source adjoint from the QP already built at the exact physical nominal.
 * No duplicate rollout, Jacobian build, objective or controller. */
int mpc_rti_qp_source_gradient(const MpcRtiProblem_t *problem,
    const MpcModelControl_t controls[PREDICTION_HORIZON], int increment_qp,
    double gradient[PREDICTION_HORIZON][MPC_RTI_NU]);

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

void mpc_rti_rebase_solver_warm_start(MpcRtiMemory_t *,const MpcRtiNominal_t *,
    const float *,RiccatiAdmmConfig_t *,int stage_shift=1);

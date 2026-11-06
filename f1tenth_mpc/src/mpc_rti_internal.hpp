#pragma once

// Shared definitions for the ONE original-source nonlinear MPC.
// No alternative QP, Riccati/ADMM, line-search, feasibility-restoration,
// multiple-shooting, terminal-recovery or fallback solver declarations.
#include "mpc_rti.h"
#include <cmath>

double monotonic_microseconds(void);
double thread_cpu_microseconds(void);
int finite_rti_state(const MpcRtiState_t *state);
int finite_reference(const MpcRtiReference_t *reference);
int valid_configuration(const MpcRtiConfiguration_t *configuration);
void serialize_state(const MpcRtiState_t *state, float x[MPC_RTI_NX]);

float corridor_margin_at_prediction(
    const MpcRtiConfiguration_t *configuration,int prediction_index);
float overspeed_weight_for_reference(
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiReference_t *reference);
float underspeed_weight_for_reference(
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiReference_t *reference);
float longitudinal_slip_weight_for_reference(
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiReference_t *reference);
float steering_reference_increment(
    const MpcRtiReference_t *current,const MpcRtiReference_t *next);
double source_slip_cost(
    const MpcRtiConfiguration_t *configuration,
    const MpcModelState_t *plant,
    const MpcRtiReference_t *reference);

inline constexpr double kMpcSourceFrontLateralSlipReserveRad=.05;

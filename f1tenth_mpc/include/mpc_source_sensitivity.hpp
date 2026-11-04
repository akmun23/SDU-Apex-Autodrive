#pragma once
#include "mpc_rti.h"
#include <cstddef>

namespace sdu_apex::source_sqp {
// Differentiates all 30 source stages, not the smaller MPC/QP state.
// Limits: 60 independent normalized control coordinates, 20 dual lanes
// per source interval, physical source-state size 26, 30 scored stages.
// These are LOCAL derivatives of the piecewise source model. They do not
// substitute for full nonlinear acceptance and independent certification.
struct FullSourceDerivatives {
    double gradient[60]{};
    double positive_diagonal_curvature[60]{};
    double footprint[66][60]{};
    double actuator_rate[120][60]{};
    double gn_hessian[60][60]{};
    int dual_plant_intervals{};
    int score_evaluations{};
    double dual_plant_us{};
    double tangent_scoring_us{};
    bool deadline_exhausted{};

};
bool full_source_derivatives(
    const MpcRtiNominal_t *baseline,
    const MpcTrajectorySample_t *trajectory,std::size_t trajectory_count,
    double lap_length,float dt,const MpcRtiCycleConfiguration_t *cfg,
    int decision_count,double normalized_step,double baseline_cost,
    double absolute_deadline_us,FullSourceDerivatives *out,
    bool build_gauss_newton_hessian=false,
    bool use_stage_local_jacobians=false,
    bool compress_positive_drive_passive_timers=false,
    bool corridor_only_phase_one=false);
} // namespace sdu_apex::source_sqp

#pragma once
#include "mpc_rti.h"
#include <cstddef>

namespace sdu_apex::source_sqp {
// Differentiates all 30 source stages, not the smaller MPC/QP state.
// Limits: 60 independent normalized control coordinates, 20 dual lanes
// per source interval, physical source-state size 26, 30 scored stages.
// These are LOCAL derivatives of the piecewise source model. They do not
// substitute for full nonlinear acceptance and independent certification.
// The directional condensed source Hessian contribution of the ORIGINAL
// longitudinal tire-slip loss. This is NOT Gauss-Newton: it includes
// 2*w*r*d2(r)/du2 from the physical 26-state source transition, including
// second-order propagation through previous wheel/suspension states.
// A directional contraction is not a complete 60x60 cost Hessian.
struct SourceSlipDirectionalCurvature {
    double gauss_newton{};
    double residual_weighted{};
    // Separate ORIGINAL front lateral-slip Hessian contribution; zero when
    // its source scorer weight is disabled.
    double front_lateral_gauss_newton{};
    double front_lateral_residual_weighted{};
    double total{};
    double max_first_source_mismatch{};
    int stages{};
    double elapsed_us{};
};
bool source_slip_directional_curvature(
    const MpcRtiNominal_t *nominal,
    const MpcRtiCycleConfiguration_t *configuration,float dt,
    const double normalized_control_direction[60],
    SourceSlipDirectionalCurvature *out,
    bool reuse_precomputed_stage_jacobians=false);

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
    // Original loaded-path / physical-footprint analytic derivative rows.
    // Count per STAGE, not per control, to expose nonsmooth branches.
    int exact_reference_corridor_stages{};
    int nonsmooth_reference_corridor_stages{};
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
    bool corridor_only_phase_one=false,
    bool exact_stage_corridor_fd=false);
} // namespace sdu_apex::source_sqp

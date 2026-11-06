#pragma once
#include "mpc_source_exact_nlp.hpp"
#include <cmath>

namespace sdu_apex::source_nlp {
// Smooth except at ORIGINAL raceline speed-interpolation knots. The
// nonlinear feasibility bound is sampled at the ACTUAL full-source terminal
// position and speed, not frozen at the seed progress or a bicycle model.
// This is a soft terminal *speed-operating* condition, not a guarantee of
// future collision avoidance or a second control policy.
struct TerminalSpeedViability {
    double margin_mps{};
    double reference_speed_mps{};
    double slope_forward{};
    double slope_backward{};
    bool at_reference_knot{};
};

inline bool terminal_source_speed_viability(
    const MpcTrajectorySample_t *track,size_t count,double lap_length,
    const MpcRtiNominal_t &source_nominal,double speed_allowance_mps,
    TerminalSpeedViability *out)
{
    if(!track || !out || !source_nominal.valid ||
       source_nominal.horizon!=kStages ||
       !std::isfinite(speed_allowance_mps) ||
       speed_allowance_mps<0.)return false;
    const double progress=source_nominal.progress[kStages];
    MpcTrajectorySample_t sample{};
    MpcTrajectoryProgressDifferential_t differential{};
    if(!mpc_trajectory_sample(track,count,lap_length,progress,&sample) ||
       !mpc_trajectory_sample_progress_differential(
           track,count,lap_length,progress,&differential))
        return false;
    const double speed=source_nominal.states[kStages].plant.u;
    if(!std::isfinite(sample.speed) || sample.speed<=0. ||
       !std::isfinite(speed) ||
       !std::isfinite(differential.forward.speed) ||
       !std::isfinite(differential.backward.speed))return false;
    *out={
        sample.speed+speed_allowance_mps-speed,
        sample.speed,
        differential.forward.speed,
        differential.backward.speed,
        differential.at_knot!=0
    };
    return true;
}
}

#pragma once
// Load-time physical-corridor-aware *objective* target smoothing. This does
// not alter the source IPOPT geometry, native speed/witness, Frenet bounds,
// vehicle dynamics, optimizer constraints, or collision validator.
#include "mpc_reference.h"
#include "mpc_types.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace f1tenth_mpc {

inline double source_body_lateral_target(
    const MpcTrajectorySample_t &p, double hard_margin_m,
    double desired_extra_margin_m, double maximum_offset_m)
{
    const double heading_error = p.source_witness_valid
        ? p.source_heading_error : 0.;
    const double sn=std::sin(heading_error);
    const double common = hard_margin_m +
        .5 * MPC_CAR_WIDTH_M * std::abs(std::cos(heading_error));
    const double left_margin = common + std::max(
        MPC_REAR_AXLE_TO_FRONT_BUMPER_M*sn, -MPC_REAR_OVERHANG_M*sn);
    const double right_margin = common + std::max(
        -MPC_REAR_AXLE_TO_FRONT_BUMPER_M*sn, MPC_REAR_OVERHANG_M*sn);
    const double lower = right_margin-p.right_bound;
    const double upper = p.left_bound-left_margin;
    if (!std::isfinite(lower)||!std::isfinite(upper)||
        lower>=upper || !(maximum_offset_m>0.))
        return 0.;
    const double preferred_lower = lower + desired_extra_margin_m;
    const double preferred_upper = upper - desired_extra_margin_m;
    if (preferred_lower>preferred_upper) {
        // If the track is physically narrow, aim at the safest center,
        // not at an arbitrarily infeasible one-sided offset.
        return std::clamp(.5*(lower+upper),
                          -maximum_offset_m, maximum_offset_m);
    }
    return std::clamp(std::clamp(0.,preferred_lower,preferred_upper),
                      -maximum_offset_m,maximum_offset_m);
}

struct RobustContourSummary {
    std::size_t affected_knots{};
    double max_abs_offset_m{};
    double max_neighbor_slope{};
};

inline RobustContourSummary precompute_robust_contour(
    std::vector<MpcTrajectorySample_t> &trajectory,
    double lap_length, double hard_margin_m,
    double target_extra_margin_m, double maximum_offset_m,
    double smooth_sigma_m = .70, double forward_preview_m = .45)
{
    RobustContourSummary result{};
    if (trajectory.size()<3 || !(lap_length>0.) ||
        !std::isfinite(lap_length) ||
        !(target_extra_margin_m>0.) ||
        !(maximum_offset_m>0.) ||
        !(smooth_sigma_m>0.))
        return result;
    // Frozen source geometry is sampled for each filter tap, so
    // computation cost is O(knots * fixed taps * binary_search), once.
    // The kernel is spatial, periodic, and shifted forward to begin
    // repositioning before a tight section rather than at the wall.
    constexpr int kSteps=64;
    constexpr double kSpacing=.125;
    std::vector<double> raw(trajectory.size());
    for (std::size_t i=0;i<trajectory.size();++i)
        raw[i]=source_body_lateral_target(trajectory[i], hard_margin_m,
            target_extra_margin_m,maximum_offset_m);
    std::vector<double> smooth(trajectory.size(),0.);
    for (std::size_t i=0;i<trajectory.size();++i) {
        double weighted=0., total=0.;
        const double base=trajectory[i].s;
        for (int tap=-kSteps/2;tap<=kSteps/2;++tap) {
            const double delta=kSpacing*tap;
            const double z=(delta-forward_preview_m)/smooth_sigma_m;
            const double w=std::exp(-.5*z*z);
            MpcTrajectorySample_t sampled{};
            if (!mpc_trajectory_sample(trajectory.data(),trajectory.size(),
                    lap_length,base+delta,&sampled))
                continue;
            // Compute the raw correction from precisely this *physical*
            // location, not from a stale discrete neighbor.
            weighted+=w*source_body_lateral_target(sampled,
                hard_margin_m,target_extra_margin_m,maximum_offset_m);
            total+=w;
        }
        const double candidate=total>0.?weighted/total:raw[i];
        // Never shift the objective reference outside the currently
        // physically permitted (original) oriented body corridor.
        const auto &p=trajectory[i];
        const double heading_error=p.source_witness_valid
            ? p.source_heading_error:0.;
        const double sn=std::sin(heading_error);
        const double common=hard_margin_m+
            .5*MPC_CAR_WIDTH_M*std::abs(std::cos(heading_error));
        const double lower=common+
            std::max(-MPC_REAR_AXLE_TO_FRONT_BUMPER_M*sn,
                      MPC_REAR_OVERHANG_M*sn)-p.right_bound+.005;
        const double upper=p.left_bound-common-
            std::max(MPC_REAR_AXLE_TO_FRONT_BUMPER_M*sn,
                     -MPC_REAR_OVERHANG_M*sn)-.005;
        smooth[i]=lower<=upper?
            std::clamp(candidate,
                std::max(-maximum_offset_m,lower),
                std::min(maximum_offset_m,upper)):0.;
    }
    for (std::size_t i=0;i<trajectory.size();++i) {
        trajectory[i].robust_contour_lateral_offset_m=smooth[i];
        const double abs_offset=std::abs(smooth[i]);
        result.max_abs_offset_m=std::max(result.max_abs_offset_m,abs_offset);
        if(abs_offset>.001)++result.affected_knots;
    }
    // The desired heading follows the tangent of the offset curve,
    // not the tangent of the *unshifted* IPOPT path. A curvature
    // correction provides consistent yaw-rate and steering feedforward.
    // Evaluate derivatives at a fixed physical arc-length scale (not
    // minuscule, irregular IPOPT knot spacing). The raw source state
    // remains untouched; these are objective references only.
    constexpr double kDifferentialDs=.35;
    for(std::size_t i=0;i<trajectory.size();++i) {
        MpcTrajectorySample_t before{}, after{};
        const double progress=trajectory[i].s;
        if (!mpc_trajectory_sample(trajectory.data(),trajectory.size(),
                lap_length,progress-kDifferentialDs,&before) ||
            !mpc_trajectory_sample(trajectory.data(),trajectory.size(),
                lap_length,progress+kDifferentialDs,&after))
            continue;
        const double lateral=smooth[i];
        const double first=(after.robust_contour_lateral_offset_m-
                            before.robust_contour_lateral_offset_m)/
                           (2*kDifferentialDs);
        const double second=(after.robust_contour_lateral_offset_m-
                             2*lateral+
                             before.robust_contour_lateral_offset_m)/
                            (kDifferentialDs*kDifferentialDs);
        const double curvature=trajectory[i].curvature;
        const double denominator=std::max(.5,1-curvature*lateral);
        trajectory[i].robust_contour_heading_offset_rad=
            std::clamp(std::atan(first/denominator),-.20,.20);
        trajectory[i].robust_contour_curvature_delta_radpm=
            std::clamp(second+curvature*curvature*lateral,-.5,.5);
    }
    for(std::size_t i=1;i<trajectory.size();++i){
        const double ds=trajectory[i].s-trajectory[i-1].s;
        if(ds>0.)result.max_neighbor_slope=std::max(result.max_neighbor_slope,
           std::abs(smooth[i]-smooth[i-1])/ds);
    }
    return result;
}
} // namespace f1tenth_mpc

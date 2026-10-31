#pragma once

#include "mpc_reference.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace f1tenth_mpc {

struct SpeedEnvelopeSummary {
    std::size_t reduced_points{};
    double minimum_speed_mps{};
    double maximum_speed_mps{};
    bool witness_invalidated{};
};

// Apply a track-independent, CLOSED-LOOP distance-domain anticipatory speed
// envelope once at track load, never inside the 200 Hz control callback.
// The lateral-acceleration envelope limits speed as sqrt(ay/|curvature|).
// The backward pass adds a finite longitudinal deceleration feasibility bound,
// v(s)^2 <= v(s+ds)^2+2*deceleration*ds.  This is only a speed REFERENCE:
// the unchanged nonlinear source-plant/collision validator remains mandatory.
// Source witness (actuator/motion state) was certified for the original timing;
// modifying that timing invalidates the entire witness, not just one sample.
inline SpeedEnvelopeSummary apply_curvature_speed_envelope(
    std::vector<MpcTrajectorySample_t> &samples, double lap_length_m,
    double lateral_acceleration_mps2, double deceleration_mps2)
{
    if (samples.size()<3 || !std::isfinite(lap_length_m) || lap_length_m<=0 ||
        !std::isfinite(lateral_acceleration_mps2) || lateral_acceleration_mps2<=0 ||
        !std::isfinite(deceleration_mps2) || deceleration_mps2<=0)
        throw std::invalid_argument("invalid curvature speed envelope input");
    std::vector<double> original(samples.size());
    for (std::size_t i=0;i<samples.size();++i) {
        const auto &p=samples[i];
        if (!std::isfinite(p.s) || !std::isfinite(p.curvature) ||
            !std::isfinite(p.speed) || p.speed<=0 ||
            (i && p.s<=samples[i-1].s))
            throw std::invalid_argument("invalid trajectory speed or stations");
        original[i]=p.speed;
    }
    const double seam = lap_length_m - (samples.back().s-samples.front().s);
    if (!(seam>0)) throw std::invalid_argument("non-positive closed-loop seam");
    for (auto &point:samples)
        point.speed=std::min(point.speed,
            std::sqrt(lateral_acceleration_mps2/
                std::max(1e-5,std::abs(point.curvature))));
    // Two backwards passes resolve the cyclic last->first dependence. The
    // first pass propagates through the linear array; the second propagates
    // the wrap constraint through the whole lap.
    for (int pass=0;pass<2;++pass)
        for (std::size_t backwards=samples.size();backwards>0;--backwards) {
            const std::size_t i=backwards-1;
            const std::size_t j=(i+1)%samples.size();
            const double ds=j ? samples[j].s-samples[i].s : seam;
            if (!(ds>0)) throw std::invalid_argument("bad segment distance");
            const double safe=std::sqrt(samples[j].speed*samples[j].speed+
                2*deceleration_mps2*ds);
            samples[i].speed=std::min(samples[i].speed,safe);
        }
    SpeedEnvelopeSummary out{};
    out.minimum_speed_mps=samples.front().speed;
    out.maximum_speed_mps=samples.front().speed;
    for (std::size_t i=0;i<samples.size();++i) {
        auto &point=samples[i];
        if (point.speed < original[i]-1e-6) ++out.reduced_points;
        out.minimum_speed_mps=std::min(out.minimum_speed_mps,point.speed);
        out.maximum_speed_mps=std::max(out.maximum_speed_mps,point.speed);
    }
    if (out.reduced_points) {
        for (auto &point:samples) point.source_witness_valid=0;
        out.witness_invalidated=true;
    }
    return out;
}
}  // namespace f1tenth_mpc

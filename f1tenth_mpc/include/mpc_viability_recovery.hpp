#pragma once

// Track-independent terminal viability screening and smooth trial commands.
// This is not a collision check: every trial must separately pass the exact
// full-horizon production source-plant and corridor validation.
#include "mpc_rti.h"
#include <algorithm>
#include <cmath>

namespace f1tenth_mpc {
inline bool terminal_instability_predicted(const MpcModelState_t &s)
{
    if (!std::isfinite(s.e_y) || !std::isfinite(s.e_psi) ||
        !std::isfinite(s.u) || !std::isfinite(s.v)) return true;
    return std::abs(s.e_psi) > 0.35f ||
        std::abs(s.v) > std::max(0.75f, 0.25f * std::abs(s.u));
}

inline bool terminal_recovered(const MpcModelState_t &s)
{
    return std::isfinite(s.e_y) && std::isfinite(s.e_psi) &&
        std::isfinite(s.u) && std::isfinite(s.v) &&
        std::abs(s.e_psi) < 0.35f &&
        std::abs(s.v) < std::max(0.60f, 0.25f * std::abs(s.u));
}

inline double viability_score(const MpcRtiNominal_t &plan,
                              float expected_terminal_speed)
{
    if (!plan.valid || plan.horizon < 1 ||
        !std::isfinite(plan.validation_minimum_corridor_slack_m)) return INFINITY;
    const MpcModelState_t &s = plan.states[plan.horizon].plant;
    if (!std::isfinite(s.e_y) || !std::isfinite(s.e_psi) ||
        !std::isfinite(s.v) || !std::isfinite(s.u)) return INFINITY;
    return std::abs(s.v) + 2.0 * std::abs(s.e_psi) +
        0.5 * std::abs(s.e_y) +
        10.0 * std::max(0.0f, 0.02f - plan.validation_minimum_corridor_slack_m) +
        0.12 * std::abs(s.u - expected_terminal_speed);
}

// Seed a physically validated steady-actuation bridge from the *current*
// actuator/geometry, not from arbitrary map-indexed steering guesses. This
// expands the recovery search beyond the locally infeasible SQP nominal.
inline void steady_viability_controls(
    MpcModelControl_t *out, int horizon, float steering_target,
    float throttle_target, float steering_limit)
{
    const MpcModelControl_t command{
        std::clamp(steering_target, -steering_limit, steering_limit),
        std::clamp(throttle_target, MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX)};
    for (int k = 0; k < horizon; ++k) out[k] = command;
}

inline void taper_viability_controls(
    const MpcModelControl_t *base,
    MpcModelControl_t *out, int horizon, int taper_steps,
    float steering_correction, float throttle_correction, float steering_limit)
{
    for (int k = 0; k < horizon; ++k) {
        out[k] = base[k];
        if (k >= taper_steps) continue;
        const float fraction = 1.0f - (float)k / (float)taper_steps;
        out[k].steering_target_rad = std::clamp(
            base[k].steering_target_rad + fraction * steering_correction,
            -steering_limit, steering_limit);
        out[k].throttle_command = std::clamp(
            base[k].throttle_command + fraction * throttle_correction,
            MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
    }
}
// Apply a smooth, finite correction before a predicted future wall violation.
// Unlike a beginning-only taper, this leaves the initial commands untouched
// when a certified early prefix already exists. The actuator delay and the
// whole 750 ms source-plant rollout remain authoritative at validation time.
// This merely generates candidates; it never certifies a plan.
inline void targeted_viability_controls(
    const MpcModelControl_t *base, MpcModelControl_t *out,
    int horizon, int first_stage, int failure_stage,
    float steering_correction, float throttle_correction, float steering_limit)
{
    if (horizon <= 0) return;
    const int last = std::clamp(failure_stage, 1, horizon - 1);
    const int first = std::clamp(first_stage, 0, last - 1);
    const int midpoint = first + (last - first) / 2;
    for (int k = 0; k < horizon; ++k) {
        out[k] = base[k];
        if (k < first || k >= last) continue;
        // A short ramp at either end avoids injecting an actuator command
        // discontinuity at the target window boundaries.
        const float up = float(k - first + 1) /
            float(std::max(1, midpoint - first + 1));
        const float down = float(last - k) /
            float(std::max(1, last - midpoint));
        const float fraction = std::min(1.f, std::min(up, down));
        out[k].steering_target_rad = std::clamp(
            base[k].steering_target_rad + fraction * steering_correction,
            -steering_limit, steering_limit);
        out[k].throttle_command = std::clamp(
            base[k].throttle_command + fraction * throttle_correction,
            MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
    }
}

// During a late lateral-wall violation the current triangular correction
// peaks too late relative to the queued actuators and steering delay.
// Start correcting immediately, keep pressure on through the hazard, then
// taper to the source commands. This is only a search seed; exact complete
// source-plant rollout decides whether it is safe to publish.
inline void early_hazard_viability_controls(
    const MpcModelControl_t *base, MpcModelControl_t *out,
    int horizon, int failure_stage, float steering_correction,
    float throttle_correction, float steering_limit)
{
    const int last=std::clamp(failure_stage-2,3,horizon);
    for(int k=0;k<horizon;++k) {
        out[k]=base[k];
        if(k>=last) continue;
        const float onset= k==0 ? 0.55f : 1.0f;
        const float tail= k>=last-4
            ? float(last-k)/4.0f : 1.0f;
        const float amount=std::min(onset,tail);
        out[k].steering_target_rad=std::clamp(
            base[k].steering_target_rad+amount*steering_correction,
            -steering_limit,steering_limit);
        out[k].throttle_command=std::clamp(
            base[k].throttle_command+amount*throttle_correction,
            MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
    }
}

// Signed near-wall rejection determines the corrective steering direction
// from actual Frenet error (left positive). Never steer farther into a wall.
inline float corridor_corrective_sign(float e_y,float lower,float upper)
{
    if(!std::isfinite(e_y)||!std::isfinite(lower)||!std::isfinite(upper))
        return 0.0f;
    if(e_y<lower)return +1.0f;
    if(e_y>upper)return -1.0f;
    return 0.0f;
}

// A full QP step that points through a wall followed by a heavily damped
// accepted line search reveals rapidly shrinking nonlinear feasibility.
// Only the rare combination with small certified wall slack triggers
// anticipatory recovery, preserving the normal racing objective elsewhere.
inline bool line_search_wall_early_warning(
    MpcRtiCycleStatus_t status,int full_step_reason,float alpha,
    float full_horizon_slack_m,float max_alpha,float trigger_slack_m)
{
    return status==MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH &&
        full_step_reason==MPC_RTI_ROLLOUT_CORRIDOR &&
        std::isfinite(alpha) && alpha>0.f && alpha<=max_alpha &&
        std::isfinite(full_horizon_slack_m) &&
        full_horizon_slack_m>=0.f &&
        full_horizon_slack_m<trigger_slack_m;
}

// Expand ten future path-curvature feedforward knots into a *candidate*
// 30-stage command. Holds a legal throttle and linearly interpolates the
// steering target. Never certifies a command; callers MUST use source-plant
// full-horizon body/corridor validation on the live state and actuator queue.
inline bool geometric_feasibility_controls(
    const float steering_knots[10],float steering_gain,float throttle,
    float steering_limit,MpcModelControl_t out[PREDICTION_HORIZON])
{
    if (!steering_knots || !out ||
        !std::isfinite(steering_gain) || steering_gain<0.f ||
        !std::isfinite(throttle) ||
        throttle<MPC_THROTTLE_MIN_DRIVE || throttle>MPC_THROTTLE_MAX ||
        !std::isfinite(steering_limit) || steering_limit<=0.f)
        return false;
    for(int i=0;i<10;++i)
        if (!std::isfinite(steering_knots[i])) return false;
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        const float knot=float(k)*9.f/float(PREDICTION_HORIZON-1);
        const int left=std::min(8,int(knot));
        const float fraction=knot-float(left);
        // First interpolate feed-forward curvature steering *then* apply
        // gain and physical steering clamp (matches the source seed grid).
        const float steer=(1.f-fraction)*steering_knots[left]+
                          fraction*steering_knots[left+1];
        out[k].steering_target_rad=std::clamp(
            steer*steering_gain,-steering_limit,steering_limit);
        out[k].throttle_command=throttle;
    }
    return true;
}

// Move a rejected full-horizon nominal toward a geometric safe steering
// sequence without discarding its high-speed throttle and source timing.
// Both inputs are bounded. This constructs candidates only; every published
// blend requires full exact nonlinear source-plant/corridor certification.
inline bool blended_viability_controls(
    const MpcModelControl_t *geometric,const MpcModelControl_t *nominal,
    int horizon,float nominal_fraction,float steering_limit,
    MpcModelControl_t *out)
{
    if(!geometric || !nominal || !out ||
       horizon<1 || horizon>PREDICTION_HORIZON ||
       !std::isfinite(nominal_fraction) ||
       nominal_fraction<0.f || nominal_fraction>1.f ||
       !std::isfinite(steering_limit) || steering_limit<=0.f)
       return false;
    for(int k=0;k<horizon;++k) {
        const auto &a=geometric[k],&b=nominal[k];
        if(!std::isfinite(a.steering_target_rad) ||
           !std::isfinite(a.throttle_command) ||
           !std::isfinite(b.steering_target_rad) ||
           !std::isfinite(b.throttle_command) ||
           a.throttle_command<MPC_THROTTLE_MIN_DRIVE ||
           b.throttle_command<MPC_THROTTLE_MIN_DRIVE ||
           a.throttle_command>MPC_THROTTLE_MAX ||
           b.throttle_command>MPC_THROTTLE_MAX ||
           std::abs(a.steering_target_rad)>steering_limit ||
           std::abs(b.steering_target_rad)>steering_limit)
           return false;
        out[k].steering_target_rad=
            (1.f-nominal_fraction)*a.steering_target_rad+
            nominal_fraction*b.steering_target_rad;
        out[k].throttle_command=
            (1.f-nominal_fraction)*a.throttle_command+
            nominal_fraction*b.throttle_command;
    }
    return true;
}

} // namespace f1tenth_mpc

#pragma once

#include <algorithm>
#include <cmath>

namespace f1tenth_mpc {

struct PrefixSelection {
    bool choose_candidate{};
    bool clearance_override{};
    bool stale_prior_refresh{};
    bool near_boundary_override{};
};

template <typename Control>
inline bool prefix_control_changes_bounded(
    const Control *candidate, const Control *prior, int count,
    double maximum_steering_delta_rad, double maximum_throttle_delta)
{
    if (!candidate || !prior || count < 1 ||
        !std::isfinite(maximum_steering_delta_rad) ||
        !std::isfinite(maximum_throttle_delta) ||
        maximum_steering_delta_rad < 0.0 || maximum_throttle_delta < 0.0)
        return false;
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(candidate[i].steering_target_rad) ||
            !std::isfinite(prior[i].steering_target_rad) ||
            !std::isfinite(candidate[i].throttle_command) ||
            !std::isfinite(prior[i].throttle_command) ||
            std::abs(candidate[i].steering_target_rad -
                     prior[i].steering_target_rad) >
                maximum_steering_delta_rad ||
            std::abs(candidate[i].throttle_command -
                     prior[i].throttle_command) >
                maximum_throttle_delta)
            return false;
    }
    return true;
}

/* A short exact-valid prefix can still leave the plant rotating away from a
 * strong turn. If the retained prefix has the correct yaw direction, do not
 * replace it with a candidate whose terminal yaw rate materially opposes the
 * path's curvature-speed yaw-rate reference. This is intentionally a narrow
 * direction guard, not a magnitude tracker: the source-plant rollout and
 * objective continue to decide among directionally viable prefixes. */
inline bool candidate_prefix_yaw_direction_conflicts(
    double reference_yaw_rate,
    double candidate_yaw_rate,
    double retained_yaw_rate,
    double minimum_reference_yaw_rate = 1.0,
    double minimum_opposing_candidate_yaw_rate = 0.25)
{
    if (!std::isfinite(reference_yaw_rate) ||
        !std::isfinite(candidate_yaw_rate) ||
        !std::isfinite(retained_yaw_rate) ||
        !std::isfinite(minimum_reference_yaw_rate) ||
        !std::isfinite(minimum_opposing_candidate_yaw_rate) ||
        minimum_reference_yaw_rate < 0.0 ||
        minimum_opposing_candidate_yaw_rate < 0.0) return false;

    return std::abs(reference_yaw_rate) >= minimum_reference_yaw_rate &&
        candidate_yaw_rate * reference_yaw_rate <=
            -minimum_opposing_candidate_yaw_rate &&
        retained_yaw_rate * reference_yaw_rate > 0.0;
}

/* Include the action just issued at index zero; the next-cycle nominal shifts
 * exactly once. Never pad an expired plan and call it a genuine suffix. */
template <typename Control>
inline int copy_retained_plan_warm_start(
    const Control *plan, int count, int issued_index, int capacity,
    Control *destination)
{
    if (!plan || !destination || issued_index < 0 || issued_index >= count ||
        capacity < 1) return 0;
    const int remaining = std::min(capacity, count - issued_index);
    for (int k = 0; k < remaining; ++k)
        destination[k] = plan[issued_index + k];
    return remaining;
}

/* Compose the next nominal after publishing an exact-valid short candidate
 * prefix. The rejected full-horizon QP tail is deliberately excluded: only
 * the validated prefix may come from that candidate. If a retained certified
 * plan exists, continue it at the matching future epoch; otherwise return a
 * prefix-only seed so the nominal builder initializes the unsolved tail from
 * the path. Tail initialization is not a certified or published command.
 *
 * retained_start_index is the retained plan's action aligned with the current
 * callback. Therefore the first retained action after a prefix of length P is
 * retained_start_index + P, not retained_start_index. */
template <typename Control>
inline int compose_candidate_prefix_warm_start(
    const Control *validated_prefix,
    int validated_prefix_count,
    const Control *retained_plan,
    int retained_plan_horizon,
    int retained_start_index,
    int horizon,
    Control *destination)
{
    if (!validated_prefix || !destination || validated_prefix_count < 1 ||
        horizon < validated_prefix_count || retained_plan_horizon < 0 ||
        (retained_plan_horizon > 0 && !retained_plan)) return 0;

    for (int k = 0; k < validated_prefix_count; ++k)
        destination[k] = validated_prefix[k];

    const bool retained_suffix_available = retained_plan &&
        retained_plan_horizon > 0 && retained_start_index >= 0 &&
        retained_start_index < retained_plan_horizon;
    if (!retained_suffix_available) return validated_prefix_count;

    const int available = std::min(horizon,
        retained_plan_horizon - retained_start_index);
    for (int k = validated_prefix_count; k < available; ++k)
        destination[k] = retained_plan[retained_start_index + k];
    return std::max(validated_prefix_count, available);
}

inline bool iteration_limited_prefix_meets_residual_quality(
    double primal_residual,
    double dual_residual,
    double solver_tolerance,
    double tolerance_multiplier)
{
    return std::isfinite(primal_residual) &&
        std::isfinite(dual_residual) && std::isfinite(solver_tolerance) &&
        std::isfinite(tolerance_multiplier) && solver_tolerance > 0.0 &&
        tolerance_multiplier >= 1.0 && primal_residual <=
            solver_tolerance * tolerance_multiplier && dual_residual <=
            solver_tolerance * tolerance_multiplier;
}

// Progress reward permits negative objectives. A positive-only ratio would
// silently disable meaningful improvement checks for every such candidate.
inline bool objective_improves_materially(
    double candidate, double prior, double minimum_relative_gain)
{
    return std::isfinite(candidate) && std::isfinite(prior) &&
        prior - candidate >= std::clamp(minimum_relative_gain, 0.0, 1.0) *
            std::max(std::abs(prior), 1.e-9);
}

/* Select between equal-duration, exact-validated prefixes. A small safety
 * reserve may override objective ordering only when the retained plan is near
 * the corridor boundary and the candidate provides a measurable clearance
 * gain. */
inline PrefixSelection select_validated_prefix(
    bool candidate_valid,
    double candidate_objective,
    double candidate_slack_m,
    bool prior_valid,
    double prior_objective,
    double prior_slack_m,
    double clearance_reserve_m,
    double minimum_clearance_gain_m)
{
    if (candidate_valid && !prior_valid) return {true, false};
    if (!candidate_valid || !prior_valid) return {false, false};

    const bool clearance_override =
        std::isfinite(prior_slack_m) && std::isfinite(candidate_slack_m) &&
        prior_slack_m < clearance_reserve_m &&
        candidate_slack_m > prior_slack_m + minimum_clearance_gain_m;
    if (clearance_override) return {true, true};

    if (std::isfinite(candidate_objective) &&
        std::isfinite(prior_objective))
        return {candidate_objective < prior_objective, false};
    if (std::isfinite(candidate_objective)) return {true, false};
    if (std::isfinite(prior_objective)) return {false, false};
    return {candidate_slack_m > prior_slack_m, false};
}

/* Compare equal-duration prefixes only after both passed exact nonlinear
 * plant/corridor validation. A well-converged candidate uses the normal
 * objective/clearance ordering. A poor-residual iterate normally cannot
 * replace a valid prior prefix; it may do so for a material objective gain,
 * or under the separately gated stale-prior refresh below when a chain of
 * retained actions has aged, the exact candidate prefix preserves margin,
 * and its exact objective is no worse than the retained prefix. After a
 * longer stale chain, a materially clearer candidate may also refresh despite
 * a bounded cost increase; this avoids replaying a less-safe plan solely due
 * to an iteration-limited full-horizon QP. */
inline PrefixSelection select_validated_recovery_prefix(
    bool candidate_valid,
    double candidate_objective,
    double candidate_slack_m,
    bool candidate_solver_converged,
    bool candidate_residual_quality_acceptable,
    bool prior_valid,
    double prior_objective,
    double prior_slack_m,
    double minimum_relative_objective_gain,
    double clearance_reserve_m,
    double minimum_clearance_gain_m,
    int consecutive_prior_plan_actions = 0,
    int stale_prior_refresh_after_actions = 0,
    double stale_prior_max_clearance_loss_m = 0.0,
    double stale_prior_objective_loss_fraction = 0.0,
    double near_boundary_minimum_clearance_gain_m = 0.0,
    double near_boundary_max_objective_ratio = 0.0,
    bool near_boundary_control_changes_bounded = false)
{
    if (!candidate_valid) return {false, false};
    if (!prior_valid) {
        /* If the retained sequence has no exact-valid prefix, an exact
         * source-plant-validated candidate prefix is the only bounded action
         * available. Solver residuals still describe the rejected full QP
         * horizon, not this short nonlinear-checked prefix. Publishing only
         * this validated prefix lets the next 25 ms cycle replan; it never
         * certifies or seeds the rejected tail. Exact validation must also
         * have produced finite cost and nonnegative corridor slack. */
        const bool standalone_prefix_is_well_formed =
            std::isfinite(candidate_objective) &&
            std::isfinite(candidate_slack_m) && candidate_slack_m >= 0.0;
        return {standalone_prefix_is_well_formed, false};
    }
    /* Exact nonlinear validation establishes that both prefixes are safe
     * through the command-effect horizon. If the retained prefix is already
     * below the configured clearance reserve, prefer a materially safer
     * candidate even when its QP residual is poor and its short-horizon cost
     * is higher. Otherwise a stale, lower-cost prior can be replayed until it
     * loses corridor validity. */
    const bool clearance_override = std::isfinite(prior_slack_m) &&
        std::isfinite(candidate_slack_m) &&
        std::isfinite(clearance_reserve_m) &&
        std::isfinite(minimum_clearance_gain_m) &&
        prior_slack_m < clearance_reserve_m &&
        candidate_slack_m > prior_slack_m + minimum_clearance_gain_m;
    if (clearance_override) return {true, true};

    /* Near a hard wall, a small but real increase in exact-validated reserve
     * can matter more than the short-horizon objective. Permit this only when
     * the prior is already below its reserve, the candidate is better by the
     * configured minimum, the objective penalty is bounded, and the first
     * validated controls do not introduce a steering/throttle discontinuity.
     * The hard nonlinear corridor check remains authoritative. */
    const bool near_boundary_override =
        std::isfinite(prior_slack_m) && std::isfinite(candidate_slack_m) &&
        std::isfinite(clearance_reserve_m) &&
        std::isfinite(near_boundary_minimum_clearance_gain_m) &&
        std::isfinite(near_boundary_max_objective_ratio) &&
        prior_slack_m < clearance_reserve_m &&
        near_boundary_minimum_clearance_gain_m > 0.0 &&
        near_boundary_max_objective_ratio >= 1.0 &&
        near_boundary_control_changes_bounded &&
        candidate_slack_m >= prior_slack_m +
            near_boundary_minimum_clearance_gain_m &&
        candidate_objective <= near_boundary_max_objective_ratio *
            std::max(std::abs(prior_objective), 1.e-9);
    if (near_boundary_override) return {true, true, false, true};

    if (candidate_solver_converged) {
        return select_validated_prefix(candidate_valid, candidate_objective,
            candidate_slack_m, prior_valid, prior_objective, prior_slack_m,
            clearance_reserve_m, minimum_clearance_gain_m);
    }

    /* Stale-plan refresh is restricted to iteration-limited candidates. A
     * converged candidate uses ordinary exact-objective ordering above. For
     * an iteration-limited solve, refresh only after a retained-plan chain
     * when both prefixes preserve reserve, clearance loss and objective cost
     * are bounded, and the first validated controls remain within configured
     * change limits. */
    const bool stale_chain_ready = stale_prior_refresh_after_actions > 0 &&
        consecutive_prior_plan_actions >= stale_prior_refresh_after_actions &&
        std::isfinite(candidate_slack_m) && std::isfinite(prior_slack_m) &&
        std::isfinite(stale_prior_max_clearance_loss_m) &&
        stale_prior_max_clearance_loss_m >= 0.0 &&
        candidate_slack_m >= clearance_reserve_m &&
        prior_slack_m >= clearance_reserve_m;
    const bool stale_prior_refresh = stale_chain_ready &&
        std::isfinite(candidate_objective) && std::isfinite(prior_objective) &&
        std::isfinite(stale_prior_objective_loss_fraction) &&
        stale_prior_objective_loss_fraction >= 0.0 &&
        candidate_objective <= prior_objective +
            stale_prior_objective_loss_fraction *
                std::max(std::abs(prior_objective), 1.e-9) &&
        candidate_slack_m >= prior_slack_m -
            stale_prior_max_clearance_loss_m &&
        near_boundary_control_changes_bounded;
    if (stale_prior_refresh) return {true, false, true};

    if (candidate_solver_converged) {
        return select_validated_prefix(candidate_valid, candidate_objective,
            candidate_slack_m, prior_valid, prior_objective, prior_slack_m,
            clearance_reserve_m, minimum_clearance_gain_m);
    }

    if (!candidate_residual_quality_acceptable) {
        constexpr double kPoorResidualMinimumObjectiveGain = 0.10;
        const double minimum_gain = std::max(
            kPoorResidualMinimumObjectiveGain,
            std::clamp(minimum_relative_objective_gain, 0.0, 1.0));
        const bool clearance_preserved =
            std::isfinite(candidate_slack_m) &&
            std::isfinite(prior_slack_m) &&
            candidate_slack_m >= prior_slack_m -
                std::max(0.0, minimum_clearance_gain_m);
        const bool materially_better = objective_improves_materially(
            candidate_objective, prior_objective, minimum_gain);
        return {clearance_preserved && materially_better, false};
    }

    const double minimum_gain = std::clamp(
        minimum_relative_objective_gain, 0.0, 1.0);
    if (objective_improves_materially(
            candidate_objective, prior_objective, minimum_gain)) {
        return {true, false};
    }
    return {false, false};
}

}  // namespace f1tenth_mpc

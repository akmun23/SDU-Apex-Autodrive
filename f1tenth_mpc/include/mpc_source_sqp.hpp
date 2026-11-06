#pragma once
/* Trust-region SQP over the 30 steering/throttle pairs of the horizon.
 *
 * Every candidate is integrated through the source-derived Unity plant and
 * scored by the exact racing objective; the QP is only a local model:
 *
 *   * Gauss-Newton curvature of all quadratic racing residuals,
 *   * the longitudinal tire-slip cost w*max(0, r(U))^2 modelled by an exact
 *     convex epigraph of the first-order residual, so a slip limit that the
 *     QP step would cross is seen *before* the step is taken,
 *   * the exact first-order body-corner, actuator-rate and actuator-box rows,
 *   * a box trust region whose radius is adapted from the ratio of exact to
 *     predicted improvement and carried from cycle to cycle.
 *
 * The loop is anytime: it stops at the deadline, never discards the best
 * exactly verified iterate and reports whether that iterate is feasible.
 */
#include "mpc_rti.h"
#include "unity_source_plant_c.h"

namespace sdu_apex::source_sqp {

inline constexpr int kStages = PREDICTION_HORIZON;
inline constexpr int kPhysical = UNITY_SOURCE_PLANT_STATE_SIZE;
// Physical source state plus tracking error, heading error and progress.
inline constexpr int kState = kPhysical + 3;
inline constexpr int kControls = 2 * kStages;
// QP variables are controls divided by these scales.
inline constexpr double kSteeringScale = 0.15;
inline constexpr double kThrottleScale = 0.25;

struct Options {
    int maximum_iterations{4};
    // Absolute monotonic deadline for the whole solve (0 = none). No new
    // iteration starts unless the measured cost of one still fits.
    double deadline_us{};
    // Initial and maximum trust radius, in scaled control units.
    double trust_radius{0.5};
    double minimum_trust_radius{1.e-3};
    double maximum_trust_radius{4.};
    // Gauss-Newton curvature is shifted by this much to keep the QP strictly
    // convex in directions that no residual observes.
    double curvature_floor{1.};
    // Weight of the exact L1 penalty used while the iterate violates a
    // physical row; large against any cost multiplier of a feasible row.
    double violation_penalty{1.e6};
    const MpcRtiNominal_t *same_epoch_nominal{nullptr};
};

enum class Termination {
    Converged,        // local model predicts no further improvement
    IterationLimit,
    Deadline,
    TrustRegionCollapsed,
    LinearizationFailed,
    QpFailed,
};

struct Iteration {
    bool feasible_before{};
    bool accepted{};
    bool qp_converged{};
    bool trust_region_active{};
    double trust_radius{};
    double step_inf{};
    double cost_before{};
    double cost_trial{};
    double predicted_reduction{};
    double actual_reduction{};
    double ratio{};
    double violation_before{};
    double violation_trial{};
    // Slip cost change predicted by the model versus the exact change.
    double slip_cost_before{};
    double slip_cost_trial{};
    int newly_active_slip{};
    int qp_iterations{};
    int rejected_trials{};
    double linearization_ms{};
    double qp_ms{};
    double rollout_ms{};
    double total_ms{};
};

struct Result {
    Termination termination{Termination::IterationLimit};
    bool initial_feasible{};
    bool final_feasible{};
    bool converged{};
    int accepted_iterations{};
    int rollouts{};
    double initial_cost{};
    double final_cost{};
    double initial_violation{};
    double final_violation{};
    double final_minimum_corridor_m{};
    double trust_radius{};
    MpcRtiNominal_t optimal;
    std::vector<Iteration> history;
};

// Returns false only when the seed itself cannot be integrated/scored.
bool solve(const MpcRtiState_t &initial, double initial_progress,
           const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
           double lap_length, float dt,
           const MpcRtiCycleConfiguration_t &configuration,
           const MpcModelControl_t seed[kStages], const Options &options,
           Result *result);

} // namespace sdu_apex::source_sqp

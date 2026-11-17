#pragma once
/* Sequential quadratic programming over the 30 steering/throttle pairs of the horizon.
 *
 * Every candidate is integrated through the source-derived Unity plant and
 * scored by the exact racing objective; the QP is only a local model:
 *
 *   * exact selected-branch objective curvature through all physical states,
 *     with a separate reflected metric for the convex direction solve,
 *   * the longitudinal tire-slip cost w*max(0, r(U))^2 modelled by an exact
 *     convex epigraph of the first-order residual, so a slip limit that the
 *     QP step would cross is seen *before* the step is taken,
 *   * the exact first-order body-corner rows and the legal command interval,
 *   * a step bound (radius) that is adapted only from the measured ratio of
 *     exact to predicted improvement, and no curvature shift or other damping.
 *
 * The loop is anytime: it stops at the deadline, never discards the best
 * exactly verified iterate and reports whether that iterate is feasible.
 */
#include "mpc_cycle.h"
#include "unity_source_plant_c.h"

#include <cstdint>
#include <vector>

namespace sdu_apex::source_sqp {

inline constexpr int kStages = PREDICTION_HORIZON;
inline constexpr int kPhysical = UNITY_SOURCE_PLANT_STATE_SIZE;
// Physical source state plus tracking error, heading error and progress.
inline constexpr int kState = kPhysical + 3;
// Core coordinates carry the second-order model: forward speed, lateral speed,
// yaw rate, steering rack, tracking error, heading error and progress. First
// derivatives carry the full state. Wheel, body and suspension coordinates
// relax within one stage; see docs/mpc.md#derivative-cost-and-convergence.
enum CoreState { kU, kV, kR, kRack, kEy, kEpsi, kS, kCore };
inline constexpr int kCoreIndex[kCore] = {
    UNITY_SOURCE_U, UNITY_SOURCE_V, UNITY_SOURCE_YAW_RATE,
    UNITY_SOURCE_PHYSICAL_STEERING, kPhysical, kPhysical + 1, kPhysical + 2};
// Second-order inputs: the core state, then steering target and throttle.
inline constexpr int kExpansion = kCore + 2;
inline constexpr int kControls = 2 * kStages;
// QP variables are controls divided by these scales.
inline constexpr double kSteeringScale = 0.15;
inline constexpr double kThrottleScale = 0.25;

struct Options {
    int maximum_iterations{4};
    // Absolute monotonic deadline for the whole solve (0 = none). No new
    // iteration starts unless the measured cost of one still fits.
    double deadline_us{};
    // Recycled stage curvature (see MpcCycleCurvatureCache_t) and the epoch of
    // this solve's first command stage. Without a cache every stage's
    // curvature is computed at the first feasible iterate.
    MpcCycleCurvatureCache_t *curvature_cache{nullptr};
    std::int64_t command_epoch_ns{-1};
    // Weight of the exact L1 penalty used while the iterate violates a
    // physical row; large against any cost multiplier of a feasible row.
    double violation_penalty{1.e6};
    // Step bound in scaled control units: the box over which the exact
    // rollout confirms the model. Carried from cycle to cycle and adapted only
    // from the measured ratio of exact to predicted improvement.
    double radius{0.5};
    double minimum_radius{1.e-3};
    double maximum_radius{4.};
    // A racing trial qualifies if its exact reduction is within this fraction
    // of the starting cost of the model's prediction. A trial outside it
    // adds the measured curvature error along its step to the model (at most
    // maximum_corrections times per iteration) and is re-solved; beyond that
    // the radius is halved.
    double accuracy_tolerance{.05};
    int maximum_corrections{2};
    // Concurrent radii: racing probes r, r/2, ...; restoration probes both
    // sides of r on a symmetric logarithmic grid. Every trial is verified.
    int speculative_trials{4};
    int maximum_rounds{4};
    double sufficient_decrease{1.e-4};
    const MpcCycleNominal_t *same_epoch_nominal{nullptr};
    // Differentiate with the plant's own substep schedule from the first iteration
    // (the previous solve found the capped derivatives unusable for this plan).
    bool exact_derivatives{};
};

enum class Termination {
    Converged,        // local model predicts no further improvement
    IterationLimit,
    Deadline,
    RadiusCollapsed,
    LinearizationFailed,
    QpFailed,
};

struct Iteration {
    bool feasible_before{};
    bool accepted{};
    // The accepted step predicted its exact reduction within the tolerance.
    bool accurate{};
    bool qp_converged{};
    bool radius_active{};
    double radius{};
    double step_inf{};
    int rejected_trials{};
    int qp_rounds{};
    double cost_before{};
    double cost_trial{};
    double predicted_reduction{};
    double actual_reduction{};
    double ratio{};
    double violation_before{};
    double violation_trial{};
    // Slip cost change predicted by the model versus the exact change.
    double grip_cost_before{};
    double grip_cost_trial{};
    double grip_cost_predicted{};
    int newly_active_grip{};
    int qp_iterations{};
    double linearization_ms{};
    double model_ms{};
    double qp_ms{};
    double rollout_ms{};
    double total_ms{};
    int corrections{};
    int attempts{};
};

// Why the seed plan could not be integrated and scored (the solve then returns false).
enum class SeedFailure {
    None,
    PlantStep,            // the exact plant refused a stage
    StageReference,       // the track could not be sampled at a stage
    Scoring,              // the corridor / cost scorer refused the plan
    ReferenceDerivatives, // the reference width slopes are ambiguous (preview edge on a knot, float min parity)
    CorridorParity,       // the two corridor computations disagree
    NonFinite,
};

struct Result {
    Termination termination{Termination::IterationLimit};
    bool initial_feasible{};
    bool final_feasible{};
    // The final plan passes the independent publication check (the reserve is not required).
    bool final_publishable{};
    bool converged{};
    int accepted_iterations{};
    int rollouts{};
    // The first search ended infeasible and a coasting restart also ran.
    bool restarted{};
    SeedFailure seed_failure{};
    // An iteration no capped trial survived was repeated with exact derivatives.
    bool exact_derivatives{};
    double initial_cost{};
    double final_cost{};
    double initial_violation{};
    double final_violation{};
    double final_minimum_corridor_m{};
    double radius{};
    MpcCycleNominal_t optimal;
    std::vector<Iteration> history;
};

// Returns false only when the seed itself cannot be integrated/scored.
bool solve(const MpcCycleState_t &initial, double initial_progress,
           const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
           double lap_length, float dt,
           const MpcCycleConfiguration_t &configuration,
           const MpcModelControl_t seed[kStages], const Options &options,
           Result *result);

} // namespace sdu_apex::source_sqp

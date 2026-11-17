// Include the production model, as the attribution utility does, to check its
// private assembly against the independent exact scorer without a test API.
#include "../src/mpc_source_sqp.cpp"
#include "mpc_trajectory_io.hpp"
#include "../src/mpc_cycle_internal.hpp"

#include <gtest/gtest.h>
#include <fstream>

namespace sdu_apex::source_sqp {
MpcCycleConfiguration_t captured_practice_configuration()
{
    MpcCycleConfiguration_t cfg{};
        cfg.model.max_speed_mps = 16.0;
        cfg.model.weight_e_y = 4500.0;
        cfg.model.weight_e_psi = 1450.0;
        cfg.model.weight_u = 65.0;
        cfg.model.weight_u_overspeed = 20.0;
        cfg.model.overspeed_curvature_gain_m = 0.5;
        cfg.model.underspeed_curvature_gain_m = 0.6;
        cfg.model.weight_grip_loss = 200.0;
        cfg.model.grip_loss_tolerance = 0.2;
        cfg.model.weight_v = 5.0;
        cfg.model.weight_r = 7.8125;
        cfg.model.weight_steering_command = 4.5;
        cfg.model.weight_steering_target = 5.0;
        cfg.model.weight_throttle_command = 0.1;
        cfg.model.weight_steering_slew = 1500.0;
        cfg.model.weight_throttle_slew = 50.0;
        cfg.model.startup_nominal_throttle_seed = 0.25;
        cfg.model.terminal_multiplier = 3.0;
        cfg.model.max_steering_rad = 0.5235987756;
        cfg.model.corridor_margin_m = 0.02;
        cfg.model.first_prediction_corridor_margin_m = 0.02;
        cfg.model.corridor_preview_halfwidth_m = 0.1;
        cfg.model.nonlinear_corridor_tolerance_m = 0.0;
        cfg.model.jacobian_threads = 8;
        cfg.model.active_speed_ceiling_mps = cfg.model.max_speed_mps;
        cfg.model.jacobian_threads=4;
    return cfg;
}

// A fast warm start that leaves the corridor: the same scenario for the tests
// that need an infeasible seed.
struct UnsafeFastStart {
    MpcTrajectorySample_t track[8]{};
    size_t count = 8;
    double lap = 0.;
    MpcCycleConfiguration_t cfg{};
    MpcCycleObjectiveSchedule_t schedule{};
    MpcCycleState_t initial{};
    MpcModelControl_t controls[kStages];

    UnsafeFastStart()
    {
        const double stations[8] = {0., 3.1, 3.2, 100., 200., 300., 400., 500.};
        const double xs[8] = {0., 3.1, 3.2, 100., 100., 0., -100., -100.};
        const double ys[8] = {0., 0., 0., 0., 100., 100., 100., 0.};
        for (int k = 0; k < 8; ++k) {
            track[k].s = stations[k]; track[k].x = xs[k]; track[k].y = ys[k];
            track[k].speed = 3.;
            track[k].left_bound = 1.;
            track[k].right_bound = k < 2 ? 1. : .2;
        }
        EXPECT_TRUE(mpc_trajectory_prepare(track, &count, &lap));
        cfg.model.weight_e_y = 4000.f; cfg.model.weight_u = 35.f;
        cfg.model.weight_throttle_slew = 50.f;
        cfg.model.terminal_multiplier = 3.f;
        cfg.model.max_speed_mps = cfg.model.active_speed_ceiling_mps = 16.f;
        cfg.model.max_steering_rad = .5235987756f;
        cfg.model.corridor_margin_m = cfg.model.first_prediction_corridor_margin_m = .05f;
        cfg.model.corridor_preview_halfwidth_m = .1f;
        cfg.model.jacobian_threads = 1;
        schedule.valid = 1; schedule.horizon = kStages;
        for (int k = 0; k <= kStages; ++k) {
            schedule.references[k].u = 3.f;
            schedule.speed_weights[k] = 35.f;
            schedule.e_y_weights[k] = cfg.model.weight_e_y;
            schedule.e_psi_weights[k] = cfg.model.weight_e_psi;
        }
        cfg.objective_schedule = &schedule;
        initial.plant.u = 3.f; initial.plant.e_y = -.1f;
        EXPECT_TRUE(vehicle_model_initialize_plant_state(&initial.plant));
        for (auto &u : controls) u = {0.f, .5f};
    }
    Problem problem() const { return {&initial, .5, track, count, lap, .025f, &cfg}; }
};

struct ExplicitPlanMetrics {
    double objective{};
    float minimum_slack{INFINITY};
    float corridor[kStages + 1][2]{};
    MpcCycleGripMetrics_t grip{};
};

void explicit_plan_metrics(const Problem &p, const MpcCycleNominal_t &nominal,
                           ExplicitPlanMetrics *out)
{
    *out = {};
    const auto &model = p.configuration->model;
    const auto *schedule = p.configuration->objective_schedule;
    MpcCycleReference_t references[kStages + 1]{};
    for (int k = 0; k <= nominal.horizon; ++k) {
        ASSERT_TRUE(mpc_cycle_reference_at_progress(p.trajectory, p.trajectory_count,
            p.lap_length, nominal.progress[k], &model, &references[k]));
        if (nominal.speed_reference_valid) references[k].u = nominal.speed_reference[k];
        if (!references[k].source_witness_valid)
            references[k].r = references[k].path_curvature * references[k].u;
    }
    const auto *objective_references = schedule ? schedule->references : references;
    double previous_steering = nominal.states[0].plant.delayed_steering_command_2;
    double previous_throttle = nominal.states[0].plant.delayed_throttle_command_2;
    for (int k = 0; k <= nominal.horizon; ++k) {
        const auto &reference = objective_references[k];
        const auto &state = nominal.states[k].plant;
        if (k > 0) {
            ASSERT_TRUE(mpc_cycle_compute_corridor_slack(&state, references[k].left_bound,
                references[k].right_bound, &model, k, &out->corridor[k][0], &out->corridor[k][1]));
            out->minimum_slack = fminf(out->minimum_slack,
                fminf(out->corridor[k][0], out->corridor[k][1]));
        }
        const double terminal_scale = k == nominal.horizon ? model.terminal_multiplier : 1.0;
        const double errors[] = {
            state.e_y - reference.e_y, state.e_psi - reference.e_psi,
            state.u - reference.u, state.v - reference.v, state.r - reference.r,
            state.delayed_steering_command_2 - reference.steering_command,
            state.actual_steering_angle - reference.steering_command};
        const double speed_weight = schedule ? schedule->speed_weights[k] :
            (state.u > reference.u ? overspeed_weight_for_reference(&model, &reference)
                                   : underspeed_weight_for_reference(&model, &reference));
        const double weights[] = {
            schedule ? schedule->e_y_weights[k] : lateral_weight_for_reference(&model, &reference),
            schedule ? schedule->e_psi_weights[k] : heading_weight_for_reference(&model, &reference),
            speed_weight, model.weight_v, model.weight_r,
            model.weight_steering_command, model.weight_steering_command};
        for (size_t i = 0; i < sizeof(errors) / sizeof(errors[0]); ++i)
            out->objective += terminal_scale * weights[i] * errors[i] * errors[i];
        if (k > 0)
            for (int wheel = 0; wheel < 4; ++wheel) {
                const double excess = fmax(0.0, nominal.states[k].incoming_wheel_grip_loss[wheel] -
                    (double)model.grip_loss_tolerance * model.grip_loss_tolerance);
                out->objective += terminal_scale * model.weight_grip_loss * excess * excess;
            }
        if (k == nominal.horizon) continue;
        const auto &control = nominal.controls[k];
        const double target_error = control.steering_target_rad - reference.steering_command;
        out->objective += model.weight_steering_target * target_error * target_error +
            model.weight_throttle_command * control.throttle_command * control.throttle_command;
        const double steering_delta = (double)control.steering_target_rad - previous_steering;
        const double throttle_delta = (double)control.throttle_command - previous_throttle;
        const double steering_correction_delta = steering_delta -
            steering_reference_increment(&objective_references[k], &objective_references[k + 1]);
        out->objective += model.weight_steering_slew * steering_correction_delta * steering_correction_delta +
            model.weight_throttle_slew * throttle_delta * throttle_delta;
        previous_steering = control.steering_target_rad;
        previous_throttle = control.throttle_command;
    }
    // SQP diagnostics sum grip independently, in interval/wheel order.
    for (int k = 0; k < nominal.horizon; ++k)
        for (int wheel = 0; wheel < 4; ++wheel) {
            const double tolerance = model.grip_loss_tolerance;
            const double excess = std::max(0.,
                nominal.states[k + 1].incoming_wheel_grip_loss[wheel] - tolerance * tolerance);
            out->grip.excess[k][wheel] = excess;
            out->grip.cost += model.weight_grip_loss *
                (k + 1 == kStages ? model.terminal_multiplier : 1.) * excess * excess;
        }
}

void explicit_corridor_corners(const Problem &p, const MpcCycleNominal_t &nominal,
                              int k, MpcCycleSourceCorridorCorners_t *out)
{
    *out = {};
    const auto &cfg = p.configuration->model;
    const auto &state = nominal.states[k].plant;
    const double progress = nominal.progress[k];
    MpcCycleReference_t ref{};
    ASSERT_TRUE(mpc_cycle_reference_at_progress(p.trajectory, p.trajectory_count,
        p.lap_length, progress, &cfg, &ref));
    float right, left, physical_right, physical_left;
    ASSERT_TRUE(mpc_cycle_compute_corridor_slack(&state, ref.left_bound,
        ref.right_bound, &cfg, k, &right, &left));
    auto physical = cfg;
    physical.corridor_margin_m = physical.first_prediction_corridor_margin_m = 0.f;
    physical.nonlinear_corridor_tolerance_m = 0.f;
    ASSERT_TRUE(mpc_cycle_compute_corridor_slack(&state, ref.left_bound,
        ref.right_bound, &physical, k, &physical_right, &physical_left));
    const double psi = state.e_psi;
    const double sine = std::sin(psi), cosine = std::cos(psi);
    const double x[2] = {-double(MPC_REAR_OVERHANG_M), double(MPC_REAR_AXLE_TO_FRONT_BUMPER_M)};
    const double y[2] = {-.5 * double(MPC_CAR_WIDTH_M), .5 * double(MPC_CAR_WIDTH_M)};
    double lat[4], derivative[4], minimum = INFINITY, maximum = -INFINITY;
    for (int along = 0; along < 2; ++along)
        for (int across = 0; across < 2; ++across) {
            const int j = 2 * along + across;
            lat[j] = x[along] * sine + y[across] * cosine;
            derivative[j] = x[along] * cosine - y[across] * sine;
            minimum = std::min(minimum, lat[j]);
            maximum = std::max(maximum, lat[j]);
        }
    for (int j = 0; j < 4; ++j) {
        out->left[j] = double(left) + (maximum - lat[j]);
        out->right[j] = double(right) + (lat[j] - minimum);
        out->physical_left[j] = double(physical_left) + (maximum - lat[j]);
        out->physical_right[j] = double(physical_right) + (lat[j] - minimum);
        out->left_d_psi[j] = -derivative[j];
        out->right_d_psi[j] = derivative[j];
    }
    struct Width { float value; double forward, backward; };
    const auto update = [](Width &w, float value, double forward, double backward) {
        if (value < w.value) w = {value, forward, backward};
        else if (value == w.value) {
            w.forward = std::min(w.forward, forward);
            w.backward = std::max(w.backward, backward);
        }
    };
    MpcTrajectorySample_t center{};
    MpcTrajectoryProgressDifferential_t differential{};
    ASSERT_TRUE(mpc_trajectory_sample(p.trajectory, p.trajectory_count, p.lap_length, progress, &center));
    ASSERT_TRUE(mpc_trajectory_sample_progress_differential(
        p.trajectory, p.trajectory_count, p.lap_length, progress, &differential));
    Width lw{float(center.left_bound), differential.forward.left_bound, differential.backward.left_bound};
    Width rw{float(center.right_bound), differential.forward.right_bound, differential.backward.right_bound};
    const double half = cfg.corridor_preview_halfwidth_m;
    if (half > 0.) {
        for (double q : {progress - half, progress + half}) {
            MpcTrajectorySample_t endpoint{};
            ASSERT_TRUE(mpc_trajectory_sample(p.trajectory, p.trajectory_count, p.lap_length, q, &endpoint));
            ASSERT_TRUE(mpc_trajectory_sample_progress_differential(
                p.trajectory, p.trajectory_count, p.lap_length, q, &differential));
            update(lw, float(endpoint.left_bound), differential.forward.left_bound, differential.backward.left_bound);
            update(rw, float(endpoint.right_bound), differential.forward.right_bound, differential.backward.right_bound);
        }
        const double first = p.trajectory[0].s, lap = p.lap_length;
        double wrapped = fmod(progress - first, lap);
        if (wrapped < 0.) wrapped += lap;
        wrapped += first;
        const double lo = wrapped - half, hi = wrapped + half;
        const auto knots = [&](double lower, double upper) {
            const double tolerance = 1.e-11 * std::max(1., lap);
            const bool moving_lower = std::abs(lower-lo) <= tolerance || std::abs(lower-(lo+lap)) <= tolerance;
            const bool moving_upper = std::abs(upper-hi) <= tolerance || std::abs(upper-(hi-lap)) <= tolerance;
            for (size_t i = 0; i < p.trajectory_count; ++i) {
                const auto &point = p.trajectory[i];
                if (point.s < lower || point.s > upper) continue;
                if ((moving_lower && std::abs(point.s-lower) <= tolerance) ||
                    (moving_upper && std::abs(point.s-upper) <= tolerance)) {
                    // A knot at a moving endpoint is the endpoint: value only, no slope of its own.
                    lw.value = std::min(lw.value, float(point.left_bound));
                    rw.value = std::min(rw.value, float(point.right_bound));
                    continue;
                }
                update(lw, float(point.left_bound), 0., 0.);
                update(rw, float(point.right_bound), 0., 0.);
            }
        };
        if (lo < first) { knots(lo+lap, first+lap); knots(first, hi); }
        else if (hi >= first+lap) { knots(lo, first+lap); knots(first, hi-lap); }
        else knots(lo, hi);
    }
    EXPECT_EQ(lw.value, ref.left_bound);
    EXPECT_EQ(rw.value, ref.right_bound);
    out->reference_derivatives_valid = 1;
    out->left_d_progress_forward = lw.forward;
    out->left_d_progress_backward = lw.backward;
    out->right_d_progress_forward = rw.forward;
    out->right_d_progress_backward = rw.backward;
    const double ls = std::max({1., std::abs(lw.forward), std::abs(lw.backward)});
    const double rs = std::max({1., std::abs(rw.forward), std::abs(rw.backward)});
    out->reference_progress_kink = std::abs(lw.forward-lw.backward) > 1.e-10*ls ||
        std::abs(rw.forward-rw.backward) > 1.e-10*rs;
}

TEST(SourceSqp, SharedStageGeometryMatchesExplicitSamplingAndScoring)
{
    UnsafeFastStart start;
    auto at = std::make_unique<Snapshot>();
    ASSERT_TRUE(rollout(start.problem(), start.controls, nullptr, at.get()));
    const double queries[] = {-.037, 0., .037, 3.1, 3.15, 3.2, 3.225, 499.99};
    start.cfg.model.weight_e_psi = 1450.f;
    start.cfg.model.weight_v = 5.f; start.cfg.model.weight_r = 7.8125f;
    start.cfg.model.weight_steering_command = 4.5f;
    start.cfg.model.weight_steering_target = 5.f;
    start.cfg.model.weight_throttle_command = .1f;
    start.cfg.model.weight_steering_slew = 1500.f;
    for (int i = 0; i < 8; ++i) {
        auto &point = start.track[i];
        point.left_bound = i%3 == 0 ? .7 : 1.1;
        point.curvature = .01*(i+1);
        point.robust_contour_lateral_offset_m = .015*i;
        point.robust_contour_heading_offset_rad = -.004*i;
        point.robust_contour_curvature_delta_radpm = .002*i;
        point.source_heading_error = .007*i;
        point.source_v_rear = .031*i; point.source_yaw_rate = .12*i;
        point.source_physical_steering = .017*i;
        point.source_steering_target = .023*i;
    }
    for (float half : {0.f, .1f, .125f})
        for (bool witness : {false, true})
            for (float ceiling : {2.f, 16.f})
                for (bool scheduled : {false, true}) {
                    start.cfg.objective_schedule = scheduled ? &start.schedule : nullptr;
                    start.cfg.model.corridor_preview_halfwidth_m = half;
                    start.cfg.model.weight_grip_loss = 200.25f;
                    start.cfg.model.terminal_multiplier = 3.1f;
                    start.cfg.model.active_speed_ceiling_mps = ceiling;
                    for (auto &point : start.track) {
                        point.source_witness_valid = witness;
                        point.source_throttle = .25;
                    }
                    for (int k = 0; k <= kStages; ++k) {
                        at->nominal.progress[k] = queries[k % 8];
                        at->nominal.states[k].plant.e_psi = .017f * (k - 11);
                        at->nominal.states[k].plant.e_y = .03f * (k - 7);
                    }
                    const Problem p = start.problem();
                    // Kinks retain geometry but can make the SQP evaluation unusable.
                    const bool valid = evaluate(p, at->nominal, &at->evaluation);
                    ExplicitPlanMetrics expected;
                    explicit_plan_metrics(p, at->nominal, &expected);
                    EXPECT_EQ(at->evaluation.objective, expected.objective);
                    EXPECT_EQ(at->evaluation.grip.cost, expected.grip.cost);
                    double scored_cost;
                    float scored_minimum, scored_corridor[kStages+1][2]{};
                    MpcCycleGripMetrics_t scored_grip{};
                    ASSERT_TRUE(mpc_cycle_score_candidate_plan_and_corridor_for_diagnostics(
                        &at->nominal,p.trajectory,p.trajectory_count,p.lap_length,&start.cfg,
                        &scored_cost,&scored_minimum,scored_corridor,&scored_grip));
                    EXPECT_EQ(scored_cost,expected.objective);
                    EXPECT_EQ(scored_minimum,expected.minimum_slack);
                    EXPECT_EQ(scored_grip.cost,expected.grip.cost);
                    bool derivatives_valid = true;
                    double violation = 0., minimum_corner = INFINITY, minimum_physical = INFINITY;
                    for (const auto &margins : at->evaluation.speed_margin)
                        for (double margin : margins) violation += std::max(0., -margin);
                    for (int k = 0; k <= kStages; ++k) {
                        MpcCycleReference_t reference{};
                        ASSERT_TRUE(mpc_cycle_reference_at_progress(p.trajectory, p.trajectory_count,
                            p.lap_length, at->nominal.progress[k], &start.cfg.model, &reference));
                        EXPECT_EQ(at->nominal.speed_reference[k], k == 0 ?
                            at->nominal.states[0].plant.u : reference.u);
                    }
                    for (int k = 1; k <= kStages; ++k) {
                        EXPECT_EQ(scored_corridor[k][0],expected.corridor[k][0]);
                        EXPECT_EQ(scored_corridor[k][1],expected.corridor[k][1]);
                        MpcCycleSourceCorridorCorners_t corners{};
                        explicit_corridor_corners(p, at->nominal, k, &corners);
                        const auto &actual = at->evaluation.corners[k-1];
                        derivatives_valid &= corners.reference_derivatives_valid;
                        EXPECT_EQ(actual.reference_derivatives_valid, corners.reference_derivatives_valid);
                        EXPECT_EQ(actual.reference_progress_kink, corners.reference_progress_kink);
                        EXPECT_EQ(actual.left_d_progress_forward, corners.left_d_progress_forward);
                        EXPECT_EQ(actual.left_d_progress_backward, corners.left_d_progress_backward);
                        EXPECT_EQ(actual.right_d_progress_forward, corners.right_d_progress_forward);
                        EXPECT_EQ(actual.right_d_progress_backward, corners.right_d_progress_backward);
                        for (int j = 0; j < 4; ++j) {
                            EXPECT_EQ(actual.left[j], corners.left[j]);
                            EXPECT_EQ(actual.right[j], corners.right[j]);
                            EXPECT_EQ(actual.physical_left[j], corners.physical_left[j]);
                            EXPECT_EQ(actual.physical_right[j], corners.physical_right[j]);
                            EXPECT_EQ(actual.left_d_psi[j], corners.left_d_psi[j]);
                            EXPECT_EQ(actual.right_d_psi[j], corners.right_d_psi[j]);
                            violation += std::max(0., -corners.right[j]) + std::max(0., -corners.left[j]);
                            minimum_corner = std::min({minimum_corner, corners.right[j], corners.left[j]});
                            minimum_physical = std::min({minimum_physical, corners.physical_left[j], corners.physical_right[j]});
                        }
                        for (int wheel = 0; wheel < 4; ++wheel) {
                            EXPECT_EQ(at->evaluation.grip.excess[k-1][wheel], expected.grip.excess[k-1][wheel]);
                            EXPECT_EQ(scored_grip.excess[k-1][wheel], expected.grip.excess[k-1][wheel]);
                        }
                    }
                    EXPECT_EQ(valid, derivatives_valid);
                    if (valid) {
                        EXPECT_EQ(at->evaluation.violation, violation);
                        EXPECT_EQ(at->evaluation.minimum_corner, minimum_corner);
                        EXPECT_EQ(at->evaluation.minimum_physical, minimum_physical);
                        const bool physical = minimum_physical >= 0. && at->evaluation.minimum_speed >= 0. &&
                            at->evaluation.minimum_box >= -kBoxTolerance;
                        EXPECT_EQ(at->evaluation.physical_feasible, physical);
                        EXPECT_EQ(at->evaluation.feasible, physical && minimum_corner >= 0.);
                    }
                }
}

TEST(SourceSqp, SinglePassGripAndCachedModelCostMatchExplicitArithmetic)
{
    UnsafeFastStart start;
    for (auto &point : start.track) point.left_bound = point.right_bound = 10.;
    const Problem p = start.problem();
    auto at = std::make_unique<Snapshot>();
    ASSERT_TRUE(rollout(p, start.controls, nullptr, at.get()));
    auto lin = std::make_unique<Linearization>();
    for (auto &sensitivity : lin->sensitivity) sensitivity.setZero();
    Model model;
    for (float weight : {200.25f, 0.f}) {
        start.cfg.model.weight_grip_loss = weight;
        start.cfg.model.grip_loss_tolerance = .36f;
        start.cfg.model.terminal_multiplier = 3.1f;
        const double tolerance = double(start.cfg.model.grip_loss_tolerance) * start.cfg.model.grip_loss_tolerance;
        for (int k = 0; k < kStages; ++k)
            for (int wheel = 0; wheel < 4; ++wheel) {
                const double residual = (k + wheel) % 3 == 0 ? -.01 : .0013 * (k + 1) * (wheel + 1);
                at->nominal.states[k + 1].incoming_wheel_grip_loss[wheel] = tolerance + residual;
                lin->stages[k].grip_residual[wheel] = residual;
                lin->stages[k].grip_control[wheel][1] = .01 * (wheel + 1);
            }
        ExplicitPlanMetrics expected;
        explicit_plan_metrics(p, at->nominal, &expected);
        ASSERT_TRUE(evaluate(p, at->nominal, &at->evaluation));
        EXPECT_EQ(at->evaluation.objective, expected.objective);
        EXPECT_EQ(at->evaluation.grip.cost, expected.grip.cost);
        for (int k = 0; k < kStages; ++k)
            for (int wheel = 0; wheel < 4; ++wheel)
                EXPECT_EQ(at->evaluation.grip.excess[k][wheel], expected.grip.excess[k][wheel]);
        ASSERT_TRUE(build_model(p, *at, *lin, &model));
        double explicit_nominal_cost = 0.;
        for (const auto &row : model.grip)
            explicit_nominal_cost += row.weight * row.clipped * row.clipped;
        EXPECT_EQ(model.nominal_grip_cost, explicit_nominal_cost);
        Model corrected = model;
        // Curvature and wall corrections retain the same grip rows/cache.
        corrected.hessian += Matrix::Identity(kControls, kControls);
        corrected.direction_hessian = corrected.hessian;
        for (double radius : {.125, .25}) {
            Step step;
            Options options;
            ASSERT_TRUE(solve_step(corrected, true, options, radius, &step));
            const double old_prediction = -(corrected.gradient.dot(step.x) +
                0.5 * step.x.dot(corrected.hessian * step.x) +
                grip_model_cost(corrected, step.x) - explicit_nominal_cost);
            EXPECT_EQ(step.predicted_reduction, old_prediction);
        }
    }
}

TEST(SourceSqp, ValidationMetricsAndGeometryMatchExplicitScoring)
{
    UnsafeFastStart start;
    for (auto &point : start.track) point.left_bound = point.right_bound = 10.;
    start.cfg.model.weight_grip_loss = 200.25f;
    start.cfg.model.grip_loss_tolerance = .36f;
    start.cfg.model.terminal_multiplier = 3.1f;
    start.cfg.model.weight_steering_target = 5.f;
    start.cfg.model.weight_throttle_command = .7f;
    const Problem p = start.problem();
    auto exact = std::make_unique<Snapshot>();
    ASSERT_TRUE(rollout(p, start.controls, nullptr, exact.get()));
    for (bool scheduled : {true, false}) {
        start.cfg.objective_schedule = scheduled ? &start.schedule : nullptr;
        auto checked = std::make_unique<MpcCycleNominal_t>(exact->nominal);
        checked->states[1].plant.u = NAN;
        checked->progress[1] = NAN;
        MpcCyclePlanMetrics_t metrics{-1., -1.f};
        int failed = 99;
        ASSERT_EQ(mpc_cycle_validate_candidate_plan(&start.initial, p.progress, start.controls,
            p.trajectory, p.trajectory_count, p.lap_length, kStages, p.dt, &start.cfg,
            checked.get(), &metrics, &failed), MPC_CYCLE_ROLLOUT_OK);
        EXPECT_EQ(failed, -1);
        ASSERT_TRUE(checked->valid);
        EXPECT_FALSE(checked->speed_reference_valid);
        // The safety rollout must overwrite the supplied plan independently.
        for (int k = 0; k <= kStages; ++k) {
            EXPECT_EQ(checked->progress[k], exact->nominal.progress[k]);
            EXPECT_EQ(checked->states[k].plant.u, exact->nominal.states[k].plant.u);
            EXPECT_EQ(checked->states[k].plant.e_y, exact->nominal.states[k].plant.e_y);
            EXPECT_EQ(checked->states[k].plant.e_psi, exact->nominal.states[k].plant.e_psi);
            for (int wheel = 0; wheel < 4; ++wheel)
                EXPECT_EQ(checked->states[k].incoming_wheel_grip_loss[wheel],
                          exact->nominal.states[k].incoming_wheel_grip_loss[wheel]);
        }
        ExplicitPlanMetrics expected;
        explicit_plan_metrics(p, *checked, &expected);
        EXPECT_EQ(metrics.objective, expected.objective);
        EXPECT_EQ(metrics.minimum_corridor_slack_m, expected.minimum_slack);
        // Clearance uses the same float footprint/minimum, including negatives.
        for (float offset : {0.f, 20.f}) {
            auto shifted = std::make_unique<MpcCycleNominal_t>(exact->nominal);
            for (int k = 1; k <= kStages; ++k) shifted->states[k].plant.e_y += offset;
            explicit_plan_metrics(p, *shifted, &expected);
            float minimum;
            ASSERT_TRUE(minimum_candidate_corridor_slack(shifted.get(), p.trajectory,
                p.trajectory_count, p.lap_length, &start.cfg.model, &minimum));
            EXPECT_EQ(minimum, expected.minimum_slack);
            EXPECT_EQ(minimum >= 0.f, expected.minimum_slack >= 0.f);
        }
    }
    for (int stage : {0, 1, kStages}) {
        auto invalid = std::make_unique<MpcCycleNominal_t>(exact->nominal);
        invalid->progress[stage] = NAN;
        float minimum;
        EXPECT_FALSE(minimum_candidate_corridor_slack(invalid.get(), p.trajectory,
            p.trajectory_count, p.lap_length, &start.cfg.model, &minimum));
        double cost;
        float corridor[kStages + 1][2]{};
        EXPECT_FALSE(mpc_cycle_score_candidate_plan_and_corridor_for_diagnostics(invalid.get(),
            p.trajectory, p.trajectory_count, p.lap_length, &start.cfg,
            &cost, &minimum, corridor, nullptr));
    }
}

TEST(SourceSqp, ValidationFailuresKeepTheirStageAndLeaveMetricsUnwritten)
{
    UnsafeFastStart start;
    for (auto &point : start.track) point.left_bound = point.right_bound = 10.;
    const Problem p = start.problem();
    auto checked = std::make_unique<MpcCycleNominal_t>();
    for (int stage : {0, 3, kStages - 1}) {
        MpcModelControl_t controls[kStages];
        std::copy_n(start.controls, kStages, controls);
        controls[stage].throttle_command = 2.f;
        MpcCyclePlanMetrics_t metrics{3.125, .25f};
        int failed = 99;
        EXPECT_EQ(mpc_cycle_validate_candidate_plan(&start.initial, p.progress, controls,
            p.trajectory, p.trajectory_count, p.lap_length, kStages, p.dt, &start.cfg,
            checked.get(), &metrics, &failed), MPC_CYCLE_ROLLOUT_COMMAND_LIMIT);
        EXPECT_EQ(failed, stage);
        EXPECT_FALSE(checked->valid);
        EXPECT_EQ(metrics.objective, 3.125);
        EXPECT_EQ(metrics.minimum_corridor_slack_m, .25f);
    }
    start.schedule.valid = 0;
    MpcCyclePlanMetrics_t metrics{3.125, .25f};
    int failed = 99;
    EXPECT_EQ(mpc_cycle_validate_candidate_plan(&start.initial, p.progress, start.controls,
        p.trajectory, p.trajectory_count, p.lap_length, kStages, p.dt, &start.cfg,
        checked.get(), &metrics, &failed), MPC_CYCLE_ROLLOUT_INVALID_MODEL);
    EXPECT_EQ(failed, -1);
    EXPECT_FALSE(checked->valid);
    EXPECT_EQ(metrics.objective, 3.125);
    EXPECT_EQ(metrics.minimum_corridor_slack_m, .25f);
}

TEST(SourceSqp, PublicationMatchesIndependentValidationAndExplicitRescoring)
{
    for (float lateral : {-.1f, 20.f}) {
        UnsafeFastStart start;
        for (auto &point : start.track) point.left_bound = point.right_bound = 10.;
        start.initial.plant.e_y = lateral;
        start.cfg.sqp_iterations = 1;
        start.cfg.sqp_violation_penalty = 1.e6f;
        start.cfg.sqp_accuracy_tolerance = .05f;
        auto memory = std::make_unique<MpcCycleMemory_t>();
        auto result = std::make_unique<MpcCycleCycleResult_t>();
        const auto status = mpc_cycle_solve(&start.initial, .5, start.track, start.count,
            start.lap, .025f, kStages, &start.cfg, memory.get(), result.get());
        ASSERT_EQ(result->source_optimized_control_count, kStages);
        auto physical = start.cfg;
        physical.objective_schedule = &result->comparison_objective;
        physical.model.corridor_margin_m = physical.model.first_prediction_corridor_margin_m = 0.f;
        physical.model.nonlinear_corridor_tolerance_m = 0.f;
        const Problem p{&start.initial, .5, start.track, start.count, start.lap, .025f, &physical};
        auto checked = std::make_unique<MpcCycleNominal_t>();
        int failed = -1;
        const auto old_validation = mpc_cycle_validate_candidate_plan(&start.initial, .5,
            result->source_optimized_controls, start.track, start.count, start.lap,
            kStages, .025f, &physical, checked.get(), nullptr, &failed);
        const bool feasible = old_validation == MPC_CYCLE_ROLLOUT_OK && checked->valid;
        if (!feasible) {
            auto fallback = std::make_unique<Snapshot>();
            ASSERT_TRUE(rollout(p, result->source_optimized_controls, nullptr, fallback.get()));
            *checked = fallback->nominal;
        }
        ExplicitPlanMetrics rescored;
        explicit_plan_metrics(p, *checked, &rescored);
        const bool publish = feasible && rescored.minimum_slack >= 0.f;
        EXPECT_EQ(status, publish ? MPC_CYCLE_CYCLE_ACCEPTED_FEASIBLE
                                 : MPC_CYCLE_CYCLE_REJECTED_NONLINEAR_ROLLOUT);
        EXPECT_EQ(result->nonlinear_rollout_status, old_validation);
        EXPECT_EQ(result->nonlinear_failure_stage, failed);
        EXPECT_EQ(result->source_exact_nlp_final_physical_margin_m, rescored.minimum_slack);
        EXPECT_EQ(result->source_exact_nlp_final_source_feasible, publish);
        EXPECT_EQ(result->candidate_control_count, publish ? kStages : 0);
        if (publish) {
            for (int k = 0; k < kStages; ++k) {
                EXPECT_EQ(result->candidate_controls[k].steering_target_rad,
                          result->source_optimized_controls[k].steering_target_rad);
                EXPECT_EQ(result->candidate_controls[k].throttle_command,
                          result->source_optimized_controls[k].throttle_command);
            }
            EXPECT_EQ(result->published_steering_command, checked->controls[0].steering_target_rad);
            EXPECT_EQ(result->published_throttle, checked->controls[0].throttle_command);
        } else {
            EXPECT_FALSE(memory->warm_start_controls_valid);
            EXPECT_FALSE(memory->nominal.valid);
        }
        MpcCycleReference_t references[kStages + 1]{};
        ASSERT_TRUE(mpc_cycle_build_nominal_same_epoch(&start.initial, .5, nullptr,
            result->nominal_controls, kStages, start.track, start.count, start.lap,
            .025f, kStages, &physical.model, checked.get(), references));
        explicit_plan_metrics(p, *checked, &rescored);
        EXPECT_EQ(result->source_exact_nlp_initial_source_feasible, rescored.minimum_slack >= 0.f);
    }
}

TEST(SourceSqp, DistinctClampedRadiiMatchEveryExplicitTrial)
{
    UnsafeFastStart start;
    for (auto &point : start.track) point.left_bound = point.right_bound = 10.;
    const Problem p = start.problem();
    auto at = std::make_unique<Snapshot>();
    ASSERT_TRUE(rollout(p, start.controls, nullptr, at.get()));
    Model model;
    model.hessian = model.direction_hessian = Matrix::Identity(kControls, kControls);
    model.gradient = Vector::Zero(kControls);
    model.gradient(0) = -.3;
    model.gradient(1) = -.2;
    model.lower = Vector::Constant(kControls, -1.);
    model.upper = Vector::Constant(kControls, 1.);
    Options options;
    options.minimum_radius = .125;
    options.maximum_radius = .5;
    for (bool feasible : {true, false}) {
        for (double radius : {.125, .25, .5}) {
            const auto radii = speculative_radii(feasible, radius, options);
            std::array<Step, 4> explicit_steps, distinct_steps;
            std::array<std::unique_ptr<Snapshot>, 4> explicit_plans, distinct_plans;
            int explicit_best = -1, mapped_best = -1;
            int distinct_count = 0;
            for (int j = 0; j < options.speculative_trials; ++j) {
                // The full search computes and solves every radius in order.
                const int exponent = feasible ? -j : options.speculative_trials - 1 - 2 * j;
                const double old_radius = std::clamp(radius * std::pow(2., exponent),
                    options.minimum_radius, options.maximum_radius);
                EXPECT_EQ(radii[j].value, old_radius);
                int first = 0;
                while (first < j && radii[first].value != old_radius) ++first;
                EXPECT_EQ(radii[j].first, first);
                ASSERT_TRUE(solve_step(model, feasible, options, old_radius, &explicit_steps[j]));
                MpcModelControl_t controls[kStages];
                controls_after_step(*at, explicit_steps[j].x, 1., start.cfg.model, controls);
                explicit_plans[j] = std::make_unique<Snapshot>();
                ASSERT_TRUE(rollout(p, controls, &at->nominal, explicit_plans[j].get()));
                if (first == j) {
                    ++distinct_count;
                    ASSERT_TRUE(solve_step(model, feasible, options, radii[j].value, &distinct_steps[j]));
                    controls_after_step(*at, distinct_steps[j].x, 1., start.cfg.model, controls);
                    distinct_plans[j] = std::make_unique<Snapshot>();
                    ASSERT_TRUE(rollout(p, controls, &at->nominal, distinct_plans[j].get()));
                }
                const auto &old_step = explicit_steps[j];
                const auto &mapped_step = distinct_steps[first];
                EXPECT_EQ(mapped_step.predicted_reduction, old_step.predicted_reduction);
                EXPECT_EQ(mapped_step.model_grip_cost, old_step.model_grip_cost);
                EXPECT_EQ(mapped_step.iterations, old_step.iterations);
                EXPECT_EQ(mapped_step.rounds, old_step.rounds);
                for (int i = 0; i < kControls; ++i) EXPECT_EQ(mapped_step.x(i), old_step.x(i));
                const auto &old_plan = *explicit_plans[j];
                const auto &mapped_plan = *distinct_plans[first];
                EXPECT_EQ(mapped_plan.evaluation.objective, old_plan.evaluation.objective);
                EXPECT_EQ(mapped_plan.evaluation.violation, old_plan.evaluation.violation);
                EXPECT_EQ(mapped_plan.evaluation.feasible, old_plan.evaluation.feasible);
                for (int k = 0; k < kStages; ++k) {
                    EXPECT_EQ(mapped_plan.nominal.controls[k].steering_target_rad,
                              old_plan.nominal.controls[k].steering_target_rad);
                    EXPECT_EQ(mapped_plan.nominal.controls[k].throttle_command,
                              old_plan.nominal.controls[k].throttle_command);
                }
                // Both merits use strict comparisons, retaining the first tie.
                const double old_actual = at->evaluation.objective - old_plan.evaluation.objective -
                    (feasible ? 0. : options.violation_penalty * old_plan.evaluation.violation);
                const double mapped_actual = at->evaluation.objective - mapped_plan.evaluation.objective -
                    (feasible ? 0. : options.violation_penalty * mapped_plan.evaluation.violation);
                EXPECT_EQ(mapped_actual, old_actual);
                if (explicit_best < 0 || (feasible
                        ? old_actual > at->evaluation.objective - explicit_plans[explicit_best]->evaluation.objective
                        : old_plan.evaluation.objective < explicit_plans[explicit_best]->evaluation.objective))
                    explicit_best = j;
                if (mapped_best < 0 || (feasible
                        ? mapped_actual > at->evaluation.objective - distinct_plans[radii[mapped_best].first]->evaluation.objective
                        : mapped_plan.evaluation.objective < distinct_plans[radii[mapped_best].first]->evaluation.objective))
                    mapped_best = j;
            }
            EXPECT_EQ(mapped_best, explicit_best);
            EXPECT_EQ(radii[mapped_best].first, mapped_best);
            if (feasible && radius == options.minimum_radius) {
                EXPECT_EQ(distinct_count, 1);
                EXPECT_EQ(mapped_best, 0);
            }
            if (!feasible && radius == options.maximum_radius) EXPECT_EQ(distinct_count, 3);
        }
    }
}

TEST(SourceSqp, StageWallFeaturesMatchExplicitPerCornerArithmetic)
{
    UnsafeFastStart start;
    auto at = std::make_unique<Snapshot>();
    ASSERT_TRUE(rollout(start.problem(), start.controls, nullptr, at.get()));
    auto lin = std::make_unique<Linearization>();
    for (int k = 0; k < kStages; ++k) {
        lin->sensitivity[k].setZero();
        for (int i = 0; i < kState; ++i)
            for (int j = 0; j < 2*k+2; ++j)
                lin->sensitivity[k](i,j) = (i+j)%5 == 0 ? -0. :
                    .0013 * (i+1) * (j-k);
    }
    Model model;
    ASSERT_TRUE(build_model(start.problem(), *at, *lin, &model));
    for (int k = 0; k < kStages; ++k)
        for (int side = 0; side < 2; ++side)
            for (int corner = 0; corner < 4; ++corner) {
                const auto &g = at->evaluation.corners[k];
                const bool left = side == 0;
                const double heading = left ? g.left_d_psi[corner] : g.right_d_psi[corner];
                const double path = left ? g.left_d_progress_forward : g.right_d_progress_forward;
                const Row expected = (left ? -1. : 1.) * state_feature(*lin, k, unit(kEy)) +
                    heading * state_feature(*lin, k, unit(kEpsi)) +
                    path * state_feature(*lin, k, unit(kS));
                const auto &actual = model.physical[8*k + 4*side + corner];
                EXPECT_EQ(actual.margin, left ? g.left[corner] : g.right[corner]);
                EXPECT_EQ(actual.physical_margin, left ? g.physical_left[corner] : g.physical_right[corner]);
                for (int j = 0; j < kControls; ++j) {
                    EXPECT_EQ(actual.slope(j), expected(j));
                    EXPECT_EQ(std::signbit(actual.slope(j)), std::signbit(expected(j)));
                }
            }
}

TEST(SourceSqp, APlanThatMissesOnlyTheClearanceReserveIsPublishableAndDoesNotRestart)
{
    UnsafeFastStart start;
    MpcModelControl_t coasting[kStages];
    std::copy_n(start.controls, kStages, coasting);
    for (auto &u : coasting) u.throttle_command = MPC_THROTTLE_MIN_DRIVE;
    auto at = std::make_unique<Snapshot>();
    ASSERT_TRUE(rollout(start.problem(), coasting, nullptr, at.get()));
    ASSERT_TRUE(at->evaluation.feasible);
    // A reserve no plan can keep: the walls stay physically clear but the reserve rows are violated.
    start.cfg.model.corridor_margin_m = .9f;
    ASSERT_TRUE(rollout(start.problem(), coasting, nullptr, at.get()));
    ASSERT_FALSE(at->evaluation.feasible);
    ASSERT_TRUE(at->evaluation.physical_feasible);
    ASSERT_TRUE(at->evaluation.publishable);
    Options options;
    options.maximum_iterations = 1;
    options.deadline_us = mpc_cycle_monotonic_time_us() - 1.;
    auto result = std::make_unique<Result>();
    ASSERT_TRUE(solve(start.initial, .5, start.track, start.count, start.lap, .025f, start.cfg,
                      coasting, options, result.get()));
    EXPECT_FALSE(result->final_feasible);
    EXPECT_TRUE(result->final_publishable);
    EXPECT_FALSE(result->restarted);
    EXPECT_EQ(result->rollouts, 1);
}

TEST(SourceSqp, AKnotExactlyAtAPreviewEndpointStillHasReferenceWidthSlopes)
{
    // The track has knots at 3.1 and 3.2 and the preview half-width is 0.1: at progress
    // 3.2 the lower endpoint is the knot at 3.1 (and 3.3 the knot at 3.2 for the upper).
    // Endpoint slopes remain usable for scoring the seed plan.
    UnsafeFastStart start;
    MpcCycleSourceCorridorCorners_t corners{};
    for (double progress : {3.2, 3.3}) {
        ASSERT_TRUE(mpc_cycle_source_exact_corridor_corners(
            &start.initial.plant, start.track, start.count, start.lap, progress, &start.cfg.model, 1, &corners));
        EXPECT_TRUE(corners.reference_derivatives_valid) << progress;
    }
    // The slopes agree with the neighbouring progress values off the event.
    MpcCycleSourceCorridorCorners_t beside{};
    ASSERT_TRUE(mpc_cycle_source_exact_corridor_corners(
        &start.initial.plant, start.track, start.count, start.lap, 3.2 + 1.e-6, &start.cfg.model, 1, &beside));
    ASSERT_TRUE(mpc_cycle_source_exact_corridor_corners(
        &start.initial.plant, start.track, start.count, start.lap, 3.2, &start.cfg.model, 1, &corners));
    EXPECT_NEAR(corners.right_d_progress_forward, beside.right_d_progress_forward, 1.e-9);
    EXPECT_NEAR(corners.left_d_progress_forward, beside.left_d_progress_forward, 1.e-9);
}

TEST(SourceSqp, RestorationRepairsAnUnsafeFastWarmStart)
{
    UnsafeFastStart start;
    const Problem p = start.problem();
    auto at = std::make_unique<Snapshot>();
    ASSERT_TRUE(rollout(p, start.controls, nullptr, at.get()));
    ASSERT_FALSE(at->evaluation.physical_feasible);
    Options options;
    options.maximum_iterations = 1;
    auto result = std::make_unique<Result>();
    ASSERT_TRUE(solve(start.initial, .5, start.track, start.count, start.lap, .025f, start.cfg,
                      start.controls, options, result.get()));
    ASSERT_TRUE(rollout(p, result->optimal.controls, nullptr, at.get()));
    EXPECT_TRUE(at->evaluation.physical_feasible);
    for (const auto &u : result->optimal.controls)
        EXPECT_GE(u.throttle_command, MPC_THROTTLE_MIN_DRIVE);
}

TEST(SourceSqp, AnIterationNoCappedTrialSurvivesIsRepeatedWithTheExactSubstepSchedule)
{
    // A coasting plan at the throttle floor: the one-substep derivatives misdescribe it, so
    // every trial of the first iteration is rejected and requires exact derivatives.
    UnsafeFastStart start;
    start.initial.plant.u = 3.f;
    ASSERT_TRUE(vehicle_model_initialize_plant_state(&start.initial.plant));
    for (auto &u : start.controls) u = {.2f, MPC_THROTTLE_MIN_DRIVE};
    start.cfg.derivative_force_substeps = 1;
    Options options;
    options.maximum_iterations = 4;
    auto result = std::make_unique<Result>();
    ASSERT_TRUE(solve(start.initial, .5, start.track, start.count, start.lap, .025f, start.cfg,
                      start.controls, options, result.get()));
    ASSERT_EQ(result->history.size(), 4u);
    EXPECT_FALSE(result->history[0].accepted);
    EXPECT_EQ(result->accepted_iterations, 3);
    EXPECT_LT(result->final_cost, result->initial_cost);
    EXPECT_TRUE(result->exact_derivatives);

    // The next solve of such a plan starts with exact derivatives: no iteration is wasted.
    options.exact_derivatives = true;
    auto next = std::make_unique<Result>();
    ASSERT_TRUE(solve(start.initial, .5, start.track, start.count, start.lap, .025f, start.cfg,
                      start.controls, options, next.get()));
    ASSERT_EQ(next->history.size(), 4u);
    EXPECT_EQ(next->accepted_iterations, 4);
    EXPECT_FALSE(next->exact_derivatives);
}

TEST(SourceSqp, RestorationAcceptsItsBestMeritDecreaseInTheFirstRoundWhenNoTrialIsFeasible)
{
    // Wall rows are nonlinear in the controls, so a restoration step often cannot reach
    // feasibility at once. Waiting for a feasible trial costs a whole iteration of rounds
    // (a corrected model, then a smaller radius); the best verified decrease of the
    // merit is taken immediately and the next iteration continues from it.
    UnsafeFastStart start;
    start.initial.plant.u = 6.f;
    start.initial.plant.e_y = -.3f;
    ASSERT_TRUE(vehicle_model_initialize_plant_state(&start.initial.plant));
    for (auto &u : start.controls) u = {0.f, .5f};
    Options options;
    options.maximum_iterations = 5;
    auto result = std::make_unique<Result>();
    ASSERT_TRUE(solve(start.initial, .5, start.track, start.count, start.lap, .025f, start.cfg,
                      start.controls, options, result.get()));
    ASSERT_FALSE(result->initial_feasible);
    ASSERT_GE(result->history.size(), 3u);
    ASSERT_FALSE(result->history[1].feasible_before);  // the first step left the plan infeasible
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(result->history[i].accepted);
        EXPECT_EQ(result->history[i].attempts, 1);
    }
    EXPECT_TRUE(result->final_feasible);
}

TEST(SourceSqp, TrackingWeightsFollowTheReferenceCurvatureOnlyWhenGainsAreSet)
{
    MpcSolverConfiguration_t model{};
    model.weight_e_y = 4000.f;
    model.weight_e_psi = 1500.f;
    MpcCycleReference_t straight{}, hairpin{};
    hairpin.path_curvature = -1.f;  // the sign of the turn does not matter
    // Zero gains preserve the uniform weights on any path.
    EXPECT_FLOAT_EQ(lateral_weight_for_reference(&model, &hairpin), 4000.f);
    EXPECT_FLOAT_EQ(heading_weight_for_reference(&model, &hairpin), 1500.f);
    model.lateral_curvature_gain_m = 0.5f;
    model.heading_curvature_gain_m = 0.5f;
    EXPECT_FLOAT_EQ(lateral_weight_for_reference(&model, &straight), 4000.f);
    EXPECT_FLOAT_EQ(lateral_weight_for_reference(&model, &hairpin), 6000.f);  // 4000 * (1 + 0.5)
    EXPECT_FLOAT_EQ(heading_weight_for_reference(&model, &straight), 1500.f);
    EXPECT_FLOAT_EQ(heading_weight_for_reference(&model, &hairpin), 2250.f);  // 1500 * (1 + 0.5)
}

TEST(SourceSqp, ARolloutPastItsDeadlineIsAbandoned)
{
    UnsafeFastStart start;
    Problem p = start.problem();
    auto at = std::make_unique<Snapshot>();
    ASSERT_TRUE(rollout(p, start.controls, nullptr, at.get()));
    p.deadline_us = mpc_cycle_monotonic_time_us() - 1.;
    EXPECT_FALSE(rollout(p, start.controls, nullptr, at.get()));
    EXPECT_FALSE(at->valid);
}

TEST(SourceSqp, ASolveWhoseDeadlinePassedKeepsAFeasibleSeedAndStartsNoTrial)
{
    // A feasible seed is kept whole: no trial starts and no restart is needed.
    UnsafeFastStart start;
    auto at = std::make_unique<Snapshot>();
    const Problem p = start.problem();
    MpcModelControl_t coasting[kStages];
    std::copy_n(start.controls, kStages, coasting);
    for (auto &u : coasting) u.throttle_command = MPC_THROTTLE_MIN_DRIVE;
    ASSERT_TRUE(rollout(p, coasting, nullptr, at.get()));
    ASSERT_TRUE(at->evaluation.feasible);
    Options options;
    options.maximum_iterations = 4;
    options.deadline_us = mpc_cycle_monotonic_time_us() - 1.;
    auto result = std::make_unique<Result>();
    ASSERT_TRUE(solve(start.initial, .5, start.track, start.count, start.lap, .025f, start.cfg,
                      coasting, options, result.get()));
    EXPECT_EQ(result->termination, Termination::Deadline);
    EXPECT_EQ(result->history.size(), 0u);
    EXPECT_FALSE(result->restarted);
    EXPECT_EQ(result->rollouts, 1);  // the seed's own integration; no trial was started
}

TEST(SourceSqp, AnInfeasibleSearchRestartsFromACoastingPlanEvenOutOfTime)
{
    UnsafeFastStart start;
    auto at = std::make_unique<Snapshot>();
    const Problem p = start.problem();
    MpcModelControl_t coasting[kStages];
    std::copy_n(start.controls, kStages, coasting);
    for (auto &u : coasting) u.throttle_command = MPC_THROTTLE_MIN_DRIVE;
    ASSERT_TRUE(rollout(p, start.controls, nullptr, at.get()));
    ASSERT_FALSE(at->evaluation.feasible);
    ASSERT_TRUE(rollout(p, coasting, nullptr, at.get()));
    ASSERT_TRUE(at->evaluation.feasible);
    Options options;
    options.maximum_iterations = 4;
    options.deadline_us = mpc_cycle_monotonic_time_us() - 1.;
    auto result = std::make_unique<Result>();
    ASSERT_TRUE(solve(start.initial, .5, start.track, start.count, start.lap, .025f, start.cfg,
                      start.controls, options, result.get()));
    EXPECT_TRUE(result->restarted);
    EXPECT_FALSE(result->initial_feasible);
    EXPECT_TRUE(result->final_feasible);
    EXPECT_EQ(result->rollouts, 2);  // both seeds are integrated; no trial was started
    for (const auto &u : result->optimal.controls) EXPECT_EQ(u.throttle_command, MPC_THROTTLE_MIN_DRIVE);
}

TEST(SourceSqp, RestorationRetainsTrackingAndDiscoversSlipCrossings)
{
    Model model;
    model.hessian = model.direction_hessian = Matrix::Identity(kControls, kControls);
    model.gradient = Vector::Zero(kControls);
    model.gradient(1) = -.8;
    model.lower = Vector::Constant(kControls, -1.);
    model.upper = Vector::Constant(kControls, 1.);
    PhysicalRow wall{};
    wall.margin = -.1;
    wall.slope = Row::Zero();
    wall.slope(0) = 1.;
    model.physical.push_back(wall);
    GripRow grip{};
    grip.residual = -.3; // Initially omitted, crossed by the tracking-only step.
    grip.weight = 1000.;
    grip.slope = Row::Zero();
    grip.slope(1) = 1.;
    model.grip.push_back(grip);
    Options options;
    Step step;
    ASSERT_TRUE(solve_step(model, false, options, 1., &step));
    EXPECT_LT(linearized_violation(model, step.x), 1.e-6);
    EXPECT_GE(step.rounds, 2);
    // Minimize x^2/2 - .8*x + 1000*max(0, x-.3)^2.
    EXPECT_NEAR(step.x(1), (.8 + 2000. * .3) / 2001., 1.e-4);
}

bool explicit_tracking_second_order(const MpcTrajectorySample_t *points, size_t count,
    double length, const double q[6], double seed, double jacobian[3][6], double hessian[3][6][6])
{
    const auto wrap = [](double angle) {
        constexpr double pi = 3.14159265358979323846;
        return fabs(angle) <= pi ? angle : remainder(angle, 2.0 * pi);
    };
    const auto frame_at = [&](double progress, MpcPathFrame_t *frame) {
        if (!mpc_trajectory_sample(points, count, length, progress, &frame->sample)) return false;
        double wrapped = fmod(progress-points[0].s, length);
        if (wrapped < 0.) wrapped += length;
        wrapped += points[0].s;
        size_t lower = count-1, upper = 0;
        if (wrapped < points[count-1].s) {
            lower = 0; upper = count-1;
            while (upper-lower > 1) {
                const size_t mid = lower+(upper-lower)/2;
                if (points[mid].s <= wrapped) lower = mid;
                else upper = mid;
            }
        }
        const auto &a = points[lower], &b = points[upper];
        const double span = (upper == 0 ? b.s+length : b.s)-a.s;
        const double heading = upper == 0 ? a.heading+wrap(b.heading-a.heading) : b.heading;
        frame->dx_ds = (b.x-a.x)/span;
        frame->dy_ds = (b.y-a.y)/span;
        frame->dheading_ds = (heading-a.heading)/span;
        return true;
    };
    const double ey=q[0], epsi=q[1], dx=q[2], dy=q[3], progress=q[5];
    MpcPathFrame_t start_frame{}, end_frame{};
    if (!frame_at(progress, &start_frame)) return false;
    const auto start = start_frame.sample;
    const double c0=cos(start.heading), s0=sin(start.heading);
    const double yaw=start.heading+epsi, cb=cos(yaw), sb=sin(yaw);
    const double x=start.x-s0*ey+cb*dx-sb*dy;
    const double y=start.y+c0*ey+sb*dx+cb*dy;
    double station=seed, next_ey=0.;
    bool projected=false;
    for (int iteration=0; iteration<8; ++iteration) {
        if (!frame_at(station, &end_frame)) return false;
        const auto &sample=end_frame.sample;
        const double c=cos(sample.heading), s=sin(sample.heading);
        const double ex=x-sample.x, ey_world=y-sample.y;
        const double tangent_error=c*ex+s*ey_world;
        next_ey=-s*ex+c*ey_world;
        const double tangent_scale=c*end_frame.dx_ds+s*end_frame.dy_ds-end_frame.dheading_ds*next_ey;
        if (!(tangent_scale > .05) || !std::isfinite(tangent_error)) return false;
        if (fabs(tangent_error) <= 1.e-8) { projected=true; break; }
        station += fmax(-.5, fmin(.5, tangent_error/tangent_scale));
    }
    if (!projected) return false;
    {
        const double c=cos(end_frame.sample.heading), s=sin(end_frame.sample.heading);
        const double a=c*end_frame.dx_ds+s*end_frame.dy_ds-end_frame.dheading_ds*next_ey;
        const double b=-s*end_frame.dx_ds+c*end_frame.dy_ds;
        const double h0=start_frame.dheading_ds;
        const double px[6]={-s0,-sb*dx-cb*dy,cb,-sb,0.,
            start_frame.dx_ds-c0*ey*h0+(-sb*dx-cb*dy)*h0};
        const double py[6]={c0,cb*dx-sb*dy,sb,cb,0.,
            start_frame.dy_ds-s0*ey*h0+(cb*dx-sb*dy)*h0};
        for (int j=0;j<6;++j) {
            const double ds=(c*px[j]+s*py[j])/a;
            jacobian[0][j]=-s*px[j]+c*py[j]-b*ds;
            jacobian[1][j]=(j==5?h0:(j==1||j==4?1.:0.))-end_frame.dheading_ds*ds;
            jacobian[2][j]=ds;
        }
    }
    // Curvature evaluates both loaded frames and their trigonometry explicitly.
    if (!frame_at(progress, &start_frame) || !frame_at(station, &end_frame)) return false;
    const double hc0=cos(start_frame.sample.heading), hs0=sin(start_frame.sample.heading);
    const double hyaw=start_frame.sample.heading+epsi, hcb=cos(hyaw), hsb=sin(hyaw);
    const double c=cos(end_frame.sample.heading), s=sin(end_frame.sample.heading);
    const double h0=start_frame.dheading_ds, h1=end_frame.dheading_ds;
    const double a=c*end_frame.dx_ds+s*end_frame.dy_ds-h1*next_ey;
    const double b=-s*end_frame.dx_ds+c*end_frame.dy_ds;
    for (int i=0;i<6;++i) for (int j=0;j<6;++j) {
        const double hi=i==5?h0:0., hj=j==5?h0:0.;
        const double wi=i==1?1.:hi, wj=j==1?1.:hj;
        const double dix=i==2?1.:0., diy=i==3?1.:0.;
        const double djx=j==2?1.:0., djy=j==3?1.:0.;
        const double mixed=(i==0?hj:0.)+(j==0?hi:0.);
        double px=-hc0*mixed+ey*hs0*hi*hj-(hcb*dx-hsb*dy)*wi*wj+
            (-hsb*djx-hcb*djy)*wi+(-hsb*dix-hcb*diy)*wj;
        double py=-hs0*mixed-ey*hc0*hi*hj-(hsb*dx+hcb*dy)*wi*wj+
            (hcb*djx-hsb*djy)*wi+(hcb*dix-hsb*diy)*wj;
        const double mixed_end=jacobian[0][i]*jacobian[2][j]+jacobian[0][j]*jacobian[2][i];
        const double pure_end=next_ey*h1*h1*jacobian[2][i]*jacobian[2][j];
        px+=h1*c*mixed_end-s*pure_end;
        py+=h1*s*mixed_end+c*pure_end;
        const double ds=(c*px+s*py)/a;
        hessian[0][i][j]=-s*px+c*py-b*ds;
        hessian[1][i][j]=-h1*ds;
        hessian[2][i][j]=ds;
    }
    return true;
}

TEST(SourceSqp, TrackingCurvatureMatchesExplicitDoubleFrameArithmetic)
{
    MpcTrajectorySample_t points[4]{};
    for (int i=0;i<4;++i) {
        points[i].s=i; points[i].speed=6.; points[i].left_bound=points[i].right_bound=1.;
    }
    constexpr double quarter_pi = .78539816339744830962;
    points[0].heading=-quarter_pi;
    points[1].x=1.; points[1].heading=quarter_pi;
    points[2].x=1.; points[2].y=1.; points[2].heading=3.*quarter_pi;
    points[3].y=1.; points[3].heading=5.*quarter_pi;
    for (double progress : {-.03, 0., .3, .97, 1.03, 2.05, 3.92, 4.3})
        for (double displacement : {0., .03}) {
            const double q[6]={.05123456789,.10123456789,displacement,.00123456789,.02,progress};
            double actual_jac[3][6], expected_jac[3][6], actual_hess[3][6][6], expected_hess[3][6][6];
            ASSERT_TRUE(explicit_tracking_second_order(points,4,4.,q,progress+displacement,
                expected_jac,expected_hess));
            ASSERT_TRUE(mpc_trajectory_tracking_second_order(points,4,4.,q[5],q[0],q[1],q[2],q[3],q[4],
                progress+displacement,actual_jac,actual_hess));
            for (int row=0;row<3;++row) for (int i=0;i<6;++i) {
                EXPECT_EQ(actual_jac[row][i],expected_jac[row][i]);
                EXPECT_EQ(std::signbit(actual_jac[row][i]),std::signbit(expected_jac[row][i]));
                for (int j=0;j<6;++j) {
                    EXPECT_EQ(actual_hess[row][i][j],expected_hess[row][i][j]);
                    EXPECT_EQ(std::signbit(actual_hess[row][i][j]),std::signbit(expected_hess[row][i][j]));
                }
            }
        }
}

TEST(SourceSqp, AnalyticTrackingHessianMatchesJacobianDifferencesWithinSegment)
{
    MpcTrajectorySample_t points[4]{};
    for(int i=0;i<4;++i) {
        points[i].s=i;points[i].speed=6.;points[i].left_bound=points[i].right_bound=1.;
    }
    points[1].x=1.;points[1].y=.1;points[1].heading=.2;
    points[2].x=1.;points[2].y=1.1;points[2].heading=1.5;
    points[3].y=1.;points[3].heading=3.;
    double q[6]={.05,.1,.03,.01,.02,.3},jac[3][6],hess[3][6][6];
    // Seed at the nominal inverse: finite differences then resolve the
    // Hessian without crossing the Newton stopping boundary.
    double seed;float ey,epsi;
    ASSERT_TRUE(mpc_trajectory_advance_tracking(points,4,4.,q[5],q[0],q[1],q[2],q[3],q[4],.33,&seed,&ey,&epsi,nullptr));
    ASSERT_TRUE(mpc_trajectory_tracking_second_order(points,4,4.,q[5],q[0],q[1],q[2],q[3],q[4],seed,jac,hess));
    for(int i=0;i<6;++i) {
        constexpr double epsilon=1.e-4;
        double plus[3][6],minus[3][6],unused[3][6][6];
        q[i]+=epsilon;
        ASSERT_TRUE(mpc_trajectory_tracking_second_order(points,4,4.,q[5],q[0],q[1],q[2],q[3],q[4],seed,plus,unused));
        q[i]-=2.*epsilon;
        ASSERT_TRUE(mpc_trajectory_tracking_second_order(points,4,4.,q[5],q[0],q[1],q[2],q[3],q[4],seed,minus,unused));
        q[i]+=epsilon;
        for(int m=0;m<3;++m)for(int j=0;j<6;++j) {
            EXPECT_NEAR(hess[m][j][i],(plus[m][j]-minus[m][j])/(2.*epsilon),1.e-7);
            EXPECT_NEAR(hess[m][i][j],hess[m][j][i],1.e-12);
        }
    }
}

TEST(SourceSqp, CommandQuadraticMatchesExactRolledOutCost)
{
    std::ifstream csv(MPC_TEST_RACELINE);
    ASSERT_TRUE(csv.good());
    double length = 0.;
    const auto trajectory = f1tenth_mpc::read_mpc_controller_trajectory(csv, &length);
    MpcCycleConfiguration_t cfg{};
    auto &m = cfg.model;
    m.weight_steering_target = 5.f;
    m.weight_throttle_command = .7f;
    m.weight_steering_command = 4.5f;
    m.weight_steering_slew = 1500.f;
    m.weight_throttle_slew = 50.f;
    m.terminal_multiplier = 3.f;
    m.max_speed_mps = m.active_speed_ceiling_mps = 16.f;
    m.max_steering_rad = .5235987756f;
    m.jacobian_threads = 1;
    MpcCycleObjectiveSchedule_t schedule{};
    schedule.valid = 1;
    schedule.horizon = kStages;
    for (int k = 0; k <= kStages; ++k)
        schedule.references[k].steering_command = .02f * std::sin(.4f * k);
    cfg.objective_schedule = &schedule;
    MpcCycleState_t initial{};
    initial.plant.u = 3.f;
    initial.plant.actual_steering_angle = .01f;
    initial.plant.delayed_steering_command_2 = .01f;
    initial.plant.delayed_throttle_command_2 = .1f;
    ASSERT_TRUE(vehicle_model_initialize_plant_state(&initial.plant));
    MpcModelControl_t controls[kStages];
    for (auto &u : controls) u = {.01f, .1f};
    const Problem p{&initial,5.,trajectory.data(),trajectory.size(),length,.025f,&cfg};
    auto at = std::make_unique<Snapshot>(), trial = std::make_unique<Snapshot>();
    auto lin = std::make_unique<Linearization>();
    Model model;
    ASSERT_TRUE(rollout(p,controls,nullptr,at.get()));
    ASSERT_EQ(linearize(p,at->nominal,1,0.,all_stages(),lin.get(),nullptr),LinearizeStatus::Ok);
    ASSERT_TRUE(build_model(p,*at,*lin,&model));
    // Small rack targets are reached within each stage. All nonzero costs
    // are therefore exactly quadratic in the commands, even though the full
    // nonlinear source plant (including the tires) is rolled out each time.
    for (int sample = 0; sample < 6; ++sample) {
        Vector requested = Vector::Zero(kControls);
        if (sample < 4) requested(sample < 2 ? sample : kControls-4+sample) = .02;
        else for (int j = 0; j < kControls; ++j)
            requested(j) = (sample == 4 ? 1. : -1.) * .02 * std::sin(.7*j+.3);
        MpcModelControl_t changed[kStages];
        controls_after_step(*at,requested,1.,m,changed);
        ASSERT_TRUE(rollout(p,changed,&at->nominal,trial.get()));
        Vector applied(kControls);
        for (int k = 0; k < kStages; ++k) {
            applied(2*k) = (double(changed[k].steering_target_rad)-controls[k].steering_target_rad)/kSteeringScale;
            applied(2*k+1) = (double(changed[k].throttle_command)-controls[k].throttle_command)/kThrottleScale;
        }
        const double predicted = model.gradient.dot(applied)+.5*applied.dot(model.hessian*applied);
        EXPECT_NEAR(predicted,trial->evaluation.objective-at->evaluation.objective,
                    3.e-6*std::max(1.,at->evaluation.objective)) << "sample " << sample;
    }
}

TEST(SourceSqp, HairpinRestorationCannotTradeTinyReserveGainForLargeCostLoss)
{
    // With no sensor gap or deferral, restoration bounds racing-cost loss
    // as well as improving reserve slack.
    std::ifstream csv(MPC_TEST_RACELINE);
    ASSERT_TRUE(csv.good());
    double length{};
    const auto trajectory = f1tenth_mpc::read_mpc_controller_trajectory(csv, &length);
    auto cfg = captured_practice_configuration();
    MpcCycleState_t initial{};
    initial.plant.e_y = -0.04899672046303749;
    initial.plant.e_psi = 0.0029965874273329973;
    initial.plant.u = 3.6863181591033936;
    initial.plant.v = 0.10190234333276749;
    initial.plant.r = -2.952548027038574;
    initial.plant.delayed_steering_command_1 = -0.2764601707458496;
    initial.plant.delayed_steering_command_2 = -0.2497566193342209;
    initial.plant.actual_steering_angle = -0.2497566193342209;
    initial.plant.delayed_throttle_command_1 = 0.16599999368190765;
    initial.plant.delayed_throttle_command_2 = 0.1899999976158142;
    const double internal[MPC_PLANT_INTERNAL_STATE_COUNT] = {
        79.86048004503537, 81.06899558336475, 79.2702127684859, 80.65113347444711,
        0.004135450427087444, -0.00030334158310220573, -0.09166000067425639, 0.08085277691934786,
        0.025806772919601255, 0.1829618200748103, 0.00999230385273836, -0.011299892688622748,
        0.011537576209936035, -0.010155181565608823, 0, 0,
        0, 0, 0, 0,
        0, 0,
    };
    std::copy_n(internal, MPC_PLANT_INTERNAL_STATE_COUNT, initial.plant.source_plant_internal_state);
    initial.plant.source_plant_initialized = 1;
    initial.plant.source_suspension_history_initialized = 1;
    initial.plant.source_imu_state_initialized = 1;
    const MpcModelControl_t controls[kStages] = {
        {-0.23301935195922852f, 0.1860942393541336f},
        {-0.22255651652812958f, 0.17906764149665833f},
        {-0.21165189146995544f, 0.17485247552394867f},
        {-0.2012857049703598f, 0.17365625500679016f},
        {-0.19819758832454681f, 0.17415878176689148f},
        {-0.18261286616325378f, 0.17574574053287506f},
        {-0.1764868199825287f, 0.1783457100391388f},
        {-0.1757206916809082f, 0.1800304651260376f},
        {-0.16184991598129272f, 0.18051733076572418f},
        {-0.14836527407169342f, 0.22524447739124298f},
        {-0.13609939813613892f, 0.18423700332641602f},
        {-0.12511402368545532f, 0.20122221112251282f},
        {-0.12032046169042587f, 0.24355845153331757f},
        {-0.11924672871828079f, 0.2108941674232483f},
        {-0.1265202909708023f, 0.2068987339735031f},
        {-0.1131775975227356f, 0.2170969843864441f},
        {-0.08709847927093506f, 0.23473796248435974f},
        {-0.09235640615224838f, 0.2297801822423935f},
        {-0.0872340202331543f, 0.2374732941389084f},
        {-0.08058512210845947f, 0.25148889422416687f},
        {-0.05925091356039047f, 0.26523950695991516f},
        {-0.05323132872581482f, 0.27267706394195557f},
        {-0.047460149973630905f, 0.2594253420829773f},
        {-0.04257158562541008f, 0.2590855062007904f},
        {-0.0393463559448719f, 0.2590855062007904f},
        {-0.035527512431144714f, 0.28467246890068054f},
        {-0.031079672276973724f, 0.28467246890068054f},
        {-0.029352281242609024f, 0.28467246890068054f},
        {-0.027253836393356323f, 0.2759811580181122f},
        {-0.03062419965863228f, 0.2759811580181122f},
    };
    const double progress = 120.56574983928854;
    auto nominal = std::make_unique<MpcCycleNominal_t>();
    MpcCycleReference_t refs[kStages + 1]{};
    ASSERT_TRUE(mpc_cycle_build_nominal_same_epoch(&initial, progress, nullptr,
        controls, kStages, trajectory.data(), trajectory.size(), length, .025f,
        kStages, &cfg.model, nominal.get(), refs));
    MpcCycleObjectiveSchedule_t schedule{};
    ASSERT_TRUE(mpc_cycle_build_objective_schedule(nominal.get(), refs, &cfg.model, &schedule));
    cfg.objective_schedule = &schedule;
    Options options;
    options.maximum_iterations = 1;
    options.radius = 0.015592433425025215;
    auto result = std::make_unique<Result>();
    ASSERT_TRUE(solve(initial, progress, trajectory.data(), trajectory.size(),
        length, .025f, cfg, controls, options, result.get()));
    // The recorded problem is the 30-stage one; other horizons change both.
    if (PREDICTION_HORIZON == 30) {
        EXPECT_NEAR(result->initial_cost, 165553.00962062061, 1.e-6);
        EXPECT_NEAR(result->initial_violation, .0023892149329185486, 1.e-9);
    }
    const double initial_merit = result->initial_cost + options.violation_penalty * result->initial_violation;
    const double final_merit = result->final_cost + options.violation_penalty * result->final_violation;
    EXPECT_LE(final_merit, initial_merit);
    const Problem problem{&initial, progress, trajectory.data(), trajectory.size(), length, .025f, &cfg};
    auto final = std::make_unique<Snapshot>();
    ASSERT_TRUE(rollout(problem, result->optimal.controls, nullptr, final.get()));
    EXPECT_TRUE(final->evaluation.physical_feasible);
}

TEST(MpcCycleCorridor, FarStageToleranceGrowsLinearlyOverTheLateStagesOnly)
{
    MpcSolverConfiguration_t model{};
    model.nonlinear_corridor_tolerance_m = 0.005f;
    // Without a far tolerance every stage keeps the constant tolerance.
    for (int stage = 1; stage <= PREDICTION_HORIZON; ++stage)
        EXPECT_EQ(corridor_tolerance_at_prediction(&model, stage), 0.005f);
    model.far_stage_corridor_tolerance_m = 0.06f;
    model.far_stage_tolerance_start_stage = 10;
    EXPECT_EQ(corridor_tolerance_at_prediction(&model, 10), 0.005f);
    EXPECT_NEAR(corridor_tolerance_at_prediction(&model, 20),
                0.005f + 0.06f * 10.f / (PREDICTION_HORIZON - 10), 1e-7);
    EXPECT_NEAR(corridor_tolerance_at_prediction(&model, PREDICTION_HORIZON), 0.005f + 0.06f, 1e-7);
    // The slack of a state exactly on the wall gains the stage's tolerance.
    MpcModelState_t state{};
    float lower = NAN, upper = NAN, near_lower = NAN, near_upper = NAN;
    ASSERT_TRUE(mpc_cycle_compute_corridor_slack(&state, 0.4f, -0.4f, &model, PREDICTION_HORIZON,
                                               &lower, &upper));
    model.far_stage_corridor_tolerance_m = 0.f;
    ASSERT_TRUE(mpc_cycle_compute_corridor_slack(&state, 0.4f, -0.4f, &model, PREDICTION_HORIZON,
                                               &near_lower, &near_upper));
    EXPECT_NEAR(lower - near_lower, 0.06f, 1e-6);
    EXPECT_NEAR(upper - near_upper, 0.06f, 1e-6);
}

TEST(MpcCycleCorridor, AReserveNoLateralPositionFitsGivesNegativeSlackNotAFailedQuery)
{
    MpcSolverConfiguration_t model{};
    model.corridor_margin_m = model.first_prediction_corridor_margin_m = 0.05f;
    MpcModelState_t state{};
    float lower_slack = NAN, upper_slack = NAN;
    // A 0.2 m wide corridor is narrower than the 0.273 m car plus the reserve.
    ASSERT_TRUE(mpc_cycle_compute_corridor_slack(&state, 0.1f, -0.1f, &model, 1,
                                               &lower_slack, &upper_slack));
    const float footprint = 0.05f + 0.5f * MPC_CAR_WIDTH_M;
    EXPECT_NEAR(lower_slack, -(footprint + 0.1f), 1e-6);
    EXPECT_NEAR(upper_slack, 0.1f - footprint, 1e-6);
    EXPECT_LT(lower_slack, 0.f);
    EXPECT_LT(upper_slack, 0.f);
}

TEST(SourceSqp, RestoresTheSpeedEnvelopeUsedByIndependentPublicationValidation)
{
    if (PREDICTION_HORIZON != 30) GTEST_SKIP() << "captured at the 30-stage horizon";
    // Wall clearance alone is insufficient; the SQP must
    // enforce the speed bounds checked by independent publication validation.
    std::ifstream csv(MPC_TEST_RACELINE);
    ASSERT_TRUE(csv.good());
    double length{};
    const auto trajectory = f1tenth_mpc::read_mpc_controller_trajectory(csv, &length);
    auto cfg = captured_practice_configuration();
    MpcCycleState_t initial{};
    initial.plant.e_y = -0.029934357851743698;
    initial.plant.e_psi = 0.01897285133600235;
    initial.plant.u = 4.11430025100708;
    initial.plant.v = 0.1771426498889923;
    initial.plant.r = -2.251274824142456;
    initial.plant.delayed_steering_command_1 = -0.15707963705062866;
    initial.plant.delayed_steering_command_2 = -0.14608405530452728;
    initial.plant.actual_steering_angle = -0.14608405530452728;
    initial.plant.delayed_throttle_command_1 = 0.1809999942779541;
    initial.plant.delayed_throttle_command_2 = 0.18799999356269836;
    const double internal[MPC_PLANT_INTERNAL_STATE_COUNT] = {
        79.38535733837664, 80.13605752852354, 79.02971121919107, 79.69724180493628,
        0.00408838270279509, -3.268673364771974e-05, -0.09442806564868911, 0.004971244619076623,
        0.018130765261419785, 0.17842057158762217, 0.009049633139153579, -0.012887456136188746,
        0.013147969906096957, -0.009201812702103768, 0, 0,
        0, 0, 0, 0,
        0, 0,
    };
    std::copy_n(internal, MPC_PLANT_INTERNAL_STATE_COUNT, initial.plant.source_plant_internal_state);
    initial.plant.source_plant_initialized = 1;
    initial.plant.source_suspension_history_initialized = 1;
    initial.plant.source_imu_state_initialized = 1;
    const MpcModelControl_t controls[kStages] = {
        {-0.13432523608207703f, 0.19716134667396545f},
        {-0.12623660266399384f, 0.20441433787345886f},
        {-0.12032667547464371f, 0.21244840323925018f},
        {-0.12041174620389938f, 0.21491192281246185f},
        {-0.11487662047147751f, 0.22510211169719696f},
        {-0.1028035432100296f, 0.22714440524578094f},
        {-0.09424315392971039f, 0.23368382453918457f},
        {-0.07888753712177277f, 0.23820647597312927f},
        {-0.06847059726715088f, 0.24754759669303894f},
        {-0.056756533682346344f, 0.2569026052951813f},
        {-0.04596450552344322f, 0.26345890760421753f},
        {-0.034585077315568924f, 0.2703455686569214f},
        {-0.02179543487727642f, 0.27627745270729065f},
        {-0.001035958295688033f, 0.004808655474334955f},
        {0.018242431804537773f, 0.004808654543012381f},
        {0.03379671275615692f, 0.004808654543012381f},
        {0.04642948880791664f, 0.004808654543012381f},
        {0.05403219535946846f, 0.004808654077351093f},
        {0.060637954622507095f, 0.004808654077351093f},
        {0.04762372002005577f, 0.004808654077351093f},
        {0.06122521683573723f, 0.004808654077351093f},
        {0.05882633849978447f, 0.004808654077351093f},
        {0.06002708524465561f, 0.004808654077351093f},
        {0.05936800688505173f, 0.004808653146028519f},
        {0.04278611019253731f, 0.004808654543012381f},
        {0.08239439129829407f, 0.004808653611689806f},
        {0.06444311887025833f, 0.004808654543012381f},
        {0.06248227134346962f, 0.004678841680288315f},
        {0.05499512329697609f, 0.0044248392805457115f},
        {0.05619767680764198f, 0.0044248392805457115f},
    };
    const double progress = 108.89104917929455;
    auto nominal = std::make_unique<MpcCycleNominal_t>();
    MpcCycleReference_t refs[kStages + 1]{};
    ASSERT_TRUE(mpc_cycle_build_nominal_same_epoch(&initial, progress, nullptr,
        controls, kStages, trajectory.data(), trajectory.size(), length, .025f,
        kStages, &cfg.model, nominal.get(), refs));
    MpcCycleObjectiveSchedule_t schedule{};
    ASSERT_TRUE(mpc_cycle_build_objective_schedule(nominal.get(), refs, &cfg.model, &schedule));
    cfg.objective_schedule = &schedule;
    const Problem problem{&initial, progress, trajectory.data(), trajectory.size(), length, .025f, &cfg};
    auto at = std::make_unique<Snapshot>();
    ASSERT_TRUE(rollout(problem, controls, nullptr, at.get()));
    EXPECT_GT(at->evaluation.minimum_physical, 0.);
    EXPECT_LT(at->evaluation.minimum_speed, 0.);
    EXPECT_FALSE(at->evaluation.physical_feasible);
    Options options;
    options.maximum_iterations = 8;
    auto result = std::make_unique<Result>();
    ASSERT_TRUE(solve(initial, progress, trajectory.data(), trajectory.size(),
        length, .025f, cfg, controls, options, result.get()));
    ASSERT_TRUE(rollout(problem, result->optimal.controls, nullptr, at.get()));
    ASSERT_TRUE(at->evaluation.physical_feasible);
    auto physical = cfg;
    physical.model.corridor_margin_m = physical.model.first_prediction_corridor_margin_m = 0.f;
    physical.model.nonlinear_corridor_tolerance_m = 0.f;
    int failed;
    EXPECT_EQ(mpc_cycle_validate_candidate_plan(&initial, progress, result->optimal.controls,
        trajectory.data(), trajectory.size(), length, kStages, .025f, &physical,
        nominal.get(), nullptr, &failed), MPC_CYCLE_ROLLOUT_OK) << "stage " << failed;
}
} // namespace sdu_apex::source_sqp

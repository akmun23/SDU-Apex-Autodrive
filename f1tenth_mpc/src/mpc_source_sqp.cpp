#include "mpc_source_sqp.hpp"
#include "mpc_source_qp.hpp"
#include "mpc_cycle_internal.hpp"
#include "vehicle_model.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sdu_apex::source_sqp {
namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Tangent = Eigen::Matrix<double, kState, kControls, Eigen::RowMajor>;
using Core = Eigen::Matrix<double, kCore, 1>;
using Full = Eigen::Matrix<double, kState, 1>;
using Expansion = Eigen::Matrix<double, kExpansion, kExpansion>;

constexpr double kRearAxleToCom = 0.15532;
constexpr double kBoxTolerance = 1.e-7;
constexpr double kScale[2] = {kSteeringScale, kThrottleScale};

double elapsed_ms(std::chrono::steady_clock::time_point since)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - since).count();
}

// ---------------------------------------------------------------------------
// Exact evaluation of one control sequence.
// ---------------------------------------------------------------------------
struct Evaluation {
    double objective{};
    MpcCycleGripMetrics_t grip{};
    MpcCycleSourceCorridorCorners_t corners[kStages]{};
    double box[kStages][4]{};
    double minimum_corner{};
    double minimum_physical{};
    double minimum_box{};
    double speed_margin[kStages][2]{};
    double minimum_speed{};
    // L1 violation of the wall clearance and publication speed rows.
    double violation{};
    bool feasible{};
    bool physical_feasible{};
    // The independent publication check passes: physical rows with the far-stage
    // tolerance, speed and command box (the clearance reserve is not required).
    bool publishable{};
    // Why the evaluation was refused (SeedFailure; 0 = none).
    SeedFailure failure{SeedFailure::None};
};

// The wall violation the publication check tolerates at prediction stage k.
double publication_tolerance(const MpcCycleConfiguration_t &configuration, int k)
{
    const int start = configuration.publication_tolerance_start_stage;
    if (!(configuration.publication_far_tolerance_m > 0.f) || k <= start) return 0.;
    return double(configuration.publication_far_tolerance_m) *
           std::min(1., double(k - start) / double(kStages - start));
}

struct Snapshot {
    MpcCycleNominal_t nominal;
    Evaluation evaluation;
    bool valid{};
};

struct Problem {
    const MpcCycleState_t *initial;
    double progress;
    const MpcTrajectorySample_t *trajectory;
    size_t trajectory_count;
    double lap_length;
    float dt;
    const MpcCycleConfiguration_t *configuration;
    // Absolute monotonic deadline (0 = none). An exact rollout that is still
    // integrating when it passes is abandoned, so no trial can overrun it.
    double deadline_us{};
};

double grip_weight(const Problem &p, int state_index)
{
    const auto &model = p.configuration->model;
    return model.weight_grip_loss * (state_index == kStages ? model.terminal_multiplier : 1.);
}

// Grip lost beyond the tolerated fraction: the plant reports the interval's
// worst loss_x^2 + loss_y^2; the tolerance is squared to compare with it.
double grip_tolerance_squared(const Problem &p)
{
    const double tolerance = p.configuration->model.grip_loss_tolerance;
    return tolerance * tolerance;
}

bool evaluate(const Problem &p, MpcCycleNominal_t &nominal, Evaluation *out)
{
    *out = {};
    const auto &model = p.configuration->model;
    out->minimum_box = INFINITY;
    out->minimum_speed = INFINITY;
    const double maximum_speed = vehicle_model_get_parameters().maximum_command_speed_mps;
    for (int k = 0; k < kStages; ++k) {
        const auto &u = nominal.controls[k];
        const double limit = model.max_steering_rad;
        out->box[k][0] = limit - double(u.steering_target_rad);
        out->box[k][1] = limit + double(u.steering_target_rad);
        out->box[k][2] = double(MPC_THROTTLE_MAX) - double(u.throttle_command);
        out->box[k][3] = double(u.throttle_command) - double(MPC_THROTTLE_MIN_DRIVE);
        for (double v : out->box[k]) out->minimum_box = std::min(out->minimum_box, v);
        const double speed = nominal.states[k + 1].plant.u;
        out->speed_margin[k][0] = speed + MPC_PHYSX_BRAKE_REVERSE_TRANSIENT_MPS;
        out->speed_margin[k][1] = maximum_speed - speed;
        for (double margin : out->speed_margin[k]) {
            out->minimum_speed = std::min(out->minimum_speed, margin);
            out->violation += std::max(0., -margin);
        }
    }
    float minimum_slack = 0.f;
    float corridor[kStages + 1][2]{};
    MpcCycleStageReference references[kStages + 1];
    for (int k = 0; k <= kStages; ++k) {
        if (!mpc_cycle_sample_stage_reference(p.trajectory, p.trajectory_count,
                p.lap_length, nominal.progress[k], &model, &references[k])) {
            out->failure = SeedFailure::StageReference;
            return false;
        }
        auto &ref = references[k].reference;
        if (k == 0) ref.u = nominal.states[0].plant.u;
        if (!ref.source_witness_valid) ref.r = ref.path_curvature * ref.u;
        nominal.speed_reference[k] = ref.u;
    }
    nominal.speed_reference_valid = 1;
    if (!mpc_cycle_score_candidate_with_references(
            &nominal, p.trajectory, p.trajectory_count, p.lap_length,
            p.configuration, references, &out->objective, &minimum_slack,
            1, corridor, &out->grip, out->corners)) {
        out->failure = SeedFailure::Scoring;
        return false;
    }
    out->minimum_corner = INFINITY;
    out->minimum_physical = INFINITY;
    double minimum_tolerated = INFINITY;
    double parity = 0.;
    for (int k = 1; k <= kStages; ++k) {
        auto &g = out->corners[k - 1];
        if (!g.reference_derivatives_valid) {
            out->failure = SeedFailure::ReferenceDerivatives;
            return false;
        }
        double right = INFINITY, left = INFINITY;
        for (int corner = 0; corner < 4; ++corner) {
            right = std::min(right, g.right[corner]);
            left = std::min(left, g.left[corner]);
            out->violation += std::max(0., -g.right[corner]) + std::max(0., -g.left[corner]);
            out->minimum_physical = std::min({out->minimum_physical,
                g.physical_left[corner], g.physical_right[corner]});
            minimum_tolerated = std::min({minimum_tolerated,
                g.physical_left[corner] + publication_tolerance(*p.configuration, k),
                g.physical_right[corner] + publication_tolerance(*p.configuration, k)});
        }
        out->minimum_corner = std::min({out->minimum_corner, right, left});
        parity = std::max({parity, std::abs(right - double(corridor[k][0])),
                           std::abs(left - double(corridor[k][1]))});
    }
    if (parity > 1.e-9 || !std::isfinite(out->objective) ||
        !std::isfinite(out->violation)) {
        out->failure = parity > 1.e-9 ? SeedFailure::CorridorParity : SeedFailure::NonFinite;
        return false;
    }
    out->physical_feasible = out->minimum_physical >= 0. &&
                             out->minimum_speed >= 0. &&
                             out->minimum_box >= -kBoxTolerance;
    out->feasible = out->physical_feasible && out->minimum_corner >= 0.;
    out->publishable = minimum_tolerated >= 0. && out->minimum_speed >= 0. &&
                       out->minimum_box >= -kBoxTolerance;
    return true;
}

// Integrates `controls` through the source plant from the common initial
// state, reusing the identical already-integrated prefix of `reuse`.
bool rollout(const Problem &p, const MpcModelControl_t controls[kStages],
             const MpcCycleNominal_t *reuse, Snapshot *out)
{
    out->valid = false;
    auto &nominal = out->nominal;
    int first_changed = 0;
    if (reuse && reuse->valid && reuse->horizon == kStages &&
        reuse->progress[0] == p.progress &&
        std::memcmp(&reuse->states[0], p.initial, sizeof(*p.initial)) == 0) {
        while (first_changed < kStages &&
               reuse->controls[first_changed].steering_target_rad ==
                   controls[first_changed].steering_target_rad &&
               reuse->controls[first_changed].throttle_command ==
                   controls[first_changed].throttle_command)
            ++first_changed;
        if (first_changed > 0) {
            std::copy_n(reuse->states, first_changed + 1, nominal.states);
            std::copy_n(reuse->progress, first_changed + 1, nominal.progress);
        }
    }
    if (first_changed == 0) {
        nominal.states[0] = *p.initial;
        nominal.progress[0] = p.progress;
    }
    nominal.horizon = kStages;
    nominal.valid = 0;
    std::copy_n(controls, kStages, nominal.controls);
    for (int k = first_changed; k < kStages; ++k) {
        if (p.deadline_us > 0. && mpc_cycle_monotonic_time_us() >= p.deadline_us) return false;
        if (!mpc_cycle_source_exact_stage(
                &nominal.states[k], nominal.progress[k], &nominal.controls[k],
                p.trajectory, p.trajectory_count, p.lap_length, p.dt,
                &nominal.states[k + 1], &nominal.progress[k + 1])) {
            out->evaluation = {};
            out->evaluation.failure = SeedFailure::PlantStep;
            return false;
        }
    }
    nominal.valid = 1;
    if (!evaluate(p, nominal, &out->evaluation)) return false;
    out->valid = true;
    return true;
}

// ---------------------------------------------------------------------------
// Exact source sensitivities of the rollout.
// ---------------------------------------------------------------------------
struct Stage {
    double A[kState][kState]{};
    double B[kState][2]{};
    // Worst grip loss over the stage's substeps, less the tolerance, and its tangent.
    double grip_residual[4]{};
    double grip_state[4][kState]{};
    double grip_control[4][2]{};
    // Exact second derivatives in (full state, steering target, throttle).
    std::array<Expansion, kCore> hessian;
    std::array<Expansion, 4> grip_hessian;

    void reset_first_order()
    {
        std::memset(A, 0, sizeof A);
        std::memset(B, 0, sizeof B);
        std::memset(grip_residual, 0, sizeof grip_residual);
        std::memset(grip_state, 0, sizeof grip_state);
        std::memset(grip_control, 0, sizeof grip_control);
    }
};

struct Linearization {
    bool curvature{};  // second-order data is used by this iteration's model
    std::array<bool, kStages> valid{};  // stage has second-order data
    std::array<Stage, kStages> stages;
    std::array<Tangent, kStages> sensitivity;  // d state[k+1] / d scaled controls
};

enum class LinearizeStatus { Ok, Failed, Deadline };

// Exact second derivatives of one stage in the core coordinates: plant
// Hessians in (u, v, r, rack, steer, throttle) composed with the analytic
// inverse of the loaded tracking frame, whose inputs are the plant's body
// displacement and the incoming tracking state.
bool second_order_stage(const Problem &p, const MpcCycleNominal_t &trace, int k, Stage *out)
{
    const auto &previous = trace.states[k];
    const auto &exact = trace.states[k + 1];
    UnitySourcePlantHessianC plant;
    if (!vehicle_model_source_interval_hessian(&previous.plant, &trace.controls[k], p.dt,
            static_cast<unsigned int>(p.configuration->derivative_force_substeps), &plant))
        return false;
    double jac[3][6], hess[3][6][6];
    if (!mpc_trajectory_tracking_second_order(
            p.trajectory, p.trajectory_count, p.lap_length, trace.progress[k],
            previous.plant.e_y, previous.plant.e_psi, exact.incoming_body_forward_delta_m,
            exact.incoming_body_lateral_delta_m, exact.incoming_body_yaw_delta_rad,
            trace.progress[k + 1], jac, hess))
        return false;
    constexpr int outputs = UNITY_SOURCE_HESSIAN_OUTPUTS, inputs = UNITY_SOURCE_HESSIAN_INPUTS;
    constexpr int slot[inputs] = {kU, kV, kR, kCore, kCore + 1};
    Eigen::Matrix<double, outputs, kExpansion> gradient = Eigen::Matrix<double, outputs, kExpansion>::Zero();
    std::array<Expansion, outputs> curvature;
    for (int o = 0; o < outputs; ++o) {
        curvature[o].setZero();
        for (int i = 0; i < inputs; ++i) {
            gradient(o, slot[i]) = plant.gradient[o][i];
            for (int j = 0; j < inputs; ++j) curvature[o](slot[i], slot[j]) = plant.hessian[o][i][j];
        }
    }
    for (int o = 0; o < 4; ++o) out->hessian[o] = curvature[o];
    // Tracking-frame inputs (e_y, e_psi, dx, dy, dyaw, s) in the expansion.
    Eigen::Matrix<double, 6, kExpansion> tau = Eigen::Matrix<double, 6, kExpansion>::Zero();
    tau(0, kEy) = 1.;
    tau(1, kEpsi) = 1.;
    tau(5, kS) = 1.;
    for (int i = 0; i < 3; ++i) tau.row(2 + i) = gradient.row(4 + i);
    for (int m = 0; m < 3; ++m) {
        Eigen::Matrix<double, 6, 6> second;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) second(i, j) = hess[m][i][j];
        out->hessian[kEy + m] = tau.transpose() * second * tau;
        for (int i = 0; i < 3; ++i) out->hessian[kEy + m] += jac[m][2 + i] * curvature[4 + i];
    }
    for (int w = 0; w < 4; ++w) out->grip_hessian[w] = curvature[7 + w];
    for (const auto &h : out->hessian) if (!h.allFinite()) return false;
    for (const auto &h : out->grip_hessian) if (!h.allFinite()) return false;
    return true;
}

// Wall-clock share of the second-order tasks in one combined pass (their task
// time spread over the workers).
struct PassTimes {
    double curvature_ms{};
};

// One parallel pass over the exact sensitivities of every stage and, when
// `curvature_stages` is not empty, the second-order data of those stages.
// The second-order tasks are about three times as long as the first-order
// ones; handing them out first keeps the workers busy to the end of the pass.
// A second-order task that is not reached before the deadline leaves its
// stage without data (the caller keeps whatever it had); a first-order task
// that is not reached ends the solve.
LinearizeStatus linearize(const Problem &p, const MpcCycleNominal_t &trace,
                          int threads, double deadline_us,
                          const std::vector<int> &curvature_stages,
                          Linearization *out, PassTimes *times)
{
    // The sticky-tire timers are reset every stage at the positive throttle
    // the controller always commands, so they carry no sensitivity: the 18
    // other plant coordinates and the two commands get one AD lane each.
    constexpr int tracked = UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL;
    constexpr int L = UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;
    static_assert(tracked + 2 <= 20, "the tracked coordinates fit the 20-lane AD kernel");
    std::atomic<bool> failed{false}, late{false};
    const auto stage = [&](int k) {
        if (failed.load(std::memory_order_relaxed)) return;
        if (deadline_us > 0. && mpc_cycle_monotonic_time_us() >= deadline_us) {
            late.store(true, std::memory_order_relaxed);
            return;
        }
        const auto &previous = trace.states[k];
        const auto &exact = trace.states[k + 1];
        if (!exact.incoming_transition_valid || !exact.incoming_tracking_jacobian_valid) {
            failed.store(true);
            return;
        }
        double seed[kPhysical][L]{}, control[2][L]{};
        double physical[kPhysical][L]{}, body[3][L]{}, residual[4]{}, peak[4][L]{};
        for (int i = 0; i < tracked; ++i) seed[i][i] = 1.;
        control[0][tracked] = 1.;
        control[1][tracked + 1] = 1.;
        if (!vehicle_model_full_source_interval_sensitivities(
                &previous.plant, &trace.controls[k], p.dt, seed, control,
                static_cast<unsigned int>(p.configuration->derivative_force_substeps),
                physical, body, residual, peak)) {
            failed.store(true);
            return;
        }
        const auto &g = exact.incoming_tracking_jacobian;
        auto &s = out->stages[k];
        // First-order data is rebuilt every iteration; the second-order data of
        // this cycle stays in place until the cycle's curvature pass replaces it.
        s.reset_first_order();
        for (int column = 0; column < tracked + 2; ++column) {
            const bool control_column = column >= tracked;
            const int state_column = control_column ? column - tracked : column;
            for (int state = 0; state < kPhysical; ++state) {
                if (!control_column) s.A[state][column] = physical[state][column];
                else s.B[state][state_column] = physical[state][column];
            }
            for (int row = 0; row < 3; ++row) {
                double tangent = 0.;
                for (int i = 0; i < 3; ++i) tangent += g[row][i + 2] * body[i][column];
                if (!control_column) s.A[kPhysical + row][column] = tangent;
                else s.B[kPhysical + row][state_column] = tangent;
            }
            for (int wheel = 0; wheel < 4; ++wheel) {
                if (!control_column) s.grip_state[wheel][column] = peak[wheel][column];
                else s.grip_control[wheel][state_column] = peak[wheel][column];
            }
        }
        for (int row = 0; row < 3; ++row) {
            s.A[kPhysical + row][kPhysical] = g[row][0];
            s.A[kPhysical + row][kPhysical + 1] = g[row][1];
            s.A[kPhysical + row][kPhysical + 2] = g[row][5];
        }
        for (int wheel = 0; wheel < 4; ++wheel) s.grip_residual[wheel] = residual[wheel] - grip_tolerance_squared(p);
    };
    std::atomic<bool> curvature_failed{false};
    std::array<double, kStages> curvature_us{};
    const int curvature_count = static_cast<int>(curvature_stages.size());
    const auto curvature = [&](int i) {
        if (curvature_failed.load(std::memory_order_relaxed)) return;
        if (deadline_us > 0. && mpc_cycle_monotonic_time_us() >= deadline_us) return;
        const auto begin = std::chrono::steady_clock::now();
        const int k = curvature_stages[i];
        if (second_order_stage(p, trace, k, &out->stages[k])) out->valid[k] = true;
        else curvature_failed.store(true);
        curvature_us[i] = 1000. * elapsed_ms(begin);
    };
    const int tasks = curvature_count + kStages;
    const auto task = [&](int i) {
        if (i < curvature_count) curvature(i);
        else stage(i - curvature_count);
    };
    int workers = 1;
#ifdef _OPENMP
    workers = std::min({16, omp_get_max_threads(), std::max(1, threads)});
    if (workers > 1) {
#pragma omp parallel for num_threads(workers) schedule(dynamic, 1)
        for (int i = 0; i < tasks; ++i) task(i);
    } else
#endif
    {
        for (int i = 0; i < tasks; ++i) task(i);
    }
    if (times) {
        double total_us = 0.;
        for (int i = 0; i < curvature_count; ++i) total_us += curvature_us[i];
        times->curvature_ms = total_us / (1000. * workers);
    }
    if (failed.load() || curvature_failed.load()) return LinearizeStatus::Failed;
    if (late.load()) return LinearizeStatus::Deadline;
    // Chain rule: S[k] = A[k] S[k-1] + B[k] placed at controls (2k, 2k+1),
    // expressed in the scaled QP variables.
    for (int k = 0; k < kStages; ++k) {
        const auto &s = out->stages[k];
        auto &S = out->sensitivity[k];
        if (k == 0) S.setZero();
        else {
            const Eigen::Map<const Eigen::Matrix<double, kState, kState, Eigen::RowMajor>>
                A(&s.A[0][0]);
            S.noalias() = A * out->sensitivity[k - 1];
        }
        for (int i = 0; i < kState; ++i) {
            S(i, 2 * k) += s.B[i][0] * kSteeringScale;
            S(i, 2 * k + 1) += s.B[i][1] * kThrottleScale;
        }
        if (!S.allFinite()) return LinearizeStatus::Failed;
    }
    return LinearizeStatus::Ok;
}

std::vector<int> all_stages()
{
    std::vector<int> list;
    for (int k = 0; k < kStages; ++k) list.push_back(k);
    return list;
}

// ---------------------------------------------------------------------------
// Curvature recycled across cycles.
// ---------------------------------------------------------------------------
// Stage curvature changes slowly along a converged receding-horizon plan, and
// costs several first-order passes to recompute. The head stages are refreshed
// every cycle; the stages without data (the new tail, or everything after a
// cold start or a long gap) fill in over the next cycles. One thread wave of
// stages per cycle bounds the time spent.
constexpr int kHeadRefresh = 3;
constexpr int kStagesPerCycle = 8;

// Loads the cache shifted by the elapsed command stages. A gap longer than the
// horizon or an epoch off the stage grid leaves the cache empty.
void load_curvature(const Options &options, float dt, Linearization *lin)
{
    lin->valid.fill(false);
    for (auto &stage : lin->stages) {
        for (auto &h : stage.hessian) h.setZero();
        for (auto &h : stage.grip_hessian) h.setZero();
    }
    const MpcCycleCurvatureCache_t *cache = options.curvature_cache;
    if (!cache || cache->epoch_ns < 0 || options.command_epoch_ns < 0) return;
    const std::int64_t stage_ns = std::llround(1.e9 * double(dt));
    const std::int64_t elapsed = options.command_epoch_ns - cache->epoch_ns;
    const std::int64_t shift = (elapsed + stage_ns / 2) / stage_ns;
    if (elapsed < 0 || shift >= kStages || std::llabs(elapsed - shift * stage_ns) > 1000000) return;
    for (int k = 0; k + shift < kStages; ++k) {
        if (!cache->valid[k + shift]) continue;
        auto &stage = lin->stages[k];
        for (int m = 0; m < kCore; ++m)
            stage.hessian[m] = Eigen::Map<const Expansion>(cache->hessian[k + shift][m]);
        for (int w = 0; w < 4; ++w)
            stage.grip_hessian[w] = Eigen::Map<const Expansion>(cache->hessian[k + shift][kCore + w]);
        lin->valid[k] = true;
    }
}

void store_curvature(const Options &options, const Linearization &lin)
{
    MpcCycleCurvatureCache_t *cache = options.curvature_cache;
    if (!cache || options.command_epoch_ns < 0) return;
    cache->epoch_ns = options.command_epoch_ns;
    for (int k = 0; k < kStages; ++k) {
        cache->valid[k] = lin.valid[k];
        if (!lin.valid[k]) continue;
        for (int m = 0; m < kCore; ++m)
            Eigen::Map<Expansion>(cache->hessian[k][m]) = lin.stages[k].hessian[m];
        for (int w = 0; w < 4; ++w)
            Eigen::Map<Expansion>(cache->hessian[k][kCore + w]) = lin.stages[k].grip_hessian[w];
    }
}

// Stages whose curvature this cycle (re)computes: all of them without a
// cache; otherwise the head and then the stages without data, capped.
std::vector<int> stages_to_expand(const Options &options, const Linearization &lin)
{
    std::vector<int> list;
    if (!options.curvature_cache || options.command_epoch_ns < 0) return all_stages();
    for (int k = 0; k < kHeadRefresh; ++k) list.push_back(k);
    for (int k = kHeadRefresh; k < kStages && int(list.size()) < kStagesPerCycle; ++k)
        if (!lin.valid[k]) list.push_back(k);
    return list;
}

// ---------------------------------------------------------------------------
// Local model: quadratic racing residuals + grip epigraph + linear rows.
// ---------------------------------------------------------------------------
using Row = Eigen::Matrix<double, kControls, 1>;

struct GripRow {
    double weight;   // cost = weight * max(0, residual + slope.x)^2
    double residual; // signed peak residual at the iterate
    double clipped;  // max(0, residual)
    Row slope;
};

struct PhysicalRow {
    double margin;   // >= 0 is legal; the linearization is margin + slope.x
    double physical_margin;
    Row slope;
};

struct Model {
    bool physical_feasible{};
    Matrix hessian;  // True selected-branch cost curvature, excluding grip GN epigraph.
    Matrix direction_hessian;  // Convex direction metric; never used for prediction.
    Vector gradient;
    std::vector<PhysicalRow> physical;
    std::vector<GripRow> grip;
    double nominal_grip_cost{};
    Vector lower, upper;  // legal command interval, as a step
    // Direct gradient of the racing cost with respect to each reduced state.
    std::array<Core, kStages + 2> state_gradient;
};

bool reflect(Model *model);

Core unit(int index)
{
    Core v = Core::Zero();
    v(index) = 1.;
    return v;
}

// Derivative of the combination `coefficients . core(z[interval + 1])` with
// respect to the scaled controls.
Row state_feature(const Linearization &lin, int interval, const Core &coefficients)
{
    Row feature = Row::Zero();
    for (int i = 0; i < kCore; ++i)
        if (coefficients(i) != 0.)
            feature.noalias() += coefficients(i) * lin.sensitivity[interval].row(kCoreIndex[i]).transpose();
    return feature;
}

bool build_model(const Problem &p, const Snapshot &at, const Linearization &lin,
                 Model *model)
{
    const auto &cfg = p.configuration->model;
    const auto &schedule = *p.configuration->objective_schedule;
    const auto &nominal = at.nominal;
    const auto &evaluation = at.evaluation;
    model->physical_feasible = evaluation.physical_feasible;
    model->hessian = Matrix::Zero(kControls, kControls);
    model->gradient = Vector::Zero(kControls);
    model->physical.clear();
    model->grip.clear();
    model->nominal_grip_cost = 0.;

    const auto residual_term = [&](double weight, double residual, const Row &feature) {
        if (weight <= 0.) return true;
        if (!std::isfinite(weight) || !std::isfinite(residual) || !feature.allFinite())
            return false;
        model->hessian.selfadjointView<Eigen::Lower>().rankUpdate(feature, 2. * weight);
        model->gradient.noalias() += (2. * weight * residual) * feature;
        return true;
    };
    // Command residuals have one nonzero derivative (two for slew).
    // Accumulate only those entries; a dense 60x60 outer product performs
    // the same zero additions for every other command at every stage.
    const auto control_residual = [&](double weight, double residual,
                                      int stage, int axis, bool slew = false) {
        if (weight <= 0.) return true;
        if (!std::isfinite(weight) || !std::isfinite(residual)) return false;
        const int index = 2 * stage + axis;
        const double feature = kScale[axis];
        const double curvature = (2. * weight) * feature;
        const double gradient = (2. * weight * residual) * feature;
        model->hessian(index, index) += curvature * feature;
        model->gradient(index) += gradient;
        if (slew && stage > 0) {
            model->hessian(index, index - 2) += curvature * -feature;
            model->hessian(index - 2, index) += -curvature * feature;
            model->hessian(index - 2, index - 2) += -curvature * -feature;
            model->gradient(index - 2) += -gradient;
        }
        return true;
    };

    // gz[k] is the direct gradient of the racing cost with respect to z[k].
    auto &gz = model->state_gradient;
    gz.fill(Core::Zero());
    const Core rear_axle_lateral = unit(kV) - kRearAxleToCom * unit(kR);
    const auto state_residual = [&](double weight, double residual, int k, const Core &coefficients) {
        if (weight <= 0.) return true;
        if (!residual_term(weight, residual, state_feature(lin, k - 1, coefficients))) return false;
        gz[k] += (2. * weight * residual) * coefficients;
        return true;
    };
    for (int k = 1; k <= kStages; ++k) {
        const auto &plant = nominal.states[k].plant;
        const auto &ref = schedule.references[k];
        const double terminal = k == kStages ? cfg.terminal_multiplier : 1.;
        const int interval = k - 1;
        if (!state_residual(terminal * schedule.e_y_weights[k], double(plant.e_y) - ref.e_y, k, unit(kEy)) ||
            !state_residual(terminal * schedule.e_psi_weights[k], double(plant.e_psi) - ref.e_psi, k, unit(kEpsi)) ||
            !state_residual(terminal * schedule.speed_weights[k], double(plant.u) - ref.u, k, unit(kU)) ||
            !state_residual(terminal * cfg.weight_v, double(plant.v) - ref.v, k, rear_axle_lateral) ||
            !state_residual(terminal * cfg.weight_r, double(plant.r) - ref.r, k, unit(kR)) ||
            !state_residual(terminal * cfg.weight_steering_command,
                            double(plant.actual_steering_angle) - ref.steering_command, k, unit(kRack)) ||
            !control_residual(terminal * cfg.weight_steering_command,
                              double(plant.delayed_steering_command_2) - ref.steering_command,
                              interval, 0))
            return false;
    }
    double previous_steering = nominal.states[0].plant.delayed_steering_command_2;
    double previous_throttle = nominal.states[0].plant.delayed_throttle_command_2;
    for (int k = 0; k < kStages; ++k) {
        const auto &ref = schedule.references[k];
        const auto &u = nominal.controls[k];
        const double increment = steering_reference_increment(&ref, &schedule.references[k + 1]);
        if (!control_residual(cfg.weight_steering_target,
                           double(u.steering_target_rad) - ref.steering_command,
                           k, 0) ||
            !control_residual(cfg.weight_throttle_command, u.throttle_command,
                           k, 1) ||
            !control_residual(cfg.weight_steering_slew,
                           double(u.steering_target_rad) - previous_steering - increment,
                           k, 0, true) ||
            !control_residual(cfg.weight_throttle_slew,
                           double(u.throttle_command) - previous_throttle, k, 1, true))
            return false;
        previous_steering = u.steering_target_rad;
        previous_throttle = u.throttle_command;
    }

    // Grip loss: one cut per wheel and stage on the worst-substep residual.
    for (int k = 0; k < kStages; ++k) {
        const auto &stage = lin.stages[k];
        const double weight = grip_weight(p, k + 1);
        if (weight <= 0.) continue;
        for (int wheel = 0; wheel < 4; ++wheel) {
            GripRow row;
            row.weight = weight;
            row.residual = stage.grip_residual[wheel];
            row.clipped = std::max(0., row.residual);
            row.slope = Row::Zero();
            if (k > 0)
                for (int s = 0; s < kPhysical; ++s)
                    row.slope.noalias() += stage.grip_state[wheel][s] *
                                           lin.sensitivity[k - 1].row(s).transpose();
            row.slope(2 * k) += stage.grip_control[wheel][0] * kSteeringScale;
            row.slope(2 * k + 1) += stage.grip_control[wheel][1] * kThrottleScale;
            if (!row.slope.allFinite() || !std::isfinite(row.residual)) return false;
            model->nominal_grip_cost += row.weight * row.clipped * row.clipped;
            model->grip.push_back(std::move(row));
        }
    }

    for (int k = 0; k < kStages; ++k) {
        const auto &g = evaluation.corners[k];
        const Row lateral = state_feature(lin, k, unit(kEy));
        const Row heading_feature = state_feature(lin, k, unit(kEpsi));
        const Row progress = state_feature(lin, k, unit(kS));
        for (int side = 0; side < 2; ++side)
            for (int corner = 0; corner < 4; ++corner) {
                const bool left = side == 0;
                PhysicalRow row;
                row.margin = left ? g.left[corner] : g.right[corner];
                row.physical_margin = left ? g.physical_left[corner] : g.physical_right[corner];
                const double heading = left ? g.left_d_psi[corner] : g.right_d_psi[corner];
                const double path = left ? g.left_d_progress_forward : g.right_d_progress_forward;
                row.slope = (left ? -1. : 1.) * lateral +
                            heading * heading_feature + path * progress;
                if (!row.slope.allFinite()) return false;
                model->physical.push_back(std::move(row));
            }
    }
    // Match the independent publication validator's speed envelope.
    // Otherwise a wall-clear slowing tail can reverse and be falsely called
    // feasible by the SQP, only to fail the final gate.
    for (int k = 0; k < kStages; ++k) {
        const Row speed = state_feature(lin, k, unit(kU));
        for (int side = 0; side < 2; ++side) {
            PhysicalRow row;
            row.margin = row.physical_margin = evaluation.speed_margin[k][side];
            row.slope = side == 0 ? speed : -speed;
            model->physical.push_back(std::move(row));
        }
    }
    // Physical command interval, separate from the step bound.
    model->lower = Vector(kControls);
    model->upper = Vector(kControls);
    for (int k = 0; k < kStages; ++k)
        for (int axis = 0; axis < 2; ++axis) {
            const int j = 2 * k + axis;
            model->upper(j) = std::max(0., evaluation.box[k][2 * axis]) / kScale[axis];
            model->lower(j) = -std::max(0., evaluation.box[k][2 * axis + 1]) / kScale[axis];
        }
    // Only the lower triangle is updated by the rank-one Gauss-Newton terms.
    // Restore exact symmetry once before adding the signed second-order block.
    for (int j = 1; j < kControls; ++j)
        for (int i = 0; i < j; ++i)
            model->hessian(i, j) = model->hessian(j, i);
    if (lin.curvature) {
        // Exact curvature. The adjoint of the full state carries each stage
        // output's multiplier; the core-output Hessians are mapped to the
        // controls through the sensitivities of the core inputs.
        const auto embed = [](const Core &core) {
            Full full = Full::Zero();
            for (int i = 0; i < kCore; ++i) full(kCoreIndex[i]) = core(i);
            return full;
        };
        std::array<Full, kStages + 2> lambda;
        lambda.fill(Full::Zero());
        lambda[kStages] = embed(gz[kStages]);
        for (int k = kStages - 1; k >= 1; --k) {
            const Eigen::Map<const Eigen::Matrix<double, kState, kState, Eigen::RowMajor>> A(&lin.stages[k].A[0][0]);
            lambda[k] = embed(gz[k]) + A.transpose() * lambda[k + 1];
            // Interval k grip cost depends on its incoming physical state; its
            // gradient weights the curvature of every upstream stage.
            for (int w = 0; w < 4; ++w) {
                const auto &stage = lin.stages[k];
                const double multiplier = 2. * grip_weight(p, k + 1) * std::max(0., stage.grip_residual[w]);
                for (int i = 0; i < kPhysical; ++i) lambda[k](i) += multiplier * stage.grip_state[w][i];
            }
        }
        Matrix curvature = Matrix::Zero(kControls, kControls);
        for (int k = 0; k < kStages; ++k) {
            const auto &stage = lin.stages[k];
            Expansion q = Expansion::Zero();
            for (int m = 0; m < kCore; ++m) q += lambda[k + 1](kCoreIndex[m]) * stage.hessian[m];
            for (int wheel = 0; wheel < 4; ++wheel) {
                const double multiplier = 2. * grip_weight(p, k + 1) * std::max(0., stage.grip_residual[wheel]);
                if (multiplier > 0.) q += multiplier * stage.grip_hessian[wheel];
            }
            const int columns = 2 * k + 2;
            Eigen::Matrix<double, kExpansion, Eigen::Dynamic> map =
                Eigen::Matrix<double, kExpansion, Eigen::Dynamic>::Zero(kExpansion, columns);
            if (k > 0)
                for (int i = 0; i < kCore; ++i)
                    map.row(i) = lin.sensitivity[k - 1].row(kCoreIndex[i]).leftCols(columns);
            map(kCore, 2 * k) = kSteeringScale;
            map(kCore + 1, 2 * k + 1) = kThrottleScale;
            curvature.topLeftCorner(columns, columns).noalias() += map.transpose() * q * map;
        }
        model->hessian += 0.5 * (curvature + curvature.transpose());
    }
    // Reflect the direction metric to keep the subproblem convex;
    // prediction requires the unmodified, possibly indefinite curvature.
    if (lin.curvature) {
        if (!reflect(model)) return false;
    } else {
        model->direction_hessian = model->hessian;
    }
    return model->hessian.allFinite() && model->gradient.allFinite();
}

double grip_model_cost(const Model &model, const Vector &x)
{
    double cost = 0.;
    for (const auto &row : model.grip) {
        const double t = std::max(0., row.residual + row.slope.dot(x));
        cost += row.weight * t * t;
    }
    return cost;
}

double linearized_violation(const Model &model, const Vector &x)
{
    double violation = 0.;
    for (const auto &row : model.physical)
        violation += std::max(0., -(row.margin + row.slope.dot(x)));
    return violation;
}

struct Step {
    Vector x;
    bool deadline{};
    int iterations{};
    int rounds{};
    double qp_ms{};
    // Predicted racing-cost reduction; restoration adds its L1 wall penalty.
    double predicted_reduction{};
    double model_grip_cost{};
};

// Solves the local model exactly inside the radius and the legal command box. While the iterate is feasible the physical
// rows are hard; x = 0 satisfies them, so the QP is always feasible.
// Otherwise they enter as an exact L1 penalty. Rows enter the QP lazily:
// the solve is repeated until every omitted row is satisfied (grip rows:
// contributes nothing) at the solution, so the result equals the solution
// of the complete model.
bool solve_step(const Model &model, bool feasible, const Options &options,
                double radius, Step *step)
{
    const auto begin = std::chrono::steady_clock::now();
    constexpr double kInitialReach = 0.25;  // wall rows this close are started with
    constexpr double kInitialGripReach = 0.05;  // grip rows (loss^2 units) likewise
    const int n = kControls;
    std::vector<char> physical_in(model.physical.size(), 0), grip_in(model.grip.size(), 0);
    for (size_t i = 0; i < model.physical.size(); ++i)
        physical_in[i] = model.physical[i].margin < kInitialReach;
    // Grip cuts remain active during restoration.
    for (size_t i = 0; i < model.grip.size(); ++i)
        grip_in[i] = model.grip[i].residual > -kInitialGripReach;
    const Vector lower = model.lower.cwiseMax(-radius), upper = model.upper.cwiseMin(radius);
    const double tolerance =
        std::max(1.e-4, std::min(1.e-2, 1.e-4 * model.gradient.lpNorm<Eigen::Infinity>()));
    for (int round = 0; round < 12; ++round) {
        int physical = 0, grip = 0;
        for (char in : physical_in) physical += in;
        for (char in : grip_in) grip += in;
        const int hard = feasible || model.physical_feasible ? physical : 0;
        const int elastic = grip + (feasible ? 0 : physical);
        ConvexQp qp;
        // The same tracking, command-slew and grip model regularizes
        // restoration. A Euclidean minimum-control correction can move along
        // cost-sensitive null directions and destroy grip while reducing a
        // wall violation. The L1 wall penalty still gives feasibility priority.
        qp.hessian = model.direction_hessian;
        qp.gradient = model.gradient;
        qp.lower = lower;
        qp.upper = upper;
        qp.hard_jacobian.resize(hard, n);
        qp.hard_margin.resize(hard);
        qp.elastic_jacobian.resize(elastic, n);
        qp.elastic_margin.resize(elastic);
        qp.elastic_weight = Vector::Zero(elastic);
        qp.elastic_linear_cost = Vector::Zero(elastic);
        int e = 0, h = 0;
        for (size_t i = 0; i < model.grip.size(); ++i) {
            if (!grip_in[i]) continue;
            // t >= residual + slope.x  <=>  -slope.x - residual + t >= 0
            qp.elastic_jacobian.row(e) = -model.grip[i].slope.transpose();
            qp.elastic_margin(e) = -model.grip[i].residual;
            qp.elastic_weight(e++) = 2. * model.grip[i].weight;
        }
        for (size_t i = 0; i < model.physical.size(); ++i) {
            if (!physical_in[i]) continue;
            const auto &row = model.physical[i];
            if (feasible) {
                qp.hard_jacobian.row(h) = row.slope.transpose();
                qp.hard_margin(h++) = row.margin;
            } else {
                if (model.physical_feasible) {
                    qp.hard_jacobian.row(h) = row.slope.transpose();
                    qp.hard_margin(h++) = row.physical_margin;
                }
                qp.elastic_jacobian.row(e) = row.slope.transpose();
                qp.elastic_margin(e) = row.margin;
                qp.elastic_weight(e) = 1.;
                qp.elastic_linear_cost(e++) = options.violation_penalty;
            }
        }
        ConvexQpResult solved;
        const bool ok = solve_convex_qp(qp, 80, tolerance, &solved, options.deadline_us);
        step->iterations += solved.iterations;
        step->rounds = round + 1;
        step->deadline = solved.deadline_exceeded;
        if (!ok || !solved.converged || !solved.step.allFinite()) {
            step->qp_ms = elapsed_ms(begin);
            return false;
        }
        step->x = solved.step.cwiseMax(lower).cwiseMin(upper);
        bool complete = true;
        for (size_t i = 0; i < model.physical.size(); ++i)
            if (!physical_in[i] &&
                model.physical[i].margin + model.physical[i].slope.dot(step->x) < 0.) {
                physical_in[i] = 1;
                complete = false;
            }
        for (size_t i = 0; i < model.grip.size(); ++i)
            if (!grip_in[i] &&
                model.grip[i].residual + model.grip[i].slope.dot(step->x) > 0.) {
                grip_in[i] = 1;
                complete = false;
            }
        if (complete) break;
        if (options.deadline_us > 0. && mpc_cycle_monotonic_time_us() >= options.deadline_us) {
            step->deadline = true;
            step->qp_ms = elapsed_ms(begin);
            return false;
        }
        if (round == 11) {
            step->qp_ms = elapsed_ms(begin);
            return false;
        }
    }
    step->qp_ms = elapsed_ms(begin);
    step->model_grip_cost = grip_model_cost(model, step->x);
    step->predicted_reduction =
        -(model.gradient.dot(step->x) + 0.5 * step->x.dot(model.hessian * step->x) +
          step->model_grip_cost - model.nominal_grip_cost);
    return true;
}

void controls_after_step(const Snapshot &at, const Vector &x, double alpha,
                         const MpcSolverConfiguration_t &model,
                         MpcModelControl_t out[kStages])
{
    for (int k = 0; k < kStages; ++k) {
        const auto &u = at.nominal.controls[k];
        out[k] = u;
        out[k].steering_target_rad = static_cast<float>(std::clamp(
            double(u.steering_target_rad) + alpha * kSteeringScale * x(2 * k),
            -double(model.max_steering_rad), double(model.max_steering_rad)));
        out[k].throttle_command = static_cast<float>(std::clamp(
            double(u.throttle_command) + alpha * kThrottleScale * x(2 * k + 1),
            double(MPC_THROTTLE_MIN_DRIVE), double(MPC_THROTTLE_MAX)));
    }
}

// Direction metric of the convex solve: the eigenvalue-absolute reflection of
// the (possibly indefinite) curvature. Prediction keeps the true curvature.
bool reflect(Model *model)
{
    // A positive-definite Hessian needs no eigenvalue reflection.
    Eigen::LLT<Matrix> positive(model->hessian);
    if (positive.info() == Eigen::Success) {
        model->direction_hessian = model->hessian;
        return true;
    }
    Eigen::SelfAdjointEigenSolver<Matrix> eigen(model->hessian);
    if (eigen.info() != Eigen::Success) return false;
    model->direction_hessian = eigen.eigenvectors() * eigen.eigenvalues().cwiseAbs().asDiagonal() *
                               eigen.eigenvectors().transpose();
    return true;
}

// What one exact trial measured along its step x. The cost error is quadratic
// in the step fraction, so the curvature along x absorbs it; a wall row that the
// trial found closer than its linear prediction becomes the secant through the
// exact point.
void correct_wall_rows(const Snapshot &trial, const Vector &x, Model *model)
{
    const double norm2 = x.squaredNorm();
    size_t row = 0;
    for (int k = 0; k < kStages; ++k)
        for (int side = 0; side < 2; ++side)
            for (int corner = 0; corner < 4; ++corner, ++row) {
                const auto &g = trial.evaluation.corners[k];
                const double exact = side == 0 ? g.left[corner] : g.right[corner];
                auto &wall = model->physical[row];
                const double shortfall = wall.margin + wall.slope.dot(x) - exact;
                if (shortfall > 0.) wall.slope -= shortfall * x.transpose() / norm2;
            }
    for (int k = 0; k < kStages; ++k)
        for (int side = 0; side < 2; ++side, ++row) {
            auto &bound = model->physical[row];
            const double shortfall = bound.margin + bound.slope.dot(x) -
                trial.evaluation.speed_margin[k][side];
            if (shortfall > 0.) bound.slope -= shortfall * x.transpose() / norm2;
        }
}

void add_measurement(const Snapshot &trial, const Vector &x, double predicted, double actual,
                     Model *model)
{
    const double norm2 = x.squaredNorm();
    if (std::isfinite(actual))
        model->hessian -= 2. * (actual - predicted) * x * x.transpose() / (norm2 * norm2);
    correct_wall_rows(trial, x, model);
}

int newly_active_grip(const Evaluation &before, const Evaluation &after)
{
    int count = 0;
    for (int k = 0; k < kStages; ++k)
        for (int wheel = 0; wheel < 4; ++wheel)
            count += before.grip.excess[k][wheel] <= 1.e-8 && after.grip.excess[k][wheel] > 1.e-5;
    return count;
}

struct SpeculativeRadius {
    double value;
    int first;
};

std::vector<SpeculativeRadius> speculative_radii(bool feasible, double radius,
                                                const Options &options)
{
    std::vector<SpeculativeRadius> radii;
    radii.reserve(options.speculative_trials);
    for (int j = 0; j < options.speculative_trials; ++j) {
        const int exponent = feasible ? -j : options.speculative_trials - 1 - 2 * j;
        const double value = std::clamp(radius * std::pow(2., exponent),
            options.minimum_radius, options.maximum_radius);
        int first = 0;
        while (first < j && radii[first].value != value) ++first;
        radii.push_back({value, first});
    }
    return radii;
}
} // namespace

// One local search from `seed`.
static bool solve_from_seed(const MpcCycleState_t &initial, double initial_progress,
                            const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
                            double lap_length, float dt,
                            const MpcCycleConfiguration_t &configuration,
                            const MpcModelControl_t seed[kStages], const Options &options,
                            Result *result)
{
    if (!result || !trajectory || trajectory_count == 0 || !seed ||
        !configuration.objective_schedule || !configuration.objective_schedule->valid ||
        configuration.objective_schedule->horizon != kStages ||
        options.maximum_iterations < 1 ||
        !(options.minimum_radius > 0.) || options.minimum_radius > options.maximum_radius)
        return false;
    *result = {};
    const Problem problem{&initial, initial_progress, trajectory, trajectory_count,
                          lap_length, dt, &configuration};
    // The seed must be integrated whatever the clock says; the trials of the
    // search are abandoned at the deadline.
    Problem search = problem;
    search.deadline_us = options.deadline_us;
    auto current = std::make_unique<Snapshot>();
    auto trial = std::make_unique<Snapshot>();
    auto fallback = std::make_unique<Snapshot>();
    auto lin = std::make_unique<Linearization>();
    auto model = std::make_unique<Model>();
    if (!rollout(problem, seed, options.same_epoch_nominal, current.get())) {
        result->seed_failure = current->evaluation.failure;
        return false;
    }
    ++result->rollouts;
    result->initial_feasible = current->evaluation.feasible;
    result->initial_cost = current->evaluation.objective;
    result->initial_violation = current->evaluation.violation;

    // The radius is the extent over which an exact rollout last confirmed
    // the physical model. Keep it as the search center on a merit change;
    // restoration probes both larger and smaller steps and verifies them.
    double radius = std::clamp(options.radius, options.minimum_radius, options.maximum_radius);
    double measured_iteration_us = 0.;
    bool curvature_done = false;
    load_curvature(options, dt, lin.get());
    const auto late = [&](double needed_us) {
        return options.deadline_us > 0. &&
               mpc_cycle_monotonic_time_us() + needed_us >= options.deadline_us;
    };
    // Capped derivative passes are cheap but can misdescribe the plant below a few
    // m/s, where every trial is then rejected; an iteration that no trial survives
    // is repeated once with the plant's own substep schedule, and the rest of the
    // solve keeps it.
    auto exact_configuration = std::make_unique<MpcCycleConfiguration_t>(configuration);
    exact_configuration->derivative_force_substeps = 0;
    Problem exact_problem = problem;
    exact_problem.configuration = exact_configuration.get();
    bool exact_derivatives = configuration.derivative_force_substeps == 0 || options.exact_derivatives;
    result->termination = Termination::IterationLimit;
    for (int iteration = 0; iteration < options.maximum_iterations; ++iteration) {
        const bool feasible = current->evaluation.feasible;
        // Backtracking is charged to the previous iteration. Its elapsed
        // time is not a lower bound on the next restoration iteration.
        // Restoration gets the remaining hard budget; stage AD and the QP
        // still check their absolute deadlines.
        if (late(feasible && iteration > 0 ? measured_iteration_us : 0.)) {
            result->termination = Termination::Deadline;
            break;
        }
        const auto begin = std::chrono::steady_clock::now();
        const double radius_before = radius;
        Iteration record;
        record.feasible_before = feasible;
        record.cost_before = current->evaluation.objective;
        record.violation_before = current->evaluation.violation;
        record.grip_cost_before = current->evaluation.grip.cost;

        const auto lin_begin = std::chrono::steady_clock::now();
        // Exact curvature belongs to racing. Restoration uses the positive
        // Gauss-Newton tracking/grip model to regularize its L1 wall correction.
        // The second-order data is computed once per cycle, in the same pass as
        // the first iteration's sensitivities; later racing iterations reuse it
        // with refreshed first-order data.
        const bool refresh_curvature = feasible && !curvature_done;
        PassTimes pass;
        const LinearizeStatus status = linearize(
            exact_derivatives ? exact_problem : problem, current->nominal,
            configuration.model.jacobian_threads,
            options.deadline_us,
            refresh_curvature ? stages_to_expand(options, *lin) : std::vector<int>{},
            lin.get(), &pass);
        const double curvature_ms = pass.curvature_ms;
        curvature_done = curvature_done || refresh_curvature;
        lin->curvature = feasible;
        record.linearization_ms = elapsed_ms(lin_begin);
        if (status != LinearizeStatus::Ok) {
            result->termination = status == LinearizeStatus::Deadline
                                      ? Termination::Deadline : Termination::LinearizationFailed;
            break;
        }
        const auto model_begin = std::chrono::steady_clock::now();
        const bool modelled = build_model(problem, *current, *lin, model.get());
        record.model_ms = elapsed_ms(model_begin);
        if (!modelled) {
            result->termination = Termination::LinearizationFailed;
            break;
        }
        // Use the QP's same objective during exact acceptance. Judging only
        // wall violation can trade a fraction of a millimeter of reserve for
        // a large tracking/grip loss, despite the cost-aware QP direction.
        const double origin = current->evaluation.objective +
            (feasible ? 0. : options.violation_penalty * current->evaluation.violation);
        const double scale_value = std::max(1., origin);
        bool accepted = false, stop = false;
        {
            // Each round solves the QP at several radii at once and
            // verifies every result by an exact rollout. Racing takes the best
            // accurate decrease; restoration takes a feasible trial with the
            // lowest cost. Otherwise measured cost curvature (racing only) and
            // wall secants correct the model before the radius is shrunk.
            struct Candidate {
                double radius{}, predicted{}, actual{-INFINITY}, error{INFINITY};
                Step step;
                std::unique_ptr<Snapshot> after = std::make_unique<Snapshot>();
                bool solved{}, evaluated{};
            };
            std::vector<Candidate> candidates(options.speculative_trials);
            Model corrected;
            const Model *active = model.get();
            int corrections = 0;
            // Retain the best exact decrease while searching: racing can fall
            // back outside the accuracy gate; restoration can return its best
            // merit-decreasing trial if no feasible step fits before the deadline.
            int fallback_from = -1;
            double fallback_actual = 0.;
            Iteration fallback_record;
            double round_ms_seen = 0.;
            const auto fill = [&](const Candidate &c, Iteration *out) {
                out->radius = c.radius;
                out->step_inf = c.step.x.lpNorm<Eigen::Infinity>();
                out->radius_active = out->step_inf >= 0.99 * c.radius;
                out->predicted_reduction = c.predicted;
                out->grip_cost_predicted = c.step.model_grip_cost;
                out->actual_reduction = c.actual;
                out->ratio = c.actual / c.predicted;
                out->cost_trial = c.after->evaluation.objective;
                out->violation_trial = c.after->evaluation.violation;
                out->grip_cost_trial = c.after->evaluation.grip.cost;
                out->newly_active_grip = newly_active_grip(current->evaluation, c.after->evaluation);
            };
            const auto carried = [&](const Iteration &r, double error) {
                // The error grows with the square of the step fraction, so the
                // radius that would have given half the tolerance follows from
                // the measured error; it may grow at most fourfold per step.
                const double fit = r.step_inf * std::sqrt(
                    .5 * options.accuracy_tolerance / std::max(error, 1.e-3 * options.accuracy_tolerance));
                return std::clamp(std::min(4. * r.radius, fit), options.minimum_radius,
                                  options.maximum_radius);
            };
            for (int round = 0; round < options.maximum_rounds && !accepted && !stop; ++round) {
                if (late(1000. * round_ms_seen)) {
                    result->termination = Termination::Deadline;
                    stop = true;
                    break;
                }
                const auto round_begin = std::chrono::steady_clock::now();
                // Equal clamped radii share their first trial; candidate order
                // still governs ties, corrections and logical trial counts.
                const auto radii = speculative_radii(feasible, radius, options);
                for (int j = 0; j < options.speculative_trials; ++j) {
                    auto &c = candidates[j];
                    // A changed measurement can require a correction larger
                    // than the last racing step. Probe on both sides of that
                    // earned radius instead of resetting it to a fixed value
                    // or only shrinking a step that cannot reach feasibility.
                    c.radius = radii[j].value;
                    c.step = Step{};
                    c.solved = c.evaluated = false;
                    c.actual = -INFINITY;
                    c.error = INFINITY;
                }
#pragma omp parallel for num_threads(options.speculative_trials) schedule(static, 1)
                for (int j = 0; j < options.speculative_trials; ++j) {
                    if (radii[j].first != j) continue;
                    auto &c = candidates[j];
                    c.solved = solve_step(*active, feasible, options, c.radius, &c.step);
                    if (!c.solved) continue;
                    c.predicted = c.step.predicted_reduction + (feasible ? 0. :
                        options.violation_penalty * (current->evaluation.violation -
                            linearized_violation(*active, c.step.x)));
                    if (!(c.predicted > 1.e-9 * scale_value)) continue;
                    MpcModelControl_t controls[kStages];
                    controls_after_step(*current, c.step.x, 1., configuration.model, controls);
                    c.evaluated = rollout(search, controls, &current->nominal, c.after.get());
                    if (!c.evaluated) continue;
                    c.actual = feasible
                        ? (c.after->evaluation.feasible ? origin - c.after->evaluation.objective : -INFINITY)
                        : (current->evaluation.physical_feasible && !c.after->evaluation.physical_feasible
                            ? -INFINITY : origin - c.after->evaluation.objective -
                                options.violation_penalty * c.after->evaluation.violation);
                    c.error = std::abs(c.predicted - c.actual) / scale_value;
                }
                round_ms_seen = elapsed_ms(round_begin);
                record.attempts = round + 1;
                record.rollout_ms += round_ms_seen;
                bool deadline = false;
                for (int j = 0; j < options.speculative_trials; ++j) {
                    const auto &c = candidates[radii[j].first];
                    record.qp_ms = std::max(record.qp_ms, c.step.qp_ms);
                    record.qp_iterations += c.step.iterations;
                    record.qp_rounds = std::max(record.qp_rounds, c.step.rounds);
                    result->rollouts += c.evaluated;
                    deadline = deadline || c.step.deadline;
                }
                record.qp_converged = candidates[0].solved;
                // A trial abandoned inside its rollout is the same event as a
                // QP stopped by the clock.
                deadline = deadline || late(0.);
                if (deadline) {
                    result->termination = Termination::Deadline;
                    stop = true;
                    // Other workers may already have verified a usable step.
                    // Keep those results before honoring the batch deadline.
                }
                if (candidates[0].solved &&
                    !(candidates[0].predicted > 1.e-9 * scale_value)) {
                    // Nothing is left that the local model can improve.
                    result->termination = Termination::Converged;
                    result->converged = true;
                    stop = true;
                    break;
                }
                int chosen = -1, lead = -1;
                for (int j = 0; j < options.speculative_trials; ++j) {
                    const auto &c = candidates[radii[j].first];
                    if (!c.evaluated) continue;
                    if (lead < 0 || (!feasible && c.actual > candidates[radii[lead].first].actual)) lead = j;
                    const bool decrease = c.actual > options.sufficient_decrease * c.predicted &&
                        c.actual > 32. * std::numeric_limits<double>::epsilon() * scale_value;
                    const bool qualified = feasible ? c.error <= options.accuracy_tolerance
                                                     : c.after->evaluation.feasible;
                    if (decrease && qualified &&
                        (chosen < 0 || (feasible ? c.actual > candidates[radii[chosen].first].actual
                            : c.after->evaluation.objective < candidates[radii[chosen].first].after->evaluation.objective)))
                        chosen = j;
                    else if (decrease && !qualified && c.actual > fallback_actual) {
                        fill(c, &fallback_record);
                        // Keep the measured trial intact: this same candidate
                        // supplies the wall secants below when the round has
                        // no accurate step. Copy its snapshot so both paths
                        // use the evaluated rollout.
                        *fallback = *c.after;
                        fallback_actual = c.actual;
                        fallback_from = j;
                    }
                }
                if (chosen >= 0) {
                    auto &c = candidates[radii[chosen].first];
                    fill(c, &record);
                    std::swap(trial, c.after);
                    if (feasible)
                        radius = record.radius_active ? carried(record, c.error) : c.radius;
                    else
                        radius = record.ratio > .75 && record.radius_active
                            ? std::min(options.maximum_radius, 2. * c.radius) : c.radius;
                    accepted = true;
                    break;
                }
                if (stop) break;
                // Restoration takes the best verified merit decrease as soon as no trial
                // reaches feasibility: further rounds cost a whole iteration each.
                if (!feasible && fallback_from >= 0) break;
                if (lead >= 0 && corrections < options.maximum_corrections &&
                    (feasible || candidates[radii[lead].first].actual > 0.)) {
                    const auto &c = candidates[radii[lead].first];
                    if (corrections == 0) corrected = *model;
                    if (feasible) {
                        add_measurement(*c.after, c.step.x, c.predicted, c.actual, &corrected);
                        reflect(&corrected);
                    } else {
                        correct_wall_rows(*c.after, c.step.x, &corrected);
                    }
                    active = &corrected;
                    ++corrections;
                } else {
                    radius = std::max(options.minimum_radius,
                                      radius * std::pow(.5, options.speculative_trials));
                }
                record.rejected_trials += options.speculative_trials;
            }
            record.corrections = corrections;
            if (!accepted && fallback_from >= 0) {
                // Neither merit may discard its best verified improvement
                // merely because the fully qualified search ran out of time.
                std::swap(trial, fallback);
                record.radius = fallback_record.radius;
                record.step_inf = fallback_record.step_inf;
                record.radius_active = fallback_record.radius_active;
                record.predicted_reduction = fallback_record.predicted_reduction;
                record.grip_cost_predicted = fallback_record.grip_cost_predicted;
                record.actual_reduction = fallback_record.actual_reduction;
                record.ratio = fallback_record.ratio;
                record.cost_trial = fallback_record.cost_trial;
                record.violation_trial = fallback_record.violation_trial;
                record.grip_cost_trial = fallback_record.grip_cost_trial;
                record.newly_active_grip = fallback_record.newly_active_grip;
                const double error = std::abs(record.predicted_reduction - record.actual_reduction) /
                                     scale_value;
                if (feasible)
                    radius = std::clamp(std::min(.9 * record.step_inf, carried(record, error)),
                                        options.minimum_radius, options.maximum_radius);
                else
                    radius = std::clamp(record.ratio < .25 ? .25 * record.step_inf
                        : (record.ratio > .75 && record.radius_active ? 2. * record.radius
                                                                     : record.radius),
                        options.minimum_radius, options.maximum_radius);
                accepted = true;
            }
        }
        record.accepted = accepted;
        record.accurate = accepted && (!feasible || std::abs(record.predicted_reduction -
            record.actual_reduction) <= options.accuracy_tolerance * scale_value);
        if (accepted) {
            std::swap(current, trial);
            ++result->accepted_iterations;
        } else if (!stop && !exact_derivatives) {
            exact_derivatives = true;
            result->exact_derivatives = true;
            curvature_done = false;
            radius = radius_before;
            result->history.push_back(record);
            continue;
        } else if (!stop) {
            result->termination = Termination::RadiusCollapsed;
        }
        record.total_ms = elapsed_ms(begin);
        // The next iteration resembles this one without the cycle's one-off
        // curvature refresh.
        measured_iteration_us = 1150. * (record.total_ms - curvature_ms);
        result->history.push_back(record);
        if (stop || !accepted) break;
    }
    result->final_feasible = current->evaluation.feasible;
    result->final_publishable = current->evaluation.publishable;
    result->final_cost = current->evaluation.objective;
    result->final_violation = current->evaluation.violation;
    result->final_minimum_corridor_m = current->evaluation.minimum_corner;
    result->radius = radius;
    store_curvature(options, *lin);
    result->optimal = current->nominal;
    return true;
}

bool solve(const MpcCycleState_t &initial, double initial_progress,
           const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
           double lap_length, float dt,
           const MpcCycleConfiguration_t &configuration,
           const MpcModelControl_t seed[kStages], const Options &options,
           Result *result)
{
    if (!solve_from_seed(initial, initial_progress, trajectory, trajectory_count, lap_length,
                         dt, configuration, seed, options, result))
        return false;
    // The warm start can sit in a basin whose restoration stalls outside the
    // wall (the L1 correction stops at a local infeasible stationary point) while a
    // feasible plan exists: a coasting plan usually is one. Restart from it when the
    // search ended with a plan the publication check would reject, even out of time
    // (the seed is integrated whatever the clock says, which alone can yield a
    // publishable plan). A plan that misses only the clearance reserve stays: the
    // coasting plan is much slower. Every other cycle pays nothing.
    if (result->final_publishable) return true;
    MpcModelControl_t coasting[kStages];
    std::copy_n(seed, kStages, coasting);
    for (auto &control : coasting) control.throttle_command = MPC_THROTTLE_MIN_DRIVE;
    // The cached curvature belongs to the first search's plan and misdescribes the
    // coasting plan, so the restart starts with an empty cache of its own; the
    // shared one keeps the first search's unless the restart's plan is published.
    MpcCycleCurvatureCache_t restart_cache{};
    restart_cache.epoch_ns = -1;
    Options restart_options = options;
    if (options.curvature_cache) restart_options.curvature_cache = &restart_cache;
    auto restart = std::make_unique<Result>();
    if (!solve_from_seed(initial, initial_progress, trajectory, trajectory_count, lap_length,
                         dt, configuration, coasting, restart_options, restart.get()))
        return true;
    const bool better = restart->final_publishable || restart->final_violation < result->final_violation;
    const bool exact_derivatives = result->exact_derivatives || restart->exact_derivatives;
    restart->rollouts += result->rollouts;
    restart->history.insert(restart->history.begin(), result->history.begin(), result->history.end());
    result->rollouts = restart->rollouts;
    if (!better) {
        result->history = std::move(restart->history);
        result->restarted = true;
        result->exact_derivatives = exact_derivatives;
        return true;
    }
    if (options.curvature_cache) *options.curvature_cache = restart_cache;
    // The cycle's starting point is the warm start, not the restart seed.
    restart->initial_feasible = result->initial_feasible;
    restart->initial_cost = result->initial_cost;
    restart->initial_violation = result->initial_violation;
    *result = std::move(*restart);
    result->restarted = true;
    result->exact_derivatives = exact_derivatives;
    return true;
}
} // namespace sdu_apex::source_sqp

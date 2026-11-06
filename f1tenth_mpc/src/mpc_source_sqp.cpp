#include "mpc_source_sqp.hpp"
#include "mpc_source_qp.hpp"
#include "mpc_rti_internal.hpp"
#include "vehicle_model.h"

#include <Eigen/Core>
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
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sdu_apex::source_sqp {
namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Tangent = Eigen::Matrix<double, kState, kControls, Eigen::RowMajor>;

constexpr double kRearAxleToCom = 0.15532;
constexpr double kBoxTolerance = 1.e-7;
constexpr double kScale[2] = {kSteeringScale, kThrottleScale};

double scale_of(int variable) { return kScale[variable % 2]; }

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
    double slip_cost{};
    double peak_slip[kStages][4]{};
    MpcRtiSourceCorridorCorners_t corners[kStages]{};
    double box[kStages][4]{};
    double minimum_corner{};
    double minimum_box{};
    // L1 violation of the wall clearance rows.
    double violation{};
    bool feasible{};
};

struct Snapshot {
    MpcRtiNominal_t nominal;
    Evaluation evaluation;
    bool valid{};
};

struct Problem {
    const MpcRtiState_t *initial;
    double progress;
    const MpcTrajectorySample_t *trajectory;
    size_t trajectory_count;
    double lap_length;
    float dt;
    const MpcRtiCycleConfiguration_t *configuration;
};

double slip_weight(const Problem &p, int state_index)
{
    const auto &model = p.configuration->model;
    const auto &ref = p.configuration->objective_schedule->references[state_index];
    return longitudinal_slip_weight_for_reference(&model, &ref) *
           (state_index == kStages ? model.terminal_multiplier : 1.);
}

bool evaluate(const Problem &p, MpcRtiNominal_t &nominal, Evaluation *out)
{
    *out = {};
    const auto &model = p.configuration->model;
    out->minimum_box = INFINITY;
    for (int k = 0; k < kStages; ++k) {
        const auto &u = nominal.controls[k];
        const double limit = model.max_steering_rad;
        out->box[k][0] = limit - double(u.steering_target_rad);
        out->box[k][1] = limit + double(u.steering_target_rad);
        out->box[k][2] = double(MPC_THROTTLE_MAX) - double(u.throttle_command);
        out->box[k][3] = double(u.throttle_command) - double(MPC_THROTTLE_MIN_DRIVE);
        for (double v : out->box[k]) out->minimum_box = std::min(out->minimum_box, v);
        for (int wheel = 0; wheel < 4; ++wheel) {
            const double excess = nominal.states[k + 1].incoming_wheel_slip_excess_mps[wheel];
            out->peak_slip[k][wheel] = excess;
            out->slip_cost += slip_weight(p, k + 1) * excess * excess;
        }
    }
    float minimum_slack = 0.f;
    float corridor[kStages + 1][2]{};
    if (!mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
            &nominal, p.trajectory, p.trajectory_count, p.lap_length,
            p.configuration, &out->objective, &minimum_slack, corridor))
        return false;
    out->minimum_corner = INFINITY;
    double parity = 0.;
    for (int k = 1; k <= kStages; ++k) {
        auto &g = out->corners[k - 1];
        if (!mpc_rti_source_exact_corridor_corners(
                &nominal.states[k].plant, p.trajectory, p.trajectory_count,
                p.lap_length, nominal.progress[k], &model, k, &g) ||
            !g.reference_derivatives_valid)
            return false;
        double right = INFINITY, left = INFINITY;
        for (int corner = 0; corner < 4; ++corner) {
            right = std::min(right, g.right[corner]);
            left = std::min(left, g.left[corner]);
            out->violation += std::max(0., -g.right[corner]) + std::max(0., -g.left[corner]);
        }
        out->minimum_corner = std::min({out->minimum_corner, right, left});
        parity = std::max({parity, std::abs(right - double(corridor[k][0])),
                           std::abs(left - double(corridor[k][1]))});
    }
    if (parity > 1.e-9 || !std::isfinite(out->objective) ||
        !std::isfinite(out->violation))
        return false;
    out->feasible = out->minimum_corner >= 0. &&
                    out->minimum_box >= -kBoxTolerance;
    return true;
}

// Integrates `controls` through the source plant from the common initial
// state, reusing the identical already-integrated prefix of `reuse`.
bool rollout(const Problem &p, const MpcModelControl_t controls[kStages],
             const MpcRtiNominal_t *reuse, Snapshot *out)
{
    out->valid = false;
    const auto &model = p.configuration->model;
    if (!vehicle_model_set_active_target_speed_ceiling(model.active_speed_ceiling_mps))
        return false;
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
    for (int k = first_changed; k < kStages; ++k)
        if (!mpc_rti_source_exact_stage(
                &nominal.states[k], nominal.progress[k], &nominal.controls[k],
                p.trajectory, p.trajectory_count, p.lap_length, p.dt,
                &nominal.states[k + 1], &nominal.progress[k + 1]))
            return false;
    nominal.valid = 1;
    MpcRtiReference_t refs[kStages + 1]{};
    if (!mpc_rti_references_from_exact_nominal(
            &nominal, p.trajectory, p.trajectory_count, p.lap_length, &model, refs))
        return false;
    for (int k = 0; k <= kStages; ++k) nominal.speed_reference[k] = refs[k].u;
    nominal.speed_reference_valid = 1;
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
    // Signed maximum slip residual over the stage's substeps and its tangent.
    double slip_residual[4]{};
    double slip_state[4][kState]{};
    double slip_control[4][2]{};
};

struct Linearization {
    std::array<Stage, kStages> stages;
    std::array<Tangent, kStages> sensitivity;  // d state[k+1] / d scaled controls
};

enum class LinearizeStatus { Ok, Failed, Deadline };

LinearizeStatus linearize(const Problem &p, const MpcRtiNominal_t &trace,
                          int threads, double deadline_us, Linearization *out)
{
    constexpr int L = UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;
    static_assert(kPhysical + 2 <= L, "one sensitivity batch must cover all seeds");
    std::atomic<bool> failed{false}, late{false};
    const auto stage = [&](int k) {
        if (failed.load(std::memory_order_relaxed)) return;
        if (deadline_us > 0. && mpc_rti_monotonic_time_us() >= deadline_us) {
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
        for (int i = 0; i < kPhysical; ++i) seed[i][i] = 1.;
        control[0][kPhysical] = 1.;
        control[1][kPhysical + 1] = 1.;
        double physical[kPhysical][L]{}, body[3][L]{}, residual[4]{}, peak[4][L]{};
        if (!vehicle_model_full_source_interval_sensitivities(
                &previous.plant, &trace.controls[k], p.dt, seed, control,
                physical, body, residual, peak)) {
            failed.store(true);
            return;
        }
        const auto &g = exact.incoming_tracking_jacobian;
        auto &s = out->stages[k];
        s = {};
        for (int column = 0; column < kPhysical + 2; ++column) {
            for (int state = 0; state < kPhysical; ++state) {
                if (column < kPhysical) s.A[state][column] = physical[state][column];
                else s.B[state][column - kPhysical] = physical[state][column];
            }
            for (int row = 0; row < 3; ++row) {
                double tangent = 0.;
                for (int i = 0; i < 3; ++i) tangent += g[row][i + 2] * body[i][column];
                if (column < kPhysical) s.A[kPhysical + row][column] = tangent;
                else s.B[kPhysical + row][column - kPhysical] = tangent;
            }
            for (int wheel = 0; wheel < 4; ++wheel) {
                if (column < kPhysical) s.slip_state[wheel][column] = peak[wheel][column];
                else s.slip_control[wheel][column - kPhysical] = peak[wheel][column];
            }
        }
        for (int row = 0; row < 3; ++row) {
            s.A[kPhysical + row][kPhysical] = g[row][0];
            s.A[kPhysical + row][kPhysical + 1] = g[row][1];
            s.A[kPhysical + row][kPhysical + 2] = g[row][5];
        }
        for (int wheel = 0; wheel < 4; ++wheel) s.slip_residual[wheel] = residual[wheel];
    };
#ifdef _OPENMP
    const int workers = std::min({8, omp_get_max_threads(), std::max(1, threads)});
    if (workers > 1) {
#pragma omp parallel for num_threads(workers) schedule(dynamic, 1)
        for (int k = 0; k < kStages; ++k) stage(k);
    } else
#endif
    {
        for (int k = 0; k < kStages; ++k) stage(k);
    }
    if (failed.load()) return LinearizeStatus::Failed;
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

// ---------------------------------------------------------------------------
// Local model: quadratic racing residuals + slip epigraph + linear rows.
// ---------------------------------------------------------------------------
using Row = Eigen::Matrix<double, kControls, 1>;

struct SlipRow {
    int stage;       // interval index k (state k+1)
    int wheel;
    double weight;   // cost = weight * max(0, residual + slope.x)^2
    double residual; // signed peak residual at the iterate
    double clipped;  // max(0, residual)
    Row slope;
};

struct PhysicalRow {
    double margin;   // >= 0 is legal; the linearization is margin + slope.x
    Row slope;
};

struct Model {
    Matrix hessian;  // Gauss-Newton curvature of the quadratic residuals
    Vector gradient;
    std::vector<PhysicalRow> physical;
    std::vector<SlipRow> slip;
    Vector lower, upper;  // legal command interval, as a step
};

Row state_feature(const Linearization &lin, int interval, int state_index)
{
    return lin.sensitivity[interval].row(state_index).transpose();
}

bool build_model(const Problem &p, const Snapshot &at, const Linearization &lin,
                 Model *model)
{
    const auto &cfg = p.configuration->model;
    const auto &schedule = *p.configuration->objective_schedule;
    const auto &nominal = at.nominal;
    const auto &evaluation = at.evaluation;
    model->hessian = Matrix::Zero(kControls, kControls);
    model->gradient = Vector::Zero(kControls);
    model->physical.clear();
    model->slip.clear();

    const auto residual_term = [&](double weight, double residual, const Row &feature) {
        if (weight <= 0.) return true;
        if (!std::isfinite(weight) || !std::isfinite(residual) || !feature.allFinite())
            return false;
        model->hessian.noalias() += (2. * weight) * feature * feature.transpose();
        model->gradient.noalias() += (2. * weight * residual) * feature;
        return true;
    };
    const auto control_feature = [&](int stage, int axis) {
        Row feature = Row::Zero();
        feature(2 * stage + axis) = kScale[axis];
        return feature;
    };

    for (int k = 1; k <= kStages; ++k) {
        const auto &plant = nominal.states[k].plant;
        const auto &ref = schedule.references[k];
        const double terminal = k == kStages ? cfg.terminal_multiplier : 1.;
        const int interval = k - 1;
        if (!residual_term(terminal * cfg.weight_e_y, double(plant.e_y) - ref.e_y,
                           state_feature(lin, interval, kPhysical)) ||
            !residual_term(terminal * cfg.weight_e_psi, double(plant.e_psi) - ref.e_psi,
                           state_feature(lin, interval, kPhysical + 1)) ||
            !residual_term(terminal * schedule.speed_weights[k], double(plant.u) - ref.u,
                           state_feature(lin, interval, UNITY_SOURCE_U)) ||
            !residual_term(terminal * cfg.weight_v, double(plant.v) - ref.v,
                           state_feature(lin, interval, UNITY_SOURCE_V) -
                               kRearAxleToCom * state_feature(lin, interval, UNITY_SOURCE_YAW_RATE)) ||
            !residual_term(terminal * cfg.weight_r, double(plant.r) - ref.r,
                           state_feature(lin, interval, UNITY_SOURCE_YAW_RATE)) ||
            !residual_term(terminal * cfg.weight_steering_command,
                           double(plant.actual_steering_angle) - ref.steering_command,
                           state_feature(lin, interval, UNITY_SOURCE_PHYSICAL_STEERING)) ||
            !residual_term(terminal * cfg.weight_steering_command,
                           double(plant.delayed_steering_command_2) - ref.steering_command,
                           control_feature(interval, 0)))
            return false;
    }
    double previous_steering = nominal.states[0].plant.delayed_steering_command_2;
    double previous_throttle = nominal.states[0].plant.delayed_throttle_command_2;
    for (int k = 0; k < kStages; ++k) {
        const auto &ref = schedule.references[k];
        const auto &u = nominal.controls[k];
        const double increment = steering_reference_increment(&ref, &schedule.references[k + 1]);
        Row steering_slew = control_feature(k, 0);
        Row throttle_slew = control_feature(k, 1);
        if (k > 0) {
            steering_slew -= control_feature(k - 1, 0);
            throttle_slew -= control_feature(k - 1, 1);
        }
        if (!residual_term(cfg.weight_steering_target,
                           double(u.steering_target_rad) - ref.steering_command,
                           control_feature(k, 0)) ||
            !residual_term(cfg.weight_throttle_command, u.throttle_command,
                           control_feature(k, 1)) ||
            !residual_term(cfg.weight_steering_slew,
                           double(u.steering_target_rad) - previous_steering - increment,
                           steering_slew) ||
            !residual_term(cfg.weight_throttle_slew,
                           double(u.throttle_command) - previous_throttle, throttle_slew))
            return false;
        previous_steering = u.steering_target_rad;
        previous_throttle = u.throttle_command;
    }

    // Longitudinal slip: one cut per wheel and stage on the signed peak residual.
    for (int k = 0; k < kStages; ++k) {
        const auto &stage = lin.stages[k];
        const double weight = slip_weight(p, k + 1);
        if (weight <= 0.) continue;
        for (int wheel = 0; wheel < 4; ++wheel) {
            SlipRow row;
            row.stage = k;
            row.wheel = wheel;
            row.weight = weight;
            row.residual = stage.slip_residual[wheel];
            row.clipped = std::max(0., row.residual);
            row.slope = Row::Zero();
            if (k > 0)
                for (int s = 0; s < kPhysical; ++s)
                    row.slope.noalias() += stage.slip_state[wheel][s] *
                                           lin.sensitivity[k - 1].row(s).transpose();
            row.slope(2 * k) += stage.slip_control[wheel][0] * kSteeringScale;
            row.slope(2 * k + 1) += stage.slip_control[wheel][1] * kThrottleScale;
            if (!row.slope.allFinite() || !std::isfinite(row.residual)) return false;
            model->slip.push_back(std::move(row));
        }
    }

    for (int k = 0; k < kStages; ++k) {
        const auto &g = evaluation.corners[k];
        for (int side = 0; side < 2; ++side)
            for (int corner = 0; corner < 4; ++corner) {
                const bool left = side == 0;
                PhysicalRow row;
                row.margin = left ? g.left[corner] : g.right[corner];
                const double heading = left ? g.left_d_psi[corner] : g.right_d_psi[corner];
                const double path = left ? g.left_d_progress_forward : g.right_d_progress_forward;
                row.slope = (left ? -1. : 1.) * state_feature(lin, k, kPhysical) +
                            heading * state_feature(lin, k, kPhysical + 1) +
                            path * state_feature(lin, k, kPhysical + 2);
                if (!row.slope.allFinite()) return false;
                model->physical.push_back(std::move(row));
            }
    }
    // The legal command interval: the only restriction on the step besides
    // the physical rows.
    model->lower = Vector(kControls);
    model->upper = Vector(kControls);
    for (int k = 0; k < kStages; ++k)
        for (int axis = 0; axis < 2; ++axis) {
            const int j = 2 * k + axis;
            model->upper(j) = std::max(0., evaluation.box[k][2 * axis]) / kScale[axis];
            model->lower(j) = -std::max(0., evaluation.box[k][2 * axis + 1]) / kScale[axis];
        }
    return model->hessian.allFinite() && model->gradient.allFinite();
}

double slip_model_cost(const Model &model, const Vector &x)
{
    double cost = 0.;
    for (const auto &row : model.slip) {
        const double t = std::max(0., row.residual + row.slope.dot(x));
        cost += row.weight * t * t;
    }
    return cost;
}

double nominal_slip_cost(const Model &model)
{
    double cost = 0.;
    for (const auto &row : model.slip) cost += row.weight * row.clipped * row.clipped;
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
    // Model change of the quantity being reduced (cost while feasible,
    // violation otherwise) and its slope along the step at alpha = 0.
    double predicted_reduction{};
    double slope{};
    double model_slip_cost{};
};

// Solves the local model exactly inside the radius and the legal command box. While the iterate is feasible the physical
// rows are hard; x = 0 satisfies them, so the QP is always feasible.
// Otherwise they enter as an exact L1 penalty. Rows enter the QP lazily:
// the solve is repeated until every omitted row is satisfied (slip rows:
// contributes nothing) at the solution, so the result equals the solution
// of the complete model.
bool solve_step(const Model &model, bool feasible, const Options &options,
                double radius, Step *step)
{
    const auto begin = std::chrono::steady_clock::now();
    constexpr double kInitialReach = 0.25;  // rows this close are started with
    const int n = kControls;
    std::vector<char> physical_in(model.physical.size(), 0), slip_in(model.slip.size(), 0);
    for (size_t i = 0; i < model.physical.size(); ++i)
        physical_in[i] = model.physical[i].margin < kInitialReach;
    for (size_t i = 0; i < model.slip.size(); ++i)
        slip_in[i] = model.slip[i].residual > -kInitialReach;
    const Vector lower = model.lower.cwiseMax(-radius), upper = model.upper.cwiseMin(radius);
    const double tolerance =
        std::max(1.e-4, std::min(1.e-2, 1.e-4 * model.gradient.lpNorm<Eigen::Infinity>()));
    for (int round = 0; round < 12; ++round) {
        int physical = 0, slip = 0;
        for (char in : physical_in) physical += in;
        for (char in : slip_in) slip += in;
        const int hard = feasible ? physical : 0;
        const int elastic = slip + (feasible ? 0 : physical);
        ConvexQp qp;
        qp.hessian = model.hessian;
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
        for (size_t i = 0; i < model.slip.size(); ++i) {
            if (!slip_in[i]) continue;
            // t >= residual + slope.x  <=>  -slope.x - residual + t >= 0
            qp.elastic_jacobian.row(e) = -model.slip[i].slope.transpose();
            qp.elastic_margin(e) = -model.slip[i].residual;
            qp.elastic_weight(e++) = 2. * model.slip[i].weight;
        }
        for (size_t i = 0; i < model.physical.size(); ++i) {
            if (!physical_in[i]) continue;
            const auto &row = model.physical[i];
            if (feasible) {
                qp.hard_jacobian.row(h) = row.slope.transpose();
                qp.hard_margin(h++) = row.margin;
            } else {
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
        for (size_t i = 0; i < model.slip.size(); ++i)
            if (!slip_in[i] &&
                model.slip[i].residual + model.slip[i].slope.dot(step->x) > 0.) {
                slip_in[i] = 1;
                complete = false;
            }
        if (complete) break;
        if (options.deadline_us > 0. && mpc_rti_monotonic_time_us() >= options.deadline_us) {
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
    step->model_slip_cost = slip_model_cost(model, step->x);
    if (feasible) {
        step->predicted_reduction =
            -(model.gradient.dot(step->x) + 0.5 * step->x.dot(model.hessian * step->x) +
              step->model_slip_cost - nominal_slip_cost(model));
        // Directional derivative of the true cost at alpha = 0 (right-hand
        // derivative of each slip hinge).
        double slope = model.gradient.dot(step->x);
        for (const auto &row : model.slip)
            if (row.residual > 0.) slope += 2. * row.weight * row.residual * row.slope.dot(step->x);
        step->slope = slope;
    } else {
        step->predicted_reduction = 0.;
        // violation_before is supplied by the caller; slope = linearized change.
        step->slope = 0.;
    }
    return true;
}

void controls_after_step(const Snapshot &at, const Vector &x, double alpha,
                         const MpcRtiConfiguration_t &model,
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

int newly_active_slip(const Evaluation &before, const Evaluation &after)
{
    int count = 0;
    for (int k = 0; k < kStages; ++k)
        for (int wheel = 0; wheel < 4; ++wheel)
            count += before.peak_slip[k][wheel] <= 1.e-8 && after.peak_slip[k][wheel] > 1.e-5;
    return count;
}
} // namespace

bool solve(const MpcRtiState_t &initial, double initial_progress,
           const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
           double lap_length, float dt,
           const MpcRtiCycleConfiguration_t &configuration,
           const MpcModelControl_t seed[kStages], const Options &options,
           Result *result)
{
    if (!result || !trajectory || trajectory_count == 0 || !seed ||
        !configuration.objective_schedule || !configuration.objective_schedule->valid ||
        configuration.objective_schedule->horizon != kStages ||
        options.maximum_iterations < 1 ||
        !(options.minimum_radius > 0.) || options.minimum_radius > options.maximum_radius ||
        !(options.restoration_radius > 0.))
        return false;
    *result = {};
    const Problem problem{&initial, initial_progress, trajectory, trajectory_count,
                          lap_length, dt, &configuration};
    auto current = std::make_unique<Snapshot>();
    auto trial = std::make_unique<Snapshot>();
    auto lin = std::make_unique<Linearization>();
    auto model = std::make_unique<Model>();
    if (!rollout(problem, seed, options.same_epoch_nominal, current.get())) return false;
    ++result->rollouts;
    result->initial_feasible = current->evaluation.feasible;
    result->initial_cost = current->evaluation.objective;
    result->initial_violation = current->evaluation.violation;

    // The radius is the extent over which the exact rollout last confirmed the
    // model. Restoring wall feasibility is judged on the exact violation, so
    // it is not held to a radius earned on racing-cost steps.
    double radius = std::clamp(options.radius, options.minimum_radius, options.maximum_radius);
    if (!current->evaluation.feasible)
        radius = std::max(radius, std::min(options.restoration_radius, options.maximum_radius));
    double measured_iteration_us = options.minimum_iteration_us;
    double qp_ms_seen = 0., rollout_ms_seen = 0.;
    const auto late = [&](double needed_us) {
        return options.deadline_us > 0. &&
               mpc_rti_monotonic_time_us() + needed_us >= options.deadline_us;
    };
    result->termination = Termination::IterationLimit;
    for (int iteration = 0; iteration < options.maximum_iterations; ++iteration) {
        if (late(measured_iteration_us)) {
            result->termination = Termination::Deadline;
            break;
        }
        const auto begin = std::chrono::steady_clock::now();
        Iteration record;
        const bool feasible = current->evaluation.feasible;
        record.feasible_before = feasible;
        record.cost_before = current->evaluation.objective;
        record.violation_before = current->evaluation.violation;
        record.slip_cost_before = current->evaluation.slip_cost;

        const auto lin_begin = std::chrono::steady_clock::now();
        const LinearizeStatus status = linearize(
            problem, current->nominal, configuration.model.jacobian_threads,
            options.deadline_us, lin.get());
        record.linearization_ms = elapsed_ms(lin_begin);
        if (status != LinearizeStatus::Ok) {
            result->termination = status == LinearizeStatus::Deadline
                                      ? Termination::Deadline : Termination::LinearizationFailed;
            break;
        }
        if (!build_model(problem, *current, *lin, model.get())) {
            result->termination = Termination::LinearizationFailed;
            break;
        }
        const double origin = feasible ? current->evaluation.objective
                                       : current->evaluation.violation;
        const double scale_value = std::max(1., origin);
        bool accepted = false, stop = false;
        // The linearization is reused while the radius is reduced: the model
        // is the same, only the region it is trusted over changes.
        for (int attempt = 0; attempt < options.maximum_attempts && !accepted && !stop; ++attempt) {
            if (late(1000. * (qp_ms_seen + rollout_ms_seen))) {
                result->termination = Termination::Deadline;
                stop = true;
                break;
            }
            Step step;
            const bool solved = solve_step(*model, feasible, options, radius, &step);
            record.qp_ms += step.qp_ms;
            record.qp_iterations += step.iterations;
            record.qp_rounds = std::max(record.qp_rounds, step.rounds);
            qp_ms_seen = std::max(qp_ms_seen, step.qp_ms);
            if (!solved) {
                if (step.deadline) {
                    result->termination = Termination::Deadline;
                    stop = true;
                } else if (radius <= options.minimum_radius) {
                    result->termination = Termination::QpFailed;
                    stop = true;
                }
                radius = std::max(options.minimum_radius, .25 * radius);
                ++record.rejected_trials;
                continue;
            }
            const double predicted = feasible
                ? step.predicted_reduction
                : current->evaluation.violation - linearized_violation(*model, step.x);
            const double step_inf = step.x.lpNorm<Eigen::Infinity>();
            record.radius = radius;
            record.step_inf = step_inf;
            record.radius_active = step_inf >= 0.99 * radius;
            record.predicted_reduction = predicted;
            record.slip_cost_predicted = step.model_slip_cost;
            if (predicted <= 1.e-9 * scale_value) {
                // Nothing is left that the local model can improve.
                result->termination = Termination::Converged;
                result->converged = feasible;
                stop = true;
                break;
            }
            if (late(1000. * rollout_ms_seen)) {
                result->termination = Termination::Deadline;
                stop = true;
                break;
            }
            MpcModelControl_t controls[kStages];
            controls_after_step(*current, step.x, 1., configuration.model, controls);
            const auto roll_begin = std::chrono::steady_clock::now();
            const bool evaluated = rollout(problem, controls, &current->nominal, trial.get());
            const double roll_ms = elapsed_ms(roll_begin);
            record.rollout_ms += roll_ms;
            rollout_ms_seen = std::max(rollout_ms_seen, roll_ms);
            ++result->rollouts;
            double actual = -INFINITY;
            if (evaluated) {
                const auto &after = trial->evaluation;
                record.cost_trial = after.objective;
                record.violation_trial = after.violation;
                record.slip_cost_trial = after.slip_cost;
                record.newly_active_slip = newly_active_slip(current->evaluation, after);
                if (feasible) actual = after.feasible ? origin - after.objective : -INFINITY;
                else actual = after.feasible ? predicted : origin - after.violation;
            }
            record.actual_reduction = actual;
            record.ratio = actual / predicted;
            accepted = evaluated && actual > options.sufficient_decrease * predicted &&
                       actual > 32. * std::numeric_limits<double>::epsilon() * scale_value;
            if (accepted && record.ratio > 0.75 && record.radius_active)
                radius = std::min(options.maximum_radius, 2. * radius);
            else if (!accepted || record.ratio < 0.25)
                radius = std::max(options.minimum_radius, .25 * std::min(radius, step_inf));
            if (!accepted) ++record.rejected_trials;
        }
        record.accepted = accepted;
        if (accepted) {
            std::swap(current, trial);
            ++result->accepted_iterations;
        } else if (!stop) {
            result->termination = Termination::RadiusCollapsed;
        }
        record.total_ms = elapsed_ms(begin);
        measured_iteration_us = 1150. * record.total_ms;  // the next one resembles this one
        result->history.push_back(record);
        if (stop || !accepted) break;
    }
    result->final_feasible = current->evaluation.feasible;
    result->final_cost = current->evaluation.objective;
    result->final_violation = current->evaluation.violation;
    result->final_minimum_corridor_m = current->evaluation.minimum_corner;
    result->radius = radius;
    result->optimal = current->nominal;
    return true;
}
} // namespace sdu_apex::source_sqp

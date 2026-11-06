#include "mpc_source_qp.hpp"
#include "mpc_rti.h"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>

namespace sdu_apex::source_sqp {
namespace {
using V = Eigen::VectorXd;
using M = Eigen::MatrixXd;

double inf(const V &v) { return v.size() ? v.lpNorm<Eigen::Infinity>() : 0.; }

// Largest alpha <= 1 keeping s + alpha*ds >= (1 - safety) * s.
double boundary_step(const V &s, const V &ds, double safety)
{
    double alpha = 1.;
    for (Eigen::Index i = 0; i < s.size(); ++i)
        if (ds(i) < 0.) alpha = std::min(alpha, -safety * s(i) / ds(i));
    return alpha;
}

// Every inequality block of the primal-dual iterate: slack s and multiplier l.
struct Block {
    V s, l;
    Block() = default;
    explicit Block(const V &slack) : s(slack), l(V::Ones(slack.size())) {}
    double products(V *out) const { *out = s.cwiseProduct(l); return out->size() ? out->sum() : 0.; }
};
} // namespace

bool solve_convex_qp(const ConvexQp &p, int maximum_iterations, double tolerance,
                     ConvexQpResult *out, double absolute_deadline_monotonic_us)
{
    const int n = static_cast<int>(p.gradient.size());
    const int mh = static_cast<int>(p.hard_margin.size());
    const int me = static_cast<int>(p.elastic_margin.size());
    if (!out || n < 1 || maximum_iterations < 1 || !(tolerance > 0.) ||
        !std::isfinite(tolerance) || p.hessian.rows() != n || p.hessian.cols() != n ||
        p.lower.size() != n || p.upper.size() != n || p.hard_jacobian.rows() != mh ||
        (mh && p.hard_jacobian.cols() != n) || p.elastic_jacobian.rows() != me ||
        (me && p.elastic_jacobian.cols() != n) || p.elastic_weight.size() != me ||
        p.elastic_linear_cost.size() != me || !p.hessian.allFinite() ||
        !p.gradient.allFinite() || !p.lower.allFinite() || !p.upper.allFinite() ||
        !p.hard_jacobian.allFinite() || !p.hard_margin.allFinite() ||
        !p.elastic_jacobian.allFinite() || !p.elastic_margin.allFinite() ||
        !p.elastic_weight.allFinite() || !p.elastic_linear_cost.allFinite() ||
        (me && (p.elastic_weight.minCoeff() < 0. || p.elastic_linear_cost.minCoeff() < 0.)) ||
        (p.upper - p.lower).minCoeff() < 0.)
        return false;
    *out = {};

    // Scaling changes neither the minimizer nor the physical multipliers.
    const double objective_scale = std::max(
        {1., inf(p.gradient), p.hessian.diagonal().cwiseAbs().maxCoeff(),
         inf(p.elastic_weight), inf(p.elastic_linear_cost)});
    const M H = p.hessian / objective_scale;
    const V g = p.gradient / objective_scale;
    const V w = p.elastic_weight / objective_scale;
    const V c = p.elastic_linear_cost / objective_scale;
    M A = p.hard_jacobian;
    V b = p.hard_margin;
    for (int i = 0; i < mh; ++i) {
        const double scale = 1. / std::max({1.e-3, A.row(i).norm(), std::abs(b(i))});
        A.row(i) *= scale;
        b(i) *= scale;
    }
    const M &C = p.elastic_jacobian;
    const V &d = p.elastic_margin;
    // A zero-width interval has no interior; give it a hair of width.
    V lo = p.lower, hi = p.upper;
    for (int j = 0; j < n; ++j)
        if (hi(j) - lo(j) < 1.e-9) {
            lo(j) -= 5.e-10;
            hi(j) += 5.e-10;
        }

    V x(n);
    for (int j = 0; j < n; ++j) {
        const double margin = std::min(.1, .25 * (hi(j) - lo(j)));
        x(j) = std::clamp(0., lo(j) + margin, hi(j) - margin);
    }
    V t = (-(C * x + d)).cwiseMax(0.);
    Block hard(V(b + A * x).cwiseMax(1.)), below(x - lo), above(hi - x),
        row((C * x + d + t).cwiseMax(1.)), sign(t.cwiseMax(1.));
    const int count = mh + 2 * n + 2 * me;

    M normal(n, n);
    V products;
    for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
        if (absolute_deadline_monotonic_us > 0. &&
            mpc_rti_monotonic_time_us() >= absolute_deadline_monotonic_us) {
            out->deadline_exceeded = true;
            break;
        }
        out->iterations = iteration + 1;
        const V Ax = mh ? V(A * x) : V(0), Cx = me ? V(C * x) : V(0);
        const V rx = H * x + g - (mh ? V(A.transpose() * hard.l) : V::Zero(n)) - below.l +
                     above.l - (me ? V(C.transpose() * row.l) : V::Zero(n));
        const V rt = w.cwiseProduct(t) + c - row.l - sign.l;
        const V ph = Ax + b - hard.s, pl = x - lo - below.s, pu = hi - x - above.s,
                pr = Cx + d + t - row.s, pt = t - sign.s;
        double total = hard.products(&products);
        total += below.products(&products) + above.products(&products) +
                 row.products(&products) + sign.products(&products);
        const double mu = total / count;
        const double complementarity =
            std::max({inf(hard.s.cwiseProduct(hard.l)), inf(below.s.cwiseProduct(below.l)),
                      inf(above.s.cwiseProduct(above.l)), inf(row.s.cwiseProduct(row.l)),
                      inf(sign.s.cwiseProduct(sign.l))}) * objective_scale;
        const double primal = std::max({inf(ph), inf(pl), inf(pu), inf(pr), inf(pt)});
        if (inf(rx) * objective_scale <= tolerance && inf(rt) * objective_scale <= tolerance &&
            complementarity <= tolerance && primal <= 1.e-8) {
            out->converged = true;
            break;
        }

        const V Dh = hard.l.cwiseQuotient(hard.s), Dl = below.l.cwiseQuotient(below.s),
                Du = above.l.cwiseQuotient(above.s), Dr = row.l.cwiseQuotient(row.s),
                Dt = sign.l.cwiseQuotient(sign.s);
        const V T = w + Dr + Dt;
        const V kappa = Dr.cwiseProduct(w + Dt).cwiseQuotient(T);
        normal = H;
        normal.diagonal() += Dl + Du;
        if (mh + me) {
            M weighted(mh + me, n);
            if (mh) weighted.topRows(mh) = Dh.cwiseSqrt().asDiagonal() * A;
            if (me) weighted.bottomRows(me) = kappa.cwiseSqrt().asDiagonal() * C;
            normal.selfadjointView<Eigen::Lower>().rankUpdate(weighted.transpose());
        }
        // Symmetric equilibration protects nearly parallel active rows.
        const V equilibration =
            normal.diagonal().cwiseMax(1.e-300).cwiseSqrt().cwiseInverse();
        M scaled = normal.selfadjointView<Eigen::Lower>();
        scaled = scaled.cwiseProduct(equilibration * equilibration.transpose());
        Eigen::LLT<M> factor(scaled);
        if (factor.info() != Eigen::Success) break;
        const auto solve = [&](const V &rhs) -> V {
            return equilibration.asDiagonal() *
                   factor.solve(equilibration.asDiagonal() * rhs);
        };

        struct Direction {
            V x, t, sh, sl, su, sr, st, lh, ll, lu, lr, lt;
        };
        const auto newton = [&](double target, const Direction *cross) -> Direction {
            const auto q = [&](const Block &blk, const V &residual, const V *cs,
                               const V *cl) {
                V r = V::Constant(blk.s.size(), target) - blk.s.cwiseProduct(blk.l) -
                      blk.l.cwiseProduct(residual);
                if (cs) r -= cs->cwiseProduct(*cl);
                return r;
            };
            const V qh = q(hard, ph, cross ? &cross->sh : nullptr, cross ? &cross->lh : nullptr),
                    ql = q(below, pl, cross ? &cross->sl : nullptr, cross ? &cross->ll : nullptr),
                    qu = q(above, pu, cross ? &cross->su : nullptr, cross ? &cross->lu : nullptr),
                    qr = q(row, pr, cross ? &cross->sr : nullptr, cross ? &cross->lr : nullptr),
                    qt = q(sign, pt, cross ? &cross->st : nullptr, cross ? &cross->lt : nullptr);
            const V qh_s = qh.cwiseQuotient(hard.s), qr_s = qr.cwiseQuotient(row.s);
            V rhs = -rx + ql.cwiseQuotient(below.s) - qu.cwiseQuotient(above.s);
            if (mh) rhs += A.transpose() * qh_s;
            if (me) rhs += C.transpose() * qr_s;
            const V rhs_t = -rt + qr_s + qt.cwiseQuotient(sign.s);
            if (me) rhs -= C.transpose() * Dr.cwiseProduct(rhs_t).cwiseQuotient(T);
            Direction dir;
            dir.x = solve(rhs);
            const V Cdx = me ? V(C * dir.x) : V(0), Adx = mh ? V(A * dir.x) : V(0);
            dir.t = (rhs_t - Dr.cwiseProduct(Cdx)).cwiseQuotient(T);
            dir.sh = Adx + ph;
            dir.sl = dir.x + pl;
            dir.su = -dir.x + pu;
            dir.sr = Cdx + dir.t + pr;
            dir.st = dir.t + pt;
            dir.lh = qh_s - Dh.cwiseProduct(Adx);
            dir.ll = ql.cwiseQuotient(below.s) - Dl.cwiseProduct(dir.x);
            dir.lu = qu.cwiseQuotient(above.s) + Du.cwiseProduct(dir.x);
            dir.lr = qr_s - Dr.cwiseProduct(Cdx + dir.t);
            dir.lt = qt.cwiseQuotient(sign.s) - Dt.cwiseProduct(dir.t);
            return dir;
        };
        const auto lengths = [&](const Direction &dir, double safety) {
            return std::pair{
                std::min({boundary_step(hard.s, dir.sh, safety), boundary_step(below.s, dir.sl, safety),
                          boundary_step(above.s, dir.su, safety), boundary_step(row.s, dir.sr, safety),
                          boundary_step(sign.s, dir.st, safety)}),
                std::min({boundary_step(hard.l, dir.lh, safety), boundary_step(below.l, dir.ll, safety),
                          boundary_step(above.l, dir.lu, safety), boundary_step(row.l, dir.lr, safety),
                          boundary_step(sign.l, dir.lt, safety)})};
        };

        const Direction affine = newton(0., nullptr);
        if (!affine.x.allFinite() || !affine.t.allFinite()) break;
        const auto [affine_primal, affine_dual] = lengths(affine, 1.);
        const double mu_affine =
            ((hard.s + affine_primal * affine.sh).dot(hard.l + affine_dual * affine.lh) +
             (below.s + affine_primal * affine.sl).dot(below.l + affine_dual * affine.ll) +
             (above.s + affine_primal * affine.su).dot(above.l + affine_dual * affine.lu) +
             (row.s + affine_primal * affine.sr).dot(row.l + affine_dual * affine.lr) +
             (sign.s + affine_primal * affine.st).dot(sign.l + affine_dual * affine.lt)) / count;
        const double sigma = std::pow(std::clamp(mu_affine / std::max(1.e-300, mu), 0., 1.), 3.);
        const Direction step = newton(sigma * mu, &affine);
        if (!step.x.allFinite() || !step.t.allFinite() || !step.lh.allFinite() ||
            !step.lr.allFinite() || !step.ll.allFinite() || !step.lu.allFinite() ||
            !step.lt.allFinite())
            break;
        const auto [alpha_primal, alpha_dual] = lengths(step, .995);
        x += alpha_primal * step.x;
        t += alpha_primal * step.t;
        hard.s += alpha_primal * step.sh;
        below.s += alpha_primal * step.sl;
        above.s += alpha_primal * step.su;
        row.s += alpha_primal * step.sr;
        sign.s += alpha_primal * step.st;
        hard.l += alpha_dual * step.lh;
        below.l += alpha_dual * step.ll;
        above.l += alpha_dual * step.lu;
        row.l += alpha_dual * step.lr;
        sign.l += alpha_dual * step.lt;
        if (!x.allFinite() || !t.allFinite()) break;
    }
    out->step = x.cwiseMax(p.lower).cwiseMin(p.upper);
    return true;
}
} // namespace sdu_apex::source_sqp

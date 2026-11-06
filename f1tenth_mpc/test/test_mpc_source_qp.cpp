#include "mpc_source_qp.hpp"

#include <gtest/gtest.h>

using sdu_apex::source_sqp::ConvexQp;
using sdu_apex::source_sqp::ConvexQpResult;
using sdu_apex::source_sqp::solve_convex_qp;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
ConvexQp unconstrained(const VectorXd &diagonal, const VectorXd &gradient)
{
    const int n = static_cast<int>(diagonal.size());
    ConvexQp qp;
    qp.hessian = MatrixXd(diagonal.asDiagonal());
    qp.gradient = gradient;
    qp.lower = VectorXd::Constant(n, -100.);
    qp.upper = VectorXd::Constant(n, 100.);
    qp.hard_jacobian.resize(0, n);
    qp.hard_margin.resize(0);
    qp.elastic_jacobian.resize(0, n);
    qp.elastic_margin.resize(0);
    qp.elastic_weight.resize(0);
    qp.elastic_linear_cost.resize(0);
    return qp;
}
} // namespace

TEST(SourceQp, UnconstrainedMinimizerIsNewtonStep)
{
    ConvexQp qp = unconstrained(Eigen::Vector2d(2., 8.), Eigen::Vector2d(-4., 8.));
    ConvexQpResult result;
    ASSERT_TRUE(solve_convex_qp(qp, 60, 1.e-9, &result));
    ASSERT_TRUE(result.converged);
    EXPECT_NEAR(result.step(0), 2., 1.e-6);
    EXPECT_NEAR(result.step(1), -1., 1.e-6);
}

TEST(SourceQp, BoundsActiveAtTheirLimit)
{
    ConvexQp qp = unconstrained(Eigen::Vector2d(1., 1.), Eigen::Vector2d(-10., 0.5));
    qp.lower = Eigen::Vector2d(-1., -1.);
    qp.upper = Eigen::Vector2d(0.5, 1.);
    ConvexQpResult result;
    ASSERT_TRUE(solve_convex_qp(qp, 60, 1.e-9, &result));
    ASSERT_TRUE(result.converged);
    EXPECT_NEAR(result.step(0), 0.5, 1.e-6);
    EXPECT_NEAR(result.step(1), -0.5, 1.e-6);
}

TEST(SourceQp, HardRowIsRespected)
{
    // minimize 1/2 |x|^2 - [3 3]x  subject to  x0 + x1 <= 2  (2 - x0 - x1 >= 0)
    ConvexQp qp = unconstrained(Eigen::Vector2d(1., 1.), Eigen::Vector2d(-3., -3.));
    qp.hard_jacobian = MatrixXd(1, 2);
    qp.hard_jacobian << -1., -1.;
    qp.hard_margin = VectorXd::Constant(1, 2.);
    ConvexQpResult result;
    ASSERT_TRUE(solve_convex_qp(qp, 60, 1.e-9, &result));
    ASSERT_TRUE(result.converged);
    EXPECT_NEAR(result.step(0), 1., 1.e-6);
    EXPECT_NEAR(result.step(1), 1., 1.e-6);
}

TEST(SourceQp, ElasticRowWithoutLinearCostIsAnExactSquaredHinge)
{
    // minimize 1/2 x^2 - x + 1/2 w max(0, x - 0.5)^2 with the row
    // t >= x - 0.5 written as  -x + 0.5 + t >= 0.  For x > 0.5 the optimum is
    // (1 + 0.5 w) / (1 + w).
    const double w = 3.;
    ConvexQp qp = unconstrained(VectorXd::Constant(1, 1.), VectorXd::Constant(1, -1.));
    qp.elastic_jacobian = MatrixXd::Constant(1, 1, -1.);
    qp.elastic_margin = VectorXd::Constant(1, 0.5);
    qp.elastic_weight = VectorXd::Constant(1, w);
    qp.elastic_linear_cost = VectorXd::Zero(1);
    ConvexQpResult result;
    ASSERT_TRUE(solve_convex_qp(qp, 80, 1.e-9, &result));
    ASSERT_TRUE(result.converged);
    EXPECT_NEAR(result.step(0), (1. + 0.5 * w) / (1. + w), 1.e-6);
}

TEST(SourceQp, ElasticRowInactiveBelowItsHinge)
{
    ConvexQp qp = unconstrained(VectorXd::Constant(1, 1.), VectorXd::Constant(1, -0.25));
    qp.elastic_jacobian = MatrixXd::Constant(1, 1, -1.);
    qp.elastic_margin = VectorXd::Constant(1, 0.5);  // violated only for x > 0.5
    qp.elastic_weight = VectorXd::Constant(1, 100.);
    qp.elastic_linear_cost = VectorXd::Zero(1);
    ConvexQpResult result;
    ASSERT_TRUE(solve_convex_qp(qp, 80, 1.e-9, &result));
    ASSERT_TRUE(result.converged);
    EXPECT_NEAR(result.step(0), 0.25, 1.e-6);
}

TEST(SourceQp, LargeLinearPenaltyIsAnExactL1Constraint)
{
    // The penalty exceeds the multiplier of x <= 1, so the optimum sits on it.
    ConvexQp qp = unconstrained(VectorXd::Constant(1, 1.), VectorXd::Constant(1, -5.));
    qp.elastic_jacobian = MatrixXd::Constant(1, 1, -1.);
    qp.elastic_margin = VectorXd::Constant(1, 1.);
    qp.elastic_weight = VectorXd::Constant(1, 1.e-3);
    qp.elastic_linear_cost = VectorXd::Constant(1, 1.e4);
    ConvexQpResult result;
    ASSERT_TRUE(solve_convex_qp(qp, 120, 1.e-6, &result));
    ASSERT_TRUE(result.converged);
    EXPECT_NEAR(result.step(0), 1., 1.e-3);
}

TEST(SourceQp, MalformedInputIsRejected)
{
    ConvexQp qp = unconstrained(Eigen::Vector2d(1., 1.), Eigen::Vector2d(0., 0.));
    qp.upper = Eigen::Vector2d(-200., 100.);  // upper < lower
    ConvexQpResult result;
    EXPECT_FALSE(solve_convex_qp(qp, 60, 1.e-9, &result));
}

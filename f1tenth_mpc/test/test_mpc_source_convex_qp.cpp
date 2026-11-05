#include "mpc_source_convex_qp.hpp"
#include <gtest/gtest.h>

namespace sdu_apex::source_nlp {
namespace {
SourceConvexQp scalar(double gradient,double hard_margin=0.) {
    SourceConvexQp p;
    p.hessian=Eigen::MatrixXd::Identity(1,1);
    p.gradient=Eigen::VectorXd::Constant(1,gradient);
    p.hard_jacobian=Eigen::MatrixXd::Ones(1,1);
    p.hard_margin=Eigen::VectorXd::Constant(1,hard_margin);
    p.elastic_jacobian.resize(0,1);p.elastic_margin.resize(0);p.elastic_weight.resize(0);
    return p;
}
TEST(SourceConvexQp, PhysicalBoundaryCannotBeTradedForCost) {
    auto p=scalar(10000.);
    SourceConvexQpResult r;
    ASSERT_TRUE(solve_source_convex_qp(p,40,1.e-6,&r));
    ASSERT_TRUE(r.converged)<<r.primal_inf<<' '<<r.dual_inf<<' '<<r.complementarity_inf;
    EXPECT_GE(r.step(0),-1.e-9);
    EXPECT_NEAR(r.step(0),0.,1.e-8);
    EXPECT_NEAR(r.hard_multipliers(0),10000.,1.e-4);
}
TEST(SourceConvexQp, PreferredReserveRemainsElastic) {
    auto p=scalar(20.);
    p.elastic_jacobian=Eigen::MatrixXd::Ones(1,1);
    p.elastic_margin=Eigen::VectorXd::Constant(1,-.03);
    p.elastic_weight=Eigen::VectorXd::Constant(1,1000.);
    SourceConvexQpResult r;
    ASSERT_TRUE(solve_source_convex_qp(p,40,1.e-6,&r));
    ASSERT_TRUE(r.converged)<<r.primal_inf<<' '<<r.dual_inf<<' '<<r.complementarity_inf;
    EXPECT_NEAR(r.step(0),10./1001.,1.e-7);
    EXPECT_LT(r.step(0),.03);
    EXPECT_GE(r.step(0),0.);
    EXPECT_NEAR(r.hard_multipliers(0),0.,1.e-5);
}
TEST(SourceConvexQp, InconsistentHardRowsNeverConverge) {
    auto p=scalar(0.);
    p.hard_jacobian.resize(2,1);p.hard_jacobian<<1.,-1.;
    p.hard_margin.resize(2);p.hard_margin<<-1.,0.;
    SourceConvexQpResult r;
    const bool valid=solve_source_convex_qp(p,30,1.e-6,&r);
    if(valid) {
        EXPECT_FALSE(r.converged);
        EXPECT_GE(r.primal_inf,.5-1.e-8);
    }
}
TEST(SourceConvexQp, ConstraintScalingPreservesPhysicalSolutionAndDual) {
    auto p=scalar(2.,.02);
    SourceConvexQpResult a,b;
    ASSERT_TRUE(solve_source_convex_qp(p,40,1.e-6,&a));
    p.hard_jacobian*=1000.;p.hard_margin*=1000.;
    ASSERT_TRUE(solve_source_convex_qp(p,40,1.e-6,&b));
    EXPECT_TRUE(a.converged);EXPECT_TRUE(b.converged);
    EXPECT_NEAR(a.step(0),b.step(0),1.e-8);
    EXPECT_NEAR(a.hard_multipliers(0),1000.*b.hard_multipliers(0),1.e-6);
}
TEST(SourceConvexQp, ElasticOnlyProblemUsesSameOptimalityEquations) {
    auto p=scalar(-1.);
    p.hard_jacobian.resize(0,1);p.hard_margin.resize(0);
    p.elastic_jacobian=Eigen::MatrixXd::Constant(1,1,-1.);
    p.elastic_margin=Eigen::VectorXd::Constant(1,.2);
    p.elastic_weight=Eigen::VectorXd::Constant(1,10.);
    SourceConvexQpResult r;
    ASSERT_TRUE(solve_source_convex_qp(p,40,1.e-6,&r));
    EXPECT_TRUE(r.converged)<<r.dual_inf;
    EXPECT_NEAR(r.step(0),3./11.,1.e-7);
}
TEST(SourceConvexQp, SharedPeakKeepsBothBranchesAtKink) {
    auto p=scalar(0.);
    p.hard_jacobian.resize(0,1);p.hard_margin.resize(0);
    p.elastic_jacobian.resize(2,1);p.elastic_jacobian<<1.,-1.;
    p.elastic_margin=Eigen::VectorXd::Constant(2,-1.);
    p.elastic_groups=Eigen::VectorXi::Zero(2);
    p.elastic_weight=Eigen::VectorXd::Constant(1,10.);
    SourceConvexQpResult r;
    ASSERT_TRUE(solve_source_convex_qp(p,60,1.e-6,&r));
    EXPECT_TRUE(r.converged)<<r.dual_inf<<' '<<r.complementarity_inf;
    EXPECT_NEAR(r.step(0),0.,1.e-7);
    EXPECT_NEAR(r.elastic_multipliers.sum(),10.,1.e-5);
    EXPECT_NEAR(r.elastic_multipliers(0),r.elastic_multipliers(1),1.e-5);
}
TEST(SourceConvexQp, ExactL1LinearizationPenaltyDoesNotForceLargeStep) {
    auto p=scalar(0.);
    p.hessian(0,0)=1.e6;
    p.hard_jacobian.resize(0,1);p.hard_margin.resize(0);
    p.elastic_jacobian=Eigen::MatrixXd::Ones(1,1);
    p.elastic_margin=Eigen::VectorXd::Constant(1,-1.);
    p.elastic_weight=Eigen::VectorXd::Zero(1);
    p.elastic_linear_cost=Eigen::VectorXd::Constant(1,100.);
    SourceConvexQpResult r;
    ASSERT_TRUE(solve_source_convex_qp(p,60,1.e-6,&r));
    EXPECT_TRUE(r.converged)<<r.dual_inf<<' '<<r.complementarity_inf;
    EXPECT_NEAR(r.step(0),.0001,1.e-8);
    EXPECT_NEAR(r.elastic_multipliers(0),100.,1.e-5);
}
} // namespace
} // namespace sdu_apex::source_nlp

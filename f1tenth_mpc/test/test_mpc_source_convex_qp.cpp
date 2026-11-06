#include "mpc_source_convex_qp.hpp"
#include "mpc_source_exact_qp.hpp"
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
TEST(SourceConstrainedQp, OriginalStationarityExcludesPreferredReservePenalty) {
    SparseNonlinearProgram program;
    program.valid=true;
    program.equality_rows=kStages*kState;
    program.original_source_stage_derivatives.resize(kStages);
    program.rows.resize(program.equality_rows);
    for(auto &row:program.rows)row.type=NonlinearRowType::DynamicsEquality;
    const int row=program.rows.size();
    program.rows.push_back({NonlinearRowType::BodyCorner,0,0,-1,-1,.01});
    program.jacobian.push_back({row,program.equality_rows,1.,1.});
    double original_gradient[kStages][2]{};
    original_gradient[0][0]=4.;
    MpcModelControl_t controls[kStages]{};
    for(auto &control:controls)control.throttle_command=.5f;
    SourceConstrainedQpOptions options;
    options.requested_corner_clearance_m=.03;
    options.inequality_penalty_weight=1000.;
    options.maximum_qp_iterations=60;
    SourceConstrainedQpResult result;
    ASSERT_TRUE(source_exact_constrained_qp_direction(program,original_gradient,
        controls,options,&result));
    ASSERT_TRUE(result.converged);
    // The original Lagrangian contains only J - lambda*g. The finite
    // preferred reserve penalty is absent from this KKT certificate.
    const double expected=std::abs((4.-result.inequality_multipliers[row])*.15);
    EXPECT_NEAR(result.original_nlp_stationarity_inf,expected,1.e-12);
    EXPECT_GT(result.original_nlp_stationarity_inf,.5);
}
TEST(SourceConstrainedQp, InconsistentHardRowsUseOneNormalTangentialStep) {
    SparseNonlinearProgram program;
    program.valid=true;program.equality_rows=kStages*kState;
    program.original_source_stage_derivatives.resize(kStages);
    program.rows.resize(program.equality_rows);
    for(auto &row:program.rows)row.type=NonlinearRowType::DynamicsEquality;
    const int first=program.rows.size();
    program.rows.push_back({NonlinearRowType::BodyCorner,0,0,-1,-1,-.10});
    program.rows.push_back({NonlinearRowType::BodyCorner,0,1,-1,-1,0.});
    program.jacobian.push_back({first,program.equality_rows,1.,1.});
    program.jacobian.push_back({first+1,program.equality_rows,-1.,-1.});
    // The real 60-control problem has a finite legal box. These rows also
    // bound the normal problem's otherwise unconstrained nullspace.
    for(int j=0;j<2*kStages;++j)for(int side=0;side<2;++side) {
        const int row=program.rows.size();
        const double margin=j%2?(j==1?(side?.199:.8):(side?.499:.5)):.5;
        program.rows.push_back({NonlinearRowType::PhysicalCommandBox,j/2,2*(j%2)+side,-1,-1,margin});
        program.jacobian.push_back({row,program.equality_rows+j,side?1.:-1.,side?1.:-1.});
    }
    double gradient[kStages][2]{};gradient[0][0]=-1.;
    MpcModelControl_t controls[kStages]{};
    for(auto &u:controls)u.throttle_command=.5f;
    controls[0].throttle_command=.2f;
    SourceConstrainedQpOptions options;options.maximum_qp_iterations=60;
    SourceConstrainedQpResult result;
    ASSERT_TRUE(source_exact_constrained_qp_direction(program,gradient,controls,options,&result));
    ASSERT_TRUE(result.normal_step_computed);
    ASSERT_TRUE(result.converged)<<result.primal_residual<<' '<<result.dual_residual;
    EXPECT_NEAR(result.normal_direction[0][0],.05,1.e-5);
    EXPECT_NEAR(result.direction[0][0],.05,1.e-5);
    EXPECT_NEAR(result.linearized_physical_violation_inf,.05,1.e-5);
    EXPECT_FALSE(result.original_duals_trustworthy);
    EXPECT_EQ(result.inequality_multipliers[first],0.);
    EXPECT_EQ(result.inequality_multipliers[first+1],0.);
    EXPECT_NEAR(result.normal_linearized_violation,.0025,1.e-9);
    // Racing damping must not change the affine physical normal optimum.
    options.curvature_regularization=1.e10;
    SourceConstrainedQpResult damped;
    ASSERT_TRUE(source_exact_constrained_qp_direction(program,gradient,controls,options,&damped));
    ASSERT_TRUE(damped.normal_step_computed);
    EXPECT_NEAR(damped.normal_direction[0][0],result.normal_direction[0][0],1.e-9);
    EXPECT_NEAR(damped.normal_linearized_violation,result.normal_linearized_violation,1.e-12);
}
TEST(SourceConvexQp, DependentNormalEquationsUseExactNullspace) {
    SourceConvexQp p;
    p.hessian=Eigen::MatrixXd::Identity(2,2);
    p.gradient.resize(2);p.gradient<<0.,-2.;
    p.equality_jacobian.resize(2,2);p.equality_jacobian<<1.,1.,2.,2.;
    p.equality_rhs.resize(2);p.equality_rhs<<1.,2.;
    p.hard_jacobian.resize(2,2);p.hard_jacobian=Eigen::MatrixXd::Identity(2,2);
    p.hard_margin=Eigen::VectorXd::Zero(2);
    p.elastic_jacobian.resize(0,2);p.elastic_margin.resize(0);p.elastic_weight.resize(0);
    SourceConvexQpResult r;
    ASSERT_TRUE(solve_source_convex_qp(p,60,1.e-7,&r));
    ASSERT_TRUE(r.converged)<<r.primal_inf<<' '<<r.dual_inf;
    EXPECT_NEAR(r.step(0),0.,1.e-7);EXPECT_NEAR(r.step(1),1.,1.e-7);
    EXPECT_LE((p.equality_jacobian*r.step-p.equality_rhs).norm(),1.e-8);
    EXPECT_GE(r.hard_multipliers.minCoeff(),0.);
}
TEST(SourceConvexQp, ExpiredBudgetCannotCertifyDirectionOrDuals) {
    auto p=scalar(-1.);
    SourceConvexQpResult r;
    // Any positive epoch this small is already in the monotonic past.
    solve_source_convex_qp(p,60,1.e-7,&r,1.);
    EXPECT_TRUE(r.deadline_exceeded);
    EXPECT_FALSE(r.converged);
    EXPECT_EQ(r.iterations,0);
}
TEST(SourceConstrainedQp, NormalStepRespectsSaturatedPhysicalBounds) {
    SparseNonlinearProgram program;
    program.valid=true;program.equality_rows=kStages*kState;
    program.original_source_stage_derivatives.resize(kStages);
    program.rows.resize(program.equality_rows);
    for(auto &row:program.rows)row.type=NonlinearRowType::DynamicsEquality;
    const int wall=program.rows.size();
    program.rows.push_back({NonlinearRowType::BodyCorner,0,0,-1,-1,-.1});
    program.jacobian.push_back({wall,program.equality_rows,1.,1.});
    program.jacobian.push_back({wall,program.equality_rows+1,1.,1.});
    for(int j=0;j<2*kStages;++j)for(int side=0;side<2;++side) {
        const int row=program.rows.size();
        const double margin=j<2?(side?(j?.969:.97):.03):(j%2?(side?.499:.5):.5);
        program.rows.push_back({NonlinearRowType::PhysicalCommandBox,j/2,2*(j%2)+side,-1,-1,margin});
        program.jacobian.push_back({row,program.equality_rows+j,side?1.:-1.,side?1.:-1.});
    }
    MpcModelControl_t controls[kStages]{};
    for(auto &u:controls)u.throttle_command=.5f;
    controls[0]={.47f,.97f};
    double gradient[kStages][2]{};
    SourceConstrainedQpOptions options;options.maximum_qp_iterations=80;
    SourceConstrainedQpResult result;
    ASSERT_TRUE(source_exact_constrained_qp_direction(program,gradient,controls,options,&result))
        <<result.failure_reason;
    ASSERT_TRUE(result.normal_step_computed);
    EXPECT_NEAR(result.normal_direction[0][0],.03,1.e-7);
    EXPECT_NEAR(result.normal_direction[0][1],.03,1.e-7);
    EXPECT_NEAR(result.normal_linearized_violation,.5*.04*.04,1.e-9);
    EXPECT_FALSE(result.original_duals_trustworthy);
}
TEST(SourceConvexQp, EqualityScalingRetainsIndependentPhysicalEquations) {
    SourceConvexQp p;
    p.hessian=Eigen::MatrixXd::Identity(2,2);p.gradient=Eigen::VectorXd::Zero(2);
    p.equality_jacobian=Eigen::MatrixXd::Zero(2,2);
    p.equality_jacobian(0,0)=1.e-8;p.equality_jacobian(1,1)=1.e8;
    p.equality_rhs.resize(2);p.equality_rhs<<2.e-8,3.e8;
    p.hard_jacobian.resize(0,2);p.hard_margin.resize(0);
    p.elastic_jacobian.resize(0,2);p.elastic_margin.resize(0);p.elastic_weight.resize(0);
    SourceConvexQpResult r;
    ASSERT_TRUE(solve_source_convex_qp(p,60,1.e-7,&r));
    ASSERT_TRUE(r.converged);
    EXPECT_NEAR(r.step(0),2.,1.e-12);EXPECT_NEAR(r.step(1),3.,1.e-12);
    EXPECT_NEAR(r.equality_multipliers(0),2.e8,1.e-6);
    EXPECT_NEAR(r.equality_multipliers(1),3.e-8,1.e-20);
    EXPECT_LE(r.equality_dual_reconstruction_inf,1.e-12);
}
TEST(SourceConstrainedQp, ConsistentHardLinearizationRetainsHardDirectionAndDuals) {
    SparseNonlinearProgram program;
    program.valid=true;program.equality_rows=kStages*kState;
    program.original_source_stage_derivatives.resize(kStages);
    program.rows.resize(program.equality_rows);
    for(auto &row:program.rows)row.type=NonlinearRowType::DynamicsEquality;
    const int row=program.rows.size();
    program.rows.push_back({NonlinearRowType::BodyCorner,0,0,-1,-1,-.1});
    program.jacobian.push_back({row,program.equality_rows,1.,1.});
    double gradient[kStages][2]{};gradient[0][1]=-10.;
    MpcModelControl_t controls[kStages]{};
    for(auto &u:controls)u.throttle_command=.5f;
    SourceConstrainedQpOptions options;options.maximum_qp_iterations=60;
    SourceConstrainedQpResult result;
    ASSERT_TRUE(source_exact_constrained_qp_direction(program,gradient,controls,options,&result));
    ASSERT_TRUE(result.converged);
    EXPECT_FALSE(result.normal_step_computed);
    EXPECT_TRUE(result.original_duals_trustworthy);
    EXPECT_GE(result.direction[0][0],.1-1.e-8);
    EXPECT_LE(result.linearized_physical_violation_inf,1.e-8);
    // QP: min 500*z_steer^2 - 2.5*z_throttle, with
    // -.1+.15*z_steer >= 0. Its hard multiplier is 1000*.1/.15^2.
    EXPECT_NEAR(result.inequality_multipliers[row],1000.*.1/(.15*.15),1.e-2);
    EXPECT_GT(result.direction[0][1],0.);
}
TEST(SourceConvexQp, FullyFixedNormalStepKeepsElasticStationarity) {
    auto p=scalar(-1.e5);
    p.equality_jacobian=Eigen::MatrixXd::Ones(1,1);
    p.equality_rhs=Eigen::VectorXd::Constant(1,.25);
    p.elastic_jacobian=Eigen::MatrixXd::Ones(1,1);
    p.elastic_margin=Eigen::VectorXd::Constant(1,-.3);
    p.elastic_weight=Eigen::VectorXd::Constant(1,1000.);
    SourceConvexQpResult r;
    ASSERT_TRUE(solve_source_convex_qp(p,60,1.e-7,&r));
    ASSERT_TRUE(r.converged)<<r.primal_inf<<' '<<r.dual_inf;
    EXPECT_NEAR(r.step(0),.25,1.e-12);
    EXPECT_NEAR(r.elastic_values(0),.05,1.e-12);
    EXPECT_NEAR(r.elastic_multipliers(0),50.,1.e-8);
    EXPECT_LE(r.dual_inf,1.e-7);
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
TEST(SourceConvexQp, StiffIndependentElasticsDoNotCancelControlCurvature) {
    // Analytic solution x_j=1/(H_jj+2w), all hard bounds inactive.
    // The old subtractive Schur complement loses the positive curvature
    // when elastic barrier ratios grow and terminates without convergence.
    constexpr int n=5;
    SourceConvexQp p;
    p.hessian=Eigen::MatrixXd::Identity(n,n);
    p.hessian.diagonal()<<1.,1.e-4,1.e-8,1.e-4,1.;
    p.gradient=Eigen::VectorXd::Constant(n,-1.);
    p.hard_jacobian.resize(2*n,n);
    p.hard_jacobian.topRows(n)=Eigen::MatrixXd::Identity(n,n);
    p.hard_jacobian.bottomRows(n)=-Eigen::MatrixXd::Identity(n,n);
    p.hard_margin=Eigen::VectorXd::Constant(2*n,2.);
    p.elastic_jacobian=p.hard_jacobian;
    p.elastic_margin=Eigen::VectorXd::Constant(2*n,-.03);
    p.elastic_weight=Eigen::VectorXd::Constant(2*n,1.e15);
    SourceConvexQpResult r;
    ASSERT_TRUE(solve_source_convex_qp(p,100,1.e-6,&r));
    EXPECT_TRUE(r.converged)<<r.dual_inf<<' '<<r.complementarity_inf;
    EXPECT_LE(r.dual_inf,1.e-6);
    EXPECT_LE(r.complementarity_inf,1.e-6);
    for(int j=0;j<n;++j)EXPECT_NEAR(r.step(j),1./(p.hessian(j,j)+2.e15),1.e-12);
}
TEST(SourceConvexQp, SharedElasticSchurCovarianceKeepsCrossControlCurvature) {
    auto p=scalar(0.);
    p.hessian=Eigen::MatrixXd::Identity(2,2);
    p.gradient=Eigen::VectorXd::Zero(2);
    p.hard_jacobian.resize(0,2);p.hard_margin.resize(0);
    p.elastic_jacobian=Eigen::MatrixXd::Identity(2,2);
    p.elastic_margin=Eigen::VectorXd::Constant(2,-1.);
    p.elastic_groups=Eigen::VectorXi::Zero(2);
    p.elastic_weight=Eigen::VectorXd::Constant(1,10.);
    SourceConvexQpResult r;
    ASSERT_TRUE(solve_source_convex_qp(p,60,1.e-7,&r));
    ASSERT_TRUE(r.converged)<<r.dual_inf;
    EXPECT_NEAR(r.step(0),10./12.,1.e-7);
    EXPECT_NEAR(r.step(1),10./12.,1.e-7);
    EXPECT_NEAR(r.elastic_values(0),2./12.,1.e-7);
    EXPECT_NEAR(r.elastic_multipliers(0),10./12.,1.e-6);
    EXPECT_NEAR(r.elastic_multipliers(1),10./12.,1.e-6);
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

#include "mpc_rti.h"
#include "riccati_state_polygon.h"
#include "mpc_qp_elastic_polygon.hpp"
#include "mpc_qp_convexity.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

TEST(MpcCorridorPolygon, MatchesEveryCornerAndHeadingZeroSwitch)
{
  MpcRtiConfiguration_t config{};
  config.corridor_margin_m = .05f;
  config.first_prediction_corridor_margin_m = .03f;
  const float left = .6f, right = .7f;
  for (float angle : {-3.f, -1.f, -.1f, 0.f, .1f, 1.f, 3.f}) {
    for (float ey : {-.3f, 0.f, .3f}) {
      for (int stage : {1, 2, 30}) {
        MpcModelState_t state{};
        state.e_y = ey; state.e_psi = angle;
        RiccatiStatePolygon_t polygon{};
        ASSERT_TRUE(mpc_rti_build_corridor_polygon(&state, left, right,
            &config, stage, &polygon));
        ASSERT_EQ(polygon.count, 8);
        float min_left = INFINITY, min_right = INFINITY;
        for (int i = 0; i < polygon.count; ++i) {
          const float slack = polygon.upper[i] - polygon.normal[i][0]*ey -
              polygon.normal[i][1]*angle;
          if (i % 2 == 0) min_left = std::min(min_left, slack);
          else min_right = std::min(min_right, slack);
        }
        float expected_right, expected_left;
        ASSERT_TRUE(mpc_rti_compute_corridor_slack(&state, left, right,
            &config, stage, &expected_right, &expected_left));
        EXPECT_NEAR(min_left, expected_left, 2.e-7f);
        EXPECT_NEAR(min_right, expected_right, 2.e-7f);
      }
    }
  }
  // The former heading-frozen ey box passes this point; the new QP must not.
  MpcModelState_t nominal{};
  RiccatiStatePolygon_t polygon{};
  ASSERT_TRUE(mpc_rti_build_corridor_polygon(&nominal, .3f, .3f,
      &config, 2, &polygon));
  MpcModelState_t candidate{};
  candidate.e_y = .1f; candidate.e_psi = .1f;
  EXPECT_LT(candidate.e_y, .3f-.05f-.5f*MPC_CAR_WIDTH_M);
  int violated = 0;
  for (int i = 0; i < polygon.count; ++i)
    violated += polygon.normal[i][0]*candidate.e_y +
        polygon.normal[i][1]*candidate.e_psi > polygon.upper[i];
  EXPECT_GT(violated, 0);
  float lo, hi;
  ASSERT_TRUE(mpc_rti_compute_corridor_slack(&candidate, .3f, .3f,
      &config, 2, &lo, &hi));
  EXPECT_LT(hi, 0.f);
}

TEST(RiccatiStatePolygon, WeightedFaceCornerAndEmptyIntersection)
{
  RiccatiStatePolygon_t polygon{};
  polygon.count = 1;
  polygon.normal[0][0] = polygon.normal[0][1] = 1.f;
  polygon.upper[0] = 1.f;
  const float lower[2] = {-10.f, -10.f}, upper[2] = {10.f, 10.f};
  const float metric[2] = {1.f, 4.f}, input[2] = {1.f, 1.f};
  float output[2]{};
  ASSERT_TRUE(riccati_project_state_polygon(&polygon, input, lower, upper, metric, output));
  EXPECT_NEAR(output[0], .2f, 1.e-7f);
  EXPECT_NEAR(output[1], .8f, 1.e-7f);
  const float positive_lower[2] = {0.f, 0.f}, box_upper[2] = {1.f, 1.f};
  const float outside[2] = {2.f, 2.f};
  ASSERT_TRUE(riccati_project_state_polygon(&polygon, outside,
      positive_lower, box_upper, metric, output));
  EXPECT_FLOAT_EQ(output[0], 0.f);
  EXPECT_FLOAT_EQ(output[1], 1.f);
  polygon.count = 2;
  polygon.normal[0][1] = 0.f; polygon.upper[0] = 0.f;
  polygon.normal[1][0] = -1.f; polygon.upper[1] = -1.f;
  EXPECT_FALSE(riccati_project_state_polygon(&polygon, input, lower, upper, metric, output));
  polygon.count = 1; polygon.normal[0][0] = NAN;
  EXPECT_FALSE(riccati_project_state_polygon(&polygon, input, lower, upper, metric, output));
}

TEST(RiccatiStatePolygon, RotatedFaceMatchesIndependentAnalyticProjection)
{
  const float input[2] = {1.f, .4f}, lower[2] = {-10.f, -10.f};
  const float upper[2] = {10.f, 10.f}, metric[2] = {2.f, 7.f};
  for (int degrees = -180; degrees < 180; ++degrees) {
    RiccatiStatePolygon_t polygon{};
    polygon.count = 1;
    const double angle = degrees * .017453292519943295;
    const float a = (float)std::cos(angle), b = (float)std::sin(angle);
    polygon.normal[0][0] = a; polygon.normal[0][1] = b;
    polygon.upper[0] = .3f;
    const double violation = a*(double)input[0] + b*(double)input[1] - .3f;
    const double lambda = std::max(0., violation) / (a*(double)a/2. + b*(double)b/7.);
    float result[2];
    ASSERT_TRUE(riccati_project_state_polygon(&polygon, input, lower, upper, metric, result));
    EXPECT_NEAR(result[0], input[0] - lambda*a/2., 1.e-7);
    EXPECT_NEAR(result[1], input[1] - lambda*b/7., 1.e-7);
  }
}

TEST(RiccatiStatePolygon, ScaledAndPhysicalAdmmKeepTheCoupledConstraint)
{
  RiccatiStepData_t steps[RICCATI_HORIZON]{};
  float terminal_Q[RICCATI_MAX_NX]{}, terminal_q[RICCATI_MAX_NX]{};
  float lo[RICCATI_MAX_NX], hi[RICCATI_MAX_NX], initial[RICCATI_MAX_NX]{};
  std::fill_n(lo, RICCATI_MAX_NX, -BIG_BOUND);
  std::fill_n(hi, RICCATI_MAX_NX, BIG_BOUND);
  lo[0] = lo[1] = -2.f; hi[0] = hi[1] = 2.f;
  for (auto &step : steps) {
    std::copy_n(lo, RICCATI_MAX_NX, step.x_lb);
    std::copy_n(hi, RICCATI_MAX_NX, step.x_ub);
    for (int axis = 0; axis < 2; ++axis) {
      step.A[axis][axis] = step.B[axis][axis] = 1.f;
      step.Q_diag[axis] = 2.f; step.q[axis] = -2.f;
      step.R_diag[axis] = .1f;
      step.u_lb[axis] = -1.f; step.u_ub[axis] = 1.f;
      terminal_Q[axis] = 2.f; terminal_q[axis] = -2.f;
    }
  }
  RiccatiAdmmConfig_t config{};
  config.rho = config.rho_u = 7.f;
  // Deliberately unequal state scales stress the weighted projection. This
  // is a converged mathematical parity test, not the runtime iteration cap.
  config.tolerance = 1.e-5f; config.max_iterations = 5000;
  config.over_relaxation = 1.6f;
  for (auto &polygon : config.state_polygons) {
    polygon.count = 1;
    polygon.normal[0][0] = polygon.normal[0][1] = 1.f;
    polygon.upper[0] = .5f;
  }
  RiccatiSolution_t solutions[2]{};
  for (int scaled = 0; scaled < 2; ++scaled) {
    config.use_scaling = scaled;
    std::fill_n(config.state_scale, RICCATI_MAX_NX, 1.f);
    config.state_scale[0] = .1f; config.state_scale[1] = .5f;
    config.input_scale[0] = .4f; config.input_scale[1] = .7f;
    RiccatiAdmmState_t memory{};
    ASSERT_EQ(riccati_admm_solve(steps, terminal_Q, terminal_q, lo, hi,
        initial, &config, &memory, &solutions[scaled]), RICCATI_STATUS_OPTIMAL)
        << "scaled=" << scaled << " primal=" << solutions[scaled].primal_residual
        << " dual=" << solutions[scaled].dual_residual << " end="
        << solutions[scaled].x[RICCATI_HORIZON][0] << ','
        << solutions[scaled].x[RICCATI_HORIZON][1];
    for (int k = 1; k <= RICCATI_HORIZON; ++k)
      EXPECT_LE(solutions[scaled].x[k][0] + solutions[scaled].x[k][1], .5001f);
    EXPECT_NEAR(solutions[scaled].x[RICCATI_HORIZON][0], .25f, 2.e-4f);
    EXPECT_NEAR(solutions[scaled].x[RICCATI_HORIZON][1], .25f, 2.e-4f);
  }
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    for (int axis = 0; axis < 2; ++axis)
      EXPECT_NEAR(solutions[0].u[k][axis], solutions[1].u[k][axis], 2.e-4f);
}

TEST(RiccatiStatePolygon, ExactWeightedThreeAxisProgressProjection)
{
  RiccatiStatePolygon_t poly{};
  poly.count=1;
  poly.normal[0][0]=1.f;
  poly.progress_normal[0]=1.f;
  poly.upper[0]=.5f;
  const float input[3]={1.f,0.f,1.f};
  const float lo[3]={-2.f,-2.f,-2.f},hi[3]={2.f,2.f,2.f};
  const float metric[3]={1.f,4.f,1.f};
  float projected[3]{};
  ASSERT_TRUE(riccati_project_state_progress_polygon(
      &poly,input,lo,hi,metric,projected));
  EXPECT_NEAR(projected[0],.25f,1.e-7);
  EXPECT_NEAR(projected[1],0.f,1.e-7);
  EXPECT_NEAR(projected[2],.25f,1.e-7);
  // With a high progress penalty, more of the correction must move the car.
  const float expensive_progress[3]={1.f,4.f,100.f};
  ASSERT_TRUE(riccati_project_state_progress_polygon(
      &poly,input,lo,hi,expensive_progress,projected));
  EXPECT_LT(projected[0],-.45f);
  EXPECT_GT(projected[2],.98f);
}

TEST(RiccatiStatePolygon, ThreeAxisProjectionHandlesVertexAndEmptySet)
{
  RiccatiStatePolygon_t poly{};
  poly.count=3;
  poly.normal[0][0]=1.f;poly.upper[0]=0.f;
  poly.normal[1][1]=1.f;poly.upper[1]=0.f;
  poly.progress_normal[2]=1.f;poly.upper[2]=0.f;
  const float input[3]={1.f,1.f,1.f},lo[3]={-2.f,-2.f,-2.f};
  const float hi[3]={2.f,2.f,2.f},metric[3]={1.f,2.f,4.f};
  float projected[3]{};
  ASSERT_TRUE(riccati_project_state_progress_polygon(
      &poly,input,lo,hi,metric,projected));
  for(float value:projected) EXPECT_NEAR(value,0.f,1.e-7f);
  poly.count=2;
  poly.normal[0][0]=1.f;poly.upper[0]=0.f;
  poly.normal[1][0]=-1.f;poly.normal[1][1]=0.f;
  poly.upper[1]=-1.f;
  EXPECT_FALSE(riccati_project_state_progress_polygon(
      &poly,input,lo,hi,metric,projected));
}

TEST(RiccatiStatePolygon, AdmmActuallyEnforcesProgressCoupledHalfspace)
{
  auto steps=std::make_unique<RiccatiStepData_t[]>(RICCATI_HORIZON);
  auto config=std::make_unique<RiccatiAdmmConfig_t>();
  auto memory=std::make_unique<RiccatiAdmmState_t>();
  auto solution=std::make_unique<RiccatiSolution_t>();
  float terminal_q[RICCATI_MAX_NX]{},terminal_Q[RICCATI_MAX_NX]{};
  float lb[RICCATI_MAX_NX]{},ub[RICCATI_MAX_NX]{},x0[RICCATI_MAX_NX]{};
  std::fill_n(lb,RICCATI_MAX_NX,-BIG_BOUND);
  std::fill_n(ub,RICCATI_MAX_NX, BIG_BOUND);
  for(int i:{MPC_OPT_IDX_EY,MPC_OPT_IDX_EPSI,
             MPC_OPT_IDX_PROGRESS_DEVIATION}) {
    lb[i]=-2.f;ub[i]=2.f;
  }
  constexpr int S=MPC_OPT_IDX_EY,P=MPC_OPT_IDX_PROGRESS_DEVIATION;
  for(int k=0;k<RICCATI_HORIZON;++k) {
    auto &stage=steps[k];
    std::copy_n(lb,RICCATI_MAX_NX,stage.x_lb);
    std::copy_n(ub,RICCATI_MAX_NX,stage.x_ub);
    for(int i=0;i<RICCATI_MAX_NX;++i)stage.A[i][i]=1.f;
    stage.B[S][0]=1.f;
    stage.B[P][1]=1.f;
    stage.R_diag[0]=stage.R_diag[1]=.1f;
    stage.Q_diag[S]=stage.Q_diag[P]=2.f;
    stage.q[S]=stage.q[P]=-2.f;
    stage.u_lb[0]=stage.u_lb[1]=-1.f;
    stage.u_ub[0]=stage.u_ub[1]= 1.f;
  }
  terminal_Q[S]=terminal_Q[P]=2.f;
  terminal_q[S]=terminal_q[P]=-2.f;
  for(int k=1;k<=RICCATI_HORIZON;++k) {
    auto &poly=config->state_polygons[k];
    poly.count=1;poly.normal[0][0]=1.f;
    poly.progress_normal[0]=1.f;
    poly.upper[0]=.5f;
  }
  config->rho=config->rho_u=7.f;
  config->rho_max=32.f;
  config->max_iterations=2500;
  config->tolerance=2.e-4f;
  config->over_relaxation=1.6f;
  config->adaptive_rho_update_period=10;
  std::fill_n(config->state_scale,RICCATI_MAX_NX,1.f);
  std::fill_n(config->input_scale,RICCATI_MAX_NU,1.f);
  const auto status=riccati_admm_solve(steps.get(),terminal_Q,terminal_q,
      lb,ub,x0,config.get(),memory.get(),solution.get());
  ASSERT_TRUE(status==RICCATI_STATUS_OPTIMAL ||
      status==RICCATI_STATUS_MAX_ITERATIONS);
  EXPECT_LE(solution->primal_residual,1.e-2f);
  for(int k=1;k<=RICCATI_HORIZON;++k)
    EXPECT_LE(solution->x[k][S]+solution->x[k][P],.51f)
        << "constraint stage " << k;
}


TEST(MpcQpConvexity, FloatRoundedSourceGaussNewtonRemainsPsd)
{
    auto stage=std::make_unique<RiccatiStepData_t>();
    constexpr int body=MPC_OPT_IDX_BODY_HEIGHT;
    constexpr int jounce=MPC_OPT_IDX_SUSPENSION_JOUNCE_FL;
    // A correctly PSD rank-one full-source GN residual, independently
    // rounded to binary32 and summed, can acquire a small negative
    // eigenvalue at real high-slip Hessian magnitudes.
    stage->Q_diag[body]=350000.f;
    stage->Q_diag[jounce]=50000.f;
    stage->Q_offdiag[body][jounce]=132287.578125f;
    stage->Q_offdiag[jounce][body]=132287.578125f;
    float nominal[MPC_RTI_NX]{};
    nominal[body]=.015f;
    nominal[jounce]=.027f;
    stage->q[body]=-400.f;
    stage->q[jounce]=90.f;
    const float nominal_input[MPC_RTI_NU]{};
    const double original_offdiag=stage->Q_offdiag[body][jounce];
    const double original_qbody=stage->q[body];
    const double original_qjounce=stage->q[jounce];
    const double original_hbody=stage->Q_diag[body];
    const double original_hjounce=stage->Q_diag[jounce];
    const double det_before=original_hbody*original_hjounce-
        double(stage->Q_offdiag[body][jounce])*
        stage->Q_offdiag[jounce][body];
    ASSERT_LT(det_before,0.);
    float numerical_shift=NAN;
    ASSERT_TRUE(mpc_rti_convexify_float_source_hessian(
        stage.get(),nominal,nominal_input,&numerical_shift));
    EXPECT_GT(numerical_shift,0.);
    const double det_after=double(stage->Q_diag[body])*
        stage->Q_diag[jounce]-
        double(stage->Q_offdiag[body][jounce])*
        stage->Q_offdiag[jounce][body];
    EXPECT_GE(det_after,-.001);
    EXPECT_NEAR(double(stage->Q_diag[body]-original_hbody)*
        nominal[body]+(stage->Q_offdiag[body][jounce]-original_offdiag)*
        nominal[jounce]+stage->q[body]-original_qbody,0.,.0002);
    EXPECT_NEAR(double(stage->Q_diag[jounce]-original_hjounce)*
        nominal[jounce]+(stage->Q_offdiag[jounce][body]-original_offdiag)*
        nominal[body]+stage->q[jounce]-original_qjounce,0.,.0002);
}

TEST(MpcQpConvexity, RejectsRealNonconvexityWithoutDisguisingPhysicalCost)
{
    auto stage=std::make_unique<RiccatiStepData_t>();
    constexpr int a=MPC_OPT_IDX_BODY_HEIGHT;
    constexpr int b=MPC_OPT_IDX_SUSPENSION_JOUNCE_FL;
    stage->Q_diag[a]=1.f;
    stage->Q_diag[b]=1.f;
    stage->Q_offdiag[a][b]=stage->Q_offdiag[b][a]=2.f;
    float nominal[MPC_RTI_NX]{};
    const float nominal_input[MPC_RTI_NU]{};
    EXPECT_FALSE(mpc_rti_convexify_float_source_hessian(
        stage.get(),nominal,nominal_input,nullptr));
    EXPECT_FLOAT_EQ(stage->Q_diag[a],1.f);
    EXPECT_FLOAT_EQ(stage->Q_diag[b],1.f);
    stage->Q_offdiag[a][b]=stage->Q_offdiag[b][a]=.5f;
    float shift=NAN;
    ASSERT_TRUE(mpc_rti_convexify_float_source_hessian(
        stage.get(),nominal,nominal_input,&shift));
    EXPECT_FLOAT_EQ(shift,0.f);
    EXPECT_FLOAT_EQ(stage->Q_diag[a],1.f);
    EXPECT_FLOAT_EQ(stage->Q_diag[b],1.f);
}

TEST(RiccatiElasticCorridor, ExactSquareSlackPenaltyPreservesOriginalWallPlanes)
{
  RiccatiStatePolygon_t poly{};
  poly.count=2;
  poly.normal[0][0]=1.f;poly.upper[0]=.3f;
  poly.normal[1][0]=-1.f;poly.upper[1]=-.4f;
  poly.elastic_weight=100.f;
  // The hard intersection is impossible (e_y<=.3 and e_y>=.4).
  // Mathematically explicit slack makes the convex QP feasible without
  // silently shifting the ORIGINAL physical walls.
  const float input[3]={.4f,.08f,.1f};
  const float lo[3]={-2.f,-2.f,-2.f},hi[3]={2.f,2.f,2.f};
  const float metric[3]={2.f,4.f,6.f};
  float output[3]{};
  EXPECT_FALSE(riccati_project_state_progress_polygon(
      &poly,input,lo,hi,metric,output));
  ASSERT_TRUE(riccati_elastic_polygon_prox(
      &poly,input,lo,hi,metric,output));
  // KKT: rho*(x-.4)+100*(x-.3)-100*(.4-x)=0
  const double expected=(2.*.4+100.*(.3+.4))/(2.+2.*100.);
  EXPECT_NEAR(output[0],expected,1.e-6);
  EXPECT_FLOAT_EQ(output[1],input[1]);
  EXPECT_FLOAT_EQ(output[2],input[2]);
  EXPECT_GT(output[0],.3f);
  EXPECT_LT(output[0],.4f);
  const double left_slack=std::max(0.,double(output[0])-.3);
  const double right_slack=std::max(0.,.4-double(output[0]));
  EXPECT_GT(left_slack,0.);
  EXPECT_GT(right_slack,0.);
  // Must not claim a hard-legal physical state from its penalized QP solve.
  EXPECT_GT(left_slack+right_slack,.09);
}

TEST(RiccatiElasticCorridor, ProgressCoupledSquareHingeMatchesAnalyticOnePlaneProx)
{
  RiccatiStatePolygon_t poly{};
  poly.count=1;
  poly.normal[0][0]=1.f;
  poly.progress_normal[0]=1.f;
  poly.upper[0]=.5f;
  poly.elastic_weight=3.f;
  const float input[3]={.8f,0.f,.7f};
  const float lo[3]={-2.f,-2.f,-2.f},hi[3]={2.f,2.f,2.f};
  const float metric[3]={1.f,4.f,2.f};
  float output[3]{};
  ASSERT_TRUE(riccati_elastic_polygon_prox(
      &poly,input,lo,hi,metric,output));
  const double violation=.8+.7-.5;
  const double dual=3.*violation/(1.+3.*(1.+.5));
  EXPECT_NEAR(output[0],.8-dual,2.e-7);
  EXPECT_NEAR(output[2],.7-dual/2.,2.e-7);
  EXPECT_NEAR(output[1],0.,1.e-9);
}

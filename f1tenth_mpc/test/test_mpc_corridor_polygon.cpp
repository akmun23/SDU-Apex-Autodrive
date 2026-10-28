#include "mpc_rti.h"
#include "riccati_state_polygon.h"
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

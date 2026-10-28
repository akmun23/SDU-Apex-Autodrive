#include "../../tools/mpc_qp_fixture_format.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <unistd.h>

TEST(MpcQpFixture, RoundTripsActualProductionDimensions)
{
  char path[] = "/tmp/sdu_mpc_qp_roundtrip_XXXXXX";
  const int fd = mkstemp(path);
  ASSERT_GE(fd, 0);
  close(fd);
  struct Cleanup {
    const char *path;
    ~Cleanup() { std::remove(path); }
  } cleanup{path};
  MpcRtiQpCapture_t capture{};
  capture.valid = capture.capture_count = 1;
  capture.problem.horizon = RICCATI_HORIZON;
  capture.problem.terminal_q[MPC_OPT_IDX_PROGRESS_DEVIATION] = -125.0f;
  capture.problem.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] = 0.2f;
  auto &config = capture.solver_config;
  config.rho = config.rho_u = 7.0f;
  config.rho_max = 32.0f;
  config.tolerance = 0.02f;
  config.max_iterations = 150;
  config.adaptive_rho_update_period = 10;
  config.over_relaxation = 1.6f;
  std::fill_n(config.state_scale, RICCATI_MAX_NX, 1.0f);
  std::fill_n(config.input_scale, RICCATI_MAX_NU, 1.0f);
  ASSERT_TRUE(sdu_apex::qp_fixture::write_capture(path, capture));
  MpcRtiQpCapture_t decoded{};
  ASSERT_TRUE(sdu_apex::qp_fixture::read_capture(path, &decoded));
  EXPECT_EQ(decoded.problem.horizon, RICCATI_HORIZON);
  EXPECT_FLOAT_EQ(decoded.problem.terminal_q[MPC_OPT_IDX_PROGRESS_DEVIATION],
      -125.0f);
  EXPECT_FLOAT_EQ(decoded.problem.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND],
      0.2f);
}

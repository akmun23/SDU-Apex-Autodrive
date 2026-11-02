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
  capture.problem.steps[3].Q_offdiag[2][4] = -0.375f;
  capture.problem.steps[3].Q_offdiag[4][2] = -0.375f;
  capture.problem.steps[3].R_cross = 0.021f;
  capture.problem.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] = 0.2f;
  auto &config = capture.solver_config;
  config.rho = config.rho_u = 7.0f;
  config.rho_max = 32.0f;
  config.tolerance = 0.02f;
  config.max_iterations = 150;
  config.reuse_same_epoch = 1;
  config.state_polygons[1].count = 1;
  config.state_polygons[1].normal[0][0] = 1.f;
  config.state_polygons[1].normal[0][1] = .43f;
  config.state_polygons[1].upper[0] = .2f;
  config.adaptive_rho_update_period = 10;
  config.over_relaxation = 1.6f;
  std::fill_n(config.state_scale, RICCATI_MAX_NX, 1.0f);
  std::fill_n(config.input_scale, RICCATI_MAX_NU, 1.0f);
  config.terminal_constraints.count=1;
  config.terminal_constraints.rho=7.f;
  config.terminal_constraints.normal[0][MPC_OPT_IDX_U]=.25f;
  config.terminal_constraints.normal[0][MPC_OPT_IDX_R]=-.4f;
  config.terminal_constraints.upper[0]=.2f;
  ASSERT_TRUE(sdu_apex::qp_fixture::write_capture(path, capture));
  MpcRtiQpCapture_t decoded{};
  ASSERT_TRUE(sdu_apex::qp_fixture::read_capture(path, &decoded));
  EXPECT_EQ(decoded.problem.horizon, RICCATI_HORIZON);
  EXPECT_EQ(decoded.solver_config.terminal_constraints.count,1);
  EXPECT_FLOAT_EQ(decoded.solver_config.terminal_constraints.normal[0][MPC_OPT_IDX_U],.25f);
  EXPECT_FLOAT_EQ(decoded.solver_config.terminal_constraints.normal[0][MPC_OPT_IDX_R],-.4f);
  EXPECT_FLOAT_EQ(decoded.solver_config.terminal_constraints.upper[0],.2f);

  EXPECT_FLOAT_EQ(decoded.problem.steps[3].Q_offdiag[2][4], -0.375f);
  EXPECT_FLOAT_EQ(decoded.problem.steps[3].Q_offdiag[4][2], -0.375f);
  EXPECT_FLOAT_EQ(decoded.problem.steps[3].R_cross, 0.021f);
  EXPECT_EQ(decoded.solver_config.reuse_same_epoch, 1);
  EXPECT_EQ(decoded.solver_config.state_polygons[1].count, 1);
  EXPECT_FLOAT_EQ(decoded.solver_config.state_polygons[1].normal[0][1], .43f);
  EXPECT_FLOAT_EQ(decoded.problem.state_polygons[1].upper[0], .2f);
  EXPECT_FLOAT_EQ(decoded.problem.terminal_q[MPC_OPT_IDX_PROGRESS_DEVIATION],
      -125.0f);
  EXPECT_FLOAT_EQ(decoded.problem.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND],
      0.2f);
}

#include "../src/mpc_qp_failure_recorder.hpp"
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

TEST(MpcQpFixture, AsyncFailureRecorderPreservesBothPassesAndPhysicalEpoch)
{
  char directory[] = "/tmp/sdu_mpc_failure_capture_XXXXXX";
  ASSERT_NE(mkdtemp(directory), nullptr);
  const std::filesystem::path base(directory);
  auto first = std::make_unique<MpcRtiQpCapture_t>();
  auto second = std::make_unique<MpcRtiQpCapture_t>();
  for (auto *c : {first.get(), second.get()}) {
    c->valid = 1;
    c->capture_count = 1;
    c->problem.horizon = RICCATI_HORIZON;
    c->solver_config.rho = 7.0f;
    c->solver_config.rho_u = 7.0f;
    c->solver_config.rho_max = 32.0f;
    c->solver_config.tolerance = 0.002f;
    c->solver_config.max_iterations = 150;
    c->solver_config.adaptive_rho_update_period = 10;
    c->solver_config.over_relaxation = 1.6f;
    for (int k = 0; k < RICCATI_HORIZON; ++k) {
      for (int i = 0; i < RICCATI_MAX_NX; ++i) {
        c->problem.steps[k].x_lb[i] = -50.0f;
        c->problem.steps[k].x_ub[i] = 50.0f;
      }
      for (int i = 0; i < RICCATI_MAX_NU; ++i) {
        c->problem.steps[k].u_lb[i] = -1.0f;
        c->problem.steps[k].u_ub[i] = 1.0f;
      }
    }
    for (int i = 0; i < RICCATI_MAX_NX; ++i) {
      c->problem.terminal_x_lb[i] = -50.0f;
      c->problem.terminal_x_ub[i] = 50.0f;
      c->solver_config.state_scale[i] = 1.0f;
      c->initial_solver_state.state_scale[i] = 1.0f;
      c->final_solver_state.state_scale[i] = 1.0f;
    }
    for (int i = 0; i < RICCATI_MAX_NU; ++i) {
      c->solver_config.input_scale[i] = 1.0f;
      c->initial_solver_state.input_scale[i] = 1.0f;
      c->final_solver_state.input_scale[i] = 1.0f;
    }
    c->initial_solver_state.rho = 11.0f;
    c->initial_solver_state.y_x[4][2] = -0.33f;
  }
  first->problem.steps[2].q[3] = 12.5f;
  second->problem.steps[2].q[3] = -13.5f;
  second->initial_solver_state.y_x[4][2] = 0.88f;
  MpcRtiState_t initial{};
  initial.plant.e_y = -0.17f;
  initial.plant.u = 4.25f;
  initial.plant.source_plant_initialized = 1;
  initial.plant.source_imu_state_initialized = 1;
  initial.plant.source_plant_internal_state[0] = 71.0;
  MpcRtiCycleResult_t result{};
  result.nominal_control_count = RICCATI_HORIZON;
  result.nominal_controls[0].steering_target_rad = 0.3f;
  result.nominal_controls[0].throttle_command = 0.25f;
  result.admm_status = 1;
  result.primal_residual = 0.5f;
  {
    f1tenth_mpc::MpcQpFailureRecorder recorder(base);
    recorder.submit(1793573160746756676LL, 59.2778, 16.0f,
        initial, MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT,
        result, first.get(), second.get());
    // Joining the writer must drain pending snapshots before teardown.
  }
  auto entry = base / "epoch_1793573160746756676_case_0";
  auto decoded = std::make_unique<MpcRtiQpCapture_t>();
  ASSERT_TRUE(sdu_apex::qp_fixture::read_capture(
      (entry / "r1.fqp").string(), decoded.get()));
  EXPECT_FLOAT_EQ(decoded->problem.steps[2].q[3], 12.5f);
  EXPECT_FLOAT_EQ(decoded->initial_solver_state.y_x[4][2], -0.33f);
  ASSERT_TRUE(sdu_apex::qp_fixture::read_capture(
      (entry / "r2.fqp").string(), decoded.get()));
  EXPECT_FLOAT_EQ(decoded->problem.steps[2].q[3], -13.5f);
  EXPECT_FLOAT_EQ(decoded->initial_solver_state.y_x[4][2], 0.88f);
  std::ifstream state(entry / "physical_state.csv");
  std::string line;
  ASSERT_TRUE(std::getline(state, line));
  EXPECT_EQ(std::count(line.begin(), line.end(), ','), 37);
  std::ifstream metadata(entry / "context.json");
  ASSERT_TRUE(std::getline(metadata, line));
  EXPECT_NE(line.find("\"r2_captured\":true"), std::string::npos);
  std::filesystem::remove_all(base);
}
